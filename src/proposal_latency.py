"""Original-cohort latency from decoded CT in host RAM to native-grid CPU labels."""
import json
import math
import platform
import os
import subprocess
import time

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from experiment import PROC, ROOT, init_runtime, positions, predict_volume
from models import UNet
from native_resample import restore_on_gpu
from prepare_data import resample

OUT = ROOT/'results/slice_selection/proposal_latency'
MODELS = ['2d','2p5d','3d']
DEVICE = torch.device('cuda:0')


def decoded(case):
    meta = json.loads((PROC/case/'metadata.json').read_text())
    image = nib.load(ROOT/meta['raw_image'])
    return np.asarray(image.dataobj,dtype=np.float32), image.affine, meta


def preprocess(raw, affine):
    transform = nib.orientations.io_orientation(affine)
    canonical = nib.orientations.apply_orientation(raw,transform)
    original_affine = affine @ nib.orientations.inv_ornt_aff(transform,raw.shape)
    shape = np.array(canonical.shape)
    spacing = nib.affines.voxel_sizes(original_affine)
    target_shape = np.ceil((shape-1)*spacing/3).astype(int)+1
    model_affine = np.diag([3.,3.,3.,1.])
    model_affine[:3,3]=original_affine[:3,3]
    img=resample(canonical,original_affine,model_affine,target_shape,1,-1000.)
    img=((np.clip(img,-200,300)+200)/500).transpose(2,1,0).astype(np.float16)
    return img,dict(original_shape_xyz=shape, original_affine=original_affine,
                    model_affine_xyz=model_affine)


@torch.inference_mode()
def predict_3d(model, image, batch):
    d,h,w=image.shape
    dd,hh,ww=max(d,32),math.ceil(h/16)*16,math.ceil(w/16)*16
    x=torch.as_tensor(np.array(image,dtype=np.float32,copy=True),device=DEVICE)
    x=F.pad(x,(0,ww-w,0,hh-h,0,dd-d))
    out=torch.zeros((5,dd,hh,ww),device=DEVICE)
    norm=torch.zeros(dd,device=DEVICE)
    z=torch.arange(32,device=DEVICE,dtype=torch.float32)-15.5
    weight=torch.exp(-.5*(z/4)**2).clamp_min(1e-3)
    starts=positions(dd,32,16)
    for idx in range(0,len(starts),batch):
        slabs=starts[idx:idx+batch]
        inp=torch.stack([x[k:k+32] for k in slabs])[:,None]
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits=model(inp)
        for j,k in enumerate(slabs):
            out[:,k:k+32]+=logits[j].float().softmax(0)*weight[None,:,None,None]
            norm[k:k+32]+=weight
    out/=norm[None,:,None,None]
    return out[:,:d,:h,:w].cpu().numpy()


def timed(model,kind,batch,case_data):
    raw,affine,_=case_data
    torch.cuda.synchronize()
    start=time.perf_counter()
    image,geometry=preprocess(raw,affine)
    prepared=time.perf_counter()
    prob=predict_3d(model,image,batch) if kind=='3d' else predict_volume(model,image,kind,DEVICE,32,batch)
    labels=restore_on_gpu(prob,geometry,DEVICE)
    torch.cuda.synchronize()
    finish=time.perf_counter()
    return (finish-start,prepared-start,finish-prepared),labels


def exclusive_gpu():
    gpu_id = subprocess.check_output(
        ['nvidia-smi','-i','0','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    tasks = subprocess.check_output(
        ['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    for line in tasks.splitlines():
        fields = [value.strip() for value in line.split(',')]
        if len(fields)==2 and fields[0]==gpu_id and fields[1]!=str(os.getpid()):
            raise RuntimeError(f'Other GPU work detected during timing: {line}')
    return gpu_id


def fit_table():
    rows=[]
    for kind in MODELS:
        for seed in [11,22,33]:
            run=ROOT/'results/runs'/f'{kind}_seed{seed}'
            t=json.loads((run/'complete.json').read_text())
            rows.append(dict(kind=kind,seed=seed,steps=t['training_steps'],
                             elapsed_hours_including_validation=t['total_seconds']/3600))
    pd.DataFrame(rows).to_csv(OUT/'training_time.csv',index=False)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    split=json.loads((ROOT/'data/metadata/split.json').read_text())['splits']
    init_runtime(11)
    torch.backends.cudnn.benchmark=False
    models={}
    for kind in MODELS:
        ckpt=torch.load(ROOT/'results/runs'/f'{kind}_seed11'/'best.pt',map_location='cpu',weights_only=False)
        models[kind]=UNet(kind,16).to(DEVICE).eval()
        models[kind].load_state_dict(ckpt['state_dict'])
    dev=sorted(split['val'],key=lambda c: np.prod(json.loads((PROC/c/'metadata.json').read_text())['model_shape_dhw']))
    subset=[dev[0],dev[len(dev)//2],dev[-1]]
    tuning=[]; batch_sizes={}
    dev_raw={c:decoded(c) for c in subset}
    for c,data in dev_raw.items():
        model_input,geometry=preprocess(*data[:2])
        assert np.array_equal(model_input,np.load(PROC/c/'image_f1.npy'))
        assert np.allclose(geometry['original_affine'],data[2]['original_affine'])
    for kind,model in models.items():
        scores={}
        for batch in [1,2,4,8,16,32]:
            seconds=[]
            try:
                for c in subset:
                    timed(model,kind,batch,dev_raw[c])
                    measurements,_=timed(model,kind,batch,dev_raw[c])
                    seconds.append(measurements[0])
                    tuning.append(dict(kind=kind,batch=batch,case=c,seconds=seconds[-1]))
                scores[batch]=float(np.median(seconds))
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                break
        best=min(scores.values())
        batch_sizes[kind]=min(b for b,t in scores.items() if t<=1.02*best)
    pd.DataFrame(tuning).to_csv(OUT/'development_tuning.csv',index=False)
    (OUT/'settings.json').write_text(json.dumps(dict(batch_sizes=batch_sizes,dev_tuning_cases=subset,
        GPU=torch.cuda.get_device_name(),device_index=0,torch=torch.__version__,python=platform.python_version(),
        cuda=torch.version.cuda,threads=torch.get_num_threads(),warmup_per_model=5,seed=11),indent=2))
    for kind,model in models.items():
        for c in dev[:5]:
            timed(model,kind,batch_sizes[kind],decoded(c))
    rows=[]
    for rep in range(3):
        exclusive_gpu()
        for case in split['test']:
            data=decoded(case)
            for kind in MODELS[rep:]+MODELS[:rep]:
                t,pred=timed(models[kind],kind,batch_sizes[kind],data)
                reference=ROOT/'results/runs'/f'{kind}_seed11'/'predictions/f1'/f'{case}.nii.gz'
                old=np.asarray(nib.load(reference).dataobj)
                assert old.shape==pred.shape
                disagreement=float(np.mean(old!=pred))
                rows.append(dict(round=rep+1,case=case,kind=kind,end_to_end_s=t[0],preprocess_s=t[1],
                            prediction_restore_s=t[2],fraction_different_from_evaluation=disagreement))
            pd.DataFrame(rows).to_csv(OUT/'measurements.csv',index=False)
        exclusive_gpu()
        print('Latency round',rep+1,'complete',flush=True)
    frame=pd.DataFrame(rows)
    wide=frame.groupby(['case','kind']).end_to_end_s.median().unstack()
    assert wide.shape==(40,3) and len(frame)==360
    wide.describe(percentiles=[.25,.5,.75]).T.to_csv(OUT/'latency_summary.csv')
    ratios=[]
    samples=np.random.default_rng(20261119).integers(0,40,(2000,40))
    for a,b in [('2p5d','2d'),('3d','2d'),('2p5d','3d')]:
        values=(wide[a]/wide[b]).to_numpy()
        lo,hi=np.quantile(np.median(values[samples],axis=1),[.025,.975])
        ratios.append(dict(numerator=a,denominator=b,median_paired_ratio=np.median(values),ci_low=lo,ci_high=hi))
    pd.DataFrame(ratios).to_csv(OUT/'paired_ratios.csv',index=False)
    fit_table()
    (OUT/'complete.json').write_text(json.dumps(dict(measurements=len(frame),patients=len(wide),
        timer='decoded CT in CPU memory to native labels in CPU memory',rounds=3,
        max_label_disagreement=frame.fraction_different_from_evaluation.max()),indent=2))


if __name__=='__main__':
    main()
