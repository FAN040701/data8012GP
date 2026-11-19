"""Native-grid, patient-level segmentation evaluation, independent of model fitting."""
import argparse
import concurrent.futures as cf
import hashlib
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from scipy import ndimage as ndi

from experiment import ROOT,PROC,init_runtime,predict_volume
from models import UNet
from prepare_data import NAMES,remap_labels,resample
from native_resample import restore_on_gpu


def restore_probabilities(prob,meta):
    shape=tuple(meta['original_shape_xyz'])
    best=np.full(shape,-np.inf,dtype=np.float32)
    labels=np.zeros(shape,dtype=np.uint8)
    src=np.array(meta['model_affine_xyz'])
    dst=np.array(meta['original_affine'])
    for k in range(5):
        p=resample(prob[k].transpose(2,1,0),src,dst,shape,1,0)
        update=p>best
        best[update]=p[update]
        labels[update]=k
    return labels


def binary_metrics(pred,target,spacing):
    npred=int(pred.sum()); ntrue=int(target.sum())
    # Fix the scored-organ set by reference presence, independently of predictions.
    dice=float(2*np.count_nonzero(pred&target)/(npred+ntrue)) if ntrue else np.nan
    if not npred or not ntrue:
        state='both_empty' if not npred and not ntrue else ('prediction_empty' if not npred else 'reference_empty')
        return {'dice':dice,'hd95_mm':np.nan,'status':state,'pred_voxels':npred,'true_voxels':ntrue,'relative_volume_error':float(abs(npred-ntrue)/ntrue) if ntrue else np.nan,'absent_reference_false_positive_ml':float(npred*np.prod(spacing)/1000) if not ntrue else 0.0}
    union=pred|target
    bounds=[]
    for axis in range(3):
        axes=tuple(i for i in range(3) if i!=axis)
        nonzero=np.where(union.any(axis=axes))[0]
        bounds.append(slice(max(int(nonzero[0])-1,0),min(int(nonzero[-1])+2,union.shape[axis])))
    p=pred[tuple(bounds)]; t=target[tuple(bounds)]
    structure=ndi.generate_binary_structure(3,1)
    ps=p^ndi.binary_erosion(p,structure=structure,border_value=0)
    ts=t^ndi.binary_erosion(t,structure=structure,border_value=0)
    d_to_t=ndi.distance_transform_edt(~ts,sampling=spacing)[ps]
    d_to_p=ndi.distance_transform_edt(~ps,sampling=spacing)[ts]
    hd95=float(np.percentile(np.concatenate([d_to_t,d_to_p]),95))
    return {'dice':dice,'hd95_mm':hd95,'status':'ok','pred_voxels':npred,'true_voxels':ntrue,'relative_volume_error':float(abs(npred-ntrue)/ntrue),'absent_reference_false_positive_ml':0.0}


def score_case(pred,case,factor,kind,seed,run,infer_seconds):
    meta=json.loads((PROC/case/'metadata.json').read_text())
    raw=nib.as_closest_canonical(nib.load(ROOT/meta['raw_label']))
    target=remap_labels(np.asarray(raw.dataobj,dtype=np.uint8))
    spacing=meta['original_spacing_xyz_mm']
    pred_dir=run/'predictions'/f'f{factor}'
    pred_dir.mkdir(parents=True,exist_ok=True)
    output=nib.Nifti1Image(pred,np.array(meta['original_affine']))
    output.header.set_xyzt_units('mm')
    nib.save(output,pred_dir/f'{case}.nii.gz')
    rows=[]
    for k in range(1,5):
        result=binary_metrics(pred==k,target==k,spacing)
        rows.append({'kind':kind,'seed':seed,'case':case,'factor':factor,'organ':NAMES[k],
                     'native_z_spacing_mm':spacing[2],'effective_z_sampling_mm':spacing[2]*factor,
                     'inference_seconds':infer_seconds,**result})
    return rows


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--kind',required=True)
    ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--workers',type=int,default=6)
    args=ap.parse_args()
    init_runtime(args.seed)
    run=ROOT/'results/runs'/f'{args.kind}_seed{args.seed}'
    if (run/'evaluation_complete.json').exists():
        print('Evaluation already completed',run,flush=True)
        return
    checkpoint=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
    model=UNet(args.kind,width=checkpoint['config']['width'])
    model.load_state_dict(checkpoint['state_dict'])
    device=torch.device(args.device)
    model=model.to(device).eval()
    test_ids=json.loads((ROOT/'data/metadata/split.json').read_text())['splits']['test']
    missing=[c for c in test_ids if not (PROC/c/'metadata.json').exists()]
    if missing: raise RuntimeError(f'Incomplete locked test data: {missing}')
    cfg=checkpoint['config']
    rows=[]; jobs={}; started=time.time()
    with cf.ThreadPoolExecutor(args.workers) as pool:
        for factor in [1,2,3]:
            for case in test_ids:
                image=np.load(PROC/case/f'image_f{factor}.npy')
                if device.type=='cuda': torch.cuda.synchronize(device)
                before=time.time()
                prob=predict_volume(model,image,args.kind,device,cfg['patch'][0])
                meta=json.loads((PROC/case/'metadata.json').read_text())
                pred=restore_on_gpu(prob,meta,device) if device.type=='cuda' else restore_probabilities(prob,meta)
                if device.type=='cuda': torch.cuda.synchronize(device)
                elapsed=time.time()-before
                f=pool.submit(score_case,pred,case,factor,args.kind,args.seed,run,elapsed)
                jobs[f]=(case,factor)
                if len(jobs)>=2*args.workers:
                    done,_=cf.wait(jobs,return_when=cf.FIRST_COMPLETED)
                    for f in done:
                        result=f.result(); rows.extend(result)
                        c,fac=jobs.pop(f)
                        print(f'{args.kind} seed={args.seed} {c} f={fac} Dice={np.nanmean([r["dice"] for r in result]):.4f} completed={len(rows)//4}/120',flush=True)
                    pd.DataFrame(rows).to_csv(run/'metrics.csv',index=False)
        for f in cf.as_completed(jobs):
            rows.extend(f.result())
    frame=pd.DataFrame(rows).sort_values(['factor','case','organ'])
    frame.to_csv(run/'metrics.csv',index=False)
    summary=frame.groupby('factor')[['dice','hd95_mm','relative_volume_error']].mean().reset_index().to_dict('records')
    complete={'kind':args.kind,'seed':args.seed,'cases':len(test_ids),'conditions':[1,2,3],'rows':len(rows),'checkpoint_step':checkpoint['step'],'elapsed_seconds':time.time()-started,'mean_organ_scores':summary,'hd95_definition':'pooled bidirectional 95th percentile of surface-voxel-centre distances in native mm','native_probability_resampling':True}
    complete['inverse_resampling_backend']='validated GPU physical trilinear grid_sample' if device.type=='cuda' else 'scipy CPU physical affine_transform'
    complete['evaluation_source_sha256']={name:hashlib.sha256((ROOT/'src'/name).read_bytes()).hexdigest() for name in ['evaluate.py','native_resample.py']}
    (run/'evaluation_complete.json').write_text(json.dumps(complete,indent=2))
    print(json.dumps(complete),flush=True)


if __name__=='__main__': main()
