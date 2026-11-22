"""Time whole model grids already on an H100, up to probabilities still on that GPU."""
import json
import math
import os
import subprocess
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from experiment import PROC, init_runtime, positions, predict_volume
from models import UNet
from proposal_latency import predict_3d
from review_refits import ROOT, OUT


@torch.inference_mode()
def gpu_grid(model, x, kind, batch):
    d, h, w = x.shape
    dd = max(d,32) if kind == '3d' else d
    hh, ww = math.ceil(h/16)*16, math.ceil(w/16)*16
    x = F.pad(x,(0,ww-w,0,hh-h,0,dd-d))
    if kind == '3d':
        output = torch.zeros((5,dd,hh,ww),device=x.device)
        norm = torch.zeros(dd, device=x.device)
        z = torch.arange(32,device=x.device,dtype=torch.float32)-15.5
        weight = torch.exp(-.5*(z/4)**2).clamp_min(1e-3)
        starts = positions(dd,32,16)
        for i in range(0,len(starts),batch):
            slices = starts[i:i+batch]
            slab = torch.stack([x[z:z+32] for z in slices])[:,None]
            with torch.autocast('cuda',dtype=torch.bfloat16):
                pred = model(slab)
            for j,z in enumerate(slices):
                output[:,z:z+32] += pred[j].float().softmax(0)*weight[None,:,None,None]
                norm[z:z+32] += weight
        output /= norm[None,:,None,None]
    else:
        output = torch.empty((5,d,hh,ww),device=x.device)
        for i in range(0,d,batch):
            j = min(i+batch,d)
            centre = x[i:j]
            if kind == '2d':
                stack = centre[:,None]
            else:
                ix = torch.arange(i,j,device=x.device)
                a = x[(ix-1).clamp(0,d-1)].clone()
                b = x[(ix+1).clamp(0,d-1)].clone()
                a[ix==0] = 0
                b[ix==d-1] = 0
                stack = torch.stack([a,centre,b],dim=1)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                pred = model(stack)
            output[:,i:j] = pred.float().softmax(1).permute(1,0,2,3)
    return output[:,:d,:h,:w]


def idle_check():
    identifier = os.environ.get('CUDA_VISIBLE_DEVICES','0').split(',')[0]
    uuid = subprocess.check_output(['nvidia-smi','-i',identifier,
        '--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    processes = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader,nounits'],text=True).splitlines()
    if any(row.split(',')[0].strip()==uuid and int(row.split(',')[1])!=os.getpid()
           for row in processes if ',' in row):
        raise RuntimeError(f'Another process is using measurement GPU {identifier}')


def main():
    dest = OUT/'cost'
    dest.mkdir(parents=True,exist_ok=True)
    split = json.loads((ROOT/'data/metadata/split.json').read_text())['splits']
    # Development-selected whole-case batches from the frozen latency experiment.
    batches = {'2d':32,'2p5d':16,'3d':4}
    settings = ROOT/'results/slice_selection/proposal_latency/settings.json'
    if settings.exists():
        batches=json.loads(settings.read_text())['batch_sizes']
    init_runtime(11)
    torch.backends.cudnn.benchmark=False
    models = {}
    for kind in batches:
        m=UNet(kind,16).cuda().eval()
        ckpt=torch.load(ROOT/'results/runs'/f'{kind}_seed11/best.pt',map_location='cpu',weights_only=False)
        m.load_state_dict(ckpt['state_dict'])
        models[kind]=m
    idle_check()
    def load(case):
        raw=np.load(PROC/case/'image_f1.npy')
        return raw, torch.as_tensor(raw,device='cuda',dtype=torch.float32)
    for case in split['val'][:10]:
        raw,x=load(case)
        for k,m in models.items():
            prob=gpu_grid(m,x,k,batches[k])
            if case==split['val'][0]:
                ref=predict_3d(m,raw,batches[k]) if k=='3d' else predict_volume(
                    m,raw,k,torch.device('cuda:0'),batch=batches[k])
                np.testing.assert_array_equal(prob.cpu().numpy(),ref)
    rows=[]
    kinds=list(models)
    for n in range(3):
        idle_check()
        for case in split['test']:
            raw,x=load(case)
            for k in kinds[n:]+kinds[:n]:
                torch.cuda.synchronize()
                start=time.perf_counter()
                p=gpu_grid(models[k],x,k,batches[k])
                torch.cuda.synchronize()
                dt=time.perf_counter()-start
                rows.append(dict(kind=k,case=case,round=n+1,seconds=dt,batch=batches[k],
                                 voxels=x.numel(),voxels_per_s=x.numel()/dt))
                del p
        idle_check()
        print('GPU timing round',n+1,flush=True)
    result=pd.DataFrame(rows)
    assert len(result)==360 and not result.duplicated(['kind','case','round']).any()
    result.to_csv(dest/'gpu_measurements.csv',index=False)
    (dest/'gpu_environment.json').write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,cuda=torch.version.cuda,warmup_cases=10,rounds=3,
        input='preprocessed float32 on GPU',output='float32 class probabilities on model grid on GPU'),indent=2))


if __name__=='__main__':
    main()
