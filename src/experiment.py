"""Train and reconstruct matched 2D / 2.5D / 3D segmentation experiments."""
import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from models import UNet, planar_input, volume_loss

ROOT=Path(__file__).resolve().parents[1]
PROC=ROOT/'data/processed'


def init_runtime(seed,threads=4):
    torch.set_num_threads(threads)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True


class SlabSampler:
    def __init__(self,ids,seed,patch,augmentation=False):
        self.ids=ids
        self.rng=np.random.default_rng(seed)
        self.augmentation=augmentation
        self.aug_rng=np.random.default_rng(seed+100000)
        self.patch=np.array(patch)
        self.pad=self.patch//2+np.array([1,0,0])
        self.data=[]
        for case in ids:
            image=np.load(PROC/case/'image_f1.npy').astype(np.float32)
            label=np.load(PROC/case/'label.npy')
            with np.load(PROC/case/'foreground.npz') as f:
                coords=[f[str(i)] for i in range(1,5) if len(f[str(i)])]
            pairs=tuple((int(p),int(p)) for p in self.pad)
            images=[np.pad(image,pairs)]
            if augmentation:
                images.extend(np.pad(np.load(PROC/case/f'image_f{f}.npy').astype(np.float32),pairs) for f in [2,3])
            self.data.append((images,np.pad(label,pairs),np.array(label.shape),coords))

    def next(self):
        idx=int(self.rng.integers(len(self.data)))
        images,label,shape,coords=self.data[idx]
        image=images[int(self.aug_rng.choice(3,p=[0.5,0.25,0.25]))] if self.augmentation else images[0]
        if self.rng.random()<0.5 and coords:
            group=coords[int(self.rng.integers(len(coords)))]
            center=group[int(self.rng.integers(len(group)))].astype(int)
        else:
            center=np.array([self.rng.integers(int(n)) for n in shape])
        start=center+self.pad-self.patch//2
        end=start+self.patch
        z,y,x=start; zz,yy,xx=end
        halo=image[z-1:zz+1,y:yy,x:xx].copy()
        target=label[z:zz,y:yy,x:xx].astype(np.int64)
        scale=self.rng.uniform(0.9,1.1)
        shift=self.rng.uniform(-0.03,0.03)
        halo=np.clip(halo*scale+shift,0,1)
        return halo,target,idx,start-self.pad


def make_logits(model,halo,kind):
    if kind=='3d':
        return model(halo[None,None,1:-1])
    return model(planar_input(halo,kind)).permute(1,0,2,3).unsqueeze(0)


def positions(n,patch,stride):
    if n<=patch: return [0]
    return sorted(set(list(range(0,n-patch+1,stride))+[n-patch]))


@torch.inference_mode()
def predict_volume(model,image,kind,device,depth=32,batch=24):
    """Entire in-plane FOV; 3D has 50% overlapping depth windows only."""
    benchmark_before=torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark=False  # Avoid autotuning every patient's different FOV.
    d,h,w=image.shape
    hh=int(math.ceil(h/16)*16); ww=int(math.ceil(w/16)*16)
    dd=max(d,depth) if kind=='3d' else d
    x=torch.as_tensor(np.array(image,dtype=np.float32,copy=True),device=device)
    x=F.pad(x,(0,ww-w,0,hh-h,0,dd-d),value=0)
    amp=device.type=='cuda'
    if kind!='3d':
        out=torch.empty((5,d,hh,ww),device=device,dtype=torch.float32)
        for begin in range(0,d,batch):
            stop=min(begin+batch,d)
            center=x[begin:stop]
            if kind=='2d':
                inp=center[:,None]
            elif kind=='repeat':
                inp=center[:,None].expand(-1,3,-1,-1).contiguous()
            else:
                indices=torch.arange(begin,stop,device=device)
                # At acquired scan boundaries, the missing neighbour is air.
                previous=x[(indices-1).clamp(0,dd-1)].clone()
                following=x[(indices+1).clamp(0,dd-1)].clone()
                previous[indices==0]=0
                following[indices==d-1]=0
                inp=torch.stack([previous,center,following],1)
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=amp):
                logits=model(inp)
            out[:,begin:stop]=logits.float().softmax(1).permute(1,0,2,3)
    else:
        out=torch.zeros((5,dd,hh,ww),device=device)
        norm=torch.zeros(dd,device=device)
        zz=torch.arange(depth,device=device,dtype=torch.float32)-(depth-1)/2
        weight=torch.exp(-0.5*(zz/(depth/8))**2).clamp_min(1e-3)
        for begin in positions(dd,depth,depth//2):
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=amp):
                logits=model(x[None,None,begin:begin+depth])
            out[:,begin:begin+depth]+=logits[0].float().softmax(0)*weight[None,:,None,None]
            norm[begin:begin+depth]+=weight
        out/=norm[None,:,None,None]
    result=out[:,:d,:h,:w].cpu().numpy()
    torch.backends.cudnn.benchmark=benchmark_before
    return result


def dice_scores(pred,target):
    values=[]
    for k in range(1,5):
        p=pred==k; t=target==k
        denom=p.sum()+t.sum()
        values.append(float(2*(p&t).sum()/denom) if t.any() else float('nan'))
    return values


def validate(model,ids,kind,device,depth):
    model.eval()
    rows=[]
    for case in ids:
        image=np.load(PROC/case/'image_f1.npy')
        y=np.load(PROC/case/'label.npy')
        pred=predict_volume(model,image,kind,device,depth).argmax(0)
        scores=dice_scores(pred,y)
        rows.append({'case':case,'dice':scores,'macro_dice':float(np.nanmean(scores))})
    model.train()
    return float(np.mean([r['macro_dice'] for r in rows])),rows


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--kind',choices=['2d','repeat','2p5d','3d','2p5d_aug'],required=True)
    ap.add_argument('--seed',type=int,default=11)
    ap.add_argument('--steps',type=int,default=None)
    ap.add_argument('--pilot',action='store_true')
    ap.add_argument('--device',default='cuda:0')
    args=ap.parse_args()
    confpath=ROOT/'configs/experiment.json'
    cfg=json.loads(confpath.read_text()) if confpath.exists() else {'steps':6000,'patch':[32,128,128],'width':16,'lr':0.0003,'val_every':1000}
    if args.steps is not None: cfg['steps']=args.steps
    split=json.loads((ROOT/'data/metadata/split.json').read_text())['splits']
    train_ids=split['train']; val_ids=split['val']
    if args.pilot:
        train_ids=[c for c in train_ids if (PROC/c/'metadata.json').exists()][:2]
        val_ids=train_ids[:1]
        if not train_ids: raise RuntimeError('Pilot needs prepared fitting cases')
    else:
        missing=[c for c in train_ids+val_ids if not (PROC/c/'metadata.json').exists()]
        if missing: raise RuntimeError(f'Incomplete fit/development data: {missing}')
    run_id=f'{args.kind}_seed{args.seed}'
    run=ROOT/('results/pilot' if args.pilot else 'results/runs')/(f'{run_id}_steps{cfg["steps"]}' if args.pilot else run_id)
    run.mkdir(parents=True,exist_ok=True)
    init_runtime(args.seed)
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA requires host execution, not the restricted sandbox')
    model=UNet(args.kind,width=cfg['width']).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=1e-5)
    sampler=SlabSampler(train_ids,args.seed,cfg['patch'],augmentation=args.kind=='2p5d_aug')
    params=sum(p.numel() for p in model.parameters())
    modelhash=hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()
    metadata={'kind':args.kind,'seed':args.seed,'parameters':params,'initial_parameter_sha256':modelhash,'config':cfg,'train_ids':train_ids,'val_ids':val_ids,'torch_version':torch.__version__,'cuda_version':torch.version.cuda,'device':torch.cuda.get_device_name(device) if device.type=='cuda' else 'cpu','start_unix':time.time(),'pilot':args.pilot}
    metadata['source_sha256']={name:hashlib.sha256((ROOT/'src'/name).read_bytes()).hexdigest() for name in ['experiment.py','models.py','prepare_data.py','evaluate.py']}
    (run/'metadata.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata),flush=True)
    model.train()
    started=time.time(); last=started; best=-1; best_step=0
    val_every=min(cfg['val_every'],cfg['steps'])
    rows=[]
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
    for step in range(1,cfg['steps']+1):
        halo,y,patient_idx,start=sampler.next()
        x=torch.from_numpy(halo).to(device)
        target=torch.from_numpy(y)[None].to(device)
        optimizer.zero_grad(set_to_none=True)
        lr=cfg['lr']*(0.1+0.9*0.5*(1+math.cos(math.pi*(step-1)/cfg['steps'])))
        for group in optimizer.param_groups: group['lr']=lr
        with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
            logits=make_logits(model,x,args.kind)
            loss=volume_loss(logits,target)
        if not torch.isfinite(loss): raise RuntimeError(f'Nonfinite loss at step {step}')
        loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),12.0)
        if not torch.isfinite(grad) or float(grad)==0: raise RuntimeError(f'Invalid gradient at step {step}')
        optimizer.step()
        if step==1 or step%50==0:
            now=time.time()
            row={'step':step,'loss':float(loss),'grad_norm':float(grad),'lr':lr,'elapsed_seconds':now-started,'interval_seconds':now-last,'patient':train_ids[patient_idx]}
            rows.append(row)
            with (run/'train.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
            print(f"{run_id} step={step}/{cfg['steps']} loss={float(loss):.4f} elapsed={now-started:.1f}s",flush=True)
            last=now
        if step%val_every==0 or step==cfg['steps']:
            score,case_rows=validate(model,val_ids,args.kind,device,cfg['patch'][0])
            record={'step':step,'macro_dice':score,'cases':case_rows,'elapsed_seconds':time.time()-started}
            with (run/'validation.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
            print(f'{run_id} validation step={step} dice={score:.5f}',flush=True)
            checkpoint={'state_dict':model.state_dict(),'kind':args.kind,'seed':args.seed,'step':step,'validation_dice':score,'config':cfg}
            if score>best:
                best=score; best_step=step
                torch.save(checkpoint,run/'best.pt')
            torch.save(checkpoint,run/'last.pt')
    finished={'best_validation_macro_dice':best,'best_step':best_step,'training_steps':cfg['steps'],'total_seconds':time.time()-started,'peak_gpu_allocated_mb':torch.cuda.max_memory_allocated(device)/2**20 if device.type=='cuda' else None,'finish_unix':time.time()}
    (run/'complete.json').write_text(json.dumps(finished,indent=2))
    print(json.dumps(finished),flush=True)


if __name__=='__main__': main()
