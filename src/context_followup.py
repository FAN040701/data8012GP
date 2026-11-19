"""Sequential, fresh-patient check of ±9 mm versus ±3 mm context.

Only the additional neighbour offsets change. Original study code and outputs
are imported without modification. See configs/followup_context_protocol.md.
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import math
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from experiment import ROOT,PROC,SlabSampler,init_runtime,predict_volume,dice_scores
from models import UNet,volume_loss
from evaluate import score_case
from native_resample import restore_on_gpu

OUT=ROOT/'results/followup_context'
META=ROOT/'data/metadata/followup_context'


def source_hashes():
    return {n:hashlib.sha256((ROOT/'src'/n).read_bytes()).hexdigest() for n in
            ['context_followup.py','experiment.py','models.py','evaluate.py','native_resample.py']}


class WideSampler(SlabSampler):
    def __init__(self,ids,seed,patch):
        super().__init__(ids,seed,patch)
        self.pad[0]+=2
        self.data=[([np.pad(images[0],((2,2),(0,0),(0,0)))],
                    np.pad(label,((2,2),(0,0),(0,0))),shape,coords)
                   for images,label,shape,coords in self.data]

    def next(self):
        idx=int(self.rng.integers(len(self.data)))
        images,label,shape,coords=self.data[idx]
        if self.rng.random()<.5 and coords:
            group=coords[int(self.rng.integers(len(coords)))]
            center=group[int(self.rng.integers(len(group)))].astype(int)
        else: center=np.array([self.rng.integers(int(n)) for n in shape])
        start=center+self.pad-self.patch//2
        end=start+self.patch
        z,y,x=start; zz,yy,xx=end
        halo=images[0][z-3:zz+3,y:yy,x:xx].copy()
        target=label[z:zz,y:yy,x:xx].astype(np.int64)
        scale=self.rng.uniform(.9,1.1); shift=self.rng.uniform(-.03,.03)
        halo=np.clip(halo*scale+shift,0,1)
        return halo,target,idx,start-self.pad


def wide_input(halo):
    return torch.stack([halo[:-6],halo[3:-3],halo[6:]],1)


@torch.inference_mode()
def predict_wide(model,image,device,batch=24,stride=3):
    previous_benchmark=torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark=False
    d,h,w=image.shape; hh=math.ceil(h/16)*16; ww=math.ceil(w/16)*16
    x=torch.as_tensor(np.array(image,dtype=np.float32,copy=True),device=device)
    x=F.pad(x,(0,ww-w,0,hh-h),value=0)
    out=torch.empty((5,d,hh,ww),device=device,dtype=torch.float32)
    for begin in range(0,d,batch):
        stop=min(begin+batch,d); indices=torch.arange(begin,stop,device=device)
        earlier=x[(indices-stride).clamp(0,d-1)].clone()
        later=x[(indices+stride).clamp(0,d-1)].clone()
        earlier[indices<stride]=0; later[indices>=d-stride]=0
        inp=torch.stack([earlier,x[begin:stop],later],1)
        with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
            logits=model(inp)
        out[:,begin:stop]=logits.float().softmax(1).permute(1,0,2,3)
    result=out[:,:,:h,:w].cpu().numpy()
    torch.backends.cudnn.benchmark=previous_benchmark
    return result


def check():
    OUT.mkdir(exist_ok=True)
    split=json.loads((ROOT/'data/metadata/split.json').read_text())['splits']
    ids=split['train'][:2]; patch=[32,128,128]
    narrow=SlabSampler(ids,11,patch); wide=WideSampler(ids,11,patch)
    for _ in range(100):
        a,y,i,start=narrow.next(); b,yy,j,ss=wide.next()
        assert i==j and np.array_equal(start,ss) and np.array_equal(y,yy)
        assert np.array_equal(a,b[2:-2]) and b.shape[0]==38
    halo=torch.arange(38)[:,None,None].expand(-1,8,8)
    encoded=wide_input(halo)
    assert torch.equal(encoded[:,0,0,0],torch.arange(32))
    assert torch.equal(encoded[:,1,0,0],torch.arange(3,35))
    assert torch.equal(encoded[:,2,0,0],torch.arange(6,38))
    init_runtime(11)
    model=UNet('2p5d',16).to('cuda').eval()
    case=ids[0]; image=np.load(PROC/case/'image_f1.npy')
    baseline=predict_volume(model,image,'2p5d',torch.device('cuda'),32)
    equivalent=predict_wide(model,image,torch.device('cuda'),stride=1)
    maxerr=float(np.abs(baseline-equivalent).max())
    assert maxerr<1e-6
    (OUT/'pipeline_checks.json').write_text(json.dumps({'matched_slabs_and_intensities_100_draws':True,
        'offset_encoding_3_grid_steps':True,'physical_neighbour_offset_mm':9,
        'stride1_reconstruction_matches_original_probability_max_error':maxerr,
        'checked_on_fitting_case':case},indent=2))
    print('Follow-up input checks passed',flush=True)


def train(seed):
    run=OUT/f'wide_seed{seed}'; run.mkdir(parents=True,exist_ok=True)
    if (run/'complete.json').exists(): return
    cfg=json.loads((ROOT/'configs/experiment.json').read_text())
    split=json.loads((ROOT/'data/metadata/split.json').read_text())['splits']
    init_runtime(seed); device=torch.device('cuda')
    model=UNet('2p5d',cfg['width']).to(device)
    initial=hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()
    original=json.loads((ROOT/'results/runs'/f'2p5d_seed{seed}'/'metadata.json').read_text())
    assert initial==original['initial_parameter_sha256']
    sampler=WideSampler(split['train'],seed,cfg['patch'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=1e-5)
    metadata={'kind':'wide','seed':seed,'config':cfg,'context_offset_mm':9,'context_stride':3,
              'initial_parameter_sha256':initial,'parameters':sum(p.numel() for p in model.parameters()),
              'source_sha256':source_hashes(),'train_ids':split['train'],'val_ids':split['val'],
              'start_unix':time.time(),'device':torch.cuda.get_device_name()}
    (run/'metadata.json').write_text(json.dumps(metadata,indent=2))
    started=time.time(); best=-1; best_step=0
    model.train()
    for step in range(1,cfg['steps']+1):
        halo,target,idx,start=sampler.next()
        x=torch.from_numpy(halo).to(device); y=torch.from_numpy(target)[None].to(device)
        optimizer.zero_grad(set_to_none=True)
        lr=cfg['lr']*(.1+.9*.5*(1+math.cos(math.pi*(step-1)/cfg['steps'])))
        for group in optimizer.param_groups: group['lr']=lr
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
            logits=model(wide_input(x)).permute(1,0,2,3).unsqueeze(0)
            loss=volume_loss(logits,y)
        assert torch.isfinite(loss)
        loss.backward(); grad=torch.nn.utils.clip_grad_norm_(model.parameters(),12.0)
        assert torch.isfinite(grad) and grad>0
        optimizer.step()
        if step==1 or step%50==0:
            row={'step':step,'loss':float(loss),'grad_norm':float(grad),'lr':lr,'patient':split['train'][idx],'elapsed_seconds':time.time()-started}
            with (run/'train.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
            print('wide',seed,step,float(loss),flush=True)
        if step%cfg['val_every']==0 or step==cfg['steps']:
            model.eval(); rows=[]
            for case in split['val']:
                image=np.load(PROC/case/'image_f1.npy'); target=np.load(PROC/case/'label.npy')
                prediction=predict_wide(model,image,device).argmax(0)
                rows.append({'case':case,'macro_dice':float(np.nanmean(dice_scores(prediction,target)))})
            score=float(np.mean([r['macro_dice'] for r in rows]))
            with (run/'validation.jsonl').open('a') as f: f.write(json.dumps({'step':step,'macro_dice':score,'cases':rows})+'\n')
            checkpoint={'state_dict':model.state_dict(),'seed':seed,'kind':'wide','context_stride':3,'step':step,'validation_dice':score,'config':cfg}
            if score>best:
                best=score; best_step=step; torch.save(checkpoint,run/'best.pt')
            torch.save(checkpoint,run/'last.pt')
            print('validation',seed,step,score,flush=True)
            model.train()
    (run/'complete.json').write_text(json.dumps({'training_steps':cfg['steps'],'best_step':best_step,
        'best_validation_macro_dice':best,'total_seconds':time.time()-started},indent=2))
    original_schedule=pd.read_json(ROOT/'results/runs'/f'2p5d_seed{seed}'/'train.jsonl',lines=True)
    schedule=pd.read_json(run/'train.jsonl',lines=True)
    assert list(zip(schedule.step,schedule.patient))==list(zip(original_schedule.step,original_schedule.patient))


def evaluate(kind,seed):
    run=OUT/f'{kind}_seed{seed}'; run.mkdir(parents=True,exist_ok=True)
    if (run/'evaluation_complete.json').exists(): return
    if kind=='wide': checkpoint_path=run/'best.pt'
    else: checkpoint_path=ROOT/'results/runs'/f'{"repeat" if kind=="repeat" else "2p5d"}_seed{seed}'/'best.pt'
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    init_runtime(seed); device=torch.device('cuda')
    model=UNet('repeat' if kind=='repeat' else '2p5d',checkpoint['config']['width']).to(device).eval()
    model.load_state_dict(checkpoint['state_dict'])
    ids=json.loads((META/'split.json').read_text())['test']
    rows=[]; jobs={}; started=time.time()
    with cf.ThreadPoolExecutor(6) as pool:
        for factor in [1,2,3]:
            for case in ids:
                image=np.load(PROC/case/f'image_f{factor}.npy')
                before=time.time()
                if kind=='wide': prob=predict_wide(model,image,device)
                else: prob=predict_volume(model,image,'repeat' if kind=='repeat' else '2p5d',device,32)
                meta=json.loads((PROC/case/'metadata.json').read_text())
                pred=restore_on_gpu(prob,meta,device)
                f=pool.submit(score_case,pred,case,factor,kind,seed,run,time.time()-before)
                jobs[f]=(case,factor)
                if len(jobs)>=12:
                    done,_=cf.wait(jobs,return_when=cf.FIRST_COMPLETED)
                    for f in done:
                        rows.extend(f.result()); jobs.pop(f)
                    print(kind,seed,'evaluated',len(rows)//4,'/ 120',flush=True)
        for f in cf.as_completed(jobs): rows.extend(f.result())
    frame=pd.DataFrame(rows).sort_values(['factor','case','organ'])
    assert len(frame)==480
    frame.to_csv(run/'metrics.csv',index=False)
    (run/'evaluation_complete.json').write_text(json.dumps({'kind':kind,'seed':seed,'cases':40,'rows':480,
        'checkpoint':str(checkpoint_path.relative_to(ROOT)),'checkpoint_step':checkpoint['step'],
        'source_sha256':source_hashes(),'elapsed_seconds':time.time()-started},indent=2))


def launch():
    OUT.mkdir(exist_ok=True)
    while len(list((ROOT/'results/runs').glob('*/evaluation_complete.json')))<15 or not (META/'complete.json').exists():
        print('Waiting for original study and fresh data',flush=True); time.sleep(20)
    check()
    jobs=queue.Queue()
    for seed in [11,22,33]: jobs.put(('wide',seed))
    for seed in [11,22,33]:
        jobs.put(('repeat',seed)); jobs.put(('narrow',seed))
    events=[]; lock=threading.Lock(); errors=[]
    def worker(gpu):
        env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']=str(gpu)
        while True:
            try: kind,seed=jobs.get_nowait()
            except queue.Empty: return
            try:
                with lock:
                    events.append({'status':'started','kind':kind,'seed':seed,'gpu':gpu,'unix':time.time()})
                    (OUT/'queue_status.json').write_text(json.dumps(events,indent=2))
                with (ROOT/'logs'/f'followup_{kind}_{seed}.log').open('a') as log:
                    for mode in (['train','evaluate'] if kind=='wide' else ['evaluate']):
                        cmd=[sys.executable,'-u',str(Path(__file__).resolve()),mode,'--kind',kind,'--seed',str(seed)]
                        subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                with lock:
                    events.append({'status':'complete','kind':kind,'seed':seed,'gpu':gpu,'unix':time.time()})
                    (OUT/'queue_status.json').write_text(json.dumps(events,indent=2))
                print('Follow-up complete',kind,seed,flush=True)
            except Exception as e:
                errors.append({'kind':kind,'seed':seed,'error':repr(e)})
                print('Follow-up failure',errors[-1],flush=True)
            finally: jobs.task_done()
    threads=[threading.Thread(target=worker,args=(g,)) for g in [0,1]]
    for t in threads:t.start()
    for t in threads:t.join()
    (OUT/'queue_complete.json').write_text(json.dumps({'errors':errors,'finish_unix':time.time()},indent=2))
    assert not errors


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['check','train','evaluate','launch'])
    parser.add_argument('--kind',choices=['repeat','narrow','wide'],default='wide')
    parser.add_argument('--seed',type=int,default=11)
    args=parser.parse_args()
    if args.mode=='check': check()
    elif args.mode=='train': train(args.seed)
    elif args.mode=='evaluate': evaluate(args.kind,args.seed)
    else: launch()
