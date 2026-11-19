"""Two-GPU queue with explicit task logs, completion markers, and a time budget."""
import argparse
import datetime as dt
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--gpus',default='0,1')
    ap.add_argument('--deadline',default='2026-11-14T22:00:00+00:00')
    args=ap.parse_args()
    cfg=json.loads((ROOT/'configs/experiment.json').read_text())
    deadline=dt.datetime.fromisoformat(args.deadline).timestamp()
    jobs=queue.Queue()
    kinds=['repeat','2p5d','3d','2d']
    for seed in cfg['seeds']:
        for kind in kinds: jobs.put((kind,seed))
    if cfg.get('include_augmentation',False):
        for seed in cfg['seeds']: jobs.put(('2p5d_aug',seed))
    lock=threading.Lock()
    events=[]
    errors=[]
    def record(row):
        with lock:
            row['utc']=dt.datetime.now(dt.timezone.utc).isoformat()
            events.append(row)
            (ROOT/'logs/study_status.json').write_text(json.dumps(events,indent=2))
            print(json.dumps(row),flush=True)
    def worker(gpu):
        env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']=str(gpu)
        while True:
            try: kind,seed=jobs.get_nowait()
            except queue.Empty: return
            run=ROOT/'results/runs'/f'{kind}_seed{seed}'
            if time.time()>deadline-300:
                record({'status':'deferred_time_budget','kind':kind,'seed':seed,'gpu':gpu})
                jobs.task_done(); continue
            try:
                if not (run/'complete.json').exists():
                    record({'status':'training','kind':kind,'seed':seed,'gpu':gpu})
                    cmd=[sys.executable,'-u',str(ROOT/'src/experiment.py'),'--kind',kind,'--seed',str(seed),'--device','cuda:0']
                    with (ROOT/'logs'/f'train_{kind}_{seed}.log').open('a') as f:
                        subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
                record({'status':'evaluating','kind':kind,'seed':seed,'gpu':gpu})
                cmd=[sys.executable,'-u',str(ROOT/'src/evaluate.py'),'--kind',kind,'--seed',str(seed),'--device','cuda:0']
                with (ROOT/'logs'/f'eval_{kind}_{seed}.log').open('a') as f:
                    subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
                record({'status':'complete','kind':kind,'seed':seed,'gpu':gpu})
            except Exception as e:
                errors.append({'kind':kind,'seed':seed,'error':repr(e)})
                record({'status':'error','kind':kind,'seed':seed,'gpu':gpu,'error':repr(e)})
            finally: jobs.task_done()
    threads=[threading.Thread(target=worker,args=(gpu,)) for gpu in args.gpus.split(',')]
    for t in threads: t.start()
    for t in threads: t.join()
    (ROOT/'logs/study_queue_complete.json').write_text(json.dumps({'events':len(events),'errors':errors,'finish_unix':time.time()},indent=2))
    if errors: raise RuntimeError(f'{len(errors)} run(s) failed; inspect per-run logs')


if __name__=='__main__': main()
