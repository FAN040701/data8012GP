"""Evaluate a development-selected offset model on the paired replication cases."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
import time

import numpy as np
import pandas as pd
import torch

from evaluate import score_case
from experiment import PROC, ROOT, init_runtime
from mechanism_common import sha
from models import UNet
from native_resample import restore_on_gpu
from slice_selection import ARMS, CFG, PROTOCOL, predict, run_dir, offsets


def evaluate(arm, seed, cohort='replication', checkpoint='best'):
    training = run_dir(arm, seed)
    run = training if (cohort, checkpoint) == ('replication', 'best') else (
        ROOT / 'results/final_study' / cohort / checkpoint / training.name)
    run.mkdir(parents=True, exist_ok=True)
    marker = run/'evaluation_complete.json'
    if marker.exists():
        return
    record = json.loads((training/'complete.json').read_text())
    model_path = training / f'{checkpoint}.pt'
    if checkpoint == 'best':
        assert sha(model_path) == record['best_checkpoint_sha256']
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    assert ckpt['offsets'] == offsets(arm) and ckpt['step'] == record['best_step' if checkpoint == 'best' else 'training_steps']
    init_runtime(seed)
    device = torch.device('cuda')
    model = UNet('2p5d',16).to(device).eval()
    model.load_state_dict(ckpt['state_dict'])
    ids = json.loads((ROOT/CFG['test_split']).read_text())['test']
    if cohort == 'original':
        ids = json.loads((ROOT/'data/metadata/split.json').read_text())['splits']['test']
    pending, rows = set(), []
    with ThreadPoolExecutor(5) as pool:
        for factor in CFG['factors']:
            for case in ids:
                image = np.load(PROC/case/f'image_f{factor}.npy')
                meta = json.loads((PROC/case/'metadata.json').read_text())
                start = time.perf_counter()
                prob = predict(model, image, device, arm)
                labels = restore_on_gpu(prob, meta, device)
                pending.add(pool.submit(score_case, labels, case, factor, arm, seed, run,
                                        time.perf_counter()-start))
                if len(pending)>=8:
                    done, pending=wait(pending,return_when=FIRST_COMPLETED)
                    for future in done:
                        rows.extend(future.result())
        for future in pending:
            rows.extend(future.result())
    frame = pd.DataFrame(rows).sort_values(['factor','case','organ'])
    assert len(ids)==40 and len(frame)==320 and not frame.duplicated(['factor','case','organ']).any()
    frame.to_csv(run/'metrics.csv',index=False)
    marker.write_text(json.dumps(dict(arm=arm,seed=seed,cohort=cohort,checkpoint=checkpoint,checkpoint_sha256=sha(model_path),
        checkpoint_step=ckpt['step'],protocol_sha256=sha(PROTOCOL),predictions=80,rows=len(frame),
        metrics_sha256=sha(run/'metrics.csv')),indent=2))
    print('Completed',arm,seed,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--arm',required=True,choices=ARMS)
    ap.add_argument('--seed',required=True,type=int,choices=[11,22,33,44,55,66])
    ap.add_argument('--cohort',choices=['replication','original'],default='replication')
    ap.add_argument('--checkpoint',choices=['best','last'],default='best')
    args=ap.parse_args()
    evaluate(args.arm,args.seed,args.cohort,args.checkpoint)
