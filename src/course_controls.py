"""Evaluate the matched three-channel controls on the replication test set."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
import time

import numpy as np
import pandas as pd
import torch

from evaluate import score_case
from experiment import PROC, ROOT, init_runtime, predict_volume
from models import UNet
from native_resample import restore_on_gpu
from seed_stability_common import checkpoint_path, BASE, sha


def evaluate(seed, arm):
    output = ROOT/'results/course_controls'/f'{arm}_seed{seed}'
    output.mkdir(exist_ok=True,parents=True)
    checkpoint = checkpoint_path(arm,seed)
    digest = sha(checkpoint)
    marker = output/'complete.json'
    if marker.exists():
        record=json.loads(marker.read_text())
        if record['checkpoint_sha256']!=digest or sha(output/'metrics.csv')!=record['metrics_sha256']:
            raise ValueError(f'Stale evaluation at {output}')
        return
    init_runtime(seed)
    device=torch.device('cuda')
    model=UNet(BASE[arm],16).to(device).eval()
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['state_dict'])
    ids=json.loads((ROOT/'data/metadata/mechanism_replication/split.json').read_text())['test']
    rows=[]; pending=set()
    with ThreadPoolExecutor(5) as pool:
        for factor in [1,3]:
            for patient in ids:
                image=np.load(PROC/patient/f'image_f{factor}.npy')
                meta=json.loads((PROC/patient/'metadata.json').read_text())
                start=time.perf_counter()
                pred=restore_on_gpu(predict_volume(model,image,BASE[arm],device),meta,device)
                pending.add(pool.submit(score_case,pred,patient,factor,arm,seed,output,time.perf_counter()-start))
                if len(pending)>=8:
                    done,pending=wait(pending,return_when=FIRST_COMPLETED)
                    for f in done: rows.extend(f.result())
        for f in pending: rows.extend(f.result())
    table=pd.DataFrame(rows).sort_values(['case','factor','organ']).assign(
        selection='best',cohort='new40_replication',arm=arm)
    assert len(table)==320 and not table.duplicated(['case','factor','organ']).any()
    table.to_csv(output/'metrics.csv',index=False)
    marker.write_text(json.dumps(dict(checkpoint_sha256=digest,metrics_sha256=sha(output/'metrics.csv'))))


def gather():
    paths=[ROOT/'results/course_controls'/f'{arm}_seed{seed}' for arm in BASE for seed in [11,22,33,44,55,66]]
    for p in paths:
        assert sha(p/'metrics.csv')==json.loads((p/'complete.json').read_text())['metrics_sha256']
    dest=ROOT/'results/seed_stability/analysis/all_organ_metrics.csv'
    if dest.exists():
        old=pd.read_csv(dest)
        if set(old.cohort)!={'new40_replication'}:
            raise ValueError('Preserve the archived multi-cohort results. Use a fresh checkout for reproduction.')
    dest.parent.mkdir(exist_ok=True,parents=True)
    pd.concat([pd.read_csv(p/'metrics.csv') for p in paths]).to_csv(dest,index=False)
