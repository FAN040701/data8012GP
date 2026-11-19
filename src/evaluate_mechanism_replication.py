"""Frozen model evaluation on the prospectively selected additional 40 patients."""
import argparse
import concurrent.futures as cf
import json
import time

import numpy as np
import pandas as pd
import torch

from mechanism_common import ROOT, CFG, ComponentModel, run_dir, sha, sources
from experiment import PROC, init_runtime, predict_volume
from models import UNet
from evaluate import score_case
from native_resample import restore_on_gpu

OUT = ROOT/'results/mechanism/replication'
META = ROOT/'data/metadata/mechanism_replication'
PROTOCOL = ROOT/'configs/mechanism_replication.json'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=CFG['new_arms']+CFG['reused_arms'], required=True)
    ap.add_argument('--seed', type=int, required=True)
    args = ap.parse_args()
    arm, seed = args.arm, args.seed
    dest = OUT/'runs'/f'{arm}_seed{seed}'
    dest.mkdir(parents=True, exist_ok=True)
    if (dest/'complete.json').exists(): return
    assert (META/'complete.json').exists()
    specification = json.loads(PROTOCOL.read_text())
    split = json.loads((META/'split.json').read_text())
    ids = split['test']; assert len(ids) == 40
    is_component = arm in CFG['new_arms']
    path = run_dir(arm, seed)/'best.pt' if is_component else ROOT/'results/continuation/b40000'/f'{arm}_seed{seed}'/'best.pt'
    digest = sha(path)
    if not is_component: assert digest == split['primary_checkpoint_sha256'][str(path.relative_to(ROOT))]
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    init_runtime(seed)
    device = torch.device('cuda')
    base = 'repeat' if arm.startswith('repeat') else '2p5d'
    if is_component:
        model = ComponentModel(checkpoint['component'], CFG['width']).to(device).eval()
        model.network.load_state_dict(checkpoint['state_dict'])
    else:
        model = UNet(base, CFG['width']).to(device).eval()
        model.load_state_dict(checkpoint['state_dict'])
    code = sources(['evaluate_mechanism_replication.py', 'mechanism_common.py', 'experiment.py',
                    'models.py', 'evaluate.py', 'native_resample.py'])
    protocol_hash = sha(PROTOCOL)
    rows, pending, started = [], {}, time.time()
    with cf.ThreadPoolExecutor(6) as pool:
        for factor in specification['conditions']:
            for case in ids:
                image = np.load(PROC/case/f'image_f{factor}.npy')
                meta = json.loads((PROC/case/'metadata.json').read_text())
                before = time.time()
                prob = predict_volume(model, image, base, device, CFG['patch'][0])
                pred = restore_on_gpu(prob, meta, device)
                task = pool.submit(score_case, pred, case, factor, arm, seed, dest, time.time()-before)
                pending[task] = (case, factor)
                if len(pending) >= 12:
                    ready, _ = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
                    for task in ready:
                        rows.extend(task.result()); pending.pop(task)
                    print('replication', arm, seed, len(rows)//4, '/80', flush=True)
        for task in cf.as_completed(pending): rows.extend(task.result())
    frame = pd.DataFrame(rows).sort_values(['factor', 'case', 'organ'])
    frame['arm'], frame['budget'], frame['cohort'] = arm, 40000, 'new40_replication'
    frame['prediction_file'] = [str((dest/'predictions'/f'f{f}'/f'{c}.nii.gz').relative_to(ROOT))
                                for c, f in zip(frame.case, frame.factor)]
    assert len(frame) == 320 and not frame.duplicated(['case', 'factor', 'organ']).any()
    frame.to_csv(dest/'metrics.csv', index=False)
    assert sources(list(code)) == code and sha(path) == digest and sha(PROTOCOL) == protocol_hash
    (dest/'complete.json').write_text(json.dumps({'arm': arm, 'seed': seed, 'patients': 40,
        'rows': len(frame), 'native_predictions': len(frame)//4, 'conditions': specification['conditions'],
        'checkpoint': str(path.relative_to(ROOT)), 'checkpoint_sha256': digest,
        'checkpoint_step': checkpoint['step'], 'source_sha256': code,
        'protocol_sha256': protocol_hash, 'split_sha256': sha(META/'split.json'),
        'elapsed_seconds': time.time()-started, 'finish_unix': time.time(), 'report_generated': False}, indent=2))
    print('Replication complete', arm, seed, flush=True)


if __name__ == '__main__': main()
