"""Within-patient fixed-physical-resolution evaluation for the 21 thin-slice cases."""
import argparse
import concurrent.futures as cf
import json
import time

import numpy as np
import pandas as pd
import torch

from mechanism_common import ROOT, sha, sources
from experiment import PROC, init_runtime, predict_volume
from models import UNet
from evaluate import score_case
from evaluate_sampling_phase import interior_metrics
from native_resample import restore_on_gpu

OUT = ROOT/'results/mechanism/absolute_spacing'
META = ROOT/'data/metadata/absolute_spacing'
PROTOCOL = ROOT/'configs/absolute_spacing.json'


def score(pred, case, target_mm, arm, seed, dest, elapsed):
    meta = json.loads((PROC/case/'metadata.json').read_text())
    ratio = target_mm/meta['original_spacing_xyz_mm'][2]
    rows = score_case(pred, case, ratio, arm, seed, dest, elapsed)
    interior = interior_metrics(pred, case)
    for row in rows:
        row.update(interior[row['organ']])
        row['target_sampling_mm'] = target_mm
        row['prediction_file'] = str((dest/'predictions'/f'f{ratio}'/f'{case}.nii.gz').relative_to(ROOT))
        assert np.isclose(row['effective_z_sampling_mm'], target_mm)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=['repeat', 'context', 'repeat_aug', 'context_aug'], required=True)
    ap.add_argument('--seed', type=int, required=True)
    args = ap.parse_args()
    arm, seed = args.arm, args.seed
    run = OUT/'runs'/f'{arm}_seed{seed}'; run.mkdir(parents=True, exist_ok=True)
    if (run/'complete.json').exists(): return
    assert (META/'complete.json').exists()
    ids = json.loads((META/'split.json').read_text())['cases']
    path = ROOT/'results/continuation/b40000'/f'{arm}_seed{seed}'/'best.pt'
    digest = sha(path)
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    init_runtime(seed)
    device = torch.device('cuda')
    base = 'repeat' if arm.startswith('repeat') else '2p5d'
    model = UNet(base, 16).to(device).eval(); model.load_state_dict(checkpoint['state_dict'])
    code = sources(['evaluate_absolute_spacing.py', 'evaluate_sampling_phase.py', 'experiment.py',
                    'models.py', 'evaluate.py', 'native_resample.py', 'prepare_data.py'])
    protocol_hash, started, all_rows = sha(PROTOCOL), time.time(), []
    for target_mm in [6., 15.]:
        dest = run/f'target{int(target_mm)}mm'; dest.mkdir(parents=True, exist_ok=True)
        pending = {}
        with cf.ThreadPoolExecutor(6) as pool:
            for case in ids:
                image = np.load(ROOT/'data/processed_absolute_spacing'/case/f'image_z{int(target_mm)}mm.npy')
                meta = json.loads((PROC/case/'metadata.json').read_text())
                before = time.time()
                prob = predict_volume(model, image, base, device, 32)
                pred = restore_on_gpu(prob, meta, device)
                task = pool.submit(score, pred, case, target_mm, arm, seed, dest, time.time()-before)
                pending[task] = case
                if len(pending) >= 12:
                    ready, _ = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
                    for task in ready: all_rows.extend(task.result()); pending.pop(task)
            for task in cf.as_completed(pending): all_rows.extend(task.result())
        print('absolute spacing', target_mm, arm, seed, len(ids), 'patients', flush=True)
    frame = pd.DataFrame(all_rows).sort_values(['target_sampling_mm', 'case', 'organ'])
    frame['arm'], frame['budget'], frame['cohort'] = arm, 40000, 'prior_thin_native_spacing'
    assert len(frame) == len(ids)*2*4 and not frame.duplicated(['target_sampling_mm', 'case', 'organ']).any()
    frame.to_csv(run/'metrics.csv', index=False)
    assert sha(path) == digest and sources(list(code)) == code and sha(PROTOCOL) == protocol_hash
    (run/'complete.json').write_text(json.dumps({'arm': arm, 'seed': seed, 'patients': len(ids),
        'rows': len(frame), 'new_native_predictions': len(ids)*2,
        'checkpoint_sha256': digest, 'checkpoint_step': checkpoint['step'], 'source_sha256': code,
        'protocol_sha256': protocol_hash, 'split_sha256': sha(META/'split.json'),
        'elapsed_seconds': time.time()-started, 'finish_unix': time.time(), 'report_generated': False}, indent=2))


if __name__ == '__main__': main()
