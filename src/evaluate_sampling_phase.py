"""Controlled native-slice sampling-phase diagnostic with frozen existing models."""
import argparse
import concurrent.futures as cf
import json
import time

import nibabel as nib
import numpy as np
import pandas as pd
import torch

from mechanism_common import ROOT, sha, sources
from experiment import PROC, init_runtime, predict_volume
from models import UNet
from evaluate import score_case
from native_resample import restore_on_gpu
from prepare_data import remap_labels, NAMES

OUT = ROOT/'results/mechanism/sampling_phase'
META = ROOT/'data/metadata/sampling_phase'
PROTOCOL = ROOT/'configs/sampling_phase.json'


def interior_metrics(pred, case):
    meta = json.loads((PROC/case/'metadata.json').read_text())
    target = remap_labels(np.asarray(nib.as_closest_canonical(nib.load(ROOT/meta['raw_label'])).dataobj,
                                    dtype=np.uint8))
    z_mm = np.arange(target.shape[2])*meta['original_spacing_xyz_mm'][2]
    kept = (z_mm >= 9) & (z_mm <= z_mm[-1]-9)
    p, t = pred[:, :, kept], target[:, :, kept]
    result = {}
    for k, name in enumerate(NAMES[1:], 1):
        npred, ntrue = int(np.count_nonzero(p == k)), int(np.count_nonzero(t == k))
        tp = int(np.count_nonzero((p == k) & (t == k)))
        result[name] = {'interior_dice': 2*tp/(npred+ntrue) if ntrue else np.nan,
                        'interior_reference_voxels': ntrue, 'interior_prediction_voxels': npred,
                        'interior_retained_z_planes': int(kept.sum())}
    return result


def score(pred, case, phase, arm, seed, dest, elapsed):
    rows = score_case(pred, case, 3, arm, seed, dest, elapsed)
    interior = interior_metrics(pred, case)
    for row in rows:
        row.update(interior[row['organ']])
        row['phase'] = phase
        row['prediction_file'] = str((dest/'predictions/f3'/f'{case}.nii.gz').relative_to(ROOT))
    return rows


def reuse_phase_zero(row):
    pred = np.asarray(nib.load(ROOT/row['prediction_file']).dataobj, dtype=np.uint8)
    return row['case'], interior_metrics(pred, row['case'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=['repeat', 'context', 'repeat_aug', 'context_aug'], required=True)
    ap.add_argument('--seed', type=int, required=True)
    args = ap.parse_args()
    arm, seed = args.arm, args.seed
    run = OUT/'runs'/f'{arm}_seed{seed}'; run.mkdir(parents=True, exist_ok=True)
    if (run/'complete.json').exists(): return
    assert (META/'complete.json').exists()
    ids = json.loads((ROOT/'data/metadata/continuation/split.json').read_text())['test']
    path = ROOT/'results/continuation/b40000'/f'{arm}_seed{seed}'/'best.pt'
    digest = sha(path)
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    init_runtime(seed)
    device = torch.device('cuda')
    base = 'repeat' if arm.startswith('repeat') else '2p5d'
    model = UNet(base, 16).to(device).eval()
    model.load_state_dict(checkpoint['state_dict'])
    code = sources(['evaluate_sampling_phase.py', 'experiment.py', 'models.py',
                    'evaluate.py', 'native_resample.py', 'prepare_data.py'])
    protocol_hash, started = sha(PROTOCOL), time.time()
    oldfile = ROOT/'results/continuation/analysis/all_organ_metrics.csv'
    old = pd.read_csv(oldfile)
    baseline = old[(old.cohort == 'fresh20') & (old.budget == 40000) & (old.arm == arm) &
                   (old.seed == seed) & (old.factor == 3)].copy()
    baseline['phase'] = 0
    assert len(baseline) == 80
    with cf.ThreadPoolExecutor(4) as pool:
        interiors = dict(pool.map(reuse_phase_zero, baseline.drop_duplicates('case').to_dict('records')))
    for field in ['interior_dice', 'interior_reference_voxels', 'interior_prediction_voxels', 'interior_retained_z_planes']:
        baseline[field] = [interiors[c][o][field] for c, o in zip(baseline.case, baseline.organ)]
    frames = [baseline]
    for phase in [1, 2]:
        dest = run/f'phase{phase}'; dest.mkdir(parents=True, exist_ok=True)
        rows, pending = [], {}
        with cf.ThreadPoolExecutor(6) as pool:
            for case in ids:
                image = np.load(ROOT/'data/processed_phase'/case/f'image_f3_p{phase}.npy')
                meta = json.loads((PROC/case/'metadata.json').read_text())
                before = time.time()
                prob = predict_volume(model, image, base, device, 32)
                pred = restore_on_gpu(prob, meta, device)
                task = pool.submit(score, pred, case, phase, arm, seed, dest, time.time()-before)
                pending[task] = case
                if len(pending) >= 12:
                    ready, _ = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
                    for task in ready: rows.extend(task.result()); pending.pop(task)
            for task in cf.as_completed(pending): rows.extend(task.result())
        frame = pd.DataFrame(rows)
        frame['arm'], frame['budget'], frame['cohort'] = arm, 40000, 'fresh20'
        frames.append(frame)
        print('phase', phase, arm, seed, len(rows)//4, 'patients', flush=True)
    combined = pd.concat(frames, ignore_index=True).sort_values(['phase', 'case', 'organ'])
    assert len(combined) == 240 and not combined.duplicated(['phase', 'case', 'organ']).any()
    combined.to_csv(run/'metrics.csv', index=False)
    assert sha(path) == digest and sources(list(code)) == code and sha(PROTOCOL) == protocol_hash
    (run/'complete.json').write_text(json.dumps({'arm': arm, 'seed': seed, 'patients': 20,
        'rows': len(combined), 'new_native_predictions': 40, 'reused_native_predictions': 20,
        'checkpoint_sha256': digest, 'checkpoint_step': checkpoint['step'],
        'source_sha256': code, 'protocol_sha256': protocol_hash, 'prior_metrics_sha256': sha(oldfile),
        'elapsed_seconds': time.time()-started, 'finish_unix': time.time(), 'report_generated': False}, indent=2))


if __name__ == '__main__': main()
