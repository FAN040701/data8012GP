"""New seeds: two unchanged cohorts and controlled spacing; all seeds: last checkpoint."""
import argparse
import concurrent.futures as cf
import json
import time

import numpy as np
import pandas as pd
import torch

from seed_stability_common import ROOT, OUT, CFG, BASE, PROTOCOL, sha, source_hashes, checkpoint_path
from experiment import PROC, init_runtime, predict_volume
from models import UNet
from evaluate import score_case
from evaluate_absolute_spacing import score as score_spacing
from native_resample import restore_on_gpu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=BASE, required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--selection', choices=['best', 'last'], required=True)
    args = ap.parse_args()
    arm, seed, selection = args.arm, args.seed, args.selection
    assert seed in CFG['old_seeds'] + CFG['new_seeds']
    assert selection == 'last' or seed in CFG['new_seeds']
    dest = OUT / 'evaluations' / selection / f'{arm}_seed{seed}'
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'complete.json').exists():
        return
    path = checkpoint_path(arm, seed, selection)
    digest, protocol_hash = sha(path), sha(PROTOCOL)
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if selection == 'last':
        assert checkpoint['step'] == 40000
    init_runtime(seed)
    device = torch.device('cuda')
    base = BASE[arm]
    model = UNet(base, CFG['width']).to(device).eval()
    model.load_state_dict(checkpoint['state_dict'])
    code = source_hashes(['evaluate_seed_stability.py', 'seed_stability_common.py', 'evaluate_absolute_spacing.py', 'evaluate_sampling_phase.py', 'experiment.py', 'models.py', 'evaluate.py', 'native_resample.py', 'prepare_data.py'])
    groups = [('new40_replication', CFG['new40_ids'], [1, 3]), ('prior_fresh20', CFG['prior20_ids'], [1, 3]), ('absolute_spacing', CFG['spacing_ids'], [6, 15])]
    if selection == 'last':
        groups = [('new40_replication', CFG['new40_ids'], [3])]
    frames, started = [], time.time()
    for cohort, ids, conditions in groups:
        rows, pending = [], {}
        evaldir = dest / cohort
        with cf.ThreadPoolExecutor(6) as pool:
            for condition in conditions:
                for case in ids:
                    absolute = cohort == 'absolute_spacing'
                    image_path = ROOT / 'data/processed_absolute_spacing' / case / f'image_z{condition}mm.npy' if absolute else PROC / case / f'image_f{condition}.npy'
                    image = np.load(image_path)
                    meta = json.loads((PROC / case / 'metadata.json').read_text())
                    before = time.time()
                    prob = predict_volume(model, image, base, device, CFG['patch'][0])
                    pred = restore_on_gpu(prob, meta, device)
                    elapsed = time.time() - before
                    if absolute:
                        task = pool.submit(score_spacing, pred, case, float(condition), arm, seed, evaldir / f'target{condition}mm', elapsed)
                    else:
                        task = pool.submit(score_case, pred, case, condition, arm, seed, evaldir, elapsed)
                    pending[task] = case
                    if len(pending) >= 12:
                        ready, _ = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
                        for task in ready:
                            rows.extend(task.result())
                            pending.pop(task)
            for task in cf.as_completed(pending):
                rows.extend(task.result())
        frame = pd.DataFrame(rows)
        if cohort != 'absolute_spacing':
            frame['prediction_file'] = [str((evaldir / 'predictions' / f'f{factor}' / f'{case}.nii.gz').relative_to(ROOT)) for factor, case in zip(frame.factor, frame.case)]
        frame['cohort'], frame['arm'], frame['selection'], frame['budget'] = cohort, arm, selection, 40000
        frame = frame.sort_values(['factor', 'case', 'organ'])
        assert len(frame) == len(ids) * len(conditions) * 4
        assert not frame.duplicated(['case', 'factor', 'organ']).any()
        evaldir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(evaldir / 'metrics.csv', index=False)
        frames.append(frame)
        print(selection, arm, seed, cohort, len(frame)//4, 'predictions complete', flush=True)
    raw = pd.concat(frames, ignore_index=True)
    raw.to_csv(dest / 'metrics.csv', index=False)
    assert sha(path) == digest and sha(PROTOCOL) == protocol_hash and source_hashes(list(code)) == code
    (dest / 'complete.json').write_text(json.dumps({'arm': arm, 'seed': seed, 'selection': selection, 'native_predictions': len(raw)//4, 'rows': len(raw), 'checkpoint': str(path.relative_to(ROOT)), 'checkpoint_sha256': digest, 'checkpoint_step': checkpoint['step'], 'protocol_sha256': protocol_hash, 'source_sha256': code, 'elapsed_seconds': time.time()-started, 'finish_unix': time.time(), 'report_generated': False}, indent=2))


if __name__ == '__main__':
    main()
