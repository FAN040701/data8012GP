"""Evaluate wider 40k models with matched narrow controls and separate cohorts."""
import argparse
import concurrent.futures as cf
import json
import time

import numpy as np
import pandas as pd
import torch

from mechanism_common import ROOT, sha, sources
from width_budget_common import OUT, CFG, run_dir
from experiment import PROC, init_runtime, predict_volume
from context_followup import predict_wide
from models import UNet
from evaluate import score_case
from native_resample import restore_on_gpu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=['wide', 'wide_aug', 'context', 'context_aug'], required=True)
    ap.add_argument('--seed', type=int, required=True)
    args = ap.parse_args(); arm, seed = args.arm, args.seed
    run = run_dir(arm, seed); run.mkdir(parents=True, exist_ok=True)
    if (run/'evaluation_complete.json').exists(): return
    wide = arm.startswith('wide')
    path = run/'best.pt' if wide else ROOT/'results/continuation/b40000'/f'{arm}_seed{seed}'/'best.pt'
    checkpoint_hash = sha(path)
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    init_runtime(seed); device = torch.device('cuda')
    model = UNet('2p5d', 16).to(device).eval(); model.load_state_dict(checkpoint['state_dict'])
    cohorts = {'previous_width40': json.loads((ROOT/'data/metadata/followup_context/split.json').read_text())['test'],
               'new40_replication': json.loads((ROOT/'data/metadata/mechanism_replication/split.json').read_text())['test']}
    code = sources(['evaluate_width_budget.py', 'width_budget_common.py', 'context_followup.py',
                    'experiment.py', 'models.py', 'evaluate.py', 'native_resample.py'])
    protocol = sha(ROOT/'configs/width_budget.json'); started = time.time()
    frames, reused = [], {}
    for cohort, ids in cohorts.items():
        if cohort == 'new40_replication' and not wide:
            source = ROOT/'results/mechanism/replication/runs'/f'{arm}_seed{seed}'/'metrics.csv'
            frame = pd.read_csv(source); reused[str(source.relative_to(ROOT))] = sha(source)
        else:
            dest = run/'evaluations'/cohort; dest.mkdir(parents=True, exist_ok=True)
            rows, pending = [], {}
            with cf.ThreadPoolExecutor(6) as pool:
                for factor in [1, 3]:
                    for case in ids:
                        image = np.load(PROC/case/f'image_f{factor}.npy')
                        meta = json.loads((PROC/case/'metadata.json').read_text())
                        before = time.time()
                        prob = predict_wide(model, image, device) if wide else predict_volume(model, image, '2p5d', device, 32)
                        pred = restore_on_gpu(prob, meta, device)
                        task = pool.submit(score_case, pred, case, factor, arm, seed, dest, time.time()-before)
                        pending[task] = (case, factor)
                        if len(pending) >= 12:
                            ready, _ = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
                            for task in ready: rows.extend(task.result()); pending.pop(task)
                            print('width evaluation', arm, seed, cohort, len(rows)//4, '/80', flush=True)
                for task in cf.as_completed(pending): rows.extend(task.result())
            frame = pd.DataFrame(rows)
            frame['prediction_file'] = [str((dest/'predictions'/f'f{f}'/f'{c}.nii.gz').relative_to(ROOT))
                                        for c, f in zip(frame.case, frame.factor)]
        frame['arm'], frame['budget'], frame['cohort'] = arm, 40000, cohort
        assert len(frame) == 320 and set(frame.case) == set(ids)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True).sort_values(['cohort', 'factor', 'case', 'organ'])
    assert len(combined) == 640 and not combined.duplicated(['cohort', 'case', 'factor', 'organ']).any()
    combined.to_csv(run/'metrics.csv', index=False)
    assert sources(list(code)) == code and sha(path) == checkpoint_hash
    assert sha(ROOT/'configs/width_budget.json') == protocol
    for p, digest in reused.items(): assert sha(ROOT/p) == digest
    (run/'evaluation_complete.json').write_text(json.dumps({'arm': arm, 'seed': seed,
        'rows': len(combined), 'new_native_predictions': 160 if wide else 80,
        'reused_native_predictions': 0 if wide else 80, 'source_sha256': code,
        'protocol_sha256': protocol, 'checkpoint_sha256': checkpoint_hash,
        'checkpoint': str(path.relative_to(ROOT)), 'checkpoint_step': checkpoint['step'],
        'reused_metrics_sha256': reused, 'elapsed_seconds': time.time()-started,
        'finish_unix': time.time(), 'report_generated': False}, indent=2))


if __name__ == '__main__': main()
