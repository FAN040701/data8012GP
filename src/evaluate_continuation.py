"""Frozen native-space evaluation, keeping fresh and previously seen cohorts separate."""
import argparse
import concurrent.futures as cf
import json
import time

import numpy as np
import pandas as pd
import torch

from continuation_common import ROOT, OUT, META, CFG, BASE, OLD, run_dir, checkpoint_path, sha, source_hashes
from experiment import PROC, init_runtime, predict_volume
from models import UNet
from evaluate import score_case
from native_resample import restore_on_gpu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=BASE, required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--steps', type=int, required=True)
    args = ap.parse_args()
    arm, seed, steps = args.arm, args.seed, args.steps
    run = run_dir(arm, seed, steps)
    run.mkdir(parents=True, exist_ok=True)
    if (run/'evaluation_complete.json').exists(): return
    assert (META/'complete.json').exists()
    fresh = json.loads((META/'split.json').read_text())['test']
    original = json.loads((ROOT/'data/metadata/split.json').read_text())['splits']
    previous_followup = json.loads((ROOT/'data/metadata/followup_context/split.json').read_text())['test']
    assert len(fresh) == 20 and not set(fresh) & set(sum(original.values(), [])+previous_followup)
    ckpt_path = checkpoint_path(arm, seed, steps)
    ckpt_digest = sha(ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    init_runtime(seed)
    device = torch.device('cuda')
    model = UNet(BASE[arm], CFG['width']).to(device).eval()
    model.load_state_dict(checkpoint['state_dict'])
    sources = source_hashes(['evaluate_continuation.py', 'continuation_common.py',
                             'experiment.py', 'models.py', 'evaluate.py', 'native_resample.py'])
    started = time.time()
    all_frames = []
    reused_metric_hashes = {}
    for cohort, ids in [('fresh20', fresh), ('original40_exploratory', original['test'])]:
        evaldir = run/'evaluations'/cohort
        evaldir.mkdir(parents=True, exist_ok=True)
        factors = CFG['evaluation_conditions'][cohort]
        reuse = steps == CFG['short_steps'] and arm in OLD and cohort == 'original40_exploratory'
        if reuse:
            oldrun = ROOT/'results/runs'/f'{OLD[arm]}_seed{seed}'
            reused_metric_hashes[str((oldrun/'metrics.csv').relative_to(ROOT))] = sha(oldrun/'metrics.csv')
            frame = pd.read_csv(oldrun/'metrics.csv')
            frame = frame[frame.factor.isin(factors)].copy()
            frame['kind'] = arm
            frame['prediction_file'] = [str((oldrun/'predictions'/f'f{f}'/f'{c}.nii.gz').relative_to(ROOT))
                                        for c, f in zip(frame.case, frame.factor)]
            frame['metric_source'] = 'reused_original_completed_evaluation'
        else:
            rows, jobs = [], {}
            with cf.ThreadPoolExecutor(6) as pool:
                for factor in factors:
                    for case in ids:
                        image = np.load(PROC/case/f'image_f{factor}.npy')
                        meta = json.loads((PROC/case/'metadata.json').read_text())
                        before = time.time()
                        prob = predict_volume(model, image, BASE[arm], device, CFG['patch'][0])
                        pred = restore_on_gpu(prob, meta, device)
                        future = pool.submit(score_case, pred, case, factor, arm, seed,
                                             evaldir, time.time()-before)
                        jobs[future] = (case, factor)
                        if len(jobs) >= 12:
                            ready, _ = cf.wait(jobs, return_when=cf.FIRST_COMPLETED)
                            for future in ready:
                                rows.extend(future.result())
                                jobs.pop(future)
                            print(arm, seed, steps, cohort, 'completed', len(rows)//4,
                                  '/', len(ids)*len(factors), flush=True)
                for future in cf.as_completed(jobs): rows.extend(future.result())
            frame = pd.DataFrame(rows)
            frame['prediction_file'] = [str((evaldir/'predictions'/f'f{f}'/f'{c}.nii.gz').relative_to(ROOT))
                                        for c, f in zip(frame.case, frame.factor)]
            frame['metric_source'] = 'new_native_grid_evaluation'
        frame['arm'] = arm
        frame['budget'] = steps
        frame['cohort'] = cohort
        frame = frame.sort_values(['factor', 'case', 'organ'])
        assert len(frame) == len(ids)*len(factors)*4
        assert set(frame.case) == set(ids) and not frame.duplicated(['case', 'factor', 'organ']).any()
        assert all((ROOT/p).exists() for p in frame.prediction_file.unique())
        frame.to_csv(evaldir/'metrics.csv', index=False)
        all_frames.append(frame)
    combined = pd.concat(all_frames, ignore_index=True)
    combined.to_csv(run/'metrics.csv', index=False)
    assert sha(ckpt_path) == ckpt_digest and source_hashes(list(sources)) == sources
    assert all(sha(ROOT/path) == digest for path, digest in reused_metric_hashes.items())
    complete = {'arm': arm, 'seed': seed, 'budget': steps, 'rows': len(combined),
                'cohort_patients': {'fresh20': 20, 'original40_exploratory': 40},
                'conditions': CFG['evaluation_conditions'],
                'checkpoint': str(ckpt_path.relative_to(ROOT)), 'checkpoint_sha256': ckpt_digest,
                'checkpoint_step': checkpoint['step'],
                'reused_metric_sha256': reused_metric_hashes,
                'source_sha256': sources, 'protocol_sha256': sha(ROOT/'configs/continuation.json'),
                'elapsed_seconds': time.time()-started, 'finish_unix': time.time(),
                'no_test_checkpoint_selection': True}
    (run/'evaluation_complete.json').write_text(json.dumps(complete, indent=2))
    print('Evaluation complete', arm, seed, steps, len(combined), flush=True)


if __name__ == '__main__': main()
