"""Original-acquisition-grid metrics for frozen component models."""
import argparse
import concurrent.futures as cf
import json
import time

import numpy as np
import pandas as pd
import torch

from mechanism_common import ROOT, CFG, ComponentModel, run_dir, sha, sources, cohorts
from experiment import PROC, init_runtime, predict_volume
from evaluate import score_case
from native_resample import restore_on_gpu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=CFG['new_arms'], required=True)
    ap.add_argument('--seed', type=int, required=True)
    args = ap.parse_args()
    arm, seed = args.arm, args.seed
    run = run_dir(arm, seed)
    if (run/'evaluation_complete.json').exists(): return
    assert (run/'complete.json').exists()
    checkpoint = torch.load(run/'best.pt', map_location='cpu', weights_only=False)
    checkpoint_hash = sha(run/'best.pt')
    init_runtime(seed)
    device = torch.device('cuda')
    model = ComponentModel(checkpoint['component'], CFG['width']).to(device).eval()
    model.network.load_state_dict(checkpoint['state_dict'])
    code = sources(['evaluate_mechanism.py', 'mechanism_common.py', 'experiment.py',
                    'models.py', 'evaluate.py', 'native_resample.py'])
    protocol = sha(ROOT/'configs/mechanism.json')
    started, frames = time.time(), []
    for cohort, ids in cohorts().items():
        dest = run/'evaluations'/cohort
        dest.mkdir(parents=True, exist_ok=True)
        rows, pending = [], {}
        with cf.ThreadPoolExecutor(6) as pool:
            for factor in CFG['evaluation_conditions'][cohort]:
                for case in ids:
                    meta = json.loads((PROC/case/'metadata.json').read_text())
                    image = np.load(PROC/case/f'image_f{factor}.npy')
                    before = time.time()
                    prob = predict_volume(model, image, '2p5d', device, CFG['patch'][0])
                    pred = restore_on_gpu(prob, meta, device)
                    task = pool.submit(score_case, pred, case, factor, arm, seed, dest, time.time()-before)
                    pending[task] = (case, factor)
                    if len(pending) >= 12:
                        ready, _ = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
                        for task in ready:
                            rows.extend(task.result()); pending.pop(task)
                        print(arm, seed, cohort, 'predictions', len(rows)//4, flush=True)
            for task in cf.as_completed(pending): rows.extend(task.result())
        frame = pd.DataFrame(rows)
        frame['arm'], frame['budget'], frame['cohort'] = arm, CFG['steps'], cohort
        frame['prediction_file'] = [str((dest/'predictions'/f'f{f}'/f'{c}.nii.gz').relative_to(ROOT))
                                    for c, f in zip(frame.case, frame.factor)]
        frame = frame.sort_values(['factor', 'case', 'organ'])
        assert len(frame) == len(ids)*len(CFG['evaluation_conditions'][cohort])*4
        assert not frame.duplicated(['case', 'factor', 'organ']).any()
        frame.to_csv(dest/'metrics.csv', index=False)
        frames.append(frame)
    all_rows = pd.concat(frames, ignore_index=True)
    all_rows.to_csv(run/'metrics.csv', index=False)
    assert sources(list(code)) == code and sha(run/'best.pt') == checkpoint_hash
    assert sha(ROOT/'configs/mechanism.json') == protocol
    complete = {'arm': arm, 'seed': seed, 'rows': len(all_rows), 'native_predictions': len(all_rows)//4,
                'checkpoint_step': checkpoint['step'], 'checkpoint_sha256': checkpoint_hash,
                'source_sha256': code, 'protocol_sha256': protocol,
                'elapsed_seconds': time.time()-started, 'finish_unix': time.time(),
                'all_results_exploratory': True, 'report_generated': False}
    (run/'evaluation_complete.json').write_text(json.dumps(complete, indent=2))
    print('Evaluation complete', arm, seed, flush=True)


if __name__ == '__main__': main()
