"""Fixed-checkpoint context sensitivity on the previously evaluated cohort only."""
import argparse
import concurrent.futures as cf
import json
import math
import os
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from continuation_common import ROOT, OUT, CFG, BASE, run_dir, checkpoint_path, sha, source_hashes
from experiment import PROC, init_runtime, predict_volume
from models import UNet
from native_resample import restore_on_gpu
from evaluate import score_case
from analyze_continuation import effect_stats

DEST = OUT/'context_interventions'
SPEC = json.loads((ROOT/'configs/context_interventions.json').read_text())


@torch.inference_mode()
def predict(model, image, device, mode, batch=24):
    if mode == 'repeat_centre': return predict_volume(model, image, 'repeat', device, 32, batch)
    before = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False
    d, h, w = image.shape
    hh, ww = math.ceil(h/16)*16, math.ceil(w/16)*16
    x = torch.as_tensor(np.array(image, dtype=np.float32, copy=True), device=device)
    x = F.pad(x, (0, ww-w, 0, hh-h))
    out = torch.empty((5, d, hh, ww), device=device, dtype=torch.float32)
    for begin in range(0, d, batch):
        stop = min(d, begin+batch)
        indices = torch.arange(begin, stop, device=device)
        previous, following = x[(indices-1).clamp(0, d-1)].clone(), x[(indices+1).clamp(0, d-1)].clone()
        previous[indices == 0] = 0
        following[indices == d-1] = 0
        inp = torch.stack([previous, x[begin:stop], following], 1)
        if mode == 'swap_neighbours': inp = inp[:, [2, 1, 0]]
        elif mode != 'identity': raise ValueError(mode)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16): logits = model(inp)
        out[:, begin:stop] = logits.float().softmax(1).permute(1, 0, 2, 3)
    result = out[:, :, :h, :w].cpu().numpy()
    torch.backends.cudnn.benchmark = before
    return result


def check():
    DEST.mkdir(parents=True, exist_ok=True)
    split = json.loads((ROOT/'data/metadata/split.json').read_text())['splits']
    case = split['train'][0]
    image = np.load(PROC/case/'image_f1.npy')
    init_runtime(11)
    model = UNet('2p5d', 16).cuda().eval()
    reference = predict_volume(model, image, '2p5d', torch.device('cuda'))
    same = predict(model, image, torch.device('cuda'), 'identity')
    error = float(np.abs(reference-same).max())
    assert error == 0
    x = torch.randn(4, 3, 16, 16)
    swapped = x[:, [2, 1, 0]]
    assert torch.equal(x[:, 1], swapped[:, 1]) and torch.equal(x, swapped[:, [2, 1, 0]])
    repeated = x[:, 1:2].expand(-1, 3, -1, -1)
    assert torch.equal(repeated, repeated[:, [2, 1, 0]])
    (DEST/'pipeline_checks.json').write_text(json.dumps({'fitting_case': case,
        'identity_probability_max_error': error, 'central_channel_unchanged': True,
        'swap_is_involution': True, 'repeated_input_swap_negative_control': True,
        'fresh_test_patients_unused': True}, indent=2))


def evaluate(arm, seed):
    run = DEST/f'{arm}_seed{seed}'
    run.mkdir(parents=True, exist_ok=True)
    if (run/'complete.json').exists(): return
    ids = json.loads((ROOT/'data/metadata/split.json').read_text())['splits']['test']
    path = checkpoint_path(arm, seed, 40000)
    digest = sha(path)
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    init_runtime(seed)
    device = torch.device('cuda')
    model = UNet(BASE[arm], 16).to(device).eval()
    model.load_state_dict(ckpt['state_dict'])
    frames = []
    started = time.time()
    for mode in SPEC['modes']:
        mode_dir = run/mode
        rows, jobs = [], {}
        with cf.ThreadPoolExecutor(6) as pool:
            for factor in SPEC['factors']:
                for case in ids:
                    image = np.load(PROC/case/f'image_f{factor}.npy')
                    meta = json.loads((PROC/case/'metadata.json').read_text())
                    start = time.time()
                    probability = predict(model, image, device, mode)
                    labels = restore_on_gpu(probability, meta, device)
                    future = pool.submit(score_case, labels, case, factor, arm, seed,
                                         mode_dir, time.time()-start)
                    jobs[future] = (case, factor)
                    if len(jobs) >= 12:
                        ready, _ = cf.wait(jobs, return_when=cf.FIRST_COMPLETED)
                        for future in ready: rows.extend(future.result()); jobs.pop(future)
            for future in cf.as_completed(jobs): rows.extend(future.result())
        frame = pd.DataFrame(rows).sort_values(['case', 'factor', 'organ'])
        frame['mode'] = mode
        frame['prediction_file'] = [str((mode_dir/'predictions'/f'f{f}'/f'{c}.nii.gz').relative_to(ROOT))
                                    for c, f in zip(frame.case, frame.factor)]
        assert len(frame) == 320 and set(frame.case) == set(ids)
        frame.to_csv(mode_dir/'metrics.csv', index=False)
        frames.append(frame)
    pd.concat(frames, ignore_index=True).to_csv(run/'metrics.csv', index=False)
    assert sha(path) == digest
    (run/'complete.json').write_text(json.dumps({'arm': arm, 'seed': seed, 'rows': 640,
        'checkpoint': str(path.relative_to(ROOT)), 'checkpoint_sha256': digest,
        'protocol_sha256': sha(ROOT/'configs/context_interventions.json'),
        'source_sha256': source_hashes(['intervene_context.py', 'analyze_continuation.py', 'continuation_common.py', 'evaluate.py', 'native_resample.py', 'experiment.py', 'models.py']),
        'elapsed_seconds': time.time()-started, 'finish_unix': time.time()}, indent=2))


def analyze():
    frames, differences = [], []
    for arm in SPEC['arms']:
        for seed in SPEC['seeds']:
            run = DEST/f'{arm}_seed{seed}'
            complete = json.loads((run/'complete.json').read_text())
            for name, digest in complete['source_sha256'].items(): assert sha(ROOT/'src'/name) == digest
            assert sha(ROOT/complete['checkpoint']) == complete['checkpoint_sha256']
            frame = pd.read_csv(run/'metrics.csv')
            intact = pd.read_csv(run_dir(arm, seed, 40000)/'evaluations/original40_exploratory/metrics.csv')
            intact['mode'] = 'intact'
            frame = pd.concat([frame, intact], ignore_index=True)
            frame['arm'] = arm
            frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    patient = raw.groupby(['arm', 'seed', 'case', 'factor', 'mode'], as_index=False)[['dice', 'hd95_mm']].mean()
    raw.to_csv(DEST/'all_organ_metrics.csv', index=False)
    patient.to_csv(DEST/'patient_metrics.csv', index=False)
    effects, paired = [], []
    for (arm, factor), sub in patient.groupby(['arm', 'factor']):
        table = sub.pivot(index=['seed', 'case'], columns='mode', values='dice')
        for mode in SPEC['modes']:
            delta = table[mode]-table['intact']
            stats, mat = effect_stats(delta)
            key = {'arm': arm, 'factor': int(factor), 'contrast': mode+'_minus_intact'}
            effects.append({**key, **stats})
            paired += [{**key, 'case': c, 'difference': float(v)} for c, v in mat.mean(1).items()]
    for factor, sub in patient.groupby('factor'):
        table = sub.pivot(index=['seed', 'case'], columns=['arm', 'mode'], values='dice')
        for mode in SPEC['modes']:
            delta = (table[('context_aug', mode)]-table[('context_aug', 'intact')]
                     -table[('context', mode)]+table[('context', 'intact')])
            stats, mat = effect_stats(delta)
            key = {'arm': 'context_aug_minus_context', 'factor': int(factor),
                   'contrast': mode+'_minus_intact'}
            effects.append({**key, **stats})
            paired += [{**key, 'case': c, 'difference': float(v)} for c, v in mat.mean(1).items()]
    pd.DataFrame(effects).to_csv(DEST/'paired_contrasts.csv', index=False)
    pd.DataFrame(paired).to_csv(DEST/'paired_patient_differences.csv', index=False)
    (DEST/'analysis.json').write_text(json.dumps({'effects': effects,
        'interpretation_limit': SPEC['mechanistic_scope'], 'fresh20_unused': True,
        'new_predictions': 960, 'completed_utc': datetime.now(timezone.utc).isoformat()}, indent=2))
    print(pd.DataFrame(effects).to_string(index=False), flush=True)


def launch():
    assert json.loads((OUT/'verification.json').read_text())['all_passed']
    check()
    q = queue.Queue()
    for seed in SPEC['seeds']:
        for arm in SPEC['arms']: q.put((arm, seed))
    errors = []
    deadline = datetime.fromisoformat(CFG['deadline_utc']).timestamp()
    def worker(gpu):
        env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        while True:
            try: arm, seed = q.get_nowait()
            except queue.Empty: return
            try:
                if deadline-time.time() < 360: raise RuntimeError('Insufficient time for another diagnostic run')
                with (ROOT/'logs'/f'intervention_{arm}_{seed}.log').open('a') as log:
                    subprocess.run([sys.executable, '-u', str(ROOT/'src/intervene_context.py'), 'evaluate',
                                    '--arm', arm, '--seed', str(seed)], env=env, cwd=ROOT,
                                   stdout=log, stderr=subprocess.STDOUT, check=True,
                                   timeout=max(1, deadline-time.time()-30))
                print('Intervention complete', arm, seed, flush=True)
            except Exception as exc: errors.append({'arm': arm, 'seed': seed, 'error': repr(exc)})
            finally: q.task_done()
    workers = [threading.Thread(target=worker, args=(gpu,)) for gpu in [0, 1]]
    for t in workers: t.start()
    for t in workers: t.join()
    (DEST/'queue_complete.json').write_text(json.dumps({'errors': errors}, indent=2))
    if errors: raise RuntimeError(errors)
    analyze()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['launch', 'evaluate', 'analyze', 'check'])
    ap.add_argument('--arm', choices=SPEC['arms'])
    ap.add_argument('--seed', type=int)
    args = ap.parse_args()
    if args.mode == 'launch': launch()
    elif args.mode == 'evaluate': evaluate(args.arm, args.seed)
    elif args.mode == 'check': check()
    else: analyze()
