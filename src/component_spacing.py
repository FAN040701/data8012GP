"""Bounded inference-only follow-up of component representations and sampling distance."""
import argparse
import concurrent.futures as cf
from datetime import datetime, timezone
import json
import os
import queue
import subprocess
import sys
import threading
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from mechanism_common import ROOT, ComponentModel, run_dir, sha, sources
from experiment import PROC, init_runtime, predict_volume
from evaluate_absolute_spacing import score
from native_resample import restore_on_gpu
from mechanism_statistics import component_contrasts, effect
from verify_mechanism import check_header, check_mask

OUT = ROOT/'results/mechanism/component_spacing'
PROTOCOL = ROOT/'configs/component_spacing.json'
CFG = json.loads(PROTOCOL.read_text())
CODE = ['component_spacing.py', 'mechanism_common.py', 'experiment.py', 'models.py',
        'evaluate_absolute_spacing.py', 'evaluate_sampling_phase.py', 'evaluate.py', 'native_resample.py', 'prepare_data.py']


def evaluate(arm, seed):
    dest = OUT/'runs'/f'{arm}_seed{seed}'; dest.mkdir(parents=True, exist_ok=True)
    if (dest/'complete.json').exists(): return
    freeze = json.loads((OUT/'freeze.json').read_text())
    checkpoint_path = run_dir(arm, seed)/'best.pt'
    digest = sha(checkpoint_path)
    assert digest == freeze['checkpoints_sha256'][str(checkpoint_path.relative_to(ROOT))]
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    init_runtime(seed); device = torch.device('cuda')
    model = ComponentModel(checkpoint['component'], 16).to(device).eval()
    model.network.load_state_dict(checkpoint['state_dict'])
    ids = json.loads((ROOT/'data/metadata/absolute_spacing/split.json').read_text())['cases']
    code, protocol_hash, started, rows = sources(CODE), sha(PROTOCOL), time.time(), []
    for target in CFG['target_sampling_mm']:
        folder = dest/f'target{int(target)}mm'; folder.mkdir(parents=True, exist_ok=True)
        pending = {}
        with cf.ThreadPoolExecutor(6) as pool:
            for case in ids:
                image = np.load(ROOT/'data/processed_absolute_spacing'/case/f'image_z{int(target)}mm.npy')
                meta = json.loads((PROC/case/'metadata.json').read_text())
                before = time.time()
                prob = predict_volume(model, image, '2p5d', device, 32)
                pred = restore_on_gpu(prob, meta, device)
                task = pool.submit(score, pred, case, target, arm, seed, folder, time.time()-before)
                pending[task] = case
                if len(pending) >= 12:
                    ready, _ = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
                    for task in ready: rows.extend(task.result()); pending.pop(task)
            for task in cf.as_completed(pending): rows.extend(task.result())
        print('component spacing', arm, seed, target, len(ids), flush=True)
    frame = pd.DataFrame(rows).sort_values(['target_sampling_mm', 'case', 'organ'])
    frame['arm'], frame['budget'], frame['cohort'] = arm, 40000, 'prior_thin_native_spacing'
    assert len(frame) == 168 and not frame.duplicated(['case', 'organ', 'target_sampling_mm']).any()
    frame.to_csv(dest/'metrics.csv', index=False)
    assert sources(CODE) == code and sha(checkpoint_path) == digest and sha(PROTOCOL) == protocol_hash
    (dest/'complete.json').write_text(json.dumps({'completed_utc': datetime.now(timezone.utc).isoformat(),
        'arm': arm, 'seed': seed, 'new_predictions': 42, 'checkpoint_sha256': digest,
        'source_sha256': code, 'protocol_sha256': protocol_hash, 'elapsed_seconds': time.time()-started,
        'report_generated': False}, indent=2))


def analyze():
    assert not json.loads((OUT/'queue_complete.json').read_text())['errors']
    frames, records = [], []
    for arm in CFG['new_arms']+CFG['reused_arms']:
        for seed in CFG['seeds']:
            root = OUT if arm in CFG['new_arms'] else ROOT/'results/mechanism/absolute_spacing'
            folder = root/'runs'/f'{arm}_seed{seed}'
            frame = pd.read_csv(folder/'metrics.csv'); assert len(frame) == 168
            frames.append(frame)
            if arm in CFG['new_arms']:
                record = json.loads((folder/'complete.json').read_text())
                assert record['protocol_sha256'] == sha(PROTOCOL)
                assert record['source_sha256'] == sources(CODE)
                assert record['checkpoint_sha256'] == sha(run_dir(arm, seed)/'best.pt')
                records.append(record)
    raw = pd.concat(frames, ignore_index=True)
    key = ['arm', 'seed', 'case', 'target_sampling_mm']
    assert len(raw) == 4032 and not raw.duplicated(key+['organ']).any()
    raw.to_csv(OUT/'all_organ_metrics.csv', index=False)
    patient = raw.groupby(key, as_index=False)[['dice', 'interior_dice']].mean()
    patient.to_csv(OUT/'patient_metrics.csv', index=False)
    comparisons = []
    for metric in ['dice', 'interior_dice']:
        wide = patient.pivot(index=['seed', 'case', 'target_sampling_mm'], columns='arm', values=metric)
        for name, weights in component_contrasts().items():
            delta = sum(wide[a]*v for a, v in weights.items())
            conditions = {f'target{int(t)}mm': delta.xs(t, level='target_sampling_mm') for t in [6., 15.]}
            conditions['target15_minus_target6'] = conditions['target15mm']-conditions['target6mm']
            for condition, values in conditions.items():
                stats, _ = effect(values, 2026111513)
                comparisons.append({'metric': metric, 'contrast': name, 'condition': condition, **stats})
    comparisons = pd.DataFrame(comparisons)
    comparisons.to_csv(OUT/'paired_contrasts.csv', index=False)
    new = raw[raw.arm.isin(CFG['new_arms'])]
    files = new[['prediction_file', 'case']].drop_duplicates(); assert len(files) == 504
    with cf.ThreadPoolExecutor(8) as pool: assert all(pool.map(check_header, files.itertuples(index=False, name=None)))
    samples = []
    for _, rows in new.groupby(['arm', 'seed', 'target_sampling_mm']):
        cases = sorted(rows.case.unique())
        samples.extend(rows[rows.case == c] for c in [cases[0], cases[-1]])
    tasks = [(s.prediction_file.iloc[0], s.case.iloc[0], s.to_dict('records')) for s in samples]
    with cf.ProcessPoolExecutor(4) as pool: errors = list(pool.map(check_mask, tasks))
    freeze = json.loads((OUT/'freeze.json').read_text())
    for path, digest in freeze['input_files_sha256'].items(): assert sha(ROOT/path) == digest
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    for ax, suffix in zip(axes, ['without_aug', 'with_aug']):
        for prefix, label, color in [('full_context_gain', 'Full neighbours − repeat', '#0072B2'),
                                     ('slope_gain', 'First difference − repeat', '#009E73'),
                                     ('curvature_gain', 'Second difference − repeat', '#D55E00')]:
            sub = comparisons[(comparisons.metric == 'dice') & (comparisons.contrast == prefix+'_'+suffix)]
            sub = sub.set_index('condition').loc[['target6mm', 'target15mm']]
            y = sub['mean'].to_numpy()*100
            ax.errorbar([6, 15], y, yerr=np.stack((y-sub.ci_low.to_numpy()*100, sub.ci_high.to_numpy()*100-y)),
                        marker='o', capsize=3, color=color, label=label)
        ax.axhline(0, color='black', linewidth=.7); ax.grid(alpha=.2); ax.set_xticks([6, 15])
        ax.set_xlabel('Fixed physical sampling distance (mm)'); ax.set_ylabel('Paired Dice difference (percentage points)')
        ax.set_title('Without augmentation' if suffix == 'without_aug' else 'With augmentation')
    axes[0].legend(fontsize=8)
    fig.suptitle('Same 21 patients: component representations across sampling distances\n'
                 'Frozen 40k models; exploratory unadjusted patient-bootstrap intervals', fontsize=11)
    artifacts = []
    for ext in ['png', 'svg']:
        p = OUT/f'component_spacing_effects.{ext}'; fig.savefig(p, dpi=180, bbox_inches='tight', pad_inches=.15)
        artifacts.append({'path': str(p.relative_to(ROOT)), 'sha256': sha(p)})
    plt.close(fig)
    (OUT/'analysis_and_verification.json').write_text(json.dumps({'completed_utc': datetime.now(timezone.utc).isoformat(),
        'all_passed': True, 'new_predictions': 504, 'new_training_runs': 0, 'patients': 21,
        'new_evaluation_configurations': len(records), 'native_masks_independently_checked': len(samples),
        'maximum_dice_error': max(errors), 'protocol_sha256': sha(PROTOCOL), 'source_sha256': sha(__file__),
        'all_results_exploratory': True, 'artifacts': artifacts, 'report_generated': False}, indent=2))
    print('Component spacing analysis and verification complete', flush=True)


def launch():
    OUT.mkdir(parents=True, exist_ok=True)
    deadline = datetime.fromisoformat(CFG['deadline_utc']).timestamp()
    marker = ROOT/'results/mechanism/absolute_spacing/queue_complete.json'
    while not marker.exists():
        if deadline-time.time() < 1800:
            (OUT/'not_started.json').write_text(json.dumps({'recorded_utc': datetime.now(timezone.utc).isoformat(),
                'reason': 'Fewer than 30 minutes remained before the main queue completed', 'new_predictions': 0,
                'protocol_sha256': sha(PROTOCOL)}, indent=2)); return
        time.sleep(15)
    assert not json.loads(marker.read_text())['errors']
    if deadline-time.time() < 1800:
        (OUT/'not_started.json').write_text(json.dumps({'recorded_utc': datetime.now(timezone.utc).isoformat(),
            'reason': 'Fewer than 30 minutes remained at queue handover', 'new_predictions': 0,
            'protocol_sha256': sha(PROTOCOL)}, indent=2)); return
    (OUT/'start.json').write_text(json.dumps({'started_utc': datetime.now(timezone.utc).isoformat(),
        'remaining_seconds': deadline-time.time(), 'protocol_sha256': sha(PROTOCOL)}, indent=2))
    q = queue.Queue()
    for arm in CFG['new_arms']:
        for seed in CFG['seeds']: q.put((arm, seed))
    events, errors, lock = [], [], threading.Lock()
    def worker(gpu):
        env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        while True:
            try: arm, seed = q.get_nowait()
            except queue.Empty: return
            if (OUT/'runs'/f'{arm}_seed{seed}'/'complete.json').exists(): q.task_done(); continue
            try:
                with lock:
                    events.append({'arm': arm, 'seed': seed, 'status': 'started', 'unix': time.time()})
                    (OUT/'queue_status.json').write_text(json.dumps(events, indent=2))
                with (ROOT/'logs'/f'component_spacing_{arm}_{seed}.log').open('a') as log:
                    subprocess.run([sys.executable, '-u', __file__, 'evaluate', '--arm', arm, '--seed', str(seed)],
                                   env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True,
                                   timeout=max(1, deadline-time.time()-120))
                with lock:
                    events.append({'arm': arm, 'seed': seed, 'status': 'complete', 'unix': time.time()})
                    (OUT/'queue_status.json').write_text(json.dumps(events, indent=2))
                print('Complete', arm, seed, flush=True)
            except Exception as exc:
                with lock: errors.append({'arm': arm, 'seed': seed, 'error': repr(exc)})
            finally: q.task_done()
    threads = [threading.Thread(target=worker, args=(g,)) for g in [0, 1]]
    for t in threads: t.start()
    for t in threads: t.join()
    (OUT/'queue_complete.json').write_text(json.dumps({'events': events, 'errors': errors,
        'completed_utc': datetime.now(timezone.utc).isoformat()}, indent=2))
    assert not errors, errors
    analyze()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['launch', 'evaluate', 'analyze'])
    parser.add_argument('--arm', choices=CFG['new_arms'])
    parser.add_argument('--seed', type=int, choices=CFG['seeds'])
    args = parser.parse_args()
    if args.mode == 'evaluate': evaluate(args.arm, args.seed)
    elif args.mode == 'analyze': analyze()
    else: launch()
