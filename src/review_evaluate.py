"""Score blinded-channel and refit controls on the original native label grids."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json

import nibabel as nib
import numpy as np
import pandas as pd
import torch

from experiment import PROC, init_runtime, predict_volume
from mechanism_common import sha
from models import UNet
from native_resample import restore_on_gpu
from proposal_quality import distances
from review_refits import ARMS, OUT, ROOT, SEEDS
from slice_selection import predict, run_dir
from verify_mechanism import NAMES, target

MIRRORS = ('near_far', 'far_near', 'near_far_aug', 'far_near_aug')
SWAP_SEEDS = (44, 55, 66)
BANDS = ('inferior', 'middle', 'superior')


def read(p):
    return json.loads(p.read_text())


class Swap(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x[:, [2, 1, 0]])


def measure(case, labels, group, archived, ordinary):
    truth, meta = target(case)
    assert labels.shape == truth.shape and labels.max() <= 4
    spacing = meta['original_spacing_xyz_mm']
    if group == 'swap':
        hdr = nib.load(archived/'predictions/f3'/f'{case}.nii.gz')
        assert np.allclose(hdr.affine, meta['original_affine'])
        archived_mask = np.asarray(hdr.dataobj, dtype=np.uint8)
        assert ordinary.shape == archived_mask.shape == labels.shape
        inference_drift = np.count_nonzero(ordinary != archived_mask) / labels.size
        mismatch = np.count_nonzero(labels != ordinary) / labels.size
    rows = []
    for k, organ in enumerate(NAMES[1:], 1):
        p, t = labels == k, truth == k
        nt, npred = np.count_nonzero(t), np.count_nonzero(p)
        assert nt > 0
        dice = 2 * np.count_nonzero(p & t) / (nt + npred)
        row = dict(case=case, organ=organ, dice=dice, true_voxels=nt, pred_voxels=npred)
        if group == 'capacity':
            row['hd95_mm'] = max(distances(p, t, spacing)[:2]) if npred else np.inf
        if group == 'swap':
            before = ordinary == k
            archived_organ = archived_mask == k
            row['archived_dice'] = 2 * np.count_nonzero(archived_organ & t) / (nt + np.count_nonzero(archived_organ))
            row['ordinary_fraction_different_from_archive'] = inference_drift
            row['ordinary_dice'] = 2 * np.count_nonzero(before & t) / (nt + np.count_nonzero(before))
            row['fraction_labels_changed'] = mismatch
            indices = np.flatnonzero(t.any(axis=(0, 1)))
            cuts = indices[0] + np.array([1, 2]) * (indices[-1] - indices[0] + 1) / 3
            zband = np.digitize(np.arange(t.shape[-1]), cuts)
            for field, new, old in [('fp', p & ~t, before & ~t), ('fn', ~p & t, ~before & t)]:
                difference = new.sum(axis=(0, 1), dtype=np.int64) - old.sum(axis=(0, 1), dtype=np.int64)
                for b, name in enumerate(BANDS):
                    row[f'{field}_delta_ml_{name}'] = difference[zband == b].sum() * np.prod(spacing) / 1000
        rows.append(row)
    return rows


def score(group, arm, seed, repeat=None, checkpoint_name='best'):
    if group == 'factorial':
        workspace = ROOT if repeat == 0 else OUT/f'refit{repeat}'
        folder = workspace/'results/continuation/b40000'/f'{arm}_seed{seed}'
        base = 'repeat' if arm.startswith('repeat') else '2p5d'
    elif group == 'capacity':
        folder, base = OUT/'capacity/results/runs'/f'3d_seed{seed}', '3d'
    else:
        folder, base = run_dir(arm, seed), '2p5d'
    suffix = f'run{repeat}' if repeat is not None else 'best'
    if checkpoint_name == 'last':
        if group != 'factorial':
            raise ValueError('Final checkpoints are used for the factorial control only.')
        suffix += '_last'
    output = OUT/'evaluation'/group/f'{arm}_seed{seed}'/suffix
    output.mkdir(parents=True, exist_ok=True)
    ckpt = folder/f'{checkpoint_name}.pt'
    digest = sha(ckpt)
    marker, csv = output/'complete.json', output/'organs.csv'
    if marker.exists():
        meta = read(marker)
        assert digest == meta['checkpoint_sha256'] and sha(csv) == meta['metrics_sha256']
        return
    init_runtime(seed)
    device = torch.device('cuda')
    model = UNet(base, 9 if group == 'capacity' else 16).to(device).eval()
    checkpoint = torch.load(ckpt, map_location='cpu', weights_only=False)
    which_step = 'training_steps' if checkpoint_name == 'last' else 'best_step'
    assert checkpoint['step'] == read(folder/'complete.json')[which_step]
    model.load_state_dict(checkpoint['state_dict'])
    if group == 'swap':
        swapped = Swap(model).eval()
        cases = read(ROOT/'data/metadata/mechanism_replication/split.json')['test']
        assert read(folder/'evaluation_complete.json')['checkpoint_sha256'] == digest
    elif group == 'capacity':
        cases = read(ROOT/'data/metadata/split.json')['splits']['test']
    else:
        cases = read(ROOT/'data/metadata/mechanism_replication/split.json')['test']
    assert len(set(cases)) == len(cases) == 40
    factor = 1 if group == 'capacity' else 3
    rows, pending = [], set()
    with ThreadPoolExecutor(max_workers=3) as pool:
        for case in cases:
            image = np.load(PROC/case/f'image_f{factor}.npy')
            meta = read(PROC/case/'metadata.json')
            ordinary = None
            if group == 'swap':
                ordinary = restore_on_gpu(predict(model, image, device, arm), meta, device)
                prob = predict(swapped, image, device, arm)
            else:
                prob = predict_volume(model, image, base, device)
            pred = restore_on_gpu(prob, meta, device)
            pending.add(pool.submit(measure, case, pred, group, folder, ordinary))
            if len(pending) >= 4:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for job in done:
                    rows.extend(job.result())
        for job in pending:
            rows.extend(job.result())
    frame = pd.DataFrame(rows).sort_values(['case','organ'])
    assert len(frame) == 160 and not frame.duplicated(['case','organ']).any()
    if group == 'swap':
        old = pd.read_csv(folder/'metrics.csv').query('factor==3')
        both = frame.merge(old, on=['case','organ'], validate='one_to_one', suffixes=('', '_old'))
        assert len(both) == 160 and np.allclose(both.archived_dice,both.dice_old,atol=1e-12,rtol=0)
    frame.to_csv(csv, index=False)
    marker.write_text(json.dumps(dict(rows=len(frame), patients=len(cases), factor=factor,
        checkpoint_step=checkpoint['step'], checkpoint_sha256=digest, metrics_sha256=sha(csv)),indent=2))
    print('Scored', group, arm, seed, repeat, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study',choices=['swap','capacity','factorial'])
    parser.add_argument('--repeat',type=int,choices=[0,1,2])
    parser.add_argument('--checkpoint', choices=['best','last'], default='best')
    parser.add_argument('--arm', choices=ARMS, help='Evaluate a single factorial arm')
    args = parser.parse_args()
    if args.study != 'factorial' and (args.checkpoint != 'best' or args.arm or args.repeat is not None):
        parser.error('--arm, --repeat and final checkpoints apply to factorial fits only')
    if args.study == 'factorial':
        repeats = [0,1,2] if args.checkpoint == 'last' else [1,2]
        for repeat in ([args.repeat] if args.repeat is not None else repeats):
            for arm in ([args.arm] if args.arm else ARMS):
                score('factorial',arm,11,repeat,args.checkpoint)
    elif args.study == 'capacity':
        for seed in SEEDS:
            score('capacity','3d_w9',seed)
    else:
        for seed in SWAP_SEEDS:
            for arm in MIRRORS:
                score('swap',arm,seed)
