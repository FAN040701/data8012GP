"""Matched 40k slice-offset fits, with and without degradation augmentation."""
import argparse
import hashlib
import json
import math
import time

import numpy as np
import torch

from mechanism_common import ROOT, sha, sources
from width_budget_common import WideAugSampler
from slice_selection import OUT, ARMS, PROTOCOL, train_input, predict, offsets, run_dir
from experiment import PROC, init_runtime, dice_scores
from models import UNet, volume_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=ARMS, required=True)
    ap.add_argument('--seed', type=int, required=True)
    args = ap.parse_args()
    arm, seed = args.arm, args.seed
    design = json.loads((ROOT/'configs/final_study.json').read_text())
    assert arm.removesuffix('_aug') in design['new_policies'] and seed in design['new_seeds']
    probe = torch.arange(38)[:,None,None]
    assert torch.equal(train_input(probe, arm)[:,1], probe[3:-3])
    assert train_input(probe,arm)[0,:,0,0].tolist() == [3+i for i in offsets(arm)]
    run = run_dir(arm, seed); run.mkdir(parents=True, exist_ok=True)
    if (run/'complete.json').exists(): return
    if (run/'train.jsonl').exists(): raise RuntimeError('Refuse to overwrite incomplete fit')
    split = json.loads((ROOT/'data/metadata/split.json').read_text())['splits']
    init_runtime(seed); device = torch.device('cuda')
    model = UNet('2p5d', 16).to(device)
    initial = hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()
    prior = json.loads((ROOT/'results/seed_stability/runs'/f'repeat_seed{seed}'/'metadata.json').read_text())
    assert initial == prior['initial_parameter_sha256']
    code = sources(['train_final_offsets.py', 'width_budget_common.py', 'slice_selection.py', 'experiment.py', 'models.py'])
    protocol = sha(PROTOCOL)
    (run/'metadata.json').write_text(json.dumps({'arm': arm, 'seed': seed, 'parameters': sum(p.numel() for p in model.parameters()),
        'initial_parameter_sha256': initial, 'train_ids': split['train'], 'val_ids': split['val'],
        'patch': [32, 128, 128], 'source_sha256': code, 'protocol_sha256': protocol,
        'start_unix': time.time(), 'device': torch.cuda.get_device_name()}, indent=2))
    augmented = arm.endswith('_aug')
    sampler = WideAugSampler(split['train'], seed, [32, 128, 128], augmented)
    factor_rng = np.random.default_rng(seed+100000); factors = {1: 0, 2: 0, 3: 0}
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    best, best_step, started, trace = -1, 0, time.time(), hashlib.sha256()
    model.train()
    for step in range(1, 40001):
        halo, y, idx, start = sampler.next()
        factor = int(factor_rng.choice([1, 2, 3], p=[.5, .25, .25])) if augmented else 1
        factors[factor] += 1; trace.update(np.array([idx, *start], dtype='<i8').tobytes())
        x, target = torch.from_numpy(halo).to(device), torch.from_numpy(y)[None].to(device)
        optimizer.zero_grad(set_to_none=True)
        lr = 3e-4*(.1+.9*.5*(1+math.cos(math.pi*(step-1)/40000)))
        for group in optimizer.param_groups: group['lr'] = lr
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(train_input(x,arm)).permute(1, 0, 2, 3).unsqueeze(0)
            loss = volume_loss(logits, target)
        assert torch.isfinite(loss)
        loss.backward(); grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 12.)
        assert torch.isfinite(grad) and float(grad) > 0
        optimizer.step()
        if step == 1 or step % 50 == 0:
            with (run/'train.jsonl').open('a') as f:
                f.write(json.dumps({'step': step, 'loss': float(loss), 'grad_norm': float(grad), 'lr': lr,
                                    'patient': split['train'][idx], 'start': start.tolist(), 'factor': factor,
                                    'elapsed_seconds': time.time()-started})+'\n')
        if step == 1 or step % 1000 == 0: print(arm, seed, step, 'loss', float(loss), 'seconds', time.time()-started, flush=True)
        if step % 2000 == 0:
            model.eval(); cases = []
            for case in split['val']:
                image = np.load(PROC/case/'image_f1.npy'); y = np.load(PROC/case/'label.npy')
                scores = dice_scores(predict(model, image, device, arm).argmax(0), y)
                cases.append({'case': case, 'dice': scores, 'macro_dice': float(np.nanmean(scores))})
            score = float(np.mean([r['macro_dice'] for r in cases]))
            with (run/'validation.jsonl').open('a') as f:
                f.write(json.dumps({'step': step, 'macro_dice': score, 'cases': cases})+'\n')
            checkpoint = {'state_dict': model.state_dict(), 'arm': arm, 'seed': seed, 'step': step,
                          'validation_dice': score, 'offsets': offsets(arm), 'sampling_sha256': trace.hexdigest()}
            if score > best:
                best, best_step = score, step; torch.save(checkpoint, run/'best.pt')
            torch.save(checkpoint, run/'last.pt')
            model.train(); print('validation', arm, seed, step, score, flush=True)
    prior_complete = json.loads((ROOT/'results/seed_stability/runs'/f'repeat_seed{seed}'/'complete.json').read_text())
    assert trace.hexdigest() == prior_complete['sampling_sha256']
    assert sources(list(code)) == code and sha(PROTOCOL) == protocol
    (run/'complete.json').write_text(json.dumps({'training_steps': 40000, 'best_step': best_step,
        'best_validation_macro_dice': best, 'sampling_sha256': trace.hexdigest(), 'factor_counts': factors,
        'best_checkpoint_sha256': sha(run/'best.pt'), 'total_seconds': time.time()-started,
        'finish_unix': time.time()}, indent=2))


if __name__ == '__main__': main()
