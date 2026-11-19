"""Matched retraining after retaining slope-only or curvature-only context."""
import argparse
import hashlib
import json
import math
import time

import numpy as np
import torch

from mechanism_common import ROOT, OUT, CFG, ComponentModel, run_dir, sha, sources
from experiment import SlabSampler, init_runtime, make_logits, validate
from models import volume_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=CFG['new_arms'], required=True)
    ap.add_argument('--seed', type=int, required=True)
    args = ap.parse_args()
    arm, seed = args.arm, args.seed
    assert (OUT/'pipeline_checks.json').exists()
    assert seed in CFG['seeds']
    run = run_dir(arm, seed)
    run.mkdir(parents=True, exist_ok=True)
    if (run/'complete.json').exists(): return
    if (run/'train.jsonl').exists(): raise RuntimeError('Refuse to overwrite incomplete fit')
    split = json.loads((ROOT/'data/metadata/split.json').read_text())['splits']
    init_runtime(seed)
    device = torch.device('cuda')
    model = ComponentModel(arm.removesuffix('_aug'), CFG['width']).to(device)
    initial = hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes()
                                    for p in model.parameters())).hexdigest()
    prior = json.loads((ROOT/'results/continuation/b40000'/f'repeat_seed{seed}'/'metadata.json').read_text())
    assert initial == prior['initial_parameter_sha256']
    code = sources(['train_mechanism.py', 'mechanism_common.py', 'experiment.py', 'models.py'])
    protocol = sha(ROOT/'configs/mechanism.json')
    metadata = {'arm': arm, 'seed': seed, 'config': CFG, 'initial_parameter_sha256': initial,
                'parameters': sum(p.numel() for p in model.parameters()),
                'train_ids': split['train'], 'val_ids': split['val'], 'source_sha256': code,
                'protocol_sha256': protocol, 'device': torch.cuda.get_device_name(),
                'torch_version': torch.__version__, 'start_unix': time.time()}
    (run/'metadata.json').write_text(json.dumps(metadata, indent=2))
    augmented = arm.endswith('_aug')
    sampler = SlabSampler(split['train'], seed, CFG['patch'], augmentation=augmented)
    factor_rng = np.random.default_rng(seed+100000)
    counts = {1: 0, 2: 0, 3: 0}
    trace = hashlib.sha256()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=1e-5)
    best, best_step, started = -1, 0, time.time()
    model.train()
    torch.cuda.reset_peak_memory_stats()
    for step in range(1, CFG['steps']+1):
        halo, y, idx, start = sampler.next()
        factor = int(factor_rng.choice([1, 2, 3], p=[.5, .25, .25])) if augmented else 1
        counts[factor] += 1
        trace.update(np.array([idx, *start], dtype='<i8').tobytes())
        x, target = torch.from_numpy(halo).to(device), torch.from_numpy(y)[None].to(device)
        optimizer.zero_grad(set_to_none=True)
        lr = CFG['lr']*(.1+.9*.5*(1+math.cos(math.pi*(step-1)/CFG['steps'])))
        for group in optimizer.param_groups: group['lr'] = lr
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss = volume_loss(make_logits(model, x, '2p5d'), target)
        if not torch.isfinite(loss): raise RuntimeError(f'Nonfinite loss {step}')
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 12.)
        if not torch.isfinite(grad) or float(grad) == 0: raise RuntimeError(f'Invalid gradient {step}')
        optimizer.step()
        if step == 1 or step % 50 == 0:
            record = {'step': step, 'loss': float(loss), 'grad_norm': float(grad), 'lr': lr,
                      'patient': split['train'][idx], 'start': start.tolist(), 'factor': factor,
                      'elapsed_seconds': time.time()-started}
            with (run/'train.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
        if step == 1 or step % 1000 == 0:
            print(arm, seed, step, 'loss', float(loss), 'seconds', round(time.time()-started, 1), flush=True)
        if step % CFG['val_every'] == 0:
            score, cases = validate(model, split['val'], '2p5d', device, CFG['patch'][0])
            with (run/'validation.jsonl').open('a') as f:
                f.write(json.dumps({'step': step, 'macro_dice': score, 'cases': cases,
                                    'elapsed_seconds': time.time()-started})+'\n')
            checkpoint = {'state_dict': model.network.state_dict(), 'arm': arm, 'seed': seed,
                          'component': model.component, 'step': step, 'validation_dice': score,
                          'config': CFG, 'sampling_sha256': trace.hexdigest()}
            if score > best:
                best, best_step = score, step
                torch.save(checkpoint, run/'best.pt')
            torch.save(checkpoint, run/'last.pt')
            print('validation', arm, seed, step, score, flush=True)
    prior_complete = json.loads((ROOT/'results/continuation/b40000'/f'repeat_seed{seed}'/'complete.json').read_text())
    assert trace.hexdigest() == prior_complete['sampling_sha256']
    assert sources(list(code)) == code and sha(ROOT/'configs/mechanism.json') == protocol
    complete = {'training_steps': CFG['steps'], 'best_step': best_step, 'best_validation_macro_dice': best,
                'sampling_sha256': trace.hexdigest(), 'factor_counts': counts,
                'total_seconds': time.time()-started, 'finish_unix': time.time(),
                'best_checkpoint_sha256': sha(run/'best.pt'),
                'peak_gpu_allocated_mb': torch.cuda.max_memory_allocated()/2**20}
    (run/'complete.json').write_text(json.dumps(complete, indent=2))
    print('Training complete', arm, seed, flush=True)


if __name__ == '__main__': main()
