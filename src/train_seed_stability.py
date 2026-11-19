"""Additional seeds with the unchanged 40k factorial training implementation."""
import argparse
import hashlib
import json
import math
import time

import numpy as np
import torch

from seed_stability_common import ROOT, OUT, CFG, BASE, run_dir, sha, source_hashes
from experiment import SlabSampler, init_runtime, make_logits, validate
from models import UNet, volume_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=BASE, required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--steps', type=int, required=True)
    args = ap.parse_args()
    assert args.steps == CFG['long_steps']
    assert args.seed in CFG['new_seeds']
    assert (OUT/'pipeline_checks.json').exists()
    arm, seed, steps = args.arm, args.seed, args.steps
    run = run_dir(arm, seed, steps)
    run.mkdir(parents=True, exist_ok=True)
    if (run/'complete.json').exists():
        print('Already complete', run, flush=True)
        return
    if (run/'train.jsonl').exists():
        raise RuntimeError('Refuse to mix an incomplete existing run with a new fit')
    split = json.loads((ROOT/'data/metadata/split.json').read_text())['splits']
    train_ids, val_ids = split['train'], split['val']
    init_runtime(seed)
    device = torch.device('cuda')
    base = BASE[arm]
    augmented = arm.endswith('_aug')
    model = UNet(base, CFG['width']).to(device)
    initial = hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes()
                                    for p in model.parameters())).hexdigest()
    expected = json.loads((OUT/'initializations.json').read_text())
    assert initial == expected[str(seed)]
    cfg = {k: CFG[k] for k in ['patch', 'width', 'lr', 'val_every']}
    cfg['steps'] = steps
    sources = source_hashes(['train_seed_stability.py', 'seed_stability_common.py',
                             'experiment.py', 'models.py'])
    metadata = {'arm': arm, 'base_kind': base, 'seed': seed, 'config': cfg,
                'augmentation': augmented, 'initial_parameter_sha256': initial,
                'parameters': sum(p.numel() for p in model.parameters()),
                'train_ids': train_ids, 'val_ids': val_ids,
                'source_sha256': sources, 'protocol_sha256': sha(ROOT/'configs/seed_stability.json'),
                'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
                'device': torch.cuda.get_device_name(), 'start_unix': time.time()}
    (run/'metadata.json').write_text(json.dumps(metadata, indent=2))
    sampler = SlabSampler(train_ids, seed, CFG['patch'], augmentation=augmented)
    factor_rng = np.random.default_rng(seed+100000)
    factors = {1: 0, 2: 0, 3: 0}
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=1e-5)
    schedule_hash = hashlib.sha256()
    started = time.time()
    best, best_step = -1, 0
    model.train()
    torch.cuda.reset_peak_memory_stats()
    for step in range(1, steps+1):
        halo, y, idx, start = sampler.next()
        factor = int(factor_rng.choice([1, 2, 3], p=[.5, .25, .25])) if augmented else 1
        factors[factor] += 1
        schedule_hash.update(np.array([idx, *start], dtype='<i8').tobytes())
        x = torch.from_numpy(halo).to(device)
        target = torch.from_numpy(y)[None].to(device)
        optimizer.zero_grad(set_to_none=True)
        lr = CFG['lr']*(.1+.9*.5*(1+math.cos(math.pi*(step-1)/steps)))
        for group in optimizer.param_groups: group['lr'] = lr
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss = volume_loss(make_logits(model, x, base), target)
        if not torch.isfinite(loss): raise RuntimeError(f'Nonfinite loss at {step}')
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 12.)
        if not torch.isfinite(grad) or float(grad) == 0: raise RuntimeError(f'Invalid gradient at {step}')
        optimizer.step()
        if step == 1 or step % 50 == 0:
            row = {'step': step, 'loss': float(loss), 'grad_norm': float(grad), 'lr': lr,
                   'patient': train_ids[idx], 'start': start.tolist(), 'factor': factor,
                   'elapsed_seconds': time.time()-started}
            with (run/'train.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
        if step == 1 or step % 1000 == 0:
            print(arm, seed, f'{step}/{steps}', 'loss', float(loss),
                  'elapsed', round(time.time()-started, 1), flush=True)
        if step % CFG['val_every'] == 0 or step == steps:
            score, cases = validate(model, val_ids, base, device, CFG['patch'][0])
            with (run/'validation.jsonl').open('a') as f:
                f.write(json.dumps({'step': step, 'macro_dice': score, 'cases': cases,
                                    'elapsed_seconds': time.time()-started})+'\n')
            checkpoint = {'state_dict': model.state_dict(), 'kind': base, 'arm': arm,
                          'seed': seed, 'step': step, 'validation_dice': score,
                          'config': cfg, 'sampling_sha256': schedule_hash.hexdigest()}
            if score > best:
                best, best_step = score, step
                torch.save(checkpoint, run/'best.pt')
            torch.save(checkpoint, run/'last.pt')
            print('validation', arm, seed, step, score, flush=True)
    complete = {'training_steps': steps, 'best_step': best_step,
                'best_validation_macro_dice': best, 'total_seconds': time.time()-started,
                'sampling_sha256': schedule_hash.hexdigest(), 'factor_counts': factors,
                'peak_gpu_allocated_mb': torch.cuda.max_memory_allocated()/2**20,
                'best_checkpoint_sha256': sha(run/'best.pt'), 'finish_unix': time.time()}
    assert source_hashes(list(sources)) == sources
    (run/'complete.json').write_text(json.dumps(complete, indent=2))
    print('Training complete', arm, seed, steps, flush=True)


if __name__ == '__main__': main()
