"""Run isolated copies of the published trainers for same-seed and capacity controls."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/review_controls'
ARMS = ('repeat', 'context', 'repeat_aug', 'context_aug')
SEEDS = (11, 22, 33)


def prepare(name):
    work = OUT / name
    for directory in ('src', 'configs', 'results/runs'):
        (work / directory).mkdir(parents=True, exist_ok=True)
    for f in ('experiment.py', 'models.py', 'prepare_data.py', 'evaluate.py',
              'train_continuation.py', 'continuation_common.py'):
        src = ROOT / 'src' / f
        dst = work / 'src' / f
        if dst.exists() and dst.read_bytes() != src.read_bytes():
            raise ValueError(f'Trainer changed since setup: {dst}')
        if not dst.exists():
            shutil.copyfile(src, dst)
    data = work / 'data'
    if not data.exists():
        data.symlink_to(ROOT / 'data', target_is_directory=True)
    for cfg in ('experiment', 'continuation'):
        filename = cfg + '.json'
        value = json.loads((ROOT / 'configs' / filename).read_text())
        if cfg == 'experiment' and name == 'capacity':
            value['width'] = 9
        dst = work / 'configs' / filename
        if dst.exists() and value != json.loads(dst.read_text()):
            raise ValueError(f'Existing settings differ: {dst}')
        dst.write_text(json.dumps(value, indent=2)+'\n')
    return work


def run(work, rel, trainer, args):
    dest = work / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    with (dest.parent / (dest.name + '.lock')).open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if (dest / 'complete.json').exists():
            return
        if (dest / 'train.jsonl').exists():
            # Keep the incomplete run and start with a clean set of checkpoints.
            dest.rename(dest.with_name(f'{dest.name}.interrupted.{time.time_ns()}'))
        log = work / (dest.name + '.log')
        with log.open('a', buffering=1) as handle:
            subprocess.run([sys.executable, str(work / 'src' / trainer), *args], cwd=work,
                           stdout=handle, stderr=subprocess.STDOUT, check=True)
    print('Completed', dest, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study', choices=['factorial', 'capacity'])
    opts = parser.parse_args()
    if opts.study == 'capacity':
        work = prepare('capacity')
        for seed in SEEDS:
            run(work, f'results/runs/3d_seed{seed}', 'experiment.py',
                ['--kind', '3d', '--seed', str(seed)])
        return
    for repeat in (1, 2):
        work = prepare(f'refit{repeat}')
        # All four arms initialise from the same seed-specific planar distribution.
        prior = ROOT / 'results/runs/repeat_seed11/metadata.json'
        dst = work / 'results/runs/repeat_seed11'
        dst.mkdir(exist_ok=True)
        if prior.exists():
            (dst / 'metadata.json').write_bytes(prior.read_bytes())
        else:
            import torch
            from experiment import init_runtime
            from models import UNet
            init_runtime(11)
            model = UNet('repeat', 16)
            weights = b''.join(p.detach().numpy().tobytes() for p in model.parameters())
            (dst / 'metadata.json').write_text(json.dumps(
                {'initial_parameter_sha256':hashlib.sha256(weights).hexdigest()}))
        for arm in ARMS:
            run(work, f'results/continuation/b40000/{arm}_seed11', 'train_continuation.py',
                ['--arm', arm, '--seed', '11', '--steps', '40000'])


if __name__ == '__main__':
    main()
