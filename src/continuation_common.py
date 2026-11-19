"""Frozen identifiers and provenance helpers for the second experimental session."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/continuation'
META = ROOT/'data/metadata/continuation'
CFG = json.loads((ROOT/'configs/continuation.json').read_text())
BASE = {'repeat': 'repeat', 'context': '2p5d', 'repeat_aug': 'repeat',
        'context_aug': '2p5d', 'volume': '3d', 'volume_aug': '3d'}
OLD = CFG['reused_short_arms']


def run_dir(arm, seed, steps):
    return OUT/f'b{steps}'/f'{arm}_seed{seed}'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes(names):
    return {n: sha(ROOT/'src'/n) for n in names}


def checkpoint_path(arm, seed, steps):
    if steps == CFG['short_steps'] and arm in OLD:
        return ROOT/'results/runs'/f'{OLD[arm]}_seed{seed}'/'best.pt'
    return run_dir(arm, seed, steps)/'best.pt'


def all_evaluations():
    return [(arm, seed, steps) for steps, arms in
            [(CFG['short_steps'], list(OLD)+CFG['new_short_arms']),
             (CFG['long_steps'], CFG['long_arms'])]
            for arm in arms for seed in CFG['seeds']]


def new_trainings():
    # Core long-budget factorial first, then missing short controls, then 3D.
    return [(arm, seed, steps) for steps, arms in
            [(CFG['long_steps'], [a for a in CFG['long_arms'] if not a.startswith('volume')]),
             (CFG['short_steps'], CFG['new_short_arms']),
             (CFG['long_steps'], [a for a in CFG['long_arms'] if a.startswith('volume')])]
            for seed in CFG['seeds'] for arm in arms]
