"""Paths and immutable protocol for the additional training-seed experiment."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/seed_stability'
PROTOCOL = ROOT / 'configs/seed_stability.json'
CFG = json.loads(PROTOCOL.read_text())
BASE = {'repeat': 'repeat', 'context': '2p5d', 'repeat_aug': 'repeat', 'context_aug': '2p5d'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes(names):
    return {name: sha(ROOT / 'src' / name) for name in names}


def run_dir(arm, seed, steps=40000):
    assert steps == CFG['long_steps']
    return OUT / 'runs' / f'{arm}_seed{seed}'


def checkpoint_path(arm, seed, selection='best'):
    folder = run_dir(arm, seed) if seed in CFG['new_seeds'] else ROOT / 'results/continuation/b40000' / f'{arm}_seed{seed}'
    return folder / f'{selection}.pt'


def jobs():
    return [(arm, seed) for seed in CFG['new_seeds'] for arm in CFG['arms']]
