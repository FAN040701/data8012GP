"""Local, frozen settings and exact contextual-input component interventions."""
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from models import UNet

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/mechanism'
CFG = json.loads((ROOT/'configs/mechanism.json').read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources(names):
    return {n: sha(ROOT/'src'/n) for n in names}


def run_dir(arm, seed):
    return OUT/'runs'/f'{arm}_seed{seed}'


def transform(inp, component):
    if component == 'full':
        return inp
    left, centre, right = inp.unbind(1)
    if component == 'slope':
        slope = (right-left)*0.5
        return torch.stack((centre-slope, centre, centre+slope), 1)
    if component == 'curvature':
        average = (left+right)*0.5
        return torch.stack((average, centre, average), 1)
    if component == 'repeat':
        return centre[:, None].expand(-1, 3, -1, -1).contiguous()
    raise ValueError(component)


class ComponentModel(nn.Module):
    def __init__(self, component, width=16):
        super().__init__()
        self.network = UNet('2p5d', width)
        self.component = component

    def forward(self, inp):
        # Inputs remain float32 here; convolution autocasting happens afterwards.
        return self.network(transform(inp, self.component))


def jobs():
    return [(arm, seed) for seed in CFG['seeds'] for arm in CFG['new_arms']]


def cohorts():
    old = json.loads((ROOT/'data/metadata/split.json').read_text())['splits']['test']
    recent = json.loads((ROOT/'data/metadata/continuation/split.json').read_text())['test']
    assert len(old) == 40 and len(recent) == 20 and not set(old) & set(recent)
    return {'prior_fresh20': recent, 'original40_exploratory': old}
