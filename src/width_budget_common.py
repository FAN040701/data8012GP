"""Augmentation-compatible wider context, preserving the original sampler trace."""
import json
import numpy as np

from mechanism_common import ROOT
from experiment import SlabSampler

OUT = ROOT/'results/mechanism/width_budget'
CFG = json.loads((ROOT/'configs/width_budget.json').read_text())


class WideAugSampler(SlabSampler):
    def __init__(self, ids, seed, patch, augmentation=False):
        super().__init__(ids, seed, patch, augmentation=augmentation)
        self.pad[0] += 2
        self.data = [([np.pad(image, ((2, 2), (0, 0), (0, 0))) for image in images],
                      np.pad(label, ((2, 2), (0, 0), (0, 0))), shape, coords)
                     for images, label, shape, coords in self.data]

    def next(self):
        idx = int(self.rng.integers(len(self.data)))
        images, label, shape, coords = self.data[idx]
        image = images[int(self.aug_rng.choice(3, p=[.5, .25, .25]))] if self.augmentation else images[0]
        if self.rng.random() < .5 and coords:
            group = coords[int(self.rng.integers(len(coords)))]
            center = group[int(self.rng.integers(len(group)))].astype(int)
        else: center = np.array([self.rng.integers(int(n)) for n in shape])
        start = center+self.pad-self.patch//2
        end = start+self.patch
        z, y, x = start; zz, yy, xx = end
        halo = image[z-3:zz+3, y:yy, x:xx].copy()
        target = label[z:zz, y:yy, x:xx].astype(np.int64)
        scale = self.rng.uniform(.9, 1.1); shift = self.rng.uniform(-.03, .03)
        halo = np.clip(halo*scale+shift, 0, 1)
        return halo, target, idx, start-self.pad


def run_dir(arm, seed): return OUT/'runs'/f'{arm}_seed{seed}'
