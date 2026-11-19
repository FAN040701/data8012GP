"""Three-channel slice policies. Offsets use the shared 3-mm model grid."""
import json
import math

import numpy as np
import torch
from torch.nn import functional as F

from experiment import ROOT

PROTOCOL = ROOT/'configs/slice_selection.json'
CFG = json.loads(PROTOCOL.read_text())
OUT = ROOT/'results/slice_selection'
ARMS = [name + suffix for name in CFG['offsets'] for suffix in ('', '_aug')]


def offsets(arm):
    return CFG['offsets'][arm.removesuffix('_aug')]


def run_dir(arm, seed):
    return OUT/'runs'/f'{arm}_seed{seed}'


def train_input(halo, arm):
    depth = len(halo) - 6
    return torch.stack([halo[3+offset:3+offset+depth] for offset in offsets(arm)], dim=1)


@torch.inference_mode()
def predict(model, image, device, arm, batch=24):
    previous = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False
    try:
        depth, height, width = image.shape
        hpad, wpad = math.ceil(height/16)*16, math.ceil(width/16)*16
        x = torch.as_tensor(np.asarray(image, dtype=np.float32), device=device)
        halo = F.pad(x, (0,wpad-width,0,hpad-height,3,3))
        out = torch.empty((5,depth,hpad,wpad), dtype=torch.float32, device=device)
        for begin in range(0,depth,batch):
            end = min(begin+batch,depth)
            planes = train_input(halo[begin:end+6], arm)
            with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                logits = model(planes)
            out[:,begin:end] = logits.float().softmax(1).permute(1,0,2,3)
        return out[:,:,:height,:width].cpu().numpy()
    finally:
        torch.backends.cudnn.benchmark = previous
