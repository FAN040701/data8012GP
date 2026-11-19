"""Check mathematical interventions and unchanged training/inference plumbing."""
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import torch

from mechanism_common import ROOT, OUT, CFG, ComponentModel, transform, sha, sources
from experiment import PROC, SlabSampler, init_runtime, make_logits, predict_volume
from models import UNet, volume_loss, planar_input


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert torch.cuda.is_available(), 'Use the host GPU execution context'
    init_runtime(11)
    rng = np.random.default_rng(2026111503)
    inp = torch.from_numpy(rng.random((7, 3, 32, 48), dtype=np.float32))
    centre = inp[:, 1]
    s, k = (inp[:, 2]-inp[:, 0])/2, (inp[:, 0]+inp[:, 2])/2-centre
    full_reconstructed = torch.stack((centre+k-s, centre, centre+k+s), 1)
    identity_error = float((full_reconstructed-inp).abs().max())
    assert identity_error < 2e-7
    for component in ['slope', 'curvature', 'repeat']:
        x = transform(inp, component)
        assert torch.equal(x[:, 1], centre)
        assert float((transform(x, component)-x).abs().max()) < 2e-7
        if component == 'slope':
            assert float(((x[:, 2]-x[:, 0])/2-s).abs().max()) < 1e-7
            assert float(((x[:, 0]+x[:, 2])/2-centre).abs().max()) < 1e-7
        if component == 'curvature':
            assert torch.equal(x[:, 0], x[:, 2])
            assert float(((x[:, 0]+x[:, 2])/2-centre-k).abs().max()) < 1e-7
    decomposition_error = float((transform(inp, 'slope')+transform(inp, 'curvature')-
                                  transform(inp, 'repeat')-inp).abs().max())
    assert decomposition_error < 2e-7
    repeated = transform(inp, 'repeat')
    assert torch.equal(transform(repeated, 'slope'), repeated)
    assert torch.equal(transform(repeated, 'curvature'), repeated)
    split = json.loads((ROOT/'data/metadata/split.json').read_text())['splits']
    case = split['train'][0]
    sampler = SlabSampler([case], 11, [4, 32, 48], augmentation=True)
    halo, target, _, _ = sampler.next()
    device = torch.device('cuda')
    gradient_checks = {}
    for component in ['slope', 'curvature']:
        init_runtime(11)
        model = ComponentModel(component).to(device)
        initial = hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes()
                                        for p in model.parameters())).hexdigest()
        old = json.loads((ROOT/'results/continuation/b40000/repeat_seed11/metadata.json').read_text())
        assert initial == old['initial_parameter_sha256']
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss = volume_loss(make_logits(model, torch.from_numpy(halo).to(device), '2p5d'),
                               torch.from_numpy(target)[None].to(device))
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 12.)
        assert torch.isfinite(loss) and torch.isfinite(grad) and float(grad) > 0
        gradient_checks[component] = {'loss': float(loss), 'grad': float(grad)}
    # The wrapper's full mode must reproduce existing reconstruction exactly,
    # including missing-neighbour zero padding and a non-divisible field of view.
    init_runtime(11)
    original = UNet('2p5d').to(device).eval()
    wrapper = ComponentModel('full').to(device).eval()
    wrapper.network.load_state_dict(original.state_dict())
    image = rng.random((5, 29, 37), dtype=np.float32)
    p = predict_volume(original, image, '2p5d', device, batch=3)
    q = predict_volume(wrapper, image, '2p5d', device, batch=3)
    assert np.array_equal(p, q)
    # Explicit mathematical inference/halo agreement, including the first/last slice.
    padded = np.pad(image, ((1, 1), (0, 0), (0, 0)))
    channels = planar_input(torch.from_numpy(padded), '2p5d')
    for mode in ['slope', 'curvature']:
        bulk = transform(channels, mode)
        for z in range(len(image)):
            one = torch.stack((torch.from_numpy(padded[z]), torch.from_numpy(padded[z+1]),
                               torch.from_numpy(padded[z+2])))[None]
            assert torch.equal(transform(one, mode)[0], bulk[z])
    result = {'completed_utc': datetime.now(timezone.utc).isoformat(), 'all_passed': True,
              'algebra_identity_max_error': identity_error,
              'component_decomposition_max_error': decomposition_error,
              'full_wrapper_prediction_max_error': float(np.max(np.abs(p-q))),
              'centre_unchanged': True, 'repeated_input_negative_control': True,
              'train_inference_boundary_channels_identical': True,
              'initialization_matches_previous': True, 'gradient_checks': gradient_checks,
              'protocol_sha256': sha(ROOT/'configs/mechanism.json'),
              'source_sha256': sources(['check_mechanism.py', 'mechanism_common.py', 'models.py', 'experiment.py']),
              'device': torch.cuda.get_device_name(), 'report_generated': False}
    (OUT/'pipeline_checks.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__': main()
