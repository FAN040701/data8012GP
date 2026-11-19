"""Check the new factorial encoding and shared sampling on fitting cases only."""
import hashlib
import json

import numpy as np
import torch

from continuation_common import ROOT, OUT, CFG, BASE, new_trainings
from experiment import SlabSampler, init_runtime, make_logits
from models import UNet, volume_loss


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    split = json.loads((ROOT/'data/metadata/split.json').read_text())['splits']
    ids = split['train'][:2]
    original = SlabSampler(ids, 11, CFG['patch'])
    aug_a = SlabSampler(ids, 11, CFG['patch'], True)
    aug_b = SlabSampler(ids, 11, CFG['patch'], True)
    factors = np.random.default_rng(100011)
    counts = {1: 0, 2: 0, 3: 0}
    for _ in range(100):
        h, y, idx, start = original.next()
        ha, ya, ia, sa = aug_a.next()
        hb, yb, ib, sb = aug_b.next()
        factor = int(factors.choice([1, 2, 3], p=[.5, .25, .25]))
        counts[factor] += 1
        assert idx == ia == ib and np.array_equal(start, sa) and np.array_equal(sa, sb)
        assert np.array_equal(y, ya) and np.array_equal(ya, yb) and np.array_equal(ha, hb)
        if factor == 1: assert np.array_equal(h, ha)
    hashes = {}
    for arm, base in BASE.items():
        init_runtime(11)
        model = UNet(base, CFG['width']).cuda()
        digest = hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()
        family = '3d' if base == '3d' else 'planar'
        if family in hashes: assert hashes[family] == digest
        hashes[family] = digest
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = make_logits(model, torch.from_numpy(ha).cuda(), base)
            assert logits.shape == (1, 5, *CFG['patch'])
            loss = volume_loss(logits, torch.from_numpy(ya)[None].cuda())
        loss.backward()
        assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        del model, logits, loss
    rows = {'fitting_cases_only': ids, 'matched_sampling_and_labels_100_draws': True,
            'matched_augmented_inputs_100_draws': True,
            'native_draws_exactly_match_unaugmented_inputs': counts[1],
            'six_arm_finite_loss_and_gradient': True,
            'matched_initial_hashes_by_family': hashes,
            'planned_new_training_runs': len(new_trainings()), 'no_test_inference': True}
    (OUT/'pipeline_checks.json').write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))


if __name__ == '__main__': main()
