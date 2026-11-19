"""Check added augmentation support without changing the supervised patch trace."""
from datetime import datetime, timezone
import json

import numpy as np
import torch

from mechanism_common import ROOT, sha
from width_budget_common import OUT, WideAugSampler
from experiment import SlabSampler
from context_followup import WideSampler, wide_input


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ids = json.loads((ROOT/'data/metadata/split.json').read_text())['splits']['train'][:2]
    for augmented in [False, True]:
        narrow = SlabSampler(ids, 11, [32, 128, 128], augmentation=augmented)
        wide = WideAugSampler(ids, 11, [32, 128, 128], augmentation=augmented)
        old_wide = WideSampler(ids, 11, [32, 128, 128]) if not augmented else None
        for _ in range(100):
            a, y, i, start = narrow.next(); b, yy, j, ss = wide.next()
            assert i == j and np.array_equal(start, ss) and np.array_equal(y, yy)
            assert np.array_equal(a, b[2:-2]) and b.shape[0] == 38
            if old_wide:
                c, yc, ic, sc = old_wide.next()
                assert np.array_equal(b, c) and np.array_equal(yy, yc) and j == ic and np.array_equal(ss, sc)
    x = torch.arange(38)[:, None, None].expand(-1, 8, 8)
    encoded = wide_input(x)
    assert torch.equal(encoded[:, 0, 0, 0], torch.arange(32))
    assert torch.equal(encoded[:, 1, 0, 0], torch.arange(3, 35))
    assert torch.equal(encoded[:, 2, 0, 0], torch.arange(6, 38))
    (OUT/'pipeline_checks.json').write_text(json.dumps({'completed_utc': datetime.now(timezone.utc).isoformat(),
        'all_passed': True, 'matched_slabs_targets_intensities_with_and_without_augmentation': 100,
        'unaugmented_sampler_matches_prior_wide_exactly': True, 'offset_encoding_mm': 9,
        'existing_prediction_implementation_reused': 'context_followup.predict_wide',
        'protocol_sha256': sha(ROOT/'configs/width_budget.json'), 'source_sha256': sha(__file__),
        'report_generated': False}, indent=2))
    print('Width-budget input checks passed', flush=True)


if __name__ == '__main__': main()
