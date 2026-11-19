"""Compare anatomy and CT intensities actually supplied by the five slice policies."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_data import NAMES


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/slice_selection/input_diagnostic'


def measure_case(case, policies):
    folder = ROOT / 'data/processed' / case
    label = np.load(folder / 'label.npy', mmap_mode='r')
    z, y, x = np.nonzero(label[3:-3])
    z += 3
    centre_labels = label[z, y, x]
    rows = []
    for factor in [1, 3]:
        image = np.load(folder / f'image_f{factor}.npy', mmap_mode='r')
        assert image.shape == label.shape
        centre = image[z, y, x].astype(np.float32)
        for policy, (lo, _, hi) in policies.items():
            left = image[z + lo, y, x].astype(np.float32)
            right = image[z + hi, y, x].astype(np.float32)
            same_left = centre_labels == label[z + lo, y, x]
            same_right = centre_labels == label[z + hi, y, x]
            # Distance-weighted linear interpolation at the known centre position.
            reconstructed = (hi * left - lo * right) / (hi - lo)
            for k, organ in enumerate(NAMES[1:], 1):
                mask = centre_labels == k
                if not mask.any():
                    continue
                matched_left, matched_right = same_left[mask], same_right[mask]
                rows.append(dict(
                    case=case, factor=factor, policy=policy, organ=organ,
                    central_voxels=int(mask.sum()),
                    both_same_organ=np.mean(matched_left & matched_right),
                    any_same_organ=np.mean(matched_left | matched_right),
                    mean_neighbor_abs_hu=250 * (
                        np.abs(left[mask] - centre[mask]).mean(dtype=np.float64)
                        + np.abs(right[mask] - centre[mask]).mean(dtype=np.float64)),
                    interpolated_centre_abs_error_hu=500 * np.abs(
                        reconstructed[mask] - centre[mask]).mean(dtype=np.float64)))
    return rows


def main():
    cfg = json.loads((ROOT / 'configs/slice_selection.json').read_text())
    cases = json.loads((ROOT / cfg['test_split']).read_text())['test']
    policies = {'context': [-1, 0, 1], **cfg['offsets']}
    rows = [row for case in cases for row in measure_case(case, policies)]
    frame = pd.DataFrame(rows)
    keys = ['case', 'factor', 'policy', 'organ']
    assert set(frame.case) == set(cases) and not frame.duplicated(keys).any()
    assert frame.groupby(['case', 'organ']).size().eq(2 * len(policies)).all()
    for col in ['both_same_organ', 'any_same_organ']:
        wide = frame.pivot(index=['case', 'organ', 'policy'], columns='factor', values=col)
        assert np.array_equal(wide[1], wide[3])
    assert frame.both_same_organ.between(0, 1).all()
    assert (frame.both_same_organ <= frame.any_same_organ).all() and (frame.any_same_organ <= 1).all()
    metrics = ['both_same_organ', 'any_same_organ', 'mean_neighbor_abs_hu', 'interpolated_centre_abs_error_hu']
    patient = frame.groupby(['case', 'policy', 'factor'])[metrics].mean()
    summary = patient.groupby(['policy', 'factor']).mean()
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / 'organ_by_patient.csv', index=False)
    summary.to_csv(OUT / 'policy_means.csv')
    (OUT / 'definitions.json').write_text(json.dumps(dict(
        patients=len(cases), policies=policies, factors=[1, 3],
        cohort=cfg['test_split'], rows=len(rows), spacing_mm=3,
        population='Reference foreground on common grid. Drop three outermost axial planes at each end for all policies.',
        aggregation='Within each patient and present organ, average over centre voxels. Give present organs equal weight within each patient, then average patients.',
        same_organ='At the same in-plane location on the common-grid reference label, both or at least one of the two input neighbours has the centre organ label.',
        intensity='HU after the fixed [-200, 300] clipping and float16 storage. Linear centre interpolation weights are proportional to distance from the opposite neighbour.',
        purpose='Descriptive input diagnostic on previously observed patients. No model predictions are used.',
    ), indent=2))
    print(summary.round(4).to_string())


if __name__ == '__main__':
    main()
