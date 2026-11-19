"""Audit the deliberately unclipped slope projection over the full model field."""
import concurrent.futures as cf
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/mechanism/input_components'


def one(item):
    cohort, case = item
    folder = ROOT/'data/processed'/case
    label = np.load(folder/'label.npy')
    rows = []
    for factor in [1, 3]:
        image = np.load(folder/f'image_f{factor}.npy').astype(np.float32)
        padded = np.pad(image, ((1, 1), (0, 0), (0, 0)))
        slope = (padded[2:]-padded[:-2])*.5
        previous, following = image-slope, image+slope
        excursions = np.maximum(-previous, 0)+np.maximum(previous-1, 0)+np.maximum(-following, 0)+np.maximum(following-1, 0)
        count = ((previous < -1e-7) | (previous > 1+1e-7)).astype(np.uint8)
        count += ((following < -1e-7) | (following > 1+1e-7)).astype(np.uint8)
        masks = {'all_model_fov': np.ones(image.shape, dtype=bool), 'reference_foreground': label > 0,
                 'saturated_centre_zero': image == 0, 'saturated_centre_one': image == 1}
        for region, mask in masks.items():
            n = int(mask.sum())
            if not n: continue
            rows.append({'cohort': cohort, 'case': case, 'factor': factor, 'region': region,
                         'centre_pixels': n, 'out_of_range_neighbour_fraction': float(count[mask].sum()/(2*n)),
                         'mean_excursion_per_neighbour': float(excursions[mask].sum()/(2*n)),
                         'minimum_projected_intensity': float(min(previous[mask].min(), following[mask].min())),
                         'maximum_projected_intensity': float(max(previous[mask].max(), following[mask].max()))})
    return rows


def main():
    split = json.loads((ROOT/'data/metadata/split.json').read_text())['splits']
    recent = json.loads((ROOT/'data/metadata/continuation/split.json').read_text())['test']
    jobs = [('fitting80', c) for c in split['train']]+[('original40_exploratory', c) for c in split['test']]+[('prior_fresh20', c) for c in recent]
    with cf.ProcessPoolExecutor(4) as pool: rows = [r for part in pool.map(one, jobs) for r in part]
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT/'slope_projection_full_field_range.csv', index=False)
    frame.groupby(['cohort', 'factor', 'region']).mean(numeric_only=True).reset_index().to_csv(
        OUT/'slope_projection_range_summary.csv', index=False)
    (OUT/'projection_range_complete.json').write_text(json.dumps({'completed_utc': datetime.now(timezone.utc).isoformat(),
        'patients': len(jobs), 'rows': len(frame), 'scope': 'Full actual common-grid FOV including zero-padded missing neighbours, before mild random training intensity augmentation; no additional clipping',
        'interpretation': 'This quantifies a representation-distribution caveat of the component intervention; small global fractions alone do not establish negligible performance effects.',
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'report_generated': False}, indent=2))


if __name__ == '__main__': main()
