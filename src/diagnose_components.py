"""Quantify the actual retained input components and transformation range effects."""
import concurrent.futures as cf
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_data import NAMES

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/mechanism/input_components'


def one(item):
    cohort, case, factors = item
    folder = ROOT/'data/processed'/case
    label = np.load(folder/'label.npy')
    meta = json.loads((folder/'metadata.json').read_text())
    rows = []
    for factor in factors:
        image = np.load(folder/f'image_f{factor}.npy').astype(np.float32)
        left, centre, right = image[:-2], image[1:-1], image[2:]
        slope, curvature = (right-left)*.5, (left+right)*.5-centre
        projected_left, projected_right = centre-slope, centre+slope
        masks = {'unsaturated_centre': (centre > 0) & (centre < 1),
                 'reference_foreground': label[1:-1] > 0}
        masks.update({name: label[1:-1] == k for k, name in enumerate(NAMES[1:], 1)})
        for region, mask in masks.items():
            count = int(mask.sum())
            if not count: continue
            s, k = slope[mask].astype(np.float64), curvature[mask].astype(np.float64)
            p, q = projected_left[mask], projected_right[mask]
            s2, k2 = float(np.mean(s*s)), float(np.mean(k*k))
            excursion = np.concatenate((np.maximum(-p, 0)+np.maximum(p-1, 0),
                                        np.maximum(-q, 0)+np.maximum(q-1, 0)))
            rows.append({'cohort': cohort, 'case': case, 'factor': factor, 'region': region,
                         'native_z_spacing_mm': meta['original_spacing_xyz_mm'][2],
                         'effective_sampling_mm': meta['original_spacing_xyz_mm'][2]*factor,
                         'sampled_pixels': count, 'slope_rms': s2**.5, 'curvature_rms': k2**.5,
                         'curvature_energy_fraction': k2/(s2+k2) if s2+k2 else 0,
                         'near_zero_curvature_fraction': float(np.mean(np.abs(k) <= .001)),
                         'slope_neighbor_out_of_range_fraction': float(np.mean(excursion > 1e-7)),
                         'slope_neighbor_mean_excursion': float(excursion.mean()),
                         'slope_neighbor_max_excursion': float(excursion.max()),
                         'slope_projection_min': float(min(p.min(), q.min())),
                         'slope_projection_max': float(max(p.max(), q.max()))})
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    split = json.loads((ROOT/'data/metadata/split.json').read_text())['splits']
    recent = json.loads((ROOT/'data/metadata/continuation/split.json').read_text())['test']
    jobs = [('fitting80', case, [1, 2, 3]) for case in split['train']]
    jobs += [('original40_exploratory', case, [1, 3]) for case in split['test']]
    jobs += [('prior_fresh20', case, [1, 3, 4]) for case in recent]
    rows = []
    with cf.ProcessPoolExecutor(4) as pool:
        for i, values in enumerate(pool.map(one, jobs), 1):
            rows.extend(values)
            if i % 10 == 0: print('Input components', i, '/', len(jobs), flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT/'patient_input_components.csv', index=False)
    metrics = ['slope_rms', 'curvature_rms', 'curvature_energy_fraction', 'near_zero_curvature_fraction',
               'slope_neighbor_out_of_range_fraction', 'slope_neighbor_mean_excursion',
               'slope_neighbor_max_excursion']
    frame.groupby(['cohort', 'factor', 'region'])[metrics].mean().reset_index().to_csv(
        OUT/'component_summary.csv', index=False)
    (OUT/'complete.json').write_text(json.dumps({'completed_utc': datetime.now(timezone.utc).isoformat(),
        'patients': len(jobs), 'rows': len(frame), 'scope': 'Interior axial positions, actual f1/f2/f3/f4 input arrays before mild random training intensity changes',
        'limitations': 'Component energy is not independent information content. Out-of-range slopes are retained deliberately; representation and optimization effects remain possible.',
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'report_generated': False}, indent=2))


if __name__ == '__main__': main()
