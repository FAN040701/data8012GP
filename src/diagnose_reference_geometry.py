"""Descriptive label-geometry/sampling diagnostics; no anatomical diagnosis."""
import concurrent.futures as cf
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from prepare_data import NAMES, remap_labels

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/mechanism/reference_geometry'


def one(item):
    cohort, case = item
    meta = json.loads((ROOT/'data/processed'/case/'metadata.json').read_text())
    image = nib.as_closest_canonical(nib.load(ROOT/meta['raw_label']))
    y = remap_labels(np.asarray(image.dataobj, dtype=np.uint8))
    spacing = np.asarray(meta['original_spacing_xyz_mm'])
    rows = []
    for k, organ in enumerate(NAMES[1:], 1):
        counts = np.count_nonzero(y == k, axis=(0, 1))
        positions = np.flatnonzero(counts)
        if not len(positions): continue
        total = int(counts.sum())
        extent = (positions[-1]-positions[0]+1)*spacing[2]
        row = {'cohort': cohort, 'case': case, 'organ': organ,
               'reference_voxels': total, 'reference_ml': total*np.prod(spacing)/1000,
               'native_z_spacing_mm': spacing[2], 'reference_axial_extent_mm': extent,
               'reference_native_axial_planes': len(positions),
               'f3_extent_over_sampling_distance': extent/(3*spacing[2]),
               'reference_at_first_scan_slice': bool(counts[0]),
               'reference_at_last_scan_slice': bool(counts[-1]),
               'label_nearest_neighbor_roundtrip_dice': meta['reference_roundtrip_dice'][organ]}
        for phase in [0, 1, 2]:
            retained = counts[phase::3]
            row[f'f3_phase{phase}_reference_nonempty_retained_planes'] = int(np.count_nonzero(retained))
            row[f'f3_phase{phase}_unsmoothed_label_sampling_volume_bias'] = 3*int(retained.sum())/total-1
        rows.append(row)
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    original = json.loads((ROOT/'data/metadata/split.json').read_text())['splits']['test']
    recent = json.loads((ROOT/'data/metadata/continuation/split.json').read_text())['test']
    jobs = [('original40_exploratory', c) for c in original]+[('fresh20', c) for c in recent]
    with cf.ProcessPoolExecutor(4) as pool:
        rows = [r for part in pool.map(one, jobs) for r in part]
    geometry = pd.DataFrame(rows)
    geometry.to_csv(OUT/'reference_geometry.csv', index=False)
    metrics = pd.read_csv(ROOT/'results/mechanism/case_study/native_error_metrics.csv')
    per_case = metrics.groupby(['cohort', 'case', 'organ', 'arm', 'factor']).dice.mean().unstack('factor')
    per_case['f3_minus_f1'] = per_case[3]-per_case[1]
    per_case = per_case.reset_index().rename(columns={1: 'f1_dice', 3: 'f3_dice'})
    joined = per_case.merge(geometry, on=['cohort', 'case', 'organ'], validate='many_to_one')
    joined.to_csv(OUT/'geometry_and_model_outcomes.csv', index=False)
    correlations = []
    for (cohort, organ, arm), sub in joined.groupby(['cohort', 'organ', 'arm']):
        for feature in ['reference_ml', 'reference_axial_extent_mm', 'f3_extent_over_sampling_distance',
                        'label_nearest_neighbor_roundtrip_dice']:
            for outcome in ['f3_dice', 'f3_minus_f1']:
                rho = spearmanr(sub[feature], sub[outcome]).statistic
                correlations.append({'cohort': cohort, 'organ': organ, 'arm': arm,
                                     'feature': feature, 'outcome': outcome, 'patients': len(sub),
                                     'spearman_rho': float(rho)})
    pd.DataFrame(correlations).to_csv(OUT/'descriptive_correlations.csv', index=False)
    selected = json.loads((ROOT/'results/mechanism/case_study/case_selection.json').read_text())['selection']
    examples = []
    for choice in selected:
        row = geometry[(geometry.case == choice['case']) & (geometry.organ == choice['organ'])].iloc[0]
        peers = geometry[(geometry.cohort == row.cohort) & (geometry.organ == row.organ)]
        examples.append({**choice, **row.to_dict(),
                         'reference_volume_rank_smallest_first': int((peers.reference_ml < row.reference_ml).sum())+1,
                         'same_cohort_same_organ_patients': len(peers)})
    pd.DataFrame(examples).to_csv(OUT/'preselected_case_geometry.csv', index=False)
    # Show how equal-organ and volume-weighted metrics weight these same errors.
    # Neither alternative replaces the original equal-organ patient-macro endpoint.
    weighted = metrics.groupby(['cohort', 'case', 'arm', 'seed', 'factor']).agg(
        tp=('tp_voxels', 'sum'), reference=('true_voxels', 'sum'), prediction=('pred_voxels', 'sum'),
        macro_dice=('dice', 'mean')).reset_index()
    weighted['foreground_micro_dice'] = 2*weighted.tp/(weighted.reference+weighted.prediction)
    metric_rows = []
    for metric in ['macro_dice', 'foreground_micro_dice']:
        wide = weighted.pivot(index=['cohort', 'case', 'seed', 'factor'], columns='arm', values=metric)
        value = wide.context_aug-wide.repeat_aug-wide.context+wide.repeat
        for (cohort, factor), values in value.groupby(['cohort', 'factor']):
            metric_rows.append({'cohort': cohort, 'factor': int(factor), 'metric': metric,
                                'mean_context_x_augmentation': float(values.mean()),
                                'scope': 'Descriptive change of weighting, not a replacement primary endpoint or a significance test'})
    pd.DataFrame(metric_rows).to_csv(OUT/'metric_weighting_sensitivity.csv', index=False)
    (OUT/'complete.json').write_text(json.dumps({'completed_utc': datetime.now(timezone.utc).isoformat(),
        'patients': len(jobs), 'reference_organ_rows': len(geometry),
        'limitations': [
            'Small reference-label volume does not establish actual anatomical absence, pathology or annotation error.',
            'Nearest-neighbor label round-trip Dice is a discretization diagnostic, not an upper bound on model Dice.',
            'Unsmoothed label sampling bias describes geometry only; CT images were Gaussian-filtered before sampling.',
            'Correlations are post hoc, descriptive and potentially confounded by acquisition and case mix; no causal or multiplicity-adjusted claim.'
        ], 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'report_generated': False}, indent=2))
    print(pd.DataFrame(examples)[['case', 'organ', 'reference_ml', 'reference_axial_extent_mm',
                                'f3_extent_over_sampling_distance', 'reference_volume_rank_smallest_first']].to_string(index=False), flush=True)


if __name__ == '__main__': main()
