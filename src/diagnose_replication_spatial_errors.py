"""Localize errors in all new40 primary predictions; retain the fixed primary endpoint."""
import concurrent.futures as cf
from datetime import datetime, timezone
import json

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from mechanism_common import ROOT, OUT as MAIN, sha
from mechanism_statistics import effect
from prepare_data import NAMES, remap_labels

OUT = MAIN/'replication_spatial_errors'
ARMS = ['repeat', 'context', 'repeat_aug', 'context_aug']
PARTS = ['fp_within_10mm', 'fp_beyond_10mm', 'fn_boundary_3mm', 'fn_deeper_3mm']


def one(item):
    case, source_rows, selected_organ = item
    meta = json.loads((ROOT/'data/processed'/case/'metadata.json').read_text())
    target = remap_labels(np.asarray(nib.as_closest_canonical(nib.load(ROOT/meta['raw_label'])).dataobj, dtype=np.uint8))
    spacing = np.array(meta['original_spacing_xyz_mm'])
    voxel_ml = float(np.prod(spacing)/1000)
    rois = {}
    for k, organ in enumerate(NAMES[1:], 1):
        mask = target == k
        coords = [np.flatnonzero(mask.any(axis=tuple(j for j in range(3) if j != axis))) for axis in range(3)]
        assert all(len(c) for c in coords)
        margin = np.ceil(10/spacing).astype(int)+1
        region = tuple(slice(max(0, int(c[0])-int(m)), min(n, int(c[-1])+int(m)+1))
                       for c, m, n in zip(coords, margin, target.shape))
        gt = mask[region].copy()
        surface = gt & ~ndi.binary_erosion(gt, structure=ndi.generate_binary_structure(3, 1), border_value=0)
        boundary_band = gt & (ndi.distance_transform_edt(~surface, sampling=spacing) <= 3+1e-8)
        near_outside = ~gt & (ndi.distance_transform_edt(~gt, sampling=spacing) <= 10+1e-8)
        rois[organ] = (k, region, gt, boundary_band, near_outside, int(gt.sum()))
    source = pd.DataFrame(source_rows)
    rows, slices = [], []
    for (arm, seed, path), group in source.groupby(['arm', 'seed', 'prediction_file']):
        pred = np.asarray(nib.load(ROOT/path).dataobj, dtype=np.uint8)
        assert pred.shape == target.shape
        counts = np.bincount(pred.ravel(), minlength=5)
        for row in group.to_dict('records'):
            organ = row['organ']; k, region, gt, band, near, nt = rois[organ]
            p = pred[region] == k
            tp, npred = int(np.count_nonzero(p & gt)), int(counts[k])
            fp, fn = npred-tp, nt-tp
            assert nt == row['true_voxels'] and npred == row['pred_voxels']
            assert abs(2*tp/(nt+npred)-row['dice']) < 1e-12
            fp_near = int(np.count_nonzero(p & near))
            fn_boundary = int(np.count_nonzero(~p & band))
            parts = {'fp_within_10mm': fp_near, 'fp_beyond_10mm': fp-fp_near,
                     'fn_boundary_3mm': fn_boundary, 'fn_deeper_3mm': fn-fn_boundary}
            assert min(parts.values()) >= 0 and sum(parts.values()) == fp+fn
            assert abs(sum(parts.values())/(nt+npred)-(1-row['dice'])) < 1e-12
            rows.append({**row, 'tp_voxels': tp, 'fp_voxels': fp, 'fn_voxels': fn,
                         'fp_ml': fp*voxel_ml, 'fn_ml': fn*voxel_ml, 'reference_ml': nt*voxel_ml,
                         'reference_boundary_3mm_voxels': int(band.sum()), **parts,
                         **{name+'_loss': count/(nt+npred) for name, count in parts.items()},
                         **{name+'_per_reference': count/nt for name, count in parts.items()}})
            if organ == selected_organ and seed == 11:
                g = target == k; full = pred == k
                fp_z = np.count_nonzero(full & ~g, axis=(0, 1))
                fn_z = np.count_nonzero(~full & g, axis=(0, 1))
                nt_z = np.count_nonzero(g, axis=(0, 1))
                slices.extend({'case': case, 'organ': organ, 'arm': arm, 'seed': int(seed), 'factor': 3,
                               'z': z, 'fp_voxels': int(fp_z[z]), 'fn_voxels': int(fn_z[z]),
                               'reference_voxels': int(nt_z[z])} for z in range(target.shape[2]))
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT/'per_patient'/f'{case}_native.csv', index=False)
    if selected_organ is not None: pd.DataFrame(slices).to_csv(OUT/'per_patient'/f'{case}_slices.csv', index=False)
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True); (OUT/'per_patient').mkdir(exist_ok=True)
    source_path = MAIN/'analysis/all_organ_metrics.csv'
    raw = pd.read_csv(source_path)
    raw = raw[(raw.cohort == 'new40_replication') & (raw.factor == 3) & raw.arm.isin(ARMS)].copy()
    assert len(raw) == 1920 and raw.case.nunique() == 40
    # Freeze illustrative extremes before reading their image appearance. Every case is quantified.
    organ = raw.groupby(['case', 'organ', 'arm']).dice.mean().unstack('arm')
    organ['interaction'] = organ.context_aug-organ.repeat_aug-organ.context+organ.repeat
    patient = organ.interaction.groupby('case').mean().sort_values()
    selection = []
    for group, ids in [('extreme_negative', list(patient.index[:2])), ('extreme_positive', list(patient.index[-2:]))]:
        for case in ids:
            values = organ.xs(case).interaction
            selected = values.idxmin() if group == 'extreme_negative' else values.idxmax()
            selection.append({'case': case, 'organ': selected, 'selection_group': group,
                              'patient_interaction': float(patient.loc[case]), 'organ_interaction': float(values.loc[selected])})
    chosen = {r['case']: r['organ'] for r in selection}
    freeze = OUT/'case_selection.json'
    expected = {'selection': selection,
                'criterion': 'Two most negative and two most positive patient interactions; organ with the largest contribution in the matching direction; fixed visual seed 11. Illustrative extremes, not representative frequency estimates.',
                'all_40_patients_quantified': True, 'source_metrics_sha256': sha(source_path)}
    if freeze.exists():
        old = json.loads(freeze.read_text()); assert old['selection'] == selection
    else:
        freeze.write_text(json.dumps({'frozen_utc': datetime.now(timezone.utc).isoformat(), **expected}, indent=2))
    protocol = OUT/'diagnostic_semantics.json'
    if not protocol.exists():
        protocol.write_text(json.dumps({'frozen_utc': datetime.now(timezone.utc).isoformat(),
            'scope': 'Post hoc descriptive diagnostic, complete new40 primary f3 predictions, all four arms and all three seeds.',
            'false_positive_distance': 'Euclidean distance between native-grid voxel centres and the nearest reference-positive voxel centre. Near <=10 mm; far >10 mm.',
            'false_negative_distance': 'Euclidean distance to reference boundary voxel centres; boundary uses 6-connected binary erosion with exterior background. Boundary band <=3 mm; deeper >3 mm.',
            'thresholds': 'Fixed diagnostic definitions, not clinical boundaries or optimized thresholds.',
            'statistics': 'Patient macro, three-seed averages, 10000 paired-patient bootstrap replicates with seed 2026111515. All intervals exploratory and unadjusted.',
            'limitations': 'Spatial compartments describe errors, not independent causal mechanisms. FP/(T+P) and FN/(T+P) use model-dependent denominators. Reference geometry does not establish pathology or annotation error.',
            'primary_unchanged': True, 'report_generated': False}, indent=2))
    jobs = [(case, rows.to_dict('records'), chosen.get(case)) for case, rows in raw.groupby('case', sort=True)]
    values = []
    with cf.ProcessPoolExecutor(4) as pool:
        for i, rows in enumerate(pool.map(one, jobs), 1):
            values.extend(rows); print('Native spatial error checks', i, '/40', flush=True)
    frame = pd.DataFrame(values); assert len(frame) == 1920
    frame.to_csv(OUT/'native_spatial_errors.csv', index=False)
    metrics = ['dice']+[p+'_loss' for p in PARTS]+[p+'_per_reference' for p in PARTS]
    patient = frame.groupby(['case', 'arm', 'seed'], as_index=False)[metrics].mean()
    weights = {'context_without_aug': {'context': 1, 'repeat': -1},
               'context_with_aug': {'context_aug': 1, 'repeat_aug': -1},
               'augmentation_context': {'context_aug': 1, 'context': -1},
               'augmentation_repeat': {'repeat_aug': 1, 'repeat': -1},
               'context_x_augmentation': {'context_aug': 1, 'repeat_aug': -1, 'context': -1, 'repeat': 1}}
    effects, paired = [], []
    for metric in metrics:
        wide = patient.pivot(index=['seed', 'case'], columns='arm', values=metric)
        for name, w in weights.items():
            value = sum(wide[a]*v for a, v in w.items())
            if metric.endswith('_loss'): value = -value
            stats, matrix = effect(value, 2026111515)
            effects.append({'metric': metric, 'contrast': name, **stats})
            paired.extend({'metric': metric, 'contrast': name, 'case': c, 'difference': float(v)}
                          for c, v in matrix.mean(axis=1).items())
    effects, paired = pd.DataFrame(effects), pd.DataFrame(paired)
    effects.to_csv(OUT/'paired_spatial_contrasts.csv', index=False)
    paired.to_csv(OUT/'patient_spatial_contrasts.csv', index=False)
    for contrast in weights:
        s = paired[paired.contrast == contrast].pivot(index='case', columns='metric', values='difference')
        assert np.max(np.abs(s.dice-s[[p+'_loss' for p in PARTS]].sum(axis=1))) < 1e-12
    primary = json.loads((MAIN/'analysis/analysis.json').read_text())['replication_primary']['mean']
    observed = effects[(effects.metric == 'dice') & (effects.contrast == 'context_x_augmentation')]['mean'].iloc[0]
    assert abs(primary-observed) < 1e-12
    # Reuse the existing validated plotting routine with only the output/data directory changed.
    import plot_case_errors
    plot_case_errors.OUT = OUT
    plot_case_errors.main()
    (OUT/'complete.json').write_text(json.dumps({'completed_utc': datetime.now(timezone.utc).isoformat(),
        'all_passed': True, 'patients': 40, 'native_predictions_independently_recounted': 480,
        'organ_rows': len(frame), 'illustrative_extreme_patients': 4,
        'all_spatial_partitions_sum_to_original_error_counts': True,
        'all_native_dice_values_and_primary_mean_reproduced': True,
        'new_training_runs': 0, 'new_native_predictions': 0, 'source_metrics_sha256': sha(source_path),
        'source_sha256': sha(__file__), 'plot_source_sha256': sha(ROOT/'src/plot_case_errors.py'),
        'diagnostic_semantics_sha256': sha(protocol), 'report_generated': False}, indent=2))
    print('Spatial error diagnosis and case figures complete', flush=True)


if __name__ == '__main__': main()
