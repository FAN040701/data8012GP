"""All-patient native-grid error anatomy, plus frozen illustrative case selection.

No model fitting, no report, no exclusions or changes to the prior primary result.
"""
import concurrent.futures as cf
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import nibabel as nib
import numpy as np
import pandas as pd

from prepare_data import NAMES, remap_labels

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/mechanism/case_study'
SOURCE = ROOT/'results/continuation/analysis/all_organ_metrics.csv'
ARMS = ['repeat', 'context', 'repeat_aug', 'context_aug']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def select_cases(raw):
    recent = raw[(raw.cohort == 'fresh20') & (raw.factor == 3)]
    wide = recent.pivot(index=['seed', 'case', 'organ'], columns='arm', values='dice')
    delta = wide.context_aug-wide.repeat_aug-wide.context+wide.repeat
    organ = delta.groupby(['case', 'organ']).mean().rename('organ_interaction').reset_index()
    patient = organ.groupby('case').organ_interaction.mean().rename('patient_interaction').reset_index()
    selections, chosen = [], set()
    order = {name: i for i, name in enumerate(NAMES)}
    for group, ordered in [
        ('largest_positive', patient.sort_values(['patient_interaction', 'case'], ascending=[False, True])),
        ('most_negative', patient.sort_values(['patient_interaction', 'case'])),
        ('closest_to_zero', patient.assign(absolute=patient.patient_interaction.abs()).sort_values(['absolute', 'case']))]:
        added = 0
        for row in ordered.to_dict('records'):
            case = row['case']
            if case in chosen: continue
            chosen.add(case); added += 1
            organs = organ[organ.case == case].copy()
            organs['absolute'] = organs.organ_interaction.abs()
            organs['organ_order'] = organs.organ.map(order)
            which = organs.sort_values(['absolute', 'organ_order'], ascending=[False, True]).iloc[0]
            selections.append({'case': case, 'selection_group': group,
                               'patient_interaction': float(row['patient_interaction']),
                               'organ': which.organ, 'organ_interaction': float(which.organ_interaction),
                               'visual_seed': 11, 'factor': 3})
            if added == 2: break
    record = {'selected_before_image_inspection_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'Post hoc illustrations, not an unbiased case sample',
              'source_sha256': sha(SOURCE), 'protocol_sha256': sha(ROOT/'configs/mechanism.json'),
              'selection': selections, 'visual_slices': 'Largest reference area and largest four-arm total FP+FN; ties lowest z'}
    path = OUT/'case_selection.json'
    if path.exists():
        previous = json.loads(path.read_text())
        assert previous['selection'] == selections
    else: path.write_text(json.dumps(record, indent=2))
    organ.to_csv(OUT/'all_fresh20_organ_interactions.csv', index=False)
    patient.to_csv(OUT/'all_fresh20_patient_interactions.csv', index=False)
    return selections


def analyze_patient(item):
    cohort, case, records = item
    meta = json.loads((ROOT/'data/processed'/case/'metadata.json').read_text())
    reference = nib.as_closest_canonical(nib.load(ROOT/meta['raw_label']))
    target = remap_labels(np.asarray(reference.dataobj, dtype=np.uint8))
    spacing = np.asarray(meta['original_spacing_xyz_mm'])
    voxel_ml = float(np.prod(spacing)/1000)
    depth = target.shape[2]
    truth_z = np.stack([np.count_nonzero(target == k, axis=(0, 1)) for k in range(1, 5)])
    records = pd.DataFrame(records)
    native_rows, slice_rows = [], []
    for (arm, seed, factor), group in records.groupby(['arm', 'seed', 'factor']):
        path = ROOT/group.prediction_file.iloc[0]
        volume = nib.load(path)
        assert volume.shape == target.shape and np.allclose(volume.affine, reference.affine, atol=1e-4)
        pred = np.asarray(volume.dataobj, dtype=np.uint8)
        assert pred.max() <= 4
        by_z = np.stack([np.bincount((target[:, :, z]*5+pred[:, :, z]).ravel(), minlength=25).reshape(5, 5)
                         for z in range(depth)])
        assert by_z.sum() == target.size
        matrix = by_z.sum(0)
        for k in range(1, 5):
            name = NAMES[k]
            true_count = int(matrix[k].sum()); pred_count = int(matrix[:, k].sum())
            tp = int(matrix[k, k]); fn, fp = true_count-tp, pred_count-tp
            dice = 2*tp/(true_count+pred_count) if true_count else np.nan
            old = group[group.organ == name].iloc[0]
            assert int(old.true_voxels) == true_count and int(old.pred_voxels) == pred_count
            assert np.isclose(dice, old.dice, atol=1e-12, rtol=0, equal_nan=True)
            present = np.flatnonzero(truth_z[k-1])
            assert len(present), 'This fixed cohort has all four reference organs present'
            lo, hi = int(present[0]), int(present[-1])
            zpos = np.arange(depth)
            regions = np.where(zpos < lo+(hi-lo+1)/3, 'inferior',
                               np.where(zpos < lo+2*(hi-lo+1)/3, 'middle', 'superior'))
            tp_z = by_z[:, k, k]
            fn_z = by_z[:, k].sum(1)-tp_z
            fp_z = by_z[:, :, k].sum(1)-tp_z
            row = {'cohort': cohort, 'case': case, 'arm': arm, 'seed': int(seed), 'factor': int(factor),
                   'organ': name, 'native_z_spacing_mm': float(spacing[2]), 'dice': dice,
                   'true_voxels': true_count, 'pred_voxels': pred_count, 'tp_voxels': tp,
                   'fn_voxels': fn, 'fp_voxels': fp, 'fn_ml': fn*voxel_ml, 'fp_ml': fp*voxel_ml,
                   'reference_ml': true_count*voxel_ml,
                   'fn_fraction_of_reference': fn/true_count,
                   'fp_fraction_of_reference': fp/true_count,
                   'recall': tp/true_count, 'precision': tp/pred_count if pred_count else np.nan,
                   'signed_volume_error': (pred_count-true_count)/true_count,
                   'fn_to_background_voxels': int(matrix[k, 0]),
                   'fn_to_other_organ_voxels': int(matrix[k, 1:].sum()-tp),
                   'fp_from_background_voxels': int(matrix[0, k]),
                   'fp_from_other_organ_voxels': int(matrix[1:, k].sum()-tp),
                   'reference_inferior_z': lo, 'reference_superior_z': hi,
                   'prediction_file': str(path.relative_to(ROOT))}
            for region in ['inferior', 'middle', 'superior']:
                mask = regions == region
                row[f'{region}_fn_voxels'] = int(fn_z[mask].sum())
                row[f'{region}_fp_voxels'] = int(fp_z[mask].sum())
                row[f'{region}_reference_voxels'] = int(truth_z[k-1, mask].sum())
            assert sum(row[f'{r}_fn_voxels'] for r in ['inferior', 'middle', 'superior']) == fn
            assert sum(row[f'{r}_fp_voxels'] for r in ['inferior', 'middle', 'superior']) == fp
            native_rows.append(row)
            if seed == 11 and factor == 3:
                for z in np.flatnonzero(truth_z[k-1]+fp_z):
                    slice_rows.append({'cohort': cohort, 'case': case, 'arm': arm, 'seed': int(seed),
                                       'factor': int(factor), 'organ': name, 'z': int(z),
                                       'reference_voxels': int(truth_z[k-1, z]), 'tp_voxels': int(tp_z[z]),
                                       'fn_voxels': int(fn_z[z]), 'fp_voxels': int(fp_z[z]),
                                       'region': str(regions[z])})
    folder = OUT/'per_patient'; folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(native_rows).to_csv(folder/f'{case}_native.csv', index=False)
    pd.DataFrame(slice_rows).to_csv(folder/f'{case}_slices.csv', index=False)
    return {'case': case, 'cohort': cohort, 'organ_rows': len(native_rows), 'predictions': len(records)//4,
            'native_grid_dice_matches_prior': True}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(SOURCE)
    raw = raw[(raw.budget == 40000) & raw.arm.isin(ARMS) & raw.factor.isin([1, 3])].copy()
    assert len(raw) == 60*4*3*2*4
    select_cases(raw)
    jobs = [(cohort, case, sub.to_dict('records')) for (cohort, case), sub in raw.groupby(['cohort', 'case'])]
    started, statuses = time.time(), []
    with cf.ProcessPoolExecutor(4) as pool:
        tasks = [pool.submit(analyze_patient, job) for job in jobs]
        for task in cf.as_completed(tasks):
            statuses.append(task.result())
            print('error anatomy', len(statuses), '/', len(jobs), statuses[-1]['case'], flush=True)
    native = pd.concat([pd.read_csv(OUT/'per_patient'/f'{case}_native.csv') for _, case, _ in jobs])
    slices = pd.concat([pd.read_csv(OUT/'per_patient'/f'{case}_slices.csv') for _, case, _ in jobs])
    assert len(native) == len(raw)
    native.to_csv(OUT/'native_error_metrics.csv', index=False)
    slices.to_csv(OUT/'slice_error_metrics.csv', index=False)
    result = {'completed_utc': datetime.now(timezone.utc).isoformat(), 'patients': len(jobs),
              'predictions': len(raw)//4, 'organ_rows': len(native), 'slice_rows': len(slices),
              'all_native_dice_and_voxel_counts_match_previous': True,
              'source_metrics_sha256': sha(SOURCE), 'script_sha256': sha(__file__),
              'protocol_sha256': sha(ROOT/'configs/mechanism.json'),
              'elapsed_seconds': time.time()-started, 'statuses': statuses, 'report_generated': False}
    (OUT/'quantification_complete.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != 'statuses'}, indent=2), flush=True)


if __name__ == '__main__': main()
