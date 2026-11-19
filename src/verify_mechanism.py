"""Independent provenance, native-mask and primary-statistic verification."""
import concurrent.futures as cf
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zlib

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/mechanism'
ARMS = ['slope', 'curvature', 'slope_aug', 'curvature_aug']
NAMES = ['background', 'spleen', 'right_kidney', 'left_kidney', 'liver']


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def target(case):
    meta = json.loads((ROOT/'data/processed'/case/'metadata.json').read_text())
    source = nib.as_closest_canonical(nib.load(ROOT/meta['raw_label']))
    raw = np.asarray(source.dataobj, dtype=np.uint8)
    y = np.zeros(raw.shape, dtype=np.uint8)
    for old, new in [(1, 1), (2, 2), (3, 3), (6, 4)]: y[raw == old] = new
    return y, meta


def check_header(item):
    path, case = item
    meta = json.loads((ROOT/'data/processed'/case/'metadata.json').read_text())
    image = nib.load(ROOT/path)
    assert list(image.shape) == meta['original_shape_xyz']
    assert np.allclose(image.affine, meta['original_affine'], atol=1e-4)
    assert image.header.get_xyzt_units()[0] == 'mm'
    assert image.get_data_dtype() == np.dtype('uint8')
    return True


def check_mask(item):
    path, case, rows = item
    y, meta = target(case)
    p = np.asarray(nib.load(ROOT/path).dataobj, dtype=np.uint8)
    maximum = 0.
    for row in rows:
        k = NAMES.index(row['organ'])
        nt, npred = int(np.count_nonzero(y == k)), int(np.count_nonzero(p == k))
        tp = int(np.count_nonzero((y == k) & (p == k)))
        value = 2*tp/(nt+npred) if nt else np.nan
        assert nt == row['true_voxels'] and npred == row['pred_voxels']
        assert np.isclose(value, row['dice'], atol=1e-12, rtol=0, equal_nan=True)
        if nt: maximum = max(maximum, abs(value-row['dice']))
    return maximum


def check_hd95(frame):
    case, path = frame.case.iloc[0], frame.prediction_file.iloc[0]
    y, meta = target(case)
    p = np.asarray(nib.load(ROOT/path).dataobj, dtype=np.uint8)
    spacing = np.array(meta['original_spacing_xyz_mm'])
    results = []
    for organ in ['spleen', 'right_kidney']:
        row = frame[frame.organ == organ].iloc[0]
        k = NAMES.index(organ)
        a, b = p == k, y == k
        if not a.any() or not b.any(): continue
        structure = ndi.generate_binary_structure(3, 1)
        aa = np.argwhere(a ^ ndi.binary_erosion(a, structure=structure, border_value=0))*spacing
        bb = np.argwhere(b ^ ndi.binary_erosion(b, structure=structure, border_value=0))*spacing
        first = cKDTree(bb).query(aa, workers=1)[0]
        second = cKDTree(aa).query(bb, workers=1)[0]
        value = float(np.percentile(np.r_[first, second], 95))
        error = abs(value-row.hd95_mm)
        assert error < 1e-6
        results.append({'case': case, 'organ': organ, 'prediction_file': path, 'error_mm': error})
    return results


def main():
    analysis = json.loads((OUT/'analysis/analysis.json').read_text())
    assert (OUT/'sampling_phase/analysis.json').exists() and (OUT/'absolute_spacing/analysis.json').exists()
    old_snapshot = json.loads((OUT/'pre_experiment_snapshot.json').read_text())
    for path, digest in old_snapshot['files'].items(): assert sha(ROOT/path) == digest, path
    training_rows = []
    for arm in ARMS:
        for seed in [11, 22, 33]:
            run = OUT/'runs'/f'{arm}_seed{seed}'
            record = json.loads((run/'complete.json').read_text())
            metadata = json.loads((run/'metadata.json').read_text())
            prior_run = ROOT/'results/continuation/b40000'/f'{"repeat_aug" if arm.endswith("_aug") else "repeat"}_seed{seed}'
            prior = json.loads((prior_run/'complete.json').read_text())
            prior_meta = json.loads((prior_run/'metadata.json').read_text())
            assert record['training_steps'] == 40000 and metadata['parameters'] == 1964165
            assert metadata['initial_parameter_sha256'] == prior_meta['initial_parameter_sha256']
            assert record['sampling_sha256'] == prior['sampling_sha256']
            assert record['factor_counts'] == prior['factor_counts']
            assert metadata['train_ids'] == prior_meta['train_ids'] and metadata['val_ids'] == prior_meta['val_ids']
            for name, digest in metadata['source_sha256'].items(): assert sha(ROOT/'src'/name) == digest
            assert metadata['protocol_sha256'] == sha(ROOT/'configs/mechanism.json')
            trace = pd.read_json(run/'train.jsonl', lines=True)
            oldtrace = pd.read_json(prior_run/'train.jsonl', lines=True)
            for field in ['step', 'patient', 'factor', 'lr']:
                assert trace[field].tolist() == oldtrace[field].tolist(), (arm, seed, field)
            assert trace.start.tolist() == oldtrace.start.tolist()
            validation = [json.loads(line) for line in (run/'validation.jsonl').read_text().splitlines()]
            assert len(validation) == 20 and all(len(v['cases']) == 20 for v in validation)
            best = max(validation, key=lambda v: (v['macro_dice'], -v['step']))
            assert best['step'] == record['best_step'] and best['macro_dice'] == record['best_validation_macro_dice']
            assert sha(run/'best.pt') == record['best_checkpoint_sha256']
            training_rows.append({'arm': arm, 'seed': seed, 'best_step': record['best_step']})
    # Recheck the exact newly downloaded bytes against independent official ZIP CRCs.
    split = json.loads((ROOT/'data/metadata/mechanism_replication/split.json').read_text())
    prior_ids = set(split['excluded_prior_ids'])
    assert len(prior_ids) == 200 and len(split['test']) == 40 and not set(split['test']) & prior_ids
    index = {r['name']: r for r in json.loads((ROOT/'data/metadata/archive_index.json').read_text())}
    checksums = json.loads((ROOT/'data/metadata/mechanism_replication/download_checksums.json').read_text())
    assert len(checksums) == 80
    for row in checksums:
        data = (ROOT/'data/raw'/row['member']).read_bytes()
        assert hashlib.sha256(data).hexdigest() == row['sha256']
        assert len(data) == index[row['member']]['size']
        assert zlib.crc32(data) & 0xffffffff == index[row['member']]['crc32']
    for path, digest in split['primary_checkpoint_sha256'].items(): assert sha(ROOT/path) == digest
    processed = json.loads((ROOT/'data/metadata/mechanism_replication/processed_inputs_sha256.json').read_text())
    for path, digest in processed['files'].items(): assert sha(ROOT/path) == digest
    evaluation_records = []
    for folder, marker, protocol in [
        (OUT/'runs', 'evaluation_complete.json', 'mechanism.json'),
        (OUT/'replication/runs', 'complete.json', 'mechanism_replication.json'),
        (OUT/'sampling_phase/runs', 'complete.json', 'sampling_phase.json'),
        (OUT/'absolute_spacing/runs', 'complete.json', 'absolute_spacing.json')]:
        for path in sorted(folder.glob('*/'+marker)):
            record = json.loads(path.read_text())
            assert record['protocol_sha256'] == sha(ROOT/'configs'/protocol)
            for name, digest in record['source_sha256'].items(): assert sha(ROOT/'src'/name) == digest
            evaluation_records.append(record)
    assert len(evaluation_records) == 60
    for folder in ['sampling_phase', 'absolute_spacing']:
        for row in json.loads((ROOT/'data/metadata'/folder/'input_audit.json').read_text()):
            assert sha(ROOT/row['path']) == row['sha256']
    # Verify all prediction paths and headers; select fixed first/last cases for full-mask checks.
    main_raw = pd.read_csv(OUT/'analysis/all_organ_metrics.csv')
    phase_raw = pd.read_csv(OUT/'sampling_phase/all_organ_metrics.csv')
    absolute_raw = pd.read_csv(OUT/'absolute_spacing/all_organ_metrics.csv')
    new_main = main_raw[main_raw.prediction_file.str.startswith('results/mechanism/')]
    new_phase = phase_raw[phase_raw.phase > 0]
    all_new = pd.concat([new_main, new_phase, absolute_raw], ignore_index=True)
    files = all_new[['prediction_file', 'case']].drop_duplicates()
    assert len(files) == 4584 and len(all_new) == 4584*4
    assert all((ROOT/p).exists() for p in files.prediction_file)
    with cf.ThreadPoolExecutor(8) as pool:
        assert all(pool.map(check_header, files.itertuples(index=False, name=None)))
    samples = []
    for _, group in new_main[new_main.factor == 3].groupby(['cohort', 'arm', 'seed']):
        cases = sorted(group.case.unique())
        samples.extend(group[group.case == c] for c in [cases[0], cases[-1]])
    for _, group in new_phase.groupby(['arm', 'seed', 'phase']):
        samples.append(group[group.case == sorted(group.case.unique())[0]])
    for _, group in absolute_raw.groupby(['arm', 'seed', 'target_sampling_mm']):
        samples.append(group[group.case == sorted(group.case.unique())[0]])
    tasks = [(g.prediction_file.iloc[0], g.case.iloc[0], g.to_dict('records')) for g in samples]
    with cf.ProcessPoolExecutor(4) as pool:
        mask_errors = list(pool.map(check_mask, tasks))
    distances = []
    for source in [new_main[new_main.cohort == 'prior_fresh20'], new_main[new_main.cohort == 'new40_replication'],
                   new_phase, absolute_raw]:
        path = sorted(source.prediction_file.unique())[0]
        distances.extend(check_hd95(source[source.prediction_file == path]))
    # Reconstruct the new primary directly from organ rows, independently of the analysis helpers.
    sub = main_raw[(main_raw.cohort == 'new40_replication') & (main_raw.factor == 3) &
                   main_raw.arm.isin(['repeat', 'context', 'repeat_aug', 'context_aug'])]
    values = []
    for case, rows in sub.groupby('case', sort=True):
        paired = []
        for seed in [11, 22, 33]:
            r = rows[rows.seed == seed]
            scores = {arm: float(r[r.arm == arm].dice.mean()) for arm in ['repeat', 'context', 'repeat_aug', 'context_aug']}
            paired.append(scores['context_aug']-scores['repeat_aug']-scores['context']+scores['repeat'])
        values.append(float(np.mean(paired)))
    values = np.asarray(values); assert len(values) == 40
    rng = np.random.default_rng(2026111504)
    bootstrap = values[rng.integers(40, size=(10000, 40))].mean(1)
    low, high = np.quantile(bootstrap, [.025, .975])
    primary = analysis['replication_primary']
    assert abs(values.mean()-primary['mean']) < 1e-12
    assert abs(low-primary['ci_low']) < 1e-12 and abs(high-primary['ci_high']) < 1e-12
    # Existing summaries must be unaffected by importing/using their model scores.
    old = pd.read_csv(ROOT/'results/continuation/analysis/model_summary.csv')
    patient = pd.read_csv(OUT/'analysis/patient_metrics.csv')
    for _, row in old[(old.budget == 40000) & old.arm.isin(['repeat', 'context', 'repeat_aug', 'context_aug']) & old.factor.isin([1, 3])].iterrows():
        cohort = 'prior_fresh20' if row.cohort == 'fresh20' else row.cohort
        value = patient[(patient.cohort == cohort) & (patient.arm == row.arm) & (patient.factor == row.factor)].dice.mean()
        assert abs(value-row.dice) < 1e-12
    assert json.loads((OUT/'sampling_geometry_checks.json').read_text())['all_passed']
    assert not (ROOT/'reports').exists()
    result = {'verified_utc': datetime.now(timezone.utc).isoformat(), 'all_passed': True,
              'new_training_runs': len(training_rows), 'new_training_updates': 480000,
              'new_native_predictions_checked': len(files), 'native_masks_independently_recomputed': len(tasks),
              'maximum_native_dice_error': max(mask_errors), 'independent_kdtree_hd95_checks': distances,
              'new_raw_members_crc32_sha256_rechecked': len(checksums), 'new_patients_disjoint_from_prior_200': True,
              'new40_processed_inputs_unchanged_since_pre_evaluation_freeze': True,
              'evaluation_code_and_protocol_records_verified': len(evaluation_records),
              'original_source_config_analysis_snapshot_unchanged': len(old_snapshot['files']),
              'initializations_sampling_augmentation_schedules_match': True,
              'all_checkpoint_selections_recomputed': True,
              'new40_primary_and_bootstrap_independently_recomputed': True,
              'replication_primary': primary, 'training_checks': training_rows,
              'report_outputs_absent': True, 'source_sha256': sha(__file__)}
    (OUT/'verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ['training_checks', 'independent_kdtree_hd95_checks']}, indent=2), flush=True)


if __name__ == '__main__': main()
