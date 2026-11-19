"""Independent checks of pairing, native masks, distances, and factorial estimates."""
from datetime import datetime, timezone
import hashlib
import json

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from continuation_common import ROOT, OUT, META, CFG, BASE, OLD, all_evaluations, new_trainings, run_dir, sha


def main():
    split = json.loads((ROOT/'data/metadata/split.json').read_text())['splits']
    fresh = json.loads((META/'split.json').read_text())['test']
    follow = json.loads((ROOT/'data/metadata/followup_context/split.json').read_text())['test']
    assert len(set(fresh)) == 20 and not set(fresh) & set(sum(split.values(), [])+follow)
    fresh_audit = json.loads((META/'preprocessing_audit.json').read_text())
    assert {r['case'] for r in fresh_audit} == set(fresh)
    for row in fresh_audit:
        assert sha(ROOT/row['factor4_file']) == row['factor4_sha256']
    protocol_hash = sha(ROOT/'configs/continuation.json')
    legacy = json.loads((ROOT/'results/original_experiment_fingerprints.json').read_text())
    for name, info in legacy['files'].items(): assert sha(ROOT/name) == info['sha256']
    records, train_records, native_files = [], [], set()
    for arm, seed, steps in all_evaluations():
        run = run_dir(arm, seed, steps)
        completed = json.loads((run/'evaluation_complete.json').read_text())
        assert completed['protocol_sha256'] == protocol_hash
        assert sha(ROOT/completed['checkpoint']) == completed['checkpoint_sha256']
        for name, digest in completed['reused_metric_sha256'].items(): assert sha(ROOT/name) == digest
        for name, digest in completed['source_sha256'].items(): assert sha(ROOT/'src'/name) == digest
        frame = pd.read_csv(run/'metrics.csv')
        for cohort, ids in [('fresh20', fresh), ('original40_exploratory', split['test'])]:
            sub = frame[frame.cohort == cohort]
            assert set(sub.case) == set(ids)
            assert len(sub) == len(ids)*len(CFG['evaluation_conditions'][cohort])*4
            for factor in CFG['evaluation_conditions'][cohort]:
                part = sub[sub.factor == factor]
                assert len(part) == len(ids)*4 and set(part.case) == set(ids)
            assert not sub.duplicated(['case', 'factor', 'organ']).any()
        for path in frame.prediction_file.unique():
            assert (ROOT/path).is_file()
            native_files.add(path)
        source = ROOT/'results/runs'/f'{OLD[arm]}_seed{seed}' if steps == 20000 and arm in OLD else run
        meta = json.loads((source/'metadata.json').read_text())
        finish = json.loads((source/'complete.json').read_text())
        assert meta['train_ids'] == split['train'] and meta['val_ids'] == split['val']
        assert meta['config']['steps'] == steps
        assert finish['training_steps'] == steps
        validations = [json.loads(l) for l in (source/'validation.jsonl').read_text().splitlines()]
        assert [r['step'] for r in validations] == list(range(2000, steps+1, 2000))
        for row in validations:
            assert {c['case'] for c in row['cases']} == set(split['val'])
            assert np.isclose(row['macro_dice'], np.mean([c['macro_dice'] for c in row['cases']]), atol=1e-14)
        winner = max(validations, key=lambda x: x['macro_dice'])
        assert winner['step'] == finish['best_step'] == completed['checkpoint_step']
        if (arm, seed, steps) in new_trainings():
            assert meta['protocol_sha256'] == protocol_hash
            for name, digest in meta['source_sha256'].items(): assert sha(ROOT/'src'/name) == digest
            assert finish['best_checkpoint_sha256'] == completed['checkpoint_sha256']
        train_records.append({'arm': arm, 'seed': seed, 'budget': steps, 'metadata': meta, 'complete': finish, 'source': source})
        records.append(frame)
    raw = pd.concat(records, ignore_index=True)
    assert len(raw) == 19200 and len(native_files) == 4800
    assert not raw.duplicated(['budget', 'arm', 'seed', 'cohort', 'case', 'factor', 'organ']).any()
    for steps in [20000, 40000]:
        for seed in CFG['seeds']:
            subset = [r for r in train_records if r['budget'] == steps and r['seed'] == seed]
            for family in ['planar', '3d']:
                group = [r for r in subset if ('3d' if BASE[r['arm']] == '3d' else 'planar') == family]
                if group: assert len({r['metadata']['initial_parameter_sha256'] for r in group}) == 1
            schedules = [pd.read_json(r['source']/'train.jsonl', lines=True) for r in subset]
            baseline = schedules[0][['step', 'patient']]
            assert all(t[['step', 'patient']].equals(baseline) for t in schedules)
            if steps == 40000:
                assert len({r['complete']['sampling_sha256'] for r in subset}) == 1
                augmented = [r for r in subset if r['arm'].endswith('_aug')]
                assert len({json.dumps(r['complete']['factor_counts'], sort_keys=True) for r in augmented}) == 1
                factor_rng = np.random.default_rng(seed+100000)
                values = factor_rng.choice([1, 2, 3], size=steps, p=[.5, .25, .25])
                expected = {str(f): int((values == f).sum()) for f in [1, 2, 3]}
                assert all(r['complete']['factor_counts'] == expected for r in augmented)
    for seed in CFG['seeds']:
        for arm in ['repeat', 'context', 'repeat_aug', 'context_aug']:
            short = next(r for r in train_records if r['arm'] == arm and r['seed'] == seed and r['budget'] == 20000)
            long = next(r for r in train_records if r['arm'] == arm and r['seed'] == seed and r['budget'] == 40000)
            assert short['metadata']['initial_parameter_sha256'] == long['metadata']['initial_parameter_sha256']
            a = pd.read_json(short['source']/'train.jsonl', lines=True)
            b = pd.read_json(long['source']/'train.jsonl', lines=True)
            b = b[b.step <= 20000].reset_index(drop=True)
            assert a[['step', 'patient']].equals(b[['step', 'patient']])
            if arm == 'repeat_aug':
                assert a.start.tolist() == b.start.tolist() and a.factor.tolist() == b.factor.tolist()
    # Independently reconstruct the four-arm primary contrast from organ rows.
    differences = []
    seed_differences = {s: [] for s in CFG['seeds']}
    for case in sorted(fresh):
        seeds = []
        for seed in CFG['seeds']:
            values = {}
            for arm in ['repeat', 'context', 'repeat_aug', 'context_aug']:
                row = raw[(raw.case == case) & (raw.seed == seed) & (raw.arm == arm) &
                          (raw.budget == 40000) & (raw.factor == 3)]
                assert len(row) == 4
                refpresent = row[row.true_voxels > 0].dice.to_numpy()
                values[arm] = float(refpresent.sum()/len(refpresent))
            effect = values['context_aug']-values['repeat_aug']-values['context']+values['repeat']
            seeds.append(effect)
            seed_differences[seed].append(effect)
        differences.append(float(np.mean(seeds)))
    differences = np.array(differences)
    rng = np.random.default_rng(CFG['primary']['bootstrap_seed'])
    bootstrap = rng.choice(differences, size=(10000, len(differences)), replace=True).mean(1)
    primary = json.loads((OUT/'analysis/analysis.json').read_text())['primary']
    assert np.isclose(differences.mean(), primary['mean'], rtol=0, atol=1e-12)
    assert np.allclose(np.quantile(bootstrap, [.025, .975]), [primary['ci_low'], primary['ci_high']], rtol=0, atol=1e-12)
    assert np.isclose(min(np.mean(v) for v in seed_differences.values()), primary['seed_min'], atol=1e-12)
    # Two predetermined patients, all models and conditions on fresh patient,
    # plus the native condition on the old patient; selection is performance-free.
    cases = [('fresh20', sorted(fresh)[0], [1, 2, 3, 4]),
             ('original40_exploratory', sorted(split['test'])[0], [1])]
    organs = [('spleen', 1), ('right_kidney', 2), ('left_kidney', 3), ('liver', 6)]
    mask_checks, dice_error, hd95_errors = 0, 0.0, []
    for cohort, case, factors in cases:
        meta = json.loads((ROOT/'data/processed'/case/'metadata.json').read_text())
        ref = nib.as_closest_canonical(nib.load(ROOT/meta['raw_label']))
        reference = np.asarray(ref.dataobj, dtype=np.uint8)
        spacing = np.array(ref.header.get_zooms()[:3], dtype=float)
        subset = raw[(raw.cohort == cohort) & (raw.case == case) & raw.factor.isin(factors)]
        for (budget, arm, seed, factor), group in subset.groupby(['budget', 'arm', 'seed', 'factor']):
            path = group.prediction_file.iloc[0]
            img = nib.load(ROOT/path)
            assert img.shape == ref.shape and np.allclose(img.affine, ref.affine, rtol=0, atol=1e-5)
            assert img.header.get_xyzt_units()[0] == 'mm'
            labels = np.asarray(img.dataobj, dtype=np.uint8)
            assert set(np.unique(labels)) <= set(range(5))
            for output_id, (organ, reference_id) in enumerate(organs, 1):
                p, t = labels == output_id, reference == reference_id
                row = group[group.organ == organ].iloc[0]
                assert int(p.sum()) == row.pred_voxels and int(t.sum()) == row.true_voxels
                if t.any():
                    independent_dice = 2*np.count_nonzero(p & t)/(p.sum()+t.sum())
                    dice_error = max(dice_error, abs(independent_dice-row.dice))
                else:
                    assert np.isnan(row.dice)
                    assert np.isclose(p.sum()*np.prod(spacing)/1000, row.absent_reference_false_positive_ml)
                if cohort == 'fresh20' and budget == 40000 and seed == 11 and (
                        (arm == 'repeat' and factor == 1) or (arm == 'volume_aug' and factor == 4)) and p.any() and t.any():
                    ps = np.argwhere(p & ~ndi.binary_erosion(p))*spacing
                    ts = np.argwhere(t & ~ndi.binary_erosion(t))*spacing
                    distances = np.concatenate([cKDTree(ts).query(ps, workers=2)[0],
                                                cKDTree(ps).query(ts, workers=2)[0]])
                    hd95_errors.append(abs(float(np.quantile(distances, .95))-row.hd95_mm))
            mask_checks += 1
    assert mask_checks == 150 and dice_error < 1e-12 and max(hd95_errors) < 1e-6
    assert not (ROOT/'reports').exists() and not (ROOT/'GroupProject_Review_Bundle.zip').exists()
    result = {'verified_utc': datetime.now(timezone.utc).isoformat(), 'all_passed': True,
              'evaluations': len(records), 'new_fits': len(new_trainings()), 'raw_rows': len(raw),
              'native_predictions_referenced': len(native_files), 'new_native_predictions': 4080,
              'fresh_patients_disjoint_from_prior_180': True,
              'all_fresh_factor4_input_hashes_match_pre_evaluation_audit': True,
              'retained_original_experiment_snapshot_unchanged': True,
              'initializations_and_full_sampling_traces_match': True,
              'short_long_budgets_have_matching_logged_sampling_prefixes': True,
              'all_development_checkpoint_selections_recomputed': True,
              'primary_patient_contrast_and_bootstrap_independently_recomputed': True,
              'native_masks_independently_checked': mask_checks,
              'maximum_dice_disagreement': dice_error,
              'independent_kdtree_hd95_checks': len(hd95_errors),
              'maximum_hd95_disagreement_mm': max(hd95_errors),
              'report_outputs_absent': True}
    (OUT/'verification.json').write_text(json.dumps(result, indent=2))
    manifest = {}
    paths = list((OUT/'analysis').glob('*.csv'))+list((OUT/'analysis').glob('*.json'))
    paths += [OUT/'verification.json', ROOT/'configs/continuation.json', ROOT/'configs/continuation_analysis.json']
    paths += list((ROOT/'src').glob('*continuation*.py'))
    paths += [ROOT/'src/diagnose_inputs.py', ROOT/'src/intervene_context.py', ROOT/'configs/context_interventions.json']
    paths += [ROOT/'src/diagnose_kidney_labels.py']
    paths += [OUT/'preprocessing_cache_checks.json']
    paths += [ROOT/'results/original_experiment_fingerprints.json']
    paths += list((OUT/'input_diagnostics').glob('*.csv'))+list((OUT/'input_diagnostics').glob('*.json'))
    paths += list((OUT/'kidney_label_diagnostics').glob('*.csv'))+list((OUT/'kidney_label_diagnostics').glob('*.json'))
    for job in all_evaluations(): paths += [run_dir(*job)/'metrics.csv', run_dir(*job)/'evaluation_complete.json']
    for job in new_trainings(): paths += [run_dir(*job)/'best.pt', run_dir(*job)/'metadata.json', run_dir(*job)/'complete.json']
    for path in paths: manifest[str(path.relative_to(ROOT))] = {'bytes': path.stat().st_size, 'sha256': sha(path)}
    (OUT/'artifact_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
