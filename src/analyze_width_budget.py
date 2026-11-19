"""Matched width/budget/augmentation analysis and independent artifact checks."""
from datetime import datetime, timezone
import concurrent.futures as cf
import json

import numpy as np
import pandas as pd

from mechanism_common import ROOT, sha
from width_budget_common import OUT, CFG, run_dir
from mechanism_statistics import interval, effect
from verify_mechanism import check_header, check_mask


def main():
    assert not json.loads((OUT/'evaluate_queue_complete.json').read_text())['errors']
    frames, evaluation_records, training_records = [], [], []
    for arm in ['wide', 'wide_aug', 'context', 'context_aug']:
        for seed in CFG['seeds']:
            run = run_dir(arm, seed)
            record = json.loads((run/'evaluation_complete.json').read_text())
            assert record['protocol_sha256'] == sha(ROOT/'configs/width_budget.json')
            for name, digest in record['source_sha256'].items(): assert sha(ROOT/'src'/name) == digest
            assert sha(ROOT/record['checkpoint']) == record['checkpoint_sha256']
            for path, digest in record['reused_metrics_sha256'].items(): assert sha(ROOT/path) == digest
            frame = pd.read_csv(run/'metrics.csv'); assert len(frame) == 640
            frames.append(frame); evaluation_records.append(record)
            if arm.startswith('wide'):
                metadata = json.loads((run/'metadata.json').read_text())
                complete = json.loads((run/'complete.json').read_text())
                prior_arm = 'context_aug' if arm.endswith('_aug') else 'context'
                prior = ROOT/'results/continuation/b40000'/f'{prior_arm}_seed{seed}'
                original = json.loads((prior/'metadata.json').read_text())
                original_complete = json.loads((prior/'complete.json').read_text())
                assert metadata['parameters'] == 1964165 and complete['training_steps'] == 40000
                assert metadata['initial_parameter_sha256'] == original['initial_parameter_sha256']
                assert complete['sampling_sha256'] == original_complete['sampling_sha256']
                assert complete['factor_counts'] == original_complete['factor_counts']
                assert metadata['train_ids'] == original['train_ids'] and metadata['val_ids'] == original['val_ids']
                assert complete['best_checkpoint_sha256'] == sha(run/'best.pt')
                for name, digest in metadata['source_sha256'].items(): assert sha(ROOT/'src'/name) == digest
                trace, old = pd.read_json(run/'train.jsonl', lines=True), pd.read_json(prior/'train.jsonl', lines=True)
                for field in ['step', 'patient', 'factor', 'lr', 'start']: assert trace[field].tolist() == old[field].tolist()
                validation = [json.loads(line) for line in (run/'validation.jsonl').read_text().splitlines()]
                assert len(validation) == 20 and all(len(v['cases']) == 20 for v in validation)
                best = max(validation, key=lambda v: (v['macro_dice'], -v['step']))
                assert best['step'] == complete['best_step'] and best['macro_dice'] == complete['best_validation_macro_dice']
                training_records.append({'arm': arm, 'seed': seed, 'best_step': complete['best_step']})
    source = ROOT/'results/followup_context/analysis/all_organ_metrics.csv'
    short = pd.read_csv(source)
    short = short[short.kind.isin(['narrow', 'wide']) & short.factor.isin([1, 3])].copy()
    short['prediction_file'] = [str((ROOT/'results/followup_context'/f'{k}_seed{s}'/'predictions'/f'f{f}'/f'{c}.nii.gz').relative_to(ROOT))
                                for k, s, f, c in zip(short.kind, short.seed, short.factor, short.case)]
    short['arm'] = short.kind.replace({'narrow': 'context'})
    short['budget'], short['cohort'] = 20000, 'previous_width40'
    frames.append(short)
    raw = pd.concat(frames, ignore_index=True)
    assert len(raw) == 9600 and not raw.duplicated(['cohort', 'budget', 'arm', 'seed', 'case', 'factor', 'organ']).any()
    raw.to_csv(OUT/'all_organ_metrics.csv', index=False)
    patient = raw.groupby(['cohort', 'budget', 'arm', 'seed', 'case', 'factor'], as_index=False).agg(
        dice=('dice', 'mean'), hd95_mm=('hd95_mm', 'mean'), reference_present_organs=('dice', 'count'))
    assert patient.groupby(['cohort', 'case', 'factor']).reference_present_organs.nunique().max() == 1
    patient.to_csv(OUT/'patient_metrics.csv', index=False)
    summaries = []
    for (cohort, budget, arm, factor), sub in patient.groupby(['cohort', 'budget', 'arm', 'factor']):
        mean, lo, hi = interval(sub.groupby('case').dice.mean().sort_index(), 2026111509)
        seeds = sub.groupby('seed').dice.mean()
        summaries.append({'cohort': cohort, 'budget': int(budget), 'arm': arm, 'factor': int(factor),
                          'dice': mean, 'ci_low': lo, 'ci_high': hi, 'hd95_mm': float(sub.hd95_mm.mean()),
                          'seed_min': float(seeds.min()), 'seed_max': float(seeds.max())})
    pd.DataFrame(summaries).to_csv(OUT/'model_summary.csv', index=False)
    weights = {'width_without_aug': {'wide': 1, 'context': -1},
               'width_with_aug': {'wide_aug': 1, 'context_aug': -1},
               'augmentation_narrow': {'context_aug': 1, 'context': -1},
               'augmentation_wide': {'wide_aug': 1, 'wide': -1},
               'width_x_augmentation': {'wide_aug': 1, 'context_aug': -1, 'wide': -1, 'context': 1}}
    effects, paired = [], []
    def add(series, meta):
        stats, matrix = effect(series, 2026111509)
        effects.append({**meta, **stats})
        paired.extend({**meta, 'case': case, 'difference': float(v)} for case, v in matrix.mean(1).items())
    for (cohort, budget), sub in patient.groupby(['cohort', 'budget']):
        wide = sub.pivot(index=['seed', 'case', 'factor'], columns='arm', values='dice')
        for name, w in weights.items():
            if not set(w) <= set(wide.columns): continue
            delta = sum(wide[a]*v for a, v in w.items())
            for factor in [1, 3]:
                add(delta.xs(factor, level='factor'), {'cohort': cohort, 'budget': int(budget), 'contrast': name, 'condition': f'f{factor}'})
            add(delta.xs(3, level='factor')-delta.xs(1, level='factor'),
                {'cohort': cohort, 'budget': int(budget), 'contrast': name, 'condition': 'f3_minus_f1'})
    sub = patient[(patient.cohort == 'previous_width40') & patient.arm.isin(['context', 'wide'])]
    wide = sub.pivot(index=['seed', 'case', 'factor', 'arm'], columns='budget', values='dice')
    difference = wide[40000]-wide[20000]
    for factor in [1, 3]:
        for arm in ['context', 'wide']:
            add(difference.xs((factor, arm), level=('factor', 'arm')),
                {'cohort': 'previous_width40', 'budget': 40000, 'contrast': 'budget_'+arm, 'condition': f'f{factor}'})
        interaction = difference.xs((factor, 'wide'), level=('factor', 'arm'))-difference.xs((factor, 'context'), level=('factor', 'arm'))
        add(interaction, {'cohort': 'previous_width40', 'budget': 40000,
                          'contrast': 'budget_modification_of_width_gain', 'condition': f'f{factor}'})
    table = pd.DataFrame(effects)
    table.to_csv(OUT/'paired_contrasts.csv', index=False)
    pd.DataFrame(paired).to_csv(OUT/'paired_patient_differences.csv', index=False)
    old = pd.read_csv(ROOT/'results/followup_context/analysis/paired_contrasts.csv')
    for factor in [1, 3]:
        expected = old[(old.contrast == 'wide_minus_narrow') & (old.condition == f'f{factor}')]['mean'].iloc[0]
        observed = table[(table.budget == 20000) & (table.contrast == 'width_without_aug') & (table.condition == f'f{factor}')]['mean'].iloc[0]
        assert abs(expected-observed) < 1e-12
    new = raw[raw.prediction_file.str.startswith('results/mechanism/width_budget/')]
    files = new[['prediction_file', 'case']].drop_duplicates()
    assert len(files) == 1440
    with cf.ThreadPoolExecutor(8) as pool: assert all(pool.map(check_header, files.itertuples(index=False, name=None)))
    samples = []
    for _, group in new[new.factor == 3].groupby(['cohort', 'arm', 'seed']):
        ids = sorted(group.case.unique())
        samples.extend(group[group.case == case] for case in [ids[0], ids[-1]])
    with cf.ProcessPoolExecutor(4) as pool:
        errors = list(pool.map(check_mask, [(s.prediction_file.iloc[0], s.case.iloc[0], s.to_dict('records')) for s in samples]))
    result = {'completed_utc': datetime.now(timezone.utc).isoformat(), 'all_passed': True,
              'new_training_runs': 6, 'training_updates': 240000, 'evaluated_run_configurations': 12,
              'new_native_predictions': 1440, 'native_masks_independently_checked': len(samples),
              'maximum_dice_error': max(errors), 'all_initializations_sampling_and_augmentation_traces_match': True,
              'checkpoint_selection_recomputed': True, 'prior_short_budget_results_unchanged': True,
              'training_records': training_records, 'protocol_sha256': sha(ROOT/'configs/width_budget.json'),
              'all_intervals_exploratory': True, 'report_generated': False}
    (OUT/'analysis_and_verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__': main()
