"""Compute prespecified factorial statistics; write tables/JSON only, no report."""
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from continuation_common import ROOT, OUT, CFG, BASE, OLD, all_evaluations, new_trainings, run_dir, sha

AN = OUT/'analysis'
BOOT_SEED = CFG['primary']['bootstrap_seed']
CONTRASTS = {
    'context_without_aug': {'context': 1, 'repeat': -1},
    'context_with_aug': {'context_aug': 1, 'repeat_aug': -1},
    'augmentation_repeat': {'repeat_aug': 1, 'repeat': -1},
    'augmentation_context': {'context_aug': 1, 'context': -1},
    'context_x_augmentation': {'context_aug': 1, 'repeat_aug': -1, 'context': -1, 'repeat': 1},
    'augmentation_volume': {'volume_aug': 1, 'volume': -1},
    'volume_minus_context_without_aug': {'volume': 1, 'context': -1},
    'volume_minus_context_with_aug': {'volume_aug': 1, 'context_aug': -1},
}


def patient_interval(values):
    values = np.asarray(values, float)
    assert values.ndim == 1 and np.isfinite(values).all()
    rng = np.random.default_rng(BOOT_SEED)
    draws = values[rng.integers(len(values), size=(10000, len(values)))].mean(1)
    lo, hi = np.quantile(draws, [.025, .975])
    return float(values.mean()), float(lo), float(hi)


def effect_stats(series):
    """Input index is seed/case; paired differences retain both crossed axes."""
    mat = series.unstack('seed').sort_index().sort_index(axis=1)
    assert mat.shape[1] == 3 and not mat.isna().any().any()
    values = mat.to_numpy()
    mean, lo, hi = patient_interval(values.mean(1))
    rng = np.random.default_rng(BOOT_SEED)
    patients = rng.integers(len(values), size=(10000, len(values)))
    seeds = rng.integers(3, size=(10000, 3))
    crossed = values[patients[:, :, None], seeds[:, None, :]].mean((1, 2))
    seed_means = values.mean(0)
    low2, high2 = np.quantile(crossed, [.025, .975])
    return {'mean': mean, 'ci_low': lo, 'ci_high': hi,
            'crossed_bootstrap_ci_low': float(low2), 'crossed_bootstrap_ci_high': float(high2),
            'patients': len(values), 'seeds': 3,
            'seed_min': float(seed_means.min()), 'seed_max': float(seed_means.max()),
            'seed_sd': float(seed_means.std(ddof=1))}, mat


def main():
    AN.mkdir(parents=True, exist_ok=True)
    frames, provenance = [], []
    expected_protocol = sha(ROOT/'configs/continuation.json')
    for arm, seed, steps in all_evaluations():
        run = run_dir(arm, seed, steps)
        complete = json.loads((run/'evaluation_complete.json').read_text())
        assert complete['protocol_sha256'] == expected_protocol
        assert sha(ROOT/complete['checkpoint']) == complete['checkpoint_sha256']
        for name, digest in complete['source_sha256'].items(): assert sha(ROOT/'src'/name) == digest
        frame = pd.read_csv(run/'metrics.csv')
        assert len(frame) == 640 and not frame.duplicated(['cohort', 'case', 'factor', 'organ']).any()
        assert set(frame.arm) == {arm} and set(frame.budget) == {steps} and set(frame.seed) == {seed}
        frames.append(frame)
        provenance.append(complete)
    raw = pd.concat(frames, ignore_index=True)
    assert len(raw) == 19200
    assert raw[raw.true_voxels > 0].dice.notna().all()
    assert raw[raw.true_voxels == 0].dice.isna().all()
    raw.to_csv(AN/'all_organ_metrics.csv', index=False)
    keys = ['cohort', 'budget', 'arm', 'seed', 'case', 'factor']
    patient = raw.groupby(keys, as_index=False).agg(
        dice=('dice', 'mean'), hd95_mm=('hd95_mm', 'mean'),
        reference_present_organs=('dice', 'count'), hd95_available_organs=('hd95_mm', 'count'))
    expected_counts = patient.groupby(['cohort', 'case', 'factor']).reference_present_organs.nunique()
    assert expected_counts.max() == 1
    assert patient.dice.notna().all()
    patient.to_csv(AN/'patient_metrics.csv', index=False)
    seed_summary = patient.groupby(['cohort', 'budget', 'arm', 'seed', 'factor'], as_index=False)[['dice', 'hd95_mm']].mean()
    seed_summary.to_csv(AN/'seed_summary.csv', index=False)
    summaries = []
    for (cohort, budget, arm, factor), sub in patient.groupby(['cohort', 'budget', 'arm', 'factor']):
        mean, lo, hi = patient_interval(sub.groupby('case').dice.mean().sort_index())
        seedmeans = sub.groupby('seed').dice.mean()
        organ_rows = raw[(raw.cohort == cohort) & (raw.budget == budget) &
                         (raw.arm == arm) & (raw.factor == factor)]
        summaries.append({'cohort': cohort, 'budget': int(budget), 'arm': arm, 'factor': int(factor),
                          'dice': mean, 'ci_low': lo, 'ci_high': hi,
                          'hd95_mm': float(sub.hd95_mm.mean()),
                          'patients': sub.case.nunique(), 'seed_min': float(seedmeans.min()),
                          'seed_max': float(seedmeans.max()), 'seed_sd': float(seedmeans.std()),
                          'empty_predictions_reference_present': int((organ_rows.status == 'prediction_empty').sum()),
                          'reference_absent_case_organ_seed_rows': int((organ_rows.true_voxels == 0).sum())})
    pd.DataFrame(summaries).to_csv(AN/'model_summary.csv', index=False)
    effects, paired, seed_effects = [], [], []
    def add_effect(series, meta):
        stats, mat = effect_stats(series)
        effects.append({**meta, **stats})
        paired.extend({**meta, 'case': case, 'difference': float(value)} for case, value in mat.mean(1).items())
        seed_effects.extend({**meta, 'seed': int(seed), 'difference': float(value)} for seed, value in mat.mean(0).items())
    for (cohort, budget), sub in patient.groupby(['cohort', 'budget']):
        wide = sub.pivot(index=['seed', 'case', 'factor'], columns='arm', values='dice')
        for name, weights in CONTRASTS.items():
            if not set(weights) <= set(wide.columns): continue
            delta = sum(wide[a]*w for a, w in weights.items())
            for factor in sorted(sub.factor.unique()):
                add_effect(delta.xs(factor, level='factor'),
                           {'cohort': cohort, 'budget': int(budget), 'contrast': name,
                            'condition': f'f{factor}', 'metric': 'dice'})
            for factor in sorted(set(sub.factor)-{1}):
                add_effect(delta.xs(factor, level='factor')-delta.xs(1, level='factor'),
                           {'cohort': cohort, 'budget': int(budget), 'contrast': name,
                            'condition': f'f{factor}_minus_f1', 'metric': 'dice'})
    for (cohort, arm), sub in patient.groupby(['cohort', 'arm']):
        if sub.budget.nunique() != 2: continue
        wide = sub.pivot(index=['seed', 'case', 'factor'], columns='budget', values='dice')
        delta = wide[CFG['long_steps']]-wide[CFG['short_steps']]
        for factor in sorted(sub.factor.unique()):
            add_effect(delta.xs(factor, level='factor'),
                       {'cohort': cohort, 'budget': 40000, 'reference_budget': 20000, 'contrast': f'budget_{arm}',
                        'condition': f'f{factor}', 'metric': 'dice'})
    effects_frame = pd.DataFrame(effects)
    effects_frame.to_csv(AN/'paired_contrasts.csv', index=False)
    pd.DataFrame(paired).to_csv(AN/'paired_patient_differences.csv', index=False)
    pd.DataFrame(seed_effects).to_csv(AN/'paired_seed_differences.csv', index=False)
    primary = next(e for e in effects if e['cohort'] == 'fresh20' and e['budget'] == 40000
                   and e['contrast'] == 'context_x_augmentation' and e['condition'] == 'f3')
    organ = raw.groupby(['cohort', 'budget', 'arm', 'factor', 'organ'], as_index=False).agg(
        dice=('dice', 'mean'), hd95_mm=('hd95_mm', 'mean'), reference_present_rows=('dice', 'count'),
        false_positive_ml_reference_absent=('absent_reference_false_positive_ml', 'sum'))
    organ.to_csv(AN/'organ_summary.csv', index=False)
    learning = []
    for arm, seed, steps in all_evaluations():
        run = run_dir(arm, seed, steps)
        source = ROOT/'results/runs'/f'{OLD[arm]}_seed{seed}' if steps == 20000 and arm in OLD else run
        rows = [json.loads(l) for l in (source/'validation.jsonl').read_text().splitlines()]
        complete = json.loads((source/'complete.json').read_text())
        for row in rows:
            learning.append({'arm': arm, 'seed': seed, 'budget': steps, 'step': row['step'],
                             'development_dice': row['macro_dice'],
                             'selected': row['step'] == complete['best_step']})
    learning = pd.DataFrame(learning)
    learning.to_csv(AN/'development_learning_curves.csv', index=False)
    selected = learning[learning.selected]
    absent = raw[raw.true_voxels == 0][['cohort', 'case', 'organ']].drop_duplicates().to_dict('records')
    result = {'completed_utc': datetime.now(timezone.utc).isoformat(),
              'new_training_runs': len(new_trainings()), 'evaluated_run_configurations': len(provenance),
              'organ_rows': len(raw), 'primary': primary, 'summaries': summaries,
              'reference_absent_case_organs': absent,
              'empty_predictions_of_reference_present_organs': int((raw.status == 'prediction_empty').sum()),
              'final_checkpoint_selections': {str(b): int((g.step == b).sum()) for b, g in selected.groupby('budget')},
              'bootstrap': {'replicates': 10000, 'unit': 'patient', 'seed': BOOT_SEED,
                            'seed_aggregation': 'Mean of seed-specific patient metrics; predictions are not ensembled',
                            'primary_interval_conditions_on_three_trained_seeds': True,
                            'crossed_interval_is_exploratory_with_only_three_seed_clusters': True},
              'report_generated': False}
    (AN/'analysis.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({'primary': primary, 'evaluations': len(provenance), 'organ_rows': len(raw)}, indent=2))


if __name__ == '__main__': main()
