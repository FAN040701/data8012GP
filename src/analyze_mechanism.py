"""Complete predeclared component contrasts, keeping all cohorts separate."""
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd

from mechanism_common import ROOT, OUT, CFG, jobs, run_dir, sha
from mechanism_statistics import interval, effect, component_contrasts

AN = OUT/'analysis'


def main():
    AN.mkdir(parents=True, exist_ok=True)
    assert not json.loads((OUT/'evaluate_queue_complete.json').read_text())['errors']
    assert not json.loads((OUT/'replication/queue_complete.json').read_text())['errors']
    frames, sources = [], {}
    for arm, seed in jobs():
        run = run_dir(arm, seed)
        record = json.loads((run/'evaluation_complete.json').read_text())
        assert record['protocol_sha256'] == sha(ROOT/'configs/mechanism.json')
        frame = pd.read_csv(run/'metrics.csv')
        assert len(frame) == 560 and set(frame.arm) == {arm} and set(frame.seed) == {seed}
        frames.append(frame); sources[str((run/'metrics.csv').relative_to(ROOT))] = sha(run/'metrics.csv')
    path = ROOT/'results/continuation/analysis/all_organ_metrics.csv'
    prior = pd.read_csv(path)
    prior = prior[(prior.budget == 40000) & prior.arm.isin(CFG['reused_arms'])].copy()
    prior['cohort'] = prior.cohort.replace({'fresh20': 'prior_fresh20'})
    prior = prior[prior.apply(lambda r: r.factor in CFG['evaluation_conditions'][r.cohort], axis=1)]
    frames.append(prior); sources[str(path.relative_to(ROOT))] = sha(path)
    for arm in CFG['reused_arms']+CFG['new_arms']:
        for seed in CFG['seeds']:
            run = OUT/'replication/runs'/f'{arm}_seed{seed}'
            marker = json.loads((run/'complete.json').read_text())
            assert marker['protocol_sha256'] == sha(ROOT/'configs/mechanism_replication.json')
            frame = pd.read_csv(run/'metrics.csv')
            assert len(frame) == 320
            frames.append(frame); sources[str((run/'metrics.csv').relative_to(ROOT))] = sha(run/'metrics.csv')
    raw = pd.concat(frames, ignore_index=True)
    key = ['cohort', 'arm', 'seed', 'case', 'factor', 'organ']
    assert not raw.duplicated(key).any() and len(raw) == 21120
    assert raw[raw.true_voxels > 0].dice.notna().all()
    assert raw[raw.true_voxels == 0].dice.isna().all()
    raw.to_csv(AN/'all_organ_metrics.csv', index=False)
    patient = raw.groupby(key[:-1], as_index=False).agg(dice=('dice', 'mean'), hd95_mm=('hd95_mm', 'mean'),
                                                        reference_present_organs=('dice', 'count'),
                                                        hd95_available_organs=('hd95_mm', 'count'))
    assert patient.groupby(['cohort', 'case', 'factor']).reference_present_organs.nunique().max() == 1
    patient.to_csv(AN/'patient_metrics.csv', index=False)
    summaries = []
    for (cohort, arm, factor), sub in patient.groupby(['cohort', 'arm', 'factor']):
        bootstrap_seed = 2026111504 if cohort == 'new40_replication' else CFG['bootstrap_seed']
        mean, lo, hi = interval(sub.groupby('case').dice.mean().sort_index(), bootstrap_seed)
        seeds = sub.groupby('seed').dice.mean()
        rows = raw[(raw.cohort == cohort) & (raw.arm == arm) & (raw.factor == factor)]
        summaries.append({'cohort': cohort, 'arm': arm, 'factor': int(factor), 'patients': sub.case.nunique(),
                          'dice': mean, 'ci_low': lo, 'ci_high': hi, 'hd95_mm': float(sub.hd95_mm.mean()),
                          'seed_min': float(seeds.min()), 'seed_max': float(seeds.max()),
                          'empty_predictions_reference_present': int((rows.status == 'prediction_empty').sum()),
                          'reference_absent_rows': int((rows.true_voxels == 0).sum())})
    pd.DataFrame(summaries).to_csv(AN/'model_summary.csv', index=False)
    patient.groupby(['cohort', 'arm', 'seed', 'factor'])[['dice', 'hd95_mm']].mean().reset_index().to_csv(
        AN/'seed_summary.csv', index=False)
    raw.groupby(['cohort', 'arm', 'factor', 'organ'])[['dice', 'hd95_mm']].mean().reset_index().to_csv(
        AN/'organ_summary.csv', index=False)
    effects, paired, seed_effects = [], [], []
    def add(series, meta, bootstrap_seed):
        stats, matrix = effect(series, bootstrap_seed)
        effects.append({**meta, **stats})
        paired.extend({**meta, 'case': case, 'difference': float(v)} for case, v in matrix.mean(1).items())
        seed_effects.extend({**meta, 'seed': int(seed), 'difference': float(v)} for seed, v in matrix.mean(0).items())
    contrast_weights = component_contrasts()
    for cohort, sub in patient.groupby('cohort'):
        bootstrap_seed = 2026111504 if cohort == 'new40_replication' else CFG['bootstrap_seed']
        wide = sub.pivot(index=['seed', 'case', 'factor'], columns='arm', values='dice')
        for name, weights in contrast_weights.items():
            delta = sum(wide[a]*w for a, w in weights.items())
            for factor in sorted(sub.factor.unique()):
                add(delta.xs(factor, level='factor'), {'cohort': cohort, 'contrast': name, 'condition': f'f{factor}'}, bootstrap_seed)
            add(delta.xs(3, level='factor')-delta.xs(1, level='factor'),
                {'cohort': cohort, 'contrast': name, 'condition': 'f3_minus_f1'}, bootstrap_seed)
    table = pd.DataFrame(effects)
    table.to_csv(AN/'paired_contrasts.csv', index=False)
    pd.DataFrame(paired).to_csv(AN/'paired_patient_differences.csv', index=False)
    pd.DataFrame(seed_effects).to_csv(AN/'paired_seed_differences.csv', index=False)
    organ_effects = []
    for (cohort, organ, factor), sub in raw.groupby(['cohort', 'organ', 'factor']):
        wide = sub.pivot(index=['seed', 'case'], columns='arm', values='dice').dropna()
        for name, weights in contrast_weights.items():
            series = sum(wide[a]*w for a, w in weights.items())
            stats, _ = effect(series, 2026111504 if cohort == 'new40_replication' else CFG['bootstrap_seed'])
            organ_effects.append({'cohort': cohort, 'organ': organ, 'factor': int(factor), 'contrast': name, **stats})
    pd.DataFrame(organ_effects).to_csv(AN/'organ_paired_contrasts.csv', index=False)
    primary = table[(table.cohort == 'new40_replication') &
                    (table.contrast == 'augmentation_x_full_context_gain') & (table.condition == 'f3')]
    assert len(primary) == 1
    learning = []
    for arm, seed in jobs():
        run = run_dir(arm, seed)
        best = json.loads((run/'complete.json').read_text())['best_step']
        for row in map(json.loads, (run/'validation.jsonl').read_text().splitlines()):
            learning.append({'arm': arm, 'seed': seed, 'step': row['step'],
                             'development_dice': row['macro_dice'], 'selected': row['step'] == best})
    pd.DataFrame(learning).to_csv(AN/'development_learning_curves.csv', index=False)
    result = {'completed_utc': datetime.now(timezone.utc).isoformat(), 'new_fits': 12,
              'new_component_native_predictions_prior_cohorts': 1680,
              'new_replication_native_predictions': 1920, 'organ_rows_including_reused': len(raw),
              'cohort_patients': patient.groupby('cohort').case.nunique().to_dict(),
              'replication_primary': primary.iloc[0].to_dict(), 'sources_sha256': sources,
              'statistics': 'Patient bootstrap after averaging seed-specific metrics; crossed seed bootstrap exploratory with only three seeds; secondary/component intervals unadjusted',
              'reference_absent_case_organs': raw[raw.true_voxels == 0][['cohort', 'case', 'organ']].drop_duplicates().to_dict('records'),
              'report_generated': False}
    (AN/'analysis.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != 'sources_sha256'}, indent=2), flush=True)


if __name__ == '__main__': main()
