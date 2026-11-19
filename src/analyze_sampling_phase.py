"""Paired phase sensitivity, including the prespecified scan-interior check."""
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd

from mechanism_common import ROOT, OUT, sha
from mechanism_statistics import interval, effect

DEST = OUT/'sampling_phase'
WEIGHTS = {'context_without_aug': {'context': 1, 'repeat': -1},
           'context_with_aug': {'context_aug': 1, 'repeat_aug': -1},
           'augmentation_repeat': {'repeat_aug': 1, 'repeat': -1},
           'augmentation_context': {'context_aug': 1, 'context': -1},
           'context_x_augmentation': {'context_aug': 1, 'repeat_aug': -1, 'context': -1, 'repeat': 1}}


def main():
    assert not json.loads((DEST/'queue_complete.json').read_text())['errors']
    frames, sources = [], {}
    for arm in ['repeat', 'context', 'repeat_aug', 'context_aug']:
        for seed in [11, 22, 33]:
            p = DEST/'runs'/f'{arm}_seed{seed}'/'metrics.csv'
            frame = pd.read_csv(p); assert len(frame) == 240
            frames.append(frame); sources[str(p.relative_to(ROOT))] = sha(p)
    raw = pd.concat(frames, ignore_index=True)
    assert len(raw) == 2880
    raw.to_csv(DEST/'all_organ_metrics.csv', index=False)
    patient = raw.groupby(['arm', 'seed', 'case', 'phase'], as_index=False).agg(
        dice=('dice', 'mean'), interior_dice=('interior_dice', 'mean'),
        interior_scored_organs=('interior_dice', 'count'))
    assert patient.groupby(['case', 'phase']).interior_scored_organs.nunique().max() == 1
    patient.to_csv(DEST/'patient_metrics.csv', index=False)
    summaries, contrasts, paired = [], [], []
    for metric in ['dice', 'interior_dice']:
        for (arm, phase), sub in patient.groupby(['arm', 'phase']):
            mean, lo, hi = interval(sub.groupby('case')[metric].mean().sort_index(), 2026111505)
            summaries.append({'arm': arm, 'phase': int(phase), 'metric': metric, 'mean': mean, 'ci_low': lo, 'ci_high': hi})
        def add(series, meta):
            stats, matrix = effect(series, 2026111505)
            contrasts.append({'metric': metric, **meta, **stats})
            paired.extend({'metric': metric, **meta, 'case': case, 'difference': float(v)}
                          for case, v in matrix.mean(1).items())
        wide = patient.pivot(index=['seed', 'case', 'phase'], columns='arm', values=metric)
        for name, weights in WEIGHTS.items():
            delta = sum(wide[arm]*weight for arm, weight in weights.items())
            for phase in [0, 1, 2]:
                add(delta.xs(phase, level='phase'), {'contrast': name, 'condition': f'phase{phase}'})
            for phase in [1, 2]:
                add(delta.xs(phase, level='phase')-delta.xs(0, level='phase'),
                    {'contrast': name, 'condition': f'phase{phase}_minus_phase0'})
        for arm in ['repeat', 'context', 'repeat_aug', 'context_aug']:
            delta = wide[arm]
            for phase in [1, 2]:
                add(delta.xs(phase, level='phase')-delta.xs(0, level='phase'),
                    {'contrast': arm+'_phase_effect', 'condition': f'phase{phase}_minus_phase0'})
    pd.DataFrame(summaries).to_csv(DEST/'model_summary.csv', index=False)
    pd.DataFrame(contrasts).to_csv(DEST/'paired_contrasts.csv', index=False)
    paired = pd.DataFrame(paired); paired.to_csv(DEST/'paired_patient_differences.csv', index=False)
    organ = raw.groupby(['arm', 'case', 'organ', 'phase'], as_index=False).dice.mean()
    ranges = organ.groupby(['arm', 'case', 'organ']).dice.agg(['min', 'max']).reset_index()
    ranges['phase_range'] = ranges['max']-ranges['min']
    ranges.to_csv(DEST/'patient_organ_phase_ranges.csv', index=False)
    chosen = json.loads((OUT/'case_study/case_selection.json').read_text())['selection']
    examples = []
    for choice in chosen:
        for row in organ[(organ.case == choice['case']) & (organ.organ == choice['organ'])].to_dict('records'):
            examples.append({**choice, **row})
    pd.DataFrame(examples).to_csv(DEST/'preselected_case_phase_results.csv', index=False)
    primary_patient = paired[(paired.metric == 'dice') & (paired.contrast == 'context_x_augmentation') &
                             paired.condition.isin(['phase0', 'phase1', 'phase2'])].pivot(
                                 index='case', columns='condition', values='difference')
    primary_patient['phase_range'] = primary_patient.max(axis=1)-primary_patient.min(axis=1)
    primary_patient['sign_changes_across_phases'] = ((primary_patient[['phase0', 'phase1', 'phase2']].min(1) < 0) &
                                                    (primary_patient[['phase0', 'phase1', 'phase2']].max(1) > 0))
    primary_patient.to_csv(DEST/'patient_interaction_phase_sensitivity.csv')
    result = {'completed_utc': datetime.now(timezone.utc).isoformat(), 'patients': 20,
              'new_predictions': 480, 'reused_predictions': 240, 'organ_rows': len(raw),
              'patients_with_interaction_sign_change': int(primary_patient.sign_changes_across_phases.sum()),
              'mean_patient_interaction_phase_range': float(primary_patient.phase_range.mean()),
              'phase0_native_input_exact_checks': 20,
              'interior_reference_absent_organs': raw[raw.interior_reference_voxels == 0][['case', 'organ']].drop_duplicates().to_dict('records'),
              'sources_sha256': sources, 'protocol_sha256': sha(ROOT/'configs/sampling_phase.json'),
              'all_results_exploratory': True, 'report_generated': False}
    (DEST/'analysis.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != 'sources_sha256'}, indent=2), flush=True)


if __name__ == '__main__': main()
