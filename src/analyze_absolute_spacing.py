"""Same-patient absolute-resolution effects, without native scanner-group confounding."""
from datetime import datetime, timezone
import json

import pandas as pd

from mechanism_common import ROOT, OUT, sha
from mechanism_statistics import effect, interval

DEST = OUT/'absolute_spacing'
WEIGHTS = {'context_without_aug': {'context': 1, 'repeat': -1},
           'context_with_aug': {'context_aug': 1, 'repeat_aug': -1},
           'augmentation_repeat': {'repeat_aug': 1, 'repeat': -1},
           'augmentation_context': {'context_aug': 1, 'context': -1},
           'context_x_augmentation': {'context_aug': 1, 'repeat_aug': -1, 'context': -1, 'repeat': 1}}


def main():
    assert not json.loads((DEST/'queue_complete.json').read_text())['errors']
    cases = json.loads((ROOT/'data/metadata/absolute_spacing/split.json').read_text())['cases']
    frames, sources = [], {}
    for arm in ['repeat', 'context', 'repeat_aug', 'context_aug']:
        for seed in [11, 22, 33]:
            p = DEST/'runs'/f'{arm}_seed{seed}'/'metrics.csv'
            frame = pd.read_csv(p); assert len(frame) == len(cases)*2*4
            frames.append(frame); sources[str(p.relative_to(ROOT))] = sha(p)
    raw = pd.concat(frames, ignore_index=True)
    raw.to_csv(DEST/'all_organ_metrics.csv', index=False)
    patient = raw.groupby(['arm', 'seed', 'case', 'target_sampling_mm'], as_index=False).agg(
        dice=('dice', 'mean'), interior_dice=('interior_dice', 'mean'),
        interior_scored_organs=('interior_dice', 'count'), hd95_mm=('hd95_mm', 'mean'))
    assert patient.groupby(['case', 'target_sampling_mm']).interior_scored_organs.nunique().max() == 1
    patient.to_csv(DEST/'patient_metrics.csv', index=False)
    summaries, contrasts, paired = [], [], []
    for metric in ['dice', 'interior_dice']:
        for (arm, target), sub in patient.groupby(['arm', 'target_sampling_mm']):
            mean, lo, hi = interval(sub.groupby('case')[metric].mean().sort_index(), 2026111506)
            summaries.append({'arm': arm, 'target_sampling_mm': target, 'metric': metric,
                              'mean': mean, 'ci_low': lo, 'ci_high': hi, 'patients': len(cases)})
        def add(series, meta):
            stats, matrix = effect(series, 2026111506)
            contrasts.append({'metric': metric, **meta, **stats})
            paired.extend({'metric': metric, **meta, 'case': case, 'difference': float(v)}
                          for case, v in matrix.mean(1).items())
        wide = patient.pivot(index=['seed', 'case', 'target_sampling_mm'], columns='arm', values=metric)
        for name, weights in WEIGHTS.items():
            delta = sum(wide[arm]*weight for arm, weight in weights.items())
            for target in [6., 15.]:
                add(delta.xs(target, level='target_sampling_mm'),
                    {'contrast': name, 'condition': f'target{int(target)}mm'})
            add(delta.xs(15., level='target_sampling_mm')-delta.xs(6., level='target_sampling_mm'),
                {'contrast': name, 'condition': 'target15_minus_target6'})
        for arm in ['repeat', 'context', 'repeat_aug', 'context_aug']:
            add(wide[arm].xs(15., level='target_sampling_mm')-wide[arm].xs(6., level='target_sampling_mm'),
                {'contrast': arm+'_resolution_effect', 'condition': 'target15_minus_target6'})
    pd.DataFrame(summaries).to_csv(DEST/'model_summary.csv', index=False)
    pd.DataFrame(contrasts).to_csv(DEST/'paired_contrasts.csv', index=False)
    pd.DataFrame(paired).to_csv(DEST/'paired_patient_differences.csv', index=False)
    organ = raw.groupby(['arm', 'case', 'organ', 'target_sampling_mm'], as_index=False).dice.mean()
    organ.to_csv(DEST/'patient_organ_means.csv', index=False)
    result = {'completed_utc': datetime.now(timezone.utc).isoformat(), 'patients': len(cases),
              'new_predictions': len(cases)*2*12, 'organ_rows': len(raw),
              'exact_existing_f3_preprocessing_controls': 39,
              'all_results_exploratory': True, 'paired_patient_intervention': True,
              'sources_sha256': sources, 'protocol_sha256': sha(ROOT/'configs/absolute_spacing.json'),
              'report_generated': False}
    (DEST/'analysis.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != 'sources_sha256'}, indent=2), flush=True)


if __name__ == '__main__': main()
