"""Separate exact Dice accounting from fixed-reference false-negative rates."""
from datetime import datetime, timezone
import json

import pandas as pd

from mechanism_common import ROOT, OUT, sha
from mechanism_statistics import effect


def main():
    folder = OUT/'case_study'
    raw = pd.read_csv(folder/'native_error_metrics.csv')
    patient = raw.groupby(['cohort', 'arm', 'seed', 'case', 'factor'])[
        ['fn_fraction_of_reference', 'fp_fraction_of_reference']].mean().reset_index()
    contrasts = {'augmentation_context': {'context_aug': 1, 'context': -1},
                 'augmentation_repeat': {'repeat_aug': 1, 'repeat': -1},
                 'context_with_aug': {'context_aug': 1, 'repeat_aug': -1},
                 'context_without_aug': {'context': 1, 'repeat': -1}}
    rows = []
    for (cohort, factor), sub in patient.groupby(['cohort', 'factor']):
        for metric in ['fn_fraction_of_reference', 'fp_fraction_of_reference']:
            wide = sub.pivot(index=['seed', 'case'], columns='arm', values=metric)
            for name, weights in contrasts.items():
                value = sum(wide[a]*w for a, w in weights.items())
                stats, _ = effect(value, 2026111510)
                rows.append({'cohort': cohort, 'factor': int(factor), 'metric': metric, 'contrast': name,
                             'direction': 'negative = less error per fixed reference volume', **stats})
    pd.DataFrame(rows).to_csv(folder/'fixed_reference_error_rate_contrasts.csv', index=False)
    (folder/'error_accounting_semantics.json').write_text(json.dumps({
        'completed_utc': datetime.now(timezone.utc).isoformat(),
        'exact_dice_identity': '1-Dice = FP/(T+P) + FN/(T+P), where T and P are reference and prediction counts',
        'normalization_caveat': 'The T+P denominator changes across models, so these are exact score contributions, not independently manipulated causal error mechanisms.',
        'additional_check': 'FP/T and FN/T use the same fixed reference denominator across models and directly quantify voxel error changes. FP/T is not one-minus-precision.',
        'aggregation': 'Equal reference-present organs within patient, then equal patients; absolute mL averages use a different weighting and can move in a different direction.',
        'all_intervals_exploratory': True, 'source_sha256': sha(__file__), 'report_generated': False}, indent=2))


if __name__ == '__main__': main()
