"""Distinguish negative interaction from absolute model degradation on every new40 case."""
from datetime import datetime, timezone
import json

import pandas as pd

from mechanism_common import OUT, sha


def main():
    folder = OUT/'replication_spatial_errors'
    source = OUT/'analysis/patient_metrics.csv'
    raw = pd.read_csv(source)
    raw = raw[(raw.cohort == 'new40_replication') & (raw.factor == 3) &
              raw.arm.isin(['repeat', 'context', 'repeat_aug', 'context_aug'])]
    table = raw.groupby(['case', 'arm']).dice.mean().unstack('arm')
    assert len(table) == 40 and not table.isna().any().any()
    table['interaction'] = table.context_aug-table.repeat_aug-table.context+table.repeat
    table['negative_interaction'] = table.interaction < 0
    table['combined_above_repeat'] = table.context_aug > table.repeat
    table['combined_best_of_four'] = table.context_aug >= table[['repeat', 'context', 'repeat_aug']].max(axis=1)
    table['combined_below_context'] = table.context_aug < table.context
    table['combined_below_repeat_aug'] = table.context_aug < table.repeat_aug
    table.to_csv(folder/'patient_absolute_scores_and_interactions.csv')
    negative = table[table.negative_interaction]
    (folder/'interaction_sign_semantics.json').write_text(json.dumps({
        'completed_utc': datetime.now(timezone.utc).isoformat(), 'all_passed': True,
        'patients': 40, 'negative_interaction_patients': len(negative),
        'among_negative_interactions': {field: int(negative[field].sum()) for field in
            ['combined_above_repeat', 'combined_best_of_four', 'combined_below_context', 'combined_below_repeat_aug']},
        'scope': 'Descriptive comparisons of patient-macro Dice averaged across three seeds, not individual confidence statements. Flags overlap and are not mutually exclusive categories. All 40 cases retained.',
        'interpretation': 'Negative interaction means the change in context gain under augmentation is negative; it does not by itself mean the combined model is worse than baseline.',
        'source_data_sha256': sha(source), 'source_sha256': sha(__file__), 'report_generated': False}, indent=2))


if __name__ == '__main__': main()
