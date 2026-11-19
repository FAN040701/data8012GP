"""Post hoc influence diagnostics; retain every patient and keep primary unchanged."""
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd

from continuation_common import ROOT, OUT, CFG, sha


def main():
    assert json.loads((OUT/'verification.json').read_text())['all_passed']
    source = OUT/'analysis/paired_patient_differences.csv'
    table = pd.read_csv(source)
    selected = table[(table.cohort == 'fresh20') & (table.budget == 40000) &
                     (table.contrast == 'context_x_augmentation') & (table.condition == 'f3')]
    selected = selected.sort_values('case').reset_index(drop=True)
    assert len(selected) == selected.case.nunique() == 20
    primary = json.loads((OUT/'analysis/analysis.json').read_text())['primary']
    values = selected.difference.to_numpy()
    assert np.isclose(values.mean(), primary['mean'], rtol=0, atol=1e-12)
    rows = []
    for i, row in selected.iterrows():
        rest = np.delete(values, i)
        rng = np.random.default_rng(CFG['primary']['bootstrap_seed'])
        boot = rng.choice(rest, size=(10000, len(rest)), replace=True).mean(1)
        lo, hi = np.quantile(boot, [.025, .975])
        mean = float(rest.mean())
        assert abs(mean-(20*values.mean()-values[i])/19) < 1e-14
        rows.append({'omitted_case_for_diagnostic_only': row['case'],
                     'patient_interaction_dice': float(values[i]),
                     'remaining_patients': len(rest), 'leave_one_out_mean': mean,
                     'leave_one_out_ci_low': float(lo), 'leave_one_out_ci_high': float(hi),
                     'mean_change_from_full_primary': float(mean-values.mean())})
    folder = OUT/'primary_influence'; folder.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(folder/'all_leave_one_patient_out.csv', index=False)
    summary = {'completed_utc': datetime.now(timezone.utc).isoformat(),
               'scope': 'Post hoc diagnostic after the primary outcome was inspected; not a confirmatory analysis, no patient is excluded from the primary result.',
               'primary_patients': 20, 'positive_patient_interactions': int((values > 0).sum()),
               'negative_patient_interactions': int((values < 0).sum()),
               'zero_patient_interactions': int((values == 0).sum()),
               'patient_effect_min': float(values.min()), 'patient_effect_median': float(np.median(values)),
               'patient_effect_max': float(values.max()),
               'leave_one_out_mean_min': float(result.leave_one_out_mean.min()),
               'leave_one_out_mean_max': float(result.leave_one_out_mean.max()),
               'maximum_absolute_mean_change': float(result.mean_change_from_full_primary.abs().max()),
               'leave_one_out_intervals_covering_zero': int(((result.leave_one_out_ci_low <= 0) &
                                                            (result.leave_one_out_ci_high >= 0)).sum()),
               'all_twenty_deletions_retained': True,
               'primary_unchanged': primary, 'input_sha256': sha(source),
               'source_sha256': sha(ROOT/'src/diagnose_primary_influence.py'),
               'report_generated': False}
    (folder/'complete.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__': main()
