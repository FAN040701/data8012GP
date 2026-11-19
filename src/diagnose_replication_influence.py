"""Complete leave-one-out sensitivity; original primary retains all patients and seeds."""
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd

from mechanism_common import OUT, sha
from mechanism_statistics import interval


def main():
    dest = OUT/'replication_influence'; dest.mkdir(parents=True, exist_ok=True)
    source = OUT/'analysis/patient_metrics.csv'
    raw = pd.read_csv(source)
    sub = raw[(raw.cohort == 'new40_replication') & (raw.factor == 3)]
    wide = sub.pivot(index=['case', 'seed'], columns='arm', values='dice')
    delta = wide.context_aug-wide.repeat_aug-wide.context+wide.repeat
    matrix = delta.unstack('seed').sort_index()
    assert matrix.shape == (40, 3) and not matrix.isna().any().any()
    patient = matrix.mean(axis=1)
    mean, low, high = interval(patient, 2026111504)
    primary = json.loads((OUT/'analysis/analysis.json').read_text())['replication_primary']
    assert np.allclose([mean, low, high], [primary['mean'], primary['ci_low'], primary['ci_high']], atol=1e-12, rtol=0)
    rows = []
    for case in matrix.index:
        m, lo, hi = interval(patient.drop(case), 2026111514)
        rows.append({'omitted_case_for_diagnostic_only': case, 'omitted_case_interaction': patient.loc[case],
                     'patients': 39, 'mean': m, 'ci_low': lo, 'ci_high': hi})
    patients = pd.DataFrame(rows)
    patients.to_csv(dest/'leave_one_patient_out.csv', index=False)
    rows = []
    for seed in matrix.columns:
        m, lo, hi = interval(matrix.drop(columns=seed).mean(axis=1), 2026111514)
        rows.append({'omitted_seed_for_diagnostic_only': int(seed), 'patients': 40, 'seeds': 2,
                     'mean': m, 'ci_low': lo, 'ci_high': hi})
    pd.DataFrame(rows).to_csv(dest/'leave_one_seed_out.csv', index=False)
    (dest/'complete.json').write_text(json.dumps({'completed_utc': datetime.now(timezone.utc).isoformat(),
        'all_passed': True, 'full_primary_reproduced': True,
        'leave_one_patient_out_mean_range': [float(patients['mean'].min()), float(patients['mean'].max())],
        'leave_one_patient_out_intervals_excluding_zero': int((patients.ci_low > 0).sum()),
        'leave_one_patient_out_checks': len(patients),
        'interpretation': 'All 40 leave-one-patient-out and all three leave-one-seed-out results retained. Post hoc sensitivity, not a rule for exclusion or a replacement endpoint. Overlapping diagnostics are not independent tests.',
        'primary_keeps_all_40_patients_and_all_three_seeds': True,
        'source_metrics_sha256': sha(source), 'source_sha256': sha(__file__), 'report_generated': False}, indent=2))
    print(patients[['mean', 'ci_low', 'ci_high']].agg(['min', 'max']).to_string(), flush=True)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == '__main__': main()
