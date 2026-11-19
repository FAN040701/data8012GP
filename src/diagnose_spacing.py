"""Fixed exploratory spacing strata, using old-cohort saved metrics only."""
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd

from continuation_common import ROOT, OUT, sha
from analyze_continuation import CONTRASTS, effect_stats


def main():
    assert json.loads((OUT/'verification.json').read_text())['all_passed']
    protocol = ROOT/'configs/spacing_diagnostic.json'
    spec = json.loads(protocol.read_text())
    folder = OUT/'spacing_diagnostic'
    folder.mkdir(parents=True, exist_ok=True)
    acquisition_path = OUT/'input_diagnostics/cohort_acquisition_metadata.csv'
    acquisition = pd.read_csv(acquisition_path)
    acquisition = acquisition[acquisition.cohort == 'original_test40'].copy()
    acquisition['stratum'] = np.where(acquisition.z_spacing_mm <= 2.5+1e-4, 'thin', 'thick')
    assert acquisition.groupby('stratum').case.nunique().to_dict() == spec['expected_patients']
    assert not acquisition.case.duplicated().any()
    raw_path = OUT/'analysis/all_organ_metrics.csv'
    raw = pd.read_csv(raw_path)
    raw = raw[(raw.cohort == spec['cohort']) & (raw.budget == spec['budget']) &
              raw.arm.isin(spec['arms']) & raw.factor.isin(spec['factors'])].copy()
    assert len(raw) == 3840 and (raw.true_voxels > 0).all()
    patient = raw.groupby(['arm', 'seed', 'case', 'factor'], as_index=False).dice.mean()
    patient = patient.merge(acquisition.drop(columns='cohort'), on='case', validate='many_to_one')
    assert len(patient) == 960 and not patient.stratum.isna().any()
    patient.to_csv(folder/'patient_metrics.csv', index=False)
    groups, paired, between, independent_error = [], [], [], 0.
    for name in spec['contrasts']:
        weights = CONTRASTS[name]
        table = patient.pivot(index=['seed', 'case', 'factor'], columns='arm', values='dice')
        delta = sum(table[a]*w for a, w in weights.items())
        for condition in spec['conditions']:
            series = (delta.xs(3, level='factor')-delta.xs(1, level='factor')
                      if condition == 'f3_minus_f1' else delta.xs(int(condition[1:]), level='factor'))
            means = {}
            for stratum in ['thin', 'thick']:
                ids = sorted(acquisition.loc[acquisition.stratum == stratum, 'case'])
                part = series[series.index.get_level_values('case').isin(ids)]
                stats, mat = effect_stats(part)
                groups.append({'stratum': stratum, 'contrast': name, 'condition': condition, **stats})
                means[stratum] = mat.mean(1).sort_index().to_numpy()
                paired += [{'stratum': stratum, 'contrast': name, 'condition': condition,
                            'case': case, 'difference': float(value)} for case, value in mat.mean(1).items()]
                # Independent reconstruction directly from four organ metrics.
                values = []
                for case in ids:
                    seed_effects = []
                    for seed in spec['seeds']:
                        effect = 0.
                        for factor, factor_weight in ([(3, 1), (1, -1)] if condition == 'f3_minus_f1'
                                                      else [(int(condition[1:]), 1)]):
                            for arm, arm_weight in weights.items():
                                rows = raw[(raw.case == case) & (raw.seed == seed) &
                                           (raw.arm == arm) & (raw.factor == factor)]
                                assert len(rows) == 4
                                effect += factor_weight*arm_weight*sum(rows.dice)/4
                        seed_effects.append(effect)
                    values.append(sum(seed_effects)/len(seed_effects))
                independent_error = max(independent_error, float(np.max(np.abs(np.array(values)-means[stratum]))))
                rng = np.random.default_rng(spec['bootstrap']['seed'])
                boot = rng.choice(values, size=(10000, len(values)), replace=True).mean(1)
                independent_error = max(independent_error, abs(np.mean(values)-stats['mean']),
                                        float(np.max(np.abs(np.quantile(boot, [.025, .975])-
                                                            [stats['ci_low'], stats['ci_high']]))))
            rng = np.random.default_rng(spec['bootstrap']['seed'])
            a, b = means['thick'], means['thin']
            draws = a[rng.integers(len(a), size=(10000, len(a)))].mean(1)
            draws -= b[rng.integers(len(b), size=(10000, len(b)))].mean(1)
            lo, hi = np.quantile(draws, [.025, .975])
            between.append({'comparison': 'thick_minus_thin', 'contrast': name, 'condition': condition,
                            'mean': float(a.mean()-b.mean()), 'ci_low': float(lo), 'ci_high': float(hi),
                            'thick_patients': len(a), 'thin_patients': len(b)})
    assert independent_error < 1e-12
    pd.DataFrame(groups).to_csv(folder/'within_stratum_contrasts.csv', index=False)
    pd.DataFrame(paired).to_csv(folder/'paired_patient_differences.csv', index=False)
    pd.DataFrame(between).to_csv(folder/'between_stratum_contrasts.csv', index=False)
    complete = {'completed_utc': datetime.now(timezone.utc).isoformat(), 'all_passed': True,
                'patients': 40, 'source_organ_rows': len(raw), 'patient_model_condition_rows': len(patient),
                'within_stratum_contrasts': len(groups), 'between_stratum_contrasts': len(between),
                'maximum_independent_difference': independent_error, 'fresh20_used': False,
                'protocol_sha256': sha(protocol), 'source_sha256': sha(ROOT/'src/diagnose_spacing.py'),
                'input_sha256': {str(p.relative_to(ROOT)): sha(p) for p in [raw_path, acquisition_path]},
                'limitations': spec['interpretation_limit'], 'report_generated': False}
    (folder/'complete.json').write_text(json.dumps(complete, indent=2))
    paths = [protocol, ROOT/'src/diagnose_spacing.py', ROOT/'src/analyze_continuation.py']
    paths += list(folder.glob('*.csv'))+[folder/'complete.json']
    (folder/'artifact_manifest.json').write_text(json.dumps({str(p.relative_to(ROOT)): sha(p) for p in paths}, indent=2))
    print(json.dumps(complete, indent=2))


if __name__ == '__main__': main()
