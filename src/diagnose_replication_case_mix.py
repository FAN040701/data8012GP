"""Descriptive acquisition-mix accounting; fixed bins, no patient exclusions."""
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd

from mechanism_common import ROOT, OUT, sha
from mechanism_statistics import effect

DEST = OUT/'replication_case_mix'


def main():
    assert (OUT/'analysis/analysis.json').exists()
    DEST.mkdir(parents=True, exist_ok=True)
    source = OUT/'analysis/patient_metrics.csv'
    raw = pd.read_csv(source)
    raw = raw[(raw.factor == 3) & raw.arm.isin(['repeat', 'context', 'repeat_aug', 'context_aug'])]
    acquire = pd.read_csv(ROOT/'data/metadata/labeled_data_meta_0000_0599.csv').set_index('amos_id')
    rows = []
    for cohort, case in raw[['cohort', 'case']].drop_duplicates().itertuples(index=False, name=None):
        meta = json.loads((ROOT/'data/processed'/case/'metadata.json').read_text())
        z = meta['original_spacing_xyz_mm'][2]
        rows.append({'cohort': cohort, 'case': case, 'native_z_mm': z,
                     'spacing_group': 'thin_le_2p5mm' if z <= 2.5 else 'thick_gt_2p5mm',
                     'f3_sampling_mm': 3*z, 'manufacturer': acquire.loc[int(case.split('_')[1]), 'Manufacturer']})
    patients = pd.DataFrame(rows)
    assert patients.groupby('cohort').size().to_dict() == {'new40_replication': 40, 'original40_exploratory': 40, 'prior_fresh20': 20}
    patients.to_csv(DEST/'patient_acquisition_groups.csv', index=False)
    patients.groupby(['cohort', 'spacing_group', 'native_z_mm', 'manufacturer']).size().rename('patients').reset_index().to_csv(
        DEST/'acquisition_composition.csv', index=False)
    joined = raw.merge(patients, on=['cohort', 'case'], validate='many_to_one')
    weights = {'context_without_aug': {'context': 1, 'repeat': -1},
               'context_with_aug': {'context_aug': 1, 'repeat_aug': -1},
               'context_x_augmentation': {'context_aug': 1, 'repeat_aug': -1, 'context': -1, 'repeat': 1}}
    estimates = []
    for (cohort, group), sub in joined.groupby(['cohort', 'spacing_group']):
        wide = sub.pivot(index=['seed', 'case'], columns='arm', values='dice')
        for name, w in weights.items():
            stats, _ = effect(sum(wide[a]*v for a, v in w.items()), 2026111512)
            estimates.append({'cohort': cohort, 'spacing_group': group, 'contrast': name, **stats})
    estimates = pd.DataFrame(estimates)
    estimates.to_csv(DEST/'exploratory_group_contrasts.csv', index=False)
    mixture = patients.groupby(['cohort', 'spacing_group']).size().unstack().astype(float)
    mixture = mixture.div(mixture.sum(axis=1), axis=0)
    decomposed = []
    for previous in ['prior_fresh20', 'original40_exploratory']:
        for contrast in weights:
            means = estimates[estimates.contrast == contrast].pivot(index='cohort', columns='spacing_group', values='mean')
            p_old, p_new = mixture.loc[previous], mixture.loc['new40_replication']
            mu_old, mu_new = means.loc[previous], means.loc['new40_replication']
            old_mean, new_mean = float(p_old@mu_old), float(p_new@mu_new)
            composition = float((p_new-p_old)@mu_old)
            within_groups = float(p_new@(mu_new-mu_old))
            assert np.isclose(new_mean-old_mean, composition+within_groups, atol=1e-14)
            decomposed.append({'previous_cohort': previous, 'contrast': contrast,
                               'previous_observed_mean': old_mean, 'new40_observed_mean': new_mean,
                               'observed_difference': new_mean-old_mean,
                               'composition_term_using_previous_group_means': composition,
                               'within_group_term_using_new40_weights': within_groups,
                               'new40_standardized_to_previous_spacing_weights': float(p_old@mu_new)})
    pd.DataFrame(decomposed).to_csv(DEST/'descriptive_mixture_accounting.csv', index=False)
    (DEST/'complete.json').write_text(json.dumps({'completed_utc': datetime.now(timezone.utc).isoformat(),
        'all_passed': True, 'patients': len(patients), 'spacing_threshold_mm': 2.5,
        'limitations': ['All subgroup intervals are post hoc, exploratory and unadjusted.',
                       'Spacing groups also differ in scanner/case mix and f3 physical severity; these are not causal spacing effects.',
                       'Mixture decomposition is descriptive, reference-dependent algebra, not an adjusted replacement primary endpoint.',
                       'Only acquisition metadata needed for this check are exported; no demographic identifiers are used.',
                       'All cohorts retain their original patients and primary weights.'],
        'source_metrics_sha256': sha(source), 'source_sha256': sha(__file__), 'report_generated': False}, indent=2))
    print('Acquisition mix accounting complete', flush=True)


if __name__ == '__main__': main()
