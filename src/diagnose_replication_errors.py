"""Post hoc error accounting on the complete replication; no outcome-based exclusions."""
from datetime import datetime, timezone
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mechanism_common import ROOT, OUT, sha
from mechanism_statistics import component_contrasts, effect

DEST = OUT/'replication_error_diagnostic'


def main():
    assert (OUT/'analysis/analysis.json').exists()
    DEST.mkdir(parents=True, exist_ok=True)
    source = OUT/'analysis/all_organ_metrics.csv'
    raw = pd.read_csv(source)
    # Integer intersections follow exactly from Dice and the two integer mask sizes.
    present = raw.true_voxels > 0
    intersections = raw.dice*(raw.true_voxels+raw.pred_voxels)/2
    error = np.abs(intersections[present]-np.rint(intersections[present]))
    assert error.max() < 1e-6
    raw['tp_voxels'] = np.where(present, np.rint(intersections.fillna(0)), 0).astype(np.int64)
    assert ((raw.tp_voxels >= 0) & (raw.tp_voxels <= raw[['true_voxels', 'pred_voxels']].min(axis=1))).all()
    raw['fp_voxels'] = raw.pred_voxels-raw.tp_voxels
    raw['fn_voxels'] = raw.true_voxels-raw.tp_voxels
    for name, count in [('fp_loss', 'fp_voxels'), ('fn_loss', 'fn_voxels')]:
        raw[name] = np.where(present, raw[count]/(raw.true_voxels+raw.pred_voxels), np.nan)
    for name, count in [('fp_per_reference', 'fp_voxels'), ('fn_per_reference', 'fn_voxels')]:
        raw[name] = np.where(present, raw[count]/raw.true_voxels.replace(0, np.nan), np.nan)
    assert np.max(np.abs(1-raw.loc[present, 'dice']-raw.loc[present, 'fp_loss']-raw.loc[present, 'fn_loss'])) < 1e-12
    # Check recovered counts against earlier full-mask counting on all shared rows.
    old = pd.read_csv(OUT/'case_study/native_error_metrics.csv')
    old.cohort = old.cohort.replace({'fresh20': 'prior_fresh20'})
    key = ['cohort', 'case', 'arm', 'seed', 'factor', 'organ']
    joined = raw.merge(old[key+['tp_voxels', 'fp_voxels', 'fn_voxels']], on=key, validate='one_to_one', suffixes=('', '_mask'))
    assert len(joined) == 5760
    for col in ['tp_voxels', 'fp_voxels', 'fn_voxels']:
        assert np.array_equal(joined[col], joined[col+'_mask'])
    raw.to_csv(DEST/'all_organ_error_counts.csv', index=False)
    metrics = ['dice', 'fp_loss', 'fn_loss', 'fp_per_reference', 'fn_per_reference']
    patient = raw.groupby(key[:-1], as_index=False)[metrics].mean()
    names = ['full_context_gain_without_aug', 'full_context_gain_with_aug',
             'augmentation_context', 'augmentation_repeat', 'augmentation_x_full_context_gain',
             'curvature_given_slope_without_aug', 'curvature_given_slope_with_aug',
             'slope_given_curvature_without_aug', 'slope_given_curvature_with_aug']
    weights = component_contrasts()
    effects, cases = [], []
    for (cohort, factor), sub in patient.groupby(['cohort', 'factor']):
        for metric in metrics:
            wide = sub.pivot(index=['seed', 'case'], columns='arm', values=metric)
            for name in names:
                w = weights[name]
                value = sum(wide[a]*v for a, v in w.items())
                # Loss contributions have the sign of a Dice improvement; fixed-reference rates retain their raw sign.
                if metric in ['fp_loss', 'fn_loss']: value = -value
                stats, matrix = effect(value, 2026111511)
                effects.append({'cohort': cohort, 'factor': factor, 'contrast': name, 'metric': metric, **stats})
                cases.extend({'cohort': cohort, 'factor': factor, 'contrast': name, 'metric': metric,
                              'case': case, 'difference': float(v)} for case, v in matrix.mean(1).items())
    effects, cases = pd.DataFrame(effects), pd.DataFrame(cases)
    effects.to_csv(DEST/'paired_error_contrasts.csv', index=False)
    cases.to_csv(DEST/'patient_error_contrasts.csv', index=False)
    sub = cases[(cases.cohort == 'new40_replication') & (cases.factor == 3) &
                (cases.contrast == 'augmentation_x_full_context_gain')]
    wide = sub.pivot(index='case', columns='metric', values='difference').sort_values('dice')
    assert len(wide) == 40 and np.max(np.abs(wide.dice-wide.fp_loss-wide.fn_loss)) < 1e-12
    fig, ax = plt.subplots(figsize=(10, 12), layout='constrained')
    y = np.arange(40); positive = np.zeros(40); negative = np.zeros(40)
    for field, label, color in [('fp_loss', 'False-positive contribution', '#CC79A7'),
                                ('fn_loss', 'False-negative contribution', '#0072B2')]:
        values = wide[field].to_numpy()*100
        ax.barh(y, values, left=np.where(values >= 0, positive, negative), height=.7, color=color, label=label)
        positive += np.maximum(values, 0); negative += np.minimum(values, 0)
    ax.scatter(wide.dice*100, y, color='black', marker='D', s=18, label='Total interaction', zorder=4)
    ax.set_yticks(y, wide.index, fontsize=8); ax.axvline(0, color='black', linewidth=.8)
    ax.grid(axis='x', alpha=.2); ax.legend(fontsize=8, loc='lower right')
    ax.set_xlabel('Contribution to context × augmentation (Dice percentage points)')
    ax.set_title('All 40 new replication patients | f3 | three-seed means\n'
                 'Post hoc score accounting; no patients excluded')
    artifacts = []
    for ext in ['png', 'svg']:
        p = DEST/f'all40_interaction_error_contributions.{ext}'
        fig.savefig(p, dpi=180, bbox_inches='tight', pad_inches=.15)
        artifacts.append({'path': str(p.relative_to(ROOT)), 'sha256': sha(p)})
    plt.close(fig)
    (DEST/'complete.json').write_text(json.dumps({'completed_utc': datetime.now(timezone.utc).isoformat(),
        'all_passed': True, 'integer_intersection_maximum_rounding_error': float(error.max()),
        'old_organ_rows_exactly_matching_independent_mask_counts': len(joined),
        'new40_positive_patient_interactions': int((wide.dice > 0).sum()),
        'new40_negative_patient_interactions': int((wide.dice < 0).sum()),
        'limitations': ['Post hoc exploratory and unadjusted; not additional confirmatory tests.',
                       'FP/(T+P) and FN/(T+P) are exact score contributions with model-dependent denominators, not causal components.',
                       'FP/T and FN/T use a fixed reference denominator; FP/T is not one-minus-precision.',
                       'No cases are removed or reweighted based on this accounting.'],
        'source_metrics_sha256': sha(source), 'source_sha256': sha(__file__), 'artifacts': artifacts,
        'report_generated': False}, indent=2))
    print('Complete replication error accounting for all 40 patients', flush=True)


if __name__ == '__main__': main()
