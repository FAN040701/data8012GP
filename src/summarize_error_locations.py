"""Exact FP/FN decomposition of Dice contrasts, with all patients retained."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/mechanism/case_study'
WEIGHTS = {
    'context_without_aug': {'context': 1, 'repeat': -1},
    'context_with_aug': {'context_aug': 1, 'repeat_aug': -1},
    'augmentation_repeat': {'repeat_aug': 1, 'repeat': -1},
    'augmentation_context': {'context_aug': 1, 'context': -1},
    'context_x_augmentation': {'context_aug': 1, 'repeat_aug': -1, 'context': -1, 'repeat': 1},
}
COMPONENTS = [f'{region}_{error}' for region in ['inferior', 'middle', 'superior'] for error in ['fp', 'fn']]


def statistics(series):
    matrix = series.unstack('seed').sort_index().sort_index(axis=1)
    assert matrix.shape[1] == 3 and not matrix.isna().any().any()
    values = matrix.to_numpy()
    patient = values.mean(1)
    rng = np.random.default_rng(2026111501)
    means = patient[rng.integers(len(patient), size=(10000, len(patient)))].mean(1)
    lo, hi = np.quantile(means, [.025, .975])
    return {'mean': float(patient.mean()), 'ci_low': float(lo), 'ci_high': float(hi),
            'patients': len(patient), 'seed_min': float(values.mean(0).min()),
            'seed_max': float(values.mean(0).max())}


def main():
    source = OUT/'native_error_metrics.csv'
    raw = pd.read_csv(source)
    denominator = raw.true_voxels+raw.pred_voxels
    raw['fp_loss'] = raw.fp_voxels/denominator
    raw['fn_loss'] = raw.fn_voxels/denominator
    assert np.max(np.abs(raw.dice-(1-raw.fp_loss-raw.fn_loss))) < 1e-12
    for name in COMPONENTS: raw[name] = raw[f'{name}_voxels']/denominator
    assert np.max(np.abs(raw[COMPONENTS].sum(1)-(1-raw.dice))) < 1e-12
    raw.to_csv(OUT/'error_loss_components.csv', index=False)
    metrics = ['dice', 'fp_loss', 'fn_loss', *COMPONENTS]
    macro = raw.groupby(['cohort', 'case', 'arm', 'seed', 'factor'])[metrics].mean().reset_index()
    macro['organ'] = 'patient_macro'
    whole = pd.concat([raw, macro], ignore_index=True)
    rows, paired = [], []
    for (cohort, factor, organ), sub in whole.groupby(['cohort', 'factor', 'organ']):
        wide = sub.pivot(index=['seed', 'case'], columns='arm', values=metrics)
        for contrast, weights in WEIGHTS.items():
            total = sum(wide['dice'][arm]*weight for arm, weight in weights.items())
            parts = {}
            for metric in metrics:
                value = sum(wide[metric][arm]*weight for arm, weight in weights.items())
                # Positive contributions always mean an increase in Dice.
                if metric != 'dice': value = -value
                parts[metric] = value
                record = {'cohort': cohort, 'factor': int(factor), 'organ': organ,
                          'contrast': contrast, 'component': metric, **statistics(value)}
                rows.append(record)
                paired.extend({'cohort': cohort, 'factor': int(factor), 'organ': organ,
                               'contrast': contrast, 'component': metric, 'case': case,
                               'contribution': float(v)} for case, v in value.groupby('case').mean().items())
            assert np.max(np.abs(sum(parts[c] for c in COMPONENTS)-total)) < 1e-12
            assert np.max(np.abs(parts['fp_loss']+parts['fn_loss']-total)) < 1e-12
    table = pd.DataFrame(rows)
    table.to_csv(OUT/'error_contribution_contrasts.csv', index=False)
    pd.DataFrame(paired).to_csv(OUT/'patient_error_contributions.csv', index=False)
    group = raw.groupby(['cohort', 'arm', 'factor', 'organ'])
    summary = group[['dice', 'fp_ml', 'fn_ml', 'fn_fraction_of_reference', 'fp_fraction_of_reference',
                     'fp_loss', 'fn_loss', *COMPONENTS]].mean().reset_index()
    summary.to_csv(OUT/'error_summary.csv', index=False)
    selected = json.loads((OUT/'case_selection.json').read_text())['selection']
    case_rows = []
    for choice in selected:
        sub = raw[(raw.case == choice['case']) & (raw.organ == choice['organ']) & (raw.factor == 3)]
        for arm, frame in sub.groupby('arm'):
            case_rows.append({**choice, 'arm': arm, 'seed_mean_dice': float(frame.dice.mean()),
                              'seed_mean_fp_ml': float(frame.fp_ml.mean()),
                              'seed_mean_fn_ml': float(frame.fn_ml.mean()),
                              'seed_mean_reference_ml': float(frame.reference_ml.mean()),
                              'seed_min_dice': float(frame.dice.min()), 'seed_max_dice': float(frame.dice.max())})
    pd.DataFrame(case_rows).to_csv(OUT/'selected_case_three_seed_metrics.csv', index=False)
    # Exact accounting, not a causal attribution or adjusted inference.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True, layout='constrained')
    names = ['context_without_aug', 'context_with_aug', 'augmentation_repeat',
             'augmentation_context', 'context_x_augmentation']
    labels = ['Context, no aug.', 'Context, with aug.', 'Aug., repeated centre',
              'Aug., full neighbours', 'Context × augmentation']
    for ax, cohort in zip(axes, ['fresh20', 'original40_exploratory']):
        subset = table[(table.cohort == cohort) & (table.factor == 3) & (table.organ == 'patient_macro')]
        for j, (component, label, color) in enumerate([('fp_loss', 'Reduced false positives', '#CC79A7'),
                                                       ('fn_loss', 'Reduced false negatives', '#0072B2')]):
            vals = np.array([subset[(subset.contrast == name) & (subset.component == component)]['mean'].iloc[0]
                             for name in names])*100
            ax.barh(np.arange(len(names))+(j-.5)*.27, vals, height=.25, label=label, color=color)
        total = np.array([subset[(subset.contrast == name) & (subset.component == 'dice')]['mean'].iloc[0]
                          for name in names])*100
        ax.scatter(total, np.arange(len(names)), marker='D', s=30, color='black', label='Total Dice difference', zorder=5)
        ax.set_yticks(np.arange(len(names)), labels)
        ax.axvline(0, color='#777777', linewidth=.8); ax.grid(axis='x', alpha=.2)
        ax.set_title('Previous 20-patient cohort' if cohort == 'fresh20' else 'Original 40-patient cohort')
        ax.set_xlabel('Contribution to Dice difference (percentage points)')
        ax.invert_yaxis()
    axes[0].legend(loc='best', fontsize=8)
    fig.suptitle('f3: exact false-positive / false-negative accounting\n'
                 'Patient macro averages across organs and three seeds; post hoc exploratory analysis', fontsize=12)
    fig.savefig(OUT/'figures/error_contributions_f3.png', dpi=180, bbox_inches='tight', pad_inches=.15)
    plt.close(fig)
    (OUT/'error_decomposition_complete.json').write_text(json.dumps({
        'completed_utc': datetime.now(timezone.utc).isoformat(), 'organ_rows': len(raw),
        'contrasts_and_components': len(table), 'exact_identity': '1-Dice = FP/(reference+prediction) + FN/(reference+prediction)',
        'region_definition': 'Inferior/middle/superior thirds of reference axial extent; all FP outside that extent assigned to the corresponding end; whole-organ denominator shared across regions',
        'all_per_patient_contrasts_reconstruct_exactly': True,
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope': 'Post hoc descriptive error localization after viewing selected case images; not identification of a biological cause; intervals unadjusted',
        'report_generated': False}, indent=2))
    print(table[(table.organ == 'patient_macro') & (table.factor == 3) &
                table.component.isin(['dice', 'fp_loss', 'fn_loss'])].to_string(index=False), flush=True)


if __name__ == '__main__': main()
