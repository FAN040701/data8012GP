"""Standalone scientific figures for completed controlled experiments."""
from datetime import datetime, timezone
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mechanism_common import ROOT, OUT

DEST = OUT/'figures'
COLORS = {'repeat': '#666666', 'context': '#0072B2', 'slope': '#009E73', 'curvature': '#D55E00'}
LABELS = {'repeat': 'Repeated centre', 'context': 'Full neighbours',
          'slope': 'First difference only', 'curvature': 'Second difference only'}


def save(fig, name, artifacts):
    for ext in ['png', 'svg']:
        path = DEST/f'{name}.{ext}'
        fig.savefig(path, dpi=180, bbox_inches='tight', pad_inches=.16)
        artifacts.append({'path': str(path.relative_to(ROOT)),
                          'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    plt.close(fig)


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(OUT/'analysis/model_summary.csv')
    effects = pd.read_csv(OUT/'analysis/paired_contrasts.csv')
    artifacts = []
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    for col, cohort in enumerate(['prior_fresh20', 'new40_replication']):
        for row, augmented in enumerate([False, True]):
            ax = axes[row, col]
            for arm in ['repeat', 'context', 'slope', 'curvature']:
                name = arm+'_aug' if augmented else arm
                sub = raw[(raw.cohort == cohort) & (raw.arm == name)].sort_values('factor')
                mean = sub.dice.to_numpy()*100
                errors = np.stack((mean-sub.ci_low.to_numpy()*100, sub.ci_high.to_numpy()*100-mean))
                ax.errorbar(sub.factor, mean, yerr=errors, color=COLORS[arm], marker='o', capsize=3,
                            linewidth=1.6, label=LABELS[arm], alpha=.9)
            ax.set_xticks([1, 3, 4] if cohort == 'prior_fresh20' else [1, 3])
            ax.set_xlabel('Synthetic native-z downsampling factor (1 = no extra degradation)')
            ax.set_ylabel('Patient macro Dice (%)'); ax.grid(alpha=.2)
            ax.set_title(('Previous 20 patients' if cohort == 'prior_fresh20' else 'New 40-patient replication')+
                         (' | with augmentation' if augmented else ' | without augmentation'))
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Retrained context components | identical planar network, 40k steps, three seeds\n'
                 'Intervals are patient-bootstrap uncertainty in model means; comparisons use paired contrasts', fontsize=12)
    save(fig, 'component_model_results', artifacts)
    contrast_names = ['full_context_gain', 'slope_gain', 'curvature_gain',
                      'curvature_given_slope', 'slope_given_curvature', 'component_interaction']
    contrast_labels = ['Full − repeat', 'First diff. − repeat', 'Second diff. − repeat',
                       'Full − first diff.', 'Full − second diff.', 'Full − first diff. − second diff. + repeat']
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=True, layout='constrained')
    for col, cohort in enumerate(['prior_fresh20', 'new40_replication']):
        for row, augmented in enumerate([False, True]):
            ax = axes[row, col]
            suffix = '_with_aug' if augmented else '_without_aug'
            sub = effects[(effects.cohort == cohort) & (effects.condition == 'f3')].set_index('contrast')
            for i, name in enumerate(contrast_names):
                r = sub.loc[name+suffix]
                ax.errorbar(r['mean']*100, i, xerr=np.array([[r['mean']-r.ci_low], [r.ci_high-r['mean']]])*100,
                            fmt='o', color='#0072B2' if augmented else '#555555', capsize=3)
            ax.axvline(0, color='black', linewidth=.7); ax.grid(axis='x', alpha=.2)
            ax.set_yticks(np.arange(len(contrast_names)), contrast_labels)
            ax.set_xlabel('Paired Dice difference (percentage points)')
            ax.set_title(('Previous 20 patients' if cohort == 'prior_fresh20' else 'New 40 patients')+
                         (' | with augmentation' if augmented else ' | without augmentation'))
    axes[0, 0].invert_yaxis()
    fig.suptitle('f3 context-component contrasts | positive values favour the first term\n'
                 'All component intervals exploratory and unadjusted; full contrast family saved in CSV', fontsize=12)
    save(fig, 'component_paired_contrasts_f3', artifacts)
    phase = pd.read_csv(OUT/'sampling_phase/paired_contrasts.csv')
    absolute = pd.read_csv(OUT/'absolute_spacing/paired_contrasts.csv')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    for ax, data, conditions, x, title in [
        (axes[0], phase, ['phase0', 'phase1', 'phase2'], [0, 1, 2], 'Same 20 patients: sampling origin'),
        (axes[1], absolute, ['target6mm', 'target15mm'], [6, 15], 'Same 21 thin-acquisition patients: sampling distance')]:
        data = data[data.metric == 'dice']
        for contrast, label, color in [('context_without_aug', 'Context gain, no augmentation', '#666666'),
                                        ('context_with_aug', 'Context gain, with augmentation', '#0072B2'),
                                        ('context_x_augmentation', 'Context × augmentation', '#009E73')]:
            sub = data[data.contrast == contrast].set_index('condition').loc[conditions]
            y = sub['mean'].to_numpy()*100
            error = np.stack((y-sub.ci_low.to_numpy()*100, sub.ci_high.to_numpy()*100-y))
            ax.errorbar(x, y, yerr=error, marker='o', capsize=3, color=color, label=label)
        ax.set_title(title, fontsize=11); ax.axhline(0, color='black', linewidth=.7)
        ax.grid(alpha=.2); ax.set_xticks(x); ax.set_ylabel('Paired Dice difference (percentage points)')
    axes[0].set_xlabel('Sampling offset in native slices (factor remains 3)')
    axes[1].set_xlabel('Fixed physical sampling distance (mm)')
    axes[0].legend(fontsize=8)
    fig.suptitle('Controlled sampling diagnostics | frozen models, unchanged references\n'
                 'Exploratory patient-bootstrap intervals; synthetic acquisition changes', fontsize=12)
    save(fig, 'controlled_sampling_effects', artifacts)
    phase_patients = pd.read_csv(OUT/'sampling_phase/patient_interaction_phase_sensitivity.csv')
    fig, ax = plt.subplots(figsize=(8, 5), layout='constrained')
    for _, row in phase_patients.iterrows():
        ax.plot([0, 1, 2], row[['phase0', 'phase1', 'phase2']].to_numpy(dtype=float)*100,
                color='#888888', alpha=.4, linewidth=.9)
    means = phase_patients[['phase0', 'phase1', 'phase2']].mean().to_numpy()*100
    ax.plot([0, 1, 2], means, color='#0072B2', marker='o', linewidth=2.5, label='Mean across 20 patients')
    ax.axhline(0, color='black', linewidth=.7); ax.set_xticks([0, 1, 2])
    ax.set_xlabel('Sampling offset in native slices'); ax.set_ylabel('Context × augmentation (Dice percentage points)')
    ax.set_title('Individual-patient interaction sensitivity to sampling alignment\n'
                 'Each grey line is one patient, averaged across three seeds')
    ax.legend(); ax.grid(alpha=.2)
    save(fig, 'patient_sampling_phase_sensitivity', artifacts)
    (DEST/'manifest.json').write_text(json.dumps({'completed_utc': datetime.now(timezone.utc).isoformat(),
        'artifacts': artifacts, 'source_sha256': hashlib.sha256((ROOT/'src/plot_mechanism_results.py').read_bytes()).hexdigest(),
        'report_generated': False}, indent=2))
    print('Saved', len(artifacts), 'scientific figure files', flush=True)


if __name__ == '__main__': main()
