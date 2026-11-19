"""Show every prior confirmation patient, without selecting or excluding cases."""
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


def main():
    table = pd.read_csv(OUT/'patient_error_contributions.csv')
    sub = table[(table.cohort == 'fresh20') & (table.factor == 3) & (table.organ == 'patient_macro') &
                (table.contrast == 'context_x_augmentation')]
    wide = sub.pivot(index='case', columns='component', values='contribution').sort_values('dice')
    assert len(wide) == 20 and np.max(np.abs(wide.dice-wide.fp_loss-wide.fn_loss)) < 1e-12
    fig, ax = plt.subplots(figsize=(10, 8), layout='constrained')
    y = np.arange(len(wide)); positive = np.zeros(len(wide)); negative = np.zeros(len(wide))
    for field, label, color in [('fp_loss', 'False-positive contribution', '#CC79A7'),
                                ('fn_loss', 'False-negative contribution', '#0072B2')]:
        values = wide[field].to_numpy()*100
        left = np.where(values >= 0, positive, negative)
        ax.barh(y, values, left=left, height=.65, color=color, label=label, alpha=.85)
        positive += np.maximum(values, 0); negative += np.minimum(values, 0)
    ax.scatter(wide.dice*100, y, color='black', marker='D', s=24, label='Total interaction', zorder=4)
    ax.set_yticks(y, wide.index); ax.axvline(0, color='black', linewidth=.8)
    ax.set_xlabel('Contribution to context × augmentation (Dice percentage points)')
    ax.set_title('All 20 prior confirmation patients | f3 | three-seed means\n'
                 'False-positive and false-negative contributions sum to each black point')
    ax.grid(axis='x', alpha=.2); ax.legend(fontsize=9, loc='lower right')
    p = OUT/'figures/all20_patient_interaction_error_contributions.png'
    fig.savefig(p, dpi=180, bbox_inches='tight', pad_inches=.15); plt.close(fig)
    organ = pd.read_csv(OUT/'all_fresh20_organ_interactions.csv').pivot(index='case', columns='organ', values='organ_interaction')
    # barh displays its last row at the top, whereas imshow starts at the top.
    organ = organ.loc[wide.index[::-1], ['liver', 'spleen', 'left_kidney', 'right_kidney']]
    values = organ.to_numpy()*100
    limit = max(1, float(np.max(np.abs(values))))
    fig, ax = plt.subplots(figsize=(8, 9), layout='constrained')
    im = ax.imshow(values, cmap='RdBu', vmin=-limit, vmax=limit, aspect='auto')
    ax.set_xticks(np.arange(4), ['Liver', 'Spleen', 'Left kidney', 'Right kidney'])
    ax.set_yticks(np.arange(20), organ.index)
    for i in range(20):
        for j in range(4):
            ax.text(j, i, f'{values[i,j]:+.2f}', ha='center', va='center', fontsize=8,
                    color='white' if abs(values[i,j]) > .6*limit else 'black')
    fig.colorbar(im, ax=ax, label='Organ interaction (Dice percentage points)', shrink=.7)
    ax.set_title('Organ-level interaction for every prior confirmation patient\n'
                 'Same patient order as the error-contribution figure; descriptive three-seed means')
    q = OUT/'figures/all20_organ_interaction_heatmap.png'
    fig.savefig(q, dpi=180, bbox_inches='tight', pad_inches=.15); plt.close(fig)
    dominant = np.where(wide.fp_loss.abs() > wide.fn_loss.abs(), 'false_positive', 'false_negative')
    result = {'completed_utc': datetime.now(timezone.utc).isoformat(), 'patients': len(wide),
              'positive_interactions': int((wide.dice > 0).sum()), 'negative_interactions': int((wide.dice < 0).sum()),
              'larger_absolute_error_contribution_counts': pd.Series(dominant).value_counts().to_dict(),
              'all_patients_retained': True, 'figures': [{'path': str(f.relative_to(ROOT)),
                 'sha256': hashlib.sha256(f.read_bytes()).hexdigest()} for f in [p, q]],
              'scope': 'Post hoc descriptive accounting; reference labels and primary patient/organ weights unchanged',
              'report_generated': False}
    (OUT/'patient_heterogeneity_figures.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__': main()
