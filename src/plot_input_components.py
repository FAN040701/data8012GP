"""Paired input measurements for every fitting patient; no outcome selection."""
from datetime import datetime, timezone
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from mechanism_common import ROOT, OUT, sha


def main():
    dest = OUT/'input_components'
    source = dest/'patient_input_components.csv'
    raw = pd.read_csv(source)
    sub = raw[(raw.cohort == 'fitting80') & (raw.region == 'reference_foreground') & raw.factor.isin([1, 3])]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout='constrained')
    specs = [('slope_rms', 'First-difference magnitude', 'RMS normalized intensity', 1, '#009E73'),
             ('curvature_rms', 'Second-difference magnitude', 'RMS normalized intensity', 1, '#D55E00'),
             ('near_zero_curvature_fraction', 'Near-zero second difference', 'Foreground pixels with |k| ≤ 0.001 (%)', 100, '#0072B2')]
    maximum = sub[['slope_rms', 'curvature_rms']].max().max()*1.12
    for ax, (metric, title, ylabel, scale, color) in zip(axes, specs):
        wide = sub.pivot(index='case', columns='factor', values=metric).sort_index()*scale
        assert wide.shape == (80, 2) and not wide.isna().any().any()
        for values in wide.to_numpy(): ax.plot([1, 3], values, color='#888888', alpha=.18, linewidth=.7)
        mean = wide.mean()
        ax.plot([1, 3], mean, color=color, marker='o', linewidth=2.5, label='Mean across 80 patients')
        for x in [1, 3]:
            ax.annotate(f'{mean[x]:.1f}%' if scale == 100 else f'{mean[x]:.4f}', (x, mean[x]),
                        xytext=(0, 12), textcoords='offset points', ha='center', color=color, fontsize=9)
        ax.set_xticks([1, 3], ['f1', 'f3']); ax.set_xlim(.65, 3.35)
        ax.set_ylim(0, 100 if scale == 100 else maximum)
        ax.set_title(title, fontsize=11); ax.set_ylabel(ylabel); ax.grid(alpha=.15)
        ax.set_xlabel('Synthetic native-z downsampling factor')
    axes[0].legend(fontsize=8, loc='upper right')
    fig.suptitle('Actual input components: s = (right − left) / 2; k = (left + right) / 2 − centre\n'
                 'All 80 fitting patients, reference foreground, interior slices; energy is not information content', fontsize=11)
    artifacts = []
    for ext in ['png', 'svg']:
        p = dest/f'paired_input_component_measurements.{ext}'
        fig.savefig(p, dpi=180, bbox_inches='tight', pad_inches=.15)
        artifacts.append({'path': str(p.relative_to(ROOT)), 'sha256': sha(p)})
    plt.close(fig)
    (dest/'figure_manifest.json').write_text(json.dumps({'completed_utc': datetime.now(timezone.utc).isoformat(),
        'patients': 80, 'source_data_sha256': sha(source), 'source_sha256': sha(__file__),
        'artifacts': artifacts, 'report_generated': False}, indent=2))


if __name__ == '__main__': main()
