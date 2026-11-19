"""Plot the fixed-span comparison and its companion training-seed experiment."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/slice_selection'
LABELS = {'near_far': '−3, +9 mm', 'far_near': '−9, +3 mm', 'mid': '±6 mm', 'far': '±9 mm'}


def interval(ax, row, y, color):
    mean, low, high = 100 * row[['mean', 'ci_low', 'ci_high']].astype(float)
    ax.plot([low, high], [y, y], color=color, lw=2)
    ax.scatter(mean, y, color=color, s=34, zorder=3)


def main():
    first = pd.read_csv(OUT / 'analysis/paired_effects.csv', dtype={'condition': str})
    first = first[first.condition == '3'].set_index('contrast')
    repeats = pd.read_csv(OUT / 'repeat_analysis/paired_effects.csv', dtype={'condition': str})
    policy = json.loads((OUT / 'selection.json').read_text())['selected_policy']
    repeats = repeats[(repeats.condition == '3') &
                      (repeats.contrast == policy + '_aug_minus_context_aug')].set_index('seed_group')
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.8, 4.8), sharex=True)
    for ax in [left, right]:
        ax.axvline(0, linewidth=1, color='#747b86', linestyle='--')
        ax.spines[['top','right','left']].set_visible(False)
        ax.tick_params(axis='y', length=0, pad=6)
        ax.set_xlabel('Difference in Dice (percentage points)')
    for y, name in enumerate(LABELS):
        interval(left, first.loc[name + '_aug_minus_context_aug'], 3-y, '#216a79')
    interval(left, first.loc['mixed_average_minus_mid_aug'], 5, '#923c28')
    left.set_yticks([5,3,2,1,0],
                    ['Asymmetric mean vs ±6 mm'] + [label + ' vs ±3 mm' for label in LABELS.values()])
    left.set_ylim(-.7,5.8)
    left.set_title('Offset comparisons across the first three seeds', fontsize=10, loc='left', pad=12)
    means = json.loads(repeats.loc['all6','seed_means'])
    ticks = []
    for i, seed in enumerate(['11','22','33','44','55','66']):
        y = 7-i
        right.scatter(100*means[seed], y, color=('#216a79' if i<3 else '#b56325'), s=26)
        ticks.append((y, 'Seed ' + seed))
    interval(right, repeats.loc['all6'], 0, '#253343')
    ticks.append((0, 'All six seeds'))
    right.set_yticks([y for y, _ in ticks], [s for _, s in ticks])
    right.set_ylim(-.7,7.9)
    right.set_title('Development-selected ' + LABELS[policy] + ' vs ±3 mm', fontsize=10, loc='left', pad=12)
    fig.suptitle('How does slice selection change 2.5D segmentation?', x=.03, y=.98,
                 ha='left', fontsize=15, fontweight='bold')
    fig.text(.03,.89,'40 AMOS patients, threefold axial downsampling, degradation-augmented training',
             fontsize=10, color='#45525c')
    fig.text(.03,.045,'Intervals resample paired patients and training seeds. Individual seed points average the same 40 patients.',
             fontsize=9, color='#45525c')
    fig.subplots_adjust(left=.21,right=.98,top=.75,bottom=.21,wspace=.58)
    fig.savefig(OUT / 'offset_effects.pdf', bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
