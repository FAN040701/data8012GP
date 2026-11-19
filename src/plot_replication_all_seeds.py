"""Show all seeds for the same four selected cases without choosing a favourable seed."""
from datetime import datetime, timezone
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import nibabel as nib
import numpy as np
import pandas as pd

from mechanism_common import ROOT, OUT as MAIN, sha
from prepare_data import NAMES, remap_labels, resample

OUT = MAIN/'replication_spatial_errors'
ARMS = ['repeat', 'context', 'repeat_aug', 'context_aug']
LABELS = ['Repeated centre', 'Full neighbours', 'Repeated centre + aug.', 'Full neighbours + aug.']


def main():
    artifacts, slice_choices = [], []
    for choice in json.loads((OUT/'case_selection.json').read_text())['selection']:
        case, organ = choice['case'], choice['organ']; k = NAMES.index(organ)
        rows = pd.read_csv(OUT/'per_patient'/f'{case}_native.csv')
        rows = rows[rows.organ == organ].set_index(['seed', 'arm'])
        meta = json.loads((ROOT/'data/processed'/case/'metadata.json').read_text())
        target = remap_labels(np.asarray(nib.as_closest_canonical(nib.load(ROOT/meta['raw_label'])).dataobj, dtype=np.uint8)) == k
        masks = {(seed, arm): np.asarray(nib.load(ROOT/rows.loc[(seed, arm), 'prediction_file']).dataobj, dtype=np.uint8) == k
                 for seed in [11, 22, 33] for arm in ARMS}
        errors = sum(np.count_nonzero(p != target, axis=(0, 1)) for p in masks.values())
        z = int(errors.argmax())
        union = target.any(axis=2)
        for p in masks.values(): union |= p.any(axis=2)
        spacing = np.asarray(meta['original_spacing_xyz_mm'])
        xx, yy = np.flatnonzero(union.any(axis=1)), np.flatnonzero(union.any(axis=0))
        margin = np.ceil(15/spacing[:2]).astype(int)
        region = (slice(max(0, xx[0]-margin[0]), min(target.shape[0], xx[-1]+margin[0]+1)),
                  slice(max(0, yy[0]-margin[1]), min(target.shape[1], yy[-1]+margin[1]+1)))
        extent = (region[0].start*spacing[0], region[0].stop*spacing[0], region[1].start*spacing[1], region[1].stop*spacing[1])
        image = np.load(ROOT/'data/processed'/case/'image_f3.npy').transpose(2, 1, 0)
        image = resample(image.astype(np.float32), np.asarray(meta['model_affine_xyz']),
                         np.asarray(meta['original_affine']), target.shape, 1, 0)*500-200
        base, gt = image[:, :, z][region].T, target[:, :, z][region].T
        fig, axes = plt.subplots(3, 4, figsize=(15, 10.5), layout='constrained')
        for i, seed in enumerate([11, 22, 33]):
            scores = rows.loc[seed].dice
            interaction = scores['context_aug']-scores['repeat_aug']-scores['context']+scores['repeat']
            for j, arm in enumerate(ARMS):
                ax = axes[i, j]; p = masks[(seed, arm)][:, :, z][region].T
                overlay = np.zeros((*gt.shape, 4))
                overlay[p & gt] = [.13, .85, .36, .40]
                overlay[p & ~gt] = [1., .19, .65, .8]
                overlay[~p & gt] = [.05, .8, 1., .8]
                ax.imshow(base, cmap='gray', vmin=-200, vmax=300, origin='lower', extent=extent)
                ax.imshow(overlay, origin='lower', extent=extent)
                ax.set_xticks([]); ax.set_yticks([])
                r = rows.loc[(seed, arm)]
                ax.set_title(f'{LABELS[j]}\nWhole-organ Dice {r.dice*100:.2f}%; FP {r.fp_ml:.1f} / FN {r.fn_ml:.1f} mL', fontsize=9)
                if j == 0: ax.set_ylabel(f'Seed {seed}\nOrgan interaction {interaction*100:+.2f} pp', fontsize=10)
        fig.suptitle(f'{case} | {organ.replace("_", " ")} | all three seeds | f3\n'
                     f'Organ interaction averaged across seeds {choice["organ_interaction"]*100:+.2f} pp; '
                     'negative interaction does not necessarily mean worse absolute accuracy\n'
                     f'Shared native slice z={z}: largest combined error across all 12 masks; '
                     f'{int(target[:, :, z].sum())} reference-positive pixels on this slice\n'
                     'Shared background: degraded f3 input mapped to the native grid', fontsize=11)
        fig.legend(handles=[Patch(facecolor='#21d95c', label='Correct foreground'),
                            Patch(facecolor='#ff30a6', label='False positive'),
                            Patch(facecolor='#0dccff', label='False negative')],
                   loc='outside lower center', ncol=3, frameon=False, fontsize=9)
        path = OUT/'figures'/f'{case}_{organ}_all_seeds_z{z}.png'
        fig.savefig(path, dpi=160); plt.close(fig)
        artifacts.append({'path': str(path.relative_to(ROOT)), 'sha256': sha(path)})
        slice_choices.append({'case': case, 'organ': organ, 'native_z': z,
                              'selection': 'Maximum summed error over every seed and arm',
                              'reference_pixels': int(target[:, :, z].sum())})
        print('All-seed figure', case, flush=True)
    (OUT/'all_seed_figures_complete.json').write_text(json.dumps({'completed_utc': datetime.now(timezone.utc).isoformat(),
        'all_three_seeds_shown_for_all_four_preselected_cases': True,
        'slice_selection': slice_choices, 'artifacts': artifacts, 'source_sha256': sha(__file__),
        'source_metrics_sha256': sha(OUT/'native_spatial_errors.csv'), 'report_generated': False}, indent=2))


if __name__ == '__main__': main()
