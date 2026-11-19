"""Standalone, reproducible scientific case figures; no document or report."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import nibabel as nib
import numpy as np
import pandas as pd

from prepare_data import NAMES, remap_labels, resample

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/mechanism/case_study'
ARMS = ['repeat', 'context', 'repeat_aug', 'context_aug']
LABELS = {'repeat': 'Repeated centre', 'context': 'Full neighbours',
          'repeat_aug': 'Repeated centre + aug.', 'context_aug': 'Full neighbours + aug.'}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    selection = json.loads((OUT/'case_selection.json').read_text())
    dest = OUT/'figures'; dest.mkdir(parents=True, exist_ok=True)
    artifacts, chosen_slices = [], []
    for choice in selection['selection']:
        case, organ = choice['case'], choice['organ']
        table = pd.read_csv(OUT/'per_patient'/f'{case}_native.csv')
        table = table[(table.organ == organ) & (table.seed == 11) & (table.factor == 3)]
        slices = pd.read_csv(OUT/'per_patient'/f'{case}_slices.csv')
        slices = slices[(slices.organ == organ) & (slices.seed == 11) & (slices.factor == 3)]
        meta = json.loads((ROOT/'data/processed'/case/'metadata.json').read_text())
        ct_obj = nib.as_closest_canonical(nib.load(ROOT/meta['raw_image']))
        ct = np.asarray(ct_obj.dataobj, dtype=np.float32)
        ref_obj = nib.as_closest_canonical(nib.load(ROOT/meta['raw_label']))
        target = remap_labels(np.asarray(ref_obj.dataobj, dtype=np.uint8))
        k = NAMES.index(organ)
        reference = target == k
        spacing = np.asarray(meta['original_spacing_xyz_mm'])
        degraded = np.load(ROOT/'data/processed'/case/'image_f3.npy').transpose(2, 1, 0)
        degraded = resample(degraded.astype(np.float32), np.asarray(meta['model_affine_xyz']),
                            np.asarray(meta['original_affine']), ct.shape, 1, 0)*500-200
        predictions = {arm: np.asarray(nib.load(ROOT/table[table.arm == arm].prediction_file.iloc[0]).dataobj,
                                       dtype=np.uint8) == k for arm in ARMS}
        reference_z = reference.sum((0, 1))
        error_z = np.zeros(ct.shape[2], dtype=np.int64)
        for arm in ARMS: error_z += np.count_nonzero(predictions[arm] != reference, axis=(0, 1))
        choices = [('largest_reference_area', int(reference_z.argmax())),
                   ('largest_total_error', int(error_z.argmax()))]
        if choices[0][1] == choices[1][1]: choices = [('largest_reference_area_and_total_error', choices[0][1])]
        # Global organ-centred ROI shared across all panels/slices; visual crop only.
        union = reference.copy()
        for pred in predictions.values(): union |= pred
        xx, yy = np.flatnonzero(union.any((1, 2))), np.flatnonzero(union.any((0, 2)))
        margin = np.ceil(15/spacing[:2]).astype(int)
        x0, x1 = max(0, int(xx[0])-margin[0]), min(ct.shape[0], int(xx[-1])+margin[0]+1)
        y0, y1 = max(0, int(yy[0])-margin[1]), min(ct.shape[1], int(yy[-1])+margin[1]+1)
        region = (slice(x0, x1), slice(y0, y1))
        extent = (x0*spacing[0], x1*spacing[0], y0*spacing[1], y1*spacing[1])
        for reason, z in choices:
            chosen_slices.append({**choice, 'slice_reason': reason, 'z': z,
                                  'z_world_mm': float(meta['original_affine'][2][3]+z*spacing[2]),
                                  'reference_pixels': int(reference_z[z]),
                                  'sum_four_arm_error_pixels': int(error_z[z]),
                                  'crop_x0_x1_y0_y1': [int(x0), int(x1), int(y0), int(y1)]})
            fig, axes = plt.subplots(2, 3, figsize=(12, 8.6), layout='constrained')
            gt = reference[:, :, z][region].T
            base = degraded[:, :, z][region].T
            for ax in axes.flat:
                ax.set_facecolor('#161616'); ax.set_xticks([]); ax.set_yticks([])
            for ax, values, title in [(axes[0, 0], ct[:, :, z][region].T, 'Acquired CT'),
                                      (axes[0, 1], base, 'Model input: f3, mapped to native grid')]:
                ax.imshow(values, cmap='gray', vmin=-200, vmax=300, origin='lower', extent=extent)
                if gt.any() and not gt.all():
                    ax.contour(gt, levels=[.5], colors=['#ffdf40'], linewidths=1.0,
                               origin='lower', extent=extent)
                ax.set_title(title, fontsize=11)
            for ax, arm in zip([axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2]], ARMS):
                p = predictions[arm][:, :, z][region].T
                overlay = np.zeros((*gt.shape, 4), dtype=float)
                overlay[p & gt] = [0.13, 0.85, 0.36, .40]
                overlay[p & ~gt] = [1.0, .19, .65, .8]
                overlay[~p & gt] = [.05, .8, 1.0, .8]
                ax.imshow(base, cmap='gray', vmin=-200, vmax=300, origin='lower', extent=extent)
                ax.imshow(overlay, origin='lower', extent=extent)
                row = table[table.arm == arm].iloc[0]
                ax.set_title(f'{LABELS[arm]}\nWhole-organ Dice {row.dice*100:.2f}%; '
                             f'FP {row.fp_ml:.1f} / FN {row.fn_ml:.1f} mL', fontsize=10)
            group = choice['selection_group'].replace('_', ' ')
            fig.suptitle(f'{case} | {organ.replace("_", " ")} | {group}\n'
                         f'Patient interaction {choice["patient_interaction"]*100:+.2f} pp; '
                         f'organ interaction {choice["organ_interaction"]*100:+.2f} pp (3-seed means)\n'
                         f'Shown masks: fixed seed 11; native z={z}; {reason.replace("_", " ")}', fontsize=12)
            fig.legend(handles=[Patch(facecolor='#ffdf40', label='Reference contour'),
                                Patch(facecolor='#21d95c', label='Correct foreground'),
                                Patch(facecolor='#ff30a6', label='False positive'),
                                Patch(facecolor='#0dccff', label='False negative')],
                       loc='outside lower center', ncol=4, frameon=False, fontsize=10)
            path = dest/f'{case}_{organ}_{reason}_z{z}.png'
            fig.savefig(path, dpi=160); plt.close(fig)
            artifacts.append({'path': str(path.relative_to(ROOT)), 'sha256': sha(path)})
        # Through-plane error curves include every original-grid slice.
        fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True, layout='constrained')
        zmm = np.arange(ct.shape[2])*spacing[2]
        axes[0].plot(zmm, reference_z*spacing[0]*spacing[1], color='black', label='Reference organ')
        axes[0].set_ylabel('Reference area (mm²)')
        colors = ['#555555', '#0072B2', '#D55E00', '#009E73']
        for arm, color in zip(ARMS, colors):
            p = predictions[arm]
            fp = np.count_nonzero(p & ~reference, axis=(0, 1))*spacing[0]*spacing[1]
            fn = np.count_nonzero(~p & reference, axis=(0, 1))*spacing[0]*spacing[1]
            axes[1].plot(zmm, fp, label=LABELS[arm], color=color)
            axes[2].plot(zmm, fn, label=LABELS[arm], color=color)
        axes[1].set_ylabel('False-positive area (mm²)')
        axes[2].set_ylabel('False-negative area (mm²)')
        axes[2].set_xlabel('Inferior → superior position relative to scan start (mm)')
        for ax in axes:
            ax.grid(alpha=.2)
            for _, z in choices: ax.axvline(zmm[z], color='#999999', linewidth=.8, linestyle='--')
        axes[1].legend(ncol=2, fontsize=9)
        fig.suptitle(f'{case} | {organ.replace("_", " ")} | f3 | seed 11\n'
                     'All slices; dashed lines identify the saved axial views', fontsize=12)
        path = dest/f'{case}_{organ}_through_plane_errors.png'
        fig.savefig(path, dpi=160); plt.close(fig)
        artifacts.append({'path': str(path.relative_to(ROOT)), 'sha256': sha(path)})
        print('Figures complete', case, organ, flush=True)
    pd.DataFrame(chosen_slices).to_csv(OUT/'selected_slices.csv', index=False)
    (OUT/'figures_complete.json').write_text(json.dumps({
        'completed_utc': datetime.now(timezone.utc).isoformat(), 'patients': len(selection['selection']),
        'figures': artifacts, 'source_sha256': sha(__file__),
        'selection_sha256': sha(OUT/'case_selection.json'), 'report_generated': False}, indent=2))


if __name__ == '__main__': main()
