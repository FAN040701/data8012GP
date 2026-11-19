"""Post hoc error accounting for the earlier negative context-width experiment."""
import concurrent.futures as cf
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
from mechanism_statistics import effect

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/mechanism/wide_context_diagnostic'
SOURCE = ROOT/'results/followup_context/analysis/all_organ_metrics.csv'


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def region_errors(item):
    case, records = item
    meta = json.loads((ROOT/'data/processed'/case/'metadata.json').read_text())
    target = remap_labels(np.asarray(nib.as_closest_canonical(nib.load(ROOT/meta['raw_label'])).dataobj,
                                    dtype=np.uint8))
    reference_z = np.stack([np.count_nonzero(target == k, axis=(0, 1)) for k in range(1, 5)])
    rows = []
    for (kind, seed, factor), group in pd.DataFrame(records).groupby(['kind', 'seed', 'factor']):
        path = ROOT/'results/followup_context'/f'{kind}_seed{seed}'/'predictions'/f'f{factor}'/f'{case}.nii.gz'
        pred = np.asarray(nib.load(path).dataobj, dtype=np.uint8)
        matrices = np.stack([np.bincount((target[:, :, z]*5+pred[:, :, z]).ravel(), minlength=25).reshape(5, 5)
                             for z in range(target.shape[2])])
        for k, organ in enumerate(NAMES[1:], 1):
            old = group[group.organ == organ].iloc[0]
            tp = matrices[:, k, k]
            fn = matrices[:, k].sum(1)-tp
            fp = matrices[:, :, k].sum(1)-tp
            ntrue, npred = int(reference_z[k-1].sum()), int((tp+fp).sum())
            assert ntrue == old.true_voxels and npred == old.pred_voxels
            dice = 2*tp.sum()/(ntrue+npred) if ntrue else np.nan
            assert np.isclose(dice, old.dice, rtol=0, atol=1e-12, equal_nan=True)
            row = {'case': case, 'kind': kind, 'seed': int(seed), 'factor': int(factor),
                   'organ': organ, 'dice': dice, 'reference_voxels': ntrue, 'prediction_voxels': npred}
            present = np.flatnonzero(reference_z[k-1])
            for region in ['inferior', 'middle', 'superior']:
                row[region+'_fp_loss'] = np.nan; row[region+'_fn_loss'] = np.nan
            if len(present):
                lo, hi = present[[0, -1]]; z = np.arange(target.shape[2])
                labels = np.where(z < lo+(hi-lo+1)/3, 'inferior',
                                  np.where(z < lo+2*(hi-lo+1)/3, 'middle', 'superior'))
                for region in ['inferior', 'middle', 'superior']:
                    row[region+'_fp_loss'] = float(fp[labels == region].sum()/(ntrue+npred))
                    row[region+'_fn_loss'] = float(fn[labels == region].sum()/(ntrue+npred))
            rows.append(row)
    return rows


def plot_selected(selection, raw):
    dest = OUT/'figures'; dest.mkdir(parents=True, exist_ok=True)
    plots = []
    for choice in selection:
        case, organ = choice['case'], choice['organ']; k = NAMES.index(organ)
        folder = ROOT/'data/processed'/case
        meta = json.loads((folder/'metadata.json').read_text())
        ct = np.asarray(nib.as_closest_canonical(nib.load(ROOT/meta['raw_image'])).dataobj, dtype=np.float32)
        target = remap_labels(np.asarray(nib.as_closest_canonical(nib.load(ROOT/meta['raw_label'])).dataobj,
                                        dtype=np.uint8)) == k
        common = np.load(folder/'image_f1.npy').transpose(2, 1, 0).astype(np.float32)
        common = resample(common, np.array(meta['model_affine_xyz']), np.array(meta['original_affine']),
                          ct.shape, 1, 0)*500-200
        predictions = {kind: np.asarray(nib.load(ROOT/'results/followup_context'/f'{kind}_seed11'/
                                               'predictions/f1'/f'{case}.nii.gz').dataobj, dtype=np.uint8) == k
                       for kind in ['narrow', 'wide']}
        errors = sum(np.count_nonzero(p != target, axis=(0, 1)) for p in predictions.values())
        z = int(errors.argmax())
        union = target.copy()
        for p in predictions.values(): union |= p
        x, y = np.flatnonzero(union.any((1, 2))), np.flatnonzero(union.any((0, 2)))
        spacing = np.array(meta['original_spacing_xyz_mm'])
        margin = np.ceil(15/spacing[:2]).astype(int)
        x0, x1 = max(0, int(x[0])-margin[0]), min(ct.shape[0], int(x[-1])+margin[0]+1)
        y0, y1 = max(0, int(y[0])-margin[1]), min(ct.shape[1], int(y[-1])+margin[1]+1)
        sl = (slice(x0, x1), slice(y0, y1)); extent = (x0*spacing[0], x1*spacing[0], y0*spacing[1], y1*spacing[1])
        gt = target[:, :, z][sl].T; base = common[:, :, z][sl].T
        fig, axes = plt.subplots(2, 2, figsize=(9, 9), layout='constrained')
        for ax in axes.flat: ax.set_xticks([]); ax.set_yticks([])
        for ax, data, title in [(axes[0, 0], ct[:, :, z][sl].T, 'Acquired CT + reference'),
                                 (axes[0, 1], base, 'Common 3-mm input; no extra degradation')]:
            ax.imshow(data, cmap='gray', vmin=-200, vmax=300, origin='lower', extent=extent)
            if gt.any() and not gt.all(): ax.contour(gt, levels=[.5], colors=['#ffdf40'], origin='lower', extent=extent)
            ax.set_title(title, fontsize=10)
        for ax, kind, label in [(axes[1, 0], 'narrow', 'Neighbours at ±3 mm'),
                                 (axes[1, 1], 'wide', 'Neighbours at ±9 mm')]:
            pred = predictions[kind][:, :, z][sl].T
            overlay = np.zeros((*gt.shape, 4))
            overlay[pred & gt] = [.13, .85, .36, .4]
            overlay[pred & ~gt] = [1, .19, .65, .8]
            overlay[~pred & gt] = [.05, .8, 1, .8]
            ax.imshow(base, cmap='gray', vmin=-200, vmax=300, origin='lower', extent=extent)
            ax.imshow(overlay, origin='lower', extent=extent)
            row = raw[(raw.case == case) & (raw.organ == organ) & (raw.kind == kind) & (raw.seed == 11) & (raw.factor == 1)].iloc[0]
            ax.set_title(f'{label}\nWhole-organ Dice {row.dice*100:.2f}%; HD95 {row.hd95_mm:.1f} mm', fontsize=10)
        fig.suptitle(f'{case} | {organ.replace("_", " ")} | {choice["selection_group"]}\n'
                     f'Wide − narrow: patient {choice["patient_difference"]*100:+.2f} pp, '
                     f'organ {choice["organ_difference"]*100:+.2f} pp (3-seed means)\n'
                     f'Shown masks: seed 11, z={z}; largest combined error slice; 20k-step models', fontsize=11)
        fig.legend(handles=[Patch(facecolor='#21d95c', label='Correct foreground'),
                            Patch(facecolor='#ff30a6', label='False positive'),
                            Patch(facecolor='#0dccff', label='False negative')],
                   loc='outside lower center', ncol=3, frameon=False)
        path = dest/f'{case}_{organ}_wide_vs_narrow.png'
        fig.savefig(path, dpi=160, bbox_inches='tight', pad_inches=.15); plt.close(fig)
        plots.append({'case': case, 'organ': organ, 'z': z, 'path': str(path.relative_to(ROOT)), 'sha256': sha(path)})
    return plots


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(SOURCE)
    denom = raw.pred_voxels+raw.true_voxels
    tp_float = raw.dice.fillna(0)*denom/2
    tp = np.rint(tp_float).astype(np.int64)
    assert np.max(np.abs(tp_float-tp)) < 1e-7
    raw['fp_voxels'] = raw.pred_voxels-tp; raw['fn_voxels'] = raw.true_voxels-tp
    raw['fp_loss'] = np.where(raw.true_voxels > 0, raw.fp_voxels/denom, np.nan)
    raw['fn_loss'] = np.where(raw.true_voxels > 0, raw.fn_voxels/denom, np.nan)
    raw.to_csv(OUT/'all_organ_error_metrics.csv', index=False)
    patient = raw.groupby(['seed', 'case', 'kind', 'factor'])[['dice', 'fp_loss', 'fn_loss', 'hd95_mm']].mean().reset_index()
    patient.to_csv(OUT/'patient_metrics.csv', index=False)
    native = patient[patient.factor == 1].pivot(index=['seed', 'case'], columns='kind', values='dice')
    differences = (native.wide-native.narrow).groupby('case').mean().sort_values()
    organ = raw[(raw.factor == 1) & raw.kind.isin(['narrow', 'wide'])].groupby(['case', 'organ', 'kind']).dice.mean().unstack('kind')
    organ['difference'] = organ.wide-organ.narrow
    choices = []
    for group, cases in [('largest losses', differences.index[:2]), ('largest gains', differences.index[-2:])]:
        for case in cases:
            values = organ.loc[case].difference.dropna()
            which = values.abs().sort_values(ascending=False).index[0]
            choices.append({'case': case, 'organ': which, 'selection_group': group,
                            'patient_difference': float(differences[case]), 'organ_difference': float(values[which])})
    (OUT/'case_selection.json').write_text(json.dumps({'selected_before_figure_generation_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Post hoc selection after aggregate FP/FN decomposition; extremes are illustrations only',
        'selection': choices, 'source_sha256': sha(SOURCE)}, indent=2))
    rows = []
    for factor in [1, 2, 3]:
        for metric in ['dice', 'fp_loss', 'fn_loss', 'hd95_mm']:
            w = patient[patient.factor == factor].pivot(index=['seed', 'case'], columns='kind', values=metric)
            value = w.wide-w.narrow
            if metric in ['fp_loss', 'fn_loss']: value = -value
            stats, _ = effect(value, 2026111508)
            rows.append({'factor': factor, 'component': metric, **stats})
    table = pd.DataFrame(rows); table.to_csv(OUT/'wide_minus_narrow_contrasts.csv', index=False)
    sub = raw[raw.kind.isin(['narrow', 'wide']) & raw.factor.isin([1, 3])]
    with cf.ProcessPoolExecutor(4) as pool:
        region_rows = [r for part in pool.map(region_errors, [(c, g.to_dict('records')) for c, g in sub.groupby('case')]) for r in part]
    regions = pd.DataFrame(region_rows)
    regions.to_csv(OUT/'regional_organ_error_metrics.csv', index=False)
    columns = [f'{r}_{e}_loss' for r in ['inferior', 'middle', 'superior'] for e in ['fp', 'fn']]
    regional_patient = regions.groupby(['seed', 'case', 'kind', 'factor'])[columns].mean().reset_index()
    region_effects = []
    for factor in [1, 3]:
        for name in columns:
            w = regional_patient[regional_patient.factor == factor].pivot(index=['seed', 'case'], columns='kind', values=name)
            stats, _ = effect(w.narrow-w.wide, 2026111508)
            region_effects.append({'factor': factor, 'component': name, **stats})
        total = sum(r['mean'] for r in region_effects if r['factor'] == factor)
        expected = table[(table.factor == factor) & (table.component == 'dice')]['mean'].iloc[0]
        assert abs(total-expected) < 1e-12
    pd.DataFrame(region_effects).to_csv(OUT/'regional_contributions.csv', index=False)
    plots = plot_selected(choices, raw)
    (OUT/'complete.json').write_text(json.dumps({'completed_utc': datetime.now(timezone.utc).isoformat(),
        'patients': 40, 'reused_native_predictions_region_checked': 480,
        'source_sha256': sha(SOURCE), 'script_sha256': sha(__file__), 'figures': plots,
        'reference_absent_case_organs': raw[raw.true_voxels == 0][['case', 'organ']].drop_duplicates().to_dict('records'),
        'all_results_post_hoc_exploratory': True, 'report_generated': False}, indent=2))
    print(table.to_string(index=False), flush=True)


if __name__ == '__main__': main()
