"""Fixed secondary endcap sensitivity using already saved, unchanged predictions."""
import concurrent.futures as cf
from datetime import datetime, timezone
import json

import nibabel as nib
import numpy as np
import pandas as pd

from continuation_common import ROOT, OUT, run_dir, sha
from analyze_continuation import CONTRASTS, effect_stats

SPEC = json.loads((ROOT/'configs/endcap_diagnostic.json').read_text())
ORGANS = [('spleen', 1), ('right_kidney', 2), ('left_kidney', 3), ('liver', 6)]


def one(case, frame, check_case):
    meta = json.loads((ROOT/'data/processed'/case/'metadata.json').read_text())
    ref = nib.as_closest_canonical(nib.load(ROOT/meta['raw_label']))
    spacing = meta['original_spacing_xyz_mm'][2]
    zstart = max(0, int(np.ceil((ref.shape[2]-1)-9/spacing-1e-9)))
    reference_end = np.asarray(ref.dataobj[:, :, zstart:], dtype=np.uint8)
    source_extent = (ref.shape[2]-1)*spacing
    model_extent = (meta['model_shape_dhw'][0]-1)*3
    outside = model_extent > source_extent+1e-5
    native = np.load(ROOT/'data/processed'/case/'image_f1.npy', mmap_mode='r')[-1]
    degraded = np.load(ROOT/'data/processed'/case/'image_f3.npy', mmap_mode='r')[-1]
    if outside: assert native.max() == 0
    audit = {'case': case, 'native_z_spacing_mm': spacing, 'native_endcap_start_index': zstart,
             'excluded_planes': ref.shape[2]-zstart, 'common_last_plane_outside_source': outside,
             'common_grid_extension_mm': max(0, model_extent-source_extent),
             'native_last_plane_nonair_fraction': float((native > 0).mean()),
             'factor3_last_plane_nonair_fraction': float((degraded > 0).mean())}
    rows, check_error = [], None
    for (arm, seed, factor), group in frame[frame.case == case].groupby(['arm', 'seed', 'factor']):
        img = nib.load(ROOT/group.prediction_file.iloc[0])
        prediction_end = np.asarray(img.dataobj[:, :, zstart:], dtype=np.uint8)
        for output_id, (organ, reference_id) in enumerate(ORGANS, 1):
            row = group[group.organ == organ].iloc[0]
            p, t = prediction_end == output_id, reference_end == reference_id
            full_tp_float = row.dice*(row.pred_voxels+row.true_voxels)/2
            assert abs(full_tp_float-round(full_tp_float)) < 1e-6
            tp = round(full_tp_float)-int(np.count_nonzero(p & t))
            npred, ntrue = int(row.pred_voxels-p.sum()), int(row.true_voxels-t.sum())
            assert 0 <= tp <= min(npred, ntrue)
            score = 2*tp/(npred+ntrue) if ntrue else np.nan
            if case == check_case and arm == 'repeat' and seed == 11 and factor == 1:
                full_p = np.asarray(img.dataobj, dtype=np.uint8)[:, :, :zstart] == output_id
                full_t = np.asarray(ref.dataobj, dtype=np.uint8)[:, :, :zstart] == reference_id
                independent = 2*np.count_nonzero(full_p & full_t)/(full_p.sum()+full_t.sum()) if full_t.any() else np.nan
                if np.isfinite(independent):
                    check_error = max(check_error or 0., abs(independent-score))
            rows.append({'case': case, 'arm': arm, 'seed': int(seed), 'factor': int(factor),
                         'organ': organ, 'full_dice': float(row.dice), 'interior_dice': score,
                         'reference_voxels_in_endcap': int(t.sum()),
                         'predicted_voxels_in_endcap': int(p.sum()),
                         'interior_reference_voxels': ntrue,
                         'endcap_reference_fraction': float(t.sum()/row.true_voxels)})
    return rows, audit, check_error


def main():
    assert json.loads((OUT/'verification.json').read_text())['all_passed']
    folder = OUT/'endcap_diagnostic'; folder.mkdir(parents=True, exist_ok=True)
    ids = sorted(json.loads((ROOT/'data/metadata/split.json').read_text())['splits']['test'])
    frames = [pd.read_csv(run_dir(arm, seed, 40000)/'evaluations/original40_exploratory/metrics.csv')
              for arm in SPEC['arms'] for seed in SPEC['seeds']]
    raw = pd.concat(frames, ignore_index=True)
    assert len(raw) == 3840
    with cf.ThreadPoolExecutor(4) as pool:
        results = list(pool.map(lambda case: one(case, raw, ids[0]), ids))
    frame = pd.DataFrame([r for rows, _, _ in results for r in rows])
    frame.to_csv(folder/'organ_metrics.csv', index=False)
    audits = pd.DataFrame([a for _, a, _ in results])
    audits.to_csv(folder/'endpoint_input_audit.csv', index=False)
    errors = [e for _, _, e in results if e is not None]
    assert len(errors) == 1 and max(errors) < 1e-12
    patient = frame.groupby(['arm', 'seed', 'case', 'factor'], as_index=False)[['full_dice', 'interior_dice']].mean()
    patient.to_csv(folder/'patient_metrics.csv', index=False)
    effects = []
    for metric in ['full_dice', 'interior_dice']:
        for factor in SPEC['factors']:
            table = patient[patient.factor == factor].pivot(index=['seed', 'case'], columns='arm', values=metric)
            for name, weights in CONTRASTS.items():
                if not set(weights) <= set(table.columns): continue
                delta = sum(table[a]*w for a, w in weights.items())
                stats, _ = effect_stats(delta)
                effects.append({'scope': metric, 'factor': factor, 'contrast': name, **stats})
    pd.DataFrame(effects).to_csv(folder/'paired_contrasts.csv', index=False)
    complete = {'completed_utc': datetime.now(timezone.utc).isoformat(),
                'cases': 40, 'prediction_volumes_analyzed': 960,
                'common_final_plane_outside_native_extent_cases': int(audits.common_last_plane_outside_source.sum()),
                'reference_absent_in_interior_case_organs': frame[frame.interior_reference_voxels == 0][['case', 'organ']].drop_duplicates().to_dict('records'),
                'independent_full_mask_crosscheck_case': ids[0], 'maximum_dice_check_error': max(errors),
                'protocol_sha256': sha(ROOT/'configs/endcap_diagnostic.json'),
                'source_sha256': sha(ROOT/'src/diagnose_endcap.py'), 'fresh20_used': False,
                'primary_results_and_models_unchanged': True, 'report_generated': False}
    (folder/'complete.json').write_text(json.dumps(complete, indent=2))
    print(json.dumps(complete, indent=2))


if __name__ == '__main__': main()
