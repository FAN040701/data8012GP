"""Matched absolute sampling distances, independent of each patient's native spacing."""
import concurrent.futures as cf
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

from prepare_data import resample, atomic_npy

ROOT = Path(__file__).resolve().parents[1]
META = ROOT/'data/metadata/absolute_spacing'
DEST = ROOT/'data/processed_absolute_spacing'


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def prepare(item):
    case, targets, control = item
    folder = ROOT/'data/processed'/case
    meta = json.loads((folder/'metadata.json').read_text())
    image = nib.as_closest_canonical(nib.load(ROOT/meta['raw_image']))
    ct = np.asarray(image.dataobj, dtype=np.float32)
    native_spacing = float(meta['original_spacing_xyz_mm'][2])
    target_shape = tuple(reversed(meta['model_shape_dhw']))
    common_affine = np.array(meta['model_affine_xyz'])
    rows = []
    for target in targets:
        ratio = target/native_spacing
        assert ratio >= 1
        filtered = ndi.gaussian_filter1d(ct, sigma=.5*np.sqrt(ratio*ratio-1), axis=2,
                                        mode='nearest', truncate=3.0)
        affine = image.affine.copy(); affine[:3, 2] *= ratio
        depth = int(np.floor((ct.shape[2]-1)*native_spacing/target+1e-9))+1
        sampled = resample(filtered, image.affine, affine, (ct.shape[0], ct.shape[1], depth), 1, -1000)
        sampled = np.concatenate([sampled, sampled[:, :, -1:]], axis=2)
        common = resample(sampled, affine, common_affine, target_shape, 1, -1000)
        result = ((np.clip(common, -200, 300)+200)/500).transpose(2, 1, 0).astype(np.float16)
        if control:
            old = np.load(folder/'image_f3.npy')
            assert np.array_equal(result, old), f'Integer-ratio identity failed for {case}'
            path = folder/'image_f3.npy'
        else:
            dest = DEST/case; dest.mkdir(parents=True, exist_ok=True)
            path = dest/f'image_z{int(target)}mm.npy'; atomic_npy(path, result)
        rows.append({'case': case, 'native_z_spacing_mm': native_spacing,
                     'target_sampling_mm': target, 'sampling_ratio': ratio,
                     'path': str(path.relative_to(ROOT)), 'sha256': sha(path),
                     'integer_ratio_control': control, 'baseline_exact_match': control})
    return rows


def main():
    META.mkdir(parents=True, exist_ok=True)
    original = json.loads((ROOT/'data/metadata/split.json').read_text())['splits']['test']
    recent = json.loads((ROOT/'data/metadata/continuation/split.json').read_text())['test']
    thin, controls = [], []
    for case in original+recent:
        meta = json.loads((ROOT/'data/processed'/case/'metadata.json').read_text())
        dz = meta['original_spacing_xyz_mm'][2]
        if dz <= 2.5: thin.append(case)
        elif np.isclose(dz, 5): controls.append(case)
    manifest = {'cases': sorted(thin), 'five_mm_control_cases': sorted(controls),
                'frozen_before_input_generation_utc': datetime.now(timezone.utc).isoformat(),
                'protocol_sha256': sha(ROOT/'configs/absolute_spacing.json')}
    p = META/'split.json'
    if p.exists(): assert json.loads(p.read_text())['cases'] == sorted(thin)
    else: p.write_text(json.dumps(manifest, indent=2))
    tasks = [(c, [6., 15.], False) for c in sorted(thin)]+[(c, [15.], True) for c in sorted(controls)]
    with cf.ProcessPoolExecutor(4) as pool:
        audit = [row for part in pool.map(prepare, tasks) for row in part]
    (META/'input_audit.json').write_text(json.dumps(audit, indent=2))
    (META/'complete.json').write_text(json.dumps({'completed_utc': datetime.now(timezone.utc).isoformat(),
        'patients': len(thin), 'new_inputs': 2*len(thin), 'exact_integer_ratio_controls': len(controls),
        'protocol_sha256': sha(ROOT/'configs/absolute_spacing.json'), 'source_sha256': sha(__file__),
        'report_generated': False}, indent=2))
    print('Absolute-spacing preparation', len(thin), 'patients;', len(controls), 'exact controls', flush=True)


if __name__ == '__main__': main()
