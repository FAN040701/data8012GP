"""Shift the synthetic acquisition sampling origin while retaining fixed spacing."""
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
META = ROOT/'data/metadata/sampling_phase'
DEST = ROOT/'data/processed_phase'


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def prepare(case):
    folder = ROOT/'data/processed'/case
    meta = json.loads((folder/'metadata.json').read_text())
    image = nib.as_closest_canonical(nib.load(ROOT/meta['raw_image']))
    ct = np.asarray(image.dataobj, dtype=np.float32)
    filtered = ndi.gaussian_filter1d(ct, sigma=.5*np.sqrt(8), axis=2, mode='nearest', truncate=3.0)
    target_shape = tuple(reversed(meta['model_shape_dhw']))
    target_affine = np.array(meta['model_affine_xyz'])
    records = []
    for phase in [0, 1, 2]:
        sampled = filtered[:, :, phase::3]
        affine = image.affine.copy()
        affine[:3, 3] += phase*image.affine[:3, 2]
        affine[:3, 2] *= 3
        if phase:
            sampled = np.concatenate([sampled[:, :, :1], sampled], axis=2)
            affine[:3, 3] -= affine[:3, 2]
        sampled = np.concatenate([sampled, sampled[:, :, -1:]], axis=2)
        common = resample(sampled, affine, target_affine, target_shape, 1, -1000)
        result = ((np.clip(common, -200, 300)+200)/500).transpose(2, 1, 0).astype(np.float16)
        if phase == 0:
            baseline = np.load(folder/'image_f3.npy')
            assert np.array_equal(result, baseline), f'Baseline changed: {case}'
            path = folder/'image_f3.npy'
        else:
            dest = DEST/case; dest.mkdir(parents=True, exist_ok=True)
            path = dest/f'image_f3_p{phase}.npy'; atomic_npy(path, result)
        records.append({'case': case, 'phase': phase, 'path': str(path.relative_to(ROOT)),
                        'sha256': sha(path), 'native_z_spacing_mm': meta['original_spacing_xyz_mm'][2],
                        'offset_mm': phase*meta['original_spacing_xyz_mm'][2],
                        'effective_sampling_mm': 3*meta['original_spacing_xyz_mm'][2],
                        'baseline_exact_match': phase == 0, 'degraded_affine': affine.tolist()})
    return records


def main():
    META.mkdir(parents=True, exist_ok=True)
    cases = json.loads((ROOT/'data/metadata/continuation/split.json').read_text())['test']
    assert len(cases) == 20
    frozen = {'frozen_before_phase_generation_utc': datetime.now(timezone.utc).isoformat(),
              'cases': cases, 'protocol_sha256': sha(ROOT/'configs/sampling_phase.json')}
    p = META/'freeze.json'
    if p.exists(): assert json.loads(p.read_text())['cases'] == cases
    else: p.write_text(json.dumps(frozen, indent=2))
    with cf.ProcessPoolExecutor(4) as pool:
        rows = [r for part in pool.map(prepare, cases) for r in part]
    (META/'input_audit.json').write_text(json.dumps(rows, indent=2))
    (META/'complete.json').write_text(json.dumps({'completed_utc': datetime.now(timezone.utc).isoformat(),
        'cases': len(cases), 'new_inputs': 40, 'all_20_baseline_inputs_exactly_reproduced': True,
        'protocol_sha256': sha(ROOT/'configs/sampling_phase.json'), 'source_sha256': sha(__file__),
        'report_generated': False}, indent=2))
    print('Sampling-phase preparation complete', flush=True)


if __name__ == '__main__': main()
