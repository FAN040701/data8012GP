"""Independent scalar-coordinate check of the new sampling inputs.

Reconstruct sampled voxels using explicit Gaussian weights, native-index linear
interpolation and bilinear x/y interpolation, without scipy affine_transform.
"""
import concurrent.futures as cf
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/mechanism'


def scalar_value(ct, native_point, ratio, phase):
    x, y, z = native_point
    if x < 0 or y < 0 or x > ct.shape[0]-1 or y > ct.shape[1]-1:
        return 0.
    sigma = .5*np.sqrt(ratio*ratio-1)
    radius = int(3*sigma+.5)
    offsets = np.arange(-radius, radius+1)
    weights = np.exp(-.5*(offsets/sigma)**2); weights /= weights.sum()
    max_j = int(np.floor((ct.shape[2]-1-phase)/ratio+1e-9))
    location = (z-phase)/ratio
    j = int(np.floor(location)); alpha = location-j
    points = [phase+ratio*np.clip(j, 0, max_j), phase+ratio*np.clip(j+1, 0, max_j)]
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = min(x0+1, ct.shape[0]-1), min(y0+1, ct.shape[1]-1)
    wx, wy = x-x0, y-y0
    def sample(zpoint):
        z0 = int(np.floor(zpoint)); wz = zpoint-z0
        z1 = min(z0+1, ct.shape[2]-1)
        value = 0.
        for xx, a in [(x0, 1-wx), (x1, wx)]:
            for yy, b in [(y0, 1-wy), (y1, wy)]:
                low = np.dot(ct[xx, yy, np.clip(z0+offsets, 0, ct.shape[2]-1)].astype(float), weights)
                high = np.dot(ct[xx, yy, np.clip(z1+offsets, 0, ct.shape[2]-1)].astype(float), weights)
                value += a*b*((1-wz)*low+wz*high)
        return value
    hu = (1-alpha)*sample(points[0])+alpha*sample(points[1])
    return float(np.clip((hu+200)/500, 0, 1))


def check(item):
    case, variants = item
    folder = ROOT/'data/processed'/case
    meta = json.loads((folder/'metadata.json').read_text())
    ct = np.asarray(nib.as_closest_canonical(nib.load(ROOT/meta['raw_image'])).dataobj, dtype=np.float32)
    label = np.load(folder/'label.npy')
    rng = np.random.default_rng(2026111507+int(case.split('_')[1]))
    fg = np.argwhere(label > 0)
    points = [list(x) for x in fg[rng.choice(len(fg), size=24, replace=False)]]
    points += [list(x) for x in np.stack([rng.integers(n, size=24) for n in label.shape], axis=1)]
    # Explicit acquired-start/end planes supplement interior random samples.
    middle = np.array(label.shape)//2
    points += [[z, int(middle[1]), int(middle[2])] for z in [0, 1, len(label)-2, len(label)-1]]
    mapping = np.linalg.inv(np.asarray(meta['original_affine']))@np.asarray(meta['model_affine_xyz'])
    rows = []
    for variant in variants:
        image = np.load(ROOT/variant['path'])
        errors = []
        for z, y, x in points:
            native_point = (mapping@np.array([x, y, z, 1.]))[:3]
            expected = scalar_value(ct, native_point, variant['ratio'], variant['phase'])
            actual = float(image[z, y, x])
            errors.append(abs(actual-expected))
        maximum = float(max(errors))
        # Float16 quantization permits at most half an ULP below one, plus
        # small float32 interpolation error in the independent reconstruction.
        assert maximum <= 2**-11+2e-6, (case, variant, maximum)
        rows.append({'case': case, **variant, 'checked_voxels': len(points),
                     'max_normalized_intensity_error': maximum,
                     'mean_normalized_intensity_error': float(np.mean(errors))})
    return rows


def main():
    assert (ROOT/'data/metadata/sampling_phase/complete.json').exists()
    assert (ROOT/'data/metadata/absolute_spacing/complete.json').exists()
    tasks = {}
    phase = json.loads((ROOT/'data/metadata/sampling_phase/input_audit.json').read_text())
    for row in phase:
        if row['phase'] == 0: continue
        tasks.setdefault(row['case'], []).append({'kind': 'sampling_phase', 'path': row['path'],
                                                'ratio': 3., 'phase': row['phase']})
    absolute = json.loads((ROOT/'data/metadata/absolute_spacing/input_audit.json').read_text())
    for row in absolute:
        if row['integer_ratio_control']: continue
        tasks.setdefault(row['case'], []).append({'kind': 'absolute_spacing', 'path': row['path'],
                                                'ratio': row['sampling_ratio'], 'phase': 0})
    with cf.ProcessPoolExecutor(4) as pool:
        rows = [r for part in pool.map(check, sorted(tasks.items())) for r in part]
    result = {'completed_utc': datetime.now(timezone.utc).isoformat(), 'all_passed': True,
              'input_arrays': len(rows), 'checked_voxels': sum(r['checked_voxels'] for r in rows),
              'maximum_error': max(r['max_normalized_intensity_error'] for r in rows),
              'tolerance': 2**-11+2e-6,
              'independent_method': 'Explicit scalar Gaussian kernel and native-index interpolation; no scipy affine_transform',
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'checks': rows, 'report_generated': False}
    (OUT/'sampling_geometry_checks.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != 'checks'}, indent=2), flush=True)


if __name__ == '__main__': main()
