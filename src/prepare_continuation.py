"""Reserve the final unused AMOS CT holdout and add unseen factor-4 inputs."""
import concurrent.futures as cf
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

import download_amos as dl
from prepare_data import prepare_one, resample, atomic_npy

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / 'data/metadata/continuation'


def prepare_extra(member):
    meta = prepare_one(('test', member))
    case = meta['case']
    path = ROOT / 'data/processed' / case / 'image_f4.npy'
    img = nib.as_closest_canonical(nib.load(ROOT / meta['raw_image']))
    assert img.header.get_xyzt_units()[0] == 'mm'
    ct = np.asarray(img.dataobj, dtype=np.float32)
    filtered = ndi.gaussian_filter1d(ct, sigma=.5*np.sqrt(15), axis=2,
                                    mode='nearest', truncate=3.0)
    sampled = filtered[:, :, ::4]
    sampled = np.concatenate([sampled, sampled[:, :, -1:]], axis=2)
    affine = img.affine.copy()
    affine[:3, 2] *= 4
    target_affine = np.array(meta['model_affine_xyz'])
    shape = tuple(reversed(meta['model_shape_dhw']))
    common = resample(sampled, affine, target_affine, shape, 1, -1000)
    result = ((np.clip(common, -200, 300)+200)/500).transpose(2, 1, 0).astype(np.float16)
    assert result.shape == tuple(meta['model_shape_dhw']) and np.isfinite(result).all()
    atomic_npy(path, result)
    return {**meta, 'factor4_file': str(path.relative_to(ROOT)),
            'factor4_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'factor4_sigma_native_voxels': float(.5*np.sqrt(15)),
            'factor4_effective_z_sampling_mm': float(img.header.get_zooms()[2]*4)}


def main():
    META.mkdir(parents=True, exist_ok=True)
    original = json.loads((ROOT/'data/metadata/split.json').read_text())
    follow = json.loads((ROOT/'data/metadata/followup_context/split.json').read_text())
    excluded = set(sum(original['splits'].values(), [])) | set(follow['test'])
    index = json.loads((ROOT/'data/metadata/archive_index.json').read_text())
    lookup = {e['name']: e for e in index}
    members = sorted(n for n in lookup if '/imagesVa/amos_' in n and n.endswith('.nii.gz')
                     and int(Path(n).name.split('_')[1].split('.')[0]) < 500
                     and Path(n).name.removesuffix('.nii.gz') not in excluded
                     and n.replace('images', 'labels') in lookup)
    assert len(members) == 20
    manifest = {'test': [Path(n).name.removesuffix('.nii.gz') for n in members],
                'members': members, 'excluded_prior_ids': sorted(excluded),
                'selection': 'All remaining 20 official validation CT patients; no performance-based selection',
                'protocol': 'configs/continuation.json'}
    splitfile = META/'split.json'
    if splitfile.exists(): assert json.loads(splitfile.read_text()) == manifest
    else: splitfile.write_text(json.dumps(manifest, indent=2))
    dl.DOWNLOAD_URL = original['download_url']
    names = [m for n in members for m in [n.replace('images', 'labels'), n]]
    tasks = [(n, lookup[n]['size'], lookup[n]['crc32'], lookup[n]['header_offset'],
              lookup[n]['compressed_size'], lookup[n]['method']) for n in names]
    records = []
    with cf.ThreadPoolExecutor(8) as pool:
        for future in cf.as_completed([pool.submit(dl.fetch_member, t) for t in tasks]):
            name, size, digest, status = future.result()
            records.append({'member': name, 'bytes': size, 'sha256': digest, 'status': status})
            (META/'download_checksums.json').write_text(json.dumps(records, indent=2))
            print('Downloaded', len(records), '/', len(tasks), name, flush=True)
    with cf.ProcessPoolExecutor(6) as pool:
        audit = list(pool.map(prepare_extra, members))
    older = json.loads((ROOT/'data/metadata/preprocessing_audit.json').read_text())
    older += json.loads((ROOT/'data/metadata/followup_context/preprocessing_audit.json').read_text())
    hashes = [r['image_sha256'] for r in audit+older]
    assert len(hashes) == len(set(hashes)) == 200
    (META/'preprocessing_audit.json').write_text(json.dumps(audit, indent=2))
    (META/'complete.json').write_text(json.dumps({'cases': 20, 'total_unique_cases': 200,
        'disjoint_from_previous_180': True, 'image_hashes_unique': True,
        'completed_utc': datetime.now(timezone.utc).isoformat()}, indent=2))
    print('Fresh holdout and unseen severity prepared', flush=True)


if __name__ == '__main__': main()
