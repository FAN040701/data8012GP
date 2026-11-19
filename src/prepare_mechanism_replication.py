"""Select/freeze, download and prepare 40 previously unused AMOS patients."""
import concurrent.futures as cf
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

import download_amos as dl
from prepare_data import prepare_one

ROOT = Path(__file__).resolve().parents[1]
META = ROOT/'data/metadata/mechanism_replication'
CFG = json.loads((ROOT/'configs/mechanism_replication.json').read_text())


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    META.mkdir(parents=True, exist_ok=True)
    original = json.loads((ROOT/'data/metadata/split.json').read_text())
    follow = json.loads((ROOT/'data/metadata/followup_context/split.json').read_text())
    recent = json.loads((ROOT/'data/metadata/continuation/split.json').read_text())
    excluded = set(sum(original['splits'].values(), [])) | set(follow['test']) | set(recent['test'])
    assert len(excluded) == 200
    splitfile = META/'split.json'
    index = json.loads((ROOT/'data/metadata/archive_index.json').read_text())
    lookup = {e['name']: e for e in index}
    eligible = sorted(n for n in lookup if '/imagesTr/amos_' in n and n.endswith('.nii.gz')
                      and int(Path(n).name.split('_')[1].split('.')[0]) < 500
                      and Path(n).name.removesuffix('.nii.gz') not in excluded
                      and n.replace('images', 'labels') in lookup)
    assert len(eligible) == 100
    members = sorted(random.Random(CFG['selection_seed']).sample(eligible, CFG['patients']))
    cases = [Path(n).name.removesuffix('.nii.gz') for n in members]
    assert not set(cases) & excluded
    if splitfile.exists():
        manifest = json.loads(splitfile.read_text())
        assert manifest['members'] == members and manifest['test'] == cases
    else:
        assert all(not (ROOT/'data/processed'/case).exists() for case in cases)
        checkpoint_hashes = {str(p.relative_to(ROOT)): sha(p)
                             for arm in ['repeat', 'context', 'repeat_aug', 'context_aug']
                             for seed in CFG['seeds']
                             for p in [ROOT/'results/continuation/b40000'/f'{arm}_seed{seed}'/'best.pt']}
        manifest = {'test': cases, 'members': members, 'excluded_prior_ids': sorted(excluded),
                    'eligible_unused_ct': len(eligible), 'selection_seed': CFG['selection_seed'],
                    'frozen_before_download_utc': datetime.now(timezone.utc).isoformat(),
                    'protocol_sha256': sha(ROOT/'configs/mechanism_replication.json'),
                    'primary_checkpoint_sha256': checkpoint_hashes,
                    'archive_index_sha256': sha(ROOT/'data/metadata/archive_index.json')}
        splitfile.write_text(json.dumps(manifest, indent=2))
    dl.DOWNLOAD_URL = original['download_url']
    names = [m for n in members for m in [n.replace('images', 'labels'), n]]
    tasks = [(n, lookup[n]['size'], lookup[n]['crc32'], lookup[n]['header_offset'],
              lookup[n]['compressed_size'], lookup[n]['method']) for n in names]
    print('Frozen', len(cases), 'new patients; download GB', sum(t[1] for t in tasks)/1e9, flush=True)
    records = []
    with cf.ThreadPoolExecutor(8) as pool:
        for task in cf.as_completed([pool.submit(dl.fetch_member, t) for t in tasks]):
            name, size, digest, status = task.result()
            records.append({'member': name, 'bytes': size, 'sha256': digest, 'status': status})
            (META/'download_checksums.json').write_text(json.dumps(records, indent=2))
            print('Downloaded', len(records), '/', len(tasks), name, flush=True)
    with cf.ProcessPoolExecutor(6) as pool:
        audit = list(pool.map(prepare_one, [('test', m) for m in members]))
    older = []
    for p in ['data/metadata/preprocessing_audit.json',
              'data/metadata/followup_context/preprocessing_audit.json',
              'data/metadata/continuation/preprocessing_audit.json']:
        older.extend(json.loads((ROOT/p).read_text()))
    hashes = [r['image_sha256'] for r in audit+older]
    assert len(hashes) == len(set(hashes)) == 240
    for p, digest in manifest['primary_checkpoint_sha256'].items(): assert sha(ROOT/p) == digest
    (META/'preprocessing_audit.json').write_text(json.dumps(audit, indent=2))
    (META/'complete.json').write_text(json.dumps({'completed_utc': datetime.now(timezone.utc).isoformat(),
        'patients': 40, 'total_unique_patients': 240, 'disjoint_from_prior_200': True,
        'raw_image_hashes_unique': True, 'primary_checkpoints_unchanged': True,
        'protocol_sha256': sha(ROOT/'configs/mechanism_replication.json'),
        'source_sha256': sha(__file__), 'report_generated': False}, indent=2))
    print('Replication data complete', flush=True)


if __name__ == '__main__': main()
