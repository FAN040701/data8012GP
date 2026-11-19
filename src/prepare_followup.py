"""Fresh held-out patients for the explicitly sequential context-width experiment."""
import concurrent.futures as cf
import json
import random
from pathlib import Path
import download_amos as dl
from prepare_data import prepare_one

ROOT=Path(__file__).resolve().parents[1]
META=ROOT/'data/metadata/followup_context'


def main():
    META.mkdir(exist_ok=True)
    original=json.loads((ROOT/'data/metadata/split.json').read_text())
    index=json.loads((ROOT/'data/metadata/archive_index.json').read_text())
    lookup={e['name']:e for e in index}
    excluded=set(sum(original['splits'].values(),[]))
    eligible=sorted(n for n in lookup if '/imagesVa/amos_' in n and n.endswith('.nii.gz')
                    and int(Path(n).name.split('_')[1].split('.')[0])<500
                    and Path(n).name.removesuffix('.nii.gz') not in excluded
                    and n.replace('images','labels') in lookup)
    assert len(eligible)==60
    random.Random(20261115).shuffle(eligible)
    members=eligible[:40]
    manifest={'seed':20261115,'test':[Path(n).name.removesuffix('.nii.gz') for n in members],
              'members':members,'excluded_original_ids':sorted(excluded),
              'eligible_remaining_patients':60,'protocol':'configs/followup_context_protocol.md'}
    path=META/'split.json'
    if path.exists(): assert json.loads(path.read_text())==manifest
    else: path.write_text(json.dumps(manifest,indent=2))
    dl.DOWNLOAD_URL=original['download_url']
    names=[m for n in members for m in [n.replace('images','labels'),n]]
    tasks=[(n,lookup[n]['size'],lookup[n]['crc32'],lookup[n]['header_offset'],lookup[n]['compressed_size'],lookup[n]['method']) for n in names]
    records=[]
    with cf.ThreadPoolExecutor(8) as pool:
        for future in cf.as_completed([pool.submit(dl.fetch_member,t) for t in tasks]):
            name,size,digest,status=future.result()
            records.append({'member':name,'bytes':size,'sha256':digest,'status':status})
            (META/'download_checksums.json').write_text(json.dumps(records,indent=2))
            print('Downloaded',len(records),'/ 80',name,flush=True)
    audit=[]
    with cf.ProcessPoolExecutor(6) as pool:
        for row in pool.map(prepare_one,[('test',n) for n in members]):
            audit.append(row)
            print('Prepared',row['case'],flush=True)
    oldaudit=json.loads((ROOT/'data/metadata/preprocessing_audit.json').read_text())
    hashes=[r['image_sha256'] for r in audit+oldaudit]
    assert len(hashes)==len(set(hashes))
    (META/'preprocessing_audit.json').write_text(json.dumps(audit,indent=2))
    (META/'complete.json').write_text(json.dumps({'cases':len(audit),'disjoint_from_original':True,'all_image_hashes_unique':True},indent=2))


if __name__=='__main__': main()
