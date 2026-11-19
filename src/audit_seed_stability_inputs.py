"""Record data bytes during training, then verify they remain unchanged at completion."""
import concurrent.futures as cf
from datetime import datetime, timezone
import json

from seed_stability_common import ROOT, OUT, CFG, sha


def main():
    split=json.loads((ROOT/'data/metadata/split.json').read_text())['splits']
    paths=set()
    for case in split['train']:
        paths.update(ROOT/'data/processed'/case/name for name in ['image_f1.npy','image_f2.npy','image_f3.npy','label.npy','foreground.npz','metadata.json'])
    for case in split['val']:
        paths.update(ROOT/'data/processed'/case/name for name in ['image_f1.npy','label.npy','metadata.json'])
    for case in set(CFG['new40_ids']+CFG['prior20_ids']+CFG['spacing_ids']):
        p=ROOT/'data/processed'/case/'metadata.json'
        meta=json.loads(p.read_text())
        paths.update([p,ROOT/meta['raw_label']])
        if case in CFG['new40_ids']+CFG['prior20_ids']:
            paths.update(ROOT/'data/processed'/case/f'image_f{f}.npy' for f in [1,3])
        if case in CFG['spacing_ids']:
            paths.update(ROOT/'data/processed_absolute_spacing'/case/f'image_z{mm}mm.npy' for mm in [6,15])
    spacing_audit=json.loads((ROOT/'data/metadata/absolute_spacing/input_audit.json').read_text())
    paths.update(ROOT/row['path'] for row in spacing_audit)
    paths=sorted(paths)
    with cf.ThreadPoolExecutor(4) as pool:
        files={str(p.relative_to(ROOT)):h for p,h in zip(paths,pool.map(sha,paths))}
    # These previously frozen evaluation inputs must match their earlier independent hashes.
    prior=json.loads((ROOT/'data/metadata/mechanism_replication/processed_inputs_sha256.json').read_text())['files']
    for p,digest in prior.items():
        if p in files:
            assert files[p]==digest
    for row in spacing_audit:
        assert files[row['path']]==row['sha256']
    result={'all_passed':True,'recorded_utc':datetime.now(timezone.utc).isoformat(),'timing':'Recorded during the first new training pair, not claimed to be a before-training freeze; prior replication and absolute-spacing input hashes also checked.','files':files,'report_generated':False}
    (OUT/'input_snapshot.json').write_text(json.dumps(result,indent=2))
    print('Audited',len(files),'input files',flush=True)


if __name__=='__main__':
    main()
