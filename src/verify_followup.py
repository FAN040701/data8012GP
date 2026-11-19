"""Independent saved-mask and aggregation checks for the fresh-patient follow-up."""
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/followup_context'


def main():
    ids=json.loads((ROOT/'data/metadata/followup_context/split.json').read_text())['test']
    old=json.loads((ROOT/'data/metadata/split.json').read_text())['splits']
    assert len(ids)==40 and not set(ids)&set(sum(old.values(),[]))
    audit=json.loads((ROOT/'data/metadata/followup_context/preprocessing_audit.json').read_text())
    assert {r['case'] for r in audit}==set(ids)
    for row in audit:
        assert nib.load(ROOT/row['raw_image']).header.get_xyzt_units()[0]=='mm'
        assert nib.load(ROOT/row['raw_label']).header.get_xyzt_units()[0]=='mm'
    case=sorted(ids)[0]
    meta=next(r for r in audit if r['case']==case)
    reference=nib.as_closest_canonical(nib.load(ROOT/meta['raw_label']))
    truth=np.asarray(reference.dataobj,dtype=np.uint8)
    raw=pd.read_csv(OUT/'analysis/all_organ_metrics.csv')
    assert len(raw)==4320
    checked=0; native_masks=0; maxerr=0.0
    for kind in ['repeat','narrow','wide']:
        for seed in [11,22,33]:
            run=OUT/f'{kind}_seed{seed}'
            for factor in [1,2,3]:
                files=list((run/'predictions'/f'f{factor}').glob('*.nii.gz'))
                assert {p.name.removesuffix('.nii.gz') for p in files}==set(ids)
                native_masks+=len(files)
                pred=nib.load(run/'predictions'/f'f{factor}'/f'{case}.nii.gz')
                assert pred.shape==reference.shape and np.allclose(pred.affine,reference.affine,atol=1e-5)
                assert pred.header.get_xyzt_units()[0]=='mm'
                labels=np.asarray(pred.dataobj,dtype=np.uint8)
                for output_id,org,original_id in zip([1,2,3,4],['spleen','right_kidney','left_kidney','liver'],[1,2,3,6]):
                    p=labels==output_id;t=truth==original_id
                    row=raw[(raw.kind==kind)&(raw.seed==seed)&(raw.case==case)&(raw.factor==factor)&(raw.organ==org)].iloc[0]
                    assert p.sum()==row.pred_voxels and t.sum()==row.true_voxels
                    if t.any(): maxerr=max(maxerr,abs(2*np.count_nonzero(p&t)/(p.sum()+t.sum())-row.dice))
                    else: assert np.isnan(row.dice)
                checked+=1
    assert maxerr<1e-12
    # Build primary patient differences directly from raw rows using explicit loops.
    differences=[]
    for patient in sorted(ids):
        seeds=[]
        for seed in [11,22,33]:
            means=[]
            for kind in ['wide','narrow']:
                values=raw[(raw.case==patient)&(raw.seed==seed)&(raw.kind==kind)&(raw.factor==1)].dice.to_numpy()
                means.append(np.nanmean(values))
            seeds.append(means[0]-means[1])
        differences.append(np.mean(seeds))
    differences=np.asarray(differences)
    result=json.loads((OUT/'analysis/analysis.json').read_text())
    assert abs(differences.mean()-result['primary']['mean'])<1e-12
    rng=np.random.default_rng(20261114)
    boots=rng.choice(differences,size=(2000,40),replace=True).mean(1)
    assert np.allclose(np.quantile(boots,[.025,.975]),[result['primary']['ci_low'],result['primary']['ci_high']],rtol=0,atol=1e-12)
    checks={'utc':datetime.now(timezone.utc).isoformat(),'all_passed':True,'disjoint_new_patients':40,
            'native_units_mm_all_cases':True,'metric_rows':len(raw),'saved_native_predictions':native_masks,
            'native_mask_check_case':case,'native_masks_independently_checked':checked,
            'maximum_dice_disagreement':maxerr,'primary_estimate_and_interval_independently_recomputed':True}
    (OUT/'deliverable_checks.json').write_text(json.dumps(checks,indent=2))
    paths=list(OUT.glob('*/metrics.csv'))+list(OUT.glob('wide_seed*/best.pt'))+list((OUT/'analysis').glob('*'))
    manifest={str(p.relative_to(ROOT)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths if p.is_file()}
    (OUT/'deliverable_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
