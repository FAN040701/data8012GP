"""Sensitivity of the original-three-seed component diagnostic to 26-connectivity."""
import concurrent.futures as cf
from datetime import datetime,timezone
import json

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from seed_stability_common import ROOT,OUT,CFG,sha

DEST=OUT/'component_connectivity'
PROTOCOL=ROOT/'configs/seed_component_connectivity.json'
ORGANS=['background','spleen','right_kidney','left_kidney','liver']


def process_case(case):
    meta=json.loads((ROOT/'data/processed'/case/'metadata.json').read_text())
    raw=np.asarray(nib.as_closest_canonical(nib.load(ROOT/meta['raw_label'])).dataobj,dtype=np.uint8)
    y=np.zeros(raw.shape,dtype=np.uint8)
    for old,new in [(1,1),(2,2),(3,3),(6,4)]:
        y[raw==old]=new
    structure=np.ones((3,3,3),dtype=bool)
    voxel_ml=float(np.prod(meta['original_spacing_xyz_mm'])/1000)
    rows=[]
    for seed in CFG['old_seeds']:
        for arm in CFG['arms']:
            path=ROOT/'results/mechanism/replication/runs'/f'{arm}_seed{seed}'/'predictions/f3'/f'{case}.nii.gz'
            p=np.asarray(nib.load(path).dataobj,dtype=np.uint8)
            for k in range(1,5):
                mask=p==k
                nt,npred=int((y==k).sum()),int(mask.sum())
                before_tp=int(np.count_nonzero(mask&(y==k)))
                ncomp=kept=tp=0
                if npred:
                    coordinates=np.where(mask)
                    bounds=tuple(slice(int(a.min()),int(a.max())+1) for a in coordinates)
                    labels,ncomp=ndi.label(mask[bounds],structure=structure)
                    sizes=np.bincount(labels.ravel());sizes[0]=0
                    keep=labels==int(sizes.argmax())
                    kept=int(keep.sum());tp=int(np.count_nonzero(keep&(y[bounds]==k)))
                rows.append({'case':case,'arm':arm,'seed':seed,'organ':ORGANS[k],'components26':int(ncomp),'kept_voxels26':kept,'tp_after26':tp,'dice_after26':2*tp/(nt+kept) if nt else np.nan,'removed_true_positive_ml26':(before_tp-tp)*voxel_ml,'removed_false_positive_ml26':(npred-kept-before_tp+tp)*voxel_ml})
    return rows


def main():
    DEST.mkdir(exist_ok=True)
    assert not (DEST/'complete.json').exists()
    protocol_hash,code=sha(PROTOCOL),sha(__file__)
    original_path=OUT/'component_diagnostic/original3_organ_metrics.csv'
    original_hash=sha(original_path)
    rows=[]
    with cf.ProcessPoolExecutor(2) as pool:
        for case,result in zip(CFG['new40_ids'],pool.map(process_case,CFG['new40_ids'])):
            rows.extend(result)
            print(case,len(rows)//48,'/40 complete',flush=True)
    old=pd.read_csv(original_path)
    new=pd.DataFrame(rows)
    key=['case','arm','seed','organ']
    merged=old.merge(new,on=key,validate='one_to_one')
    assert len(merged)==1920
    assert (merged.components26<=merged.components).all()
    assert (merged.kept_voxels26>=merged.kept_voxels).all()
    merged['dice_26_minus_6']=merged.dice_after26-merged.dice_after
    merged.to_csv(DEST/'organ_metrics.csv',index=False)
    patient=merged.groupby(['case','arm','seed'],as_index=False)[['dice_before','dice_after','dice_after26','dice_26_minus_6']].mean()
    patient.to_csv(DEST/'patient_metrics.csv',index=False)
    from analyze_seed_stability import summarize,WEIGHTS
    summaries=[]
    for arm in CFG['arms']:
        sub=patient[patient.arm==arm].set_index(['case','seed'])
        for name,series in [('six_connected_gain',sub.dice_after-sub.dice_before),('twentysix_connected_gain',sub.dice_after26-sub.dice_before),('twentysix_minus_six',sub.dice_26_minus_6)]:
            summaries.append({'contrast':arm,'condition':name,**summarize(series.unstack('seed'))})
    for field in ['dice_before','dice_after','dice_after26']:
        wide=patient.pivot(index=['case','seed'],columns='arm',values=field)
        d=sum(wide[a]*w for a,w in WEIGHTS['context_x_augmentation'].items())
        summaries.append({'contrast':'context_x_augmentation','condition':field,**summarize(d.unstack('seed'))})
    pd.DataFrame(summaries).to_csv(DEST/'paired_contrasts.csv',index=False)
    means=merged.drop(columns=['seed']).groupby(['case','arm','organ'],as_index=False).mean(numeric_only=True)
    means.to_csv(DEST/'patient_organ_means.csv',index=False)
    assert sha(PROTOCOL)==protocol_hash and sha(__file__)==code and sha(original_path)==original_hash
    result={'all_passed':True,'completed_utc':datetime.now(timezone.utc).isoformat(),'patients':40,'seeds':CFG['old_seeds'],'existing_predictions_reused':480,'new_training_or_inference':0,'changed_organ_scores_vs_six_connectivity':int((merged.dice_26_minus_6.abs()>1e-12).sum()),'maximum_absolute_organ_dice_change':float(merged.dice_26_minus_6.abs().max()),'protocol_sha256':protocol_hash,'source_sha256':code,'six_connectivity_source_sha256':original_hash,'scope':'Original three seeds only. Sensitivity analysis does not replace the fixed six-connectivity diagnostic or original unprocessed model results.','report_generated':False}
    (DEST/'complete.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
