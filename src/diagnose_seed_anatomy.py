"""Locate segmentation errors using all original AMOS annotation classes."""
import argparse
import concurrent.futures as cf
from datetime import datetime, timezone
import json

import nibabel as nib
import numpy as np
import pandas as pd

from seed_stability_common import ROOT, OUT, CFG, sha

DEST=OUT/'anatomical_diagnostic'
PROTOCOL=ROOT/'configs/seed_anatomical_diagnostic.json'
MAP={1:1,2:2,3:3,4:6}
NAMES={1:'spleen',2:'right_kidney',3:'left_kidney',4:'liver'}
LABELS=json.loads((ROOT/'data/metadata/dataset.json').read_text())['labels']


def process_case(task):
    case,seeds=task
    meta=json.loads((ROOT/'data/processed'/case/'metadata.json').read_text())
    raw=np.asarray(nib.as_closest_canonical(nib.load(ROOT/meta['raw_label'])).dataobj,dtype=np.uint8)
    assert int(raw.max())<=15
    label_code=raw.astype(np.int16)*5
    voxel_ml=float(np.prod(meta['original_spacing_xyz_mm'])/1000)
    rows,organs=[] ,[]
    hashes={}
    for seed in seeds:
        for arm in CFG['arms']:
            folder=ROOT/'results/mechanism/replication/runs'/f'{arm}_seed{seed}' if seed in CFG['old_seeds'] else OUT/'evaluations/best'/f'{arm}_seed{seed}'/'new40_replication'
            metrics=pd.read_csv(folder/'metrics.csv').set_index(['case','factor','organ'])
            for factor in [1,3]:
                path=folder/'predictions'/f'f{factor}'/f'{case}.nii.gz'
                pred=np.asarray(nib.load(path).dataobj,dtype=np.uint8)
                assert pred.shape==raw.shape and int(pred.max())<=4
                counts=np.bincount((label_code+pred).ravel(),minlength=80).reshape(16,5)
                assert int(counts.sum())==raw.size
                hashes[str(path.relative_to(ROOT))]=sha(path)
                for ref in range(16):
                    for predicted in range(5):
                        rows.append({'case':case,'seed':seed,'arm':arm,'factor':factor,'reference_class':ref,'reference_name':LABELS[str(ref)],'predicted_class':predicted,'voxels':int(counts[ref,predicted]),'volume_ml':float(counts[ref,predicted]*voxel_ml)})
                for k,ref in MAP.items():
                    tp=int(counts[ref,k])
                    nt=int(counts[ref,:].sum())
                    npred=int(counts[:,k].sum())
                    fp=npred-tp
                    fn=nt-tp
                    denominator=nt+npred
                    expected=metrics.loc[(case,factor,NAMES[k])]
                    value=2*tp/denominator if nt else np.nan
                    assert nt==expected.true_voxels and npred==expected.pred_voxels
                    assert np.isclose(value,expected.dice,atol=1e-12,rtol=0,equal_nan=True)
                    components={f'fp_ref{r}_score':float(counts[r,k]/denominator) if r!=ref and nt else 0. for r in range(16)}
                    assert np.isclose(sum(components.values()),fp/denominator,atol=1e-12)
                    organs.append({'case':case,'seed':seed,'arm':arm,'factor':factor,'organ':NAMES[k],'dice':value,'true_voxels':nt,'pred_voxels':npred,'tp':tp,'fp':fp,'fn':fn,'fp_ml':fp*voxel_ml,'fn_ml':fn*voxel_ml,'fn_score':fn/denominator if nt else np.nan,**components})
    return rows,organs,hashes


def summarize():
    from analyze_seed_stability import summarize as effect, GROUPS, WEIGHTS
    organ_frames,counts=[],[]
    for group in ['original3','additional3']:
        assert json.loads((DEST/f'{group}_complete.json').read_text())['all_passed']
        organ_frames.append(pd.read_csv(DEST/f'{group}_organ_metrics.csv'))
        counts.append(pd.read_csv(DEST/f'{group}_confusion_counts.csv'))
    raw=pd.concat(organ_frames,ignore_index=True)
    confusion=pd.concat(counts,ignore_index=True)
    assert len(raw)==7680 and len(confusion)==153600
    raw.to_csv(DEST/'all_organ_metrics.csv',index=False)
    confusion.to_csv(DEST/'all_confusion_counts.csv',index=False)
    fields=[f'fp_ref{r}_score' for r in range(16)]+['fn_score']
    assert np.allclose(raw[fields].sum(axis=1),1-raw.dice,atol=1e-12)
    patient=raw.groupby(['case','seed','arm','factor'],as_index=False)[['dice']+fields].mean()
    effects=[]
    for metric in ['dice']+fields:
        wide=patient.pivot(index=['case','seed','factor'],columns='arm',values=metric)
        for name in ['context_with_aug','context_x_augmentation','augmentation_context']:
            d=sum(wide[a]*w for a,w in WEIGHTS[name].items())
            if metric!='dice':
                d=-d # Error reduction is positive and sums exactly to Dice gain.
            for factor in [1,3]:
                matrix=d.xs(factor,level='factor').unstack('seed')
                for group,seeds in GROUPS.items():
                    effects.append({'metric':metric,'contrast':name,'factor':factor,'seed_group':group,**effect(matrix[seeds])})
    table=pd.DataFrame(effects)
    table.to_csv(DEST/'paired_error_contributions.csv',index=False)
    for key,sub in table.groupby(['contrast','factor','seed_group']):
        assert np.isclose(sub[sub.metric!='dice']['mean'].sum(),sub[sub.metric=='dice']['mean'].iloc[0],atol=1e-12)
    # Full original-label breakdown for each patient and predicted target, averaged over seeds.
    means=confusion.groupby(['case','arm','factor','reference_class','reference_name','predicted_class'],as_index=False)[['voxels','volume_ml']].mean()
    means.to_csv(DEST/'patient_confusion_means.csv',index=False)
    raw.groupby(['case','arm','factor','organ'],as_index=False).mean(numeric_only=True).to_csv(DEST/'patient_organ_means.csv',index=False)
    summary={'all_passed':True,'completed_utc':datetime.now(timezone.utc).isoformat(),'existing_predictions_recounted':1920,'organ_rows':len(raw),'native_confusion_rows':len(confusion),'all_original_dice_values_reproduced':True,'error_components_sum_to_original_dice_losses':True,'new_inference_or_training':0,'protocol_sha256':sha(PROTOCOL),'source_sha256':sha(__file__),'reference_label_names_sha256':sha(ROOT/'data/metadata/dataset.json'),'interpretation':'Reference-class overlap locates errors; it does not establish their learned cause. Original label 0 is an unclassified remainder, not a specific anatomical tissue. All original scores remain unchanged.','report_generated':False}
    (DEST/'complete.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('group',choices=['original3','additional3','analyze'])
    args=ap.parse_args()
    DEST.mkdir(exist_ok=True)
    if args.group=='analyze':
        summarize()
        return
    marker=DEST/f'{args.group}_complete.json'
    if marker.exists():
        return
    seeds=CFG['old_seeds'] if args.group=='original3' else CFG['new_seeds']
    digest,code=sha(PROTOCOL),sha(__file__)
    rows,organs,hashes=[],[],{}
    with cf.ProcessPoolExecutor(2) as pool:
        for case,(r,o,h) in zip(CFG['new40_ids'],pool.map(process_case,[(c,seeds) for c in CFG['new40_ids']])):
            rows.extend(r);organs.extend(o);hashes.update(h)
            print(args.group,case,len(hashes)//24,'/40 complete',flush=True)
    assert len(rows)==76800 and len(organs)==3840 and len(hashes)==960
    pd.DataFrame(rows).to_csv(DEST/f'{args.group}_confusion_counts.csv',index=False)
    pd.DataFrame(organs).to_csv(DEST/f'{args.group}_organ_metrics.csv',index=False)
    (DEST/f'{args.group}_prediction_hashes.json').write_text(json.dumps(hashes,indent=2))
    assert sha(PROTOCOL)==digest and sha(__file__)==code
    marker.write_text(json.dumps({'all_passed':True,'completed_utc':datetime.now(timezone.utc).isoformat(),'existing_predictions_recounted':len(hashes),'organ_rows':len(organs),'protocol_sha256':digest,'source_sha256':code,'report_generated':False},indent=2))


if __name__=='__main__':
    main()
