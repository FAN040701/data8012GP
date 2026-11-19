"""Label-blind largest-component intervention on saved predictions; original scores untouched."""
import argparse
import concurrent.futures as cf
from datetime import datetime, timezone
import json

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from seed_stability_common import ROOT, OUT, CFG, sha

DEST=OUT/'component_diagnostic'
PROTOCOL=ROOT/'configs/seed_component_diagnostic.json'
ORGANS=['background','spleen','right_kidney','left_kidney','liver']


def process_case(task):
    case,seeds=task
    meta=json.loads((ROOT/'data/processed'/case/'metadata.json').read_text())
    source=nib.as_closest_canonical(nib.load(ROOT/meta['raw_label']))
    raw=np.asarray(source.dataobj,dtype=np.uint8)
    target=np.zeros(raw.shape,dtype=np.uint8)
    for old,new in [(1,1),(2,2),(3,3),(6,4)]:
        target[raw==old]=new
    voxel_ml=float(np.prod(meta['original_spacing_xyz_mm'])/1000)
    structure=ndi.generate_binary_structure(3,1)
    result=[]
    for seed in seeds:
        for arm in CFG['arms']:
            if seed in CFG['old_seeds']:
                folder=ROOT/'results/mechanism/replication/runs'/f'{arm}_seed{seed}'
            else:
                folder=OUT/'evaluations/best'/f'{arm}_seed{seed}'/'new40_replication'
            path=folder/'predictions/f3'/f'{case}.nii.gz'
            frame=pd.read_csv(folder/'metrics.csv')
            frame=frame[(frame.case==case)&(frame.factor==3)].set_index('organ')
            pred=np.asarray(nib.load(path).dataobj,dtype=np.uint8)
            assert pred.shape==target.shape
            for k in range(1,5):
                mask=pred==k
                # Cropping to the foreground bounding box preserves all connectivity.
                coordinates=np.where(mask)
                nt=int(np.count_nonzero(target==k))
                npred=int(mask.sum())
                tp=int(np.count_nonzero(mask & (target==k)))
                ncomp=kept=kept_tp=0
                if npred:
                    bounds=tuple(slice(int(x.min()),int(x.max())+1) for x in coordinates)
                    labels,ncomp=ndi.label(mask[bounds],structure=structure)
                    sizes=np.bincount(labels.ravel())
                    sizes[0]=0
                    largest=int(sizes.argmax())
                    keep=labels==largest
                    kept=int(keep.sum())
                    kept_tp=int(np.count_nonzero(keep & (target[bounds]==k)))
                before=2*tp/(nt+npred) if nt else np.nan
                after=2*kept_tp/(nt+kept) if nt else np.nan
                expected=frame.loc[ORGANS[k]]
                assert np.isclose(before,expected.dice,atol=1e-12,rtol=0,equal_nan=True)
                assert npred==expected.pred_voxels and nt==expected.true_voxels
                removed_tp=tp-kept_tp
                removed_fp=npred-kept-removed_tp
                assert removed_tp>=0 and removed_fp>=0
                result.append({'case':case,'arm':arm,'seed':seed,'organ':ORGANS[k],
                    'dice_before':before,'dice_after':after,'dice_change':after-before,
                    'true_voxels':nt,'pred_voxels':npred,'tp_before':tp,
                    'kept_voxels':kept,'tp_after':kept_tp,'components':int(ncomp),
                    'removed_true_positive_voxels':removed_tp,'removed_false_positive_voxels':removed_fp,
                    'removed_true_positive_ml':removed_tp*voxel_ml,'removed_false_positive_ml':removed_fp*voxel_ml,
                    'source_prediction':str(path.relative_to(ROOT)),'source_prediction_sha256':sha(path)})
            del pred
    return result


def summarize():
    from analyze_seed_stability import summarize as effect, GROUPS, WEIGHTS
    frames=[]
    for group in ['original3','additional3']:
        assert json.loads((DEST/f'{group}_complete.json').read_text())['all_passed']
        frames.append(pd.read_csv(DEST/f'{group}_organ_metrics.csv'))
    raw=pd.concat(frames,ignore_index=True)
    assert len(raw)==40*6*4*4 and not raw.duplicated(['case','seed','arm','organ']).any()
    raw.to_csv(DEST/'all_organ_metrics.csv',index=False)
    patient=raw.groupby(['case','seed','arm'],as_index=False)[['dice_before','dice_after','removed_true_positive_ml','removed_false_positive_ml']].mean()
    patient.to_csv(DEST/'patient_metrics.csv',index=False)
    rows=[]
    def add(series,meta):
        matrix=series.unstack('seed').sort_index().sort_index(axis=1)
        for group,seeds in GROUPS.items():
            rows.append({**meta,'seed_group':group,**effect(matrix[seeds])})
    wide={metric:patient.pivot(index=['case','seed'],columns='arm',values=metric) for metric in ['dice_before','dice_after']}
    for arm in CFG['arms']:
        add(wide['dice_after'][arm]-wide['dice_before'][arm],{'contrast':arm,'condition':'postprocessing_gain'})
    for name,weights in WEIGHTS.items():
        effects={metric:sum(wide[metric][a]*w for a,w in weights.items()) for metric in wide}
        for metric,series in effects.items():
            add(series,{'contrast':name,'condition':metric})
        add(effects['dice_after']-effects['dice_before'],{'contrast':name,'condition':'after_minus_before'})
    pd.DataFrame(rows).to_csv(DEST/'paired_contrasts.csv',index=False)
    per_case=raw.groupby(['case','arm','organ'],as_index=False).mean(numeric_only=True)
    per_case.to_csv(DEST/'patient_organ_means.csv',index=False)
    illustrative=json.loads((ROOT/'results/mechanism/replication_spatial_errors/complete.json').read_text())
    summary={'all_passed':True,'completed_utc':datetime.now(timezone.utc).isoformat(),'saved_predictions_reused':960,'organ_rows':len(raw),'new_training_or_inference':0,'protocol_sha256':sha(PROTOCOL),'source_sha256':sha(__file__),'interpretation':'Label-blind fixed postprocessing diagnostic. Reduction localizes errors to removable connected components; it does not establish a training mechanism. Removing small components can remove genuine foreground or retain a larger wrong component. Original unprocessed scores remain primary.','report_generated':False}
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
    digest=sha(PROTOCOL)
    code=sha(__file__)
    rows=[]
    with cf.ProcessPoolExecutor(2) as pool:
        for case,result in zip(CFG['new40_ids'],pool.map(process_case,[(c,seeds) for c in CFG['new40_ids']])):
            rows.extend(result)
            print(args.group,case,len(rows)//48,'/40 complete',flush=True)
    frame=pd.DataFrame(rows).sort_values(['case','seed','arm','organ'])
    assert len(frame)==1920 and sha(PROTOCOL)==digest and sha(__file__)==code
    path=DEST/f'{args.group}_organ_metrics.csv'
    frame.to_csv(path,index=False)
    marker.write_text(json.dumps({'all_passed':True,'completed_utc':datetime.now(timezone.utc).isoformat(),'saved_predictions_recounted':480,'organ_rows':len(frame),'csv_sha256':sha(path),'source_sha256':code,'protocol_sha256':digest,'report_generated':False},indent=2))


if __name__=='__main__':
    main()
