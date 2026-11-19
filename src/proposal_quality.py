"""Re-score the original three models using the proposal's directional HD95."""
import concurrent.futures
import json
from itertools import combinations
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/slice_selection/proposal_quality'
KINDS = ('2d','2p5d','3d')
ORGANS = {1:'spleen',2:'right_kidney',3:'left_kidney',4:'liver'}
SEEDS = (11,22,33)


def distances(pred, target, spacing):
    # Crop to the union so empty exterior voxels cost no distance transforms.
    points = ndi.find_objects((pred | target).astype(np.uint8))[0]
    def boundary(mask):
        region = mask[points]
        return region ^ ndi.binary_erosion(region, border_value=0)
    a, b = boundary(pred), boundary(target)
    ab = ndi.distance_transform_edt(~b, sampling=spacing)[a]
    ba = ndi.distance_transform_edt(~a, sampling=spacing)[b]
    return np.percentile(ab,95), np.percentile(ba,95), np.percentile(np.concatenate([ab,ba]),95)


def patient(case):
    dest = OUT/'cases'/f'{case}.csv'
    if dest.exists():
        return
    meta = json.loads((ROOT/'data/processed'/case/'metadata.json').read_text())
    ref = nib.as_closest_canonical(nib.load(ROOT/meta['raw_label']))
    raw = np.asarray(ref.dataobj, dtype=np.uint8)
    targets = {k: raw == label for k,label in zip(ORGANS,[1,2,3,6])}
    rows = []
    for kind in KINDS:
        for seed in SEEDS:
            for factor in [1,2,3]:
                path = ROOT/'results/runs'/f'{kind}_seed{seed}'/'predictions'/f'f{factor}'/f'{case}.nii.gz'
                image = nib.load(path)
                assert image.shape == raw.shape and np.allclose(image.affine,ref.affine,atol=1e-4)
                labels = np.asarray(image.dataobj,dtype=np.uint8)
                for k,name in ORGANS.items():
                    target, pred = targets[k], labels==k
                    nt, npred = int(target.sum()), int(pred.sum())
                    dice = 2*np.count_nonzero(target & pred)/(nt+npred) if nt else np.nan
                    forward = reverse = pooled = np.nan
                    if nt and npred:
                        forward, reverse, pooled = distances(pred,target,meta['original_spacing_xyz_mm'])
                    hd = max(forward,reverse) if nt and npred else (np.inf if nt else np.nan)
                    rows.append(dict(case=case,kind=kind,seed=seed,factor=factor,organ=name,
                        reference_voxels=nt,predicted_voxels=npred,dice=dice,pred_to_ref_mm=forward,
                        ref_to_pred_mm=reverse,hd95_mm=hd,pooled_hd95_mm=pooled))
    tmp=dest.with_suffix('.tmp')
    pd.DataFrame(rows).to_csv(tmp,index=False)
    tmp.replace(dest)


def main():
    (OUT/'cases').mkdir(exist_ok=True,parents=True)
    cases=json.loads((ROOT/'data/metadata/split.json').read_text())['splits']['test']
    with concurrent.futures.ProcessPoolExecutor(10) as pool:
        for i,_ in enumerate(pool.map(patient,cases),1):
            print('Patients checked',i,'/',len(cases),flush=True)
    raw=pd.concat([pd.read_csv(OUT/'cases'/f'{case}.csv') for case in cases])
    keys=['case','kind','seed','factor','organ']
    archive = ROOT/'results/analysis/all_organ_metrics.csv'
    if archive.is_file():
        old = pd.read_csv(archive)
        old = old[old.kind.isin(KINDS)]
    else:
        old = pd.concat([
            pd.read_csv(ROOT/'results/runs'/f'{kind}_seed{seed}'/'metrics.csv')
            for kind in KINDS for seed in SEEDS
        ], ignore_index=True)
    pair=raw.merge(old,on=keys,validate='one_to_one',suffixes=('','_old'))
    assert len(pair)==len(raw)==len(old)==4320
    assert np.allclose(pair.dice,pair.dice_old,atol=1e-12,equal_nan=True)
    assert np.allclose(pair.pooled_hd95_mm,pair.hd95_mm_old,atol=1e-5,equal_nan=True)
    raw.sort_values(keys).to_csv(OUT/'native_organ_metrics.csv',index=False)
    patient_means = raw.groupby(['kind','factor','organ','case'])[['dice','hd95_mm']].mean()
    distributions = []
    for (kind, factor, organ), patients in patient_means.groupby(['kind','factor','organ']):
        distances_mm = patients.hd95_mm[np.isfinite(patients.hd95_mm)]
        distributions.append(dict(
            kind=kind, factor=factor, organ=organ,
            present_patients=patients.dice.count(), absent_patients=patients.dice.isna().sum(),
            dice_mean=patients.dice.mean(), dice_sd=patients.dice.std(),
            hd95_finite_patients=len(distances_mm), hd95_median_mm=distances_mm.median(),
            hd95_q25_mm=distances_mm.quantile(.25), hd95_q75_mm=distances_mm.quantile(.75)))
    pd.DataFrame(distributions).to_csv(OUT/'organ_distributions.csv', index=False)
    summaries=[]; comparisons=[]
    for f,group in raw.groupby('factor'):
        # First average seeds for each patient and organ. Bootstrap whole patients.
        cases_sorted=sorted(cases)
        draws=np.random.default_rng(20261119).integers(0,len(cases),size=(2000,len(cases)))
        for metric in ['dice','hd95_mm']:
            matrices={}
            for kind in KINDS:
                mat=group[group.kind==kind].pivot_table(index='case',columns='organ',values=metric,
                          aggfunc='mean').reindex(index=cases_sorted,columns=list(ORGANS.values())).to_numpy()
                matrices[kind]=mat
            for name,mat in matrices.items():
                for i,organ in enumerate([*ORGANS.values(),'equal_organ_macro']):
                    values=mat[:,i] if i<4 else mat
                    mean=np.nanmean(values,axis=0).mean()
                    summaries.append(dict(kind=name,factor=f,organ=organ,metric=metric,mean=mean))
            for a,b in combinations(KINDS,2):
                delta=matrices[b]-matrices[a]
                for i,organ in enumerate([*ORGANS.values(),'equal_organ_macro']):
                    v=delta[:,i:i+1] if i<4 else delta
                    sample=np.nanmean(v[draws],axis=1).mean(axis=1)
                    lo,hi=np.quantile(sample,[.025,.975])
                    comparisons.append(dict(contrast=b+' minus '+a,factor=f,organ=organ,metric=metric,
                        mean=np.nanmean(v,axis=0).mean(),ci_low=lo,ci_high=hi,
                        patients=len(v),joint_observed_patient_organ_pairs=np.isfinite(v).sum()))
    pd.DataFrame(summaries).to_csv(OUT/'model_summary.csv',index=False)
    pd.DataFrame(comparisons).to_csv(OUT/'paired_comparisons.csv',index=False)
    (OUT/'verified.json').write_text(json.dumps(dict(validated_rows=len(pair),predictions=1080,
        cases=len(cases),empty_prediction_present_target=int(np.isinf(raw.hd95_mm).sum()),
        definition='maximum of two directed 95th percentiles; mm; native grid',
        pooled_hd95_and_dice_reproduce_original=True),indent=2))


if __name__=='__main__':
    main()
