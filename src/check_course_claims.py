"""Independent checks of report metrics from original masks and paired patient scores."""
import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree

from verify_mechanism import target
from prepare_data import NAMES

ROOT=Path(__file__).resolve().parents[1]
FINAL=ROOT/'results/final_study'


def original():
    source=pd.read_csv(ROOT/'results/slice_selection/proposal_quality/native_organ_metrics.csv')
    case=sorted(source.case.unique())[len(source.case.unique())//2]
    label,meta=target(case)
    rows=[]
    for kind in ['2d','2p5d','3d']:
        for factor in [1,3]:
            path=ROOT/'results/runs'/f'{kind}_seed11/predictions'/f'f{factor}'/f'{case}.nii.gz'
            pred=np.asanyarray(nib.load(path).dataobj)
            for k,name in enumerate(NAMES[1:],1):
                sets=[s==k for s in [label,pred]]
                dice=2*(sets[0]&sets[1]).sum()/(sets[0].sum()+sets[1].sum())
                points=[np.argwhere(s ^ ndimage.binary_erosion(s))*meta['original_spacing_xyz_mm'] for s in sets]
                p,q=points
                hd=max(np.percentile(cKDTree(a).query(b)[0],95) for a,b in [(p,q),(q,p)])
                measured=source[(source.case==case)&(source.kind==kind)&(source.factor==factor)&(source.seed==11)&(source.organ==name)].iloc[0]
                assert np.isclose(dice,measured.dice,atol=1e-12,rtol=0)
                assert np.isclose(hd,measured.hd95_mm,atol=1e-8,rtol=0)
                rows.append([float(abs(dice-measured.dice)),float(abs(hd-measured.hd95_mm))])
    return {'patient':case,'native_organ_measurements':len(rows),'max_Dice_error':max(x[0] for x in rows),
            'max_HD95_error_mm':max(x[1] for x in rows)}


def intervals():
    c=pd.read_csv(FINAL/'contrasts.csv')
    patients=pd.read_csv(FINAL/'patient_scores.csv')
    recs=[]
    for co,sel,group,seeds in [('replication','best','new3',[44,55,66]),('replication','best','all6',[11,22,33,44,55,66]),
                               ('original','best','first3',[11,22,33]),('replication','last','all6',[11,22,33,44,55,66])]:
        p=patients[(patients.cohort==co)&(patients.checkpoint==sel)&(patients.factor==3)&patients.seed.isin(seeds)]
        wide=p.pivot(index=['case','seed'],columns='arm',values='dice')
        delta=(wide.near_far_aug+wide.far_near)-(wide.near_far+wide.far_near_aug)
        mat=delta.unstack().sort_index().to_numpy()
        # Compute the bootstrap with multinomial counts applied to the matrix.
        # This independently reconstructs the interval generator's equal-patient/equal-seed weights.
        rng=np.random.default_rng(20261119)
        pidx=rng.integers(40,size=(10000,40));sidx=rng.integers(len(seeds),size=(10000,len(seeds)))
        pw=np.array([np.bincount(x,minlength=40) for x in pidx])
        sw=np.array([np.bincount(x,minlength=len(seeds)) for x in sidx])
        means=np.einsum('bp,ps,bs->b',pw,mat,sw)/(40*len(seeds))
        ref=c[(c.cohort==co)&(c.checkpoint==sel)&(c.group==group)&(c.contrast=='direction_by_augmentation')&(c.condition=='f3')].iloc[0]
        estimate=np.array([mat.mean(),*np.quantile(means,[.025,.975])])
        assert np.allclose(estimate,ref[['mean','ci_low','ci_high']].astype(float),atol=1e-12,rtol=0)
        recs.append(dict(cohort=co,checkpoint=sel,group=group,patients=mat.shape[0],seeds=mat.shape[1]))
    return {'independently_rebuilt_intervals':recs}


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('scope',choices=['original','intervals'])
    args=p.parse_args()
    result=original() if args.scope=='original' else intervals()
    FINAL.mkdir(parents=True,exist_ok=True)
    (FINAL/f'check_{args.scope}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result)
