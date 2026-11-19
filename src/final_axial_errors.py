"""Attribute the signed mirror-policy Dice difference to axial regions and FP/FN voxels."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from prepare_data import NAMES, remap_labels
from verify_mechanism import target
from analyze_offsets import bootstrap
from slice_selection import ROOT, CFG, run_dir

OUT = ROOT / 'results/final_study/axial'
SEEDS = [11,22,33,44,55,66]
ARMS = ['near_far','far_near','near_far_aug','far_near_aug']


def one_case(job):
    case, seeds = job
    y, meta = target(case)
    z = np.arange(y.shape[-1])
    # Three equal-length axial bands per reference organ, extending the outer bands beyond its ends.
    bands = {}
    for k in range(1,5):
        occupied = z[(y==k).any(axis=(0,1))]
        if len(occupied):
            a = occupied[0] + np.array([1,2]) * (occupied[-1]-occupied[0]+1)/3
            bands[k] = np.digitize(z,a)
    rows = []
    for seed in seeds:
        for arm in ARMS:
            folder = run_dir(arm,seed)
            file = folder/'predictions/f3'/f'{case}.nii.gz'
            prediction = np.asarray(nib.load(file).dataobj, dtype=np.uint8)
            assert prediction.shape == y.shape and prediction.max() <= 4
            checks = pd.read_csv(folder/'metrics.csv').query('case==@case and factor==3').set_index('organ')
            for k, band in bands.items():
                p, t = prediction==k, y==k
                fp = (p & ~t).sum(axis=(0,1)); fn = (~p & t).sum(axis=(0,1))
                denom = int(p.sum()+t.sum())
                assert np.isclose(1-(fp.sum()+fn.sum())/denom,checks.loc[NAMES[k],'dice'],atol=1e-12)
                for b,name in enumerate(['inferior','middle','superior']):
                    rows.append(dict(case=case,seed=seed,arm=arm,organ=NAMES[k],band=name,
                                     fp_loss=fp[band==b].sum()/denom,fn_loss=fn[band==b].sum()/denom))
    return rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('group',choices=['first3','new3','summarize'])
    action=parser.parse_args().group
    OUT.mkdir(exist_ok=True, parents=True)
    if action!='summarize':
        cases=json.loads((ROOT/CFG['test_split']).read_text())['test']
        seeds=SEEDS[:3] if action=='first3' else SEEDS[3:]
        with ProcessPoolExecutor(4) as pool:
            rows=[row for part in pool.map(one_case,[(c,seeds) for c in cases]) for row in part]
        pd.DataFrame(rows).to_csv(OUT/f'{action}.csv',index=False)
        return
    frame=pd.concat([pd.read_csv(OUT/f'{group}.csv') for group in ['first3','new3']])
    scores=frame.groupby(['case','seed','arm','band'])[['fp_loss','fn_loss']].mean()
    results=[]
    for band in ['inferior','middle','superior']:
        for measure in ['fp_loss','fn_loss']:
            w=scores.xs(band,level='band')[measure].unstack('arm')
            plain=w.far_near-w.near_far
            aug=w.far_near_aug-w.near_far_aug
            for name,term in [('direction_plain',plain),('direction_aug',aug),('direction_by_augmentation',aug-plain)]:
                results.append(dict(band=band,component=measure,contrast=name,**bootstrap(term)))
    out=pd.DataFrame(results)
    contrasts=pd.read_csv(OUT.parent/'contrasts.csv')
    full=contrasts.query("cohort=='replication' and checkpoint=='best' and group=='all6' and condition=='f3'").set_index('contrast')
    for name,q in out.groupby('contrast'):
        assert np.isclose(q['mean'].sum(),full.loc[name,'mean'],atol=1e-12,rtol=0)
    out.to_csv(OUT/'contributions.csv',index=False)


if __name__=='__main__':
    main()
