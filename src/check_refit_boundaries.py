"""Recompute native-space directional HD95 for independently refitted original models."""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from proposal_quality import distances
from prepare_data import NAMES
from verify_mechanism import target

ROOT=Path(__file__).resolve().parents[1]
FRESH=ROOT/'environment/tmp/reproduction_check'


def one(case):
    y, meta=target(case)
    rows=[]
    for kind in ['2d','2p5d','3d']:
        path=FRESH/'results/runs'/f'{kind}_seed11'/'predictions/f1'/f'{case}.nii.gz'
        image=nib.load(path)
        assert image.shape==y.shape and np.allclose(image.affine,meta['original_affine'])
        p=np.asarray(image.dataobj,dtype=np.uint8)
        for k,organ in enumerate(NAMES[1:],1):
            a,b=p==k,y==k
            assert a.any() and b.any()
            d=distances(a,b,meta['original_spacing_xyz_mm'])
            rows.append(dict(case=case,kind=kind,organ=organ,hd95_mm=max(d[:2])))
    return rows


if __name__=='__main__':
    cases=json.loads((ROOT/'data/metadata/split.json').read_text())['splits']['test']
    with ProcessPoolExecutor(8) as pool:
        rows=pd.DataFrame([r for batch in pool.map(one,cases) for r in batch])
    assert len(rows)==480
    out=ROOT/'results/final_study'
    rows.to_csv(out/'refit_native_hd95.csv',index=False)
    summary=rows.groupby('kind').hd95_mm.mean()
    summary.to_csv(out/'refit_hd95_summary.csv')
    print(summary.to_string())
