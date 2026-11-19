"""Independent checks of completed outputs, geometry, metrics and experiment provenance."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

ROOT=Path(__file__).resolve().parents[1]
KINDS=['2d','repeat','2p5d','3d','2p5d_aug']
ORGANS=['spleen','right_kidney','left_kidney','liver']
ORIGINAL_LABELS=[1,2,3,6]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''): h.update(block)
    return h.hexdigest()


def main():
    cfg=json.loads((ROOT/'configs/experiment.json').read_text())
    split=json.loads((ROOT/'data/metadata/split.json').read_text())['splits']
    cases=sorted(split['test']); seeds=cfg['seeds']
    assert len(cases)==40 and len(split['train'])==80 and len(split['val'])==20
    assert not (set(split['train'])&set(split['val']) or set(split['train'])&set(cases) or set(split['val'])&set(cases))
    kinds=KINDS if cfg['include_augmentation'] else KINDS[:-1]
    expected={(k,s) for k in kinds for s in seeds}
    completed=list((ROOT/'results/runs').glob('*/evaluation_complete.json'))
    assert len(completed)==len(expected)
    records=[]; timelines={}; count_predictions=0
    source_names=['experiment.py','models.py','prepare_data.py']
    source_hashes={n:sha(ROOT/'src'/n) for n in source_names}
    for k,s in sorted(expected):
        run=ROOT/'results/runs'/f'{k}_seed{s}'
        train=json.loads((run/'metadata.json').read_text())
        done=json.loads((run/'complete.json').read_text())
        ev=json.loads((run/'evaluation_complete.json').read_text())
        assert not train['pilot'] and train['train_ids']==split['train'] and train['val_ids']==split['val']
        assert done['training_steps']==cfg['steps'] and done['best_step']==ev['checkpoint_step']
        assert all(train['source_sha256'][n]==source_hashes[n] for n in source_names)
        assert all(digest==sha(ROOT/'src'/n) for n,digest in ev['evaluation_source_sha256'].items())
        assert ev['rows']==len(cases)*3*4 and ev['cases']==len(cases)
        frame=pd.read_csv(run/'metrics.csv')
        assert len(frame)==ev['rows'] and not frame.duplicated(['case','factor','organ']).any()
        assert frame['dice'].between(0,1).all()
        assert frame['hd95_mm'].dropna().ge(0).all()
        assert set(frame.kind)=={k} and set(frame.seed)=={s}
        for f in [1,2,3]:
            assert set(frame.loc[frame.factor==f,'case'])==set(cases)
            found={p.name.removesuffix('.nii.gz') for p in (run/'predictions'/f'f{f}').glob('*.nii.gz')}
            assert found==set(cases)
            count_predictions+=len(found)
        sampled=pd.read_json(run/'train.jsonl',lines=True)
        assert sampled.step.iloc[-1]==cfg['steps']
        assert sampled.loss.map(np.isfinite).all() and sampled.grad_norm.gt(0).all()
        timelines[k,s]=list(zip(sampled.step.tolist(),sampled.patient.tolist()))
        records.append(frame)
    for s in seeds:
        assert all(timelines[k,s]==timelines['repeat',s] for k in kinds)
    raw=pd.concat(records,ignore_index=True)
    analysis=pd.read_csv(ROOT/'results/analysis/all_organ_metrics.csv')
    keys=['kind','seed','case','factor','organ']
    pd.testing.assert_frame_equal(raw.sort_values(keys).reset_index(drop=True),analysis.sort_values(keys).reset_index(drop=True))

    # Rebuild the balanced score cube directly from raw rows, independently of analysis groupby code.
    cube=np.full((len(kinds),len(seeds),len(cases),3,4),np.nan)
    for r in raw.itertuples():
        cube[kinds.index(r.kind),seeds.index(r.seed),cases.index(r.case),r.factor-1,ORGANS.index(r.organ)]=r.dice
    assert np.isfinite(cube).all()  # All four reference organs are present in this fixed test cohort.
    patient_scores=cube.mean(axis=4).mean(axis=1)
    means=patient_scores.mean(axis=1)
    summary=pd.read_csv(ROOT/'results/analysis/model_summary.csv')
    for ki,k in enumerate(kinds):
        assert np.allclose(summary[summary.kind==k].sort_values('factor').dice_mean,means[ki],rtol=0,atol=1e-12)
    contrasts=pd.read_csv(ROOT/'results/analysis/paired_contrasts.csv')
    for a,b,name in [('2p5d','repeat','context_gain'),('3d','2d','3d_minus_2d'),('2p5d_aug','2p5d','augmentation_gain')]:
        if a not in kinds: continue
        diff=patient_scores[kinds.index(a)]-patient_scores[kinds.index(b)]
        for condition,x in [(f'f{f}',diff[:,f-1]) for f in [1,2,3]]+[(f'f{f}_minus_f1',diff[:,f-1]-diff[:,0]) for f in [2,3]]:
            row=contrasts[(contrasts.contrast==name)&(contrasts.condition==condition)].iloc[0]
            assert abs(x.mean()-row['mean'])<1e-12
            rng=np.random.default_rng(20261114)
            draws=rng.choice(x,size=(2000,len(x)),replace=True).mean(1)
            assert np.allclose(np.quantile(draws,[.025,.975]),[row.ci_low,row.ci_high],rtol=0,atol=1e-12)

    # Fixed lexical-first test case, chosen without examining performance.
    case=cases[0]
    meta=json.loads((ROOT/'data/processed'/case/'metadata.json').read_text())
    reference=nib.as_closest_canonical(nib.load(ROOT/meta['raw_label']))
    truth=np.asarray(reference.dataobj,dtype=np.uint8)
    spacing=np.asarray(reference.header.get_zooms()[:3])
    checked_masks=0; max_dice_error=0.0; distance_errors=[]
    for k,s in sorted(expected):
        for f in [1,2,3]:
            path=ROOT/'results/runs'/f'{k}_seed{s}'/'predictions'/f'f{f}'/f'{case}.nii.gz'
            predicted=nib.load(path)
            assert predicted.shape==reference.shape and np.allclose(predicted.affine,reference.affine,atol=1e-5)
            assert predicted.header.get_xyzt_units()[0]=='mm'
            labels=np.asarray(predicted.dataobj,dtype=np.uint8)
            assert set(np.unique(labels))<=set(range(5))
            for output_id,(org,original_id) in enumerate(zip(ORGANS,ORIGINAL_LABELS),1):
                p=labels==output_id; t=truth==original_id
                score=2*np.count_nonzero(p&t)/(np.count_nonzero(p)+np.count_nonzero(t))
                row=raw[(raw.kind==k)&(raw.seed==s)&(raw.case==case)&(raw.factor==f)&(raw.organ==org)].iloc[0]
                assert p.sum()==row.pred_voxels and t.sum()==row.true_voxels
                max_dice_error=max(max_dice_error,abs(score-row.dice))
                # Separate nearest-neighbour implementation checks physical HD95 against EDT results.
                if k=='repeat' and s==seeds[0]:
                    ps=np.argwhere(p&~ndi.binary_erosion(p))*spacing
                    ts=np.argwhere(t&~ndi.binary_erosion(t))*spacing
                    distances=np.concatenate([cKDTree(ts).query(ps,workers=2)[0],cKDTree(ps).query(ts,workers=2)[0]])
                    independent_hd95=np.quantile(distances,.95)
                    distance_errors.append(abs(independent_hd95-row.hd95_mm))
            checked_masks+=1
    assert max_dice_error<1e-12 and max(distance_errors)<1e-6

    checks={'utc':datetime.now(timezone.utc).isoformat(),'all_passed':True,'completed_runs':len(expected),
            'metric_rows':len(raw),'saved_native_predictions':count_predictions,
            'matched_logged_patient_schedule':True,'source_hashes_match':True,
            'independent_balanced_aggregation_and_bootstrap':True,
            'native_mask_check_case':case,'native_masks_checked':checked_masks,
            'maximum_dice_disagreement':max_dice_error,
            'hd95_independent_kdtree_checks':len(distance_errors),'maximum_hd95_disagreement_mm':max(distance_errors)}
    (ROOT/'results/deliverable_checks.json').write_text(json.dumps(checks,indent=2))
    manifest={}
    for pattern in ['src/*','configs/*','results/analysis/*.csv','results/analysis/*.json','results/*checks.json','results/runs/*/best.pt','results/runs/*/metrics.csv']:
        for p in ROOT.glob(pattern):
            if p.is_file(): manifest[str(p.relative_to(ROOT))]={'bytes':p.stat().st_size,'sha256':sha(p)}
    manifest['README.md']={'bytes':(ROOT/'README.md').stat().st_size,'sha256':sha(ROOT/'README.md')}
    (ROOT/'results/deliverable_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(checks,indent=2))


if __name__=='__main__': main()
