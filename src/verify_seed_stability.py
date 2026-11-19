"""Independent checks for new training, original results, masks and six-seed statistics."""
import concurrent.futures as cf
from datetime import datetime, timezone
import json

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from scipy import ndimage as ndi

from seed_stability_common import ROOT, OUT, CFG, PROTOCOL, sha, jobs, run_dir

NAMES=['background','spleen','right_kidney','left_kidney','liver']


def header(item):
    path,case=item
    image=nib.load(ROOT/path)
    meta=json.loads((ROOT/'data/processed'/case/'metadata.json').read_text())
    assert list(image.shape)==meta['original_shape_xyz']
    assert np.allclose(image.affine,meta['original_affine'],atol=1e-4)
    assert image.header.get_xyzt_units()[0]=='mm'
    assert image.get_data_dtype()==np.dtype('uint8')
    return True


def mask(task):
    rows,hd=task
    case,path=rows[0]['case'],rows[0]['prediction_file']
    meta=json.loads((ROOT/'data/processed'/case/'metadata.json').read_text())
    raw=np.asarray(nib.as_closest_canonical(nib.load(ROOT/meta['raw_label'])).dataobj,dtype=np.uint8)
    y=np.zeros(raw.shape,dtype=np.uint8)
    for old,new in [(1,1),(2,2),(3,3),(6,4)]:
        y[raw==old]=new
    p=np.asarray(nib.load(ROOT/path).dataobj,dtype=np.uint8)
    assert set(np.unique(p)) <= set(range(5))
    error=0.
    hd_errors=[]
    for row in rows:
        k=NAMES.index(row['organ'])
        nt,npred=int((y==k).sum()),int((p==k).sum())
        tp=int(np.count_nonzero((p==k)&(y==k)))
        value=2*tp/(nt+npred) if nt else np.nan
        assert nt==row['true_voxels'] and npred==row['pred_voxels']
        assert np.isclose(value,row['dice'],atol=1e-12,rtol=0,equal_nan=True)
        if nt:
            error=max(error,abs(value-row['dice']))
        if row['cohort']=='absolute_spacing':
            dz=meta['original_spacing_xyz_mm'][2]
            indices=np.arange(y.shape[2])
            keep=(indices*dz>=9)&((y.shape[2]-1-indices)*dz>=9)
            pp,yy=p[:,:,keep]==k,y[:,:,keep]==k
            ni,pi=int(yy.sum()),int(pp.sum())
            iv=2*np.count_nonzero(pp&yy)/(ni+pi) if ni else np.nan
            assert ni==row['interior_reference_voxels'] and pi==row['interior_prediction_voxels']
            assert np.isclose(iv,row['interior_dice'],atol=1e-12,rtol=0,equal_nan=True)
        if hd and k in [1,2] and nt and npred:
            spacing=np.array(meta['original_spacing_xyz_mm'])
            structure=ndi.generate_binary_structure(3,1)
            pp,yy=p==k,y==k
            a=np.argwhere(pp^ndi.binary_erosion(pp,structure,border_value=0))*spacing
            b=np.argwhere(yy^ndi.binary_erosion(yy,structure,border_value=0))*spacing
            distances=np.r_[cKDTree(b).query(a,workers=1)[0],cKDTree(a).query(b,workers=1)[0]]
            diff=abs(float(np.percentile(distances,95))-row['hd95_mm'])
            assert diff<1e-6
            hd_errors.append(diff)
    return {'dice_error':error,'hd95_errors':hd_errors}


def main():
    analysis=json.loads((OUT/'analysis/analysis.json').read_text())
    assert analysis['all_passed']
    frozen=json.loads((OUT/'pre_experiment_snapshot.json').read_text())['files']
    for p,digest in frozen.items():
        assert sha(ROOT/p)==digest,p
    inputs=json.loads((OUT/'input_snapshot.json').read_text())['files']
    with cf.ThreadPoolExecutor(4) as pool:
        actual=list(pool.map(sha,[ROOT/p for p in inputs]))
    assert actual==list(inputs.values())
    trace_by_seed={}
    selections=[]
    initial=json.loads((OUT/'initializations.json').read_text())
    for arm,seed in jobs():
        folder=run_dir(arm,seed)
        meta=json.loads((folder/'metadata.json').read_text())
        done=json.loads((folder/'complete.json').read_text())
        assert done['training_steps']==40000 and meta['parameters']==1964165
        assert meta['initial_parameter_sha256']==initial[str(seed)]
        assert meta['protocol_sha256']==sha(PROTOCOL)
        for name,digest in meta['source_sha256'].items():
            assert sha(ROOT/'src'/name)==digest
        trace=pd.read_json(folder/'train.jsonl',lines=True)
        assert len(trace)==801 and trace.step.iloc[-1]==40000
        assert np.isfinite(trace[['loss','grad_norm','lr']].to_numpy()).all()
        key=seed
        if key in trace_by_seed:
            previous,prior_done=trace_by_seed[key]
            for field in ['step','patient','start','lr']:
                assert trace[field].tolist()==previous[field].tolist()
            assert done['sampling_sha256']==prior_done['sampling_sha256']
        else:
            trace_by_seed[key]=(trace,done)
        counts=done['factor_counts']
        assert sum(counts.values())==40000
        if not arm.endswith('_aug'):
            assert counts=={'1':40000,'2':0,'3':0}
        else:
            rng=np.random.default_rng(seed+100000)
            factors=rng.choice([1,2,3],p=[.5,.25,.25],size=40000)
            assert counts=={str(k):int((factors==k).sum()) for k in [1,2,3]}
            assert trace.factor.tolist()==factors[trace.step.to_numpy()-1].tolist()
        validation=[json.loads(x) for x in (folder/'validation.jsonl').read_text().splitlines()]
        assert [r['step'] for r in validation]==list(range(2000,40001,2000))
        assert all(len(r['cases'])==20 for r in validation)
        chosen=max(validation,key=lambda r:(r['macro_dice'],-r['step']))
        assert chosen['step']==done['best_step'] and chosen['macro_dice']==done['best_validation_macro_dice']
        assert sha(folder/'best.pt')==done['best_checkpoint_sha256']
        for selection in ['best','last']:
            checkpoint=torch.load(folder/f'{selection}.pt',map_location='cpu',weights_only=False)
            assert checkpoint['step']==(done['best_step'] if selection=='best' else 40000)
        selections.append({'arm':arm,'seed':seed,'best_step':done['best_step']})
    newframes=[]
    records=0
    for selection,seeds in [('best',CFG['new_seeds']),('last',CFG['old_seeds']+CFG['new_seeds'])]:
        for seed in seeds:
            for arm in CFG['arms']:
                folder=OUT/'evaluations'/selection/f'{arm}_seed{seed}'
                record=json.loads((folder/'complete.json').read_text())
                assert record['protocol_sha256']==sha(PROTOCOL)
                assert sha(ROOT/record['checkpoint'])==record['checkpoint_sha256']
                for name,digest in record['source_sha256'].items():
                    assert sha(ROOT/'src'/name)==digest
                frame=pd.read_csv(folder/'metrics.csv')
                assert len(frame)==(648 if selection=='best' else 160)
                newframes.append(frame)
                records+=1
    raw=pd.concat(newframes,ignore_index=True)
    files=raw[['prediction_file','case']].drop_duplicates()
    assert len(files)==2904 and len(raw)==11616
    with cf.ThreadPoolExecutor(8) as pool:
        assert all(pool.map(header,files.itertuples(index=False,name=None)))
    tasks=[]
    for key,g in raw.groupby(['cohort','selection','arm','seed','factor']):
        cases=sorted(g.case.unique())
        for case in [cases[0],cases[-1]]:
            tasks.append((g[g.case==case].to_dict('records'),key[2]=='repeat' and key[3]==44 and case==cases[0]))
    with cf.ProcessPoolExecutor(4) as pool:
        checked=list(pool.map(mask,tasks))
    # If the selected checkpoint is already the final one, the two evaluation paths
    # must reproduce identical scores and sampled masks. This checks integration.
    identical_checkpoint_checks=[]
    for seed in CFG['old_seeds']+CFG['new_seeds']:
        for arm in CFG['arms']:
            folder=run_dir(arm,seed) if seed in CFG['new_seeds'] else ROOT/'results/continuation/b40000'/f'{arm}_seed{seed}'
            complete=json.loads((folder/'complete.json').read_text())
            if complete['best_step']!=40000:
                continue
            best=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
            last=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
            assert all(torch.equal(value,last['state_dict'][name]) for name,value in best['state_dict'].items())
            best_metrics=OUT/'evaluations/best'/f'{arm}_seed{seed}'/'new40_replication/metrics.csv' if seed in CFG['new_seeds'] else ROOT/'results/mechanism/replication/runs'/f'{arm}_seed{seed}'/'metrics.csv'
            a=pd.read_csv(best_metrics)
            a=a[a.factor==3].set_index(['case','organ']).sort_index()
            b=raw[(raw.arm==arm)&(raw.seed==seed)&(raw.selection=='last')].set_index(['case','organ']).sort_index()
            assert a.index.equals(b.index)
            for field in ['dice','pred_voxels','true_voxels','hd95_mm']:
                assert np.allclose(a[field],b[field],atol=1e-12,rtol=0,equal_nan=True),(arm,seed,field)
            cases=sorted(a.index.get_level_values('case').unique())
            for case in [cases[0],cases[-1]]:
                first=a.loc[case].prediction_file.iloc[0]
                second=b.loc[case].prediction_file.iloc[0]
                assert np.array_equal(np.asarray(nib.load(ROOT/first).dataobj),np.asarray(nib.load(ROOT/second).dataobj))
            identical_checkpoint_checks.append({'arm':arm,'seed':seed,'patients':40,'sampled_masks':2})
    # Independently aggregate original organ rows; do not reuse analyzer patient tables.
    old=pd.read_csv(ROOT/'results/mechanism/replication/runs/repeat_seed11/metrics.csv')
    matrices={}
    for seed in CFG['old_seeds']+CFG['new_seeds']:
        columns={}
        for arm in CFG['arms']:
            if seed in CFG['old_seeds']:
                p=ROOT/'results/mechanism/replication/runs'/f'{arm}_seed{seed}'/'metrics.csv'
            else:
                p=OUT/'evaluations/best'/f'{arm}_seed{seed}'/'new40_replication/metrics.csv'
            f=pd.read_csv(p)
            f=f[f.factor==3]
            columns[arm]=f.groupby('case').dice.mean().sort_index().to_numpy()
        matrices[seed]=columns['context_aug']-columns['repeat_aug']-columns['context']+columns['repeat']
    effects=pd.DataFrame(analysis['primary']).set_index('seed_group')
    for group,seeds in [('original3',CFG['old_seeds']),('additional3',CFG['new_seeds']),('all6',CFG['old_seeds']+CFG['new_seeds'])]:
        a=np.column_stack([matrices[s] for s in seeds])
        b=CFG['primary']['bootstrap_replicates']
        rng=np.random.default_rng(CFG['primary']['bootstrap_seed'])
        pi=rng.integers(40,size=(b,40))
        si=rng.integers(len(seeds),size=(b,len(seeds)))
        pw=np.zeros((b,40)); sw=np.zeros((b,len(seeds)))
        np.add.at(pw,(np.repeat(np.arange(b),40),pi.ravel()),1/40)
        np.add.at(sw,(np.repeat(np.arange(b),len(seeds)),si.ravel()),1/len(seeds))
        patient=pw@a.mean(1)
        crossed=np.einsum('bi,ij,bj->b',pw,a,sw)
        row=effects.loc[group]
        assert np.isclose(a.mean(),row['mean'],atol=1e-12,rtol=0)
        assert np.allclose(np.quantile(patient,[.025,.975]),[row.ci_low,row.ci_high],atol=1e-12,rtol=0)
        assert np.allclose(np.quantile(crossed,[.025,.975]),[row.crossed_ci_low,row.crossed_ci_high],atol=1e-12,rtol=0)
    result={'all_passed':True,'verified_utc':datetime.now(timezone.utc).isoformat(),'new_training_runs':12,'new_native_prediction_headers_checked':len(files),'native_masks_recounted':len(tasks),'maximum_dice_error':max(x['dice_error'] for x in checked),'independent_hd95_checks':sum(len(x['hd95_errors']) for x in checked),'maximum_hd95_error_mm':max([v for x in checked for v in x['hd95_errors']],default=0),'evaluation_records_verified':records,'original_files_unchanged':len(frozen),'input_files_unchanged':len(inputs),'paired_initializations_sampling_and_training_schedules_verified':True,'checkpoint_selections_recomputed':True,'identical_checkpoint_evaluation_checks':identical_checkpoint_checks,'primary_means_patient_and_crossed_intervals_independently_recomputed':True,'training_selections':selections,'source_sha256':sha(__file__),'report_generated':False}
    (OUT/'verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
