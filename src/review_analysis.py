"""Paired analyses of same-seed refits, model capacity, channel order and cost."""
import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import t
from analyze_offsets import bootstrap
from analyze_seed_stability import summarize as summarize_factorial
from mechanism_common import sha
from review_evaluate import BANDS, MIRRORS, SWAP_SEEDS, read
from review_refits import ARMS, OUT, ROOT, SEEDS

SUMMARY = OUT/'summary'


def evaluated(group, arm, seed, repeat=None, checkpoint='best'):
    suffix = f'run{repeat}' if repeat is not None else 'best'
    if checkpoint == 'last':
        suffix += '_last'
    folder=OUT/'evaluation'/group/f'{arm}_seed{seed}'/suffix
    assert read(folder/'complete.json')['metrics_sha256'] == sha(folder/'organs.csv')
    df=pd.read_csv(folder/'organs.csv')
    assert len(df)==160 and df.case.nunique()==40 and df.organ.nunique()==4
    return df.assign(arm=arm,seed=seed)


def patient_interval(series):
    values=series.sort_index().to_numpy()
    draws=np.random.default_rng(20261119).integers(len(values),size=(10000,len(values)))
    lo,hi=np.quantile(values[draws].mean(axis=1),[.025,.975])
    return dict(mean=values.mean(),ci_low=lo,ci_high=hi)


def save(rows, name):
    pd.DataFrame(rows).to_csv(SUMMARY/f'{name}.csv',index=False)


def factorial():
    original=pd.read_csv(ROOT/'results/seed_stability/analysis/all_organ_metrics.csv').query(
        "cohort=='new40_replication' and selection=='best' and factor==3")
    old=original[original.seed==11]
    old=old[old.arm.isin(ARMS)][['case','organ','arm','seed','dice']].assign(run=0)
    assert len(old)==640 and not old.duplicated(['case','organ','arm']).any()
    data=[old]
    signatures=[]
    for arm in ARMS:
        archived=ROOT/'results/continuation/b40000'/f'{arm}_seed11'
        a,b=read(archived/'metadata.json'),read(archived/'complete.json')
        for n in (1,2):
            folder=OUT/f'refit{n}/results/continuation/b40000'/f'{arm}_seed11'
            c,d=read(folder/'metadata.json'),read(folder/'complete.json')
            archive_cfg = ROOT/'configs/continuation.json'
            replay_cfg = OUT/f'refit{n}/configs/continuation.json'
            assert read(archive_cfg) == read(replay_cfg)
            assert sha(archive_cfg) == a['protocol_sha256'] and sha(replay_cfg) == c['protocol_sha256']
            assert all(a[k]==c[k] for k in ('initial_parameter_sha256','train_ids','val_ids','parameters','source_sha256','config','augmentation'))
            assert all(b[k]==d[k] for k in ('sampling_sha256','factor_counts','training_steps'))
            refit=evaluated('factorial',arm,11,n)
            assert set(refit.case)==set(old.case)
            data.append(refit.assign(run=n))
            signatures.append(dict(run=n,arm=arm,seed=11,best_step=d['best_step'],
                historical_best_step=b['best_step'],checkpoint_sha256=d['best_checkpoint_sha256'],
                historical_checkpoint_sha256=b['best_checkpoint_sha256'],
                sampling_sha256=d['sampling_sha256'],initial_sha256=c['initial_parameter_sha256']))
    save(signatures,'factorial_provenance')
    pat=pd.concat(data).groupby(['case','run','arm']).dice.mean().unstack('arm')*100
    assert len(pat)==120 and pat.notna().all().all()
    pat['plain_gain']=pat.context-pat['repeat']
    pat['aug_gain']=pat.context_aug-pat.repeat_aug
    pat['interaction']=pat.aug_gain-pat.plain_gain
    pat.to_csv(OUT/'factorial_patient_pp.csv')
    out=[]
    for n,group in pat.groupby(level='run'):
        for effect in [*ARMS,'plain_gain','aug_gain','interaction']:
            out.append(dict(run=n,measure=effect,**patient_interval(group[effect])))
    save(out,'factorial_blocks_pp')
    orig=original.groupby(['case','seed','arm']).dice.mean().unstack('arm')
    orig_all=(orig.context_aug-orig.repeat_aug-orig.context+orig['repeat'])*100
    assert len(orig_all)==240
    sensitivity=[]
    for n in (0,1,2):
        candidate=orig_all.unstack('seed').copy()
        replacement=pat.interaction.xs(n,level='run')
        assert candidate.index.equals(replacement.index)
        if n==0:
            np.testing.assert_allclose(candidate[11],replacement,atol=1e-12,rtol=0)
        candidate[11]=replacement
        pooled = summarize_factorial(candidate)
        sensitivity.append(dict(seed11_block=n, mean=pooled['mean'], ci_low=pooled['crossed_ci_low'],
            ci_high=pooled['crossed_ci_high'], patients=pooled['patients'], seeds=pooled['seeds'],
            seed_means=json.dumps(candidate.mean().to_dict())))
    save(sensitivity,'factorial_six_seed_sensitivity_pp')
    blocks=pat.interaction.groupby('run').mean()
    mean,sd=blocks.mean(),blocks.std(ddof=1)
    margin=t.ppf(.975,df=len(blocks)-1)*sd/np.sqrt(len(blocks))
    save([dict(seed=11,patients=40,blocks=len(blocks),positive_blocks=int((blocks>0).sum()),
         mean_pp=mean,sd_pp=sd,min_pp=blocks.min(),max_pp=blocks.max(),
         conditional_t_low_pp=mean-margin,conditional_t_high_pp=mean+margin)],'factorial_variation')


def fixed_step():
    raw = pd.read_csv(ROOT/'results/seed_stability/analysis/all_organ_metrics.csv')
    archived = raw.query("cohort=='new40_replication' and seed==11 and factor==3")
    rows = []
    for n in (0, 1, 2):
        effects = {}
        for selection in ('best', 'last'):
            sub = archived[archived.selection == selection] if n == 0 else None
            if sub is None or sub.empty:
                sub = pd.concat([evaluated('factorial', arm, 11, n, selection) for arm in ARMS])
            assert len(sub) == 640 and not sub.duplicated(['case','organ','arm']).any()
            patient = sub.groupby(['case','arm']).dice.mean().unstack('arm') * 100
            assert len(patient) == 40 and patient.notna().all().all()
            effects[selection] = (patient.context_aug - patient.repeat_aug) - (patient.context - patient['repeat'])
        assert effects['last'].index.equals(effects['best'].index)
        for name, contrast in [('interaction at 40000', effects['last']),
                               ('final minus selected interaction', effects['last'] - effects['best'])]:
            rows.append(dict(run=n, measure=name, **patient_interval(contrast)))
    save(rows, 'factorial_fixed_step_pp')


def capacity():
    old=pd.read_csv(ROOT/'results/slice_selection/proposal_quality/native_organ_metrics.csv')
    old=old[(old.factor==1)&old.kind.isin(['2p5d','3d'])].copy()
    old['arm']=old.kind.map({'2p5d':'2p5d_w16','3d':'3d_w16'})
    data=[old]
    for seed in SEEDS:
        folder=OUT/'capacity/results/runs'/f'3d_seed{seed}'
        orig=ROOT/'results/runs'/f'3d_seed{seed}'
        b,a=read(orig/'metadata.json'),read(folder/'metadata.json')
        assert (a['parameters'],b['parameters'])==(1863086,5885573)
        assert a['config']['width']==9 and b['config']['width']==16
        assert a['train_ids']==b['train_ids'] and a['val_ids']==b['val_ids']
        assert a['source_sha256']==b['source_sha256']
        for field in ('steps','patch','lr','val_every'): assert a['config'][field]==b['config'][field]
        cols=['step','patient','lr']
        aa,bb=[pd.read_json(x/'train.jsonl',lines=True)[cols] for x in (folder,orig)]
        pd.testing.assert_frame_equal(aa,bb)
        q=evaluated('capacity','3d_w9',seed)
        assert set(q.case)==set(old.case)
        data.append(q)
    allrows=pd.concat(data)
    assert len(allrows)==3*3*40*4 and not allrows.duplicated(['case','seed','arm','organ']).any()
    assert np.isfinite(allrows[['dice','hd95_mm']]).all().all()
    results=[]
    for metric,unit,scale in [('dice','pp',100),('hd95_mm','mm',1)]:
        wide=allrows.pivot(index=['case','organ','seed'],columns='arm',values=metric)*scale
        for contrast,series in [('3d_w9 minus 2p5d_w16',wide['3d_w9']-wide['2p5d_w16']),
                                ('3d_w16 minus 3d_w9',wide['3d_w16']-wide['3d_w9'])]:
            for org in ['all',*sorted(old.organ.unique())]:
                sub=series if org=='all' else series.xs(org,level='organ')
                results.append(dict(metric=metric,unit=unit,organ=org,contrast=contrast,
                    seed_means=json.dumps(sub.groupby('seed').mean().to_dict()),
                    **patient_interval(sub.groupby('case').mean())))
    save(results,'capacity_contrasts')
    allrows.groupby(['arm'])[['dice','hd95_mm']].mean().to_csv(SUMMARY/'capacity_scores.csv')


def swap():
    frames=[evaluated('swap',a,s) for a in MIRRORS for s in SWAP_SEEDS]
    data=pd.concat(frames)
    assert len(data)==1920 and not data.duplicated(['case','seed','arm','organ']).any()
    tab=data.groupby(['case','seed','arm'])[['dice','ordinary_dice','fraction_labels_changed']].mean()
    out=[]
    audit=data.groupby(['case','seed','arm']).ordinary_fraction_different_from_archive.first()
    save([dict(paired_native_predictions=len(audit), bitwise_identical_ordinary=int((audit==0).sum()),
               maximum_fraction_different=audit.max(),mean_fraction_different=audit.mean())], 'channel_swap_verification')
    for arm in MIRRORS:
        sub=tab.xs(arm,level='arm')
        out.append(dict(arm=arm,measure='swapped minus ordinary Dice (pp)',
            **bootstrap((sub.dice-sub.ordinary_dice)*100)))
        out.append(dict(arm=arm,measure='native labels changed (%)',
            **bootstrap(sub.fraction_labels_changed*100)))
    wide=tab.unstack('arm')
    terms={}
    for field in ['ordinary_dice','dice']:
        q=wide[field]
        direction=q.near_far-q.far_near
        direction_aug=q.near_far_aug-q.far_near_aug
        terms[field]=100*(direction_aug-direction)
        out.append(dict(arm='all',measure=field+' mirror interaction (pp)',**bootstrap(terms[field])))
    out.append(dict(arm='all',measure='swap-induced interaction change (pp)',
                    **bootstrap(terms['dice']-terms['ordinary_dice'])))
    save(out,'channel_swap')
    regions=[]
    for (organ,arm),sub in data.groupby(['organ','arm']):
        q=sub.set_index(['case','seed'])
        for measurement in ['dice_delta_pp',*[f'{k}_delta_ml_{b}' for k in ['fp','fn'] for b in BANDS]]:
            vals=(q.dice-q.ordinary_dice)*100 if measurement=='dice_delta_pp' else q[measurement]
            regions.append(dict(organ=organ,arm=arm,measure=measurement,**bootstrap(vals)))
    save(regions,'channel_swap_organs')


def cost():
    result=[]
    timings=pd.read_csv(OUT/'cost/gpu_measurements.csv')
    assert len(timings)==360 and not timings.duplicated(['case','round','kind']).any()
    deployed=pd.read_csv(ROOT/'results/slice_selection/proposal_latency/measurements.csv')
    for kind in ['2d','2p5d','3d']:
        elapsed,validation=[],[]
        for seed in SEEDS:
            folder=ROOT/'results/runs'/f'{kind}_seed{seed}'
            record=read(folder/'complete.json')
            assert record['training_steps']==20000
            elapsed.append(record['total_seconds'])
            val=pd.read_json(folder/'validation.jsonl',lines=True).set_index('step')
            log=pd.read_json(folder/'train.jsonl',lines=True).set_index('step')
            duration=val.elapsed_seconds-log.loc[val.index,'elapsed_seconds']
            assert len(duration)==10 and (duration>0).all()
            validation.append(duration.sum())
        scores=timings[timings.kind==kind].groupby('case')[['seconds','voxels_per_s']].median()
        e2e=deployed[deployed.kind==kind].groupby('case')[['end_to_end_s','preprocess_s']].median()
        assert len(scores)==40 and set(scores.index)==set(e2e.index)
        result.append(dict(kind=kind,batch=int(timings[timings.kind==kind].batch.iloc[0]),
            fit_minutes_median=np.median(elapsed)/60,seconds_per_1000_including_val=np.median(elapsed)/20,
            validation_minutes_approx_median=np.median(validation)/60,
            three_run_gpu_hours=sum(elapsed)/3600,gpu_median_ms=1000*scores.seconds.median(),
            gpu_q25_ms=1000*scores.seconds.quantile(.25),gpu_q75_ms=1000*scores.seconds.quantile(.75),
            median_Mvox_s=scores.voxels_per_s.median()/1e6,e2e_median_s=e2e.end_to_end_s.median(),
            median_preprocess_share=(e2e.preprocess_s/e2e.end_to_end_s).median()))
    save(result,'computational_cost')
    grid=timings.groupby(['case','kind']).seconds.median().unstack('kind')
    ratio=(grid['2p5d']/grid['3d']).to_numpy()
    draws=np.random.default_rng(20261119).integers(0,len(ratio),size=(10000,len(ratio)))
    lo,hi=np.quantile(np.median(ratio[draws],axis=1),[.025,.975])
    save([dict(comparison='2p5d / 3d', median=np.median(ratio),ci_low=lo,ci_high=hi,
               patients=len(ratio),rounds_per_patient=3)],'gpu_paired_ratio')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study',choices=['factorial','fixed-step','capacity','swap','cost','all'])
    args=parser.parse_args()
    SUMMARY.mkdir(parents=True,exist_ok=True)
    for key,func in [('factorial',factorial),('fixed-step',fixed_step),('capacity',capacity),('swap',swap),('cost',cost)]:
        if args.study in ('all',key): func()
