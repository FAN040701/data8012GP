"""Paired analysis of the completed fixed-offset experiment and selected-policy repeat."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json

import numpy as np
import pandas as pd

from mechanism_common import sha
from slice_selection import ROOT, OUT, CFG, run_dir, PROTOCOL
from verify_mechanism import check_header, check_mask


def logged_sampling(run):
    fields = ['step', 'patient', 'start', 'factor', 'lr']
    records = [json.loads(line) for line in (run / 'train.jsonl').read_text().splitlines()]
    return [[record[field] for field in fields] for record in records]


def checked_scores(arm,seed,test_ids):
    run=run_dir(arm,seed)
    training=json.loads((run/'complete.json').read_text())
    evaluation=json.loads((run/'evaluation_complete.json').read_text())
    metadata=json.loads((run/'metadata.json').read_text())
    control_arm='context_aug' if arm.endswith('_aug') else 'context'
    control_base='results/seed_stability/runs' if seed>33 else 'results/continuation/b40000'
    control=ROOT/control_base/f'{control_arm}_seed{seed}'
    prior=json.loads((control/'complete.json').read_text())
    prior_meta=json.loads((control/'metadata.json').read_text())
    for k in ['sampling_sha256','factor_counts','training_steps']:
        assert prior[k]==training[k],(arm,seed,k)
    assert metadata['initial_parameter_sha256']==prior_meta['initial_parameter_sha256']
    assert metadata['train_ids']==prior_meta['train_ids'] and metadata['val_ids']==prior_meta['val_ids']
    trace = logged_sampling(run)
    assert len(trace) == 801 and trace == logged_sampling(control), (arm, seed, 'training trace')
    assert all(sha(ROOT/'src'/n)==digest for n,digest in metadata['source_sha256'].items())
    assert sha(run/'best.pt')==training['best_checkpoint_sha256']==evaluation['checkpoint_sha256']
    assert metadata['protocol_sha256']==sha(PROTOCOL)==evaluation['protocol_sha256']
    assert sha(run/'metrics.csv')==evaluation['metrics_sha256']
    validations=pd.read_json(run/'validation.jsonl',lines=True)
    assert len(validations)==20
    assert all({v['case'] for v in cases}==set(metadata['val_ids']) and len(cases)==20
               for cases in validations.cases)
    assert validations.loc[validations.macro_dice.idxmax(),'step']==training['best_step']
    df=pd.read_csv(run/'metrics.csv').assign(arm=arm)
    assert set(df.case)==set(test_ids) and set(df.seed)=={seed} and len(df)==320
    def predpath(c,f):
        return str((run/'predictions'/f'f{f}'/f'{c}.nii.gz').relative_to(ROOT))
    files=[(predpath(c,f),c) for c in test_ids for f in [1,3]]
    with ThreadPoolExecutor(6) as pool:
        assert all(pool.map(check_header,files))
    for f in CFG['factors']:
        c=sorted(test_ids)[(seed+f)%len(test_ids)]
        check_mask((predpath(c,f),c,df[(df.case==c)&(df.factor==f)].to_dict('records')))
    return df


def bootstrap(series):
    mat=series.unstack('seed').sort_index().sort_index(axis=1)
    assert np.isfinite(mat).all().all()
    v=mat.to_numpy()
    rng=np.random.default_rng(20261119)
    pat=rng.integers(len(v),size=(10000,len(v)))
    sd=rng.integers(v.shape[1],size=(10000,v.shape[1]))
    lo,hi=np.quantile(v[pat[:,:,None],sd[:,None,:]].mean((1,2)),[.025,.975])
    return dict(mean=v.mean(),ci_low=lo,ci_high=hi,patients=len(v),seeds=v.shape[1],
                seed_means=json.dumps({str(s):mat[s].mean() for s in mat.columns}))


def main(repeat):
    ids=json.loads((ROOT/CFG['test_split']).read_text())['test']
    known=pd.read_csv(ROOT/'results/seed_stability/analysis/all_organ_metrics.csv')
    known=known[(known.selection=='best') & (known.cohort=='new40_replication') &
                known.arm.isin(['context','context_aug'])].copy()
    name='repeat_analysis' if repeat else 'analysis'
    dest=OUT/name
    dest.mkdir(exist_ok=True)
    policies=[json.loads((OUT/'selection.json').read_text())['selected_policy']] if repeat else list(CFG['offsets'])
    seeds=[11,22,33,44,55,66] if repeat else CFG['seeds']
    frames=[known[known.seed.isin(seeds)]]
    for policy in policies:
        for suffix in ('','_aug'):
            for seed in seeds:
                frames.append(checked_scores(policy+suffix,seed,ids))
    columns=['case','seed','arm','organ','factor','dice','pred_voxels','true_voxels']
    organs=pd.concat([f[columns] for f in frames],ignore_index=True)
    assert not organs.duplicated(columns[:5]).any()
    assert set(organs.case)==set(ids) and set(organs.factor)==set(CFG['factors'])
    assert organs.groupby(['case','seed','arm','factor']).size().eq(4).all()
    assert len(organs)==len(ids)*len(seeds)*(2+2*len(policies))*len(CFG['factors'])*4
    organs.to_csv(dest/'organ_metrics.csv',index=False)
    pat=organs.groupby(['case','seed','arm','factor']).dice.mean().unstack('arm')
    pat.to_csv(dest/'patient_scores.csv')
    effects=[]; organ_effects=[]
    def add(label,delta,group):
        for condition in [1,3,'f3_minus_f1']:
            d=delta.xs(condition,level='factor') if isinstance(condition,int) else delta.xs(3,level='factor')-delta.xs(1,level='factor')
            effects.append(dict(contrast=label,condition=condition,seed_group=group,**bootstrap(d)))
    groups={'all6':seeds,'additional3':[44,55,66],'original3':[11,22,33]} if repeat else {'original3':seeds}
    for group,ss in groups.items():
        table=pat[pat.index.get_level_values('seed').isin(ss)]
        for policy in policies:
            for suffix in ('','_aug'):
                add(policy+suffix+'_minus_context'+suffix,table[policy+suffix]-table['context'+suffix],group)
            add(policy+'_selection_x_augmentation',(table[policy+'_aug']-table.context_aug)-(table[policy]-table.context),group)
        if not repeat:
            for suffix in ('','_aug'):
                add('mixed_average_minus_mid'+suffix,(table['near_far'+suffix]+table['far_near'+suffix])/2-table['mid'+suffix],group)
                add('near_far_minus_far_near'+suffix,table['near_far'+suffix]-table['far_near'+suffix],group)
            direction_plain = table.near_far - table.far_near
            direction_aug = table.near_far_aug - table.far_near_aug
            add('mirror_direction_x_augmentation', direction_aug-direction_plain, group)
    pd.DataFrame(effects).to_csv(dest/'paired_effects.csv',index=False)
    means=pat.groupby('factor').mean().T
    means.to_csv(dest/'absolute_dice.csv')
    # Decompose organ-specific changes into missed and extra reference-grid volume.
    voxel_ml={c:np.prod(json.loads((ROOT/'data/processed'/c/'metadata.json').read_text())['original_spacing_xyz_mm'])/1000 for c in ids}
    tp=(organs.dice*(organs.pred_voxels+organs.true_voxels)/2).fillna(0).round()
    organs['fp_ml']=(organs.pred_voxels-tp)*organs.case.map(voxel_ml)
    organs['fn_ml']=(organs.true_voxels-tp)*organs.case.map(voxel_ml)
    for (organ,factor),sub in organs.groupby(['organ','factor']):
        for metric in ['dice','fp_ml','fn_ml']:
            table=sub.pivot(index=['case','seed'],columns='arm',values=metric).dropna()
            for policy in policies:
                for suffix in ('','_aug'):
                    d=table[policy+suffix]-table['context'+suffix]
                    organ_effects.append(dict(organ=organ,factor=factor,metric=metric,
                        contrast=policy+suffix+'_minus_context'+suffix,**bootstrap(d)))
            if not repeat:
                for suffix in ('','_aug'):
                    d=(table['near_far'+suffix]+table['far_near'+suffix])/2-table['mid'+suffix]
                    organ_effects.append(dict(organ=organ,factor=factor,metric=metric,contrast='mixed_average_minus_mid'+suffix,**bootstrap(d)))
                    d=table['near_far'+suffix]-table['far_near'+suffix]
                    organ_effects.append(dict(organ=organ,factor=factor,metric=metric,contrast='near_far_minus_far_near'+suffix,**bootstrap(d)))
    pd.DataFrame(organ_effects).to_csv(dest/'organ_effects.csv',index=False)
    (dest/'verified.json').write_text(json.dumps(dict(cases=len(ids),new_fits_checked=len(frames)-1,
        predictions_checked=(len(frames)-1)*80,independent_mask_recounts=(len(frames)-1)*2,
        paired_sampling_and_initializations_verified=True,bootstrap_replicates=10000),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--repeat',action='store_true')
    main(parser.parse_args().repeat)
