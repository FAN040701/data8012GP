"""Prespecified seed extension summaries, with old and added seeds kept visible."""
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd
from scipy.stats import t

from seed_stability_common import ROOT, OUT, CFG, PROTOCOL, sha

AN = OUT / 'analysis'
WEIGHTS = {
    'context_without_aug': {'context':1, 'repeat':-1},
    'context_with_aug': {'context_aug':1, 'repeat_aug':-1},
    'augmentation_repeat': {'repeat_aug':1, 'repeat':-1},
    'augmentation_context': {'context_aug':1, 'context':-1},
    'context_x_augmentation': {'context_aug':1, 'repeat_aug':-1, 'context':-1, 'repeat':1},
}
GROUPS = {'original3': CFG['old_seeds'], 'additional3': CFG['new_seeds'], 'all6': CFG['old_seeds']+CFG['new_seeds']}


def summarize(matrix):
    """Rows are patients, columns matched training seeds; no prediction ensembling."""
    a = matrix.sort_index().sort_index(axis=1).to_numpy(dtype=float)
    assert np.isfinite(a).all() and a.ndim == 2
    n, s = a.shape
    rng = np.random.default_rng(CFG['primary']['bootstrap_seed'])
    b = CFG['primary']['bootstrap_replicates']
    pi = rng.integers(n, size=(b, n))
    si = rng.integers(s, size=(b, s))
    patient = a.mean(1)[pi].mean(1)
    crossed = a[pi[:,:,None], si[:,None,:]].mean((1,2))
    lo, hi = np.quantile(patient, [.025,.975])
    clo, chi = np.quantile(crossed, [.025,.975])
    sm = a.mean(0)
    radius = t.ppf(.975,s-1)*sm.std(ddof=1)/np.sqrt(s)
    return {'mean':float(a.mean()), 'ci_low':float(lo), 'ci_high':float(hi),
            'crossed_ci_low':float(clo), 'crossed_ci_high':float(chi),
            'seed_t_ci_low':float(sm.mean()-radius), 'seed_t_ci_high':float(sm.mean()+radius),
            'seed_sd':float(sm.std(ddof=1)), 'seed_min':float(sm.min()), 'seed_max':float(sm.max()),
            'positive_seed_means':int((sm>0).sum()), 'positive_patient_seed_means':int((a.mean(1)>0).sum()),
            'patients':n, 'seeds':s}


def load():
    sources = {}
    def read(p):
        sources[str(p.relative_to(ROOT))] = sha(p)
        return pd.read_csv(p)
    old = read(ROOT / 'results/mechanism/analysis/all_organ_metrics.csv')
    old = old[old.arm.isin(CFG['arms']) & old.cohort.isin(['new40_replication','prior_fresh20']) & old.factor.isin([1,3])].copy()
    old['selection'] = 'best'
    absolute = read(ROOT / 'results/mechanism/absolute_spacing/all_organ_metrics.csv')
    absolute['cohort'], absolute['selection'] = 'absolute_spacing', 'best'
    frames = [old, absolute]
    for selection, seeds in [('best',CFG['new_seeds']), ('last',GROUPS['all6'])]:
        for seed in seeds:
            for arm in CFG['arms']:
                folder = OUT / 'evaluations' / selection / f'{arm}_seed{seed}'
                record = json.loads((folder/'complete.json').read_text())
                assert record['protocol_sha256'] == sha(PROTOCOL)
                frames.append(read(folder/'metrics.csv'))
    return pd.concat(frames, ignore_index=True), sources


def main():
    for name in ['main','last']:
        assert not json.loads((OUT / f'{name}_queue_complete.json').read_text())['errors']
    AN.mkdir(exist_ok=True)
    raw, sources = load()
    key = ['cohort','selection','arm','seed','case','factor','organ']
    assert not raw.duplicated(key).any()
    assert len(raw) == (24*162+24*40)*4
    raw.to_csv(AN/'all_organ_metrics.csv',index=False)
    patient = raw.groupby(key[:-1],as_index=False).agg(dice=('dice','mean'), interior_dice=('interior_dice','mean'), scored_organs=('dice','count'), hd95_mm=('hd95_mm','mean'))
    assert patient.groupby(['cohort','selection','case','factor']).scored_organs.nunique().max() == 1
    patient.to_csv(AN/'patient_metrics.csv',index=False)
    results, seeds_out, patients_out, model = [], [], [], []

    def add(series, meta):
        matrix = series.unstack('seed').sort_index().sort_index(axis=1)
        assert matrix.shape[1] == 6 and not matrix.isna().any().any()
        for group, seeds in GROUPS.items():
            sub = matrix[seeds]
            results.append({**meta,'seed_group':group,**summarize(sub)})
            patients_out.extend({**meta,'seed_group':group,'case':case,'difference':float(x)} for case,x in sub.mean(1).items())
        seeds_out.extend({**meta,'seed':int(seed),'difference':float(x)} for seed,x in matrix.mean().items())

    regular = patient[(patient.selection=='best') & (patient.cohort!='absolute_spacing')]
    for cohort, sub in regular.groupby('cohort'):
        wide = sub.pivot(index=['case','seed','factor'],columns='arm',values='dice')
        for name, weights in WEIGHTS.items():
            d = sum(wide[a]*w for a,w in weights.items())
            for factor in [1,3]:
                add(d.xs(factor,level='factor'),{'cohort':cohort,'selection':'best','metric':'dice','contrast':name,'condition':f'f{factor}'})
            add(d.xs(3,level='factor')-d.xs(1,level='factor'),{'cohort':cohort,'selection':'best','metric':'dice','contrast':name,'condition':'f3_minus_f1'})
    for (cohort,selection,arm,factor),sub in patient[patient.cohort!='absolute_spacing'].groupby(['cohort','selection','arm','factor']):
        for group, seeds in GROUPS.items():
            matrix=sub[sub.seed.isin(seeds)].pivot(index='case',columns='seed',values='dice')
            model.append({'cohort':cohort,'selection':selection,'arm':arm,'condition':f'f{int(factor)}','seed_group':group,**summarize(matrix)})
    # Use physical targets from source rows, not a relative degradation factor label.
    spacing = raw[raw.cohort=='absolute_spacing'].groupby(['case','seed','arm','target_sampling_mm'],as_index=False).agg(dice=('dice','mean'),interior_dice=('interior_dice','mean'))
    for (arm,target),sub in spacing.groupby(['arm','target_sampling_mm']):
        for group,seeds in GROUPS.items():
            matrix=sub[sub.seed.isin(seeds)].pivot(index='case',columns='seed',values='dice')
            model.append({'cohort':'absolute_spacing','selection':'best','arm':arm,'condition':f'target{int(target)}mm','seed_group':group,**summarize(matrix)})
    for metric in ['dice','interior_dice']:
        wide=spacing.pivot(index=['case','seed','target_sampling_mm'],columns='arm',values=metric)
        for name,weights in WEIGHTS.items():
            d=sum(wide[a]*w for a,w in weights.items())
            for target in [6.,15.]:
                add(d.xs(target,level='target_sampling_mm'),{'cohort':'absolute_spacing','selection':'best','metric':metric,'contrast':name,'condition':f'target{int(target)}mm'})
            add(d.xs(15.,level='target_sampling_mm')-d.xs(6.,level='target_sampling_mm'),{'cohort':'absolute_spacing','selection':'best','metric':metric,'contrast':name,'condition':'target15_minus_target6'})
    # Fixed last-step sensitivity is not a replacement for the original selection rule.
    rep=patient[(patient.cohort=='new40_replication') & (patient.factor==3)]
    wide=rep.pivot(index=['case','seed','selection'],columns='arm',values='dice')
    for name,weights in WEIGHTS.items():
        d=sum(wide[a]*w for a,w in weights.items())
        add(d.xs('last',level='selection'),{'cohort':'new40_replication','selection':'last','metric':'dice','contrast':name,'condition':'f3'})
        add(d.xs('last',level='selection')-d.xs('best',level='selection'),{'cohort':'new40_replication','selection':'last_minus_best','metric':'dice','contrast':name,'condition':'f3'})
    for arm in CFG['arms']:
        d=wide[arm].xs('last',level='selection')-wide[arm].xs('best',level='selection')
        add(d,{'cohort':'new40_replication','selection':'last_minus_best','metric':'dice','contrast':arm+'_absolute_score','condition':'f3'})
    effects=pd.DataFrame(results)
    effects.to_csv(AN/'paired_contrasts.csv',index=False)
    pd.DataFrame(model).to_csv(AN/'model_summary.csv',index=False)
    pd.DataFrame(seeds_out).to_csv(AN/'seed_contrasts.csv',index=False)
    pd.DataFrame(patients_out).to_csv(AN/'patient_contrasts.csv',index=False)
    # A negative interaction need not imply lower absolute accuracy.
    base_wide=rep[rep.selection=='best'].pivot(index=['case','seed'],columns='arm',values='dice')
    individual=base_wide['context_aug']-base_wide['repeat_aug']-base_wide['context']+base_wide['repeat']
    semantics=[]
    case_rows=[]
    for group,seeds in GROUPS.items():
        scores=base_wide[base_wide.index.get_level_values('seed').isin(seeds)].groupby('case').mean()
        scores['interaction']=scores['context_aug']-scores['repeat_aug']-scores['context']+scores['repeat']
        scores['combined_above_repeat']=scores['context_aug']>scores['repeat']
        scores['combined_below_context']=scores['context_aug']<scores['context']
        scores['combined_below_repeat_aug']=scores['context_aug']<scores['repeat_aug']
        scores['combined_best_of_four']=scores['context_aug']>=scores[CFG['arms']].max(axis=1)
        individual_matrix=individual.unstack('seed')[seeds]
        scores['positive_seeds']=(individual_matrix>0).sum(axis=1)
        scores['negative_seeds']=(individual_matrix<0).sum(axis=1)
        scores['seed_group']=group
        case_rows.extend(scores.reset_index().to_dict('records'))
        negative=scores[scores.interaction<0]
        positive=scores[scores.interaction>0]
        semantics.append({'seed_group':group,'negative_interaction_patients':len(negative),
                          'negative_interaction_combined_above_repeat':int(negative.combined_above_repeat.sum()),
                          'negative_interaction_combined_best_of_four':int(negative.combined_best_of_four.sum()),
                          'negative_interaction_combined_below_context':int(negative.combined_below_context.sum()),
                          'negative_interaction_combined_below_repeat_aug':int(negative.combined_below_repeat_aug.sum()),
                          'positive_interaction_patients':len(positive),
                          'positive_interaction_combined_below_repeat':int((positive['context_aug']<positive['repeat']).sum()),
                          'positive_interaction_combined_below_context':int(positive.combined_below_context.sum()),
                          'positive_interaction_combined_below_repeat_aug':int(positive.combined_below_repeat_aug.sum()),
                          'positive_in_every_seed':int((scores.positive_seeds==len(seeds)).sum()),
                          'negative_in_every_seed':int((scores.negative_seeds==len(seeds)).sum())})
    pd.DataFrame(case_rows).to_csv(AN/'patient_interaction_semantics.csv',index=False)
    (AN/'interaction_sign_semantics.json').write_text(json.dumps({'all_passed':True,'counts':semantics,'interpretation':'Patient point comparisons of seed-averaged scores, not patient-level significance tests. Flags overlap.'},indent=2))
    influence=[]
    matrix=individual.unstack('seed')[GROUPS['all6']]
    for excluded in GROUPS['all6']:
        influence.append({'excluded_seed':excluded,**summarize(matrix.drop(columns=excluded))})
    pd.DataFrame(influence).to_csv(AN/'leave_one_seed_out.csv',index=False)
    organ_effects,organ_seeds=[],[]
    organ_raw=raw[(raw.selection=='best')&(raw.cohort!='absolute_spacing')]
    for (cohort,organ,factor),sub in organ_raw.groupby(['cohort','organ','factor']):
        w=sub.pivot(index=['case','seed'],columns='arm',values='dice').dropna()
        for name in ['context_without_aug','context_with_aug','context_x_augmentation']:
            d=sum(w[a]*weight for a,weight in WEIGHTS[name].items()).unstack('seed')
            for group,seeds in GROUPS.items():
                organ_effects.append({'cohort':cohort,'organ':organ,'factor':factor,'contrast':name,'seed_group':group,**summarize(d[seeds])})
            organ_seeds.extend({'cohort':cohort,'organ':organ,'factor':factor,'contrast':name,'seed':int(seed),'difference':float(value)} for seed,value in d.mean().items())
    pd.DataFrame(organ_effects).to_csv(AN/'organ_contrasts.csv',index=False)
    pd.DataFrame(organ_seeds).to_csv(AN/'organ_seed_contrasts.csv',index=False)
    primary=effects[(effects.cohort=='new40_replication') & (effects.selection=='best') & (effects.contrast=='context_x_augmentation') & (effects.condition=='f3')]
    # Keep the historical value unchanged; bootstrap uses the new fixed seed and 20k draws.
    old_primary=json.loads((ROOT/'results/mechanism/analysis/analysis.json').read_text())['replication_primary']
    assert np.isclose(primary[primary.seed_group=='original3']['mean'].iloc[0],old_primary['mean'],atol=1e-14,rtol=0)
    record={'all_passed':True,'completed_utc':datetime.now(timezone.utc).isoformat(),'primary':primary.to_dict('records'),'new_training_runs':12,'new_training_updates':480000,'new_native_predictions':2904,'new_patients':0,'cohorts_not_pooled':True,'six_seed_result_is_sequential_exploration':True,'sources_sha256':sources,'protocol_sha256':sha(PROTOCOL),'source_sha256':sha(__file__),'interaction_sign_counts':semantics,'report_generated':False}
    (AN/'analysis.json').write_text(json.dumps(record,indent=2))
    print(json.dumps({k:v for k,v in record.items() if k!='sources_sha256'},indent=2),flush=True)


if __name__=='__main__':
    main()
