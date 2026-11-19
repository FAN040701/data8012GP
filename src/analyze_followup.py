"""Analyze the fixed sequential experiment only after all nine evaluations finish."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_results import interval

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/followup_context'
LABEL={'repeat':'Repeated centre','narrow':'Neighbours ±3 mm','wide':'Neighbours ±9 mm'}
COLORS={'repeat':'#8B5CB8','narrow':'#2376BB','wide':'#D87B28'}


def formatted(r):
    return f'{100*r["mean"]:+.2f} pp [95% CI {100*r["ci_low"]:+.2f}, {100*r["ci_high"]:+.2f}]'


def main():
    folder=OUT/'analysis'; folder.mkdir(exist_ok=True)
    original=json.loads((ROOT/'data/metadata/split.json').read_text())['splits']
    new=json.loads((ROOT/'data/metadata/followup_context/split.json').read_text())['test']
    assert len(new)==40 and not set(new)&set(sum(original.values(),[]))
    records=[]; versions=set()
    for kind in LABEL:
        for seed in [11,22,33]:
            run=OUT/f'{kind}_seed{seed}'
            complete=json.loads((run/'evaluation_complete.json').read_text())
            frame=pd.read_csv(run/'metrics.csv')
            assert len(frame)==480 and not frame.duplicated(['case','factor','organ']).any()
            assert set(frame.case)==set(new) and set(frame.seed)=={seed} and set(frame.kind)=={kind}
            for f in [1,2,3]: assert len(frame[frame.factor==f])==160
            assert all(hashlib.sha256((ROOT/'src'/n).read_bytes()).hexdigest()==s for n,s in complete['source_sha256'].items())
            versions.add(json.dumps(complete['source_sha256'],sort_keys=True))
            records.append(frame)
    assert len(versions)==1
    raw=pd.concat(records,ignore_index=True)
    raw.to_csv(folder/'all_organ_metrics.csv',index=False)
    patient=raw.groupby(['kind','seed','case','factor'],as_index=False)[['dice','hd95_mm']].mean()
    patient.to_csv(folder/'patient_metrics.csv',index=False)
    seedmeans=patient.groupby(['kind','seed','factor'],as_index=False)[['dice','hd95_mm']].mean()
    seedmeans.to_csv(folder/'seed_summary.csv',index=False)
    averaged=patient.groupby(['kind','case','factor'],as_index=False)[['dice','hd95_mm']].mean()
    summaries=[]
    for kind in LABEL:
        for f in [1,2,3]:
            sub=averaged[(averaged.kind==kind)&(averaged.factor==f)]
            mean,lo,hi=interval(sub.dice)
            summaries.append({'kind':kind,'factor':f,'dice':mean,'ci_low':lo,'ci_high':hi,'hd95_mm':float(sub.hd95_mm.mean())})
    summary=pd.DataFrame(summaries); summary.to_csv(folder/'model_summary.csv',index=False)
    contrasts=[]; paired=[]
    for a,b,name in [('wide','narrow','wide_minus_narrow'),('wide','repeat','wide_context_gain'),('narrow','repeat','narrow_context_gain')]:
        pair=patient[patient.kind==a].merge(patient[patient.kind==b],on=['seed','case','factor'],suffixes=('_a','_b'))
        pair['difference']=pair.dice_a-pair.dice_b
        per_patient=pair.groupby(['case','factor']).difference.mean().unstack()
        per_seed=pair.groupby(['seed','factor']).difference.mean().unstack()
        for condition,values,seedvalues in [(f'f{f}',per_patient[f],per_seed[f]) for f in [1,2,3]]+[(f'f{f}_minus_f1',per_patient[f]-per_patient[1],per_seed[f]-per_seed[1]) for f in [2,3]]:
            mean,lo,hi=interval(values)
            row={'contrast':name,'condition':condition,'mean':mean,'ci_low':lo,'ci_high':hi,'seed_min':float(seedvalues.min()),'seed_max':float(seedvalues.max()),'patients':len(values),'seeds':3}
            contrasts.append(row)
            for case,value in values.items(): paired.append({'contrast':name,'condition':condition,'case':case,'difference':value})
    table=pd.DataFrame(contrasts); table.to_csv(folder/'paired_contrasts.csv',index=False)
    pd.DataFrame(paired).to_csv(folder/'paired_patient_differences.csv',index=False)
    primary=next(r for r in contrasts if r['contrast']=='wide_minus_narrow' and r['condition']=='f1')
    interaction=next(r for r in contrasts if r['contrast']=='wide_minus_narrow' and r['condition']=='f3_minus_f1')
    if primary['ci_low']>0:
        en='The fresh-patient interval supports a native-condition benefit from the fixed wider context window in this implementation.'
        cn='新的患者留出集支持：在本实现中，把物理上下文从 ±3 mm 扩大到 ±9 mm 带来了原始条件下的收益。'
    elif primary['ci_high']<0:
        en='The wider context window reduced native-condition Dice on fresh patients; increasing physical span alone did not improve the primary endpoint.'
        cn='新的患者留出集上，扩大上下文窗口降低了原始条件的 Dice；单纯扩大物理范围未改善这个主要终点。'
    else:
        en='The fresh-patient interval does not resolve a clear native-condition effect of widening the context window. This is not an equivalence result.'
        cn='新的患者留出集尚未分辨出扩大物理上下文窗口的明确收益，不能解释为两种窗口等效。'
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.0))
    for kind in LABEL:
        sub=summary[summary.kind==kind]
        axes[0].plot(sub.factor,100*sub.dice,'o-',label=LABEL[kind],color=COLORS[kind])
        axes[0].fill_between(sub.factor,100*sub.ci_low,100*sub.ci_high,alpha=.08,color=COLORS[kind])
    sub=table[(table.contrast=='wide_minus_narrow')&table.condition.isin(['f1','f2','f3'])]
    m=100*sub['mean'].to_numpy()
    for seed in [11,22,33]:
        a=seedmeans[(seedmeans.kind=='wide')&(seedmeans.seed==seed)].sort_values('factor')
        b=seedmeans[(seedmeans.kind=='narrow')&(seedmeans.seed==seed)].sort_values('factor')
        axes[1].plot([1,2,3],100*(a.dice.to_numpy()-b.dice.to_numpy()),':',color='#98A2B3',alpha=.8)
    axes[1].errorbar([1,2,3],m,yerr=[m-100*sub.ci_low.to_numpy(),100*sub.ci_high.to_numpy()-m],fmt='o-',capsize=4,color=COLORS['wide'])
    axes[1].axhline(0,color='#667085',ls='--',lw=1)
    axes[0].set_ylabel('Patient macro Dice (%)'); axes[1].set_ylabel('Wide minus narrow Dice (pp)')
    axes[0].set_title('40 fresh held-out patients'); axes[1].set_title('Paired effect and individual seed means')
    for ax in axes:
        ax.set_xticks([1,2,3],['Native','Factor 2','Factor 3']); ax.grid(axis='y',alpha=.15)
    axes[0].legend(fontsize=9); fig.tight_layout()
    for ext in ['png','pdf']: fig.savefig(folder/f'context_width.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)

    checks=json.loads((OUT/'pipeline_checks.json').read_text())
    for seed in [11,22,33]:
        wide=json.loads((OUT/f'wide_seed{seed}'/'metadata.json').read_text())
        narrow=json.loads((ROOT/'results/runs'/f'2p5d_seed{seed}'/'metadata.json').read_text())
        assert wide['config']==narrow['config'] and wide['initial_parameter_sha256']==narrow['initial_parameter_sha256']
        assert wide['train_ids']==narrow['train_ids'] and wide['val_ids']==narrow['val_ids']
        a=pd.read_json(OUT/f'wide_seed{seed}'/'train.jsonl',lines=True)
        b=pd.read_json(ROOT/'results/runs'/f'2p5d_seed{seed}'/'train.jsonl',lines=True)
        assert a[['step','patient']].equals(b[['step','patient']])
    failed=int((raw.status=='prediction_empty').sum())
    absent=raw[raw.true_voxels==0][['case','organ']].drop_duplicates().to_dict('records')
    analysis={'completed_evaluations':9,'new_training_runs':3,'fresh_test_patients':40,'rows':len(raw),
              'primary':primary,'interaction':interaction,'summaries':summaries,'contrasts':contrasts,
              'conclusion':en,'interpretation':'Sequential hypothesis, new patient holdout; shared fitting/development cohort and same data source.',
              'checks':checks,'matched_initialization_config_and_sampling':True,'disjoint_new_patients':True,'empty_predictions':failed,'reference_absent_case_organs':absent}
    (folder/'analysis.json').write_text(json.dumps(analysis,indent=2))
    print(summary.to_string(index=False))
    print(table.to_string(index=False))


if __name__=='__main__': main()
