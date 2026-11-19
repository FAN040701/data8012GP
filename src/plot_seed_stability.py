"""Standalone scientific figures for seed and selection sensitivity, without a report."""
from datetime import datetime, timezone
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from seed_stability_common import ROOT, OUT, CFG, sha

FIG=OUT/'figures'
SEEDS=CFG['old_seeds']+CFG['new_seeds']
LABELS={'repeat':'Repeated centre','context':'Neighbour slices','repeat_aug':'Centre + augmentation','context_aug':'Neighbours + augmentation'}
COLORS=['#606c80','#2878ad','#bc7b2d','#26927a']


def save(fig,name):
    fig.savefig(FIG/f'{name}.png',dpi=180,bbox_inches='tight')
    fig.savefig(FIG/f'{name}.svg',bbox_inches='tight')
    plt.close(fig)


def style(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.18)
    ax.axhline(0,color='#555',lw=.7,zorder=0)


def main():
    assert json.loads((OUT/'analysis/analysis.json').read_text())['all_passed']
    FIG.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.titlesize':12,'axes.labelsize':11})
    s=pd.read_csv(OUT/'analysis/seed_contrasts.csv')
    p=pd.read_csv(OUT/'analysis/patient_metrics.csv')
    effects=pd.read_csv(OUT/'analysis/paired_contrasts.csv')
    selected=s[(s.selection=='best')&(s.metric=='dice')&(s.condition=='f3')]
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for row,cohort in enumerate(['new40_replication','prior_fresh20']):
        sub=selected[selected.cohort==cohort]
        for col,names in enumerate([['context_without_aug','context_with_aug'],['context_x_augmentation']]):
            ax=axes[row,col]
            for j,name in enumerate(names):
                v=sub[sub.contrast==name].set_index('seed').loc[SEEDS,'difference']*100
                label={'context_without_aug':'Context gain without augmentation','context_with_aug':'Context gain with augmentation','context_x_augmentation':'Context × augmentation interaction'}[name]
                color=['#2878ad','#bc7b2d'][j]
                ax.plot(range(6),v,'o',label=label,color=color,linestyle='none')
                ax.hlines([v.iloc[:3].mean(),v.iloc[3:].mean()],[-.2,2.8],[2.2,5.2],color=color,lw=1.3,alpha=.65)
            ax.axvline(2.5,color='#aaa',ls='--',lw=.8)
            ax.set_xticks(range(6),SEEDS)
            ax.set_xlabel('Training seed: original 3 | additional 3')
            ax.set_ylabel('Patient-mean effect (Dice pp)')
            ax.set_title(('Replication cohort (40 patients)' if row==0 else 'Earlier held-out cohort (20 patients)')+' · f3')
            ax.legend(fontsize=8,loc='best')
            style(ax)
    fig.suptitle('Training-seed stability with the fixed development-selected checkpoint rule\nDots: individual seeds; horizontal segments: original/additional group means',fontsize=13)
    save(fig,'seed_reproducibility')

    fig,axes=plt.subplots(1,3,figsize=(14,4.5),constrained_layout=True)
    absolute=s[(s.cohort=='absolute_spacing')&(s.selection=='best')&(s.metric=='dice')]
    for ax,name,title in zip(axes[:2],['context_with_aug','context_x_augmentation'],['Context gain with augmentation','Context × augmentation interaction']):
        for j,seed in enumerate(SEEDS):
            row=absolute[(absolute.seed==seed)&(absolute.contrast==name)].set_index('condition')
            ax.plot([6,15],[row.loc[f'target{x}mm','difference']*100 for x in [6,15]],'o-',color=plt.cm.tab10(j),label=str(seed),alpha=.8)
        ax.set_xticks([6,15],['6 mm','15 mm'])
        ax.set_xlabel('Simulated physical sampling interval')
        ax.set_ylabel('Effect (Dice pp)')
        ax.set_title(title)
        style(ax)
    sub=absolute[(absolute.contrast=='context_x_augmentation')&(absolute.condition=='target15_minus_target6')].set_index('seed')
    axes[2].bar(range(6),sub.loc[SEEDS,'difference']*100,color=[plt.cm.tab10(j) for j in range(6)],width=.7)
    axes[2].set_xticks(range(6),SEEDS)
    axes[2].set_xlabel('Training seed')
    axes[2].set_ylabel('15 mm minus 6 mm (Dice pp)')
    axes[2].set_title('Change in interaction within patients')
    style(axes[2])
    axes[0].legend(title='Seed',ncol=2,fontsize=8)
    fig.suptitle('Same 21 thin-slice patients at both severities · synthetic acquisition diagnostic',fontsize=14)
    save(fig,'resolution_seed_sensitivity')

    fig,axes=plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
    sub=s[(s.cohort=='new40_replication')&(s.condition=='f3')&(s.contrast=='context_x_augmentation')]
    for j,seed in enumerate(SEEDS):
        v=sub[(sub.seed==seed)&sub.selection.isin(['best','last'])].set_index('selection').loc[['best','last'],'difference']*100
        axes[0].plot([0,1],v,'o-',color=plt.cm.tab10(j),label=str(seed))
    axes[0].set_xticks([0,1],['Development-selected best','Fixed step 40,000'])
    axes[0].set_ylabel('Interaction (Dice pp)')
    axes[0].set_title('Interaction sensitivity to checkpoint rule')
    axes[0].legend(title='Seed',ncol=2,fontsize=8)
    for j,arm in enumerate(CFG['arms']):
        v=s[(s.selection=='last_minus_best')&(s.contrast==arm+'_absolute_score')].set_index('seed').loc[SEEDS,'difference']*100
        axes[1].scatter(np.full(6,j)+np.linspace(-.16,.16,6),v,color=COLORS[j],s=35)
        axes[1].plot([j-.2,j+.2],[v.mean(),v.mean()],color='black',lw=2)
    axes[1].set_xticks(range(4),['Centre','Neighbours','Centre\n+ aug.','Neighbours\n+ aug.'])
    axes[1].set_ylabel('Final minus best score (Dice pp)')
    axes[1].set_title('All six seeds; black line is their mean')
    for ax in axes:
        style(ax)
    fig.suptitle('40 replication patients · f3 · sensitivity analysis, not test-based checkpoint selection',fontsize=13)
    save(fig,'checkpoint_seed_sensitivity')

    rep=p[(p.cohort=='new40_replication')&(p.selection=='best')&(p.factor==3)]
    wide=rep.pivot(index=['case','seed'],columns='arm',values='dice')
    interaction=(wide['context_aug']-wide['repeat_aug']-wide['context']+wide['repeat']).unstack('seed')[SEEDS]*100
    interaction=interaction.loc[interaction.mean(1).sort_values().index]
    fig,axes=plt.subplots(1,2,figsize=(9,12),gridspec_kw={'width_ratios':[3,1]},constrained_layout=True,sharey=True)
    limit=max(abs(interaction.to_numpy().min()),abs(interaction.to_numpy().max()))
    image=axes[0].imshow(interaction,aspect='auto',cmap='RdBu_r',vmin=-limit,vmax=limit)
    axes[0].set_xticks(range(6),SEEDS)
    axes[0].set_yticks(range(40),interaction.index,fontsize=8)
    axes[0].axvline(2.5,color='black',lw=1)
    axes[0].set_xlabel('Training seed: original 3 | additional 3')
    axes[0].set_title('Patient-level interaction (Dice pp)')
    means=interaction.mean(1)
    axes[1].barh(range(40),means,color=np.where(means>=0,'#b9524b','#4b83b9'))
    axes[1].axvline(0,color='#555',lw=.7)
    axes[1].set_xlabel('Dice pp')
    axes[1].set_title('Six-seed mean')
    fig.colorbar(image,ax=axes[0],shrink=.5,label='Dice pp')
    fig.suptitle('All 40 replication patients, ordered by six-seed mean; no case exclusions',fontsize=13)
    save(fig,'patient_seed_interactions')

    semantic=pd.read_csv(OUT/'analysis/patient_interaction_semantics.csv')
    semantic=semantic[semantic.seed_group=='all6'].copy()
    semantic['overall_gain_pp']=(semantic['context_aug']-semantic['repeat'])*100
    fig,ax=plt.subplots(figsize=(8.5,6),constrained_layout=True)
    for best,color,label in [(False,'#7a8796','Combined model is not best of four'),(True,'#26927a','Combined model is best of four')]:
        sub=semantic[semantic.combined_best_of_four==best]
        ax.scatter(sub.interaction*100,sub.overall_gain_pp,color=color,label=label,s=45,alpha=.85)
    ax.axhline(0,color='#444',lw=1)
    ax.axvline(0,color='#444',lw=1)
    labels=semantic[(semantic.overall_gain_pp<0)|semantic.case.isin(['amos_0052','amos_0224'])]
    for row in labels.itertuples():
        ax.annotate(row.case,(row.interaction*100,row.overall_gain_pp),xytext=(5,6),textcoords='offset points',fontsize=9)
    ax.set_xlabel('Context × augmentation interaction (Dice pp)')
    ax.set_ylabel('Combined model minus repeated-centre baseline (Dice pp)')
    ax.spines[['top','right']].set_visible(False)
    ax.legend(fontsize=9)
    ax.set_title('Interaction sign and absolute performance are different quantities\nAll 40 patients, six-seed means; point comparisons, not individual significance tests',fontsize=12)
    save(fig,'interaction_vs_absolute_performance')

    component=OUT/'component_diagnostic/complete.json'
    if component.exists():
        raw=pd.read_csv(OUT/'component_diagnostic/all_organ_metrics.csv')
        patient=raw.groupby(['case','arm'],as_index=False)[['dice_before','dice_after','removed_true_positive_ml','removed_false_positive_ml']].mean()
        fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
        for j,arm in enumerate(CFG['arms']):
            sub=patient[patient.arm==arm].sort_values('case')
            delta=(sub.dice_after-sub.dice_before)*100
            axes[0].scatter(np.full(len(sub),j)+np.linspace(-.22,.22,len(sub)),delta,color=COLORS[j],s=15,alpha=.7)
            axes[0].plot([j-.25,j+.25],[delta.mean(),delta.mean()],color='black',lw=2)
            axes[1].scatter(sub.removed_false_positive_ml,sub.removed_true_positive_ml,color=COLORS[j],label=LABELS[arm],s=22,alpha=.7)
        axes[0].set_xticks(range(4),['Centre','Neighbours','Centre\n+ aug.','Neighbours\n+ aug.'])
        axes[0].set_ylabel('Postprocessing change (Dice pp)')
        axes[0].set_title('Each point: one patient, averaged across six seeds')
        style(axes[0])
        axes[1].set_xscale('symlog',linthresh=.1)
        axes[1].set_yscale('symlog',linthresh=.1)
        axes[1].set_xlabel('Removed false-positive volume (mL / organ)')
        axes[1].set_ylabel('Removed true-positive volume (mL / organ)')
        axes[1].set_title('Benefit and cost of discarding smaller components')
        axes[1].legend(fontsize=8)
        fig.suptitle('Fixed label-blind largest-component diagnostic · original masks remain primary',fontsize=13)
        save(fig,'connected_component_diagnostic')
        # This patient was selected in the preceding session, before the new seeds.
        # Keep every seed visible: averaging can hide a wrong-largest-component failure.
        selected=raw[(raw.case=='amos_0052')&(raw.organ=='left_kidney')]
        fig,axes=plt.subplots(2,2,figsize=(11,7.5),constrained_layout=True,sharey=True)
        for ax,arm in zip(axes.ravel(),CFG['arms']):
            sub=selected[selected.arm==arm].set_index('seed').loc[SEEDS]
            positions=np.arange(6)
            ax.bar(positions-.18,sub.dice_before*100,width=.36,color='#367da3',label='Original prediction')
            ax.bar(positions+.18,sub.dice_after*100,width=.36,color='#c98a42',label='Largest component only')
            ax.set_xticks(positions,SEEDS)
            ax.set_ylim(0,100)
            ax.axvline(2.5,color='#aaa',ls='--',lw=.8)
            ax.set_xlabel('Training seed: original 3 | additional 3')
            ax.set_ylabel('Whole-organ Dice (%)')
            ax.set_title(LABELS[arm])
            ax.spines[['top','right']].set_visible(False)
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='outside lower center',ncol=2)
        fig.suptitle('amos_0052 · left kidney · the largest predicted component can be the wrong one\nAll six seeds; fixed label-blind postprocessing diagnostic, not the primary model result',fontsize=13)
        save(fig,'largest_component_failure_all_seeds')
    paths=sorted(FIG.glob('*'))
    (FIG/'manifest.json').write_text(json.dumps({'completed_utc':datetime.now(timezone.utc).isoformat(),'files':{str(x.relative_to(ROOT)):sha(x) for x in paths if x.suffix in ['.png','.svg']},'source_sha256':sha(__file__),'report_generated':False},indent=2))
    print('Saved scientific figures',len(paths),flush=True)


if __name__=='__main__':
    main()
