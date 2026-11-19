"""Standalone annotation-overlap figures with explicit remainder and fixed case selection."""
from datetime import datetime, timezone
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from seed_stability_common import ROOT,OUT,CFG,sha

DEST=OUT/'anatomical_diagnostic'
FIG=DEST/'figures'
MAP={1:1,2:2,3:3,4:6}
NAMES={1:'spleen',2:'right_kidney',3:'left_kidney',4:'liver'}


def main():
    assert json.loads((DEST/'complete.json').read_text())['all_passed']
    FIG.mkdir(exist_ok=True)
    counts=pd.read_csv(DEST/'patient_confusion_means.csv')
    label_names=json.loads((ROOT/'data/metadata/dataset.json').read_text())['labels']
    selected=json.loads((ROOT/'results/mechanism/replication_spatial_errors/case_selection.json').read_text())['selection']
    arms=CFG['arms']
    arm_labels=['Centre','Neighbours','Centre\n+ aug.','Neighbours\n+ aug.']
    # All labels remain in the tables; group only for a readable fixed-display figure.
    category_names=['Unspecified remainder (label 0)','Bladder (label 14)','Other annotated non-target organs','Other target organs']
    colors=['#a5a8af','#d59d38','#6b9eae','#b66c9d']
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for ax,choice in zip(axes.ravel(),selected):
        case,organ=choice['case'],choice['organ']
        k=next(k for k,v in NAMES.items() if v==organ)
        sub=counts[(counts.case==case)&(counts.factor==3)&(counts.predicted_class==k)&(counts.reference_class!=MAP[k])].copy()
        sub['category']=np.where(sub.reference_class==0,0,np.where(sub.reference_class==14,1,np.where(sub.reference_class.isin(list(MAP.values())),3,2)))
        wide=sub.groupby(['arm','category']).volume_ml.sum().unstack('category',fill_value=0).reindex(index=arms,columns=range(4),fill_value=0)
        bottom=np.zeros(4)
        for category in range(4):
            values=wide[category].to_numpy()
            ax.bar(range(4),values,bottom=bottom,color=colors[category],label=category_names[category])
            bottom+=values
        ax.set_xticks(range(4),arm_labels)
        ax.set_ylabel('False-positive volume (mL)')
        ax.set_title(f'{case} · {organ.replace("_"," ")}')
        ax.spines[['top','right']].set_visible(False)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=2,fontsize=9)
    fig.suptitle('Where false positives fall in the original annotations · six-seed means at f3\nFour cases fixed by earlier extreme interactions; not representative prevalence',fontsize=13)
    fig.savefig(FIG/'fixed_case_false_positive_locations.png',dpi=180,bbox_inches='tight')
    fig.savefig(FIG/'fixed_case_false_positive_locations.svg',bbox_inches='tight')
    plt.close(fig)

    organ=pd.read_csv(DEST/'patient_organ_means.csv')
    fig,axes=plt.subplots(2,2,figsize=(10,8),constrained_layout=True)
    arm_colors=['#606c80','#2878ad','#bc7b2d','#26927a']
    for ax,choice in zip(axes.ravel(),selected):
        sub=organ[(organ.case==choice['case'])&(organ.organ==choice['organ'])]
        for j,arm in enumerate(arms):
            values=sub[sub.arm==arm].set_index('factor').loc[[1,3],'dice']*100
            ax.plot([1,3],values,'o-',label=arm_labels[j].replace('\n',' '),color=arm_colors[j])
        ax.set_xticks([1,3],['f1: common model grid','f3: synthetic degradation'])
        ax.set_ylabel('Whole-organ Dice (%)')
        ax.set_title(f'{choice["case"]} · {choice["organ"].replace("_"," ")}')
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.2)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=4,fontsize=9)
    fig.suptitle('Fixed illustrative cases at both input severities · six-seed averages\nf1 is also resampled to the common model grid',fontsize=13)
    fig.savefig(FIG/'fixed_case_degradation_comparison.png',dpi=180,bbox_inches='tight')
    fig.savefig(FIG/'fixed_case_degradation_comparison.svg',bbox_inches='tight')
    plt.close(fig)

    # All patients: class-specific FP fraction of each complete predicted organ volume.
    fig,axes=plt.subplots(1,2,figsize=(13,5.5),constrained_layout=True,sharey=True)
    images=[]
    for ax,factor in zip(axes,[1,3]):
        sub=counts[(counts.factor==factor)&counts.predicted_class.isin([1,2,3,4])]
        fractions=np.zeros((16,4))
        for k in range(1,5):
            a=sub[sub.predicted_class==k].groupby('reference_class').voxels.sum()
            fractions[:,k-1]=a.reindex(range(16),fill_value=0).to_numpy()/a.sum()*100
            fractions[MAP[k],k-1]=np.nan
        images.append(ax.imshow(fractions,aspect='auto',cmap='magma',vmin=0,vmax=25))
        ax.set_xticks(range(4),['Spleen','R. kidney','L. kidney','Liver'])
        ax.set_yticks(range(16),['Unspecified remainder' if k==0 else label_names[str(k)] for k in range(16)],fontsize=9)
        ax.set_title(f'f{factor}: all 40 patients, all four arms')
        ax.set_xlabel('Predicted target organ')
    fig.colorbar(images[0],ax=axes,label='FP category / total predicted organ volume (%)\nPooled voxel ratio; color saturates at 25%',shrink=.8)
    fig.suptitle('Original reference classes occupied by false positives · six seeds\nCorrect-class cells omitted from display; pooled volume is descriptive, not the primary patient metric',fontsize=13)
    fig.savefig(FIG/'all_patient_annotation_confusion.png',dpi=180,bbox_inches='tight')
    fig.savefig(FIG/'all_patient_annotation_confusion.svg',bbox_inches='tight')
    plt.close(fig)
    contributions=pd.read_csv(DEST/'paired_error_contributions.csv')
    components={
        'Reduced missed foreground':['fn_score'],
        'FP reduction: unspecified remainder':['fp_ref0_score'],
        'FP reduction: other target organs':[f'fp_ref{x}_score' for x in [1,2,3,6]],
        'FP reduction: other annotated organs':[f'fp_ref{x}_score' for x in [4,5,7,8,9,10,11,12,13,14,15]],
    }
    groups=['original3','additional3','all6']
    fig,axes=plt.subplots(1,2,figsize=(12,5.5),constrained_layout=True)
    component_colors=['#389080','#9399a3','#b58b48','#a36b8e']
    for ax,contrast,title in zip(axes,['context_with_aug','context_x_augmentation'],['Context gain with augmentation','Context × augmentation interaction']):
        sub=contributions[(contributions.factor==3)&(contributions.contrast==contrast)]
        positive=np.zeros(3);negative=np.zeros(3)
        for (name,metrics),color in zip(components.items(),component_colors):
            v=np.array([sub[(sub.seed_group==g)&sub.metric.isin(metrics)]['mean'].sum()*100 for g in groups])
            bottom=np.where(v>=0,positive,negative)
            ax.bar(range(3),v,bottom=bottom,color=color,label=name,width=.65)
            positive+=np.maximum(v,0);negative+=np.minimum(v,0)
        total=sub[sub.metric=='dice'].set_index('seed_group').loc[groups]
        mean=total['mean'].to_numpy()*100
        lo=total.crossed_ci_low.to_numpy()*100
        hi=total.crossed_ci_high.to_numpy()*100
        ax.errorbar(range(3),mean,yerr=[mean-lo,hi-mean],fmt='o',color='black',capsize=4,label='Total and crossed 95% interval')
        ax.axhline(0,color='#444',lw=.8)
        ax.set_xticks(range(3),['Original 3','Additional 3','All 6'])
        ax.set_ylabel('Contribution to Dice gain (pp)')
        ax.set_title(title)
        ax.spines[['top','right']].set_visible(False)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=2,fontsize=9)
    fig.suptitle('40 replication patients at f3 · exact score accounting\nModel-dependent Dice denominators: components describe the score, not independent causal effects',fontsize=13)
    fig.savefig(FIG/'score_error_contributions.png',dpi=180,bbox_inches='tight')
    fig.savefig(FIG/'score_error_contributions.svg',bbox_inches='tight')
    plt.close(fig)
    paths=sorted(x for x in FIG.iterdir() if x.suffix in ['.png','.svg'])
    (FIG/'manifest.json').write_text(json.dumps({'completed_utc':datetime.now(timezone.utc).isoformat(),'files':{str(p.relative_to(ROOT)):sha(p) for p in paths},'source_sha256':sha(__file__),'report_generated':False},indent=2))


if __name__=='__main__':
    main()
