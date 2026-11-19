"""Patient-paired analysis and scientific figures from completed runs only."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/analysis'
FIG=OUT/'figures'
ORDER=['2d','repeat','2p5d','3d','2p5d_aug']
LABEL={'2d':'2D','repeat':'Repeated centre','2p5d':'Adjacent 2.5D','3d':'3D','2p5d_aug':'2.5D + resolution aug.'}
COLOR={'2d':'#667085','repeat':'#8B5CB8','2p5d':'#2376BB','3d':'#D87B28','2p5d_aug':'#189982'}
ORGANS=['spleen','right_kidney','left_kidney','liver']


def interval(values,seed=20261114):
    a=np.asarray(values,dtype=float); a=a[np.isfinite(a)]
    if not len(a): return [None,None,None]
    rng=np.random.default_rng(seed)
    boot=a[rng.integers(0,len(a),size=(2000,len(a)))].mean(1)
    return [float(a.mean()),float(np.quantile(boot,.025)),float(np.quantile(boot,.975))]


def save(fig,name):
    fig.savefig(FIG/(name+'.png'),dpi=180,bbox_inches='tight',facecolor='white')
    fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight',facecolor='white')
    plt.close(fig)


def qualitative(data,case,label,seed):
    meta=json.loads((ROOT/'data/processed'/case/'metadata.json').read_text())
    ct=np.asarray(nib.as_closest_canonical(nib.load(ROOT/meta['raw_image'])).dataobj,dtype=np.float32)
    raw=np.asarray(nib.as_closest_canonical(nib.load(ROOT/meta['raw_label'])).dataobj,dtype=np.uint8)
    gt=np.zeros(raw.shape,np.uint8)
    for old,new in [(1,1),(2,2),(3,3),(6,4)]: gt[raw==old]=new
    z=int(((gt==1)|(gt==2)|(gt==3)).sum((0,1)).argmax())
    kinds=[k for k in ['repeat','2p5d','3d','2p5d_aug'] if (ROOT/'results/runs'/f'{k}_seed{seed}'/'predictions'/'f1'/f'{case}.nii.gz').exists()]
    fig,axes=plt.subplots(2,len(kinds)+1,figsize=(3.0*(len(kinds)+1),6.2),squeeze=False)
    cmap=matplotlib.colors.ListedColormap(['#00000000','#F4C542','#E25A62','#54A9DC','#74B85A'])
    for row,factor in enumerate([1,3]):
        for col,kind in enumerate(['reference']+kinds):
            ax=axes[row,col]
            pred=gt if kind=='reference' else np.asarray(nib.load(ROOT/'results/runs'/f'{kind}_seed{seed}'/'predictions'/f'f{factor}'/f'{case}.nii.gz').dataobj)
            ax.imshow(ct[:,:,z].T,origin='lower',cmap='gray',vmin=-200,vmax=300)
            overlay=np.ma.masked_where(pred[:,:,z].T==0,pred[:,:,z].T)
            ax.imshow(overlay,origin='lower',cmap=cmap,vmin=0,vmax=4,alpha=.50,interpolation='nearest')
            title='Reference' if kind=='reference' else LABEL[kind]
            ax.set_title(title,fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            if col==0: ax.set_ylabel('Native input' if factor==1 else 'Factor-3 input',fontsize=11)
    fig.suptitle(f'{label}: {case}, seed {seed}; axial plane selected by maximal kidney/spleen reference area',fontsize=12)
    fig.text(.5,.01,'Identical native CT background in all panels. Colours: spleen yellow; right kidney red; left kidney blue; liver green.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.03,1,.94]); save(fig,f'qualitative_{label.lower().replace(" ","_")}')
    return {'case':case,'seed':int(seed),'axial_index_canonical':z,'selection_rule':label}


def main():
    FIG.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','axes.labelcolor':'#344054','text.color':'#172B4D'})
    frames=[]; runs=[]
    for complete in sorted((ROOT/'results/runs').glob('*/evaluation_complete.json')):
        run=complete.parent
        info=json.loads(complete.read_text())
        frame=pd.read_csv(run/'metrics.csv')
        if len(frame)!=info['cases']*3*4: raise ValueError(f'Incomplete rows in {run}')
        if frame.duplicated(['kind','seed','case','factor','organ']).any(): raise ValueError('Duplicate metric rows')
        frames.append(frame)
        runs.append({**json.loads((run/'metadata.json').read_text()),**json.loads((run/'complete.json').read_text()),'evaluation_seconds':info['elapsed_seconds'],'evaluation_backend':info['inverse_resampling_backend'],'evaluation_source_sha256':info['evaluation_source_sha256']})
    if not frames: raise RuntimeError('No completed evaluations')
    data=pd.concat(frames,ignore_index=True)
    cfg=json.loads((ROOT/'configs/experiment.json').read_text())
    expected_cases=set(json.loads((ROOT/'data/metadata/split.json').read_text())['splits']['test'])
    for run in runs:
        for key in ['steps','patch','width','lr','val_every']:
            if run['config'][key]!=cfg[key]: raise ValueError(f'Unmatched training configuration: {run["kind"]} {key}')
        sub=data[(data.kind==run['kind'])&(data.seed==run['seed'])]
        for factor in [1,2,3]:
            if set(sub.loc[sub.factor==factor,'case'])!=expected_cases: raise ValueError('Unmatched test cohort')
    initialization_checks=[]
    evaluation_versions={json.dumps(r['evaluation_source_sha256'],sort_keys=True) for r in runs}
    if len(evaluation_versions)!=1: raise ValueError('Reported runs use different evaluation versions')
    for seed in sorted({r['seed'] for r in runs}):
        matched=[r for r in runs if r['seed']==seed and r['kind'] in ['repeat','2p5d','2p5d_aug']]
        hashes={r['initial_parameter_sha256'] for r in matched}
        if len(hashes)>1: raise ValueError('The context-control initial weights differ')
        initialization_checks.append({'seed':seed,'models':[r['kind'] for r in matched],'identical_initial_weights':len(hashes)==1})
    (OUT/'analysis_integrity_checks.json').write_text(json.dumps({'matching_configurations':True,'matching_test_patients':True,'unique_metric_keys':True,'matching_evaluation_source_hashes':True,'context_control_initialization':initialization_checks},indent=2))
    data.to_csv(OUT/'all_organ_metrics.csv',index=False)
    patient=data.groupby(['kind','seed','case','factor'],as_index=False).agg(dice=('dice','mean'),hd95_mm=('hd95_mm','mean'),relative_volume_error=('relative_volume_error','mean'))
    patient.to_csv(OUT/'patient_metrics.csv',index=False)
    averaged=patient.groupby(['kind','case','factor'],as_index=False)[['dice','hd95_mm','relative_volume_error']].mean()
    available=[k for k in ORDER if k in data.kind.unique()]
    summaries=[]
    for kind in available:
        for factor in [1,2,3]:
            sub=averaged[(averaged.kind==kind)&(averaged.factor==factor)]
            mean,lo,hi=interval(sub.dice)
            seedmeans=patient[(patient.kind==kind)&(patient.factor==factor)].groupby('seed').dice.mean()
            failures=data[(data.kind==kind)&(data.factor==factor)&(data.status!='ok')]
            summaries.append({'kind':kind,'factor':factor,'patients':len(sub),'seeds':int(len(seedmeans)),'dice_mean':mean,'dice_ci_low':lo,'dice_ci_high':hi,'hd95_mm_mean':float(sub.hd95_mm.mean()),'relative_volume_error_mean':float(sub.relative_volume_error.mean()),'seed_dice_min':float(seedmeans.min()),'seed_dice_max':float(seedmeans.max()),'empty_case_organ_seed_count':len(failures),'missed_reference_present_organ_count':int((failures.status=='prediction_empty').sum()),'absent_reference_false_positive_ml_total':float(data.loc[(data.kind==kind)&(data.factor==factor),'absent_reference_false_positive_ml'].sum())})
    summary=pd.DataFrame(summaries); summary.to_csv(OUT/'model_summary.csv',index=False)
    distribution=averaged.groupby(['kind','factor']).agg(hd95_patient_median_mm=('hd95_mm','median'),hd95_patient_p90_mm=('hd95_mm',lambda x:x.quantile(.9)),hd95_patient_max_mm=('hd95_mm','max'),dice_patient_median=('dice','median'),dice_patient_p10=('dice',lambda x:x.quantile(.1)))
    distribution.to_csv(OUT/'patient_metric_distribution.csv')
    data.sort_values('hd95_mm',ascending=False).head(30).to_csv(OUT/'largest_boundary_errors.csv',index=False)
    seedtable=patient.groupby(['kind','seed','factor'],as_index=False)[['dice','hd95_mm']].mean()
    seedtable.to_csv(OUT/'seed_summary.csv',index=False)
    contrasts=[]; paired_values=[]
    for a,b,label in [('2p5d','repeat','context_gain'),('3d','2d','3d_minus_2d'),('2p5d_aug','2p5d','augmentation_gain')]:
        if a not in available or b not in available: continue
        seeds=sorted(set(patient.loc[patient.kind==a,'seed']) & set(patient.loc[patient.kind==b,'seed']))
        aa=patient[(patient.kind==a)&patient.seed.isin(seeds)]
        bb=patient[(patient.kind==b)&patient.seed.isin(seeds)]
        pair=aa.merge(bb,on=['seed','case','factor'],suffixes=('_a','_b'))
        pair['difference']=pair.dice_a-pair.dice_b
        pairs=pair.groupby(['case','factor'],as_index=False).difference.mean()
        pairs['contrast']=label; paired_values.append(pairs)
        for factor in [1,2,3]:
            sub=pairs[pairs.factor==factor]
            mean,lo,hi=interval(sub.difference)
            sm=pair[pair.factor==factor].groupby('seed').difference.mean()
            contrasts.append({'contrast':label,'condition':f'f{factor}','mean':mean,'ci_low':lo,'ci_high':hi,'patients':len(sub),'matched_seeds':seeds,'seed_mean_min':float(sm.min()),'seed_mean_max':float(sm.max())})
        wide=pairs.pivot(index='case',columns='factor',values='difference')
        for factor in [2,3]:
            delta=(wide[factor]-wide[1]).dropna()
            mean,lo,hi=interval(delta)
            per_seed=pair.pivot(index=['seed','case'],columns='factor',values='difference')
            seed_delta=(per_seed[factor]-per_seed[1]).groupby('seed').mean()
            contrasts.append({'contrast':label,'condition':f'f{factor}_minus_f1','mean':mean,'ci_low':lo,'ci_high':hi,'patients':len(delta),'matched_seeds':seeds,'seed_mean_min':float(seed_delta.min()),'seed_mean_max':float(seed_delta.max())})
    contrast=pd.DataFrame(contrasts); contrast.to_csv(OUT/'paired_contrasts.csv',index=False)
    if paired_values: pd.concat(paired_values,ignore_index=True).to_csv(OUT/'paired_patient_differences.csv',index=False)
    organ=data.groupby(['kind','factor','organ'],as_index=False)[['dice','hd95_mm','relative_volume_error']].mean()
    organ.to_csv(OUT/'organ_summary.csv',index=False)

    fig,ax=plt.subplots(figsize=(8.4,4.5))
    for k in available:
        s=summary[summary.kind==k].sort_values('factor')
        ax.plot(s.factor,100*s.dice_mean,'o-',lw=2,color=COLOR[k],label=LABEL[k])
        ax.fill_between(s.factor,100*s.dice_ci_low,100*s.dice_ci_high,color=COLOR[k],alpha=.09)
    ax.set_xticks([1,2,3],['Native','2× z downsampling','3× z downsampling'])
    ax.set_ylabel('Patient macro Dice (%)'); ax.set_title('Segmentation robustness on the same held-out patients')
    ax.grid(axis='y',alpha=.18); ax.legend(fontsize=9,ncol=2)
    fig.tight_layout(); save(fig,'robustness_curves')

    fig,axes=plt.subplots(1,2,figsize=(10.6,4.0))
    for ax,name,title in zip(axes,['context_gain','augmentation_gain'],['Benefit of real neighbouring slices','Effect of resolution augmentation']):
        sub=contrast[(contrast.contrast==name)&contrast.condition.isin(['f1','f2','f3'])]
        if len(sub):
            ka,kb=('2p5d','repeat') if name=='context_gain' else ('2p5d_aug','2p5d')
            per_seed=seedtable[seedtable.kind==ka].merge(seedtable[seedtable.kind==kb],on=['seed','factor'],suffixes=('_a','_b'))
            for si,seed in enumerate(sorted(per_seed.seed.unique())):
                sm=per_seed[per_seed.seed==seed].sort_values('factor')
                ax.plot(sm.factor,100*(sm.dice_a-sm.dice_b),':',color='#98A2B3',lw=1.2,alpha=.8,label='Individual seed means' if si==0 else None)
            x=np.arange(1,4); means=sub['mean'].to_numpy()*100
            ax.errorbar(x,means,yerr=np.stack([means-sub.ci_low.to_numpy()*100,sub.ci_high.to_numpy()*100-means]),fmt='o-',capsize=4,color=COLOR['2p5d' if name=='context_gain' else '2p5d_aug'],lw=2,label='Seed-averaged patient estimate')
            ax.legend(fontsize=7,loc='best')
        else: ax.text(.5,.5,'Not completed',ha='center',transform=ax.transAxes)
        ax.axhline(0,color='#667085',ls='--',lw=1)
        ax.set_xticks([1,2,3],['Native','Factor 2','Factor 3']); ax.set_ylabel('Paired Dice difference (percentage points)'); ax.set_title(title,fontsize=11)
        ax.grid(axis='y',alpha=.15)
    fig.tight_layout(); save(fig,'paired_effects')

    fig,axes=plt.subplots(2,2,figsize=(9.4,6.3),sharex=True)
    for ax,org in zip(axes.flat,ORGANS):
        for k in available:
            s=organ[(organ.organ==org)&(organ.kind==k)].sort_values('factor')
            ax.plot(s.factor,s.dice*100,'o-',color=COLOR[k],label=LABEL[k],lw=1.7)
        ax.set_title(org.replace('_',' ').title(),fontsize=11); ax.set_xticks([1,2,3]); ax.grid(alpha=.15); ax.set_ylabel('Dice (%)')
    axes[1,0].set_xlabel('Downsampling factor'); axes[1,1].set_xlabel('Downsampling factor')
    handles,labels=axes[0,0].get_legend_handles_labels(); fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=9)
    fig.tight_layout(rect=[0,.1,1,1]); save(fig,'organ_robustness')

    fig,axes=plt.subplots(1,2,figsize=(10.5,4.1))
    for k in available:
        for run in [r for r in runs if r['kind']==k]:
            path=ROOT/'results/runs'/f'{k}_seed{run["seed"]}'
            vals=pd.read_json(path/'validation.jsonl',lines=True)
            axes[0].plot(vals.step,vals.macro_dice*100,color=COLOR[k],alpha=.65,lw=1.3,label=LABEL[k] if run['seed']==min(r['seed'] for r in runs if r['kind']==k) else None)
            train=pd.read_json(path/'train.jsonl',lines=True)
            axes[1].plot(train.step,train.loss.rolling(5,min_periods=1).mean(),color=COLOR[k],alpha=.45,lw=1)
    axes[0].set_title('Development performance; each line is one seed'); axes[0].set_ylabel('Macro Dice at 3 mm (%)'); axes[0].legend(fontsize=8)
    axes[1].set_title('Training loss (5-point moving average)'); axes[1].set_ylabel('Matched-slab CE + Dice loss')
    for ax in axes: ax.set_xlabel('Optimizer update'); ax.grid(alpha=.15)
    fig.tight_layout(); save(fig,'learning_curves')

    audits=json.loads((ROOT/'data/metadata/preprocessing_audit.json').read_text())
    fig,ax=plt.subplots(figsize=(8.1,3.1))
    for split,col in [('train','#2376BB'),('val','#D87B28'),('test','#189982')]:
        vals=[r['original_spacing_xyz_mm'][2] for r in audits if r['split']==split]
        vc=pd.Series(vals).round(2).value_counts().sort_index()
        offset={'train':-.17,'val':0,'test':.17}[split]
        ax.bar(vc.index.to_numpy()+offset,vc.values,width=.16,label=split,color=col)
    ax.set_xlabel('Original through-plane voxel sampling (mm)'); ax.set_ylabel('Patients'); ax.set_title('Acquisition sampling in the selected cohort'); ax.legend()
    fig.tight_layout(); save(fig,'spacing_distribution')
    selections=[]
    if '2p5d' in available:
        base=averaged[(averaged.kind=='2p5d')&(averaged.factor==1)].set_index('case').dice
        median_case=(base-base.median()).abs().idxmin()
        seed=min(patient.loc[patient.kind=='2p5d','seed'])
        selections.append(qualitative(data,median_case,'Median native case',seed))
        p=averaged[averaged.kind=='2p5d'].pivot(index='case',columns='factor',values='dice')
        worst_case=(p[3]-p[1]).idxmin()
        selections.append(qualitative(data,worst_case,'Largest degradation loss',seed))
    result={'completed_runs':len(runs),'models':available,'runs':runs,'summaries':summaries,'contrasts':contrasts,'qualitative_selection':selections,'bootstrap':{'replicates':2000,'unit':'patient','seed':20261114,'seed_predictions_averaged_before_resampling':True,'interpretation':'Patient uncertainty conditional on the completed trained models; seed variability reported separately'},'audit_cases':len(audits)}
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2))
    print(summary.to_string(index=False))
    print(contrast.to_string(index=False))


if __name__=='__main__': main()
