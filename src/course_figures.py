"""Build research figures in the single-column course report from measured outputs."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/course_report/figures'
TEAL, RUST, INK, GREY = '#087e80', '#bb5c38', '#223642', '#687781'
COLORS = ['#ef9f48','#53d7c4','#7eabff','#f37b90']
NAMES = ['Spleen','Right kidney','Left kidney','Liver']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,
    'axes.labelsize':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,
    'axes.edgecolor':'#9aabae','axes.labelcolor':INK,'text.color':INK,'xtick.color':INK,'ytick.color':INK})


def csv(path):
    return pd.read_csv(ROOT/path)


def save(fig, name):
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight',pad_inches=.055,dpi=220)
    plt.close(fig)


def image_panels():
    # Reference-only choice of plane: maximize the minimum of four relative organ areas.
    folder=ROOT/'data/processed/amos_0024'
    lab=np.load(folder/'label.npy')
    scans={f:np.load(folder/f'image_f{f}.npy').astype(np.float32)*500-200 for f in [1,3]}
    areas=np.array([(lab==k).sum((1,2)) for k in range(1,5)])
    z=int((areas/areas.max(axis=1,keepdims=True)).min(axis=0).argmax())
    fig,axs=plt.subplots(1,3,figsize=(7.1,2.35))
    occupied=np.argwhere(np.any(lab[z-3:z+4]>0,axis=0))
    low=np.maximum(0,occupied.min(axis=0)-12); high=np.minimum(lab.shape[1:],occupied.max(axis=0)+13)
    sl=np.s_[low[0]:high[0],low[1]:high[1]]
    for ax,d in zip(axs,[-3,0,3]):
        ct=scans[1][z+d][sl]; y=lab[z+d][sl]
        ax.imshow(ct,cmap='gray',vmin=-160,vmax=230,origin='lower')
        for k,color in enumerate(COLORS,1):
            if (y==k).any(): ax.contour(y==k,levels=[.5],colors=[color],linewidths=.85,origin='lower')
        ax.set_title({-3:'9 mm inferior',0:'Centre section',3:'9 mm superior'}[d]);ax.axis('off')
    fig.legend([Line2D([0],[0],color=c,lw=2.6) for c in COLORS],NAMES,loc='lower center',ncol=4,frameon=False)
    fig.subplots_adjust(left=0,right=1,top=.89,bottom=.19,wspace=.035)
    save(fig,'sections')
    y=int(np.median(np.nonzero(lab==2)[1]))
    coords=np.argwhere(lab[:,y,:]>0)
    lo=np.maximum(0,coords.min(axis=0)-np.array([7,8]));hi=np.minimum((lab.shape[0],lab.shape[2]),coords.max(axis=0)+np.array([8,9]))
    zz,xx=np.s_[lo[0]:hi[0]],np.s_[lo[1]:hi[1]]
    fig,axs=plt.subplots(1,3,figsize=(7.05,2.48),layout='constrained')
    images=[scans[f][zz,y,xx] for f in [1,3]]
    for i,(ax,img,title) in enumerate(zip(axs,[*images,abs(images[1]-images[0])],['Standard input','Threefold downsampling','Absolute change (HU)'])):
        im=ax.imshow(img,origin='lower',cmap='gray' if i<2 else 'magma',vmin=-160 if i<2 else 0,vmax=230 if i<2 else 80)
        ax.set_title(title);ax.axis('off')
        if i==2:fig.colorbar(im,ax=ax,shrink=.62,ticks=[0,40,80],fraction=.05)
    save(fig,'degradation')


def baselines():
    data=csv('results/slice_selection/proposal_quality/paired_comparisons.csv')
    fig,axs=plt.subplots(1,2,figsize=(7.1,2.45),layout='constrained')
    organs=['spleen','right_kidney','left_kidney','liver']
    for ax,metric,mult,title in zip(axs,['dice','hd95_mm'],[100,1],['Dice difference (points)','HD95 difference (mm)']):
        s=data[(data.contrast=='3d minus 2p5d')&(data.factor==1)&(data.metric==metric)].set_index('organ').loc[organs]
        ax.axvline(0,color=GREY,lw=.8,ls='--')
        for i,(_,r) in enumerate(s.iterrows()):
            ax.plot(mult*np.array([r.ci_low,r.ci_high]),[3-i]*2,color=TEAL,lw=2)
            ax.scatter(mult*r['mean'],3-i,color=INK,s=24,zorder=3)
        ax.set_yticks(range(3,-1,-1),NAMES if ax==axs[0] else ['']*4)
        ax.set_ylim(-.55,3.55);ax.set_xlabel(title)
    save(fig,'architectures')


def factorial():
    frame=csv('results/seed_stability/analysis/paired_contrasts.csv')
    frame=frame.query("selection=='best' and seed_group=='all6' and metric=='dice'")
    panels=[('Replication cohort · 40 patients', 'new40_replication', [
        ('context_without_aug','f3','Plain training'),
        ('context_with_aug','f3','Augmented training'),
        ('context_x_augmentation','f3','Interaction')]),
        ('Absolute spacing · 21 patients','absolute_spacing', [
        ('context_x_augmentation','target6mm','At 6 mm'),
        ('context_x_augmentation','target15mm','At 15 mm'),
        ('context_x_augmentation','target15_minus_target6','Paired change')])]
    fig,axs=plt.subplots(1,2,figsize=(7.1,2.05),layout='constrained',sharex=True)
    for ax,(title,cohort,rows) in zip(axs,panels):
        ax.axvline(0,ls='--',lw=.8,color=GREY)
        for i,(key,condition,_) in enumerate(rows):
            q=frame[(frame.cohort==cohort)&(frame.contrast==key)&(frame.condition==condition)].iloc[0]
            color=TEAL if i<2 else RUST
            ax.plot(100*np.array([q.crossed_ci_low,q.crossed_ci_high]),[2-i]*2,lw=2,color=color)
            ax.scatter(100*q['mean'],2-i,s=26,color=color,zorder=3)
        ax.set_ylim(-.4,2.4);ax.set_yticks([2,1,0],[r[2] for r in rows])
        ax.set_title(title);ax.set_xlabel('Dice difference (points)')
    save(fig,'factorial')


def geometry():
    d=csv('results/slice_selection/input_diagnostic/policy_means.csv')
    names=['context','mid','far','near_far','far_near']
    labels=['Adjacent','Symmetric 6 mm','Symmetric 9 mm','Near inferior','Near superior']
    offsets=[(-3,3),(-6,6),(-9,9),(-3,9),(-9,3)]
    fig,(a,b)=plt.subplots(1,2,figsize=(7.0,2.75),gridspec_kw={'width_ratios':[1.2,1]},layout='constrained')
    for i,(left,right) in enumerate(offsets):
        y=4-i
        a.plot([left,right],[y,y],color='#a6b7bb',lw=1.4);a.scatter([left,right],[y,y],s=33,color=TEAL)
        a.scatter(0,y,marker='s',s=25,color=INK)
    a.set_yticks(range(4,-1,-1),labels);a.set_xticks([-9,-6,-3,0,3,6,9]);a.set_xlabel('Offset from predicted section (mm)')
    for i,name in enumerate(names):
        val=float(d.query('policy==@name and factor==1').both_same_organ.iloc[0])*100
        b.barh(4-i,val,color=TEAL if i<3 else '#649a9c',height=.43)
        b.text(val+1.5,4-i,f'{val:.1f}',va='center',fontsize=8)
    b.set_xlim(0,100);b.set_yticks([]);b.set_xlabel('Both neighbours match organ (%)')
    for ax in (a,b):ax.set_ylim(-.6,4.6)
    save(fig,'geometry')


def errors():
    data=csv('results/seed_stability/component_diagnostic/all_organ_metrics.csv')
    fig,axs=plt.subplots(2,3,figsize=(7.1,4.8),layout='constrained')
    for row,(case,organ,k) in enumerate([('amos_0224','right_kidney',2),('amos_0052','left_kidney',3)]):
        rec=data.query("case==@case and organ==@organ and arm=='context_aug' and seed==44").iloc[0]
        meta=json.loads((ROOT/'data/processed'/case/'metadata.json').read_text())
        def native(path):return np.asarray(nib.as_closest_canonical(nib.load(ROOT/path)).dataobj)
        gt=native(meta['raw_label'])==k
        pred=native(rec.source_prediction)==k
        cc,n=ndimage.label(pred);counts=np.bincount(cc.ravel());counts[0]=0;kept=cc==counts.argmax()
        assert np.count_nonzero(kept&gt)==rec.tp_after and int(kept.sum())==rec.kept_voxels
        # Coronal projections retain islands distant from the kidney in the axial plane.
        masks=[a.any(axis=1).T for a in [gt,pred,kept]]
        ij=np.argwhere(np.logical_or.reduce(masks));lo=np.maximum(ij.min(axis=0)-9,0);hi=ij.max(axis=0)+10
        sl=np.s_[lo[0]:hi[0],lo[1]:hi[1]]
        for j,(ax,mask) in enumerate(zip(axs[row],masks)):
            gt2=masks[0][sl]; m=mask[sl]
            img=np.ones((*m.shape,3))*.973
            for condition,color in [(m&gt2,[.08,.45,.47]),(m&~gt2,[.83,.39,.24]),(~m&gt2,[.70,.79,.91])]:
                img[condition]=color
            asp=meta['original_spacing_xyz_mm'][2]/meta['original_spacing_xyz_mm'][0]
            ax.imshow(img,origin='lower',aspect=asp,interpolation='nearest')
            if row==0:ax.set_title(['Reference','Prediction','Largest component'][j])
            if j==0:ax.set_ylabel(case[-4:]+' · '+organ.replace('_',' '),fontsize=8)
            ax.set_xticks([]);ax.set_yticks([])
            for s in ax.spines.values():s.set_visible(False)
    fig.legend([Line2D([0],[0],color=c,lw=6) for c in ['#147378','#d4633d','#b3c9e8']],
               ['Overlap','Extra foreground','Missed foreground'],ncol=3,frameon=False,
               loc='outside lower center',fontsize=8)
    save(fig,'kidney_errors')


def final_plots():
    d=csv('results/final_study/contrasts.csv')
    cond=d[(d.condition=='f3') & (d.contrast=='direction_by_augmentation')]
    groups=[('replication','best','first3'),('replication','best','new3'),('replication','best','all6'),
            ('replication','last','all6'),('original','best','first3')]
    labels=['First three seeds','Additional three','All six seeds','Final checkpoint · six','Original cohort · first three']
    fig,ax=plt.subplots(figsize=(6.2,2.6),layout='constrained')
    ax.axvline(0,color=GREY,ls='--',lw=.9)
    for i,(co,ck,g) in enumerate(groups):
        r=cond.query('cohort==@co and checkpoint==@ck and group==@g').iloc[0];y=4-i
        ax.plot(100*np.array([r.ci_low,r.ci_high]),[y,y],color=TEAL,lw=2)
        ax.scatter(100*r['mean'],y,marker='D' if i==1 else 'o',color=RUST if i==1 else INK,s=34,zorder=3)
    ax.set_yticks(range(4,-1,-1),labels);ax.set_xlabel('Change in mirror-direction contrast with augmentation (Dice points)')
    ax.set_ylim(-.6,4.6);save(fig,'direction')
    axial=csv('results/final_study/axial/contributions.csv')
    s=axial[axial.contrast=='direction_by_augmentation'].pivot(index='band',columns='component',values='mean').loc[['inferior','middle','superior']]
    fig,ax=plt.subplots(figsize=(6.1,2.35),layout='constrained')
    for i,(col,title,color) in enumerate([('fp_loss','Extra foreground',RUST),('fn_loss','Missed foreground',TEAL)]):
        ax.bar(np.arange(3)+(.36*i-.18),s[col]*100,width=.34,label=title,color=color)
    ax.set_xticks(range(3),['Inferior third','Middle third','Superior third'])
    ax.axhline(0,color=GREY,lw=.85);ax.set_ylabel('Contribution (Dice points)')
    ax.legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,1.19),ncol=2)
    save(fig,'error_bands')


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('stage',choices=['base','final'])
    args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.stage=='base':
        image_panels(); baselines(); factorial(); geometry(); errors()
    else:
        final_plots()
