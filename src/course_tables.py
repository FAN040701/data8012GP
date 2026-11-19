"""Generate the report's numeric tables from frozen evaluation files."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/course_report'


def main():
    read=lambda path: pd.read_csv(ROOT/path)
    quality=read('results/slice_selection/proposal_quality/model_summary.csv')
    quality=quality.query("organ=='equal_organ_macro'").set_index(['kind','metric','factor'])['mean']
    baseline=[]
    for kind,label in [('2d','2D'),('2p5d','2.5D'),('3d','3D')]:
        values=[quality[kind,m,f]*(100 if m=='dice' else 1) for m in ['dice','hd95_mm'] for f in [1,2,3]]
        baseline.append([label,*values])
    factorial=read('results/seed_stability/analysis/model_summary.csv')
    factorial=factorial.query("seed_group=='all6' and selection=='best' and cohort=='new40_replication'").set_index(['arm','condition'])['mean']
    arms=[(p+s,t,a) for s,a in [('', 'Plain'),('_aug','Augmented')] for p,t in [('repeat','Repeated centre'),('context','Adjacent slices')]]
    fact=[[desc,aug,*[factorial[p,f]*100 for f in ['f1','f3']]] for p,desc,aug in arms]
    geom=read('results/final_study/absolute_scores.csv')
    geom=geom[(geom.cohort=='replication') & (geom.checkpoint=='best')].pivot(index='arm',columns='factor',values='dice')
    names={'context':'$-3,0,+3$','mid':'$-6,0,+6$','far':'$-9,0,+9$','near_far':'$-3,0,+9$','far_near':'$-9,0,+3$'}
    layouts=[[label,*[geom.loc[name+s,f]*100 for s in ['', '_aug'] for f in [1.,3.]]] for name,label in names.items()]
    raw=read('results/seed_stability/component_diagnostic/all_organ_metrics.csv')
    kidneys=[]
    for case,organ in [('amos_0224','right_kidney'),('amos_0052','left_kidney')]:
        q=raw.query("case==@case and organ==@organ and arm=='context_aug' and seed==44").iloc[0]
        kidneys.append([case[-4:],organ.replace('_',' ').capitalize(),q.dice_before*100,q.dice_after*100,
                        q.removed_false_positive_ml,q.removed_true_positive_ml])
    blocks=[]
    for key,rows in [('baselineRows',baseline),('factorRows',fact),('offsetRows',layouts),('kidneyRows',kidneys)]:
        body='\n'.join(' & '.join(v if isinstance(v,str) else f'{v:.2f}' for v in row) + r' \\' for row in rows)
        blocks.append('\\newcommand{\\'+key+'}{%\n'+body+'\n}')
    (OUT/'tables.tex').write_text('\n'.join(blocks)+'\n')


if __name__=='__main__':
    main()
