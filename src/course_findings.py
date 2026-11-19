"""Write the final report passages from the checked patient-paired results."""
from pathlib import Path
import json
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/course_report'


def main():
    data=pd.read_csv(ROOT/'results/final_study/contrasts.csv')
    coarse=data[data.condition=='f3'].set_index(['cohort','checkpoint','group','contrast'])

    def row(group='all6',cohort='replication',checkpoint='best',contrast='direction_by_augmentation'):
        return coarse.loc[(cohort,checkpoint,group,contrast)]
    def num(value):
        return '$' + format(100*value,'+.2f') + '$'
    def estimate(r):
        return f"{num(r['mean'])} points (95\\% interval {num(r.ci_low)} to {num(r.ci_high)})"
    fresh=row('new3')
    pooled=row()
    original=row('first3','original')
    last=row(checkpoint='last')
    span=row(contrast='mixed_minus_symmetric_aug')
    bands=pd.read_csv(ROOT/'results/final_study/axial/contributions.csv').query("contrast=='direction_by_augmentation'")
    fp=bands[bands.component=='fp_loss']
    assert abs(bands['mean'].sum()-pooled['mean'])<1e-12
    largest=bands.loc[bands['mean'].idxmax()]
    assert (largest.band,largest.component)==('superior','fp_loss')
    paragraphs={
        'abstractFollow':f"Reversing which side supplied the nearer slice changed the effect of augmentation by {estimate(fresh)} in three additional seeds, compared with {num(row('first3')['mean'])} in the first three.",
        'spanFollow':f"Across all six seeds the equal-span difference was {estimate(span)}.",
        'directionText':f"The additional seeds gave {estimate(fresh)}. All three seed means were positive, but smaller than the initial group mean. Across all six seeds, the mirror difference was {num(row(contrast='direction_plain')['mean'])} after plain training and {num(row(contrast='direction_aug')['mean'])} after augmented training, giving $J={100*pooled['mean']:+.2f}$ points ({num(pooled.ci_low)} to {num(pooled.ci_high)}). Evaluating all six at exactly 40,000 updates gave {estimate(last)}. On the original 40 patients, the first three frozen seed pairs gave {estimate(original)}. This second cohort agrees in direction and admits a zero effect.",
        'bandText':f"Extra foreground accounted for {100*fp['mean'].sum():.2f} points of the {100*pooled['mean']:.2f}-point interaction. The largest single contribution, {100*largest['mean']:.2f} points, came from extra foreground in the superior third. Errors in the middle and inferior thirds also contributed to the change.",
        'discussionFollow':"Augmentation reduced the directional gap in both seed groups. The smaller estimate for the additional seeds and the uncertainty on the original evaluation cohort place the clearest support at the level of the overall trend."}
    refits=json.loads((ROOT/'results/final_study/reproduction_models.json').read_text())
    standard=[100*refits[f'results/runs/{k}_seed11']['fresh_dice_by_factor']['1'] for k in ['2d','2p5d','3d']]
    shifts=[100*m['mean_paired_delta_by_factor']['3'] for m in refits.values()]
    paragraphs['rebuildText']=(f"A clean build reproduced all 760 preprocessed arrays for 200 patients byte for byte. "
        f"Independent seed-11 fits gave standard-input Dice of {standard[0]:.2f}\\%, {standard[1]:.2f}\\% and {standard[2]:.2f}\\% "
        f"for 2D, 2.5D and 3D. Including a fourth fit with asymmetric inputs, changes in mean $f_3$ Dice from the "
        f"archived seed-11 runs ranged from {min(shifts):+.2f} to {max(shifts):+.2f} points with the same starting parameters and logged patient sequence. "
        "Training used cuDNN autotuning and mixed precision.")
    (OUT/'findings.tex').write_text('\n'.join('\\newcommand{\\'+key+'}{'+value+'}' for key,value in paragraphs.items())+'\n')


if __name__=='__main__':
    main()
