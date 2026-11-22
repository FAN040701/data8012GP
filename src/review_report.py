"""Generate compact, unit-labelled report entries from the review control analyses."""
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'results/review_controls/summary'
DEST=ROOT/'reports/course_report'


def sign(x):
    return f'${x:+.2f}$'


def limits(row):
    return f"{sign(row.ci_low)} to {sign(row.ci_high)}"


def command(name, value):
    return rf"\newcommand{{\{name}}}{{{value}}}"


def stability_macros():
    blocks=pd.read_csv(SRC/'factorial_blocks_pp.csv').set_index(['run','measure'])
    var=pd.read_csv(SRC/'factorial_variation.csv').iloc[0]
    fixed = pd.read_csv(SRC/'factorial_fixed_step_pp.csv').query("measure == 'interaction at 40000'").set_index('run')
    rows=[]
    for n,label in [(0,'Original'),(1,'Refit 1'),(2,'Refit 2')]:
        inter=blocks.loc[(n,'interaction')]
        rows.append(' & '.join([label,sign(blocks.loc[(n,'plain_gain'),'mean']),
            sign(blocks.loc[(n,'aug_gain'),'mean']),f"{sign(inter['mean'])} ({limits(inter)})",
            sign(fixed.loc[n,'mean'])])+r' \\')
    stability=(f"The interaction was positive in {int(var.positive_blocks)} of three four-arm blocks at seed 11. "
        f"Its mean was {sign(var.mean_pp)} points and its between-run standard deviation was {var.sd_pp:.2f}. "
        f"A 95\\% Student \\(t\\) interval across the three block means was {sign(var.conditional_t_low_pp)} to {sign(var.conditional_t_high_pp)} points "
        "(two degrees of freedom, conditional on these patients and this seed).")
    pools=pd.read_csv(SRC/'factorial_six_seed_sensitivity_pp.csv').set_index('seed11_block')
    sensitivity=(f"Substituting each refit for the original seed-11 block while keeping the other five "
        f"seeds fixed gave pooled interactions of {sign(pools.loc[1,'mean'])} ({limits(pools.loc[1])}) and "
        f"{sign(pools.loc[2,'mean'])} ({limits(pools.loc[2])}) points, with crossed patient--seed intervals.")
    stability += f" Fixing every checkpoint at 40,000 updates gave a between-run standard deviation of {fixed['mean'].std():.2f} points."
    stability += ' ' + sensitivity
    pooled = pools['mean']
    abstract = (f"Repeating the four fits for one seed gave six-seed estimates ranging from "
                f"{pooled.min():.2f} to {pooled.max():.2f} points across the three choices of fit.")
    return [command("reviewStabilityRows", table_rows(rows)), command("reviewStability", stability),
            command("reviewStabilityAbstract", abstract)]


def cost_macros():
    costs=pd.read_csv(SRC/'computational_cost.csv').set_index('kind')
    ratio=pd.read_csv(SRC/'gpu_paired_ratio.csv').iloc[0]
    share=costs.median_preprocess_share.mul(100)
    costnote=(f"The corresponding GPU-grid ratio was {ratio['median']:.3f} ({ratio.ci_low:.3f} to {ratio.ci_high:.3f}). "
              f"CPU preprocessing accounted for about {share.mean():.0f}\\% of whole-case time in each architecture.")
    costrows=[]
    for k,l in [('2d','2D'),('2p5d','2.5D'),('3d','3D')]:
        c=costs.loc[k]
        costrows.append(' & '.join([l,str(int(c.batch)),f'{c.gpu_median_ms:.1f} ({c.gpu_q25_ms:.1f}--{c.gpu_q75_ms:.1f})',
            f'{c.median_Mvox_s:.1f}',f'{c.e2e_median_s:.2f}',f'{c.fit_minutes_median:.1f}'])+r' \\')
    return [command("reviewCostNote", costnote), command("reviewCostRows", table_rows(costrows))]


def capacity_macro():
    caps=pd.read_csv(SRC/'capacity_contrasts.csv').query("contrast=='3d_w9 minus 2p5d_w16' and organ=='all'").set_index('metric')
    scores=pd.read_csv(SRC/'capacity_scores.csv').set_index('arm')
    cap=(f"At 1.86 million parameters, the width-9 3D model achieved {100*scores.loc['3d_w9','dice']:.2f}\\% Dice "
         f"and {scores.loc['3d_w9','hd95_mm']:.2f} mm HD95. Against the 1.96-million-parameter adjacent model, "
         f"the paired differences were {sign(caps.loc['dice','mean'])} Dice points (95\\% interval {limits(caps.loc['dice'])}) "
         f"and {sign(caps.loc['hd95_mm','mean'])} mm ({limits(caps.loc['hd95_mm'])}). "
         "Both models used the same train, development and evaluation cases, three seeds and 20,000 updates. "
         "Kernel dimensionality and through-plane receptive field still differed.")
    return command("reviewCapacity", cap)


def swap_macro():
    swap=pd.read_csv(SRC/'channel_swap.csv').set_index(['arm','measure'])
    mirror=swap.loc['all']
    delta=mirror.loc['swap-induced interaction change (pp)']
    normal=mirror.loc['ordinary_dice mirror interaction (pp)']
    flipped=mirror.loc['dice mirror interaction (pp)']
    def swap_effect(arm):
        return sign(swap.loc[(arm,'swapped minus ordinary Dice (pp)'),'mean'])
    swaptext=(f"With plain training, swapping channels changed mean Dice by {swap_effect('near_far')} points "
        f"for near-inferior and {swap_effect('far_near')} for near-superior. After augmentation, the corresponding "
        f"changes were {swap_effect('near_far_aug')} and {swap_effect('far_near_aug')}. "
        f"The mirror interaction was {sign(normal['mean'])} with the trained channel order and {sign(flipped['mean'])} with "
        f"the swapped order. The paired change was {sign(delta['mean'])} points (95\\% patient--seed interval {limits(delta)}).")
    ordinary=swap.xs('native labels changed (%)',level='measure')['mean']
    swaptext+=f" Across the four conditions, {ordinary.min():.2f}--{ordinary.max():.2f}\\% of native-grid labels changed on average."
    organs=pd.read_csv(SRC/'channel_swap_organs.csv')
    organ_dice = organs[organs.measure == 'dice_delta_pp'].pivot(index='arm', columns='organ', values='mean')
    if (organ_dice.idxmin(axis=1) == 'spleen').all():
        loss = abs(organ_dice.loc['near_far_aug', 'spleen'])
        swaptext += f" Spleen Dice showed the largest loss in each condition, reaching {loss:.2f} points for the augmented near-inferior layout."
    missed = organs[organs.measure.str.startswith('fn_delta_ml_')]
    if (missed['mean'] > 0).all():
        swaptext += " In every organ and axial third, all four settings showed an increase in mean missed volume."
    return command("reviewSwap", swaptext)


def table_rows(rows):
    return "%\n" + "\n".join(rows) + "\n"


def main():
    text = [*stability_macros(), capacity_macro(), swap_macro(), *cost_macros()]
    DEST.mkdir(exist_ok=True, parents=True)
    (DEST / "controls.tex").write_text("\n".join(text) + "\n")


if __name__ == "__main__":
    main()
