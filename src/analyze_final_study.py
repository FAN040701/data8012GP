"""Rebuild six-seed slice-policy contrasts and the frozen-checkpoint sensitivity tests."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_offsets import bootstrap, checked_scores
from mechanism_common import sha
from slice_selection import CFG, ROOT, run_dir
from verify_mechanism import check_header, check_mask

OUT = ROOT / 'results/final_study'
SEEDS = [11, 22, 33, 44, 55, 66]
ORIGINAL = json.loads((ROOT / 'data/metadata/split.json').read_text())['splits']['test']
REPLICATION = json.loads((ROOT / CFG['test_split']).read_text())['test']
METRICS = ['case', 'seed', 'arm', 'factor', 'organ', 'dice']


def sensitivity_scores(cohort, checkpoint, arm, seed):
    folder = OUT / cohort / checkpoint / f'{arm}_seed{seed}'
    record = json.loads((folder / 'evaluation_complete.json').read_text())
    cases = ORIGINAL if cohort == 'original' else REPLICATION
    assert record['checkpoint_sha256'] == sha(run_dir(arm, seed) / f'{checkpoint}.pt')
    assert sha(folder / 'metrics.csv') == record['metrics_sha256']
    assert record['checkpoint'] == checkpoint and record['cohort'] == cohort
    assert record['predictions'] == 80 and record['rows'] == 320
    frame = pd.read_csv(folder / 'metrics.csv').assign(arm=arm)
    assert frame.kind.eq(arm).all() and frame.seed.eq(seed).all() and set(frame.case) == set(cases)
    assert not frame.duplicated(['case','factor','organ']).any() and len(frame) == 320
    for factor in CFG['factors']:
        files = folder / 'predictions' / f'f{factor}'
        assert {p.name for p in files.glob('*.nii.gz')} == {f'{case}.nii.gz' for case in cases}
        for case in cases:
            check_header((files / f'{case}.nii.gz', case))
        case = sorted(cases)[(seed + factor) % len(cases)]
        check_mask((files / f'{case}.nii.gz', case, frame[(frame.case==case) & (frame.factor==factor)].to_dict('records')))
    return frame[METRICS]


def effects(table):
    mirror_plain = table.near_far - table.far_near
    mirror_aug = table.near_far_aug - table.far_near_aug
    contrasts = dict(direction_plain=mirror_plain, direction_aug=mirror_aug,
                     direction_by_augmentation=mirror_aug-mirror_plain)
    if 'mid' in table:
        for suffix in ['', '_aug']:
            mixed = (table['near_far'+suffix] + table['far_near'+suffix]) / 2
            contrasts['mixed_minus_symmetric'+suffix] = mixed - table['mid'+suffix]
            for policy in CFG['offsets']:
                contrasts[policy+'_minus_narrow'+suffix] = table[policy+suffix] - table['context'+suffix]
    return contrasts


def summarize(frame, cohort, checkpoint):
    key = ['case','seed','factor','arm']
    pat = frame.groupby(key).dice.mean().unstack('arm')
    out = []
    for name, seeds in [('first3',SEEDS[:3]), ('new3',SEEDS[3:]), ('all6',SEEDS)]:
        if not set(seeds).issubset(frame.seed.unique()): continue
        table = pat.loc[pat.index.get_level_values('seed').isin(seeds)]
        assert table.notna().all().all() and len(table)==40*len(seeds)*2
        for contrast, values in effects(table).items():
            clean, coarse = (values.xs(k,level='factor') for k in (1,3))
            for condition, scores in [('f1',clean),('f3',coarse),('f3_minus_f1',coarse-clean)]:
                out.append(dict(cohort=cohort,checkpoint=checkpoint,group=name,contrast=contrast,
                                condition=condition,**bootstrap(scores)))
    return pd.DataFrame(out), pat.stack().rename('dice').reset_index()


def main():
    assert set(ORIGINAL).isdisjoint(REPLICATION)
    known = pd.read_csv(ROOT/'results/seed_stability/analysis/all_organ_metrics.csv')
    known = known[(known.cohort=='new40_replication') & (known.selection=='best') &
                  known.arm.isin(['context','context_aug'])]
    main_frames = [known[METRICS]]
    for p in CFG['offsets']:
        for suffix in ['', '_aug']:
            for seed in SEEDS:
                main_frames.append(checked_scores(p+suffix, seed, REPLICATION)[METRICS])
    cohorts = [('replication', 'best', pd.concat(main_frames))]
    for cohort, checkpoint, seeds in [('original','best',SEEDS[:3]),('replication','last',SEEDS)]:
        parts = [sensitivity_scores(cohort, checkpoint, p+s, seed) for p in ['near_far','far_near']
                 for s in ['', '_aug'] for seed in seeds]
        cohorts.append((cohort, checkpoint, pd.concat(parts)))
    contrasts, patient_scores, organs = [], [], []
    for cohort, ck, frame in cohorts:
        assert not frame.duplicated(METRICS[:-1]).any() and frame.dice.notna().all()
        results, patients = summarize(frame, cohort, ck)
        contrasts.append(results)
        patient_scores.append(patients.assign(cohort=cohort,checkpoint=ck))
        organs.append(frame.groupby(['organ','factor','arm']).dice.mean().reset_index().assign(cohort=cohort,checkpoint=ck))
    summary = pd.concat(contrasts)
    cases = pd.concat(patient_scores)
    cases.to_csv(OUT/'patient_scores.csv', index=False)
    summary.to_csv(OUT/'contrasts.csv', index=False)
    pd.concat(organs).to_csv(OUT/'organ_means.csv', index=False)
    cases.groupby(['cohort','checkpoint','factor','arm']).dice.mean().to_csv(OUT/'absolute_scores.csv')
    old = pd.read_csv(ROOT/'results/slice_selection/analysis/paired_effects.csv',dtype={'condition':str})
    original = old[(old.condition=='3') & (old.contrast=='mirror_direction_x_augmentation')].iloc[0]
    current = summary.query("cohort=='replication' and checkpoint=='best' and group=='first3' and condition=='f3' and contrast=='direction_by_augmentation'").iloc[0]
    assert np.allclose([original[k] for k in ['mean','ci_low','ci_high']],
                       [current[k] for k in ['mean','ci_low','ci_high']],atol=1e-12,rtol=0)
    record = dict(new_fits=18, new_masks=18*80 + 36*80, original_cohort_sensitivity_masks=12*80,
                  fixed_step_sensitivity_masks=24*80, repeated_seed_masks=18*80,
                  rechecked_slice_fits=48, rechecked_slice_fit_masks=48*80, primary='new3 replication best f3 direction_by_augmentation',
                  config_sha256=sha(ROOT/'configs/final_study.json'),contrast_sha256=sha(OUT/'contrasts.csv'))
    (OUT/'complete.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__':
    main()
