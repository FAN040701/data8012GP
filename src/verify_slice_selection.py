"""Validate the completed slice-policy comparison against raw metrics and both protocols."""
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/slice_selection'


def read(path):
    return json.loads(path.read_text())


def main():
    cfg = read(ROOT / 'configs/slice_selection.json')
    repeat_cfg = read(ROOT / 'configs/slice_selection_repeat.json')
    selection = read(OUT / 'selection.json')
    policies = repeat_cfg['candidates']
    seeds = cfg['seeds']
    new_seeds = repeat_cfg['new_seeds']
    dev_scores = {}
    for policy in policies:
        best = []
        for seed in seeds:
            log = OUT / 'runs' / f'{policy}_aug_seed{seed}' / 'validation.jsonl'
            best.append(max(json.loads(line)['macro_dice'] for line in log.read_text().splitlines()))
        dev_scores[policy] = np.mean(best)
    selected = max(policies, key=dev_scores.get)
    assert selected == selection['selected_policy']
    assert all(np.isclose(value, selection['mean_augmented_development_dice'][key])
               for key, value in dev_scores.items())

    patients = set(read(ROOT / cfg['test_split'])['test'])
    original = read(ROOT / 'data/metadata/split.json')['splits']
    for split in ['train', 'val', 'test']:
        assert patients.isdisjoint(original[split])
    expected = {(p + suffix, s) for p in policies for suffix in ['', '_aug'] for s in seeds}
    expected |= {(selected + suffix, s) for suffix in ['', '_aug'] for s in new_seeds}
    metrics = []
    prediction_count = 0
    for arm, seed in sorted(expected):
        run = OUT / 'runs' / f'{arm}_seed{seed}'
        record = read(run / 'evaluation_complete.json')
        assert record['arm'] == arm and record['seed'] == seed
        for factor in cfg['factors']:
            names = {f.name.removesuffix('.nii.gz') for f in
                     (run / 'predictions' / f'f{factor}').glob('*.nii.gz')}
            assert names == patients
            prediction_count += len(names)
        frame = pd.read_csv(run / 'metrics.csv')
        assert frame.kind.eq(arm).all() and frame.seed.eq(seed).all()
        metrics.append(frame)
    scores = pd.concat(metrics).groupby(['case','seed','factor','kind']).dice.mean().unstack()
    effects = pd.read_csv(OUT / 'analysis/paired_effects.csv', dtype={'condition':str})
    reps = pd.read_csv(OUT / 'repeat_analysis/paired_effects.csv', dtype={'condition':str})
    coarse = scores.xs(3, level='factor')
    primary = coarse.loc[(slice(None), seeds), :]
    primary_values = (primary.near_far_aug + primary.far_near_aug) / 2 - primary.mid_aug
    row = effects[(effects.contrast == 'mixed_average_minus_mid_aug') & (effects.condition == '3')]
    assert len(row) == 1 and np.isclose(primary_values.mean(), row['mean'].iloc[0], atol=1e-12, rtol=0)
    controls = pd.read_csv(ROOT / 'results/seed_stability/analysis/all_organ_metrics.csv')
    controls = controls[(controls.selection == 'best') & (controls.cohort == 'new40_replication') &
                        (controls.arm == 'context_aug') & (controls.factor == 3)]
    control_scores = controls.groupby(['case','seed']).dice.mean()
    differences = coarse[selected + '_aug'] - control_scores
    for group, group_seeds in [('original3',seeds),('additional3',new_seeds),('all6',seeds+new_seeds)]:
        values = differences.loc[(slice(None),group_seeds)]
        assert len(values) == len(patients) * len(group_seeds) and values.notna().all()
        row = reps[(reps.contrast == selected + '_aug_minus_context_aug') &
                   (reps.condition == '3') & (reps.seed_group == group)]
        assert len(row) == 1 and np.isclose(values.mean(), row['mean'].iloc[0], atol=1e-12, rtol=0)

    quality = read(OUT / 'proposal_quality/verified.json')
    distances = pd.read_csv(OUT / 'proposal_quality/native_organ_metrics.csv')
    assert np.allclose(np.maximum(distances.pred_to_ref_mm, distances.ref_to_pred_mm),
                       distances.hd95_mm, equal_nan=True)
    latency = read(OUT / 'proposal_latency/complete.json')
    times = pd.read_csv(OUT / 'proposal_latency/measurements.csv')
    assert set(times.case) == set(original['test']) and set(times.kind) == {'2d','2p5d','3d'}
    assert set(times['round']) == {1,2,3} and not times.duplicated(['case','kind','round']).any()
    assert len(times) == 360 and times.end_to_end_s.gt(0).all()
    assert np.allclose(times.end_to_end_s, times.preprocess_s + times.prediction_restore_s)
    first = read(OUT / 'analysis/verified.json')
    repeated = read(OUT / 'repeat_analysis/verified.json')
    assert first['new_fits_checked'] == 24 and repeated['new_fits_checked'] == 12
    assert len(expected) == 30 and prediction_count == 2400 and sum(map(len, metrics)) == 9600
    result = dict(unique_fits=len(expected), initial_fits=24, additional_seed_fits=6,
                  evaluated_patients=len(patients), unique_new_masks=prediction_count,
                  new_organ_scores=sum(map(len, metrics)), development_selected_policy=selected,
                  original_cohort_rescored_masks=quality['predictions'], latency_measurements=latency['measurements'],
                  index=['analysis/paired_effects.csv', 'repeat_analysis/paired_effects.csv',
                         'proposal_quality/model_summary.csv', 'proposal_latency/latency_summary.csv',
                         'input_diagnostic/policy_means.csv'])
    (OUT / 'complete.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
