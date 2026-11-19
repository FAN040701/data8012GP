"""Separate kidney laterality confusion from geometric overlap on saved masks."""
import concurrent.futures as cf
from datetime import datetime, timezone
import json

import nibabel as nib
import numpy as np
import pandas as pd

from continuation_common import ROOT, OUT

KINDS = ['2d', 'repeat', '2p5d', '3d', '2p5d_aug']
SEEDS = [11, 22, 33]


def one(case):
    meta = json.loads((ROOT/'data/processed'/case/'metadata.json').read_text())
    truth = np.asarray(nib.as_closest_canonical(nib.load(ROOT/meta['raw_label'])).dataobj,
                       dtype=np.uint8).ravel()
    positions = np.flatnonzero((truth == 2) | (truth == 3))
    target = truth[positions]
    del truth
    rows = []
    for kind in KINDS:
        for seed in SEEDS:
            run = ROOT/'results/runs'/f'{kind}_seed{seed}'
            metrics = pd.read_csv(run/'metrics.csv')
            for factor in [1, 3]:
                prediction = np.asarray(nib.load(run/'predictions'/f'f{factor}'/f'{case}.nii.gz').dataobj,
                                        dtype=np.uint8).ravel()[positions]
                matched = metrics[(metrics.case == case) & (metrics.factor == factor) &
                                  metrics.organ.isin(['right_kidney', 'left_kidney'])]
                assert len(matched) == 2 and int(matched.true_voxels.sum()) == len(target)
                predicted_volume = int(matched.pred_voxels.sum())
                correct = int(np.count_nonzero(prediction == target))
                wrong = int(np.count_nonzero(((target == 2) & (prediction == 3)) |
                                             ((target == 3) & (prediction == 2))))
                background = int(np.count_nonzero(prediction == 0))
                other = int(np.count_nonzero((prediction == 1) | (prediction == 4)))
                assert correct+wrong+background+other == len(target)
                denom = predicted_volume+len(target)
                rows.append({'kind': kind, 'seed': seed, 'factor': factor, 'case': case,
                             'true_kidney_voxels': len(target), 'predicted_kidney_voxels': predicted_volume,
                             'correct_kidney_label_voxels': correct, 'wrong_kidney_label_voxels': wrong,
                             'kidney_label_confusion_fraction': wrong/len(target),
                             'kidney_as_background_voxels': background, 'kidney_as_other_organ_voxels': other,
                             'kidney_union_dice': 2*(correct+wrong)/denom,
                             'kidney_label_aware_micro_dice': 2*correct/denom,
                             'label_confusion_micro_dice_cost': 2*wrong/denom})
    return rows


def main():
    dest = OUT/'kidney_label_diagnostics'
    dest.mkdir(parents=True, exist_ok=True)
    split = json.loads((ROOT/'data/metadata/split.json').read_text())['splits']
    protocol = {'created_utc': datetime.now(timezone.utc).isoformat(),
                'status': 'Post-hoc diagnostic of the previous completed study, no model changes',
                'cases': split['test'], 'models': KINDS, 'seeds': SEEDS, 'factors': [1, 3],
                'definition': 'Wrong kidney label means reference right predicted left or reference left predicted right. Union Dice ignores kidney side; label-aware micro Dice retains side. Their difference is exactly 2*wrong_label_voxels/(total reference plus total predicted kidney voxels).',
                'fresh20_used': False, 'replaces_primary_macro_dice': False}
    (dest/'protocol.json').write_text(json.dumps(protocol, indent=2))
    with cf.ThreadPoolExecutor(4) as pool:
        rows = [r for sub in pool.map(one, split['test']) for r in sub]
    frame = pd.DataFrame(rows)
    assert len(frame) == 1200
    frame.to_csv(dest/'patient_metrics.csv', index=False)
    means = frame.groupby(['kind', 'factor'], as_index=False).agg(
        wrong_label_fraction=('kidney_label_confusion_fraction', 'mean'),
        union_dice=('kidney_union_dice', 'mean'),
        label_aware_micro_dice=('kidney_label_aware_micro_dice', 'mean'),
        confusion_dice_cost=('label_confusion_micro_dice_cost', 'mean'),
        affected_seed_patient_cases=('wrong_kidney_label_voxels', lambda x: int((x > 0).sum())),
        affected_1pct_seed_patient_cases=('kidney_label_confusion_fraction', lambda x: int((x > .01).sum())))
    means.to_csv(dest/'summary.csv', index=False)
    (dest/'complete.json').write_text(json.dumps({'saved_predictions_analyzed': 1200,
        'completed_utc': datetime.now(timezone.utc).isoformat(), 'models_unchanged': True,
        'no_report_generated': True}, indent=2))
    print(means.to_string(index=False))


if __name__ == '__main__': main()
