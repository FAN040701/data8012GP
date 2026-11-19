"""Overlap of corrected baseline errors, for every new40 patient and seed."""
import concurrent.futures as cf
import argparse
from datetime import datetime, timezone
import json

import nibabel as nib
import numpy as np
import pandas as pd

from mechanism_common import ROOT, OUT as MAIN, sha
from prepare_data import NAMES, remap_labels

OUT = MAIN/'replication_spatial_errors'
ARMS = ['repeat', 'context', 'repeat_aug', 'context_aug']


def one(item):
    case, rows = item
    source = pd.DataFrame(rows)
    meta = json.loads((ROOT/'data/processed'/case/'metadata.json').read_text())
    target = remap_labels(np.asarray(nib.as_closest_canonical(nib.load(ROOT/meta['raw_label'])).dataobj, dtype=np.uint8))
    voxel_ml = float(np.prod(meta['original_spacing_xyz_mm'])/1000)
    result = []
    for seed, group in source.groupby('seed'):
        predictions = {a: np.asarray(nib.load(ROOT/group[group.arm == a].prediction_file.iloc[0]).dataobj, dtype=np.uint8)
                       for a in ARMS}
        for k, organ in enumerate(NAMES[1:], 1):
            gt = target == k
            for kind in ['fp', 'fn']:
                errors = {a: (p == k) & ~gt if kind == 'fp' else (p != k) & gt for a, p in predictions.items()}
                baseline = errors['repeat']
                repaired_context = baseline & ~errors['context']
                repaired_aug = baseline & ~errors['repeat_aug']
                intersection = int(np.count_nonzero(repaired_context & repaired_aug))
                union = int(np.count_nonzero(repaired_context | repaired_aug))
                original = group[(group.arm == 'repeat') & (group.organ == organ)].iloc[0]
                assert int(baseline.sum()) == original[f'{kind}_voxels']
                row = {'case': case, 'organ': organ, 'seed': int(seed), 'error_kind': kind,
                       'baseline_errors': int(baseline.sum()), 'reference_voxels': int(gt.sum()),
                       'corrected_by_context': int(repaired_context.sum()), 'corrected_by_augmentation': int(repaired_aug.sum()),
                       'correction_intersection': intersection, 'correction_union': union,
                       'correction_overlap_iou': intersection/union if union else np.nan,
                       'overlap_ml': intersection*voxel_ml}
                for arm in ['context', 'repeat_aug', 'context_aug']:
                    row[arm+'_corrected_baseline_errors'] = int(np.count_nonzero(baseline & ~errors[arm]))
                    row[arm+'_introduced_errors'] = int(np.count_nonzero(~baseline & errors[arm]))
                    row[arm+'_remaining_errors'] = int(errors[arm].sum())
                    assert row[arm+'_remaining_errors'] == row['baseline_errors']-row[arm+'_corrected_baseline_errors']+row[arm+'_introduced_errors']
                assert row['correction_union'] == row['corrected_by_context']+row['corrected_by_augmentation']-intersection
                result.append(row)
    return result


def main(resume_counts=False):
    source_path = OUT/'native_spatial_errors.csv'
    source = pd.read_csv(source_path)
    if resume_counts:
        # Recover the fully written numerical stage after a later summary/plotting error.
        frame = pd.read_csv(OUT/'all_patient_correction_overlap.csv')
    else:
        rows = []
        with cf.ProcessPoolExecutor(4) as pool:
            for i, part in enumerate(pool.map(one, [(c, r.to_dict('records')) for c, r in source.groupby('case')]), 1):
                rows.extend(part)
                if i % 10 == 0: print('Correction overlap', i, '/40', flush=True)
        frame = pd.DataFrame(rows)
        frame.to_csv(OUT/'all_patient_correction_overlap.csv', index=False)
    assert len(frame) == 40*3*4*2 and not frame.duplicated(['case', 'organ', 'seed', 'error_kind']).any()
    baseline = source[source.arm == 'repeat'].set_index(['case', 'organ', 'seed'])
    for row in frame.to_dict('records'):
        reference = baseline.loc[(row['case'], row['organ'], row['seed'])]
        assert row['baseline_errors'] == reference[row['error_kind']+'_voxels']
        assert row['reference_voxels'] == reference['true_voxels']
        assert row['correction_union'] == row['corrected_by_context']+row['corrected_by_augmentation']-row['correction_intersection']
        assert 0 <= row['correction_intersection'] <= min(row['corrected_by_context'], row['corrected_by_augmentation'])
        for arm in ['context', 'repeat_aug', 'context_aug']:
            assert row[arm+'_remaining_errors'] == row['baseline_errors']-row[arm+'_corrected_baseline_errors']+row[arm+'_introduced_errors']
    choices = json.loads((OUT/'case_selection.json').read_text())['selection']
    selected = []
    for choice in choices:
        sub = source[(source.case == choice['case']) & (source.organ == choice['organ'])]
        wide = sub.pivot(index='seed', columns='arm', values='dice')
        for seed, row in wide.iterrows():
            selected.append({**choice, 'seed': int(seed), **{a+'_dice': float(row[a]) for a in ARMS},
                             'seed_organ_interaction': float(row['context_aug']-row['repeat_aug']-row['context']+row['repeat']),
                             'combination_minus_repeat': float(row['context_aug']-row['repeat']),
                             'combination_minus_context': float(row['context_aug']-row['context']),
                             'combination_minus_repeat_aug': float(row['context_aug']-row['repeat_aug'])})
    pd.DataFrame(selected).to_csv(OUT/'selected_case_all_seed_effects.csv', index=False)
    (OUT/'overlap_complete.json').write_text(json.dumps({'completed_utc': datetime.now(timezone.utc).isoformat(),
        'all_passed': True, 'patients': 40, 'seeds': 3, 'organ_error_type_rows': len(frame),
        'all_correction_and_introduction_accounting_identities_pass': True,
        'interpretation': 'Negative interaction is not equivalent to poorer absolute combined-model accuracy. Shared correction of baseline errors can indicate overlapping gains, but introduced errors and the nonlinear Dice denominator also matter; overlap is not a complete causal explanation.',
        'all_results_descriptive_and_posthoc': True, 'no_changes_to_primary_endpoint': True,
        'raw_counts_reused_after_summary_error': resume_counts,
        'raw_counts_sha256': sha(OUT/'all_patient_correction_overlap.csv'),
        'source_data_sha256': sha(source_path), 'source_sha256': sha(__file__), 'report_generated': False}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume-counts', action='store_true')
    main(parser.parse_args().resume_counts)
