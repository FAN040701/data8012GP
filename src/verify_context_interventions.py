"""Independently verify saved intervention masks and paired effect calculations."""
from datetime import datetime, timezone
import json

import nibabel as nib
import numpy as np
import pandas as pd

from continuation_common import ROOT, OUT, CFG, run_dir, sha


def main():
    folder = OUT/'context_interventions'
    spec = json.loads((ROOT/'configs/context_interventions.json').read_text())
    ids = sorted(json.loads((ROOT/'data/metadata/split.json').read_text())['splits']['test'])
    raw = pd.read_csv(folder/'all_organ_metrics.csv')
    effects = pd.read_csv(folder/'paired_contrasts.csv')
    assert len(raw) == 5760 and len(effects) == 12
    assert set(raw.case) == set(ids)
    assert not raw.duplicated(['arm', 'seed', 'case', 'factor', 'organ', 'mode']).any()
    mask_paths = set()
    for arm in spec['arms']:
        for seed in spec['seeds']:
            run = folder/f'{arm}_seed{seed}'
            complete = json.loads((run/'complete.json').read_text())
            intact = json.loads((run_dir(arm, seed, 40000)/'evaluation_complete.json').read_text())
            assert complete['checkpoint_sha256'] == intact['checkpoint_sha256'] == sha(ROOT/complete['checkpoint'])
            assert complete['protocol_sha256'] == sha(ROOT/'configs/context_interventions.json')
            for name, digest in complete['source_sha256'].items(): assert sha(ROOT/'src'/name) == digest
            frame = pd.read_csv(run/'metrics.csv')
            assert len(frame) == 640
            for mode in spec['modes']:
                for factor in spec['factors']:
                    group = frame[(frame['mode'] == mode) & (frame.factor == factor)]
                    assert len(group) == 160 and set(group.case) == set(ids)
            for path in frame.prediction_file.unique():
                assert (ROOT/path).exists()
                mask_paths.add(path)
    assert len(mask_paths) == 960
    max_effect_error, max_interval_error = 0., 0.
    for row in effects.itertuples(index=False):
        mode = row.contrast.removesuffix('_minus_intact')
        arms = [('context_aug', 1), ('context', -1)] if row.arm == 'context_aug_minus_context' else [(row.arm, 1)]
        values = []
        for case in ids:
            seed_values = []
            for seed in spec['seeds']:
                effect = 0.
                for arm, sign in arms:
                    base = raw[(raw.arm == arm) & (raw.seed == seed) & (raw.case == case) & (raw.factor == row.factor)]
                    altered = base.loc[base['mode'] == mode, 'dice'].to_numpy()
                    intact = base.loc[base['mode'] == 'intact', 'dice'].to_numpy()
                    assert len(altered) == len(intact) == 4
                    effect += sign*(np.nanmean(altered)-np.nanmean(intact))
                seed_values.append(effect)
            values.append(np.mean(seed_values))
        values = np.array(values)
        rng = np.random.default_rng(CFG['primary']['bootstrap_seed'])
        boots = rng.choice(values, size=(10000, len(values)), replace=True).mean(1)
        ci = np.quantile(boots, [.025, .975])
        max_effect_error = max(max_effect_error, abs(float(values.mean())-row.mean))
        max_interval_error = max(max_interval_error, float(np.max(np.abs(ci-[row.ci_low, row.ci_high]))))
    assert max_effect_error < 1e-12 and max_interval_error < 1e-12
    case = ids[0]
    meta = json.loads((ROOT/'data/processed'/case/'metadata.json').read_text())
    reference = nib.as_closest_canonical(nib.load(ROOT/meta['raw_label']))
    target = np.asarray(reference.dataobj, dtype=np.uint8)
    organ_ids = [('spleen', 1), ('right_kidney', 2), ('left_kidney', 3), ('liver', 6)]
    selected = raw[(raw.case == case) & raw['mode'].isin(spec['modes'])]
    checked, max_dice_error = 0, 0.
    for key, rows in selected.groupby(['arm', 'seed', 'mode', 'factor']):
        pred = nib.load(ROOT/rows.prediction_file.iloc[0])
        assert pred.shape == reference.shape and np.allclose(pred.affine, reference.affine, rtol=0, atol=1e-5)
        assert pred.header.get_xyzt_units()[0] == 'mm'
        array = np.asarray(pred.dataobj, dtype=np.uint8)
        assert set(np.unique(array)) <= set(range(5))
        for label, (organ, original_label) in enumerate(organ_ids, 1):
            p, t = array == label, target == original_label
            row = rows[rows.organ == organ].iloc[0]
            assert p.sum() == row.pred_voxels and t.sum() == row.true_voxels
            dice = 2*np.count_nonzero(p & t)/(p.sum()+t.sum())
            max_dice_error = max(max_dice_error, abs(dice-row.dice))
        checked += 1
    assert checked == 24 and max_dice_error < 1e-12
    result = {'verified_utc': datetime.now(timezone.utc).isoformat(), 'all_passed': True,
              'same_checkpoint_as_intact_control': True, 'prediction_files': 960,
              'paired_contrasts_independently_checked': len(effects),
              'maximum_effect_disagreement': max_effect_error,
              'maximum_interval_disagreement': max_interval_error,
              'native_masks_checked': checked, 'maximum_dice_disagreement': max_dice_error,
              'check_case_selected_lexicographically': case,
              'fresh20_not_used': True}
    (folder/'verification.json').write_text(json.dumps(result, indent=2))
    paths = list(folder.glob('*.csv'))+list(folder.glob('*.json'))
    paths += list(folder.glob('*_seed*/metrics.csv'))+list(folder.glob('*_seed*/complete.json'))
    paths += [ROOT/'src/intervene_context.py', ROOT/'src/verify_context_interventions.py',
              ROOT/'src/analyze_continuation.py', ROOT/'configs/context_interventions.json']
    manifest = {str(p.relative_to(ROOT)): {'bytes': p.stat().st_size, 'sha256': sha(p)}
                for p in paths if p.name != 'artifact_manifest.json'}
    (folder/'artifact_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
