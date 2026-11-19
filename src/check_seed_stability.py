"""Pre-outcome code, identity, data split and prior-result checks."""
import ast
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import torch

from seed_stability_common import ROOT, OUT, CFG, PROTOCOL, BASE, sha
from experiment import init_runtime
from models import UNet, planar_input


def main():
    assert not list((OUT / 'runs').glob('*/train.jsonl'))
    original = json.loads((ROOT / 'configs/continuation.json').read_text())
    for name in ['patch', 'width', 'lr', 'val_every', 'long_steps', 'augmentation']:
        assert CFG[name] == original[name]
    split = json.loads((ROOT / 'data/metadata/split.json').read_text())['splits']
    assert len(split['train']) == 80 and len(split['val']) == 20
    assert not set(split['train']) & set(split['val'])
    for cohort in ['new40_ids', 'prior20_ids', 'spacing_ids']:
        assert not set(CFG[cohort]) & set(split['train'] + split['val'])
    assert not set(CFG['new40_ids']) & set(CFG['prior20_ids'] + CFG['spacing_ids'])
    assert [len(CFG[x]) for x in ['new40_ids', 'prior20_ids', 'spacing_ids']] == [40, 20, 21]
    initializations = {}
    for seed in CFG['new_seeds']:
        hashes = []
        for arm, base in BASE.items():
            init_runtime(seed)
            model = UNet(base, CFG['width'])
            assert sum(p.numel() for p in model.parameters()) == 1964165
            hashes.append(hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest())
        assert len(set(hashes)) == 1
        initializations[str(seed)] = hashes[0]
    assert len(set(initializations.values())) == 3
    (OUT / 'initializations.json').write_text(json.dumps(initializations, indent=2))
    halo = torch.arange(6*8*8).reshape(6,8,8).float()
    a, b = planar_input(halo, 'repeat'), planar_input(halo, '2p5d')
    assert torch.equal(a[:,1], b[:,1]) and torch.equal(a[:,0], a[:,2])
    # Training loop must be an exact AST match to the original matched-factorial loop.
    def loop(name):
        tree = ast.parse((ROOT / 'src' / name).read_text())
        return next(n for n in ast.walk(tree) if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'step')
    assert ast.dump(loop('train_seed_stability.py')) == ast.dump(loop('train_continuation.py'))
    snapshot = {}
    paths = list((ROOT / 'configs').glob('*')) + list((ROOT / 'src').glob('*.py'))
    paths += [ROOT / 'results/mechanism/session_complete.json', ROOT / 'results/mechanism/artifact_manifest.csv', ROOT / 'results/mechanism/analysis/analysis.json', ROOT / 'results/mechanism/analysis/all_organ_metrics.csv', ROOT / 'results/mechanism/absolute_spacing/all_organ_metrics.csv']
    paths += [ROOT / 'data/metadata/split.json', ROOT / 'data/metadata/mechanism_replication/split.json', ROOT / 'data/metadata/continuation/split.json', ROOT / 'data/metadata/absolute_spacing/split.json']
    for seed in CFG['old_seeds']:
        for arm in CFG['arms']:
            folder = ROOT / 'results/continuation/b40000' / f'{arm}_seed{seed}'
            paths += [folder / name for name in ['best.pt', 'last.pt', 'metadata.json', 'complete.json']]
    for p in paths:
        if p.is_file():
            snapshot[str(p.relative_to(ROOT))] = sha(p)
    (OUT / 'pre_experiment_snapshot.json').write_text(json.dumps({'created_utc':datetime.now(timezone.utc).isoformat(), 'files':snapshot}, indent=2))
    result = {'all_passed':True, 'checked_utc':datetime.now(timezone.utc).isoformat(), 'protocol_sha256':sha(PROTOCOL), 'training_loop_ast_matches_original':True, 'identical_initialization_within_seed':True, 'parameter_count':1964165, 'train_validation_test_disjoint':True, 'prior_files_frozen':len(snapshot), 'report_generated':False}
    (OUT / 'pipeline_checks.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
