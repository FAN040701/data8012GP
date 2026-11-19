"""Compare fresh runs of the public entry points to archived runs on the same inputs."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
RUNS=['results/runs/2d_seed11','results/runs/2p5d_seed11','results/runs/3d_seed11','results/slice_selection/runs/near_far_seed11']


def read(p):
    return json.loads(p.read_text())


def main():
    output={}
    fresh=ROOT/'environment/tmp/reproduction_check'
    for path in RUNS:
        src=ROOT/path
        dst=fresh/path
        old,new=[read(p/'complete.json') for p in [src,dst]]
        before,after=[read(p/'metadata.json') for p in [src,dst]]
        for key in ['initial_parameter_sha256','val_ids','train_ids']:
            assert before[key]==after[key]
        assert old['training_steps']==new['training_steps']
        traces=[[json.loads(line) for line in (p/'train.jsonl').read_text().splitlines()] for p in [src,dst]]
        for field in ['step','patient','lr']:
            assert [r[field] for r in traces[0]]==[r[field] for r in traces[1]]
        if 'start' in traces[0][0]:
            for field in ['start','factor']:
                assert [r[field] for r in traces[0]]==[r[field] for r in traces[1]]
        if 'sampling_sha256' in old: assert old['sampling_sha256']==new['sampling_sha256']
        keys=['case','factor','organ']
        a,b=[pd.read_csv(p/'metrics.csv').set_index(keys).sort_index() for p in [src,dst]]
        assert a.index.equals(b.index) and a.true_voxels.equals(b.true_voxels)
        assert np.array_equal(a.dice.isna(),b.dice.isna()) and np.isfinite(b.dice.dropna()).all()
        delta=b.dice-a.dice
        output[path]={'native_predictions':len(a)//4,'organ_scores':len(a),'original_best':old['best_step'],
                      'fresh_best':new['best_step'],'absolute_max_organ_difference':float(delta.abs().max()),
                      'original_dice_by_factor':a.dice.groupby('factor').mean().to_dict(),
                      'fresh_dice_by_factor':b.dice.groupby('factor').mean().to_dict(),
                      'same_initialization':True,'same_training_steps':True,
                      'mean_paired_delta_by_factor':delta.groupby('factor').mean().to_dict()}
    (ROOT/'results/final_study/reproduction_models.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))


if __name__=='__main__':
    main()
