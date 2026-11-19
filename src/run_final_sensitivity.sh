#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source environment/activate.sh
card="${1:?supply GPU 0 or 1}"
[[ "$card" == 0 || "$card" == 1 ]]
export CUDA_VISIBLE_DEVICES="$card"
suffix=""
if [[ "$card" == 1 ]]; then suffix=_aug; fi
while [[ ! -f "results/slice_selection/runs/far${suffix}_seed66/evaluation_complete.json" ]]; do sleep 30; done
for policy in near_far far_near; do
    for seed in 11 22 33; do
        python -u src/eval_slice_selection.py --arm "${policy}${suffix}" --seed "$seed" --cohort original
    done
done
for policy in near_far far_near; do
    for seed in 11 22 33 44 55 66; do
        python -u src/eval_slice_selection.py --arm "${policy}${suffix}" --seed "$seed" --checkpoint last
    done
done
