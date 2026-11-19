#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source environment/activate.sh
card="${1:?supply GPU 0 or 1}"
[[ "$card" == 0 || "$card" == 1 ]]
export CUDA_VISIBLE_DEVICES="$card"
suffix=""
if [[ "$card" == 1 ]]; then suffix=_aug; fi
mkdir -p logs/final_study
for policy in far_near mid far; do
    for seed in 44 55 66; do
        arm="${policy}${suffix}"
        python -u src/train_final_offsets.py --arm "$arm" --seed "$seed" >"logs/final_study/${arm}_${seed}_train.log" 2>&1
        python -u src/eval_slice_selection.py --arm "$arm" --seed "$seed" >"logs/final_study/${arm}_${seed}_eval.log" 2>&1
    done
done
