#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source environment/activate.sh
export CUDA_VISIBLE_DEVICES="$1"
mkdir -p logs/slice_selection
for seed in 11 22 33; do
    for policy in near_far far_near mid far; do
        arm="$policy"
        if [[ "$1" == 1 ]]; then arm="${policy}_aug"; fi
        python -u src/train_slice_selection.py --seed "$seed" --arm "$arm" > "logs/slice_selection/${arm}_${seed}_train.log" 2>&1
        python -u src/eval_slice_selection.py --seed "$seed" --arm "$arm" > "logs/slice_selection/${arm}_${seed}_eval.log" 2>&1
    done
done
