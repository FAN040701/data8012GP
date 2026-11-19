#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source environment/activate.sh
out=results/slice_selection
while [[ ! -f "$out/runs/far_seed33/evaluation_complete.json" || ! -f "$out/runs/far_aug_seed33/evaluation_complete.json" ]]; do sleep 45; done
python -u src/analyze_offsets.py > logs/slice_selection/analysis.log 2>&1
policy=$(python src/select_offsets.py)
printf 'Repeating development-selected policy %s\n' "$policy"
# Run the timing experiment with no experiment training on either GPU.
while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)" ]]; do sleep 30; done
CUDA_VISIBLE_DEVICES=0 python -u src/proposal_latency.py > logs/slice_selection/latency.log 2>&1
repeat_policy() {
    local card="$1" arm="$2" seed
    for seed in 44 55 66; do
        CUDA_VISIBLE_DEVICES="$card" python -u src/train_slice_selection_repeat.py --seed "$seed" --arm "$arm" > "logs/slice_selection/${arm}_${seed}_train.log" 2>&1
        CUDA_VISIBLE_DEVICES="$card" python -u src/eval_slice_selection.py --seed "$seed" --arm "$arm" > "logs/slice_selection/${arm}_${seed}_eval.log" 2>&1
    done
}
repeat_policy 0 "$policy" &
plain_pid=$!
repeat_policy 1 "${policy}_aug" &
aug_pid=$!
wait "$plain_pid"
wait "$aug_pid"
python -u src/analyze_offsets.py --repeat > logs/slice_selection/repeat_analysis.log 2>&1
