#!/usr/bin/env bash
# Source from the public checkout after following the README installation steps.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/.venv/bin/activate"
export TMPDIR="$PROJECT_DIR/environment/tmp"
export XDG_CACHE_HOME="$PROJECT_DIR/environment/cache"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export PYTHONPYCACHEPREFIX="$XDG_CACHE_HOME/pycache"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
