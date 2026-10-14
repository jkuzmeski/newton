#!/bin/bash
# One SLURM array task = one indexed run of a bilateral experiment, 1 H100 per task.
# Submitted by projects/impedance_instron/bilateral/grid/client.py, which passes
# GRID_ROOT, GRID_CODE_DIR, GRID_CODE_COMMIT, GRID_CODE_LABEL, EXPERIMENT, and SMOKE
# with --export, and the array range, time limit, job name, and log paths on the
# sbatch command line.
#SBATCH --partition=e9-h100nvl
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Profile and modules before strict mode: /etc/profile has unbound variables.
source /etc/profile
module load conda/Python-ML-2026a-pytorch
set -eo pipefail
: "${GRID_ROOT:?}" "${GRID_CODE_DIR:?}" "${EXPERIMENT:?}"

source "$GRID_ROOT/venv/bin/activate"
export WARP_CACHE_PATH="$GRID_ROOT/warp_cache"
export PYTHONUNBUFFERED=1
# Code snapshots are read-only; keep bytecode on node-local disk.
export PYTHONPYCACHEPREFIX="$TMPDIR/pycache"
cd "$GRID_CODE_DIR"

echo "task=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} experiment=$EXPERIMENT smoke=${SMOKE:-0} host=$(hostname) date=$(date)"
echo "code=$GRID_CODE_COMMIT label=$GRID_CODE_LABEL dir=$GRID_CODE_DIR"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

# Stage the run's dataset to node-local storage; datasets are a few MB, but every
# reset reads the in-memory copy, never the shared filesystem.
DATASET=$(python -m projects.impedance_instron.bilateral.grid.run --experiment "$EXPERIMENT" \
    --index "$SLURM_ARRAY_TASK_ID" --print-dataset)
LOCAL="$TMPDIR/data"
mkdir -p "$LOCAL"
cp -r "$GRID_ROOT/data/$DATASET" "$LOCAL/"

SMOKE_FLAG=()
if [[ "${SMOKE:-0}" == "1" ]]; then
    SMOKE_FLAG=(--smoke)
fi

python -u -m projects.impedance_instron.bilateral.grid.run \
    --experiment "$EXPERIMENT" \
    --index "$SLURM_ARRAY_TASK_ID" \
    --data-root "$LOCAL" \
    --out-root "$GRID_ROOT/runs" \
    --tensorboard-root "$GRID_ROOT/tb" \
    "${SMOKE_FLAG[@]}"

echo "done: $(date)"
