#!/bin/bash
# One SLURM array task = one entry of the run's tasks.json, 1 H100 per task.
# Submitted by projects/supermimic/grid/launch.py, which exports GRID_ROOT,
# GRID_CODE_DIR and RUN_DIR and sets the array, time limit, job name and log.
#SBATCH --partition=e9-h100nvl
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Profile before strict mode: /etc/profile has unbound variables.
source /etc/profile
set -eo pipefail
: "${GRID_ROOT:?}" "${GRID_CODE_DIR:?}" "${RUN_DIR:?}"

PYTHON="$GRID_ROOT/venv/bin/python"
export WARP_CACHE_PATH="$GRID_ROOT/warp_cache"
export PYTHONUNBUFFERED=1
# The code folder is read-only; keep bytecode on node-local disk.
export PYTHONPYCACHEPREFIX="$TMPDIR/pycache"
cd "$GRID_CODE_DIR"

NAME=$("$PYTHON" -c 'import json, sys; print(json.load(open(sys.argv[1]))[int(sys.argv[2])]["name"])' \
    "$RUN_DIR/tasks.json" "$SLURM_ARRAY_TASK_ID")
OUT="$RUN_DIR/$NAME"
mkdir -p "$OUT"
mapfile -t ARGS < <("$PYTHON" -c '
import json, sys
task = json.load(open(sys.argv[1]))[int(sys.argv[2])]
print("\n".join(a.format(data=sys.argv[3], out=sys.argv[4]) for a in task["args"]))
' "$RUN_DIR/tasks.json" "$SLURM_ARRAY_TASK_ID" "$GRID_ROOT/data" "$OUT")

echo "task=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} name=$NAME host=$(hostname) date=$(date)"
echo "code=$(cat GRID_COMMIT) out=$OUT"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
"$PYTHON" -u -m "${ARGS[@]}"
echo "done: $(date)"
