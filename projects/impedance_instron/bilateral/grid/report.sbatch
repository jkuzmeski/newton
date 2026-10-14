#!/bin/bash
# Experiment report: compares the runs of one array job once every task has ended.
# Submitted by projects/impedance_instron/bilateral/grid/client.py with
# --dependency=afterany:<array job> and GRID_ROOT, GRID_CODE_DIR, and RUNS_DIR
# (<runs>/<experiment>/<array job>) passed with --export. CPU only.
#SBATCH --partition=xeon-p8
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

source /etc/profile
module load conda/Python-ML-2026a-pytorch
set -eo pipefail
: "${GRID_ROOT:?}" "${GRID_CODE_DIR:?}" "${RUNS_DIR:?}"

source "$GRID_ROOT/venv/bin/activate"
export PYTHONPYCACHEPREFIX="$TMPDIR/pycache"
export MPLCONFIGDIR="$TMPDIR/matplotlib"
cd "$GRID_CODE_DIR"

echo "experiment report: $RUNS_DIR host=$(hostname) date=$(date)"
python -u -m projects.impedance_instron.bilateral.report experiment --dir "$RUNS_DIR"
echo "done: $(date)"
