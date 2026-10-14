#!/bin/bash
# Serve the TensorBoard of all bilateral grid runs ($GRID_ROOT/tb) on the LLSC portal.
# A small CPU job: TensorBoard needs no GPU. The srun step reserves the port that the
# portal forwards to; one portal URL exists per user, so run one instance at a time.
# Submitted by `client tensorboard`, which passes GRID_ROOT with --export.
#SBATCH --partition=xeon-p8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

# Profile and modules before strict mode: /etc/profile has unbound variables.
source /etc/profile
module load conda/Python-ML-2026a-pytorch
set -eo pipefail
: "${GRID_ROOT:?}"
source "$GRID_ROOT/venv/bin/activate"

srun --resv-ports=1 bash -c '
set -eo pipefail
: "${SLURM_STEP_RESV_PORTS:?no reserved port}"
name="$(id -un | tr "[:upper:]" "[:lower:]")-tensorboard"
# The forward file holds exactly one line; overwriting replaces a stale entry.
echo "http://$(hostname -s):${SLURM_STEP_RESV_PORTS}/" > "/home/gridsan/portal-url-fw/$name"
chmod u+x "/home/gridsan/portal-url-fw/$name"
echo "portal: https://$name.fn.txg-portal.llgrid.ll.mit.edu/ host=$(hostname -s) port=$SLURM_STEP_RESV_PORTS date=$(date)"
exec tensorboard --logdir "$GRID_ROOT/tb" --host "$(hostname -s)" --port "$SLURM_STEP_RESV_PORTS" --reload_interval 30
'
