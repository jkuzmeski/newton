# Running bilateral experiments on the LLSC grid

Develop on the workstation, and run on TX-Green H100 nodes from immutable code
snapshots. The `Z:` share and the grid home are the same files
(`Z:\` is `/home/gridsan/<USER>/`). Everything lives under `newton_grid/`:

| path | contents |
|---|---|
| `code/<commit>/` | Read-only code snapshots (`newton/`, `projects/`, `pyproject.toml`), one per pushed commit |
| `data/<dataset>/` | Prepared datasets, copied once; immutable (a changed dataset needs a new name) |
| `venv/` | LLSC `conda/Python-ML-2026a-pytorch` module plus `warp-lang` matching the workstation |
| `runs/<experiment>/<job>/<index>_<name>/` | Training outputs, with `<index>_<name>.run.json` beside each |
| `tb/<experiment>/<job>/<index>_<name>/` | TensorBoard scalars |
| `logs/<experiment>-<job>_<task>.out` | SLURM task logs |
| `warp_cache/` | Shared Warp kernel cache |
| `submissions.jsonl` | One record per submission: job, runs, code commit |

## Workflow

Run these from the repository root on the workstation:

```console
uv run --no-sync python -m projects.impedance_instron.bilateral.grid.client setup
uv run --no-sync python -m projects.impedance_instron.bilateral.grid.client list
uv run --no-sync python -m projects.impedance_instron.bilateral.grid.client submit pm_ablation --indices 0 --smoke --time 00:30:00
uv run --no-sync python -m projects.impedance_instron.bilateral.grid.client submit pm_ablation --time 02:00:00
uv run --no-sync python -m projects.impedance_instron.bilateral.grid.client status <job>
uv run --no-sync python -m projects.impedance_instron.bilateral.grid.client tail <job> --task 1
uv run --no-sync python -m projects.impedance_instron.bilateral.grid.client load
```

- **Experiments** are flat, indexed run lists in [`experiments.py`](experiments.py);
  one SLURM array index is one run. To add an experiment, add an entry there.
- **Pushing code.** Every `submit` (or `push`) snapshots the working tree,
  including uncommitted and untracked files, as a commit kept under
  `refs/grid/<commit>` on the workstation. The snapshot is extracted once to
  `code/<commit>/`. Dev pushes never change the code of queued or running
  jobs, and every run records the exact commit it ran.
- **Testing dev code.** Edit locally, then `submit ... --smoke` to run the
  short smoke arguments of a run. Once it passes, submit the full experiment,
  preferably from a commit so the label carries no `+dev` suffix.
- **Reusing a snapshot.** `submit ... --code <12-character commit>` reuses an
  existing snapshot without pushing.
- **Job script.** [`job.sbatch`](job.sbatch) uses one H100 per task, stages
  the run's dataset to the node-local `$TMPDIR`, and runs
  [`run.py`](run.py).
- **Results.** Outputs are readable on `Z:` from the workstation, so
  evaluation tools such as `evaluate-bilateral` run locally against
  `Z:\newton_grid\runs\...`.

## Guard rails (from the LLSC skill)

- **Size.** `submit` checks `LLfree` first. It refuses more than 4 GPUs or more
  than 8 h per task unless you pass `--allow-large` after confirming with the
  team.
- **Cancelling.** `cancel` cancels only jobs that this client submitted, by
  exact job ID. Never use `scancel -u`.
- **Monitoring.** Check status on demand; do not poll in loops. Size
  `--time` at about 1.5 times the measured runtime.

## Measured

- **2026-10-14, smoke job 16683440 (`pm_ablation` index 0).** H100 NVL,
  driver 580, CUDA 13.0, Warp 1.18.0, torch 2.9.1. 1,024 worlds ran at about
  85k samples/s after warm-up, with the environment taking 83% of each
  iteration. Peak memory was 2.4 GB; the task took 30 s for 5 iterations.
