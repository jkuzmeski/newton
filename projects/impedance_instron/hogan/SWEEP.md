# Hogan impedance tuning sweep

## Match the fitting objective to the common score

Identification defaults to `--lm-objective sample`, fitting coordinate, GRF,
and vibration mean squares plus coefficient-offset regularization. Use
`--lm-objective score` to additionally fit the existing common score's peak
force, impulse, contact-duration, and torque-effort terms. This changes the
optimizer's residuals; it preserves the reported score and its weights. Failed
rollouts remain ineligible rather than supplying incomplete residuals.

For a controlled comparison, initialize both objectives from the same successful
runner, keep the dataset and physics fixed, and use the same stricter
`--lm-tolerance 1e-7` and iteration ceiling. Regularization remains a separate
penalty relative to each fit's initialization. A restarted fit also restarts
damping and coefficient-offset bounds, so it is not an uninterrupted
continuation of the previous optimizer. Contact duration uses a force threshold
and is nonsmooth; matching the score does not guarantee convergence or a global
minimum. Choose between the frozen baseline and fitted models on common
training loss with zero training failures; report held-out metrics separately.

For independent full runs from scratch, construct each initial model directly
from `Runner.seed` with the original engineering feature weights. Initialize
both objective conditions from that unfitted model, rather than a selected
checkpoint. Use `--iterations 50 --lm-tolerance 0` to run the complete iteration
budget without small-improvement early stopping. Keep failed engineering starts
in the record; a fit requires its initial model to complete all training trials.

## Independent full fits from scratch

Use this driver when every sweep candidate must start from unfitted engineering
parameters. It does not accept or load learned checkpoints:

```bash
uv run --no-sync -m projects.impedance_instron.hogan.scratch_search \
  --dataset outputs/impedance_instron/hogan_scratch_20261009/dataset \
  --iterations 50 --budget-minutes 80 \
  --output outputs/impedance_instron/hogan_independent_scratch --run
```

Every candidate is built directly from `Runner.seed`, with the original
engineering feature weights. The 16 configurations vary initial frequency
(1.5 or 2 Hz), initial torque response time (15 or 35 ms), lagged/immediate
damping, and regularization (0 or 0.001). Each fits all 119 active parameters
jointly for up to 50 iterations, stopping at native convergence. Coefficient
offsets are bounded to ±3 relative to each unfitted initialization. Engineering
output and torque bounds, shoe, body profile, and data processing stay fixed.
There is no refinement initialized from a different completed fit.

The unfitted engineering baseline is evaluated separately. Selection uses the
lowest finite common training score with zero training failures, and held-out
scores are reporting only. Failed raw starts are retained in the logs; they are
never silently replaced with fitted weights. Jobs are serial, and new jobs stop
launching at 85% of the budget to reserve export/reporting time. A running job
can cross that boundary; `coarse_grid_complete` records whether all configurations
were attempted. The saved native traces need verified contact replay exports
and report packaging before the experiment is report-ready.

LM optimizes coordinate, GRF and vibration residuals plus regularization. The
common ranking score additionally includes peak force, impulse, contact timing
and effort, so the optimized residual objective and reported score differ.

## Warm-start search for the lowest common loss

For a search initialized from two fitted checkpoints, use the search driver. Every candidate fits all
119 active single-speed parameters jointly (117 feature coefficients, stride
frequency, and torque response time). The outer grid varies the starting model,
lagged versus immediate damping, initial response-time multiplier (0.5, 1, 2),
and LM offset regularization (0.001, 0.01). Response initialization stays within
the existing identification domain of 0.005–0.15 s. Force scoring remains fixed
at 20 Hz and vibration weight 1, so reducing a loss weight cannot win the search.

```bash
uv run --no-sync -m projects.impedance_instron.hogan.search \
  --dataset outputs/impedance_instron/hogan_scratch_20261009/dataset \
  --models outputs/impedance_instron/hogan_scratch_20261009/seed_fit/runner.json \
    outputs/impedance_instron/hogan_scratch_20261009/full/fit_vibration_1/runner.json \
  --iterations 12 --finalists 3 --refine-iterations 40 --budget-minutes 120 \
  --output outputs/impedance_instron/hogan_parameter_search --run
```

Omit `--run` to save a plan without launching fits. Two input models produce 24
coarse configurations. The strongest three completed candidates receive longer
refinement fits. Frozen input models remain eligible, so a search that fails to
improve them preserves the better baseline. Selection uses finite common
training loss with zero training rollout failures; held-out results are reported
separately and never select a checkpoint. Failed configurations remain in the
record, and the driver continues with the remaining candidates.

The budget reserves time for refinement and evidence capture. Jobs run serially;
the driver checks time between jobs, so a running fit can cross a stage boundary.
`best.json` contains the best frozen model found; `summary.json` records its
training loss, source, and selection rule.
This is a finite multi-start search, not a guarantee of the global minimum.

## Pilot vibration-weight comparison

From the Newton repository root, first create a plan in a new output directory:

```bash
uv run --no-sync -m projects.impedance_instron.hogan.sweep \
  --dataset PATH_TO_DATASET --output outputs/impedance_instron/hogan_pilot
```

Planning is the default and does not start simulation work. It writes `experiment-plan.json`, `commands.json` with reviewable argv arrays, `experiment-artifacts.json` with honest missing roles, and `summary.json`. The dataset must be a directory containing `manifest.json`; a frozen runner model must also exist. Missing inputs are recorded during planning and prevent `--run`.

After reviewing the plan and when those inputs are ready, use a **new** output directory and add `--run`:

```bash
uv run --no-sync -m projects.impedance_instron.hogan.sweep \
  --dataset PATH_TO_DATASET --model PATH_TO_RUNNER_JSON \
  --output outputs/impedance_instron/hogan_pilot_run --run
```

The default pilot performs three serial fits with force-vibration weights 0.5, 1, and 2, three LM iterations each. Every fit uses the same force filter (20 Hz), immediate damping, mount `[-0.03186147427106201, 0, 0.10943209684347802]` m, task speed 3.65 m/s, and compression limit 0.99. It also evaluates the frozen baseline and evaluates every fitted runner again at common force-vibration weight 1. The latter provides comparable reporting metrics; each fit's own objective and train/eval metrics remain separately recorded. Evaluation metrics do not select a candidate. Selection stays undecided, and held-out evaluation is reporting only.

Use `--iterations 15` for the full fitting budget after reviewing the pilot. `--device` defaults to `cuda:0`; `--chunk` defaults to 128. Outputs are never overwritten. Run summaries preserve command argv, output paths, stdout/stderr paths, return codes, timestamps, and failures. The sweep creates no report-readiness claim: replay-aligned per-column deformation capture and its validation are explicitly pending, along with the other evidence listed as missing in the artifact inventory.

To fit the controller from scratch, replace `--model` with `--from-scratch`:

```bash
uv run --no-sync -m projects.impedance_instron.hogan.sweep \
  --dataset PATH_TO_FRESH_DATASET --from-scratch --iterations 15 \
  --output outputs/impedance_instron/hogan_scratch_sweep --run
```

This first fits `Runner.seed` with lagged damping for 15 iterations, then initializes every immediate-damping candidate from that same completed model. The unfitted seed can fail with immediate damping, so the seed stage retains lagged damping. Input compatibility and rollout checks remain active. Dataset and shoe preparation are separate inputs; use the preparation recipe in the [pipeline guide](../README.md) and record the actual local shoe/profile identities when reproducing a run on another computer.
