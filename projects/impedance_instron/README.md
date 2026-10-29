# Hogan variable-impedance runner

One pipeline: prepare observations, fit a shared generative runner with
Levenberg-Marquardt, and predict from its frozen model. No measured trajectory,
force target, or inverse-dynamics feedforward enters the simulation. Only the
hip, knee, and ankle are actuated.

## Experiment reports

[`hogan/report_site/index.html`](hogan/report_site/index.html) is the committed
summary of the runner on the slider FR3 shoe: shoe and Instron fit, both
controllers, spring geometry, material sensitivity and the Dev3 check. It is one
self-contained page. Rebuild it from the committed JSON with
`uv run python projects/impedance_instron/hogan/report_site/_build/build.py`.

Open `outputs/index.html` for the results dashboard, with separate
`outputs/controller-testing/index.html` and `outputs/materials-testing/index.html`
catalogs. Each study links to its individual report and available saved motion.
Material-condition pages show the geometry and results for that specific run.
The dashboard keeps controller fitting scores separate from held-out mechanical
work and conditional marathon estimates; missing results remain marked.

Rebuild the dashboard and catalogs from saved artifacts without running physics:

```console
uv run --no-project python -m projects.impedance_instron.report_catalog --outputs outputs
```

Upgrade a qualified replay with the experiment's actual spring geometry:

```console
uv run --no-project python -m projects.impedance_instron.hogan.spring_viewer_report CONDITION_DIRECTORY
```

This reconstructs display endpoints from saved poses, shoe geometry, and captured
compression without integrating physics. It retains the original viewer and
records the display provenance in `replay_evidence/spring-geometry.json`.
Material report generators run this step before packaging their viewers.

The motion pane offers a shoe/run selector and a purple dashed comparison
overlay for the same saved example. Drag the 3D spring close-up to orbit, use
Side/Top/End presets, and zoom with the wheel or buttons. Camera position stays
fixed through playback and scrubbing. Overlays match elapsed time and center
the spring geometry on each shoe; they do not alter the simulation or metrics.
Build the saved comparison selector with:

```console
uv run --no-project python -m projects.impedance_instron.hogan.comparison_viewer_report --run RUN_DIRECTORY --default CONDITION_ID
```

The material report generators call the Instron report renderer, an external
Codex skill at `$CODEX_HOME/skills/instron-reports` that is not part of this
repository; without it they write `report-spec.json` and then fail. The renderer
dispatches catalog rebuilding to this repository generator. Existing report URLs and experiment artifacts are retained.

## Preserved full-run baseline

The current controller is the
[slider-shoe score runner](hogan/baselines/generative_runner_score_slider_20261029.json), with
[saved fit provenance](hogan/baselines/generative_runner_score_slider_20261029.provenance.json).
It was fitted on the committed slider-law FR3 shoe from the unfitted `Runner.seed`
engineering weights for all 50 LM iterations, on 98 training and 9 held-out
stances of `outputs/hogan_reset/fit_dataset_f01_20hz`:

```console
uv run --no-sync -m projects.impedance_instron fit --dataset outputs/hogan_reset/fit_dataset_f01_20hz --mount -0.03186147427106201 0 0.10943209684347802 --speed 3.65 --dt 0.000125 --compression-limit 0.99 --force-filter-hz 20 --force-gate-n 1 --force-vibration-weight 1 --iterations 50 --lm-step 0.01 --lm-damping 0.01 --lm-bound 3 --lm-regularization 0.001 --lm-tolerance 0 --chunk 128 --immediate-damping --lm-objective score --output outputs/impedance_instron/hogan_score
```

Its common score is **10.5377 train** and **10.4277 held-out**, with zero
rollout failures. Held-out results did not select the controller. Check it
without refitting:

```console
uv run --no-sync -m projects.impedance_instron evaluate --dataset outputs/hogan_reset/fit_dataset_f01_20hz --model projects/impedance_instron/hogan/baselines/generative_runner_score_slider_20261029.json --mount -0.03186147427106201 0 0.10943209684347802 --speed 3.65 --dt 0.000125 --compression-limit 0.99 --force-filter-hz 20 --force-gate-n 1 --force-vibration-weight 1 --immediate-damping --output outputs/impedance_instron/hogan_score_slider_eval
```

The paired sample-objective fit scores 11.894 train and 11.447 held-out. The
largest filtered vertical force is within about 22 N on held-out stances, but
the whole-stance GRF error stays near 93 N (Fx) and 118 N (Fz) RMSE: the first
vertical peak overshoots by about 359 N raw and contact starts about 22 ms
early. Horizontal hip error stays near 45 mm.

The earlier controllers remain as records:

- [2026-10-29 column-bed score runner](hogan/baselines/generative_runner_score_20261029.json)
  ([provenance](hogan/baselines/generative_runner_score_20261029.provenance.json)):
  10.563 train and 10.549 held-out on the earlier column-bed FR3 export
  (sha256 `a9bf61dd...`), which is not committed. Frozen on the slider shoe it
  scores 12.066 and 12.758. `test_impedance_pipeline` replays all 107 saved
  traces of its run exactly when the run is at
  `outputs/hogan_reset/fit_score_20261029` and CUDA is available.

The two below were fitted under shoe laws that predate the column bed, and
their saved datasets no longer load:

- [2026-10-10 score runner](hogan/baselines/generative_runner_score_20261010.json)
  ([provenance](hogan/baselines/generative_runner_score_20261010.provenance.json)):
  10.309 train and 10.162 held-out under its shoe law. Frozen on the column
  bed it scores 11.954 and 11.350, and it ranks the paper foams opposite to both
  controllers fitted on the column bed, so do not use it for material studies.
- [2026-10-08 F01 runner](hogan/baselines/generative_runner_f01_20261008.json):
  16.080 train and 14.796 held-out, fitted to the 6 Hz, 66 kg F01 exports.

Local motion and shoe assets are required; see
[asset provenance](../../ASSET_PROVENANCE.md).


## Run

From the repository root, inspect the available commands:

```console
uv run --no-sync -m projects.impedance_instron --help
```

Reproduce the latest fit from its original seed, including the HTML report:

```console
uv run --no-sync -m projects.impedance_instron fit --dataset outputs/hogan_reset/fit_dataset_f01_20hz --mount -0.03186147427106201 0 0.10943209684347802 --speed 3.65 --compression-limit 0.99 --iterations 15 --output outputs/impedance_instron/hogan_fit
```

Choose a new output directory. CUDA is the default; `--device cpu` selects the
reference backend, which integrates the body on the host and steps the CUDA
column bed once per step. Use `--limit-per-split 1 --iterations 1` for a small
fit. Baselines fitted before 2026-10-29 ran with earlier shoe laws; their traces
and costs are not reproduced on the two-surface column bed.
The bed advances every world's bed one step from the carrier's start-of-step
pose and writes the carrier wrench; the caller applies that wrench to the body.
`--chunk` (default 128) sets candidates integrated concurrently; 128 candidates
over 98 stances need about 2.5 GB of GPU memory,
so lower it on smaller GPUs. It changes speed, not results.

The F01 force exports are now low-passed at 20 Hz, the running convention, and
use the measured 66.5 kg body mass. New fits should score the simulated force
the way the target was filtered and let the impedance damping act without the
torque lag. See
[GRF targets and force ripple](hogan/GENERATIVE_RUNNER.md#grf-targets-and-force-ripple-2026-10-09):

```console
uv run --no-sync -m projects.impedance_instron fit --dataset outputs/hogan_reset/fit_dataset_f01_20hz --mount -0.03186147427106201 0 0.10943209684347802 --speed 3.65 --compression-limit 0.99 --iterations 15 --model projects/impedance_instron/hogan/baselines/generative_runner_f01_20261008.json --force-filter-hz 20 --immediate-damping --output outputs/impedance_instron/hogan_fit_20hz
```

That dataset repeats the 2026-10-08 recipe on the new exports:

- `prepare --data-root data/F01 --profile projects/impedance_instron/hogan/profiles/f01_66_5kg.json --shoe DigitalInstron/digital_shoe_showcase/digital_shoe.json --virtual-foot-reference sole_markers --shoe-static-pitch-rad -0.24446041090480894 --exclude FR3_2_train_003 FR3_2_eval_001 FR3_2_train_009 --output outputs/hogan_reset/fit_dataset_f01_20hz`.
- Pass `--profile` and `--shoe` explicitly; no prepared bundle is preserved to
  supply defaults. The shoe is the FR3 `digital_shoe.json` written by
  `python -m projects.digital_instron_v2.export_digital_shoe`. Datasets whose
  saved shoe predates the column bed no longer load; pass `--shoe` with a newly
  exported shoe to `fit`, `inspect`, or `evaluate` to replace every member's
  shoe.
- The leg profile is mass-scaled to 66.5 kg; the 66.5 kg profile is a
  reconstruction committed at `projects/impedance_instron/hogan/profiles/f01_66_5kg.json`.
- The committed shoe is the slider-law FR3 export that the current baseline
  was fitted on.
- The seeded split selects 110 stances. `--exclude` leaves out the three whose
  relaxed shoe starts below ground, leaving 98 training and 9 held-out stances;
  the other IDs keep their numbering. Without it, `inspect` reports the
  clearance of every stance.

For a small balanced motion/GRF tuning pilot, see the
[Hogan sweep setup](hogan/SWEEP.md). It saves a plan before running candidates
and evaluates every fitted model with the same force-scoring settings.

Leg profiles need only `masses_kg` (3), `com_local_m` (3 by 2),
`inertias_kg_m2` (3), and `provenance.inertial`. Old gain/limit fields are
accepted when loading historical files but discarded; they never set Hogan
impedance. Newly prepared profiles contain only inertial inputs.

| Command | Purpose |
|---|---|
| `prepare` | Build peak-to-peak observations from Visual3D exports |
| `inspect` | Check a dataset's initialization and ankle/shoe compatibility |
| `fit` | Fit on training stances, evaluate held-out stances, write `report.html` |
| `evaluate` | Score a frozen model on observations without refitting |
| `generate` | Predict from a frozen model and exported scenario, without observations |
| `report` | Rebuild an existing LM fit report |
| `visual3d`, `prepare-visual3d` | Inspect/import exports and prepare an individual reference |

Use `<command> --help` for its arguments. The module-specific Hogan commands
remain available; `python -m projects.impedance_instron.hogan` uses the same router.

Regenerate one saved prediction without the measurement dataset:

```console
uv run --no-sync -m projects.impedance_instron generate --model projects/impedance_instron/hogan/baselines/generative_runner_f01_20261008.json --scenario outputs/impedance_instron/generative_fit_lm_flightcom_20261008/scenario_000.json --output outputs/impedance_instron/hogan_generated
```

That run's saved scenarios embed a shoe artifact that predates the column bed;
it no longer loads and `generate` takes no `--shoe` override, so the command
above is a historical record. Regenerate predictions from a scenario written
by a current fit.

## Code and checks

- [Model and method](hogan/GENERATIVE_RUNNER.md): initialization, loss, baseline results, and limitations.
- [Visual3D inputs](VISUAL3D_INPUTS.md): clocks, reference frames, and data preparation.
- `hogan/`: CPU/CUDA dynamics, LM identification, frozen generation, and reporting.
- `cartesian/`: retained measurement/profile preparation and shoe attachment to
  the shared CUDA column bed (`projects/digital_shoe/column_bed.py`).

The retired Cartesian equilibrium searches, reference-tracking Hogan schedules,
CEM optimizer, quick-fit/recovery experiments, old controller bundles, and their
reports are removed. Shared Digital Shoe material/contact laws and Newton's
public controllers are unchanged.

Run the focused regression checks. `test_impedance_pipeline` also replays a
saved local full run when its artifacts and CUDA are available; the retained
2026-10-08 run predates the column-bed shoe, so its dataset no longer loads
and an exact replay of its 107 traces is not possible:

```console
uv run --no-sync -m unittest newton.tests.test_impedance_pipeline newton.tests.test_impedance_profile newton.tests.test_impedance_hogan newton.tests.test_impedance_runner newton.tests.test_impedance_runner_gpu newton.tests.test_impedance_least_squares newton.tests.test_impedance_gpu_workflow
```

For frozen-controller comparisons of paper-derived foam properties, see
[Materials testing](hogan/MATERIALS.md). These reports compare matched joint
work, power, and contact performance, with separate material-fit diagnostics.
