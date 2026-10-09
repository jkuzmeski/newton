# Hogan variable-impedance runner

One pipeline: prepare observations, fit a shared generative runner with
Levenberg-Marquardt, and predict from its frozen model. No measured trajectory,
force target, or inverse-dynamics feedforward enters the simulation. Only the
hip, knee, and ankle are actuated.

## Preserved full-run baseline

- Model: [F01 runner](hogan/baselines/generative_runner_f01_20261008.json).
- Saved run: `outputs/impedance_instron/generative_fit_lm_flightcom_20261008`.
- Dataset: `outputs/impedance_instron/generative_fit_dataset_flight_20261008`.
- Subject: 66 kg; belt: 3.65 m/s; 98 training and 9 held-out stances.
- Mean loss: **16.079667862151016 train**, **14.796461691170371 held-out**.

The saved model, scenarios, traces, summaries, and report are not rewritten by
cleanup. Local motion and shoe assets are required; see
[asset provenance](../../ASSET_PROVENANCE.md).

## Run

From the repository root, inspect the available commands:

```console
uv run --no-sync -m projects.impedance_instron --help
```

Reproduce the latest fit from its original seed, including the HTML report:

```console
uv run --no-sync -m projects.impedance_instron fit --dataset outputs/impedance_instron/generative_fit_dataset_flight_20261008 --mount -0.03186147427106201 0 0.10943209684347802 --speed 3.65 --compression-limit 0.99 --iterations 15 --output outputs/impedance_instron/hogan_fit
```

Choose a new output directory. CUDA is the default; `--device cpu` selects the
reference backend. Use `--limit-per-split 1 --iterations 1` for a small fit.
The full baseline fit originally took about 82 minutes on an RTX A4000 Laptop
GPU; the batched CUDA runtime now takes about 15 s per LM iteration. Its finite
differences use fast-math shoe kernels while costs and accepted steps stay
exact; `--exact-jacobian` replays the original rollouts exactly at about 20 s
per iteration. `--chunk` (default 128) sets candidates integrated
concurrently; 128 candidates over 98 stances need about 2.5 GB of GPU memory,
so lower it on smaller GPUs. It changes speed, not results.

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

## Code and checks

- [Model and method](hogan/GENERATIVE_RUNNER.md): initialization, loss, baseline results, and limitations.
- [Visual3D inputs](VISUAL3D_INPUTS.md): clocks, reference frames, and data preparation.
- `hogan/`: CPU/CUDA dynamics, LM identification, frozen generation, and reporting.
- `cartesian/`: retained measurement/profile preparation and shoe attachment;
  `cartesian/gpu/foundation.py` is the shared batched shoe adapter, not another runner model.

The retired Cartesian equilibrium searches, reference-tracking Hogan schedules,
CEM optimizer, quick-fit/recovery experiments, old controller bundles, and their
reports are removed. Shared Digital Shoe material/contact laws and Newton's
public controllers are unchanged.

Run the focused regression checks, including an exact replay of all 107 saved
full-run traces when the local dataset and CUDA are available:

```console
uv run --no-sync -m unittest newton.tests.test_impedance_pipeline newton.tests.test_impedance_profile newton.tests.test_impedance_hogan newton.tests.test_impedance_runner newton.tests.test_impedance_runner_gpu newton.tests.test_impedance_least_squares newton.tests.test_impedance_gpu_workflow
```
