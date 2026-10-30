# Hogan variable-impedance runner

The package keeps the historical name `impedance_instron`; its current purpose
is to prepare Visual3D stance observations, fit and evaluate a causal Hogan
runner, and compare the runner across shoe materials and geometry. The active
workflow does not use a Cartesian controller or measured-trajectory
feedforward. Only the hip, knee, and ankle are actuated.

## Runner workflow

The committed report page is [`hogan/report_site/index.html`](hogan/report_site/index.html).
Rebuild it from the JSON data stored beside it, without running physics:

```console
uv run python projects/impedance_instron/hogan/report_site/_build/build.py
```

Prepare the 20 Hz F01 dataset from the Visual3D exports, the 66.5 kg inertial
profile, and a current Digital Shoe export:

```console
uv run -m projects.impedance_instron prepare --data-root data/F01 --profile projects/impedance_instron/hogan/profiles/f01_66_5kg.json --shoe DigitalInstron/digital_shoe_showcase/digital_shoe.json --virtual-foot-reference sole_markers --shoe-static-pitch-rad -0.24446041090480894 --exclude FR3_2_train_003 FR3_2_eval_001 FR3_2_train_009 --output outputs/hogan_reset/fit_dataset_f01_20hz
```

The `--exclude` IDs omit three windows whose relaxed shoe starts below ground.
The retained dataset has 98 training and 9 held-out stances. Visual3D input
fields, clocks, and frame conventions are described in
[Visual3D inputs](VISUAL3D_INPUTS.md).

Evaluate the committed slider-shoe score model:

```console
uv run -m projects.impedance_instron evaluate --dataset outputs/hogan_reset/fit_dataset_f01_20hz --model projects/impedance_instron/hogan/baselines/generative_runner_score_slider_20261029.json --mount -0.03186147427106201 0 0.10943209684347802 --speed 3.65 --dt 0.000125 --compression-limit 0.99 --force-filter-hz 20 --force-gate-n 1 --force-vibration-weight 1 --immediate-damping --output outputs/impedance_instron/hogan_score_slider_eval
```

The saved-model evaluation gives mean loss **10.537715730923786** on training
and **10.427736690837257** on held-out stances, with no rollout failures. These
are fitting metrics, not physiological validation. The full numerical setup,
model limitations, and force scoring are in
[Generative runner](hogan/GENERATIVE_RUNNER.md).

Fit from the engineering seed with a small smoke run, or choose more iterations
for a full fit:

```console
uv run -m projects.impedance_instron fit --dataset outputs/hogan_reset/fit_dataset_f01_20hz --mount -0.03186147427106201 0 0.10943209684347802 --speed 3.65 --dt 0.000125 --compression-limit 0.99 --force-filter-hz 20 --force-gate-n 1 --force-vibration-weight 1 --immediate-damping --limit-per-split 1 --iterations 1 --output outputs/impedance_instron/hogan_fit_smoke
```

Use a new output directory for each run. CUDA is the default; `--device cpu`
uses the host rigid-body integrator with the shared CUDA shoe bed. Candidate
batch size is controlled by `--chunk`. The router commands are `prepare`,
`visual3d`, `prepare-visual3d`, `inspect`, `fit`, `evaluate`, `generate`, and
`report`; run `uv run -m projects.impedance_instron --help` for the options.

## Material studies

The in-repository studies use a frozen controller to compare material response,
shoe geometry, and fitted material sensitivity. Their methods and limits are in
[Materials](hogan/MATERIALS.md). The committed report page summarizes the
reported conditions; it does not run simulations during a build.

## Package contents

- `cartesian/` retains Visual3D ingestion, observation preparation, inertial
  profile loading, and the Digital Shoe contact adapter. The directory name and
  data schema are historical; no Cartesian controller is included.
- `hogan/` contains the causal runner, CPU/CUDA rollouts, identification,
  generation, fit reports, and material studies.
