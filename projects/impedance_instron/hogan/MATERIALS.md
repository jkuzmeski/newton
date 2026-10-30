# Frozen-controller material studies

The material drivers compare a fixed Hogan runner on matched shoe materials and
geometries. These are engineering simulations on prepared Visual3D stances, not
metabolic-energy estimates or proof of footwear performance. The controller is
not refitted between conditions.

## Paper-derived foam comparison

```console
uv run -m projects.impedance_instron.hogan.materials --dataset PATH_TO_DATASET --output outputs/impedance_instron/materials_paper --run
```

The driver writes a plan without `--run` and evaluates all conditions with it.
It uses the slider FR3 score model by default. Change it with `--model` to
compare another supported schema-2 runner. The paper-derived foam response is a
reduced Ogden-Hill compression approximation with an independent equilibrium
shear override; it does not reproduce the source paper's full 3D constitutive
law or measured hysteresis.

## Shoe geometry comparison

```console
uv run -m projects.impedance_instron.hogan.materials_geometry --dataset PATH_TO_DATASET --output outputs/impedance_instron/materials_geometry --run
```

This evaluates the declared materials with the original bed geometry, a 25%
longer-column geometry, and a flat 40 mm block. Initial ankle-to-ground clearance
is matched by a documented vertical translation; velocities, phase, runner,
and contact settings stay fixed. Shoe mass is not added to the runner.

## Sensitivity of the fitted shoe

```console
uv run -m projects.impedance_instron.hogan.materials_sensitivity --dataset PATH_TO_DATASET --model RUNNER_JSON --output outputs/impedance_instron/materials_sensitivity --run --instron SHOE_FOLDER MATERIAL_FIT_JSON SWEEP_FOLDER LAST_STL
```

This changes one material property at a time with the runner frozen. `--combine`
adds combinations and `--variants` limits the selected variants. `--instron`
replays the held-out bench sweep using the exporter's validation replay.

All studies save their plan, input hashes, conditions, trial traces, scenarios,
and metrics. Interpret held-out mechanical work as finite-window model output,
not running economy. Retain failures and out-of-support strain in any comparison.
The committed summary page is [`report_site/index.html`](report_site/index.html);
rebuild it from its JSON inputs with:

```console
uv run python projects/impedance_instron/hogan/report_site/_build/build.py
```
