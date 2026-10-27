<!--
SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
SPDX-License-Identifier: Apache-2.0
-->

# Digital Instron v2

## Start here

Digital Instron owns shoe preparation, sweep fitting, the Instron scenes and
export. It calls the shared laws in Digital Shoe; it does not maintain another
material or contact implementation. See the [three-project map](../README.md)
for file roles and the difference between artifact flow and shared-code
dependencies.

```bash
uv run --no-sync -m projects.digital_instron_v2 --help
uv run --no-sync -m projects.digital_instron_v2 fit --help
```

| Task | Package command | Module |
|---|---|---|
| Fit one shoe's material to its foam sweep | `fit` | `sweep_fit.py` |
| Portable artifact production | `export` | `export_digital_shoe.py` |
| Mechanical demonstrations | `view` | `example.py` |

> **Standalone presentation:** Use `projects.digital_instron_v2.export_digital_shoe`
> to produce a portable `digital_shoe.json`, then run the artifact-only Virtual
> Instron, free-body drop, and rocker examples in `projects/digital_shoe`. See
> `projects/digital_shoe/README.md`. These examples do not use gait or a human model.

## Shared mechanics implementation

`projects.digital_shoe` is the home of the material and contact laws used here:

| Module | Owns |
|---|---|
| `digital_shoe.material` | Ogden–Hill pressure and Maxwell recurrence, one expression source for NumPy and Warp |
| `digital_shoe.contact` | Symmetric neighbor coupling, the two-surface column balance and the bristle law (used by the friction example, not by the column bed) |
| `digital_shoe.column_bed` | The batched GPU column bed |
| `digital_shoe.runtime` | The material block and `FoundationConfig` |
| `digital_shoe.rendering` | Column drawing from solved bottom and top endpoints |

Each column has two surfaces. Its bottom lands on the ground plane or is fixed to
the platen. Its top is bonded to the carrier, pressed by an indenter, or free.
The same column bed serves the Instron bench and a carried shoe; only the top and
bottom modes differ.

The audited upper-last seating limitation remains: a projected support footprint
is not a solved last/insole contact interface.

## Sweep fit (`sweep_fit.py`)

Fit one shoe's material to its Instron foam sweep. Each sweep condition (stance
time x force target) is one full-foot last trace on the shoe's own column grid.
Every trace is replayed through the column bed (`last_replay.py`), one world per
trace and material. Train traces drive a multi-start least-squares fit; validate
and repeat traces are only scored.

```bash
uv run --extra dev -m projects.digital_instron_v2 fit \
  --shoe DigitalInstron/<slug> --sweep <processed sweep folder> --out <output folder>
```

The fit writes `material_fit.json`; its `"material"` block is the fitted material.
Every trace is replayed in `substeps` contact steps (default 8, the runner's
1.25e-4 s step) and the file records the value. `example.py --shoe` and
`sweep_report.py` replay with the recorded `substeps`, so they show the force curve
the fit was scored on.

## Dynamic midsole example (`example.py`)

Four scenes run the column bed in live Newton rigid-body physics:

```bash
# Displacement-controlled digital Instron: the last presses the column tops
# against the platen; record the force and hysteresis loop.
uv run -m projects.digital_instron_v2.example --mode instron

# Free, massive midsole resting on the ground; a sub-cone lateral load is held
# by Coulomb friction between the columns' bottoms and the ground.
uv run -m projects.digital_instron_v2.example --mode settle

# Prescribed carrier pose carries the bonded bed over the ground; record the
# ground reaction and centre of pressure.
uv run -m projects.digital_instron_v2.example --mode stride

# Foot-mounted shoe with mass and inertia, held to the foot by a damped
# attachment; record the stance/flight ground reaction.
uv run -m projects.digital_instron_v2.example --mode attached
```

The `settle` and `attached` carriers are free rigid bodies; the `instron` and `stride`
carriers follow a prescribed pose. In every mode the bed applies its explicit forces to
the carrier. The `instron` mode runs the bed with 64 sweeps; the other modes use 16.

Add `--fit PATH` to use the `"material"` block of a `material_fit.json`. Without
it the scenes use `core.CALIBRATED_MATERIAL`, which was calibrated for the earlier
one-unknown-per-column law. Its Instron peak is therefore not a validated
prediction for the column bed.

Add `--viewer null --num-frames N --test` to run headlessly and audit the
recorded response, or `--viewer gl` for the interactive viewer. Pass `--eager`
to disable CUDA graph capture for debugging.

## Effective Poisson ratio 0

Confined and unconfined compression of racing-shoe midsole foam agree within
scatter (McCulloch, Delp and Kuhl, arXiv:2602.12694), so
`EFFECTIVE_POISSON_RATIO = 0.0` and the Ogden-Hill exponent `beta` vanishes. This
also makes the independent-column assumption self-consistent with the
constitutive law.

## Scope on main

The standalone shoe and fitting tools do not depend on an OpenSim runtime.
The legacy jumping-leg crossover and human-shoe experiments remain on
`jkuzmeski/digital-shoe-standalone`; they are not part of this integration.

## Tests

```bash
uv run --extra dev -m unittest newton.tests.test_digital_instron_core
uv run --extra dev -m unittest newton.tests.test_digital_instron_project
uv run --extra dev -m unittest newton.tests.test_digital_instron_dynamics
uv run --extra dev -m unittest newton.tests.test_digital_instron_sweep_fit
```

`test_digital_instron_dynamics` checks the column-bed geometry, the zero-Poisson
Hyperfoam law, and that each example scene passes its physical audit.
