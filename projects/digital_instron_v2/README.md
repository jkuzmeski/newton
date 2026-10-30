<!--
SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
SPDX-License-Identifier: Apache-2.0
-->

# Digital Instron v2

## Start here

Digital Instron prepares midsole scans, processes Instron sweeps, fits material,
builds the fit report and exports a portable artifact. Shared mechanics live in
`projects.digital_shoe`.

```bash
uv run --no-sync -m projects.digital_instron_v2 --help
uv run --no-sync -m projects.digital_instron_v2 fit --help
```

| Step | Command |
|---|---|
| Split the raw sweeps | `python -m projects.digital_instron_v2.sweep_split` |
| Prepare midsole scans | `python -m projects.digital_instron_v2.shoe_prep` |
| Fit a shoe's material | `python -m projects.digital_instron_v2 fit` |
| Build the fit report | `python -m projects.digital_instron_v2.sweep_report` |
| Export the shoe | `python -m projects.digital_instron_v2 export` |

The exported `digital_shoe.json` runs in the Instron, drop and rocker demos of
[`projects/digital_shoe`](../digital_shoe/README.md).

## Shared mechanics implementation

`projects.digital_shoe` is the home of the material and contact laws used here:

| Module | Owns |
|---|---|
| `digital_shoe.material` | Ogden–Hill pressure and Maxwell recurrence, one expression source for NumPy and Warp |
| `digital_shoe.contact` | Symmetric neighbor coupling and the two-surface column balance |
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
1.25e-4 s step) and the file records the value. `sweep_report.py` replays with
the recorded substeps, so its figures show the force curve the fit was scored on.

## Effective Poisson ratio 0

Confined and unconfined compression of racing-shoe midsole foam agree within
scatter (McCulloch, Delp and Kuhl, arXiv:2602.12694), so
`EFFECTIVE_POISSON_RATIO = 0.0` and the Ogden-Hill exponent `beta` vanishes. This
also makes the independent-column assumption self-consistent with the
constitutive law.

## Tests

```bash
uv run --extra dev -m newton.tests -k test_digital_instron_
```
