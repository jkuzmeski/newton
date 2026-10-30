<!--
SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
SPDX-License-Identifier: Apache-2.0
-->

# Foam sweep fit

`sweep_split.py` divides the raw Instron sweeps into train (cycles 90–98),
validation (99–100), and repeat traces. Fits replay each trace through the
shared two-surface column bed. `sweep_report.py` builds `fit/report.html`.

## Files

- `10_13_2026_<shoe>_.../*.steps.tracking.csv`: raw Instron cycles.
- `processed/<shoe>/`: phase-averaged traces and `conditions.json`.
- `fit/<slug>/material_fit.json`: fitted material, starts, registration and
  per-condition scores.
- `fit/<slug>/registration.png`: last-to-midsole registration QC.
- `fit/report.html`: fit report.

## Reproduce

From the repository root, split the raw traces and rebuild the report:

```powershell
uv run --extra dev -m projects.digital_instron_v2.sweep_split
uv run --extra dev -m projects.digital_instron_v2.sweep_report
```

The committed FR3 and Dev3Elite fits ran on LLSC (jobs 16844885 and 16846016,
respectively) from seven starts recorded in each fit's `optimization.starts`. To rerun a fit
from those starts, pass its fit file:

```powershell
uv run --extra dev -m projects.digital_instron_v2 fit `
  --shoe DigitalInstron/puma-fast-r-nitro-elite-3 `
  --sweep DigitalInstron/FoamSweep/processed/FR3 `
  --out <output folder> `
  --starts-from DigitalInstron/FoamSweep/fit/puma-fast-r-nitro-elite-3/material_fit.json
```

A fresh fit without `--starts-from` uses the calibrated material plus six
seeded Latin-hypercube starts (seed 0).
For shoe preparation and artifact export, see
`projects/digital_instron_v2/README.md`.
