<!--
SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
SPDX-License-Identifier: Apache-2.0
-->

# Foam sweep fit

The raw Instron sweeps are split into train (cycles 90–98), validation (99–100),
and repeat traces by `sweep_split.py`. Fits replay every trace through the
shared two-surface column bed; `sweep_report.py` writes the self-contained HTML
report at `fit/report.html`.

## Files

- `10_13_2026_<shoe>_.../*.steps.tracking.csv`: raw Instron cycles.
- `processed/<shoe>/`: phase-averaged traces and `conditions.json`.
- `fit/<slug>/material_fit.json`: fitted material, starts, registration and
  per-condition scores.
- `fit/<slug>/registration.png`: last-to-midsole registration QC.
- `fit/report.html`: fit report.

## Reproduce

From the repository root:

```powershell
uv run --extra dev -m projects.digital_instron_v2.sweep_split
uv run --extra dev -m projects.digital_instron_v2 fit `
  --shoe DigitalInstron/puma-fast-r-nitro-elite-3 `
  --sweep DigitalInstron/FoamSweep/processed/FR3 `
  --out DigitalInstron/FoamSweep/fit/puma-fast-r-nitro-elite-3
uv run --extra dev -m projects.digital_instron_v2.sweep_report
```

Fit starts are the calibrated material and six seeded Latin-hypercube samples
(default seed 0). The number of random starts, evaluations and contact
substeps can be changed with fit options. For midsole preparation and artifact
export, see `projects/digital_instron_v2/README.md`.
