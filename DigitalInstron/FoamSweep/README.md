<!--
SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
SPDX-License-Identifier: Apache-2.0
-->

# Foam sweep fit

Instron foam sweeps of two PUMA racing shoes (13 October 2026) and the material
fit of the two-face column contact law. The report is
[`fit/report.html`](fit/report.html). It is one self-contained file: open it in a
browser.

## Files

| Path | Role | Made by |
|---|---|---|
| `10_13_2026_<shoe>_.../*.steps.tracking.csv` | Raw Instron cycles, one file per shoe | Instron (checksums in `../../ASSET_PROVENANCE.md`) |
| `processed/<shoe>/` | Train (cycles 90-98) and validate (cycles 99-100) traces for each stance x force condition, plus `conditions.json` | `sweep_split.py` |
| `../<slug>/` | Column grids of each midsole, from `source.glb` (Git LFS) | `shoe_prep.py`, see `../SHOE_PREP.md` |
| `../Instron Shoe Last Size 9 6drop merged attachment 1 left.STL` | Shoe last pressed into the midsole | measured asset |
| `fit/<slug>/material_fit.json` | Fitted material (`"material"`), every start, registration, and per-condition scores | `sweep_fit.py` |
| `fit/<slug>_grounded_columns/` | Stopgap fit with every column on the platen, for comparison | earlier `sweep_fit.py` |
| `fit/history/` | Earlier fits: old law A (2 and 30 evaluations), the alpha1 <= 20 fit, and the `C2nx` fits used as fit starts | earlier `sweep_fit.py` runs |
| `fit/kernel_profile.json`, `fit/instron_fr3_dev3.gif` | Kernel timing and the viewer animation shown in the report | report tooling |
| `fit/report.html` | The report | `sweep_report.py` |

## Fitted materials

| Shoe | G1 [kPa] | alpha1 | f_eq | tau [ms] | G2 [Pa] | alpha2 |
|---|---|---|---|---|---|---|
| FR3 (`puma-fast-r-nitro-elite-3`) | 571.6 | 23.14 | 0.775 | 20.9 | 0.0207 | -15.19 |
| Dev3Elite (`puma-deviate-nitro-elite-3`) | 300.0 | 9.40 | 0.825 | 36.6 | 0.733 | -20.00 |

G1/alpha1 are the main Hyperfoam term. G2/alpha2 form a densification wall;
its onset is identified, its steepness is not. f_eq is the equilibrium fraction
and tau the Maxwell relaxation time.

## Reproduce

PowerShell, from the repository root:

```powershell
git lfs pull
uv sync --extra dev

# 1. Raw sweeps -> processed/ (deterministic)
uv run --extra dev python -m projects.digital_instron_v2.sweep_split

# 2. Optional: rebuild the column grids from the midsole scans
uv run --extra dev python -m projects.digital_instron_v2.shoe_prep `
  DigitalInstron/puma-fast-r-nitro-elite-3/source.glb DigitalInstron/puma-deviate-nitro-elite-3/source.glb `
  --out-root <scratch folder>

# 3. Fit each shoe (about 17 min each on an RTX A4000 Laptop GPU)
$starts = "--start", "DigitalInstron/FoamSweep/fit/history/puma-fast-r-nitro-elite-3_C2nx/material_fit.json", `
          "--start", "DigitalInstron/FoamSweep/fit/history/puma-deviate-nitro-elite-3_C2nx/material_fit.json"
uv run --extra dev -m projects.digital_instron_v2 fit --shoe DigitalInstron/puma-fast-r-nitro-elite-3 `
  --sweep DigitalInstron/FoamSweep/processed/FR3 --out DigitalInstron/FoamSweep/fit/puma-fast-r-nitro-elite-3 @starts
uv run --extra dev -m projects.digital_instron_v2 fit --shoe DigitalInstron/puma-deviate-nitro-elite-3 `
  --sweep DigitalInstron/FoamSweep/processed/Dev3Elite --out DigitalInstron/FoamSweep/fit/puma-deviate-nitro-elite-3 @starts

# 4. Report (about 1 min)
uv run --extra dev python -m projects.digital_instron_v2.sweep_report

# 5. Interactive viewer
uv run --extra dev -m projects.digital_instron_v2.example `
  --shoe DigitalInstron/puma-fast-r-nitro-elite-3 --shoe DigitalInstron/puma-deviate-nitro-elite-3 --slow-motion 6.7
```

Each fit runs 9 starts with the defaults (80 residual evaluations per start,
seed 0, left shoe): the calibrated material, the two `C2nx` fits, and six
Latin-hypercube starts. The fit uses the GPU and float32 fast-math kernels, so
another GPU or driver can move the result in the last digits.

Fitted with Python 3.12.14, Warp 1.18.0, NumPy 2.5.3, SciPy 1.18.1 (pinned in
`uv.lock`), Windows 11, NVIDIA RTX A4000 Laptop GPU, driver 596.71.
