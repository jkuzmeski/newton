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
| `fit/<slug>/material_fit.json` | Fitted material (`"material"`), every start, registration, and per-condition scores | Dev3: `sweep_fit.py`. FR3: a grid driver, see [Provenance of the committed fits](#provenance-of-the-committed-fits) |
| `fit/<slug>_grounded_columns/` | Stopgap fit with every column on the platen, for comparison | earlier `sweep_fit.py` |
| `fit/history/` | Earlier fits: old law A (2 and 30 evaluations), the alpha1 <= 20 fit, and the `C2nx` fits used as fit starts | earlier `sweep_fit.py` runs |
| `fit/kernel_profile.json`, `fit/instron_fr3_dev3.gif` | Kernel timing and the viewer animation shown in the report | report tooling |
| `fit/report.html` | The report. **Stale:** built from the previous fits (see below) | `sweep_report.py` |

## Fitted materials

| Shoe | G1 [kPa] | alpha1 | f_eq | tau [ms] | G2 [Pa] | alpha2 |
|---|---|---|---|---|---|---|
| FR3 (`puma-fast-r-nitro-elite-3`) | 2165.9 | **40.0** | 0.587 | **5.0** | 3.69 | **-20.0** |
| Dev3Elite (`puma-deviate-nitro-elite-3`) | 312.2 | 7.61 | 0.843 | 36.2 | 12.09 | -14.24 |

G1/alpha1 are the main Hyperfoam term. G2/alpha2 form a densification wall;
its onset is identified, its steepness is not. f_eq is the equilibrium fraction
and tau the Maxwell relaxation time.

Held out, each fit passes 8 of its 24 gates (the same count on the training
traces). Every failed held-out gate is a hysteresis error of 10 % or more (one FR3
trace also exceeds 10 % unloading RMSE). The failed gates are accepted.

### Status of the committed fits

**FR3 sits on three fit bounds.** The first exponent is 40 (upper bound), the
second exponent is -20 (lower bound, -19.9998) and the relaxation time is 5 ms
(lower bound, 5.000 ms). The second modulus is also near its bound. All four
starts end at the same loss (0.00602) on these bounds, so these parameters are
not identified. A change of the bounds would move the fit. The largest column
strain in the fit replays is 0.50. The 100 MPa tangent cap starts at about
strain 0.54 for this fit, so the cap does not shape the fit, but above that
strain it, not the fitted exponents, sets the force. Read the FR3 exponents as
shape parameters of an effective bed response, not as foam constants.

**Dev3 was not refit on the final bed.** Its fit was made on the column bed of
`f4300b06` (grid job 16784231). The shear layer, the tangent cap, the
compression stop, the solver sweeps and the friction law changed afterwards. The
scores in its `material_fit.json` (8 of 24 held-out gates, mean peak error 2.5 %)
come from that earlier bed and cannot be reproduced with the committed code. The
registration of the fit still matches the committed grid. Replay it once on the
final bed before relying on it.

**`fit/report.html` and the `registration.png` files are stale.** They were
built from the previous (pre-refit) fits and show their material table, scores
and registration poses. The poses have moved since (FR3 yaw 0.0676 to 0.0338
rad, y -5.1 to -1.2 mm). Run `sweep_report.py` (and `sweep_fit.py` for the PNGs) to rebuild
them; the `<slug>_grounded_columns/` and `history/` folders belong to the older
fits.

### Provenance of the committed fits

- **Dev3** `material_fit.json` is the output of `sweep_fit.run` (13 starts, 80
  residual evaluations per start, 8 substeps) and has every key it writes.
- **FR3** `material_fit.json` was not written by `sweep_fit.run`. It is the
  result of the fit driver that ran on the LLSC grid as job 16786170 (task `P0`).
  The driver called `sweep_fit.fit_material` with the same objective, bounds,
  8 substeps and 80 evaluations per start on the final bed, from 4 starts (the
  calibrated material and three earlier FR3 fits, no random starts). It
  scored only the fitted material, because replaying the calibrated initial
  material of FR3 on the final bed is not physical and `sweep_fit.run` stops on
  that. Differences from what `sweep_fit.run` writes: no `history` per start, no
  `identifiability`, no `initial` scores per condition, and an extra
  `nonphysical_fitted` list (empty). `registration.qc`, `sweep`,
  `evaluations_per_start` and `initial_material` were carried over from the
  previous FR3 fit (same grid and pose, they are not new output). The grid
  driver is not in the repository. A second run from other starts (task `R1`)
  reached the same material.
  `sweep_report.py` and `export_digital_shoe.py` read this file without the
  missing keys.

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

Each fit runs the calibrated material, one start per `--start` file and six
Latin-hypercube starts with the defaults (80 residual evaluations per start,
seed 0, left shoe, 8 contact substeps per frame). The fit uses the GPU and
float32 fast-math kernels, so another GPU or driver can move the result in the
last digits. The timing in step 3 is from the earlier laptop fits; the committed
fits were run on the LLSC grid.
