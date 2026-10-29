# Slider-shoe material sensitivity (Phase C) — results, filled incrementally

Status: COMPLETE (stage 1 and stage 2). Started 2026-10-29.

## Setup

- User decision: the paper-foam comparison (ff_leap / ff_turbo_plus) is dropped from reporting.
- Shoe: slider FR3 export, sha256 `81ac31ba...` (embedded in `data/f01_20hz_slider_v1`); original geometry for every variant.
- Controllers held fixed (no refit): Phase 2 score fit `/home/gridsan/JO31399/newton_grid/hogan/runs/phase2_slider/20261029-115158_2a5463b1/fit_score/run/runner.json` and sample fit `/home/gridsan/JO31399/newton_grid/hogan/runs/phase2_slider/20261029-115158_2a5463b1/fit_sample/run/runner.json`.
- Dataset: `/home/gridsan/JO31399/newton_grid/hogan/data/f01_20hz_slider_v1/fit_dataset_slider` (107 trials; held-out = 9 eval windows, primary).
- Code: branch `agents/instron-spreading-fit` in the hogan-reset worktree:
  - `32f7bcf4` Test the shoe material sensitivity variants (test-only; fails without the feature: `ModuleNotFoundError`)
  - `5a1c41e7` Add a material sensitivity study of the fitted shoe. Pushed to `/home/gridsan/JO31399/newton_grid/hogan/code/5a1c41e7f0e5`.
  - New module `projects/impedance_instron/hogan/materials_sensitivity.py`: `VARIANTS` + `scale_material` (single place for all parameter edits), `build_conditions` plugged into `materials.main` (same rollout/scoring/work machinery as materials/materials_geometry; default `rematerialize` trial builder since geometry is unchanged), `--combine NAME=A+B` for combinations, `--instron SHOE FIT SWEEP LAST` for the bench replay.
  - `export_digital_shoe._validation_curves` → public `validation_curves(..., materials: list)` that batches several materials in one `LastReplay`; `build_artifact` passes `[material]`. The bench replay therefore uses the exporter's own code path (same depth = stroke + preload indentation + fitted start offset, same substeps).
  - Docs: MATERIALS.md section; changelog fragment `+hogan-material-sensitivity-5c81e2a4.added.md`. `uvx pre-commit` clean; CPU unit tests pass (7 new + materials/geometry tests).

### Which field drives column coupling

`projects/digital_shoe/column_bed.py` (line ~286): `shear = p.shear_fraction * material_shear_modulus(p)`, where `material_shear_modulus` = `equilibrium_shear_modulus_override_pa` if > 0 else `(G1 + G2) * equilibrium_fraction` (`runtime.py`). The same `mu_eq` sets the tangential bristle stiffness `k_t = f G_eq A / L`. `pasternak_n_per_m` is **reported only** (runtime.py docstring: "Nothing consumes it"). So `shear_layer_fraction` (and `mu_eq`) drive coupling. The variants rewrite the reported `pasternak_n_per_m` with the exporter rule (`shear_layer_fraction * mu_eq * mean rest length`; reproduces the original's 787.8 N/m exactly). Note: the old `materials.py`/`materials_geometry.py` rewrite `pasternak_n_per_m` as `mu_eq * mean thickness`, which has no runtime effect.

### Variant definitions (one-at-a-time, original geometry)

| Variant | Edit (in `scale_material`) |
|---|---|
| stiffness_x0.5/0.7/1.4/2 | G1, G2 × s (equilibrium + instantaneous + mu_eq, hence also shear-layer and bristle stiffness); slider unchanged |
| slider_yield_x0/0.5/2 | slider_yield_pa × s (x0 = slider off) |
| slider_modulus_x0.5/2 | slider_modulus_pa × s |
| maxwell_off | equilibrium_fraction → 1, G1,G2 × f_old (equilibrium curve unchanged); tau kept |
| maxwell_share_x2 | Maxwell share 1−f doubled: f 0.9519 → 0.9037, G1,G2 × f_old/f_new (equilibrium curve unchanged); tau kept |
| no_loss | slider off + Maxwell off (true elastic control; `materials.py` `original_elastic` kept the slider) |
| shear_layer_x0.5/2 | shear_layer_fraction × s (0.0721 → 0.0361 / 0.1442) |
| exponent_x0.7/1.4 | hyperfoam_exponent and _2 × s (13.87/−1.23 → 9.71/−0.86 or 19.42/−1.72). The Ogden-Hill small-strain tangent 2(G1+G2)f does not depend on the exponents, so the required rescale factor is exactly 1 (unit test checks the tangent numerically). |

Flag rule (pre-declared): a variant whose held-out score rises by more than 25% over the same controller's original (ratio > 1.25, the guardrail used by `materials_sweep.py`) is flagged as "controller out of fitted range" and its work change is not quoted as a speed change.

## Run registry

| Stage | Controller | Job | Run dir | Status |
|---|---|---|---|---|
| 1 one-at-a-time (+ bench replay) | score | 16846064 | `/home/gridsan/JO31399/newton_grid/hogan/runs/material_sensitivity_slider/20261029-123650_5a1c41e7/score/run` | COMPLETED 5m34s |
| 1 one-at-a-time | sample | 16846065 | `/home/gridsan/JO31399/newton_grid/hogan/runs/material_sensitivity_slider/20261029-123650_5a1c41e7/sample/run` | COMPLETED 4m24s |
| 2 combos (+ bench replay) | score | 16846071 | `/home/gridsan/JO31399/newton_grid/hogan/runs/material_sensitivity_slider/20261029-123650_5a1c41e7/score_combos/run` | COMPLETED 1m32s |
| 2 combos | sample | 16846072 | `/home/gridsan/JO31399/newton_grid/hogan/runs/material_sensitivity_slider/20261029-123650_5a1c41e7/sample_combos/run` | COMPLETED 59s |

## Stage 1 results: one-at-a-time (jobs 16846064 score 5m34s, 16846065 sample 4m24s, both COMPLETED, 0 failed rollouts in every condition)

Local copies of the small files: `material_sensitivity/runs/stage1/{score,sample}/`. Metrics come from `analyze_runs.phase3_table`, the same code as Phase 3. They are held-out means over the 9 eval windows; the proxy is the weighted work, and marathon seconds use couplings 0.5 / 0.65 / 0.8 (+ = faster). Held-out ratio = this variant's held-out score / the same controller's original. No variant crossed the 1.25 flag; the largest ratio is 1.066 (sample, stiffness_x2).

### Runner, score controller (original held-out score 10.43)

| variant | score train / held-out | held-out ratio | failed (train/eval) | W+ J | abs W- J | proxy change | marathon s @0.5 / 0.65 / 0.8 | peak Fz N | peak compression | flag |
|---|---|---|---|---|---|---|---|---|---|---|
| original | 10.54 / 10.43 | 1.000 | 0/0 | 135.7 | 183.9 | -0.00% | -0 / +0 / +0 | 1479 | 0.516 |  |
| stiffness_x0.5 | 11.00 / 11.21 | 1.075 | 0/0 | 132.4 | 178.3 | -2.56% | +127 / +164 / +202 | 1529 | 0.757 |  |
| stiffness_x0.7 | 10.64 / 10.65 | 1.021 | 0/0 | 133.5 | 179.6 | -1.73% | +85 / +111 / +136 | 1472 | 0.651 |  |
| stiffness_x1.4 | 10.75 / 10.48 | 1.005 | 0/0 | 138.4 | 189.1 | +2.22% | -110 / -143 / -177 | 1508 | 0.399 |  |
| stiffness_x2 | 11.83 / 11.15 | 1.069 | 0/0 | 142.5 | 194.8 | +5.22% | -261 / -340 / -419 | 1520 | 0.307 |  |
| slider_yield_x0 | 10.64 / 10.58 | 1.015 | 0/0 | 135.3 | 184.3 | -0.15% | +7 / +9 / +12 | 1502 | 0.520 |  |
| slider_yield_x0.5 | 10.57 / 10.48 | 1.005 | 0/0 | 135.5 | 184.0 | -0.07% | +3 / +4 / +5 | 1487 | 0.518 |  |
| slider_yield_x2 | 10.52 / 10.38 | 0.996 | 0/0 | 135.9 | 183.7 | +0.13% | -6 / -8 / -10 | 1474 | 0.511 |  |
| slider_modulus_x0.5 | 10.54 / 10.44 | 1.001 | 0/0 | 135.6 | 183.9 | -0.01% | +1 / +1 / +1 | 1486 | 0.515 |  |
| slider_modulus_x2 | 10.54 / 10.42 | 0.999 | 0/0 | 135.7 | 183.8 | +0.01% | -0 / -0 / -0 | 1474 | 0.516 |  |
| maxwell_off | 10.55 / 10.46 | 1.003 | 0/0 | 135.6 | 184.1 | -0.02% | +1 / +1 / +1 | 1485 | 0.517 |  |
| maxwell_share_x2 | 10.54 / 10.41 | 0.999 | 0/0 | 135.8 | 183.7 | +0.03% | -1 / -2 / -2 | 1474 | 0.514 |  |
| no_loss | 10.71 / 10.68 | 1.024 | 0/0 | 135.2 | 184.5 | -0.16% | +8 / +11 / +13 | 1509 | 0.522 |  |
| shear_layer_x0.5 | 10.56 / 10.57 | 1.013 | 0/0 | 135.2 | 183.4 | -0.31% | +15 / +20 / +25 | 1513 | 0.556 |  |
| shear_layer_x2 | 10.64 / 10.41 | 0.998 | 0/0 | 136.3 | 184.5 | +0.42% | -21 / -27 / -33 | 1444 | 0.474 |  |
| exponent_x0.7 | 10.66 / 10.54 | 1.010 | 0/0 | 137.1 | 187.3 | +1.23% | -61 / -79 / -98 | 1525 | 0.448 |  |
| exponent_x1.4 | 10.53 / 10.44 | 1.001 | 0/0 | 134.9 | 182.1 | -0.66% | +33 / +43 / +52 | 1473 | 0.560 |  |

### Runner, sample controller (original held-out score 11.45)

| variant | score train / held-out | held-out ratio | failed (train/eval) | W+ J | abs W- J | proxy change | marathon s @0.5 / 0.65 / 0.8 | peak Fz N | peak compression | flag |
|---|---|---|---|---|---|---|---|---|---|---|
| original | 11.89 / 11.45 | 1.000 | 0/0 | 132.8 | 184.7 | -0.00% | -0 / +0 / +0 | 1344 | 0.484 |  |
| stiffness_x0.5 | 11.57 / 11.24 | 0.982 | 0/0 | 124.6 | 175.5 | -5.88% | +289 / +375 / +461 | 1366 | 0.710 |  |
| stiffness_x0.7 | 11.77 / 11.46 | 1.001 | 0/0 | 128.1 | 179.2 | -3.41% | +168 / +218 / +268 | 1323 | 0.602 |  |
| stiffness_x1.4 | 11.89 / 11.46 | 1.001 | 0/0 | 137.2 | 189.3 | +3.18% | -158 / -206 / -254 | 1399 | 0.384 |  |
| stiffness_x2 | 12.43 / 12.20 | 1.066 | 0/0 | 142.8 | 194.3 | +7.01% | -351 / -458 / -565 | 1458 | 0.290 |  |
| slider_yield_x0 | 11.71 / 11.14 | 0.973 | 0/0 | 132.1 | 185.0 | -0.32% | +16 / +21 / +26 | 1363 | 0.488 |  |
| slider_yield_x0.5 | 11.82 / 11.31 | 0.988 | 0/0 | 132.5 | 184.8 | -0.16% | +8 / +10 / +13 | 1350 | 0.486 |  |
| slider_yield_x2 | 11.97 / 11.65 | 1.018 | 0/0 | 133.3 | 184.5 | +0.29% | -14 / -19 / -23 | 1340 | 0.481 |  |
| slider_modulus_x0.5 | 11.85 / 11.38 | 0.994 | 0/0 | 132.7 | 184.7 | -0.01% | +1 / +1 / +1 | 1350 | 0.484 |  |
| slider_modulus_x2 | 11.93 / 11.50 | 1.004 | 0/0 | 132.8 | 184.6 | +0.01% | -0 / -0 / -1 | 1339 | 0.485 |  |
| maxwell_off | 11.83 / 11.32 | 0.989 | 0/0 | 132.6 | 184.8 | -0.08% | +4 / +5 / +6 | 1348 | 0.484 |  |
| maxwell_share_x2 | 11.96 / 11.59 | 1.013 | 0/0 | 132.9 | 184.5 | +0.09% | -4 / -6 / -7 | 1339 | 0.484 |  |
| no_loss | 11.65 / 11.04 | 0.964 | 0/0 | 132.0 | 185.2 | -0.40% | +20 / +26 / +32 | 1370 | 0.488 |  |
| shear_layer_x0.5 | 11.66 / 11.12 | 0.971 | 0/0 | 131.2 | 183.5 | -1.06% | +53 / +68 / +84 | 1376 | 0.513 |  |
| shear_layer_x2 | 12.15 / 11.94 | 1.043 | 0/0 | 134.8 | 185.9 | +1.32% | -66 / -85 / -105 | 1313 | 0.454 |  |
| exponent_x0.7 | 11.77 / 11.24 | 0.982 | 0/0 | 134.9 | 187.5 | +1.62% | -80 / -104 / -129 | 1392 | 0.425 |  |
| exponent_x1.4 | 11.89 / 11.55 | 1.009 | 0/0 | 131.0 | 182.5 | -1.29% | +64 / +83 / +102 | 1333 | 0.523 |  |

### Bench replay (Instron, held-out sweep, fixed measured stroke; from `instron.json` of the score job)

Check: the `original` replay reproduces the artifact's exported validation-curve metrics within 0.12% (max relative difference over all 24 curves and all metrics). The batched `validation_curves` therefore matches the exporter's single-material path. All variants replayed physically (no NaN curves).

Peak force, secant stiffness (peak force / depth at peak) and energy loss (loop area / loading work, both integrated over the last's depth) on the 200 ms curves. Every replay drives the same measured stroke, so the depth at peak does not change; the secant change therefore equals the peak-force change. 

| variant | 800 N: peak N / secant N/mm / loss % | 1600 N: peak N / secant N/mm / loss % | Δ vs original (800 N: peak, secant, loss pts; 1600 N: peak, secant, loss pts) | mean force RMSE (all 24 curves, rel.) |
|---|---|---|---|---|
| measured (bench) | 800 / 56.9 / 11.1 | 1600 / 75.5 / 9.8 | | |
| original | 793 / 56.5 / 11.0 | 1607 / 75.8 / 9.7 | +0.0%, +0.0%, +0.0; +0.0%, +0.0%, +0.0 | 0.008 |
| stiffness_x0.5 | 415 / 29.5 / 18.1 | 827 / 39.0 / 16.0 | -47.7%, -47.7%, +7.1; -48.5%, -48.5%, +6.3 | 0.193 |
| stiffness_x0.7 | 566 / 40.3 / 14.2 | 1139 / 53.7 / 12.5 | -28.6%, -28.6%, +3.2; -29.1%, -29.1%, +2.8 | 0.116 |
| stiffness_x1.4 | 1096 / 78.0 / 8.7 | 2230 / 105.2 / 7.7 | +38.2%, +38.2%, -2.3; +38.8%, +38.8%, -1.9 | 0.155 |
| stiffness_x2 | 1551 / 110.4 / 6.9 | 3166 / 149.3 / 6.2 | +95.5%, +95.5%, -4.1; +97.1%, +97.1%, -3.5 | 0.386 |
| slider_yield_x0 | 757 / 53.9 / 2.4 | 1559 / 73.5 / 2.5 | -4.5%, -4.5%, -8.6; -2.9%, -2.9%, -7.1 | 0.015 |
| slider_yield_x0.5 | 775 / 55.2 / 7.7 | 1583 / 74.6 / 6.5 | -2.3%, -2.3%, -3.2; -1.5%, -1.5%, -3.2 | 0.011 |
| slider_yield_x2 | 828 / 58.9 / 12.9 | 1654 / 78.0 / 14.2 | +4.4%, +4.4%, +2.0; +2.9%, +2.9%, +4.5 | 0.013 |
| slider_modulus_x0.5 | 793 / 56.4 / 8.1 | 1606 / 75.7 / 8.7 | -0.1%, -0.1%, -2.8; -0.0%, -0.0%, -1.0 | 0.007 |
| slider_modulus_x2 | 794 / 56.5 / 12.5 | 1607 / 75.8 / 10.2 | +0.0%, +0.0%, +1.6; +0.0%, +0.0%, +0.5 | 0.009 |
| maxwell_off | 789 / 56.1 / 9.0 | 1596 / 75.3 / 7.4 | -0.6%, -0.6%, -2.0; -0.6%, -0.6%, -2.2 | 0.009 |
| maxwell_share_x2 | 799 / 56.9 / 13.1 | 1618 / 76.3 / 12.0 | +0.7%, +0.7%, +2.1; +0.7%, +0.8%, +2.4 | 0.009 |
| no_loss | 752 / 53.6 / 0.0 | 1549 / 73.0 / 0.0 | -5.2%, -5.2%, -11.0; -3.6%, -3.6%, -9.7 | 0.018 |
| shear_layer_x0.5 | 690 / 49.1 / 11.7 | 1459 / 68.8 / 10.0 | -13.0%, -13.0%, +0.7; -9.2%, -9.2%, +0.4 | 0.041 |
| shear_layer_x2 | 919 / 65.4 / 10.3 | 1793 / 84.5 / 9.3 | +15.8%, +15.8%, -0.6; +11.6%, +11.6%, -0.3 | 0.055 |
| exponent_x0.7 | 916 / 65.2 / 10.1 | 1945 / 91.7 / 8.5 | +15.5%, +15.5%, -0.9; +21.1%, +21.1%, -1.1 | 0.074 |
| exponent_x1.4 | 681 / 48.5 / 12.2 | 1373 / 64.7 / 11.1 | -14.2%, -14.2%, +1.2; -14.6%, -14.6%, +1.4 | 0.053 |

Tornado plot (marathon s at coupling 0.65, both controllers): `material_sensitivity/tornado_marathon065.png`

### Stage 1 reading

- **Foam stiffness dominates by far.** At 0.65: score +164 / −340 s, sample +375 / −458 s for G ×0.5 / ×2. Softer foam lowers the work proxy (−2.6% score, −5.9% sample) under the frozen controllers. It also raises peak compression to 0.76 / 0.71 (original 0.52 / 0.48) and halves bench force at the fixed stroke (−48%). So ×0.5 is a very different, much more bottomed-out shoe, not a small tweak.
- **Second: hyperfoam exponents** (+43 / −79 s score, +83 / −104 s sample). The small-strain stiffness is unchanged, but the bench force at the measured stroke still moves by ±15–21%, and in the same direction as the runner work. Larger exponents (softer at these strains, more compression) help.
- **Third: shear-layer fraction** (+20 / −27 s score, +68 / −85 s sample). Less coupling helps; bench force changes by −13% / +16%.
- **The loss branches barely matter for the runner.** Turning the slider off removes most of the bench hysteresis (loss 11.0% → 2.4% at 800 N; no_loss → 0%). Yet the runner change is only +9 / +21 s (slider off) and +11 / +26 s (no_loss). The Maxwell share and the slider modulus are each under 6 s. All loss-reducing directions help slightly: they also soften the bench curve by 3–5%.
- **Direction is consistent between controllers for every property.** The sample controller is about 1.3–3× more sensitive. The held-out scores move by at most ±7% (no flag). The ranking is stiffness >> exponents > shear layer >> slider yield ≈ no-loss > Maxwell > slider modulus.
- **Caveat:** the controllers are frozen, so these are first-order sensitivities of the work proxy, not refit predictions. The gains go with more compression (softer = better under this proxy).

## Stage 2: combinations (definitions)

Chosen from stage 1. Each knob uses its helpful (or harmful) extreme within the swept range; every property points the same way for both controllers.
- `best` = stiffness_x0.5 + exponent_x1.4 + shear_layer_x0.5 + no_loss (slider off and Maxwell off; slider modulus has no effect once the slider is off).
- `worst` = stiffness_x2 + exponent_x0.7 + shear_layer_x2 + slider_yield_x2 + slider_modulus_x2 + maxwell_share_x2.
- Expect `best` to compress far past the original (stiffness ×0.5 alone gives 0.76). Watch the flag and the bottoming.

## Stage 2 results: combinations (jobs 16846071 score 1m32s, 16846072 sample 59s, both COMPLETED, 0 failed rollouts)

Local copies: `material_sensitivity/runs/stage2/{score,sample}/`. The original row is a repeat run inside these jobs; it reproduces stage 1 exactly.

### Runner, score controller

| variant | score train / held-out | held-out ratio | failed (train/eval) | W+ J | abs W- J | proxy change | marathon s @0.5 / 0.65 / 0.8 | peak Fz N | peak compression | flag |
|---|---|---|---|---|---|---|---|---|---|---|
| original | 10.54 / 10.43 | 1.000 | 0/0 | 135.7 | 183.9 | -0.00% | -0 / +0 / +0 | 1479 | 0.516 |  |
| best | 13.28 / 14.34 | 1.375 | 0/0 | 132.1 | 182.8 | -2.19% | +108 / +141 / +173 | 1781 | 0.760 | OUT OF FITTED RANGE |
| worst | 13.13 / 12.62 | 1.210 | 0/0 | 146.4 | 198.2 | +7.86% | -394 / -514 / -634 | 1414 | 0.234 |  |

### Runner, sample controller

| variant | score train / held-out | held-out ratio | failed (train/eval) | W+ J | abs W- J | proxy change | marathon s @0.5 / 0.65 / 0.8 | peak Fz N | peak compression | flag |
|---|---|---|---|---|---|---|---|---|---|---|
| original | 11.89 / 11.45 | 1.000 | 0/0 | 132.8 | 184.7 | -0.00% | -0 / +0 / +0 | 1344 | 0.484 |  |
| best | 12.64 / 12.20 | 1.066 | 0/0 | 123.7 | 178.3 | -6.05% | +297 / +386 / +474 | 1580 | 0.720 |  |
| worst | 13.42 / 12.81 | 1.119 | 0/0 | 149.3 | 198.7 | +11.37% | -572 / -748 / -925 | 1412 | 0.222 |  |

### Bench replay (score job `instron.json`)

| variant | 800 N: peak N / secant N/mm / loss % | 1600 N: peak N / secant N/mm / loss % | Δ vs original (800 N: peak, secant, loss pts; 1600 N: peak, secant, loss pts) | mean force RMSE (all 24 curves, rel.) |
|---|---|---|---|---|
| measured (bench) | 800 / 56.9 / 11.1 | 1600 / 75.5 / 9.8 | | |
| original | 793 / 56.5 / 11.0 | 1607 / 75.8 / 9.7 | +0.0%, +0.0%, +0.0; +0.0%, +0.0%, +0.0 | 0.008 |
| best | 277 / 19.7 / 0.0 | 594 / 28.0 / 0.0 | -65.1%, -65.1%, -11.0; -63.0%, -63.0%, -9.7 | 0.241 |
| worst | 2140 / 152.3 / 11.6 | 4366 / 205.9 / 10.7 | +169.7%, +169.7%, +0.6; +171.8%, +171.9%, +1.0 | 0.661 |

### Stage 2 reading

- **`best` is out of the fitted range for the score controller** (held-out ratio 1.375 > 1.25). Its +141 s is therefore **not** a speed estimate. For the sample controller, `best` stays in range (ratio 1.066) and gives +386 s at 0.65. That is barely more than stiffness ×0.5 alone (+375 s).
- **`best` is strongly sub-additive.** The sum of its one-at-a-time effects would be +238 s (score) and +552 s (sample). Peak compression saturates at 0.76 / 0.72 (≈ stiffness ×0.5 alone: 0.76 / 0.71), and peak Fz rises 20% / 18% (1781 vs 1479 N; 1580 vs 1344 N). The shoe bottoms out, so softening beyond stiffness ×0.5 buys nothing. On the bench, `best` carries only 35–37% of the measured force at the fixed stroke, with 0% loop loss.
- **`worst` is somewhat super-additive.** It gives −514 s (score) and −748 s (sample), against summed one-at-a-time effects of -456 / -672 s. Both controllers stay in range (ratios 1.21 / 1.12, the 1.21 close to the flag), and peak compression drops to 0.23 / 0.22. On the bench, force at the fixed stroke is 2.7× the measured force.
- **Practical span** for the fitted shoe's material under frozen controllers, at coupling 0.65: about −750 to +390 s (sample controller), or −510 to +164 s (score controller, best in-range single variant = stiffness ×0.5). Almost all of it comes from foam stiffness. Exponents and the shear layer add a little in the harmful direction, and the loss branches are negligible.

## Final summary

1. **Property ranking** (runner, both controllers agree in direction): foam stiffness >> hyperfoam exponents > shear-layer fraction >> slider yield ≈ no-loss > Maxwell share > slider modulus.
2. **Hysteresis does not drive runner work.** The slider sets the bench loop loss (11% → 2.4% when off), yet it moves the runner by ≤ 26 s.
3. **Softer is better under this work proxy, up to bottoming** (compression ≈ 0.72–0.76). Beyond that, the gains saturate and peak Fz rises. The best combination is out of fitted range for the score controller.
4. **Coupling is set by** `shear_layer_fraction × mu_eq`; `pasternak_n_per_m` is reported only.
5. **All numbers come from frozen controllers** (no refit) on the original geometry, held-out windows.
