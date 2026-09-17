<!--
SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
SPDX-License-Identifier: Apache-2.0
-->

# Controlled controller refit with fixed contact mechanics

> **Current default:** Digital Shoe now uses Maxwell shear friction automatically.
> The leg-shoe baseline uses mu 0.8, equilibrium stiffness 1000 N/m per nominal
> 25 mm² column, internal viscosity 10 N s/m, and the material relaxation time.
> `friction_model="legacy"` explicitly selects the previous law. Historical
> studies below retain their original settings and do not define current defaults.

> These diagnostics default to the archived initial twelve-point manifest
> `projects/impedance_instron/baselines/baseline12_initial.json`. They do not
> replace the accepted baseline in `projects/impedance_instron/baseline.json`.
> To check another pinned bundle, set `NEWTON_BASELINE12_MANIFEST` explicitly
> and pass its matching `--baseline` directory.

This is an opt-in diagnostic, not a new production controller or friction calibration.
It compares the old equilibrium-command trajectory against refitted commands using
exactly the same Maxwell contact law, gains, initial state, masses, geometry and
normal model. Only the 12-by-4 cubic B-spline equilibrium coefficients are optimized.

## Fixed contact

- Method 7, Maxwell bristle; mu 0.8.
- Equilibrium tangential stiffness scale 0.1, internal viscosity scale 1.
- No extra direct velocity damper (`viscous_ratio=0`).
- Release dwell 0.0005 s; zero rounded yield width.
- Shear relaxation time copied explicitly from the sealed effective material artifact.
  This is an engineering shear extrapolation, not independently calibrated friction.

## Refit objective and limits

The original CUDA leg integrator and device-resident bounded optimizer are reused.
No dynamics equations, normal kernels or controller gains are edited. Simulation
forces are not filtered or downsampled for scoring. The pre-20 Hz acquisition-cleaned
input reference is interpolated to the full simulation force clock.

The objective contains:

1. Original hip/joint tracking channels and raw Fx/Fz curve residuals.
2. Soft foot-pitch and early projected-ground-point velocity residuals. Reference
   velocity is derived from 100 Hz kinematics, not directly measured rubber slip.
3. Braking/propulsive impulses, peaks, force-weighted timing and early braking terms.
4. Explicit penalties for exceeding the unchanged six original tracking tolerances.

Original equilibrium position/rate/acceleration bounds and plant screens are retained.
Additional effort guards are 1.5 times the old-controller actuator component peaks and
2 times its component peak rates, measured with this fixed contact model. These are
engineering guards, not physiological actuator limits. The source profile does not
supply explicit joint motor torque limits; none are invented or labelled as original.

A lower training loss is not acceptance. The original six-channel metrics, force
phases, foot motion, effort, independent CPU result and timestep refinement must all
be inspected. Initial exploratory results and stricter tracking-penalty results use
different objective versions and their losses must not be compared directly.

## Commands

Choose an unused output directory and set `BASELINE` to the sealed baseline directory.

```bash
uv run --no-sync -m projects.digital_shoe friction-controller-refit \
  --baseline "$BASELINE" \
  --output outputs/friction_identification/controller_refit/new_run \
  --iterations 80 --max-wall-s 600 --seed 71

uv run --no-sync -m projects.digital_shoe.friction_controller_validate \
  --baseline-dir "$BASELINE" \
  --coefficients-path outputs/friction_identification/controller_refit/new_run/coefficients.npz \
  --output-dir outputs/friction_identification/controller_refit/new_run_cpu \
  --device cpu
```

The coefficient artifact contains `coefficients` [12, 4] and scalar `duration_s`.
Channel order is hip X/Z equilibrium [m], knee/ankle equilibrium [rad].
`--initial-coefficients` continues from a saved bounded spline without overwriting it.
`--tracking-penalty` controls the explicit unchanged-threshold hinge penalty, not the
threshold values. All choices are recorded in the objective metadata.

Validation writes independent traces for both controllers, raw force metrics, original
six-channel metrics, foot diagnostics, bounds and component-wise effort guards.
`complete` means a full numerical rollout, not an accepted experimental fit.
No coefficient artifact is automatically installed as a default.

## Recorded same-stance diagnostic

The selected seed79 continuation used `--iterations 100 --seed 79` with
`--initial-fraction .002 --minimum-fraction .0002 --tracking-penalty 1000
--tracking-margin .98`, initialized from `equilibrium_constrained_seed73/coefficients.npz`.
The 0.98 factor is an interior training buffer; validation retains the original limits.
See local `outputs/friction_identification/controller_refit/RESULTS.md` for all stages
and the explicit remaining peak-timing/impulse errors. No artifact is installed.
