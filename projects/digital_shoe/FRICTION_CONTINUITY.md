<!--
SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
SPDX-License-Identifier: Apache-2.0
-->

# Raw contact-force continuity

The objective here is continuous, well-resolved **physical force**, not peak-height
fitting. No simulation-output filter is used. Normal mechanics, geometry, initial
conditions and controller parameters remain unchanged. Experimental force-fit gates
remain available but are not a certificate of constitutive continuity.

## What the discontinuity tests show

The exact-return bristle force is C0 across stick/slip yield, velocity reversal,
unloading and re-entry for continuous normal inputs and admissible histories.
Its tangent has a kink at yield; it is not globally C1. A history reset under finite
load or a discontinuity imposed in normal force is outside that continuity claim.

In the saved full-leg refinement, the consistent-deflection/direct-damper model's
largest early Fx increment shrinks as the timestep is halved:

| Step [microseconds] | Largest early raw force increment [N] |
|---:|---:|
| 62.5 | 25.48 |
| 31.25 | 13.09 |
| 15.625 | 6.67 |
| 7.8125 | 3.35 |

This supports a fast continuous response rather than a fixed finite jump. However,
its limiting force rate is steep (about 429 kN/s). Finer timesteps do not make that
physical rate slower.

## Internal viscoelastic shear stress instead of direct velocity damping

An ideal parallel dashpot applies a term proportional to relative velocity. It can
therefore produce a force jump if velocity jumps, and a steep force change during
rapid deceleration. Simply softening the elastic spring does not remove this term.

The optional Maxwell-bristle model uses an equilibrium spring in parallel with a
Maxwell branch (another spring in series with a dashpot). This entire element is
in series with the Coulomb slider. The branch force `q` is a mechanical state:

```text
branch_k = viscosity / relaxation_time
q_dot = branch_k * elastic_velocity - q / relaxation_time
traction = equilibrium_k * elastic_deflection + q
|traction| <= mu * normal_force
```

Both elastic and internal branch states are advanced together by backward Euler
and a single radial return. The total stored energy is

```text
E = 0.5 * equilibrium_k * |z|^2 + |q|^2 / (2 * branch_k)
```

The return dissipates plastic work, the dashpot dissipates `|q|^2 / viscosity`, and
backward Euler adds nonnegative numerical dissipation. Tests check this balance,
not just force appearance. There is no external smoothing of the computed force.

At fixed displacement, nonzero branch stress relaxes on its physical timescale.
That is intended viscoelastic behavior, unlike the rejected yield-shoulder scheme
whose static relaxation depended on the number of timesteps.

## Fixed-setting mechanical comparison

Same mu=0.8 and dashpot coefficient; no peak fitting. The time constant is 5.150 ms,
an explicit engineering extrapolation of the existing effective material timescale
into shear, not independent tangential calibration.

| Contact model | Largest early raw Fx step at 62.5 us [N] | Converged max early force rate [kN/s] | Raw braking peak [N] |
|---|---:|---:|---:|
| Consistent bristle + direct damper | 25.48 | 429.05 | about 410 |
| Maxwell branch, same equilibrium stiffness | 16.55 | 265.61 | about 410 |
| Maxwell branch, 0.1x equilibrium stiffness | 7.65 | 122.88 | about 403 |

The softer case has about 71% lower limiting unloading rate than the direct-damper
consistent bristle. It has about 7.2 mm maximum elastic deflection in this test.
All these are raw computed forces. The peak amplitude remains high and is not the
success criterion here. The legacy production path has a 16.22 N step at 62.5 us,
but retains the previously diagnosed non-conjugate tangential work issue.

The 0.1x compliance setting is a demonstrator, not a fitted material constant.
Normal and controller laws are unchanged; resulting motion/normal histories can
change indirectly. No universal runtime default is changed.

On the optimized Cartesian backend, an attached friction adapter selects the shared
foundation launch path so the adapter cannot be bypassed by a fused kernel. The
accepted legacy/fused default remains unchanged. Capture graphs after attachment
(or recapture after changing adapter attachment). This optional path may cost more
than the default fused evaluation; no performance equivalence is claimed.

## Reusable runnable comparison

```bash
uv run --no-sync -m projects.digital_shoe friction-continuity \
  --baseline "$BASELINE" \
  --output outputs/friction_identification/continuity_example \
  --dt-factors 1 2 4 8
```

This writes raw force plots, timestep/rate records, and an explicit candidate JSON.
Use an unused output directory. The sealed baseline is verified and never modified.
The candidate can also be checked with the independent CPU runner:

```bash
uv run --no-sync -m projects.digital_shoe friction-check \
  --baseline "$BASELINE" \
  --candidate outputs/friction_identification/continuity_example/candidate.json \
  --output outputs/friction_identification/continuity_cpu --device cpu
```

Candidate method `7` / `maxwell` uses `kt_scale` for equilibrium stiffness,
`kv_scale` for internal dashpot viscosity, and `shear_relaxation_time_s` in seconds.
Set `viscous_ratio=0`: there is no additional direct cone-limited velocity damper.
The per-world parameter table is append-only: old 7/9/10/11-field rows remain valid;
12-field rows add `shear_relaxation_time_s` after `slip_scale_m`.

## Parked amplitude experiments

Pressure-cap (method 5) and loaded-slip weakening (method 6) are optional research
hypotheses from the preceding amplitude study. They are not adopted to claim a
continuity fix and are not calibrated. Method 5 appends `pressure_scale_pa` (Pa).
Method 6 also appends `slip_scale_m` (m), with `mu` and `mu_dynamic` as cold/hot bounds.
A numerical benefit does not establish that either mechanism describes this outsole.

Force reports and new friction searches default to `raw`: simulation values are
not filtered or downsampled for metrics; only the acquisition-cleaned pre-20 Hz input
reference is interpolated. `matched` remains an explicitly secondary observation
comparison. It must not be used as evidence that computed physical force is smoother.
