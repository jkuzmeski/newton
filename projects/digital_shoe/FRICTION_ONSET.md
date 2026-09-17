<!--
SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
SPDX-License-Identifier: Apache-2.0
-->

# Friction onset: initialization and force observations

> **Current default:** Digital Shoe now uses Maxwell shear friction automatically.
> The leg-shoe baseline uses mu 0.8, equilibrium stiffness 1000 N/m per nominal
> 25 mm² column, internal viscosity 10 N s/m, and the material relaxation time.
> `friction_model="legacy"` explicitly selects the previous law. Historical
> studies below retain their original settings and do not define current defaults.

> **Current priority:** Raw physical-force continuity, not peak fitting. `friction-report`
> and `friction-fit` now default to `raw`. Filtered `matched` output is only an
> optional observation diagnostic. See [FRICTION_CONTINUITY.md](FRICTION_CONTINUITY.md).

The onset workflow separates physical force, measured-force processing, and incoming
foot kinematics. It does not alter normal mechanics, geometry, controller gains,
equilibrium commands, or the sealed baseline. No fitted friction parameters are installed.

## Initial coordinates are hip coordinates, not total-leg COM coordinates

The Cartesian leg uses `[hip_x, hip_z, thigh_angle, knee_angle, ankle_angle]`.
The first two velocities are hip translation. Segment COM offsets enter the mass
matrix separately. Foot translation also includes joint rotation through the point
Jacobian. Shared hip motion contributes to foot motion by design, not by applying
one velocity twice.

For the flat sagittal ground plane, the diagnostic decomposes

```text
ankle_velocity = J_ankle(q) @ qdot
plane_vx = hip_vx + relative_leg_rotation_vx + foot_omega * (ankle_z - ground_z)
```

An exact match at the initial frame only establishes agreement with the processed
reference. It does not establish that the estimated initial velocity is physically
accurate. The saved reference has 100 Hz kinematics and finite-difference velocities.
The diagnostic uses one Hermite position interpolant and its derivative, not
independently interpolated positions and velocities.

## Match observations; do not filter applied force

The old comparison placed instantaneous simulation forces against a 20 Hz filtered
reference whose negative vertical values had been clipped. Such processed Fx/Fz
pairs need not satisfy an instantaneous Coulomb cone near contact onset.

The new observation policy:

1. Requires `unfiltered_grf_target_n` and explicit filter provenance. This historical
   name means **pre-20 Hz**, not raw sensor data: the force was already tare-corrected,
   pooled, Hann-filtered and episode-assigned.
2. Applies a normalized Hann kernel to the full-rate prediction before native-clock
   resampling. Its physical duration comes from source metadata. The native 21-sample
   kernel spans 10 ms at 2 kHz; the fine-grid kernel is a declared approximation.
3. Excludes unsupported force endpoints instead of inventing a terminal force.
4. Reconstructs reference and prediction with the same fourth-order, forward/backward
   20 Hz Butterworth filter, common force clock, and odd-padding rule.
5. Retains signed Fx **and** Fz. Negative filtered Fz is a signal-processing artifact,
   not tensile normal contact in the simulation.
6. Defines stance from the separate pre-20 Hz measured normal signal, not clipped
   low-pass output.

No filter is applied to the force used to integrate the leg. Raw curves, peaks,
energy/cone checks, failed-rollout flags, and timestep-refinement checks remain
separate. The raw peak-aliasing guard compares raw peaks with raw native samples;
it does not mistake intended observation-filter attenuation for chatter.

Hann filtering on the shorter simulation window cannot reconstruct unavailable
full-trial context. Its zero-padding approximation and the Butterworth endpoint
padding are recorded. This matches the known linear processing, not an independently
identified force-platform transfer function. Do not infer material friction from
filtered Fx/Fz ratios near touchdown.

## Runnable workflow

Set `BASELINE` to the sealed baseline directory. Use unused output paths; tools reject
overwrites. Historical artifacts are not rewritten.

```bash
# Same normal histories, different declared tangential velocity histories.
uv run --no-sync -m projects.digital_shoe friction-onset \
  --baseline "$BASELINE" \
  --history outputs/friction_identification/history_exact.npz \
  --candidate outputs/friction_identification/candidate_effective_constant.json \
  --output-dir outputs/friction_identification/onset_followup/replay

# Corrected observations of saved full-leg runs. Original baseline is included automatically.
uv run --no-sync -m projects.digital_shoe friction-report \
  --baseline "$BASELINE" \
  --run repaired=outputs/friction_qualification/cpu_baseline_vs_consistent_original \
  --run previous-fit=outputs/friction_identification/dynamic_stribeck_raw_seed89 \
  --output outputs/friction_identification/onset_followup/comparison
```

`friction-report --score-policy matched` selects the secondary observation view; the current default is `raw`. It saves signed observation
signals, per-run metadata, original legacy metrics, and a separate raw-force HTML
view. `--score-policy legacy` is only for the old unmatched comparison.

`friction-fit --score-policy matched` selects the historical observation objective; the current default is `raw`. `legacy-full-rate` retains
the old objective explicitly. The original six-channel eligibility gates are still
reported and retained. Matched force scores and force penalties use the observation
domain. Rollouts remain on GPU, but matched observation scoring currently copies
force curves to a CPU reporting boundary. No speedup or fully GPU-resident scoring
claim is made for this policy.

The independent `friction-check --matched-observations` optionally adds observation metrics. Raw physical metrics are the default; legacy diagnostics remain separately labelled. `score_friction_trace()` remains a general supplied-signal scorer;
`score_observed_forces()` applies the declared observation policy first.

## Limits of the velocity diagnostic

Replacing the tangential velocity history while holding normal load and contact
geometry fixed is a diagnostic, not a new free-leg solution. It must not be injected
into the real force integration as though it were the body's actual velocity.
Reference-derived rigid-foot velocity is not a direct measurement of rubber slip.

The replay uses consistent deflection with zero yield width and identical normal,
stiffness, damping and geometry arrays. Normal histories are hashed before and after.
Coulomb saturation can make different positive velocities produce nearly identical
forces: lower incoming speed does not automatically fix peak force.

See the generated onset report for measured differences. Keep raw and observed peaks
labelled separately. A lower observation error is not an independent material calibration.
