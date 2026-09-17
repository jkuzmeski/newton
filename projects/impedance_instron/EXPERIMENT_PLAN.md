# Frozen-controller shoe experiments

Status: approved plan; implementation and validation are in progress under
[`experiments/`](experiments/README.md). A complete qualified campaign has not yet
run. Do not run the default fitting pipeline for shoe variants.

## Question and scope

What happens to the simulated stance when the shoe changes but the person's
fitted control law does not?

Freeze the four equilibrium splines (12 points each), their time base, and all
hip/knee/ankle stiffness and damping gains. Feedback remains active: different
motion produces different actuator forces and torques under the same law.
This tests immediate mechanical response, not controller learning or adaptation.
Do not freeze the actuator force history or prescribe the measured trajectory.

The present model is a planar three-segment leg, one shoe, and one 0.36 s stance.
It has no trunk, opposite leg, added upper-body load, or leg-ground collision.
Results concern this stance model, not full-body falls, repeated running, or a
validated prediction of a person's response. The identified shoe parameters are
effective intact-shoe parameters, not independently identified foam constants.
Changed shoes are synthetic sensitivity cases, not newly calibrated products.

## 1. Select and seal the baseline

The accepted local bundle is `outputs/impedance_instron/baseline12_accepted/`.
The versioned [baseline manifest](baseline.json) identifies the selected fresh
perturbed fit (initialization seed 2718, search seed 101, 200 iterations).
Its loss is 1.5389228967. Saved native/refined acceptance and spring replay pass.
Vertical hip RMS is 19.9883 mm against a 20 mm limit, so acceptance has very
little margin and must not be described as robustness. Planning inspection
verified all 12 local artifact hashes against `baseline.json`; no new fit or
campaign rollout was run. Saved acceptance is historical evidence, not a fresh
qualification of the proposed experiment runner.

Before experiments:

1. Verify every manifest file hash and controller identity. Never overwrite the
   accepted bundle or reuse its acceptance label for a modified shoe.
2. Replay unchanged coefficients using the current source/runtime. Repeat the
   baseline to measure numerical repeatability. Check native/half-step agreement
   and spring-contact replay. Archive source hashes, environment, device, and dt.
3. Seal a controller-only identity (coefficients, gains, spline definition/time
   base) separately from each scenario identity (shoe, initial state, contact).
   Keep the original full identity as the parent provenance. The intentional
   shoe difference is not permission to bypass unrelated identity checks.
4. Only if no usable fitted controller exists, fit a new controller from scratch
   on the original shoe, with no previous coefficients or optimizer state.
   Qualify it before sealing it. Missing motion/shoe inputs are a blocker, not
   something training can repair. Do not select a baseline using variant results.

The existing non-fitting qualification stages are:

```bash
uv run --no-sync -m projects.impedance_instron \
  --baseline outputs/impedance_instron/baseline12_accepted \
  --output outputs/impedance_instron/shoe_experiments_baseline --stage prepare
uv run --no-sync -m projects.impedance_instron \
  --output outputs/impedance_instron/shoe_experiments_baseline --stage validate
```

Choose a fresh output directory. These stages are not the proposed experiment
runner. The default `--stage all` also fits and must not be used for variants.

## 2. Freeze the comparison protocol

Keep subject geometry, inertias, masses, gravity, ground, friction, carrier
registration/pitch, numerical screens, and controller fixed. Geometry variants
retain the model's current mass convention; do not add thickness-dependent mass
in this first study. Keep the constitutive implementation unchanged.

For every primary run, restore the exact baseline initial state and velocity,
controller time zero, and zero material/friction/contact histories. Never carry
Maxwell stress, bristle displacement, passive relaxation, or a terminal leg state
from another case. Refresh derived material/device caches before graph replay.
Run the full stance dynamically, once, without optimization or rescue control.

Changing thickness at a fixed initial pose can change initial ground clearance
or penetration. Record the initial clearance, compression, normal load, and
contacting area for every case. Do not silently settle or move the leg to make
it succeed. A failure at initialization is distinct from loss of support later.

For the geometry cases, also make a separately labeled contact-matched diagnostic
series. Translate only the ground plane so the undeformed shoe's minimum initial
clearance equals the baseline clearance. Keep the leg state, velocity, controller,
and mount unchanged. Record the offset; do not mix these results into the primary
series. This diagnostic matches clearance, not preload or the full contact patch.
It separates stack-height/initial-overlap effects from later geometry response.

## 3. Test A: material sensitivity on the original raycast geometry

Vary one parameter at a time. Use multipliers **0.8, 0.9, 1.1, 1.2** relative to
the baseline, plus one shared unchanged baseline. Do not run a full factorial
search in the first pass.

| Factor | Artifact mapping | Baseline |
| --- | --- | ---: |
| Ogden-Hill term 1 modulus | `instantaneous_shear_modulus_pa` | 267615.20 Pa |
| Ogden-Hill term 2 modulus | `instantaneous_shear_modulus_2_pa` | 19487.70 Pa |
| Term 1 exponent | `hyperfoam_exponent` | 18.076976 |
| Term 2 exponent | `hyperfoam_exponent_2` | -0.585587 |
| Maxwell relaxable fraction, b | `1 - equilibrium_fraction` | 0.341328 |
| Maxwell relaxation time | `maxwell_relaxation_time_s` | 0.00515011 s |

This gives **24 material variants**. Preserve the negative second exponent's
sign: the multiplier acts on its signed value. For the Maxwell fraction use
`b_new = scale * b_baseline`, then `equilibrium_fraction_new = 1 - b_new`.
This changes the relaxable fraction at fixed instantaneous moduli. It also
changes equilibrium stiffness; it is not a pure damping-only change. Changing
tau holds instantaneous/equilibrium moduli fixed but shifts relaxation timing.

Keep Poisson ratio, contact friction, raycast anchors, lengths, areas, footprint,
and driven/passive mask unchanged. Let the existing Pasternak coupling follow
the equilibrium moduli and thickness as usual; do not manually freeze a derived
coupling coefficient. The artifact's scalar `pasternak_n_per_m` is descriptive,
not an independent runtime input.

Before a leg rollout, use the shared law to inspect each case's compression,
loading/unloading, and relaxation curves over the tested strain/rate range.
Check admissible parameters, finite compressive response, and no spurious energy
generation over a closed material-state cycle. Separate invalid material/setup
from a valid material that causes the leg simulation to terminate. Do not claim
that changing tau monotonically increases damping or improves the shoe.

After the first pass, optional follow-ups can vary both moduli together, separate
term balance from overall scale, or test a small declared interaction grid. Do
not retune the controller or choose follow-ups only because motion looks good.

## 4. Test B: spring-length geometry sensitivity

Create four variants with every raycast column's rest length multiplied by
**0.8, 0.9, 1.1, or 1.2**. The factor 1.0 is the shared baseline.

Hold each nominal physical spring top fixed in the shoe/foot frame. Extend or
shorten downward along the original column axis:

```text
L_new[i] = scale * L_original[i]
bottom_new[i] = bottom_original[i] + (L_original[i] - L_new[i]) * local_up
```

This changes thickness, not shoe length/width, material modulus, or foot size.
Keep planar sample positions, per-column areas, neighbors, driven/passive mask,
mount, and static pitch unchanged. Retain passive-surround behavior; preserving
a nominal top does not turn a passive column into a rigid foot attachment.

The material parameters stay byte-identical. Structural stiffness need not stay
constant: strain uses the new rest length, and the existing shear-layer coupling
also depends on thickness. Do not compensate either effect by refitting material.

The adapter has inherited fixture/last display offsets separate from nominal
physical spring tops. Preserve these offsets explicitly. Do not accidentally
change the physical attachment while rebuilding an artifact, or repair the old
fixture interface only for variants. Verify transformed bottoms, physical tops,
and display sites at common foot poses. In particular, shift each matched
fixture's `foam_bottom_m` by the same displacement as its bed bottom, retain
`carrier_anchor_m`, and update fixture rest lengths/free tops consistently. The
adapter adds `bed_bottom - fixture_foam_bottom` to carrier sites; changing only
the bed would silently move those sites. Render the actual modified spring bed,
not an unchanged mesh presented as the simulated geometry.

## 5. Geometry endpoint: uniform rectangular spring bed

Use the **horizontal bounding box of the midsole mesh in the intrinsic shoe
frame**, not its world-aligned box while the foot rotates. Fill it with a regular
spring grid at the original approximately 5 mm resolution. Use correct edge-cell
areas and rebuild neighbors; do not assign extra full-cell area outside the box.

Recommended primary common height: the original area-weighted mean rest length,
`h0 = sum(area[i] * L[i]) / sum(area[i])`, approximately **29.79 mm**. This is an
explicit thickness choice, not the full mesh bounding-box height. Put the top
plane at the original area-weighted nominal-top height (about 44.64 mm in the
intrinsic frame), with bottoms `h0` below it. Keep mount and static pitch fixed.

Rigidly attach every rectangular spring top to the foot carrier. Use the exact
baseline material, per-area normal law, existing coupling rule, and contact law.
Do not add a second mesh-collision force path. Keep the current foot mass.

This changes outline, contact area, top/bottom shape, thickness distribution,
and which columns are directly attached. It is deliberately a broad geometry
ablation, not a pure thickness or equal-stiffness comparison. Report total area,
volume, column count, and driven area instead of hiding these changes by rescaling
material stiffness.

Add one explanatory companion: the same constant-height/top-plane construction
on the **original sampled footprint with its original driven/passive mask**.
This separates flattening from the combined rectangle-area/attachment change.
It does not independently separate rectangle area from attachment; a further
all-driven original-footprint case can do that if needed.

The current mesh box is about 292.10 by 113.19 mm. At 5 mm resolution the rectangle
needs roughly 1357 columns, versus 910 for the baseline. It exceeds the current
1024-column fused fast-path limit. Use and validate the existing general path;
do not coarsen only this geometry to make it fit the fast path.

## 5a. Required rearfoot and fullfoot hysteresis graphs

Run a separate prescribed-displacement bench test for each primary shoe condition,
independently of whether its frozen-controller stance completes. The user confirmed
rearfoot and **fullfoot** (not forefoot-only): use the artifact's `rearfoot_punch`
and `fullfoot_last` fixtures with their exact identities and geometry.

Plot force [N] against displacement [mm], show loading/unloading direction, and
overlay the unchanged-shoe loop for the same fixture and protocol. Retain raw
time/displacement/force arrays, first-cycle and conditioned-cycle curves, peak
force, input/returned/net work [J], and closure and cap diagnostics. Do not interpret
a nonclosed cycle's net work as pure material dissipation.

Use the same declared displacement depth, time waveform, loading rate, hold times,
and conditioning-cycle policy across shoe variants for each fixture. Reset
material/friction histories between independent bench tests, not between cycles
within a conditioning sequence. Refine bench timesteps separately from stance.
A motion failure must not suppress its bench plot. A bench failure retains its
valid prefix and visible failure marker, not a fabricated closed loop. The
rectangle's all-column stance attachment is separate from the bench fixture:
rearfoot and fullfoot retain their actual carrier surfaces and footprints.

The 31 primary conditions require **62 fixture-specific hysteresis tests** before
bench timestep refinement. Clearance-matched stance diagnostics use the same shoe
artifacts and need no duplicate bench test. Missing requested fixtures must fail
closed rather than silently substitute another shape.

## 6. Measurements and failure reporting

Compare every case primarily with the unchanged simulated baseline. Measured
motion/force are secondary references: a changed shoe should not be forced to
match the old shoe's data. The original fit loss is descriptive, not the criterion
for a useful experiment or a good shoe.

Record:

- Hip x/z, thigh angle, knee/ankle angle, foot pitch, velocities, and terminal state.
- Horizontal/vertical GRF, ankle contact moment, impulses, peak values/times,
  contact onset/loss/duration, and slip. COP is valid only above a declared normal
  force threshold, using the actual contact force/moment convention.
- Hip force and knee/ankle torque histories, with peak and time-weighted RMS loads.
- Joint power from torque times relative angular velocity, plus produced,
  absorbed, net, and cumulative work from full-resolution saved force support.
  Show hip-point work and four-actuator totals separately to expose load transfer
  and avoid cancellation between actuator channels.
- Baseline-relative actuator changes and native/half-step contrast drift on
  common saved support. These diagnostics are not new qualification gates.
  Export both primary and clearance-matched actuator metrics to CSV and JSON.
- Per-column strain/pressure, contact area, peak compression, cap activation,
  and spatial load maps. If material work/dissipation is reported, distinguish
  stored energy remaining at the end from losses; stance loop area alone is not
  certified Maxwell dissipation.
- Baseline-relative RMS/max changes on shared time support. Show trajectories
  and event timing, not just a single score.

Keep every case in the result table. Separate these outcomes:

1. Completed stance, with the size and direction of motion changes.
2. Completed with diagnostic warnings (joint-range exceedance or passive caps).
3. Terminated at a declared model screen: record first event, time, reason,
   completed fraction, and last valid state, plus whether refinement supports it.
4. Numerical uncertainty: nonfinite state, timestep-sensitive behavior, solver
   sensitivity, or failed CPU/GPU/contact consistency.
5. Invalid artifact, parameter, initialization, or experiment setup.

Do not equate any numerical screen with a proven human fall. Keep current screens
and cap behavior unchanged. A diagnostic joint limit is not a new hard stop.
Preserve valid prefixes for terminated runs. Do not pad them with frozen states,
extrapolate them, or assign them deceptively good full-stance tracking scores.
Loss of support or motion failure is a result, not a reason to refit or delete a
case. An initialization failure must not be narrated as late-stance instability.

## 7. Execution order and numerical checks

1. Seal and replay the accepted baseline; measure repeated-run numerical spread.
2. Implement a separate frozen-rollout runner, variant builder, and comparison
   report. No fit/search entrypoint is reachable from a variant run.
3. Use the actual experiments as the validation work, as requested by the user.
   Do not add standalone test scaffolding. Keep controller checksums, explicit
   material/geometry changes, fixture consistency, reset rules, and failure
   reporting inside the experiment runner.
4. Start with the unchanged baseline, one material condition, and one length
   condition at native and half timestep, plus both actual fixture benches.
   Then run the rectangle through the existing general GPU path. Retain real
   motion failures rather than constructing extra dummy failure tests.
5. Run **31 primary conditions**: baseline + 24 material + 4 scaled-length +
   rectangle + original-footprint flat companion. Also run the six geometry
   variants in the separate contact-matched diagnostic series.
6. Run each at **62.5 microseconds and 31.25 microseconds** with the controller
   frozen. This is 74 native/refined rollouts before repeatability controls and
   numerical qualification. Refine further when the conclusion is step-sensitive.
7. For complete runs, retain current comparison tolerances: 2 mm hip position,
   0.01 rad joint/thigh angle, 25 N GRF maximum differences. Show actual errors;
   a shoe effect smaller than numerical spread is unresolved. For terminated
   runs, add failure-aware comparison of common valid prefixes, event category,
   and event-time convergence; the existing complete-only refinement helper is
   insufficient. Do not relax limits until a case appears successful.
8. Recheck CPU/GPU/contact agreement on representative extrema and each new
   geometry path. Check passive-solver iteration sensitivity where caps or the
   surround dominate; timestep refinement alone is not enough.
9. Produce the full comparison report, including unsuccessful and unresolved
   cases. Only then choose declared follow-up interaction/boundary tests.

Use GPU-resident rollout and material state. Same-geometry material variants can
use per-world materials after validating the cache/reset path. The current engine
shares geometry across worlds, so geometry variants need separate engines or a
separately qualified layout extension. Do not implement heterogeneous geometry
batching just to run five or six shapes. Measure elapsed time and completed useful
conditions, not utilization or padded duplicate worlds. Repeated deterministic
rollouts measure numerical spread, not biological uncertainty or subject count.

## 8. Deliverables and completion criteria

Save a new campaign directory containing:

- `plan.json`: all conditions, multipliers, parameter units, fixed fields,
  height/attachment/initialization rules, thresholds, and parent hashes.
- A sealed controller bundle and explicit per-case synthetic shoe provenance.
- Per-case traces, failure diagnostics, refinement evidence, and spring histories.
- `results.csv` / `results.json`: all conditions, not only completed cases.
- Per-condition rearfoot/fullfoot hysteresis plots and raw bench histories,
  with exact fixture identities, unavailable-fixture status, protocol, conditioning,
  work/closure diagnostics, and baseline overlays.
- An interactive HTML replay that reuses the native report renderer, not a
  second schematic renderer. Select any two simulated conditions. Overlay mode
  uses a solid primary and adjustable-opacity comparison instead of mocap.
  Side-by-side mode shows both native views with shared time. Put Play and the
  time slider between the animations and two always-side-by-side heat maps.
  Match map colors to the native renderer's acknowledged stance frames in both
  animation modes. Keep actual last meshes, each shoe's own solved spring bed,
  shared compression scales, foot close-ups, and visible termination/cap markers. Explain
  CAD context, compression colors, controller targets, and force arrows.
- A documented reusable `create` / `run` / `report` workflow plus a small end-to-end
  example. These commands are planned, not currently available.

Completion means all planned cases are accounted for, unchanged controls replay,
the controller stayed unchanged, differences can be distinguished from numerical
artifacts where claimed, and the user can inspect and rerun both successful and
failed cases. Completion does not require every shoe to support the stance.

## Current implementation references

- [Pipeline stages](pipeline.py), [bundle preparation](cartesian/gpu/baseline.py),
  [CPU rollout and screens](cartesian/run.py), [GPU engine](cartesian/gpu/engine.py).
- [Fixed shoe attachment](cartesian/shoe.py),
  [fit/refinement definitions](cartesian/fit.py),
  [GPU spring export](cartesian/gpu/springs.py).
- [Shared material law](../digital_shoe/material.py),
  [foundation/material reset API](../digital_shoe/runtime.py),
  [fused path and fallback](cartesian/gpu/foundation.py).
