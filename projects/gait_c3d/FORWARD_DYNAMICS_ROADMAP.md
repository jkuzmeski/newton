# Flagship C3D-to-forward-dynamics roadmap

This document is the implementation contract for advancing the Trial 101
instrumented-treadmill pipeline from measured-load reconstruction to predictive,
muscle-driven forward dynamics. Each stage must preserve its inputs, outputs,
provenance, and failed QC. Passing an engineering consistency check must never be
reported as predictive validation.

## Revision 1 — acceptance-order correction

Independent review after canonical integration found that an accepted residual,
timing, mass, COM, inertia, or kinematic change invalidates earlier inverse
dynamics and contact calibration. Preliminary contact work may begin early, but
residual/model closure must pass before FD-1 acceptance. The accepted model then
regenerates inverse dynamics and re-fits/revalidates contact before torque-driven
FD-1 evaluation. Stage 0 completed at `6f66a393`; both the explicit
`forward_dynamics` artifact and the default `latest` artifact were regenerated
cleanly from that commit with their dependent torque diagnostics.

## Final definitions

The project uses two explicit forward-dynamics milestones.

### FD-1: predictive torque-driven gait

FD-1 starts from a C3D-derived state, applies no measured external wrench after
initialization, generates foot loads only through Newton/OpenSim contact, actuates
only non-root coordinates, and completes a full stride with quantitative
kinematic and kinetic QC.

### FD-2: predictive muscle-driven gait

FD-2 replaces prescribed net joint torques with muscle excitations and dynamic
activation. FD-2R is the explicitly approximate rigid-tendon milestone supported
by the current implementation. FD-2E additionally requires a newly implemented
and independently validated fiber-length/tendon-equilibrium state integrator.
The pelvis remains unactuated. Any non-root reserve is bounded, penalized,
archived, and included in the pass decision.

A prescribed replay, pointwise acceleration reconstruction, measured-GRF
rollout, or pelvis-stabilized trajectory is neither FD-1 nor FD-2.

## Non-negotiable rules

- Preserve the treadmill-to-overground, force-frame, wrench-identity, and
  COP-to-foot gates introduced by `fff3172d`.
- Use a stationary ground in the virtual-overground frame.
- Never apply measured GRF/COP/free torque as an input to predictive stages.
  They are calibration and validation targets only.
- Never apply a pelvis/root actuator in an FD-1 or FD-2 pass candidate.
- Name and archive every controller, reserve, continuation force, and gain.
- Do not hide nonfinite prefixes, tracking effort, contact penetration, residuals,
  or excluded frames.
- Do not weaken a threshold after seeing a result without a documented roadmap
  amendment, rationale, and before/after artifact comparison.
- Keep human-shoe coupling out of the first FD-1 milestone. Add it only after a
  simpler foot-contact model passes.

## Stage 0 — integrate the corrected measured-load foundation

Status: complete. Canonical local and remote `jkuzmeski/biomechanics/main` point
to `6f66a393`, and clean default plus explicit artifacts identify that commit.

1. Fast-forward `jkuzmeski/biomechanics/main` through `340dddd6` and `fff3172d`.
2. Push with an explicit lease after verifying the expected remote base.
3. Regenerate clean Trial 101 artifacts from the integrated commit.
4. Confirm schema `gait_c3d_analysis_3`, cache `gait_c3d_trial_cache_5`,
   `git_dirty=false`, wrench identity, COP-foot association, pointwise ID-to-FD
   closure, and the explicit remaining failed-QC status.
5. Create `jkuzmeski/c3d-predictive-forward-dynamics` from the integrated commit
   in its own worktree.

Stage gate: canonical local and remote branches match; focused tests and
pre-commit pass; clean generated artifacts identify the canonical commit.

## Stage 1 — characterize and stabilize measured-load integration

This stage diagnoses the engineering rollout. It does not make predictive claims.

### 1.1 Time-step convergence

Run the measured-wrench/ID-torque rollout at 1.0, 0.5, 0.25, and, if needed,
0.125 ms. Compare first-nonfinite time, finite-prefix marker error, energy,
coordinate error, speed error, and mass-matrix conditioning.

### 1.2 Short-horizon restart map

From every measured frame, run 25, 50, and 100 ms open-loop windows. Archive the
error-growth rate by start time and coordinate. Identify the first event and
state component that loses stability.

### 1.3 Input decomposition

Compare:

- measured wrenches plus all ID generalized forces, explicitly labeled as using
  six root residual actuators;
- measured wrenches plus ID forces with all six root entries forced exactly to
  zero;
- measured wrenches plus bounded non-root tracking with root feed-forward and
  feedback both exactly zero;
- left and right measured loads separately;
- feed-forward torque only; and
- alternative input interpolation that does not extrapolate.

### 1.4 Diagnostic tracking controller

Add a named, bounded controller on non-root coordinates only. Track measured
coordinates and speeds while retaining measured loads. Archive feed-forward and
feedback torque separately per coordinate. Force all six pelvis entries of both
the feed-forward and feedback generalized-force vectors to exactly zero.

Report external-wrench work, feed-forward work, feedback work, kinetic and
potential energy, free-coordinate mass-matrix condition number, coordinate-range
violations, saturation fraction, and marker/coordinate error. Compare linear
interpolation with causal zero-order hold; neither may extrapolate.

Stage gate: a full measured-load stride remains finite at 1.0 and 0.5 ms; root
commanded force is exactly zero; no coordinate leaves its declared model range;
controller saturation occupies less than 1% of non-root coordinate-time samples;
marker RMS is below 30 mm and marker maximum below 60 mm; and every work, energy,
conditioning, controller-effort, and short-window error-growth metric is present.
This is labeled `engineering_measured_load_tracking`, not predictive gait.

## Stage 2 — calibrate predictive contact under prescribed motion

The scaled gait2354 artifact currently has no foot-ground contact geometry.
Author a project-local contact sidecar instead of silently changing the scaled
subject model.

### 2.1 Initial geometry and law

- Add one stationary ground half-space.
- Add heel, medial forefoot, lateral forefoot, and toe spheres to each foot.
- Start with `newton.opensim.OpenSimContact` smooth sphere-halfspace forces.
- Share parameters bilaterally unless the data reject that assumption.
- Archive body-local sphere centers, radii, force-law parameters, and units.

### 2.2 Prescribed-motion calibration

Replay exact measured kinematics without applying measured loads. Fit contact
geometry/material parameters against measured force-platform targets:

- vertical GRF waveform, peak, and impulse;
- braking and propulsion force;
- contact onset and release;
- COP trajectory; and
- vertical free moment.

Use bounded parameters and record the objective terms. Prefer a held-out test
such as fitting one stance and evaluating the other; later trials are required
before generalization claims.

### 2.3 Frozen calibration contract

The first fit uses the archived OpenSim frame and exactly the pipeline's load
mask, filter, and sample grid. COP metrics use frames with measured and predicted
vertical force at least 200 N. The ground-wrench-to-COP/free-moment calculation
is archived and unit tested. Freeze and hash the configuration before held-out
evaluation.

Initial bounds are:

- sphere-center displacement from its marker-seeded body-local position: ±30 mm
  per component;
- sphere radius: 10–60 mm;
- ground height relative to the measured force-platform plane: ±20 mm;
- stiffness: `1e5–5e7` in the smooth-contact model's native units;
- dissipation: `0–5 s/m`;
- static friction: `0.2–1.5`;
- dynamic friction: `0.1–static_friction`;
- viscous friction: `0–1 s/m`; and
- transition velocity: `0.01–0.5 m/s`.

The objective includes normalized vertical force, horizontal force, impulse, COP,
free moment, parameter regularization, and bilateral-sharing terms. A prescribed
contact time-step and parameter-sensitivity table accompanies the optimum.

### 2.4 Initial contact gates

- correct bilateral contact ordering and force signs;
- vertical peak relative error below 10%;
- vertical impulse relative error below 5%;
- onset/release error below 20 ms;
- COP RMS error below 30 mm on the 200 N loaded mask;
- AP and ML force RMS error each below 10% body weight;
- vertical free-moment RMS error below 2% body-weight-height;
- friction-cone pass;
- maximum sphere penetration below 20 mm;
- finite force at the nominal and one smaller prescribed-motion sample step; and
- no measured load passed to the contact evaluator.

If these predeclared gates prove incompatible with marker-defined foot geometry,
preserve the failure and amend this document before changing them.

Stage gate: prescribed-motion contact passes its declared targets and a held-out
stance result is reported separately.

## Stage 3 — FD-1 torque-driven predictive contact

Stage 3 prototypes may run after preliminary contact calibration, but FD-1
acceptance is blocked until Stage 4 residual/model closure passes. The accepted
model and motion must then regenerate inverse dynamics and re-fit/revalidate
Stage 2 contact before the final Stage 3 run.

### 3.1 Initialization

- Initialize from filtered C3D-derived coordinates and speeds.
- Use a prescribed-motion warm-up only to initialize contact/controller state.
- Start from a declared gait phase and archive the exact transition to free
  forward dynamics.
- Use stationary overground contact and remove all measured external loads.
- Freeze and record the accepted model, contact, controller, and reference hashes.

### 3.2 Controller progression

1. Strong bounded tracking on non-root coordinates.
2. Reduced-gain bounded tracking with per-coordinate saturation.
3. Feed-forward ID torque plus the minimum documented stabilization needed.

At every level, archive feed-forward and feedback torque separately. Report RMS
feedback torque divided by RMS total torque for each coordinate. Root commanded
force is exactly zero; contact projection onto root coordinates remains physical
and is reported separately.

The first milestone is explicitly **contact-predictive tracking**, not autonomous
gait: the controller may use current state and the frozen periodic phase/reference,
but never measured future state or measured external loads. Autonomous or
perturbation-recovery claims require Stage 6.

### 3.3 FD-1 gates

- 100% stride completion at 1.0 and 0.5 ms;
- no measured external wrench after the free-dynamics transition;
- every commanded root generalized force exactly zero;
- no hidden continuation force;
- no declared coordinate-range violation;
- maximum contact penetration below 20 mm;
- marker RMS below 30 mm and marker maximum below 60 mm;
- global RMS feedback/total non-root torque ratio below 25%, with no coordinate
  above 50%;
- controller saturation in less than 1% of non-root coordinate-time samples;
- every vertical, horizontal, impulse, timing, COP, free-moment, and friction gate
  from Stage 2;
- bounded speeds and a complete work/energy balance; and
- complete controller-effort, saturation, and information-set reports.

Stage gate: the artifact is labeled FD-1 only when every required gate passes on
the residual-accepted model. A finite but heavily tracked result remains an
engineering continuation result.

## Stage 4 — resolve residual and model inconsistency

Run preliminary sensitivity work alongside Stage 2, but require this stage before
FD-1 acceptance. Newton currently has no ready RRA tool; this is a new constrained
estimation implementation and must receive its own regression and recovery tests.

1. Sweep only a predeclared ±20 ms marker/force timing offset in 1 ms increments
   and report the full sensitivity surface.
2. Test matched marker/force filtering on a frozen grid without choosing a setting
   from one output alone.
3. Audit subject mass, segment mass distribution, inertias, and COM locations.
4. Implement an RRA-like adjustment of pelvis kinematics and body COM with bounds
   set by marker-fit tolerances.
5. Limit each body COM adjustment to 20 mm per component, total mass change to 1%,
   segment mass change to 5%, and pelvis coordinate adjustment to the existing
   30 mm RMS / 60 mm maximum marker-fit envelope unless this document is amended.
6. Archive every adjusted state and model parameter plus the before/after change.

Stage gates retain the current quantitative targets and add peak reporting:

- pelvis translational residual RMS below 10% body weight;
- pelvis translational residual peak below 25% body weight;
- pelvis rotational residual RMS below 5% body-weight-height;
- pelvis rotational residual peak below 10% body-weight-height;
- root force reserve below 10% body weight;
- non-root moment reserve below 5% body-weight-height; and
- marker-fit gates remain passed without worsening RMS or maximum by more than
  10% relative to the corrected foundation.

Crossing a threshold without an identified sensitivity mechanism is not enough;
the accepted adjustment must be reproducible and biomechanically justified. Once
accepted, regenerate IK derivatives, inverse dynamics, and torque inputs, then
re-fit and revalidate Stage 2 contact on the exact accepted model before final
FD-1 evaluation.

## Stage 5 — FD-2 muscle-driven predictive contact

### 5.0 Capability and coverage gate

Before a gait solve, audit which non-root coordinates the 54 muscles can span,
which coordinates require passive structures or reserves, and the condition of
the muscle moment-arm map across the stride. Benchmark the control method on the
full state dimension. Do not claim that the current dense project-local
collocation solver is scalable until that benchmark passes.

The first executable milestone is FD-2R, using the current rigid-tendon state
`[q, qdot, activation]`. FD-2E is a later capability milestone that first
implements and independently validates fiber-length/tendon-equilibrium states,
initialization, integration, force/energy consistency, and time-step convergence.

### 5.1 Initial muscle state and controls

- Initialize activations consistently and, for FD-2E, initialize fiber/tendon
  equilibrium states with a documented solver.
- Seed excitations from Static Optimization, Moco inverse, or an equivalent
  archived solution, while preserving their failed-reserve status.
- Replace net non-root joint torques with muscle forces wherever muscle coverage
  exists.
- Retain the residual-accepted Stage 3 predictive contact model and stationary
  ground.
- Use a controller/optimizer whose information set and scalability benchmark are
  archived; prescribed-kinematics redundancy resolution alone is not FD-2.

### 5.2 Assistance reduction

1. FD-2R muscle-driven tracking with bounded non-root reserves.
2. Penalize and reduce reserves while preserving contact and state stability.
3. Remove reserves where muscle coverage permits; root reserves remain forbidden.
4. Implement and validate FD-2E before emitting elastic-fiber/tendon claims.

### 5.3 FD-2R gates

- every FD-1 contact, kinematic, integration, and no-root-actuation gate;
- excitations and activations remain in `[0, 1]`;
- muscle forces remain finite and nonnegative where required by the model;
- reserve magnitude and work are reported per coordinate;
- root reserve is exactly absent;
- non-root reserves pass the declared Stage 4 thresholds; and
- rigid-tendon approximation is explicit in every scope/status field.

### 5.4 FD-2E additional gates

- independently validated fiber/tendon state initialization and equilibrium;
- finite, physically bounded fiber length, velocity, tendon force, and energy;
- nominal and smaller time-step convergence; and
- no rigid-tendon fallback in the accepted artifact.

Muscle and metabolic interpretation is emitted only after model compatibility,
residual, reserve, and the applicable FD-2R/FD-2E gates pass.

Stage gate: one full muscle-driven predictive stride passes at 1.0 and 0.5 ms.

## Stage 6 — continuity and generalization

Before running this stage, freeze and hash train/calibration/validation window
lists, the model, contact sidecar, controller, and every threshold. The extractor
must address windows by immutable time/frame bounds rather than silently choosing
the first successful stride. Reject belt-speed ramps with absolute acceleration
above `0.05 m/s²` until a non-inertial mapping is implemented.

Progress in order:

1. two continuous strides;
2. a fixed perturbation set containing pelvis translations of ±20 mm per axis,
   rotations of ±2 degrees per axis, and speeds of ±0.05 m/s per free root speed;
3. at least two held-out steady-speed Trial 101 windows;
4. another trial or gait speed with the same frozen controller/contact parameters;
5. subject/shoe registration; and
6. predictive human-shoe contact coupling.

A run is recovered when it completes two strides, retains every FD-1/FD-2 gate,
and returns within 50 mm pelvis-position, 5 degrees pelvis-orientation, and
0.1 m/s root-speed error of the unperturbed phase trajectory by the second
ipsilateral heel strike. A fall is any nonfinite state, ground contact by a
non-foot body, pelvis height below 0.5 m, or failure to reach the second heel
strike. All frozen unperturbed held-out windows must pass, and at least 90% of
the fixed perturbation cases must recover.

The controller must not consume future measured motion in a generalization test.
Archive the exact causal information set. Report every recovery, fall, and failed
perturbation rather than selecting successful seeds.

Predictive human-shoe coupling remains a separate capability: it requires exact
bilateral subject/shoe registration and an explicit history-state foundation-to-
OpenSim wrench bridge. The prescribed right-foot replay cannot satisfy this gate.

## Required artifact structure

Each stage writes to a distinct, non-overlapping directory and includes:

- manifest with schema, commit, dirty state, source hashes, device, and settings;
- QC summary with every gate, threshold, value, and note;
- exact model/contact/controller configuration;
- state, control, muscle, and contact trajectories needed to reproduce metrics;
- comparison with the preceding accepted milestone; and
- an explicit scope label such as `engineering`, `FD-1`, or `FD-2`.

Publication remains staged so a failed run cannot mix with an older successful
artifact.

## Implementation order

The required acceptance order is:

1. Stage 0 canonical integration and default-artifact regeneration.
2. Stage 1 divergence and measured-load tracking harness.
3. Preliminary Stage 2 prescribed-motion contact calibration.
4. Stage 4 residual/model consistency closure.
5. Regenerate inverse dynamics and re-fit/revalidate Stage 2 contact on the
   accepted model.
6. Stage 3 torque-driven predictive contact and FD-1 acceptance.
7. Stage 5.0 capability/coverage validation, then FD-2R and FD-2E.
8. Stage 6 multi-stride and cross-trial generalization.

A later stage may be prototyped early only when it does not bypass an earlier
stage gate, overwrite its artifacts, or receive an acceptance label out of order.
