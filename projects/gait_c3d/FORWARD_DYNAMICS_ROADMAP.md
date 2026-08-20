# Flagship C3D-to-forward-dynamics roadmap

This document is the implementation contract for advancing the Trial 101
instrumented-treadmill pipeline from measured-load reconstruction to predictive,
muscle-driven forward dynamics. Each stage must preserve its inputs, outputs,
provenance, and failed QC. Passing an engineering consistency check must never be
reported as predictive validation.

## Final definitions

The project uses two explicit forward-dynamics milestones.

### FD-1: predictive torque-driven gait

FD-1 starts from a C3D-derived state, applies no measured external wrench after
initialization, generates foot loads only through Newton/OpenSim contact, actuates
only non-root coordinates, and completes a full stride with quantitative
kinematic and kinetic QC.

### FD-2: predictive muscle-driven gait

FD-2 replaces prescribed net joint torques with dynamic muscle excitations,
activations, fibers, and tendons. The pelvis remains unactuated. Any non-root
reserve is bounded, penalized, archived, and included in the pass decision.

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

Status at roadmap creation: implementation complete locally at `fff3172d`; remote
integration pending.

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

- measured wrenches plus ID torques;
- measured wrenches plus bounded non-root tracking;
- left and right measured loads separately;
- feed-forward torque only; and
- alternative input interpolation that does not extrapolate.

### 1.4 Diagnostic tracking controller

Add a named, bounded controller on non-root coordinates only. Track measured
coordinates and speeds while retaining measured loads. Archive feed-forward and
feedback torque separately per coordinate. The pelvis must remain unactuated.

Stage gate: a full measured-load stride remains finite at two time steps; the
result reports controller torque, tracking error, energy, and convergence. This
is labeled `engineering_measured_load_tracking`, not predictive gait.

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

### 2.3 Initial contact gates

- correct bilateral contact ordering and force signs;
- vertical peak relative error below 10%;
- vertical impulse relative error below 5%;
- onset/release error below 20 ms;
- COP RMS error below 30 mm during sufficiently loaded contact;
- friction-cone pass;
- finite force and bounded penetration; and
- no measured load passed to the contact evaluator.

If these predeclared gates prove incompatible with marker-defined foot geometry,
preserve the failure and amend this document before changing them.

Stage gate: prescribed-motion contact passes its declared targets and a held-out
stance result is reported separately.

## Stage 3 — FD-1 torque-driven predictive contact

### 3.1 Initialization

- Initialize from filtered C3D-derived coordinates and speeds.
- Use a prescribed-motion warm-up only to initialize contact/controller state.
- Start from a declared gait phase and archive the exact transition to free
  forward dynamics.
- Use stationary overground contact and remove all measured external loads.

### 3.2 Controller progression

1. Strong bounded tracking on non-root coordinates.
2. Reduced-gain bounded tracking with per-coordinate saturation.
3. Feed-forward ID torque plus the minimum documented stabilization needed.

At every level, archive feed-forward and feedback torque separately. Report RMS
feedback torque divided by RMS total torque for each coordinate. Never actuate
the pelvis.

### 3.3 FD-1 gates

- 100% stride completion at the nominal and one smaller time step;
- no measured external wrench after the free-dynamics transition;
- no root actuator or hidden continuation force;
- bounded coordinates, speeds, energy, and contact penetration;
- declared marker/coordinate tracking thresholds;
- vertical GRF peak and impulse gates from Stage 2;
- contact timing and COP gates from Stage 2;
- correct braking/propulsion signs and friction-cone pass; and
- complete controller-effort and saturation reports.

Stage gate: the artifact is labeled FD-1 only when every required gate passes.
A finite but heavily tracked result remains an engineering continuation result.

## Stage 4 — resolve residual and model inconsistency

Run this work alongside Stages 2–3, but require it before FD-2 interpretation.

1. Sweep only a predeclared small marker/force timing offset and report the full
   sensitivity surface.
2. Test matched marker/force filtering without choosing a setting from one output
   alone.
3. Audit subject mass, segment mass distribution, inertias, and COM locations.
4. Implement an RRA-like adjustment of pelvis kinematics and body COM with bounds
   set by marker-fit tolerances.
5. Archive every adjusted state and model parameter.

Stage gates retain the current quantitative targets:

- pelvis translational residual RMS below 10% body weight;
- pelvis rotational residual RMS below 5% body-weight-height;
- root force reserve below 10% body weight;
- non-root moment reserve below 5% body-weight-height; and
- marker-fit gates remain passed.

Crossing a threshold without an identified sensitivity mechanism is not enough;
the accepted adjustment must be reproducible and biomechanically justified.

## Stage 5 — FD-2 muscle-driven predictive contact

### 5.1 Initial muscle state and controls

- Initialize activation, fiber, and tendon states consistently.
- Seed excitations from Static Optimization, Moco inverse, or an equivalent
  archived solution.
- Replace net non-root joint torques with muscle forces.
- Retain the Stage 3 predictive contact model and stationary ground.

### 5.2 Assistance reduction

1. Muscle-driven tracking with bounded non-root reserves.
2. Penalize and reduce reserves while preserving contact and state stability.
3. Remove reserves where possible; root reserves remain forbidden throughout.

### 5.3 FD-2 gates

- every FD-1 contact, kinematic, integration, and no-root-actuation gate;
- excitations and activations remain in valid bounds;
- fiber/tendon states and muscle forces remain finite and physically bounded;
- reserve magnitude and work are reported per coordinate;
- root reserve is exactly absent;
- non-root reserves pass the declared thresholds; and
- muscle and metabolic interpretation is emitted only after model-compatibility
  and residual gates pass.

Stage gate: one full muscle-driven predictive stride passes at two time steps.

## Stage 6 — continuity and generalization

Progress in order:

1. two continuous strides;
2. perturbed initial pelvis position and velocity;
3. another Trial 101 window;
4. another trial or speed;
5. subject/shoe registration; and
6. predictive human-shoe contact coupling.

The controller must not consume future measured motion in a generalization test.
Report recovery, falls, and failed perturbations rather than selecting only
successful seeds.

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

The required order is:

1. Stage 0 integration.
2. Stage 1 divergence and measured-load tracking harness.
3. Stage 2 prescribed-motion contact calibration.
4. Stage 3 torque-driven predictive contact.
5. Stage 4 residual/model consistency closure.
6. Stage 5 muscle-driven predictive contact.
7. Stage 6 multi-stride and cross-trial generalization.

A later stage may be prototyped early only when it does not bypass an earlier
stage gate or overwrite its artifacts.
