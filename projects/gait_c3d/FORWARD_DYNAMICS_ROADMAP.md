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

## Revision 2 — contact seed geometry

A prescribed-motion probe showed that raw marker-centered spheres produce only
6–11 N peaks and marker-plus-radius centers produce essentially zero force. Heel
markers remain 74–81 mm above the ground at their minimum, while forefoot markers
remain 23–40 mm above it. Therefore contact centers are not initialized directly
at markers. For each role, solve and archive the body-local center nearest its
landmark whose sphere surface is tangent to the measured ground over frozen
role-specific stance frames. Optimization remains bounded to ±30 mm per component
around that geometry-derived seed. The existing unregistered human-shoe spheres
also produced near-zero force and are not an accepted fallback.

## Revision 4 — align implementation with official OpenSim methods

A review of the primary OpenSim 4.6 implementation showed that our proposed
free-form COM and Fourier pelvis optimization was not RRA. That path is abandoned
and must not be integrated. Residual reduction now uses official `RRATool` and
`CMC` as the executable reference. Newton implementations may follow only after
canonical OpenSim fixture and Trial 101 parity.

The pinned reference source is opensim-core commit
`11036b39ca7232c604685b37f483afafc056d92b`:

- `OpenSim/Tools/RRATool.cpp`
- `OpenSim/Tools/CMC.cpp`
- `OpenSim/Tools/CMC_Joint.cpp`
- `OpenSim/Tools/ActuatorForceTarget.cpp`
- the official gait2354 RRA task and actuator resources; and
- the official 3-D Moco walking examples.

RRA automatically adjusts only the selected heavy body's COM X/Z from average
spatial residual moments, recommends but does not silently apply proportional
mass changes, then uses CMC to generate adjusted kinematics. CMC's PD law creates
desired task accelerations; it does not directly apply PD generalized torques.
Force allocation, actuator prediction, excitation inversion, scheduled controls,
and forward integration are required before a Newton feature may use the CMC
name. Muscle redundancy starts with RRA-adjusted motion and official MocoInverse;
motion/contact tracking uses torque-driven MocoTrack before muscle-driven
MocoTrack; untracked prediction is a custom MocoStudy seeded by tracking.

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

- sphere-center displacement from its archived stance-tangent geometry seed:
  ±30 mm per component;
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

## Stage 3 — torque-driven contact tracking with OpenSim Moco

Stage 3 prototypes may run after preliminary contact calibration, but FD-1
acceptance is blocked until official Stage 4 RRA passes. The accepted RRA model
and kinematics then regenerate inverse dynamics and contact calibration.

### 3.1 Official torque-driven MocoTrack bridge

Use the official OpenSim 3-D walking pattern:

1. Process the accepted model with only justified welds and explicit residual/
   reserve actuators.
2. Use the RRA-adjusted state trajectory as the tracking reference.
3. For the measured-load bridge, add corrected ExternalLoads and heavily penalize
   pelvis residual controls.
4. For predictive contact, remove measured ExternalLoads from the model, add
   SmoothSphereHalfSpace forces, and use measured ExternalLoads only as the
   `MocoContactTrackingGoal` reference.
5. Group all spheres per foot; provide alternative frame paths when contact spans
   calcaneus and toe bodies.
6. Solve torque-driven tracking first and archive its solution as the seed for
   muscle-driven tracking.

`MocoContactTrackingGoal` tracks summed force vectors only. COP and free moment
remain independent validation outputs and cannot be passed by the optimization
goal itself.

### 3.2 Treadmill/overground contract

For the verified tied-belt constant-speed Trial 101 window, use the audited
virtual-overground motion with a stationary ground. Never combine treadmill-frame
kinematics with a static floor. Belt acceleration windows remain rejected until a
non-inertial or moving-belt formulation is implemented.

### 3.3 FD-1 definition and gates

FD-1 is the torque-driven contact-tracking solution plus independent forward
replay, not a custom PD callback. Require:

- accepted official RRA model/kinematics and re-fitted contact hash;
- measured external loads absent from predictive model dynamics;
- measured loads used only as force-vector tracking references;
- no hidden root generalized force; every residual/reserve is an explicit named
  actuator with physical force/moment and optimal-force penalty;
- mesh refinement and solver convergence;
- 100% stride completion in independent forward replay;
- marker RMS <30 mm and maximum <60 mm;
- global RMS auxiliary/total non-root torque ratio below 25%, no coordinate above
  50%, and root residuals within accepted RRA component limits;
- every vertical/horizontal/impulse/timing/COP/free-moment/friction/penetration
  gate from revalidated Stage 2 contact; and
- complete model, goal, bounds, weights, solver, guess, and output provenance.

A finite state-tracking solve with large residuals remains an engineering bridge,
not FD-1. Autonomous prediction and perturbation recovery remain Stage 6.

## Stage 4 — official OpenSim RRA residual reduction

Use official OpenSim `RRATool` as the executable reference before any Newton
native port. The input is the corrected scaled model, padded IK, corrected
ExternalLoads, a replacement ideal-actuator ForceSet, and a CMC joint-task set.

### 4.1 Exact RRA preprocessing

- Clamp the requested interval to available motion.
- Pad desired motion by 60 samples.
- Apply OpenSim order-50 low-pass FIR filtering at the declared cutoff before
  completing constrained coordinates.
- Fit quintic GCV splines for q/u and differentiate u for acceleration.
- Exclude locked coordinates from tasks and reserve actuators.

### 4.2 Anthropometry pass

Construct three global pelvis PointActuators at the scaled pelvis COM and three
global TorqueActuators, plus one CoordinateActuator per unlocked internal DOF.
Compute average spatial residual `FX/FY/FZ/MX/MY/MZ` with official inverse
dynamics. For the selected heavy body (torso), reproduce current OpenSim exactly:

```text
bodyWeight = abs(gravity_y) * torso_mass
dx = clamp(MZ_average / bodyWeight, -0.100, 0.100)
dz = clamp(-MX_average / bodyWeight, -0.100, 0.100)
new_com_x = old_com_x - dx
new_com_z = old_com_z - dz
recommended_total_mass_change = FY_average / gravity_y
```

The COM X/Z change is automatic. The total mass change is a recommendation only;
if accepted for the next pass, distribute it proportional to segment mass and
archive every old/new value. Do not alter COM Y or optimize arbitrary body COMs
under the RRA label.

### 4.3 CMC kinematic adjustment

Run official regular `ActuatorForceTarget` CMC with target window 1 ms,
`kp=100`, `kv=20`, `ka=1`, official gait2354 relative task weights, explicit
residual/reserve optimal forces, IPOPT tolerance `1e-5`, and adaptive Manager
integration with maximum step 1 ms and error tolerance `1e-4`. Archive states,
q/u/udot, pErr, controls, Actuation forces/powers/speeds, average residuals, setup,
tasks, actuators, adjusted model, official version, and log.

Iterate as OpenSim prescribes: inspect the first pass, apply a documented mass
recommendation if justified, rerun on adjusted kinematics/model, and make
residuals more expensive only through explicit actuator optimal-force or control
bounds. Never replace CMC with direct joint PD torque.

### 4.4 Acceptance gates

Apply official componentwise RRA guidance and retain the project's normalized
resultant gates:

- max residual force: GOOD 0–10 N, OKAY 10–25 N, BAD >25 N;
- RMS residual force: GOOD 0–5 N, OKAY 5–10 N, BAD >10 N;
- max residual moment: GOOD 0–50 N·m, OKAY 50–75 N·m, BAD >75 N·m;
- RMS residual moment: GOOD 0–30 N·m, OKAY 30–50 N·m, BAD >50 N·m;
- coordinate pErr translation RMS <2 cm and max <2 cm for GOOD (2–4/5 cm
  is reported as OKAY, not silently promoted);
- coordinate pErr rotation RMS/max <2 degrees for GOOD (2–5 degrees OKAY);
- pelvis resultant translational RMS below 10% BW and peak below 25% BW;
- pelvis resultant rotational RMS below 5% BW-height and peak below 10%; and
- marker-fit and force/COP mapping gates remain passed.

A production candidate requires no BAD residual component and explicit review of
every OKAY component. Once an RRA pass is accepted, regenerate inverse dynamics
and re-fit/revalidate contact on the exact adjusted model and kinematics before
FD-1 evaluation.

### 4.5 Newton parity gate

A Newton-native `ResidualReduction`/`ComputedMuscleControl` implementation may be
accepted only after matching official OpenSim single-muscle, two-muscle, arm26,
gait2354, and Trial 101 controls, forces, states, task accelerations, residual
wrenches, and pErr within predeclared tolerances. Until then, official OpenSim is
the reference runtime and Newton artifacts are parity experiments.

## Stage 5 — official OpenSim muscle redundancy and tracking

### 5.1 MocoInverse baseline

Run official MocoInverse first on the accepted RRA-adjusted motion with corrected
measured ExternalLoads. Follow the official 3-D walking ModelProcessor pattern:

- add ExternalLoads;
- weld only unsupported coordinates;
- add explicit weak residuals/reserves;
- use the validated muscle model or replace with DeGrooteFregly2016 through an
  explicit compatibility decision;
- document tendon-compliance and passive-fiber assumptions;
- solve coarse-to-fine mesh (`0.05 -> 0.02 -> 0.01 s` where practical); and
- report reserve controls relative to peak inverse-dynamics moments.

MocoInverse prescribes kinematics. It is the fast muscle-redundancy baseline and
replacement for the current illustrative Static Optimization result; it is not
predictive gait.

### 5.2 Muscle-driven MocoTrack

Use the torque-driven MocoTrack states as the initial guess for muscle-driven
MocoTrack. Track the accepted RRA states and, for the predictive-contact lane,
track grouped contact force vectors. Add periodicity appropriate to a full stride,
including speeds, activations, controls, and auxiliary states; exclude forward
pelvis translation value in the overground frame while making its speed periodic.

Lower-limb reserves are removed or reduced in stages. Root residuals remain
explicit and must meet RRA limits. Validate muscle coordinate coverage and the
moment-arm map before interpreting excitations.

### 5.3 Muscle-driven prediction

Prediction is a custom MocoStudy seeded by the accepted muscle-driven tracking
solution. Remove state/contact tracking goals for a genuinely predictive problem,
or label retained tracking as hybrid. Add full-stride periodicity, average-speed
or belt-relative-distance constraints, physiological objectives, and the
validated contact model. Do not divide effort by lab-frame treadmill pelvis
displacement; use overground/belt-relative distance or a time/stride-normalized
objective.

### 5.4 Muscle gates

- every accepted RRA, FD-1 contact, kinematic, and no-hidden-root-force gate;
- excitations and activations within `[0,1]`;
- finite muscle force and applicable fiber/tendon states;
- every reserve/residual physical magnitude and work archived;
- no lower-limb reserve in final prediction unless explicitly justified;
- Moco mesh/objective/constraint convergence; and
- independent forward replay of the solved controls.

Rigid-tendon results remain explicitly approximate. Elastic fiber/tendon claims
require the official model state definitions or a Newton implementation validated
against OpenSim; they are not inferred from a rigid-tendon simulation.

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
2. Stage 1 engineering diagnostics only.
3. Preliminary Stage 2 prescribed-motion contact calibration.
4. Official OpenSim RRATool/CMC Stage 4 passes and reference artifact acceptance.
5. Optional Newton RRA/CMC implementation only after official fixture parity.
6. Regenerate inverse dynamics and re-fit/revalidate Stage 2 contact on the
   accepted RRA model and kinematics.
7. Official torque-driven MocoTrack contact bridge and FD-1 acceptance.
8. Official MocoInverse muscle-redundancy baseline.
9. Official muscle-driven MocoTrack, then custom predictive MocoStudy.
10. Stage 6 multi-stride and cross-trial generalization.

A later stage may be prototyped early only when it does not bypass an earlier
stage gate, overwrite its artifacts, or receive an acceptance label out of order.
