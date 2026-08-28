# Newton-native motion retargeting plan

## Goal

Convert subject motion-capture markers into a time series of generalized
coordinates for the subject's Newton-native MJCF model. The result must be a
reusable, validated motion artifact that can drive kinematic replay and later
initialize Newton forward-dynamics experiments.

"Motion retargeting" is the project name. Marker inverse kinematics is the
first algorithm used to produce the retargeted motion, but the output belongs
to the reduced Newton model rather than the source OpenSim coordinate system.

## Architecture contract

OpenSim is an **offline reference and conversion oracle**. Use its documented
behavior, accepted marker placements, and reference results to understand
scaling, frame conversion, marker weighting, and validation problems. Do not
copy its runtime one for one and do not rebuild an OpenSim-shaped solver inside
Newton.

After subject compilation, production motion fitting must consume only neutral,
sealed artifacts:

- subject MJCF;
- subject marker layout;
- C3D-derived marker trajectory;
- frame-registration metadata; and
- optional reference results used only for comparison.

The production fit and replay paths must use established public Newton APIs
where practical:

- `newton.ModelBuilder.add_mjcf()` and MJCF sites;
- `newton.ik.IKSolver`;
- `newton.ik.IKObjectivePosition`;
- `newton.ik.IKObjectiveRotation` only when an orientation target is justified;
- `newton.ik.IKObjectiveJointLimit`;
- `newton.eval_fk()`;
- `newton.Model`, `newton.State`, and `newton.Control`; and
- `newton.solvers.SolverFeatherstone` for the current dynamics baseline.

Do not import `newton._src` from project code, examples, or documentation. Do
not author a project-specific Warp kernel until a measured limitation of the
public API is documented, a no-custom-kernel alternative has been tested, and
the smallest necessary extension has been reviewed.

The historical OpenSim-specific Warp IK on `jkuzmeski/opensim/port` remains an
offline reference. Its exact OpenSim forward kinematics, finite-difference
Jacobian, normal-equation, Cholesky, and batched acceptance kernels are not the
production implementation for this project.

## Data flow

```text
static C3D + template + geometry
        |
        | offline subject compilation and OpenSim reference checks
        v
subject MJCF + neutral marker sites + marker_layout.json

motion C3D
        |
        | direct decode, units, axes, validity, treadmill registration
        v
sealed Newton-frame marker trajectory
        |
        | public newton.ik objectives and solver
        v
native generalized-coordinate motion
        |
        +--> newton.eval_fk() replay and marker residual QC
        +--> later Featherstone initialization/tracking experiments
```

## Shared rules

1. Missing C3D samples are never passed to IK as valid zero positions or NaNs.
2. A free pelvis is used for unregistered whole-body motion. Fixed-root fitting
   is allowed only when the input is explicitly registered to that root frame.
3. Treadmill and stationary-overground frames are distinct and recorded.
4. Marker aliases, virtual markers, interpolation, filtering, and exclusions
   are recorded in the output manifest.
5. OpenSim coordinates are not copied into the reduced native model. Compare
   marker and body trajectories in common physical frames instead.
6. IK replay is not predictive forward dynamics. Measured GRF remains a
   validation target rather than an input to a predictive result.
7. Each phase ends with a reusable runnable example and automated checks. A
   screenshot or recording is added when the example is registered in Newton's
   top-level example gallery.

## Phase 1 — neutral marker layout and MJCF sites

**Status:** complete on `jkuzmeski/mocap-native-ik`; Phase 2 is next.

Convert the accepted offline marker placement into a neutral Newton artifact.
For every marker, store:

- marker name;
- source body;
- target native body;
- target body-local position [m];
- corresponding MJCF site name; and
- source hashes and frame-conversion metadata.

The conversion follows the existing row-vector convention:

```python
ground_opensim = local @ source_rotation.T + source_translation
ground_newton = ground_opensim @ opensim_to_newton.T
target_local = (ground_newton - target_translation) @ target_rotation
```

Apply the same audited root-height offset used by the compiled visual geometry,
joint centers, COMs, inertias, and contacts. Map source calcaneus markers into
the merged native foot body. Emit hidden, non-colliding MJCF `<site>` elements
so the saved model is self-describing.

**Definition of done**

- The marker layout has a versioned, sealed schema.
- Unsafe, duplicate, missing, unknown, or nonfinite marker definitions fail.
- The subject bundle manifest references the layout.
- One-call MJCF import preserves marker labels, bodies, and local positions.
- Tests catch a transpose/frame error and payload tampering.
- The reusable `opensim_subject --marker-demo --show-markers` example displays
  imported neutral marker sites from a clean checkout without an OpenSim
  runtime dependency.

## Phase 2 — synthetic native marker IK

Generate target markers from known configurations of the native MJCF, then
recover those configurations through the public Newton IK API. Start with
Levenberg-Marquardt, analytic Jacobians, one seed, joint-limit objectives, and a
previous-frame warm start. Use `wp.ScopedDevice(model.device)` around solver
creation and execution.

Handle validity without a custom kernel first. The initial strategies, in
order, are:

1. an always-valid marker subset;
2. short-gap host interpolation with explicit provenance; and
3. cached solver instances keyed by per-frame visibility mask.

A validity-weighted custom objective is considered only if profiling proves the
public-objective strategies insufficient.

**Definition of done**

- Clean synthetic targets recover to a documented numerical tolerance.
- Millimeter noise and deterministic occlusion tests stay finite and improve
  materially over the neutral seed.
- Joint limits and free-root quaternion normalization are checked.
- Frame-to-frame motion is continuous under warm starting.
- The reusable `native_motion_fit --synthetic` example overlays target and
  predicted markers and reports coordinate and marker errors.

## Phase 3 — real C3D native motion fit

Decode the dynamic trial directly to Newton-frame SI arrays. Reuse the verified
marker alias and virtual-marker policies from the offline gait analysis, but
publish their results as neutral artifacts. Apply the documented
instrumented-treadmill to stationary-overground registration before contact-aware
replay.

Fit frames in time order and warm-start from the previous native result. Record
times, native coordinates, finite-difference velocities, targets, predictions,
validity, residuals, solver settings, source hashes, and registration metadata.

**Definition of done**

- No invalid observation enters an IK residual.
- Per-frame, per-marker, and per-body residuals are reported.
- Coordinate-limit and frame-jump diagnostics are reported.
- Results are compared with the historical OpenSim analysis only in common
  physical marker/body frames.
- The reusable `native_motion_fit --c3d <trial>` example plays the fitted native
  MJCF with measured markers, predicted markers, and residual lines.

## Phase 4 — residual-driven native model refinement

Use real residual patterns to decide whether the reduced topology needs more
standard MJCF joints. Do not add degrees of freedom solely to copy OpenSim.
Evaluate changes in this order unless the residual evidence gives a different
order:

1. lumbar/torso motion;
2. multi-axis ankle motion;
3. MTP/toe motion;
4. separate hindfoot mechanics;
5. knee translation or another standard-joint approximation; and
6. upper-body segments if their markers are in scope.

Prefer standard revolute, D6, ball, fixed, and free joints already supported by
Newton. A custom coupled-joint kernel is a last resort and requires a standalone
public-API gap analysis.

**Definition of done**

- Each topology change has before/after residual and conditioning evidence.
- Added DOFs remain observable and bounded.
- Synthetic recovery continues to pass.
- The reusable `native_motion_compare` example shows baseline and refined models
  against the same targets with per-body residual summaries.

## Phase 5 — replay and dynamics handoff

Publish the fitted trajectory as a kinematic replay artifact. Use
`newton.eval_fk()` for exact native-model replay. Compute velocities with a
recorded, tested differentiation/filter policy.

Separately demonstrate initialization and bounded non-root target tracking with
Featherstone. Keep root actuation absent and report target, feed-forward, and
feedback effort separately. This is an engineering tracking handoff, not an
FD-1 claim.

**Definition of done**

- Replay reproduces the saved marker residuals.
- Dynamics initialization preserves finite state and normalized quaternions.
- Root effort is exactly zero.
- Tracking error, control effort, saturation, contacts, and integration settings
  are archived.
- The reusable `native_motion_replay` example selects a subject motion from its
  bundle and switches visibly between kinematic replay and the declared native
  tracking experiment.

## Validation ladder

### Synthetic gates

- finite coordinates, poses, and residuals;
- clean marker recovery near numerical precision;
- noisy recovery improves substantially over the neutral seed;
- no limit violations beyond tolerance;
- normalized free-root quaternion; and
- deterministic CPU/CUDA comparison within declared float32 tolerances.

### Real-data reports

Report before setting hard acceptance bars:

- median and 95th-percentile frame RMS [m];
- maximum frame/marker error [m];
- per-marker and per-body RMS [m];
- invalid, interpolated, and excluded sample counts;
- limit proximity/violations;
- frame-to-frame coordinate and body-pose jumps;
- pelvis trajectory and segment orientations; and
- foot marker error by stance and swing.

The historical OpenSim pipeline's 30 mm p95 RMS and 60 mm maximum engineering
bars are reference context, not automatic native-model pass criteria. The
reduced model has a structural error floor until residual-driven topology work
is complete.

## Implementation record

Update this section after each phase with the commit, runnable command, artifacts,
validation results, and remaining limitations. Do not mark a phase complete
until its example and automated checks both pass.

### Phase 1

- Branch: `jkuzmeski/mocap-native-ik`.
- Implementation: `projects/gait_c3d/marker_layout.py`, marker sites in
  `projects/gait_c3d/subject_mjcf.py`, and the `--show-markers` overlay in
  `opensim_subject`.
- Artifacts: `model/marker_layout.json`, marker `<site>` elements in
  `model/subject.xml`, and the `marker_layout` reference in `subject.json`.
- Clean-checkout example: the tracked compact OpenSim-output-style fixture builds
  ten sealed marker sites without executing OpenSim.
- Canonical S001 result: 35 sealed marker sites; one-call Newton import produces
  8 bodies, 10 fixed-root inspection DOFs, and 79 shapes/sites. Imported neutral
  site world positions reproduce the converted reference positions within
  0.10 micrometers maximum error.
- Focused validation: all 34 gait tests pass with one optional dependency skip;
  the `opensim_subject` example passes on CPU and CUDA; pre-commit passes.
- Reusable command:

  ```bash
  uv run --extra dev -m newton.examples opensim_subject \
    --subject projects/gait_c3d/subjects/marker-demo \
    --overwrite \
    --marker-demo \
    --show-markers \
    --paused
  ```

- Remaining limitation: this phase publishes neutral marker attachments. It does
  not yet fit dynamic markers or solve native generalized coordinates.
