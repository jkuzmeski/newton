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
- The tracked S001 base bundle displays the 35 placed markers on the actual
  19-mesh neutral bone geometry without an OpenSim runtime.
- A static calibration C3D builds a saved personalized MJCF from the S001
  segment templates, with per-side thigh/shank/foot scales and all contacts
  tangent to the declared flat ground plane.
- Scaling from the S001 base applies one audited length factor to the MJCF
  meshes, marker sites, body frames, contacts, and inertial geometry, plus a
  separate mass factor to inertial values.
- The S001 static calibration publishes a sealed Visual3D-style segment
  artifact with CODA/Bell--Brand hip centers, medial/lateral knee and ankle
  centers, per-side lengths and widths, and a horizontal flat-foot frame. The
  S001 foot endpoint uses `LHLX`/`RHLX` as hallux markers rather than `LTOE`/
  `RTOE`.
- A calibrated subject compiler saves the static calibration, personalized
  MJCF, per-segment mesh geometry, collision proxies, and flat-ground contacts
  as a reusable subject bundle. The pelvis mesh is rebased into the calibrated
  CODA origin so it does not shift the lower-body chain.

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
- Base marker set: tracked `projects/gait_c3d/assets/s001_base`, identified as
  `S001`; it contains the final neutral MJCF, 35 placed markers, and 19 actual
  subject-specific bone meshes. The source OpenSim/C3D inputs remain external.
- Actual-geometry visualization: `opensim_subject --subject
  projects/gait_c3d/assets/s001_base --show-markers --paused` overlays imported
  marker sites on the S001 bone meshes.
- Default subject: omitting `--subject` opens the tracked
  `projects/gait_c3d/assets/s001_calibrated` bundle with per-segment static
  calibration and three bounded rotational torso axes.
- Base scaling: `scale_subject_mjcf_from_base()` and
  `scale_subject_marker_layout_from_base()` derive a self-contained target
  bundle from S001. Explicit hip-width overrides update both femur frames and
  marker-layout body transforms.
- Static segment calibration: `segment_calibration.py` averages valid C3D
  samples, constructs the CODA/Bell--Brand pelvis and hip centers, derives
  bilateral endpoint frames, and constrains both feet to a horizontal Z-up
  frame. `calibrated_subject.py` consumes that artifact and writes a saved
  personalized MJCF.
- Branch commit: `ab7eeda9` (`Add S001 base marker geometry`).
- Implementation: `projects/gait_c3d/marker_layout.py`, marker sites in
  `projects/gait_c3d/subject_mjcf.py`, and the `--show-markers` overlay in
  `opensim_subject`.
- Artifacts: `model/marker_layout.json`, marker `<site>` elements in
  `model/subject.xml`, and the `marker_layout` reference in `subject.json`.
  The tracked S001 base adds `assets/s001_base/model/subject.xml`, its actual
  `Geometry/*.obj` meshes, and sealed base/scaled manifests.
- Clean-checkout example: the tracked compact OpenSim-output-style fixture builds
  ten sealed marker sites without executing OpenSim.
- Canonical S001 result: 35 sealed marker sites; one-call Newton import produces
  8 bodies, 10 fixed-root inspection DOFs, and 79 shapes/sites. Imported neutral
  site world positions reproduce the converted reference positions within
  0.10 micrometers maximum error.
- Focused validation: the gait suite passes with one optional dependency skip
  when `ezc3d` is unavailable; calibration tests verify CODA landmarks,
  per-side dimensions, sealed provenance, and tamper rejection; base scaling
  tests verify 35 imported sites, 19 visible non-colliding meshes, scaled
  inertias, marker frames, explicit hip width, calibrated body frames, and flat
  foot contacts; the base example is runnable without OpenSim.
- Reusable commands:

  ```bash
  uv run --extra dev -m newton.examples opensim_subject \
    --subject projects/gait_c3d/subjects/marker-demo \
    --overwrite \
    --marker-demo \
    --show-markers \
    --paused
  ```

  ```bash
  uv run --extra dev -m newton.examples opensim_subject \
    --subject projects/gait_c3d/assets/s001_base \
    --show-markers \
    --paused
  ```

  ```bash
  uv run --extra dev -m newton.examples opensim_subject \
    --base-subject projects/gait_c3d/assets/s001_base \
    --subject /tmp/s001_scaled_subject \
    --height 1.80 \
    --mass 90.0 \
    --overwrite \
    --show-markers
  ```

  ```bash
  uv run --extra dev --with ezc3d -m newton.examples opensim_subject \
    --static-cal /path/to/static_calibration.c3d \
    --subject /tmp/s001_calibrated_subject \
    --overwrite \
    --show-markers \
    --show-calibration
  ```

- Remaining limitations: the static calibration path requires the offline
  `ezc3d` dependency. It now includes three bounded rotational torso axes at
  the calibrated sacrum frame, but those axes are not yet fitted to dynamic
  torso motion or validated by residual-driven topology comparisons;
  mass scaling is uniform, segment geometry is per-side, and anisotropic
  segment inertias still use a mean-square scale approximation. It does not fit
  dynamic markers or solve native generalized coordinates. The source C3D/VTP
  inputs used to produce S001 are not bundled, but the neutral base geometry,
  placement, and static calibration are tracked for reproducible model builds.

### Phase 2

- Implementation: `projects/gait_c3d/native_motion_fit.py` binds imported MJCF
  sites to public `newton.ik.IKObjectivePosition` objectives and solves with
  analytic LM, one seed, and temporal warm starts. Invalid marker handling
  begins with an explicit visible-marker subset.
- Example: `newton/examples/opensim/example_native_motion_fit.py` provides the
  reusable `native_motion_fit --synthetic` target/prediction overlay.
- Validation: clean, 1 mm noisy, deterministic occlusion, free-root
  quaternion, joint-limit, and warm-start continuity tests pass on CPU. The
  example reports marker RMS/max error and public solver cost.
- Remaining limitation: real C3D fitting, short-gap interpolation, visibility
  mask solver caching, and full dynamic diagnostics remain Phase 3 work.
