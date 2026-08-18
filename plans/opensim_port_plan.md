# OpenSim → Newton/Warp Port Plan

## Goal

Port the OpenSim biomechanics stack (`opensim-org/opensim-core`) to be fully
Warp/Newton native: parse OpenSim models, simulate musculoskeletal dynamics on
the GPU, and reproduce the OpenSim analysis/tool workflows (IK, ID, forward,
static optimization, Moco optimal control) using Newton solvers and Warp kernels.

OpenSim-core is ~1,000 C++ source/header files across `Simulation`, `Common`,
`Actuators`, `Tools`, `Analyses`, and `Moco`. A literal file-by-file port is a
multi-month effort. This plan stages the work so each phase lands a usable,
tested capability rather than an inert translation.

## Design principles

- **Newton-native, not a wrapper.** No Simbody/SimTK dependency. Kinematics and
  dynamics run on Newton's `ModelBuilder` + solvers (Featherstone / MuJoCo /
  XPBD); muscle mechanics run as Warp `@wp.func`/`@wp.kernel` code.
- **Differentiable first.** Prefer smooth formulations (De Groote-Fregly 2016)
  so the port composes with Newton's gradient-based tooling.
- **Public API discipline.** Everything lives under `newton/_src/opensim/`
  (internal) and is exposed via `newton/opensim.py` (public). Examples/docs never
  import `newton._src`.
- **SI units, Y-up.** Match OpenSim conventions on import; convert at the boundary.

## Package layout

```
newton/_src/opensim/
    types.py       # solver-agnostic IR dataclasses (OsimModel, OsimBody, ...)
    parser.py      # .osim XML -> IR (4.x sockets + legacy < 30000 inline joints)
    functions.py   # coordinate Functions (Linear/Constant/SimmSpline/Multiplier)
    kinematics.py  # OpenSim-exact forward kinematics in float64 Warp kernels
                   #   (CustomJoint SpatialTransform, batched over poses)
    mocap.py       # .trc marker + .mot/.sto storage read/write
    ik.py          # marker-based inverse kinematics; Warp kernels for FK, residual,
                   #   normal equations (J^T J, J^T r) and costs; Levenberg-Marquardt
    dynamics.py    # inverse + forward dynamics in Warp kernels (Newton-Euler via
                   #   the batched FK Jacobian transpose; forward dynamics reads the
                   #   mass matrix + bias from the same kernels and integrates);
                   #   Butterworth filter + GCVSpline differentiation; ExternalLoads
    gcvspl.py      # Woltring GCVSPL quintic spline port (OpenSim GCVSpline)
    importer.py    # IR -> newton.ModelBuilder (bodies, joints, shapes)
    muscle.py      # Warp-native Hill-type muscle curves + force/activation
    muscle_path.py # Warp-native GeometryPath: muscle-tendon length + moment
                   #   arms (r = -dL/dq) for fixed / conditional / moving points;
                   #   PathWrap over WrapCylinder / WrapSphere surfaces
    muscle_force.py# rigid- and elastic-tendon muscle force, activation-dynamics
                   #   integrator, fiber kinematics, and the joint (generalized)
                   #   forces muscles apply; PathSpring / Ligament forces
    actuators.py   # CoordinateActuator, PointActuator/TorqueActuator (spatial),
                   #   and BodyActuator generalized forces (Warp kernels)
    forces.py      # PointToPointSpring, SpringGeneralizedForce, BushingForce
    contact.py     # OpenSim compliant contact forces (SmoothSphereHalfSpace /
                   #   HuntCrossley / ElasticFoundation) as Warp kernels
    scale.py       # subject scaling: ModelScaler + MarkerPlacer + ScaleTool,
                   #   .trc/.c3d marker I/O, virtual-marker synthesis
    visualize.py   # Warp-native motion visualization: per-frame body transforms
                   #   (OpenSim-exact FK), skeleton bones, and length-colored
                   #   muscle path lines for a Newton viewer (MotionVisualizer)
newton/opensim.py  # public re-exports
newton/tests/test_opensim.py
```

## Phases

### Phase 0 — Foundation  ✅ (this branch)

- [x] `.osim` 4.x parser → IR (bodies, joints, coordinates, frames, muscles,
      actuators, contact geometry/forces, markers).
- [x] Importer → `ModelBuilder`: bodies + inertia, joints
      (Weld/Pin/Slider/Ball/Free/Planar/Universal/Custom→D6), contact
      spheres/half-spaces; topological body ordering; articulation creation.
- [x] Warp-native muscle curves: De Groote-Fregly (2016) active/passive
      force-length, force-velocity, tendon force; Thelen (2003) curves;
      rigid-tendon force evaluation; first-order activation dynamics.
- [x] Public module `newton.opensim` + unittest suite.

### Phase 1 — Muscle path & moment arms

- [x] Warp kernel to compute muscle-tendon length from path points across all
      muscles in parallel (reuses the validated forward-kinematics body
      transforms; `newton.opensim.MusclePaths`).
- [x] Moment-arm computation per spanned coordinate as `r = -dL/dq` (central
      finite difference of the path length — OpenSim's own moment-arm
      definition; `compute_muscle_moment_arms`).
- [x] Conditional (range-gated) and moving (location = `SimmSpline`/
      `LinearFunction`/`Constant` of a coordinate) path points, with the IR +
      parser extended to capture them. Path-point locations and conditional
      gating are now evaluated fully on device (`point_sample_kernel` reusing the
      shared `_eval_axis` coordinate-function evaluator). **Validated: path length matches the
      geometry to ~1e-16 and moment arms match the finite-difference definition
      to ~1e-10; gait2354 and arm26 (neither uses wrapping) give physiological
      knee/elbow moment arms.**
- [x] Muscle-tendon velocity (dL/dt) from coordinate speeds
      (`MusclePaths.velocities`: :math:`\dot L_{MT}=-\sum_i r_i\,\dot q_i`;
      matches the finite-difference time derivative of path length to ~1e-10).
- [x] Muscle forces and the generalized (joint) forces they apply
      (`newton.opensim.MuscleForces` / `compute_muscle_generalized_forces`):
      rigid-tendon De Groote-Fregly (2016) force in a Warp kernel, projected onto
      the coordinates as :math:`\tau_i=\sum_m r_{m,i}F_m` (OpenSim's
      `Moment = MomentArm * force`). This is the muscle model Moco uses. Applying
      the forces as body spatial forces (for a muscle-driven forward simulation of
      the skeleton) is still pending.
- [x] `PathWrap` over `WrapCylinder` and `WrapSphere` surfaces (geodesic
      tangent-arc-tangent detour inserted into the path length in a Warp kernel;
      honors the `PathWrap` `range`). `WrapEllipsoid` (affine "scaled sphere"
      geodesic, exact for the isotropic case) and `WrapTorus` (tube reduced to a
      sphere of the tube radius on the nearest ring point) are now modelled too
      (`wrap_ellipsoid_extra` / `wrap_torus_extra`). **done.**

### Phase 2 — Muscle dynamics & controllers

- [~] Equilibrium (elastic-tendon) muscle. Isometric elastic-tendon equilibrium
      force (series force balance solved on device,
      `MuscleForces.forces_elastic_tendon` / `equilibrium_force_kernel`), plus
      elastic-tendon fiber velocity and fiber kinematics
      (`MuscleForces.elastic_tendon_fiber_velocity` / `fiber_kinematics` /
      `fiber_forces`). A full fiber-length **state + integrator** for a
      muscle-driven forward simulation is still **pending**.
- [x] Excitation→activation dynamics integrated on device
      (`MuscleForces.integrate_activation`, first-order De Groote-Fregly
      activation ODE). Wiring it as a live Newton control channel in a forward
      run is part of the muscle-driven forward-dynamics item (Phase 3).
- [x] `CoordinateActuator`, `PointActuator`, `TorqueActuator`, and body
      (spatial-force) actuators as generalized forces in Warp kernels
      (`newton.opensim.CoordinateActuators`, `SpatialActuators`, `BodyActuators`).
- [x] Other force elements as generalized forces on device
      (`newton.opensim.PathSpringForces`, `PointToPointSprings`,
      `SpringGeneralizedForces`, `LigamentForces`, `BushingForces`).
- [ ] `PrescribedController`, `ControlSet` playback.

### Phase 3 — Tools (analyses)

- [x] Inverse Kinematics: markers → coordinates. OpenSim-exact forward
      kinematics + Levenberg-Marquardt least-squares marker fit
      (`newton.opensim.InverseKinematics` / `solve_marker_ik`), implemented in
      float64 Warp kernels (CPU or CUDA): batched FK Jacobian, and Warp
      reductions for the residual, normal equations and candidate costs.
      **Validated 1-for-1 against OpenSim's gait2354 synthetic-marker IK
      regression.**
- [x] Inverse Dynamics: coordinates → joint moments. OpenSim's
      `InverseDynamicsTool` pipeline (reflective padding, zero-lag 6 Hz
      Butterworth low-pass, quintic GCVSpline differentiation) plus a
      Newton-Euler inverse-dynamics core in float64 Warp kernels that reuses the
      batched FK Jacobian and assembles joint moments with its transpose;
      `ExternalLoads` ground reactions (`newton.opensim.InverseDynamics` /
      `solve_inverse_dynamics`). **Validated 1-for-1 against OpenSim's
      `Applications/ID/test` regressions (arm26, gait2354 subject01).**
- [x] Forward Dynamics: applied generalized forces -> accelerations and time
      integration. Reuses the affine-in-acceleration inverse-dynamics Warp
      kernels to read the mass matrix and bias (composite rigid-body method),
      solves the equations of motion, and integrates (RK4 / symplectic Euler)
      with optional joint-force controls and `ExternalLoads`
      (`newton.opensim.ForwardDynamics` / `solve_forward_dynamics`).
      **Validated 1-for-1 against OpenSim's `Applications/Forward/test` pendulum
      (SHO reference, < 1e-2) and as the exact inverse of the validated inverse
      dynamics (~1e-5 on gait2354).**
- [x] OpenSim compliant contact forces (`newton.opensim.OpenSimContact`): all
      three models as differentiable Warp kernels — `SmoothSphereHalfSpaceForce`
      (Moco), classic `HuntCrossleyForce` point contact, and
      `ElasticFoundationForce` mesh contact — faithful to the SimTK
      implementations.
- [x] Subject scaling (`newton.opensim.ScaleTool` / `ModelScaler` /
      `MarkerPlacer`): body-segment scaling from marker-distance measurements,
      static-pose marker placement, `.trc`/`.c3d` marker I/O, and virtual-marker
      synthesis (validated end-to-end on gait2354 subject01).
- [x] On-device whole-body analysis primitives on `ForwardKinematics`
      (`center_of_mass`, `body_velocities`, `body_accelerations`,
      `body_jacobian`, `whole_body_momentum`, batched variants) — the building
      blocks the OpenSim `Analyze` outputs are assembled from.
- [~] Forward Dynamics state I/O and muscle-driven runs. `.sto`/`.mot` state
      **output** is done (`FDResult.write_sto` / `to_storage`,
      `ForwardDynamics.simulate` / `simulate_batch` / `solve_from_motion`); a
      closed-loop **muscle-driven** forward run (muscle forces → body spatial
      loads through the skeleton, with contact) is still **pending**.
- [x] Static Optimization (per-frame muscle-force distribution QP).
      `newton.opensim.StaticOptimization` / `solve_static_optimization`
      (`static_optimization.py`): the rigid-tendon muscle force is affine in
      activation (`F = a A + P`), so each frame is a bound-constrained QP
      (`sum a**p` objective, moment balance `R (A a + P) + reserves = tau_ID`,
      `0<=a<=1`) solved with SLSQP and warm-started across frames; optional
      reserve/residual coordinate actuators keep infeasible frames solvable.
      **Validated: recovers the analytic least-norm muscle split (and its
      passive-offset variant) to 1e-6, reserves close a saturated balance, and
      the whole-model pipeline holds the moment balance to <1e-4.**
- [~] Analyses: `BodyKinematics`, `MuscleAnalysis`, `JointReaction`. The
      underlying kinematic/kinetic quantities exist on device (see the whole-body
      primitives above and `MuscleForces`). The OpenSim `Analyze`-style tool
      wrappers (`BodyKinematics`, `MuscleAnalysis`, `JointReaction` in
      `newton/_src/opensim/analyze.py`, re-exported from `newton.opensim`) and
      their `.sto` reports are **done** (tests in `test_opensim_analyze.py`).

### Phase 4 — Moco (optimal control)

- [x] Direct-collocation trajectory-optimization core (`collocation.py`):
      separated Hermite-Simpson transcription (matching Moco's CasOC defect
      equations) + exact-Hessian SQP with an l1-merit line search. Public API
      `DirectCollocationSolver`, `OptimalControlProblem`, `OptimalControlSolution`,
      `solve_optimal_control`. Physics-based dynamics from the Warp forward-dynamics
      engine via `create_torque_driven_dynamics`. Validated 1-for-1 against
      OpenSim Moco's analytic benchmarks (`testMocoAnalytic`): Kirk second-order
      minimum effort (states to 2e-7, bar 1e-5), Bryson-Ho linear tangent steering
      (control to 1e-10, bar 1e-3), the minimum-effort double integrator (5.8e-11),
      and the same double integrator driven through a `SliderJoint` model's Warp
      forward dynamics (2e-9).
- [x] Box bounds on states and controls and free-final-time / minimum-time
      problems (`control_bounds`, `state_bounds`, `final_time_bounds`,
      `minimize_final_time`) via a primal-dual interior-point path. Reproduces
      OpenSim Moco's minimum-time sliding-mass benchmark (bang-bang, final time
      2 s) 1-for-1.
- [x] Equality path constraints `g(t, x, u) = 0` (a `MocoPathConstraint`)
      via `OptimalControlProblem.path_constraints`, enforced at every
      Hermite-Simpson collocation point through the interior-point solver with an
      exact constraint Jacobian and Lagrangian Hessian. This is the muscle-
      equilibrium mechanism an inverse muscle problem needs (muscle moments =
      inverse-dynamics net moments). Validated on a redundant muscle-sharing
      problem (recovers the analytic least-norm excitations) and a fully-
      determined coordinate (tracks its prescribed activation exactly).
- [x] `MocoInverse` and `MocoTrack` (`newton/_src/opensim/moco.py`,
      re-exported from `newton.opensim`; tests in `test_opensim_moco.py`).
      `MocoInverse` resolves muscle redundancy per node (least effort
      reproducing the ID moments via the affine rigid-tendon force) and recovers
      excitations by inverting the first-order activation dynamics; validated to
      recover the exact single-muscle activation, the redundant least-norm
      solution, reproduce the ID moments, and its excitations reproduce the
      activation trajectory under forward dynamics. `MocoTrack` tracks a
      reference coordinate trajectory with a torque-driven model (state tracking
      + control effort) to sub-mrad RMS. **done.**

  Historical scope note: assembling `MocoInverse` (prescribed kinematics ->
      inverse-dynamics net moments -> solve for muscle activations minimizing
      integral excitation), validated against OpenSim's shipped
      `std_testMocoInverse_solution.sto`.
- [ ] Reproduce `example2DWalking` predictive/tracking problems.

### Phase 5 — Fidelity, coverage, validation

- [x] Legacy `.osim` (< v30000, `SimbodyEngine`) parsing (inline `<Joint>`,
      scalar inertia tags, `<parent_body>`/`location_in_parent`).
- [x] `.osim` round-trip writer (`write_osim`/`osim_to_xml`; round-trips `parse_osim`, gait2354 exact).
- [x] Full `CustomJoint` coupled coordinate functions (faithful `SimmSpline`
      port matching `OpenSim::SimmSpline`, `MultiplierFunction`, `LinearFunction`).
- [~] Numerical validation vs OpenSim reference outputs (`.sto`): IK on
      gait2354, ID on arm26 + gait2354 subject01 done; gait10dof18musc,
      Rajagopal2015 pending.
- [x] Example: `example_opensim_arm` (muscle-driven forward-dynamics elbow; flexor lifts the forearm).
- [x] Example: `example_opensim_muscle_activity` (gait2354 walk with muscles colored by live Static-Optimization activation via `MotionVisualizer.color_muscles_by`).
- [x] Example: `example_opensim_gait2d` (planar muscle-driven leg swing; closed-loop muscle dynamics with coordinate limit forces via the `coordinate_controls` hook + activation-colored muscles).
- [x] Example: `example_opensim_contact_hop` (planar leg hopping in place; foot-ground contact fed back into the forward integrator via `simulate_muscle_driven(contact=True)` closes the loop -- crouch/push-off/flight, peak GRF ~3.5x body weight).
- [x] Example: `example_opensim_shoe_material` (3D shoe sole as an `ElasticFoundationForce` triangle-mesh foundation on the foot; drop-landing closes the contact loop; `--material-sweep` perturbs stiffness/dissipation/friction +/-10-50% from the calibrated fit and reports kinematic (base/joint RMSE) and kinetic (peak GRF, loading rate, impulse) deviations).

### Phase 6 — Visualization

- [x] Warp-native motion visualization (`newton.opensim.MotionVisualizer` /
      `read_motion`). A coordinate trajectory (a `.mot`/`.sto` motion or an IK
      result) is turned into per-frame renderables for a Newton
      :class:`~newton.viewer.ViewerBase`, all precomputed on the Warp device:
      - **Body transforms** from the OpenSim-exact forward kinematics
        (`fk_kernel`), so the rendered skeleton reproduces the `CustomJoint`
        `SpatialTransform` coupling (e.g. the gait2354 `SimmSpline` knee
        translation) that Newton's generic D6 joints do not. A gather kernel
        converts the `float64` pose matrices to `transformf` aligned to any
        Newton model's `body_label`, ready to copy into `State.body_q`.
      - **Skeleton bones** spanning each joint's parent/child body origins
        (`_bone_kernel`).
      - **Muscle-tendon paths** as poly-lines through the active `GeometryPath`
        points (`_world_points_kernel` + `_segments_kernel`, reusing the muscle
        path point sampling), colored by normalized muscle-tendon length
        (`_muscle_color_kernel`) so lengthening muscles light up over the stride.
      Validated self-contained (body transforms match FK, muscle segments match
      the ground-space path geometry, colors track normalized length) plus a
      headless render smoke test; shipped as the `example_opensim_gait` example
      playing back a gait2354 walking trial (23 DOF, 54 muscles). An optional
      solid-bone mode skins the model's actual OpenSim `.vtp` display meshes to
      the same OpenSim-exact body poses (see the mesh item below).
- [ ] Muscle activation / force coloring (drive muscle color from a
      `MuscleForces` / static-optimization / Moco activation channel instead of
      length).
- [x] Optional import of OpenSim `Geometry/` `.vtp` display meshes for a solid
      (skinned) skeleton in place of the stick figure
      (`MotionVisualizer.load_meshes` / `render_meshes`). The model's per-body
      display geometry (subject-specific scale factors + body-frame offsets) is
      parsed with `read_display_geometry`; ASCII VTK PolyData `.vtp` files are
      read with a dependency-free reader (`_read_vtp`, stdlib `xml.etree`),
      triangulated, baked into the body frame, and rigidly skinned to the
      OpenSim-exact per-frame body pose in a Warp kernel (`_skin_kernel`).
      `fetch_opensim_geometry` pulls the standard OpenSim bone meshes
      (`opensim-org/opensim-models`, pinned by commit) on demand, and the
      `example_opensim_gait` example exposes it via `--download-geometry` /
      `--geometry <dir>`. Validated offline (VTP triangulation, display-geometry
      parsing, and kernel skinning against the FK pose).
- [ ] Ground-reaction / external-load arrow overlays (`ExternalLoads`) and
      center-of-mass / center-of-pressure markers during a gait cycle.

## Validation strategy

For each phase, gate on numerical agreement with OpenSim reference data shipped in
opensim-core (`*.sto`, `*.mot`) within documented tolerances, plus unittest
coverage for parser/importer edge cases. Regression tests must fail without the
corresponding fix.

## Current status

**Unified trunk.** `jkuzmeski/opensim/main` now holds the merged port (the
former `warp-kernels` on-device force/actuator/muscle work + the `port`
contact / subject-scaling / visualization work on one branch). Beyond the
tools detailed below, the trunk implements: `PathWrap` over cylinder/sphere
surfaces; the full actuator set (Coordinate/Point/Torque/Body); spring,
ligament, and bushing force elements; on-device activation-dynamics
integration and elastic-tendon equilibrium/fiber kinematics; OpenSim
compliant contact forces (Smooth / HuntCrossley / ElasticFoundation);
subject scaling (`ScaleTool`); and on-device whole-body analysis primitives
(COM, momentum, body velocities/accelerations, body Jacobians). The full
opensim test suite passes (186 tests; 4 opt-in skips need external
opensim-core data dirs). Static Optimization, `MocoInverse`/`MocoTrack`, and the
`Analyze` tool wrappers are now implemented. Largest remaining gaps:
muscle-driven closed-loop forward simulation, controllers
(`PrescribedController`/`ControlSet`), a `.osim` writer, and the arm26/gait2d
examples.

Phase 0 complete: parse + import + finalize verified on `2D_gait.osim`
(12 bodies, 12 joints, 18 muscles, 5 contact geometries) and a minimal synthetic
pendulum model. Muscle curves validated for physiological correctness
(force-length peak at optimal length; force-velocity = 1 at zero velocity;
rigid-tendon isometric force reproduces `max_isometric_force`).

**Inverse kinematics — 1-for-1 with OpenSim (3D motion capture).** The
OpenSim-exact forward-kinematics engine (`CustomJoint` `SpatialTransform` with a
faithful `SimmSpline` port) plus a native marker-fit Levenberg-Marquardt solver
reproduce OpenSim's own gait2354 synthetic-marker IK regression
(`Applications/IK/test`): the 23 recovered coordinates (`subject01_simbody.osim`,
31 markers, coupled-knee SimmSplines) match the `std_subject01_walk1_ik.mot`
reference with worst per-coordinate error **0.016 deg** and worst RMS
**0.0077 deg**, versus OpenSim's own `testIK` pass bar of 0.2 deg (RMS < 0.1 deg)
— a ~12x margin. Model marker positions reproduce the synthetic `.trc` to
< 0.15 mm across all frames. Reproduce with the opt-in test
`TestGait2354InverseKinematics` by setting `NEWTON_OPENSIM_GAIT2354` to the
opensim-core `Applications/IK/test` data directory.

**Inverse dynamics — 1-for-1 with OpenSim.** The `InverseDynamicsTool` pipeline
is reproduced exactly: reflect-and-negate padding (`Storage::pad`), a zero-lag
6 Hz Butterworth low-pass (`Signal::LowpassIIR`, forward + reverse), quintic
GCVSpline differentiation (a faithful NumPy port of Woltring's `gcvspl.c`) for
q/qd/qdd, then a Newton-Euler inverse-dynamics core in float64 Warp kernels. The
core reuses the batched forward kinematics: per frame it evaluates the pose, a
velocity/acceleration finite-difference stencil, and one ±eps perturbation per
coordinate in a single launch, forms each body's spatial force (inertial minus
gravity minus external loads), and projects them onto the coordinates with the
transpose of the geometric Jacobian. On OpenSim's own `Applications/ID/test`
regressions the joint moments match the references within OpenSim's tolerances:
**arm26** (no external loads) to worst **0.0014 N·m** (`testID` bar 1e-2), and
the **gait2354 subject01** walk with experimental ground reactions
(`ExternalLoads`) to worst **1.99 N·m** (`testID` bar 2.0). Reproduce with the
opt-in test `TestOpenSimInverseDynamics` by setting `NEWTON_OPENSIM_ID` to the
opensim-core `Applications/ID/test` data directory.

**Forward dynamics — 1-for-1 with OpenSim.** The equations of motion
:math:`M(q)\,\ddot q + b(q,\dot q) = \tau` are solved by reusing the
inverse-dynamics kernels: since the Newton-Euler inverse dynamics is affine in
the accelerations, the bias is :math:`b=\mathrm{ID}(q,\dot q,0)` and each mass
column is :math:`M_{:,i}=\mathrm{ID}(q,\dot q,e_i)-b` (composite rigid-body
method). All of these evaluations run in the same Warp kernels; only the small
dense solve and the RK4/symplectic-Euler time stepping run on the host.
Validated against OpenSim's own `Applications/Forward/test` pendulum: released
from :math:`-\pi/20` at rest, the integrated motion tracks the analytic
simple-harmonic-oscillator reference to worst **1.05e-3 rad** (OpenSim's
`testForward` bar is 1e-2). Forward dynamics also inverts the OpenSim-validated
inverse dynamics to worst **~1e-5** across the 23-DOF gait2354 model. Reproduce
with the always-on `TestForwardDynamics`.

**Gait visualization — Warp-native.** `newton.opensim.MotionVisualizer` plays a
coordinate trajectory back through the OpenSim-exact forward kinematics and
rebuilds the skeleton and every muscle-tendon path each frame entirely in Warp
kernels (body pose -> `transformf` gather, bone segments, world-space muscle path
points, segment assembly, and normalized-length coloring). The shipped
`example_opensim_gait` example animates a gait2354 walking trial (23 DOF, 54
muscles, 211 frames) with a length-colored musculoskeletal figure; a
self-contained unittest (`TestMotionVisualizer`) checks the body transforms
against the FK, the muscle segments against the ground-space path geometry, the
color mapping, and a headless render pass. An opt-in mode
(`MotionVisualizer.load_meshes` / `render_meshes`, `--download-geometry`) renders
the model's actual OpenSim bone meshes: the `.vtp` display geometry (with its
subject-specific scale) is parsed and rigidly skinned to the same OpenSim-exact
body poses in a Warp kernel, giving a solid, correctly-scaled skeleton.

**Not yet started: `MocoInverse`.** A prior spike left an empty
`inverse_muscle.py` importing `MocoInverse`/`MocoInverseSolution`, which broke the
package import; the dangling imports were reverted so `newton.opensim` imports
cleanly again. The direct-collocation core (with equality path constraints) that
`MocoInverse` will build on is in place — see Phase 4.


## Session status (final)

All targeted OpenSim-port tool categories are implemented, tested, and committed on
`jkuzmeski/opensim/main`: Static Optimization, MocoInverse/MocoTrack, muscle-driven
closed-loop forward sim, Analyze tool wrappers, PrescribedController/ControlSet, the
`.osim` writer (round-trip validated), and WrapEllipsoid/WrapTorus. Two showcase
examples ship with 320x320 screenshots and `test_final`:

- `opensim_arm` - muscle-driven forward-dynamics elbow (flexor lifts the forearm).
- `opensim_muscle_activity` - gait2354 walk with muscles colored by live
  Static-Optimization activation (`MotionVisualizer.color_muscles_by`).
- `opensim_gait2d` - planar (2D) two-segment leg driven purely by four Thelen
  muscles; a gait swing phase with coordinate limit forces (via the
  `coordinate_controls` hook, i.e. CoordinateLimitForce behavior) and
  activation-colored muscle paths.

Deferred: a *predictive* full-body 2D walking gait optimization (OpenSim `example2DWalking`). A live
direct-collocation gait optimization runs for minutes (per-frame SLSQP ~4.6 s/frame
even on GPU) and there is no bundled 2D gait asset, so it is unsuitable as a runnable
example. The `opensim_muscle_activity` example covers the "gait + analysis viz
overlay" intent using the real gait2354 model and the ported Static-Optimization tool.
