# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Newton-native OpenSim support.

This module ports the OpenSim (`opensim-core <https://github.com/opensim-org/opensim-core>`_)
musculoskeletal modeling stack to be Warp/Newton native. Parsed models and exact
analytic kernels retain OpenSim's native Y-up coordinates. Newton-facing imports,
visualization poses, and public contact vectors use Newton's standard Z-up world
by default through :class:`OsimFrameConverter`.

It provides:

- :func:`parse_osim` — parse an OpenSim ``.osim`` document (4.x sockets or legacy
  ``Version < 30000`` inline joints) into a solver-agnostic :class:`OsimModel`.
- :func:`write_osim` / :func:`osim_to_xml` — serialize an :class:`OsimModel` back
  to an OpenSim 4.x ``.osim`` document (round-trips :func:`parse_osim`).
- :func:`add_osim` — build the model's bodies, joints, and contact geometry into a
  :class:`~newton.ModelBuilder`, returning an :class:`OsimImportResult`.
- :class:`ForwardKinematics` — OpenSim-exact forward kinematics (``CustomJoint``
  ``SpatialTransform`` with faithful ``SimmSpline`` coordinate coupling).
- :class:`InverseKinematics` / :func:`solve_marker_ik` — marker-based inverse
  kinematics from 3D motion-capture ``.trc`` data, reproducing OpenSim IK results.
- :class:`InverseDynamics` / :func:`solve_inverse_dynamics` — joint moments from a
  coordinate trajectory (Butterworth filtering, GCVSpline differentiation, and
  optional :class:`ExternalLoads` ground reactions), reproducing OpenSim ID results.
- :class:`ForwardDynamics` / :func:`solve_forward_dynamics` — accelerations from
  applied generalized forces and time integration of the equations of motion,
  reproducing OpenSim ForwardTool results.
- :class:`MusclePaths` / :func:`compute_muscle_moment_arms` — muscle-tendon
  lengths, lengthening velocities, and moment arms from a ``GeometryPath`` (fixed,
  conditional, and moving path points), matching OpenSim's ``GeometryPath`` length
  and ``r = -dL/dq`` moment-arm definition.
- :class:`BodyActuators` — generalized forces from body (spatial-force) actuators.
- :class:`CoordinateActuators` — generalized forces from non-muscle coordinate actuators.
- :class:`SpatialActuators` — generalized forces from point and torque actuators.
- :class:`MuscleForces` / :func:`compute_muscle_generalized_forces` — rigid-tendon
  De Groote-Fregly (2016) muscle forces and the generalized (joint) forces they
  apply through the moment arms (``tau = r * F``).
- :class:`OpenSimContact` — Warp-native compliant contact forces
  (``SmoothSphereHalfSpaceForce``, ``HuntCrossleyForce``, ``ElasticFoundationForce``),
  evaluating body wrenches and generalized forces from a coordinate state,
  reproducing OpenSim/Simbody's contact force laws.
- :class:`DirectCollocationSolver` / :func:`solve_optimal_control` — direct-collocation
  trajectory optimization (separated Hermite-Simpson transcription with an SQP
  solver, plus an interior-point path for box bounds on states and controls and
  free-final-time / minimum-time problems), porting the core of OpenSim Moco and
  reproducing its analytic optimal-control benchmarks.
- :class:`StaticOptimization` / :func:`solve_static_optimization` — per-frame
  muscle-redundancy resolution (least muscle effort reproducing the
  inverse-dynamics moments), reproducing OpenSim's Static Optimization.
- :class:`MuscleMetabolicsBhargava2004` — Bhargava (2004) metabolic power and cost-of-transport
  estimation, matching OpenSim's piecewise or tanh-smoothed rate equations.
- :class:`MocoInverse` / :func:`solve_moco_inverse` — muscle excitations from
  prescribed kinematics (effort-minimizing redundancy resolution with
  activation-dynamics inversion), and :class:`MocoTrack` for coordinate
  tracking, porting OpenSim Moco's inverse and tracking tools.
- :class:`MuscleDrivenForward` / :func:`simulate_muscle_driven` — muscle-driven
  closed-loop forward simulation (excitation -> activation dynamics -> Hill-type
  muscle force -> multibody forward dynamics), and :class:`PrescribedController` /
  :class:`ControlSet` for prescribed actuator controls.
- :class:`BodyKinematics`, :class:`MuscleAnalysis`, :class:`JointReaction` —
  OpenSim Analyze-tool reports (body kinematics, muscle length/force/moment-arm,
  and joint reaction loads) over a motion, written to ``.sto``.
- :class:`MotionVisualizer` — turn an OpenSim coordinate trajectory (a ``.mot``
  motion or IK result) into per-frame Warp renderables (OpenSim-exact body
  transforms, skeleton bones, and length-colored muscle paths) for a Newton
  viewer.
- ``.trc`` and ``.mot``/``.sto`` I/O (:func:`read_trc`, :func:`read_storage`, ...).
- Warp-native Hill-type muscle-tendon curves (``muscle``),
  including the differentiable De Groote-Fregly (2016) and Thelen (2003) models.

.. experimental::

    The OpenSim port is under active development. The API may change without
    prior notice, and only a subset of OpenSim components is currently supported.
    Feedback and contributions are welcome.
"""

from ._src.opensim import (
    GAIT2354_VICON_ALIASES,
    GAIT2354_VIRTUAL_MARKERS,
    BodyActuators,
    BodyKinematics,
    BushingForces,
    ControlSet,
    CoordinateActuators,
    DirectCollocationSolver,
    ExternalForce,
    ExternalLoads,
    FDBatchResult,
    FDResult,
    ForwardDynamics,
    ForwardKinematics,
    IDResult,
    IKResult,
    InverseDynamics,
    InverseKinematics,
    JointReaction,
    LigamentForces,
    MarkerData,
    MarkerPair,
    MarkerPlacementResult,
    MarkerPlacer,
    Measurement,
    MocoInverse,
    MocoInverseSolution,
    MocoTrack,
    MocoTrackSolution,
    ModelScaler,
    MotionVisualizer,
    MuscleAnalysis,
    MuscleDrivenForward,
    MuscleForces,
    MuscleForwardResult,
    MuscleMetabolicsBhargava2004,
    MuscleMetabolicsBhargava2004Parameters,
    MuscleMetabolicsBhargava2004Result,
    MusclePaths,
    OpenSimContact,
    OptimalControlProblem,
    OptimalControlSolution,
    OsimActuator,
    OsimBody,
    OsimBushingForce,
    OsimContactForce,
    OsimContactGeometry,
    OsimCoordinate,
    OsimFrame,
    OsimFrameConverter,
    OsimGeometry,
    OsimImportResult,
    OsimJoint,
    OsimLigament,
    OsimMarker,
    OsimModel,
    OsimMuscle,
    OsimMuscleModel,
    OsimPathPoint,
    OsimPathSpring,
    OsimPointToPointSpring,
    OsimSpringGeneralizedForce,
    OsimTransform,
    OsimTransformAxis,
    OsimWrap,
    OsimWrapObject,
    PathSpringForces,
    PointToPointSprings,
    PrescribedController,
    ScaleResult,
    ScaleTool,
    SOResult,
    SpatialActuators,
    SpringGeneralizedForces,
    StaticOptimization,
    Storage,
    add_osim,
    apply_marker_assignment,
    compute_muscle_generalized_forces,
    compute_muscle_moment_arms,
    create_torque_driven_dynamics,
    euler_xyz_from_matrix,
    fetch_opensim_geometry,
    gait2354_measurement_set,
    lab_to_opensim_rotation,
    osim_to_xml,
    parse_osim,
    read_c3d,
    read_display_geometry,
    read_external_loads,
    read_motion,
    read_storage,
    read_trc,
    simulate_muscle_driven,
    solve_forward_dynamics,
    solve_frame_activations,
    solve_inverse_dynamics,
    solve_marker_ik,
    solve_moco_inverse,
    solve_optimal_control,
    solve_static_optimization,
    suggest_marker_assignment,
    synthesize_markers,
    write_osim,
    write_storage,
    write_trc,
)

__all__ = [
    "GAIT2354_VICON_ALIASES",
    "GAIT2354_VIRTUAL_MARKERS",
    "BodyActuators",
    "BodyKinematics",
    "BushingForces",
    "ControlSet",
    "CoordinateActuators",
    "DirectCollocationSolver",
    "ExternalForce",
    "ExternalLoads",
    "FDBatchResult",
    "FDResult",
    "ForwardDynamics",
    "ForwardKinematics",
    "IDResult",
    "IKResult",
    "InverseDynamics",
    "InverseKinematics",
    "JointReaction",
    "LigamentForces",
    "MarkerData",
    "MarkerPair",
    "MarkerPlacementResult",
    "MarkerPlacer",
    "Measurement",
    "MocoInverse",
    "MocoInverseSolution",
    "MocoTrack",
    "MocoTrackSolution",
    "ModelScaler",
    "MotionVisualizer",
    "MuscleAnalysis",
    "MuscleDrivenForward",
    "MuscleForces",
    "MuscleForwardResult",
    "MuscleMetabolicsBhargava2004",
    "MuscleMetabolicsBhargava2004Parameters",
    "MuscleMetabolicsBhargava2004Result",
    "MusclePaths",
    "OpenSimContact",
    "OptimalControlProblem",
    "OptimalControlSolution",
    "OsimActuator",
    "OsimBody",
    "OsimBushingForce",
    "OsimContactForce",
    "OsimContactGeometry",
    "OsimCoordinate",
    "OsimFrame",
    "OsimFrameConverter",
    "OsimGeometry",
    "OsimImportResult",
    "OsimJoint",
    "OsimLigament",
    "OsimMarker",
    "OsimModel",
    "OsimMuscle",
    "OsimMuscleModel",
    "OsimPathPoint",
    "OsimPathSpring",
    "OsimPointToPointSpring",
    "OsimSpringGeneralizedForce",
    "OsimTransform",
    "OsimTransformAxis",
    "OsimWrap",
    "OsimWrapObject",
    "PathSpringForces",
    "PointToPointSprings",
    "PrescribedController",
    "SOResult",
    "ScaleResult",
    "ScaleTool",
    "SpatialActuators",
    "SpringGeneralizedForces",
    "StaticOptimization",
    "Storage",
    "add_osim",
    "apply_marker_assignment",
    "compute_muscle_generalized_forces",
    "compute_muscle_moment_arms",
    "create_torque_driven_dynamics",
    "euler_xyz_from_matrix",
    "fetch_opensim_geometry",
    "gait2354_measurement_set",
    "lab_to_opensim_rotation",
    "osim_to_xml",
    "parse_osim",
    "read_c3d",
    "read_display_geometry",
    "read_external_loads",
    "read_motion",
    "read_storage",
    "read_trc",
    "simulate_muscle_driven",
    "solve_forward_dynamics",
    "solve_frame_activations",
    "solve_inverse_dynamics",
    "solve_marker_ik",
    "solve_moco_inverse",
    "solve_optimal_control",
    "solve_static_optimization",
    "suggest_marker_assignment",
    "synthesize_markers",
    "write_osim",
    "write_storage",
    "write_trc",
]
