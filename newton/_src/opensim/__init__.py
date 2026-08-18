# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Newton-native OpenSim support (``.osim`` parsing, import, kinematics, IK).

This package ports the OpenSim (opensim-core) musculoskeletal modeling stack to
be Warp/Newton native. Public symbols are re-exported from :mod:`newton.opensim`.
"""

from .actuators import BodyActuators, CoordinateActuators, SpatialActuators
from .analyze import BodyKinematics, JointReaction, MuscleAnalysis, euler_xyz_from_matrix
from .collocation import (
    DirectCollocationSolver,
    OptimalControlProblem,
    OptimalControlSolution,
    create_torque_driven_dynamics,
    solve_optimal_control,
)
from .contact import OpenSimContact
from .controllers import ControlSet, PrescribedController
from .dynamics import (
    ExternalForce,
    ExternalLoads,
    FDBatchResult,
    FDResult,
    ForwardDynamics,
    IDResult,
    InverseDynamics,
    read_external_loads,
    solve_forward_dynamics,
    solve_inverse_dynamics,
)
from .forces import BushingForces, PointToPointSprings, SpringGeneralizedForces
from .frame import OsimFrameConverter
from .ik import IKResult, InverseKinematics, solve_marker_ik
from .importer import OsimImportResult, OsimMuscleModel, add_osim
from .kinematics import ForwardKinematics
from .metabolics import (
    MuscleMetabolicsBhargava2004,
    MuscleMetabolicsBhargava2004Parameters,
    MuscleMetabolicsBhargava2004Result,
)
from .mocap import MarkerData, Storage, read_storage, read_trc, write_storage, write_trc
from .moco import MocoInverse, MocoInverseSolution, MocoTrack, MocoTrackSolution, solve_moco_inverse
from .muscle_force import LigamentForces, MuscleForces, PathSpringForces, compute_muscle_generalized_forces
from .muscle_forward import MuscleDrivenForward, MuscleForwardResult, simulate_muscle_driven
from .muscle_path import MusclePaths, compute_muscle_moment_arms
from .parser import parse_osim
from .scale import (
    GAIT2354_VICON_ALIASES,
    GAIT2354_VIRTUAL_MARKERS,
    MarkerPair,
    MarkerPlacementResult,
    MarkerPlacer,
    Measurement,
    ModelScaler,
    ScaleResult,
    ScaleTool,
    apply_marker_assignment,
    gait2354_measurement_set,
    lab_to_opensim_rotation,
    read_c3d,
    suggest_marker_assignment,
    synthesize_markers,
)
from .static_optimization import (
    SOResult,
    StaticOptimization,
    solve_frame_activations,
    solve_static_optimization,
)
from .types import (
    OsimActuator,
    OsimBody,
    OsimBushingForce,
    OsimContactForce,
    OsimContactGeometry,
    OsimCoordinate,
    OsimFrame,
    OsimGeometry,
    OsimJoint,
    OsimLigament,
    OsimMarker,
    OsimModel,
    OsimMuscle,
    OsimPathPoint,
    OsimPathSpring,
    OsimPointToPointSpring,
    OsimSpringGeneralizedForce,
    OsimTransform,
    OsimTransformAxis,
    OsimWrap,
    OsimWrapObject,
)
from .visualize import MotionVisualizer, fetch_opensim_geometry, read_display_geometry, read_motion
from .writer import osim_to_xml, write_osim

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
