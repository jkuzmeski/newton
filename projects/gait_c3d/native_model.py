# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Build a deliberately simple Newton-native human gait articulation.

The model uses fixed-axis hip rotations and one revolute hinge for each knee
and ankle. It is an engineering scaffold for solver and contact experiments,
not an OpenSim-equivalent model or an accepted FD-1 result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import warp as wp

import newton

ARCHITECTURE_ROLE = "native_runtime"


@dataclass(frozen=True, slots=True)
class SimpleGaitConfig:
    """Dimensions and material values for the approximate gait articulation."""

    pelvis_height: float = 1.02
    """Initial pelvis height [m]."""

    pelvis_mass: float = 13.0
    """Pelvis mass [kg]."""

    torso_mass: float = 37.0
    """Combined torso, head, and arm mass [kg]."""

    thigh_mass: float = 10.0
    """Mass of each thigh [kg]."""

    shank_mass: float = 4.0
    """Mass of each shank [kg]."""

    foot_mass: float = 1.7
    """Mass of each foot [kg]."""

    hip_half_width: float = 0.076
    """Lateral distance from the pelvis center to either hip [m]."""

    thigh_length: float = 0.45
    """Hip-to-knee distance [m]."""

    shank_length: float = 0.40
    """Knee-to-ankle distance [m]."""

    foot_length: float = 0.25
    """Approximate heel-to-toe length [m]."""

    contact_radius: float = 0.04
    """Radius of each foot contact sphere [m]."""

    ground_ke: float = 1.0e5
    """Ground contact stiffness [N/m]."""

    ground_kd: float = 1.0e3
    """Ground contact damping [N·s/m]."""

    ground_kf: float = 1.0e3
    """Ground tangential stiffness [N/m]."""

    friction: float = 0.8
    """Ground friction coefficient."""


@dataclass(frozen=True, slots=True)
class SimpleGaitBuild:
    """Builder indices and initial coordinates for one simple gait model."""

    builder: newton.ModelBuilder
    """Newton model builder containing one articulation and a ground plane."""

    body_indices: dict[str, int]
    """Body indices keyed by anatomical label."""

    joint_indices: dict[str, int]
    """Joint indices keyed by anatomical label."""

    contact_shape_indices: tuple[int, ...]
    """Foot contact shape indices."""

    initial_joint_q: np.ndarray
    """Initial generalized coordinates [m or rad]."""

    root_dof_slice: slice
    """Six unactuated free-pelvis velocity/control entries."""

    actuated_dof_indices: tuple[int, ...]
    """Internal velocity/control entries eligible for actuation."""


def _box_inertia(mass: float, dimensions: tuple[float, float, float]) -> wp.mat33:
    """Return diagonal box inertia about its center of mass [kg·m²]."""
    x, y, z = dimensions
    return wp.mat33(
        mass * (y * y + z * z) / 12.0,
        0.0,
        0.0,
        0.0,
        mass * (x * x + z * z) / 12.0,
        0.0,
        0.0,
        0.0,
        mass * (x * x + y * y) / 12.0,
    )


def _add_body(
    builder: newton.ModelBuilder,
    label: str,
    mass: float,
    dimensions: tuple[float, float, float],
) -> int:
    """Add one link with a locked approximate box inertia."""
    return builder.add_link(
        mass=mass,
        inertia=_box_inertia(mass, dimensions),
        lock_inertia=True,
        label=label,
    )


def _joint_frame(translation: tuple[float, float, float]) -> wp.transform:
    """Construct a translated identity joint frame."""
    return wp.transform(translation, wp.quat_identity())


def build_simple_gait_model(config: SimpleGaitConfig | None = None) -> SimpleGaitBuild:
    """Build a bilateral Newton articulation with hinge knees and ankles.

    Args:
        config: Approximate dimensions and contact material values.

    Returns:
        Builder metadata and initial generalized coordinates. Call
        :meth:`newton.ModelBuilder.finalize` to create the runtime model.
    """
    config = config or SimpleGaitConfig()
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    bodies = {
        "pelvis": _add_body(builder, "pelvis", config.pelvis_mass, (0.24, 0.22, 0.16)),
        "torso": _add_body(builder, "torso", config.torso_mass, (0.32, 0.25, 0.55)),
    }
    for side in ("left", "right"):
        bodies[f"femur_{side}"] = _add_body(
            builder, f"femur_{side}", config.thigh_mass, (0.12, 0.12, config.thigh_length)
        )
        bodies[f"tibia_{side}"] = _add_body(
            builder, f"tibia_{side}", config.shank_mass, (0.09, 0.09, config.shank_length)
        )
        bodies[f"foot_{side}"] = _add_body(builder, f"foot_{side}", config.foot_mass, (config.foot_length, 0.10, 0.08))

    joints: dict[str, int] = {}
    articulation: list[int] = []
    joints["pelvis_free"] = builder.add_joint_free(child=bodies["pelvis"], label="pelvis_free")
    articulation.append(joints["pelvis_free"])
    joints["lumbar_fixed"] = builder.add_joint_fixed(
        parent=bodies["pelvis"],
        child=bodies["torso"],
        parent_xform=_joint_frame((0.0, 0.0, 0.25)),
        child_xform=_joint_frame((0.0, 0.0, -0.25)),
        label="lumbar_fixed",
    )
    articulation.append(joints["lumbar_fixed"])

    for side, lateral_sign in (("left", 1.0), ("right", -1.0)):
        degrees = math.pi / 180.0
        hip_axes = [
            builder.JointDofConfig(
                axis=(0.0, -1.0, 0.0),
                limit_lower=-30.0 * degrees,
                limit_upper=120.0 * degrees,
                limit_ke=2.0e3,
                limit_kd=50.0,
                damping=0.5,
                armature=0.01,
            ),
            builder.JointDofConfig(
                axis=(lateral_sign, 0.0, 0.0),
                limit_lower=-25.0 * degrees,
                limit_upper=45.0 * degrees,
                limit_ke=2.0e3,
                limit_kd=50.0,
                damping=0.5,
                armature=0.01,
            ),
            builder.JointDofConfig(
                axis=(0.0, 0.0, -lateral_sign),
                limit_lower=-45.0 * degrees,
                limit_upper=45.0 * degrees,
                limit_ke=2.0e3,
                limit_kd=50.0,
                damping=0.3,
                armature=0.01,
            ),
        ]
        joints[f"hip_{side}"] = builder.add_joint_d6(
            parent=bodies["pelvis"],
            child=bodies[f"femur_{side}"],
            angular_axes=hip_axes,
            parent_xform=_joint_frame((0.0, lateral_sign * config.hip_half_width, -0.05)),
            child_xform=_joint_frame((0.0, 0.0, 0.5 * config.thigh_length)),
            label=f"hip_{side}",
        )
        articulation.append(joints[f"hip_{side}"])
        joints[f"knee_{side}"] = builder.add_joint_revolute(
            parent=bodies[f"femur_{side}"],
            child=bodies[f"tibia_{side}"],
            axis=newton.Axis.Y,
            parent_xform=_joint_frame((0.0, 0.0, -0.5 * config.thigh_length)),
            child_xform=_joint_frame((0.0, 0.0, 0.5 * config.shank_length)),
            limit_lower=0.0,
            limit_upper=150.0 * degrees,
            limit_ke=2.0e3,
            limit_kd=50.0,
            damping=0.3,
            armature=0.01,
            label=f"knee_{side}",
        )
        articulation.append(joints[f"knee_{side}"])
        joints[f"ankle_{side}"] = builder.add_joint_revolute(
            parent=bodies[f"tibia_{side}"],
            child=bodies[f"foot_{side}"],
            axis=(0.0, -1.0, 0.0),
            parent_xform=_joint_frame((0.0, 0.0, -0.5 * config.shank_length)),
            child_xform=_joint_frame((-0.10, 0.0, 0.04)),
            limit_lower=-50.0 * degrees,
            limit_upper=30.0 * degrees,
            limit_ke=2.0e3,
            limit_kd=50.0,
            damping=0.2,
            armature=0.005,
            label=f"ankle_{side}",
        )
        articulation.append(joints[f"ankle_{side}"])

    builder.add_articulation(articulation)
    material = builder.ShapeConfig(
        density=0.0,
        ke=config.ground_ke,
        kd=config.ground_kd,
        kf=config.ground_kf,
        mu=config.friction,
    )
    builder.add_ground_plane(cfg=material)
    contact_shapes: list[int] = []
    contact_centers = (
        (-0.08, -0.035, -0.04),
        (-0.08, 0.035, -0.04),
        (0.12, -0.035, -0.04),
        (0.12, 0.035, -0.04),
    )
    for side in ("left", "right"):
        for sphere_index, center in enumerate(contact_centers):
            contact_shapes.append(
                builder.add_shape_sphere(
                    bodies[f"foot_{side}"],
                    xform=_joint_frame(center),
                    radius=config.contact_radius,
                    cfg=material,
                    label=f"contact_{side}_{sphere_index}",
                )
            )

    initial_q = np.asarray(builder.joint_q, dtype=np.float32)
    root_start = builder.joint_q_start[joints["pelvis_free"]]
    initial_q[root_start : root_start + 7] = np.asarray((0.0, 0.0, config.pelvis_height, 0.0, 0.0, 0.0, 1.0))
    root_dof_start = builder.joint_qd_start[joints["pelvis_free"]]
    root_dof_slice = slice(root_dof_start, root_dof_start + 6)
    root_dofs = set(range(root_dof_slice.start, root_dof_slice.stop))
    actuated_dofs = tuple(index for index in range(len(builder.joint_qd)) if index not in root_dofs)
    return SimpleGaitBuild(
        builder,
        bodies,
        joints,
        tuple(contact_shapes),
        initial_q,
        root_dof_slice,
        actuated_dofs,
    )


def initialize_simple_gait_state(model: newton.Model, build: SimpleGaitBuild) -> newton.State:
    """Create a state at the build's declared neutral pose."""
    if len(build.initial_joint_q) != model.joint_coord_count:
        raise ValueError("build coordinates do not match the finalized model")
    state = model.state()
    state.joint_q.assign(build.initial_joint_q)
    state.joint_qd.zero_()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    return state
