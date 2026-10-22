# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Body parameters and the Newton articulation of the SuperMimic runner.

World z is up, x forward and y left. The pelvis carries head, arms and trunk
and has a free root at the mid-hip center; each leg is a thigh, shank and foot.
Every body origin is its proximal joint center, and at zero joint angles every
body frame has the pelvis orientation. Ten joints are actuated, in the order of
:data:`JOINT_NAMES`. :class:`BodyParams` describes the right side; the left side
is its mirror image in y.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

import newton

__all__ = [
    "BODY_NAMES",
    "JOINT_COUNT",
    "JOINT_LIMIT_DAMPING",
    "JOINT_NAMES",
    "STATE_DIM",
    "BodyParams",
    "SegmentParams",
    "add_body",
    "mirror_axis",
]

BODY_NAMES = ("pelvis", "thigh_r", "shank_r", "foot_r", "thigh_l", "shank_l", "foot_l")
"""Body labels in builder order."""

JOINT_NAMES = (
    "hip_flexion_r",
    "hip_adduction_r",
    "hip_rotation_r",
    "knee_flexion_r",
    "ankle_dorsiflexion_r",
    "hip_flexion_l",
    "hip_adduction_l",
    "hip_rotation_l",
    "knee_flexion_l",
    "ankle_dorsiflexion_l",
)
"""Actuated joint coordinates in generalized-coordinate order, after the free root."""

JOINT_COUNT = len(JOINT_NAMES)
"""Number of actuated joints."""

JOINT_LIMIT_DAMPING = 100.0
"""Damping of every leg joint limit [N m s/rad].

With Newton's default 10 N m s/rad the soft limits return more energy than
they absorb when a stiff joint torque presses into them.
"""

STATE_DIM = 13 + 2 * JOINT_COUNT
"""Width of a runner state row: root position (3), root quaternion xyzw (4), root linear and
angular velocity (6), joint angles and joint speeds in :data:`JOINT_NAMES` order."""


@dataclass(frozen=True)
class SegmentParams:
    """Inertial properties of one body."""

    mass: float
    """Mass [kg]."""
    com: tuple[float, float, float]
    """Center of mass in the body frame [m]."""
    inertia: tuple[float, float, float]
    """Principal moments of inertia about the center of mass along the body axes [kg m^2]."""


# Adult range of motion (American Academy of Orthopaedic Surgeons, Greene and
# Heckman, "The Clinical Measurement of Joint Motion", 1994), widened by 10 deg
# for running; knee extension and ankle dorsiflexion by 10 and 20 deg so that
# the F01 stance stays clear of them.
_JOINT_LIMITS = tuple(
    (float(np.radians(lower)), float(np.radians(upper)))
    for lower, upper in ((-40.0, 130.0), (-55.0, 40.0), (-55.0, 55.0), (-10.0, 145.0), (-60.0, 40.0))
)


@dataclass(frozen=True)
class BodyParams:
    """Geometry and inertia of the runner. Leg values are for the right side.

    The defaults are plausible placeholders for a 66.5 kg runner, not subject
    measurements.
    """

    pelvis: SegmentParams = SegmentParams(45.1, (0.0, 0.0, 0.33), (3.05, 3.05, 0.65))
    """Pelvis with head, arms and trunk."""
    thigh: SegmentParams = SegmentParams(6.65, (0.0, 0.0, -0.18), (0.122, 0.122, 0.025))
    shank: SegmentParams = SegmentParams(3.09, (0.0, 0.0, -0.186), (0.052, 0.052, 0.005))
    foot: SegmentParams = SegmentParams(0.96, (0.05, 0.0, -0.04), (0.001, 0.004, 0.004))
    hip_offset: tuple[float, float, float] = (0.0, -0.085, 0.0)
    """Right hip center in the pelvis frame [m]."""
    knee_offset: tuple[float, float, float] = (0.0, 0.0, -0.42)
    """Knee center in the thigh frame [m]."""
    ankle_offset: tuple[float, float, float] = (0.0, 0.0, -0.43)
    """Ankle center in the shank frame [m]."""
    hip_axes: tuple[tuple[float, float, float], ...] = ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    """Right hip flexion, adduction and internal rotation axes, composed in that order."""
    knee_axis: tuple[float, float, float] = (0.0, 1.0, 0.0)
    """Right knee flexion axis."""
    ankle_axis: tuple[float, float, float] = (0.0, -1.0, 0.0)
    """Right ankle dorsiflexion axis."""
    sole_pitch: float = 0.0
    """Rotation about the foot y axis from the foot frame to the measured shoe's frame [rad]."""
    joint_limits: tuple[tuple[float, float], ...] = _JOINT_LIMITS
    """Lower and upper angle of the five right joints in :data:`JOINT_NAMES` order [rad]; the left side is equal."""

    @property
    def mass(self) -> float:
        """Total body mass [kg]."""
        return self.pelvis.mass + 2.0 * (self.thigh.mass + self.shank.mass + self.foot.mass)


def mirror_axis(axis) -> np.ndarray:
    """Mirror a rotation axis of the right side in y onto the left side."""
    return np.asarray(axis, dtype=float) * (-1.0, 1.0, -1.0)


def _mirror_point(point, side: str) -> np.ndarray:
    return np.asarray(point, dtype=float) * ((1.0, -1.0, 1.0) if side == "l" else 1.0)


def _add_segment(builder: newton.ModelBuilder, segment: SegmentParams, side: str, label: str) -> int:
    return builder.add_link(
        mass=segment.mass,
        com=wp.vec3(*_mirror_point(segment.com, side)),
        inertia=wp.mat33(*np.diag(segment.inertia).ravel()),
        lock_inertia=True,
        label=label,
    )


def add_body(builder: newton.ModelBuilder, params: BodyParams) -> None:
    """Add one runner articulation with the bodies of :data:`BODY_NAMES` to ``builder``."""
    pelvis = _add_segment(builder, params.pelvis, "r", "pelvis")
    joints = [builder.add_joint_free(pelvis, label="root")]
    for s in ("r", "l"):
        axis = (lambda a: a) if s == "r" else mirror_axis
        thigh = _add_segment(builder, params.thigh, s, f"thigh_{s}")
        shank = _add_segment(builder, params.shank, s, f"shank_{s}")
        foot = _add_segment(builder, params.foot, s, f"foot_{s}")
        joints.append(
            builder.add_joint_d6(
                pelvis,
                thigh,
                angular_axes=[
                    newton.ModelBuilder.JointDofConfig(
                        axis=tuple(axis(a)), limit_lower=lo, limit_upper=hi, limit_kd=JOINT_LIMIT_DAMPING
                    )
                    for a, (lo, hi) in zip(params.hip_axes, params.joint_limits[:3], strict=True)
                ],
                parent_xform=wp.transform(wp.vec3(*_mirror_point(params.hip_offset, s)), wp.quat_identity()),
                label=f"hip_{s}",
            )
        )
        for parent, child, offset, joint_axis, (lo, hi), name in (
            (thigh, shank, params.knee_offset, params.knee_axis, params.joint_limits[3], "knee"),
            (shank, foot, params.ankle_offset, params.ankle_axis, params.joint_limits[4], "ankle"),
        ):
            joints.append(
                builder.add_joint_revolute(
                    parent,
                    child,
                    axis=tuple(axis(joint_axis)),
                    limit_lower=lo,
                    limit_upper=hi,
                    limit_kd=JOINT_LIMIT_DAMPING,
                    parent_xform=wp.transform(wp.vec3(*_mirror_point(offset, s)), wp.quat_identity()),
                    label=f"{name}_{s}",
                )
            )
    builder.add_articulation(joints)
