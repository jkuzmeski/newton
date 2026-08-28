# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Public-API synthetic marker inverse kinematics for native gait subjects."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

import newton
import newton.ik as ik


@dataclass(frozen=True, slots=True)
class NativeMarkerAttachment:
    """A marker point attached to one native body."""

    name: str
    """Stable marker label."""

    body: int
    """Native model body index."""

    local_position: tuple[float, float, float]
    """Marker position in the body frame [m]."""


@dataclass(frozen=True, slots=True)
class NativeIKFrame:
    """One solved native marker-IK frame and its diagnostics."""

    joint_q: np.ndarray
    """Solved generalized coordinates [m or rad]."""

    predicted_markers: np.ndarray
    """Predicted marker positions [m]."""

    target_markers: np.ndarray
    """Target marker positions [m]."""

    marker_rms: float
    """Root-mean-square marker error [m]."""

    marker_max: float
    """Maximum marker error [m]."""

    solver_cost: float = float("nan")
    """Weighted squared solver cost reported by public ``IKSolver``."""

    joint_limit_violation: float = float("nan")
    """Maximum bounded joint-limit violation [m or rad]."""


def marker_attachments_from_model(model: newton.Model) -> tuple[NativeMarkerAttachment, ...]:
    """Read marker attachments from imported MJCF sites."""
    flags = model.shape_flags.numpy()
    bodies = model.shape_body.numpy()
    transforms = model.shape_transform.numpy()
    attachments = []
    names = set()
    for index, label in enumerate(model.shape_label):
        name = label.rsplit("/", 1)[-1]
        if flags[index] & newton.ShapeFlags.SITE and name.startswith("marker_"):
            body = int(bodies[index])
            marker_name = name.removeprefix("marker_")
            if body < 0 or marker_name in names:
                raise ValueError("model marker sites must have valid bodies and unique names")
            names.add(marker_name)
            attachments.append(
                NativeMarkerAttachment(
                    marker_name,
                    body,
                    tuple(float(value) for value in transforms[index, :3]),
                )
            )
    if not attachments:
        raise ValueError("model contains no imported marker sites")
    return tuple(attachments)


def _rotate_point(quaternion_xyzw: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Rotate a point by an xyzw quaternion without a custom kernel."""
    q = quaternion_xyzw[:3]
    return point + 2.0 * np.cross(q, quaternion_xyzw[3] * point + np.cross(q, point))


def marker_positions_from_joint_q(
    model: newton.Model,
    attachments: tuple[NativeMarkerAttachment, ...],
    joint_q: np.ndarray,
) -> np.ndarray:
    """Evaluate native marker positions through public forward kinematics."""
    joint_q = np.asarray(joint_q, dtype=np.float32)
    if joint_q.shape != (model.joint_coord_count,):
        raise ValueError("joint_q has an incompatible coordinate shape")
    state = model.state()
    state.joint_q.assign(joint_q)
    state.joint_qd.zero_()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    body_q = state.body_q.numpy()
    positions = []
    for attachment in attachments:
        body_pose = body_q[attachment.body]
        local = np.asarray(attachment.local_position, dtype=np.float64)
        positions.append(body_pose[:3] + _rotate_point(body_pose[3:], local))
    return np.asarray(positions, dtype=np.float64)


def marker_ik_frame(
    model: newton.Model,
    attachments: tuple[NativeMarkerAttachment, ...],
    joint_q: np.ndarray,
    target_markers: np.ndarray,
) -> NativeIKFrame:
    """Evaluate marker residual diagnostics for one solved frame."""
    target_markers = np.asarray(target_markers, dtype=np.float64)
    if target_markers.shape != (len(attachments), 3) or not np.all(np.isfinite(target_markers)):
        raise ValueError("target_markers has an incompatible or nonfinite shape")
    predicted = marker_positions_from_joint_q(model, attachments, joint_q)
    errors = predicted - target_markers
    distances = np.linalg.norm(errors, axis=1)
    return NativeIKFrame(
        np.asarray(joint_q, dtype=np.float32).copy(),
        predicted,
        target_markers.copy(),
        float(np.sqrt(np.mean(np.sum(errors * errors, axis=1)))),
        float(np.max(distances)),
    )


def solve_marker_sequence(
    model: newton.Model,
    attachments: tuple[NativeMarkerAttachment, ...],
    target_sequence: np.ndarray,
    seed: np.ndarray,
    *,
    iterations: int = 80,
    joint_limit_weight: float = 0.1,
    lambda_initial: float = 0.001,
) -> tuple[NativeIKFrame, ...]:
    """Solve a marker sequence with public Newton LM and warm starts.

    Args:
        model: Finalized native model.
        attachments: Marker/body-local attachment definitions.
        target_sequence: Target marker positions [m], shape [frame, marker, 3].
        seed: Initial generalized coordinates [m or rad].
        iterations: LM iterations per frame.
        joint_limit_weight: Weight of the public joint-limit objective.
        lambda_initial: Initial LM damping value.

    Returns:
        Solved frames in input order.
    """
    target_sequence = np.asarray(target_sequence, dtype=np.float64)
    if target_sequence.ndim != 3 or target_sequence.shape[1:] != (len(attachments), 3):
        raise ValueError("target_sequence has an incompatible shape")
    if not np.all(np.isfinite(target_sequence)):
        raise ValueError("target_sequence must be finite")
    seed = np.asarray(seed, dtype=np.float32)
    if seed.shape != (model.joint_coord_count,) or not np.all(np.isfinite(seed)):
        raise ValueError("seed has an incompatible or nonfinite shape")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    target_arrays = [wp.zeros(1, dtype=wp.vec3, device=model.device) for _ in attachments]
    with wp.ScopedDevice(model.device):
        objectives = [
            ik.IKObjectivePosition(
                attachment.body,
                wp.vec3(*attachment.local_position),
                target_array,
                weight=1.0,
            )
            for attachment, target_array in zip(attachments, target_arrays, strict=True)
        ]
        joint_limits = ik.IKObjectiveJointLimit(
            model.joint_limit_lower,
            model.joint_limit_upper,
            weight=joint_limit_weight,
        )
        solver = ik.IKSolver(
            model,
            1,
            [*objectives, joint_limits],
            lambda_initial=lambda_initial,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )
        joint_q = wp.array(seed.reshape(1, -1), dtype=wp.float32, device=model.device)
        frames = []
        for target_frame in target_sequence:
            for objective, target in zip(objectives, target_frame, strict=True):
                objective.set_target_position(0, wp.vec3(*target))
            solver.step(joint_q, joint_q, iterations=iterations, step_size=1.0)
            solved = joint_q.numpy()[0].copy()
            quaternion_slice = free_root_quaternion_slice(model)
            if quaternion_slice is not None:
                quaternion = solved[quaternion_slice]
                norm = float(np.linalg.norm(quaternion))
                if not math_is_finite_positive(norm):
                    raise ValueError("IK produced a nonfinite free-root quaternion")
                solved[quaternion_slice] = quaternion / norm
                joint_q.assign(solved.reshape(1, -1))
            frame = marker_ik_frame(model, attachments, solved, target_frame)
            frames.append(
                NativeIKFrame(
                    frame.joint_q,
                    frame.predicted_markers,
                    frame.target_markers,
                    frame.marker_rms,
                    frame.marker_max,
                    float(solver.costs.numpy()[0]),
                    joint_limit_violation(model, solved),
                )
            )
    return tuple(frames)


def free_root_quaternion_slice(model: newton.Model) -> slice | None:
    """Return the free-root quaternion coordinate slice, if the model has one."""
    joint_types = model.joint_type.numpy()
    joint_parent = model.joint_parent.numpy()
    q_start = model.joint_q_start.numpy()
    for joint, joint_type in enumerate(joint_types):
        if joint_parent[joint] == -1 and joint_type == newton.JointType.FREE:
            start = int(q_start[joint])
            return slice(start + 3, start + 7)
    return None


def joint_limit_violation(model: newton.Model, joint_q: np.ndarray) -> float:
    """Return the maximum bounded joint-coordinate limit violation [m or rad]."""
    joint_q = np.asarray(joint_q, dtype=np.float64)
    if joint_q.shape != (model.joint_coord_count,):
        raise ValueError("joint_q has an incompatible coordinate shape")
    lower = model.joint_limit_lower.numpy()
    upper = model.joint_limit_upper.numpy()
    q_start = model.joint_q_start.numpy()
    qd_start = model.joint_qd_start.numpy()
    maximum = 0.0
    for joint in range(model.joint_count):
        for offset in range(int(qd_start[joint + 1] - qd_start[joint])):
            coordinate = int(q_start[joint] + offset)
            dof = int(qd_start[joint] + offset)
            if coordinate >= len(joint_q) or dof >= len(lower):
                continue
            if np.isfinite(lower[dof]) and np.isfinite(upper[dof]) and abs(upper[dof] - lower[dof]) < 1.0e5:
                maximum = max(maximum, float(lower[dof] - joint_q[coordinate]), float(joint_q[coordinate] - upper[dof]))
    return max(0.0, maximum)


def math_is_finite_positive(value: float) -> bool:
    """Return whether a scalar is finite and strictly positive."""
    return bool(np.isfinite(value) and value > 0.0)
