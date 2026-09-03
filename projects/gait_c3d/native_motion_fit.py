# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Public-API synthetic marker inverse kinematics for native gait subjects."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.ik as ik

from .c3d_adapter import C3DMarkerTrajectory
from .marker_clusters import TRACKING_CLUSTER_C3D_SOURCES, TRACKING_CLUSTER_MARKERS


@wp.kernel
def _normalize_free_root_quaternion(
    joint_q: wp.array2d[wp.float32],
    quaternion_start: int,
    valid: wp.array[wp.int32],
):
    """Normalize a free-root quaternion in-place without a device round trip."""
    row = wp.tid()
    if quaternion_start < 0:
        valid[row] = 1
        return
    x = joint_q[row, quaternion_start + 0]
    y = joint_q[row, quaternion_start + 1]
    z = joint_q[row, quaternion_start + 2]
    w = joint_q[row, quaternion_start + 3]
    norm = wp.sqrt(x * x + y * y + z * z + w * w)
    if wp.isfinite(norm) and norm > 0.0:
        joint_q[row, quaternion_start + 0] = x / norm
        joint_q[row, quaternion_start + 1] = y / norm
        joint_q[row, quaternion_start + 2] = z / norm
        joint_q[row, quaternion_start + 3] = w / norm
        valid[row] = 1
    else:
        valid[row] = 0


@wp.kernel
def _broadcast_seed(
    seed: wp.array2d[wp.float32],
    joint_q: wp.array2d[wp.float32],
):
    row, coordinate = wp.tid()
    joint_q[row, coordinate] = seed[0, coordinate]


@wp.kernel
def _predict_markers(
    body_q: wp.array[wp.transform],
    link_indices: wp.array[wp.int32],
    link_offsets: wp.array[wp.vec3],
    frame_idx: int,
    # outputs
    predictions: wp.array2d[wp.vec3],
):
    marker_idx = wp.tid()
    predictions[frame_idx, marker_idx] = wp.transform_point(body_q[link_indices[marker_idx]], link_offsets[marker_idx])


@wp.kernel
def _predict_markers_batched(
    body_q: wp.array2d[wp.transform],
    link_indices: wp.array[wp.int32],
    link_offsets: wp.array[wp.vec3],
    frame_start: int,
    valid_count: int,
    # outputs
    predictions: wp.array2d[wp.vec3],
):
    row, marker_idx = wp.tid()
    if row >= valid_count:
        return
    predictions[frame_start + row, marker_idx] = wp.transform_point(
        body_q[row, link_indices[marker_idx]], link_offsets[marker_idx]
    )


@wp.kernel
def _store_solution(
    joint_q: wp.array2d[wp.float32],
    costs: wp.array[wp.float32],
    frame_idx: int,
    joint_q_start: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_limit_lower: wp.array[wp.float32],
    joint_limit_upper: wp.array[wp.float32],
    joint_count: int,
    # outputs
    solutions: wp.array2d[wp.float32],
    solution_costs: wp.array[wp.float32],
    limit_violations: wp.array[wp.float32],
):
    coordinate = wp.tid()
    solutions[frame_idx, coordinate] = joint_q[0, coordinate]
    if coordinate == 0:
        maximum = float(0.0)
        for joint in range(joint_count):
            q_start = joint_q_start[joint]
            dof_start = joint_qd_start[joint]
            dof_end = joint_qd_start[joint + 1]
            for offset in range(dof_end - dof_start):
                q_idx = q_start + offset
                dof_idx = dof_start + offset
                lower = joint_limit_lower[dof_idx]
                upper = joint_limit_upper[dof_idx]
                if wp.isfinite(lower) and wp.isfinite(upper) and upper - lower < 1.0e5:
                    q = joint_q[0, q_idx]
                    maximum = wp.max(maximum, wp.max(lower - q, q - upper))
        solution_costs[frame_idx] = costs[0]
        limit_violations[frame_idx] = wp.max(maximum, 0.0)


@wp.kernel
def _store_batch_solutions(
    joint_q: wp.array2d[wp.float32],
    costs: wp.array[wp.float32],
    frame_start: int,
    valid_count: int,
    joint_q_start: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_limit_lower: wp.array[wp.float32],
    joint_limit_upper: wp.array[wp.float32],
    joint_count: int,
    # outputs
    solutions: wp.array2d[wp.float32],
    solution_costs: wp.array[wp.float32],
    limit_violations: wp.array[wp.float32],
):
    row, coordinate = wp.tid()
    if row >= valid_count:
        return
    frame_idx = frame_start + row
    solutions[frame_idx, coordinate] = joint_q[row, coordinate]
    if coordinate == 0:
        maximum = float(0.0)
        for joint in range(joint_count):
            q_start = joint_q_start[joint]
            dof_start = joint_qd_start[joint]
            dof_end = joint_qd_start[joint + 1]
            for offset in range(dof_end - dof_start):
                q_idx = q_start + offset
                dof_idx = dof_start + offset
                lower = joint_limit_lower[dof_idx]
                upper = joint_limit_upper[dof_idx]
                if wp.isfinite(lower) and wp.isfinite(upper) and upper - lower < 1.0e5:
                    q = joint_q[row, q_idx]
                    maximum = wp.max(maximum, wp.max(lower - q, q - upper))
        solution_costs[frame_idx] = costs[row]
        limit_violations[frame_idx] = wp.max(maximum, 0.0)


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
    """Predicted marker positions [m] in output-attachment order."""

    target_markers: np.ndarray
    """Target marker positions [m] in fitted-attachment order."""

    marker_rms: float
    """Root-mean-square marker error [m]."""

    marker_max: float
    """Maximum marker error [m]."""

    solver_cost: float = float("nan")
    """Weighted squared solver cost reported by public ``IKSolver``."""

    joint_limit_violation: float = float("nan")
    """Maximum bounded joint-limit violation [m or rad]."""


def marker_attachments_from_model(model: newton.Model) -> tuple[NativeMarkerAttachment, ...]:
    """Read marker attachments and collapse legacy tracking sites to centroids."""
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
    by_name = {attachment.name: attachment for attachment in attachments}
    cluster_members = {member for members in TRACKING_CLUSTER_MARKERS.values() for member in members}
    for centroid, members in TRACKING_CLUSTER_MARKERS.items():
        present = [member for member in members if member in by_name]
        if present and len(present) != len(members):
            missing = [member for member in members if member not in by_name]
            raise ValueError(f"model tracking cluster {centroid!r} is incomplete; missing {missing}")
        if centroid in by_name and present:
            raise ValueError(f"model tracking cluster {centroid!r} has both source and centroid sites")
        if present and len({by_name[member].body for member in members}) != 1:
            raise ValueError(f"model tracking cluster {centroid!r} spans multiple bodies")

    collapsed = []
    emitted: set[str] = set()
    for attachment in attachments:
        if attachment.name not in cluster_members:
            collapsed.append(attachment)
            continue
        centroid = next(name for name, members in TRACKING_CLUSTER_MARKERS.items() if attachment.name in members)
        if centroid in emitted:
            continue
        members = TRACKING_CLUSTER_MARKERS[centroid]
        source = [by_name[member] for member in members]
        collapsed.append(
            NativeMarkerAttachment(
                centroid,
                source[0].body,
                tuple(
                    float(value)
                    for value in np.mean([member.local_position for member in source], axis=0, dtype=np.float64)
                ),
            )
        )
        emitted.add(centroid)
    return tuple(collapsed)


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
    batch_size: int = 1,
    prediction_attachments: tuple[NativeMarkerAttachment, ...] | None = None,
) -> tuple[NativeIKFrame, ...]:
    """Solve a marker sequence with public Newton LM on the model device.

    Target data, solved coordinates, forward kinematics, marker predictions,
    and solver diagnostics stay on the model device until the complete
    sequence has finished. Apart from one-time static model metadata needed to
    identify a free-root quaternion, the only host transfers in the solve are
    the final result copies.

    Args:
        model: Finalized native model.
        attachments: Marker/body-local attachment definitions.
        target_sequence: Target marker positions [m], shape [frame, marker, 3].
        seed: Initial generalized coordinates [m or rad].
        iterations: LM iterations per frame.
        joint_limit_weight: Weight of the public joint-limit objective.
        lambda_initial: Initial LM damping value.
        batch_size: Number of frames solved concurrently. ``1`` preserves
            sequential warm starts. Larger values solve GPU batches; each row
            in a chunk starts from the previous chunk's final solution (or
            ``seed`` for the first chunk).
        prediction_attachments: Optional marker attachments to predict after
            solving. Use this to request full-marker predictions while fitting
            only a visible subset.

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
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    frame_count = target_sequence.shape[0]
    if frame_count == 0:
        return ()
    if prediction_attachments is None:
        prediction_attachments = attachments
    used_prediction_indices = set()
    target_prediction_indices = []
    for target_attachment in attachments:
        for prediction_index, prediction_attachment in enumerate(prediction_attachments):
            if prediction_index not in used_prediction_indices and prediction_attachment == target_attachment:
                target_prediction_indices.append(prediction_index)
                used_prediction_indices.add(prediction_index)
                break
        else:
            raise ValueError("prediction_attachments must include every fitted attachment")
    quaternion_slice = free_root_quaternion_slice(model)
    quaternion_start = -1 if quaternion_slice is None else quaternion_slice.start
    with wp.ScopedDevice(model.device):
        target_positions = wp.array(target_sequence.astype(np.float32), dtype=wp.vec3, device=model.device)
        link_indices = wp.array(
            np.asarray([attachment.body for attachment in attachments], dtype=np.int32),
            dtype=wp.int32,
            device=model.device,
        )
        link_offsets = wp.array(
            np.asarray([attachment.local_position for attachment in attachments], dtype=np.float32),
            dtype=wp.vec3,
            device=model.device,
        )
        prediction_link_indices = wp.array(
            np.asarray([attachment.body for attachment in prediction_attachments], dtype=np.int32),
            dtype=wp.int32,
            device=model.device,
        )
        prediction_link_offsets = wp.array(
            np.asarray([attachment.local_position for attachment in prediction_attachments], dtype=np.float32),
            dtype=wp.vec3,
            device=model.device,
        )
        seed_device = wp.array(seed.reshape(1, -1), dtype=wp.float32, device=model.device)
        joint_qd = wp.zeros(model.joint_dof_count, dtype=wp.float32, device=model.device)
        solutions = wp.zeros((frame_count, model.joint_coord_count), dtype=wp.float32, device=model.device)
        predictions = wp.zeros((frame_count, len(prediction_attachments)), dtype=wp.vec3, device=model.device)
        costs = wp.zeros(frame_count, dtype=wp.float32, device=model.device)
        limit_violations = wp.zeros(frame_count, dtype=wp.float32, device=model.device)
        quaternion_valid = wp.zeros(frame_count, dtype=wp.int32, device=model.device)

        def make_solver(targets: wp.array2d[wp.vec3], n_problems: int) -> ik.IKSolver:
            marker_objective = ik.IKObjectivePositionBatch(link_indices, link_offsets, targets)
            joint_limits = ik.IKObjectiveJointLimit(
                model.joint_limit_lower,
                model.joint_limit_upper,
                weight=joint_limit_weight,
            )
            return ik.IKSolver(
                model,
                n_problems,
                [marker_objective, joint_limits],
                lambda_initial=lambda_initial,
                jacobian_mode=ik.IKJacobianType.ANALYTIC,
            )

        if batch_size == 1:
            # Keep the exact temporal warm-start behavior for callers that
            # request it. All frame targets and all result buffers still stay
            # on the device throughout this loop.
            solver = make_solver(target_positions, 1)
            state = model.state()
            joint_q = wp.zeros((1, model.joint_coord_count), dtype=wp.float32, device=model.device)
            wp.copy(joint_q, seed_device)
            for frame_idx in range(frame_count):
                solver.set_problem_index(frame_idx, problem_count=frame_count)
                solver.step(joint_q, joint_q, iterations=iterations, step_size=1.0)
                wp.launch(
                    _normalize_free_root_quaternion,
                    dim=1,
                    inputs=[joint_q, quaternion_start],
                    outputs=[quaternion_valid[frame_idx : frame_idx + 1]],
                    device=model.device,
                )
                wp.launch(
                    _store_solution,
                    dim=model.joint_coord_count,
                    inputs=[
                        joint_q,
                        solver.costs,
                        frame_idx,
                        model.joint_q_start,
                        model.joint_qd_start,
                        model.joint_limit_lower,
                        model.joint_limit_upper,
                        model.joint_count,
                    ],
                    outputs=[solutions, costs, limit_violations],
                    device=model.device,
                )
                newton.eval_fk(model, joint_q[0], joint_qd, state)
                wp.launch(
                    _predict_markers,
                    dim=len(prediction_attachments),
                    inputs=[state.body_q, prediction_link_indices, prediction_link_offsets, frame_idx],
                    outputs=[predictions],
                    device=model.device,
                )
        else:
            # Larger batches trade strict frame-to-frame warm starts for much
            # higher GPU occupancy. This is the fast path for long mocap
            # sequences. Rows in a chunk start from a shared predictor, and
            # the last solved row seeds the next chunk.
            # Keep one fixed-size solver and pad only the final chunk so model
            # metadata and objective buffers are initialized once.
            solver_batch = min(batch_size, frame_count)
            target_batch = wp.zeros((solver_batch, len(attachments)), dtype=wp.vec3, device=model.device)
            solver = make_solver(target_batch, solver_batch)
            chunk_seed = wp.array(seed.reshape(1, -1), dtype=wp.float32, device=model.device)
            joint_q = wp.zeros((solver_batch, model.joint_coord_count), dtype=wp.float32, device=model.device)
            joint_qd_batch = wp.zeros((solver_batch, model.joint_dof_count), dtype=wp.float32, device=model.device)
            body_q_batch = wp.zeros((solver_batch, model.body_count), dtype=wp.transform, device=model.device)
            body_qd_batch = wp.zeros((solver_batch, model.body_count), dtype=wp.spatial_vector, device=model.device)
            batch_quaternion_valid = wp.zeros(solver_batch, dtype=wp.int32, device=model.device)
            for frame_start in range(0, frame_count, solver_batch):
                n_batch = min(solver_batch, frame_count - frame_start)
                wp.copy(target_batch[:n_batch], target_positions[frame_start : frame_start + n_batch])
                wp.launch(
                    _broadcast_seed,
                    dim=[solver_batch, model.joint_coord_count],
                    inputs=[chunk_seed],
                    outputs=[joint_q],
                    device=model.device,
                )
                solver.step(joint_q, joint_q, iterations=iterations, step_size=1.0)
                wp.launch(
                    _normalize_free_root_quaternion,
                    dim=solver_batch,
                    inputs=[joint_q, quaternion_start],
                    outputs=[batch_quaternion_valid],
                    device=model.device,
                )
                wp.copy(quaternion_valid[frame_start : frame_start + n_batch], batch_quaternion_valid[:n_batch])
                wp.copy(chunk_seed, joint_q[n_batch - 1 : n_batch])
                wp.launch(
                    _store_batch_solutions,
                    dim=[solver_batch, model.joint_coord_count],
                    inputs=[
                        joint_q,
                        solver.costs,
                        frame_start,
                        n_batch,
                        model.joint_q_start,
                        model.joint_qd_start,
                        model.joint_limit_lower,
                        model.joint_limit_upper,
                        model.joint_count,
                    ],
                    outputs=[solutions, costs, limit_violations],
                    device=model.device,
                )
                newton.eval_fk_batched(model, joint_q, joint_qd_batch, body_q_batch, body_qd_batch)
                wp.launch(
                    _predict_markers_batched,
                    dim=[solver_batch, len(prediction_attachments)],
                    inputs=[
                        body_q_batch,
                        prediction_link_indices,
                        prediction_link_offsets,
                        frame_start,
                        n_batch,
                    ],
                    outputs=[predictions],
                    device=model.device,
                )

        valid_quaternions = quaternion_valid.numpy().astype(bool)
        if not np.all(valid_quaternions):
            raise ValueError("IK produced a nonfinite free-root quaternion")
        solved_coordinates = solutions.numpy()
        predicted_markers = predictions.numpy()
        solver_costs = costs.numpy()
        solved_limit_violations = limit_violations.numpy()

    frames = []
    for frame_idx, target_frame in enumerate(target_sequence):
        predicted = predicted_markers[frame_idx].astype(np.float64)
        target = target_frame.copy()
        fitted_predictions = predicted[target_prediction_indices]
        errors = fitted_predictions - target
        distances = np.linalg.norm(errors, axis=1)
        frames.append(
            NativeIKFrame(
                solved_coordinates[frame_idx].copy(),
                predicted,
                target,
                float(np.sqrt(np.mean(np.sum(errors * errors, axis=1)))),
                float(np.max(distances)),
                float(solver_costs[frame_idx]),
                float(solved_limit_violations[frame_idx]),
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


_REAL_MARKER_SOURCES = {
    "Sternum": "STRN",
    "R.Acromium": "RSHO",
    "L.Acromium": "LSHO",
    "R.ASIS": "RASI",
    "L.ASIS": "LASI",
    "R.Thigh.Upper": "RTH2",
    "R.Thigh.Front": "RTH3",
    "R.Thigh.Rear": "RTH4",
    "R.Knee.Lat": "RKNE",
    "R.Knee.Med": "RMKNE",
    "R.Shank.Upper": "RTIB2",
    "R.Shank.Front": "RTIB3",
    "R.Shank.Rear": "RTIB4",
    "R.Ankle.Lat": "RANK",
    "R.Ankle.Med": "RMANK",
    "R.Heel": "RHEE",
    "R.Toe.Lat": "RMTH5",
    "R.Toe.Med": "RMTH1",
    "R.Toe.Tip": "RHLX",
    "L.Thigh.Upper": "LTH2",
    "L.Thigh.Front": "LTH3",
    "L.Thigh.Rear": "LTH4",
    "L.Knee.Lat": "LKNE",
    "L.Knee.Med": "LMKNE",
    "L.Shank.Upper": "LTIB2",
    "L.Shank.Front": "LTIB3",
    "L.Shank.Rear": "LTIB4",
    "L.Ankle.Lat": "LANK",
    "L.Ankle.Med": "LMANK",
    "L.Heel": "LHEE",
    "L.Toe.Lat": "LMTH5",
    "L.Toe.Med": "LMTH1",
    "L.Toe.Tip": "LHLX",
}
_REAL_MARKER_VIRTUAL = {
    "V.Sacral": ("LPSI", "RPSI"),
    "Top.Head": ("LFHD", "RFHD", "LBHD", "RBHD"),
    **TRACKING_CLUSTER_C3D_SOURCES,
}


@dataclass(frozen=True, slots=True)
class NativeC3DMarkers:
    """C3D marker targets reordered to native MJCF site order."""

    times: np.ndarray
    """Frame times [s]."""

    positions: np.ndarray
    """Newton-frame marker positions [m], shape [frame, marker, 3]."""

    valid: np.ndarray
    """Marker validity mask, shape [frame, marker]."""

    marker_names: tuple[str, ...]
    """Native marker labels in column order."""

    source_file: str
    """Source C3D basename."""

    source_sha256: str
    """Source C3D SHA-256."""


@dataclass(frozen=True, slots=True)
class NativeMotionArtifact:
    """Fitted native motion arrays and frame diagnostics."""

    times: np.ndarray
    """Frame times [s]."""

    joint_q: np.ndarray
    """Fitted native coordinates [m or rad], shape [frame, coordinate]."""

    joint_qd: np.ndarray
    """Finite-difference native velocities [m/s or rad/s]."""

    targets: np.ndarray
    """Target marker positions [m]."""

    predictions: np.ndarray
    """Predicted marker positions [m]."""

    valid: np.ndarray
    """Observed marker validity mask."""

    frame_rms: np.ndarray
    """Per-frame observed-marker RMS [m]."""

    frame_max: np.ndarray
    """Per-frame observed-marker maximum error [m]."""

    marker_rms: np.ndarray
    """Per-marker RMS over valid observations [m]."""

    marker_max: np.ndarray
    """Per-marker maximum over valid observations [m]."""

    body_names: tuple[str, ...]
    """Native body labels represented by the marker set."""

    body_rms: np.ndarray
    """Per-frame, per-body marker RMS [m]."""

    solver_cost: np.ndarray
    """Public IK weighted solver costs by frame."""

    joint_limit_violation: np.ndarray
    """Maximum bounded joint-limit violation by frame [m or rad]."""

    marker_names: tuple[str, ...]
    """Marker labels in output column order, including cluster centroids."""

    source_file: str
    """Source C3D basename."""

    source_sha256: str
    """Source C3D SHA-256."""

    registration: np.ndarray
    """4x4 row-vector registration from C3D Newton frame to model frame."""


def map_c3d_markers_to_native(
    markers: C3DMarkerTrajectory,
    attachments: tuple[NativeMarkerAttachment, ...],
) -> NativeC3DMarkers:
    """Join C3D labels to native sites and create declared virtual markers.

    Tracking-cluster targets use the centroid of available valid sources on
    each frame. A source label absent from the complete trial is therefore
    omitted from that centroid without fabricating a position for the missing
    source. Other multi-source virtual targets still require all sources.
    """
    source_index = {name: index for index, name in enumerate(markers.marker_names)}
    output = np.zeros((len(markers.times), len(attachments), 3), dtype=np.float32)
    valid = np.zeros((len(markers.times), len(attachments)), dtype=bool)
    names = tuple(attachment.name for attachment in attachments)
    if len(set(names)) != len(names):
        raise ValueError("native marker attachments must have unique names")
    for column, name in enumerate(names):
        sources = _REAL_MARKER_VIRTUAL.get(name)
        if sources is None:
            source = _REAL_MARKER_SOURCES.get(name)
            if source is None:
                raise ValueError(f"no C3D source mapping for native marker {name!r}")
            sources = (source,)
        is_tracking_cluster = name in TRACKING_CLUSTER_C3D_SOURCES
        source_columns = [source_index[source] for source in sources if source in source_index]
        if not is_tracking_cluster and len(source_columns) != len(sources):
            continue
        if not source_columns:
            continue
        source_valid = markers.valid[:, source_columns]
        if len(source_columns) == 1:
            valid[:, column] = source_valid[:, 0]
            output[:, column] = markers.positions[:, source_columns[0]]
            output[~valid[:, column], column] = 0.0
            continue
        # Tracking clusters are centroids: average every available valid source
        # on each frame so one missing marker does not discard the cluster.
        valid[:, column] = np.any(source_valid, axis=1)
        weighted_positions = np.where(source_valid[..., None], markers.positions[:, source_columns], 0.0)
        counts = np.sum(source_valid, axis=1)
        output[:, column] = np.divide(
            np.sum(weighted_positions, axis=1, dtype=np.float32),
            counts[:, None],
            out=np.zeros((len(markers.times), 3), dtype=np.float32),
            where=counts[:, None] > 0,
        )
    return NativeC3DMarkers(markers.times.copy(), output, valid, names, markers.source_file, markers.source_sha256)


def apply_marker_registration(markers: NativeC3DMarkers, registration: np.ndarray) -> NativeC3DMarkers:
    """Apply a finite row-vector 4x4 registration to marker targets."""
    registration = np.asarray(registration, dtype=np.float64)
    if registration.shape != (4, 4) or not np.all(np.isfinite(registration)):
        raise ValueError("registration must be a finite 4x4 matrix")
    rotation = registration[:3, :3]
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1.0e-7) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1.0e-7
    ):
        raise ValueError("registration rotation must be proper and orthonormal")
    positions = markers.positions.astype(np.float64) @ rotation.T + registration[:3, 3]
    positions[~markers.valid] = 0.0
    return NativeC3DMarkers(
        markers.times.copy(),
        positions.astype(np.float32),
        markers.valid.copy(),
        markers.marker_names,
        markers.source_file,
        markers.source_sha256,
    )


def finite_difference_joint_qd(model: newton.Model, joint_q: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Compute finite-difference velocities in Newton joint-DOF layout."""
    joint_q = np.asarray(joint_q, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    if joint_q.ndim != 2 or joint_q.shape[1] != model.joint_coord_count or len(times) != len(joint_q):
        raise ValueError("joint_q/times have incompatible shapes")
    if len(times) < 2 or not np.all(np.diff(times) > 0.0):
        raise ValueError("motion times must increase and contain at least two frames")
    qd = np.zeros((len(times), model.joint_dof_count), dtype=np.float64)
    q_start = model.joint_q_start.numpy()
    qd_start = model.joint_qd_start.numpy()
    joint_types = model.joint_type.numpy()
    joint_parent = model.joint_parent.numpy()
    for joint in range(model.joint_count):
        q0, q1 = int(q_start[joint]), int(q_start[joint + 1])
        d0, d1 = int(qd_start[joint]), int(qd_start[joint + 1])
        if joint_parent[joint] == -1 and joint_types[joint] == newton.JointType.FREE and q1 - q0 == 7:
            q = joint_q[:, q0 + 3 : q0 + 7].copy()
            q /= np.linalg.norm(q, axis=1)[:, None]
            for frame in range(1, len(q)):
                if np.dot(q[frame - 1], q[frame]) < 0.0:
                    q[frame] *= -1.0
            qdot = np.gradient(q, times, axis=0, edge_order=1)
            vector, scalar = q[:, :3], q[:, 3, None]
            vector_dot, scalar_dot = qdot[:, :3], qdot[:, 3, None]
            qd[:, d0 + 3 : d0 + 6] = 2.0 * (scalar * vector_dot - scalar_dot * vector + np.cross(vector, vector_dot))
            child = int(model.joint_child.numpy()[joint])
            child_xform = model.joint_X_c.numpy()[joint]
            offset = _rotate_point(
                np.r_[-child_xform[3:6], child_xform[6]], model.body_com.numpy()[child] - child_xform[:3]
            )
            rotated_offset = offset + 2.0 * np.cross(vector, scalar * offset + np.cross(vector, offset))
            qd[:, d0 : d0 + 3] = np.gradient(joint_q[:, q0 : q0 + 3] + rotated_offset, times, axis=0)
            continue
        for offset in range(d1 - d0):
            qd[:, d0 + offset] = np.gradient(joint_q[:, q0 + offset], times, edge_order=1)
    return qd.astype(np.float32)


def marker_residuals_by_body(
    model: newton.Model,
    attachments: tuple[NativeMarkerAttachment, ...],
    residuals: np.ndarray,
) -> dict[str, float]:
    """Aggregate marker RMS by native body label."""
    residuals = np.asarray(residuals, dtype=np.float64)
    if residuals.shape != (len(attachments),):
        raise ValueError("residuals has an incompatible shape")
    output = {}
    body_labels = model.body_label
    for body in sorted({attachment.body for attachment in attachments}):
        values = residuals[[attachment.body == body for attachment in attachments]]
        output[body_labels[body].rsplit("/", 1)[-1]] = float(np.sqrt(np.mean(values * values))) if len(values) else 0.0
    return output


def fit_c3d_marker_motion(
    model: newton.Model,
    attachments: tuple[NativeMarkerAttachment, ...],
    markers: NativeC3DMarkers,
    seed: np.ndarray,
    *,
    registration: np.ndarray | None = None,
    iterations: int = 40,
    joint_limit_weight: float = 0.1,
    batch_size: int = 8,
    start_frame: int = 0,
    end_frame: int | None = None,
    max_frames: int | None = None,
    stride: int = 1,
) -> NativeMotionArtifact:
    """Fit valid real-C3D marker observations and publish finite diagnostics.

    Args:
        model: Finalized native model.
        attachments: Native marker site bindings.
        markers: C3D markers reordered into attachment order.
        seed: Initial generalized coordinates [m or rad].
        registration: Row-vector 4x4 C3D-to-model transform.
        iterations: LM iterations per selected frame.
        joint_limit_weight: Weight of the public joint-limit objective.
        batch_size: Number of frames solved concurrently. Larger values use
            more GPU parallelism; use ``1`` to retain frame warm starts.
        start_frame: First frame index to fit.
        end_frame: Exclusive frame index, or ``None`` for the end.
        max_frames: Maximum selected frames, or ``None`` for no limit.
        stride: Frame step between selected frames.

    Returns:
        A finite fitted motion artifact in selected-frame order.
    """
    if stride <= 0:
        raise ValueError("stride must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if start_frame < 0 or (end_frame is not None and end_frame <= start_frame):
        raise ValueError("motion frame range is invalid")
    if registration is None:
        registration = np.eye(4, dtype=np.float64)
    registered = apply_marker_registration(markers, registration)
    frame_indices = np.arange(
        start_frame, len(registered.times) if end_frame is None else end_frame, stride, dtype=np.int32
    )
    if max_frames is not None and max_frames > 0:
        frame_indices = frame_indices[:max_frames]
    if len(frame_indices) < 2:
        raise ValueError("real motion fit needs at least two selected frames")
    selected_positions = registered.positions[frame_indices]
    selected_valid = registered.valid[frame_indices]
    visible = np.flatnonzero(np.all(selected_valid, axis=0))
    if len(visible) < 6:
        raise ValueError("real motion fit needs at least six markers valid in every selected frame")
    frames = solve_marker_sequence(
        model,
        tuple(attachments[index] for index in visible),
        selected_positions[:, visible],
        seed,
        iterations=iterations,
        joint_limit_weight=joint_limit_weight,
        batch_size=batch_size,
        prediction_attachments=attachments,
    )
    joint_q = np.asarray([frame.joint_q for frame in frames], dtype=np.float32)
    # solve_marker_sequence computes predictions for all attachments on the
    # model device, even though optimization uses only the visible subset.
    predictions = np.asarray([frame.predicted_markers for frame in frames], dtype=np.float32)
    targets = registered.positions[frame_indices].astype(np.float32)
    valid = registered.valid[frame_indices].copy()
    distances = np.linalg.norm(predictions.astype(np.float64) - targets.astype(np.float64), axis=-1)
    residuals = np.where(valid, distances, 0.0)
    counts = valid.sum(axis=1)
    frame_rms = np.sqrt(np.divide(np.sum(residuals**2, axis=1), counts, out=np.zeros(len(counts)), where=counts > 0))
    frame_max = np.max(residuals, axis=1)
    marker_counts = valid.sum(axis=0)
    marker_rms = np.sqrt(
        np.divide(
            np.sum(residuals**2, axis=0), marker_counts, out=np.zeros(len(marker_counts)), where=marker_counts > 0
        )
    )
    marker_max = np.max(residuals, axis=0)
    body_indices = tuple(sorted({attachment.body for attachment in attachments}))
    body_names = tuple(model.body_label[index].rsplit("/", 1)[-1] for index in body_indices)
    body_rms = np.zeros((len(frame_indices), len(body_indices)), dtype=np.float32)
    for body_column, body in enumerate(body_indices):
        marker_mask = np.asarray([attachment.body == body for attachment in attachments], dtype=bool)
        body_valid = valid[:, marker_mask]
        body_residuals = residuals[:, marker_mask]
        body_count = body_valid.sum(axis=1)
        body_rms[:, body_column] = np.sqrt(
            np.divide(
                np.sum(body_residuals**2, axis=1),
                body_count,
                out=np.zeros(len(body_count)),
                where=body_count > 0,
            )
        )
    joint_qd = finite_difference_joint_qd(model, joint_q, registered.times[frame_indices])
    return NativeMotionArtifact(
        registered.times[frame_indices],
        joint_q,
        joint_qd,
        targets,
        predictions,
        valid,
        frame_rms.astype(np.float32),
        frame_max.astype(np.float32),
        marker_rms.astype(np.float32),
        marker_max.astype(np.float32),
        body_names,
        body_rms,
        np.asarray([frame.solver_cost for frame in frames], dtype=np.float32),
        np.asarray([frame.joint_limit_violation for frame in frames], dtype=np.float32),
        markers.marker_names,
        markers.source_file,
        markers.source_sha256,
        registration.copy(),
    )


def load_native_motion_artifact(path: str | os.PathLike) -> NativeMotionArtifact:
    """Load and verify a previously published native motion artifact."""
    requested = Path(path).expanduser().resolve()
    root = requested.parent if requested.is_file() else requested
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"native motion manifest is missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"native motion manifest is invalid: {manifest_path}") from error
    seal = manifest.pop("seal", None) if isinstance(manifest, dict) else None
    expected_seal = {
        "algorithm": "sha256",
        "content_sha256": hashlib.sha256(_canonical_motion_json(manifest)).hexdigest(),
    }
    if seal != expected_seal or manifest.get("schema_version") != "gait_native_motion_artifact_1":
        raise ValueError("native motion manifest seal or schema mismatch")

    payload = manifest.get("payload", {})
    payload_path = (root / str(payload.get("file", ""))).resolve()
    if (
        payload_path.parent != root
        or not payload_path.is_file()
        or hashlib.sha256(payload_path.read_bytes()).hexdigest() != payload.get("sha256")
    ):
        raise ValueError("native motion payload path or hash mismatch")

    source, markers, bodies, frames = (manifest.get(name, {}) for name in ("source", "markers", "bodies", "frames"))
    marker_names = tuple(markers.get("names", ()))
    body_names = tuple(bodies.get("names", ()))
    registration = np.asarray(manifest.get("registration"), dtype=np.float64)
    if (
        not marker_names
        or len(set(marker_names)) != len(marker_names)
        or len(set(body_names)) != len(body_names)
        or registration.shape != (4, 4)
        or not np.all(np.isfinite(registration))
        or not isinstance(source.get("file"), str)
        or not isinstance(source.get("sha256"), str)
    ):
        raise ValueError("native motion manifest metadata is invalid")

    dtypes = {
        "times": np.float64,
        "joint_q": np.float32,
        "joint_qd": np.float32,
        "targets": np.float32,
        "predictions": np.float32,
        "valid": bool,
        "frame_rms": np.float32,
        "frame_max": np.float32,
        "marker_rms": np.float32,
        "marker_max": np.float32,
        "body_rms": np.float32,
        "solver_cost": np.float32,
        "joint_limit_violation": np.float32,
    }
    try:
        with np.load(payload_path, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name], dtype=dtype).copy() for name, dtype in dtypes.items()}
    except (KeyError, OSError, ValueError) as error:
        raise ValueError("native motion payload is invalid") from error

    frame_count = len(arrays["times"])
    marker_count = len(marker_names)
    expected_shapes = {
        "times": (frame_count,),
        "targets": (frame_count, marker_count, 3),
        "predictions": (frame_count, marker_count, 3),
        "valid": (frame_count, marker_count),
        "frame_rms": (frame_count,),
        "frame_max": (frame_count,),
        "marker_rms": (marker_count,),
        "marker_max": (marker_count,),
        "body_rms": (frame_count, len(body_names)),
        "solver_cost": (frame_count,),
        "joint_limit_violation": (frame_count,),
    }
    if (
        frame_count < 1
        or arrays["joint_q"].ndim != 2
        or arrays["joint_q"].shape[0] != frame_count
        or arrays["joint_qd"].ndim != 2
        or arrays["joint_qd"].shape[0] != frame_count
        or any(arrays[name].shape != shape for name, shape in expected_shapes.items())
        or any(not np.all(np.isfinite(array)) for name, array in arrays.items() if name != "valid")
        or not np.all(np.diff(arrays["times"]) > 0.0)
        or frames.get("count") != frame_count
        or markers.get("sample_count") != arrays["valid"].size
    ):
        raise ValueError("native motion payload arrays are invalid")
    return NativeMotionArtifact(
        arrays["times"],
        arrays["joint_q"],
        arrays["joint_qd"],
        arrays["targets"],
        arrays["predictions"],
        arrays["valid"],
        arrays["frame_rms"],
        arrays["frame_max"],
        arrays["marker_rms"],
        arrays["marker_max"],
        body_names,
        arrays["body_rms"],
        arrays["solver_cost"],
        arrays["joint_limit_violation"],
        marker_names,
        source["file"],
        source["sha256"],
        registration,
    )


def write_native_motion_artifact(
    motion: NativeMotionArtifact,
    output_dir: str | os.PathLike,
    *,
    model_path: str | os.PathLike | None = None,
    calibration_path: str | os.PathLike | None = None,
    settings: dict | None = None,
    marker_mapping: dict | None = None,
    overwrite: bool = False,
) -> Path:
    """Publish a sealed native fitted-motion NPZ artifact."""
    root = Path(output_dir).expanduser().absolute()
    if root.exists():
        if not overwrite:
            raise FileExistsError(root)
        if (
            root.is_symlink()
            or not root.is_dir()
            or {entry.name for entry in root.iterdir()}
            != {
                "manifest.json",
                "motion.npz",
            }
        ):
            raise ValueError(f"overwrite target is not an exact native motion artifact: {root}")
        load_native_motion_artifact(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{root.name}.", dir=root.parent) as temporary:
        staged = Path(temporary)
        payload = staged / "motion.npz"
        np.savez_compressed(
            payload,
            times=motion.times,
            joint_q=motion.joint_q,
            joint_qd=motion.joint_qd,
            targets=motion.targets,
            predictions=motion.predictions,
            valid=motion.valid,
            frame_rms=motion.frame_rms,
            frame_max=motion.frame_max,
            marker_rms=motion.marker_rms,
            marker_max=motion.marker_max,
            body_rms=motion.body_rms,
            solver_cost=motion.solver_cost,
            joint_limit_violation=motion.joint_limit_violation,
            registration=motion.registration,
        )
        marker_mapping_manifest = {
            "policy": "arithmetic centroid of complete thigh and shank tracking clusters",
            "centroids": {name: list(sources) for name, sources in TRACKING_CLUSTER_C3D_SOURCES.items()},
        }
        if marker_mapping is not None:
            marker_mapping_manifest["acquisition"] = marker_mapping
        manifest = {
            "schema_version": "gait_native_motion_artifact_1",
            "coordinate_system": {
                "frame": "Newton world",
                "length_unit": "m",
                "up_axis": "Z",
                "forward_axis": "X",
                "left_axis": "Y",
            },
            "source": {"file": motion.source_file, "sha256": motion.source_sha256},
            "markers": {
                "names": list(motion.marker_names),
                "valid_count": int(motion.valid.sum()),
                "sample_count": int(motion.valid.size),
            },
            "marker_mapping": marker_mapping_manifest,
            "bodies": {"names": list(motion.body_names), "residuals": "body_rms"},
            "frames": {
                "count": int(len(motion.times)),
                "start_s": float(motion.times[0]),
                "end_s": float(motion.times[-1]),
            },
            "registration": motion.registration.tolist(),
            "model": _artifact_file_metadata(model_path),
            "calibration": _artifact_file_metadata(calibration_path),
            "settings": settings or {},
            "payload": {"file": "motion.npz", "sha256": hashlib.sha256(payload.read_bytes()).hexdigest()},
        }
        manifest["seal"] = {
            "algorithm": "sha256",
            "content_sha256": hashlib.sha256(_canonical_motion_json(manifest)).hexdigest(),
        }
        (staged / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )
        if root.exists():
            shutil.rmtree(root)
        os.rename(staged, root)
    return root


def _artifact_file_metadata(path: str | os.PathLike | None) -> dict | None:
    """Return a safe file/hash record for optional motion provenance."""
    if path is None:
        return None
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {"file": resolved.name, "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest()}


def _canonical_motion_json(value: dict) -> bytes:
    """Serialize a motion manifest deterministically."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
