# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Public-API synthetic marker inverse kinematics for native gait subjects."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.ik as ik

from .c3d_adapter import C3DMarkerTrajectory


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
_REAL_MARKER_VIRTUAL = {"V.Sacral": ("LPSI", "RPSI"), "Top.Head": ("LFHD", "RFHD", "LBHD", "RBHD")}


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
    """Marker labels in output column order."""

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
    """Join C3D labels to native sites and create declared virtual markers."""
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
        missing = [source for source in sources if source not in source_index]
        if missing:
            raise ValueError(f"C3D is missing sources for native marker {name!r}: {missing}")
        source_columns = [source_index[source] for source in sources]
        source_valid = markers.valid[:, source_columns]
        valid[:, column] = np.all(source_valid, axis=1)
        output[:, column] = np.mean(markers.positions[:, source_columns], axis=1, dtype=np.float32)
        output[~valid[:, column], column] = 0.0
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
        if joint_parent[joint] == -1 and joint_types[joint] == newton.JointType.FREE and q1 - q0 == 7 and d1 - d0 == 6:
            q = joint_q[:, q0 : q0 + 7].copy()
            for frame in range(1, len(q)):
                if np.dot(q[frame - 1, 3:], q[frame, 3:]) < 0.0:
                    q[frame, 3:] *= -1.0
            qd[:, d0 : d0 + 3] = np.gradient(q[:, :3], times, axis=0, edge_order=1)
            qdot = np.gradient(q[:, 3:], times, axis=0, edge_order=1)
            # 2 * conjugate(q) * qdot gives angular velocity in the local frame.
            xyz = q[:, :3] * 0.0
            for frame in range(len(q)):
                x, y, z, w = q[frame, 3:]
                dx, dy, dz, dw = qdot[frame]
                xyz[frame] = 2.0 * np.asarray(
                    (
                        w * dx - x * dw - y * dz + z * dy,
                        w * dy - x * dz + y * dw - z * dx,
                        w * dz + x * dy - y * dx - z * dw,
                    )
                )
            qd[:, d0 + 3 : d0 + 6] = xyz
            continue
        for offset in range(d1 - d0):
            coordinate = q0 + offset
            qd[:, d0 + offset] = np.gradient(joint_q[:, coordinate], times, edge_order=1)
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
    start_frame: int = 0,
    end_frame: int | None = None,
    max_frames: int | None = None,
    stride: int = 1,
) -> NativeMotionArtifact:
    """Fit valid real-C3D marker observations and publish finite diagnostics."""
    if stride <= 0:
        raise ValueError("stride must be positive")
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
    always_valid = np.all(selected_valid, axis=0)
    if int(np.sum(always_valid)) < 6:
        raise ValueError("real motion fit needs at least six markers valid in every selected frame")
    visible = np.flatnonzero(always_valid)
    visible_attachments = tuple(attachments[index] for index in visible)
    frames = solve_marker_sequence(
        model, visible_attachments, selected_positions[:, visible], seed, iterations=iterations
    )
    joint_q = np.asarray([frame.joint_q for frame in frames], dtype=np.float32)
    predictions = np.asarray(
        [marker_positions_from_joint_q(model, attachments, coordinates) for coordinates in joint_q], dtype=np.float32
    )
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


def write_native_motion_artifact(
    motion: NativeMotionArtifact,
    output_dir: str | os.PathLike,
    *,
    model_path: str | os.PathLike | None = None,
    calibration_path: str | os.PathLike | None = None,
    settings: dict | None = None,
) -> Path:
    """Atomically publish a sealed native fitted-motion NPZ artifact."""
    root = Path(output_dir).expanduser().resolve()
    if root.exists():
        raise FileExistsError(root)
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
