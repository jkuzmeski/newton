# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Three-dimensional segment frames and model coordinates from Visual3D exports.

Anatomical frames come from the static trial: the pelvis from both ASIS and
PSIS markers, the thighs from the hip center and femoral epicondyles, the shanks
from the knee center and malleoli, and the shoes from the heel-to-metatarsal
heading with the intrinsic shoe level in the static pose. Each segment is then
tracked through motion by a rigid fit of its marker cluster (CAST). All frames
use Newton axes: x forward, y left, z up.

Model coordinates (:data:`COORDINATE_NAMES`) are a free pelvis at the mid-hip
center, three-axis hips (flexion, adduction, internal rotation), and hinge knees
(flexion) and ankles (dorsiflexion). Hip angles are anatomical; knees and
ankles are zero in the static pose, and their constant static offsets are part
of the model. Knee and ankle keep only their sagittal
rotation, so :func:`fit_legs` refines the legs against tracked joint centers and
shoe frames; the omitted knee and subtalar motion remains as reported error.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..cartesian.prepare_visual3d import _kabsch_rigid_transforms, _read_triplets
from ..cartesian.visual3d import load_visual3d_export

SIDES = ("right", "left")
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
COORDINATE_NAMES = ("x", "y", "z", "qx", "qy", "qz", "qw", *JOINT_NAMES)
CLUSTERS = {
    "pelvis": ("RASI", "LASI", "RPSI", "LPSI"),
    "thigh": ("TH1", "TH2", "TH3", "TH4"),
    "shank": ("SH1", "SH2", "SH3", "SH4"),
    "foot": ("CAL1", "CAL2", "CAL3", "MT1H", "MT5H", "TOE"),
}


def lateral_sign(side: str) -> float:
    """Return +1 for the right leg and -1 for the left, the sign of its medial (adduction) direction along +y."""
    return 1.0 if side == "right" else -1.0


def hip_axes(side: str) -> tuple[tuple[float, float, float], ...]:
    """Flexion, adduction, and internal-rotation axes of a hip in the pelvis frame."""
    s = lateral_sign(side)
    return ((0.0, -1.0, 0.0), (s, 0.0, 0.0), (0.0, 0.0, s))


KNEE_AXIS = (0.0, 1.0, 0.0)
"""Positive knee flexion moves the shank backward."""
ANKLE_AXIS = (0.0, -1.0, 0.0)
"""Positive dorsiflexion raises the toe."""


def axis_rotation(axis, angle) -> np.ndarray:
    """Rodrigues rotation matrices, shape [..., 3, 3], for one unit axis and angles [rad]."""
    axis = np.asarray(axis, dtype=float)
    angle = np.asarray(angle, dtype=float)
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    s, c = np.sin(angle)[..., None, None], np.cos(angle)[..., None, None]
    return np.eye(3) + s * k + (1 - c) * (k @ k)


def hip_matrix(angles: np.ndarray, side: str) -> np.ndarray:
    """Compose the intrinsic hip rotation ``R(a0, q0) R(a1, q1) R(a2, q2)``."""
    axes = hip_axes(side)
    angles = np.asarray(angles, dtype=float)
    return (
        axis_rotation(axes[0], angles[..., 0])
        @ axis_rotation(axes[1], angles[..., 1])
        @ axis_rotation(axes[2], angles[..., 2])
    )


def hip_angles(rotation: np.ndarray, side: str) -> np.ndarray:
    """Decompose relative hip rotations, shape [..., 3, 3], into flexion, adduction, rotation [rad].

    With axes ``(-y, s x, s z)`` the product is ``Ry(-q0) Rx(s q1) Rz(s q2)``,
    an intrinsic y-x-z sequence.
    """
    r = np.asarray(rotation, dtype=float)
    s = lateral_sign(side)
    beta = np.arcsin(np.clip(-r[..., 1, 2], -1.0, 1.0))
    alpha = np.arctan2(r[..., 0, 2], r[..., 2, 2])
    gamma = np.arctan2(r[..., 1, 0], r[..., 1, 1])
    return np.stack((-alpha, s * beta, s * gamma), axis=-1)


def hinge_angle(rotation: np.ndarray, axis) -> np.ndarray:
    """Return the closest rotation angle about a +/-y hinge axis [rad]."""
    r = np.asarray(rotation, dtype=float)
    angle = np.arctan2(r[..., 0, 2], r[..., 2, 2])
    return angle if axis[1] > 0 else -angle


def _frame(x_hint: np.ndarray, y_hint: np.ndarray, z_axis: np.ndarray, primary: str) -> np.ndarray:
    """Right-handed frame from a primary axis and a hint, columns x, y, z."""
    if primary == "z":
        z = z_axis / np.linalg.norm(z_axis, axis=-1, keepdims=True)
        x = np.cross(y_hint, z)
        x /= np.linalg.norm(x, axis=-1, keepdims=True)
        y = np.cross(z, x)
    else:
        y = y_hint / np.linalg.norm(y_hint, axis=-1, keepdims=True)
        z = np.cross(x_hint, y)
        z /= np.linalg.norm(z, axis=-1, keepdims=True)
        x = np.cross(y, z)
    return np.stack((x, y, z), axis=-1)


@dataclass
class Calibration:
    """Static anatomical frames, cluster templates, and model geometry of one subject trial."""

    frames: dict[str, np.ndarray]
    """Static segment rotations keyed ``pelvis``, ``thigh_r``, ``shank_r``, ``foot_r``, ... (shoe frame for feet)."""
    origins: dict[str, np.ndarray]
    """Static joint centers keyed ``hip_mid``, ``hip_r``, ``knee_r``, ``ankle_r``, ... [m]."""
    templates: dict[str, np.ndarray]
    """Static cluster marker positions keyed like ``frames`` [m]."""
    offsets: dict[str, np.ndarray]
    """Constant joint child-frame offsets ``R_parent^T R_child`` at the static pose."""
    hip_m: dict[str, np.ndarray]
    """Hip centers in the pelvis frame [m]."""
    knee_m: dict[str, np.ndarray]
    """Knee centers in the thigh frame [m]."""
    ankle_m: dict[str, np.ndarray]
    """Ankle centers in the shank frame [m]."""


def _suffix(side: str) -> str:
    return "r" if side == "right" else "l"


def _prefix(side: str) -> str:
    return "R" if side == "right" else "L"


def calibrate(trial_root: Path) -> Calibration:
    """Build static anatomical frames and cluster templates from a trial's static exports."""
    manifest = json.loads((trial_root / "visual3d_manifest.json").read_text(encoding="utf-8"))
    labels, static, valid = _read_triplets(trial_root / "static_all_targets.txt", manifest)
    center_labels, centers, center_valid = _read_triplets(trial_root / "static_joint_centers.txt", manifest)

    def point(name: str) -> np.ndarray:
        i = labels.index(name)
        if not valid[:, i].any():
            raise ValueError(f"static marker {name} is missing")
        return np.nanmean(static[valid[:, i], i], axis=0)

    def center(name: str) -> np.ndarray:
        i = center_labels.index(name)
        return np.nanmean(centers[center_valid[:, i], i], axis=0)

    frames, origins, templates, offsets = {}, {}, {}, {}
    hip = {side: center(f"{_prefix(side)}HIP") for side in SIDES}
    origins["hip_mid"] = 0.5 * (hip["right"] + hip["left"])
    asis_mid = 0.5 * (point("RASI") + point("LASI"))
    psis_mid = 0.5 * (point("RPSI") + point("LPSI"))
    frames["pelvis"] = _frame(asis_mid - psis_mid, point("LASI") - point("RASI"), None, "y")
    templates["pelvis"] = np.array([point(name) for name in CLUSTERS["pelvis"]])
    up = np.array([0.0, 0.0, 1.0])
    hip_m, knee_m, ankle_m = {}, {}, {}
    for side in SIDES:
        p, s = _prefix(side), _suffix(side)
        knee = center(f"{p}KNEE")
        ankle = center(f"{p}ANKLE")
        origins[f"hip_{s}"], origins[f"knee_{s}"], origins[f"ankle_{s}"] = hip[side], knee, ankle
        # Lateral-to-medial on the right and medial-to-lateral on the left both point along +y.
        knee_left = point(f"{p}MFC") - point(f"{p}LFC") if side == "right" else point(f"{p}LFC") - point(f"{p}MFC")
        ankle_left = point(f"{p}MMAL") - point(f"{p}LMAL") if side == "right" else point(f"{p}LMAL") - point(f"{p}MMAL")
        frames[f"thigh_{s}"] = _frame(None, knee_left, hip[side] - knee, "z")
        frames[f"shank_{s}"] = _frame(None, ankle_left, knee - ankle, "z")
        heel = np.mean([point(f"{p}CAL{i}") for i in (1, 2, 3)], axis=0)
        mth = 0.5 * (point(f"{p}MT1H") + point(f"{p}MT5H"))
        heading = mth - heel
        heading[2] = 0.0
        # The intrinsic shoe is level in the static calibration.
        frames[f"foot_{s}"] = _frame(heading, np.cross(up, heading), None, "y")
        for segment in ("thigh", "shank", "foot"):
            templates[f"{segment}_{s}"] = np.array([point(f"{p}{name}") for name in CLUSTERS[segment]])
        # Hip angles are anatomical (zero when the thigh and pelvis frames align), so the
        # static stance posture, which varies between calibrations, stays in the coordinates.
        offsets[f"hip_{s}"] = np.eye(3)
        offsets[f"knee_{s}"] = frames[f"thigh_{s}"].T @ frames[f"shank_{s}"]
        offsets[f"ankle_{s}"] = frames[f"shank_{s}"].T @ frames[f"foot_{s}"]
        hip_m[side] = frames["pelvis"].T @ (hip[side] - origins["hip_mid"])
        knee_m[side] = frames[f"thigh_{s}"].T @ (knee - hip[side])
        ankle_m[side] = frames[f"shank_{s}"].T @ (ankle - knee)
    return Calibration(frames, origins, templates, offsets, hip_m, knee_m, ankle_m)


def track(trial_root: Path, calibration: Calibration, start_s: float, end_s: float, belt_speed_m_s: float) -> dict:
    """Track segment frames and joint centers over a motion window, with treadmill translation removed.

    Returns:
        ``time_s`` [frames], ``rotations`` keyed by segment [frames, 3, 3],
        ``centers`` keyed by joint [frames, 3] in the belt frame, and the
        worst cluster fit residual ``cluster_rms_max_m`` per segment.
    """
    trial = load_visual3d_export(trial_root)
    manifest = trial.manifest
    labels, points, valid = _read_triplets(trial_root / "motion_processed_targets.txt", manifest)
    center_labels, centers, center_valid = _read_triplets(trial_root / "motion_joint_centers.txt", manifest)
    mask = (trial.marker_time_s >= start_s) & (trial.marker_time_s <= end_s)
    time = trial.marker_time_s[mask] - trial.marker_time_s[mask][0]
    shift = np.zeros((len(time), 3))
    shift[:, 0] = time * belt_speed_m_s
    rotations, residuals = {}, {}
    for name, template in calibration.templates.items():
        segment, _, s = name.partition("_")
        prefix = "" if segment == "pelvis" else ("R" if s == "r" else "L")
        markers = CLUSTERS[segment]
        ids = [labels.index(f"{prefix}{m}") if segment != "pelvis" else labels.index(m) for m in markers]
        if not np.all(valid[mask][:, ids]):
            raise ValueError(f"{name} cluster has missing markers in the window")
        cluster, _, rms, _ = _kabsch_rigid_transforms(template, points[mask][:, ids])
        rotations[name] = cluster @ calibration.frames[name]
        residuals[name] = float(rms.max())
    joints = {}
    for side in SIDES:
        p, s = _prefix(side), _suffix(side)
        for joint in ("HIP", "KNEE", "ANKLE"):
            i = center_labels.index(f"{p}{joint}")
            if not np.all(center_valid[mask][:, i]):
                raise ValueError(f"{p}{joint} has missing samples in the window")
            joints[f"{joint.lower()}_{s}"] = centers[mask][:, i] + shift
    joints["hip_mid"] = 0.5 * (joints["hip_r"] + joints["hip_l"])
    return {"time_s": time, "rotations": rotations, "centers": joints, "cluster_rms_max_m": residuals}


def _quaternion(rotation: np.ndarray) -> np.ndarray:
    """Rotation matrices [..., 3, 3] to xyzw quaternions with continuous sign."""
    r = np.asarray(rotation, dtype=float)
    w = np.sqrt(np.maximum(0.0, 1.0 + r[..., 0, 0] + r[..., 1, 1] + r[..., 2, 2])) / 2
    x = np.sqrt(np.maximum(0.0, 1.0 + r[..., 0, 0] - r[..., 1, 1] - r[..., 2, 2])) / 2
    y = np.sqrt(np.maximum(0.0, 1.0 - r[..., 0, 0] + r[..., 1, 1] - r[..., 2, 2])) / 2
    z = np.sqrt(np.maximum(0.0, 1.0 - r[..., 0, 0] - r[..., 1, 1] + r[..., 2, 2])) / 2
    x = np.copysign(x, r[..., 2, 1] - r[..., 1, 2])
    y = np.copysign(y, r[..., 0, 2] - r[..., 2, 0])
    z = np.copysign(z, r[..., 1, 0] - r[..., 0, 1])
    q = np.stack((x, y, z, w), axis=-1)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    return q


def coordinates(tracked: dict, calibration: Calibration) -> np.ndarray:
    """Return model coordinates per frame, shape [frames, 17], see :data:`COORDINATE_NAMES`."""
    rotations = tracked["rotations"]
    pelvis = rotations["pelvis"]
    columns = [tracked["centers"]["hip_mid"], _quaternion(pelvis)]
    for side in SIDES:
        s = _suffix(side)
        relative = np.swapaxes(pelvis, -1, -2) @ rotations[f"thigh_{s}"] @ calibration.offsets[f"hip_{s}"].T
        hip = np.unwrap(hip_angles(relative, side), axis=0)
        thigh = pelvis @ hip_matrix(hip, side) @ calibration.offsets[f"hip_{s}"]
        knee_relative = np.swapaxes(thigh, -1, -2) @ rotations[f"shank_{s}"] @ calibration.offsets[f"knee_{s}"].T
        knee = np.unwrap(hinge_angle(knee_relative, KNEE_AXIS))
        shank = thigh @ axis_rotation(KNEE_AXIS, knee) @ calibration.offsets[f"knee_{s}"]
        ankle_relative = np.swapaxes(shank, -1, -2) @ rotations[f"foot_{s}"] @ calibration.offsets[f"ankle_{s}"].T
        ankle = np.unwrap(hinge_angle(ankle_relative, ANKLE_AXIS))
        columns.append(np.column_stack((hip, knee, ankle)))
    return np.column_stack(columns)


def model_frames(q: np.ndarray, calibration: Calibration) -> dict[str, np.ndarray]:
    """Forward kinematics of :func:`coordinates`: body rotations and joint centers per frame."""
    q = np.asarray(q, dtype=float)
    x, y, z, w = q[:, 3], q[:, 4], q[:, 5], q[:, 6]
    pelvis = np.stack(
        (
            np.stack((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)), -1),
            np.stack((2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)), -1),
            np.stack((2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)), -1),
        ),
        axis=-2,
    )
    result = {"pelvis": pelvis, "hip_mid": q[:, :3]}
    for i, side in enumerate(SIDES):
        s = _suffix(side)
        joint = q[:, 7 + 5 * i : 12 + 5 * i]
        thigh = pelvis @ hip_matrix(joint[:, :3], side) @ calibration.offsets[f"hip_{s}"]
        shank = thigh @ axis_rotation(KNEE_AXIS, joint[:, 3]) @ calibration.offsets[f"knee_{s}"]
        foot = shank @ axis_rotation(ANKLE_AXIS, joint[:, 4]) @ calibration.offsets[f"ankle_{s}"]
        hip = q[:, :3] + pelvis @ calibration.hip_m[side]
        knee = hip + thigh @ calibration.knee_m[side]
        ankle = knee + shank @ calibration.ankle_m[side]
        result.update(
            {
                f"thigh_{s}": thigh,
                f"shank_{s}": shank,
                f"foot_{s}": foot,
                f"hip_{s}": hip,
                f"knee_{s}": knee,
                f"ankle_{s}": ankle,
            }
        )
    return result


def angle_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Geodesic angle between rotation matrices [rad]."""
    relative = np.swapaxes(a, -1, -2) @ b
    return np.arccos(np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1) / 2, -1.0, 1.0))


def _rotation_error(model: np.ndarray, target: np.ndarray) -> np.ndarray:
    relative = np.swapaxes(model, -1, -2) @ target
    return 0.5 * np.stack(
        (
            relative[..., 2, 1] - relative[..., 1, 2],
            relative[..., 0, 2] - relative[..., 2, 0],
            relative[..., 1, 0] - relative[..., 0, 1],
        ),
        axis=-1,
    )


@dataclass(frozen=True)
class LegFitWeights:
    """Tolerances of the per-frame leg fit."""

    position_m: float = 0.01
    """Knee and ankle joint-center tolerance [m]."""
    foot_rad: float = 0.1
    """Shoe orientation tolerance [rad]."""
    thigh_rad: float = 0.3
    """Weak thigh-cluster orientation tolerance that fixes hip rotation [rad]."""


def _leg_residual(
    joint: np.ndarray,
    side: str,
    pelvis: np.ndarray,
    hip: np.ndarray,
    tracked: dict,
    calibration: Calibration,
    weights: LegFitWeights,
) -> np.ndarray:
    s = _suffix(side)
    thigh = pelvis @ hip_matrix(joint[:, :3], side) @ calibration.offsets[f"hip_{s}"]
    shank = thigh @ axis_rotation(KNEE_AXIS, joint[:, 3]) @ calibration.offsets[f"knee_{s}"]
    foot = shank @ axis_rotation(ANKLE_AXIS, joint[:, 4]) @ calibration.offsets[f"ankle_{s}"]
    knee = hip + thigh @ calibration.knee_m[side]
    ankle = knee + shank @ calibration.ankle_m[side]
    targets = tracked["centers"]
    return np.concatenate(
        (
            (knee - targets[f"knee_{s}"]) / weights.position_m,
            (ankle - targets[f"ankle_{s}"]) / weights.position_m,
            _rotation_error(foot, tracked["rotations"][f"foot_{s}"]) / weights.foot_rad,
            _rotation_error(thigh, tracked["rotations"][f"thigh_{s}"]) / weights.thigh_rad,
        ),
        axis=1,
    )


def fit_legs(
    q: np.ndarray, tracked: dict, calibration: Calibration, weights: LegFitWeights | None = None, iterations: int = 15
) -> tuple[np.ndarray, dict]:
    """Refine both legs' joint coordinates against tracked joint centers and shoe frames.

    The pelvis pose stays tracked. Per frame and leg, Gauss-Newton minimizes the
    weighted knee and ankle center errors, shoe orientation error, and a weak
    thigh orientation prior, so the hinge knees and ankles absorb as much of the
    omitted knee and subtalar rotation as their single axes allow.

    Returns:
        Refined coordinates and per-side peak residuals.
    """
    weights = weights or LegFitWeights()
    q = np.array(q, dtype=float)
    report = {}
    pelvis = model_frames(q, calibration)["pelvis"]
    for i, side in enumerate(SIDES):
        columns = slice(7 + 5 * i, 12 + 5 * i)
        context = (side, pelvis, q[:, :3] + pelvis @ calibration.hip_m[side], tracked, calibration, weights)
        joint = q[:, columns].copy()
        for _ in range(iterations):
            r = _leg_residual(joint, *context)
            jacobian = np.empty((*r.shape, 5))
            for k in range(5):
                step = np.zeros(5)
                step[k] = 1e-6
                jacobian[:, :, k] = (_leg_residual(joint + step, *context) - r) / 1e-6
            normal = np.einsum("nrk,nrj->nkj", jacobian, jacobian) + 1e-9 * np.eye(5)
            joint -= np.linalg.solve(normal, np.einsum("nrk,nr->nk", jacobian, r)[:, :, None])[:, :, 0]
        q[:, columns] = joint
        r = _leg_residual(joint, *context)
        report[side] = {
            "knee_error_peak_m": float(np.linalg.norm(r[:, 0:3], axis=1).max() * weights.position_m),
            "ankle_error_peak_m": float(np.linalg.norm(r[:, 3:6], axis=1).max() * weights.position_m),
            "foot_error_peak_rad": float(np.linalg.norm(r[:, 6:9], axis=1).max() * weights.foot_rad),
            "thigh_error_peak_rad": float(np.linalg.norm(r[:, 9:12], axis=1).max() * weights.thigh_rad),
        }
    return q, report
