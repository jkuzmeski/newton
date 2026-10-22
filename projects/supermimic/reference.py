# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""F01 running reference for the SuperMimic runner.

Builds :class:`~projects.supermimic.body.BodyParams` for subject F01 and
reference trajectories in the runner's generalized coordinates from the
Visual3D exports in ``data/F01`` (trials FR3_1 and FR3_2, 200 Hz markers,
1000 Hz single treadmill force plate).

Segment frames are set in each trial's static pose. The pelvis frame is the
static stance frame: z vertical and y along the anterior superior iliac spines.
The thigh frame has z from the knee to the hip center and y along the femoral
epicondyles. The shank and foot frames are the closest frames the hinge knee
and ankle reach from the thigh and shank frames. The anatomical shank is set by
the malleoli. The anatomical foot is level and points from the heel markers to
the metatarsal heads. Markers are fixed in these frames. Each motion frame is
fitted by Gauss-Newton over the root pose and the ten joint angles, minimizing
the squared distances of all 32 markers.

Markers and plate forces are low-pass filtered by the same zero-phase
4th-order 20 Hz Butterworth filter. The reference stays in the treadmill frame:
the belt carries stance feet backward and the runner stays in place.
"""

from __future__ import annotations

import argparse
import datetime
import enum
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

from projects.gait_c3d.c3d_adapter import lab_to_newton_rotation
from projects.impedance_instron.cartesian.visual3d import load_visual3d_export

from .body import JOINT_NAMES, BodyParams, SegmentParams, mirror_axis

__all__ = [
    "COORDINATE_NAMES",
    "F01_ROOT",
    "TRIALS",
    "DeLevaTable",
    "f01_body_params",
    "fit_coordinates",
    "forward_kinematics",
    "load_body_params",
    "main",
    "prepare_reference",
    "trial_reference",
]

F01_ROOT = Path(__file__).resolve().parents[2] / "data" / "F01"
"""Visual3D exports of subject F01."""

TRIALS = ("FR3_1", "FR3_2")
"""Running trials in the FR3 shoe; each has its own static calibration."""

COORDINATE_NAMES = ("x", "y", "z", "qx", "qy", "qz", "qw", *JOINT_NAMES)
"""Generalized coordinates: root position [m], root quaternion (xyzw), joint angles [rad]."""

STATURE_M = 1.65
"""F01 stature, the Visual3D model metric ``Height`` in ``data/F01/Simple_model.mdh`` [m]."""

FILTER_CUTOFF_HZ = 20.0
CONTACT_THRESHOLD_N = 50.0
MARKER_RMS_LIMIT_M = 0.02
MARKER_MAX_LIMIT_M = 0.04
"""Frames whose marker fit exceeds either limit are rejected [m]; OpenSim's inverse-kinematics guidance
recommends an RMS marker error below 2 cm and a largest error below 2-4 cm."""
WINDOW_S = 2.0
EVAL_WINDOWS = 5
"""Windows at the end of each trial held out for evaluation; the window before them is a dropped guard."""
FLIGHT_MARGIN_FRAMES = 2
"""Episode starts keep this many frames of flight from either contact."""

SIDES = ("r", "l")
CLUSTERS = {
    "pelvis": ("RASI", "LASI", "RPSI", "LPSI"),
    "thigh": ("TH1", "TH2", "TH3", "TH4"),
    "shank": ("SH1", "SH2", "SH3", "SH4"),
    "foot": ("CAL1", "CAL2", "CAL3", "MT1H", "MT5H", "TOE"),
}
SEGMENTS = ("pelvis", "thigh_r", "shank_r", "foot_r", "thigh_l", "shank_l", "foot_l")


class DeLevaTable(enum.Enum):
    """Sex-specific table of de Leva (1996)."""

    FEMALE = "female"
    MALE = "male"


# de Leva P. (1996) Adjustments to Zatsiorsky-Seluyanov's segment inertia parameters.
# J. Biomech. 29(9):1223-1230, Table 4. Per segment: mean length [m], mass fraction,
# COM from the first endpoint and radii of gyration about the sagittal (anteroposterior),
# transverse (mediolateral) and longitudinal axes, as fractions of the length.
# Endpoints: head VERT-CERV, trunk CERV-MIDH, upper arm SJC-EJC, forearm EJC-WJC,
# hand WJC-MET3, thigh HJC-KJC, shank KJC-LMAL, foot HEEL-TTIP.
_DE_LEVA = {
    DeLevaTable.MALE: (
        1.741,
        {
            "head": (0.2429, 0.0694, 0.5976, 0.362, 0.376, 0.312),
            "trunk": (0.5319, 0.4346, 0.4486, 0.372, 0.347, 0.191),
            "upper_arm": (0.2817, 0.0271, 0.5772, 0.285, 0.269, 0.158),
            "forearm": (0.2689, 0.0162, 0.4574, 0.276, 0.265, 0.121),
            "hand": (0.0862, 0.0061, 0.7900, 0.628, 0.513, 0.401),
            "thigh": (0.4222, 0.1416, 0.4095, 0.329, 0.329, 0.149),
            "shank": (0.4340, 0.0433, 0.4459, 0.255, 0.249, 0.103),
            "foot": (0.2581, 0.0137, 0.4415, 0.257, 0.245, 0.124),
        },
    ),
    DeLevaTable.FEMALE: (
        1.735,
        {
            "head": (0.2437, 0.0668, 0.5894, 0.330, 0.359, 0.318),
            "trunk": (0.5293, 0.4257, 0.4151, 0.357, 0.339, 0.171),
            "upper_arm": (0.2751, 0.0255, 0.5754, 0.278, 0.260, 0.148),
            "forearm": (0.2643, 0.0138, 0.4559, 0.261, 0.257, 0.094),
            "hand": (0.0780, 0.0056, 0.7474, 0.531, 0.454, 0.335),
            "thigh": (0.3685, 0.1478, 0.3612, 0.369, 0.364, 0.162),
            "shank": (0.4323, 0.0481, 0.4416, 0.271, 0.267, 0.093),
            "foot": (0.2283, 0.0129, 0.4014, 0.299, 0.279, 0.139),
        },
    ),
}
SHOULDER_WIDTH = 0.259
"""Biacromial width as a fraction of stature (Winter, Biomechanics and Motor Control of Human Movement, 2009, Fig. 4.1)."""
WEIGHED_SHOE_MASS_KG = 0.173
"""Mass of one FR3 shoe [kg] (RunRepeat lab scale, US men's 9), assumed worn when F01's model mass was weighed."""


# Rotations ---------------------------------------------------------------


def _axis_rotation(axis, angle) -> np.ndarray:
    """Rotation matrices [..., 3, 3] about a unit ``axis`` by ``angle`` [rad]."""
    axis = np.asarray(axis, dtype=float)
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    angle = np.asarray(angle, dtype=float)[..., None, None]
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


def _rotvec_rotation(v: np.ndarray) -> np.ndarray:
    """Rotation matrices [..., 3, 3] of rotation vectors [..., 3]."""
    angle = np.linalg.norm(v, axis=-1)
    axis = v / np.maximum(angle, 1.0e-12)[..., None]
    k = np.zeros((*v.shape[:-1], 3, 3))
    k[..., 0, 1], k[..., 0, 2], k[..., 1, 2] = -axis[..., 2], axis[..., 1], -axis[..., 0]
    k = k - np.swapaxes(k, -1, -2)
    a = angle[..., None, None]
    return np.eye(3) + np.sin(a) * k + (1.0 - np.cos(a)) * (k @ k)


def _rotation_vector(r: np.ndarray) -> np.ndarray:
    """Rotation vectors [..., 3] of rotation matrices [..., 3, 3] (angles below pi)."""
    sine = 0.5 * np.stack((r[..., 2, 1] - r[..., 1, 2], r[..., 0, 2] - r[..., 2, 0], r[..., 1, 0] - r[..., 0, 1]), -1)
    norm = np.linalg.norm(sine, axis=-1, keepdims=True)
    angle = np.arctan2(norm, (np.trace(r, axis1=-2, axis2=-1)[..., None] - 1.0) / 2.0)
    return sine * angle / np.maximum(norm, 1.0e-12)


def _matrix_quaternion(r: np.ndarray) -> np.ndarray:
    """Quaternions (xyzw) [..., 4] of rotation matrices, with a continuous sign along the first axis."""
    w = np.sqrt(np.maximum(0.0, 1.0 + r[..., 0, 0] + r[..., 1, 1] + r[..., 2, 2])) / 2.0
    x = np.copysign(
        np.sqrt(np.maximum(0.0, 1.0 + r[..., 0, 0] - r[..., 1, 1] - r[..., 2, 2])) / 2.0, r[..., 2, 1] - r[..., 1, 2]
    )
    y = np.copysign(
        np.sqrt(np.maximum(0.0, 1.0 - r[..., 0, 0] + r[..., 1, 1] - r[..., 2, 2])) / 2.0, r[..., 0, 2] - r[..., 2, 0]
    )
    z = np.copysign(
        np.sqrt(np.maximum(0.0, 1.0 - r[..., 0, 0] - r[..., 1, 1] + r[..., 2, 2])) / 2.0, r[..., 1, 0] - r[..., 0, 1]
    )
    q = np.stack((x, y, z, w), axis=-1)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    if q.ndim == 2:
        flips = np.concatenate(([False], np.einsum("ij,ij->i", q[1:], q[:-1]) < 0.0))
        q[np.cumsum(flips) % 2 == 1] *= -1.0
    return q


def _quaternion_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrices [..., 3, 3] of quaternions (xyzw)."""
    x, y, z, w = np.moveaxis(q / np.linalg.norm(q, axis=-1, keepdims=True), -1, 0)
    return np.stack(
        (
            np.stack((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)), -1),
            np.stack((2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)), -1),
            np.stack((2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)), -1),
        ),
        axis=-2,
    )


def _side_axes(params: BodyParams, side: str) -> tuple[np.ndarray, ...]:
    """Hip (3), knee and ankle axes of one side."""
    axes = (*params.hip_axes, params.knee_axis, params.ankle_axis)
    return tuple(np.asarray(a, dtype=float) if side == "r" else mirror_axis(a) for a in axes)


def _mirror(point, side: str) -> np.ndarray:
    return np.asarray(point, dtype=float) * ((1.0, -1.0, 1.0) if side == "l" else 1.0)


def _hip_angles(r: np.ndarray, side: str) -> np.ndarray:
    """Hip flexion, adduction and internal rotation [rad] of relative rotations ``R(-y) R(s x) R(s z)``."""
    s = 1.0 if side == "r" else -1.0
    add = np.arcsin(np.clip(-r[..., 1, 2], -1.0, 1.0))
    return np.stack((-np.arctan2(r[..., 0, 2], r[..., 2, 2]), s * add, s * np.arctan2(r[..., 1, 0], r[..., 1, 1])), -1)


def _hinge_angle(r: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Closest angle [rad] about a +/-y hinge ``axis`` of relative rotations."""
    return np.sign(axis[1]) * np.arctan2(r[..., 0, 2], r[..., 2, 2])


# Kinematics ---------------------------------------------------------------


def _frames(position, rotation, angles, params: BodyParams) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """World rotations [..., 3, 3] and origins [..., 3] of the bodies of :data:`SEGMENTS`."""
    frames = {"pelvis": (rotation, position)}
    for i, side in enumerate(SIDES):
        a = angles[..., 5 * i : 5 * i + 5]
        hip, add, rot, knee, ankle = _side_axes(params, side)
        thigh = (
            rotation @ _axis_rotation(hip, a[..., 0]) @ _axis_rotation(add, a[..., 1]) @ _axis_rotation(rot, a[..., 2])
        )
        shank = thigh @ _axis_rotation(knee, a[..., 3])
        foot = shank @ _axis_rotation(ankle, a[..., 4])
        hip_p = position + rotation @ _mirror(params.hip_offset, side)
        knee_p = hip_p + thigh @ _mirror(params.knee_offset, side)
        ankle_p = knee_p + shank @ _mirror(params.ankle_offset, side)
        frames.update(
            {f"thigh_{side}": (thigh, hip_p), f"shank_{side}": (shank, knee_p), f"foot_{side}": (foot, ankle_p)}
        )
    return frames


def forward_kinematics(joint_q: np.ndarray, params: BodyParams) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Body rotations [..., 3, 3] and origins [..., 3] [m] of generalized coordinates [..., 17].

    Keys are ``pelvis``, ``thigh_r``, ``shank_r``, ``foot_r`` and their ``_l`` counterparts.
    """
    joint_q = np.asarray(joint_q, dtype=float)
    return _frames(joint_q[..., :3], _quaternion_matrix(joint_q[..., 3:7]), joint_q[..., 7:], params)


def _marker_positions(frames, local: dict[str, np.ndarray]) -> np.ndarray:
    """World marker positions [..., markers, 3] of the segments in ``local`` order."""
    return np.concatenate(
        [frames[s][1][..., None, :] + np.einsum("...ij,mj->...mi", frames[s][0], local[s]) for s in local], axis=-2
    )


def fit_coordinates(
    targets: np.ndarray,
    local: dict[str, np.ndarray],
    joint_q: np.ndarray,
    params: BodyParams,
    *,
    iterations: int = 10,
) -> np.ndarray:
    """Fit generalized coordinates to marker positions by Gauss-Newton.

    Args:
        targets: Marker positions [m], shape [frames, markers, 3], in ``local`` order.
        local: Marker positions in their segment frames [m], keyed by segment.
        joint_q: Initial coordinates, shape [frames, 17].
        params: Body geometry.
        iterations: Gauss-Newton iterations.

    Returns:
        Fitted coordinates, shape [frames, 17].
    """
    position = joint_q[:, :3].copy()
    rotation = _quaternion_matrix(joint_q[:, 3:7])
    angles = joint_q[:, 7:].copy()
    step = 1.0e-6

    def residual(p, r, a):
        return (_marker_positions(_frames(p, r, a, params), local) - targets).reshape(len(targets), -1)

    for _ in range(iterations):
        r0 = residual(position, rotation, angles)
        jacobian = np.empty((*r0.shape, 16))
        for k in range(16):
            d = np.zeros(16)
            d[k] = step
            jacobian[..., k] = (
                residual(position + d[:3], rotation @ _rotvec_rotation(d[3:6]), angles + d[6:]) - r0
            ) / step
        normal = np.einsum("nrk,nrj->nkj", jacobian, jacobian) + 1.0e-9 * np.eye(16)
        delta = -np.linalg.solve(normal, np.einsum("nrk,nr->nk", jacobian, r0)[..., None])[..., 0]
        position += delta[:, :3]
        rotation = rotation @ _rotvec_rotation(delta[:, 3:6])
        angles += delta[:, 6:]
    return np.concatenate((position, _matrix_quaternion(rotation), angles), axis=1)


def _marker_error(
    targets: np.ndarray, local: dict[str, np.ndarray], joint_q: np.ndarray, params: BodyParams
) -> np.ndarray:
    """Marker distances [m] between the fitted model and ``targets``, shape [frames, markers]."""
    return np.linalg.norm(_marker_positions(forward_kinematics(joint_q, params), local) - targets, axis=-1)


def _cluster_names(segment: str) -> list[str]:
    name, _, side = segment.partition("_")
    return list(CLUSTERS[name]) if name == "pelvis" else [side.upper() + m for m in CLUSTERS[name]]


def _kabsch(template: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Rotations [frames, 3, 3] taking centered ``template`` [m, 3] onto centered ``points`` [frames, m, 3]."""
    a = template - template.mean(0)
    b = points - points.mean(1, keepdims=True)
    u, _, vt = np.linalg.svd(np.einsum("fmi,mj->fij", b, a))
    d = np.sign(np.linalg.det(u @ vt))
    return u @ (np.eye(3) * np.stack((np.ones_like(d), np.ones_like(d), d), -1)[:, None, :]) @ vt


# Static calibration -------------------------------------------------------


def _read_columns(path: Path) -> tuple[list[str], list[str], np.ndarray]:
    """Labels, components and values [samples, columns] of a Visual3D ASCII export; blanks are NaN."""
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    rows = [[float(v) if v.strip() else np.nan for v in line.split("\t")[1:]] for line in lines[5:] if line.strip()]
    return lines[1].split("\t")[1:], lines[4].split("\t")[1:], np.asarray(rows)


def _read_mean_points(path: Path, rotation: np.ndarray) -> dict[str, np.ndarray]:
    """Mean positions [m] in Newton axes of a Visual3D point export, keyed by label."""
    labels, _, values = _read_columns(path)
    mean = np.nanmean(values.reshape(len(values), -1, 3), axis=0) @ rotation.T
    return dict(zip(labels[::3], mean, strict=True))


def _frame(primary: np.ndarray, hint: np.ndarray, *, primary_axis: str) -> np.ndarray:
    """Right-handed frame (columns x, y, z) with ``primary`` as its z or y axis and ``hint`` as y or z."""
    p = primary / np.linalg.norm(primary)
    if primary_axis == "z":
        x = np.cross(hint, p)
        x /= np.linalg.norm(x)
        return np.column_stack((x, np.cross(p, x), p))
    x = np.cross(p, hint)
    x /= np.linalg.norm(x)
    return np.column_stack((x, p, np.cross(x, p)))


@dataclass(frozen=True)
class _Static:
    """Static pose of one trial."""

    points: dict[str, np.ndarray]
    """Mean static marker positions [m]."""
    joint_q: np.ndarray
    """Generalized coordinates of the static pose, shape [17]."""
    origins: dict[str, np.ndarray]
    """Measured hip, knee and ankle centers and the mid-hip point [m]."""
    rotations: dict[str, np.ndarray]
    """Body rotations of the static pose."""
    foot_level: dict[str, np.ndarray]
    """Level anatomical foot frames per side."""


def _static_pose(trial_root: Path, params: BodyParams) -> _Static:
    """Static segment frames and coordinates; only ``params`` axes are used."""
    manifest = json.loads((trial_root / "visual3d_manifest.json").read_text(encoding="utf-8"))
    lab = lab_to_newton_rotation(manifest["up_axis"], manifest["forward_axis"])
    points = _read_mean_points(trial_root / "static_all_targets.txt", lab)
    centers = _read_mean_points(trial_root / "static_joint_centers.txt", lab)
    up = np.array([0.0, 0.0, 1.0])
    origins = {"pelvis": 0.5 * (centers["RHIP"] + centers["LHIP"])}
    pelvis = _frame(up, points["LASI"] - points["RASI"], primary_axis="z")
    rotations, foot_level, angles = {"pelvis": pelvis}, {}, []
    for side in SIDES:
        p = side.upper()
        hip, knee, ankle = centers[f"{p}HIP"], centers[f"{p}KNEE"], centers[f"{p}ANKLE"]
        # Medial minus lateral on the right and lateral minus medial on the left both point along +y.
        sign = 1.0 if side == "r" else -1.0
        thigh = _frame(hip - knee, sign * (points[f"{p}MFC"] - points[f"{p}LFC"]), primary_axis="z")
        shank_anatomical = _frame(knee - ankle, sign * (points[f"{p}MMAL"] - points[f"{p}LMAL"]), primary_axis="z")
        heel = np.mean([points[f"{p}CAL{i}"] for i in (1, 2, 3)], axis=0)
        heading = 0.5 * (points[f"{p}MT1H"] + points[f"{p}MT5H"]) - heel
        level = _frame(np.cross(up, heading), up, primary_axis="y")
        _, _, _, knee_axis, ankle_axis = _side_axes(params, side)
        hip_angles = _hip_angles(pelvis.T @ thigh, side)
        knee_angle = _hinge_angle(thigh.T @ shank_anatomical, knee_axis)
        shank = thigh @ _axis_rotation(knee_axis, knee_angle)
        ankle_angle = _hinge_angle(shank.T @ level, ankle_axis)
        rotations.update(
            {
                f"thigh_{side}": thigh,
                f"shank_{side}": shank,
                f"foot_{side}": shank @ _axis_rotation(ankle_axis, ankle_angle),
            }
        )
        origins.update({f"thigh_{side}": hip, f"shank_{side}": knee, f"foot_{side}": ankle})
        foot_level[side] = level
        angles.extend((*hip_angles, knee_angle, ankle_angle))
    joint_q = np.concatenate((origins["pelvis"], _matrix_quaternion(pelvis), angles))
    return _Static(points, joint_q, origins, rotations, foot_level)


def _vector(values) -> tuple[float, float, float]:
    return tuple(float(v) for v in values)


def _segment_inertia(mass: float, length: float, row, axes: tuple[int, int, int]) -> np.ndarray:
    """Principal inertia [kg m^2] with sagittal, transverse and longitudinal radii on body ``axes``."""
    inertia = np.empty(3)
    inertia[list(axes)] = mass * (np.asarray(row[3:]) * length) ** 2
    return inertia


def f01_body_params(table: DeLevaTable) -> BodyParams:
    """Return F01 body parameters from the static trials of :data:`TRIALS`.

    Joint offsets are averaged over the trials and the two sides. Leg masses,
    COM positions and radii of gyration follow de Leva (1996) Table 4 on the
    measured segment lengths. The pelvis carries the head, trunk and arms of
    the same table, scaled from the table's mean stature to
    :data:`STATURE_M`, upright above the mid-hip point with the arms hanging
    at a shoulder width of :data:`SHOULDER_WIDTH` times stature, scaled in
    mass so the body mass equals the Visual3D model mass less a pair of
    :data:`WEIGHED_SHOE_MASS_KG` shoes, which the simulator adds to the feet.
    ``sole_pitch`` levels the FR3 shoe frame in the static pose.

    Args:
        table: Sex-specific de Leva table; F01 is male.

    Returns:
        Body parameters of F01.
    """
    reference = BodyParams()
    hip, knee, ankle, heel, heel_to_toe, pitch, mass = [], [], [], [], [], [], []
    for trial in TRIALS:
        root = F01_ROOT / trial
        mass.append(float((root / "model_mass.txt").read_text().split()[-1]))
        static = _static_pose(root, reference)
        pelvis = static.rotations["pelvis"]
        for side in SIDES:
            m = np.array((1.0, -1.0, 1.0) if side == "l" else (1.0, 1.0, 1.0))
            o, r = static.origins, static.rotations
            hip.append(m * (pelvis.T @ (o[f"thigh_{side}"] - o["pelvis"])))
            knee.append(m * (r[f"thigh_{side}"].T @ (o[f"shank_{side}"] - o[f"thigh_{side}"])))
            ankle.append(m * (r[f"shank_{side}"].T @ (o[f"foot_{side}"] - o[f"shank_{side}"])))
            foot = r[f"foot_{side}"]
            p = side.upper()
            heel_point = np.mean([static.points[f"{p}CAL{i}"] for i in (1, 2, 3)], axis=0)
            heel.append(m * (foot.T @ (heel_point - o[f"foot_{side}"])))
            heel_to_toe.append(m * (foot.T @ (static.points[f"{p}TOE"] - heel_point)))
            pitch.append(_hinge_angle(foot.T @ static.foot_level[side], np.array((0.0, 1.0, 0.0))))
    body_mass = float(np.mean(mass))
    stature, rows = _DE_LEVA[table]
    scale = STATURE_M / stature
    hip_offset, knee_offset, ankle_offset, heel, heel_to_toe = (
        np.mean(v, axis=0) for v in (hip, knee, ankle, heel, heel_to_toe)
    )

    def leg(name: str, origin: np.ndarray, span: np.ndarray, axes: tuple[int, int, int]) -> SegmentParams:
        row = rows[name]
        mass = row[1] * body_mass
        inertia = _segment_inertia(mass, float(np.linalg.norm(span)), row, axes)
        return SegmentParams(float(mass), _vector(origin + row[2] * span), _vector(inertia))

    thigh = leg("thigh", np.zeros(3), knee_offset, (0, 1, 2))
    shank = leg("shank", np.zeros(3), ankle_offset, (0, 1, 2))
    # The shoe heel and toe markers span the foot; its longitudinal axis is x, transverse y, sagittal z.
    foot = leg("foot", heel, heel_to_toe, (2, 1, 0))

    # Head, trunk and arms, built in the pelvis frame, which is upright in the static stance.
    parts = []

    def vertical(name: str, top: np.ndarray) -> np.ndarray:
        row = rows[name]
        length = row[0] * scale
        com = top - np.array((0.0, 0.0, row[2] * length))
        parts.append((row[1] * body_mass, com, _segment_inertia(row[1] * body_mass, length, row, (0, 1, 2))))
        return top - np.array((0.0, 0.0, length))

    cervicale = np.array((0.0, 0.0, rows["trunk"][0] * scale))
    vertical("trunk", cervicale)
    vertical("head", cervicale + np.array((0.0, 0.0, rows["head"][0] * scale)))
    for y in (0.5, -0.5):
        shoulder = cervicale + np.array((0.0, y * SHOULDER_WIDTH * STATURE_M, 0.0))
        vertical("hand", vertical("forearm", vertical("upper_arm", shoulder)))
    leg_mass = thigh.mass + shank.mass + foot.mass
    part_mass = sum(m for m, _, _ in parts)
    pelvis_mass = body_mass - 2.0 * (leg_mass + WEIGHED_SHOE_MASS_KG)
    factor = pelvis_mass / part_mass
    com = sum(m * c for m, c, _ in parts) / part_mass
    inertia = sum(
        np.diag(i) + m * (np.dot(c - com, c - com) * np.eye(3) - np.outer(c - com, c - com)) for m, c, i in parts
    )
    pelvis = SegmentParams(pelvis_mass, _vector(com), _vector(factor * np.diag(inertia)))
    return BodyParams(
        pelvis=pelvis,
        thigh=thigh,
        shank=shank,
        foot=foot,
        hip_offset=_vector(hip_offset),
        knee_offset=_vector(knee_offset),
        ankle_offset=_vector(ankle_offset),
        sole_pitch=float(np.mean(pitch)),
    )


# Reference ----------------------------------------------------------------


def _low_pass(values: np.ndarray, rate_hz: float) -> np.ndarray:
    """Zero-phase 4th-order Butterworth low-pass at :data:`FILTER_CUTOFF_HZ` along the first axis."""
    return sosfiltfilt(butter(4, FILTER_CUTOFF_HZ, fs=rate_hz, output="sos"), values, axis=0)


def _intervals(mask: np.ndarray) -> np.ndarray:
    """Start (inclusive) and end (exclusive) indices of the true runs of ``mask``, shape [runs, 2]."""
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    return edges.reshape(-1, 2)


def _angular_velocity(rotation: np.ndarray, dt: float) -> np.ndarray:
    """World angular velocities [rad/s] of rotations [frames, 3, 3] by central differences."""
    omega = np.gradient(rotation, dt, axis=0) @ np.swapaxes(rotation, -1, -2)
    return 0.5 * np.stack(
        (omega[:, 2, 1] - omega[:, 1, 2], omega[:, 0, 2] - omega[:, 2, 0], omega[:, 1, 0] - omega[:, 0, 1]), -1
    )


def trial_reference(trial_root: Path, params: BodyParams) -> dict[str, np.ndarray | dict]:
    """Fit one trial and split its plate signals by foot.

    Positions are in the treadmill frame, whose belt runs backward along -x,
    with the origin on the belt surface below the mean mid-hip position. Foot index 0 is
    right and 1 is left. Off the plate a foot has zero force and free moment
    and a NaN COP.

    Args:
        trial_root: Visual3D export directory of one trial.
        params: Body parameters, for example from :func:`f01_body_params`.

    Returns:
        Marker-rate arrays ``time_s`` [s] (frames), ``joint_q`` (frames, 17),
        ``joint_qd`` (frames, 16; Newton free-joint convention: pelvis COM
        velocity [m/s] and angular velocity [rad/s] in world, then joint rates
        [rad/s]), ``valid`` (frames; marker fit within the limits),
        ``contact`` (frames, 2), ``grf`` [N], ``cop`` [m] and ``free_moment``
        [N m] (frames, 2, 3); the plate arrays at the analog rate as
        ``analog_time_s``, ``analog_grf``, ``analog_cop`` and
        ``analog_free_moment``; and the ``checks`` of ``report.json``.
    """
    trial = load_visual3d_export(trial_root)
    if not trial.marker_valid.all():
        raise ValueError(f"{trial_root} has missing marker samples")
    rate, dt = trial.point_rate_hz, 1.0 / trial.point_rate_hz
    names = [n for s in SEGMENTS for n in _cluster_names(s)]
    targets = _low_pass(trial.marker_positions_m, rate)[:, [trial.marker_names.index(n) for n in names]]

    static = _static_pose(trial_root, params)
    static_frames = forward_kinematics(static.joint_q, params)
    local = {
        s: (np.asarray([static.points[n] for n in _cluster_names(s)]) - static_frames[s][1]) @ static_frames[s][0]
        for s in SEGMENTS
    }
    bounds = dict(zip(SEGMENTS, np.cumsum([0, *(len(m) for m in local.values())]), strict=False))
    cluster = {s: targets[:, bounds[s] : bounds[s] + len(local[s])] for s in SEGMENTS}
    rotation = {s: _kabsch(local[s], cluster[s]) for s in SEGMENTS}
    pelvis = rotation["pelvis"]
    position = cluster["pelvis"].mean(1) - pelvis @ local["pelvis"].mean(0)
    angles = []
    for side in SIDES:
        _, _, _, knee_axis, ankle_axis = _side_axes(params, side)
        thigh, shank, foot = (rotation[f"{n}_{side}"] for n in ("thigh", "shank", "foot"))
        angles += [
            _hip_angles(np.swapaxes(pelvis, -1, -2) @ thigh, side),
            _hinge_angle(np.swapaxes(thigh, -1, -2) @ shank, knee_axis)[:, None],
            _hinge_angle(np.swapaxes(shank, -1, -2) @ foot, ankle_axis)[:, None],
        ]
    initial = np.concatenate((position, _matrix_quaternion(pelvis), *angles), axis=1)
    joint_q = fit_coordinates(targets, local, initial, params)
    joint_q[:, 7:] = np.unwrap(joint_q[:, 7:], axis=0)
    error = _marker_error(targets, local, joint_q, params)
    valid = (np.sqrt(np.mean(error**2, axis=1)) <= MARKER_RMS_LIMIT_M) & (error.max(axis=1) <= MARKER_MAX_LIMIT_M)
    rms = {s: np.sqrt(np.mean(error[:, bounds[s] : bounds[s] + len(local[s])] ** 2, axis=1)) for s in SEGMENTS}

    # One plate under both feet: forces and plate moments are filtered, the COP and free moment are
    # recomputed from them, and each contact belongs to the foot nearest its COP.
    analog_rate = trial.analog_rate_hz
    loaded = np.isfinite(trial.cop_m[:, 0])
    plate_z = float(np.median(trial.cop_m[loaded, 2]))
    arm = np.where(loaded[:, None], trial.cop_m, 0.0)
    force = _low_pass(trial.force_n, analog_rate)
    moment = _low_pass(np.cross(arm, trial.force_n) + np.where(loaded[:, None], trial.free_moment_nm, 0.0), analog_rate)
    on = force[:, 2] > CONTACT_THRESHOLD_N
    cop = np.full_like(force, np.nan)
    cop[on, 0] = (plate_z * force[on, 0] - moment[on, 1]) / force[on, 2]
    cop[on, 1] = (moment[on, 0] + plate_z * force[on, 1]) / force[on, 2]
    cop[on, 2] = plate_z
    free_moment = np.zeros_like(force)
    free_moment[on, 2] = moment[on, 2] - cop[on, 0] * force[on, 1] + cop[on, 1] * force[on, 0]

    frame_sample = np.rint((trial.marker_time_s - trial.analog_time_s[0]) * analog_rate).astype(int)
    if not np.allclose(trial.analog_time_s[frame_sample], trial.marker_time_s, atol=0.25 / analog_rate):
        raise ValueError(f"{trial_root} marker and analog clocks do not align")
    feet = {side: cluster[f"foot_{side}"].mean(1) for side in SIDES}
    contacts = []
    analog_foot = np.full(len(force), -1)
    for a, b in _intervals(on):
        center = np.mean(cop[a:b, :2], axis=0)
        frame = int(np.clip(np.searchsorted(frame_sample, (a + b) // 2), 0, len(frame_sample) - 1))
        distance = [float(np.linalg.norm(feet[side][frame, :2] - center)) for side in SIDES]
        contacts.append((a, b, int(np.argmin(distance)), distance))
        analog_foot[a:b] = contacts[-1][2]

    # The reference stays in the treadmill frame, centered on the mean mid-hip position. The lab
    # z = 0 is the belt surface; the plate surface lies below it.
    center = np.array((*joint_q[:, :2].mean(axis=0), 0.0))
    joint_q[:, :3] -= center
    cop -= center
    midstance_speed = []
    for a, b, foot, _ in contacts[1:-1]:
        f0, f1 = np.searchsorted(frame_sample, (a + 2 * (b - a) // 5, b - 2 * (b - a) // 5))
        midstance_speed.append(-float(np.mean(np.gradient(feet[SIDES[foot]][:, 0], dt)[f0:f1])))
    time = trial.marker_time_s - trial.marker_time_s[0]
    plate = {}
    for name, values, off in (("grf", force, 0.0), ("cop", cop, np.nan), ("free_moment", free_moment, 0.0)):
        per_foot = np.full((len(force), 2, 3), off)
        for i in range(2):
            per_foot[analog_foot == i, i] = values[analog_foot == i]
        plate[name] = per_foot

    root_rotation = _quaternion_matrix(joint_q[:, 3:7])
    com = joint_q[:, :3] + root_rotation @ np.asarray(params.pelvis.com)
    joint_qd = np.concatenate(
        (np.gradient(com, dt, axis=0), _angular_velocity(root_rotation, dt), np.gradient(joint_q[:, 7:], dt, axis=0)),
        axis=1,
    )

    # Checks against the unfiltered plate and the Visual3D joint angles.
    raw = _intervals(trial.force_n[:, 2] > CONTACT_THRESHOLD_N)
    inner = [(a, b, foot) for a, b, foot, _ in contacts[1:-1]]
    timing = []
    for a, b, _ in inner:
        match = raw[(raw[:, 0] < b) & (raw[:, 1] > a)]
        timing.append(((a - match[0, 0]) / analog_rate, (b - match[-1, 1]) / analog_rate, len(match)))
    timing = np.asarray(timing)
    duration = (contacts[-1][0] - contacts[0][0]) / analog_rate
    contact_time = {side: [(b - a) / analog_rate for a, b, foot in inner if foot == i] for i, side in enumerate(SIDES)}
    flight = [(inner[i + 1][0] - inner[i][1]) / analog_rate for i in range(len(inner) - 1)]
    labels, components, values = _read_columns(trial_root / "motion_joint_angles.txt")
    visual3d = {}
    for side in SIDES:
        for name, column in (("hip_flexion", "Hip"), ("knee_flexion", "Knee"), ("ankle_dorsiflexion", "Ankle")):
            j = next(
                k
                for k, (lab, c) in enumerate(zip(labels, components, strict=True))
                if lab == f"{side.upper()}{column}Angle" and c == "X"
            )
            ours = np.degrees(joint_q[:, 7 + JOINT_NAMES.index(f"{name}_{side}")])
            theirs = values[:, j]
            visual3d[f"{name}_{side}"] = {
                "correlation": float(np.corrcoef(ours, theirs)[0, 1]),
                "mean_difference_deg": float(np.mean(ours - theirs)),
                "visual3d_range_deg": [float(theirs.min()), float(theirs.max())],
            }
    checks = {
        "frames": len(joint_q),
        "rejected_frames": int((~valid).sum()),
        "marker_error_mm": {
            "frame_rms_p95": 1e3 * float(np.percentile(np.sqrt(np.mean(error**2, axis=1)), 95)),
            "frame_max_p95": 1e3 * float(np.percentile(error.max(axis=1), 95)),
        },
        "segment_rms_mm": {
            s: {"mean": 1e3 * float(r.mean()), "p95": 1e3 * float(np.percentile(r, 95)), "max": 1e3 * float(r.max())}
            for s, r in rms.items()
        },
        "belt_speed_manifest_m_s": float(trial.manifest["belt_speed_m_s"]),
        "midstance_foot_speed_m_s": float(np.median(midstance_speed)),
        "contacts": {"reference": len(contacts), "raw_plate": len(raw), "right": sum(c[2] == 0 for c in contacts)},
        "alternating": bool(np.all(np.diff([c[2] for c in contacts]) != 0)),
        "cop_to_foot_m": {
            "assigned_max": max(c[3][c[2]] for c in contacts),
            "other_min": min(c[3][1 - c[2]] for c in contacts),
        },
        "timing_vs_raw_plate_ms": {
            "touchdown_mean": 1e3 * float(timing[:, 0].mean()),
            "touchdown_max_abs": 1e3 * float(np.abs(timing[:, 0]).max()),
            "toe_off_mean": 1e3 * float(timing[:, 1].mean()),
            "toe_off_max_abs": 1e3 * float(np.abs(timing[:, 1]).max()),
            "raw_contacts_per_reference_contact_max": int(timing[:, 2].max()),
        },
        "cadence_steps_per_min": 60.0 * (len(contacts) - 1) / duration,
        "contact_time_ms": {
            s: {"mean": 1e3 * float(np.mean(t)), "std": 1e3 * float(np.std(t))} for s, t in contact_time.items()
        },
        "flight_time_ms": {"mean": 1e3 * float(np.mean(flight)), "std": 1e3 * float(np.std(flight))},
        "joint_range_deg": {
            name: [float(np.degrees(joint_q[valid, 7 + i].min())), float(np.degrees(joint_q[valid, 7 + i].max()))]
            for i, name in enumerate(JOINT_NAMES)
        },
        "visual3d_sagittal": visual3d,
    }
    return {
        "time_s": time,
        "joint_q": joint_q,
        "joint_qd": joint_qd,
        "valid": valid,
        "contact": analog_foot[frame_sample, None] == np.arange(2),
        **{name: values[frame_sample] for name, values in plate.items()},
        "analog_time_s": trial.analog_time_s - trial.marker_time_s[0],
        **{f"analog_{name}": values for name, values in plate.items()},
        "checks": checks,
    }


def load_body_params(path: Path) -> BodyParams:
    """Read the ``body_params.json`` written by :func:`prepare_reference`."""

    def tuples(value):
        return tuple(tuples(v) for v in value) if isinstance(value, list) else value

    data = {k: tuples(v) for k, v in json.loads(Path(path).read_text(encoding="utf-8")).items()}
    segments = {
        name: SegmentParams(**{k: tuples(v) for k, v in data.pop(name).items()})
        for name in ("pelvis", "thigh", "shank", "foot")
    }
    return BodyParams(**segments, **data)


def prepare_reference(output: Path, table: DeLevaTable) -> dict:
    """Write the F01 reference windows, body parameters and checks to ``output``.

    Each trial is cut into consecutive :data:`WINDOW_S` windows; windows with a
    rejected frame are dropped. The last :data:`EVAL_WINDOWS` windows of each
    trial are the evaluation block and the window before them is dropped as a
    guard.

    ``reference.npz`` holds per window ``joint_q``, ``joint_qd``, ``contact``,
    ``grf`` and ``cop`` as returned by :func:`trial_reference` (shape
    [windows, frames, ...]), ``trial``, ``start_frame`` and ``split``
    (``train`` or ``eval``); the episode starts ``flight_window`` and
    ``flight_frame``, where both feet are off the plate for
    :data:`FLIGHT_MARGIN_FRAMES` frames on either side; ``coordinate_names``
    and ``frame_dt`` [s]. ``body_params.json`` holds the body parameters.

    Returns:
        The checks written to ``report.json``.
    """
    params = f01_body_params(table)
    output.mkdir(parents=True, exist_ok=False)
    arrays = {k: [] for k in ("joint_q", "joint_qd", "contact", "grf", "cop", "trial", "start_frame", "split")}
    report = {
        "body_table": table.value,
        "trials": {},
        "marker_limits_m": {"rms": MARKER_RMS_LIMIT_M, "max": MARKER_MAX_LIMIT_M},
    }
    for trial in TRIALS:
        data = trial_reference(F01_ROOT / trial, params)
        report["trials"][trial] = data["checks"]
        rate = 1.0 / (data["time_s"][1] - data["time_s"][0])
        length = round(WINDOW_S * rate)
        starts = np.arange(0, len(data["joint_q"]) - length + 1, length)
        split = np.full(len(starts), "train", dtype=object)
        split[-EVAL_WINDOWS:] = "eval"
        split[-EVAL_WINDOWS - 1] = "guard"
        for start, kind in zip(starts, split, strict=True):
            window = slice(start, start + length)
            if kind == "guard" or not data["valid"][window].all():
                continue
            for key, value in (("trial", trial), ("start_frame", start), ("split", kind)):
                arrays[key].append(value)
            for key in ("joint_q", "joint_qd", "contact", "grf", "cop"):
                arrays[key].append(data[key][window])
        report["trials"][trial]["windows"] = {
            "total": len(starts),
            "kept": int(sum(t == trial for t in arrays["trial"])),
        }
    arrays = {k: np.asarray(v) for k, v in arrays.items()}
    arrays["split"] = arrays["split"].astype(str)
    airborne = ~arrays["contact"].any(-1)
    margin = FLIGHT_MARGIN_FRAMES
    clear = np.ones_like(airborne)
    for shift in range(-margin, margin + 1):
        clear &= np.roll(airborne, shift, axis=1)
    clear[:, :margin] = clear[:, -margin:] = False
    flight_window, flight_frame = np.nonzero(clear)
    np.savez_compressed(
        output / "reference.npz",
        **arrays,
        flight_window=flight_window,
        flight_frame=flight_frame,
        coordinate_names=np.asarray(COORDINATE_NAMES),
        frame_dt=1.0 / rate,
    )
    report["windows"] = {kind: int((arrays["split"] == kind).sum()) for kind in ("train", "eval")}
    report["flight_starts"] = {kind: int((arrays["split"][flight_window] == kind).sum()) for kind in ("train", "eval")}
    (output / "body_params.json").write_text(json.dumps(asdict(params), indent=2) + "\n", encoding="utf-8")
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare the F01 SuperMimic reference.")
    parser.add_argument("--table", required=True, choices=[t.value for t in DeLevaTable], help="de Leva (1996) table")
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/supermimic") / f"reference_{datetime.date.today()}"
    )
    args = parser.parse_args(argv)
    report = prepare_reference(args.output, DeLevaTable(args.table))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
