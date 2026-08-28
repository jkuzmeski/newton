# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Build Visual3D-style static segment calibration artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .c3d_adapter import C3DMarkerTrajectory

_SCHEMA = "gait_static_segment_calibration_1"
_REQUIRED_MARKERS = (
    "LASI",
    "RASI",
    "LPSI",
    "RPSI",
    "LKNE",
    "LMKNE",
    "RKNE",
    "RMKNE",
    "LANK",
    "LMANK",
    "RANK",
    "RMANK",
    "LHEE",
    "LTOE",
    "RHEE",
    "RTOE",
    "LMTH1",
    "LMTH5",
    "RMTH1",
    "RMTH5",
)
_ALL_MARKER_SOURCES = tuple(
    dict.fromkeys(
        (
            *_REQUIRED_MARKERS,
            "STRN",
            "RSHO",
            "LSHO",
            "LFHD",
            "RFHD",
            "LBHD",
            "RBHD",
            "C7",
            "CLAV",
            "RBAK",
            "LUPA",
            "LELB",
            "LFRM",
            "LWRA",
            "LWRB",
            "LFIN",
            "RUPA",
            "RELB",
            "RFRM",
            "RWRA",
            "RWRB",
            "RFIN",
            "LIC",
            "RIC",
            "LPSI",
            "RPSI",
            "LTHI",
            "RTHI",
            "LTH2",
            "LTH3",
            "LTH4",
            "RTH2",
            "RTH3",
            "RTH4",
            "LTIB2",
            "LTIB3",
            "LTIB4",
            "RTIB2",
            "RTIB3",
            "RTIB4",
            "LHLX",
            "LHEE2",
            "LHEE3",
            "LMTH1",
            "LMTH5",
            "RHLX",
            "RHEE2",
            "RHEE3",
            "RMTH1",
            "RMTH5",
            "FLeft",
            "FRight",
            "ORight",
            "BLeft",
            "BRight",
            "T10",
        )
    )
)
_CANONICAL_SOURCES = {
    "L.ASIS": "LASI",
    "R.ASIS": "RASI",
    "L.Knee.Lat": "LKNE",
    "L.Knee.Med": "LMKNE",
    "R.Knee.Lat": "RKNE",
    "R.Knee.Med": "RMKNE",
    "L.Ankle.Lat": "LANK",
    "L.Ankle.Med": "LMANK",
    "R.Ankle.Lat": "RANK",
    "R.Ankle.Med": "RMANK",
    "L.Heel": "LHEE",
    "R.Heel": "RHEE",
    "L.Toe.Tip": "LTOE",
    "R.Toe.Tip": "RTOE",
    "L.Toe.Med": "LMTH1",
    "L.Toe.Lat": "LMTH5",
    "R.Toe.Med": "RMTH1",
    "R.Toe.Lat": "RMTH5",
    "L.Thigh.Upper": "LTH2",
    "L.Thigh.Front": "LTH3",
    "L.Thigh.Rear": "LTH4",
    "R.Thigh.Upper": "RTH2",
    "R.Thigh.Front": "RTH3",
    "R.Thigh.Rear": "RTH4",
    "L.Shank.Upper": "LTIB2",
    "L.Shank.Front": "LTIB3",
    "L.Shank.Rear": "LTIB4",
    "R.Shank.Upper": "RTIB2",
    "R.Shank.Front": "RTIB3",
    "R.Shank.Rear": "RTIB4",
    "Sternum": "STRN",
    "L.Acromium": "LSHO",
    "R.Acromium": "RSHO",
}


def _canonical_json(value: dict) -> bytes:
    """Serialize calibration content deterministically for a SHA-256 seal."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _unit(vector: np.ndarray, *, name: str) -> np.ndarray:
    """Normalize a finite nonzero vector."""
    vector = np.asarray(vector, dtype=np.float64)
    length = float(np.linalg.norm(vector))
    if not np.isfinite(length) or length <= 1.0e-10:
        raise ValueError(f"{name} must be finite and nonzero")
    return vector / length


def _project(vector: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Project a vector onto the plane perpendicular to a unit normal."""
    return np.asarray(vector, dtype=np.float64) - np.dot(vector, normal) * normal


def _as_list(vector: np.ndarray) -> list[float]:
    """Convert a finite vector to JSON floats."""
    vector = np.asarray(vector, dtype=np.float64)
    if not np.all(np.isfinite(vector)):
        raise ValueError("calibration values must be finite")
    return [float(value) for value in vector]


def _basis(right: np.ndarray, anterior: np.ndarray, *, name: str) -> np.ndarray:
    """Construct a right/anterior/up orthonormal basis."""
    right = _unit(right, name=f"{name} right axis")
    anterior = _unit(_project(anterior, right), name=f"{name} anterior axis")
    up = _unit(np.cross(right, anterior), name=f"{name} up axis")
    if up[2] < 0.0:
        up = -up
        anterior = -anterior
    return np.column_stack((right, anterior, up))


def _segment_basis(proximal: np.ndarray, distal: np.ndarray, lateral: np.ndarray, *, name: str) -> np.ndarray:
    """Build a forward/left/longitudinal segment basis."""
    longitudinal = _unit(proximal - distal, name=f"{name} longitudinal axis")
    left = _unit(_project(lateral, longitudinal), name=f"{name} left axis")
    forward = _unit(np.cross(left, longitudinal), name=f"{name} forward axis")
    left = _unit(np.cross(longitudinal, forward), name=f"{name} left axis")
    if np.dot(forward, np.asarray((1.0, 0.0, 0.0))) < 0.0:
        forward = -forward
        left = -left
    return np.column_stack((forward, left, longitudinal))


def _foot_basis(heel: np.ndarray, toe: np.ndarray, lateral: np.ndarray, *, name: str) -> np.ndarray:
    """Build a horizontal forward/left/up foot basis."""
    up = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    forward = _unit(_project(toe - heel, up), name=f"{name} forward axis")
    left = _unit(np.cross(up, forward), name=f"{name} left axis")
    lateral = _project(lateral, forward)
    if np.dot(left, lateral) < 0.0:
        left = -left
        forward = -forward
    return np.column_stack((forward, left, up))


@dataclass(frozen=True, slots=True)
class SegmentCalibration:
    """A verified static marker calibration and derived segment landmarks."""

    path: Path
    """Sealed calibration JSON path."""

    marker_positions: dict[str, np.ndarray]
    """Static marker-center positions in Newton world coordinates [m]."""

    marker_radius: float
    """Physical marker radius used for anatomical ASIS correction [m]."""

    time_range: tuple[float, float]
    """Static calibration time interval [s]."""

    pelvis: dict
    """CODA pelvis origin, basis, ASIS distance, and hip centers."""

    segments: dict[str, dict]
    """Per-side segment endpoints, frames, lengths, and widths."""


def _static_marker_positions(
    markers: C3DMarkerTrajectory,
    time_range: tuple[float, float] | None,
) -> tuple[dict[str, np.ndarray], tuple[float, float]]:
    """Average valid marker centers over a static calibration interval."""
    if time_range is None:
        frame_mask = np.ones(len(markers.times), dtype=bool)
        selected_range = (float(markers.times[0]), float(markers.times[-1]))
    else:
        if len(time_range) != 2 or not all(math.isfinite(float(value)) for value in time_range):
            raise ValueError("time_range must contain two finite values")
        if time_range[1] < time_range[0]:
            raise ValueError("time_range must be increasing")
        frame_mask = (markers.times >= time_range[0]) & (markers.times <= time_range[1])
        selected_range = (float(time_range[0]), float(time_range[1]))
    if not np.any(frame_mask):
        raise ValueError("static calibration time range contains no frames")
    index = {name: column for column, name in enumerate(markers.marker_names)}
    missing = [name for name in _REQUIRED_MARKERS if name not in index]
    if missing:
        raise ValueError(f"static calibration is missing markers: {missing}")
    output = {}
    for name in _ALL_MARKER_SOURCES:
        if name not in index:
            if name in _REQUIRED_MARKERS:
                raise ValueError(f"static calibration is missing markers: [{name!r}]")
            continue
        column = index[name]
        valid = frame_mask & markers.valid[:, column]
        if not np.any(valid):
            if name in _REQUIRED_MARKERS:
                raise ValueError(f"static calibration has no valid samples for marker {name!r}")
            continue
        value = np.mean(markers.positions[valid, column], axis=0, dtype=np.float64)
        if not np.all(np.isfinite(value)):
            raise ValueError(f"static calibration marker {name!r} is nonfinite")
        output[name] = value
    if "LPSI" in output and "RPSI" in output:
        output["VSAC"] = 0.5 * (output["LPSI"] + output["RPSI"])
    if all(name in output for name in ("LFHD", "RFHD", "LBHD", "RBHD")):
        output["TOPHEAD"] = 0.25 * (output["LFHD"] + output["RFHD"] + output["LBHD"] + output["RBHD"])
    return output, selected_range


def _segment_record(
    proximal: np.ndarray,
    distal: np.ndarray,
    lateral_vector: np.ndarray,
    *,
    name: str,
    width: float,
    marker_names: tuple[str, str],
) -> dict:
    """Create one endpoint-based segment calibration record."""
    if not math.isfinite(width) or width <= 0.0:
        raise ValueError(f"{name} width must be finite and positive")
    basis = _segment_basis(proximal, distal, lateral_vector, name=name)
    return {
        "proximal_m": _as_list(proximal),
        "distal_m": _as_list(distal),
        "origin_m": _as_list(proximal),
        "basis_forward_left_longitudinal": basis.tolist(),
        "length_m": float(np.linalg.norm(proximal - distal)),
        "width_m": float(width),
        "width_markers": list(marker_names),
    }


def build_static_segment_calibration(
    markers: C3DMarkerTrajectory,
    output_path: str | os.PathLike,
    *,
    marker_radius: float,
    time_range: tuple[float, float] | None = (0.5, 1.0),
) -> SegmentCalibration:
    """Build and seal a Visual3D-style SCS calibration from static C3D markers.

    Args:
        markers: Newton-frame static marker trajectory.
        output_path: Destination calibration JSON path.
        marker_radius: Physical marker radius used for ASIS correction [m].
        time_range: Inclusive static interval [s].

    Returns:
        The verified segment calibration.
    """
    if not math.isfinite(marker_radius) or marker_radius < 0.0:
        raise ValueError("marker_radius must be finite and nonnegative")
    marker_positions, selected_range = _static_marker_positions(markers, time_range)
    lasi, rasi = marker_positions["LASI"], marker_positions["RASI"]
    lpsi, rpsi = marker_positions["LPSI"], marker_positions["RPSI"]
    right_axis = _unit(rasi - lasi, name="ASIS right axis")
    sacrum = 0.5 * (lpsi + rpsi)
    preliminary_anterior = _project(0.5 * (lasi + rasi) - sacrum, right_axis)
    pelvis_basis = _basis(right_axis, preliminary_anterior, name="pelvis")
    anatomical_lasi = lasi - marker_radius * pelvis_basis[:, 1]
    anatomical_rasi = rasi - marker_radius * pelvis_basis[:, 1]
    pelvis_origin = 0.5 * (anatomical_lasi + anatomical_rasi)
    pelvis_basis = _basis(anatomical_rasi - anatomical_lasi, pelvis_origin - sacrum, name="pelvis")
    asis_distance = float(np.linalg.norm(anatomical_rasi - anatomical_lasi))
    hip_offset = (
        0.36 * asis_distance * pelvis_basis[:, 0]
        - 0.19 * asis_distance * pelvis_basis[:, 1]
        - 0.30 * asis_distance * pelvis_basis[:, 2]
    )
    right_hip = pelvis_origin + hip_offset
    left_hip = (
        pelvis_origin
        - 0.36 * asis_distance * pelvis_basis[:, 0]
        - 0.19 * asis_distance * pelvis_basis[:, 1]
        - 0.30 * asis_distance * pelvis_basis[:, 2]
    )
    pelvis = {
        "origin_m": _as_list(pelvis_origin),
        "basis_right_anterior_up": pelvis_basis.tolist(),
        "asis_distance_m": asis_distance,
        "asis_markers": {"left": _as_list(anatomical_lasi), "right": _as_list(anatomical_rasi)},
        "sacrum_m": _as_list(sacrum),
        "hip_centers_m": {"left": _as_list(left_hip), "right": _as_list(right_hip)},
        "hip_method": "coda_bell_brand",
    }
    segments = {}
    for side, prefix in (("left", "L"), ("right", "R")):
        knee_lateral = marker_positions[f"{prefix}KNE"]
        knee_medial = marker_positions[f"{prefix}MKNE"]
        ankle_lateral = marker_positions[f"{prefix}ANK"]
        ankle_medial = marker_positions[f"{prefix}MANK"]
        knee_center = 0.5 * (knee_lateral + knee_medial)
        ankle_center = 0.5 * (ankle_lateral + ankle_medial)
        knee_width = float(np.linalg.norm(knee_lateral - knee_medial))
        ankle_width = float(np.linalg.norm(ankle_lateral - ankle_medial))
        if side == "left":
            knee_left_vector = knee_lateral - knee_medial
            ankle_left_vector = ankle_lateral - ankle_medial
            foot_lateral_vector = marker_positions[f"{prefix}MTH5"] - marker_positions[f"{prefix}MTH1"]
        else:
            knee_left_vector = knee_medial - knee_lateral
            ankle_left_vector = ankle_medial - ankle_lateral
            foot_lateral_vector = marker_positions[f"{prefix}MTH1"] - marker_positions[f"{prefix}MTH5"]
        thigh = _segment_record(
            left_hip if side == "left" else right_hip,
            knee_center,
            knee_left_vector,
            name=f"{side} thigh",
            width=knee_width,
            marker_names=(f"{prefix}KNE", f"{prefix}MKNE"),
        )
        shank = _segment_record(
            knee_center,
            ankle_center,
            ankle_left_vector,
            name=f"{side} shank",
            width=ankle_width,
            marker_names=(f"{prefix}ANK", f"{prefix}MANK"),
        )
        heel = marker_positions[f"{prefix}HEE"]
        toe = marker_positions[f"{prefix}TOE"]
        foot_basis = _foot_basis(heel, toe, foot_lateral_vector, name=f"{side} foot")
        segments[f"thigh_{side}"] = thigh
        segments[f"shank_{side}"] = shank
        segments[f"foot_{side}"] = {
            "proximal_m": _as_list(ankle_center),
            "distal_m": _as_list(toe),
            "origin_m": _as_list(ankle_center),
            "basis_forward_left_up": foot_basis.tolist(),
            "length_m": float(np.linalg.norm(_project(toe - heel, np.asarray((0.0, 0.0, 1.0))))),
            "width_m": float(np.linalg.norm(marker_positions[f"{prefix}MTH5"] - marker_positions[f"{prefix}MTH1"])),
            "heel_m": _as_list(heel),
            "toe_m": _as_list(toe),
            "ankle_center_m": _as_list(ankle_center),
            "width_markers": (f"{prefix}MTH1", f"{prefix}MTH5"),
            "flat_ground": True,
        }
    manifest = {
        "schema_version": _SCHEMA,
        "coordinate_system": {
            "frame": "Newton world",
            "length_unit": "m",
            "up_axis": "Z",
            "forward_axis": "X",
            "left_axis": "Y",
        },
        "source": {
            "file": markers.source_file,
            "sha256": markers.source_sha256,
            "first_frame": markers.first_frame,
            "point_rate_hz": markers.rate,
        },
        "calibration": {
            "time_range_s": list(selected_range),
            "marker_radius_m": float(marker_radius),
            "marker_names": list(marker_positions),
        },
        "base_marker_set": "S001" if markers.source_file.startswith("Cal 101") else None,
        "pelvis": pelvis,
        "segments": segments,
        "markers": {name: _as_list(value) for name, value in marker_positions.items()},
    }
    manifest["seal"] = {"algorithm": "sha256", "content_sha256": hashlib.sha256(_canonical_json(manifest)).hexdigest()}
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output.parent, delete=False) as stream:
        staged = Path(stream.name)
        json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    os.replace(staged, output)
    return load_static_segment_calibration(output)


def load_static_segment_calibration(path: str | os.PathLike) -> SegmentCalibration:
    """Verify and load a sealed static segment calibration."""
    calibration_path = Path(path).expanduser().resolve()
    manifest = json.loads(calibration_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("segment calibration must be a JSON object")
    seal = manifest.pop("seal", None)
    expected = hashlib.sha256(_canonical_json(manifest)).hexdigest()
    if seal != {"algorithm": "sha256", "content_sha256": expected}:
        raise ValueError("segment calibration seal mismatch")
    if manifest.get("schema_version") != _SCHEMA:
        raise ValueError("unsupported segment calibration schema")
    calibration = manifest.get("calibration")
    if not isinstance(calibration, dict):
        raise ValueError("segment calibration metadata is missing")
    radius = calibration.get("marker_radius_m")
    time_range = calibration.get("time_range_s")
    if not isinstance(radius, (int, float)) or not math.isfinite(radius) or radius < 0.0:
        raise ValueError("segment calibration marker radius is invalid")
    if (
        not isinstance(time_range, list)
        or len(time_range) != 2
        or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in time_range)
    ):
        raise ValueError("segment calibration time range is invalid")
    marker_positions = {
        name: np.asarray(value, dtype=np.float64) for name, value in manifest.get("markers", {}).items()
    }
    if not set(_REQUIRED_MARKERS).issubset(marker_positions) or any(
        value.shape != (3,) or not np.all(np.isfinite(value)) for value in marker_positions.values()
    ):
        raise ValueError("segment calibration markers are incomplete or invalid")
    segments = manifest.get("segments")
    if not isinstance(segments, dict) or set(segments) != {
        "thigh_left",
        "thigh_right",
        "shank_left",
        "shank_right",
        "foot_left",
        "foot_right",
    }:
        raise ValueError("segment calibration segments are incomplete")
    for name, segment in segments.items():
        if not isinstance(segment, dict):
            raise ValueError(f"segment calibration {name!r} is invalid")
        length = segment.get("length_m")
        if not isinstance(length, (int, float)) or not math.isfinite(length) or length <= 0.0:
            raise ValueError(f"segment calibration {name!r} length is invalid")
    return SegmentCalibration(
        calibration_path,
        marker_positions,
        float(radius),
        (float(time_range[0]), float(time_range[1])),
        manifest["pelvis"],
        segments,
    )
