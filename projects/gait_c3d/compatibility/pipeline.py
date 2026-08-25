# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""OFFLINE SOURCE/ANALYSIS REFERENCE ONLY. It uses newton.opensim compatibility mechanics; production simulation begins at the sealed neutral Newton artifact.

Build the S001 C3D-to-OpenSim gait analysis artifacts.

The raw human data remain outside the repository. This project pipeline records
all inferred treadmill timing, units, axes, and frame transformations rather
than presenting them as source metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

import newton.examples
import newton.opensim as osim

ARCHITECTURE_ROLE = "compatibility_reference"

_SCHEMA_VERSION = "gait_c3d_analysis_3"
_BELT_ANCHOR_INDICES = np.array([0, 1356, 22244, 43139, 52098, 53223], dtype=float)
_BELT_ANCHOR_TIMES = np.array([0.0, 4.68, 74.69, 144.70, 174.71, 178.46], dtype=float)
_G = 9.80665
_CACHE_SCHEMA_VERSION = "gait_c3d_trial_cache_5"
_EXPECTED_MARKER_COUNT = 35


def _signal_tools():
    """Import optional SciPy signal helpers lazily."""
    from scipy.signal import butter, sosfiltfilt

    return butter, sosfiltfilt


def _json_value(value):
    """Convert NumPy values into JSON-compatible values."""
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: str | os.PathLike, data: dict) -> None:
    """Write deterministic indented JSON."""
    Path(path).write_text(
        json.dumps(_json_value(data), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def sha256(path: str | os.PathLike) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _runtime_provenance(repository_root: Path, device: str) -> dict:
    """Record code and package versions for a generated analysis."""
    try:
        git_commit = subprocess.check_output(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"], text=True
        ).strip()
        git_dirty = bool(
            subprocess.check_output(["git", "-C", str(repository_root), "status", "--porcelain"], text=True).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        git_commit = "unknown"
        git_dirty = True

    def version(distribution: str) -> str:
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            return "unknown"

    return {
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": version("scipy"),
        "ezc3d": version("ezc3d"),
        "warp": version("warp-lang"),
        "newton": version("newton"),
        "device": device,
    }


def read_visual3d_metric(path: str | os.PathLike) -> tuple[np.ndarray, np.ndarray]:
    """Read a strict two-column Visual3D metric export.

    Returns one-based item indices and scalar values. Blank tab fields are
    ignored, while all remaining fields must be numeric and indices sequential.
    """
    rows: list[tuple[float, float]] = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines()[5:]:
        fields = [field for field in re.split(r"\s+", line.strip()) if field]
        if not fields:
            continue
        if len(fields) != 2:
            raise ValueError(f"expected two numeric fields in {path}: {line!r}")
        rows.append((float(fields[0]), float(fields[1])))
    if not rows:
        raise ValueError(f"no metric rows in {path}")
    values = np.asarray(rows, dtype=float)
    expected = np.arange(1, len(values) + 1, dtype=float)
    if not np.array_equal(values[:, 0], expected):
        raise ValueError(f"non-sequential metric item indices in {path}")
    return values[:, 0], values[:, 1]


def register_belt_clock(
    sample_count: int,
    *,
    anchor_indices: np.ndarray = _BELT_ANCHOR_INDICES,
    anchor_times: np.ndarray = _BELT_ANCHOR_TIMES,
) -> np.ndarray:
    """Map belt item indices to C3D seconds through documented onset anchors."""
    anchor_indices = np.asarray(anchor_indices, dtype=float)
    anchor_times = np.asarray(anchor_times, dtype=float)
    if anchor_indices.ndim != 1 or anchor_times.shape != anchor_indices.shape:
        raise ValueError("belt clock anchors must be equal-length 1-D arrays")
    if len(anchor_indices) < 2 or np.any(np.diff(anchor_indices) <= 0.0) or np.any(np.diff(anchor_times) <= 0.0):
        raise ValueError("belt clock anchors must increase strictly")
    if anchor_indices[0] != 0 or int(anchor_indices[-1]) != sample_count - 1:
        raise ValueError("belt clock anchors must cover the complete metric")
    return np.interp(np.arange(sample_count, dtype=float), anchor_indices, anchor_times)


def integrate_speed(times: np.ndarray, speed: np.ndarray) -> np.ndarray:
    """Integrate speed with the trapezoid rule, returning displacement from zero."""
    times = np.asarray(times, dtype=float)
    speed = np.asarray(speed, dtype=float)
    if times.ndim != 1 or speed.shape != times.shape or len(times) == 0:
        raise ValueError("times and speed must be nonempty equal-length vectors")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("times must increase strictly")
    out = np.zeros_like(times)
    if len(times) > 1:
        out[1:] = np.cumsum(0.5 * (speed[:-1] + speed[1:]) * np.diff(times))
    return out


def treadmill_to_overground(points: np.ndarray, displacement: np.ndarray, *, reference_index: int = 0) -> np.ndarray:
    """Apply the virtual-origin translation to OpenSim-ground points.

    The supplied belt metric is a positive speed magnitude. In this trial the
    backward belt velocity is OpenSim ``-X``; therefore mapped points receive
    ``+X`` displacement. A constant reference offset keeps a selected window
    near the viewer origin without changing velocities or dynamics.
    """
    points = np.asarray(points, dtype=float)
    displacement = np.asarray(displacement, dtype=float)
    if points.shape[0] != len(displacement) or points.shape[-1] != 3:
        raise ValueError("points and displacement frame counts must match")
    if not 0 <= reference_index < len(displacement):
        raise IndexError("reference_index is out of range")
    mapped = points.copy()
    relative = displacement - displacement[reference_index]
    shape = (len(relative),) + (1,) * (mapped.ndim - 2)
    mapped[..., 0] += relative.reshape(shape)
    return mapped


def transform_force_platform_arrays(
    force_lab: np.ndarray,
    cop_lab_mm: np.ndarray,
    torque_lab_nmm: np.ndarray,
    rotation: np.ndarray,
    displacement: np.ndarray,
    contact: np.ndarray,
    *,
    reference_index: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotate force-platform arrays and translate COP into the overground frame."""
    force = np.asarray(force_lab, float) @ np.asarray(rotation, float).T
    cop = np.asarray(cop_lab_mm, float) * 0.001 @ np.asarray(rotation, float).T
    torque = np.asarray(torque_lab_nmm, float) * 0.001 @ np.asarray(rotation, float).T
    cop = treadmill_to_overground(cop, displacement, reference_index=reference_index)
    active = np.asarray(contact, bool)
    force[~active] = 0.0
    torque[~active] = 0.0
    cop[~active] = np.nan
    return force, cop, torque


def contact_runs(contact: np.ndarray, *, min_frames: int = 20) -> list[tuple[int, int]]:
    """Return half-open contact runs at least ``min_frames`` long."""
    active = np.asarray(contact, dtype=bool).reshape(-1)
    edges = np.diff(np.r_[False, active, False].astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    return [(int(start), int(stop)) for start, stop in zip(starts, stops, strict=True) if stop - start >= min_frames]


def select_stride(times: np.ndarray, left_contact: np.ndarray, *, search_time: float = 20.0) -> tuple[int, int]:
    """Select the first complete left heel-strike stride after ``search_time``."""
    times = np.asarray(times, float)
    runs = contact_runs(left_contact, min_frames=max(1, int(round(0.2 / np.median(np.diff(times))))))
    starts = [start for start, _stop in runs if times[start] >= search_time]
    if len(starts) < 2:
        raise ValueError(f"fewer than two left contacts after {search_time:.3f} s")
    return starts[0], starts[1] + 1


def validate_external_load_schema(labels: list[str]) -> None:
    """Require exact ordered XYZ triplets for both external loads."""
    expected = []
    for side in ("l", "r"):
        expected += [f"ground_force_{side}_v{axis}" for axis in "xyz"]
        expected += [f"ground_force_{side}_p{axis}" for axis in "xyz"]
        expected += [f"ground_torque_{side}_{axis}" for axis in "xyz"]
    if labels != expected:
        raise ValueError(f"external-load labels do not match the required schema: {labels}")


def _marker_assignment(markers: osim.MarkerData) -> dict[str, str]:
    mapping = {
        name: osim.GAIT2354_VICON_ALIASES[name] for name in markers.marker_names if name in osim.GAIT2354_VICON_ALIASES
    }
    for name in osim.GAIT2354_VIRTUAL_MARKERS:
        if name in markers.marker_names:
            mapping[name] = name
    targets = list(mapping.values())
    if len(targets) != len(set(targets)):
        raise ValueError("marker assignment is not one-to-one")
    return mapping


def _mapped_markers(path: Path) -> osim.MarkerData:
    markers = osim.read_c3d(path, up_axis="+Z", forward_axis="-Y")
    markers = osim.synthesize_markers(markers, osim.GAIT2354_VIRTUAL_MARKERS)
    mapped = osim.apply_marker_assignment(markers, _marker_assignment(markers))
    if len(mapped.marker_names) != _EXPECTED_MARKER_COUNT:
        raise ValueError(f"expected {_EXPECTED_MARKER_COUNT} mapped markers, found {len(mapped.marker_names)}")
    return mapped


def filter_force_platform_wrench(
    force_lab: np.ndarray,
    moment_lab_nmm: np.ndarray,
    corners_lab_mm: np.ndarray,
    sos: np.ndarray,
    *,
    contact_threshold_n: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Filter a raw force-platform wrench before deriving COP and free torque.

    Args:
        force_lab: Ground-on-subject force samples in lab axes [N].
        moment_lab_nmm: Moments about the platform surface center [N·mm].
        corners_lab_mm: Platform surface corners, shape ``[3, corner_count]`` [mm].
        sos: SciPy second-order-section low-pass filter coefficients.
        contact_threshold_n: Minimum positive vertical force for contact [N].

    Returns:
        Filtered force [N], COP [mm], free torque [N·mm], contact mask,
        filtered moment about the surface center [N·mm], and maximum
        loaded-sample wrench-identity error [N·mm].
    """
    _butter, sosfiltfilt = _signal_tools()
    force = np.asarray(force_lab, float)
    moment = np.asarray(moment_lab_nmm, float)
    corners = np.asarray(corners_lab_mm, float)
    if force.ndim != 2 or force.shape[1] != 3 or moment.shape != force.shape:
        raise ValueError("force and moment must have matching [sample_count, 3] shapes")
    if corners.ndim != 2 or corners.shape[0] != 3 or corners.shape[1] < 3:
        raise ValueError("platform corners must have shape [3, corner_count]")
    if not np.all(np.isfinite(force)) or not np.all(np.isfinite(moment)) or not np.all(np.isfinite(corners)):
        raise ValueError("raw force, moment, and platform corners must be finite")
    if not np.isfinite(contact_threshold_n) or contact_threshold_n <= 0.0:
        raise ValueError("contact_threshold_n must be finite and positive")

    force = sosfiltfilt(sos, force, axis=0)
    moment = sosfiltfilt(sos, moment, axis=0)
    origin = np.mean(corners, axis=1)
    surface_z = float(np.mean(corners[2]))
    contact = force[:, 2] > contact_threshold_n
    cop = np.full_like(force, np.nan)
    torque = np.zeros_like(force)
    if np.any(contact):
        loaded_force = force[contact]
        loaded_moment = moment[contact]
        offset_z = surface_z - origin[2]
        cop[contact, 0] = origin[0] + (offset_z * loaded_force[:, 0] - loaded_moment[:, 1]) / loaded_force[:, 2]
        cop[contact, 1] = origin[1] + (loaded_moment[:, 0] + offset_z * loaded_force[:, 1]) / loaded_force[:, 2]
        cop[contact, 2] = surface_z
        lever = cop[contact] - origin
        torque[contact, 2] = loaded_moment[:, 2] - np.cross(lever, loaded_force)[:, 2]
        reconstructed = np.cross(lever, loaded_force) + torque[contact]
        identity_error = float(np.max(np.abs(reconstructed - loaded_moment)))
    else:
        identity_error = 0.0
    force[~contact] = 0.0
    return force, cop, torque, contact, moment, identity_error


def _extract_subject_mass(calibration_path: Path, cutoff_hz: float = 20.0) -> tuple[float, dict]:
    butter, sosfiltfilt = _signal_tools()
    try:
        import ezc3d  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ImportError("the gait C3D pipeline requires newton[opensim]") from exc
    c3d = ezc3d.c3d(str(calibration_path), extract_forceplat_data=True)
    analog_rate = float(c3d["parameters"]["ANALOG"]["RATE"]["value"][0])
    sos = butter(4, cutoff_hz, btype="low", fs=analog_rate, output="sos")
    vertical = []
    for platform in c3d["data"]["platform"]:
        force = sosfiltfilt(sos, np.asarray(platform["force"], float), axis=1)
        vertical.append(force[2])
    total = np.sum(vertical, axis=0)
    lo, hi = int(0.2 * len(total)), int(0.8 * len(total))
    candidate = total[lo:hi]
    median = float(np.median(candidate))
    mad = float(np.median(np.abs(candidate - median)))
    tolerance = max(5.0 * 1.4826 * mad, 1.0)
    stable = candidate[np.abs(candidate - median) <= tolerance]
    mean_force = float(np.mean(stable))
    std_force = float(np.std(stable))
    coefficient = std_force / mean_force
    if coefficient > 0.02:
        raise ValueError(f"calibration-force mass estimate is unstable: CV={coefficient:.3f}")
    return mean_force / _G, {
        "source": "calibration_force_platform_sum_divided_by_standard_gravity",
        "mean_vertical_force_N": mean_force,
        "std_vertical_force_N": std_force,
        "coefficient_of_variation": coefficient,
        "candidate_sample_range": [lo, hi],
        "stable_sample_count": int(len(stable)),
        "candidate_sample_count": int(len(candidate)),
        "median_absolute_deviation_N": mad,
        "gravity_mps2": _G,
    }


def _load_belt(incoming: Path, point_times: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    left_index, left = read_visual3d_metric(incoming / "LeftBelt101.txt")
    right_index, right = read_visual3d_metric(incoming / "RightBelt101.txt")
    if not np.array_equal(left_index, right_index) or not np.array_equal(left, right):
        raise ValueError("this pipeline currently requires tied, identical belt profiles")
    speed_change_values = None
    speed_change_path = incoming / "Speedchange101.txt"
    if speed_change_path.exists():
        _items, speed_change_values = read_visual3d_metric(speed_change_path)
        anchor_rows = np.array([0, 1, 3, 5, 6, 10], dtype=int)
        if len(speed_change_values) <= int(anchor_rows[-1]) or not np.allclose(
            speed_change_values[anchor_rows], _BELT_ANCHOR_TIMES, atol=1.0e-9
        ):
            raise ValueError("Speedchange101 values do not match the documented C3D-time anchors")
    metric_times = register_belt_clock(len(left))
    metric_displacement = integrate_speed(metric_times, left)
    speed = np.interp(point_times, metric_times, left, left=left[0], right=left[-1])
    displacement = np.interp(
        point_times,
        metric_times,
        metric_displacement,
        left=metric_displacement[0],
        right=metric_displacement[-1],
    )
    return (
        speed,
        displacement,
        {
            "kind": "onset_anchor_piecewise_clock_warp",
            "units": "m/s",
            "units_source": "inferred_from_protocol_not_encoded_in_metric",
            "direction_lab": "+Y backward",
            "direction_opensim": "-X backward",
            "left_right_identical": True,
            "anchor_item_indices_zero_based": _BELT_ANCHOR_INDICES.astype(int),
            "anchor_times_c3d_s": _BELT_ANCHOR_TIMES,
            "speed_change_values_s": speed_change_values if speed_change_values is not None else [],
            "metric_distance_m": float(metric_displacement[-1]),
            "point_grid_distance_m": float(displacement[-1]),
        },
    )


def _marker_data_meters(
    times: np.ndarray, marker_names: list[str], data_m: np.ndarray, rate_hz: float
) -> osim.MarkerData:
    """Construct marker data whose numeric values and metadata are both metres."""
    return osim.MarkerData(
        times=np.asarray(times, float),
        marker_names=list(marker_names),
        data=np.asarray(data_m, float),
        rate=float(rate_hz),
        units="m",
    )


def _cache_provenance(incoming: Path) -> dict:
    """Return every source/config value embedded in the extraction cache."""
    return {
        "schema_version": _CACHE_SCHEMA_VERSION,
        "source_hashes": {
            filename: sha256(incoming / filename)
            for filename in (
                "Trial 101.v3d.c3d",
                "LeftBelt101.txt",
                "RightBelt101.txt",
                "Speedchange101.txt",
            )
        },
        "extraction_config": {
            "up_axis": "+Z",
            "forward_axis": "-Y",
            "belt_anchor_indices": _BELT_ANCHOR_INDICES.astype(int).tolist(),
            "belt_anchor_times_s": _BELT_ANCHOR_TIMES.tolist(),
            "force_cutoff_hz": 20.0,
            "contact_threshold_N": 50.0,
            "wrench_processing": "filter_force_and_moment_then_derive_cop_and_free_torque",
            "wrench_raw_sources": ["platform.force", "platform.moment"],
            "wrench_derived_sources_not_filtered": ["platform.center_of_pressure", "platform.Tz"],
            "wrench_moment_reference": "mean platform surface corners",
            "wrench_equation": "M=(P-O)xF+T",
            "wrench_filter": {
                "family": "Butterworth",
                "order": 4,
                "representation": "second-order sections",
                "phase": "zero-phase sosfiltfilt",
                "cutoff_hz": 20.0,
            },
            "validated_platform_units": {"force": "N", "moment": "Nmm", "position": "mm"},
            "expected_marker_count": _EXPECTED_MARKER_COUNT,
            "trial_geometry": "level treadmill, lab -Y heading, no incline compensation",
        },
    }


def _extract_trial_cache(incoming: Path, cache_path: Path) -> dict[str, np.ndarray]:
    butter, _sosfiltfilt = _signal_tools()
    try:
        import ezc3d  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ImportError("the gait C3D pipeline requires newton[opensim]") from exc

    c3d_path = incoming / "Trial 101.v3d.c3d"
    c3d = ezc3d.c3d(str(c3d_path), extract_forceplat_data=True)
    point = c3d["parameters"]["POINT"]
    point_rate = float(point["RATE"]["value"][0])
    labels = [label.split(":")[-1].strip() for label in point["LABELS"]["value"]]
    data = np.asarray(c3d["data"]["points"], float)[:3].transpose(2, 1, 0)
    residual = np.asarray(c3d["data"]["meta_points"]["residuals"], float)[0].T
    missing = (residual < 0.0) | np.all(data == 0.0, axis=-1)
    data[missing] = np.nan
    rotation = osim.lab_to_opensim_rotation("+Z", "-Y")
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1.0e-12) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1.0e-12
    ):
        raise ValueError("lab-to-OpenSim axes must define a proper rotation")
    data = data @ rotation.T * 0.001
    times = np.arange(len(data), dtype=float) / point_rate
    markers = _marker_data_meters(times, labels, data, point_rate)
    markers = osim.synthesize_markers(markers, osim.GAIT2354_VIRTUAL_MARKERS)
    markers = osim.apply_marker_assignment(markers, _marker_assignment(markers))

    speed, displacement, belt_qc = _load_belt(incoming, times)
    analog_rate = float(c3d["parameters"]["ANALOG"]["RATE"]["value"][0])
    sos = butter(4, 20.0, btype="low", fs=analog_rate, output="sos")
    sample_index = np.minimum(np.rint(times * analog_rate).astype(int), len(c3d["data"]["platform"][0]["force"][0]) - 1)
    force_frames = []
    cop_frames = []
    torque_frames = []
    moment_frames = []
    contact_frames = []
    wrench_identity_errors_nmm = []
    platform_mean_x = []
    platform_surface_origins_lab_mm = []
    platforms = sorted(
        c3d["data"]["platform"],
        key=lambda platform: float(np.mean(np.asarray(platform["corners"], float)[0])),
        reverse=True,
    )
    if len(platforms) != 2:
        raise ValueError(f"expected exactly two force platforms, found {len(platforms)}")
    platform_corners = [np.asarray(platform["corners"], float) for platform in platforms]
    if not all(np.all(np.isfinite(corners)) for corners in platform_corners):
        raise ValueError("force-platform corners must be finite")
    platform_level_max_abs_z_mm = float(max(np.max(np.abs(corners[2])) for corners in platform_corners))
    platform_y_spans_mm = [float(np.ptp(corners[1])) for corners in platform_corners]
    platform_x_spans_mm = [float(np.ptp(corners[0])) for corners in platform_corners]
    if platform_level_max_abs_z_mm > 1.0e-3 or any(
        y_span <= x_span for x_span, y_span in zip(platform_x_spans_mm, platform_y_spans_mm, strict=True)
    ):
        raise ValueError("this mapping is restricted to the verified level treadmill with lab-Y heading")
    for platform in platforms:
        units = (str(platform["unit_force"]), str(platform["unit_moment"]), str(platform["unit_position"]))
        if units != ("N", "Nmm", "mm"):
            raise ValueError(f"unsupported force-platform units: {units}")
        corners = np.asarray(platform["corners"], float)
        surface_origin_lab_mm = np.mean(corners, axis=1)
        mean_x = float(surface_origin_lab_mm[0])
        platform_mean_x.append(mean_x)
        platform_surface_origins_lab_mm.append(surface_origin_lab_mm)
        force_lab, cop_lab, torque_lab, _active_analog, moment_lab, identity_error_nmm = filter_force_platform_wrench(
            np.asarray(platform["force"], float).T,
            np.asarray(platform["moment"], float).T,
            corners,
            sos,
            contact_threshold_n=50.0,
        )
        wrench_identity_errors_nmm.append(identity_error_nmm)
        force = (force_lab @ rotation.T)[sample_index]
        cop = (cop_lab * 0.001 @ rotation.T)[sample_index]
        torque = (torque_lab * 0.001 @ rotation.T)[sample_index]
        moment = (moment_lab * 0.001 @ rotation.T)[sample_index]
        active = force[:, 1] > 50.0
        force_frames.append(force)
        cop_frames.append(cop)
        torque_frames.append(torque)
        moment_frames.append(moment)
        contact_frames.append(active)
    if not (platform_mean_x[0] > 0.0 and platform_mean_x[1] < 0.0):
        raise ValueError(f"force platforms do not match left(+X)/right(-X) geometry: {platform_mean_x}")
    arrays = {
        "schema_version": np.asarray(_CACHE_SCHEMA_VERSION, dtype="U"),
        "source_sha256": np.asarray(sha256(c3d_path), dtype="U"),
        "cache_provenance_json": np.asarray(json.dumps(_cache_provenance(incoming), sort_keys=True), dtype="U"),
        "times": times,
        "point_rate_hz": np.asarray(point_rate, dtype=float),
        "analog_rate_hz": np.asarray(analog_rate, dtype=float),
        "force_cutoff_hz": np.asarray(20.0, dtype=float),
        "contact_threshold_N": np.asarray(50.0, dtype=float),
        "wrench_identity_max_abs_Nm": np.asarray(wrench_identity_errors_nmm, dtype=float) * 0.001,
        "filtered_moment_at_surface_origin_Nm": np.stack(moment_frames, axis=1),
        "platform_surface_origin_lab_mm": np.asarray(platform_surface_origins_lab_mm, dtype=float),
        "platform_surface_origin_opensim_m": np.asarray(platform_surface_origins_lab_mm, dtype=float)
        * 0.001
        @ rotation.T,
        "marker_names": np.asarray(markers.marker_names, dtype="U"),
        "markers_treadmill": np.asarray(markers.data, dtype=np.float64),
        "belt_speed": speed,
        "belt_displacement_absolute": displacement,
        "grf": np.stack(force_frames, axis=1),
        "cop_treadmill": np.stack(cop_frames, axis=1),
        "free_torque": np.stack(torque_frames, axis=1),
        "contact": np.stack(contact_frames, axis=1),
        "platform_mean_lab_x_mm": np.asarray(platform_mean_x, dtype=float),
        "platform_level_max_abs_z_mm": np.asarray(platform_level_max_abs_z_mm, dtype=float),
        "platform_x_spans_mm": np.asarray(platform_x_spans_mm, dtype=float),
        "platform_y_spans_mm": np.asarray(platform_y_spans_mm, dtype=float),
        "belt_qc_json": np.asarray(json.dumps(_json_value(belt_qc)), dtype="U"),
    }
    np.savez_compressed(cache_path, **arrays)
    return arrays


def _load_cache(incoming: Path, cache_path: Path, rebuild: bool) -> dict[str, np.ndarray]:
    source_path = incoming / "Trial 101.v3d.c3d"
    if rebuild or not cache_path.exists():
        return _extract_trial_cache(incoming, cache_path)
    with np.load(cache_path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    schema = str(arrays.get("schema_version", ""))
    source_digest = str(arrays.get("source_sha256", ""))
    cached_provenance = str(arrays.get("cache_provenance_json", ""))
    expected_provenance = json.dumps(_cache_provenance(incoming), sort_keys=True)
    if (
        schema != _CACHE_SCHEMA_VERSION
        or source_digest != sha256(source_path)
        or cached_provenance != expected_provenance
    ):
        return _extract_trial_cache(incoming, cache_path)
    return arrays


def _post_placement_static_error(
    model_path: Path,
    markers: osim.MarkerData,
    time_range: tuple[float, float],
    adjacent_range: tuple[float, float] = (0.61, 0.65),
) -> dict:
    """Evaluate marker-placement round-trip and a temporally adjacent temporal static window."""
    training = (markers.times >= time_range[0]) & (markers.times <= time_range[1])
    adjacent = (markers.times >= adjacent_range[0]) & (markers.times <= adjacent_range[1])
    if not np.any(training) or not np.any(adjacent):
        raise ValueError("static training and adjacent consistency windows must both contain frames")
    observed = {
        name: np.nanmean(markers.data[training, index], axis=0) for index, name in enumerate(markers.marker_names)
    }
    solver = osim.InverseKinematics(osim.parse_osim(model_path), batched=False)
    coordinates, rms, maximum = solver.solve_frame(observed)
    predicted = solver.fk.marker_positions(coordinates)
    adjacent_observed = {
        name: np.nanmean(markers.data[adjacent, index], axis=0) for index, name in enumerate(markers.marker_names)
    }
    adjacent_errors = np.asarray(
        [np.linalg.norm(predicted[name] - adjacent_observed[name]) for name in markers.marker_names], float
    )
    return {
        "roundtrip_rms_m": float(rms),
        "roundtrip_max_m": float(maximum),
        "adjacent_rms_m": float(np.sqrt(np.mean(adjacent_errors**2))),
        "adjacent_max_m": float(np.max(adjacent_errors)),
        "adjacent_time_range_s": list(adjacent_range),
        "adjacent_per_marker_m": dict(zip(markers.marker_names, adjacent_errors, strict=True)),
        "coordinates": dict(zip(solver.coordinate_names, coordinates, strict=True)),
        "note": "Round-trip values are circular consistency checks after marker relocation; adjacent values use later calibration frames and are not an independent trial.",
    }


def _external_load_table(grf: np.ndarray, cop: np.ndarray, torque: np.ndarray) -> tuple[list[str], np.ndarray]:
    """Pack bilateral OpenSim-ground load vectors into the strict storage schema."""
    labels: list[str] = []
    columns: list[np.ndarray] = []
    for side_index, side in enumerate(("l", "r")):
        labels += [f"ground_force_{side}_v{axis}" for axis in "xyz"]
        labels += [f"ground_force_{side}_p{axis}" for axis in "xyz"]
        labels += [f"ground_torque_{side}_{axis}" for axis in "xyz"]
        columns += [grf[:, side_index], np.nan_to_num(cop[:, side_index], nan=0.0), torque[:, side_index]]
    validate_external_load_schema(labels)
    return labels, np.concatenate(columns, axis=1)


def _write_external_loads(
    output_dir: Path,
    times: np.ndarray,
    grf: np.ndarray,
    cop: np.ndarray,
    torque: np.ndarray,
    *,
    stem: str = "trial_grf_context",
) -> tuple[Path, Path]:
    """Write a context-rich GRF storage and matching ExternalLoads XML."""
    labels, table = _external_load_table(grf, cop, torque)
    mot_path = output_dir / f"{stem}.mot"
    osim.write_storage(mot_path, times, labels, table, name="S001 treadmill GRF context", in_degrees=False)
    xml_path = output_dir / f"{stem}.xml"
    xml_path.write_text(
        f"""<?xml version="1.0" encoding="UTF-8" ?>
<OpenSimDocument Version="40000"><ExternalLoads name="S001"><objects>
<ExternalForce name="left"><applied_to_body>calcn_l</applied_to_body><force_expressed_in_body>ground</force_expressed_in_body><point_expressed_in_body>ground</point_expressed_in_body><force_identifier>ground_force_l_v</force_identifier><point_identifier>ground_force_l_p</point_identifier><torque_identifier>ground_torque_l_</torque_identifier></ExternalForce>
<ExternalForce name="right"><applied_to_body>calcn_r</applied_to_body><force_expressed_in_body>ground</force_expressed_in_body><point_expressed_in_body>ground</point_expressed_in_body><force_identifier>ground_force_r_v</force_identifier><point_identifier>ground_force_r_p</point_identifier><torque_identifier>ground_torque_r_</torque_identifier></ExternalForce>
</objects><groups/><datafile>{mot_path.name}</datafile></ExternalLoads></OpenSimDocument>
""",
        encoding="utf-8",
    )
    return mot_path, xml_path


class _SampledExternalLoads:
    """Return one validated, serialized-and-sampled wrench array without another spline fit."""

    def __init__(self, times: np.ndarray, bodies: list[str], wrenches: np.ndarray):
        self.times = np.asarray(times, float)
        self.bodies = list(bodies)
        self.wrenches = np.asarray(wrenches, float)

    def sample(self, output_times: np.ndarray) -> tuple[list[str], np.ndarray]:
        """Return exact pre-sampled wrenches at frames from the validated grid."""
        output_times = np.asarray(output_times, float)
        indices = np.searchsorted(self.times, output_times)
        if (
            output_times.ndim != 1
            or np.any(indices >= len(self.times))
            or not np.allclose(
                self.times[np.minimum(indices, len(self.times) - 1)], output_times, rtol=0.0, atol=1.0e-10
            )
        ):
            raise ValueError("exact external loads were requested outside the sampled time grid")
        return self.bodies, self.wrenches[indices].copy()


def _ik_result(model, markers: osim.MarkerData, device) -> tuple[osim.IKResult, np.ndarray]:
    solver = osim.InverseKinematics(model, device=device, batched=False, max_iters=60)
    values = np.empty((len(markers.times), len(solver.coordinate_names)), dtype=float)
    rms = np.empty(len(markers.times), dtype=float)
    maximum = np.empty(len(markers.times), dtype=float)
    previous = None
    for frame in range(len(markers.times)):
        previous, rms[frame], maximum[frame] = solver.solve_frame(markers.frame(frame), q0=previous)
        values[frame] = previous
    result = osim.IKResult(
        times=np.asarray(markers.times, float),
        coordinate_names=list(solver.coordinate_names),
        values=values,
        motion_types=list(solver.motion_types),
        marker_rms=rms,
        marker_max=maximum,
        marker_names=list(markers.marker_names),
    )
    predicted_all = solver.fk.marker_positions_batch(values)
    full_index = {name: index for index, name in enumerate(solver.fk.marker_names)}
    predicted = np.stack([predicted_all[:, full_index[name]] for name in markers.marker_names], axis=1)
    return result, predicted


def _dominant_contact_indices(active: np.ndarray, *, trim_fraction: float = 0.0) -> np.ndarray:
    """Return indices from the longest contact run, optionally trimming its edges."""
    runs = contact_runs(active, min_frames=1)
    if not runs:
        raise ValueError("contact signal has no active run")
    start, stop = max(runs, key=lambda run: run[1] - run[0])
    trim = int(trim_fraction * (stop - start))
    if stop - start - 2 * trim < 3:
        trim = 0
    return np.arange(start + trim, stop - trim)


def _stance_speed_qc(
    times: np.ndarray,
    marker_names: list[str],
    treadmill: np.ndarray,
    overground: np.ndarray,
    contact: np.ndarray,
) -> dict:
    dt = float(np.median(np.diff(times)))
    result = {}
    for side_index, (side, heel) in enumerate((("left", "L.Heel"), ("right", "R.Heel"))):
        marker = marker_names.index(heel)
        raw_velocity = np.gradient(treadmill[:, marker, 0], dt)
        mapped_velocity = np.gradient(overground[:, marker, 0], dt)
        indices = _dominant_contact_indices(contact[:, side_index], trim_fraction=0.15)
        result[side] = {
            "treadmill_mean_mps": float(np.mean(raw_velocity[indices])),
            "overground_mean_mps": float(np.mean(mapped_velocity[indices])),
            "treadmill_rms_mps": float(np.sqrt(np.mean(raw_velocity[indices] ** 2))),
            "overground_rms_mps": float(np.sqrt(np.mean(mapped_velocity[indices] ** 2))),
        }
    return result


def cop_foot_proximity_qc(
    marker_names: list[str],
    markers: np.ndarray,
    cop: np.ndarray,
    grf: np.ndarray,
    contact: np.ndarray,
    *,
    high_load_threshold_n: float = 200.0,
    max_midpoint_distance_m: float = 0.25,
    max_high_load_perpendicular_m: float = 0.05,
) -> dict:
    """Check that each loaded COP remains associated with its assigned foot."""
    markers = np.asarray(markers, float)
    cop = np.asarray(cop, float)
    grf = np.asarray(grf, float)
    contact = np.asarray(contact, bool)
    if markers.ndim != 3 or markers.shape[0] != len(cop) or markers.shape[2] != 3:
        raise ValueError("markers must have shape [frame_count, marker_count, 3]")
    if cop.shape != (len(markers), 2, 3) or grf.shape != cop.shape or contact.shape != cop.shape[:2]:
        raise ValueError("COP, GRF, and contact must contain two feet on the marker frame grid")

    index = {name: marker_names.index(name) for name in ("L.Heel", "L.Toe.Tip", "R.Heel", "R.Toe.Tip")}
    sides = {}
    for side_index, (side, prefix, opposite_prefix) in enumerate((("left", "L", "R"), ("right", "R", "L"))):
        active = contact[:, side_index]
        high_load = active & (grf[:, side_index, 1] >= high_load_threshold_n)
        if not np.any(active) or not np.any(high_load):
            raise ValueError(f"{side} foot has no loaded COP frames for proximity QC")
        heel = markers[:, index[f"{prefix}.Heel"]][:, [0, 2]]
        toe = markers[:, index[f"{prefix}.Toe.Tip"]][:, [0, 2]]
        opposite_heel = markers[:, index[f"{opposite_prefix}.Heel"]][:, [0, 2]]
        opposite_toe = markers[:, index[f"{opposite_prefix}.Toe.Tip"]][:, [0, 2]]
        point = cop[:, side_index][:, [0, 2]]
        if not np.all(np.isfinite(point[active])):
            raise ValueError(f"{side} contact-active COP contains non-finite values")
        foot_axis = toe - heel
        foot_length = np.linalg.norm(foot_axis, axis=1)
        if np.any(foot_length[active] <= 0.0):
            raise ValueError(f"{side} heel-to-toe marker axis is degenerate")
        unit_axis = foot_axis / foot_length[:, None]
        unit_perpendicular = np.column_stack((-unit_axis[:, 1], unit_axis[:, 0]))
        relative = point - heel
        longitudinal = np.sum(relative * unit_axis, axis=1)
        perpendicular = np.sum(relative * unit_perpendicular, axis=1)
        midpoint = 0.5 * (heel + toe)
        opposite_midpoint = 0.5 * (opposite_heel + opposite_toe)
        assigned_distance = np.linalg.norm(point - midpoint, axis=1)
        opposite_distance = np.linalg.norm(point - opposite_midpoint, axis=1)
        ipsilateral_closer_fraction = float(np.mean(assigned_distance[active] < opposite_distance[active]))
        side_metrics = {
            "contact_frames": int(np.count_nonzero(active)),
            "high_load_frames": int(np.count_nonzero(high_load)),
            "ipsilateral_closer_fraction": ipsilateral_closer_fraction,
            "max_contact_midpoint_distance_m": float(np.max(assigned_distance[active])),
            "median_contact_midpoint_distance_m": float(np.median(assigned_distance[active])),
            "max_high_load_perpendicular_m": float(np.max(np.abs(perpendicular[high_load]))),
            "max_high_load_anterior_to_toe_marker_m": float(
                np.max(np.maximum(0.0, longitudinal[high_load] - foot_length[high_load]))
            ),
        }
        side_metrics["passed"] = bool(
            ipsilateral_closer_fraction == 1.0
            and side_metrics["max_contact_midpoint_distance_m"] <= max_midpoint_distance_m
            and side_metrics["max_high_load_perpendicular_m"] <= max_high_load_perpendicular_m
        )
        sides[side] = side_metrics
    return {
        "passed": all(side["passed"] for side in sides.values()),
        "sides": sides,
        "threshold": {
            "ipsilateral_closer_fraction": 1.0,
            "max_contact_midpoint_distance_m": max_midpoint_distance_m,
            "high_load_vertical_force_N": high_load_threshold_n,
            "max_high_load_perpendicular_m": max_high_load_perpendicular_m,
        },
    }


def _force_qc(grf: np.ndarray, contact: np.ndarray) -> tuple[dict, dict]:
    signs = {}
    friction = {}
    for side_index, side in enumerate(("left", "right")):
        indices = _dominant_contact_indices(contact[:, side_index])
        force = grf[indices, side_index]
        count = len(force)
        early = force[int(0.15 * count) : max(int(0.4 * count), 1), 0]
        late = force[int(0.6 * count) : max(int(0.85 * count), 1), 0]
        positive = force[:, 1] > 0.0
        ratio = np.linalg.norm(force[positive][:, [0, 2]], axis=1) / force[positive, 1]
        signs[side] = {"early_braking_Fx_N": float(np.mean(early)), "late_propulsion_Fx_N": float(np.mean(late))}
        friction[side] = {
            "peak_horizontal_over_vertical": float(np.max(ratio)),
            "p95_horizontal_over_vertical": float(np.percentile(ratio, 95)),
            "minimum_vertical_force_N": float(np.min(force[positive, 1])),
        }
    return signs, friction


def run_pipeline(args: argparse.Namespace) -> Path:
    """Run the staged C3D pipeline and return the output directory."""
    incoming = Path(args.input_dir).resolve()
    final_output_dir = Path(args.output_dir).resolve()
    repository_root = Path(__file__).resolve().parents[2]
    if final_output_dir == repository_root or final_output_dir.is_relative_to(repository_root):
        raise ValueError("generated human-data artifacts must stay outside the repository")
    final_output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir = final_output_dir.parent / f".{final_output_dir.name}.staging-{os.getpid()}"
    for stale_stage in final_output_dir.parent.glob(f".{final_output_dir.name}.staging-*"):
        if stale_stage != output_dir:
            shutil.rmtree(stale_stage)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    previous_cache = final_output_dir / "trial_cache.npz"
    cache_path = output_dir / "trial_cache.npz"
    if previous_cache.exists() and not args.rebuild_cache:
        shutil.copy2(previous_cache, cache_path)
    source_paths = {
        name: incoming / filename
        for name, filename in {
            "calibration_c3d": "Cal 101.v3d.c3d",
            "trial_c3d": "Trial 101.v3d.c3d",
            "left_belt": "LeftBelt101.txt",
            "right_belt": "RightBelt101.txt",
            "speed_changes": "Speedchange101.txt",
        }.items()
    }
    for path in source_paths.values():
        if not path.exists():
            raise FileNotFoundError(path)

    cache = _load_cache(incoming, cache_path, args.rebuild_cache)
    times = np.asarray(cache["times"], float)
    marker_names = [str(value) for value in cache["marker_names"]]
    start, stop = select_stride(times, cache["contact"][:, 0], search_time=args.search_time)
    stride_stop = stop
    selection = slice(start, stop)
    segment_times = times[selection]
    displacement_absolute = np.asarray(cache["belt_displacement_absolute"], float)[selection]
    displacement_relative = displacement_absolute - displacement_absolute[0]
    treadmill_markers = np.asarray(cache["markers_treadmill"], float)[selection]
    target_markers = treadmill_to_overground(
        treadmill_markers,
        displacement_absolute,
        reference_index=0,
    )
    grf = np.asarray(cache["grf"], float)[selection]
    treadmill_cop = np.asarray(cache["cop_treadmill"], float)[selection]
    cop = treadmill_to_overground(treadmill_cop, displacement_absolute, reference_index=0)
    torque = np.asarray(cache["free_torque"], float)[selection]
    contact = np.asarray(cache["contact"], bool)[selection]
    platform_moment = np.asarray(cache["filtered_moment_at_surface_origin_Nm"], float)[selection]
    platform_origin = np.asarray(cache["platform_surface_origin_opensim_m"], float)
    point_grid_wrench_identity_errors_nm = []
    for side in range(2):
        reconstructed_moment = np.cross(treadmill_cop[:, side] - platform_origin[side], grf[:, side]) + torque[:, side]
        point_grid_wrench_identity_errors_nm.append(
            float(np.max(np.abs(reconstructed_moment[contact[:, side]] - platform_moment[contact[:, side], side])))
        )
    belt_speed = np.asarray(cache["belt_speed"], float)[selection]

    mass, mass_qc = _extract_subject_mass(source_paths["calibration_c3d"])
    calibration = _mapped_markers(source_paths["calibration_c3d"])
    stable_window = (0.10, 0.60)
    stable_mask = (calibration.times >= stable_window[0]) & (calibration.times <= stable_window[1])
    calibration_index = {name: index for index, name in enumerate(calibration.marker_names)}
    head_height = np.nanmean(calibration.data[stable_mask, calibration_index["Top.Head"], 1])
    heel_height = 0.5 * (
        np.nanmean(calibration.data[stable_mask, calibration_index["L.Heel"], 1])
        + np.nanmean(calibration.data[stable_mask, calibration_index["R.Heel"], 1])
    )
    subject_height_m = float(head_height - heel_height)
    if not 1.0 < subject_height_m < 2.5:
        raise ValueError(f"implausible marker-derived subject height: {subject_height_m:.3f} m")
    seed_model = Path(newton.examples.get_asset("gait2354_subject01.osim"))
    scaled_model_path = output_dir / "S001_scaled.osim"
    scale = osim.ScaleTool(seed_model, osim.gait2354_measurement_set()).run(
        calibration,
        scaled_model_path,
        subject_mass=mass,
        time_range=stable_window,
    )
    static_post = _post_placement_static_error(scaled_model_path, calibration, stable_window)
    if (
        not np.isfinite(static_post["adjacent_rms_m"])
        or static_post["adjacent_rms_m"] > 0.010
        or static_post["adjacent_max_m"] > 0.020
    ):
        raise ValueError(f"adjacent static consistency failed: {static_post}")
    model = osim.parse_osim(scaled_model_path)

    segment_markers = osim.MarkerData(
        times=segment_times,
        marker_names=marker_names,
        data=target_markers,
        rate=100.0,
        units="m",
    )
    ik, predicted = _ik_result(model, segment_markers, args.device)
    ik_path = output_dir / "trial_ik.mot"
    ik.write_mot(ik_path)
    residual = predicted - target_markers
    residual_norm = np.linalg.norm(residual, axis=-1)
    dynamics_padding = int(round(0.25 / float(np.median(np.diff(times)))))
    dynamics_start = max(0, start - dynamics_padding)
    dynamics_stop = min(len(times), stop + dynamics_padding)
    dynamics_slice = slice(dynamics_start, dynamics_stop)
    dynamics_times = times[dynamics_slice]
    dynamics_displacement = np.asarray(cache["belt_displacement_absolute"], float)[dynamics_slice]
    dynamics_treadmill_markers = np.asarray(cache["markers_treadmill"], float)[dynamics_slice]
    dynamics_target_markers = treadmill_to_overground(
        dynamics_treadmill_markers,
        dynamics_displacement,
        reference_index=start - dynamics_start,
    )
    dynamics_markers = osim.MarkerData(
        times=dynamics_times,
        marker_names=marker_names,
        data=dynamics_target_markers,
        rate=100.0,
        units="m",
    )
    dynamics_ik, _dynamics_predicted = _ik_result(model, dynamics_markers, args.device)
    dynamics_ik_path = output_dir / "trial_ik_dynamics_context.mot"
    dynamics_ik.write_mot(dynamics_ik_path)

    # Preserve the existing downstream adapter artifact without coupling the
    # dynamics analysis to that workflow.
    shoe_ik_path = output_dir / "trial_ik_human_shoe_context.mot"
    dynamics_ik.write_mot(shoe_ik_path)

    residual_path = output_dir / "ik_marker_residuals.sto"
    osim.write_storage(
        residual_path,
        segment_times,
        [f"{name}_error" for name in marker_names],
        residual_norm,
        name="S001 IK marker residual norms",
        in_degrees=False,
    )

    frame_dt = float(np.median(np.diff(times)))
    context_padding = int(round(2.0 / frame_dt))
    context_start = max(0, start - context_padding)
    context_stop = min(len(times), stop + context_padding)
    context_slice = slice(context_start, context_stop)
    context_times = times[context_slice]
    context_displacement = np.asarray(cache["belt_displacement_absolute"], float)[context_slice]
    context_reference = start - context_start
    context_grf = np.asarray(cache["grf"], float)[context_slice]
    context_cop = treadmill_to_overground(
        np.asarray(cache["cop_treadmill"], float)[context_slice],
        context_displacement,
        reference_index=context_reference,
    )
    context_torque = np.asarray(cache["free_torque"], float)[context_slice]
    context_contact = np.asarray(cache["contact"], bool)[context_slice]
    grf_context_path, grf_xml_path = _write_external_loads(
        output_dir,
        context_times,
        context_grf,
        context_cop,
        context_torque,
    )
    grf_labels, grf_table = _external_load_table(grf, cop, torque)
    grf_path = output_dir / "trial_grf.mot"
    osim.write_storage(grf_path, segment_times, grf_labels, grf_table, name="S001 stride GRF", in_degrees=False)

    parsed_loads = osim.read_external_loads(grf_xml_path, grf_context_path)
    external_bodies, sampled_wrenches_raw = parsed_loads.sample(segment_times)
    expected_wrenches = np.concatenate([grf, np.nan_to_num(cop, nan=0.0), torque], axis=2)

    def load_difference_by_unit(actual: np.ndarray) -> dict[str, float]:
        difference = np.abs(actual - expected_wrenches)
        return {
            "force_N": float(np.max(difference[:, :, 0:3])),
            "point_m": float(np.max(difference[:, :, 3:6])),
            "torque_Nm": float(np.max(difference[:, :, 6:9])),
        }

    load_sampling_raw_differences = load_difference_by_unit(sampled_wrenches_raw)
    sampled_wrenches = sampled_wrenches_raw.copy()
    negative_vertical_samples = 0
    for side in range(2):
        invalid = (~contact[:, side]) | (sampled_wrenches[:, side, 1] <= 0.0)
        negative_vertical_samples += int(np.count_nonzero(sampled_wrenches[:, side, 1] < 0.0))
        # COP is undefined off-contact and must not be smoothed across touchdown.
        sampled_wrenches[contact[:, side], side, 3:6] = expected_wrenches[contact[:, side], side, 3:6]
        sampled_wrenches[invalid, side] = 0.0
    load_sampling_differences = load_difference_by_unit(sampled_wrenches)
    sampled_labels, sampled_table = _external_load_table(
        sampled_wrenches[:, :, 0:3], sampled_wrenches[:, :, 3:6], sampled_wrenches[:, :, 6:9]
    )
    sampled_path = output_dir / "trial_grf_id_sampled.mot"
    osim.write_storage(
        sampled_path,
        segment_times,
        sampled_labels,
        sampled_table,
        name="S001 exact ID input wrenches",
        in_degrees=False,
    )
    exact_loads = _SampledExternalLoads(segment_times, external_bodies, sampled_wrenches)

    # Differentiate the measured motion with real context on both sides of the
    # selected stride instead of padding a one-stride signal synthetically.
    ik_storage = dynamics_ik.to_storage()
    inverse = osim.InverseDynamics(model, device=args.device)
    id_result = inverse.solve_from_motion(
        ik_storage, external_loads=exact_loads, cutoff=6.0, output_times=segment_times
    )
    if id_result.coordinates is None or id_result.speeds is None or id_result.accelerations is None:
        raise RuntimeError("inverse dynamics did not retain its filtered coordinate state")
    id_path = output_dir / "trial_id.sto"
    id_result.write_sto(id_path)
    forward = osim.ForwardDynamics(model, device=args.device)
    reconstructed_accelerations = forward.accelerations(
        id_result.coordinates,
        id_result.speeds,
        id_result.generalized_forces,
        external_bodies=external_bodies,
        external_wrenches=sampled_wrenches,
    )
    acceleration_closure_error = reconstructed_accelerations - id_result.accelerations
    rotational_coordinates = np.asarray([kind == "rotational" for kind in id_result.motion_types], bool)

    def acceleration_error_stats(mask: np.ndarray) -> dict[str, float]:
        values = acceleration_closure_error[:, mask]
        return {
            "rms": float(np.sqrt(np.mean(values**2))) if values.size else 0.0,
            "max_abs": float(np.max(np.abs(values))) if values.size else 0.0,
        }

    acceleration_closure = {
        "rotational_rad_s2": acceleration_error_stats(rotational_coordinates),
        "translational_m_s2": acceleration_error_stats(~rotational_coordinates),
    }

    treadmill_coordinates = ik.values.copy()
    pelvis_tx_index = ik.coordinate_names.index("pelvis_tx")
    treadmill_coordinates[:, pelvis_tx_index] -= displacement_relative
    treadmill_wrenches = sampled_wrenches.copy()
    treadmill_wrenches[:, :, 3] -= displacement_relative[:, None]
    treadmill_loads = _SampledExternalLoads(segment_times, external_bodies, treadmill_wrenches)
    treadmill_context_coordinates = dynamics_ik.values.copy()
    dynamics_displacement_relative = dynamics_displacement - dynamics_displacement[start - dynamics_start]
    treadmill_context_coordinates[:, pelvis_tx_index] -= dynamics_displacement_relative
    treadmill_context_ik = osim.IKResult(
        times=dynamics_ik.times,
        coordinate_names=dynamics_ik.coordinate_names,
        values=treadmill_context_coordinates,
        motion_types=dynamics_ik.motion_types,
        marker_rms=dynamics_ik.marker_rms,
        marker_max=dynamics_ik.marker_max,
        marker_names=dynamics_ik.marker_names,
    )
    treadmill_id = inverse.solve_from_motion(
        treadmill_context_ik.to_storage(),
        external_loads=treadmill_loads,
        cutoff=6.0,
        output_times=segment_times,
    )
    treadmill_id_path = output_dir / "trial_id_treadmill_frame.sto"
    treadmill_id.write_sto(treadmill_id_path)
    id_frame_difference = np.abs(id_result.generalized_forces - treadmill_id.generalized_forces)
    interior = slice(10, -10) if len(segment_times) > 20 else slice(0, len(segment_times))
    id_frame_equivalence = {
        "max_abs_all_N_or_Nm": float(np.max(id_frame_difference)),
        "max_abs_interior_N_or_Nm": float(np.max(id_frame_difference[interior])),
        "rms_interior_N_or_Nm": float(np.sqrt(np.mean(id_frame_difference[interior] ** 2))),
    }

    activations = np.zeros((len(segment_times), 0), dtype=float)
    muscle_names: list[str] = []
    reserve_summary = {"status": "skipped"}
    if not args.skip_static_optimization:
        if int(args.so_nodes) < 6:
            raise ValueError("--so-nodes must be at least 6 for spline differentiation")
        so_frame_indices = np.linspace(
            0,
            len(segment_times) - 1,
            min(int(args.so_nodes), len(segment_times)),
            dtype=int,
        )
        so_times = segment_times[so_frame_indices]
        static_optimizer = osim.StaticOptimization(model, device=args.device)
        so_result = static_optimizer.solve_from_motion(
            ik_storage,
            external_loads=exact_loads,
            cutoff=6.0,
            output_times=so_times,
        )
        muscle_names = list(so_result.muscle_names)
        activations = np.column_stack(
            [
                np.interp(segment_times, so_result.times, so_result.activations[:, index])
                for index in range(len(muscle_names))
            ]
        )
        so_result.write_sto(output_dir / "trial_static_optimization.sto")
        osim.write_storage(
            output_dir / "trial_static_optimization_resampled.sto",
            segment_times,
            [f"{name}_activation" for name in muscle_names],
            activations,
            name="S001 Static Optimization Activations (resampled)",
            in_degrees=False,
        )
        coordinate_motion = {
            coordinate.name: coordinate.motion_type for joint in model.joints for coordinate in joint.coordinates
        }
        reserve_by_coordinate = {}
        normalized_force = []
        normalized_moment = []
        normalized_non_root_force = []
        normalized_non_root_moment = []
        for index, name in enumerate(so_result.coordinate_names):
            rotational = coordinate_motion[name] == "rotational"
            values = so_result.reserve_forces[:, index]
            normalization_scale = mass * _G * subject_height_m if rotational else mass * _G
            normalized = float(np.max(np.abs(values)) / normalization_scale)
            reserve_by_coordinate[name] = {
                "unit": "N*m" if rotational else "N",
                "max_abs": float(np.max(np.abs(values))),
                "rms": float(np.sqrt(np.mean(values**2))),
                "normalization": "BW*height" if rotational else "BW",
                "max_abs_normalized": normalized,
            }
            (normalized_moment if rotational else normalized_force).append(normalized)
            if not name.startswith("pelvis_"):
                (normalized_non_root_moment if rotational else normalized_non_root_force).append(normalized)
        reserve_summary = {
            "status": "computed",
            "coordinates": reserve_by_coordinate,
            "max_force_fraction_BW": max(normalized_force, default=0.0),
            "max_moment_fraction_BW_height": max(normalized_moment, default=0.0),
            "max_non_root_force_fraction_BW": max(normalized_non_root_force, default=0.0),
            "max_non_root_moment_fraction_BW_height": max(normalized_non_root_moment, default=0.0),
            "max_moment_balance_residual_N_or_Nm": float(np.max(np.abs(so_result.moment_residuals))),
            "sampling": {
                "frame_indices": so_frame_indices,
                "times_s": so_times,
                "external_load_source": sampled_path.name,
                "max_wrench_mismatch": {"force_N": 0.0, "point_m": 0.0, "torque_Nm": 0.0},
            },
            "normalization": {"body_weight_N": mass * _G, "marker_height_m": subject_height_m},
        }

    fk = osim.ForwardKinematics(model, device=args.device)
    com = fk.center_of_mass_batch(ik.values)
    treadmill_predicted_all = fk.marker_positions_batch(treadmill_coordinates)
    fk_marker_index = {name: index for index, name in enumerate(fk.marker_names)}
    treadmill_predicted = np.stack([treadmill_predicted_all[:, fk_marker_index[name]] for name in marker_names], axis=1)
    remapped_treadmill_predicted = treadmill_to_overground(
        treadmill_predicted, displacement_relative, reference_index=0
    )
    fk_frame_equivalence_max_m = float(np.max(np.abs(remapped_treadmill_predicted - predicted)))
    ranges = {coordinate.name: coordinate.range for joint in model.joints for coordinate in joint.coordinates}
    range_violations = {}
    for index, name in enumerate(ik.coordinate_names):
        lo, hi = ranges[name]
        below = float(max(0.0, lo - np.min(ik.values[:, index])))
        above = float(max(0.0, np.max(ik.values[:, index]) - hi))
        if below > 0.0 or above > 0.0:
            range_violations[name] = {"below": below, "above": above}

    context_treadmill = np.asarray(cache["markers_treadmill"], float)[context_slice]
    context_overground = treadmill_to_overground(
        context_treadmill, context_displacement, reference_index=context_reference
    )
    stance_qc = _stance_speed_qc(
        context_times,
        marker_names,
        context_treadmill,
        context_overground,
        context_contact,
    )
    cop_foot_qc = cop_foot_proximity_qc(marker_names, target_markers, cop, grf, contact)
    analog_wrench_identity_errors_nm = np.asarray(cache["wrench_identity_max_abs_Nm"], float)
    point_grid_wrench_identity_errors_nm = np.asarray(point_grid_wrench_identity_errors_nm, float)
    signs, friction = _force_qc(context_grf, context_contact)
    pelvis_force = np.column_stack(
        [
            id_result.generalized_forces[:, id_result.coordinate_names.index(name)]
            for name in ("pelvis_tx", "pelvis_ty", "pelvis_tz")
        ]
    )
    pelvis_moment = np.column_stack(
        [
            id_result.generalized_forces[:, id_result.coordinate_names.index(name)]
            for name in ("pelvis_tilt", "pelvis_list", "pelvis_rotation")
        ]
    )
    pelvis_force_resultant = np.linalg.norm(pelvis_force, axis=1)
    pelvis_moment_resultant = np.linalg.norm(pelvis_moment, axis=1)
    pelvis_residuals = {
        "translation": {
            "rms_N": float(np.sqrt(np.mean(pelvis_force_resultant**2))),
            "peak_N": float(np.max(pelvis_force_resultant)),
            "rms_fraction_body_weight": float(np.sqrt(np.mean(pelvis_force_resultant**2)) / (mass * _G)),
            "peak_fraction_body_weight": float(np.max(pelvis_force_resultant) / (mass * _G)),
        },
        "rotation": {
            "rms_Nm": float(np.sqrt(np.mean(pelvis_moment_resultant**2))),
            "peak_Nm": float(np.max(pelvis_moment_resultant)),
            "rms_fraction_BW_height": float(
                np.sqrt(np.mean(pelvis_moment_resultant**2)) / (mass * _G * subject_height_m)
            ),
            "peak_fraction_BW_height": float(np.max(pelvis_moment_resultant) / (mass * _G * subject_height_m)),
        },
        "normalization": {"body_weight_N": mass * _G, "marker_height_m": subject_height_m},
    }
    belt_acceleration = np.gradient(belt_speed, segment_times)
    constant_speed_stats = {
        "mean_mps": float(np.mean(belt_speed)),
        "min_mps": float(np.min(belt_speed)),
        "max_mps": float(np.max(belt_speed)),
        "std_mps": float(np.std(belt_speed)),
        "range_mps": float(np.ptp(belt_speed)),
        "max_abs_acceleration_mps2": float(np.max(np.abs(belt_acceleration))),
    }
    constant_speed_gate = (
        constant_speed_stats["range_mps"] <= 1.0e-3 and constant_speed_stats["max_abs_acceleration_mps2"] <= 0.05
    )
    marker_gate = float(np.percentile(ik.marker_rms, 95)) <= 0.030 and float(np.max(ik.marker_max)) <= 0.060
    stance_gate = all(
        values["overground_rms_mps"] < 0.25 * values["treadmill_rms_mps"] for values in stance_qc.values()
    )
    force_sign_gate = all(
        values["early_braking_Fx_N"] < 0.0 and values["late_propulsion_Fx_N"] > 0.0 for values in signs.values()
    )
    friction_gate = all(values["peak_horizontal_over_vertical"] < 1.0 for values in friction.values())
    gates = {
        "constant_belt_speed_frame": {
            "passed": constant_speed_gate,
            "value": constant_speed_stats,
            "threshold": {"range_mps": 1.0e-3, "max_abs_acceleration_mps2": 0.05},
        },
        "level_aligned_trial_geometry": {
            "passed": float(cache["platform_level_max_abs_z_mm"]) <= 1.0e-3,
            "value": {
                "platform_level_max_abs_z_mm": float(cache["platform_level_max_abs_z_mm"]),
                "platform_x_spans_mm": np.asarray(cache["platform_x_spans_mm"], float),
                "platform_y_spans_mm": np.asarray(cache["platform_y_spans_mm"], float),
                "verified_heading_lab": "-Y",
            },
            "threshold": {"platform_level_max_abs_z_mm": 1.0e-3, "heading": "lab -Y"},
            "note": "Generic incline reuse is blocked because this adapter does not estimate treadmill orientation R_TR^G.",
        },
        "force_platform_wrench_identity": {
            "passed": bool(
                max(np.max(analog_wrench_identity_errors_nm), np.max(point_grid_wrench_identity_errors_nm)) <= 1.0e-8
            ),
            "value": {
                "analog_max_abs_Nm_left_then_right": analog_wrench_identity_errors_nm,
                "point_grid_max_abs_Nm_left_then_right": point_grid_wrench_identity_errors_nm,
            },
            "threshold": {"max_abs_Nm": 1.0e-8},
            "note": "Filtered force, COP, and free torque must reconstruct the jointly filtered raw platform moment.",
        },
        "cop_foot_proximity": {
            "passed": cop_foot_qc["passed"],
            "value": cop_foot_qc["sides"],
            "threshold": cop_foot_qc["threshold"],
            "note": "This checks side association and gross boundary artifacts, not containment in an anatomical support polygon.",
        },
        "serialized_load_sampling": {
            "passed": (
                load_sampling_differences["force_N"] <= 20.0
                and load_sampling_differences["point_m"] <= 0.02
                and load_sampling_differences["torque_Nm"] <= 20.0
            ),
            "value": {
                "raw_spline_max_abs_difference": load_sampling_raw_differences,
                "id_input_max_abs_difference": load_sampling_differences,
                "negative_vertical_samples_before_sanitizing": negative_vertical_samples,
                "negative_vertical_samples_passed_to_id": 0,
            },
            "threshold": {
                "force_N": 20.0,
                "point_m": 0.02,
                "torque_Nm": 20.0,
                "negative_vertical_samples_passed_to_id": 0,
            },
            "note": "ID consumes the archived context load after one documented sample/sanitize pass.",
        },
        "treadmill_overground_fk_equivalence": {
            "passed": fk_frame_equivalence_max_m <= 1.0e-9,
            "value_max_abs_m": fk_frame_equivalence_max_m,
            "threshold_m": 1.0e-9,
        },
        "treadmill_overground_id_equivalence": {
            "passed": constant_speed_gate and id_frame_equivalence["max_abs_interior_N_or_Nm"] <= 0.05,
            "value": id_frame_equivalence,
            "threshold": {"max_abs_interior_N_or_Nm": 0.05},
        },
        "adjacent_static_consistency": {
            "passed": static_post["adjacent_rms_m"] <= 0.010 and static_post["adjacent_max_m"] <= 0.020,
            "value": {
                "rms_m": static_post["adjacent_rms_m"],
                "max_m": static_post["adjacent_max_m"],
                "time_range_s": static_post["adjacent_time_range_s"],
            },
            "threshold": {"rms_m": 0.010, "max_m": 0.020},
        },
        "dynamic_marker_fit": {
            "passed": marker_gate,
            "value": {"p95_rms_m": float(np.percentile(ik.marker_rms, 95)), "max_m": float(np.max(ik.marker_max))},
            "threshold": {"p95_rms_m": 0.030, "max_m": 0.060},
        },
        "coordinate_ranges": {"passed": not range_violations, "value": len(range_violations), "threshold": 0},
        "virtual_overground_stance": {
            "passed": stance_gate,
            "value": stance_qc,
            "threshold": "overground heel-speed RMS < 25% treadmill RMS",
        },
        "braking_to_propulsion": {"passed": force_sign_gate, "value": signs, "threshold": "early Fx < 0; late Fx > 0"},
        "friction_cone_mu_1": {"passed": friction_gate, "value": friction, "threshold": 1.0},
        "inverse_dynamics_finite": {
            "passed": bool(np.all(np.isfinite(id_result.generalized_forces))),
            "value": bool(np.all(np.isfinite(id_result.generalized_forces))),
            "threshold": True,
        },
        "inverse_forward_acceleration_closure": {
            "passed": (
                acceleration_closure["rotational_rad_s2"]["max_abs"] <= 1.0e-3
                and acceleration_closure["translational_m_s2"]["max_abs"] <= 1.0e-5
            ),
            "value": acceleration_closure,
            "threshold": {"rotational_max_abs_rad_s2": 1.0e-3, "translational_max_abs_m_s2": 1.0e-5},
            "note": "Engineering ID-to-FD consistency gate; it does not validate predictive contact dynamics.",
        },
        "pelvis_residual_translation": {
            "passed": pelvis_residuals["translation"]["rms_fraction_body_weight"] < 0.10,
            "value": pelvis_residuals["translation"],
            "threshold": {"rms_fraction_body_weight": 0.10},
            "note": "Resultant-vector RMS; failure blocks quantitative kinetics and muscle interpretation.",
        },
        "pelvis_residual_rotation": {
            "passed": pelvis_residuals["rotation"]["rms_fraction_BW_height"] < 0.05,
            "value": pelvis_residuals["rotation"],
            "threshold": {"rms_fraction_BW_height": 0.05},
        },
    }
    if reserve_summary["status"] == "computed":
        reserve_pass = (
            reserve_summary["max_force_fraction_BW"] < 0.10 and reserve_summary["max_moment_fraction_BW_height"] < 0.05
        )
        gates["static_optimization_reserves"] = {
            "passed": reserve_pass,
            "severity": "warning",
            "value": {
                "max_force_fraction_BW": reserve_summary["max_force_fraction_BW"],
                "max_moment_fraction_BW_height": reserve_summary["max_moment_fraction_BW_height"],
                "max_non_root_force_fraction_BW": reserve_summary["max_non_root_force_fraction_BW"],
                "max_non_root_moment_fraction_BW_height": reserve_summary["max_non_root_moment_fraction_BW_height"],
            },
            "threshold": {"force_fraction_BW": 0.10, "moment_fraction_BW_height": 0.05},
            "note": "Root reserves duplicate pelvis residuals; non-root reserves are reported separately. Activations remain illustrative when either gate fails.",
        }
    core_pass = all(bool(gate["passed"]) for gate in gates.values() if gate.get("severity", "error") != "warning")
    source_hashes = {name: sha256(path) for name, path in source_paths.items()}
    runtime = _runtime_provenance(repository_root, str(args.device))
    belt_registration = json.loads(str(cache["belt_qc_json"]))
    warnings = [
        "The bundled gait2354 subject01 model is not a provenance-cleared generic S001 model.",
        "Belt speed units and the metric-to-C3D clock are inferred from protocol exports, not hardware timestamps.",
        "This single-stride result is not clinically validated and does not include residual reduction analysis.",
    ]
    reserve_gate = gates.get("static_optimization_reserves")
    if reserve_gate is not None and not reserve_gate["passed"]:
        warnings.append(
            "Static Optimization reserves exceed the quantitative target; activation colors are illustrative."
        )
    if not gates["pelvis_residual_translation"]["passed"]:
        warnings.append(
            "Pelvis translational residual RMS exceeds the 10% body-weight gate; quantitative kinetics are not accepted."
        )
    if not gates["force_platform_wrench_identity"]["passed"]:
        warnings.append("Filtered force-platform channels do not preserve the raw platform moment identity.")
    if not gates["cop_foot_proximity"]["passed"]:
        warnings.append("A contact-active COP is not plausibly associated with its assigned foot.")
    qc = {
        "schema_version": _SCHEMA_VERSION,
        "architecture_role": ARCHITECTURE_ROLE,
        "reference_only": True,
        "production_eligible": False,
        "status": "research_demo_passed_with_provenance_warnings" if core_pass else "research_demo_failed_qc",
        "gates": gates,
        "warnings": warnings,
        "axes": {
            "lab": "X left, Y backward, Z up",
            "opensim": "X forward, Y up, Z right",
            "newton": "X forward, Y left, Z up",
            "conversion": "up_axis=+Z, forward_axis=-Y",
        },
        "source_hashes": source_hashes,
        "runtime": runtime,
        "belt_registration": belt_registration,
        "force_processing": {
            "point_rate_hz": float(cache["point_rate_hz"]),
            "analog_rate_hz": float(cache["analog_rate_hz"]),
            "lowpass_cutoff_hz": float(cache["force_cutoff_hz"]),
            "contact_threshold_N": float(cache["contact_threshold_N"]),
            "wrench_processing": "filter_force_and_moment_then_derive_cop_and_free_torque",
            "wrench_raw_sources": ["platform.force", "platform.moment"],
            "wrench_moment_reference": "mean platform surface corners",
            "wrench_equation": "M=(P-O)xF+T",
            "wrench_filter": {
                "family": "Butterworth",
                "order": 4,
                "representation": "second-order sections",
                "phase": "zero-phase sosfiltfilt",
            },
            "validated_platform_units": {"force": "N", "moment": "Nmm", "position": "mm"},
            "analog_wrench_identity_max_abs_Nm_left_then_right": analog_wrench_identity_errors_nm,
            "point_grid_wrench_identity_max_abs_Nm_left_then_right": point_grid_wrench_identity_errors_nm,
            "platform_surface_origin_lab_mm_left_then_right": np.asarray(
                cache["platform_surface_origin_lab_mm"], float
            ),
            "platform_mean_lab_x_mm_left_then_right": np.asarray(cache["platform_mean_lab_x_mm"], float),
            "platform_assignment": ["calcn_l", "calcn_r"],
            "trial_geometry": "verified level/aligned Trial 101; lab -Y heading",
            "incline_support": "not supported without treadmill-frame orientation mapping",
            "force_units": "N",
            "cop_units": "m",
            "free_torque_units": "N*m",
        },
        "assumptions": {
            "belt_units": "m/s inferred from protocol; not encoded in metric export",
            "belt_clock": "piecewise item-time warp through four ramp onsets plus endpoints",
            "model_seed": "bundled gait2354 subject01; not a provenance-cleared generic S001 model",
            "mass": mass_qc["source"],
            "treadmill_to_overground": "Jung-Lee virtual-origin translation adapted to supplied belt command",
            "analysis_frame": "overground at steady 1.5 m/s on verified level/aligned Trial 101; constant-velocity Galilean translation",
            "incline": "unsupported; requires Jung treadmill-frame orientation R_TR^G",
            "power_frame": "no unlabeled work or power metrics are emitted",
        },
        "subject_mass": {"kg": mass, **mass_qc},
        "scale_factors": scale.scale_factors,
        "static_errors": {
            "pre_marker_relocation_rms_m": scale.static_rms,
            "pre_marker_relocation_max_m": scale.static_max,
            "marker_relocation_evaluation": static_post,
        },
        "ik_errors": {
            "median_rms_m": float(np.median(ik.marker_rms)),
            "p95_rms_m": float(np.percentile(ik.marker_rms, 95)),
            "max_frame_error_m": float(np.max(ik.marker_max)),
            "per_marker_rms_m": {
                name: float(np.sqrt(np.mean(residual_norm[:, index] ** 2))) for index, name in enumerate(marker_names)
            },
        },
        "coordinate_range_violations": range_violations,
        "stance_heel_speeds": stance_qc,
        "braking_propulsion_force_signs": signs,
        "friction_ratios": friction,
        "id_finiteness": bool(np.all(np.isfinite(id_result.generalized_forces))),
        "pelvis_residuals": pelvis_residuals,
        "external_load_sampling": {
            "context_time_range_s": [float(context_times[0]), float(context_times[-1])],
            "raw_spline_max_abs_difference": load_sampling_raw_differences,
            "id_input_max_abs_difference": load_sampling_differences,
            "negative_vertical_samples_before_sanitizing": negative_vertical_samples,
            "negative_vertical_samples_passed_to_id": 0,
            "id_input_artifact": str(sampled_path.name),
        },
        "fk_frame_equivalence_max_m": fk_frame_equivalence_max_m,
        "id_frame_equivalence": id_frame_equivalence,
        "id_generalized_force_stats": {
            name: {
                "unit": "N*m" if id_result.motion_types[index] == "rotational" else "N",
                "min": float(np.min(id_result.generalized_forces[:, index])),
                "max": float(np.max(id_result.generalized_forces[:, index])),
                "rms": float(np.sqrt(np.mean(id_result.generalized_forces[:, index] ** 2))),
            }
            for index, name in enumerate(id_result.coordinate_names)
        },
        "reserves": reserve_summary,
        "stride": {
            "start_frame": start,
            "stop_frame_exclusive": stride_stop,
            "start_time_s": float(segment_times[0]),
            "stop_time_s": float(times[stride_stop - 1]),
            "duration_s": float(times[stride_stop - 1] - segment_times[0]),
            "belt_speed_mean_mps": float(np.mean(belt_speed)),
            "belt_displacement_absolute_at_start_m": float(displacement_absolute[0]),
            "overground_progression_m": float(displacement_relative[-1]),
        },
        "contacts": {
            "left_frames": int(np.count_nonzero(contact[:, 0])),
            "right_frames": int(np.count_nonzero(contact[:, 1])),
            "peak_vertical_force_N": {
                "left": float(np.max(grf[:, 0, 1])),
                "right": float(np.max(grf[:, 1, 1])),
            },
        },
        "frame_labels": {
            "markers": "OpenSim overground X-forward/Y-up/Z-right",
            "forces": "OpenSim ground; subject-directed",
            "cop": "OpenSim overground, translated with virtual origin",
            "coordinates": "OpenSim native radians/metres",
            "id_coordinates_speeds_accelerations": "6 Hz filtered OpenSim native state used by inverse dynamics",
            "id_generalized_forces": "overground constant-velocity frame",
            "id_external_wrenches": "sanitized exact sampled [F P T] used by inverse dynamics",
        },
        "artifacts": {
            "model": scaled_model_path.name,
            "ik": ik_path.name,
            "ik_dynamics_context": dynamics_ik_path.name,
            "ik_human_shoe_context": shoe_ik_path.name,
            "ik_residuals": residual_path.name,
            "grf_stride": grf_path.name,
            "grf_context": grf_context_path.name,
            "grf_xml": grf_xml_path.name,
            "grf_id_sampled": sampled_path.name,
            "id_overground": id_path.name,
            "id_treadmill": treadmill_id_path.name,
        },
    }
    write_json(output_dir / "qc_summary.json", qc)
    write_json(
        output_dir / "manifest.json",
        {
            "schema_version": _SCHEMA_VERSION,
            "architecture_role": ARCHITECTURE_ROLE,
            "reference_only": True,
            "production_eligible": False,
            "runtime": runtime,
            "source_hashes": source_hashes,
            "input_directory": str(incoming),
            "output_directory": str(final_output_dir),
            "parameters": {
                "search_time_s": float(args.search_time),
                "so_nodes": int(args.so_nodes),
                "skip_static_optimization": bool(args.skip_static_optimization),
                "device": str(args.device),
            },
            "stride": qc["stride"],
            "status": qc["status"],
            "warnings": qc["warnings"],
        },
    )

    np.savez_compressed(
        output_dir / "analysis.npz",
        schema_version=np.asarray(_SCHEMA_VERSION, dtype="U"),
        times=segment_times,
        coords=ik.values,
        coordinate_names=np.asarray(ik.coordinate_names, dtype="U"),
        motion_types=np.asarray(ik.motion_types, dtype="U"),
        marker_names=np.asarray(marker_names, dtype="U"),
        target_markers=target_markers,
        predicted_markers=predicted,
        marker_residual=residual,
        marker_residual_norm=residual_norm,
        marker_rms=ik.marker_rms,
        marker_max=ik.marker_max,
        grf=grf,
        cop=cop,
        free_torque=torque,
        contact=contact,
        foot_names=np.asarray(["left", "right"], dtype="U"),
        belt_speed=belt_speed,
        belt_displacement_absolute=displacement_absolute,
        belt_displacement_relative=displacement_relative,
        com=com,
        activations=activations,
        muscle_names=np.asarray(muscle_names, dtype="U"),
        id_coordinates=id_result.coordinates,
        id_speeds=id_result.speeds,
        id_accelerations=id_result.accelerations,
        id_generalized_forces=id_result.generalized_forces,
        id_names=np.asarray(id_result.coordinate_names, dtype="U"),
        id_external_bodies=np.asarray(external_bodies, dtype="U"),
        id_external_wrenches=sampled_wrenches,
    )
    backup_dir = final_output_dir.parent / f".{final_output_dir.name}.previous"
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    if final_output_dir.exists():
        os.replace(final_output_dir, backup_dir)
    try:
        os.replace(output_dir, final_output_dir)
    except Exception:
        if backup_dir.exists() and not final_output_dir.exists():
            os.replace(backup_dir, final_output_dir)
        raise
    if backup_dir.exists():
        shutil.rmtree(backup_dir)

    print(f"[gait_c3d] wrote {final_output_dir}")
    print(
        f"[gait_c3d] stride {segment_times[0]:.2f}-{segment_times[-1]:.2f} s, "
        f"IK median RMS {1000.0 * np.median(ik.marker_rms):.1f} mm, "
        f"overground travel {displacement_relative[-1]:.2f} m"
    )
    return final_output_dir


def create_parser() -> argparse.ArgumentParser:
    """Create the gait-pipeline CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-only",
        action="store_true",
        help="required acknowledgement: scaling/IK/ID/SO use newton.opensim compatibility mechanics",
    )
    parser.add_argument(
        "--input-dir",
        default="/home/jo31399/newton-data/gait/incoming",
        help="Directory containing the five staged source files.",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/jo31399/newton-data/gait/processed/trial_101/latest",
        help="Destination for generated analysis artifacts.",
    )
    parser.add_argument("--search-time", type=float, default=20.0, help="Earliest time [s] for stride selection.")
    parser.add_argument("--so-nodes", type=int, default=12, help="Static-optimization nodes across the stride.")
    parser.add_argument("--device", default="cpu", help="Warp device for OpenSim kernels.")
    parser.add_argument("--rebuild-cache", action="store_true", help="Re-extract the full C3D marker/force cache.")
    parser.add_argument(
        "--skip-static-optimization",
        action="store_true",
        help="Build through inverse dynamics without coarse muscle recruitment.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the command-line pipeline."""
    parser = create_parser()
    args = parser.parse_args(argv)
    if not args.reference_only:
        parser.error("--reference-only is required; the analysis pipeline is not Newton-native mechanics")
    run_pipeline(args)
    return 0


if __name__ == "__main__":
    main()
