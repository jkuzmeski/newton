# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Evaluate project-local predictive foot contact under prescribed gait motion.

This preliminary Stage 2 adapter augments a deep copy of a scaled OpenSim model
from a strict JSON sidecar. It evaluates :class:`newton.opensim.OpenSimContact`
from archived coordinates and speeds only. Measured platform loads are read only
after evaluation, as validation targets. This module does not optimize contact
parameters and does not perform forward dynamics.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

import newton.opensim as osim

_SCHEMA = "gait_c3d_predictive_contact_sidecar_2"
_ARTIFACT_SCHEMA = "gait_c3d_prescribed_contact_2"
_ANALYSIS_SCHEMA = "gait_c3d_analysis_3"
_DEFAULT_DATA = Path("/home/jo31399/newton-data/gait/processed/trial_101/latest")
_FRAME = "opensim_x_forward_y_up_z_right"
_UNITS = {"length": "m", "force": "N", "moment": "N*m", "time": "s"}
_SIDES = ("left", "right")
_ROLES = ("heel", "medial_forefoot", "lateral_forefoot", "toe")
_MARKERS = {
    ("left", "heel"): "L.Heel",
    ("left", "medial_forefoot"): "L.Toe.Med",
    ("left", "lateral_forefoot"): "L.Toe.Lat",
    ("left", "toe"): "L.Toe.Tip",
    ("right", "heel"): "R.Heel",
    ("right", "medial_forefoot"): "R.Toe.Med",
    ("right", "lateral_forefoot"): "R.Toe.Lat",
    ("right", "toe"): "R.Toe.Tip",
}
_BODIES = {"left": "calcn_l", "right": "calcn_r"}
_MATERIAL_BOUNDS = {
    "stiffness": (1.0e5, 5.0e7),
    "dissipation": (0.0, 5.0),
    "static_friction": (0.2, 1.5),
    "dynamic_friction": (0.1, 1.5),
    "viscous_friction": (0.0, 1.0),
    "transition_velocity": (0.01, 0.5),
}
_FROZEN_SMOOTHING = {
    "constant_contact_force": 1.0e-5,
    "hertz_smoothing": 300.0,
    "hunt_crossley_smoothing": 50.0,
}
_SEED_RADIUS_M = 0.03
_VELOCITY_STENCIL_H_S = 1.0e-6


def _sha256(path: Path) -> str:
    """Return a file SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_object(value: Any, context: str, required: set[str]) -> dict[str, Any]:
    """Require an object with exactly the declared fields."""
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be an object")
    missing = required - set(value)
    unknown = set(value) - required
    if missing:
        raise ValueError(f"{context} is missing fields: {', '.join(sorted(missing))}")
    if unknown:
        raise ValueError(f"{context} has unknown fields: {', '.join(sorted(unknown))}")
    return value


def _finite_float(value: Any, context: str) -> float:
    """Convert one finite scalar to float."""
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def _positive_float(value: Any, context: str) -> float:
    """Convert one positive finite scalar to float."""
    result = _finite_float(value, context)
    if result <= 0.0:
        raise ValueError(f"{context} must be positive")
    return result


def _vec3(value: Any, context: str) -> tuple[float, float, float]:
    """Convert one finite length-three value to a tuple."""
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{context} must contain three finite values")
    return tuple(float(item) for item in array)


def _bounds(value: Any, context: str) -> tuple[float, float]:
    """Convert one increasing finite bound pair to a tuple."""
    array = np.asarray(value, dtype=float)
    if array.shape != (2,) or not np.all(np.isfinite(array)) or array[0] > array[1]:
        raise ValueError(f"{context} must be an increasing finite pair")
    return float(array[0]), float(array[1])


@dataclass(frozen=True, slots=True)
class GroundConfig:
    """Stationary OpenSim ground half-space configuration [m]."""

    name: str
    height_m: float
    platform_height_m: float
    height_bounds_m: tuple[float, float]


@dataclass(frozen=True, slots=True)
class MaterialConfig:
    """Shared bilateral smooth sphere-half-space material configuration."""

    law: str
    stiffness: float
    dissipation: float
    static_friction: float
    dynamic_friction: float
    viscous_friction: float
    transition_velocity: float
    constant_contact_force: float
    hertz_smoothing: float
    hunt_crossley_smoothing: float
    bounds: dict[str, tuple[float, float]]

    def parameters(self) -> dict[str, float]:
        """Return parameters accepted by ``SmoothSphereHalfSpaceForce``."""
        return {
            "stiffness": self.stiffness,
            "dissipation": self.dissipation,
            "static_friction": self.static_friction,
            "dynamic_friction": self.dynamic_friction,
            "viscous_friction": self.viscous_friction,
            "transition_velocity": self.transition_velocity,
            "constant_contact_force": self.constant_contact_force,
            "hertz_smoothing": self.hertz_smoothing,
            "hunt_crossley_smoothing": self.hunt_crossley_smoothing,
        }


@dataclass(frozen=True, slots=True)
class SphereConfig:
    """One prescribed-motion-seeded body-local contact sphere [m]."""

    name: str
    force_name: str
    side: str
    role: str
    body: str
    marker: str
    marker_landmark_m: tuple[float, float, float]
    geometry_seed_method: str
    geometry_seed_m: tuple[float, float, float]
    seed_radius_m: float
    phase_frame_indices: tuple[int, ...]
    tangent_residual_rms_m: float
    tangent_residual_max_abs_m: float
    center_m: tuple[float, float, float]
    radius_m: float
    center_displacement_bounds_m: tuple[float, float]
    radius_bounds_m: tuple[float, float]


@dataclass(frozen=True, slots=True)
class CalibrationConfig:
    """Frozen prescribed-motion sample, split, masks, and objective settings."""

    train_side: str
    held_out_side: str
    load_threshold_n: float
    cop_load_threshold_n: float
    prescribed_time_step_s: float
    objective_weights: dict[str, float]


@dataclass(frozen=True, slots=True)
class NormalizationConfig:
    """Subject force and moment normalizers."""

    body_weight_n: float
    body_height_m: float


@dataclass(frozen=True, slots=True)
class PredictiveContactSidecar:
    """Validated project-local contact augmentation and evaluation contract."""

    schema_version: str
    source_model_path: str
    source_model_sha256: str
    source_analysis_path: str
    source_analysis_sha256: str
    frame: str
    units: dict[str, str]
    ground: GroundConfig
    material: MaterialConfig
    spheres: tuple[SphereConfig, ...]
    calibration: CalibrationConfig
    normalization: NormalizationConfig


def _parse_material(data: Any) -> MaterialConfig:
    fields = {
        "law",
        "stiffness",
        "dissipation",
        "static_friction",
        "dynamic_friction",
        "viscous_friction",
        "transition_velocity",
        "constant_contact_force",
        "hertz_smoothing",
        "hunt_crossley_smoothing",
        "bounds",
    }
    data = _strict_object(data, "material", fields)
    bound_names = {
        "stiffness",
        "dissipation",
        "static_friction",
        "dynamic_friction",
        "viscous_friction",
        "transition_velocity",
    }
    bound_data = _strict_object(data["bounds"], "material.bounds", bound_names)
    bounds = {name: _bounds(bound_data[name], f"material.bounds.{name}") for name in sorted(bound_names)}
    if bounds != _MATERIAL_BOUNDS:
        raise ValueError("material bounds must match the frozen Stage 2 roadmap")
    values = {
        "stiffness": _positive_float(data["stiffness"], "material.stiffness"),
        "dissipation": _finite_float(data["dissipation"], "material.dissipation"),
        "static_friction": _finite_float(data["static_friction"], "material.static_friction"),
        "dynamic_friction": _finite_float(data["dynamic_friction"], "material.dynamic_friction"),
        "viscous_friction": _finite_float(data["viscous_friction"], "material.viscous_friction"),
        "transition_velocity": _positive_float(data["transition_velocity"], "material.transition_velocity"),
    }
    if data["law"] != "SmoothSphereHalfSpaceForce":
        raise ValueError("material.law must be SmoothSphereHalfSpaceForce")
    for name, value in values.items():
        low, high = bounds[name]
        if not low <= value <= high:
            raise ValueError(f"material.{name} is outside its frozen bounds")
    if values["dynamic_friction"] > values["static_friction"]:
        raise ValueError("dynamic_friction must not exceed static_friction")
    smoothing = {name: _positive_float(data[name], f"material.{name}") for name in _FROZEN_SMOOTHING}
    if smoothing != _FROZEN_SMOOTHING:
        raise ValueError("material force-law smoothing parameters must match the frozen Stage 2 contract")
    return MaterialConfig(law=data["law"], **values, **smoothing, bounds=bounds)


def _parse_sphere(data: Any, index: int) -> SphereConfig:
    fields = {
        "name",
        "force_name",
        "side",
        "role",
        "body",
        "marker",
        "marker_landmark_m",
        "geometry_seed_method",
        "geometry_seed_m",
        "seed_radius_m",
        "phase_frame_indices",
        "tangent_residual_rms_m",
        "tangent_residual_max_abs_m",
        "center_m",
        "radius_m",
        "center_displacement_bounds_m",
        "radius_bounds_m",
    }
    data = _strict_object(data, f"spheres[{index}]", fields)
    side = str(data["side"])
    role = str(data["role"])
    if side not in _SIDES or role not in _ROLES:
        raise ValueError(f"spheres[{index}] has an unsupported side or role")
    if data["body"] != _BODIES[side] or data["marker"] != _MARKERS[(side, role)]:
        raise ValueError(f"spheres[{index}] does not use the declared anatomical body and marker")
    radius = _positive_float(data["radius_m"], f"spheres[{index}].radius_m")
    radius_bounds = _bounds(data["radius_bounds_m"], f"spheres[{index}].radius_bounds_m")
    displacement_bounds = _bounds(
        data["center_displacement_bounds_m"], f"spheres[{index}].center_displacement_bounds_m"
    )
    if radius_bounds != (0.01, 0.06) or displacement_bounds != (-0.03, 0.03):
        raise ValueError("sphere bounds must match the frozen Stage 2 roadmap")
    if not radius_bounds[0] <= radius <= radius_bounds[1]:
        raise ValueError(f"spheres[{index}].radius_m is outside its bounds")
    landmark = _vec3(data["marker_landmark_m"], f"spheres[{index}].marker_landmark_m")
    geometry_seed = _vec3(data["geometry_seed_m"], f"spheres[{index}].geometry_seed_m")
    seed_radius = _positive_float(data["seed_radius_m"], f"spheres[{index}].seed_radius_m")
    if seed_radius != _SEED_RADIUS_M:
        raise ValueError(f"spheres[{index}].seed_radius_m must preserve the immutable initial seed radius")
    center = _vec3(data["center_m"], f"spheres[{index}].center_m")
    method = str(data["geometry_seed_method"])
    if method != "mean_inverse_vertical_projection_to_ground_tangent":
        raise ValueError(f"spheres[{index}] has an unsupported geometry seed derivation")
    phase_data = data["phase_frame_indices"]
    if not isinstance(phase_data, list) or not phase_data:
        raise ValueError(f"spheres[{index}].phase_frame_indices must be a nonempty array")
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in phase_data):
        raise ValueError(f"spheres[{index}].phase_frame_indices must contain nonnegative integers")
    phase_frames = tuple(phase_data)
    if list(phase_frames) != sorted(set(phase_frames)):
        raise ValueError(f"spheres[{index}].phase_frame_indices must be unique and increasing")
    tangent_rms = _finite_float(data["tangent_residual_rms_m"], f"spheres[{index}].tangent_residual_rms_m")
    tangent_max = _finite_float(data["tangent_residual_max_abs_m"], f"spheres[{index}].tangent_residual_max_abs_m")
    if tangent_rms < 0.0 or tangent_max < 0.0 or tangent_max + 1.0e-15 < tangent_rms:
        raise ValueError(f"spheres[{index}] has invalid tangent residual metrics")
    displacement = np.asarray(center) - np.asarray(geometry_seed)
    if np.any(displacement < displacement_bounds[0]) or np.any(displacement > displacement_bounds[1]):
        raise ValueError(f"spheres[{index}].center_m is outside its prescribed-motion geometry-seed bounds")
    name = str(data["name"])
    force_name = str(data["force_name"])
    if not name or not force_name:
        raise ValueError(f"spheres[{index}] names must not be empty")
    return SphereConfig(
        name=name,
        force_name=force_name,
        side=side,
        role=role,
        body=str(data["body"]),
        marker=str(data["marker"]),
        marker_landmark_m=landmark,
        geometry_seed_method=method,
        geometry_seed_m=geometry_seed,
        seed_radius_m=seed_radius,
        phase_frame_indices=phase_frames,
        tangent_residual_rms_m=tangent_rms,
        tangent_residual_max_abs_m=tangent_max,
        center_m=center,
        radius_m=radius,
        center_displacement_bounds_m=displacement_bounds,
        radius_bounds_m=radius_bounds,
    )


def load_contact_sidecar(path: str | os.PathLike) -> PredictiveContactSidecar:
    """Load and strictly validate a Stage 2 contact sidecar JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    fields = {
        "schema_version",
        "source_model_path",
        "source_model_sha256",
        "source_analysis_path",
        "source_analysis_sha256",
        "frame",
        "units",
        "ground",
        "material",
        "spheres",
        "calibration",
        "normalization",
    }
    data = _strict_object(data, "sidecar", fields)
    if data["schema_version"] != _SCHEMA:
        raise ValueError(f"schema_version must be {_SCHEMA}")
    if data["frame"] != _FRAME:
        raise ValueError(f"frame must be {_FRAME}")
    units = _strict_object(data["units"], "units", set(_UNITS))
    if units != _UNITS:
        raise ValueError("units must use the frozen SI contract")
    source_path = str(data["source_model_path"])
    source_hash = str(data["source_model_sha256"])
    analysis_path = str(data["source_analysis_path"])
    analysis_hash = str(data["source_analysis_sha256"])
    if not source_path or not analysis_path:
        raise ValueError("source model and analysis paths must not be empty")
    for name, digest in (("source_model_sha256", source_hash), ("source_analysis_sha256", analysis_hash)):
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"{name} must be a lowercase SHA-256 digest")

    ground_data = _strict_object(data["ground"], "ground", {"name", "height_m", "platform_height_m", "height_bounds_m"})
    ground = GroundConfig(
        name=str(ground_data["name"]),
        height_m=_finite_float(ground_data["height_m"], "ground.height_m"),
        platform_height_m=_finite_float(ground_data["platform_height_m"], "ground.platform_height_m"),
        height_bounds_m=_bounds(ground_data["height_bounds_m"], "ground.height_bounds_m"),
    )
    if not ground.name:
        raise ValueError("ground.name must not be empty")
    if ground.height_bounds_m != (-0.02, 0.02):
        raise ValueError("ground height bounds must be +/-20 mm")
    offset = ground.height_m - ground.platform_height_m
    if not ground.height_bounds_m[0] <= offset <= ground.height_bounds_m[1]:
        raise ValueError("ground.height_m is outside the platform-relative bounds")

    material = _parse_material(data["material"])
    if not isinstance(data["spheres"], list):
        raise TypeError("spheres must be an array")
    spheres = tuple(_parse_sphere(value, index) for index, value in enumerate(data["spheres"]))
    pairs = [(sphere.side, sphere.role) for sphere in spheres]
    if len(spheres) != 8 or set(pairs) != set(_MARKERS) or len(set(pairs)) != 8:
        raise ValueError("spheres must contain exactly one sphere for every bilateral anatomical role")
    names = [sphere.name for sphere in spheres] + [sphere.force_name for sphere in spheres] + [ground.name]
    if len(names) != len(set(names)):
        raise ValueError("ground, sphere, and force names must be unique")
    for role in _ROLES:
        radii = [sphere.radius_m for sphere in spheres if sphere.role == role]
        if radii[0] != radii[1]:
            raise ValueError(f"initial {role} radius must be shared bilaterally")

    calibration_data = _strict_object(
        data["calibration"],
        "calibration",
        {
            "train_side",
            "held_out_side",
            "load_threshold_n",
            "cop_load_threshold_n",
            "prescribed_time_step_s",
            "objective_weights",
        },
    )
    weights_data = _strict_object(
        calibration_data["objective_weights"],
        "calibration.objective_weights",
        {"vertical_force", "horizontal_force", "impulse", "cop", "free_moment", "regularization", "bilateral"},
    )
    weights = {name: _positive_float(value, f"objective_weights.{name}") for name, value in weights_data.items()}
    train_side = str(calibration_data["train_side"])
    held_out_side = str(calibration_data["held_out_side"])
    if {train_side, held_out_side} != set(_SIDES):
        raise ValueError("train_side and held_out_side must be distinct left/right sides")
    calibration = CalibrationConfig(
        train_side=train_side,
        held_out_side=held_out_side,
        load_threshold_n=_positive_float(calibration_data["load_threshold_n"], "calibration.load_threshold_n"),
        cop_load_threshold_n=_positive_float(
            calibration_data["cop_load_threshold_n"], "calibration.cop_load_threshold_n"
        ),
        prescribed_time_step_s=_positive_float(
            calibration_data["prescribed_time_step_s"], "calibration.prescribed_time_step_s"
        ),
        objective_weights=weights,
    )
    if calibration.load_threshold_n != 50.0 or calibration.cop_load_threshold_n != 200.0:
        raise ValueError("load thresholds must match the frozen pipeline and Stage 2 contract")
    if calibration.prescribed_time_step_s != 0.001:
        raise ValueError("prescribed_time_step_s must match the frozen 1 ms Stage 2 contract")

    normalization_data = _strict_object(data["normalization"], "normalization", {"body_weight_n", "body_height_m"})
    normalization = NormalizationConfig(
        body_weight_n=_positive_float(normalization_data["body_weight_n"], "normalization.body_weight_n"),
        body_height_m=_positive_float(normalization_data["body_height_m"], "normalization.body_height_m"),
    )
    return PredictiveContactSidecar(
        schema_version=_SCHEMA,
        source_model_path=source_path,
        source_model_sha256=source_hash,
        source_analysis_path=analysis_path,
        source_analysis_sha256=analysis_hash,
        frame=_FRAME,
        units=dict(_UNITS),
        ground=ground,
        material=material,
        spheres=spheres,
        calibration=calibration,
        normalization=normalization,
    )


def sidecar_to_dict(sidecar: PredictiveContactSidecar) -> dict[str, Any]:
    """Convert a validated sidecar to its canonical JSON representation."""
    data = asdict(sidecar)
    data["spheres"] = [asdict(sphere) for sphere in sidecar.spheres]
    return data


def _json_value(value: Any) -> Any:
    """Convert NumPy and nonfinite values to strict JSON values."""
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        result = float(value)
        return result if np.isfinite(result) else None
    return value


def _write_json(path: Path, value: Any) -> None:
    """Write stable strict JSON, using null for failed nonfinite metrics."""
    path.write_text(
        json.dumps(_json_value(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _array_sha256(value: np.ndarray) -> str:
    """Return a stable SHA-256 digest of one contiguous array."""
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _git_runtime(repository_root: Path) -> dict[str, Any]:
    """Return the exact Git revision and dirty-worktree identity."""
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=repository_root,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    return {
        "git_commit": commit,
        "git_dirty": bool(status),
        "git_status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
    }


def _runtime_provenance(repository_root: Path, device: str) -> dict[str, Any]:
    """Describe the code, interpreter, packages, device, and Git state."""
    packages = {}
    for label, distribution in (("newton", "newton"), ("numpy", "numpy"), ("warp", "warp-lang")):
        try:
            packages[label] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[label] = None
    return {
        **_git_runtime(repository_root),
        "device": device,
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
        "predictive_contact_sha256": _sha256(Path(__file__).resolve()),
    }


def _load_source_manifest(path: Path) -> dict[str, Any]:
    """Load and validate the Stage 0 manifest used by the comparison."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError("source manifest must be an object")
    if data.get("schema_version") != _ANALYSIS_SCHEMA:
        raise ValueError(f"source manifest schema_version must be {_ANALYSIS_SCHEMA}")
    if "status" not in data or not isinstance(data["status"], str) or not data["status"]:
        raise ValueError("source manifest status must be a nonempty string")
    runtime = data.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError("source manifest runtime must be an object")
    if "git_dirty" in runtime and not isinstance(runtime["git_dirty"], bool):
        raise ValueError("source manifest runtime.git_dirty must be boolean")
    if "frame" in data and data["frame"] != _FRAME:
        raise ValueError(f"source manifest frame must be {_FRAME}")
    if "units" in data and data["units"] != _UNITS:
        raise ValueError("source manifest units do not match the frozen SI contract")
    return data


def _contact_body_indices(body_names: list[str] | tuple[str, ...], context: str) -> list[int]:
    """Return indices that map the evaluator's body order to anatomical left/right."""
    actual = list(body_names)
    expected = [_BODIES[side] for side in _SIDES]
    if len(actual) != len(expected) or len(set(actual)) != len(actual) or set(actual) != set(expected):
        raise ValueError(f"{context} bodies must contain exactly calcn_l and calcn_r")
    return [actual.index(body) for body in expected]


def _uniform_time_grid(times: np.ndarray, time_step_s: float) -> np.ndarray:
    """Construct an endpoint-preserving uniform grid at the requested step."""
    source = np.asarray(times, dtype=float)
    if source.ndim != 1 or len(source) < 2 or not np.all(np.isfinite(source)) or np.any(np.diff(source) <= 0.0):
        raise ValueError("source times must contain at least two finite increasing values")
    time_step_s = _positive_float(time_step_s, "time_step_s")
    interval_count = int(round(float((source[-1] - source[0]) / time_step_s)))
    if interval_count < 1 or not np.isclose(
        source[0] + interval_count * time_step_s,
        source[-1],
        rtol=0.0,
        atol=max(1.0e-12, 1.0e-9 * time_step_s),
    ):
        raise ValueError("source duration must contain an integer number of prescribed time steps")
    grid = source[0] + np.arange(interval_count + 1, dtype=float) * time_step_s
    grid[-1] = source[-1]
    return grid


def _interpolate_numeric(times: np.ndarray, values: np.ndarray, sample_times: np.ndarray) -> np.ndarray:
    """Linearly interpolate a finite numeric array along its first axis."""
    source = np.asarray(values, dtype=float)
    if source.shape[0] != len(times) or not np.all(np.isfinite(source)):
        raise ValueError("interpolated source arrays must be finite and match source times")
    flat = source.reshape(len(times), -1)
    sampled = np.column_stack([np.interp(sample_times, times, flat[:, index]) for index in range(flat.shape[1])])
    return sampled.reshape((len(sample_times), *source.shape[1:]))


def _interpolate_optional_numeric(times: np.ndarray, values: np.ndarray, sample_times: np.ndarray) -> np.ndarray:
    """Interpolate each column only across its finite source support."""
    source = np.asarray(values, dtype=float)
    if source.shape[0] != len(times):
        raise ValueError("interpolated source arrays must match source times")
    flat = source.reshape(len(times), -1)
    sampled = np.full((len(sample_times), flat.shape[1]), np.nan)
    for index in range(flat.shape[1]):
        finite = np.isfinite(flat[:, index])
        if np.count_nonzero(finite) == 1:
            sampled[np.isclose(sample_times, times[finite][0]), index] = flat[finite, index][0]
        elif np.count_nonzero(finite) > 1:
            support_times = times[finite]
            inside = (sample_times >= support_times[0]) & (sample_times <= support_times[-1])
            sampled[inside, index] = np.interp(sample_times[inside], support_times, flat[finite, index])
    return sampled.reshape((len(sample_times), *source.shape[1:]))


def _sample_mask_nearest(times: np.ndarray, mask: np.ndarray, sample_times: np.ndarray) -> np.ndarray:
    """Sample a categorical mask from the nearest archived frame."""
    source = np.asarray(mask)
    if source.dtype != np.bool_ or source.shape[0] != len(times):
        raise ValueError("sampled mask must be boolean and match source times")
    right = np.searchsorted(times, sample_times, side="left")
    right = np.clip(right, 0, len(times) - 1)
    left = np.maximum(right - 1, 0)
    choose_left = np.abs(sample_times - times[left]) <= np.abs(times[right] - sample_times)
    indices = np.where(choose_left, left, right)
    return source[indices]


def _role_phase_frames(contact_frames: np.ndarray, role: str) -> np.ndarray:
    """Freeze an early, middle, or late subset of one measured stance."""
    frames = np.asarray(contact_frames, dtype=int)
    if frames.ndim != 1 or not len(frames) or np.any(np.diff(frames) <= 0):
        raise ValueError("each side must have a nonempty increasing measured stance")
    segments = np.split(frames, np.flatnonzero(np.diff(frames) > 1) + 1)
    frames = max(segments, key=len)
    count = max(1, int(np.ceil(0.2 * len(frames))))
    if role == "heel":
        selected = frames[:count]
    elif role == "toe":
        selected = frames[-count:]
    else:
        start = max(0, (len(frames) - count) // 2)
        selected = frames[start : start + count]
    return selected


def derive_tangent_geometry_seed(
    body_transforms: np.ndarray,
    marker_landmark_m: np.ndarray,
    radius_m: float,
    ground_height_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Derive the landmark-nearest local center tangent over frozen frames.

    For every frozen phase frame, the landmark is transformed to ground. Its
    vertical coordinate is projected to ``ground_height_m + radius_m`` while
    ground-frame AP/ML coordinates remain unchanged. The candidate is transformed
    back to body local coordinates. Their mean is the single frozen sphere center.

    Returns:
        The body-local geometry seed and signed surface-height residual per frame.
    """
    transforms = np.asarray(body_transforms, dtype=float)
    landmark = np.asarray(marker_landmark_m, dtype=float)
    if transforms.ndim != 3 or transforms.shape[1:] != (4, 4) or not len(transforms):
        raise ValueError("body_transforms must have shape [phase_frame, 4, 4]")
    if landmark.shape != (3,) or not np.all(np.isfinite(transforms)) or not np.all(np.isfinite(landmark)):
        raise ValueError("tangent seed inputs must be finite")
    radius_m = _positive_float(radius_m, "radius_m")
    ground_height_m = _finite_float(ground_height_m, "ground_height_m")
    homogeneous_landmark = np.array([*landmark, 1.0])
    candidates = []
    for transform in transforms:
        landmark_ground = transform @ homogeneous_landmark
        target_ground = landmark_ground.copy()
        target_ground[1] = ground_height_m + radius_m
        candidates.append((np.linalg.inv(transform) @ target_ground)[:3])
    seed = np.mean(candidates, axis=0)
    homogeneous_seed = np.array([*seed, 1.0])
    surface_height = (transforms @ homogeneous_seed)[:, 1] - radius_m
    return seed, surface_height - ground_height_m


def write_initial_contact_sidecar(
    model_path: str | os.PathLike,
    analysis_path: str | os.PathLike,
    output_path: str | os.PathLike,
    *,
    platform_height_m: float = 0.0,
    body_height_m: float,
    train_side: str = "left",
    device: str = "cpu",
) -> Path:
    """Write an atomic prescribed-motion-seeded sidecar without changing the model.

    Args:
        model_path: Scaled gait2354 OpenSim model.
        analysis_path: Schema-3 archive that freezes state, load mask, and phase frames.
        output_path: New JSON sidecar path.
        platform_height_m: Verified force-platform surface height [m].
        body_height_m: Subject standing height used for moment normalization [m].
        train_side: Predeclared calibration side; the other side is held out.
        device: Warp device used only for prescribed forward kinematics.

    Returns:
        Path to the completed sidecar.
    """
    model_path = Path(model_path).resolve()
    analysis_path = Path(analysis_path).resolve()
    output_path = Path(output_path).resolve()
    for source in (model_path, analysis_path):
        if not source.is_file():
            raise FileNotFoundError(source)
    if output_path.exists():
        raise FileExistsError(output_path)
    if train_side not in _SIDES:
        raise ValueError("train_side must be left or right")
    platform_height_m = _finite_float(platform_height_m, "platform_height_m")
    body_height_m = _positive_float(body_height_m, "body_height_m")
    model = osim.parse_osim(model_path)
    body_names = {body.name for body in model.bodies}
    marker_by_name = {marker.name: marker for marker in model.markers}
    if not set(_BODIES.values()).issubset(body_names):
        raise ValueError("scaled model is missing the bilateral calcaneus bodies")
    with np.load(analysis_path, allow_pickle=False) as archive:
        if (
            "schema_version" not in archive.files
            or str(np.asarray(archive["schema_version"]).item()) != _ANALYSIS_SCHEMA
        ):
            raise ValueError(f"analysis schema must be {_ANALYSIS_SCHEMA}; rerun the C3D pipeline")
        coordinates = _require_numeric(archive, "id_coordinates", 2)
        coordinate_names = [str(value) for value in np.asarray(archive["id_names"])]
        foot_names = [str(value) for value in np.asarray(archive["foot_names"])]
        if "contact" not in archive.files:
            raise ValueError("analysis archive is missing the exact pipeline contact mask")
        measured_contact = np.asarray(archive["contact"])
    if foot_names != list(_SIDES):
        raise ValueError("analysis foot order must be exactly [left, right]")
    if measured_contact.dtype != np.bool_ or measured_contact.shape != (len(coordinates), 2):
        raise ValueError("analysis contact must be a boolean [time, 2] pipeline mask")
    if coordinates.shape != (len(coordinates), len(coordinate_names)):
        raise ValueError("analysis coordinates do not match the archived coordinate names")
    fk = osim.ForwardKinematics(model, device=device)
    if coordinate_names != list(fk.coordinate_names):
        raise ValueError("analysis coordinate order does not match ForwardKinematics")
    transforms = np.asarray(fk.body_transforms_batch(coordinates), dtype=float)
    if transforms.shape != (len(coordinates), len(fk.body_names), 4, 4):
        raise ValueError("ForwardKinematics returned an unexpected body-transform shape")
    body_index = {name: index for index, name in enumerate(fk.body_names)}

    spheres = []
    for side_index, side in enumerate(_SIDES):
        contact_frames = np.flatnonzero(measured_contact[:, side_index])
        for role in _ROLES:
            marker_name = _MARKERS[(side, role)]
            marker = marker_by_name.get(marker_name)
            if marker is None or marker.body != _BODIES[side]:
                raise ValueError(f"scaled model marker {marker_name!r} is missing or attached to the wrong body")
            if marker.body not in body_index:
                raise ValueError(f"ForwardKinematics is missing body {marker.body!r}")
            phase_frames = _role_phase_frames(contact_frames, role)
            landmark = np.asarray(marker.location, dtype=float)
            radius = 0.03
            seed, tangent_residual = derive_tangent_geometry_seed(
                transforms[phase_frames, body_index[marker.body]],
                landmark,
                radius,
                platform_height_m,
            )
            short_side = side[0]
            spheres.append(
                {
                    "name": f"contact_{short_side}_{role}",
                    "force_name": f"force_{short_side}_{role}",
                    "side": side,
                    "role": role,
                    "body": _BODIES[side],
                    "marker": marker_name,
                    "marker_landmark_m": landmark.tolist(),
                    "geometry_seed_method": "mean_inverse_vertical_projection_to_ground_tangent",
                    "geometry_seed_m": seed.tolist(),
                    "phase_frame_indices": phase_frames.tolist(),
                    "tangent_residual_rms_m": float(np.sqrt(np.mean(tangent_residual**2))),
                    "tangent_residual_max_abs_m": float(np.max(np.abs(tangent_residual))),
                    "center_m": seed.tolist(),
                    "seed_radius_m": radius,
                    "radius_m": radius,
                    "center_displacement_bounds_m": [-0.03, 0.03],
                    "radius_bounds_m": [0.01, 0.06],
                }
            )
    total_mass = sum(float(body.mass) for body in model.bodies)
    if not np.isfinite(total_mass) or total_mass <= 0.0:
        raise ValueError("scaled model must have positive finite total body mass")
    held_out = "right" if train_side == "left" else "left"
    data = {
        "schema_version": _SCHEMA,
        "source_model_path": str(model_path),
        "source_model_sha256": _sha256(model_path),
        "source_analysis_path": str(analysis_path),
        "source_analysis_sha256": _sha256(analysis_path),
        "frame": _FRAME,
        "units": dict(_UNITS),
        "ground": {
            "name": "trial101_ground",
            "height_m": platform_height_m,
            "platform_height_m": platform_height_m,
            "height_bounds_m": [-0.02, 0.02],
        },
        "material": {
            "law": "SmoothSphereHalfSpaceForce",
            "stiffness": 1.0e6,
            "dissipation": 2.0,
            "static_friction": 0.9,
            "dynamic_friction": 0.8,
            "viscous_friction": 0.5,
            "transition_velocity": 0.1,
            "constant_contact_force": 1.0e-5,
            "hertz_smoothing": 300.0,
            "hunt_crossley_smoothing": 50.0,
            "bounds": {
                "stiffness": [1.0e5, 5.0e7],
                "dissipation": [0.0, 5.0],
                "static_friction": [0.2, 1.5],
                "dynamic_friction": [0.1, 1.5],
                "viscous_friction": [0.0, 1.0],
                "transition_velocity": [0.01, 0.5],
            },
        },
        "spheres": spheres,
        "calibration": {
            "train_side": train_side,
            "held_out_side": held_out,
            "load_threshold_n": 50.0,
            "cop_load_threshold_n": 200.0,
            "prescribed_time_step_s": 0.001,
            "objective_weights": {
                "vertical_force": 1.0,
                "horizontal_force": 1.0,
                "impulse": 1.0,
                "cop": 1.0,
                "free_moment": 1.0,
                "regularization": 1.0,
                "bilateral": 1.0,
            },
        },
        "normalization": {"body_weight_n": total_mass * 9.80665, "body_height_m": body_height_m},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{output_path.name}.", dir=output_path.parent)
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        _write_json(temporary_path, data)
        load_contact_sidecar(temporary_path)
        os.rename(temporary_path, output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return output_path


def augment_contact_model(model: osim.OsimModel, sidecar: PredictiveContactSidecar) -> osim.OsimModel:
    """Return a deep-copied model augmented with the frozen contact sidecar."""
    augmented = copy.deepcopy(model)
    body_names = {body.name for body in augmented.bodies}
    marker_by_name = {marker.name: marker for marker in augmented.markers}
    existing = {geometry.name for geometry in augmented.contact_geometry} | {
        force.name for force in augmented.contact_forces
    }
    requested = (
        {sidecar.ground.name}
        | {sphere.name for sphere in sidecar.spheres}
        | {sphere.force_name for sphere in sidecar.spheres}
    )
    collisions = existing & requested
    if collisions:
        raise ValueError(f"contact sidecar names already exist: {', '.join(sorted(collisions))}")
    for sphere in sidecar.spheres:
        if sphere.body not in body_names:
            raise KeyError(f"body {sphere.body!r} not found")
        marker = marker_by_name.get(sphere.marker)
        if marker is None or marker.body != sphere.body:
            raise ValueError(f"marker {sphere.marker!r} is missing or attached to the wrong body")
        if not np.allclose(marker.location, sphere.marker_landmark_m, rtol=0.0, atol=1.0e-10):
            raise ValueError(f"marker {sphere.marker!r} no longer matches the frozen sidecar landmark")
    augmented.contact_geometry.append(
        osim.OsimContactGeometry(
            name=sidecar.ground.name,
            type="ContactHalfSpace",
            body="ground",
            location=(0.0, sidecar.ground.height_m, 0.0),
            orientation=(0.0, 0.0, -0.5 * np.pi),
        )
    )
    for sphere in sidecar.spheres:
        augmented.contact_geometry.append(
            osim.OsimContactGeometry(
                name=sphere.name,
                type="ContactSphere",
                body=sphere.body,
                location=sphere.center_m,
                radius=sphere.radius_m,
            )
        )
        augmented.contact_forces.append(
            osim.OsimContactForce(
                name=sphere.force_name,
                type=sidecar.material.law,
                sphere=sphere.name,
                half_space=sidecar.ground.name,
                params=sidecar.material.parameters(),
            )
        )
    return augmented


def ground_wrench_to_cop_free_moment(
    wrenches: np.ndarray,
    ground_height_m: float,
    *,
    load_threshold_n: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert OpenSim ``[F, P, T]`` ground wrenches to COP and free moment.

    The input torque is the resultant couple at ``P``. COP and vertical free
    moment are defined only when the upward force exceeds ``load_threshold_n``.

    Args:
        wrenches: Wrenches with shape ``[..., 9]`` in the OpenSim ground frame.
        ground_height_m: Stationary plane height [m].
        load_threshold_n: Minimum upward force used to define the result [N].

    Returns:
        A pair ``(cop, free_moment_y)`` with shapes ``[..., 3]`` and ``[...]``.
    """
    values = np.asarray(wrenches, dtype=float)
    if values.ndim < 1 or values.shape[-1] != 9:
        raise ValueError("wrenches must have trailing shape 9")
    ground_height_m = _finite_float(ground_height_m, "ground_height_m")
    load_threshold_n = _positive_float(load_threshold_n, "load_threshold_n")
    force = values[..., :3]
    point = values[..., 3:6]
    couple = values[..., 6:9]
    moment_origin = np.cross(point, force) + couple
    normal = np.array([0.0, 1.0, 0.0])
    vertical = force[..., 1]
    loaded = (vertical >= load_threshold_n) & np.all(np.isfinite(values), axis=-1)
    cop = np.full(force.shape, np.nan)
    numerator = np.cross(normal, moment_origin) + ground_height_m * force
    cop[loaded] = numerator[loaded] / vertical[loaded, None]
    free_moment = np.full(vertical.shape, np.nan)
    free_moment[loaded] = (moment_origin[loaded] - np.cross(cop[loaded], force[loaded])) @ normal
    return cop, free_moment


def sphere_penetrations(
    body_transforms: np.ndarray,
    body_names: list[str] | tuple[str, ...],
    sidecar: PredictiveContactSidecar,
) -> np.ndarray:
    """Compute nonnegative sphere penetration from prescribed body transforms [m]."""
    transforms = np.asarray(body_transforms, dtype=float)
    if transforms.ndim != 4 or transforms.shape[2:] != (4, 4) or transforms.shape[1] != len(body_names):
        raise ValueError("body_transforms must have shape [time, body, 4, 4]")
    if not np.all(np.isfinite(transforms)):
        raise ValueError("body_transforms must contain finite values")
    indices = {name: index for index, name in enumerate(body_names)}
    penetration = np.empty((len(transforms), len(sidecar.spheres)))
    ground_origin = np.array([0.0, sidecar.ground.height_m, 0.0])
    normal_into = np.array([0.0, -1.0, 0.0])
    for index, sphere in enumerate(sidecar.spheres):
        if sphere.body not in indices:
            raise KeyError(f"body {sphere.body!r} not found in prescribed transforms")
        homogeneous = np.array([*sphere.center_m, 1.0])
        centers = (transforms[:, indices[sphere.body]] @ homogeneous)[:, :3]
        indentation = sphere.radius_m + (centers - ground_origin) @ normal_into
        penetration[:, index] = np.maximum(indentation, 0.0)
    return penetration


def _rms(values: np.ndarray) -> float:
    """Return root-mean-square, or infinity for an empty/nonfinite set."""
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.all(np.isfinite(values)):
        return float("inf")
    return float(np.sqrt(np.mean(values**2)))


def _event_indices(mask: np.ndarray, onset: bool) -> np.ndarray:
    """Return periodic contact onset or last-loaded release sample indices."""
    mask = np.asarray(mask, dtype=bool)
    neighbor = np.roll(mask, 1 if onset else -1)
    return np.flatnonzero(mask & ~neighbor)


def _matched_event(
    times: np.ndarray,
    target_mask: np.ndarray,
    predicted_mask: np.ndarray,
    *,
    onset: bool,
) -> tuple[float | None, float | None, float]:
    """Match the nearest predicted event to a target on the periodic sample grid."""
    target_indices = _event_indices(target_mask, onset)
    predicted_indices = _event_indices(predicted_mask, onset)
    if not len(target_indices) or not len(predicted_indices):
        return None, None, float("inf")
    sample_dt = float(np.median(np.diff(times)))
    period = float(times[-1] - times[0] + sample_dt)
    best: tuple[float, int, int] | None = None
    for target_index in target_indices:
        for predicted_index in predicted_indices:
            raw = abs(float(times[predicted_index] - times[target_index]))
            error = min(raw, period - raw)
            candidate = (error, int(target_index), int(predicted_index))
            if best is None or candidate < best:
                best = candidate
    assert best is not None
    return float(times[best[1]]), float(times[best[2]]), best[0]


def compute_contact_qc(
    times: np.ndarray,
    measured_grf: np.ndarray,
    measured_cop: np.ndarray,
    measured_free_torque: np.ndarray,
    measured_contact: np.ndarray,
    predicted_grf: np.ndarray,
    predicted_cop: np.ndarray,
    predicted_free_moment_y: np.ndarray,
    penetrations: np.ndarray,
    sidecar: PredictiveContactSidecar,
    *,
    nominal_finite: bool = True,
    smaller_step_finite: bool = True,
    body_order_valid: bool = True,
) -> dict[str, Any]:
    """Compute all preliminary Stage 2 prescribed-contact metrics and gates."""
    times = np.asarray(times, dtype=float)
    measured_grf = np.asarray(measured_grf, dtype=float)
    measured_cop = np.asarray(measured_cop, dtype=float)
    measured_free_torque = np.asarray(measured_free_torque, dtype=float)
    measured_contact = np.asarray(measured_contact, dtype=bool)
    predicted_grf = np.asarray(predicted_grf, dtype=float)
    predicted_cop = np.asarray(predicted_cop, dtype=float)
    predicted_free_moment_y = np.asarray(predicted_free_moment_y, dtype=float)
    penetrations = np.asarray(penetrations, dtype=float)
    ntime = len(times)
    if times.ndim != 1 or ntime < 2 or np.any(np.diff(times) <= 0.0):
        raise ValueError("times must contain at least two strictly increasing values")
    if measured_grf.shape != (ntime, 2, 3) or predicted_grf.shape != (ntime, 2, 3):
        raise ValueError("GRF arrays must have shape [time, 2, 3]")
    if measured_cop.shape != (ntime, 2, 3) or predicted_cop.shape != (ntime, 2, 3):
        raise ValueError("COP arrays must have shape [time, 2, 3]")
    if measured_free_torque.shape != (ntime, 2, 3):
        raise ValueError("measured_free_torque must have shape [time, 2, 3]")
    if measured_contact.shape != (ntime, 2) or predicted_free_moment_y.shape != (ntime, 2):
        raise ValueError("contact/free-moment arrays must have shape [time, 2]")
    if penetrations.ndim != 2 or penetrations.shape[0] != ntime:
        raise ValueError("penetrations must have shape [time, sphere]")
    body_weight = sidecar.normalization.body_weight_n
    body_weight_height = body_weight * sidecar.normalization.body_height_m
    load_threshold = sidecar.calibration.load_threshold_n
    cop_threshold = sidecar.calibration.cop_load_threshold_n
    maximum_penetration = float(np.max(penetrations)) if penetrations.size else float("inf")
    all_finite = bool(
        np.all(np.isfinite(measured_grf))
        and np.all(np.isfinite(measured_free_torque))
        and np.all(np.isfinite(predicted_grf))
        and np.all(np.isfinite(penetrations))
    )
    side_metrics: dict[str, Any] = {}
    all_side_gates: list[bool] = []
    for side_index, side in enumerate(_SIDES):
        target_force = measured_grf[:, side_index]
        predicted_force = predicted_grf[:, side_index]
        target_mask = measured_contact[:, side_index]
        predicted_mask = predicted_force[:, 1] >= load_threshold
        peak_target = float(np.max(target_force[:, 1]))
        peak_predicted = float(np.max(predicted_force[:, 1]))
        peak_relative_error = abs(peak_predicted - peak_target) / peak_target if peak_target > 0.0 else float("inf")
        impulse_target = float(np.trapezoid(target_force[:, 1], times))
        impulse_predicted = float(np.trapezoid(predicted_force[:, 1], times))
        impulse_relative_error = (
            abs(impulse_predicted - impulse_target) / abs(impulse_target) if impulse_target != 0.0 else float("inf")
        )
        target_onset, predicted_onset, onset_error = _matched_event(times, target_mask, predicted_mask, onset=True)
        target_release, predicted_release, release_error = _matched_event(
            times, target_mask, predicted_mask, onset=False
        )
        cop_mask = (
            (target_force[:, 1] >= cop_threshold)
            & (predicted_force[:, 1] >= cop_threshold)
            & np.all(np.isfinite(measured_cop[:, side_index]), axis=1)
            & np.all(np.isfinite(predicted_cop[:, side_index]), axis=1)
            & np.isfinite(predicted_free_moment_y[:, side_index])
        )
        cop_error = predicted_cop[cop_mask, side_index][:, (0, 2)] - measured_cop[cop_mask, side_index][:, (0, 2)]
        cop_rms = _rms(np.linalg.norm(cop_error, axis=1))
        free_error = predicted_free_moment_y[cop_mask, side_index] - measured_free_torque[cop_mask, side_index, 1]
        free_rms = _rms(free_error)
        ap_rms = _rms(predicted_force[:, 0] - target_force[:, 0])
        ml_rms = _rms(predicted_force[:, 2] - target_force[:, 2])
        vertical_rms = _rms(predicted_force[:, 1] - target_force[:, 1])
        minimum_normal_force = float(np.min(predicted_force[:, 1]))
        upward_force_direction = bool(np.isfinite(minimum_normal_force) and minimum_normal_force >= -1.0e-9)
        predicted_loaded_force = predicted_force[predicted_mask]
        friction_excess = (
            np.hypot(predicted_loaded_force[:, 0], predicted_loaded_force[:, 2])
            - sidecar.material.static_friction * predicted_loaded_force[:, 1]
        )
        friction_max_excess = float(np.max(friction_excess)) if friction_excess.size else float("inf")
        gates = {
            "vertical_peak_relative_error_below_0_10": peak_relative_error < 0.10,
            "vertical_impulse_relative_error_below_0_05": impulse_relative_error < 0.05,
            "onset_error_below_0_020_s": onset_error < 0.020,
            "release_error_below_0_020_s": release_error < 0.020,
            "cop_rms_below_0_030_m": cop_rms < 0.030,
            "ap_force_rms_below_0_10_bw": ap_rms / body_weight < 0.10,
            "ml_force_rms_below_0_10_bw": ml_rms / body_weight < 0.10,
            "free_moment_rms_below_0_02_bwh": free_rms / body_weight_height < 0.02,
            "friction_cone": friction_max_excess <= 1.0e-9,
            "opensim_y_normal_force_is_upward": upward_force_direction,
        }
        all_side_gates.extend(gates.values())
        side_metrics[side] = {
            "split": "fit" if side == sidecar.calibration.train_side else "held_out",
            "vertical_force": {
                "rms_N": vertical_rms,
                "rms_body_weight": vertical_rms / body_weight,
                "peak_measured_N": peak_target,
                "peak_predicted_N": peak_predicted,
                "peak_relative_error": peak_relative_error,
                "impulse_measured_Ns": impulse_target,
                "impulse_predicted_Ns": impulse_predicted,
                "impulse_relative_error": impulse_relative_error,
            },
            "horizontal_force": {
                "ap_rms_N": ap_rms,
                "ml_rms_N": ml_rms,
                "braking_impulse_measured_Ns": float(np.trapezoid(np.minimum(target_force[:, 0], 0.0), times)),
                "braking_impulse_predicted_Ns": float(np.trapezoid(np.minimum(predicted_force[:, 0], 0.0), times)),
                "propulsion_impulse_measured_Ns": float(np.trapezoid(np.maximum(target_force[:, 0], 0.0), times)),
                "propulsion_impulse_predicted_Ns": float(np.trapezoid(np.maximum(predicted_force[:, 0], 0.0), times)),
            },
            "timing": {
                "measured_onset_s": target_onset,
                "predicted_onset_s": predicted_onset,
                "onset_error_s": onset_error,
                "measured_release_s": target_release,
                "predicted_release_s": predicted_release,
                "release_error_s": release_error,
            },
            "cop": {"loaded_frame_count": int(np.count_nonzero(cop_mask)), "rms_m": cop_rms},
            "vertical_free_moment": {"rms_Nm": free_rms, "rms_body_weight_height": free_rms / body_weight_height},
            "force_direction": {
                "up_axis": "+Y",
                "minimum_normal_force_N": minimum_normal_force,
            },
            "friction": {"coefficient": sidecar.material.static_friction, "maximum_excess_N": friction_max_excess},
            "objective_terms": {
                "vertical_force": vertical_rms / body_weight,
                "horizontal_force": np.hypot(ap_rms, ml_rms) / body_weight,
                "impulse": impulse_relative_error,
                "cop": cop_rms / 0.03,
                "free_moment": free_rms / body_weight_height,
                "regularization": 0.0,
                "bilateral": 0.0,
            },
            "gates": gates,
            "passed": bool(all(gates.values())),
        }
    global_gates = {
        "finite_nominal_step": bool(nominal_finite),
        "finite_smaller_step": bool(smaller_step_finite),
        "all_required_arrays_finite": all_finite,
        "maximum_penetration_below_0_020_m": maximum_penetration < 0.020,
        "contact_body_order_left_then_right": bool(body_order_valid),
        "no_measured_load_passed_to_contact_evaluator": True,
    }
    return {
        "schema_version": _ARTIFACT_SCHEMA,
        "status": "preliminary_prescribed_contact_passed"
        if all(all_side_gates) and all(global_gates.values())
        else "preliminary_prescribed_contact_failed_qc",
        "scope": "preliminary Stage 2 prescribed-motion contact; no optimization and no forward dynamics",
        "load_masks": {"pipeline_contact_threshold_N": load_threshold, "cop_threshold_N": cop_threshold},
        "sides": side_metrics,
        "maximum_sphere_penetration_m": maximum_penetration,
        "global_gates": global_gates,
        "passed": bool(all(all_side_gates) and all(global_gates.values())),
    }


def _require_numeric(archive: np.lib.npyio.NpzFile, name: str, ndim: int, *, finite: bool = True) -> np.ndarray:
    """Read one strict numeric analysis array."""
    if name not in archive.files:
        raise ValueError(f"analysis archive is missing {name!r}; rerun the C3D pipeline")
    value = np.asarray(archive[name])
    if value.ndim != ndim or not np.issubdtype(value.dtype, np.number):
        raise ValueError(f"analysis field {name!r} must be a {ndim}-D numeric array")
    if finite and not np.all(np.isfinite(value)):
        raise ValueError(f"analysis field {name!r} must contain finite values")
    return value.astype(float, copy=False)


def run_prescribed_contact(
    data_dir: str | os.PathLike = _DEFAULT_DATA,
    sidecar_path: str | os.PathLike | None = None,
    output_dir: str | os.PathLike | None = None,
    *,
    device: str = "cpu",
) -> Path:
    """Evaluate frozen contact under prescribed motion and publish an atomic artifact.

    Args:
        data_dir: Completed schema-3 C3D pipeline artifact.
        sidecar_path: Frozen Stage 2 contact sidecar JSON.
        output_dir: New non-overlapping artifact directory outside the repository.
        device: Warp device used by OpenSim contact and kinematics.

    Returns:
        Path to the completed preliminary Stage 2 artifact.
    """
    data_dir = Path(data_dir).resolve()
    if sidecar_path is None:
        raise ValueError("sidecar_path is required so contact configuration is frozen before evaluation")
    sidecar_path = Path(sidecar_path).resolve()
    output_dir = (
        Path(output_dir).resolve()
        if output_dir is not None
        else data_dir.parent / f"{data_dir.name}_prescribed_contact"
    )
    repository_root = Path(__file__).resolve().parents[2]
    if output_dir == repository_root or output_dir.is_relative_to(repository_root):
        raise ValueError("generated contact artifacts must stay outside the repository")
    if output_dir == data_dir or output_dir.is_relative_to(data_dir) or data_dir.is_relative_to(output_dir):
        raise ValueError("contact and source artifact directories must not overlap")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    model_path = data_dir / "S001_scaled.osim"
    analysis_path = data_dir / "analysis.npz"
    manifest_path = data_dir / "manifest.json"
    for path in (model_path, analysis_path, manifest_path, sidecar_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    source_manifest = _load_source_manifest(manifest_path)
    sidecar = load_contact_sidecar(sidecar_path)
    source_model = Path(sidecar.source_model_path)
    if not source_model.is_absolute():
        source_model = (sidecar_path.parent / source_model).resolve()
    source_analysis = Path(sidecar.source_analysis_path)
    if not source_analysis.is_absolute():
        source_analysis = (sidecar_path.parent / source_analysis).resolve()
    if source_model != model_path or _sha256(model_path) != sidecar.source_model_sha256:
        raise ValueError("sidecar source model path or SHA-256 does not match the Stage 0 artifact")
    if source_analysis != analysis_path or _sha256(analysis_path) != sidecar.source_analysis_sha256:
        raise ValueError("sidecar source analysis path or SHA-256 does not match the frozen sample grid")

    with np.load(analysis_path, allow_pickle=False) as archive:
        if "schema_version" not in archive.files or np.asarray(archive["schema_version"]).shape != ():
            raise ValueError("analysis schema_version must be a scalar")
        if str(np.asarray(archive["schema_version"]).item()) != _ANALYSIS_SCHEMA:
            raise ValueError(f"analysis schema must be {_ANALYSIS_SCHEMA}; rerun the C3D pipeline")
        if "frame" in archive.files and str(np.asarray(archive["frame"]).item()) != _FRAME:
            raise ValueError(f"analysis frame must be {_FRAME}")
        times = _require_numeric(archive, "times", 1)
        coordinates = _require_numeric(archive, "id_coordinates", 2)
        speeds = _require_numeric(archive, "id_speeds", 2)
        measured_grf_source = _require_numeric(archive, "grf", 3)
        measured_cop_source = _require_numeric(archive, "cop", 3, finite=False)
        measured_free_torque_source = _require_numeric(archive, "free_torque", 3)
        if "contact" not in archive.files:
            raise ValueError("analysis archive is missing the exact pipeline contact mask")
        measured_contact_source = np.asarray(archive["contact"])
        if "id_names" not in archive.files or np.asarray(archive["id_names"]).ndim != 1:
            raise ValueError("analysis id_names must be a one-dimensional array")
        if "foot_names" not in archive.files or np.asarray(archive["foot_names"]).ndim != 1:
            raise ValueError("analysis foot_names must be a one-dimensional array")
        coordinate_names = [str(value) for value in np.asarray(archive["id_names"])]
        foot_names = [str(value) for value in np.asarray(archive["foot_names"])]
    if foot_names != list(_SIDES):
        raise ValueError("analysis foot order must be exactly [left, right]")
    if coordinates.shape != speeds.shape or coordinates.shape != (len(times), len(coordinate_names)):
        raise ValueError("prescribed coordinates and speeds do not share the archived state shape")
    if len(times) < 2 or np.any(np.diff(times) <= 0.0):
        raise ValueError("analysis times must contain at least two strictly increasing samples")
    if measured_contact_source.dtype != np.bool_ or measured_contact_source.shape != (len(times), 2):
        raise ValueError("analysis contact must be a boolean [time, 2] pipeline mask")
    for name, value in (
        ("grf", measured_grf_source),
        ("cop", measured_cop_source),
        ("free_torque", measured_free_torque_source),
    ):
        if value.shape != (len(times), 2, 3):
            raise ValueError(f"analysis {name} must have shape [time, 2, 3]")
    if not np.all(np.isfinite(measured_cop_source[measured_contact_source])):
        raise ValueError("measured COP must be finite on the exact pipeline load mask")

    nominal_dt = sidecar.calibration.prescribed_time_step_s
    smaller_dt = 0.5 * nominal_dt
    nominal_times = _uniform_time_grid(times, nominal_dt)
    smaller_times = _uniform_time_grid(times, smaller_dt)
    if len(smaller_times) != 2 * len(nominal_times) - 1 or not np.array_equal(smaller_times[::2], nominal_times):
        raise ValueError("half-step prescribed grid must contain every nominal grid sample")
    nominal_coordinates = _interpolate_numeric(times, coordinates, nominal_times)
    nominal_speeds = _interpolate_numeric(times, speeds, nominal_times)
    smaller_coordinates = _interpolate_numeric(times, coordinates, smaller_times)
    smaller_speeds = _interpolate_numeric(times, speeds, smaller_times)
    measured_grf = _interpolate_numeric(times, measured_grf_source, nominal_times)
    measured_cop = _interpolate_optional_numeric(times, measured_cop_source, nominal_times)
    measured_free_torque = _interpolate_numeric(times, measured_free_torque_source, nominal_times)
    measured_contact = _sample_mask_nearest(times, measured_contact_source, nominal_times)

    model = osim.parse_osim(model_path)
    augmented = augment_contact_model(model, sidecar)
    contact = osim.OpenSimContact(augmented, device=device)
    if coordinate_names != list(contact.coordinate_names):
        raise ValueError("analysis coordinate order does not match OpenSimContact")
    body_names, nominal_wrenches = contact.body_wrenches(
        nominal_coordinates,
        nominal_speeds,
        h=_VELOCITY_STENCIL_H_S,
        frame="opensim",
    )
    smaller_body_names, smaller_wrenches = contact.body_wrenches(
        smaller_coordinates,
        smaller_speeds,
        h=_VELOCITY_STENCIL_H_S,
        frame="opensim",
    )
    body_indices = _contact_body_indices(body_names, "OpenSimContact")
    smaller_body_indices = _contact_body_indices(smaller_body_names, "OpenSimContact half-step")
    expected_body_names = [_BODIES[side] for side in _SIDES]
    predicted_wrenches = np.asarray(nominal_wrenches, dtype=float)[:, body_indices]
    smaller_predicted_wrenches = np.asarray(smaller_wrenches, dtype=float)[:, smaller_body_indices]
    if predicted_wrenches.shape != (len(nominal_times), 2, 9):
        raise ValueError("OpenSimContact returned an unexpected nominal body-wrench shape")
    if smaller_predicted_wrenches.shape != (len(smaller_times), 2, 9):
        raise ValueError("OpenSimContact returned an unexpected half-step body-wrench shape")
    nominal_finite = bool(np.all(np.isfinite(predicted_wrenches)))
    smaller_finite = bool(np.all(np.isfinite(smaller_predicted_wrenches)))
    predicted_grf = predicted_wrenches[..., :3]
    predicted_cop, predicted_free_moment_y = ground_wrench_to_cop_free_moment(
        predicted_wrenches,
        sidecar.ground.height_m,
        load_threshold_n=sidecar.calibration.load_threshold_n,
    )

    fk = osim.ForwardKinematics(augmented, device=device)
    if coordinate_names != list(fk.coordinate_names):
        raise ValueError("analysis coordinate order does not match ForwardKinematics")
    source_transforms = np.asarray(fk.body_transforms_batch(coordinates), dtype=float)
    nominal_transforms = np.asarray(fk.body_transforms_batch(nominal_coordinates), dtype=float)
    if source_transforms.shape != (len(times), len(fk.body_names), 4, 4):
        raise ValueError("ForwardKinematics returned an unexpected source transform shape")
    if nominal_transforms.shape != (len(nominal_times), len(fk.body_names), 4, 4):
        raise ValueError("ForwardKinematics returned an unexpected nominal transform shape")
    fk_body_index = {name: index for index, name in enumerate(fk.body_names)}
    for sphere in sidecar.spheres:
        expected_frames = _role_phase_frames(
            np.flatnonzero(measured_contact_source[:, _SIDES.index(sphere.side)]), sphere.role
        )
        if tuple(expected_frames) != sphere.phase_frame_indices:
            raise ValueError(f"sphere {sphere.name!r} phase frames do not match the frozen pipeline mask")
        if sphere.body not in fk_body_index or max(sphere.phase_frame_indices) >= len(times):
            raise ValueError(f"sphere {sphere.name!r} has invalid frozen phase frames")
        seed, residual = derive_tangent_geometry_seed(
            source_transforms[np.asarray(sphere.phase_frame_indices), fk_body_index[sphere.body]],
            np.asarray(sphere.marker_landmark_m),
            sphere.seed_radius_m,
            sidecar.ground.height_m,
        )
        if not np.allclose(seed, sphere.geometry_seed_m, rtol=0.0, atol=1.0e-10):
            raise ValueError(f"sphere {sphere.name!r} geometry seed does not reproduce from frozen motion")
        residual_rms = float(np.sqrt(np.mean(residual**2)))
        residual_max = float(np.max(np.abs(residual)))
        if not np.isclose(residual_rms, sphere.tangent_residual_rms_m, rtol=0.0, atol=1.0e-12) or not np.isclose(
            residual_max, sphere.tangent_residual_max_abs_m, rtol=0.0, atol=1.0e-12
        ):
            raise ValueError(f"sphere {sphere.name!r} tangent residuals do not reproduce from frozen motion")
    penetrations = sphere_penetrations(nominal_transforms, list(fk.body_names), sidecar)
    qc = compute_contact_qc(
        nominal_times,
        measured_grf,
        measured_cop,
        measured_free_torque,
        measured_contact,
        predicted_grf,
        predicted_cop,
        predicted_free_moment_y,
        penetrations,
        sidecar,
        nominal_finite=nominal_finite,
        smaller_step_finite=smaller_finite,
        body_order_valid=True,
    )
    smaller_on_nominal = smaller_predicted_wrenches[::2]
    force_delta = smaller_on_nominal[..., :3] - predicted_wrenches[..., :3]
    qc["nonfinite_counts"] = {
        "nominal_body_wrenches": int(np.count_nonzero(~np.isfinite(predicted_wrenches))),
        "smaller_step_body_wrenches": int(np.count_nonzero(~np.isfinite(smaller_predicted_wrenches))),
        "sphere_penetration": int(np.count_nonzero(~np.isfinite(penetrations))),
    }
    qc["prescribed_step_sensitivity"] = {
        "nominal_time_step_s": nominal_dt,
        "smaller_time_step_s": smaller_dt,
        "velocity_stencil_h_s": _VELOCITY_STENCIL_H_S,
        "nominal_frame_count": len(nominal_times),
        "smaller_frame_count": len(smaller_times),
        "comparison": "half-step samples at nominal timestamps (half_step[::2])",
        "force_rms_difference_N": _rms(force_delta),
        "force_max_abs_difference_N": float(np.max(np.abs(force_delta)))
        if np.all(np.isfinite(force_delta))
        else float("inf"),
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        _write_json(temporary / "contact_sidecar.json", sidecar_to_dict(sidecar))
        _write_json(temporary / "qc_summary.json", qc)
        np.savez_compressed(
            temporary / "contact_analysis.npz",
            schema_version=np.asarray(_ARTIFACT_SCHEMA),
            frame=np.asarray(_FRAME),
            times=nominal_times,
            smaller_step_times=smaller_times,
            foot_names=np.asarray(_SIDES),
            contact_body_names=np.asarray(expected_body_names),
            coordinate_names=np.asarray(coordinate_names),
            predicted_grf=predicted_grf,
            predicted_cop=predicted_cop,
            predicted_free_moment_y=predicted_free_moment_y,
            predicted_body_wrenches=predicted_wrenches,
            smaller_step_body_wrenches=smaller_predicted_wrenches,
            sphere_names=np.asarray([sphere.name for sphere in sidecar.spheres]),
            sphere_penetration=penetrations,
            measured_contact_mask=measured_contact,
        )
        artifact_files = {
            "contact_sidecar": "contact_sidecar.json",
            "analysis": "contact_analysis.npz",
            "qc": "qc_summary.json",
        }
        artifact_manifest = {
            "schema_version": _ARTIFACT_SCHEMA,
            "status": qc["status"],
            "scope": qc["scope"],
            "runtime": _runtime_provenance(repository_root, device),
            "source": {
                "data_dir": str(data_dir),
                "schema_version": source_manifest["schema_version"],
                "frame": _FRAME,
                "frame_validation": "gait_c3d_analysis_3 schema contract and optional archive/manifest field",
                "status": source_manifest["status"],
                "runtime": source_manifest["runtime"],
                "manifest_sha256": _sha256(manifest_path),
                "analysis_sha256": _sha256(analysis_path),
                "model_sha256": _sha256(model_path),
                "input_sidecar_sha256": _sha256(sidecar_path),
                "source_times_sha256": _array_sha256(times),
            },
            "information_set": {
                "contact_evaluator_inputs": ["interpolated_id_coordinates", "interpolated_id_speeds"],
                "measured_load_input": False,
                "measured_targets_used_only_for_post_evaluation_metrics": ["grf", "cop", "free_torque", "contact"],
            },
            "comparison_provenance": {
                "state_resampling": "independent linear interpolation of archived coordinates and speeds",
                "target_resampling": "linear numeric interpolation; nearest-frame boolean contact mask",
                "nominal_grid": {
                    "time_step_s": nominal_dt,
                    "frame_count": len(nominal_times),
                    "times_sha256": _array_sha256(nominal_times),
                },
                "half_step_grid": {
                    "time_step_s": smaller_dt,
                    "frame_count": len(smaller_times),
                    "times_sha256": _array_sha256(smaller_times),
                },
                "force_comparison": "half-step body wrenches at half_step[::2] minus nominal body wrenches",
                "velocity_stencil_h_s": _VELOCITY_STENCIL_H_S,
            },
            "settings": {
                "frame": _FRAME,
                "prescribed_time_step_s": nominal_dt,
                "smaller_time_step_s": smaller_dt,
                "velocity_stencil_h_s": _VELOCITY_STENCIL_H_S,
                "load_threshold_n": sidecar.calibration.load_threshold_n,
                "cop_load_threshold_n": sidecar.calibration.cop_load_threshold_n,
                "train_side": sidecar.calibration.train_side,
                "held_out_side": sidecar.calibration.held_out_side,
                "contact_body_order": expected_body_names,
                "upward_axis": "+Y",
                "optimization_performed": False,
            },
            "artifacts": {
                name: {"path": relative_path, "sha256": _sha256(temporary / relative_path)}
                for name, relative_path in artifact_files.items()
            },
        }
        _write_json(temporary / "manifest.json", artifact_manifest)
        os.rename(temporary, output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output_dir


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the preliminary Stage 2 command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    initialize = subparsers.add_parser("init", help="write a strict marker-seeded contact sidecar")
    initialize.add_argument("--model", required=True, help="Scaled S001_scaled.osim path")
    initialize.add_argument("--analysis", required=True, help="Schema-3 analysis.npz path")
    initialize.add_argument("--output", required=True, help="New sidecar JSON path")
    initialize.add_argument("--platform-height", type=float, default=0.0, help="Verified platform height [m]")
    initialize.add_argument("--body-height", type=float, required=True, help="Subject height [m]")
    initialize.add_argument("--train-side", choices=_SIDES, default="left")
    initialize.add_argument("--device", default="cpu")
    evaluate = subparsers.add_parser("evaluate", help="write an atomic prescribed-contact artifact")
    evaluate.add_argument("--data-dir", default=str(_DEFAULT_DATA))
    evaluate.add_argument("--sidecar", required=True)
    evaluate.add_argument("--output-dir")
    evaluate.add_argument("--device", default="cpu")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run sidecar initialization or prescribed-contact evaluation."""
    args = _build_arg_parser().parse_args(argv)
    if args.command == "init":
        output = write_initial_contact_sidecar(
            args.model,
            args.analysis,
            args.output,
            platform_height_m=args.platform_height,
            body_height_m=args.body_height,
            train_side=args.train_side,
            device=args.device,
        )
    else:
        output = run_prescribed_contact(args.data_dir, args.sidecar, args.output_dir, device=args.device)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
