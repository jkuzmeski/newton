# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run a preliminary Stage 4 timing and model-inertia sensitivity audit.

This module intentionally does not select or accept a timing correction. It sweeps
only the predeclared wrench lag grid and archives every inverse-dynamics result.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import newton.opensim as osim

_ANALYSIS_SCHEMA = "gait_c3d_analysis_3"
_SCHEMA = "gait_c3d_residual_sensitivity_1"
_LAG_MS = np.arange(-20, 21, dtype=np.int64)
_REQUIRED_ARRAYS = (
    "times",
    "id_coordinates",
    "id_speeds",
    "id_accelerations",
    "id_names",
    "id_external_bodies",
    "id_external_wrenches",
)


@dataclass(frozen=True)
class AnalysisInputs:
    """Validated schema-3 inputs needed by the timing sweep."""

    data_dir: Path
    analysis_path: Path
    qc_path: Path
    manifest_path: Path
    model_path: Path
    times: np.ndarray
    coordinates: np.ndarray
    speeds: np.ndarray
    accelerations: np.ndarray
    coordinate_names: tuple[str, ...]
    external_bodies: tuple[str, ...]
    external_wrenches: np.ndarray
    body_weight_N: float
    subject_height_m: float
    measured_mass_kg: float


def _as_string_tuple(values: np.ndarray, field: str) -> tuple[str, ...]:
    """Validate and convert a one-dimensional string array."""
    values = np.asarray(values)
    if values.ndim != 1 or values.dtype.kind not in "US":
        raise ValueError(f"{field} must be a one-dimensional string array")
    result = tuple(str(value) for value in values.tolist())
    if not result or any(not value for value in result) or len(set(result)) != len(result):
        raise ValueError(f"{field} must contain unique nonempty names")
    return result


def _finite_array(values: np.ndarray, field: str, ndim: int) -> np.ndarray:
    """Return a finite float array with the required rank."""
    result = np.asarray(values, dtype=float)
    if result.ndim != ndim or not np.all(np.isfinite(result)):
        raise ValueError(f"{field} must be a finite rank-{ndim} array")
    return result


def _strict_json_object(path: Path, expected_schema: str) -> dict[str, Any]:
    """Load a JSON object and require the exact analysis schema."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exception:
        raise ValueError(f"cannot load strict JSON from {path}") from exception
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    if value.get("schema_version") != expected_schema:
        raise ValueError(f"{path} schema_version must be {expected_schema}")
    return value


def load_schema3(data_dir: str | os.PathLike) -> AnalysisInputs:
    """Load the exact schema-3 ID state without legacy or inferred fallbacks."""
    data_dir = Path(data_dir).resolve()
    analysis_path = data_dir / "analysis.npz"
    qc_path = data_dir / "qc_summary.json"
    manifest_path = data_dir / "manifest.json"
    qc = _strict_json_object(qc_path, _ANALYSIS_SCHEMA)
    _strict_json_object(manifest_path, _ANALYSIS_SCHEMA)

    try:
        with np.load(analysis_path, allow_pickle=False) as archive:
            if "schema_version" not in archive.files:
                raise ValueError("analysis.npz has no schema_version")
            schema = np.asarray(archive["schema_version"])
            if schema.ndim != 0 or schema.dtype.kind not in "US" or str(schema.item()) != _ANALYSIS_SCHEMA:
                raise ValueError(f"analysis.npz schema_version must be {_ANALYSIS_SCHEMA}")
            missing = sorted(set(_REQUIRED_ARRAYS) - set(archive.files))
            if missing:
                raise ValueError(f"analysis.npz is missing required arrays: {missing}")
            times = _finite_array(archive["times"], "times", 1)
            coordinates = _finite_array(archive["id_coordinates"], "id_coordinates", 2)
            speeds = _finite_array(archive["id_speeds"], "id_speeds", 2)
            accelerations = _finite_array(archive["id_accelerations"], "id_accelerations", 2)
            coordinate_names = _as_string_tuple(archive["id_names"], "id_names")
            external_bodies = _as_string_tuple(archive["id_external_bodies"], "id_external_bodies")
            external_wrenches = _finite_array(archive["id_external_wrenches"], "id_external_wrenches", 3)
    except OSError as exception:
        raise ValueError(f"cannot load {analysis_path}") from exception

    if len(times) < 2 or not np.all(np.diff(times) > 0.0):
        raise ValueError("times must contain at least two strictly increasing samples")
    expected_state_shape = (len(times), len(coordinate_names))
    if (
        coordinates.shape != expected_state_shape
        or speeds.shape != expected_state_shape
        or accelerations.shape != expected_state_shape
    ):
        raise ValueError(f"ID state arrays must all have shape {expected_state_shape}")
    expected_wrench_shape = (len(times), len(external_bodies), 9)
    if external_wrenches.shape != expected_wrench_shape:
        raise ValueError(f"id_external_wrenches must have shape {expected_wrench_shape}")

    try:
        normalization = qc["pelvis_residuals"]["normalization"]
        body_weight_N = float(normalization["body_weight_N"])
        subject_height_m = float(normalization["marker_height_m"])
        measured_mass_kg = float(qc["subject_mass"]["kg"])
        model_name = qc["artifacts"]["model"]
    except (KeyError, TypeError, ValueError) as exception:
        raise ValueError("qc_summary.json lacks strict Stage 4 normalization/model fields") from exception
    if not isinstance(model_name, str) or not model_name or Path(model_name).name != model_name:
        raise ValueError("qc_summary.json artifacts.model must be a file name within data_dir")
    scalars = np.asarray([body_weight_N, subject_height_m, measured_mass_kg])
    if not np.all(np.isfinite(scalars)) or np.any(scalars <= 0.0):
        raise ValueError("body weight, subject height, and measured mass must be positive and finite")
    model_path = data_dir / model_name
    if not model_path.is_file():
        raise ValueError(f"scaled model does not exist: {model_path}")

    return AnalysisInputs(
        data_dir=data_dir,
        analysis_path=analysis_path,
        qc_path=qc_path,
        manifest_path=manifest_path,
        model_path=model_path,
        times=times,
        coordinates=coordinates,
        speeds=speeds,
        accelerations=accelerations,
        coordinate_names=coordinate_names,
        external_bodies=external_bodies,
        external_wrenches=external_wrenches,
        body_weight_N=body_weight_N,
        subject_height_m=subject_height_m,
        measured_mass_kg=measured_mass_kg,
    )


def common_non_extrapolated_indices(times: np.ndarray, lag_ms: np.ndarray = _LAG_MS) -> np.ndarray:
    """Return source indices valid for every lag without extrapolation."""
    times = _finite_array(times, "times", 1)
    lag_s = np.asarray(lag_ms, dtype=float) * 1.0e-3
    if lag_s.ndim != 1 or lag_s.size == 0 or not np.all(np.isfinite(lag_s)):
        raise ValueError("lag_ms must be a nonempty finite vector")
    # A positive measurement delay is corrected with measured(t + lag).
    query_min = times[:, None] + lag_s[None, :]
    tolerance = 32.0 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(times))))
    valid = np.all((query_min >= times[0] - tolerance) & (query_min <= times[-1] + tolerance), axis=1)
    indices = np.flatnonzero(valid)
    if indices.size < 2:
        raise ValueError("the common non-extrapolated lag interior has fewer than two samples")
    return indices


def interpolate_wrench_lags(
    times: np.ndarray,
    wrenches: np.ndarray,
    lag_ms: np.ndarray = _LAG_MS,
    sample_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate all wrench lags on one common, non-extrapolated time grid.

    A positive lag means that the measured wrench timestamps lag the motion. The
    aligned wrench is therefore ``measured_wrench(t + lag)``. No endpoint is extended.
    """
    times = _finite_array(times, "times", 1)
    wrenches = _finite_array(wrenches, "wrenches", 3)
    if wrenches.shape[0] != len(times) or wrenches.shape[-1] != 9:
        raise ValueError("wrenches must have shape [len(times), num_external_bodies, 9]")
    lag_ms = np.asarray(lag_ms)
    if lag_ms.ndim != 1 or lag_ms.size == 0 or not np.all(np.isfinite(lag_ms)):
        raise ValueError("lag_ms must be a nonempty finite vector")
    if sample_indices is None:
        sample_indices = common_non_extrapolated_indices(times, lag_ms)
    sample_indices = np.asarray(sample_indices, dtype=np.int64)
    if (
        sample_indices.ndim != 1
        or sample_indices.size < 2
        or np.any(sample_indices < 0)
        or np.any(sample_indices >= len(times))
    ):
        raise ValueError("sample_indices must select at least two source samples")
    target_times = times[sample_indices]
    lag_s = lag_ms.astype(float) * 1.0e-3
    queries = target_times[None, :] + lag_s[:, None]
    tolerance = 32.0 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(times))))
    if np.min(queries) < times[0] - tolerance or np.max(queries) > times[-1] + tolerance:
        raise ValueError("wrench interpolation would extrapolate")
    queries = np.clip(queries, times[0], times[-1])
    flat = wrenches.reshape(len(times), -1)
    shifted = np.empty((len(lag_ms), len(target_times), flat.shape[1]), dtype=float)
    for column in range(flat.shape[1]):
        shifted[:, :, column] = np.interp(queries, times, flat[:, column])
    return target_times, shifted.reshape(len(lag_ms), len(target_times), *wrenches.shape[1:])


def structural_root_groups(model: osim.OsimModel, coordinate_names: tuple[str, ...] | list[str]) -> dict[str, Any]:
    """Group root coordinates from joints attached to ground, not name patterns."""
    coordinate_names = tuple(coordinate_names)
    name_to_index = {name: index for index, name in enumerate(coordinate_names)}
    if len(name_to_index) != len(coordinate_names):
        raise ValueError("coordinate_names must be unique")
    root_joints = [joint for joint in model.joints if joint.parent_body == "ground"]
    if not root_joints:
        raise ValueError("model has no joint structurally attached to ground")
    translation: list[str] = []
    rotation: list[str] = []
    for joint in root_joints:
        for coordinate in joint.coordinates:
            if coordinate.name not in name_to_index:
                raise ValueError(f"root coordinate {coordinate.name!r} is absent from inverse dynamics")
            if coordinate.motion_type == "translational":
                translation.append(coordinate.name)
            elif coordinate.motion_type == "rotational":
                rotation.append(coordinate.name)
            else:
                raise ValueError(
                    f"root coordinate {coordinate.name!r} has unsupported motion type {coordinate.motion_type!r}"
                )
    translation.sort(key=name_to_index.__getitem__)
    rotation.sort(key=name_to_index.__getitem__)
    if not translation or not rotation:
        raise ValueError("structural root must contain translational and rotational coordinates")
    return {
        "joint_names": [joint.name for joint in root_joints],
        "translation_names": translation,
        "translation_indices": [name_to_index[name] for name in translation],
        "rotation_names": rotation,
        "rotation_indices": [name_to_index[name] for name in rotation],
    }


def resultant_metrics(
    generalized_forces: np.ndarray,
    translation_indices: list[int] | np.ndarray,
    rotation_indices: list[int] | np.ndarray,
    body_weight_N: float,
    subject_height_m: float,
) -> dict[str, np.ndarray]:
    """Compute per-lag vector-resultant RMS and peak residual metrics."""
    values = _finite_array(generalized_forces, "generalized_forces", 3)
    translation = np.take(values, np.asarray(translation_indices, dtype=np.int64), axis=-1)
    rotation = np.take(values, np.asarray(rotation_indices, dtype=np.int64), axis=-1)
    if translation.shape[-1] == 0 or rotation.shape[-1] == 0:
        raise ValueError("translation and rotation groups must be nonempty")
    normalization = np.asarray([body_weight_N, subject_height_m], dtype=float)
    if not np.all(np.isfinite(normalization)) or np.any(normalization <= 0.0):
        raise ValueError("normalization values must be positive and finite")
    # Reduce components last, then time. np.take(axis=-1) avoids NumPy advanced-
    # indexing moving the selected coordinate axis ahead of the lag/time axes.
    force_resultant = np.linalg.norm(translation, axis=-1)
    moment_resultant = np.linalg.norm(rotation, axis=-1)
    force_rms = np.sqrt(np.mean(force_resultant * force_resultant, axis=1))
    moment_rms = np.sqrt(np.mean(moment_resultant * moment_resultant, axis=1))
    denominator_moment = body_weight_N * subject_height_m
    return {
        "translation_rms_N": force_rms,
        "translation_peak_N": np.max(force_resultant, axis=1),
        "translation_rms_fraction_BW": force_rms / body_weight_N,
        "translation_peak_fraction_BW": np.max(force_resultant, axis=1) / body_weight_N,
        "rotation_rms_Nm": moment_rms,
        "rotation_peak_Nm": np.max(moment_resultant, axis=1),
        "rotation_rms_fraction_BW_height": moment_rms / denominator_moment,
        "rotation_peak_fraction_BW_height": np.max(moment_resultant, axis=1) / denominator_moment,
    }


def audit_model_inertia(model: osim.OsimModel, measured_mass_kg: float) -> dict[str, Any]:
    """Archive every segment mass, COM, and full inertia tensor without adjustment."""
    segments = []
    total_mass = float(sum(float(body.mass) for body in model.bodies))
    for body in model.bodies:
        packed = np.asarray(body.inertia, dtype=float)
        com = np.asarray(body.mass_center, dtype=float)
        if packed.shape != (6,) or com.shape != (3,):
            raise ValueError(f"body {body.name!r} has malformed COM or inertia")
        tensor = np.asarray(
            [[packed[0], packed[3], packed[4]], [packed[3], packed[1], packed[5]], [packed[4], packed[5], packed[2]]],
            dtype=float,
        )
        finite = bool(np.isfinite(float(body.mass)) and np.all(np.isfinite(com)) and np.all(np.isfinite(tensor)))
        principal = np.linalg.eigvalsh(tensor) if finite else np.full(3, np.nan)
        segments.append(
            {
                "name": body.name,
                "mass_kg": float(body.mass),
                "mass_fraction": float(body.mass / total_mass) if total_mass > 0.0 else None,
                "com_body_m": com.tolist(),
                "inertia_about_com_kg_m2": tensor.tolist(),
                "principal_inertia_kg_m2": principal.tolist(),
                "finite": finite,
                "nonnegative_mass": bool(float(body.mass) >= 0.0),
                "positive_semidefinite_inertia": bool(finite and np.min(principal) >= -1.0e-12),
            }
        )
    measured_mass_kg = float(measured_mass_kg)
    if not np.isfinite(measured_mass_kg) or measured_mass_kg <= 0.0:
        raise ValueError("measured_mass_kg must be positive and finite")
    finite_audit = all(segment["finite"] for segment in segments) and np.isfinite(total_mass)
    return {
        "model_total_mass_kg": total_mass,
        "measured_subject_mass_kg": measured_mass_kg,
        "mass_difference_kg": total_mass - measured_mass_kg,
        "mass_difference_fraction_measured": (total_mass - measured_mass_kg) / measured_mass_kg,
        "within_one_percent_measured_mass": bool(abs(total_mass - measured_mass_kg) / measured_mass_kg <= 0.01),
        "all_properties_finite": bool(finite_audit),
        "all_masses_nonnegative": all(segment["nonnegative_mass"] for segment in segments),
        "all_inertias_positive_semidefinite": all(segment["positive_semidefinite_inertia"] for segment in segments),
        "segments": segments,
        "adjustments_applied": False,
    }


def solve_lag_surface(
    inputs: AnalysisInputs,
    model: osim.OsimModel,
    device: str | None = None,
    solver: Any | None = None,
) -> dict[str, Any]:
    """Evaluate all 41 lag cases in exactly one batched inverse-dynamics solve."""
    indices = common_non_extrapolated_indices(inputs.times, _LAG_MS)
    common_times, lagged_wrenches = interpolate_wrench_lags(inputs.times, inputs.external_wrenches, _LAG_MS, indices)
    roots = structural_root_groups(model, inputs.coordinate_names)
    if solver is None:
        solver = osim.InverseDynamics(model, device=device)
    solver_names = tuple(solver.coordinate_names)
    if solver_names != inputs.coordinate_names:
        raise ValueError("inverse-dynamics coordinate order does not exactly match analysis.npz id_names")
    count_lag = len(_LAG_MS)
    count_time = len(indices)
    q = np.ascontiguousarray(
        np.broadcast_to(inputs.coordinates[indices], (count_lag, count_time, inputs.coordinates.shape[1])).reshape(
            -1, inputs.coordinates.shape[1]
        )
    )
    qd = np.ascontiguousarray(
        np.broadcast_to(inputs.speeds[indices], (count_lag, count_time, inputs.speeds.shape[1])).reshape(
            -1, inputs.speeds.shape[1]
        )
    )
    qdd = np.ascontiguousarray(
        np.broadcast_to(inputs.accelerations[indices], (count_lag, count_time, inputs.accelerations.shape[1])).reshape(
            -1, inputs.accelerations.shape[1]
        )
    )
    tau_flat = solver.solve(
        q,
        qd,
        qdd,
        external_bodies=list(inputs.external_bodies),
        external_wrenches=np.ascontiguousarray(lagged_wrenches.reshape(-1, len(inputs.external_bodies), 9)),
    )
    tau = _finite_array(tau_flat, "inverse-dynamics result", 2)
    if tau.shape != q.shape:
        raise ValueError(f"inverse-dynamics result must have shape {q.shape}")
    tau = tau.reshape(count_lag, count_time, -1)
    metrics = resultant_metrics(
        tau,
        roots["translation_indices"],
        roots["rotation_indices"],
        inputs.body_weight_N,
        inputs.subject_height_m,
    )
    qualified = (
        (metrics["translation_rms_fraction_BW"] < 0.10)
        & (metrics["translation_peak_fraction_BW"] < 0.25)
        & (metrics["rotation_rms_fraction_BW_height"] < 0.05)
        & (metrics["rotation_peak_fraction_BW_height"] < 0.10)
    )
    return {
        "lag_ms": _LAG_MS.copy(),
        "source_indices": indices,
        "times": common_times,
        "lagged_external_wrenches": lagged_wrenches,
        "generalized_forces": tau,
        "metrics": metrics,
        "qualified": qualified,
        "root_groups": roots,
    }


def _sha256(path: Path) -> str:
    """Return a file SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _runtime(repository_root: Path, device: str | None, wall_time_s: float) -> dict[str, Any]:
    """Record code, dirty state, package versions, and runtime."""
    try:
        commit = subprocess.check_output(["git", "-C", str(repository_root), "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(
            subprocess.check_output(["git", "-C", str(repository_root), "status", "--porcelain"], text=True).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
        dirty = True

    def version(distribution: str) -> str:
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            return "unknown"

    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "warp": version("warp-lang"),
        "newton": version("newton"),
        "device": str(device or "cpu"),
        "wall_time_s": float(wall_time_s),
    }


def _json_value(value: Any) -> Any:
    """Convert NumPy values and reject nonfinite JSON numbers."""
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
        if not np.isfinite(result):
            raise ValueError("strict JSON cannot contain NaN or infinity")
        return result
    if value is None or isinstance(value, str):
        return value
    raise TypeError(f"unsupported JSON value {type(value).__name__}")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    """Write deterministic strict JSON."""
    path.write_text(json.dumps(_json_value(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def run(data_dir: str | os.PathLike, output_dir: str | os.PathLike, device: str | None = None) -> Path:
    """Run and publish the preliminary audit into a new output directory."""
    started = time.monotonic()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing publication: {output_dir}")
    inputs = load_schema3(data_dir)
    model = osim.parse_osim(inputs.model_path)
    surface = solve_lag_surface(inputs, model, device=device)
    audit = audit_model_inertia(model, inputs.measured_mass_kg)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.stage-", dir=output_dir.parent))
    try:
        archive_name = "residual_sensitivity.npz"
        np.savez_compressed(
            stage / archive_name,
            schema_version=np.asarray(_SCHEMA),
            lag_ms=surface["lag_ms"],
            source_indices=surface["source_indices"],
            times=surface["times"],
            coordinate_names=np.asarray(inputs.coordinate_names),
            external_bodies=np.asarray(inputs.external_bodies),
            lagged_external_wrenches=surface["lagged_external_wrenches"],
            generalized_forces=surface["generalized_forces"],
            qualified=surface["qualified"],
            **surface["metrics"],
        )
        lag_rows = []
        for index, lag in enumerate(surface["lag_ms"]):
            lag_rows.append(
                {
                    "lag_ms": int(lag),
                    **{name: float(values[index]) for name, values in surface["metrics"].items()},
                    "residual_thresholds_qualified": bool(surface["qualified"][index]),
                }
            )
        source_code = Path(__file__).resolve()
        source_code_name = "residual_sensitivity_source.py"
        shutil.copy2(source_code, stage / source_code_name)
        repository_root = source_code.parents[2]
        source_hashes = {
            "analysis.npz": _sha256(inputs.analysis_path),
            "qc_summary.json": _sha256(inputs.qc_path),
            "manifest.json": _sha256(inputs.manifest_path),
            inputs.model_path.name: _sha256(inputs.model_path),
            source_code_name: _sha256(stage / source_code_name),
        }
        lock_path = repository_root / "uv.lock"
        if lock_path.is_file():
            source_hashes["uv.lock"] = _sha256(lock_path)
        summary = {
            "schema_version": _SCHEMA,
            "status": "preliminary_sensitivity_only_no_timing_accepted",
            "accepted_timing_lag_ms": None,
            "timing_adjustment_applied": False,
            "qualified_lags_ms": surface["lag_ms"][surface["qualified"]].tolist(),
            "qualification_scope": "four pelvis root resultant residual thresholds only; qualification is not acceptance",
            "lag_sign_convention": "positive wrench lag means the measured timestamps lag motion; aligned(t)=measured(t+lag)",
            "common_non_extrapolated_interval_s": [float(surface["times"][0]), float(surface["times"][-1])],
            "common_sample_count": int(len(surface["times"])),
            "root_groups": surface["root_groups"],
            "normalization": {
                "body_weight_N": inputs.body_weight_N,
                "subject_height_m": inputs.subject_height_m,
                "moment_denominator_BW_height_Nm": inputs.body_weight_N * inputs.subject_height_m,
                "method": "vector resultant per frame, then RMS or peak over time",
            },
            "thresholds": {
                "translation_rms_fraction_BW_strictly_below": 0.10,
                "translation_peak_fraction_BW_strictly_below": 0.25,
                "rotation_rms_fraction_BW_height_strictly_below": 0.05,
                "rotation_peak_fraction_BW_height_strictly_below": 0.10,
            },
            "surface": lag_rows,
            "model_inertial_audit": audit,
            "scope": {
                "included": [
                    "wrench timing lag from -20 through +20 ms at 1 ms",
                    "subject/model mass comparison",
                    "per-segment mass, body-frame COM, and inertia tensor",
                ],
                "excluded": [
                    "timing selection or acceptance",
                    "filter sensitivity",
                    "pelvis kinematic adjustment",
                    "body COM or inertia adjustment",
                    "RRA-like optimization",
                    "contact refit",
                    "FD-1 acceptance",
                ],
                "inverse_dynamics_calls": 1,
                "motion_state": "frozen schema-3 ID coordinates, speeds, and accelerations",
            },
            "source_hashes": source_hashes,
            "artifacts": {
                archive_name: _sha256(stage / archive_name),
                source_code_name: _sha256(stage / source_code_name),
            },
            "runtime": _runtime(repository_root, device, time.monotonic() - started),
        }
        _write_json(stage / "residual_sensitivity.json", summary)
        if output_dir.exists():
            raise FileExistsError(f"publication target appeared during staging: {output_dir}")
        os.rename(stage, output_dir)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return output_dir


def main(argv: list[str] | None = None) -> int:
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Completed gait_c3d schema-3 directory")
    parser.add_argument("--output-dir", required=True, help="New nonexisting publication directory")
    parser.add_argument("--device", default="cpu", help="Warp inverse-dynamics device")
    args = parser.parse_args(argv)
    result = run(args.data_dir, args.output_dir, args.device)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
