# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Publish the accepted official RRA motion as a schema-3 contact input.

This adapter does not run RRA and does not recompute inverse dynamics.  It
verifies one completed :mod:`opensim_rra_reference` artifact, imports the
adjusted model and official CMC kinematics, and resamples only the measured
validation data needed by prescribed contact analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_RRA_SCHEMA = "gait_c3d_official_opensim_rra_reference_1"
_RRA_SCOPE = "official_opensim_rra_reference_not_newton_native_prediction"
_ANALYSIS_SCHEMA = "gait_c3d_analysis_3"
_IMPORT_SCHEMA = "gait_c3d_rra_adjusted_contact_input_1"
ARCHITECTURE_ROLE = "source_adapter"

_SCOPE = "accepted_official_rra_adjusted_contact_input"
_FRAME = "opensim_x_forward_y_up_z_right"
_UNITS = {"length": "m", "force": "N", "moment": "N*m", "time": "s"}
_FOOT_NAMES = ("left", "right")
_CONTACT_THRESHOLD_N = 50.0


@dataclass(frozen=True, slots=True)
class StorageTable:
    """One finite OpenSim Storage table and its declared angular unit."""

    labels: tuple[str, ...]
    values: np.ndarray
    in_degrees: bool


@dataclass(frozen=True, slots=True)
class AdjustedKinematics:
    """Official RRA q, u, and udot on their shared time grid in SI units."""

    times: np.ndarray
    coordinate_names: tuple[str, ...]
    motion_types: tuple[str, ...]
    coordinates: np.ndarray
    speeds: np.ndarray
    accelerations: np.ndarray


def _sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _artifact_hashes(directory: Path, *, excluded: set[Path] | None = None) -> dict[str, str]:
    excluded = {path.resolve() for path in (excluded or set())}
    return {
        path.relative_to(directory).as_posix(): _sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.resolve() not in excluded
    }


def _json_object(path: Path, context: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be a JSON object")
    return value


def _verify_path_hashes(entries: Any, *, kind: str) -> None:
    if not isinstance(entries, dict) or not entries:
        raise ValueError(f"{kind} hashes must be a nonempty object")
    for raw_path, metadata in entries.items():
        if not isinstance(raw_path, str) or not isinstance(metadata, dict):
            raise ValueError(f"{kind} hash entry is invalid")
        expected = metadata.get("sha256")
        path = Path(raw_path)
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"{kind} SHA-256 is invalid for {path}")
        if not path.is_file():
            raise FileNotFoundError(path)
        if _sha256(path) != expected:
            raise ValueError(f"current {kind} hash does not match for {path}")


def parse_storage(path: str | os.PathLike[str]) -> StorageTable:
    """Parse a rectangular OpenSim Storage table, including ``inDegrees``.

    The official q/u/dudt products must declare the angular unit.  All numeric
    values must be finite and time must be strictly increasing.
    """

    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    try:
        end = next(index for index, line in enumerate(lines) if line.strip().lower() == "endheader")
    except StopIteration as exception:
        raise ValueError(f"{path} has no endheader") from exception
    if end + 1 >= len(lines):
        raise ValueError(f"{path} has no column labels")
    header: dict[str, str] = {}
    for line in lines[:end]:
        if "=" in line:
            key, value = line.split("=", 1)
            header[key.strip().lower()] = value.strip()
    angular = header.get("indegrees", "").lower()
    if angular not in {"yes", "no"}:
        raise ValueError(f"{path} must declare inDegrees=yes or inDegrees=no")
    labels = tuple(lines[end + 1].split())
    rows: list[list[float]] = []
    for line in lines[end + 2 :]:
        if line.strip():
            try:
                rows.append([float(value) for value in line.split()])
            except ValueError as exception:
                raise ValueError(f"{path} contains a nonnumeric Storage value") from exception
    if not labels or labels[0].lower() != "time" or not rows or any(len(row) != len(labels) for row in rows):
        raise ValueError(f"{path} is not a rectangular nonempty time Storage table")
    values = np.asarray(rows, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{path} contains nonfinite values")
    if len(values) < 2 or np.any(np.diff(values[:, 0]) <= 0.0):
        raise ValueError(f"{path} time must be strictly increasing")
    for key, actual in (("nrows", len(values)), ("ncolumns", len(labels))):
        if key in header:
            try:
                declared = int(header[key])
            except ValueError as exception:
                raise ValueError(f"{path} has an invalid {key} header") from exception
            if declared != actual:
                raise ValueError(f"{path} {key} header does not match its table")
    return StorageTable(labels, values, angular == "yes")


def _model_coordinates(path: Path) -> tuple[tuple[str, ...], tuple[str | None, ...]]:
    root = ET.parse(path).getroot()
    names: list[str] = []
    motion_types: list[str | None] = []
    for coordinate in root.iter("Coordinate"):
        name = (coordinate.get("name") or "").strip()
        motion_type = (coordinate.findtext("motion_type") or "").strip().lower()
        if not name or motion_type not in {"", "rotational", "translational"}:
            raise ValueError(f"model coordinate {name!r} has an unsupported motion_type")
        names.append(name)
        motion_types.append(motion_type or None)
    if not names or len(set(names)) != len(names):
        raise ValueError("model must contain unique coordinates")
    return tuple(names), tuple(motion_types)


def load_adjusted_kinematics(
    model_path: str | os.PathLike[str],
    q_path: str | os.PathLike[str],
    u_path: str | os.PathLike[str],
    dudt_path: str | os.PathLike[str],
    *,
    motion_type_model_path: str | os.PathLike[str] | None = None,
) -> AdjustedKinematics:
    """Load official RRA Kinematics_q/u/dudt and convert angular columns to SI.

    OpenSim 4.x omits the legacy ``motion_type`` elements when it serializes the
    adjusted model.  In that case, ``motion_type_model_path`` supplies the
    original model metadata, but its coordinate order must exactly match the
    adjusted model.
    """

    model_names, adjusted_motion_types = _model_coordinates(Path(model_path))
    if all(value is not None for value in adjusted_motion_types):
        motion_types = tuple(str(value) for value in adjusted_motion_types)
    else:
        if motion_type_model_path is None:
            raise ValueError("adjusted model omits motion_type; the source model is required for unit conversion")
        source_names, source_motion_types = _model_coordinates(Path(motion_type_model_path))
        if source_names != model_names or any(value is None for value in source_motion_types):
            raise ValueError("source and adjusted model coordinate metadata do not match")
        motion_types = tuple(str(value) for value in source_motion_types)
    tables = [parse_storage(path) for path in (q_path, u_path, dudt_path)]
    expected_labels = ("time", *model_names)
    for table in tables:
        if table.labels != expected_labels:
            raise ValueError("RRA kinematics coordinate order does not match the adjusted model")
        if table.values.shape != tables[0].values.shape or not np.array_equal(
            table.values[:, 0], tables[0].values[:, 0]
        ):
            raise ValueError("RRA Kinematics_q/u/dudt must use exactly the same time grid")
    arrays: list[np.ndarray] = []
    rotational = np.asarray([value == "rotational" for value in motion_types])
    for table in tables:
        values = table.values[:, 1:].copy()
        if table.in_degrees:
            values[:, rotational] = np.deg2rad(values[:, rotational])
        arrays.append(values)
    return AdjustedKinematics(
        times=tables[0].values[:, 0].copy(),
        coordinate_names=model_names,
        motion_types=motion_types,
        coordinates=arrays[0],
        speeds=arrays[1],
        accelerations=arrays[2],
    )


def _interpolate_numeric(times: np.ndarray, values: np.ndarray, sample_times: np.ndarray, context: str) -> np.ndarray:
    source_times = np.asarray(times, dtype=float)
    source = np.asarray(values, dtype=float)
    targets = np.asarray(sample_times, dtype=float)
    if source_times.ndim != 1 or len(source_times) < 2 or np.any(np.diff(source_times) <= 0.0):
        raise ValueError("source analysis times must be strictly increasing")
    if source.shape[0] != len(source_times) or not np.all(np.isfinite(source)):
        raise ValueError(f"source {context} must be finite and match source times")
    tolerance = 1.0e-12 * max(1.0, abs(float(source_times[0])), abs(float(source_times[-1])))
    if targets.ndim != 1 or len(targets) < 2 or np.any(np.diff(targets) <= 0.0):
        raise ValueError("RRA sample times must be strictly increasing")
    if targets[0] < source_times[0] - tolerance or targets[-1] > source_times[-1] + tolerance:
        raise ValueError(f"RRA time grid would extrapolate source {context}")
    flat = source.reshape(len(source_times), -1)
    sampled = np.column_stack([np.interp(targets, source_times, flat[:, index]) for index in range(flat.shape[1])])
    return sampled.reshape((len(targets), *source.shape[1:]))


def _interpolate_optional_numeric(
    times: np.ndarray, values: np.ndarray, sample_times: np.ndarray, context: str
) -> np.ndarray:
    """Interpolate columns only inside their finite support, never extrapolating."""

    source_times = np.asarray(times, dtype=float)
    source = np.asarray(values, dtype=float)
    targets = np.asarray(sample_times, dtype=float)
    if source.shape[0] != len(source_times):
        raise ValueError(f"source {context} must match source times")
    # Validate the common time contract even if every optional value is missing.
    _interpolate_numeric(source_times, np.zeros((len(source_times), 1)), targets, context)
    flat = source.reshape(len(source_times), -1)
    sampled = np.full((len(targets), flat.shape[1]), np.nan)
    for index in range(flat.shape[1]):
        finite = np.isfinite(flat[:, index])
        count = int(np.count_nonzero(finite))
        if count == 1:
            exact = np.isclose(targets, source_times[finite][0], rtol=0.0, atol=1.0e-12)
            sampled[exact, index] = flat[finite, index][0]
        elif count > 1:
            support_times = source_times[finite]
            inside = (targets >= support_times[0]) & (targets <= support_times[-1])
            sampled[inside, index] = np.interp(targets[inside], support_times, flat[finite, index])
    return sampled.reshape((len(targets), *source.shape[1:]))


def _require_archive_array(
    archive: np.lib.npyio.NpzFile,
    name: str,
    ndim: int,
    *,
    finite: bool = True,
) -> np.ndarray:
    if name not in archive.files:
        raise ValueError(f"source analysis is missing {name}")
    value = np.asarray(archive[name])
    if value.ndim != ndim or not np.issubdtype(value.dtype, np.number):
        raise ValueError(f"source analysis {name} must be a numeric {ndim}-D array")
    if finite and not np.all(np.isfinite(value)):
        raise ValueError(f"source analysis {name} must be finite")
    return value.astype(float, copy=False)


def _verify_rra_reference(rra_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    summary_path = rra_dir / "summary.json"
    runtime_path = rra_dir / "run_runtime.json"
    prepare_path = rra_dir / "prepare_manifest.json"
    for path in (summary_path, runtime_path, prepare_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    summary = _json_object(summary_path, "RRA summary")
    runtime = _json_object(runtime_path, "RRA runtime")
    prepare = _json_object(prepare_path, "RRA prepare manifest")
    for value, context in ((summary, "summary"), (runtime, "runtime"), (prepare, "prepare manifest")):
        if value.get("schema_version") != _RRA_SCHEMA or value.get("scope") != _RRA_SCOPE:
            raise ValueError(f"RRA {context} has the wrong schema or scope")

    gates = summary.get("gates")
    required_gates = (
        "runtime_success",
        "no_bad_residual_component",
        "no_bad_perr_coordinate",
        "no_silent_mass_application",
        "normalized_resultants_passed",
        "production_candidate",
    )
    if not isinstance(gates, dict) or any(gates.get(name) is not True for name in required_gates):
        raise ValueError("official RRA summary is not an accepted production candidate")
    if runtime.get("success") is not True or summary.get("runtime") != runtime:
        raise ValueError("official RRA summary does not describe the current successful runtime")
    run_id = runtime.get("run_id")
    if (
        not isinstance(run_id, str)
        or len(run_id) != 32
        or any(character not in "0123456789abcdef" for character in run_id)
    ):
        raise ValueError("official RRA runtime has no valid run id")
    if runtime.get("artifact_linkage", {}).get("run_id") != run_id:
        raise ValueError("official RRA runtime linkage does not match its run id")
    linkage = summary.get("runtime_linkage")
    if (
        not isinstance(linkage, dict)
        or linkage.get("run_id") != run_id
        or linkage.get("runtime_sha256") != _sha256(runtime_path)
    ):
        raise ValueError("official RRA summary runtime hash or run id is stale")
    if runtime.get("prepare_manifest_sha256") != _sha256(prepare_path):
        raise ValueError("official RRA runtime was not produced from the current prepare manifest")

    expected_summary_hashes = summary.get("artifacts")
    current_summary_hashes = _artifact_hashes(rra_dir, excluded={summary_path})
    if expected_summary_hashes != current_summary_hashes:
        raise ValueError("current official RRA artifacts do not match the accepted summary hashes")
    deferred = runtime.get("deferred_artifacts_finalized_after_process_exit", [])
    if not isinstance(deferred, list) or any(not isinstance(name, str) for name in deferred):
        raise ValueError("official RRA deferred artifact list is invalid")
    deferred_paths = {rra_dir / name for name in deferred}
    current_runtime_hashes = _artifact_hashes(rra_dir, excluded={runtime_path, summary_path, *deferred_paths})
    if runtime.get("artifacts") != current_runtime_hashes:
        raise ValueError("current official RRA stable artifacts do not match the successful runtime hashes")
    for path in deferred_paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"official RRA deferred artifact is missing or empty: {path}")
    _verify_path_hashes(prepare.get("source_inputs"), kind="RRA source input")
    _verify_path_hashes(prepare.get("generated_inputs"), kind="RRA generated input")
    return summary, runtime, prepare


def _paths_for_rra_products(rra_dir: Path, prepare: dict[str, Any]) -> dict[str, Path]:
    tool_name = prepare.get("tool_name")
    if not isinstance(tool_name, str) or not tool_name:
        raise ValueError("RRA prepare manifest has no tool name")
    products = {
        "model": rra_dir / "results" / f"{tool_name}_adjusted.osim",
        "q": rra_dir / "results" / f"{tool_name}_Kinematics_q.sto",
        "u": rra_dir / "results" / f"{tool_name}_Kinematics_u.sto",
        "dudt": rra_dir / "results" / f"{tool_name}_Kinematics_dudt.sto",
    }
    for path in products.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    return products


def _runtime_provenance(opensim_version: str, run_id: str) -> dict[str, Any]:
    try:
        numpy_version = importlib.metadata.version("numpy")
    except importlib.metadata.PackageNotFoundError:
        numpy_version = np.__version__
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "numpy": numpy_version,
        "official_opensim_version": opensim_version,
        "official_rra_run_id": run_id,
        "rra_adjusted_contact_input_sha256": _sha256(Path(__file__).resolve()),
    }


def publish_rra_adjusted_contact_input(
    rra_dir: str | os.PathLike[str],
    data_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
) -> Path:
    """Verify and atomically publish an accepted official RRA contact input.

    ``output_dir`` must be new, outside the repository, and disjoint from both
    source directories.  The original inverse-dynamics generalized forces are
    deliberately not copied because they do not belong to the adjusted motion.
    """

    rra_dir = Path(rra_dir).resolve()
    data_dir = Path(data_dir).resolve()
    output_dir = Path(output_dir).resolve()
    repository_root = Path(__file__).resolve().parents[2]
    if not rra_dir.is_dir() or not data_dir.is_dir():
        raise FileNotFoundError("RRA reference and source data directories must exist")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if output_dir == repository_root or output_dir.is_relative_to(repository_root):
        raise ValueError("generated RRA-adjusted contact input must stay outside the repository")
    for source in (rra_dir, data_dir):
        if output_dir == source or output_dir.is_relative_to(source) or source.is_relative_to(output_dir):
            raise ValueError("output and source directories must not overlap")

    summary, runtime, prepare = _verify_rra_reference(rra_dir)
    products = _paths_for_rra_products(rra_dir, prepare)
    original_analysis_path = data_dir / "analysis.npz"
    original_manifest_path = data_dir / "manifest.json"
    original_model_path = data_dir / "S001_scaled.osim"
    for path in (original_analysis_path, original_manifest_path, original_model_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    source_manifest = _json_object(original_manifest_path, "source schema-3 manifest")
    if source_manifest.get("schema_version") != _ANALYSIS_SCHEMA:
        raise ValueError(f"source manifest schema_version must be {_ANALYSIS_SCHEMA}")
    if source_manifest.get("frame", _FRAME) != _FRAME:
        raise ValueError(f"source manifest frame must be {_FRAME}")
    if source_manifest.get("units", _UNITS) != _UNITS:
        raise ValueError("source manifest units do not match the SI contract")
    if not isinstance(source_manifest.get("runtime"), dict):
        raise ValueError("source manifest runtime must be an object")

    source_model_entries = [
        (Path(path).resolve(), metadata)
        for path, metadata in prepare["source_inputs"].items()
        if Path(path).name == "S001_scaled.osim"
    ]
    if len(source_model_entries) != 1 or source_model_entries[0][0] != original_model_path:
        raise ValueError("official RRA was not prepared from this source schema-3 model")

    kinematics = load_adjusted_kinematics(
        products["model"],
        products["q"],
        products["u"],
        products["dudt"],
        motion_type_model_path=original_model_path,
    )
    with np.load(original_analysis_path, allow_pickle=False) as archive:
        if "schema_version" not in archive.files or np.asarray(archive["schema_version"]).shape != ():
            raise ValueError("source analysis schema_version must be a scalar")
        if str(np.asarray(archive["schema_version"]).item()) != _ANALYSIS_SCHEMA:
            raise ValueError(f"source analysis schema_version must be {_ANALYSIS_SCHEMA}")
        source_times = _require_archive_array(archive, "times", 1)
        source_grf = _require_archive_array(archive, "grf", 3)
        source_cop = _require_archive_array(archive, "cop", 3, finite=False)
        source_free_torque = _require_archive_array(archive, "free_torque", 3)
        source_target_markers = _require_archive_array(archive, "target_markers", 3, finite=False)
        if "id_names" not in archive.files or np.asarray(archive["id_names"]).ndim != 1:
            raise ValueError("source analysis id_names must be one-dimensional")
        if "motion_types" not in archive.files or np.asarray(archive["motion_types"]).ndim != 1:
            raise ValueError("source analysis motion_types must be one-dimensional")
        if "marker_names" not in archive.files or np.asarray(archive["marker_names"]).ndim != 1:
            raise ValueError("source analysis marker_names must be one-dimensional")
        if "foot_names" not in archive.files or np.asarray(archive["foot_names"]).ndim != 1:
            raise ValueError("source analysis foot_names must be one-dimensional")
        source_coordinate_names = tuple(str(value) for value in np.asarray(archive["id_names"]))
        source_motion_types = tuple(str(value).lower() for value in np.asarray(archive["motion_types"]))
        marker_names = tuple(str(value) for value in np.asarray(archive["marker_names"]))
        foot_names = tuple(str(value) for value in np.asarray(archive["foot_names"]))
    if source_coordinate_names != kinematics.coordinate_names or source_motion_types != kinematics.motion_types:
        raise ValueError("source schema-3 coordinate order or motion types do not match the adjusted model")
    if foot_names != _FOOT_NAMES:
        raise ValueError("source analysis foot order must be exactly [left, right]")
    sample_count = len(source_times)
    if source_grf.shape != (sample_count, 2, 3) or source_cop.shape != (sample_count, 2, 3):
        raise ValueError("source grf and cop must have shape [time, 2, 3]")
    if source_free_torque.shape != (sample_count, 2, 3):
        raise ValueError("source free_torque must have shape [time, 2, 3]")
    if source_target_markers.shape != (sample_count, len(marker_names), 3):
        raise ValueError("source target_markers must have shape [time, marker, 3]")

    times = kinematics.times
    grf = _interpolate_numeric(source_times, source_grf, times, "corrected GRF")
    free_torque = _interpolate_numeric(source_times, source_free_torque, times, "corrected free torque")
    cop = _interpolate_optional_numeric(source_times, source_cop, times, "corrected COP")
    target_markers = _interpolate_optional_numeric(
        source_times, source_target_markers, times, "target marker validation data"
    )
    contact = grf[:, :, 1] >= _CONTACT_THRESHOLD_N
    cop[~contact] = np.nan
    if not np.all(np.isfinite(cop[contact])):
        raise ValueError("corrected COP has no finite interpolation support inside an RRA loaded run")

    opensim_version = runtime.get("opensim_version")
    run_id = runtime["run_id"]
    if not isinstance(opensim_version, str) or not opensim_version.strip():
        raise ValueError("successful official RRA runtime has no OpenSim version")
    source_hashes = {
        "original_analysis": _sha256(original_analysis_path),
        "original_manifest": _sha256(original_manifest_path),
        "original_model": _sha256(original_model_path),
        "rra_summary": _sha256(rra_dir / "summary.json"),
        "rra_runtime": _sha256(rra_dir / "run_runtime.json"),
        "rra_prepare_manifest": _sha256(rra_dir / "prepare_manifest.json"),
        "rra_adjusted_model": _sha256(products["model"]),
        "rra_kinematics_q": _sha256(products["q"]),
        "rra_kinematics_u": _sha256(products["u"]),
        "rra_kinematics_dudt": _sha256(products["dudt"]),
    }
    residual_grades = {
        name: value.get("grades", {})
        for name, value in summary.get("residual_components", {}).items()
        if isinstance(value, dict)
    }
    perr_grades = {
        name: {
            "grades": value.get("grades", {}),
            "included_in_no_bad_perr_gate": value.get("included_in_no_bad_perr_gate"),
        }
        for name, value in summary.get("perr", {}).items()
        if isinstance(value, dict)
    }
    accepted_gates = {
        name: summary["gates"][name]
        for name in (
            "runtime_success",
            "no_bad_residual_component",
            "no_bad_perr_coordinate",
            "no_silent_mass_application",
            "normalized_resultants_passed",
            "production_candidate",
        )
    }
    qc = {
        "schema_version": _IMPORT_SCHEMA,
        "scope": _SCOPE,
        "status": "production_candidate",
        "frame": _FRAME,
        "units": dict(_UNITS),
        "rra_run_id": run_id,
        "opensim_version": opensim_version,
        "source_hashes": source_hashes,
        "rra_acceptance": {
            "gates": accepted_gates,
            "residual_component_grades": residual_grades,
            "perr_coordinate_grades": perr_grades,
            "okay_components_requiring_explicit_review": summary["gates"].get(
                "okay_components_require_explicit_review", []
            ),
        },
        "data": {
            "sample_count": len(times),
            "coordinate_count": len(kinematics.coordinate_names),
            "marker_count": len(marker_names),
            "foot_order": list(_FOOT_NAMES),
            "contact_threshold_N": _CONTACT_THRESHOLD_N,
            "contact_frame_counts": {
                side: int(np.count_nonzero(contact[:, index])) for index, side in enumerate(_FOOT_NAMES)
            },
            "time_range_s": [float(times[0]), float(times[-1])],
            "target_marker_nonfinite_count": int(np.count_nonzero(~np.isfinite(target_markers))),
            "cop_nonfinite_on_contact_count": int(np.count_nonzero(~np.isfinite(cop[contact]))),
        },
        "gates": {
            "coordinate_order_matches_adjusted_model": True,
            "shared_rra_time_grid": True,
            "interpolation_without_time_extrapolation": True,
            "cop_finite_on_loaded_runs": True,
            "official_rra_artifact_hashes_current": True,
            "original_id_generalized_forces_excluded": True,
            "shapes_valid": True,
        },
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        shutil.copy2(products["model"], temporary / "S001_scaled.osim")
        np.savez_compressed(
            temporary / "analysis.npz",
            schema_version=np.asarray(_ANALYSIS_SCHEMA),
            frame=np.asarray(_FRAME),
            times=times,
            coords=kinematics.coordinates,
            coordinate_names=np.asarray(kinematics.coordinate_names),
            motion_types=np.asarray(kinematics.motion_types),
            marker_names=np.asarray(marker_names),
            target_markers=target_markers,
            grf=grf,
            cop=cop,
            free_torque=free_torque,
            contact=contact,
            foot_names=np.asarray(_FOOT_NAMES),
            id_coordinates=kinematics.coordinates,
            id_speeds=kinematics.speeds,
            id_accelerations=kinematics.accelerations,
            id_names=np.asarray(kinematics.coordinate_names),
        )
        _write_json(temporary / "qc_summary.json", qc)
        artifacts = {
            name: _sha256(temporary / name) for name in ("S001_scaled.osim", "analysis.npz", "qc_summary.json")
        }
        manifest = {
            "schema_version": _ANALYSIS_SCHEMA,
            "artifact_schema_version": _IMPORT_SCHEMA,
            "scope": _SCOPE,
            "status": "production_candidate",
            "input_directory": str(data_dir),
            "rra_reference_directory": str(rra_dir),
            "output_directory": str(output_dir),
            "frame": _FRAME,
            "units": dict(_UNITS),
            "runtime": _runtime_provenance(opensim_version, run_id),
            "source_hashes": source_hashes,
            "artifacts": artifacts,
            "rra_reference": {
                "run_id": run_id,
                "opensim_version": opensim_version,
                "accepted_gates": accepted_gates,
                "current_summary_artifact_hash_count": len(summary["artifacts"]),
                "current_runtime_artifact_hash_count": len(runtime["artifacts"]),
            },
            "resampling": {
                "time_grid": "official RRA Kinematics_q time grid",
                "angular_conversion": "rotational q/u/dudt degrees converted to radians from adjusted-model motion_type",
                "numeric_targets": "linear interpolation without time extrapolation",
                "optional_targets": "per-column finite-support interpolation without extrapolation",
                "cop": "optional interpolation, retained only where interpolated vertical GRF is at least 50 N",
                "contact": "interpolated corrected GRF Fy >= 50 N",
            },
            "information_set": {
                "adjusted_state_source": ["Kinematics_q", "Kinematics_u", "Kinematics_dudt"],
                "validation_data_source": ["corrected_grf", "corrected_cop", "corrected_free_torque", "target_markers"],
                "original_id_generalized_forces_carried_forward": False,
                "inverse_dynamics_must_be_regenerated_for_adjusted_motion": True,
            },
        }
        _write_json(temporary / "manifest.json", manifest)
        if output_dir.exists():
            raise FileExistsError(output_dir)
        temporary.rename(output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output_dir


# Explicit aliases keep the API readable for callers that think of this as an
# import or build step rather than a publication step.
import_rra_adjusted_contact_input = publish_rra_adjusted_contact_input
build_rra_adjusted_contact_input = publish_rra_adjusted_contact_input


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rra-reference", required=True, help="Accepted official opensim_rra_reference directory")
    parser.add_argument("--data-dir", required=True, help="Original gait_c3d schema-3 latest directory")
    parser.add_argument("--output", required=True, help="New non-overlapping output directory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = create_parser().parse_args(argv)
    output = publish_rra_adjusted_contact_input(arguments.rra_reference, arguments.data_dir, arguments.output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
