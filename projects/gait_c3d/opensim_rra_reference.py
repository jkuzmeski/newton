# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Prepare, run, and summarize the official OpenSim RRA reference for Trial 101.

This adapter is intentionally scoped to the official OpenSim ``RRATool``.  It is
not a Newton-native residual-reduction or predictive-forward-dynamics result.
OpenSim is optional and imported only by :func:`run_reference`.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import math
import os
import platform
import re
import shutil
import sys
import time
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SCHEMA = "gait_c3d_official_opensim_rra_reference_1"
_SCOPE = "official_opensim_rra_reference_not_newton_native_prediction"
_DEFAULT_DATA = Path("/home/jo31399/newton-data/gait/processed/trial_101/latest")
_PINNED_COMMIT = "11036b39ca7232c604685b37f483afafc056d92b"
_RAW_ROOT = f"https://raw.githubusercontent.com/opensim-org/opensim-core/{_PINNED_COMMIT}"
_SOURCE_FILES = {
    "Applications/RRA/test/gait2354_RRA_Tasks.xml": "366170335acfe521ec68eb2885b91b42f965bad259a2889f59b0fe305c6de6c4",
    "Applications/RRA/test/gait2354_RRA_Actuators.xml": "a633d79da6f2171889035c722276df2429fb6063fe4aa24764ed4a4de36088ff",
    "OpenSim/Tools/RRATool.cpp": "088d988e4e978cb11ed98e926fc2ec34eadaabac7b1cfb469f3ddac2b151632f",
    "OpenSim/Tools/CMC.cpp": "191bc1868d6276469becb3390dc21f366cc05430e9cd82969f84e4bf3a1b779a",
    "OpenSim/Tools/CMC_Joint.cpp": "39472e421a432aa7e4412d4860a757bd30053fc7af98a4195aa2044811d03af1",
    "OpenSim/Tools/ActuatorForceTarget.cpp": "c07239d4339624b24390cda037e2e78b5874a2f36bfaecddde9706dfb7b441cb",
}

# These are the complete gait2354 values embedded from the two pinned resources.
_TASK_WEIGHTS = {
    "pelvis_tz": 5.0,
    "pelvis_tx": 5.0,
    "pelvis_ty": 5.0,
    "pelvis_tilt": 1000.0,
    "pelvis_list": 500.0,
    "pelvis_rotation": 100.0,
    "hip_flexion_r": 20.0,
    "hip_adduction_r": 20.0,
    "hip_rotation_r": 20.0,
    "knee_angle_r": 20.0,
    "ankle_angle_r": 20.0,
    "subtalar_angle_r": 20.0,
    "mtp_angle_r": 20.0,
    "hip_flexion_l": 20.0,
    "hip_adduction_l": 20.0,
    "hip_rotation_l": 20.0,
    "knee_angle_l": 20.0,
    "ankle_angle_l": 20.0,
    "subtalar_angle_l": 20.0,
    "mtp_angle_l": 20.0,
    "lumbar_extension": 50.0,
    "lumbar_bending": 50.0,
    "lumbar_rotation": 10.0,
}
_COORDINATE_OPTIMAL_FORCES = {
    "hip_flexion_r": 300.0,
    "hip_adduction_r": 200.0,
    "hip_rotation_r": 100.0,
    "knee_angle_r": 300.0,
    "ankle_angle_r": 300.0,
    "subtalar_angle_r": 100.0,
    "mtp_angle_r": 100.0,
    "hip_flexion_l": 300.0,
    "hip_adduction_l": 200.0,
    "hip_rotation_l": 100.0,
    "knee_angle_l": 300.0,
    "ankle_angle_l": 300.0,
    "subtalar_angle_l": 100.0,
    "mtp_angle_l": 100.0,
    "lumbar_extension": 200.0,
    "lumbar_bending": 200.0,
    "lumbar_rotation": 200.0,
}
_PELVIS_COORDINATES = {
    "pelvis_tilt",
    "pelvis_list",
    "pelvis_rotation",
    "pelvis_tx",
    "pelvis_ty",
    "pelvis_tz",
}
_GAINS = (100.0, 20.0, 1.0)


@dataclass(frozen=True)
class CoordinateSpec:
    """Coordinate metadata read without importing OpenSim."""

    name: str
    motion_type: str
    locked: bool


@dataclass(frozen=True)
class PreparedReference:
    """Paths produced by :func:`prepare_reference`."""

    output_dir: Path
    setup_path: Path
    manifest_path: Path


def _sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_hash_entries(entries: Any, *, kind: str) -> None:
    """Verify every file in a manifest hash mapping."""
    if not isinstance(entries, dict) or not entries:
        raise ValueError(f"prepare manifest has no {kind} hash entries")
    for raw_path, metadata in entries.items():
        if not isinstance(raw_path, str) or not isinstance(metadata, dict):
            raise ValueError(f"prepare manifest has an invalid {kind} hash entry")
        expected = metadata.get("sha256")
        path = Path(raw_path)
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError(f"prepare manifest has an invalid {kind} SHA-256 for {path}")
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"{kind} hash changed for {path}: expected {expected}, got {actual}")


def _resolve_external_loads_datafile(external_loads_path: Path) -> Path:
    """Resolve the one Storage data file referenced by an ExternalLoads XML."""
    values = [(element.text or "").strip() for element in ET.parse(external_loads_path).getroot().iter("datafile")]
    values = [value for value in values if value]
    if len(values) != 1:
        raise ValueError(f"{external_loads_path} must reference exactly one ExternalLoads datafile")
    referenced = Path(values[0]).expanduser()
    if not referenced.is_absolute():
        referenced = external_loads_path.parent / referenced
    referenced = referenced.resolve()
    if not referenced.is_file():
        raise FileNotFoundError(referenced)
    return referenced


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _text(element: ET.Element, tag: str) -> str:
    return (element.findtext(tag) or "").strip()


def parse_model_spec(path: str | os.PathLike[str]) -> tuple[list[CoordinateSpec], tuple[float, float, float]]:
    """Read coordinate locking and scaled pelvis COM from an OpenSim XML model.

    A clamped zero-width range is treated as locked.  This is how the scaled
    S001 MTP coordinates are fixed even though their legacy ``locked`` field is
    false.
    """
    root = ET.parse(path).getroot()
    coordinates: list[CoordinateSpec] = []
    for coordinate in root.iter("Coordinate"):
        name = coordinate.get("name", "").strip()
        motion_type = _text(coordinate, "motion_type").lower()
        if not name or motion_type not in {"rotational", "translational"}:
            raise ValueError(f"coordinate {name!r} has no supported motion_type")
        explicit = _text(coordinate, "locked") or _text(coordinate, "default_locked")
        locked = explicit.lower() == "true"
        range_values = _text(coordinate, "range").split()
        if _text(coordinate, "clamped").lower() == "true" and len(range_values) == 2:
            locked = locked or math.isclose(float(range_values[0]), float(range_values[1]), abs_tol=1.0e-15)
        coordinates.append(CoordinateSpec(name, motion_type, locked))
    if not coordinates or len({item.name for item in coordinates}) != len(coordinates):
        raise ValueError("model must contain unique coordinates")

    pelvis = next((body for body in root.iter("Body") if body.get("name") == "pelvis"), None)
    if pelvis is None:
        raise ValueError("model has no pelvis body")
    values = _text(pelvis, "mass_center").split()
    if len(values) != 3:
        raise ValueError("pelvis mass_center must contain three values")
    pelvis_com = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in pelvis_com):
        raise ValueError("pelvis mass_center must be finite")
    return coordinates, pelvis_com  # type: ignore[return-value]


def _element(
    parent: ET.Element, tag: str, text: str | float | int | bool | None = None, **attributes: str
) -> ET.Element:
    child = ET.SubElement(parent, tag, attributes)
    if text is not None:
        if isinstance(text, bool):
            child.text = "true" if text else "false"
        elif isinstance(text, float):
            child.text = f"{text:.8f}"
        else:
            child.text = str(text)
    return child


def _write_xml(path: Path, root: ET.Element) -> None:
    ET.indent(root, space="  ")
    data = ET.tostring(root, encoding="utf-8", xml_declaration=True, short_empty_elements=True)
    path.write_bytes(data + b"\n")


def build_actuator_xml(
    coordinates: list[CoordinateSpec], pelvis_com: tuple[float, float, float], *, fy_optimal_force: float = 8.0
) -> ET.Element:
    """Build the pinned gait2354 ideal-actuator set for this scaled model.

    The six pelvis generalized-force slots are supplied by three point and three
    torque residual actuators.  Every upstream internal-coordinate actuator is
    retained, including fixed/clamped coordinates, because RRATool's inverse
    dynamics requires a force-complete system. Official parity also retains the
    upstream locked-coordinate CMC tasks; their pErr is diagnostic only.
    """
    if not math.isfinite(fy_optimal_force) or fy_optimal_force <= 0.0:
        raise ValueError("FY optimal force must be positive and finite")
    names = {coordinate.name for coordinate in coordinates}
    expected = set(_TASK_WEIGHTS)
    if names != expected:
        raise ValueError(f"model coordinates do not match embedded gait2354 specification: {sorted(names ^ expected)}")
    if set(_COORDINATE_OPTIMAL_FORCES) != names - _PELVIS_COORDINATES:
        raise AssertionError("embedded actuator specification is incomplete")

    document = ET.Element("OpenSimDocument", {"Version": "20302"})
    force_set = _element(document, "ForceSet", None, name="gait2354_RRA_S001")
    objects = _element(force_set, "objects")
    point_text = " ".join(f"{value:.8f}" for value in pelvis_com)
    directions = {"FX": "1 0 0", "FY": "0 1 0", "FZ": "0 0 1"}
    for name, optimal_force in (("FX", 4.0), ("FY", fy_optimal_force), ("FZ", 4.0)):
        actuator = _element(objects, "PointActuator", None, name=name)
        _element(actuator, "isDisabled", False)
        _element(actuator, "min_control", "-infinity")
        _element(actuator, "max_control", "infinity")
        _element(actuator, "body", "pelvis")
        _element(actuator, "point", point_text)
        _element(actuator, "point_is_global", False)
        _element(actuator, "direction", directions[name])
        _element(actuator, "force_is_global", True)
        _element(actuator, "optimal_force", optimal_force)
    axes = {"MX": "1 0 0", "MY": "0 1 0", "MZ": "0 0 1"}
    for name in ("MX", "MY", "MZ"):
        actuator = _element(objects, "TorqueActuator", None, name=name)
        _element(actuator, "isDisabled", False)
        _element(actuator, "min_control", "-infinity")
        _element(actuator, "max_control", "infinity")
        _element(actuator, "bodyA", "pelvis")
        _element(actuator, "bodyB", "ground")
        _element(actuator, "torque_is_global", True)
        _element(actuator, "axis", axes[name])
        _element(actuator, "optimal_force", 2.0)
    for coordinate in coordinates:
        if coordinate.name in _PELVIS_COORDINATES:
            continue
        actuator = _element(objects, "CoordinateActuator", None, name=coordinate.name)
        _element(actuator, "isDisabled", False)
        _element(actuator, "min_control", "-infinity")
        _element(actuator, "max_control", "infinity")
        _element(actuator, "coordinate", coordinate.name)
        _element(actuator, "optimal_force", _COORDINATE_OPTIMAL_FORCES[coordinate.name])
    _element(force_set, "groups")
    return document


def build_task_xml(coordinates: list[CoordinateSpec], *, omit_locked_tasks: bool = False) -> ET.Element:
    """Build the exact upstream task set, or an explicit locked-task experiment.

    Official parity retains all 23 gait2354 tasks. The optional omission mode is
    an experiment only; Trial 101 full-run evidence found it IPOPT-infeasible.
    """
    names = {coordinate.name for coordinate in coordinates}
    if names != set(_TASK_WEIGHTS):
        raise ValueError("model coordinates do not match embedded gait2354 task specification")
    document = ET.Element("OpenSimDocument", {"Version": "20302"})
    task_set = _element(document, "CMC_TaskSet", None, name="gait2354_RRA_S001")
    objects = _element(task_set, "objects")
    for coordinate in coordinates:
        if omit_locked_tasks and coordinate.locked:
            continue
        task = _element(objects, "CMC_Joint", None, name=coordinate.name)
        _element(task, "on", True)
        _element(task, "weight", _TASK_WEIGHTS[coordinate.name])
        _element(task, "wrt_body", "-1")
        _element(task, "express_body", "-1")
        _element(task, "active", "true false false")
        _element(task, "kp", f"{_GAINS[0]:.8f} 1.00000000 1.00000000")
        _element(task, "kv", f"{_GAINS[1]:.8f} 1.00000000 1.00000000")
        _element(task, "ka", f"{_GAINS[2]:.8f} 1.00000000 1.00000000")
        for axis in range(3):
            _element(task, f"r{axis}", "0 0 0")
        _element(task, "coordinate", coordinate.name)
        _element(task, "limit", 0.0)
    _element(task_set, "groups")
    return document


def _storage_time_range(path: Path) -> tuple[float, float]:
    lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    try:
        header = next(index for index, line in enumerate(lines) if line.strip().lower() == "endheader")
    except StopIteration as exception:
        raise ValueError(f"{path} has no endheader") from exception
    rows = [line.split() for line in lines[header + 2 :] if line.strip()]
    if len(rows) < 2:
        raise ValueError(f"{path} has fewer than two data rows")
    times = [float(row[0]) for row in rows]
    if any(not math.isfinite(value) for value in times) or any(b <= a for a, b in itertools.pairwise(times)):
        raise ValueError(f"{path} times must be finite and strictly increasing")
    return times[0], times[-1]


def _default_stride(data_dir: Path, available: tuple[float, float]) -> tuple[float, float]:
    qc_path = data_dir / "qc_summary.json"
    if qc_path.is_file():
        qc = json.loads(qc_path.read_text(encoding="utf-8"))
        try:
            return float(qc["stride"]["start_time_s"]), float(qc["stride"]["stop_time_s"])
        except (KeyError, TypeError, ValueError):
            pass
    return available


def build_setup_xml(
    *,
    tool_name: str,
    model_path: Path,
    actuator_path: Path,
    results_dir: Path,
    external_loads_path: Path,
    motion_path: Path,
    task_path: Path,
    output_model_path: Path,
    initial_time: float,
    final_time: float,
) -> ET.Element:
    """Build an RRATool setup with the pinned official numerical settings."""
    document = ET.Element("OpenSimDocument", {"Version": "40600"})
    tool = _element(document, "RRATool", None, name=tool_name)
    properties: list[tuple[str, str | float | int | bool | None]] = [
        ("model_file", str(model_path)),
        ("replace_force_set", True),
        ("force_set_files", str(actuator_path)),
        ("results_directory", str(results_dir)),
        ("output_precision", 16),
        ("initial_time", initial_time),
        ("final_time", final_time),
        ("solve_for_equilibrium_for_auxiliary_states", False),
        ("maximum_number_of_integrator_steps", 200000),
        ("maximum_integrator_step_size", 0.001),
        ("minimum_integrator_step_size", 1.0e-8),
        ("integrator_error_tolerance", 1.0e-4),
    ]
    for tag, value in properties:
        _element(tool, tag, value)
    analyses = _element(tool, "AnalysisSet", None, name="Analyses")
    _element(analyses, "objects")
    _element(analyses, "groups")
    controllers = _element(tool, "ControllerSet", None, name="Controllers")
    _element(controllers, "objects")
    _element(controllers, "groups")
    for tag, value in [
        ("external_loads_file", str(external_loads_path)),
        ("desired_points_file", None),
        ("desired_kinematics_file", str(motion_path)),
        ("task_set_file", str(task_path)),
        ("constraints_file", None),
        ("lowpass_cutoff_frequency", 6.0),
        ("optimizer_algorithm", "ipopt"),
        ("numerical_derivative_step_size", 1.0e-4),
        ("optimization_convergence_tolerance", 1.0e-5),
        ("adjust_com_to_reduce_residuals", True),
        ("initial_time_for_com_adjustment", -1.0),
        ("final_time_for_com_adjustment", -1.0),
        ("adjusted_com_body", "torso"),
        ("output_model_file", str(output_model_path)),
        ("use_verbose_printing", False),
    ]:
        _element(tool, tag, value)
    return document


def prepare_reference(
    data_dir: str | os.PathLike[str] = _DEFAULT_DATA,
    output_dir: str | os.PathLike[str] | None = None,
    *,
    initial_time: float | None = None,
    final_time: float | None = None,
    tool_name: str = "trial101_official_opensim_rra_reference",
    omit_locked_tasks: bool = False,
    fy_optimal_force: float = 8.0,
) -> PreparedReference:
    """Generate deterministic official RRA inputs without importing OpenSim."""
    data_dir = Path(data_dir).resolve()
    output_dir = Path(output_dir).resolve() if output_dir else data_dir.parent / "opensim_rra_reference_generated"
    model_path = data_dir / "S001_scaled.osim"
    motion_path = data_dir / "trial_ik_dynamics_context.mot"
    external_loads_path = data_dir / "trial_grf_context.xml"
    for path in (model_path, motion_path, external_loads_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    external_loads_data_path = _resolve_external_loads_datafile(external_loads_path)
    qc_path = data_dir / "qc_summary.json"
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", tool_name):
        raise ValueError("tool_name must contain only letters, digits, dot, underscore, or hyphen")

    repository_root = Path(__file__).resolve().parents[2]
    if output_dir == repository_root or output_dir.is_relative_to(repository_root):
        raise ValueError("official RRA generated data must stay outside the repository")
    if output_dir == data_dir or output_dir.is_relative_to(data_dir) or data_dir.is_relative_to(output_dir):
        raise ValueError("RRA output and source data directories must not overlap")
    inputs_dir = output_dir / "inputs"
    results_dir = output_dir / "results"
    for generated_dir in (inputs_dir, results_dir):
        if generated_dir.exists():
            shutil.rmtree(generated_dir)
        generated_dir.mkdir(parents=True)
    for stale in (
        output_dir / "opensim.log",
        output_dir / "rratool.log",
        output_dir / "run_runtime.json",
        output_dir / "summary.json",
    ):
        stale.unlink(missing_ok=True)
    for stale in output_dir.glob("delete_this_to_stop_optimization__*.txt"):
        stale.unlink()

    coordinates, pelvis_com = parse_model_spec(model_path)
    available = _storage_time_range(motion_path)
    default_initial, default_final = _default_stride(data_dir, available)
    requested = (
        default_initial if initial_time is None else initial_time,
        default_final if final_time is None else final_time,
    )
    if not all(math.isfinite(value) for value in requested):
        raise ValueError("time range must be finite")
    effective = (max(requested[0], available[0]), min(requested[1], available[1]))
    if effective[1] <= effective[0]:
        raise ValueError(f"requested time range {requested} has no interval inside motion range {available}")

    actuator_path = inputs_dir / "gait2354_RRA_Actuators_S001.xml"
    task_path = inputs_dir / "gait2354_RRA_Tasks_S001.xml"
    setup_path = inputs_dir / "Setup_RRA.xml"
    adjusted_model_path = results_dir / f"{tool_name}_adjusted.osim"
    _write_xml(actuator_path, build_actuator_xml(coordinates, pelvis_com, fy_optimal_force=fy_optimal_force))
    _write_xml(task_path, build_task_xml(coordinates, omit_locked_tasks=omit_locked_tasks))
    _write_xml(
        setup_path,
        build_setup_xml(
            tool_name=tool_name,
            model_path=model_path,
            actuator_path=actuator_path,
            results_dir=results_dir,
            external_loads_path=external_loads_path,
            motion_path=motion_path,
            task_path=task_path,
            output_model_path=adjusted_model_path,
            initial_time=effective[0],
            final_time=effective[1],
        ),
    )
    locked = [coordinate.name for coordinate in coordinates if coordinate.locked]
    source_paths = [model_path, motion_path, external_loads_path, external_loads_data_path]
    if qc_path.is_file():
        source_paths.append(qc_path)
    normalization_source: dict[str, Any] | None = None
    if qc_path.is_file():
        qc = json.loads(qc_path.read_text(encoding="utf-8"))
        try:
            normalization = qc["pelvis_residuals"]["normalization"]
            body_weight = float(normalization["body_weight_N"])
            marker_height = float(normalization["marker_height_m"])
            if not all(math.isfinite(value) and value > 0.0 for value in (body_weight, marker_height)):
                raise ValueError
            normalization_source = {
                "path": str(qc_path),
                "sha256": _sha256(qc_path),
                "json_path": "pelvis_residuals.normalization",
            }
        except (KeyError, TypeError, ValueError):
            normalization_source = None
    manifest = {
        "schema_version": _SCHEMA,
        "scope": _SCOPE,
        "official_runtime_required_only_for_run": True,
        "tool_name": tool_name,
        "time_range_s": {"available": available, "requested": requested, "effective": effective},
        "source_inputs": {str(path): {"sha256": _sha256(path)} for path in source_paths},
        "generated_inputs": {str(path): {"sha256": _sha256(path)} for path in (actuator_path, task_path, setup_path)},
        "external_loads_datafile": {
            "external_loads_path": str(external_loads_path),
            "path": str(external_loads_data_path),
            "sha256": _sha256(external_loads_data_path),
        },
        "normalization_source": normalization_source,
        "model": {
            "coordinate_count": len(coordinates),
            "locked_coordinates": locked,
            "pelvis_com_m": pelvis_com,
            "coordinate_force_slots": {
                "pelvis_residual_actuators": sorted(_PELVIS_COORDINATES),
                "coordinate_actuators": [
                    coordinate.name for coordinate in coordinates if coordinate.name not in _PELVIS_COORDINATES
                ],
            },
        },
        "method": {
            "lowpass_fir_order": 50,
            "lowpass_cutoff_hz": 6.0,
            "motion_padding_samples": 60,
            "cmc_target_window_s": 0.001,
            "task_gains": {"kp": 100.0, "kv": 20.0, "ka": 1.0},
            "optimizer": "ipopt",
            "optimizer_tolerance": 1.0e-5,
            "maximum_integrator_step_s": 0.001,
            "integrator_error_tolerance": 1.0e-4,
            "adjusted_com_body": "torso",
            "locked_task_policy": "omit_experiment" if omit_locked_tasks else "exact_upstream_included",
            "fy_optimal_force_N": fy_optimal_force,
            "fy_optimal_force_is_upstream_default": fy_optimal_force == 8.0,
            "mass_recommendation_automatically_applied": False,
        },
        "pinned_upstream": {
            "repository": "https://github.com/opensim-org/opensim-core",
            "commit": _PINNED_COMMIT,
            "files": {path: {"url": f"{_RAW_ROOT}/{path}", "sha256": digest} for path, digest in _SOURCE_FILES.items()},
        },
    }
    manifest_path = output_dir / "prepare_manifest.json"
    _write_json(manifest_path, manifest)
    return PreparedReference(output_dir, setup_path, manifest_path)


def _import_official_opensim() -> Any:
    try:
        import opensim  # noqa: PLC0415
    except ImportError as exception:
        raise RuntimeError("official OpenSim Python bindings are required only for the 'run' command") from exception
    return opensim


def _artifact_hashes(directory: Path, *, excluded: set[Path] | None = None) -> dict[str, str]:
    excluded = {path.resolve() for path in (excluded or set())}
    return {
        str(path.relative_to(directory)): _sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.resolve() not in excluded
    }


def run_reference(output_dir: str | os.PathLike[str]) -> Path:
    """Execute official ``opensim.RRATool`` after verifying prepared inputs."""
    output_dir = Path(output_dir).resolve()
    manifest_path = output_dir / "prepare_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != _SCHEMA or manifest.get("scope") != _SCOPE:
        raise ValueError("output directory is not a prepared official RRA reference")
    _verify_hash_entries(manifest.get("source_inputs"), kind="source input")
    _verify_hash_entries(manifest.get("generated_inputs"), kind="generated input")

    setup_candidates = [Path(path) for path in manifest["generated_inputs"] if Path(path).name == "Setup_RRA.xml"]
    if len(setup_candidates) != 1:
        raise ValueError("prepare manifest does not identify exactly one RRATool setup")
    setup_path = setup_candidates[0]
    if setup_path != output_dir / "inputs" / "Setup_RRA.xml":
        raise ValueError("prepare manifest RRATool setup is outside the expected output location")

    log_path = output_dir / "rratool.log"
    runtime_path = output_dir / "run_runtime.json"
    results_dir = output_dir / "results"
    # A rerun is a new attempt. Remove every previous result and success record
    # before OpenSim starts so a failed rerun cannot inherit stale evidence.
    if results_dir.exists():
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True)
    for stale in (log_path, output_dir / "opensim.log", runtime_path, output_dir / "summary.json"):
        stale.unlink(missing_ok=True)
    for stale in output_dir.glob("delete_this_to_stop_optimization__*.txt"):
        stale.unlink()

    run_id = uuid.uuid4().hex
    manifest_sha256 = _sha256(manifest_path)
    started = time.perf_counter()
    success = False
    error: str | None = None
    opensim: Any | None = None
    tool: Any | None = None
    logger_added = False
    previous_directory = Path.cwd()
    try:
        opensim = _import_official_opensim()
        opensim.Logger.addFileSink(str(log_path))
        logger_added = True
        # RRATool creates its default log and optimization stop sentinel in the
        # process working directory. Keep those generated files out of the repo.
        os.chdir(output_dir)
        tool = opensim.RRATool(str(setup_path))
        result = tool.run()
        success = result is not False
        if not success:
            raise RuntimeError("official RRATool returned false")
    except Exception as exception:
        error = f"{type(exception).__name__}: {exception}"
        raise
    finally:
        elapsed = time.perf_counter() - started
        if tool is not None:
            del tool
            tool = None
            gc.collect()
        if logger_added:
            opensim.Logger.removeFileSink()
        os.chdir(previous_directory)
        artifact_hashes = _artifact_hashes(output_dir, excluded={runtime_path})
        runtime = {
            "schema_version": _SCHEMA,
            "scope": _SCOPE,
            "run_id": run_id,
            "success": success,
            "error": error,
            "wall_time_s": elapsed,
            "opensim_version": opensim.GetVersionAndDate() if opensim is not None else None,
            "python_version": sys.version,
            "platform": platform.platform(),
            "prepare_manifest_sha256": manifest_sha256,
            "setup_sha256": _sha256(setup_path),
            "artifact_linkage": {"run_id": run_id, "root": str(output_dir)},
            "artifacts": artifact_hashes,
            "mass_recommendation_automatically_applied": False,
        }
        _write_json(runtime_path, runtime)
    return runtime_path


def parse_storage(path: str | os.PathLike[str]) -> tuple[list[str], list[list[float]]]:
    """Parse an OpenSim Storage table without OpenSim or NumPy."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    try:
        index = next(i for i, line in enumerate(lines) if line.strip().lower() == "endheader")
    except StopIteration as exception:
        raise ValueError(f"{path} has no endheader") from exception
    if index + 1 >= len(lines):
        raise ValueError(f"{path} has no column labels")
    labels = lines[index + 1].split()
    rows = [[float(value) for value in line.split()] for line in lines[index + 2 :] if line.strip()]
    if not labels or not rows or any(len(row) != len(labels) for row in rows):
        raise ValueError(f"{path} is not a rectangular nonempty Storage table")
    if any(not all(math.isfinite(value) for value in row) for row in rows):
        raise ValueError(f"{path} contains nonfinite values")
    return labels, rows


def parse_average_residuals(text: str) -> dict[str, float]:
    """Parse the official ``*_avgResiduals.txt`` format."""
    values = {
        name: float(value) for name, value in re.findall(r"\b(FX|FY|FZ|MX|MY|MZ)\s+average\s*=\s*([-+0-9.eE]+)", text)
    }
    if set(values) != {"FX", "FY", "FZ", "MX", "MY", "MZ"}:
        raise ValueError("average residual report does not contain all six components")
    return values


def parse_rra_log(text: str) -> dict[str, Any]:
    """Parse official COM and mass recommendations from a captured RRA log."""
    number = r"([-+0-9.eE]+)"
    body_matches = re.findall(r"Body adjusted:\s*([^\s*]+)", text)
    adjustment_matches = re.findall(rf"Mass Center \(COM\) adjustment:\s*dx\s*={number},\s*dz\s*={number}", text)
    location_matches = re.findall(rf"New COM location:\s*~?\[\s*{number}\s*,\s*{number}\s*,\s*{number}\s*\]", text)
    mass_matches = re.findall(rf"Total mass change:\s*{number}", text)
    body_mass_matches = re.findall(
        rf"^\s*\*\s+([A-Za-z0-9_]+):\s*orig mass\s*=\s*{number},\s*new mass\s*=\s*{number}", text, re.MULTILINE
    )
    return {
        "adjusted_body": body_matches[-1] if body_matches else None,
        "com_adjustment_m": (
            {"dx": float(adjustment_matches[-1][0]), "dz": float(adjustment_matches[-1][1])}
            if adjustment_matches
            else None
        ),
        "new_com_location_m": [float(value) for value in location_matches[-1]] if location_matches else None,
        "recommended_total_mass_change_kg": float(mass_matches[-1]) if mass_matches else None,
        "recommended_body_masses_kg": {
            name: {"original": float(original), "recommended": float(recommended)}
            for name, original, recommended in body_mass_matches
        },
        "mass_recommendation_automatically_applied": False,
    }


def _series_metrics(values: list[float]) -> dict[str, float]:
    return {
        "mean": sum(values) / len(values),
        "rms": math.sqrt(sum(value * value for value in values) / len(values)),
        "max_abs": max(abs(value) for value in values),
    }


def _grade(value: float, good: float, okay: float) -> str:
    if value <= good:
        return "GOOD"
    if value <= okay:
        return "OKAY"
    return "BAD"


def _worst(*grades: str) -> str:
    order = {"GOOD": 0, "OKAY": 1, "BAD": 2}
    return max(grades, key=order.__getitem__)


def summarize_residual_table(labels: list[str], rows: list[list[float]]) -> dict[str, Any]:
    """Summarize and grade official residual actuator force histories."""
    columns = {label: index for index, label in enumerate(labels)}
    if not all(name in columns for name in ("FX", "FY", "FZ", "MX", "MY", "MZ")):
        raise ValueError("actuation table does not contain six residual columns")
    components: dict[str, Any] = {}
    for name in ("FX", "FY", "FZ", "MX", "MY", "MZ"):
        metrics = _series_metrics([row[columns[name]] for row in rows])
        if name.startswith("F"):
            grades = {"rms": _grade(metrics["rms"], 5.0, 10.0), "max_abs": _grade(metrics["max_abs"], 10.0, 25.0)}
            unit = "N"
        else:
            grades = {"rms": _grade(metrics["rms"], 30.0, 50.0), "max_abs": _grade(metrics["max_abs"], 50.0, 75.0)}
            unit = "N*m"
        components[name] = {**metrics, "unit": unit, "grades": {**grades, "overall": _worst(*grades.values())}}
    return components


def summarize_perr(
    labels: list[str],
    rows: list[list[float]],
    coordinates: list[CoordinateSpec],
    *,
    locked_task_policy: str = "exact_upstream_included",
) -> dict[str, Any]:
    """Summarize pErr using translation metres and rotation degrees.

    Exact-upstream runs must report every task, including locked-coordinate
    diagnostics. Only the explicit omission experiment may lack locked columns.
    """
    if locked_task_policy not in {"exact_upstream_included", "omit_experiment"}:
        raise ValueError(f"unsupported locked task policy: {locked_task_policy!r}")
    columns = {label: index for index, label in enumerate(labels)}
    result: dict[str, Any] = {}
    for coordinate in coordinates:
        if coordinate.name not in columns:
            if coordinate.locked and locked_task_policy == "omit_experiment":
                continue
            raise ValueError(f"pErr table is missing required task {coordinate.name}")
        metrics = _series_metrics([row[columns[coordinate.name]] for row in rows])
        if coordinate.motion_type == "rotational":
            metrics = {name: math.degrees(value) for name, value in metrics.items()}
            grades = {"rms": _grade(metrics["rms"], 2.0, 5.0), "max_abs": _grade(metrics["max_abs"], 2.0, 5.0)}
            unit = "deg"
        else:
            grades = {"rms": _grade(metrics["rms"], 0.02, 0.04), "max_abs": _grade(metrics["max_abs"], 0.02, 0.05)}
            unit = "m"
        result[coordinate.name] = {
            **metrics,
            "unit": unit,
            "locked_coordinate_diagnostic": coordinate.locked,
            "included_in_no_bad_perr_gate": not coordinate.locked,
            "grades": {**grades, "overall": _worst(*grades.values())},
        }
    return result


def _mass_map(path: Path) -> dict[str, float]:
    root = ET.parse(path).getroot()
    return {
        body.get("name", ""): float(_text(body, "mass"))
        for body in root.iter("Body")
        if body.get("name") and _text(body, "mass")
    }


def summarize_reference(output_dir: str | os.PathLike[str]) -> Path:
    """Parse official RRA outputs and write deterministic scientific QC JSON."""
    output_dir = Path(output_dir).resolve()
    manifest_path = output_dir / "prepare_manifest.json"
    runtime_path = output_dir / "run_runtime.json"
    summary_path = output_dir / "summary.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != _SCHEMA or manifest.get("scope") != _SCOPE:
        raise ValueError("output directory is not a prepared official RRA reference")
    _verify_hash_entries(manifest.get("source_inputs"), kind="source input")
    _verify_hash_entries(manifest.get("generated_inputs"), kind="generated input")

    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    if runtime.get("schema_version") != _SCHEMA or runtime.get("scope") != _SCOPE:
        raise ValueError("runtime does not belong to this official RRA reference")
    if runtime.get("success") is not True:
        raise ValueError("the current official RRA runtime did not succeed")
    run_id = runtime.get("run_id")
    if not isinstance(run_id, str) or not re.fullmatch(r"[0-9a-f]{32}", run_id):
        raise ValueError("runtime has no valid run id")
    if runtime.get("artifact_linkage", {}).get("run_id") != run_id:
        raise ValueError("runtime artifact linkage does not match its run id")
    if runtime.get("prepare_manifest_sha256") != _sha256(manifest_path):
        raise ValueError("runtime was not produced from the current prepare manifest")
    expected_artifacts = runtime.get("artifacts")
    if not isinstance(expected_artifacts, dict):
        raise ValueError("runtime has no artifact hash set")
    current_artifacts = _artifact_hashes(output_dir, excluded={runtime_path, summary_path})
    if current_artifacts != expected_artifacts:
        raise ValueError("current official RRA artifacts do not match the successful runtime")

    tool_name = manifest["tool_name"]
    results_dir = output_dir / "results"

    force_candidates = sorted(results_dir.glob(f"{tool_name}*_Actuation_force.sto"))
    perr_candidates = sorted(results_dir.glob(f"{tool_name}*_pErr.sto"))
    average_candidates = sorted(results_dir.glob(f"{tool_name}*_avgResiduals.txt"))
    if len(force_candidates) != 1 or len(perr_candidates) != 1 or len(average_candidates) != 1:
        raise FileNotFoundError("expected exactly one Actuation_force, pErr, and avgResiduals result")
    force_labels, force_rows = parse_storage(force_candidates[0])
    perr_labels, perr_rows = parse_storage(perr_candidates[0])
    components = summarize_residual_table(force_labels, force_rows)
    model_candidates = [Path(path) for path in manifest["source_inputs"] if Path(path).name == "S001_scaled.osim"]
    if len(model_candidates) != 1:
        raise ValueError("prepare manifest does not identify exactly one source model")
    model_path = model_candidates[0]
    coordinates, _pelvis_com = parse_model_spec(model_path)
    locked_task_policy = manifest.get("method", {}).get("locked_task_policy")
    perr = summarize_perr(perr_labels, perr_rows, coordinates, locked_task_policy=locked_task_policy)
    averages = parse_average_residuals(average_candidates[0].read_text(encoding="utf-8"))
    recommendations = parse_rra_log((output_dir / "rratool.log").read_text(encoding="utf-8", errors="replace"))

    adjusted_model = results_dir / f"{tool_name}_adjusted.osim"
    original_masses = _mass_map(model_path)
    adjusted_masses = _mass_map(adjusted_model)
    mass_changes = {
        name: adjusted_masses[name] - value
        for name, value in original_masses.items()
        if name in adjusted_masses and not math.isclose(adjusted_masses[name], value, rel_tol=0.0, abs_tol=1.0e-12)
    }

    normalized: dict[str, Any] | None = None
    normalization_source = manifest.get("normalization_source")
    if normalization_source is not None:
        if not isinstance(normalization_source, dict):
            raise ValueError("prepare manifest normalization source is invalid")
        qc_path = Path(normalization_source.get("path", ""))
        source_metadata = manifest["source_inputs"].get(str(qc_path))
        expected_qc_hash = normalization_source.get("sha256")
        if (
            normalization_source.get("json_path") != "pelvis_residuals.normalization"
            or not isinstance(source_metadata, dict)
            or source_metadata.get("sha256") != expected_qc_hash
            or _sha256(qc_path) != expected_qc_hash
        ):
            raise ValueError("normalization source does not match the verified QC source hash")
        qc = json.loads(qc_path.read_text(encoding="utf-8"))
        try:
            body_weight = float(qc["pelvis_residuals"]["normalization"]["body_weight_N"])
            height = float(qc["pelvis_residuals"]["normalization"]["marker_height_m"])
            columns = {label: index for index, label in enumerate(force_labels)}
            force_norms = [math.sqrt(sum(row[columns[name]] ** 2 for name in ("FX", "FY", "FZ"))) for row in force_rows]
            moment_norms = [
                math.sqrt(sum(row[columns[name]] ** 2 for name in ("MX", "MY", "MZ"))) for row in force_rows
            ]
            force_rms = math.sqrt(sum(value * value for value in force_norms) / len(force_norms)) / body_weight
            force_peak = max(force_norms) / body_weight
            moment_scale = body_weight * height
            moment_rms = math.sqrt(sum(value * value for value in moment_norms) / len(moment_norms)) / moment_scale
            moment_peak = max(moment_norms) / moment_scale
            normalized = {
                "force": {
                    "rms_fraction_bw": force_rms,
                    "peak_fraction_bw": force_peak,
                    "passed": force_rms < 0.10 and force_peak < 0.25,
                },
                "moment": {
                    "rms_fraction_bw_height": moment_rms,
                    "peak_fraction_bw_height": moment_peak,
                    "passed": moment_rms < 0.05 and moment_peak < 0.10,
                },
            }
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            normalized = None

    no_bad_component = all(item["grades"]["overall"] != "BAD" for item in components.values())
    no_bad_perr = all(
        item["grades"]["overall"] != "BAD" for item in perr.values() if item["included_in_no_bad_perr_gate"]
    )
    summary = {
        "schema_version": _SCHEMA,
        "scope": _SCOPE,
        "official_reference_only": True,
        "time_range_s": manifest["time_range_s"]["effective"],
        "average_residuals": averages,
        "thresholds": {
            "residual_force_N": {
                "maximum": {"GOOD": [0.0, 10.0], "OKAY": [10.0, 25.0], "BAD_above": 25.0},
                "rms": {"GOOD": [0.0, 5.0], "OKAY": [5.0, 10.0], "BAD_above": 10.0},
            },
            "residual_moment_Nm": {
                "maximum": {"GOOD": [0.0, 50.0], "OKAY": [50.0, 75.0], "BAD_above": 75.0},
                "rms": {"GOOD": [0.0, 30.0], "OKAY": [30.0, 50.0], "BAD_above": 50.0},
            },
            "perr_translation_m": {"GOOD_rms_and_max_below": 0.02, "OKAY_rms_at_most": 0.04, "OKAY_max_at_most": 0.05},
            "perr_rotation_deg": {"GOOD_rms_and_max_below": 2.0, "OKAY_rms_and_max_at_most": 5.0},
            "normalized_resultants": {
                "force_rms_fraction_bw_below": 0.10,
                "force_peak_fraction_bw_below": 0.25,
                "moment_rms_fraction_bw_height_below": 0.05,
                "moment_peak_fraction_bw_height_below": 0.10,
            },
        },
        "residual_components": components,
        "perr": perr,
        "normalized_resultants": normalized,
        "anthropometry": {
            **recommendations,
            "source_to_adjusted_model_mass_changes_kg": mass_changes,
            "silent_mass_application_detected": bool(mass_changes),
        },
        "gates": {
            "runtime_success": runtime["success"],
            "no_bad_residual_component": no_bad_component,
            "no_bad_perr_coordinate": no_bad_perr,
            "perr_coordinates_excluded_as_locked_diagnostics": [
                name for name, item in perr.items() if not item["included_in_no_bad_perr_gate"]
            ],
            "no_silent_mass_application": not mass_changes,
            "normalized_resultants_passed": normalized is not None
            and normalized["force"]["passed"]
            and normalized["moment"]["passed"],
            "production_candidate": runtime["success"]
            and no_bad_component
            and no_bad_perr
            and not mass_changes
            and normalized is not None
            and normalized["force"]["passed"]
            and normalized["moment"]["passed"],
            "okay_components_require_explicit_review": [
                name for name, item in components.items() if item["grades"]["overall"] == "OKAY"
            ],
        },
        "runtime": runtime,
        "runtime_linkage": {"run_id": run_id, "runtime_sha256": _sha256(runtime_path)},
        "artifacts": _artifact_hashes(output_dir, excluded={summary_path}),
    }
    _write_json(summary_path, summary)
    return summary_path


def create_parser() -> argparse.ArgumentParser:
    """Create the official RRA reference command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="Generate deterministic RRATool XML without importing OpenSim.")
    prepare.add_argument("--data-dir", default=str(_DEFAULT_DATA))
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--initial-time", type=float)
    prepare.add_argument("--final-time", type=float)
    prepare.add_argument("--tool-name", default="trial101_official_opensim_rra_reference")
    prepare.add_argument(
        "--omit-locked-tasks",
        action="store_true",
        help="Experimental only: omit locked-coordinate tasks; official parity includes them.",
    )
    prepare.add_argument(
        "--fy-optimal-force",
        type=float,
        default=8.0,
        help="FY residual optimal force [N]; upstream is 8, accepted Trial probe used 4.",
    )
    run = subparsers.add_parser("run", help="Run the prepared setup with official OpenSim.")
    run.add_argument("--output-dir", required=True)
    summarize = subparsers.add_parser("summarize", help="Parse outputs and write scientific QC.")
    summarize.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the selected CLI command."""
    args = create_parser().parse_args(argv)
    if args.command == "prepare":
        prepared = prepare_reference(
            args.data_dir,
            args.output_dir,
            initial_time=args.initial_time,
            final_time=args.final_time,
            tool_name=args.tool_name,
            omit_locked_tasks=args.omit_locked_tasks,
            fy_optimal_force=args.fy_optimal_force,
        )
        print(prepared.manifest_path)
    elif args.command == "run":
        print(run_reference(args.output_dir))
    else:
        print(summarize_reference(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
