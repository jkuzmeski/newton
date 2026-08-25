# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Adapter for the official OpenSim ``MocoInverse`` walking example.

The input to this adapter is an *accepted* output directory from
:mod:`projects.gait_c3d.opensim_rra_reference`.  MocoInverse prescribes the
RRA-adjusted motion; it is therefore an inverse reference, not a Newton-native
prediction and not a forward-dynamics validation.

Preparing and summarizing do not import OpenSim.  The optional official Python
bindings are imported lazily by :func:`run_reference` only.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import uuid
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from projects.gait_c3d import rra_adjusted_contact_input as _rra_contact

ARCHITECTURE_ROLE = "official_oracle"

_SCHEMA = "gait_c3d_official_opensim_moco_inverse_reference_1"
_SCOPE = "official_opensim_moco_inverse_prescribed_rra_motion_not_predictive_forward_dynamics"
_RRA_SCHEMA = "gait_c3d_official_opensim_rra_reference_1"
_PINNED_COMMIT = "11036b39ca7232c604685b37f483afafc056d92b"
_EXAMPLE_PATH = "Bindings/Python/examples/Moco/example3DWalking/exampleMocoInverse.py"
_EXAMPLE_SHA256 = "c86eb4961d894ea5fec67a153491c85908a70cb06a30a746e387ba3f6917de46"
_EXAMPLE_URL = f"https://raw.githubusercontent.com/opensim-org/opensim-core/{_PINNED_COMMIT}/{_EXAMPLE_PATH}"
_WELDED_JOINTS = ("mtp_r", "mtp_l")
_DEFAULT_MESH_INTERVAL = 0.02
_DEFAULT_MAX_ITERATIONS = 1000
_MUSCLE_OUTPUT_PATHS = (
    ".*activation",
    ".*normalized_fiber_length",
    ".*normalized_tendon_force",
    ".*active_force_length_multiplier",
    ".*passive_force_multiplier",
)


@dataclass(frozen=True)
class CoordinateInfo:
    """The information needed to turn a legacy coordinate label into a state path."""

    coordinate: str
    joint: str
    rotational: bool

    @property
    def state_path(self) -> str:
        return f"/jointset/{self.joint}/{self.coordinate}/value"


@dataclass(frozen=True)
class PreparedReference:
    """Files created by :func:`prepare_reference`."""

    output_dir: Path
    manifest_path: Path
    config_path: Path
    kinematics_path: Path


def _sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _artifact_hashes(root: Path, *, excluded: Iterable[Path] = ()) -> dict[str, str]:
    excluded_resolved = {path.resolve() for path in excluded}
    return {
        str(path.relative_to(root)): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.resolve() not in excluded_resolved
    }


def _verify_hash_mapping(entries: Any, *, description: str) -> None:
    if not isinstance(entries, dict) or not entries:
        raise ValueError(f"manifest has no {description} hashes")
    for raw_path, metadata in entries.items():
        if not isinstance(raw_path, str) or not isinstance(metadata, dict):
            raise ValueError(f"invalid {description} hash entry")
        expected = metadata.get("sha256")
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError(f"invalid {description} SHA-256 for {raw_path}")
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"{description} hash changed for {path}: expected {expected}, got {actual}")


def _resolve_rra_root(path: str | os.PathLike[str]) -> Path:
    candidate = Path(path).expanduser().resolve()
    if candidate.is_file():
        if candidate.name != "summary.json":
            raise ValueError("RRA input file must be summary.json")
        candidate = candidate.parent
    if candidate.name == "results" and (candidate.parent / "summary.json").is_file():
        candidate = candidate.parent
    if not candidate.is_dir():
        raise FileNotFoundError(candidate)
    return candidate


def _tag_name(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _external_loads_from_verified_prepare(prepare: dict[str, Any]) -> tuple[Path, Path]:
    """Select one ExternalLoads XML and its verified data file.

    This function must only be called after the shared strict RRA verifier has
    checked every source and generated hash in ``prepare``.
    """
    entries = prepare.get("source_inputs")
    if not isinstance(entries, dict):
        raise ValueError("RRA prepare manifest source inputs are invalid")
    candidates: list[tuple[Path, ET.Element]] = []
    for raw_path in entries:
        if not isinstance(raw_path, str):
            raise ValueError("RRA prepare manifest source input path is invalid")
        path = Path(raw_path).expanduser().resolve()
        if path.suffix.lower() != ".xml":
            continue
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError as exception:
            raise ValueError(f"RRA source XML is malformed: {path}") from exception
        nodes = [node for node in root.iter() if _tag_name(node) == "ExternalLoads"]
        if len(nodes) > 1:
            raise ValueError(f"RRA source XML contains multiple ExternalLoads objects: {path}")
        if nodes:
            candidates.append((path, nodes[0]))
    if len(candidates) != 1:
        raise ValueError(
            f"accepted RRA prepare inputs must contain exactly one ExternalLoads XML; found {len(candidates)}"
        )

    external_loads_path, external_loads = candidates[0]
    data_nodes = [node for node in external_loads.iter() if _tag_name(node).lower() == "datafile"]
    if len(data_nodes) != 1 or not (data_nodes[0].text or "").strip():
        raise ValueError("accepted RRA ExternalLoads XML must contain exactly one datafile")
    data_path = Path((data_nodes[0].text or "").strip()).expanduser()
    if not data_path.is_absolute():
        data_path = external_loads_path.parent / data_path
    data_path = data_path.resolve()
    matching_inputs = [
        raw_path
        for raw_path in entries
        if isinstance(raw_path, str) and Path(raw_path).expanduser().resolve() == data_path
    ]
    if len(matching_inputs) != 1:
        raise ValueError("RRA ExternalLoads datafile is not exactly one verified prepare source input")
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    return external_loads_path, data_path


def _validate_accepted_rra(rra_root: Path) -> dict[str, Path]:
    # Reuse the acceptance boundary used by the RRA-adjusted contact adapter.
    # It verifies schema/scope, all six gates, exact summary/runtime equality,
    # run linkage, prepare linkage, complete artifact maps (including deferred
    # products), and every source/generated prepare hash.
    _summary, _runtime, prepare = _rra_contact._verify_rra_reference(rra_root)
    products = _rra_contact._paths_for_rra_products(rra_root, prepare)
    external_loads_path, external_loads_data_path = _external_loads_from_verified_prepare(prepare)
    return {
        "summary_path": rra_root / "summary.json",
        "manifest_path": rra_root / "prepare_manifest.json",
        "runtime_path": rra_root / "run_runtime.json",
        "model_path": products["model"],
        "kinematics_path": products["q"],
        "external_loads_path": external_loads_path,
        "external_loads_data_path": external_loads_data_path,
    }


def _coordinate_rotation_map(joint: ET.Element) -> dict[str, bool]:
    result: dict[str, bool] = {}
    spatial = joint.find("SpatialTransform")
    if spatial is not None:
        rotational_names: set[str] = set()
        translational_names: set[str] = set()
        for axis in spatial.findall("TransformAxis"):
            axis_name = (axis.get("name") or "").lower()
            names = set((axis.findtext("coordinates") or "").split())
            if axis_name.startswith("rotation"):
                rotational_names.update(names)
            elif axis_name.startswith("translation"):
                translational_names.update(names)
        # A rotational coordinate can also drive coupled translation (the gait
        # knee is the canonical example). Rotation therefore takes precedence.
        result.update(dict.fromkeys(translational_names - rotational_names, False))
        result.update(dict.fromkeys(rotational_names, True))
    tag = joint.tag.rsplit("}", 1)[-1]
    if tag in {"PinJoint", "BallJoint", "EllipsoidJoint"}:
        default: bool | None = True
    elif tag in {"SliderJoint"}:
        default = False
    else:
        default = None
    for coordinate in joint.findall("./coordinates/Coordinate"):
        name = (coordinate.get("name") or "").strip()
        motion_type = (coordinate.findtext("motion_type") or "").strip().lower()
        if motion_type in {"rotational", "translational"}:
            result[name] = motion_type == "rotational"
        elif name not in result and default is not None:
            result[name] = default
    return result


def coordinate_info_from_model(path: str | os.PathLike[str]) -> dict[str, CoordinateInfo]:
    """Read coordinate-to-joint paths without importing OpenSim.

    ``CustomJoint`` coordinates are classified through their SpatialTransform.
    This avoids the unsafe shortcut of converting pelvis translations from
    degrees to radians.
    """
    root = ET.parse(path).getroot()
    result: dict[str, CoordinateInfo] = {}
    for joint in root.iter():
        coordinates = joint.findall("./coordinates/Coordinate")
        if not coordinates:
            continue
        joint_name = (joint.get("name") or "").strip()
        rotations = _coordinate_rotation_map(joint)
        if not joint_name:
            raise ValueError("model contains an unnamed joint")
        for coordinate in coordinates:
            name = (coordinate.get("name") or "").strip()
            if not name or name in result:
                raise ValueError(f"model contains an invalid or duplicate coordinate {name!r}")
            if name not in rotations:
                raise ValueError(f"cannot classify coordinate {name!r} in joint {joint_name!r}")
            result[name] = CoordinateInfo(name, joint_name, rotations[name])
    if not result:
        raise ValueError("model contains no coordinates")
    return result


def parse_storage(path: str | os.PathLike[str]) -> tuple[dict[str, str], list[str], list[list[float]]]:
    """Parse a numeric OpenSim Storage file without importing OpenSim."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        end = next(index for index, line in enumerate(lines) if line.strip().lower() == "endheader")
    except StopIteration as exception:
        raise ValueError(f"{path} has no endheader") from exception
    metadata: dict[str, str] = {}
    for line in lines[:end]:
        if "=" in line:
            key, value = line.split("=", 1)
            metadata[key.strip().lower()] = value.strip()
    data_lines = [line for line in lines[end + 1 :] if line.strip()]
    if not data_lines:
        raise ValueError(f"{path} has no column labels")
    labels = data_lines[0].split()
    if not labels or labels[0].lower() != "time" or len(set(labels)) != len(labels):
        raise ValueError(f"{path} has invalid or duplicate column labels")
    rows: list[list[float]] = []
    for line_number, line in enumerate(data_lines[1:], start=end + 3):
        fields = line.split()
        if len(fields) != len(labels):
            raise ValueError(f"{path}:{line_number} has {len(fields)} fields; expected {len(labels)}")
        row = [float(field) for field in fields]
        if not all(math.isfinite(value) for value in row):
            raise ValueError(f"{path}:{line_number} contains a non-finite value")
        rows.append(row)
    if not rows:
        raise ValueError(f"{path} has no data rows")
    if any(right[0] <= left[0] for left, right in itertools.pairwise(rows)):
        raise ValueError(f"{path} time values must increase strictly")
    return metadata, labels, rows


def _write_storage(path: Path, labels: Sequence[str], rows: Sequence[Sequence[float]], *, name: str) -> None:
    lines = [
        name,
        "version=1",
        f"nRows={len(rows)}",
        f"nColumns={len(labels)}",
        "inDegrees=no",
        "endheader",
        "\t".join(labels),
    ]
    lines.extend("\t".join(format(float(value), ".17g") for value in row) for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def convert_rra_kinematics(
    source_path: str | os.PathLike[str],
    model_path: str | os.PathLike[str],
    destination_path: str | os.PathLike[str],
    *,
    welded_joints: Sequence[str] = _WELDED_JOINTS,
) -> Path:
    """Convert RRA ``Kinematics_q`` into deterministic Moco state paths.

    RRA emits legacy short labels and marks its rotational values as degrees.
    Rotations are converted to radians, translations retain their length units,
    and coordinates belonging to welded MTP joints are omitted.  Omitting the
    MTP columns is intentional: MocoInverse otherwise sees states that no longer
    exist after ``ModOpReplaceJointsWithWelds``.
    """
    metadata, labels, rows = parse_storage(source_path)
    infos = coordinate_info_from_model(model_path)
    if metadata.get("indegrees", "").lower() not in {"yes", "true"}:
        raise ValueError("RRA Kinematics_q must declare inDegrees=yes")
    welded = set(welded_joints)
    source_labels = labels[1:]
    if any(label.startswith("/") for label in source_labels):
        raise ValueError("RRA Kinematics_q must use legacy short coordinate labels")
    unknown = sorted(set(source_labels) - set(infos))
    duplicate = sorted({label for label in source_labels if source_labels.count(label) > 1})
    missing = sorted(set(infos) - set(source_labels))
    if unknown or duplicate or missing:
        raise ValueError(
            f"RRA coordinate labels do not match model (unknown={unknown}, duplicate={duplicate}, missing={missing})"
        )

    keep_indices = [index for index, label in enumerate(source_labels, start=1) if infos[label].joint not in welded]
    output_labels = ["time"] + [infos[labels[index]].state_path for index in keep_indices]
    output_rows: list[list[float]] = []
    for row in rows:
        converted = [row[0]]
        for index in keep_indices:
            value = row[index]
            if infos[labels[index]].rotational:
                value = math.radians(value)
            converted.append(value)
        output_rows.append(converted)
    destination = Path(destination_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_storage(destination, output_labels, output_rows, name="RRA kinematics for official OpenSim MocoInverse")
    return destination


# A descriptive alias makes the conversion API easy to find at call sites.
convert_rra_kinematics_to_moco = convert_rra_kinematics


def prepare_reference(
    rra_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    mesh_interval: float = _DEFAULT_MESH_INTERVAL,
    max_iterations: int = _DEFAULT_MAX_ITERATIONS,
    guess_file: str | os.PathLike[str] | None = None,
) -> PreparedReference:
    """Prepare a sealed official MocoInverse run from an accepted RRA result."""
    if not math.isfinite(mesh_interval) or mesh_interval <= 0:
        raise ValueError("mesh interval must be positive and finite")
    if isinstance(max_iterations, bool) or max_iterations < 0:
        raise ValueError("max iterations must be a nonnegative integer")
    rra_root = _resolve_rra_root(rra_path)
    sources = _validate_accepted_rra(rra_root)
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir == rra_root or output_dir.is_relative_to(rra_root) or rra_root.is_relative_to(output_dir):
        raise ValueError("MocoInverse output and accepted RRA directories must not overlap")
    inputs_dir = output_dir / "inputs"
    results_dir = output_dir / "results"
    for directory in (inputs_dir, results_dir):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)
    for stale in (
        output_dir / "run_runtime.json",
        output_dir / "summary.json",
        output_dir / "moco.log",
        output_dir / "child_process.json",
    ):
        stale.unlink(missing_ok=True)

    guess_path: Path | None = None
    if guess_file is not None:
        guess_path = Path(guess_file).expanduser().resolve()
        if not guess_path.is_file():
            raise FileNotFoundError(guess_path)

    kinematics_path = inputs_dir / "rra_kinematics_moco.sto"
    convert_rra_kinematics(sources["kinematics_path"], sources["model_path"], kinematics_path)
    _, _, kinematics_rows = parse_storage(kinematics_path)
    config_path = inputs_dir / "moco_inverse_config.json"
    config = {
        "schema_version": _SCHEMA,
        "scope": _SCOPE,
        "model_path": str(sources["model_path"]),
        "external_loads_path": str(sources["external_loads_path"]),
        "kinematics_path": str(kinematics_path),
        "initial_time_s": kinematics_rows[0][0],
        "final_time_s": kinematics_rows[-1][0],
        "mesh_interval_s": float(mesh_interval),
        "max_iterations": int(max_iterations),
        "guess_file": str(guess_path) if guess_path else None,
        "model_processor_operations": [
            {"operation": "ModOpAddExternalLoads", "arguments": [str(sources["external_loads_path"])]},
            {"operation": "ModOpReplaceJointsWithWelds", "arguments": list(_WELDED_JOINTS)},
            {"operation": "ModOpAddResiduals", "arguments": [250.0, 50.0, 1.0]},
            {"operation": "ModOpIgnoreTendonCompliance", "arguments": []},
            {"operation": "ModOpReplaceMusclesWithDeGrooteFregly2016", "arguments": []},
            {"operation": "ModOpIgnorePassiveFiberForcesDGF", "arguments": []},
            {"operation": "ModOpScaleActiveFiberForceCurveWidthDGF", "arguments": [1.5]},
            {"operation": "ModOpAddReserves", "arguments": [1.0]},
        ],
        "prescribed_motion_scope": {
            "mechanism": "MocoInverse PositionMotion created from the converted RRA generalized coordinates",
            "prescribed": "all non-MTP generalized coordinate values in rra_kinematics_moco.sto",
            "optimized": "muscle/tendon states and actuator controls compatible with that prescribed motion",
            "not_claimed": ["predictive motion", "forward dynamics", "Newton-native result"],
        },
    }
    _write_json(config_path, config)

    source_paths = list(sources.values())
    if guess_path:
        source_paths.append(guess_path)
    manifest = {
        "schema_version": _SCHEMA,
        "scope": _SCOPE,
        "official_runtime_required_only_for_run": True,
        "accepted_rra_root": str(rra_root),
        "source_inputs": {str(path): {"sha256": _sha256(path)} for path in source_paths},
        "generated_inputs": {str(path): {"sha256": _sha256(path)} for path in (kinematics_path, config_path)},
        "configuration": config,
        "pinned_upstream": {
            "repository": "https://github.com/opensim-org/opensim-core",
            "commit": _PINNED_COMMIT,
            "files": {_EXAMPLE_PATH: {"url": _EXAMPLE_URL, "sha256": _EXAMPLE_SHA256}},
        },
    }
    manifest_path = output_dir / "prepare_manifest.json"
    _write_json(manifest_path, manifest)
    return PreparedReference(output_dir, manifest_path, config_path, kinematics_path)


def _import_official_opensim() -> Any:
    try:
        import opensim  # noqa: PLC0415
    except ImportError as exception:
        raise RuntimeError("official OpenSim Python bindings are required only for the 'run' command") from exception
    return opensim


def build_model_processor(opensim: Any, config: dict[str, Any]) -> Any:
    """Build the pinned example ModelProcessor. Kept separate for pure tests."""
    processor = opensim.ModelProcessor(config["model_path"])
    processor.append(opensim.ModOpAddExternalLoads(config["external_loads_path"]))
    joints = opensim.StdVectorString()
    for name in _WELDED_JOINTS:
        joints.append(name)
    processor.append(opensim.ModOpReplaceJointsWithWelds(joints))
    processor.append(opensim.ModOpAddResiduals(250.0, 50.0, 1.0))
    processor.append(opensim.ModOpIgnoreTendonCompliance())
    processor.append(opensim.ModOpReplaceMusclesWithDeGrooteFregly2016())
    processor.append(opensim.ModOpIgnorePassiveFiberForcesDGF())
    processor.append(opensim.ModOpScaleActiveFiberForceCurveWidthDGF(1.5))
    processor.append(opensim.ModOpAddReserves(1.0))
    return processor


def _solution_success(solution: Any) -> bool:
    method = getattr(solution, "success", None)
    return bool(method()) if callable(method) else False


def write_solution_or_failed_guess(solution: Any, solution_path: Path, failed_guess_path: Path) -> tuple[Path, bool]:
    """Write a solution, unsealing a failed solve into a reusable guess.

    Official Moco seals unsuccessful solutions. Calling ``write()`` directly on
    such a result raises and loses the most useful diagnostic artifact. This
    helper deliberately unseals only the failed result and labels it as a guess.
    """
    if _solution_success(solution):
        solution.write(str(solution_path))
        return solution_path, True
    unseal = getattr(solution, "unseal", None)
    if not callable(unseal):
        raise RuntimeError("failed Moco solution is sealed and cannot be unsealed")
    unsealed = unseal()
    writable = unsealed if unsealed is not None and hasattr(unsealed, "write") else solution
    writable.write(str(failed_guess_path))
    return failed_guess_path, False


def _write_reserve_controls(solution_file: Path, destination: Path) -> bool:
    metadata, labels, rows = parse_storage(solution_file)
    del metadata
    indices = [index for index, label in enumerate(labels) if index == 0 or re.search(r"(?:^|/)reserve_[^/]*$", label)]
    if len(indices) == 1:
        return False
    _write_storage(
        destination,
        [labels[index] for index in indices],
        [[row[index] for index in indices] for row in rows],
        name="MocoInverse reserve controls",
    )
    return True


def _run_reference_in_process(output_dir: str | os.PathLike[str]) -> Path:
    output_dir = Path(output_dir).expanduser().resolve()
    manifest_path = output_dir / "prepare_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != _SCHEMA or manifest.get("scope") != _SCOPE:
        raise ValueError("output directory is not a prepared official MocoInverse reference")
    _verify_hash_mapping(manifest.get("source_inputs"), description="source input")
    _verify_hash_mapping(manifest.get("generated_inputs"), description="generated input")
    config = manifest["configuration"]
    results_dir = output_dir / "results"
    if results_dir.exists():
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True)
    log_path = output_dir / "moco.log"
    runtime_path = output_dir / "run_runtime.json"
    for stale in (log_path, runtime_path, output_dir / "summary.json"):
        stale.unlink(missing_ok=True)

    started = time.perf_counter()
    run_id = uuid.uuid4().hex
    opensim: Any | None = None
    logger_added = False
    success = False
    sealed_failure_captured = False
    error: str | None = None
    result_path: Path | None = None
    muscle_outputs_path: Path | None = None
    reserve_controls_path: Path | None = None
    try:
        opensim = _import_official_opensim()
        opensim.Logger.addFileSink(str(log_path))
        logger_added = True
        processor = build_model_processor(opensim, config)
        inverse = opensim.MocoInverse()
        inverse.setModel(processor)
        inverse.setKinematics(opensim.TableProcessor(config["kinematics_path"]))
        inverse.set_initial_time(float(config["initial_time_s"]))
        inverse.set_final_time(float(config["final_time_s"]))
        inverse.set_mesh_interval(float(config["mesh_interval_s"]))
        inverse.set_max_iterations(int(config["max_iterations"]))
        inverse.set_kinematics_allow_extra_columns(False)
        study = inverse.initialize()
        solver = opensim.MocoCasADiSolver.safeDownCast(study.updSolver())
        if solver is None:
            raise RuntimeError("MocoInverse did not initialize a MocoCasADiSolver")
        if config.get("guess_file"):
            solver.setGuessFile(config["guess_file"])
        solution = study.solve()
        result_path, success = write_solution_or_failed_guess(
            solution,
            results_dir / "moco_inverse_solution.sto",
            results_dir / "moco_inverse_failed_guess.sto",
        )
        sealed_failure_captured = not success
        if success:
            reserve_candidate = results_dir / "reserve_controls.sto"
            if _write_reserve_controls(result_path, reserve_candidate):
                reserve_controls_path = reserve_candidate
            paths = opensim.StdVectorString()
            for pattern in _MUSCLE_OUTPUT_PATHS:
                paths.append(pattern)
            outputs = study.analyze(solution, paths)
            muscle_outputs_path = results_dir / "muscle_outputs.sto"
            opensim.STOFileAdapter.write(outputs, str(muscle_outputs_path))
        else:
            error = "official Moco solver returned a sealed unsuccessful solution; captured as failed guess"
    except Exception as exception:  # Runtime record must survive official/SWIG failures.
        error = f"{type(exception).__name__}: {exception}"
        success = False
    finally:
        if logger_added:
            try:
                opensim.Logger.removeFileSink()
            except Exception:
                pass
        runtime = {
            "schema_version": _SCHEMA,
            "scope": _SCOPE,
            "run_id": run_id,
            "success": success,
            "sealed_failure_captured_as_guess": sealed_failure_captured,
            "error": error,
            "wall_time_s": time.perf_counter() - started,
            "opensim_version": opensim.GetVersionAndDate() if opensim is not None else None,
            "python_version": sys.version,
            "platform": platform.platform(),
            "prepare_manifest_sha256": _sha256(manifest_path),
            "configuration_sha256": _sha256(Path(config["kinematics_path"]).parent / "moco_inverse_config.json"),
            "artifact_linkage": {"run_id": run_id, "root": str(output_dir)},
            "result_path": str(result_path.relative_to(output_dir)) if result_path else None,
            "reserve_controls_path": str(reserve_controls_path.relative_to(output_dir))
            if reserve_controls_path
            else None,
            "muscle_outputs_path": str(muscle_outputs_path.relative_to(output_dir)) if muscle_outputs_path else None,
            "artifacts": _artifact_hashes(output_dir, excluded={runtime_path}),
            "prescribed_motion_scope": config["prescribed_motion_scope"],
        }
        _write_json(runtime_path, runtime)
    return runtime_path


def run_reference(output_dir: str | os.PathLike[str], *, child_process: bool = False) -> Path:
    """Run MocoInverse, optionally isolating the official runtime in a child."""
    output_dir = Path(output_dir).expanduser().resolve()
    if not child_process:
        return _run_reference_in_process(output_dir)
    lifecycle_path = output_dir / "child_process.json"
    command = [sys.executable, str(Path(__file__).resolve()), "_run-child", str(output_dir)]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    _write_json(
        lifecycle_path,
        {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        },
    )
    runtime_path = output_dir / "run_runtime.json"
    if not runtime_path.is_file():
        raise RuntimeError(
            f"MocoInverse child exited {completed.returncode} without a runtime record; see {lifecycle_path}"
        )
    return runtime_path


def summarize_reference(output_dir: str | os.PathLike[str]) -> Path:
    """Create a hash-linked summary for either a solution or a failed guess."""
    output_dir = Path(output_dir).expanduser().resolve()
    manifest_path = output_dir / "prepare_manifest.json"
    runtime_path = output_dir / "run_runtime.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != _SCHEMA or runtime.get("schema_version") != _SCHEMA:
        raise ValueError("prepared and runtime schemas do not match official MocoInverse")
    if runtime.get("prepare_manifest_sha256") != _sha256(manifest_path):
        raise ValueError("runtime is not linked to the current prepare manifest")
    for relative, expected in runtime.get("artifacts", {}).items():
        path = output_dir / relative
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"runtime artifact hash mismatch for {relative}")
    result_relative = runtime.get("result_path")
    result = None
    if result_relative:
        result_path = output_dir / result_relative
        metadata, labels, rows = parse_storage(result_path)
        result = {
            "path": result_relative,
            "sha256": _sha256(result_path),
            "row_count": len(rows),
            "column_count": len(labels),
            "time_range_s": [rows[0][0], rows[-1][0]],
            "in_degrees": metadata.get("indegrees"),
            "kind": "solution" if runtime.get("success") else "failed_guess",
        }
    summary_path = output_dir / "summary.json"
    summary = {
        "schema_version": _SCHEMA,
        "scope": _SCOPE,
        "official_reference_only": True,
        "success": runtime.get("success") is True,
        "sealed_failure_captured_as_guess": runtime.get("sealed_failure_captured_as_guess") is True,
        "error": runtime.get("error"),
        "result": result,
        "outputs": {
            "reserve_controls": runtime.get("reserve_controls_path"),
            "muscle_outputs": runtime.get("muscle_outputs_path"),
        },
        "prescribed_motion_scope": manifest["configuration"]["prescribed_motion_scope"],
        "source_and_runtime": {
            "pinned_upstream": manifest["pinned_upstream"],
            "opensim_version": runtime.get("opensim_version"),
            "python_version": runtime.get("python_version"),
            "platform": runtime.get("platform"),
            "wall_time_s": runtime.get("wall_time_s"),
            "prepare_manifest_sha256": _sha256(manifest_path),
            "run_runtime_sha256": _sha256(runtime_path),
        },
        "artifacts": _artifact_hashes(output_dir, excluded={summary_path}),
    }
    _write_json(summary_path, summary)
    return summary_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="prepare from an accepted official RRA directory")
    prepare.add_argument("rra_path", type=Path)
    prepare.add_argument("output_dir", type=Path)
    prepare.add_argument("--mesh-interval", type=float, default=_DEFAULT_MESH_INTERVAL)
    prepare.add_argument("--max-iterations", type=int, default=_DEFAULT_MAX_ITERATIONS)
    prepare.add_argument("--guess-file", type=Path)
    run = subparsers.add_parser("run", help="run official MocoInverse in an isolated child process")
    run.add_argument("output_dir", type=Path)
    run.add_argument("--in-process", action="store_true")
    summarize = subparsers.add_parser("summarize", help="summarize a completed or failed run")
    summarize.add_argument("output_dir", type=Path)
    child = subparsers.add_parser("_run-child", help=argparse.SUPPRESS)
    child.add_argument("output_dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "prepare":
        prepared = prepare_reference(
            arguments.rra_path,
            arguments.output_dir,
            mesh_interval=arguments.mesh_interval,
            max_iterations=arguments.max_iterations,
            guess_file=arguments.guess_file,
        )
        print(prepared.manifest_path)
    elif arguments.command == "run":
        print(run_reference(arguments.output_dir, child_process=not arguments.in_process))
    elif arguments.command == "_run-child":
        runtime_path = _run_reference_in_process(arguments.output_dir)
        print(runtime_path)
        return 0 if json.loads(runtime_path.read_text(encoding="utf-8")).get("success") else 1
    else:
        print(summarize_reference(arguments.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
