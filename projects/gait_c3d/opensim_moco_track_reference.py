# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Official torque-driven MocoTrack contact reference for S001 Trial 101.

The adapter follows the torque-driven lane in the pinned OpenSim
``example3DWalking``. Measured ``ExternalLoads`` are supplied only to
``MocoContactTrackingGoal``. They are never applied to the model. The output is
an official OpenSim reference. It is not a Newton prediction and it does not
satisfy the project's predictive/FD1 gate.

``prepare`` is pure Python. Only ``run`` imports the optional OpenSim bindings.
The CLI runs OpenSim in a child process by default so a SWIG/runtime failure
cannot destroy the prepare manifest or the failure record.
"""

from __future__ import annotations

import argparse
import hashlib
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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from projects.gait_c3d import opensim_moco_contact_reference as _contact
from projects.gait_c3d import opensim_moco_inverse_reference as _inverse

_SCHEMA = "gait_c3d_official_opensim_moco_track_contact_reference_1"
_SCOPE = "official_opensim_torque_driven_moco_track_contact_reference_not_predictive_or_fd1"
_CONTACT_SCHEMA = "gait_c3d_opensim_moco_contact_reference_1"
_PINNED_COMMIT = "11036b39ca7232c604685b37f483afafc056d92b"
_EXAMPLE_PATH = "Bindings/Python/examples/Moco/example3DWalking/example3DWalking.py"
_EXAMPLE_SHA256 = "5a4d7ee014c91ce0b09453f49b0ce33da6b1296e5a23801772d2a3b9fd1ca5e2"
_EXAMPLE_URL = f"https://raw.githubusercontent.com/opensim-org/opensim-core/{_PINNED_COMMIT}/{_EXAMPLE_PATH}"
_DEFAULT_MESH_INTERVAL = 0.02
_DEFAULT_MAX_ITERATIONS = 1000
_GENERATED_CONTACT_FILES = (
    "S001_ContactGeometrySet.xml",
    "S001_ContactForceSet.xml",
    "S001_newton_contact_augmentation.json",
    "S001_MocoContactTrackingGoal_groups.json",
)
_MTP_COORDINATES = ("mtp_angle_l", "mtp_angle_r")
_OFFICIAL_TOE_EXPRESSION = "-25.0*q-2.0*qdot"
_TOE_VALUE_BOUNDS = [-1.0, 1.0]
_TOE_SPEED_BOUNDS = [-20.0, 20.0]


@dataclass(frozen=True)
class PreparedReference:
    """Files created by :func:`prepare_reference`."""

    output_dir: Path
    manifest_path: Path
    config_path: Path
    states_reference_path: Path


@dataclass(frozen=True)
class PeriodicitySpec:
    """State pairs for one full overground stride."""

    value_paths: tuple[str, ...]
    speed_paths: tuple[str, ...]


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
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"{description} hash changed for {path}: expected {expected}, got {actual}")


def _xml_local_name(tag: str) -> str:
    return tag.rsplit("}", maxsplit=1)[-1]


def _mtp_locked_states_from_model_file(model_path: str | os.PathLike[str]) -> dict[str, bool]:
    """Read the default MTP locked properties without importing OpenSim."""
    root = ET.parse(Path(model_path).expanduser().resolve()).getroot()
    states: dict[str, bool] = {}
    for element in root.iter():
        if _xml_local_name(element.tag) != "Coordinate":
            continue
        name = element.get("name")
        if name not in _MTP_COORDINATES:
            continue
        locked_elements = [child for child in element if _xml_local_name(child.tag) == "locked"]
        if len(locked_elements) > 1:
            raise ValueError(f"model coordinate {name} has multiple locked properties")
        raw = locked_elements[0].text.strip().lower() if locked_elements and locked_elements[0].text else "false"
        if raw not in {"true", "false"}:
            raise ValueError(f"model coordinate {name} has invalid locked property {raw!r}")
        if name in states:
            raise ValueError(f"model has duplicate coordinate {name}")
        states[name] = raw == "true"
    missing = set(_MTP_COORDINATES) - set(states)
    if missing:
        raise ValueError(f"model is missing MTP coordinates: {sorted(missing)}")
    return states


def _toe_policy(model_path: str | os.PathLike[str]) -> dict[str, Any]:
    infos = _inverse.coordinate_info_from_model(model_path)
    locked_states = _mtp_locked_states_from_model_file(model_path)
    coordinates: dict[str, dict[str, Any]] = {}
    for name in _MTP_COORDINATES:
        state_path = infos[name].state_path
        locked = locked_states[name]
        coordinates[name] = {
            "locked": locked,
            "mode": "locked_fixed_to_calcaneus" if locked else "unlocked_official_example",
            "value_state_path": state_path,
            "speed_state_path": state_path.removesuffix("/value") + "/speed",
            "passive_force": None
            if locked
            else {"name": f"PassiveToeDamping_{name[-1]}", "expression": _OFFICIAL_TOE_EXPRESSION},
            "weak_actuator": None
            if locked
            else {"name": f"{name}_actuator", "optimal_force": 10.0, "control_bounds": [-1.0, 1.0]},
            "state_bounds": None if locked else {"value": _TOE_VALUE_BOUNDS, "speed": _TOE_SPEED_BOUNDS},
        }
    return {
        "coordinates": coordinates,
        "locked_policy": (
            "locked/fixed-to-calcaneus: fixed toes to their calcaneus; omit PassiveToeDamping, weak MTP actuators, and MTP state bounds"
        ),
        "unlocked_policy": (
            "apply official example PassiveToeDamping (-25*q-2*qdot), weak MTP actuation, and MTP state bounds"
        ),
        "contact_alternative_frames_preserved": ["/bodyset/toes_l", "/bodyset/toes_r"],
    }


def _opensim_mtp_locked_states(model: Any) -> dict[str, bool]:
    """Inspect effective MTP locks from an initialized official OpenSim model."""
    state = model.initSystem()
    coordinates = model.getCoordinateSet()
    locked: dict[str, bool] = {}
    for name in _MTP_COORDINATES:
        coordinate = coordinates.get(name)
        get_locked = getattr(coordinate, "getLocked", None)
        if callable(get_locked):
            locked[name] = bool(get_locked(state))
        else:
            locked[name] = bool(coordinate.get_locked())
    return locked


def _validate_runtime_toe_policy(model: Any, config: dict[str, Any]) -> dict[str, bool]:
    expected_entries = config.get("toe_policy", {}).get("coordinates")
    if not isinstance(expected_entries, dict) or set(expected_entries) != set(_MTP_COORDINATES):
        raise ValueError("configuration has no exact MTP toe policy")
    expected = {name: entry.get("locked") for name, entry in expected_entries.items()}
    if any(not isinstance(value, bool) for value in expected.values()):
        raise ValueError("configuration has invalid MTP locked state")
    actual = _opensim_mtp_locked_states(model)
    if actual != expected:
        raise ValueError(f"runtime MTP locked states changed: expected {expected}, got {actual}")
    return actual


def _load_canonical_config(output: Path, manifest: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    """Load the hash-sealed config and reject either direction of divergence."""
    config_path = (output / "inputs" / "moco_track_config.json").resolve()
    generated = manifest.get("generated_inputs")
    if not isinstance(generated, dict):
        raise ValueError("manifest has no generated input hashes")
    metadata = generated.get(str(config_path))
    if not isinstance(metadata, dict) or metadata.get("sha256") != _sha256(config_path):
        raise ValueError("canonical MocoTrack configuration hash does not match manifest")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config != manifest.get("configuration"):
        raise ValueError("canonical MocoTrack configuration diverges from prepare manifest")
    if config.get("schema_version") != _SCHEMA or config.get("scope") != _SCOPE:
        raise ValueError("canonical MocoTrack configuration has invalid schema or scope")
    return config_path, config


def _resolve_contact_manifest(path: str | os.PathLike[str]) -> Path:
    candidate = Path(path).expanduser().resolve()
    if candidate.is_dir():
        candidate = candidate / "manifest.json"
    if candidate.name != "manifest.json" or not candidate.is_file():
        raise ValueError("contact reference must be its generated directory or manifest.json")
    return candidate


def _validate_contact_reference(
    contact_path: str | os.PathLike[str], external_loads_path: str | os.PathLike[str]
) -> dict[str, Path]:
    """Validate all generated contact files and their measured-load linkage."""
    manifest_path = _resolve_contact_manifest(contact_path)
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Normalize tuples from dataclass provenance to their JSON array form.
    expected_provenance = json.loads(json.dumps(_contact.provenance(), allow_nan=False))
    for key in ("schema_version", "scope", "frame", "units", "pinned_upstream", "s001_vertical_alignment"):
        if manifest.get(key) != expected_provenance[key]:
            raise ValueError(f"contact reference has invalid {key}")
    if manifest["schema_version"] != _CONTACT_SCHEMA:
        raise ValueError("contact reference schema is not supported")
    contract = manifest.get("contact_tracking_contract")
    if contract != expected_provenance["contact_tracking_contract"]:
        raise ValueError("contact tracking contract changed")
    if contract.get("external_loads_added_to_predictive_model") is not False:
        raise ValueError("contact reference applies measured ExternalLoads to the model")

    generated = manifest.get("generated")
    if not isinstance(generated, dict) or set(generated) != set(_GENERATED_CONTACT_FILES):
        raise ValueError("contact manifest generated-file map is incomplete")
    paths = {name: root / name for name in _GENERATED_CONTACT_FILES}
    for name, path in paths.items():
        if not path.is_file() or _sha256(path) != generated[name]:
            raise ValueError(f"contact generated hash mismatch for {name}")

    # Hashes alone do not prevent a selectively resealed manifest. Compare the
    # scientific XML and goal-group specification with the pinned generator.
    if paths["S001_ContactGeometrySet.xml"].read_bytes() != _contact.xml_bytes(_contact.build_contact_geometry_xml()):
        raise ValueError("contact geometry is not the pinned S001 geometry")
    if paths["S001_ContactForceSet.xml"].read_bytes() != _contact.xml_bytes(_contact.build_force_xml()):
        raise ValueError("contact force set is not the pinned S001 force set")
    groups_payload = json.loads(paths["S001_MocoContactTrackingGoal_groups.json"].read_text(encoding="utf-8"))
    if groups_payload.get("model_added_external_loads") is not False:
        raise ValueError("contact group specification adds measured ExternalLoads")
    expected_groups = json.loads(
        json.dumps([asdict(group) for group in _contact.moco_contact_groups()], allow_nan=False)
    )
    if groups_payload.get("groups") != expected_groups:
        raise ValueError("contact group specification changed")

    external = Path(external_loads_path).expanduser().resolve()
    data = _contact.validate_external_loads_reference(external)
    measured = manifest.get("measured_reference")
    expected_measured = {
        "external_loads_path": str(external),
        "external_loads_sha256": _sha256(external),
        "data_path": str(data),
        "data_sha256": _sha256(data),
    }
    if measured != expected_measured:
        raise ValueError("corrected ExternalLoads reference is not hash-linked to the contact manifest")
    if groups_payload.get("external_loads_reference") != str(external):
        raise ValueError("contact groups are not linked to the corrected ExternalLoads reference")
    return {"manifest": manifest_path, "external_loads": external, "external_loads_data": data, **paths}


def convert_rra_states_reference(
    source_path: str | os.PathLike[str],
    model_path: str | os.PathLike[str],
    destination_path: str | os.PathLike[str],
) -> Path:
    """Write absolute RRA coordinate values in SI units.

    Speeds are intentionally absent. At run time the pinned
    ``TabOpAppendCoordinateValueDerivativesAsSpeeds`` operation derives them
    from OpenSim splines, exactly as in ``example3DWalking.runTrackingStudy``.
    """
    metadata, labels, rows = _inverse.parse_storage(source_path)
    infos = _inverse.coordinate_info_from_model(model_path)
    if metadata.get("indegrees", "").lower() not in {"yes", "true"}:
        raise ValueError("RRA Kinematics_q must declare inDegrees=yes")
    source_labels = labels[1:]
    if any(label.startswith("/") for label in source_labels):
        raise ValueError("RRA Kinematics_q must use legacy short coordinate labels")
    if len(source_labels) != len(set(source_labels)) or set(source_labels) != set(infos):
        raise ValueError("RRA coordinate labels do not match the accepted model")
    output_labels = ["time"] + [infos[label].state_path for label in source_labels]
    output_rows: list[list[float]] = []
    for row in rows:
        converted = [row[0]]
        for index, label in enumerate(source_labels, start=1):
            value = math.radians(row[index]) if infos[label].rotational else row[index]
            converted.append(value)
        output_rows.append(converted)
    destination = Path(destination_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    _inverse._write_storage(
        destination,
        output_labels,
        output_rows,
        name="Absolute RRA state values for official torque-driven MocoTrack",
    )
    return destination


def periodicity_spec(model_path: str | os.PathLike[str]) -> PeriodicitySpec:
    """Return full-stride state pairs, excluding only overground pelvis_tx value."""
    infos = _inverse.coordinate_info_from_model(model_path)
    locked_mtp = _mtp_locked_states_from_model_file(model_path)
    values: list[str] = []
    speeds: list[str] = []
    for info in infos.values():
        # Coupled beta coordinates, if present, are dependent coordinates in the
        # pinned example and cannot receive independent periodicity pairs.
        if locked_mtp.get(info.coordinate, False):
            continue
        if not info.coordinate.endswith("_beta") and info.coordinate != "pelvis_tx":
            values.append(info.state_path)
        if not info.coordinate.endswith("_beta"):
            speeds.append(info.state_path.removesuffix("/value") + "/speed")
    return PeriodicitySpec(tuple(values), tuple(speeds))


def contact_goal_groups() -> list[dict[str, Any]]:
    """Return the exact left/right force groups as JSON-friendly dictionaries."""
    return json.loads(json.dumps([asdict(group) for group in _contact.moco_contact_groups()], allow_nan=False))


def prepare_reference(
    rra_path: str | os.PathLike[str],
    contact_reference_path: str | os.PathLike[str],
    external_loads_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    mesh_interval: float = _DEFAULT_MESH_INTERVAL,
    max_iterations: int = _DEFAULT_MAX_ITERATIONS,
    guess_file: str | os.PathLike[str] | None = None,
) -> PreparedReference:
    """Prepare a hash-sealed torque-driven MocoTrack reference."""
    if not math.isfinite(mesh_interval) or mesh_interval <= 0.0:
        raise ValueError("mesh interval must be positive and finite")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 0:
        raise ValueError("max iterations must be a nonnegative integer")

    rra_root = _inverse._resolve_rra_root(rra_path)
    rra_sources = _inverse._validate_accepted_rra(rra_root)
    contact_sources = _validate_contact_reference(contact_reference_path, external_loads_path)
    if contact_sources["external_loads"] != rra_sources["external_loads_path"]:
        raise ValueError("corrected ExternalLoads must be the accepted RRA ExternalLoads input")
    if contact_sources["external_loads_data"] != rra_sources["external_loads_data_path"]:
        raise ValueError("corrected ExternalLoads data must be the accepted RRA measured data")
    _contact.assert_model_has_no_external_loads(rra_sources["model_path"])

    output = Path(output_dir).expanduser().resolve()
    protected_roots = {rra_root, contact_sources["manifest"].parent}
    if any(output == root or output.is_relative_to(root) or root.is_relative_to(output) for root in protected_roots):
        raise ValueError("MocoTrack output must not overlap accepted input directories")
    inputs_dir = output / "inputs"
    results_dir = output / "results"
    for directory in (inputs_dir, results_dir):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)
    for stale in (
        output / "prepare_manifest.json",
        output / "run_runtime.json",
        output / "summary.json",
        output / "moco.log",
        output / "child_process.json",
    ):
        stale.unlink(missing_ok=True)

    guess_path: Path | None = None
    if guess_file is not None:
        guess_path = Path(guess_file).expanduser().resolve()
        if not guess_path.is_file():
            raise FileNotFoundError(guess_path)

    states_path = inputs_dir / "rra_absolute_state_values.sto"
    convert_rra_states_reference(rra_sources["kinematics_path"], rra_sources["model_path"], states_path)
    _, state_labels, state_rows = _inverse.parse_storage(states_path)
    periodicity = periodicity_spec(rra_sources["model_path"])
    toe_policy = _toe_policy(rra_sources["model_path"])
    config_path = inputs_dir / "moco_track_config.json"
    config = {
        "schema_version": _SCHEMA,
        "scope": _SCOPE,
        "model_path": str(rra_sources["model_path"]),
        "contact_geometry_path": str(contact_sources["S001_ContactGeometrySet.xml"]),
        "contact_force_path": str(contact_sources["S001_ContactForceSet.xml"]),
        "external_loads_reference_path": str(contact_sources["external_loads"]),
        "states_reference_path": str(states_path),
        "initial_time_s": state_rows[0][0],
        "final_time_s": state_rows[-1][0],
        "mesh_interval_s": float(mesh_interval),
        "max_iterations": max_iterations,
        "guess_file": str(guess_path) if guess_path else None,
        "model_processor_operations": [
            {"operation": "ModOpRemoveMuscles", "arguments": []},
            {"operation": "ModOpAddReserves", "arguments": [500.0, 1.0, True, True]},
        ],
        "reference_table_operations": [
            "TabOpUseAbsoluteStateNames",
            "TabOpAppendCoupledCoordinateValues",
            "TabOpAppendCoordinateValueDerivativesAsSpeeds",
        ],
        "goals": [
            {"type": "MocoStateTrackingGoal", "name": "state_tracking", "weight": 0.05},
            {"type": "MocoControlGoal", "name": "control_effort", "weight": 0.1},
            {
                "type": "MocoContactTrackingGoal",
                "name": "grf_tracking",
                "weight": 5.0e-3,
                "external_loads_usage": "reference_only",
                "groups": contact_goal_groups(),
            },
            {"type": "MocoPeriodicityGoal", "name": "periodicity"},
        ],
        "state_weight_overrides": {
            "/jointset/ground_pelvis/pelvis_ty/value": 0.0,
            "/jointset/ground_pelvis/pelvis_ty/speed": 0.1,
        },
        "toe_policy": toe_policy,
        "periodicity": {
            "value_state_pairs": list(periodicity.value_paths),
            "speed_state_pairs": list(periodicity.speed_paths),
            "excluded_value_states": ["/jointset/ground_pelvis/pelvis_tx/value"],
            "coordinate_actuator_controls": True,
        },
        "solver": {
            "transcription_scheme": "legendre-gauss-radau-3",
            "kinematic_constraint_method": "Bordalba2023",
            "optim_convergence_tolerance": 1.0e-2,
            "optim_constraint_tolerance": 1.0e-4,
        },
        "reference_contract": {
            "absolute_state_value_columns": len(state_labels) - 1,
            "speeds": "OpenSim spline derivatives from TabOpAppendCoordinateValueDerivativesAsSpeeds",
            "full_stride": True,
            "measured_external_loads": "MocoContactTrackingGoal reference only; absent from model dynamics",
            "not_claimed_until_downstream_gates": ["Newton predictive result", "forward dynamics", "FD1"],
        },
    }
    _write_json(config_path, config)

    source_paths = set(rra_sources.values()) | set(contact_sources.values())
    if guess_path:
        source_paths.add(guess_path)
    manifest = {
        "schema_version": _SCHEMA,
        "scope": _SCOPE,
        "official_runtime_required_only_for_run": True,
        "accepted_rra_root": str(rra_root),
        "contact_reference_root": str(contact_sources["manifest"].parent),
        "source_inputs": {str(path): {"sha256": _sha256(path)} for path in sorted(source_paths)},
        "generated_inputs": {str(path): {"sha256": _sha256(path)} for path in (states_path, config_path)},
        "configuration": config,
        "pinned_upstream": {
            "repository": "https://github.com/opensim-org/opensim-core",
            "commit": _PINNED_COMMIT,
            "files": {_EXAMPLE_PATH: {"url": _EXAMPLE_URL, "sha256": _EXAMPLE_SHA256}},
        },
    }
    manifest_path = output / "prepare_manifest.json"
    _write_json(manifest_path, manifest)
    return PreparedReference(output, manifest_path, config_path, states_path)


def _import_official_opensim() -> Any:
    try:
        import opensim  # noqa: PLC0415
    except ImportError as exception:
        raise RuntimeError("official OpenSim Python bindings are required only for the 'run' command") from exception
    return opensim


def build_model_processor(opensim: Any, config: dict[str, Any]) -> tuple[Any, Any]:
    """Build the official contact model and torque-lane ModelProcessor."""
    model = opensim.Model(config["model_path"])
    _contact.assert_model_has_no_external_loads(model)
    locked_states = _validate_runtime_toe_policy(model, config)
    geometry_set = opensim.ContactGeometrySet(config["contact_geometry_path"])
    for index in range(geometry_set.getSize()):
        model.addContactGeometry(geometry_set.get(index).clone())
    force_set = opensim.ForceSet(config["contact_force_path"])
    for index in range(force_set.getSize()):
        model.addComponent(force_set.get(index).clone())
    for coordinate_name in _MTP_COORDINATES:
        if locked_states[coordinate_name]:
            continue
        passive = opensim.ExpressionBasedCoordinateForce()
        passive.setName(f"PassiveToeDamping_{coordinate_name[-1]}")
        passive.set_coordinate(coordinate_name)
        passive.set_expression(_OFFICIAL_TOE_EXPRESSION)
        model.addForce(passive)
        actuator = opensim.CoordinateActuator(coordinate_name)
        actuator.setName(f"{coordinate_name}_actuator")
        actuator.setOptimalForce(10.0)
        actuator.setMinControl(-1.0)
        actuator.setMaxControl(1.0)
        model.addForce(actuator)
    model.finalizeConnections()
    model.initSystem()
    _contact.assert_model_has_no_external_loads(model)
    processor = opensim.ModelProcessor(model)
    processor.append(opensim.ModOpRemoveMuscles())
    processor.append(opensim.ModOpAddReserves(500.0, 1.0, True, True))
    return model, processor


def build_reference_table_processor(opensim: Any, config: dict[str, Any]) -> Any:
    """Build the pinned absolute-state and spline-speed TableProcessor."""
    processor = opensim.TableProcessor(config["states_reference_path"])
    processor.append(opensim.TabOpUseAbsoluteStateNames())
    processor.append(opensim.TabOpAppendCoupledCoordinateValues())
    processor.append(opensim.TabOpAppendCoordinateValueDerivativesAsSpeeds())
    return processor


def configure_contact_tracking_goal(opensim: Any, config: dict[str, Any]) -> Any:
    """Create the reference-only left/right Moco contact goal."""
    return _contact.configure_moco_contact_tracking_goal(
        opensim,
        config["external_loads_reference_path"],
        name="grf_tracking",
        weight=5.0e-3,
    )


def _add_periodicity_goal(opensim: Any, problem: Any, model: Any, config: dict[str, Any]) -> Any:
    goal = opensim.MocoPeriodicityGoal("periodicity")
    for path in config["periodicity"]["value_state_pairs"] + config["periodicity"]["speed_state_pairs"]:
        goal.addStatePair(opensim.MocoPeriodicityGoalPair(path))
    actuators = model.getActuators()
    for index in range(actuators.getSize()):
        actuator = opensim.CoordinateActuator.safeDownCast(actuators.get(index))
        if actuator is not None:
            goal.addControlPair(opensim.MocoPeriodicityGoalPair(actuator.getAbsolutePathString()))
    problem.addGoal(goal)
    return goal


def _solution_success(solution: Any) -> bool:
    method = getattr(solution, "success", None)
    return bool(method()) if callable(method) else False


def write_solution_or_failed_guess(solution: Any, solution_path: Path, failed_guess_path: Path) -> tuple[Path, bool]:
    """Preserve Moco's sealed unsuccessful solution as a reusable guess."""
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


def _run_reference_in_process(output_dir: str | os.PathLike[str]) -> Path:
    output = Path(output_dir).expanduser().resolve()
    manifest_path = output / "prepare_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != _SCHEMA or manifest.get("scope") != _SCOPE:
        raise ValueError("output directory is not a prepared official MocoTrack reference")
    _verify_hash_mapping(manifest.get("source_inputs"), description="source input")
    _verify_hash_mapping(manifest.get("generated_inputs"), description="generated input")
    config_path, config = _load_canonical_config(output, manifest)
    results_dir = output / "results"
    if results_dir.exists():
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True)
    runtime_path = output / "run_runtime.json"
    log_path = output / "moco.log"
    for stale in (runtime_path, log_path, output / "summary.json"):
        stale.unlink(missing_ok=True)

    started = time.perf_counter()
    run_id = uuid.uuid4().hex
    opensim: Any | None = None
    logger_added = False
    success = False
    sealed_failure = False
    error: str | None = None
    result_path: Path | None = None
    model_path: Path | None = None
    ground_reactions_path: Path | None = None
    try:
        opensim = _import_official_opensim()
        opensim.Logger.addFileSink(str(log_path))
        logger_added = True
        source_model, model_processor = build_model_processor(opensim, config)
        table_processor = build_reference_table_processor(opensim, config)
        track = opensim.MocoTrack()
        track.setName("torque_driven_tracking")
        track.setModel(model_processor)
        track.setStatesReference(table_processor)
        track.set_states_global_tracking_weight(0.05)
        track.set_control_effort_weight(0.1)
        track.set_allow_unused_references(True)
        track.set_track_reference_position_derivatives(True)
        track.set_initial_time(float(config["initial_time_s"]))
        track.set_final_time(float(config["final_time_s"]))
        track.set_mesh_interval(float(config["mesh_interval_s"]))
        weights = opensim.MocoWeightSet()
        for path, weight in config["state_weight_overrides"].items():
            weights.cloneAndAppend(opensim.MocoWeight(path, float(weight)))
        track.set_states_weight_set(weights)
        study = track.initialize()
        problem = study.updProblem()
        for toe in config["toe_policy"]["coordinates"].values():
            if toe["locked"]:
                continue
            problem.setStateInfo(toe["value_state_path"], toe["state_bounds"]["value"])
            problem.setStateInfo(toe["speed_state_path"], toe["state_bounds"]["speed"])
        problem.addGoal(configure_contact_tracking_goal(opensim, config))

        updated = table_processor.process(source_model)
        index = updated.getNearestRowIndexForTime(float(config["initial_time_s"]))
        for label in updated.getColumnLabels():
            values = updated.getDependentColumn(label).to_numpy()
            width = 0.1 if "/speed" in label else 0.05
            problem.setStateInfo(label, [], [float(values[index]) - width, float(values[index]) + width])

        processed_model = model_processor.process()
        processed_model.initSystem()
        _contact.assert_model_has_no_external_loads(processed_model)
        _add_periodicity_goal(opensim, problem, processed_model, config)
        solver = opensim.MocoCasADiSolver.safeDownCast(study.updSolver())
        if solver is None:
            raise RuntimeError("MocoTrack did not initialize a MocoCasADiSolver")
        solver.set_transcription_scheme(config["solver"]["transcription_scheme"])
        solver.set_kinematic_constraint_method(config["solver"]["kinematic_constraint_method"])
        solver.set_optim_convergence_tolerance(config["solver"]["optim_convergence_tolerance"])
        solver.set_optim_constraint_tolerance(config["solver"]["optim_constraint_tolerance"])
        solver.set_optim_max_iterations(int(config["max_iterations"]))
        solver.resetProblem(problem)
        if config.get("guess_file"):
            solver.setGuessFile(config["guess_file"])
        else:
            solver.setGuess(solver.createGuess())
        solution = study.solve()
        result_path, success = write_solution_or_failed_guess(
            solution,
            results_dir / "moco_track_solution.sto",
            results_dir / "moco_track_failed_guess.sto",
        )
        sealed_failure = not success
        model_path = results_dir / "moco_track_model.osim"
        processed_model.printToXML(str(model_path))
        if success:
            groups = {group.side: group for group in _contact.moco_contact_groups()}
            right_forces = opensim.StdVectorString()
            left_forces = opensim.StdVectorString()
            for path in groups["right"].contact_force_paths:
                right_forces.append(path)
            for path in groups["left"].contact_force_paths:
                left_forces.append(path)
            ground_reactions = opensim.createExternalLoadsTableForGait(
                processed_model, solution, right_forces, left_forces
            )
            ground_reactions_path = results_dir / "moco_track_ground_reactions.sto"
            opensim.STOFileAdapter.write(ground_reactions, str(ground_reactions_path))
        else:
            error = "official Moco solver returned a sealed unsuccessful solution; captured as failed guess"
    except Exception as exception:  # Preserve records across OpenSim/SWIG errors.
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
            "sealed_failure_captured_as_guess": sealed_failure,
            "error": error,
            "wall_time_s": time.perf_counter() - started,
            "opensim_version": opensim.GetVersionAndDate() if opensim is not None else None,
            "python_version": sys.version,
            "platform": platform.platform(),
            "prepare_manifest_sha256": _sha256(manifest_path),
            "configuration_sha256": _sha256(config_path),
            "artifact_linkage": {"run_id": run_id, "root": str(output)},
            "result_path": str(result_path.relative_to(output)) if result_path else None,
            "model_path": str(model_path.relative_to(output)) if model_path else None,
            "ground_reactions_path": (
                str(ground_reactions_path.relative_to(output)) if ground_reactions_path else None
            ),
            "artifacts": _artifact_hashes(output, excluded={runtime_path}),
        }
        _write_json(runtime_path, runtime)
    return runtime_path


def run_reference(output_dir: str | os.PathLike[str], *, child_process: bool = False) -> Path:
    """Run official MocoTrack, optionally in an isolated child process."""
    output = Path(output_dir).expanduser().resolve()
    if not child_process:
        return _run_reference_in_process(output)
    lifecycle_path = output / "child_process.json"
    command = [sys.executable, str(Path(__file__).resolve()), "_run-child", str(output)]
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
    runtime_path = output / "run_runtime.json"
    if not runtime_path.is_file():
        raise RuntimeError(
            f"MocoTrack child exited {completed.returncode} without a runtime record; see {lifecycle_path}"
        )
    return runtime_path


def summarize_reference(output_dir: str | os.PathLike[str]) -> Path:
    """Create a strictly hash-linked summary without making FD1 claims."""
    output = Path(output_dir).expanduser().resolve()
    manifest_path = output / "prepare_manifest.json"
    runtime_path = output / "run_runtime.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != _SCHEMA or runtime.get("schema_version") != _SCHEMA:
        raise ValueError("prepared and runtime schemas do not match official MocoTrack")
    if runtime.get("scope") != _SCOPE or runtime.get("prepare_manifest_sha256") != _sha256(manifest_path):
        raise ValueError("runtime is not linked to the current prepare manifest")
    _verify_hash_mapping(manifest.get("source_inputs"), description="source input")
    _verify_hash_mapping(manifest.get("generated_inputs"), description="generated input")
    _, config = _load_canonical_config(output, manifest)
    for relative, expected in runtime.get("artifacts", {}).items():
        path = output / relative
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"runtime artifact hash mismatch for {relative}")
    result = None
    if runtime.get("result_path"):
        result_path = output / runtime["result_path"]
        metadata, labels, rows = _inverse.parse_storage(result_path)
        result = {
            "path": runtime["result_path"],
            "sha256": _sha256(result_path),
            "row_count": len(rows),
            "column_count": len(labels),
            "time_range_s": [rows[0][0], rows[-1][0]],
            "in_degrees": metadata.get("indegrees"),
            "kind": "solution" if runtime.get("success") else "failed_guess",
        }
    gates = {
        "runtime_success": runtime.get("success") is True,
        "source_hashes_verified": True,
        "measured_external_loads_reference_only": True,
        "full_stride_periodicity_configured": True,
        "official_contact_solution_available": runtime.get("success") is True and result is not None,
    }
    summary_path = output / "summary.json"
    summary = {
        "schema_version": _SCHEMA,
        "scope": _SCOPE,
        "official_reference_only": True,
        "success": runtime.get("success") is True,
        "sealed_failure_captured_as_guess": runtime.get("sealed_failure_captured_as_guess") is True,
        "error": runtime.get("error"),
        "result": result,
        "outputs": {
            "model": runtime.get("model_path"),
            "ground_reactions": runtime.get("ground_reactions_path"),
        },
        "gates": gates,
        "claims": {
            "official_torque_driven_moco_track_reference": all(gates.values()),
            "newton_predictive_forward_dynamics": False,
            "fd1": False,
            "reason": "official contact tracking is only an input gate for later predictive/FD1 work",
        },
        "configuration": config,
        "source_and_runtime": {
            "pinned_upstream": manifest["pinned_upstream"],
            "opensim_version": runtime.get("opensim_version"),
            "python_version": runtime.get("python_version"),
            "platform": runtime.get("platform"),
            "wall_time_s": runtime.get("wall_time_s"),
            "prepare_manifest_sha256": _sha256(manifest_path),
            "run_runtime_sha256": _sha256(runtime_path),
        },
        "artifacts": _artifact_hashes(output, excluded={summary_path}),
    }
    _write_json(summary_path, summary)
    return summary_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="prepare from accepted RRA and generated contact references")
    prepare.add_argument("rra_path", type=Path)
    prepare.add_argument("contact_reference_path", type=Path)
    prepare.add_argument("external_loads_path", type=Path)
    prepare.add_argument("output_dir", type=Path)
    prepare.add_argument("--mesh-interval", type=float, default=_DEFAULT_MESH_INTERVAL)
    prepare.add_argument("--max-iterations", type=int, default=_DEFAULT_MAX_ITERATIONS)
    prepare.add_argument("--guess-file", type=Path)
    run = subparsers.add_parser("run", help="run official MocoTrack in an isolated child process")
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
            arguments.contact_reference_path,
            arguments.external_loads_path,
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
