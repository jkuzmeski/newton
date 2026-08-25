# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""OFFLINE COMPATIBILITY REFERENCE ONLY. Despite the historical name, this uses OpenSim-shaped contact; production contact uses ``newton_contact_calibration``.

Calibrate the exact 12-sphere Moco contact topology on accepted RRA motion.

The optimizer evaluates contact from prescribed q/qd only. Measured platform
wrenches are held by the objective, never by the contact evaluator. The output
is a prescribed-motion calibration artifact, not forward dynamics or FD-1.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

import newton.opensim as osim
from projects.gait_c3d.compatibility import predictive_contact
from projects.gait_c3d.oracles import opensim_moco_contact_reference as reference

ARCHITECTURE_ROLE = "compatibility_reference"

_SCHEMA = "gait_c3d_moco_contact_calibration_1"
_SCOPE = "exact_12_sphere_rra_adjusted_prescribed_contact_calibration_not_forward_dynamics"
_DEFAULT_RRA_INPUT = Path("/home/jo31399/newton-data/gait/processed/trial_101/rra_adjusted_contact_input")
_BODY_WEIGHT_N = 803.5
_BODY_HEIGHT_M = 1.695898298375747
_LOAD_THRESHOLD_N = 50.0
_COP_THRESHOLD_N = 200.0
_MAX_PENETRATION_M = 0.020
_VELOCITY_STENCIL_H_S = 1.0e-6
_SIDE_ORDER = ("left", "right")


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    """Return a stable digest including an array's dtype and shape."""
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    """Convert NumPy values and non-finite floats to strict JSON values."""
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return _json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Any) -> None:
    """Write deterministic strict JSON."""
    path.write_text(json.dumps(_json_value(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def _runtime_provenance() -> dict[str, Any]:
    """Return runtime and repository provenance."""
    root = Path(__file__).resolve().parents[3]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True, text=True).stdout
    )
    return {"git_commit": commit, "git_dirty": dirty, "python": os.sys.version, "numpy": np.__version__}


@dataclass(frozen=True, slots=True)
class ContactCandidate:
    """One exact-topology candidate and its shared material."""

    spheres: tuple[reference.SphereSpec, ...]
    material: dict[str, float]


@dataclass(frozen=True, slots=True)
class ContactEvaluation:
    """Predicted foot wrenches and sphere penetrations."""

    foot_wrenches: np.ndarray
    penetrations_m: np.ndarray


def prepare_contact_model(
    model: osim.OsimModel,
    coordinate_names: Sequence[str],
    coordinates: np.ndarray,
) -> tuple[osim.OsimModel, dict[str, Any]]:
    """Repair the legacy zero-width MTP range without changing accepted states.

    The accepted RRA motion contains small nonzero MTP values, while the legacy
    model declares both coordinates unlocked and clamped to ``[0, 0]``. The
    predictive model uses the pinned 3-D walking range of +/-30 degrees and
    records the repair explicitly.
    """
    prepared = copy.deepcopy(model)
    names = tuple(str(name) for name in coordinate_names)
    q = np.asarray(coordinates, dtype=float)
    repairs: dict[str, Any] = {}
    bound = math.radians(30.0)
    for joint in prepared.joints:
        for coordinate in joint.coordinates:
            if coordinate.name not in ("mtp_angle_l", "mtp_angle_r"):
                continue
            index = names.index(coordinate.name)
            observed = (float(np.min(q[:, index])), float(np.max(q[:, index])))
            old_range = coordinate.range
            if coordinate.locked:
                raise ValueError(f"{coordinate.name} is locked but accepted RRA motion is nonzero")
            if old_range is None or old_range[0] == old_range[1]:
                if observed[0] < -bound or observed[1] > bound:
                    raise ValueError(f"{coordinate.name} accepted motion exceeds the pinned MTP range")
                coordinate.range = (-bound, bound)
                coordinate.clamped = True
                repairs[coordinate.name] = {
                    "old_range_rad": old_range,
                    "new_range_rad": coordinate.range,
                    "observed_range_rad": observed,
                    "source": "pinned OpenSim example3DWalking +/-30 degree MTP range",
                }
    if set(repairs) != {"mtp_angle_l", "mtp_angle_r"}:
        raise ValueError("expected both legacy zero-width MTP ranges to require explicit repair")
    return prepared, repairs


def write_contact_ready_model(
    source: str | os.PathLike,
    destination: str | os.PathLike,
    mtp_repairs: dict[str, Any],
) -> None:
    """Patch only MTP ranges in the official-valid accepted RRA XML model."""
    source_path = Path(source).resolve()
    tree = ET.parse(source_path)
    found: set[str] = set()
    for coordinate in tree.getroot().iter("Coordinate"):
        name = coordinate.get("name", "")
        if name not in mtp_repairs:
            continue
        range_node = coordinate.find("range")
        clamped_node = coordinate.find("clamped")
        if range_node is None or clamped_node is None:
            raise ValueError(f"{name} is missing range/clamped XML fields")
        low, high = mtp_repairs[name]["new_range_rad"]
        range_node.text = f"{float(low):.17g} {float(high):.17g}"
        clamped_node.text = "true"
        found.add(name)
    if found != set(mtp_repairs):
        raise ValueError("could not patch every declared MTP range repair")
    ET.indent(tree, space="  ")
    tree.write(destination, encoding="utf-8", xml_declaration=True)


def subject_sphere_specs(model: osim.OsimModel) -> tuple[reference.SphereSpec, ...]:
    """Retarget the pinned six-role topology to S001 anatomical landmarks.

    Calcaneus roles use the scaled S001 heel and medial/lateral forefoot marker
    coordinates. Toe roles transform those landmarks into the actual articulated
    ``toes_*`` body frames at the model default pose. No measured force enters
    this geometry seed.
    """
    markers = {marker.name: marker for marker in model.markers}
    kinematics = osim.ForwardKinematics(model, device="cpu")
    defaults = {coordinate.name: coordinate.default_value for joint in model.joints for coordinate in joint.coordinates}
    q0 = np.asarray([[defaults.get(name, 0.0) for name in kinematics.coordinate_names]], dtype=float)
    transforms = np.asarray(kinematics.body_transforms_batch(q0), dtype=float)[0]
    body_transform = {name: transforms[index] for index, name in enumerate(kinematics.body_names)}
    result: list[reference.SphereSpec] = []
    for side in _SIDE_ORDER:
        suffix = "l" if side == "left" else "r"
        prefix = "L" if side == "left" else "R"
        required = {
            "heel": f"{prefix}.Heel",
            "lateral": f"{prefix}.Toe.Lat",
            "medial": f"{prefix}.Toe.Med",
        }
        if not set(required.values()).issubset(markers):
            raise KeyError(f"scaled S001 model is missing contact landmarks for {side}")
        heel = np.asarray(markers[required["heel"]].location, dtype=float)
        lateral = np.asarray(markers[required["lateral"]].location, dtype=float)
        medial = np.asarray(markers[required["medial"]].location, dtype=float)
        if any(markers[name].body != f"calcn_{suffix}" for name in required.values()):
            raise ValueError("S001 contact landmarks must be expressed in the calcaneus frame")
        calc_points = {
            "heel": heel,
            "lateralRearfoot": heel + 0.44 * (lateral - heel),
            "lateralMidfoot": heel + 0.87 * (lateral - heel),
            "medialMidfoot": medial,
            "lateralToe": lateral,
            "medialToe": medial,
        }
        calcaneus = f"calcn_{suffix}"
        toes = f"toes_{suffix}"
        calc_to_world = body_transform[calcaneus]
        world_to_toes = np.linalg.inv(body_transform[toes])
        for role in reference._ROLE_ORDER:
            body = f"{reference._ROLE_BODY[role]}_{suffix}"
            point = calc_points[role].copy()
            if body == toes:
                point = (world_to_toes @ calc_to_world @ np.asarray((*point, 1.0)))[:3]
            point[1] = reference.S001_ALIGNMENT.offset_m
            default = next(sphere for sphere in reference.sphere_specs() if sphere.side == side and sphere.role == role)
            result.append(replace(default, body=body, location_m=tuple(float(value) for value in point)))
    return reference._validate_sphere_specs(result)


class ContactParameterization:
    """Encode subject calibration without changing the pinned 12-sphere topology."""

    def __init__(self, seed_spheres: Sequence[reference.SphereSpec] | None = None) -> None:
        self._seed_spheres = reference._validate_sphere_specs(
            reference.sphere_specs() if seed_spheres is None else seed_spheres
        )
        roles = reference._ROLE_ORDER
        self._names = (
            *(f"{role}_shared_y_offset_m" for role in roles),
            "right_foot_y_offset_m",
            "fore_aft_scale",
            "medio_lateral_scale",
            "fore_aft_offset_m",
            "log10_stiffness",
            "dissipation",
            "coulomb_friction",
            "viscous_friction",
        )
        self._x0 = np.asarray([*([0.0] * 7), 1.0, 1.0, 0.0, 6.0, 2.0, 0.8, 0.5])
        self._lower = np.asarray([*([-0.05] * 6), -0.05, 0.70, 0.70, -0.04, 4.0, 0.0, 0.0, 0.0])
        self._upper = np.asarray([*([0.05] * 6), 0.05, 1.30, 1.30, 0.04, math.log10(5.0e7), 5.0, 1.5, 1.0])

    @property
    def names(self) -> tuple[str, ...]:
        """Return parameter names in encoded order."""
        return self._names

    @property
    def x0(self) -> np.ndarray:
        """Return the initial point."""
        return self._x0.copy()

    @property
    def lower_bounds(self) -> np.ndarray:
        """Return lower bounds."""
        return self._lower.copy()

    @property
    def upper_bounds(self) -> np.ndarray:
        """Return upper bounds."""
        return self._upper.copy()

    def values(self, encoded: np.ndarray) -> dict[str, float]:
        """Return named encoded values."""
        vector = self._validate(encoded)
        return {name: float(value) for name, value in zip(self.names, vector, strict=True)}

    def decode(self, encoded: np.ndarray) -> ContactCandidate:
        """Decode a candidate while preserving names, order, bodies, and radii."""
        vector = self._validate(encoded)
        if np.any(vector < self._lower) or np.any(vector > self._upper):
            raise ValueError("contact calibration parameters are outside their bounds")
        role_offsets = dict(zip(reference._ROLE_ORDER, vector[:6], strict=True))
        right_offset = float(vector[6])
        x_scale, z_scale, x_offset = (float(item) for item in vector[7:10])
        spheres = []
        for sphere in self._seed_spheres:
            x, y, z = sphere.location_m
            spheres.append(
                replace(
                    sphere,
                    location_m=(
                        x_scale * x + x_offset,
                        y + float(role_offsets[sphere.role]) + (right_offset if sphere.side == "right" else 0.0),
                        z_scale * z,
                    ),
                )
            )
        material = dict(reference._NEWTON_MATERIAL)
        material.update(
            stiffness=float(10.0 ** vector[10]),
            dissipation=float(vector[11]),
            static_friction=float(vector[12]),
            dynamic_friction=float(vector[12]),
            viscous_friction=float(vector[13]),
        )
        return ContactCandidate(reference._validate_sphere_specs(spheres), material)

    def regularization(self, encoded: np.ndarray) -> np.ndarray:
        """Return weak dimensionless priors about the official seed."""
        vector = self._validate(encoded)
        scales = np.asarray([*([0.05] * 7), 0.30, 0.30, 0.04, 2.0, 5.0, 1.5, 1.0])
        return 0.02 * (vector - self._x0) / scales / math.sqrt(len(vector))

    def _validate(self, encoded: np.ndarray) -> np.ndarray:
        vector = np.asarray(encoded, dtype=float)
        if vector.shape != self._x0.shape or not np.all(np.isfinite(vector)):
            raise ValueError(f"contact calibration parameters must contain {len(self._x0)} finite values")
        return vector


class PrescribedContactEvaluator:
    """Evaluate candidates using only immutable q/qd and model state."""

    def __init__(
        self,
        model: osim.OsimModel,
        coordinate_names: Sequence[str],
        coordinates: np.ndarray,
        speeds: np.ndarray,
        *,
        device: str,
    ) -> None:
        self._model = copy.deepcopy(model)
        self._coordinate_names = tuple(str(name) for name in coordinate_names)
        self._coordinates = np.array(coordinates, dtype=float, copy=True)
        self._speeds = np.array(speeds, dtype=float, copy=True)
        if self._coordinates.shape != self._speeds.shape or self._coordinates.ndim != 2:
            raise ValueError("coordinates and speeds must have the same [time, coordinate] shape")
        if not np.all(np.isfinite(self._coordinates)) or not np.all(np.isfinite(self._speeds)):
            raise ValueError("prescribed q/qd must be finite")
        self._coordinates.setflags(write=False)
        self._speeds.setflags(write=False)
        self._device = device
        kinematics = osim.ForwardKinematics(self._model, device=device)
        if tuple(kinematics.coordinate_names) != self._coordinate_names:
            raise ValueError("prescribed coordinate order does not match ForwardKinematics")
        self._body_names = tuple(kinematics.body_names)
        self._body_transforms = np.asarray(kinematics.body_transforms_batch(self._coordinates), dtype=float)

    def provenance(self) -> dict[str, Any]:
        """Return evidence that measured loads are absent from this evaluator."""
        return {
            "kind": "prescribed_q_qd_only",
            "measured_loads_available_to_evaluator": False,
            "coordinate_names": self._coordinate_names,
            "q_sha256": _array_sha256(self._coordinates),
            "qd_sha256": _array_sha256(self._speeds),
            "device": self._device,
        }

    def __call__(self, candidate: ContactCandidate) -> ContactEvaluation:
        """Evaluate one candidate."""
        augmented = reference.augment_opensim_compat_model(
            self._model, spheres=candidate.spheres, material=candidate.material
        )
        contact = osim.OpenSimContact(augmented, device=self._device)
        if tuple(contact.coordinate_names) != self._coordinate_names:
            raise ValueError("prescribed coordinate order does not match OpenSimContact")
        body_names, values = contact.body_wrenches(
            self._coordinates, self._speeds, h=_VELOCITY_STENCIL_H_S, frame="opensim"
        )
        body_names = tuple(body_names)
        missing = set(reference._BODY_ORDER) - set(body_names)
        if missing:
            raise ValueError(f"contact evaluation omitted bodies: {sorted(missing)}")
        selected = np.asarray(values, dtype=float)[:, [body_names.index(name) for name in reference._BODY_ORDER]]
        foot = reference.aggregate_body_wrenches(reference._BODY_ORDER, selected)
        body_index = {name: index for index, name in enumerate(self._body_names)}
        penetration = np.empty((len(self._coordinates), len(candidate.spheres)))
        normal_into = np.asarray((0.0, -1.0, 0.0))
        for sphere_index, sphere in enumerate(candidate.spheres):
            transforms = self._body_transforms[:, body_index[sphere.body]]
            center = (transforms @ np.asarray((*sphere.location_m, 1.0)))[:, :3]
            penetration[:, sphere_index] = np.maximum(sphere.radius_m + center @ normal_into, 0.0)
        if not np.all(np.isfinite(foot)) or not np.all(np.isfinite(penetration)):
            raise ValueError("contact evaluation returned non-finite values")
        return ContactEvaluation(foot, penetration)


class ContactObjective:
    """Fixed-length full-wrench objective for prescribed contact calibration."""

    def __init__(
        self,
        times_s: np.ndarray,
        measured_wrenches: np.ndarray,
        measured_contact: np.ndarray,
        parameterization: ContactParameterization,
        evaluator: PrescribedContactEvaluator,
        *,
        fit_sides: Sequence[str] = _SIDE_ORDER,
    ) -> None:
        self.times = np.asarray(times_s, dtype=float)
        self.measured = np.asarray(measured_wrenches, dtype=float)
        self.contact = np.asarray(measured_contact, dtype=bool)
        self.parameterization = parameterization
        self.evaluator = evaluator
        self.fit_indices = tuple(_SIDE_ORDER.index(side) for side in fit_sides)
        if self.measured.shape != (len(self.times), 2, 9) or self.contact.shape != (len(self.times), 2):
            raise ValueError("measured targets must have shapes [time,2,9] and [time,2]")
        if not np.all(np.isfinite(self.measured)):
            raise ValueError("measured target wrenches must be finite")
        self.trace: list[dict[str, Any]] = []

    def __call__(self, encoded: np.ndarray, *, purpose: str = "optimizer") -> np.ndarray:
        """Return the frozen residual vector and record its component sums."""
        candidate = self.parameterization.decode(encoded)
        evaluation = self.evaluator(candidate)
        predicted = evaluation.foot_wrenches[:, self.fit_indices]
        target = self.measured[:, self.fit_indices]
        sample_scale = math.sqrt(predicted.shape[0] * predicted.shape[1])
        force_weights = np.asarray((0.5, 1.0, 0.5))
        force = ((predicted[..., :3] - target[..., :3]) / _BODY_WEIGHT_N / sample_scale * force_weights).ravel()
        peak = 2.0 * (np.max(predicted[..., 1], axis=0) - np.max(target[..., 1], axis=0)) / _BODY_WEIGHT_N
        duration = float(self.times[-1] - self.times[0])
        impulse = (
            2.0
            * (np.trapezoid(predicted[..., 1], self.times, axis=0) - np.trapezoid(target[..., 1], self.times, axis=0))
            / (_BODY_WEIGHT_N * duration)
        )
        braking = (
            np.trapezoid(np.minimum(predicted[..., 0], 0.0), self.times, axis=0)
            - np.trapezoid(np.minimum(target[..., 0], 0.0), self.times, axis=0)
        ) / (_BODY_WEIGHT_N * duration)
        propulsion = (
            np.trapezoid(np.maximum(predicted[..., 0], 0.0), self.times, axis=0)
            - np.trapezoid(np.maximum(target[..., 0], 0.0), self.times, axis=0)
        ) / (_BODY_WEIGHT_N * duration)

        moment = np.cross(predicted[..., 3:6], predicted[..., :3]) + predicted[..., 6:9]
        normal = np.asarray((0.0, 1.0, 0.0))
        safe_vertical = np.maximum(predicted[..., 1], 1.0)
        predicted_cop = np.cross(normal, moment) / safe_vertical[..., None]
        target_force = target[..., :3]
        target_moment = np.cross(target[..., 3:6], target_force) + target[..., 6:9]
        target_cop = np.cross(normal, target_moment) / np.maximum(target_force[..., 1], 1.0)[..., None]
        cop_mask = target_force[..., 1] >= _COP_THRESHOLD_N
        cop = (predicted_cop[..., (0, 2)] - target_cop[..., (0, 2)])[cop_mask] / 0.030
        cop = cop.ravel() / math.sqrt(max(cop.size, 1))
        load_hinge = np.maximum(_COP_THRESHOLD_N - predicted[..., 1][cop_mask], 0.0)
        load_hinge = load_hinge / _BODY_WEIGHT_N / math.sqrt(max(load_hinge.size, 1))
        predicted_free = moment[..., 1] - np.cross(predicted_cop, predicted[..., :3])[..., 1]
        target_free = target_moment[..., 1] - np.cross(target_cop, target_force)[..., 1]
        free = (predicted_free[cop_mask] - target_free[cop_mask]) / (_BODY_WEIGHT_N * _BODY_HEIGHT_M)
        free = 0.5 * free / math.sqrt(max(free.size, 1))
        penetration = np.maximum(evaluation.penetrations_m - _MAX_PENETRATION_M, 0.0)
        penetration = 4.0 * penetration.ravel() / _MAX_PENETRATION_M / math.sqrt(penetration.size)
        regularization = self.parameterization.regularization(encoded)
        terms = {
            "force_waveform": force,
            "vertical_peak": peak,
            "vertical_impulse": impulse,
            "braking_impulse": braking,
            "propulsion_impulse": propulsion,
            "cop": cop,
            "cop_load_hinge": load_hinge,
            "free_moment": free,
            "penetration_above_0_020_m": penetration,
            "regularization": regularization,
        }
        result = np.concatenate(tuple(terms.values()))
        self.trace.append(
            {
                "evaluation_index": len(self.trace),
                "purpose": purpose,
                "parameters": self.parameterization.values(encoded),
                "term_sumsq": {name: float(value @ value) for name, value in terms.items()},
                "residual_sumsq": float(result @ result),
                "maximum_penetration_m": float(np.max(evaluation.penetrations_m)),
            }
        )
        self.last_evaluation = evaluation
        return result


@dataclass(frozen=True, slots=True)
class CalibrationInputs:
    """Hash-validated accepted RRA arrays and model."""

    root: Path
    model_path: Path
    times: np.ndarray
    coordinate_names: tuple[str, ...]
    coordinates: np.ndarray
    speeds: np.ndarray
    measured_wrenches: np.ndarray
    measured_contact: np.ndarray
    manifest: dict[str, Any]


def load_calibration_inputs(root: str | os.PathLike = _DEFAULT_RRA_INPUT) -> CalibrationInputs:
    """Load only an accepted, hash-current RRA-adjusted contact input."""
    directory = Path(root).resolve()
    manifest_path = directory / "manifest.json"
    qc_path = directory / "qc_summary.json"
    analysis_path = directory / "analysis.npz"
    model_path = directory / "S001_scaled.osim"
    for path in (manifest_path, qc_path, analysis_path, model_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = json.loads(manifest_path.read_text())
    qc = json.loads(qc_path.read_text())
    if manifest.get("status") != "production_candidate" or qc.get("status") != "production_candidate":
        raise ValueError("contact calibration requires the accepted RRA production candidate")
    for name, expected in manifest.get("artifacts", {}).items():
        path = directory / name
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"RRA-adjusted input artifact is stale: {name}")
    archive = np.load(analysis_path, allow_pickle=False)
    times = np.asarray(archive["times"], dtype=float)
    coordinates = np.asarray(archive["id_coordinates"], dtype=float)
    speeds = np.asarray(archive["id_speeds"], dtype=float)
    names = tuple(str(value) for value in archive["id_names"])
    force = np.asarray(archive["grf"], dtype=float)
    cop = np.asarray(archive["cop"], dtype=float)
    free = np.asarray(archive["free_torque"], dtype=float)
    contact = np.asarray(archive["contact"], dtype=bool)
    measured = np.zeros((len(times), 2, 9), dtype=float)
    measured[..., :3] = force
    measured[..., 3:6] = np.where(np.isfinite(cop), cop, 0.0)
    measured[..., 6:9] = free
    if coordinates.shape != speeds.shape or coordinates.shape != (len(times), len(names)):
        raise ValueError("accepted RRA state arrays have inconsistent shapes")
    return CalibrationInputs(directory, model_path, times, names, coordinates, speeds, measured, contact, manifest)


def _candidate_dict(candidate: ContactCandidate) -> dict[str, Any]:
    return {"spheres": [asdict(value) for value in candidate.spheres], "material": candidate.material}


def _qc_sidecar(candidate: ContactCandidate) -> predictive_contact.PredictiveContactSidecar:
    material = predictive_contact.MaterialConfig(
        law="SmoothSphereHalfSpaceForce",
        stiffness=candidate.material["stiffness"],
        dissipation=candidate.material["dissipation"],
        static_friction=candidate.material["static_friction"],
        dynamic_friction=candidate.material["dynamic_friction"],
        viscous_friction=candidate.material["viscous_friction"],
        transition_velocity=candidate.material["transition_velocity"],
        constant_contact_force=candidate.material["constant_contact_force"],
        hertz_smoothing=candidate.material["hertz_smoothing"],
        hunt_crossley_smoothing=candidate.material["hunt_crossley_smoothing"],
        bounds={},
    )
    calibration = predictive_contact.CalibrationConfig(
        train_side="left",
        held_out_side="right",
        load_threshold_n=_LOAD_THRESHOLD_N,
        cop_load_threshold_n=_COP_THRESHOLD_N,
        prescribed_time_step_s=0.001,
        objective_weights={},
    )
    return predictive_contact.PredictiveContactSidecar(
        schema_version="qc_adapter_only",
        source_model_path="",
        source_model_sha256="",
        source_analysis_path="",
        source_analysis_sha256="",
        frame=reference._FRAME,
        units=dict(reference._UNITS),
        ground=predictive_contact.GroundConfig("floor", 0.0, 0.0, (0.0, 0.0)),
        material=material,
        spheres=(),
        calibration=calibration,
        normalization=predictive_contact.NormalizationConfig(_BODY_WEIGHT_N, _BODY_HEIGHT_M),
    )


def compute_full_qc(
    inputs: CalibrationInputs,
    candidate: ContactCandidate,
    evaluation: ContactEvaluation,
    *,
    smaller_step_finite: bool,
) -> dict[str, Any]:
    """Apply the existing strict Stage 2 gates to a 12-sphere evaluation."""
    predicted_cop, predicted_free = reference.cop_and_free_moment(
        evaluation.foot_wrenches, load_threshold_n=_COP_THRESHOLD_N
    )
    measured = inputs.measured_wrenches
    measured_cop = measured[..., 3:6].copy()
    measured_cop[~inputs.measured_contact] = np.nan
    measured_free = np.zeros((len(inputs.times), 2, 3))
    measured_free[..., 1] = measured[..., 7]
    return predictive_contact.compute_contact_qc(
        inputs.times,
        measured[..., :3],
        measured_cop,
        measured_free,
        inputs.measured_contact,
        evaluation.foot_wrenches[..., :3],
        predicted_cop,
        predicted_free,
        evaluation.penetrations_m,
        _qc_sidecar(candidate),
        nominal_finite=True,
        smaller_step_finite=smaller_step_finite,
        body_order_valid=True,
    )


def write_diagnostic_report(
    directory: str | os.PathLike,
    times: np.ndarray,
    predicted_wrenches: np.ndarray,
    measured_wrenches: np.ndarray,
    penetrations_m: np.ndarray,
    trace: Sequence[dict[str, Any]],
    qc: dict[str, Any],
    candidate: ContactCandidate,
) -> tuple[str, ...]:
    """Write diagnostic figures and a Markdown run log for one frozen fit."""
    os.environ["MPLBACKEND"] = "Agg"
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    root = Path(directory)
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    t = np.asarray(times, dtype=float)
    predicted = np.asarray(predicted_wrenches, dtype=float)
    measured = np.asarray(measured_wrenches, dtype=float)
    penetration = np.asarray(penetrations_m, dtype=float)
    if predicted.shape != measured.shape or predicted.shape != (len(t), 2, 9):
        raise ValueError("diagnostic wrenches must have shape [time,2,9]")
    if penetration.shape != (len(t), 12):
        raise ValueError("diagnostic penetration must have shape [time,12]")
    time = t - t[0]
    colors = {"measured": "#111111", "predicted": "#0072B2"}
    sides = ("Left", "Right")
    components = ((0, "AP force", "N"), (1, "Vertical force", "N"), (2, "ML force", "N"))

    fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True, constrained_layout=True)
    for side in range(2):
        for row, (component, label, unit) in enumerate(components):
            ax = axes[row, side]
            ax.plot(time, measured[:, side, component], color=colors["measured"], lw=1.8, label="Measured")
            ax.plot(time, predicted[:, side, component], color=colors["predicted"], lw=1.4, label="Predicted")
            ax.axhline(0.0, color="0.75", lw=0.7)
            ax.set_ylabel(f"{label} [{unit}]")
            ax.grid(alpha=0.25)
            if row == 0:
                ax.set_title(f"{sides[side]} foot")
            if row == 2:
                ax.set_xlabel("Time from RRA window start [s]")
    axes[0, 0].legend(loc="upper right")
    fig.suptitle("Measured versus calibrated 12-sphere ground-reaction forces")
    fig.savefig(figures / "grf_tracking.png", dpi=170)
    plt.close(fig)

    predicted_cop, _ = reference.cop_and_free_moment(predicted, load_threshold_n=_COP_THRESHOLD_N)
    measured_cop, _ = reference.cop_and_free_moment(measured, load_threshold_n=_COP_THRESHOLD_N)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for side, ax in enumerate(axes):
        mask = (
            (predicted[:, side, 1] >= _COP_THRESHOLD_N)
            & (measured[:, side, 1] >= _COP_THRESHOLD_N)
            & np.all(np.isfinite(predicted_cop[:, side]), axis=1)
            & np.all(np.isfinite(measured_cop[:, side]), axis=1)
        )
        measured_path = measured_cop[mask, side][:, (0, 2)].copy()
        predicted_path = predicted_cop[mask, side][:, (0, 2)].copy()
        for path in (measured_path, predicted_path):
            jumps = np.flatnonzero(np.linalg.norm(np.diff(path, axis=0), axis=1) > 0.10) + 1
            path[jumps] = np.nan
        ax.plot(measured_path[:, 0], measured_path[:, 1], color=colors["measured"], lw=2.0, label="Measured")
        ax.plot(
            predicted_path[:, 0],
            predicted_path[:, 1],
            color=colors["predicted"],
            lw=1.6,
            label="Predicted",
        )
        ax.scatter(measured_cop[mask, side, 0][::30], measured_cop[mask, side, 2][::30], s=10, color=colors["measured"])
        ax.scatter(
            predicted_cop[mask, side, 0][::30], predicted_cop[mask, side, 2][::30], s=10, color=colors["predicted"]
        )
        rms_mm = 1000.0 * qc["sides"][sides[side].lower()]["cop"]["rms_m"]
        ax.set_title(f"{sides[side]} COP, RMS error {rms_mm:.1f} mm")
        ax.set_xlabel("Anterior position x [m]")
        ax.set_ylabel("Rightward position z [m]")
        ax.axis("equal")
        ax.grid(alpha=0.25)
    axes[0].legend(loc="best")
    fig.suptitle("Loaded-frame center-of-pressure paths (both vertical forces >= 200 N)")
    fig.savefig(figures / "cop_tracking.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, constrained_layout=True)
    for side, ax in enumerate(axes):
        sphere_indices = [index for index, sphere in enumerate(candidate.spheres) if sphere.side == _SIDE_ORDER[side]]
        for index in sphere_indices:
            sphere = candidate.spheres[index]
            ax.plot(time, 1000.0 * penetration[:, index], lw=1.2, label=sphere.role)
        ax.axhline(1000.0 * _MAX_PENETRATION_M, color="#D55E00", ls="--", lw=1.5, label="20 mm limit")
        ax2 = ax.twinx()
        ax2.fill_between(time, 0.0, measured[:, side, 1] >= _LOAD_THRESHOLD_N, color="0.2", alpha=0.08, step="mid")
        ax2.fill_between(
            time, 0.0, predicted[:, side, 1] >= _LOAD_THRESHOLD_N, color=colors["predicted"], alpha=0.08, step="mid"
        )
        ax2.set_ylim(0.0, 1.0)
        ax2.set_yticks([])
        ax.set_ylabel("Sphere penetration [mm]")
        ax.set_title(f"{sides[side]} sphere penetration and contact coverage")
        ax.grid(alpha=0.25)
    axes[1].set_xlabel("Time from RRA window start [s]")
    axes[0].legend(ncol=4, fontsize=8, loc="upper right")
    fig.suptitle("Per-sphere penetration; shaded bands mark measured and predicted contact")
    fig.savefig(figures / "penetration_and_timing.png", dpi=170)
    plt.close(fig)

    objective = np.asarray([item["residual_sumsq"] for item in trace], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].semilogy(np.arange(len(objective)), objective, color="#0072B2", lw=1.5)
    axes[0].set_xlabel("Evaluator call")
    axes[0].set_ylabel("Residual sum of squares")
    axes[0].set_title("Optimization convergence")
    axes[0].grid(alpha=0.25)
    final_terms = trace[-1]["term_sumsq"]
    term_names = list(final_terms)
    term_values = np.asarray([final_terms[name] for name in term_names])
    order = np.argsort(term_values)
    axes[1].barh(np.asarray(term_names)[order], term_values[order], color="#56B4E9")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Final residual sum of squares")
    axes[1].set_title("Final objective composition")
    axes[1].grid(axis="x", alpha=0.25)
    fig.savefig(figures / "optimization_convergence.png", dpi=170)
    plt.close(fig)

    gate_rows: list[tuple[str, bool]] = []
    for side in _SIDE_ORDER:
        for name, passed in qc["sides"][side]["gates"].items():
            gate_rows.append((f"{side}: {name}", bool(passed)))
    for name, passed in qc["global_gates"].items():
        gate_rows.append((f"global: {name}", bool(passed)))
    fig_height = max(6.0, 0.29 * len(gate_rows))
    fig, ax = plt.subplots(figsize=(13, fig_height), constrained_layout=True)
    y = np.arange(len(gate_rows))
    values = np.asarray([1.0 if passed else 0.0 for _, passed in gate_rows])
    ax.barh(y, np.ones_like(values), color=["#009E73" if passed else "#D55E00" for _, passed in gate_rows])
    ax.set_yticks(y, [name for name, _ in gate_rows], fontsize=8)
    ax.set_xticks([])
    ax.invert_yaxis()
    for row, (_, passed) in enumerate(gate_rows):
        ax.text(0.99, row, "PASS" if passed else "FAIL", ha="right", va="center", color="white", fontweight="bold")
    passed_count = int(np.count_nonzero(values))
    ax.set_title(
        f"Contact QC gates: {passed_count}/{len(gate_rows)} pass; overall {'PASS' if qc['passed'] else 'FAIL'}"
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.savefig(figures / "qc_gate_dashboard.png", dpi=170)
    plt.close(fig)

    report = [
        "# Exact 12-sphere contact calibration diagnostic log",
        "",
        f"**Overall prescribed-contact QC:** `{'PASS' if qc['passed'] else 'FAIL'}`",
        "",
        "The figures below compare the frozen Newton contact result against corrected measured platform loads.",
        "Passing numerical parity or individual gates does not imply FD-1.",
        "",
        "## Force tracking",
        "",
        "![Measured versus predicted force](figures/grf_tracking.png)",
        "",
        "## Center of pressure",
        "",
        "![COP tracking](figures/cop_tracking.png)",
        "",
        "## Penetration and timing",
        "",
        "![Penetration and timing](figures/penetration_and_timing.png)",
        "",
        "## Optimization",
        "",
        "![Optimization convergence](figures/optimization_convergence.png)",
        "",
        "## QC gate dashboard",
        "",
        "![QC gate dashboard](figures/qc_gate_dashboard.png)",
        "",
        "## Key frozen metrics",
        "",
    ]
    for side in _SIDE_ORDER:
        metrics = qc["sides"][side]
        report.extend(
            [
                f"### {side.title()}",
                "",
                f"- Vertical peak relative error: {metrics['vertical_force']['peak_relative_error']:.3%}",
                f"- Vertical impulse relative error: {metrics['vertical_force']['impulse_relative_error']:.3%}",
                f"- COP RMS error: {1000.0 * metrics['cop']['rms_m']:.2f} mm",
                f"- AP/ML force RMS: {metrics['horizontal_force']['ap_rms_N']:.2f} / {metrics['horizontal_force']['ml_rms_N']:.2f} N",
                f"- Onset/release error: {1000.0 * metrics['timing']['onset_error_s']:.1f} / {1000.0 * metrics['timing']['release_error_s']:.1f} ms",
                "",
            ]
        )
    report.extend(
        [
            f"- Maximum sphere penetration: {1000.0 * qc['maximum_sphere_penetration_m']:.2f} mm",
            "",
            "## Scope",
            "",
            "This is prescribed-motion contact calibration. It applies no measured load to the contact evaluator and makes no forward-dynamics, FD-1, or cross-trial-generalization claim.",
            "",
        ]
    )
    (root / "calibration_report.md").write_text("\n".join(report), encoding="utf-8")
    plain = [
        f"overall_qc={'PASS' if qc['passed'] else 'FAIL'}",
        "report=calibration_report.md",
        "figures/grf_tracking.png",
        "figures/cop_tracking.png",
        "figures/penetration_and_timing.png",
        "figures/optimization_convergence.png",
        "figures/qc_gate_dashboard.png",
    ]
    (root / "run.log").write_text("\n".join(plain) + "\n", encoding="utf-8")
    return (
        "calibration_report.md",
        "run.log",
        "figures/grf_tracking.png",
        "figures/cop_tracking.png",
        "figures/penetration_and_timing.png",
        "figures/optimization_convergence.png",
        "figures/qc_gate_dashboard.png",
    )


def run_calibration(
    output_dir: str | os.PathLike,
    *,
    rra_input: str | os.PathLike = _DEFAULT_RRA_INPUT,
    device: str = "cuda:0",
    stride: int = 4,
    max_nfev: int = 60,
) -> Path:
    """Fit, validate, and atomically publish the exact 12-sphere contact artifact."""
    from scipy.optimize import least_squares

    output = Path(output_dir).resolve()
    root = Path(__file__).resolve().parents[3]
    if output.exists():
        raise FileExistsError(output)
    if output == root or output.is_relative_to(root):
        raise ValueError("calibration artifacts must be outside the repository")
    if stride < 1 or max_nfev < 1:
        raise ValueError("stride and max_nfev must be positive")
    inputs = load_calibration_inputs(rra_input)
    if output == inputs.root or output.is_relative_to(inputs.root) or inputs.root.is_relative_to(output):
        raise ValueError("output and RRA input directories must not overlap")
    source_model = osim.parse_osim(inputs.model_path)
    model, mtp_repairs = prepare_contact_model(source_model, inputs.coordinate_names, inputs.coordinates)
    seed_spheres = subject_sphere_specs(model)
    parameterization = ContactParameterization(seed_spheres)
    indices = np.arange(0, len(inputs.times), stride)
    evaluator = PrescribedContactEvaluator(
        model,
        inputs.coordinate_names,
        inputs.coordinates[indices],
        inputs.speeds[indices],
        device=device,
    )
    objective = ContactObjective(
        inputs.times[indices],
        inputs.measured_wrenches[indices],
        inputs.measured_contact[indices],
        parameterization,
        evaluator,
    )
    result = least_squares(
        objective,
        parameterization.x0,
        bounds=(parameterization.lower_bounds, parameterization.upper_bounds),
        method="trf",
        jac="2-point",
        diff_step=1.0e-3,
        max_nfev=max_nfev,
    )
    candidate = parameterization.decode(result.x)
    objective(result.x, purpose="frozen_post_fit_optimization_grid")
    full_evaluator = PrescribedContactEvaluator(
        model, inputs.coordinate_names, inputs.coordinates, inputs.speeds, device=device
    )
    full_evaluation = full_evaluator(candidate)

    half_times = np.arange(inputs.times[0], inputs.times[-1] + 0.00025, 0.0005)
    flat_q = np.column_stack(
        [
            np.interp(half_times, inputs.times, inputs.coordinates[:, index])
            for index in range(inputs.coordinates.shape[1])
        ]
    )
    flat_u = np.column_stack(
        [np.interp(half_times, inputs.times, inputs.speeds[:, index]) for index in range(inputs.speeds.shape[1])]
    )
    half_evaluator = PrescribedContactEvaluator(model, inputs.coordinate_names, flat_q, flat_u, device=device)
    half_evaluation = half_evaluator(candidate)
    smaller_finite = bool(
        np.all(np.isfinite(half_evaluation.foot_wrenches)) and np.all(np.isfinite(half_evaluation.penetrations_m))
    )
    qc = compute_full_qc(inputs, candidate, full_evaluation, smaller_step_finite=smaller_finite)
    qc["optimizer_success"] = bool(result.success)
    qc["exact_topology"] = {
        "sphere_count": len(candidate.spheres),
        "role_order": reference._ROLE_ORDER,
        "body_order": reference._BODY_ORDER,
        "radii_unchanged": all(sphere.radius_m == 0.035 for sphere in candidate.spheres),
    }
    qc["fit_information"] = {
        "sides_used": list(_SIDE_ORDER),
        "held_out_trial": False,
        "note": "Both Trial 101 stances fit; cross-trial validation remains required.",
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        _write_json(temporary / "calibrated_contact.json", _candidate_dict(candidate))
        _write_json(temporary / "mtp_range_repairs.json", mtp_repairs)
        write_contact_ready_model(inputs.model_path, temporary / "S001_contact_ready.osim", mtp_repairs)
        (temporary / "ContactGeometrySet.xml").write_bytes(
            reference.xml_bytes(reference.build_contact_geometry_xml(spheres=candidate.spheres))
        )
        official_material = {name: candidate.material[name] for name in reference._MATERIAL}
        (temporary / "ContactForceSet.xml").write_bytes(
            reference.xml_bytes(reference.build_force_xml(spheres=candidate.spheres, material=official_material))
        )
        _write_json(
            temporary / "optimizer_result.json",
            {
                "success": bool(result.success),
                "status": int(result.status),
                "message": str(result.message),
                "cost": float(result.cost),
                "optimality": float(result.optimality),
                "nfev": int(result.nfev),
                "njev": int(result.njev) if result.njev is not None else None,
                "max_nfev": max_nfev,
                "stride": stride,
                "parameter_names": parameterization.names,
                "initial_parameters": parameterization.values(parameterization.x0),
                "final_parameters": parameterization.values(result.x),
                "lower_bounds": parameterization.values(parameterization.lower_bounds),
                "upper_bounds": parameterization.values(parameterization.upper_bounds),
                "active_mask": result.active_mask,
            },
        )
        _write_json(temporary / "evaluation_trace.json", objective.trace)
        _write_json(temporary / "qc_summary.json", qc)
        np.savez_compressed(
            temporary / "evaluation.npz",
            times=inputs.times,
            predicted_foot_wrenches=full_evaluation.foot_wrenches,
            measured_foot_wrenches=inputs.measured_wrenches,
            penetrations_m=full_evaluation.penetrations_m,
            half_step_times=half_times,
            half_step_foot_wrenches=half_evaluation.foot_wrenches,
        )
        diagnostic_artifacts = write_diagnostic_report(
            temporary,
            inputs.times,
            full_evaluation.foot_wrenches,
            inputs.measured_wrenches,
            full_evaluation.penetrations_m,
            objective.trace,
            qc,
            candidate,
        )
        artifacts = (
            "calibrated_contact.json",
            "mtp_range_repairs.json",
            "S001_contact_ready.osim",
            "ContactGeometrySet.xml",
            "ContactForceSet.xml",
            "optimizer_result.json",
            "evaluation_trace.json",
            "qc_summary.json",
            "evaluation.npz",
            *diagnostic_artifacts,
        )
        manifest = {
            "schema_version": _SCHEMA,
            "architecture_role": ARCHITECTURE_ROLE,
            "reference_only": True,
            "production_eligible": False,
            "scope": _SCOPE,
            "status": "prescribed_contact_passed"
            if qc["passed"] and result.success
            else "prescribed_contact_failed_qc",
            "claims": {
                "prescribed_contact_calibration": True,
                "forward_dynamics": False,
                "fd_1": False,
                "cross_trial_generalization": False,
            },
            "source": {
                "rra_input": str(inputs.root),
                "rra_manifest_sha256": _sha256(inputs.root / "manifest.json"),
                "model_sha256": _sha256(inputs.model_path),
                "analysis_sha256": _sha256(inputs.root / "analysis.npz"),
            },
            "evaluator": full_evaluator.provenance(),
            "geometry_seed": {
                "method": "scaled S001 heel/forefoot landmarks with toe-frame transformation",
                "force_targets_used": False,
                "spheres": [asdict(value) for value in seed_spheres],
                "mtp_range_repairs": mtp_repairs,
            },
            "runtime": _runtime_provenance(),
            "artifacts": {name: _sha256(temporary / name) for name in artifacts},
        }
        _write_json(temporary / "manifest.json", manifest)
        os.rename(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def add_official_parity_to_artifact(directory: str | os.PathLike) -> dict[str, Any]:
    """Run and plot official OpenSim parity for a frozen calibrated candidate."""
    official_opensim = importlib.import_module("opensim")

    os.environ["MPLBACKEND"] = "Agg"
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    root = Path(directory).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    inputs = load_calibration_inputs(manifest["source"]["rra_input"])
    archive = np.load(root / "evaluation.npz", allow_pickle=False)
    candidate_data = json.loads((root / "calibrated_contact.json").read_text())
    candidate = ContactCandidate(
        reference._validate_sphere_specs(reference.SphereSpec(**sphere) for sphere in candidate_data["spheres"]),
        {name: float(value) for name, value in candidate_data["material"].items()},
    )
    mtp_repairs = json.loads((root / "mtp_range_repairs.json").read_text())
    write_contact_ready_model(inputs.model_path, root / "S001_contact_ready.osim", mtp_repairs)
    official_material = {name: candidate.material[name] for name in reference._MATERIAL}
    official = reference.evaluate_official_prescribed(
        official_opensim,
        root / "S001_contact_ready.osim",
        inputs.coordinate_names,
        inputs.coordinates,
        inputs.speeds,
        times_s=inputs.times,
        spheres=candidate.spheres,
        material=official_material,
    )
    newton = np.asarray(archive["predicted_foot_wrenches"], dtype=float)
    difference = newton - official.foot_wrenches
    force_error = np.abs(difference[..., :3])
    torque_error = np.abs(difference[..., 6:9])
    force_limit = 1.0e-3 + 1.0e-4 * np.abs(official.foot_wrenches[..., :3])
    torque_limit = 1.0e-4 + 1.0e-4 * np.abs(official.foot_wrenches[..., 6:9])
    summary = {
        "schema_version": "gait_c3d_calibrated_contact_official_parity_1",
        "scope": "frozen_calibrated_contact_official_opensim_vs_newton",
        "passed": bool(np.all(force_error <= force_limit) and np.all(torque_error <= torque_limit)),
        "force": {
            "max_abs_N": float(np.max(force_error)),
            "rms_N": float(np.sqrt(np.mean(np.square(difference[..., :3])))),
            "atol_N": 1.0e-3,
            "rtol": 1.0e-4,
            "passed": bool(np.all(force_error <= force_limit)),
        },
        "torque": {
            "max_abs_Nm": float(np.max(torque_error)),
            "rms_Nm": float(np.sqrt(np.mean(np.square(difference[..., 6:9])))),
            "atol_Nm": 1.0e-4,
            "rtol": 1.0e-4,
            "passed": bool(np.all(torque_error <= torque_limit)),
        },
        "measured_loads_used": False,
        "official_opensim_version": str(official_opensim.GetVersionAndDate()),
    }
    _write_json(root / "official_parity.json", summary)
    np.savez_compressed(
        root / "official_parity.npz",
        times=inputs.times,
        official_foot_wrenches=official.foot_wrenches,
        newton_foot_wrenches=newton,
        difference=difference,
    )
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    time = inputs.times - inputs.times[0]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, constrained_layout=True)
    for side, label in enumerate(("Left", "Right")):
        axes[0, side].plot(time, official.foot_wrenches[:, side, 1], color="#111111", lw=2.1, label="Official OpenSim")
        axes[0, side].plot(time, newton[:, side, 1], color="#E69F00", lw=1.0, ls="--", label="Newton")
        axes[0, side].set_title(f"{label} vertical force")
        axes[0, side].set_ylabel("Force [N]")
        axes[0, side].grid(alpha=0.25)
        axes[1, side].semilogy(
            time,
            np.maximum(np.max(force_error[:, side], axis=1), 1.0e-16),
            color="#0072B2",
            label="Force max component",
        )
        axes[1, side].semilogy(
            time,
            np.maximum(np.max(torque_error[:, side], axis=1), 1.0e-16),
            color="#CC79A7",
            label="Torque max component",
        )
        axes[1, side].set_xlabel("Time from RRA window start [s]")
        axes[1, side].set_ylabel("Absolute difference")
        axes[1, side].grid(alpha=0.25)
    axes[0, 0].legend(loc="best")
    axes[1, 0].legend(loc="best")
    fig.suptitle(
        "Frozen calibrated contact: official OpenSim versus Newton "
        f"({'PASS' if summary['passed'] else 'FAIL'}; max force delta {summary['force']['max_abs_N']:.3g} N)"
    )
    fig.savefig(figures / "official_newton_parity.png", dpi=170)
    plt.close(fig)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["official_parity"] = summary
    manifest.setdefault("artifacts", {})["S001_contact_ready.osim"] = _sha256(root / "S001_contact_ready.osim")
    manifest.setdefault("artifacts", {}).update(
        {
            name: _sha256(root / name)
            for name in (
                "official_parity.json",
                "official_parity.npz",
                "figures/official_newton_parity.png",
            )
        }
    )
    _write_json(root / "manifest.json", manifest)
    report_path = root / "calibration_report.md"
    with report_path.open("a", encoding="utf-8") as stream:
        stream.write(
            "\n## Official OpenSim parity\n\n"
            f"**Parity gate:** `{'PASS' if summary['passed'] else 'FAIL'}`  \n"
            f"Maximum force difference: `{summary['force']['max_abs_N']:.6g} N`  \n"
            f"Maximum torque difference: `{summary['torque']['max_abs_Nm']:.6g} N*m`\n\n"
            "![Official OpenSim versus Newton parity](figures/official_newton_parity.png)\n"
        )
    with (root / "run.log").open("a", encoding="utf-8") as stream:
        stream.write(f"official_parity={'PASS' if summary['passed'] else 'FAIL'}\nfigures/official_newton_parity.png\n")
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["artifacts"]["calibration_report.md"] = _sha256(report_path)
    manifest["artifacts"]["run.log"] = _sha256(root / "run.log")
    _write_json(root / "manifest.json", manifest)
    return summary


def add_diagnostics_to_artifact(directory: str | os.PathLike) -> tuple[str, ...]:
    """Generate figures for an existing completed calibration and refresh its manifest."""
    root = Path(directory).resolve()
    required = (
        root / "evaluation.npz",
        root / "evaluation_trace.json",
        root / "qc_summary.json",
        root / "calibrated_contact.json",
        root / "manifest.json",
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    archive = np.load(root / "evaluation.npz", allow_pickle=False)
    candidate_data = json.loads((root / "calibrated_contact.json").read_text())
    candidate = ContactCandidate(
        reference._validate_sphere_specs(reference.SphereSpec(**sphere) for sphere in candidate_data["spheres"]),
        {name: float(value) for name, value in candidate_data["material"].items()},
    )
    qc = json.loads((root / "qc_summary.json").read_text())
    artifacts = write_diagnostic_report(
        root,
        archive["times"],
        archive["predicted_foot_wrenches"],
        archive["measured_foot_wrenches"],
        archive["penetrations_m"],
        json.loads((root / "evaluation_trace.json").read_text()),
        qc,
        candidate,
    )
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["diagnostics"] = {
        "report": "calibration_report.md",
        "figures": [name for name in artifacts if name.endswith(".png")],
        "overall_qc_passed": qc["passed"],
    }
    manifest.setdefault("artifacts", {}).update({name: _sha256(root / name) for name in artifacts})
    _write_json(root / "manifest.json", manifest)
    return artifacts


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-only",
        action="store_true",
        help="required acknowledgement: this uses OpenSim-shaped compatibility contact",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--add-diagnostics",
        action="store_true",
        help="add figures and a Markdown log to an existing --output-dir",
    )
    parser.add_argument(
        "--official-parity",
        action="store_true",
        help="run official OpenSim parity and add its figure to an existing artifact",
    )
    parser.add_argument("--rra-input", default=str(_DEFAULT_RRA_INPUT))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--max-nfev", type=int, default=60)
    return parser


def main() -> None:
    """Run the command-line contact calibration."""
    parser = _build_parser()
    args = parser.parse_args()
    if not args.reference_only:
        parser.error("--reference-only is required; use newton_contact_calibration for production contact")
    if args.add_diagnostics:
        print("\n".join(add_diagnostics_to_artifact(args.output_dir)))
        if args.official_parity:
            print(json.dumps(add_official_parity_to_artifact(args.output_dir), indent=2))
    elif args.official_parity:
        print(json.dumps(add_official_parity_to_artifact(args.output_dir), indent=2))
    else:
        print(
            run_calibration(
                args.output_dir,
                rra_input=args.rra_input,
                device=args.device,
                stride=args.stride,
                max_nfev=args.max_nfev,
            )
        )


if __name__ == "__main__":
    main()
