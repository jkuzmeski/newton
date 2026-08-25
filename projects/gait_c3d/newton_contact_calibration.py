# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Calibrate S001 foot contact through Newton's neutral contact pipeline.

This runtime intentionally does not import ``newton.opensim``. It consumes the
one-time converted body-pose artifact and uses only ``newton.ModelBuilder``,
``newton.Model``, ``newton.State``, ``newton.CollisionPipeline``,
``newton.Contacts``, and ``newton.solvers.SolverSemiImplicit``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverSemiImplicit

ARCHITECTURE_ROLE = "native_runtime"

_SCHEMA = "gait_c3d_newton_native_contact_calibration_1"
_DEFAULT_INPUT = Path("/home/jo31399/newton-data/gait/processed/trial_101/newton_contact_input_v1")
_BODY_ORDER = ("calcn_l", "toes_l", "calcn_r", "toes_r")
_SIDE_ORDER = ("left", "right")
_ROLE_ORDER = (
    "heel",
    "lateralRearfoot",
    "lateralMidfoot",
    "medialMidfoot",
    "lateralToe",
    "medialToe",
)
_BODY_WEIGHT_N = 803.5
_BODY_HEIGHT_M = 1.695898298375747
_LOAD_THRESHOLD_N = 50.0
_COP_THRESHOLD_N = 200.0
_MAX_PENETRATION_M = 0.020


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
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
    path.write_text(json.dumps(_json_value(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


@dataclass(frozen=True, slots=True)
class NativeContactInput:
    """Neutral Newton body motion, targets, and topology."""

    root: Path
    times: np.ndarray
    body_pose: np.ndarray
    body_velocity: np.ndarray
    measured_wrenches: np.ndarray
    measured_contact: np.ndarray
    topology: dict[str, Any]
    manifest: dict[str, Any]

    def subset(self, indices: np.ndarray) -> NativeContactInput:
        """Return a frame subset while preserving the neutral topology."""
        return NativeContactInput(
            self.root,
            self.times[indices],
            self.body_pose[indices],
            self.body_velocity[indices],
            self.measured_wrenches[indices],
            self.measured_contact[indices],
            self.topology,
            self.manifest,
        )


def load_native_contact_input(root: str | os.PathLike = _DEFAULT_INPUT) -> NativeContactInput:
    """Load and validate a source-converted neutral Newton artifact."""
    directory = Path(root).resolve()
    manifest_path = directory / "manifest.json"
    topology_path = directory / "topology.json"
    motion_path = directory / "motion_and_targets.npz"
    for path in (manifest_path, topology_path, motion_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != "gait_c3d_newton_contact_input_1":
        raise ValueError("unexpected neutral Newton contact-input schema")
    if manifest.get("architecture", {}).get("opensim_used_after_this_boundary") is not False:
        raise ValueError("contact input does not seal the OpenSim adapter boundary")
    for name, expected in manifest["artifacts"].items():
        if _sha256(directory / name) != expected:
            raise ValueError(f"neutral contact input is stale: {name}")
    topology = json.loads(topology_path.read_text())
    if tuple(topology["body_order"]) != _BODY_ORDER or len(topology["spheres"]) != 12:
        raise ValueError("neutral topology must contain four ordered bodies and 12 spheres")
    archive = np.load(motion_path, allow_pickle=False)
    times = np.asarray(archive["times"], dtype=float)
    pose = np.asarray(archive["body_pose"], dtype=float)
    velocity = np.asarray(archive["body_velocity"], dtype=float)
    measured = np.asarray(archive["measured_foot_wrenches"], dtype=float)
    contact = np.asarray(archive["measured_contact"], dtype=bool)
    if pose.shape != (len(times), 4, 7) or velocity.shape != (len(times), 4, 6):
        raise ValueError("neutral body state arrays have invalid shapes")
    if measured.shape != (len(times), 2, 9) or contact.shape != (len(times), 2):
        raise ValueError("neutral measured-target arrays have invalid shapes")
    if not all(np.all(np.isfinite(value)) for value in (times, pose, velocity, measured)):
        raise ValueError("neutral contact input must be finite")
    return NativeContactInput(directory, times, pose, velocity, measured, contact, topology, manifest)


@dataclass(frozen=True, slots=True)
class NativeCandidate:
    """Neutral sphere geometry and Newton ShapeConfig material parameters."""

    spheres: tuple[dict[str, Any], ...]
    ke: float
    kd: float
    kf: float
    mu: float


class NativeParameterization:
    """Bounded geometry and native Newton material parameterization."""

    def __init__(self, topology: dict[str, Any]) -> None:
        self._topology = topology
        self.names = (
            *(f"{role}_shared_y_offset_m" for role in _ROLE_ORDER),
            "right_foot_y_offset_m",
            "fore_aft_scale",
            "medio_lateral_scale",
            "fore_aft_offset_m",
            "log10_ke",
            "kd",
            "kf",
            "mu",
        )
        self.x0 = np.asarray([*([0.0] * 7), 1.0, 1.0, 0.0, 4.0, 0.0, 500.0, 0.5])
        self.lower = np.asarray([*([-0.06] * 6), -0.06, 0.65, 0.65, -0.05, 3.0, 0.0, 0.0, 0.0])
        self.upper = np.asarray([*([0.06] * 6), 0.06, 1.35, 1.35, 0.05, 5.5, 2000.0, 5000.0, 1.5])

    def values(self, encoded: np.ndarray) -> dict[str, float]:
        vector = self._validate(encoded)
        return {name: float(value) for name, value in zip(self.names, vector, strict=True)}

    def decode(self, encoded: np.ndarray) -> NativeCandidate:
        vector = self._validate(encoded)
        if np.any(vector < self.lower) or np.any(vector > self.upper):
            raise ValueError("native contact parameters are outside their bounds")
        role_y = dict(zip(_ROLE_ORDER, vector[:6], strict=True))
        spheres = []
        for source in self._topology["spheres"]:
            sphere = dict(source)
            x, y, z = (float(value) for value in source["location_m"])
            sphere["location_m"] = [
                float(vector[7] * x + vector[9]),
                float(y + role_y[source["role"]] + (vector[6] if source["side"] == "right" else 0.0)),
                float(vector[8] * z),
            ]
            spheres.append(sphere)
        return NativeCandidate(
            tuple(spheres), 10.0 ** float(vector[10]), float(vector[11]), float(vector[12]), float(vector[13])
        )

    def regularization(self, encoded: np.ndarray) -> np.ndarray:
        vector = self._validate(encoded)
        scales = np.asarray([*([0.06] * 7), 0.35, 0.35, 0.05, 2.5, 2000.0, 5000.0, 1.5])
        return 0.01 * (vector - self.x0) / scales / math.sqrt(len(vector))

    def _validate(self, encoded: np.ndarray) -> np.ndarray:
        vector = np.asarray(encoded, dtype=float)
        if vector.shape != self.x0.shape or not np.all(np.isfinite(vector)):
            raise ValueError(f"native contact parameters must contain {len(self.x0)} finite values")
        return vector


@dataclass(frozen=True, slots=True)
class NativeEvaluation:
    """Newton-native foot wrenches, penetration, and timing."""

    foot_wrenches: np.ndarray
    penetrations_m: np.ndarray
    contact_count: int
    timings_s: dict[str, float]


def _rotate_local(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate one local vector by batches of xyzw quaternions."""
    xyz = quaternion[..., :3]
    w = quaternion[..., 3:4]
    broadcast = np.broadcast_to(vector, xyz.shape)
    return broadcast + 2.0 * np.cross(xyz, np.cross(xyz, broadcast) + w * broadcast)


class NewtonContactEvaluator:
    """Evaluate prescribed foot contact through Newton's public core interfaces."""

    def __init__(self, inputs: NativeContactInput, *, device: str = "cuda:0") -> None:
        self.inputs = inputs
        self.device = device

    def __call__(self, candidate: NativeCandidate) -> NativeEvaluation:
        """Build and evaluate one native Newton contact candidate."""
        cfg = newton.ModelBuilder.ShapeConfig(
            ke=candidate.ke,
            kd=candidate.kd,
            kf=candidate.kf,
            mu=candidate.mu,
            density=1000.0,
        )
        blueprint = newton.ModelBuilder(up_axis=newton.Axis.Z)
        bodies = {
            name: blueprint.add_link(
                mass=1.0,
                inertia=wp.mat33(1.0),
                lock_inertia=True,
                is_kinematic=True,
                label=name,
            )
            for name in _BODY_ORDER
        }
        sphere_shapes = []
        for sphere in candidate.spheres:
            sphere_shapes.append(
                blueprint.add_shape_sphere(
                    bodies[sphere["body"]],
                    xform=wp.transform(tuple(sphere["location_m"]), wp.quat_identity()),
                    radius=float(sphere["radius_m"]),
                    cfg=cfg,
                    label=sphere["name"],
                )
            )
        for first, shape_a in enumerate(sphere_shapes):
            for shape_b in sphere_shapes[first + 1 :]:
                blueprint.add_shape_collision_filter_pair(shape_a, shape_b)

        scene = newton.ModelBuilder(up_axis=newton.Axis.Z)
        scene.add_ground_plane(cfg=cfg)
        for frame in range(len(self.inputs.times)):
            scene.add_world(blueprint, label_prefix=f"frame_{frame}")
        start = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            model = scene.finalize(device=self.device)
        finalize_s = time.perf_counter() - start
        state = model.state()
        state_out = model.state()
        state.body_q.assign(self.inputs.body_pose.reshape(-1, 7).astype(np.float32))
        state.body_qd.assign(self.inputs.body_velocity.reshape(-1, 6).astype(np.float32))
        state.clear_forces()
        pipeline = newton.CollisionPipeline(model, broad_phase="explicit")
        contacts = pipeline.contacts()
        start = time.perf_counter()
        pipeline.collide(state, contacts)
        wp.synchronize_device(model.device)
        collide_s = time.perf_counter() - start
        solver = SolverSemiImplicit(model, angular_damping=0.0, friction_smoothing=0.01)
        start = time.perf_counter()
        solver.step(state, state_out, None, contacts, 1.0e-4)
        wp.synchronize_device(model.device)
        solve_s = time.perf_counter() - start
        body_force = state.body_f.numpy().reshape(len(self.inputs.times), 4, 6).astype(float)
        foot = np.zeros((len(self.inputs.times), 2, 9), dtype=float)
        positions = self.inputs.body_pose[..., :3]
        for side, body_indices in enumerate(((0, 1), (2, 3))):
            for body in body_indices:
                force = body_force[:, body, :3]
                torque = body_force[:, body, 3:]
                foot[:, side, :3] += force
                foot[:, side, 6:] += torque + np.cross(positions[:, body], force)

        body_index = {name: index for index, name in enumerate(_BODY_ORDER)}
        penetration = np.empty((len(self.inputs.times), 12), dtype=float)
        for sphere_index, sphere in enumerate(candidate.spheres):
            index = body_index[sphere["body"]]
            center = self.inputs.body_pose[:, index, :3] + _rotate_local(
                self.inputs.body_pose[:, index, 3:], np.asarray(sphere["location_m"], dtype=float)
            )
            penetration[:, sphere_index] = np.maximum(float(sphere["radius_m"]) - center[:, 2], 0.0)
        if not np.all(np.isfinite(foot)) or not np.all(np.isfinite(penetration)):
            raise ValueError("Newton contact evaluation returned non-finite values")
        return NativeEvaluation(
            foot,
            penetration,
            int(contacts.rigid_contact_count.numpy()[0]),
            {"finalize": finalize_s, "collide": collide_s, "solve": solve_s},
        )


class NativeObjective:
    """Full-wrench objective for the native Newton contact model."""

    def __init__(
        self, inputs: NativeContactInput, parameterization: NativeParameterization, evaluator: NewtonContactEvaluator
    ):
        self.inputs = inputs
        self.parameterization = parameterization
        self.evaluator = evaluator
        self.trace: list[dict[str, Any]] = []
        self.last_evaluation: NativeEvaluation | None = None

    def __call__(self, encoded: np.ndarray, *, purpose: str = "optimizer") -> np.ndarray:
        candidate = self.parameterization.decode(encoded)
        evaluation = self.evaluator(candidate)
        predicted = evaluation.foot_wrenches
        target = self.inputs.measured_wrenches
        sample_scale = math.sqrt(len(predicted) * 2)
        force_weights = np.asarray((0.5, 0.5, 1.0))
        force = ((predicted[..., :3] - target[..., :3]) / _BODY_WEIGHT_N / sample_scale * force_weights).ravel()
        peak = 2.0 * (np.max(predicted[..., 2], axis=0) - np.max(target[..., 2], axis=0)) / _BODY_WEIGHT_N
        duration = float(self.inputs.times[-1] - self.inputs.times[0])
        impulse = (
            3.0
            * (
                np.trapezoid(predicted[..., 2], self.inputs.times, axis=0)
                - np.trapezoid(target[..., 2], self.inputs.times, axis=0)
            )
            / (_BODY_WEIGHT_N * duration)
        )
        braking = (
            np.trapezoid(np.minimum(predicted[..., 0], 0.0), self.inputs.times, axis=0)
            - np.trapezoid(np.minimum(target[..., 0], 0.0), self.inputs.times, axis=0)
        ) / (_BODY_WEIGHT_N * duration)
        propulsion = (
            np.trapezoid(np.maximum(predicted[..., 0], 0.0), self.inputs.times, axis=0)
            - np.trapezoid(np.maximum(target[..., 0], 0.0), self.inputs.times, axis=0)
        ) / (_BODY_WEIGHT_N * duration)
        normal = np.asarray((0.0, 0.0, 1.0))
        predicted_moment = np.cross(predicted[..., 3:6], predicted[..., :3]) + predicted[..., 6:9]
        target_moment = np.cross(target[..., 3:6], target[..., :3]) + target[..., 6:9]
        predicted_cop = np.cross(normal, predicted_moment) / np.maximum(predicted[..., 2], 1.0)[..., None]
        target_cop = np.cross(normal, target_moment) / np.maximum(target[..., 2], 1.0)[..., None]
        mask = target[..., 2] >= _COP_THRESHOLD_N
        cop = (predicted_cop[..., :2] - target_cop[..., :2])[mask] / 0.030
        cop = cop.ravel() / math.sqrt(max(cop.size, 1))
        load_hinge = np.maximum(_COP_THRESHOLD_N - predicted[..., 2][mask], 0.0)
        load_hinge = load_hinge / _BODY_WEIGHT_N / math.sqrt(max(load_hinge.size, 1))
        predicted_free = predicted_moment[..., 2] - np.cross(predicted_cop, predicted[..., :3])[..., 2]
        target_free = target_moment[..., 2] - np.cross(target_cop, target[..., :3])[..., 2]
        free = 0.5 * (predicted_free[mask] - target_free[mask]) / (_BODY_WEIGHT_N * _BODY_HEIGHT_M)
        free = free / math.sqrt(max(free.size, 1))
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
                "native_timings_s": evaluation.timings_s,
                "contact_count": evaluation.contact_count,
            }
        )
        self.last_evaluation = evaluation
        return result


def _coarse_initial_point(objective: NativeObjective, parameterization: NativeParameterization) -> np.ndarray:
    """Select a deterministic contact-engaging seed before local optimization."""
    candidates = []
    for common_y in (0.0, -0.015, -0.030, -0.045):
        for log_ke in (3.7, 4.0, 4.3):
            encoded = parameterization.x0.copy()
            encoded[:6] = common_y
            encoded[10] = log_ke
            residual = objective(encoded, purpose="coarse_seed")
            candidates.append((float(residual @ residual), encoded))
    return min(candidates, key=lambda item: item[0])[1]


def run_native_calibration(
    output_dir: str | os.PathLike,
    *,
    input_dir: str | os.PathLike = _DEFAULT_INPUT,
    device: str = "cuda:0",
    stride: int = 4,
    max_nfev: int = 40,
) -> Path:
    """Optimize and publish a Newton-core prescribed-contact artifact."""
    from scipy.optimize import least_squares

    output = Path(output_dir).resolve()
    repository = Path(__file__).resolve().parents[2]
    if output.exists():
        raise FileExistsError(output)
    if output == repository or output.is_relative_to(repository):
        raise ValueError("native calibration outputs must remain outside the repository")
    full_input = load_native_contact_input(input_dir)
    indices = np.arange(0, len(full_input.times), stride)
    fit_input = full_input.subset(indices)
    parameterization = NativeParameterization(full_input.topology)
    fit_evaluator = NewtonContactEvaluator(fit_input, device=device)
    objective = NativeObjective(fit_input, parameterization, fit_evaluator)
    initial = _coarse_initial_point(objective, parameterization)
    result = least_squares(
        objective,
        initial,
        bounds=(parameterization.lower, parameterization.upper),
        method="trf",
        jac="2-point",
        diff_step=1.0e-3,
        max_nfev=max_nfev,
    )
    candidate = parameterization.decode(result.x)
    full_evaluation = NewtonContactEvaluator(full_input, device=device)(candidate)
    predicted = full_evaluation.foot_wrenches
    target = full_input.measured_wrenches
    vertical = {
        side: {
            "peak_relative_error": float(
                abs(np.max(predicted[:, index, 2]) - np.max(target[:, index, 2])) / np.max(target[:, index, 2])
            ),
            "impulse_relative_error": float(
                abs(
                    np.trapezoid(predicted[:, index, 2], full_input.times)
                    - np.trapezoid(target[:, index, 2], full_input.times)
                )
                / np.trapezoid(target[:, index, 2], full_input.times)
            ),
            "ap_rms_N": float(np.sqrt(np.mean(np.square(predicted[:, index, 0] - target[:, index, 0])))),
            "ml_rms_N": float(np.sqrt(np.mean(np.square(predicted[:, index, 1] - target[:, index, 1])))),
        }
        for index, side in enumerate(_SIDE_ORDER)
    }
    qc = {
        "schema_version": _SCHEMA,
        "scope": "newton_native_prescribed_contact_not_forward_dynamics",
        "optimizer_success": bool(result.success),
        "all_finite": bool(np.all(np.isfinite(predicted))),
        "maximum_penetration_m": float(np.max(full_evaluation.penetrations_m)),
        "maximum_penetration_below_0_020_m": bool(np.max(full_evaluation.penetrations_m) < _MAX_PENETRATION_M),
        "vertical_and_horizontal": vertical,
        "native_runtime": {
            "types": ["newton.Model", "newton.State", "newton.Contacts"],
            "builder_methods": ["add_link", "add_shape_sphere", "add_ground_plane", "add_world"],
            "collision": "newton.CollisionPipeline.collide",
            "response": "newton.solvers.SolverSemiImplicit.step",
            "opensim_runtime_calls": False,
        },
        "full_evaluation_timings_s": full_evaluation.timings_s,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        _write_json(
            temporary / "calibrated_newton_contact.json",
            {
                "spheres": candidate.spheres,
                "shape_config": {"ke": candidate.ke, "kd": candidate.kd, "kf": candidate.kf, "mu": candidate.mu},
            },
        )
        _write_json(
            temporary / "optimizer_result.json",
            {
                "success": bool(result.success),
                "message": str(result.message),
                "cost": float(result.cost),
                "optimality": float(result.optimality),
                "nfev": int(result.nfev),
                "parameter_names": parameterization.names,
                "initial_parameters": parameterization.values(initial),
                "final_parameters": parameterization.values(result.x),
                "lower_bounds": parameterization.values(parameterization.lower),
                "upper_bounds": parameterization.values(parameterization.upper),
                "active_mask": result.active_mask,
            },
        )
        _write_json(temporary / "evaluation_trace.json", objective.trace)
        _write_json(temporary / "qc_summary.json", qc)
        np.savez_compressed(
            temporary / "evaluation.npz",
            times=full_input.times,
            predicted_foot_wrenches=predicted,
            measured_foot_wrenches=target,
            penetrations_m=full_evaluation.penetrations_m,
        )
        artifacts = (
            "calibrated_newton_contact.json",
            "optimizer_result.json",
            "evaluation_trace.json",
            "qc_summary.json",
            "evaluation.npz",
        )
        manifest = {
            "schema_version": _SCHEMA,
            "status": "native_contact_calibrated_pending_full_qc",
            "scope": "newton_native_prescribed_contact_not_forward_dynamics",
            "architecture": qc["native_runtime"],
            "source": {
                "input_dir": str(full_input.root),
                "input_manifest_sha256": _sha256(full_input.root / "manifest.json"),
            },
            "claims": {"newton_native_contact": True, "forward_dynamics": False, "fd_1": False},
            "artifacts": {name: _sha256(temporary / name) for name in artifacts},
        }
        _write_json(temporary / "manifest.json", manifest)
        os.rename(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--input-dir", default=str(_DEFAULT_INPUT))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--max-nfev", type=int, default=40)
    args = parser.parse_args()
    print(
        run_native_calibration(
            args.output_dir, input_dir=args.input_dir, device=args.device, stride=args.stride, max_nfev=args.max_nfev
        )
    )


if __name__ == "__main__":
    main()
