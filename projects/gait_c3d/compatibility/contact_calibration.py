# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""OFFLINE COMPATIBILITY REFERENCE ONLY. Production contact calibration uses ``newton_contact_calibration`` and neutral Newton core APIs.

Fit a bounded first normal-contact model under prescribed gait motion.

This module is intentionally narrower than the complete Stage 2 contract. It
fits only ground height, four bilateral role-shared vertical center offsets,
and log10 stiffness. It uses only the declared training side in the objective.
The other side is evaluated once the optimum is frozen. Horizontal force, COP,
free moment, contact timing, parameter sensitivity, and forward dynamics remain
outside this preliminary fit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

import newton.opensim as osim
from projects.gait_c3d.compatibility import predictive_contact

ARCHITECTURE_ROLE = "compatibility_reference"

_SCHEMA = "gait_c3d_preliminary_normal_contact_calibration_1"
_SCOPE = "preliminary_stage_2_normal_contact_only"
_SIDES = ("left", "right")
_ROLES = ("heel", "medial_forefoot", "lateral_forefoot", "toe")
_GROUND_RELATIVE_BOUNDS_M = (-0.02, 0.02)
_CENTER_VERTICAL_OFFSET_BOUNDS_M = (-0.03, 0.03)
_STIFFNESS_BOUNDS = (1.0e5, 5.0e7)
_MAX_PENETRATION_M = 0.02
_PENETRATION_SUMSQ_WEIGHT = 100.0
_VELOCITY_STENCIL_H_S = 1.0e-6
_DEFAULT_DATA = Path("/home/jo31399/newton-data/gait/processed/trial_101/latest")


def _sha256(path: Path) -> str:
    """Return a file SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    """Return a stable digest of one numeric array and its metadata."""
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _runtime_provenance() -> dict[str, Any]:
    """Record the exact repository, code, and dependency-lock state."""
    repository_root = Path(__file__).resolve().parents[2]

    def git(*args: str) -> str:
        completed = subprocess.run(
            ("git", "-C", str(repository_root), *args),
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    try:
        git_head = git("rev-parse", "HEAD")
        git_status = git("status", "--porcelain=v1", "--untracked-files=all").splitlines()
    except (OSError, subprocess.CalledProcessError):
        git_head = None
        git_status = None
    lock_path = repository_root / "uv.lock"
    code_path = Path(__file__).resolve()
    return {
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "git": {
            "head_sha": git_head,
            "working_tree_dirty": None if git_status is None else bool(git_status),
            "status_porcelain": git_status,
        },
        "code": {
            "path": str(code_path),
            "repository_relative_path": str(code_path.relative_to(repository_root)),
            "sha256": _sha256(code_path),
        },
        "dependency_lock": {
            "path": str(lock_path),
            "sha256": _sha256(lock_path) if lock_path.is_file() else None,
        },
    }


def _json_value(value: Any) -> Any:
    """Convert NumPy and optimizer values to strict JSON values."""
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("calibration artifacts must not contain nonfinite JSON values")
    return value


def _write_json(path: Path, value: Any) -> None:
    """Write deterministic strict JSON."""
    path.write_text(
        json.dumps(_json_value(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


@dataclass(frozen=True, slots=True)
class NormalContactEvaluation:
    """One normal-contact evaluator result.

    Attributes:
        vertical_force_n: Predicted vertical force, shape ``[time, 2]`` in
            ``[left, right]`` order [N].
        sphere_penetration_m: Sphere penetration, shape ``[time, sphere]`` [m].
    """

    vertical_force_n: np.ndarray
    sphere_penetration_m: np.ndarray


class NormalContactParameterization:
    """Encode the six frozen parameters of the preliminary normal-contact fit."""

    def __init__(self, sidecar: predictive_contact.PredictiveContactSidecar):
        self.sidecar = sidecar
        if sidecar.ground.height_bounds_m != _GROUND_RELATIVE_BOUNDS_M:
            raise ValueError("ground bounds do not match the frozen +/-20 mm contract")
        if sidecar.material.bounds.get("stiffness") != _STIFFNESS_BOUNDS:
            raise ValueError("stiffness bounds do not match the frozen Stage 2 contract")

        offsets: dict[str, float] = {}
        for role in _ROLES:
            role_spheres = [sphere for sphere in sidecar.spheres if sphere.role == role]
            if {sphere.side for sphere in role_spheres} != set(_SIDES) or len(role_spheres) != 2:
                raise ValueError(f"role {role!r} must have exactly one sphere on each side")
            role_offsets = []
            for sphere in role_spheres:
                if sphere.center_displacement_bounds_m != _CENTER_VERTICAL_OFFSET_BOUNDS_M:
                    raise ValueError("sphere center bounds do not match the frozen +/-30 mm contract")
                displacement = np.asarray(sphere.center_m) - np.asarray(sphere.geometry_seed_m)
                if not np.allclose(displacement[(0, 2),], 0.0, rtol=0.0, atol=1.0e-12):
                    raise ValueError("the preliminary fit permits only vertical center displacement")
                role_offsets.append(float(displacement[1]))
            if not np.isclose(role_offsets[0], role_offsets[1], rtol=0.0, atol=1.0e-12):
                raise ValueError("initial vertical center offsets must be shared bilaterally by role")
            offsets[role] = role_offsets[0]

        self._x0 = np.asarray(
            [
                sidecar.ground.height_m,
                *(offsets[role] for role in _ROLES),
                np.log10(sidecar.material.stiffness),
            ],
            dtype=float,
        )
        self._lower = np.asarray(
            [
                sidecar.ground.platform_height_m + _GROUND_RELATIVE_BOUNDS_M[0],
                *(_CENTER_VERTICAL_OFFSET_BOUNDS_M[0] for _ in _ROLES),
                np.log10(_STIFFNESS_BOUNDS[0]),
            ],
            dtype=float,
        )
        self._upper = np.asarray(
            [
                sidecar.ground.platform_height_m + _GROUND_RELATIVE_BOUNDS_M[1],
                *(_CENTER_VERTICAL_OFFSET_BOUNDS_M[1] for _ in _ROLES),
                np.log10(_STIFFNESS_BOUNDS[1]),
            ],
            dtype=float,
        )
        if np.any(self._x0 < self._lower) or np.any(self._x0 > self._upper):
            raise ValueError("initial normal-contact parameters are outside the frozen bounds")

    @property
    def names(self) -> tuple[str, ...]:
        """Return optimizer parameter names in encoded order."""
        return (
            "ground_height_m",
            *(f"{role}_shared_vertical_center_offset_m" for role in _ROLES),
            "log10_stiffness",
        )

    @property
    def x0(self) -> np.ndarray:
        """Return a copy of the encoded initial point."""
        return self._x0.copy()

    @property
    def lower_bounds(self) -> np.ndarray:
        """Return a copy of the exact lower bounds."""
        return self._lower.copy()

    @property
    def upper_bounds(self) -> np.ndarray:
        """Return a copy of the exact upper bounds."""
        return self._upper.copy()

    def values(self, encoded: np.ndarray) -> dict[str, float]:
        """Return named physical/encoded optimizer values."""
        vector = self._validate_vector(encoded)
        return {name: float(value) for name, value in zip(self.names, vector, strict=True)}

    def decode(self, encoded: np.ndarray) -> predictive_contact.PredictiveContactSidecar:
        """Build a candidate sidecar while keeping geometry seeds immutable."""
        vector = self._validate_vector(encoded)
        if np.any(vector < self._lower) or np.any(vector > self._upper):
            raise ValueError("encoded parameters are outside the frozen bounds")
        offsets = dict(zip(_ROLES, vector[1:5], strict=True))
        spheres = tuple(
            replace(
                sphere,
                center_m=(
                    sphere.geometry_seed_m[0],
                    sphere.geometry_seed_m[1] + float(offsets[sphere.role]),
                    sphere.geometry_seed_m[2],
                ),
            )
            for sphere in self.sidecar.spheres
        )
        return replace(
            self.sidecar,
            ground=replace(self.sidecar.ground, height_m=float(vector[0])),
            material=replace(self.sidecar.material, stiffness=float(10.0 ** vector[5])),
            spheres=spheres,
        )

    @staticmethod
    def _validate_vector(encoded: np.ndarray) -> np.ndarray:
        """Require one finite six-parameter vector."""
        vector = np.asarray(encoded, dtype=float)
        if vector.shape != (6,) or not np.all(np.isfinite(vector)):
            raise ValueError("encoded normal-contact parameters must contain six finite values")
        return vector


def _coerce_evaluation(
    value: Any,
    ntime: int,
    nsphere: int,
    required_force_index: int,
    required_sphere_indices: list[int],
) -> NormalContactEvaluation:
    """Validate shapes and only the evaluator values required by the fit."""
    if isinstance(value, Mapping):
        value = NormalContactEvaluation(
            vertical_force_n=np.asarray(value["vertical_force_n"]),
            sphere_penetration_m=np.asarray(value["sphere_penetration_m"]),
        )
    if not isinstance(value, NormalContactEvaluation):
        raise TypeError("evaluator must return NormalContactEvaluation or an equivalent mapping")
    force = np.asarray(value.vertical_force_n, dtype=float)
    penetration = np.asarray(value.sphere_penetration_m, dtype=float)
    if force.shape != (ntime, 2):
        raise ValueError("evaluator vertical_force_n must have shape [time, 2]")
    if penetration.shape != (ntime, nsphere):
        raise ValueError("evaluator sphere_penetration_m must have shape [time, sphere]")
    required_force = force[:, required_force_index]
    required_penetration = penetration[:, required_sphere_indices]
    if not np.all(np.isfinite(required_force)) or not np.all(np.isfinite(required_penetration)):
        raise ValueError("training-side normal-contact evaluator results must be finite")
    if np.any(required_penetration < 0.0):
        raise ValueError("training-side sphere penetration must be nonnegative")
    return NormalContactEvaluation(force, penetration)


def _side_metrics(
    side: str,
    split: str,
    times: np.ndarray,
    target: np.ndarray,
    evaluation: NormalContactEvaluation,
    sidecar: predictive_contact.PredictiveContactSidecar,
) -> dict[str, Any]:
    """Compute metrics, or report invalid frozen held-out values without coercion."""
    side_index = _SIDES.index(side)
    predicted = evaluation.vertical_force_n[:, side_index]
    target_side = target[:, side_index]
    sphere_indices = [index for index, sphere in enumerate(sidecar.spheres) if sphere.side == side]
    penetration = evaluation.sphere_penetration_m[:, sphere_indices]
    validity = {
        "measured_vertical_force_finite": bool(np.all(np.isfinite(target_side))),
        "predicted_vertical_force_finite": bool(np.all(np.isfinite(predicted))),
        "sphere_penetration_finite": bool(np.all(np.isfinite(penetration))),
        "sphere_penetration_nonnegative": bool(np.all(penetration >= 0.0)),
    }
    valid = all(validity.values())
    metrics: dict[str, Any] = {
        "scope": _SCOPE,
        "split": split,
        "side": side,
        "used_by_optimizer": split == "fit",
        "status": "valid" if valid else "invalid_frozen_evaluation",
        "valid": valid,
        "validity": validity,
    }
    if not valid:
        metrics["invalid_reasons"] = [name for name, passed in validity.items() if not passed]
        return metrics

    target_peak = float(np.max(target_side))
    predicted_peak = float(np.max(predicted))
    target_impulse = float(np.trapezoid(target_side, times))
    predicted_impulse = float(np.trapezoid(predicted, times))
    metrics.update(
        {
            "vertical_waveform": {
                "rms_error_N": float(np.sqrt(np.mean((predicted - target_side) ** 2))),
                "rms_error_body_weight": float(
                    np.sqrt(np.mean((predicted - target_side) ** 2)) / sidecar.normalization.body_weight_n
                ),
            },
            "vertical_peak": {
                "target_N": target_peak,
                "predicted_N": predicted_peak,
                "relative_error": abs(predicted_peak - target_peak) / target_peak if target_peak > 0.0 else 0.0,
            },
            "vertical_impulse": {
                "target_N_s": target_impulse,
                "predicted_N_s": predicted_impulse,
                "relative_error": (
                    abs(predicted_impulse - target_impulse) / abs(target_impulse) if target_impulse != 0.0 else 0.0
                ),
            },
            "maximum_sphere_penetration_m": float(np.max(penetration)),
            "penetration_limit_m": _MAX_PENETRATION_M,
            "penetration_limit_passed": bool(np.max(penetration) < _MAX_PENETRATION_M),
        }
    )
    return metrics


def _validate_output_path(output_dir: Path, sidecar_path: Path, source_dir: Path | None) -> Path:
    """Require a new non-overlapping artifact directory outside the repository."""
    output_dir = output_dir.resolve()
    repository_root = Path(__file__).resolve().parents[2]
    if output_dir == repository_root or output_dir.is_relative_to(repository_root):
        raise ValueError("generated calibration artifacts must stay outside the repository")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if output_dir == sidecar_path.parent or sidecar_path.is_relative_to(output_dir):
        raise ValueError("calibration output must not contain or replace its input sidecar")
    if source_dir is not None:
        source_dir = source_dir.resolve()
        if output_dir == source_dir or output_dir.is_relative_to(source_dir) or source_dir.is_relative_to(output_dir):
            raise ValueError("calibration and source artifact directories must not overlap")
    return output_dir


def calibrate_normal_contact(
    sidecar_path: str | os.PathLike,
    output_dir: str | os.PathLike,
    times_s: np.ndarray,
    measured_vertical_force_n: np.ndarray,
    evaluator: Callable[[predictive_contact.PredictiveContactSidecar], NormalContactEvaluation | Mapping[str, Any]],
    *,
    max_nfev: int = 8,
    source_dir: str | os.PathLike | None = None,
) -> Path:
    """Run and atomically publish the bounded training-side-only fit.

    The evaluator call interface supplies only a candidate sidecar. An injected
    callable can still capture external state, so its measured-input isolation is
    explicitly reported as unverifiable. Its two predicted sides are returned in
    a fixed order. Only the declared training-side force and sphere penetration
    enter the residual. Held-out validity and metrics are computed after fitting.

    Args:
        sidecar_path: Strict predictive-contact sidecar used as the immutable seed.
        output_dir: New artifact directory outside the repository and source data.
        times_s: Frozen prescribed-motion target times [s].
        measured_vertical_force_n: Target vertical GRF in ``[left, right]`` order
            [N], shape ``[time, 2]``.
        evaluator: Callable whose explicit argument is only the candidate sidecar.
        max_nfev: Small positive cap on optimizer function evaluations.
        source_dir: Optional source artifact directory used for overlap checks.

    Returns:
        Path to the atomically published preliminary calibration artifact.
    """
    # SciPy is an optional Newton dependency, so keep it out of module import.
    from scipy.optimize import least_squares

    sidecar_path = Path(sidecar_path).resolve()
    if not sidecar_path.is_file():
        raise FileNotFoundError(sidecar_path)
    source_path = Path(source_dir).resolve() if source_dir is not None else None
    output_path = _validate_output_path(Path(output_dir), sidecar_path, source_path)
    if not isinstance(max_nfev, int) or isinstance(max_nfev, bool) or max_nfev < 1:
        raise ValueError("max_nfev must be a positive integer")

    sidecar = predictive_contact.load_contact_sidecar(sidecar_path)
    parameterization = NormalContactParameterization(sidecar)
    times = np.asarray(times_s, dtype=float)
    target = np.asarray(measured_vertical_force_n, dtype=float)
    if times.ndim != 1 or len(times) < 2 or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
        raise ValueError("times_s must contain at least two increasing finite samples")
    if target.shape != (len(times), 2):
        raise ValueError("measured_vertical_force_n must have shape [time, 2]")

    train_side = sidecar.calibration.train_side
    held_out_side = sidecar.calibration.held_out_side
    train_index = _SIDES.index(train_side)
    if not np.all(np.isfinite(target[:, train_index])):
        raise ValueError("training-side measured_vertical_force_n must be finite")
    train_spheres = [index for index, sphere in enumerate(sidecar.spheres) if sphere.side == train_side]
    body_weight = sidecar.normalization.body_weight_n
    duration = times[-1] - times[0]
    weights = sidecar.calibration.objective_weights
    vertical_waveform_sumsq_weight = float(weights["vertical_force"])
    vertical_peak_sumsq_weight = float(weights["vertical_force"])
    impulse_sumsq_weight = float(weights["impulse"])
    geometry_regularization_sumsq_weight = float(weights["regularization"])
    ground_regularization_sumsq_weight = float(weights["regularization"])
    if type(evaluator) is _PrescribedNormalEvaluator:
        evaluator_provenance = evaluator.provenance()
    else:
        evaluator_provenance = {
            "kind": "injected_callable",
            "measured_input_isolation": "unverifiable",
            "reason": "an injected callable can capture external state outside the calibration call interface",
        }
    trace: list[dict[str, Any]] = []

    def residual(encoded: np.ndarray, *, purpose: str = "optimizer") -> np.ndarray:
        candidate = parameterization.decode(encoded)
        evaluated = _coerce_evaluation(
            evaluator(candidate),
            len(times),
            len(candidate.spheres),
            train_index,
            train_spheres,
        )
        predicted = evaluated.vertical_force_n[:, train_index]
        target_train = target[:, train_index]
        waveform = (
            np.sqrt(vertical_waveform_sumsq_weight)
            * (predicted - target_train)
            / (body_weight * np.sqrt(predicted.size))
        )
        peak = np.asarray(
            [np.sqrt(vertical_peak_sumsq_weight) * (np.max(predicted) - np.max(target_train)) / body_weight]
        )
        impulse_scale = body_weight * duration
        impulse = np.asarray(
            [
                np.sqrt(impulse_sumsq_weight)
                * (np.trapezoid(predicted, times) - np.trapezoid(target_train, times))
                / impulse_scale
            ]
        )
        train_penetration = evaluated.sphere_penetration_m[:, train_spheres]
        penetration = (
            np.sqrt(_PENETRATION_SUMSQ_WEIGHT)
            * np.maximum(train_penetration - _MAX_PENETRATION_M, 0.0)
            / (_MAX_PENETRATION_M * np.sqrt(train_penetration.size))
        )
        geometry_regularization = (
            np.sqrt(geometry_regularization_sumsq_weight)
            * encoded[1:5]
            / (_CENTER_VERTICAL_OFFSET_BOUNDS_M[1] * np.sqrt(len(_ROLES)))
        )
        ground_regularization = np.asarray(
            [
                np.sqrt(ground_regularization_sumsq_weight)
                * (encoded[0] - sidecar.ground.platform_height_m)
                / _GROUND_RELATIVE_BOUNDS_M[1]
            ]
        )
        result = np.concatenate(
            (waveform, peak, impulse, penetration.ravel(), geometry_regularization, ground_regularization)
        )
        trace.append(
            {
                "evaluation_index": len(trace),
                "purpose": purpose,
                "parameters": parameterization.values(encoded),
                "objective_side": train_side,
                "held_out_side": held_out_side,
                "held_out_side_used_in_objective": False,
                "residual_term_sumsq": {
                    "vertical_waveform": float(waveform @ waveform),
                    "vertical_peak": float(peak @ peak),
                    "vertical_impulse": float(impulse @ impulse),
                    "training_side_penetration_above_0_020_m": float(penetration.ravel() @ penetration.ravel()),
                    "geometry_seed_regularization": float(geometry_regularization @ geometry_regularization),
                    "ground_plane_regularization": float(ground_regularization @ ground_regularization),
                },
                "residual_sumsq": float(result @ result),
                "training_side_maximum_penetration_m": float(np.max(train_penetration)),
            }
        )
        residual.last_evaluation = evaluated
        return result

    residual.last_evaluation = None
    result = least_squares(
        residual,
        parameterization.x0,
        bounds=(parameterization.lower_bounds, parameterization.upper_bounds),
        method="trf",
        jac="2-point",
        max_nfev=max_nfev,
    )
    calibrated = parameterization.decode(result.x)
    residual(result.x, purpose="frozen_post_fit_metrics")
    final_evaluation = residual.last_evaluation
    assert isinstance(final_evaluation, NormalContactEvaluation)
    fit_metrics = _side_metrics(train_side, "fit", times, target, final_evaluation, calibrated)
    held_out_metrics = _side_metrics(held_out_side, "held_out", times, target, final_evaluation, calibrated)

    optimizer_result = {
        "schema_version": _SCHEMA,
        "architecture_role": ARCHITECTURE_ROLE,
        "scope": _SCOPE,
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "cost": float(result.cost),
        "optimality": float(result.optimality),
        "nfev_reported_by_scipy": int(result.nfev),
        "njev_reported_by_scipy": int(result.njev) if result.njev is not None else None,
        "traced_evaluator_call_count": len(trace),
        "max_nfev": max_nfev,
        "method": "scipy.optimize.least_squares(method='trf', jac='2-point')",
        "parameter_names": parameterization.names,
        "initial_parameters": parameterization.values(parameterization.x0),
        "final_parameters": parameterization.values(result.x),
        "lower_bounds": parameterization.values(parameterization.lower_bounds),
        "upper_bounds": parameterization.values(parameterization.upper_bounds),
        "active_mask": result.active_mask,
        "objective_sumsq_weights": {
            "vertical_waveform": vertical_waveform_sumsq_weight,
            "vertical_peak": vertical_peak_sumsq_weight,
            "vertical_impulse": impulse_sumsq_weight,
            "training_side_penetration_above_0_020_m": _PENETRATION_SUMSQ_WEIGHT,
            "geometry_seed_regularization": geometry_regularization_sumsq_weight,
            "ground_plane_regularization": ground_regularization_sumsq_weight,
        },
        "residual_normalization": {
            "vertical_waveform": "sqrt(time_sample_count)",
            "training_side_penetration": "sqrt(time_sample_count_times_training_sphere_count)",
            "shared_vertical_center_offset_regularization": "sqrt(role_count)",
        },
        "objective_terms": [
            "training-side normalized vertical waveform",
            "training-side normalized vertical peak",
            "training-side normalized vertical impulse",
            "training-side sphere penetration above 0.020 m",
            "shared vertical center offset regularization about immutable geometry seeds",
            "ground height regularization about measured platform height",
        ],
        "held_out_side_used_in_objective": False,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_path.name}.", dir=output_path.parent))
    try:
        _write_json(temporary / "calibrated_contact_sidecar.json", predictive_contact.sidecar_to_dict(calibrated))
        # Re-load the published representation before any directory rename.
        predictive_contact.load_contact_sidecar(temporary / "calibrated_contact_sidecar.json")
        _write_json(temporary / "optimizer_result.json", optimizer_result)
        _write_json(temporary / "fit_metrics.json", fit_metrics)
        _write_json(temporary / "held_out_metrics.json", held_out_metrics)
        _write_json(temporary / "evaluation_trace.json", trace)
        artifact_names = (
            "calibrated_contact_sidecar.json",
            "optimizer_result.json",
            "fit_metrics.json",
            "held_out_metrics.json",
            "evaluation_trace.json",
        )
        manifest = {
            "schema_version": _SCHEMA,
            "architecture_role": ARCHITECTURE_ROLE,
            "reference_only": True,
            "production_eligible": False,
            "scope": _SCOPE,
            "status": (
                "preliminary_normal_contact_fit_succeeded"
                if result.success
                else "preliminary_normal_contact_fit_optimizer_unsuccessful"
            ),
            "claims": {
                "optimizer_success": bool(result.success),
                "complete_stage_2_calibration": False,
                "forward_dynamics": False,
                "fd_1": False,
                "held_out_side_fitted": False,
            },
            "limitations": [
                "normal vertical contact only",
                "no horizontal-force, timing, COP, or free-moment objective",
                "no parameter-sensitivity table",
                "prescribed kinematics, not forward dynamics",
                "full Stage 2 gates require predictive_contact.run_prescribed_contact",
            ],
            "split": {"fit": train_side, "held_out": held_out_side},
            "information_set": {
                "evaluator_inputs": (
                    ["candidate_contact_sidecar", "frozen_coordinates", "frozen_speeds"]
                    if evaluator_provenance["kind"] == "built_in_prescribed_normal_evaluator"
                    else ["candidate_contact_sidecar", "injected_callable_external_state_unverifiable"]
                ),
                "measured_load_passed_as_call_argument": False,
                "measured_input_isolation": evaluator_provenance["measured_input_isolation"],
                "optimizer_target": ["fit_side_vertical_force"],
                "held_out_target_used_by_optimizer": False,
                "held_out_validity_checked_after_parameters_frozen": True,
            },
            "evaluator_provenance": evaluator_provenance,
            "parameter_contract": {
                "names": parameterization.names,
                "lower_bounds": parameterization.values(parameterization.lower_bounds),
                "upper_bounds": parameterization.values(parameterization.upper_bounds),
                "bilateral_role_offsets_shared": True,
                "geometry_seed_fields_immutable": True,
                "radii_and_other_material_parameters_frozen": True,
                "maximum_penetration_m": _MAX_PENETRATION_M,
            },
            "runtime_provenance": _runtime_provenance(),
            "source": {
                "input_sidecar_path": str(sidecar_path),
                "input_sidecar_sha256": _sha256(sidecar_path),
                "source_dir": str(source_path) if source_path is not None else None,
                "times_sha256": _array_sha256(times),
                "measured_vertical_force_sha256": _array_sha256(target),
                "model_sha256": calibrated.source_model_sha256,
                "analysis_sha256": calibrated.source_analysis_sha256,
            },
            "artifacts": {name: {"path": name, "sha256": _sha256(temporary / name)} for name in artifact_names},
        }
        _write_json(temporary / "manifest.json", manifest)
        os.rename(temporary, output_path)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output_path


class _PrescribedNormalEvaluator:
    """Evaluate candidates from prescribed state only, without measured loads."""

    def __init__(
        self,
        model: osim.OsimModel,
        coordinates: np.ndarray,
        speeds: np.ndarray,
        coordinate_names: tuple[str, ...],
        device: str,
    ):
        self._model = model
        self._coordinates = np.array(coordinates, dtype=float, copy=True)
        self._speeds = np.array(speeds, dtype=float, copy=True)
        self._coordinates.setflags(write=False)
        self._speeds.setflags(write=False)
        self._coordinate_names = coordinate_names
        self._device = device

    def provenance(self) -> dict[str, Any]:
        """Return trusted prescribed-state provenance for the built-in evaluator."""
        return {
            "kind": "built_in_prescribed_normal_evaluator",
            "measured_input_isolation": "verified",
            "q_sha256": _array_sha256(self._coordinates),
            "qd_sha256": _array_sha256(self._speeds),
            "coordinate_names": self._coordinate_names,
            "device": self._device,
            "velocity_stencil_h_s": _VELOCITY_STENCIL_H_S,
        }

    def __call__(self, sidecar: predictive_contact.PredictiveContactSidecar) -> NormalContactEvaluation:
        """Return predicted vertical force and penetration for one sidecar."""
        augmented = predictive_contact.augment_contact_model(self._model, sidecar)
        contact = osim.OpenSimContact(augmented, device=self._device)
        if tuple(contact.coordinate_names) != self._coordinate_names:
            raise ValueError("analysis coordinate order does not match OpenSimContact")
        body_names, wrenches = contact.body_wrenches(
            self._coordinates,
            self._speeds,
            h=_VELOCITY_STENCIL_H_S,
            frame="opensim",
        )
        body_index = {name: index for index, name in enumerate(body_names)}
        expected = ["calcn_l", "calcn_r"]
        if not set(expected).issubset(body_index):
            raise ValueError("OpenSimContact did not return both calcaneus bodies")
        selected = np.asarray(wrenches, dtype=float)[:, [body_index[name] for name in expected]]
        if selected.shape != (len(self._coordinates), 2, 9):
            raise ValueError("OpenSimContact returned an unexpected body-wrench shape")
        kinematics = osim.ForwardKinematics(augmented, device=self._device)
        if tuple(kinematics.coordinate_names) != self._coordinate_names:
            raise ValueError("analysis coordinate order does not match ForwardKinematics")
        transforms = np.asarray(kinematics.body_transforms_batch(self._coordinates), dtype=float)
        penetration = predictive_contact.sphere_penetrations(transforms, list(kinematics.body_names), sidecar)
        return NormalContactEvaluation(selected[..., 1], penetration)


def _uniform_time_grid(times: np.ndarray, step_s: float) -> np.ndarray:
    """Build the same endpoint-preserving uniform grid as prescribed QC."""
    duration = float(times[-1] - times[0])
    intervals = int(round(duration / step_s))
    if intervals < 1 or not np.isclose(intervals * step_s, duration, rtol=0.0, atol=1.0e-10):
        raise ValueError("prescribed interval must be an integer multiple of the frozen time step")
    return times[0] + np.arange(intervals + 1, dtype=float) * step_s


def _interpolate(times: np.ndarray, values: np.ndarray, sample_times: np.ndarray) -> np.ndarray:
    """Linearly interpolate finite numeric columns."""
    flat = values.reshape(len(times), -1)
    result = np.column_stack([np.interp(sample_times, times, flat[:, index]) for index in range(flat.shape[1])])
    return result.reshape((len(sample_times), *values.shape[1:]))


def run_contact_calibration(
    data_dir: str | os.PathLike = _DEFAULT_DATA,
    sidecar_path: str | os.PathLike | None = None,
    output_dir: str | os.PathLike | None = None,
    *,
    max_nfev: int = 8,
    device: str = "cpu",
    evaluator: Callable[[predictive_contact.PredictiveContactSidecar], NormalContactEvaluation | Mapping[str, Any]]
    | None = None,
    prescribed_qc_output_dir: str | os.PathLike | None = None,
) -> Path:
    """Load the frozen Trial 101 sample and run the preliminary calibration.

    ``prescribed_qc_output_dir`` optionally runs the existing complete prescribed
    contact evaluator after publication. Its separate artifact keeps this fit's
    scope unchanged and does not convert the result into Stage 2 acceptance or FD.
    """
    data_path = Path(data_dir).resolve()
    if sidecar_path is None:
        raise ValueError("sidecar_path is required")
    sidecar_file = Path(sidecar_path).resolve()
    if output_dir is None:
        output_dir = data_path.parent / f"{data_path.name}_preliminary_normal_contact_calibration"
    output_path = Path(output_dir).resolve()
    qc_output = Path(prescribed_qc_output_dir).resolve() if prescribed_qc_output_dir is not None else None
    if qc_output is not None and (
        qc_output == output_path or qc_output.is_relative_to(output_path) or output_path.is_relative_to(qc_output)
    ):
        raise ValueError("full prescribed QC must use a separate non-overlapping artifact directory")
    sidecar = predictive_contact.load_contact_sidecar(sidecar_file)
    model_path = data_path / "S001_scaled.osim"
    analysis_path = data_path / "analysis.npz"
    for path in (model_path, analysis_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    source_model = Path(sidecar.source_model_path)
    if not source_model.is_absolute():
        source_model = sidecar_file.parent / source_model
    source_analysis = Path(sidecar.source_analysis_path)
    if not source_analysis.is_absolute():
        source_analysis = sidecar_file.parent / source_analysis
    if source_model.resolve() != model_path or _sha256(model_path) != sidecar.source_model_sha256:
        raise ValueError("sidecar source model path or SHA-256 does not match the source artifact")
    if source_analysis.resolve() != analysis_path or _sha256(analysis_path) != sidecar.source_analysis_sha256:
        raise ValueError("sidecar source analysis path or SHA-256 does not match the frozen sample")
    with np.load(analysis_path, allow_pickle=False) as archive:
        if (
            "schema_version" not in archive.files
            or np.asarray(archive["schema_version"]).shape != ()
            or str(np.asarray(archive["schema_version"]).item()) != "gait_c3d_analysis_3"
        ):
            raise ValueError("analysis must use the frozen gait_c3d_analysis_3 schema")
        if "frame" in archive.files and str(np.asarray(archive["frame"]).item()) != sidecar.frame:
            raise ValueError("analysis frame does not match the contact sidecar")
        times = np.asarray(archive["times"], dtype=float)
        coordinates = np.asarray(archive["id_coordinates"], dtype=float)
        speeds = np.asarray(archive["id_speeds"], dtype=float)
        grf = np.asarray(archive["grf"], dtype=float)
        names = [str(value) for value in np.asarray(archive["foot_names"])]
        coordinate_names = tuple(str(value) for value in np.asarray(archive["id_names"]))
    if (
        times.ndim != 1
        or len(times) < 2
        or not np.all(np.isfinite(times))
        or np.any(np.diff(times) <= 0.0)
        or coordinates.shape != speeds.shape
        or coordinates.shape != (len(times), len(coordinate_names))
        or not np.all(np.isfinite(coordinates))
        or not np.all(np.isfinite(speeds))
    ):
        raise ValueError("analysis must contain a finite ordered prescribed state sample")
    if names != list(_SIDES) or grf.shape != (len(times), 2, 3):
        raise ValueError("analysis must use exact [left, right] force ordering")
    sample_times = _uniform_time_grid(times, sidecar.calibration.prescribed_time_step_s)
    sample_coordinates = _interpolate(times, coordinates, sample_times)
    sample_speeds = _interpolate(times, speeds, sample_times)
    sample_vertical_force = _interpolate(times, grf[..., 1], sample_times)
    if evaluator is None:
        model = osim.parse_osim(model_path)
        evaluator = _PrescribedNormalEvaluator(
            model,
            sample_coordinates,
            sample_speeds,
            coordinate_names,
            device,
        )
    result = calibrate_normal_contact(
        sidecar_file,
        output_path,
        sample_times,
        sample_vertical_force,
        evaluator,
        max_nfev=max_nfev,
        source_dir=data_path,
    )
    if qc_output is not None:
        predictive_contact.run_prescribed_contact(
            data_path,
            result / "calibrated_contact_sidecar.json",
            qc_output,
            device=device,
        )
    return result


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the bounded calibration command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-only",
        action="store_true",
        help="required acknowledgement: this uses newton.opensim compatibility mechanics, not production Newton",
    )
    parser.add_argument("--data-dir", type=Path, default=_DEFAULT_DATA)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-nfev", type=int, default=8, help="small positive least-squares evaluation cap")
    parser.add_argument(
        "--prescribed-qc-output-dir",
        type=Path,
        help="optional separate full prescribed-QC artifact; does not change preliminary scope",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the bounded preliminary normal-contact calibration CLI."""
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    if not args.reference_only:
        parser.error("--reference-only is required; use newton_contact_calibration for production contact")
    output = run_contact_calibration(
        args.data_dir,
        args.sidecar,
        args.output_dir,
        max_nfev=args.max_nfev,
        device=args.device,
        prescribed_qc_output_dir=args.prescribed_qc_output_dir,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
