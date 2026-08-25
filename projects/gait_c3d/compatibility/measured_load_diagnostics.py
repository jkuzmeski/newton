# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""OFFLINE COMPATIBILITY REFERENCE ONLY. Its rollouts use ``newton.opensim`` and are excluded from production Newton dependency manifests.

Characterize measured-load forward integration for the C3D gait project.

This Stage 1 harness replays archived measured external loads. It is an
engineering diagnostic and never a predictive-gait result. The pelvis/root is
structurally identified from the OpenSim ground joint. The stage-gate controller
commands only bounded non-root generalized forces.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import shutil
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

import newton.opensim as osim

ARCHITECTURE_ROLE = "compatibility_reference"

_DEFAULT_DATA = Path("/home/jo31399/newton-data/gait/processed/trial_101/forward_dynamics")
_SCOPE = "engineering_measured_load_tracking"
_SCHEMA = "gait_c3d_measured_load_diagnostics_1"
_SAMPLE_TOLERANCE_S = 1.0e-10
SamplingMode = Literal["linear", "zoh"]


@dataclass(frozen=True)
class StrictSampler:
    """Sample finite archived values without extrapolation."""

    times: np.ndarray
    values: np.ndarray
    mode: SamplingMode = "linear"

    def __post_init__(self) -> None:
        times = np.asarray(self.times, dtype=float).copy()
        values = np.asarray(self.values, dtype=float).copy()
        if self.mode not in ("linear", "zoh"):
            raise ValueError("sampling mode must be 'linear' or 'zoh'")
        if times.ndim != 1 or len(times) < 2 or np.any(np.diff(times) <= 0.0):
            raise ValueError("source times must be one-dimensional and increase strictly")
        if values.ndim < 1 or values.shape[0] != len(times):
            raise ValueError("values must use the source time axis as their leading axis")
        if not np.all(np.isfinite(times)) or not np.all(np.isfinite(values)):
            raise ValueError("source times and values must be finite")
        times.setflags(write=False)
        values.setflags(write=False)
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "values", values)

    def sample(self, output_times: np.ndarray | list[float] | float) -> np.ndarray:
        """Return samples at requested times, including both exact endpoints."""
        requested = np.asarray(output_times, dtype=float)
        scalar = requested.ndim == 0
        requested_1d = np.atleast_1d(requested)
        if requested_1d.ndim != 1 or not np.all(np.isfinite(requested_1d)):
            raise ValueError("requested times must be finite and one-dimensional")
        if np.any(requested_1d < self.times[0] - _SAMPLE_TOLERANCE_S) or np.any(
            requested_1d > self.times[-1] + _SAMPLE_TOLERANCE_S
        ):
            raise ValueError("requested time is outside the archived trajectory")
        clipped = np.clip(requested_1d, self.times[0], self.times[-1])
        if self.mode == "zoh":
            indices = np.searchsorted(self.times, clipped, side="right") - 1
            sampled = self.values[indices]
        else:
            flat = self.values.reshape(len(self.times), -1)
            sampled = np.column_stack(
                [np.interp(clipped, self.times, flat[:, index]) for index in range(flat.shape[1])]
            ).reshape((len(clipped), *self.values.shape[1:]))
        return sampled[0] if scalar else sampled


@dataclass(frozen=True)
class MeasuredLoadTrajectory:
    """Validated measured state, inverse-dynamics force, and wrench archive."""

    times: np.ndarray
    coordinates: np.ndarray
    speeds: np.ndarray
    generalized_forces: np.ndarray
    coordinate_names: tuple[str, ...]
    motion_types: tuple[str, ...]
    external_bodies: tuple[str, ...]
    external_wrenches: np.ndarray

    def __post_init__(self) -> None:
        times = np.asarray(self.times, dtype=float).copy()
        coordinates = np.asarray(self.coordinates, dtype=float).copy()
        speeds = np.asarray(self.speeds, dtype=float).copy()
        forces = np.asarray(self.generalized_forces, dtype=float).copy()
        wrenches = np.asarray(self.external_wrenches, dtype=float).copy()
        names = tuple(str(value) for value in self.coordinate_names)
        motion_types = tuple(str(value) for value in self.motion_types)
        bodies = tuple(str(value) for value in self.external_bodies)
        if times.ndim != 1 or len(times) < 2 or np.any(np.diff(times) <= 0.0):
            raise ValueError("analysis times must increase strictly")
        expected = (len(times), len(names))
        if coordinates.shape != expected or speeds.shape != expected or forces.shape != expected:
            raise ValueError("state and generalized-force arrays must share shape [time, coordinate]")
        if len(set(names)) != len(names) or len(motion_types) != len(names):
            raise ValueError("coordinate names must be unique and match motion types")
        if wrenches.shape != (len(times), len(bodies), 9):
            raise ValueError("external wrenches must have shape [time, body, 9]")
        if len(set(bodies)) != len(bodies):
            raise ValueError("external body names must be unique")
        for value in (times, coordinates, speeds, forces, wrenches):
            if not np.all(np.isfinite(value)):
                raise ValueError("measured-load source arrays must be finite")
            value.setflags(write=False)
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "speeds", speeds)
        object.__setattr__(self, "generalized_forces", forces)
        object.__setattr__(self, "coordinate_names", names)
        object.__setattr__(self, "motion_types", motion_types)
        object.__setattr__(self, "external_bodies", bodies)
        object.__setattr__(self, "external_wrenches", wrenches)

    @classmethod
    def from_analysis(cls, path: str | os.PathLike, motion_types: list[str]) -> MeasuredLoadTrajectory:
        """Read the required Stage 1 fields from ``analysis.npz``."""
        with np.load(path, allow_pickle=False) as archive:
            required = (
                "times",
                "id_coordinates",
                "id_speeds",
                "id_generalized_forces",
                "id_names",
                "id_external_bodies",
                "id_external_wrenches",
            )
            missing = [name for name in required if name not in archive.files]
            if missing:
                raise ValueError(f"analysis archive is missing {missing!r}")
            return cls(
                times=archive["times"],
                coordinates=archive["id_coordinates"],
                speeds=archive["id_speeds"],
                generalized_forces=archive["id_generalized_forces"],
                coordinate_names=tuple(str(value) for value in archive["id_names"]),
                motion_types=tuple(motion_types),
                external_bodies=tuple(str(value) for value in archive["id_external_bodies"]),
                external_wrenches=archive["id_external_wrenches"],
            )

    def sampler(self, field_name: str, mode: SamplingMode) -> StrictSampler:
        """Build a strict sampler for one source field."""
        values = {
            "coordinates": self.coordinates,
            "speeds": self.speeds,
            "generalized_forces": self.generalized_forces,
            "external_wrenches": self.external_wrenches,
        }.get(field_name)
        if values is None:
            raise ValueError(f"unknown trajectory field {field_name!r}")
        return StrictSampler(self.times, values, mode)


def structural_root_mask(model: Any, coordinate_names: list[str] | tuple[str, ...]) -> np.ndarray:
    """Identify root coordinates from joints attached directly to ground."""
    names = tuple(coordinate_names)
    if len(set(names)) != len(names):
        raise ValueError("coordinate names must be unique")
    ground_names: set[str] = set()
    for joint in model.joints:
        if joint.parent_body == "ground":
            ground_names.update(coordinate.name for coordinate in joint.coordinates)
    unknown = ground_names.difference(names)
    if unknown:
        raise ValueError(f"ground-joint coordinates are absent from the dynamics order: {sorted(unknown)!r}")
    mask = np.asarray([name in ground_names for name in names], dtype=bool)
    if not np.any(mask):
        raise ValueError("model has no structurally identifiable ground-joint coordinate")
    return mask


def _side_indices(bodies: tuple[str, ...], side: Literal["left", "right"]) -> list[int]:
    suffix = "_l" if side == "left" else "_r"
    indices = [index for index, body in enumerate(bodies) if body.lower().endswith(suffix)]
    if not indices:
        raise ValueError(f"no {side} external-load body is identifiable by the {suffix!r} suffix")
    return indices


def select_load_variant(
    bodies: tuple[str, ...] | list[str],
    wrenches: np.ndarray,
    selection: Literal["all", "left", "right", "none"],
) -> tuple[list[str], np.ndarray]:
    """Copy one bilateral load variant while retaining source body order."""
    body_tuple = tuple(str(value) for value in bodies)
    values = np.asarray(wrenches, dtype=float)
    if values.ndim != 3 or values.shape[1:] != (len(body_tuple), 9):
        raise ValueError("wrenches must have shape [time, body, 9]")
    if selection not in ("all", "left", "right", "none"):
        raise ValueError(f"unknown load selection {selection!r}")
    result = values.copy()
    if selection == "none":
        result.fill(0.0)
    elif selection != "all":
        keep = set(_side_indices(body_tuple, selection))
        for index in range(len(body_tuple)):
            if index not in keep:
                result[:, index] = 0.0
    return list(body_tuple), result


@dataclass(frozen=True)
class ControllerConfig:
    """Frozen unit-aware gains and bounds for diagnostic non-root tracking."""

    name: str = "bounded_nonroot_idff_pd_v1"
    rotational_kp: float = 200.0
    rotational_kd: float = 20.0
    rotational_effort_limit: float = 250.0
    translational_kp: float = 2000.0
    translational_kd: float = 200.0
    translational_effort_limit: float = 2000.0

    def arrays(self, motion_types: tuple[str, ...] | list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Expand scalar settings in coordinate order."""
        rotational = np.asarray([kind == "rotational" for kind in motion_types], dtype=bool)
        supported = rotational | np.asarray([kind == "translational" for kind in motion_types], dtype=bool)
        if not np.all(supported):
            raise ValueError("controller requires rotational or translational motion types")
        values = asdict(self)
        for name, value in values.items():
            if name != "name" and (not np.isfinite(value) or value < 0.0):
                raise ValueError(f"controller setting {name!r} must be finite and nonnegative")
        kp = np.where(rotational, self.rotational_kp, self.translational_kp)
        kd = np.where(rotational, self.rotational_kd, self.translational_kd)
        limit = np.where(rotational, self.rotational_effort_limit, self.translational_effort_limit)
        if np.any(limit <= 0.0):
            raise ValueError("effort limits must be positive")
        return kp, kd, limit


@dataclass(frozen=True)
class ControlBreakdown:
    """One evaluated generalized-force decomposition."""

    raw_feedforward: np.ndarray
    feedforward: np.ndarray
    raw_feedback: np.ndarray
    feedback: np.ndarray
    total: np.ndarray
    saturated: np.ndarray


class BoundedNonRootController:
    """Apply bounded inverse-dynamics feed-forward plus PD on non-root coordinates."""

    def __init__(
        self,
        trajectory: MeasuredLoadTrajectory,
        root_mask: np.ndarray,
        config: ControllerConfig,
        mode: SamplingMode = "linear",
        *,
        feedback_enabled: bool = True,
    ) -> None:
        self.trajectory = trajectory
        self.root_mask = np.asarray(root_mask, dtype=bool).copy()
        if self.root_mask.shape != (len(trajectory.coordinate_names),):
            raise ValueError("root mask must match the coordinate order")
        self.nonroot_mask = ~self.root_mask
        self.config = config
        self.mode = mode
        self.feedback_enabled = feedback_enabled
        self.kp, self.kd, self.limit = config.arrays(trajectory.motion_types)
        self._q = trajectory.sampler("coordinates", mode)
        self._qd = trajectory.sampler("speeds", mode)
        self._tau = trajectory.sampler("generalized_forces", mode)

    def evaluate(self, t: float, q: np.ndarray, qd: np.ndarray) -> ControlBreakdown:
        """Evaluate and assert the exact-zero root and total-bound invariants."""
        q = np.asarray(q, dtype=float)
        qd = np.asarray(qd, dtype=float)
        if q.shape != self.root_mask.shape or qd.shape != self.root_mask.shape:
            raise ValueError("controller states must match the coordinate order")
        q_ref = self._q.sample(t)
        qd_ref = self._qd.sample(t)
        tau_ref = self._tau.sample(t)
        raw_ff = np.where(self.nonroot_mask, tau_ref, 0.0)
        ff = np.clip(raw_ff, -self.limit, self.limit)
        if self.feedback_enabled:
            raw_fb = np.where(self.nonroot_mask, self.kp * (q_ref - q) + self.kd * (qd_ref - qd), 0.0)
        else:
            raw_fb = np.zeros_like(raw_ff)
        feedback = np.clip(raw_fb, -self.limit - ff, self.limit - ff)
        total = ff + feedback
        saturated = (ff != raw_ff) | (feedback != raw_fb)
        for value in (raw_ff, ff, raw_fb, feedback, total):
            if not np.array_equal(value[self.root_mask], np.zeros(np.count_nonzero(self.root_mask))):
                raise AssertionError("root generalized force is not exactly zero")
        if np.any(np.abs(total[self.nonroot_mask]) > self.limit[self.nonroot_mask] + 1.0e-12):
            raise AssertionError("bounded controller exceeded a coordinate effort limit")
        return ControlBreakdown(raw_ff, ff, raw_fb, feedback, total, saturated)

    def __call__(self, t: float, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        """Return total generalized force for :class:`ForwardDynamics`."""
        return self.evaluate(t, q, qd).total


@dataclass(frozen=True)
class VariantSpec:
    """Define one predeclared input-decomposition variant."""

    name: str
    interpolation: SamplingMode
    load_selection: Literal["all", "left", "right", "none"]
    control: Literal["full_id", "zero_root_id", "bounded_tracking", "bounded_feedforward"]


_VARIANTS = {
    value.name: value
    for value in (
        VariantSpec("full_id_all_linear", "linear", "all", "full_id"),
        VariantSpec("zero_root_id_all_linear", "linear", "all", "zero_root_id"),
        VariantSpec("bounded_nonroot_tracking_linear", "linear", "all", "bounded_tracking"),
        VariantSpec("left_only_full_id_linear", "linear", "left", "full_id"),
        VariantSpec("right_only_full_id_linear", "linear", "right", "full_id"),
        VariantSpec("generalized_force_only_no_external_load_linear", "linear", "none", "full_id"),
        VariantSpec("bounded_nonroot_feedforward_linear", "linear", "all", "bounded_feedforward"),
        VariantSpec("full_id_all_zoh", "zoh", "all", "full_id"),
    )
}


@dataclass(frozen=True)
class Stage1Config:
    """Frozen settings for the complete Stage 1 diagnostic sweep."""

    timesteps_s: tuple[float, ...] = (0.001, 0.0005, 0.00025)
    refinement_timestep_s: float = 0.000125
    convergence_relative_tolerance: float = 0.05
    restart_horizons_s: tuple[float, ...] = (0.025, 0.050, 0.100)
    restart_timestep_s: float = 0.001
    integrator: str = "rk4"
    condition_number_limit: float = 1.0e12
    mass_symmetry_relative_tolerance: float = 1.0e-10
    rotational_error_scale: float = 0.1
    translational_error_scale: float = 0.05
    rotational_speed_error_scale: float = 1.0
    translational_speed_error_scale: float = 0.5
    metric_chunk_size: int = 16
    controller: ControllerConfig = field(default_factory=ControllerConfig)

    def validate(self) -> None:
        """Reject settings that make requested coverage or metrics ambiguous."""
        positive = (
            *self.timesteps_s,
            self.refinement_timestep_s,
            self.convergence_relative_tolerance,
            *self.restart_horizons_s,
            self.restart_timestep_s,
            self.condition_number_limit,
            self.mass_symmetry_relative_tolerance,
            self.rotational_error_scale,
            self.translational_error_scale,
            self.rotational_speed_error_scale,
            self.translational_speed_error_scale,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("Stage 1 numeric settings must be finite and positive")
        if len(set(self.timesteps_s)) != len(self.timesteps_s):
            raise ValueError("convergence timesteps must be unique")
        archived_timesteps = tuple(dict.fromkeys((*self.timesteps_s, self.refinement_timestep_s)))
        archive_keys = [_dt_key(value) for value in archived_timesteps]
        if len(set(archive_keys)) != len(archive_keys):
            raise ValueError("convergence timesteps collide after NPZ key encoding")
        if self.integrator not in ("rk4", "semi_implicit"):
            raise ValueError("integrator must be 'rk4' or 'semi_implicit'")
        if self.metric_chunk_size < 1:
            raise ValueError("metric_chunk_size must be positive")
        self.controller.arrays(("rotational", "translational"))


@dataclass
class RolloutRecord:
    """Trajectory, input decomposition, and metrics for one rollout."""

    variant: str
    requested_dt_s: float
    actual_dt_s: float
    start_index: int
    requested_duration_s: float
    status: str
    times: np.ndarray
    coordinates: np.ndarray
    speeds: np.ndarray
    reference_coordinates: np.ndarray
    reference_speeds: np.ndarray
    coordinate_error: np.ndarray
    speed_error: np.ndarray
    raw_feedforward: np.ndarray
    feedforward: np.ndarray
    raw_feedback: np.ndarray
    feedback: np.ndarray
    total_control: np.ndarray
    saturation: np.ndarray
    marker_error_m: np.ndarray
    kinetic_energy_j: np.ndarray
    potential_energy_j: np.ndarray
    total_energy_j: np.ndarray
    mass_condition_number: np.ndarray
    mass_symmetry_relative_error: np.ndarray
    mass_min_eigenvalue: np.ndarray
    mass_cholesky_success: np.ndarray
    external_power_w: np.ndarray
    feedforward_power_w: np.ndarray
    feedback_power_w: np.ndarray
    metrics: dict[str, Any]


@dataclass(frozen=True)
class RestartCell:
    """One requested start/horizon pair, including unavailable edge cells."""

    start_index: int
    start_time_s: float
    horizon_s: float
    status: Literal["scheduled", "unavailable_source_boundary"]


class _SampledExternalLoads:
    """Adapt a strict wrench sampler to the ForwardDynamics load protocol."""

    def __init__(self, times: np.ndarray, bodies: list[str], wrenches: np.ndarray, mode: SamplingMode) -> None:
        self.bodies = list(bodies)
        self.sampler = StrictSampler(times, wrenches, mode)

    def sample(self, times: np.ndarray) -> tuple[list[str], np.ndarray]:
        """Return ground-frame ``[F P T]`` values."""
        return self.bodies, self.sampler.sample(times)


def restart_schedule(times: np.ndarray, horizons_s: tuple[float, ...] | list[float]) -> list[RestartCell]:
    """Emit every restart cell and preserve unavailable source-boundary cells."""
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or len(times) < 2 or np.any(np.diff(times) <= 0.0):
        raise ValueError("restart times must increase strictly")
    if any(not np.isfinite(value) or value <= 0.0 for value in horizons_s):
        raise ValueError("restart horizons must be finite and positive")
    return [
        RestartCell(
            start_index=index,
            start_time_s=float(start_time),
            horizon_s=float(horizon),
            status=(
                "scheduled"
                if start_time + horizon <= times[-1] + _SAMPLE_TOLERANCE_S
                else "unavailable_source_boundary"
            ),
        )
        for index, start_time in enumerate(times)
        for horizon in horizons_s
    ]


def _finite_prefix(coordinates: np.ndarray, speeds: np.ndarray) -> int:
    finite = np.all(np.isfinite(coordinates), axis=1) & np.all(np.isfinite(speeds), axis=1)
    nonfinite = np.flatnonzero(~finite)
    return int(nonfinite[0]) if len(nonfinite) else len(finite)


def _rms(values: np.ndarray) -> float | None:
    return float(np.sqrt(np.mean(np.asarray(values, dtype=float) ** 2))) if np.size(values) else None


def _max_abs(values: np.ndarray) -> float | None:
    return float(np.max(np.abs(np.asarray(values, dtype=float)))) if np.size(values) else None


def _cumulative_trapezoid(times: np.ndarray, values: np.ndarray) -> np.ndarray:
    result = np.zeros(len(times), dtype=float)
    if len(times) > 1:
        result[1:] = np.cumsum(0.5 * np.diff(times) * (values[:-1] + values[1:]))
    return result


def _controller_archive(
    spec: VariantSpec,
    trajectory: MeasuredLoadTrajectory,
    root_mask: np.ndarray,
    controller_config: ControllerConfig,
    times: np.ndarray,
    coordinates: np.ndarray,
    speeds: np.ndarray,
) -> tuple[np.ndarray, ...]:
    nc = len(trajectory.coordinate_names)
    arrays = [np.zeros((len(times), nc), dtype=float) for _ in range(5)]
    saturation = np.zeros((len(times), nc), dtype=bool)
    if spec.control in ("bounded_tracking", "bounded_feedforward"):
        controller = BoundedNonRootController(
            trajectory,
            root_mask,
            controller_config,
            spec.interpolation,
            feedback_enabled=spec.control == "bounded_tracking",
        )
        for index, (sample_time, q, qd) in enumerate(zip(times, coordinates, speeds, strict=True)):
            if not np.all(np.isfinite(q)) or not np.all(np.isfinite(qd)):
                arrays[0][index:] = np.nan
                arrays[1][index:] = np.nan
                arrays[2][index:] = np.nan
                arrays[3][index:] = np.nan
                arrays[4][index:] = np.nan
                break
            value = controller.evaluate(float(sample_time), q, qd)
            arrays[0][index] = value.raw_feedforward
            arrays[1][index] = value.feedforward
            arrays[2][index] = value.raw_feedback
            arrays[3][index] = value.feedback
            arrays[4][index] = value.total
            saturation[index] = value.saturated
    else:
        tau = trajectory.sampler("generalized_forces", spec.interpolation).sample(times)
        arrays[0][:] = tau
        arrays[1][:] = tau
        if spec.control == "zero_root_id":
            arrays[1][:, root_mask] = 0.0
        arrays[4][:] = arrays[1]
    return (*arrays, saturation)


def _coordinate_ranges(model: Any, names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    by_name = {coordinate.name: coordinate for joint in model.joints for coordinate in joint.coordinates}
    lower = np.full(len(names), -np.inf)
    upper = np.full(len(names), np.inf)
    for index, name in enumerate(names):
        coordinate = by_name.get(name)
        if coordinate is not None and coordinate.range is not None:
            lower[index], upper[index] = coordinate.range
    return lower, upper


def _energy_and_power(
    forward: Any,
    fk: Any,
    model: Any,
    times: np.ndarray,
    coordinates: np.ndarray,
    speeds: np.ndarray,
    feedforward: np.ndarray,
    feedback: np.ndarray,
    external_bodies: list[str],
    external_wrenches: np.ndarray,
    chunk_size: int,
    mass_symmetry_relative_tolerance: float,
) -> tuple[np.ndarray, ...]:
    """Evaluate energy, power, and mass-matrix validity diagnostics."""
    count = len(times)
    kinetic = np.empty(count)
    condition = np.empty(count)
    potential = np.empty(count)
    symmetry_error = np.empty(count)
    minimum_eigenvalue = np.empty(count)
    cholesky_success = np.zeros(count, dtype=bool)
    total_mass = float(sum(float(body.mass) for body in model.bodies))
    gravity = np.asarray(model.gravity, dtype=float)
    for begin in range(0, count, chunk_size):
        stop = min(count, begin + chunk_size)
        mass = np.asarray(forward.mass_matrix(coordinates[begin:stop]), dtype=float)
        expected = (stop - begin, speeds.shape[1], speeds.shape[1])
        if mass.shape != expected:
            raise ValueError(f"mass matrix has shape {mass.shape}, expected {expected}")
        velocity = speeds[begin:stop]
        kinetic[begin:stop] = 0.5 * np.einsum("bi,bij,bj->b", velocity, mass, velocity)
        condition[begin:stop] = np.linalg.cond(mass)
        scale = np.maximum(np.max(np.abs(mass), axis=(1, 2)), np.finfo(float).tiny)
        symmetry_error[begin:stop] = np.max(np.abs(mass - np.swapaxes(mass, 1, 2)), axis=(1, 2)) / scale
        symmetric_mass = 0.5 * (mass + np.swapaxes(mass, 1, 2))
        minimum_eigenvalue[begin:stop] = np.linalg.eigvalsh(symmetric_mass)[:, 0]
        for offset, matrix in enumerate(symmetric_mass):
            sample = begin + offset
            if symmetry_error[sample] > mass_symmetry_relative_tolerance:
                continue
            try:
                np.linalg.cholesky(matrix)
            except np.linalg.LinAlgError:
                continue
            cholesky_success[sample] = True
        com = np.asarray(fk.center_of_mass_batch(coordinates[begin:stop]), dtype=float)
        potential[begin:stop] = -total_mass * (com @ gravity)
    feedforward_power = np.sum(feedforward * speeds, axis=1)
    feedback_power = np.sum(feedback * speeds, axis=1)
    external_power = np.zeros(count)
    if external_bodies:
        body_index = {name: index for index, name in enumerate(fk.body_names)}
        missing = set(external_bodies).difference(body_index)
        if missing:
            raise ValueError(f"external-load bodies are absent from kinematics: {sorted(missing)!r}")
        velocities = fk.body_velocities_batch(coordinates, speeds)
        transforms = fk.body_transforms_batch(coordinates)
        angular = velocities["angular_velocity"]
        linear = velocities["linear_velocity"]
        for load_index, body in enumerate(external_bodies):
            index = body_index[body]
            wrench = external_wrenches[:, load_index]
            force = wrench[:, :3]
            point = wrench[:, 3:6]
            torque = wrench[:, 6:9]
            origin = transforms[:, index, :3, 3]
            point_velocity = linear[:, index] + np.cross(angular[:, index], point - origin)
            external_power += np.sum(force * point_velocity, axis=1) + np.sum(torque * angular[:, index], axis=1)
    result = (
        kinetic,
        potential,
        kinetic + potential,
        condition,
        symmetry_error,
        minimum_eigenvalue,
        cholesky_success,
        external_power,
        feedforward_power,
        feedback_power,
    )
    if not all(np.all(np.isfinite(values)) for values in result):
        raise ValueError("work, energy, or mass-matrix evaluation produced a nonfinite value")
    return result


def _empty_metric_arrays(length: int) -> tuple[np.ndarray, ...]:
    empty = np.full(length, np.nan)
    return tuple(empty.copy() for _ in range(10))


def run_window(
    forward: Any,
    fk: Any,
    model: Any,
    trajectory: MeasuredLoadTrajectory,
    root_mask: np.ndarray,
    *,
    start_index: int,
    duration_s: float,
    requested_dt_s: float,
    variant: str,
    config: Stage1Config,
) -> RolloutRecord:
    """Run one exact-duration window and retain its finite prefix and metrics."""
    config.validate()
    if variant not in _VARIANTS:
        raise ValueError(f"unknown Stage 1 variant {variant!r}")
    spec = _VARIANTS[variant]
    if start_index < 0 or start_index >= len(trajectory.times):
        raise ValueError("start index is outside the measured trajectory")
    if duration_s <= 0.0 or requested_dt_s <= 0.0:
        raise ValueError("duration and timestep must be positive")
    start_time = float(trajectory.times[start_index])
    if start_time + duration_s > trajectory.times[-1] + _SAMPLE_TOLERANCE_S:
        raise ValueError("requested window crosses the measured source boundary")
    steps = max(1, int(math.ceil(duration_s / requested_dt_s - 1.0e-12)))
    actual_dt = duration_s / steps
    tau_sampler = trajectory.sampler("generalized_forces", spec.interpolation)
    controller: BoundedNonRootController | None = None
    if spec.control in ("bounded_tracking", "bounded_feedforward"):
        controller = BoundedNonRootController(
            trajectory,
            root_mask,
            config.controller,
            spec.interpolation,
            feedback_enabled=spec.control == "bounded_tracking",
        )

    def controls(t: float, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        if controller is not None:
            return controller(t, q, qd)
        value = np.asarray(tau_sampler.sample(t), dtype=float).copy()
        if spec.control == "zero_root_id":
            value[np.asarray(root_mask, dtype=bool)] = 0.0
        return value

    bodies, selected_wrenches = select_load_variant(
        trajectory.external_bodies, trajectory.external_wrenches, spec.load_selection
    )
    loads = _SampledExternalLoads(trajectory.times, bodies, selected_wrenches, spec.interpolation)
    try:
        rollout = forward.simulate(
            trajectory.coordinates[start_index],
            trajectory.speeds[start_index],
            duration_s,
            actual_dt,
            start_time=start_time,
            controls=controls,
            external_loads=loads,
            integrator=config.integrator,
            use_graph=False,
        )
        times = np.asarray(rollout.times, dtype=float)
        coordinates = np.asarray(rollout.coordinates, dtype=float)
        speeds = np.asarray(rollout.speeds, dtype=float)
        status = "completed"
        exception_note = None
    except (RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exception:
        # Preserve numerical solver failures. Programming and schema errors must propagate.
        times = np.asarray([start_time])
        coordinates = trajectory.coordinates[start_index : start_index + 1].copy()
        speeds = trajectory.speeds[start_index : start_index + 1].copy()
        status = "solver_exception"
        exception_note = f"{type(exception).__name__}: {exception}"
    if coordinates.shape != speeds.shape or coordinates.shape != (len(times), len(trajectory.coordinate_names)):
        raise ValueError("ForwardDynamics returned an invalid state trajectory shape")
    time_grid_complete = bool(
        len(times) == steps + 1
        and len(times) > 1
        and np.all(np.diff(times) > 0.0)
        and abs(float(times[0]) - start_time) <= _SAMPLE_TOLERANCE_S
        and abs(float(times[-1]) - (start_time + duration_s)) <= _SAMPLE_TOLERANCE_S
    )
    if status == "completed" and not time_grid_complete:
        status = "incomplete_time_grid"
    finite_stop = _finite_prefix(coordinates, speeds)
    if finite_stop == 0:
        raise RuntimeError("forward rollout did not retain its finite initial state")
    if finite_stop < len(times):
        status = "nonfinite"
    reference_q = np.full_like(coordinates, np.nan)
    reference_qd = np.full_like(speeds, np.nan)
    reference_q[:finite_stop] = trajectory.sampler("coordinates", spec.interpolation).sample(times[:finite_stop])
    reference_qd[:finite_stop] = trajectory.sampler("speeds", spec.interpolation).sample(times[:finite_stop])
    q_error = coordinates - reference_q
    qd_error = speeds - reference_qd
    raw_ff, ff, raw_fb, fb, total, saturation = _controller_archive(
        spec, trajectory, np.asarray(root_mask, dtype=bool), config.controller, times, coordinates, speeds
    )
    marker_error = np.full((len(times), len(getattr(fk, "marker_names", []))), np.nan)
    if finite_stop and hasattr(fk, "marker_positions_batch"):
        actual_markers = np.asarray(fk.marker_positions_batch(coordinates[:finite_stop]), dtype=float)
        reference_markers = np.asarray(fk.marker_positions_batch(reference_q[:finite_stop]), dtype=float)
        marker_error = np.full((len(times), actual_markers.shape[1]), np.nan)
        marker_error[:finite_stop] = np.linalg.norm(actual_markers - reference_markers, axis=2)
    metric_arrays = _empty_metric_arrays(len(times))
    metric_error = None
    try:
        selected_at_output = StrictSampler(trajectory.times, selected_wrenches, spec.interpolation).sample(
            times[:finite_stop]
        )
        computed = _energy_and_power(
            forward,
            fk,
            model,
            times[:finite_stop],
            coordinates[:finite_stop],
            speeds[:finite_stop],
            ff[:finite_stop],
            fb[:finite_stop],
            bodies,
            selected_at_output,
            config.metric_chunk_size,
            config.mass_symmetry_relative_tolerance,
        )
        metric_arrays = tuple(
            np.concatenate((values, np.full(len(times) - finite_stop, np.nan))) for values in computed
        )
    except Exception as exception:  # State failure remains primary; metric failure is explicit in QC.
        metric_error = f"{type(exception).__name__}: {exception}"
    (
        kinetic,
        potential,
        energy,
        condition,
        mass_symmetry_error,
        mass_min_eigenvalue,
        mass_cholesky_success,
        external_power,
        ff_power,
        fb_power,
    ) = metric_arrays
    lower, upper = _coordinate_ranges(model, trajectory.coordinate_names)
    finite_q = coordinates[:finite_stop]
    range_violation = (finite_q < lower - 1.0e-10) | (finite_q > upper + 1.0e-10)
    nonroot = ~np.asarray(root_mask, dtype=bool)
    root_exact = bool(
        np.array_equal(ff[:finite_stop, root_mask], np.zeros((finite_stop, np.count_nonzero(root_mask))))
        and np.array_equal(fb[:finite_stop, root_mask], np.zeros((finite_stop, np.count_nonzero(root_mask))))
    )
    if spec.control == "full_id":
        root_exact = False
    sat_fraction = (
        float(np.count_nonzero(saturation[:finite_stop, nonroot]) / (finite_stop * np.count_nonzero(nonroot)))
        if finite_stop and np.any(nonroot)
        else 0.0
    )
    works = {}
    if metric_error is None:
        works = {
            "external_work_j": float(_cumulative_trapezoid(times[:finite_stop], external_power[:finite_stop])[-1]),
            "feedforward_work_j": float(_cumulative_trapezoid(times[:finite_stop], ff_power[:finite_stop])[-1]),
            "feedback_work_j": float(_cumulative_trapezoid(times[:finite_stop], fb_power[:finite_stop])[-1]),
        }
        works["work_energy_balance_error_j"] = float(
            energy[finite_stop - 1]
            - energy[0]
            - works["external_work_j"]
            - works["feedforward_work_j"]
            - works["feedback_work_j"]
        )
    rotational = np.asarray([kind == "rotational" for kind in trajectory.motion_types], dtype=bool)
    grouped_errors = {}
    for group_name, group_mask in (("rotational", rotational), ("translational", ~rotational)):
        grouped_errors[group_name] = {
            "coordinate_rms": _rms(q_error[:finite_stop, group_mask]),
            "coordinate_max_abs": _max_abs(q_error[:finite_stop, group_mask]),
            "speed_rms": _rms(qd_error[:finite_stop, group_mask]),
            "speed_max_abs": _max_abs(qd_error[:finite_stop, group_mask]),
        }
    per_coordinate = {}
    for index, name in enumerate(trajectory.coordinate_names):
        ff_coordinate_power = ff[:finite_stop, index] * speeds[:finite_stop, index]
        fb_coordinate_power = fb[:finite_stop, index] * speeds[:finite_stop, index]
        per_coordinate[name] = {
            "coordinate_rms": _rms(q_error[:finite_stop, index]),
            "coordinate_max_abs": _max_abs(q_error[:finite_stop, index]),
            "speed_rms": _rms(qd_error[:finite_stop, index]),
            "speed_max_abs": _max_abs(qd_error[:finite_stop, index]),
            "feedforward_rms": _rms(ff[:finite_stop, index]),
            "feedforward_max_abs": _max_abs(ff[:finite_stop, index]),
            "feedback_rms": _rms(fb[:finite_stop, index]),
            "feedback_max_abs": _max_abs(fb[:finite_stop, index]),
            "total_control_rms": _rms(total[:finite_stop, index]),
            "total_control_max_abs": _max_abs(total[:finite_stop, index]),
            "feedforward_work_j": float(_cumulative_trapezoid(times[:finite_stop], ff_coordinate_power)[-1]),
            "feedback_work_j": float(_cumulative_trapezoid(times[:finite_stop], fb_coordinate_power)[-1]),
            "saturation_fraction": float(np.mean(saturation[:finite_stop, index])),
        }
    metrics = {
        "status": status,
        "solver_exception": exception_note,
        "metric_error": metric_error,
        "completed": bool(status == "completed" and finite_stop == len(times)),
        "finite_sample_count": finite_stop,
        "sample_count": len(times),
        "time_grid_complete": time_grid_complete,
        "first_nonfinite_time_s": None if finite_stop == len(times) else float(times[finite_stop]),
        "coordinate_rms": _rms(q_error[:finite_stop]),
        "coordinate_max_abs": _max_abs(q_error[:finite_stop]),
        "speed_rms": _rms(qd_error[:finite_stop]),
        "speed_max_abs": _max_abs(qd_error[:finite_stop]),
        "error_by_motion_type": grouped_errors,
        "per_coordinate": per_coordinate,
        "marker_rms_m": _rms(marker_error[:finite_stop]),
        "marker_max_m": _max_abs(marker_error[:finite_stop]),
        "kinetic_energy_min_j": float(np.nanmin(kinetic)) if metric_error is None else None,
        "kinetic_energy_max_j": float(np.nanmax(kinetic)) if metric_error is None else None,
        "potential_energy_min_j": float(np.nanmin(potential)) if metric_error is None else None,
        "potential_energy_max_j": float(np.nanmax(potential)) if metric_error is None else None,
        "total_energy_min_j": float(np.nanmin(energy)) if metric_error is None else None,
        "total_energy_max_j": float(np.nanmax(energy)) if metric_error is None else None,
        "mass_condition_number_max": float(np.nanmax(condition)) if metric_error is None else None,
        "mass_condition_number_median": float(np.nanmedian(condition)) if metric_error is None else None,
        "mass_symmetry_relative_error_max": (float(np.nanmax(mass_symmetry_error)) if metric_error is None else None),
        "mass_min_eigenvalue_min": float(np.nanmin(mass_min_eigenvalue)) if metric_error is None else None,
        "mass_cholesky_all_success": (
            bool(np.all(mass_cholesky_success[:finite_stop])) if metric_error is None else False
        ),
        "mass_condition_within_limit": (
            bool(np.all(condition[:finite_stop] <= config.condition_number_limit)) if metric_error is None else False
        ),
        "mass_matrix_valid_spd": (
            bool(
                np.all(mass_symmetry_error[:finite_stop] <= config.mass_symmetry_relative_tolerance)
                and np.all(mass_min_eigenvalue[:finite_stop] > 0.0)
                and np.all(mass_cholesky_success[:finite_stop])
            )
            if metric_error is None
            else False
        ),
        "range_violation_count": int(np.count_nonzero(range_violation)),
        "range_violation_coordinates": [
            trajectory.coordinate_names[index] for index in np.flatnonzero(np.any(range_violation, axis=0))
        ],
        "saturation_fraction_nonroot": sat_fraction,
        "root_feedforward_feedback_exact_zero": root_exact,
        "work_quadrature": "output_grid_trapezoidal",
        **works,
    }
    return RolloutRecord(
        variant=variant,
        requested_dt_s=requested_dt_s,
        actual_dt_s=actual_dt,
        start_index=start_index,
        requested_duration_s=duration_s,
        status=status,
        times=times,
        coordinates=coordinates,
        speeds=speeds,
        reference_coordinates=reference_q,
        reference_speeds=reference_qd,
        coordinate_error=q_error,
        speed_error=qd_error,
        raw_feedforward=raw_ff,
        feedforward=ff,
        raw_feedback=raw_fb,
        feedback=fb,
        total_control=total,
        saturation=saturation,
        marker_error_m=marker_error,
        kinetic_energy_j=kinetic,
        potential_energy_j=potential,
        total_energy_j=energy,
        mass_condition_number=condition,
        mass_symmetry_relative_error=mass_symmetry_error,
        mass_min_eigenvalue=mass_min_eigenvalue,
        mass_cholesky_success=mass_cholesky_success,
        external_power_w=external_power,
        feedforward_power_w=ff_power,
        feedback_power_w=fb_power,
        metrics=metrics,
    )


_CONVERGENCE_FIELDS = {
    "coordinate": "coordinates",
    "speed": "speeds",
    "marker_error": "marker_error_m",
    "total_energy": "total_energy_j",
}


def _common_grid_comparison(
    coarse: RolloutRecord,
    fine: RolloutRecord,
    source_times: np.ndarray,
) -> dict[str, Any]:
    """Compare two trajectories directly after interpolation to one source grid."""
    coarse_stop = int(coarse.metrics["finite_sample_count"])
    fine_stop = int(fine.metrics["finite_sample_count"])
    grid = np.asarray(source_times, dtype=float)
    if coarse_stop and fine_stop:
        start_time = max(coarse.times[0], fine.times[0])
        stop_time = min(coarse.times[coarse_stop - 1], fine.times[fine_stop - 1])
        grid = grid[(grid >= start_time - _SAMPLE_TOLERANCE_S) & (grid <= stop_time + _SAMPLE_TOLERANCE_S)]
    else:
        grid = np.empty(0)
    entry: dict[str, Any] = {
        "coarse_dt_s": coarse.actual_dt_s,
        "fine_dt_s": fine.actual_dt_s,
        "common_grid_sample_count": len(grid),
        "metrics": {},
    }
    for metric_name, attribute in _CONVERGENCE_FIELDS.items():
        coarse_values = np.asarray(getattr(coarse, attribute))[:coarse_stop]
        fine_values = np.asarray(getattr(fine, attribute))[:fine_stop]
        metric = {
            "rms_difference": None,
            "max_abs_difference": None,
            "relative_rms_difference": None,
            "relative_max_abs_difference": None,
        }
        if (
            len(grid) < 2
            or coarse_stop < 2
            or fine_stop < 2
            or not np.all(np.isfinite(coarse_values))
            or not np.all(np.isfinite(fine_values))
        ):
            entry["metrics"][metric_name] = metric
            continue
        coarse_grid = StrictSampler(coarse.times[:coarse_stop], coarse_values).sample(grid)
        fine_grid = StrictSampler(fine.times[:fine_stop], fine_values).sample(grid)
        difference = coarse_grid - fine_grid
        coarse_rms = _rms(coarse_grid)
        fine_rms = _rms(fine_grid)
        coarse_max = _max_abs(coarse_grid)
        fine_max = _max_abs(fine_grid)
        if None in (coarse_rms, fine_rms, coarse_max, fine_max):
            entry["metrics"][metric_name] = metric
            continue
        rms_difference = _rms(difference)
        max_difference = _max_abs(difference)
        metric = {
            "rms_difference": rms_difference,
            "max_abs_difference": max_difference,
            "relative_rms_difference": rms_difference / max(coarse_rms, fine_rms, 1.0e-12),
            "relative_max_abs_difference": max_difference / max(coarse_max, fine_max, 1.0e-12),
        }
        entry["metrics"][metric_name] = metric
    return entry


def _comparison_within_tolerance(comparison: dict[str, Any], tolerance: float) -> bool:
    """Require every declared direct error to be present and within tolerance."""
    if comparison["common_grid_sample_count"] < 2:
        return False
    for metric in comparison["metrics"].values():
        errors = (metric["relative_rms_difference"], metric["relative_max_abs_difference"])
        if any(value is None or not np.isfinite(value) or value > tolerance for value in errors):
            return False
    return True


def finest_convergence_check(
    records: list[RolloutRecord], source_times: np.ndarray, tolerance: float
) -> tuple[bool, dict[str, Any] | None, str]:
    """Check the finest completed pair using direct common-grid trajectory errors."""
    if len(records) < 2:
        return False, None, "fewer_than_two_timestep_runs"
    fine, coarse = sorted(records, key=lambda record: record.actual_dt_s)[:2]
    if not fine.metrics.get("completed") or not coarse.metrics.get("completed"):
        return False, None, "fewer_than_two_finest_runs_completed"
    comparison = _common_grid_comparison(coarse, fine, source_times)
    if not _comparison_within_tolerance(comparison, tolerance):
        return False, comparison, "finest_pair_direct_error_exceeded_or_missing"
    return True, comparison, "finest_pair_direct_error_accepted"


def should_run_refinement(records: list[RolloutRecord], tolerance: float, source_times: np.ndarray) -> tuple[bool, str]:
    """Trigger refinement unless the finest direct common-grid errors are accepted."""
    accepted, _, reason = finest_convergence_check(records, source_times, tolerance)
    return not accepted, reason


def summarize_convergence(records: list[RolloutRecord], source_times: np.ndarray) -> dict[str, Any]:
    """Compare successive timestep trajectories on the archived measurement grid."""
    ordered = sorted(records, key=lambda record: record.actual_dt_s, reverse=True)
    comparisons = [_common_grid_comparison(coarse, fine, source_times) for coarse, fine in itertools.pairwise(ordered)]
    observed_order: dict[str, float | None] = {}
    if len(comparisons) >= 2:
        first, second = comparisons[-2:]
        ratio_1 = first["coarse_dt_s"] / first["fine_dt_s"]
        ratio_2 = second["coarse_dt_s"] / second["fine_dt_s"]
        for metric_name in _CONVERGENCE_FIELDS:
            difference_1 = first["metrics"][metric_name]["rms_difference"]
            difference_2 = second["metrics"][metric_name]["rms_difference"]
            if (
                difference_1 is None
                or difference_2 is None
                or difference_1 <= 0.0
                or difference_2 <= 0.0
                or not np.isclose(ratio_1, ratio_2, rtol=1.0e-6)
            ):
                observed_order[metric_name] = None
            else:
                observed_order[metric_name] = float(np.log(difference_1 / difference_2) / np.log(ratio_1))
    else:
        observed_order = dict.fromkeys(_CONVERGENCE_FIELDS)
    return {"successive_common_grid_comparisons": comparisons, "observed_order": observed_order}


def run_timestep_convergence(
    forward: Any,
    fk: Any,
    model: Any,
    trajectory: MeasuredLoadTrajectory,
    root_mask: np.ndarray,
    config: Stage1Config,
    *,
    refinement: Literal["auto", "always", "never"] = "auto",
) -> tuple[dict[str, list[RolloutRecord]], dict[str, Any]]:
    """Run full-interval legacy and bounded-tracking timestep sweeps."""
    if refinement not in ("auto", "always", "never"):
        raise ValueError("refinement must be 'auto', 'always', or 'never'")
    duration = float(trajectory.times[-1] - trajectory.times[0])
    result: dict[str, list[RolloutRecord]] = {}
    trigger_report: dict[str, Any] = {}
    variants = ("full_id_all_linear", "bounded_nonroot_tracking_linear")
    for variant in variants:
        records = [
            run_window(
                forward,
                fk,
                model,
                trajectory,
                root_mask,
                start_index=0,
                duration_s=duration,
                requested_dt_s=dt,
                variant=variant,
                config=config,
            )
            for dt in config.timesteps_s
        ]
        triggered, reason = should_run_refinement(records, config.convergence_relative_tolerance, trajectory.times)
        run_refinement = refinement == "always" or (refinement == "auto" and triggered)
        refinement_executed = False
        if run_refinement and config.refinement_timestep_s not in config.timesteps_s:
            records.append(
                run_window(
                    forward,
                    fk,
                    model,
                    trajectory,
                    root_mask,
                    start_index=0,
                    duration_s=duration,
                    requested_dt_s=config.refinement_timestep_s,
                    variant=variant,
                    config=config,
                )
            )
            refinement_executed = True
        accepted, final_comparison, final_reason = finest_convergence_check(
            records, trajectory.times, config.convergence_relative_tolerance
        )
        result[variant] = records
        trigger_report[variant] = {
            "triggered": triggered,
            "reason": reason,
            "policy": refinement,
            "refinement_executed": refinement_executed,
            "finest_pair_accepted": accepted,
            "finest_pair_reason": final_reason,
            "finest_pair_comparison": final_comparison,
        }
    return result, trigger_report


def _growth_slopes(times: np.ndarray, error: np.ndarray) -> np.ndarray:
    elapsed = np.asarray(times, dtype=float) - float(times[0])
    values = np.abs(np.asarray(error, dtype=float))
    if len(elapsed) < 2 or np.sum((elapsed - np.mean(elapsed)) ** 2) == 0.0:
        return np.zeros(values.shape[1])
    centered = elapsed - np.mean(elapsed)
    return centered @ (values - np.mean(values, axis=0)) / np.sum(centered**2)


def _restart_metrics(
    record: RolloutRecord,
    trajectory: MeasuredLoadTrajectory,
    config: Stage1Config,
) -> dict[str, Any]:
    finite_stop = int(record.metrics["finite_sample_count"])
    q_error = record.coordinate_error[:finite_stop]
    qd_error = record.speed_error[:finite_stop]
    rotational = np.asarray([kind == "rotational" for kind in trajectory.motion_types])
    q_scale = np.where(rotational, config.rotational_error_scale, config.translational_error_scale)
    qd_scale = np.where(rotational, config.rotational_speed_error_scale, config.translational_speed_error_scale)
    normalized = np.maximum(np.abs(q_error) / q_scale, np.abs(qd_error) / qd_scale)
    events: list[tuple[float, str, int | None]] = []
    if finite_stop < len(record.times):
        events.append((float(record.times[finite_stop]), "nonfinite", None))
    crossing = np.argwhere(normalized > 1.0)
    if crossing.size:
        sample, coordinate = crossing[0]
        events.append((float(record.times[sample]), "normalized_state_error", int(coordinate)))
    finite_condition = record.mass_condition_number[:finite_stop]
    condition_crossing = np.flatnonzero(finite_condition > config.condition_number_limit)
    if len(condition_crossing):
        sample = int(condition_crossing[0])
        events.append((float(record.times[sample]), "mass_condition_number", None))
    if record.status == "solver_exception":
        events.append((float(record.times[-1]), "solver_exception", None))
    elif record.metrics.get("metric_error") is not None:
        events.append((float(record.times[finite_stop - 1]), "metric_evaluation_failure", None))
    if events:
        event_time, event_type, event_coordinate = min(events, key=lambda event: event[0])
    else:
        event_time, event_type, event_coordinate = None, None, None
    largest_error_coordinate = (
        int(np.unravel_index(np.argmax(normalized), normalized.shape)[1]) if normalized.size else None
    )
    metrics_acceptable = bool(
        record.metrics.get("completed")
        and record.metrics.get("metric_error") is None
        and record.metrics.get("mass_condition_within_limit")
        and record.metrics.get("mass_matrix_valid_spd")
    )
    return {
        "status": record.status,
        "completed": bool(record.metrics.get("completed")),
        "metric_error": record.metrics.get("metric_error"),
        "metrics_acceptable": metrics_acceptable,
        "endpoint_coordinate_error": q_error[-1].copy(),
        "maximum_coordinate_error": np.max(np.abs(q_error), axis=0),
        "endpoint_speed_error": qd_error[-1].copy(),
        "maximum_speed_error": np.max(np.abs(qd_error), axis=0),
        "coordinate_error_growth_slope": _growth_slopes(record.times[:finite_stop], q_error),
        "speed_error_growth_slope": _growth_slopes(record.times[:finite_stop], qd_error),
        "finite_duration_s": float(record.times[finite_stop - 1] - record.times[0]),
        "event_time_s": event_time,
        "event_type": event_type,
        "event_coordinate_index": event_coordinate,
        "event_coordinate": (trajectory.coordinate_names[event_coordinate] if event_coordinate is not None else None),
        "largest_error_coordinate_index": largest_error_coordinate,
        "largest_error_coordinate": (
            trajectory.coordinate_names[largest_error_coordinate] if largest_error_coordinate is not None else None
        ),
    }


def build_restart_map(
    forward: Any,
    fk: Any,
    model: Any,
    trajectory: MeasuredLoadTrajectory,
    root_mask: np.ndarray,
    config: Stage1Config,
    *,
    start_limit: int | None = None,
) -> list[dict[str, Any]]:
    """Run all available open-loop restart windows and retain unavailable cells."""
    schedule = restart_schedule(trajectory.times, config.restart_horizons_s)
    results: list[dict[str, Any]] = []
    for cell in schedule:
        base = asdict(cell)
        if cell.status == "unavailable_source_boundary":
            results.append(base)
            continue
        if start_limit is not None and cell.start_index >= start_limit:
            results.append({**base, "status": "not_executed_by_requested_limit"})
            continue
        record = run_window(
            forward,
            fk,
            model,
            trajectory,
            root_mask,
            start_index=cell.start_index,
            duration_s=cell.horizon_s,
            requested_dt_s=config.restart_timestep_s,
            variant="full_id_all_linear",
            config=config,
        )
        results.append({**base, **_restart_metrics(record, trajectory, config)})
    return results


def run_input_decomposition(
    forward: Any,
    fk: Any,
    model: Any,
    trajectory: MeasuredLoadTrajectory,
    root_mask: np.ndarray,
    config: Stage1Config,
    *,
    reused_full_id: RolloutRecord | None = None,
) -> dict[str, RolloutRecord]:
    """Run every predeclared nominal-timestep load/control decomposition."""
    duration = float(trajectory.times[-1] - trajectory.times[0])
    result: dict[str, RolloutRecord] = {}
    for variant in _VARIANTS:
        if variant == "full_id_all_linear" and reused_full_id is not None:
            result[variant] = reused_full_id
            continue
        result[variant] = run_window(
            forward,
            fk,
            model,
            trajectory,
            root_mask,
            start_index=0,
            duration_s=duration,
            requested_dt_s=config.timesteps_s[0],
            variant=variant,
            config=config,
        )
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    """Convert nested values to strict JSON, mapping nonfinite numbers to null."""
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_value(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _record_arrays(prefix: str, record: RolloutRecord) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        f"{prefix}__times": record.times,
        f"{prefix}__coordinates": record.coordinates,
        f"{prefix}__speeds": record.speeds,
        f"{prefix}__reference_coordinates": record.reference_coordinates,
        f"{prefix}__reference_speeds": record.reference_speeds,
        f"{prefix}__coordinate_error": record.coordinate_error,
        f"{prefix}__speed_error": record.speed_error,
        f"{prefix}__raw_feedforward": record.raw_feedforward,
        f"{prefix}__feedforward": record.feedforward,
        f"{prefix}__raw_feedback": record.raw_feedback,
        f"{prefix}__feedback": record.feedback,
        f"{prefix}__total_control": record.total_control,
        f"{prefix}__saturation": record.saturation,
        f"{prefix}__marker_error_m": record.marker_error_m,
        f"{prefix}__kinetic_energy_j": record.kinetic_energy_j,
        f"{prefix}__potential_energy_j": record.potential_energy_j,
        f"{prefix}__total_energy_j": record.total_energy_j,
        f"{prefix}__mass_condition_number": record.mass_condition_number,
        f"{prefix}__mass_symmetry_relative_error": record.mass_symmetry_relative_error,
        f"{prefix}__mass_min_eigenvalue": record.mass_min_eigenvalue,
        f"{prefix}__mass_cholesky_success": record.mass_cholesky_success,
        f"{prefix}__external_power_w": record.external_power_w,
        f"{prefix}__feedforward_power_w": record.feedforward_power_w,
        f"{prefix}__feedback_power_w": record.feedback_power_w,
    }
    return arrays


def _dt_key(value: float) -> str:
    return f"dt_{value:.9f}".replace(".", "p")


def _restart_arrays(results: list[dict[str, Any]], coordinate_count: int) -> dict[str, np.ndarray]:
    scalar_fields = ("start_index", "start_time_s", "horizon_s", "finite_duration_s", "event_time_s")
    arrays: dict[str, np.ndarray] = {}
    for field_name in scalar_fields:
        arrays[field_name] = np.asarray(
            [np.nan if result.get(field_name) is None else result.get(field_name) for result in results], dtype=float
        )
    arrays["status"] = np.asarray([result["status"] for result in results], dtype="U")
    arrays["event_type"] = np.asarray([result.get("event_type") or "" for result in results], dtype="U")
    for field_name in ("event_coordinate_index", "largest_error_coordinate_index"):
        arrays[field_name] = np.asarray(
            [-1 if result.get(field_name) is None else result[field_name] for result in results],
            dtype=int,
        )
    for field_name in (
        "endpoint_coordinate_error",
        "maximum_coordinate_error",
        "endpoint_speed_error",
        "maximum_speed_error",
        "coordinate_error_growth_slope",
        "speed_error_growth_slope",
    ):
        values = np.full((len(results), coordinate_count), np.nan)
        for index, result in enumerate(results):
            if field_name in result:
                values[index] = result[field_name]
        arrays[field_name] = values
    return arrays


def _git_runtime(repository_root: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository_root, check=True, text=True, capture_output=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], cwd=repository_root, check=True, text=True, capture_output=True
        ).stdout.strip()
    )
    return {"git_commit": commit, "git_dirty": dirty}


def _validate_paths(data_dir: Path, output_dir: Path, repository_root: Path) -> None:
    if output_dir == repository_root or output_dir.is_relative_to(repository_root):
        raise ValueError("generated human-data diagnostics must stay outside the repository")
    if output_dir == data_dir or output_dir.is_relative_to(data_dir) or data_dir.is_relative_to(output_dir):
        raise ValueError("diagnostic and source artifact directories must not overlap")
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError("diagnostic output path exists and is not a directory")


def _record_metrics_acceptable(record: RolloutRecord, config: Stage1Config) -> bool:
    """Require completion plus finite, conditioned, symmetric positive-definite mass diagnostics."""
    metrics = record.metrics
    condition = metrics.get("mass_condition_number_max")
    symmetry = metrics.get("mass_symmetry_relative_error_max")
    minimum_eigenvalue = metrics.get("mass_min_eigenvalue_min")
    return bool(
        metrics.get("completed")
        and metrics.get("metric_error") is None
        and condition is not None
        and condition <= config.condition_number_limit
        and symmetry is not None
        and symmetry <= config.mass_symmetry_relative_tolerance
        and minimum_eigenvalue is not None
        and minimum_eigenvalue > 0.0
        and metrics.get("mass_cholesky_all_success")
    )


def _tracking_gate(
    convergence: dict[str, list[RolloutRecord]],
    restart_results: list[dict[str, Any]],
    decomposition: dict[str, RolloutRecord],
    sections: tuple[str, ...],
    config: Stage1Config,
    source_times: np.ndarray,
) -> dict[str, Any]:
    """Gate direct convergence and the actual completion of every canonical path."""
    records = convergence.get("bounded_nonroot_tracking_linear", [])
    by_requested = {round(record.requested_dt_s, 9): record for record in records}
    required = [by_requested.get(round(value, 9)) for value in (0.001, 0.0005)]
    complete = len(required) == 2 and all(record is not None and record.metrics["completed"] for record in required)
    nominal = by_requested.get(round(0.001, 9))
    metric_names = (
        "external_work_j",
        "feedforward_work_j",
        "feedback_work_j",
        "kinetic_energy_min_j",
        "potential_energy_min_j",
        "mass_condition_number_max",
        "mass_symmetry_relative_error_max",
        "mass_min_eigenvalue_min",
        "coordinate_rms",
        "marker_rms_m",
    )
    metric_presence = nominal is not None and all(nominal.metrics.get(name) is not None for name in metric_names)
    convergence_accepted, finest_comparison, convergence_reason = finest_convergence_check(
        records, source_times, config.convergence_relative_tolerance
    )
    scheduled_restarts = [result for result in restart_results if result.get("status") != "unavailable_source_boundary"]
    restarts_completed = bool(
        scheduled_restarts
        and all(result.get("metrics_acceptable") is True for result in scheduled_restarts)
        and all(
            result.get("status") == "unavailable_source_boundary" or result.get("metrics_acceptable") is True
            for result in restart_results
        )
    )
    decomposition_completed = bool(
        set(decomposition) == set(_VARIANTS)
        and all(_record_metrics_acceptable(record, config) for record in decomposition.values())
    )
    gates = {
        "full_interval_finite_at_1ms_and_0p5ms": complete,
        "finest_pair_direct_common_grid_error_within_tolerance": convergence_accepted,
        "root_commanded_force_exact_zero": bool(nominal and nominal.metrics["root_feedforward_feedback_exact_zero"]),
        "no_declared_coordinate_range_violation": bool(nominal and nominal.metrics["range_violation_count"] == 0),
        "controller_saturation_below_1_percent": bool(
            nominal and nominal.metrics["saturation_fraction_nonroot"] < 0.01
        ),
        "marker_rms_below_30mm": bool(
            nominal and nominal.metrics["marker_rms_m"] is not None and nominal.metrics["marker_rms_m"] < 0.03
        ),
        "marker_max_below_60mm": bool(
            nominal and nominal.metrics["marker_max_m"] is not None and nominal.metrics["marker_max_m"] < 0.06
        ),
        "required_metrics_present": bool(metric_presence),
        "mass_condition_number_within_limit": bool(
            nominal
            and nominal.metrics.get("mass_condition_number_max") is not None
            and nominal.metrics["mass_condition_number_max"] <= config.condition_number_limit
        ),
        "mass_matrix_symmetric_positive_definite": bool(
            nominal
            and nominal.metrics.get("mass_symmetry_relative_error_max") is not None
            and nominal.metrics["mass_symmetry_relative_error_max"] <= config.mass_symmetry_relative_tolerance
            and nominal.metrics.get("mass_min_eigenvalue_min") is not None
            and nominal.metrics["mass_min_eigenvalue_min"] > 0.0
            and nominal.metrics.get("mass_cholesky_all_success")
        ),
        "all_scheduled_restarts_completed_with_acceptable_metrics": restarts_completed,
        "all_decomposition_variants_completed_with_acceptable_metrics": decomposition_completed,
        "all_canonical_sections_selected": set(sections) == {"convergence", "restarts", "decomposition"},
    }
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "nominal_metrics": None if nominal is None else nominal.metrics,
        "convergence_tolerance": config.convergence_relative_tolerance,
        "finest_pair_reason": convergence_reason,
        "finest_pair_comparison": finest_comparison,
    }


def _recover_staged_publication(output_dir: Path) -> str:
    """Restore an accepted backup left by an interrupted staged replacement."""
    parent = output_dir.parent
    backups = sorted(parent.glob(f".{output_dir.name}.previous-*"))
    staging_dirs = sorted(parent.glob(f".{output_dir.name}.staging-*"))
    if output_dir.exists():
        for path in (*backups, *staging_dirs):
            shutil.rmtree(path)
        return "accepted_output_present"
    if len(backups) > 1:
        raise RuntimeError(f"multiple interrupted publication backups require manual recovery: {backups!r}")
    if backups:
        os.replace(backups[0], output_dir)
        for path in staging_dirs:
            shutil.rmtree(path)
        return "restored_previous_output"
    for path in staging_dirs:
        shutil.rmtree(path)
    return "no_previous_output"


def publish_artifacts(
    output_dir: Path,
    manifest: dict[str, Any],
    qc: dict[str, Any],
    controller_config: ControllerConfig,
    convergence: dict[str, list[RolloutRecord]],
    restart_results: list[dict[str, Any]],
    decomposition: dict[str, RolloutRecord],
    coordinate_count: int,
) -> Path:
    """Publish by staged replacement with next-run recovery after interruption.

    Directory replacement has a short interval in which ``output_dir`` is absent.
    A surviving backup is restored at the start of the next publication attempt.
    """
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    _recover_staged_publication(output_dir)
    staging = output_dir.parent / f".{output_dir.name}.staging-{os.getpid()}-{uuid.uuid4().hex}"
    backup = output_dir.parent / f".{output_dir.name}.previous-{os.getpid()}-{uuid.uuid4().hex}"
    staging.mkdir(parents=True)
    try:
        _write_json(staging / "manifest.json", manifest)
        _write_json(staging / "qc_summary.json", qc)
        _write_json(staging / "controller_config.json", asdict(controller_config))
        convergence_arrays: dict[str, np.ndarray] = {}
        for variant, records in convergence.items():
            run_keys = [_dt_key(record.requested_dt_s) for record in records]
            if len(set(run_keys)) != len(run_keys):
                raise ValueError(f"convergence records for {variant!r} collide after NPZ key encoding")
            for run_key, record in zip(run_keys, records, strict=True):
                convergence_arrays.update(_record_arrays(f"{variant}__{run_key}", record))
        np.savez_compressed(staging / "convergence.npz", **convergence_arrays)
        np.savez_compressed(staging / "restart_map.npz", **_restart_arrays(restart_results, coordinate_count))
        decomposition_arrays: dict[str, np.ndarray] = {}
        for variant, record in decomposition.items():
            decomposition_arrays.update(_record_arrays(variant, record))
        np.savez_compressed(staging / "input_decomposition.npz", **decomposition_arrays)
        if output_dir.exists():
            os.replace(output_dir, backup)
        try:
            os.replace(staging, output_dir)
        except Exception:
            if backup.exists() and not output_dir.exists():
                os.replace(backup, output_dir)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        if backup.exists() and not output_dir.exists():
            os.replace(backup, output_dir)
        raise
    return output_dir


def run_stage1(
    data_dir: str | os.PathLike = _DEFAULT_DATA,
    output_dir: str | os.PathLike | None = None,
    *,
    device: str = "cpu",
    config: Stage1Config | None = None,
    sections: tuple[str, ...] = ("convergence", "restarts", "decomposition"),
    refinement: Literal["auto", "always", "never"] = "auto",
    restart_start_limit: int | None = None,
) -> Path:
    """Execute selected heavy Stage 1 paths and publish all artifact files by staged replacement.

    Args:
        data_dir: Completed Trial 101 gait-analysis artifact.
        output_dir: Separate Stage 1 destination outside the repository.
        device: Warp device used for OpenSim dynamics and kinematics.
        config: Frozen diagnostic configuration.
        sections: Heavy paths to execute. Omitted sections are archived as empty
            and make the stage gate fail.
        refinement: Policy for the optional 0.125 ms convergence run.
        restart_start_limit: Optional probe-only start-frame limit. Limited cells
            remain explicit and cannot pass the complete-section gate.

    Returns:
        Path to the completed Stage 1 artifact.
    """
    started = time.monotonic()
    config = Stage1Config() if config is None else config
    config.validate()
    if (
        not sections
        or len(set(sections)) != len(sections)
        or set(sections).difference({"convergence", "restarts", "decomposition"})
    ):
        raise ValueError("sections must be unique selections of convergence, restarts, and decomposition")
    if restart_start_limit is not None and restart_start_limit < 0:
        raise ValueError("restart_start_limit must be nonnegative")
    data_dir = Path(data_dir).resolve()
    output_dir = (
        Path(output_dir).resolve()
        if output_dir is not None
        else data_dir.parent / "stage1_engineering_measured_load_tracking"
    )
    repository_root = Path(__file__).resolve().parents[3]
    _validate_paths(data_dir, output_dir, repository_root)
    model_path = data_dir / "S001_scaled.osim"
    analysis_path = data_dir / "analysis.npz"
    source_manifest_path = data_dir / "manifest.json"
    for path in (model_path, analysis_path, source_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    model = osim.parse_osim(model_path)
    forward = osim.ForwardDynamics(model, device=device)
    fk = osim.ForwardKinematics(model, device=device)
    trajectory = MeasuredLoadTrajectory.from_analysis(analysis_path, list(forward.motion_types))
    if list(trajectory.coordinate_names) != list(forward.coordinate_names):
        raise ValueError("analysis coordinate order does not match ForwardDynamics")
    root_mask = structural_root_mask(model, trajectory.coordinate_names)
    convergence: dict[str, list[RolloutRecord]] = {}
    trigger_report: dict[str, Any] = {}
    if "convergence" in sections:
        convergence, trigger_report = run_timestep_convergence(
            forward, fk, model, trajectory, root_mask, config, refinement=refinement
        )
    restart_results: list[dict[str, Any]] = []
    if "restarts" in sections:
        restart_results = build_restart_map(
            forward,
            fk,
            model,
            trajectory,
            root_mask,
            config,
            start_limit=restart_start_limit,
        )
    decomposition: dict[str, RolloutRecord] = {}
    if "decomposition" in sections:
        reused = None
        baseline = convergence.get("full_id_all_linear", [])
        if baseline:
            reused = min(baseline, key=lambda record: abs(record.requested_dt_s - config.timesteps_s[0]))
        decomposition = run_input_decomposition(
            forward, fk, model, trajectory, root_mask, config, reused_full_id=reused
        )
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    gate = _tracking_gate(convergence, restart_results, decomposition, sections, config, trajectory.times)
    if restart_start_limit is not None:
        gate["gates"]["all_scheduled_restarts_completed_with_acceptable_metrics"] = False
        gate["passed"] = False
    manifest = {
        "schema_version": _SCHEMA,
        "architecture_role": ARCHITECTURE_ROLE,
        "reference_only": True,
        "production_eligible": False,
        "scope": _SCOPE,
        "status": "engineering_stage1_passed" if gate["passed"] else "engineering_stage1_failed_or_incomplete",
        "runtime": {**_git_runtime(repository_root), "device": device, "wall_time_s": time.monotonic() - started},
        "source": {
            "data_dir": str(data_dir),
            "analysis_sha256": _sha256(analysis_path),
            "model_sha256": _sha256(model_path),
            "manifest_sha256": _sha256(source_manifest_path),
            "status": source_manifest.get("status"),
            "git_commit": source_manifest.get("runtime", {}).get("git_commit"),
        },
        "settings": asdict(config),
        "executed_sections": list(sections),
        "refinement": trigger_report,
        "restart_start_limit": restart_start_limit,
        "coordinate_names": list(trajectory.coordinate_names),
        "motion_types": list(trajectory.motion_types),
        "root_coordinates": list(np.asarray(trajectory.coordinate_names)[root_mask]),
        "nonroot_coordinates": list(np.asarray(trajectory.coordinate_names)[~root_mask]),
        "source_interval_s": [float(trajectory.times[0]), float(trajectory.times[-1])],
        "warnings": [
            "Measured external wrenches are replayed; this result is not predictive gait.",
            "The legacy full-ID variants explicitly include six root residual generalized forces.",
            "Only bounded_nonroot_tracking_linear is eligible for the Stage 1 tracking gate.",
        ],
    }
    qc = {
        "schema_version": _SCHEMA,
        "scope": _SCOPE,
        "status": manifest["status"],
        "gate": gate,
        "convergence": {
            variant: {
                "runs": [
                    {
                        "requested_dt_s": record.requested_dt_s,
                        "actual_dt_s": record.actual_dt_s,
                        "metrics": record.metrics,
                    }
                    for record in records
                ],
                **summarize_convergence(records, trajectory.times),
            }
            for variant, records in convergence.items()
        },
        "restart_map": {
            "cell_count": len(restart_results),
            "status_counts": {
                status: sum(result["status"] == status for result in restart_results)
                for status in sorted({result["status"] for result in restart_results})
            },
            "cells": restart_results,
        },
        "input_decomposition": {variant: record.metrics for variant, record in decomposition.items()},
        "comparison_to_stage0": {
            "source_status": source_manifest.get("status"),
            "note": "Stage 1 adds integration, restart, controller, work, energy, and conditioning diagnostics; it does not change Stage 0 inputs.",
        },
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    result = publish_artifacts(
        output_dir,
        manifest,
        qc,
        config.controller,
        convergence,
        restart_results,
        decomposition,
        len(trajectory.coordinate_names),
    )
    print(f"[gait_c3d] wrote {result}")
    print(f"[gait_c3d] scope={_SCOPE} status={manifest['status']}")
    return result


def create_parser() -> argparse.ArgumentParser:
    """Build the measured-load diagnostic command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-only",
        action="store_true",
        help="required acknowledgement: this uses newton.opensim compatibility dynamics",
    )
    parser.add_argument("--data-dir", default=str(_DEFAULT_DATA))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--section",
        dest="sections",
        action="append",
        choices=("convergence", "restarts", "decomposition"),
        help="Heavy path to run; repeat as needed. The default runs all three.",
    )
    parser.add_argument("--refinement", choices=("auto", "always", "never"), default="auto")
    parser.add_argument(
        "--restart-start-limit",
        type=int,
        default=None,
        help="Probe only the first N restart frames; this makes the artifact incomplete.",
    )
    return parser


def main() -> None:
    """Run the Stage 1 command-line harness."""
    parser = create_parser()
    args = parser.parse_args()
    if not args.reference_only:
        parser.error("--reference-only is required; this diagnostic is not a Newton-native rollout")
    run_stage1(
        args.data_dir,
        args.output_dir,
        device=args.device,
        sections=tuple(args.sections or ("convergence", "restarts", "decomposition")),
        refinement=args.refinement,
        restart_start_limit=args.restart_start_limit,
    )


if __name__ == "__main__":
    main()
