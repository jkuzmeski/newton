# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental fixed-impedance response, separate from the policy rig.

Intent mode applies frozen reduced-rig inverse-dynamics loads at runtime. This
is declared nominal assistance, not predictive anatomy or a measured-force
reward. It does not remove the frozen model's unactuated force residuals.
Equilibrium mode keeps the existing equilibrium schedules. Neither mode fits,
shifts, or mutates the reference, shoe geometry, or constitutive law.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import ClassVar

import numpy as np
import warp as wp

from .reference import Reference
from .rig import TRACE_NAMES, Rig, RigConfig, _identity


@dataclass(frozen=True)
class ResponseConfig:
    """Select fixed gains, a static plane, and one smooth upper-body push.

    Stiffness units are [N/m] and [N·m/rad]. Damping units are [N·s/m] and
    [N·m·s/rad]. Omitted damping resolves from the frozen nominal construction,
    independently of the selected stiffness. Explicit zero damping is valid.
    ``ground_height_m`` is an absolute static plane offset [m], not a timed
    step. ``push_force_x_n`` and ``push_force_z_n`` are signed peak forces [N].
    The pulse is ``0.5*peak*(1-cos(2*pi*(t-start)/duration))`` inside its support.
    Its exact impulse is ``peak*duration/2`` [N·s]. Each integration interval
    receives its analytic average force, including fractional boundary cells.
    """

    controller_mode: str = "equilibrium"
    leg_stiffness_n_m: float = 12000.0
    ankle_stiffness_n_m_rad: float = 4000.0
    leg_damping_n_s_m: float | None = None
    ankle_damping_n_m_s_rad: float | None = None
    ground_height_m: float = 0.0
    push_start_s: float = 0.0
    push_duration_s: float = 0.0
    push_force_x_n: float = 0.0
    push_force_z_n: float = 0.0

    def __post_init__(self):
        if self.controller_mode not in ("equilibrium", "intent"):
            raise ValueError("controller_mode must be equilibrium or intent")
        for name, value in self.to_dict().items():
            if name == "controller_mode" or (
                value is None and name in ("leg_damping_n_s_m", "ankle_damping_n_m_s_rad")
            ):
                continue
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or abs(value) > float(np.finfo(np.float32).max)
            ):
                raise ValueError(f"{name} must be finite in float32")
        if min(self.leg_stiffness_n_m, self.ankle_stiffness_n_m_rad) <= 0:
            raise ValueError("Response stiffness must be positive")
        for value in (self.leg_damping_n_s_m, self.ankle_damping_n_m_s_rad):
            if value is not None and value < 0:
                raise ValueError("Response damping must be nonnegative")
        if min(self.push_start_s, self.push_duration_s) < 0:
            raise ValueError("Push start and duration must be nonnegative")
        if self.push_duration_s == 0 and (self.push_force_x_n != 0 or self.push_force_z_n != 0):
            raise ValueError("A nonzero push needs positive duration")

    def to_dict(self) -> dict:
        """Return every response setting without altering the policy rig config."""
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> ResponseConfig:
        """Restore response settings and reject unknown control fields."""
        return cls(**value)

    def resolved(self, reference: Reference) -> ResponseConfig:
        """Resolve omitted damping from nominal gains, never selected stiffness."""
        cfg = reference.provenance.get("config", {})
        inverse = reference.provenance.get("inverse_dynamics", {})
        values = {}
        for name, ratio, gain, mass in (
            (
                "leg_damping_n_s_m",
                "leg_damping_ratio",
                "nominal_leg_stiffness_n_m",
                reference.mass_kg - cfg["foot_mass_kg"],
            ),
            (
                "ankle_damping_n_m_s_rad",
                "ankle_damping_ratio",
                "nominal_ankle_stiffness_n_m_rad",
                cfg["pitch_inertia_kg_m2"],
            ),
        ):
            if getattr(self, name) is None:
                values[name] = float(inverse[name]) if name in inverse else 2 * cfg[ratio] * math.sqrt(cfg[gain] * mass)
        return replace(self, **values)

    def push_interval_averages(self, *, duration_s: float, sample_count: int) -> np.ndarray:
        """Return analytic interval-average upper-body forces, shape [N, 2] [N]."""
        if not math.isfinite(duration_s) or duration_s <= 0 or sample_count < 1 or int(sample_count) != sample_count:
            raise ValueError("A positive episode duration and integer sample count are required")
        if self.push_start_s + self.push_duration_s > duration_s:
            raise ValueError("Push support must be contained in the episode")
        if self.push_duration_s == 0:
            return np.zeros((sample_count, 2))
        dt = duration_s / sample_count
        edges = np.arange(sample_count + 1, dtype=float) * dt
        phase = np.clip((edges - self.push_start_s) / self.push_duration_s, 0.0, 1.0)
        primitive = self.push_duration_s * (0.5 * phase - np.sin(2 * np.pi * phase) / (4 * np.pi))
        weights = np.diff(primitive) / dt
        return weights[:, None] * np.array([self.push_force_x_n, self.push_force_z_n])


RESPONSE_TRACE_NAMES = (
    "movement_leg_length_m",
    "movement_leg_rate_m_s",
    "inverse_leg_force_n",
    "inverse_ankle_torque_n_m",
    "leg_nominal_force_n",
    "ankle_nominal_torque_n_m",
    "leg_feedback_force_n",
    "ankle_feedback_torque_n_m",
    "push_force_x_n",
    "push_force_z_n",
    "push_power_w",
    "ground_height_m",
    "leg_nominal_body_power_w",
    "ankle_nominal_body_power_w",
    "leg_target_motion_power_w",
    "ankle_target_motion_power_w",
    "leg_stiffness_source_power_w",
    "ankle_stiffness_source_power_w",
)


class RigResponse(Rig):
    """Run fixed equilibrium or nominal-assisted movement-intent response.

    Args:
        reference: Frozen optical movement and reduced-rig ID schedules.
        artifact_path: Identified shoe; only same-geometry material may differ.
        response_config: Experimental controller and perturbation settings.
        config: Original physical rig settings with unchanged construction checks.
        num_worlds: Independent copies of the same response experiment.
        device: Warp device such as ``"cpu"`` or ``"cuda:0"``.

    Intent force is ``F_ID + K*(Lref-L) + B*(Lrefdot-Ldot)``. Pitch torque is
    ``tau_ID + Ka*wrap(pitchref-pitch) + Ba*(pitchrefdot-omega)``. Measured leg
    movement comes from ID provenance, NOT ``Reference.leg_length_m`` (which
    stores the old equilibrium). ID loads use the original linear interpolation.
    The existing signed limits still apply to the sum of nominal and feedback
    loads. In intent traces, legacy equilibrium fields denote the movement
    target. Stored spring energy uses tracking error; nominal assistance enters
    source power through actual body velocity. Push work is external and separate.
    """

    _trace_names = TRACE_NAMES + RESPONSE_TRACE_NAMES
    action_contract: ClassVar[dict] = {
        "version": "experimental_fixed_response_v1",
        "names": ["unused_zero_leg", "unused_zero_ankle"],
        "bounds": [0.0, 0.0],
        "mapping": "zero actions only; response gains fixed independently of policy controls",
        "clock": "fixed nominal physical time; no phase adaptation",
    }

    def __init__(
        self,
        reference: Reference,
        artifact_path: str | Path,
        *,
        response_config: ResponseConfig | None = None,
        config: RigConfig | None = None,
        num_worlds: int = 1,
        device: str | None = None,
    ):
        self.requested_response_config = response_config or ResponseConfig()
        self.response_config = self.requested_response_config.resolved(reference)
        self._ground_height_m = self.response_config.ground_height_m
        super().__init__(reference, artifact_path, config=config, num_worlds=num_worlds, device=device)
        response = self.response_config
        self.response_fingerprints = {
            "response_config_sha256": _identity(response.to_dict()),
            "response_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "rig_source_sha256": self.input_fingerprints["rig_source_sha256"],
        }
        self.metadata.update(
            {
                "response_config": response.to_dict(),
                "requested_response_config": self.requested_response_config.to_dict(),
                "response_fingerprints": self.response_fingerprints,
                "experimental_response": True,
                "nominal_assistance": "frozen reduced-rig ID loads applied only in intent mode; not predictive anatomy",
                "stiffness_interpolation": "fixed response stiffness; no learned gain controls",
                "source_power": (
                    "nominal_load*body_rate + feedback_load*target_rate + 0.5*Kdot*error^2 + limit_intervention_power"
                    if response.controller_mode == "intent"
                    else self.metadata["source_power"]
                ),
                "contact": "unchanged shoe law; static ground plane; no geometry or reference shift",
                "observation_height_frame": "pelvis, foot and reference pelvis heights relative to static plane",
                "response_target_fields": "legacy equilibrium fields are movement targets in intent mode",
                "movement_reference_available": self._movement_reference_available,
                "movement_reference_missing": "zero diagnostic placeholders for legacy equilibrium references only",
                "load_split": "nominal and feedback are pre-limit loads; only their sum is clipped",
                "shoe_contact_power_scope": "rigid carrier wrench power, not a full foam/bristle energy ledger",
                "push_profile": "raised cosine; signed peak force; exact interval average on upper body",
                "requested_push_impulse_n_s": [
                    response.push_force_x_n * response.push_duration_s / 2,
                    response.push_force_z_n * response.push_duration_s / 2,
                ],
                "discrete_push_impulse_n_s": (
                    self._push_device.numpy().astype(float).sum(axis=0) * self.sim_dt
                ).tolist(),
            }
        )

    def _prepare_runtime(self):
        response, reference = self.response_config, self.reference
        inverse = reference.provenance.get("inverse_dynamics", {})
        self._movement_reference_available = all(
            name in inverse for name in ("leg_measured_length_m", "leg_measured_rate_m_s")
        )
        columns = []
        for name in ("leg_measured_length_m", "leg_measured_rate_m_s"):
            if name not in inverse:
                if response.controller_mode == "intent":
                    raise ValueError(f"Intent mode requires frozen inverse_dynamics.{name}")
                # Legacy equilibrium references need no nominal movement data.
                value = np.zeros_like(reference.time_s)
            else:
                value = np.asarray(inverse[name], dtype=float)
                if value.shape != reference.time_s.shape or not np.isfinite(value).all():
                    raise ValueError(f"inverse_dynamics.{name} must match finite reference knots")
                if name == "leg_measured_length_m" and np.any(value <= 0):
                    raise ValueError("Measured movement leg length must be positive")
            columns.append(value)
        columns.extend((reference.inverse_leg_force_n, reference.inverse_ankle_torque_n_m))
        self._reference_device = wp.array(
            np.column_stack((self._reference_device.numpy(), *columns)), dtype=wp.float32, device=self.device
        )
        self._push_device = wp.array(
            response.push_interval_averages(duration_s=self.duration, sample_count=self.sample_count),
            dtype=wp.float32,
            device=self.device,
        )
        selected = np.array([response.leg_stiffness_n_m, response.ankle_stiffness_n_m_rad])
        if not np.allclose(selected, np.exp(self._log_initial), rtol=1e-14, atol=0):
            self._log_initial = np.log(selected)

    def _make_parameters(self):
        prm = super()._make_parameters()
        prm.response_mode = 2 if self.response_config.controller_mode == "intent" else 1
        prm.damping_leg = self.response_config.leg_damping_n_s_m
        prm.damping_ankle = self.response_config.ankle_damping_n_m_s_rad
        return prm

    def reset(self) -> np.ndarray:
        """Restore the static plane as well as all inherited body/contact history."""
        observation = super().reset()
        self.foundation.z_free.fill_(self._ground_height_m)
        return observation

    def _update_stiffness(self, action):
        if np.any(action != 0):
            raise ValueError("RigResponse accepts only zero actions; response gains are fixed")

    def step(self, action) -> tuple[np.ndarray, np.ndarray, bool, dict]:
        """Advance with exactly zero [worlds, 2] actions; reject policy controls."""
        observation, reward, done, info = super().step(action)
        info["response_fingerprints"] = self.response_fingerprints
        return observation, reward, done, info

    def trace(self, world: int = 0) -> dict[str, np.ndarray]:
        """Return actuator accounting and separate external push work [J]."""
        result = super().trace(world)
        result["push_work_j"] = np.cumsum(result["push_power_w"]) * self.sim_dt
        return result
