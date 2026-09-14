# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Freeze an offline reduced-rig inverse-dynamics reference, never a force command.

Only optical pelvis height and heel-cluster pitch enter :func:`tracking_error`.
Measured platform wrenches construct nominal equilibrium schedules offline and
remain evaluation data at runtime. The two-mass model cannot in general reproduce
the acquisition: its unactuated force residuals are retained, not fitted away.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

SCHEMA = "impedance_simple_reference_1"
_PAIRS = {
    "pelvis_z_m": "pelvis_vz_m_s",
    "pitch_rad": "pitch_rate_rad_s",
    "leg_length_m": "leg_rate_m_s",
    "ankle_equilibrium_rad": "ankle_equilibrium_rate_rad_s",
}
_LINEAR = ("reference_fz_n", "reference_fx_n", "inverse_leg_force_n", "inverse_ankle_torque_n_m")
_VECTORS = ("foot_position_m", "upper_position_m", "foot_velocity_m_s", "upper_velocity_m_s")
_DEFAULT_CONFIG = {
    "foot_mass_kg": 2.0,
    "pitch_inertia_kg_m2": 0.025,
    "ankle_mount_local_m": [-0.075, 0.0, 0.105],
    "nominal_leg_stiffness_n_m": 12000.0,
    "nominal_ankle_stiffness_n_m_rad": 4000.0,
    "leg_damping_ratio": 0.25,
    "ankle_damping_ratio": 0.5,
    "gravity_m_s2": 9.80665,
    "smoothing_cutoff_hz": 12.0,
    "source_side": "right",
    "target_side": "left",
    "contact_force_fraction": 0.02,
}


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _identity(value) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def geometry_identity(shoe) -> str:
    """Hash oriented geometry, excluding material and unrelated provenance.

    Args:
        shoe: A public ``DigitalShoe`` returned by ``orient_shoe``.
    """
    return _identity(
        {key: shoe.raw[key] for key in ("coordinate_system", "column_bed", "visual_meshes", "instron_fixtures")}
    )


def tracking_error(pelvis_z, pitch, reference_z, reference_pitch, pelvis_scale, pitch_scale):
    """Return half the sum of two normalized squared optical tracking errors.

    Heights and height scale are in [m]. Angles and angle scale are in [rad].
    Angle differences wrap to [-pi, pi). Inputs broadcast using NumPy rules.
    There are no force, work, contact, duration, clipping, or safety terms.
    """
    values = [
        np.asarray(value, dtype=float)
        for value in (pelvis_z, pitch, reference_z, reference_pitch, pelvis_scale, pitch_scale)
    ]
    if any(not np.all(np.isfinite(value)) for value in values):
        raise ValueError("tracking inputs must be finite")
    z, angle, target_z, target_angle, sz, sa = values
    if np.any(sz <= 0.0) or np.any(sa <= 0.0):
        raise ValueError("tracking scales must be positive")
    delta = (angle - target_angle + np.pi) % (2.0 * np.pi) - np.pi
    return 0.5 * (((z - target_z) / sz) ** 2 + (delta / sa) ** 2)


@dataclass(frozen=True)
class Reference:
    """Portable frozen schedules on a physical clock starting at zero.

    All scalar curves have shape [sample_count]. Initial vectors have shape [3]
    in the +X-forward, +Y-left, +Z-up rig frame. ``mass_kg`` is the TOTAL two-mass
    rig mass. ``leg_length_m`` and ``leg_rate_m_s`` are equilibrium L0 and L0dot,
    NOT measured leg length. ``pitch_rad`` is the processed optical target, NOT
    the ankle equilibrium. ``provenance`` includes raw optical knots, processing,
    mounting assumptions, resolved nominal gains, source hashes, and ID residuals.
    """

    time_s: np.ndarray
    pelvis_z_m: np.ndarray
    pelvis_vz_m_s: np.ndarray
    pitch_rad: np.ndarray
    pitch_rate_rad_s: np.ndarray
    leg_length_m: np.ndarray
    leg_rate_m_s: np.ndarray
    ankle_equilibrium_rad: np.ndarray
    ankle_equilibrium_rate_rad_s: np.ndarray
    inverse_leg_force_n: np.ndarray
    inverse_ankle_torque_n_m: np.ndarray
    reference_fz_n: np.ndarray
    reference_fx_n: np.ndarray
    foot_position_m: np.ndarray
    upper_position_m: np.ndarray
    foot_velocity_m_s: np.ndarray
    upper_velocity_m_s: np.ndarray
    mass_kg: float
    gravity_m_s2: float
    contact_start_s: float
    contact_duration_s: float
    pelvis_scale_m: float
    pitch_scale_rad: float
    provenance: dict

    def __post_init__(self):
        time = np.asarray(self.time_s, dtype=float)
        if time.ndim != 1 or len(time) < 2 or time[0] != 0 or np.any(np.diff(time) <= 0):
            raise ValueError("reference time must increase strictly from zero")
        for name in ("time_s", *_PAIRS, *_PAIRS.values(), *_LINEAR, *_VECTORS):
            array = np.array(getattr(self, name), dtype=float, copy=True)
            shape = (3,) if name in _VECTORS else time.shape
            if array.shape != shape or not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must be finite with shape {shape}")
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        for name in ("mass_kg", "gravity_m_s2", "contact_duration_s", "pelvis_scale_m", "pitch_scale_rad"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)
        start = float(self.contact_start_s)
        if not np.isfinite(start) or start < 0 or start + self.contact_duration_s > time[-1] + 1e-9:
            raise ValueError("reference must contain its measured contact interval")
        object.__setattr__(self, "contact_start_s", start)
        if not isinstance(self.provenance, dict):
            raise ValueError("provenance must be an object")
        _canonical(self.provenance)
        object.__setattr__(self, "provenance", copy.deepcopy(self.provenance))

    @property
    def duration_s(self) -> float:
        """Full exported physical duration, including flight padding [s]."""
        return float(self.time_s[-1])

    def _body(self) -> dict:
        result = {field.name: getattr(self, field.name) for field in fields(self)}
        result = {
            name: value.tolist() if isinstance(value, np.ndarray) else copy.deepcopy(value)
            for name, value in result.items()
        }
        return {"schema_version": SCHEMA, **result}

    @property
    def identity(self) -> str:
        """SHA-256 of all frozen schedules, scales, and provenance."""
        return _identity(self._body())

    def to_dict(self) -> dict:
        """Return a detached JSON object with a verifiable content seal."""
        body = self._body()
        return {**body, "seal": {"algorithm": "sha256", "content_sha256": _identity(body)}}

    @classmethod
    def from_dict(cls, value: dict) -> Reference:
        """Validate a sealed reference without reading any source artifacts."""
        if not isinstance(value, dict) or value.get("schema_version") != SCHEMA:
            raise ValueError("unsupported simple reference schema")
        body = {key: item for key, item in value.items() if key != "seal"}
        if value.get("seal") != {"algorithm": "sha256", "content_sha256": _identity(body)}:
            raise ValueError("simple reference content seal mismatch")
        if set(body) != {"schema_version", *(field.name for field in fields(cls))}:
            raise ValueError("unsupported simple reference fields")
        return cls(**{key: item for key, item in body.items() if key != "schema_version"})

    def save(self, path: str | Path) -> Path:
        """Write a sealed, self-contained JSON reference and return its path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> Reference:
        """Load and verify a reference using only its embedded data."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_object))

    def sample(self, time_s) -> dict[str, np.ndarray]:
        """Sample consistent cubic-Hermite positions and derivatives.

        Values clamp outside the supplied context. Clamped derivatives are zero.
        ID and evaluation forces use linear interpolation. No source files are read.
        """
        query = np.asarray(time_s, dtype=float)
        if not np.all(np.isfinite(query)):
            raise ValueError("reference sample time must be finite")
        time = np.clip(query, self.time_s[0], self.time_s[-1])
        index = np.clip(np.searchsorted(self.time_s, time, side="right") - 1, 0, len(self.time_s) - 2)
        h = self.time_s[index + 1] - self.time_s[index]
        u = (time - self.time_s[index]) / h
        clamped = (query < self.time_s[0]) | (query > self.time_s[-1])
        result = {}
        for position, velocity in _PAIRS.items():
            y, v = getattr(self, position), getattr(self, velocity)
            y0, y1, v0, v1 = y[index], y[index + 1], v[index], v[index + 1]
            result[position] = (
                (2 * u**3 - 3 * u**2 + 1) * y0
                + (u**3 - 2 * u**2 + u) * h * v0
                + (-2 * u**3 + 3 * u**2) * y1
                + (u**3 - u**2) * h * v1
            )
            derivative = (
                (6 * u**2 - 6 * u) * y0 / h
                + (3 * u**2 - 4 * u + 1) * v0
                + (-6 * u**2 + 6 * u) * y1 / h
                + (3 * u**2 - 2 * u) * v1
            )
            result[velocity] = np.where(clamped, 0.0, derivative)
        for name in _LINEAR:
            result[name] = np.asarray(np.interp(time, self.time_s, getattr(self, name)))
        return result


def _equilibrium_error(time, force, stiffness, damping):
    """Solve B edot + K e = force exactly for piecewise-linear forcing."""
    error = np.empty_like(force)
    # This is a declared boundary condition, not a fitted preload or F/K schedule.
    error[0] = force[0] / stiffness
    tau = damping / stiffness
    for i, dt in enumerate(np.diff(time)):
        slope = (force[i + 1] - force[i]) / dt
        decay = np.exp(-dt / tau)
        error[i + 1] = (
            error[i] * decay
            + force[i] / stiffness * (-np.expm1(-dt / tau))
            + slope / stiffness * (dt + tau * np.expm1(-dt / tau))
        )
    return error, (force - stiffness * error) / damping


def _statistics(vector):
    magnitude = np.linalg.norm(vector, axis=-1) if vector.ndim > 1 else np.abs(vector)
    result = {"rms": float(np.sqrt(np.mean(magnitude**2))), "max": float(magnitude.max())}
    if vector.ndim > 1:
        result["components"] = {
            axis: {
                "rms": float(np.sqrt(np.mean(vector[:, index] ** 2))),
                "max_abs": float(np.max(np.abs(vector[:, index]))),
                "mean": float(np.mean(vector[:, index])),
            }
            for index, axis in ((0, "x"), (2, "z"))
        }
    return result


def prepare_reference(profile_path, variability_path, artifact_path, *, config: dict | None = None) -> Reference:
    """Construct a reference by OFFLINE inverse dynamics of measured optical motion.

    Args:
        profile_path: Sealed v3 stance with measured pelvis and heel optical knots.
        variability_path: Sealed subject stride-variability artifact with pelvis and pitch SDs.
        artifact_path: Digital Shoe geometry used for the one-time mounting registration.
        config: Overrides of declared preparation constants, not optimized controls.

    The foot COM and axial leg attachment coincide with the ankle mount. Pelvis
    and heel translations receive the same measured-frame belt translation. The
    heel centroid maps to the last's rear X and the static heel centroid height;
    this is an explicit cross-shoe assumption, not anatomical registration.
    Only a constant foot Z offset registers virgin sole entry at touchdown.
    Absolute pelvis height is never shifted. Pitch, heel and pelvis use the same
    offline second-difference smoothing plus natural C2 cubic on optical context.
    """
    # Keep source adapters and Warp outside the portable reference load/sample path.
    from projects.digital_shoe import load_artifact  # noqa: PLC0415
    from projects.impedance_instron.orientation import orient_shoe  # noqa: PLC0415
    from projects.impedance_instron.profile import PELVIS_SCHEMA, load_profile  # noqa: PLC0415
    from projects.impedance_instron.trajectory import TrajectoryCubic  # noqa: PLC0415
    from projects.impedance_instron.variability import contact_bounds, load_variability  # noqa: PLC0415

    cfg = copy.deepcopy(_DEFAULT_CONFIG)
    if config is not None:
        if not isinstance(config, dict) or set(config) - set(cfg):
            raise ValueError("unknown reference preparation config keys")
        cfg.update(copy.deepcopy(config))
    for key in set(cfg) - {"ankle_mount_local_m", "source_side", "target_side"}:
        cfg[key] = float(cfg[key])
        if not np.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"{key} must be positive and finite")
    mount = np.asarray(cfg["ankle_mount_local_m"], dtype=float)
    if mount.shape != (3,) or not np.all(np.isfinite(mount)) or mount[1] != 0:
        raise ValueError("ankle_mount_local_m must be a finite planar vector")
    cfg["ankle_mount_local_m"] = mount.tolist()
    profile = load_profile(profile_path)
    if profile["schema_version"] != PELVIS_SCHEMA:
        raise ValueError("offline inverse dynamics requires measured v3 pelvis and heel kinematics")
    variability = load_variability(variability_path)
    if profile["side"] != variability.side or profile["side"] != cfg["target_side"]:
        raise ValueError("profile, variability and oriented shoe sides must agree")
    mass = float(profile["mass_kg"])
    upper_mass = mass - cfg["foot_mass_kg"]
    if upper_mass <= 0:
        raise ValueError("foot mass must be smaller than total measured subject mass")
    if not np.isclose(variability.body["subject"]["mass_kg"], mass, rtol=0, atol=1e-9):
        raise ValueError("profile and variability subject masses disagree")
    shoe, orientation = orient_shoe(load_artifact(artifact_path), cfg["target_side"], source_side=cfg["source_side"])
    time = np.asarray(profile["time_s"], dtype=float)
    force_x, force_z = (np.asarray(profile[key], dtype=float) for key in ("reference_fx_n", "reference_fz_n"))
    start, end = contact_bounds(force_z, cfg["contact_force_fraction"] * mass * cfg["gravity_m_s2"])
    contact_time = float(time[start])
    pitch_ref, pelvis_ref = profile["pitch_reference"], profile["pelvis_reference"]
    knots = np.asarray(pitch_ref["knot_time_s"], dtype=float)
    if not np.allclose(knots, pelvis_ref["knot_time_s"], rtol=0, atol=1e-10):
        raise ValueError("heel and pelvis optical clocks must agree")
    heel_raw = np.asarray(pitch_ref["heel_marker_xyz_m"], dtype=float).mean(axis=1)
    pelvis_raw = np.asarray(pelvis_ref["centroid_m"], dtype=float)
    processing = {}

    def fit(name, raw):
        curve = TrajectoryCubic.fit(knots, raw, cutoff_hz=cfg["smoothing_cutoff_hz"])
        metadata = dict(curve.smoothing)
        metadata.pop("max_knot_change_deg")
        metadata["max_knot_change"] = float(np.max(np.abs(curve.angle_rad - raw)))
        metadata["unit"] = "rad" if name == "pitch" else "m"
        processing[name] = metadata
        return curve.evaluate(time)

    pitch, pitch_rate, pitch_acceleration = fit("pitch", np.asarray(pitch_ref["pitch_rad"], dtype=float))
    heel = np.zeros((3, len(time), 3))
    upper = np.zeros_like(heel)
    for axis, label in ((0, "x"), (2, "z")):
        heel[:, :, axis] = np.asarray(fit("heel_" + label, heel_raw[:, axis]))
        upper[:, :, axis] = np.asarray(fit("pelvis_" + label, pelvis_raw[:, axis]))
    registration = profile["provenance"]["registration"]
    origin = np.asarray(registration["heel_origin_newton_lab_m"], dtype=float)
    belt = np.asarray(registration["virtual_origin_x_m"], dtype=float)
    belt_speed = float(profile["provenance"]["running"]["belt_speed_m_s"])
    if not np.allclose(belt, time * belt_speed, rtol=0, atol=1e-8):
        raise ValueError("only a constant-speed virtual belt transform is supported")
    for series in (heel, upper):
        series[0, :, 0] += belt - origin[0]
        series[1, :, 0] += belt_speed
    last = shoe.visual_meshes["fullfoot_last"].vertices_m
    static_height = float(np.asarray(pitch_ref["static_template"]["position_m"])[:, 2].mean())
    heel_local = np.array([last[:, 0].min(), 0.0, static_height])
    offset = mount - heel_local
    c, s = np.cos(pitch), np.sin(pitch)
    rotated = np.column_stack((c * offset[0] + s * offset[2], np.zeros_like(time), -s * offset[0] + c * offset[2]))
    tangent = np.column_stack((rotated[:, 2], np.zeros_like(time), -rotated[:, 0]))
    foot = heel.copy()
    foot[0] += rotated
    foot[1] += pitch_rate[:, None] * tangent
    foot[2] += pitch_acceleration[:, None] * tangent - pitch_rate[:, None] ** 2 * rotated
    bottom = shoe.column_bed.anchor_bottom_m - mount
    bottom_height = -np.sin(pitch[start]) * bottom[:, 0] + np.cos(pitch[start]) * bottom[:, 2]
    foot_z_offset = -float(foot[0, start, 2] + bottom_height.min())
    foot[0, :, 2] += foot_z_offset

    delta = upper[0] - foot[0]
    length = np.linalg.norm(delta, axis=1)
    if np.any(length < 0.1):
        raise ValueError("mapped pelvis and ankle are too close to define the leg")
    direction = delta / length[:, None]
    length_rate = np.sum(direction * (upper[1] - foot[1]), axis=1)
    gravity = np.array([0.0, 0.0, -cfg["gravity_m_s2"]])
    upper_required = upper_mass * (upper[2] - gravity)
    inverse_force = np.sum(upper_required * direction, axis=1)
    axial_vector = inverse_force[:, None] * direction
    upper_residual = upper_required - axial_vector
    measured_force = np.column_stack((force_x, np.zeros_like(time), force_z))
    foot_residual = cfg["foot_mass_kg"] * (foot[2] - gravity) - (measured_force - axial_vector)
    channels = profile["provenance"]["kinetics"]["platform_channels"]
    moment_lab = np.asarray(channels["moment_about_lab_origin_nm"], dtype=float).sum(axis=1)
    # Transform the ankle back to the fixed lab origin of the measured moment.
    ankle_lab_x = foot[0, :, 0] - belt + origin[0]
    contact_moment = moment_lab[:, 1] - (foot[0, :, 2] * force_x - ankle_lab_x * force_z)
    inverse_torque = cfg["pitch_inertia_kg_m2"] * pitch_acceleration - contact_moment
    k_leg, k_ankle = cfg["nominal_leg_stiffness_n_m"], cfg["nominal_ankle_stiffness_n_m_rad"]
    b_leg = 2 * cfg["leg_damping_ratio"] * np.sqrt(k_leg * upper_mass)
    b_ankle = 2 * cfg["ankle_damping_ratio"] * np.sqrt(k_ankle * cfg["pitch_inertia_kg_m2"])
    error, error_rate = _equilibrium_error(time, inverse_force, k_leg, b_leg)
    angle_error, angle_error_rate = _equilibrium_error(time, inverse_torque, k_ankle, b_ankle)
    source_paths = {"profile": profile_path, "variability": variability_path, "shoe": artifact_path}
    sources = {
        key: {"path": str(path), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
        for key, path in source_paths.items()
    }
    contact_slice = slice(start, end + 1)
    residual_summary = {
        "upper_orthogonal_force_n": _statistics(upper_residual[contact_slice]),
        "foot_balance_force_n": _statistics(foot_residual[contact_slice]),
    }
    provenance = {
        "construction": "offline reduced two-mass rig inverse dynamics; no equilibrium optimization",
        "config": cfg,
        "sources": sources,
        "source_acquisition": copy.deepcopy(profile["provenance"]["sources"]),
        "source_stance_s": profile["provenance"]["running"]["selected_stance_source_s"],
        "source_profile_start_s": profile["source_time_s"][0],
        "source_reproduction_options": profile["provenance"]["reproduction_options"],
        "geometry_identity": geometry_identity(shoe),
        "orientation": orientation,
        "variability": {
            "stance_count": variability.stance_count,
            "windows_s": variability.windows_s,
            "scale_channels": ["pelvis_height_m", "foot_pitch_rad"],
            "scale_processing": "original unsmoothed 83-stance optical variability, not refitted to smoothing",
        },
        "processing": processing,
        "preparation_code_sha256": {
            str(path.relative_to(Path(__file__).resolve().parents[3])): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__).resolve(),
                *(
                    Path(__file__).resolve().parent.parent / name
                    for name in ("trajectory.py", "orientation.py", "profile.py", "variability.py")
                ),
            )
        },
        "raw_optical": {
            "knot_time_s": knots.tolist(),
            "pelvis_centroid_lab_m": pelvis_raw.tolist(),
            "heel_centroid_lab_m": heel_raw.tolist(),
            "pitch_rad": pitch_ref["pitch_rad"],
        },
        "mapping": {
            "frame": "right-handed +X forward, +Y left, +Z up; Y motion projected away",
            "lab_to_newton": registration["lab_to_newton"],
            "belt_speed_m_s": belt_speed,
            "heel_origin_newton_lab_m": origin.tolist(),
            "heel_proxy_local_m": heel_local.tolist(),
            "foot_z_registration_m": foot_z_offset,
            "sole_registration_time_s": contact_time,
            "initial_sole_clearance_m": float(
                np.min(foot[0, 0, 2] - np.sin(pitch[0]) * bottom[:, 0] + np.cos(pitch[0]) * bottom[:, 2])
            ),
            "assumption": "heel centroid X at last rear, static heel centroid Z above sole; fixed sagittal offset to ankle",
            "upper_height_shift_m": 0.0,
            "anatomical_registration_validated": False,
            "foot_com_equals_ankle_mount": True,
            "capture_shoe_matches_modeled_shoe": False,
        },
        "inverse_dynamics": {
            "upper_mass_kg": upper_mass,
            "leg_damping_n_s_m": float(b_leg),
            "ankle_damping_n_m_s_rad": float(b_ankle),
            "boundary_condition": "e(0)=F_ID(0)/K; edot(0)=0; angular analogue; not optimized preload",
            "ode": "B*edot+K*e=F_ID; exact update for piecewise-linear ID forcing on native force clock",
            "contact_moment": "sum measured platform moments about lab origin minus ankle cross measured force; +Y toe-down",
            "contact_moment_n_m": contact_moment.tolist(),
            "leg_measured_length_m": length.tolist(),
            "leg_measured_rate_m_s": length_rate.tolist(),
            "upper_position_m": upper[0].tolist(),
            "upper_velocity_m_s": upper[1].tolist(),
            "upper_acceleration_m_s2": upper[2].tolist(),
            "foot_position_m": foot[0].tolist(),
            "foot_velocity_m_s": foot[1].tolist(),
            "foot_acceleration_m_s2": foot[2].tolist(),
            "pitch_acceleration_rad_s2": pitch_acceleration.tolist(),
            "upper_orthogonal_force_n": upper_residual.tolist(),
            "foot_balance_force_n": foot_residual.tolist(),
            "residual_summary_contact": residual_summary,
            "residual_contact_window_s": [float(time[start]), float(time[end])],
            "residual_scope": "full recorded sagittal motion consistency, not a lower bound on pelvis-z/pitch loss; "
            "unrewarded pelvis X and foot translations may differ in rollout",
            "residual_definition": "required minus supplied planar force; upper orthogonal residual is unactuated",
            "exact_compatibility_claimed": False,
        },
        "limitations": [
            "Pelvis centroid is not whole-body COM; its assigned upper mass is a reduced-model assumption.",
            "Heel pitch uses an assumed flat static reference, not a measured Puma sole frame.",
            "The world-reacted pitch actuator cannot cancel upper transverse force or foot force-balance residuals.",
            "No force-integrated position/velocity, old objective, optimized command, or runtime wrench feedforward is used.",
            "Platform baseline noise remains in offline moments and evaluation forces, including flight.",
        ],
        "rights": copy.deepcopy(profile["provenance"]["rights"]),
    }
    return Reference(
        time_s=time,
        pelvis_z_m=upper[0, :, 2],
        pelvis_vz_m_s=upper[1, :, 2],
        pitch_rad=pitch,
        pitch_rate_rad_s=pitch_rate,
        leg_length_m=length + error,
        leg_rate_m_s=length_rate + error_rate,
        ankle_equilibrium_rad=pitch + angle_error,
        ankle_equilibrium_rate_rad_s=pitch_rate + angle_error_rate,
        inverse_leg_force_n=inverse_force,
        inverse_ankle_torque_n_m=inverse_torque,
        reference_fz_n=force_z,
        reference_fx_n=force_x,
        foot_position_m=foot[0, 0],
        upper_position_m=upper[0, 0],
        foot_velocity_m_s=foot[1, 0],
        upper_velocity_m_s=upper[1, 0],
        mass_kg=mass,
        gravity_m_s2=cfg["gravity_m_s2"],
        contact_start_s=contact_time,
        contact_duration_s=float(time[end] - time[start]),
        pelvis_scale_m=variability.normaliser("pelvis_height_m"),
        pitch_scale_rad=variability.normaliser("foot_pitch_rad"),
        provenance=provenance,
    )
