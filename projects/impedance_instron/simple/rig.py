# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Newton-native planar rig with two bounded variable-stiffness actuators.

Only leg stiffness and world-reacted ankle stiffness are actions. The bilateral
mechanical leg spring can push AND pull; there is no unilateral release gate.
Offline inverse rig dynamics fixes both equilibria. The pelvis target is the UPPER body's height,
not the two-body center of mass. The foundation material and passive surround are
used without a stiffness multiplier or a geometry change.

Substep traces describe pre-integration force evaluations. Reward is minus the
full fixed-time integral of normalized optical motion error. Work and GRF never
enter that reward. Terminal state is available as ``state_0``; no extra terminal
foundation evaluation advances its history.
"""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
import warp as wp

import newton
from projects.digital_shoe import FoundationConfig, MidsoleFoundation, load_artifact
from projects.digital_shoe.provenance import physics_source_identity
from projects.digital_shoe.runtime import SurroundConfig
from projects.impedance_instron.orientation import orient_shoe

from .reference import Reference, geometry_identity


@dataclass(frozen=True)
class RigConfig:
    """Frozen rig, contact, timing and safety settings in SI units.

    The quintic log-stiffness ramp has zero velocity and acceleration at frame
    boundaries. Its endpoint displacement is limited by ``log_stiffness_slew_s``
    divided by its peak normalized slope (1.875), so the instantaneous log-rate,
    not just its frame average, obeys the bound. Zero action targets the geometric
    midpoint of each absolute stiffness range; it is not an additive update.

    All equilibria and reward targets use the same fixed physical-time clock.
    Nominal phase is ``(t - reference.contact_start_s) / contact_duration_s``.
    Actual touchdown is a diagnostic only. No schedule resets, contact gates,
    toe-off alignment, phase correction or final-time warps are used.
    """

    frame_rate_hz: float = 120.0
    substeps: int = 64
    use_graph: bool = True
    foot_mass_kg: float = 2.0
    pitch_inertia_kg_m2: float = 0.025
    ankle_mount_m: tuple[float, float, float] = (-0.075, 0.0, 0.105)
    source_shoe_side: str = "right"
    shoe_side: str = "left"
    leg_stiffness_min_n_m: float = 3000.0
    leg_stiffness_max_n_m: float = 48000.0
    ankle_stiffness_min_n_m_rad: float = 1000.0
    ankle_stiffness_max_n_m_rad: float = 16000.0
    log_stiffness_slew_s: float = 12.0
    leg_damping_ratio: float = 0.25
    ankle_damping_ratio: float = 0.5
    force_limit_bw: float = 5.0
    ankle_torque_limit_n_m: float = 400.0
    contact_threshold_n: float = 20.0
    friction_mu: float = 0.8
    contact_kt_n_m: float = 10000.0
    contact_kd_n_s_m: float = 10.0
    friction_viscous_ratio: float = 0.2
    friction_release_dwell_s: float = 0.0005
    stretch_floor: float = 0.05
    outer_relaxation_s: float = 0.002
    outer_coupling_scale: float = 1.0
    outer_max_strain: float = 0.9
    outer_sweeps: int = 4
    minimum_last_clearance_m: float = -0.002
    minimum_pelvis_height_m: float = 0.3
    maximum_pelvis_height_m: float = 2.0
    maximum_pitch_rad: float = 1.6
    minimum_leg_length_m: float = 0.25
    maximum_leg_length_m: float = 1.5
    maximum_speed_m_s: float = 20.0

    def __post_init__(self):
        values = self.to_dict()
        if any(isinstance(v, (int, float)) and not math.isfinite(v) for v in values.values()):
            raise ValueError("Rig settings must be finite")
        positive = (
            self.frame_rate_hz,
            self.foot_mass_kg,
            self.pitch_inertia_kg_m2,
            self.leg_stiffness_min_n_m,
            self.ankle_stiffness_min_n_m_rad,
            self.log_stiffness_slew_s,
            self.force_limit_bw,
            self.ankle_torque_limit_n_m,
            self.contact_threshold_n,
            self.outer_relaxation_s,
            self.maximum_pitch_rad,
            self.minimum_leg_length_m,
            self.maximum_speed_m_s,
        )
        if min(positive) <= 0 or self.substeps < 1 or int(self.substeps) != self.substeps:
            raise ValueError("Positive physical settings and integer substeps are required")
        if self.outer_sweeps < 1 or int(self.outer_sweeps) != self.outer_sweeps:
            raise ValueError("outer_sweeps must be a positive integer")
        if (
            min(
                self.leg_damping_ratio,
                self.ankle_damping_ratio,
                self.friction_mu,
                self.contact_kt_n_m,
                self.contact_kd_n_s_m,
                self.outer_coupling_scale,
                self.friction_viscous_ratio,
                self.friction_release_dwell_s,
            )
            < 0
        ):
            raise ValueError("Damping, friction and surround coupling must be nonnegative")
        if not (
            self.leg_stiffness_max_n_m > self.leg_stiffness_min_n_m
            and self.ankle_stiffness_max_n_m_rad > self.ankle_stiffness_min_n_m_rad
            and self.maximum_pelvis_height_m > self.minimum_pelvis_height_m
            and self.maximum_leg_length_m > self.minimum_leg_length_m
        ):
            raise ValueError("Upper bounds must exceed lower bounds")
        if not 0 < self.stretch_floor < 1 or not 0 < self.outer_max_strain < 1:
            raise ValueError("Foam strain bounds must be inside (0, 1)")
        mount = np.asarray(self.ankle_mount_m, dtype=float)
        if mount.shape != (3,) or not np.isfinite(mount).all():
            raise ValueError("ankle_mount_m needs three finite coordinates")
        object.__setattr__(self, "ankle_mount_m", tuple(float(x) for x in mount))
        if self.source_shoe_side not in ("left", "right") or self.shoe_side not in ("left", "right"):
            raise ValueError("Shoe side must be explicit")

    def to_dict(self) -> dict:
        """Return every frozen setting as JSON-compatible data."""
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> RigConfig:
        """Restore settings, rejecting unknown legacy modes and action fields."""
        return cls(**value)


def _identity(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _sagittal_hull(vertices: np.ndarray) -> np.ndarray:
    """Keep exact support extrema while avoiding a full mesh scan per substep."""
    points = sorted(set(map(tuple, np.asarray(vertices)[:, (0, 2)])))

    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    halves = []
    for sequence in (points, points[::-1]):
        half = []
        for point in sequence:
            while len(half) >= 2 and cross(half[-2], half[-1], point) <= 0:
                half.pop()
            half.append(point)
        halves.extend(half[:-1])
    return np.asarray(halves or points, np.float32)


TRACE_NAMES = (
    "time_s",
    "ref_time_s",
    "ref_clock_rate",
    "pelvis_z_m",
    "reference_pelvis_z_m",
    "pitch_rad",
    "reference_pitch_rad",
    "leg_stiffness_n_m",
    "ankle_stiffness_n_m_rad",
    "leg_damping_n_s_m",
    "ankle_damping_n_m_s_rad",
    "leg_equilibrium_m",
    "ankle_equilibrium_rad",
    "shoe_fz_n",
    "shoe_fx_n",
    "reference_fz_n",
    "reference_fx_n",
    "leg_source_power_w",
    "ankle_source_power_w",
    "leg_damping_power_w",
    "ankle_damping_power_w",
    "compression_m",
    "last_clearance_m",
    "tracking_error",
    "foot_x_m",
    "foot_z_m",
    "pelvis_x_m",
    "foot_vx_m_s",
    "foot_vz_m_s",
    "pelvis_vx_m_s",
    "pelvis_vz_m_s",
    "pitch_rate_rad_s",
    "leg_length_m",
    "leg_rate_m_s",
    "leg_equilibrium_rate_m_s",
    "ankle_equilibrium_rate_rad_s",
    "leg_stiffness_rate_n_m_s",
    "ankle_stiffness_rate_n_m_rad_s",
    "leg_force_n",
    "ankle_torque_n_m",
    "leg_raw_force_n",
    "ankle_raw_torque_n_m",
    "leg_spring_energy_j",
    "ankle_spring_energy_j",
    "leg_body_power_w",
    "ankle_body_power_w",
    "leg_spring_energy_rate_w",
    "ankle_spring_energy_rate_w",
    "leg_limit_power_w",
    "ankle_limit_power_w",
    "leg_force_limited",
    "ankle_torque_limited",
    "shoe_contact_power_w",
    "shoe_moment_y_n_m",
    "detected_touchdown_s",
    "safety_flags",
)

SAFETY_REASONS = {
    1: "nonfinite_state_or_force",
    2: "rigid_last_ground_intersection",
    4: "pelvis_height_out_of_bounds",
    8: "pitch_out_of_bounds",
    16: "leg_length_out_of_bounds",
    32: "speed_out_of_bounds",
    64: "leg_force_limit",
    128: "ankle_torque_limit",
    256: "no_detected_contact",
}


@wp.struct
class _Parameters:
    dt: float
    frame_dt: float
    substeps: int
    mass_upper: float
    mass_total: float
    gravity: float
    pitch_inertia: float
    zeta_leg: float
    zeta_ankle: float
    force_limit: float
    torque_limit: float
    contact_threshold: float
    contact_start: float
    contact_duration: float
    z_scale: float
    pitch_scale: float
    clearance_min: float
    pelvis_min: float
    pelvis_max: float
    pitch_max: float
    leg_min: float
    leg_max: float
    speed_max: float
    response_mode: int
    damping_leg: float
    damping_ankle: float
    ground_height: float


@wp.func
def _smooth(u: float) -> float:
    return u * u * u * (10.0 + u * (-15.0 + 6.0 * u))


@wp.func
def _smooth_rate(u: float) -> float:
    return 30.0 * u * u * (1.0 - u) * (1.0 - u)


@wp.func
def _interval(t: float, knots: wp.array[float]) -> int:
    lo, hi = int(0), knots.shape[0] - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if knots[mid] <= t:
            lo = mid
        else:
            hi = mid
    return lo


@wp.func
def _curve(t: float, knots: wp.array[float], data: wp.array2d[float], col: int) -> wp.vec2:
    if t < knots[0]:
        return wp.vec2(data[0, col], 0.0)
    end = knots.shape[0] - 1
    if t > knots[end]:
        return wp.vec2(data[end, col], 0.0)
    i = _interval(t, knots)
    h = knots[i + 1] - knots[i]
    u = (t - knots[i]) / h
    a, b = data[i, col], data[i + 1, col]
    da, db = data[i, col + 1], data[i + 1, col + 1]
    value = (2.0 * u * u * u - 3.0 * u * u + 1.0) * a + (u * u * u - 2.0 * u * u + u) * h * da
    value += (-2.0 * u * u * u + 3.0 * u * u) * b + (u * u * u - u * u) * h * db
    rate = ((6.0 * u * u - 6.0 * u) * a + (-6.0 * u * u + 6.0 * u) * b) / h
    rate += (3.0 * u * u - 4.0 * u + 1.0) * da + (3.0 * u * u - 2.0 * u) * db
    return wp.vec2(value, rate)


@wp.func
def _linear(t: float, knots: wp.array[float], data: wp.array2d[float], col: int) -> float:
    tc = wp.clamp(t, knots[0], knots[knots.shape[0] - 1])
    i = _interval(tc, knots)
    u = (tc - knots[i]) / (knots[i + 1] - knots[i])
    return (1.0 - u) * data[i, col] + u * data[i + 1, col]


@wp.func
def _pitch(q: wp.transform) -> float:
    rotation = wp.transform_get_rotation(q)
    return 2.0 * wp.atan2(rotation[1], rotation[3])


@wp.func
def _wrap(angle: float) -> float:
    return wp.atan2(wp.sin(angle), wp.cos(angle))


@wp.kernel
def _advance(index: wp.array[int]):
    index[0] += 1


@wp.kernel
def _planar(q: wp.array[wp.transform], qd: wp.array[wp.spatial_vector]):
    w = wp.tid()
    f, p = 2 * w, 2 * w + 1
    x, z = wp.transform_get_translation(q[f]), wp.transform_get_translation(q[p])
    vf, vp = wp.spatial_top(qd[f]), wp.spatial_top(qd[p])
    q[f] = wp.transform(wp.vec3(x[0], 0.0, x[2]), wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), _pitch(q[f])))
    q[p] = wp.transform(wp.vec3(z[0], 0.0, z[2]), wp.quat_identity())
    qd[f] = wp.spatial_vector(wp.vec3(vf[0], 0.0, vf[2]), wp.vec3(0.0, wp.spatial_bottom(qd[f])[1], 0.0))
    qd[p] = wp.spatial_vector(wp.vec3(vp[0], 0.0, vp[2]), wp.vec3(0.0))


@wp.kernel
def _actuate_record(
    prm: _Parameters,
    index: wp.array[int],
    knots: wp.array[float],
    ref: wp.array2d[float],
    log_start: wp.array2d[float],
    log_end: wp.array2d[float],
    push: wp.array2d[float],
    touchdown: wp.array[float],
    q: wp.array[wp.transform],
    qd: wp.array[wp.spatial_vector],
    body_f: wp.array[wp.spatial_vector],
    shoe_force: wp.array[wp.vec3],
    shoe_moment: wp.array[wp.vec3],
    shoe_power: wp.array[float],
    compression: wp.array[float],
    last_hull: wp.array[wp.vec2],
    flags: wp.array[int],
    frame_loss: wp.array[float],
    trace: wp.array3d[float],
):
    w = wp.tid()
    i = index[0]
    t = float(i) * prm.dt
    f, p = 2 * w, 2 * w + 1
    x, pelvis = wp.transform_get_translation(q[f]), wp.transform_get_translation(q[p])
    vf, vp = wp.spatial_top(qd[f]), wp.spatial_top(qd[p])
    angle, omega = _pitch(q[f]), wp.spatial_bottom(qd[f])[1]
    length = wp.max(wp.length(pelvis - x), 1.0e-8)
    axis = (pelvis - x) / length
    rate = wp.dot(axis, vp - vf)
    if touchdown[w] < 0.0 and shoe_force[w][2] >= prm.contact_threshold:
        touchdown[w] = t
    clock = wp.vec2(t, 1.0)
    zr = _curve(clock[0], knots, ref, 0)
    pitchr = _curve(clock[0], knots, ref, 2)
    lr = _curve(clock[0], knots, ref, 4)
    ar = _curve(clock[0], knots, ref, 6)
    l0, l0dot = lr[0], lr[1] * clock[1]
    a0, a0dot = ar[0], ar[1] * clock[1]
    movement = wp.vec2(0.0)
    inverse_l, inverse_a = float(0.0), float(0.0)
    nominal_l, nominal_a = float(0.0), float(0.0)
    push_x, push_z = float(0.0), float(0.0)
    if prm.response_mode != 0:
        movement = _curve(t, knots, ref, 10)
        inverse_l = _linear(t, knots, ref, 12)
        inverse_a = _linear(t, knots, ref, 13)
        push_x, push_z = float(push[i, 0]), float(push[i, 1])
        if prm.response_mode == 2:
            l0, l0dot = movement[0], movement[1]
            a0, a0dot = pitchr[0], pitchr[1]
            nominal_l, nominal_a = inverse_l, inverse_a
    u = float(i % prm.substeps) / float(prm.substeps)
    s, ds = _smooth(u), _smooth_rate(u) / prm.frame_dt
    kl = wp.exp(log_start[w, 0] + s * (log_end[w, 0] - log_start[w, 0]))
    ka = wp.exp(log_start[w, 1] + s * (log_end[w, 1] - log_start[w, 1]))
    kldot = kl * ds * (log_end[w, 0] - log_start[w, 0])
    kadot = ka * ds * (log_end[w, 1] - log_start[w, 1])
    bl = 2.0 * prm.zeta_leg * wp.sqrt(kl * prm.mass_upper)
    ba = 2.0 * prm.zeta_ankle * wp.sqrt(ka * prm.pitch_inertia)
    if prm.response_mode != 0:
        bl, ba = float(prm.damping_leg), float(prm.damping_ankle)
    el, ea = length - l0, _wrap(angle - a0)
    sl, sa = rate - l0dot, omega - a0dot
    feedback_l, feedback_a = -kl * el - bl * sl, -ka * ea - ba * sa
    raw_l, raw_a = feedback_l, feedback_a
    if prm.response_mode == 2:
        raw_l += nominal_l
        raw_a += nominal_a
    force, torque = (
        wp.clamp(raw_l, -prm.force_limit, prm.force_limit),
        wp.clamp(raw_a, -prm.torque_limit, prm.torque_limit),
    )
    body_f[f] = body_f[f] + wp.spatial_vector(-force * axis, wp.vec3(0.0, torque, 0.0))
    body_f[p] = body_f[p] + wp.spatial_vector(force * axis, wp.vec3(0.0))
    if prm.response_mode != 0:
        body_f[p] = body_f[p] + wp.spatial_vector(wp.vec3(push_x, 0.0, push_z), wp.vec3(0.0))
    limit_l, limit_a = (force - raw_l) * rate, (torque - raw_a) * omega
    source_l = raw_l * l0dot + 0.5 * kldot * el * el + limit_l
    source_a = raw_a * a0dot + 0.5 * kadot * ea * ea + limit_a
    if prm.response_mode == 2:
        # The nominal assistance does body work, not tracking-spring storage.
        source_l = nominal_l * rate + feedback_l * l0dot + 0.5 * kldot * el * el + limit_l
        source_a = nominal_a * omega + feedback_a * a0dot + 0.5 * kadot * ea * ea + limit_a
    damp_l, damp_a = -bl * sl * sl, -ba * sa * sa
    spring_rate_l = 0.5 * kldot * el * el + kl * el * sl
    spring_rate_a = 0.5 * kadot * ea * ea + ka * ea * sa
    clearance = float(1.0e6)
    for vertex in range(last_hull.shape[0]):
        local = last_hull[vertex]
        clearance = wp.min(clearance, x[2] - wp.sin(angle) * local[0] + wp.cos(angle) * local[1])
    clearance -= prm.ground_height
    dz, da = (pelvis[2] - zr[0]) / prm.z_scale, _wrap(angle - pitchr[0]) / prm.pitch_scale
    error = 0.5 * (dz * dz + da * da)
    status = flags[w]
    if (
        not wp.isfinite(error)
        or not wp.isfinite(length)
        or not wp.isfinite(raw_l)
        or not wp.isfinite(raw_a)
        or not wp.isfinite(shoe_force[w][0])
        or not wp.isfinite(shoe_force[w][2])
        or not wp.isfinite(wp.length(vf))
        or not wp.isfinite(wp.length(vp))
    ):
        status = status | 1
    if clearance < prm.clearance_min:
        status = status | 2
    if pelvis[2] - prm.ground_height < prm.pelvis_min or pelvis[2] - prm.ground_height > prm.pelvis_max:
        status = status | 4
    if wp.abs(angle) > prm.pitch_max:
        status = status | 8
    if length < prm.leg_min or length > prm.leg_max:
        status = status | 16
    if wp.length(vf) > prm.speed_max or wp.length(vp) > prm.speed_max:
        status = status | 32
    if wp.abs(force - raw_l) > 1.0e-3:
        status = status | 64
    if wp.abs(torque - raw_a) > 1.0e-3:
        status = status | 128
    flags[w] = status
    frame_loss[w] += prm.dt * error
    trace[i, w, 0] = t
    trace[i, w, 1] = clock[0]
    trace[i, w, 2] = clock[1]
    trace[i, w, 3] = pelvis[2]
    trace[i, w, 4] = zr[0]
    trace[i, w, 5] = angle
    trace[i, w, 6] = pitchr[0]
    trace[i, w, 7] = kl
    trace[i, w, 8] = ka
    trace[i, w, 9] = bl
    trace[i, w, 10] = ba
    trace[i, w, 11] = l0
    trace[i, w, 12] = a0
    trace[i, w, 13] = shoe_force[w][2]
    trace[i, w, 14] = shoe_force[w][0]
    trace[i, w, 15] = _linear(clock[0], knots, ref, 8)
    trace[i, w, 16] = _linear(clock[0], knots, ref, 9)
    trace[i, w, 17] = source_l
    trace[i, w, 18] = source_a
    trace[i, w, 19] = damp_l
    trace[i, w, 20] = damp_a
    trace[i, w, 21] = compression[w]
    trace[i, w, 22] = clearance
    trace[i, w, 23] = error
    trace[i, w, 24] = x[0]
    trace[i, w, 25] = x[2]
    trace[i, w, 26] = pelvis[0]
    trace[i, w, 27] = vf[0]
    trace[i, w, 28] = vf[2]
    trace[i, w, 29] = vp[0]
    trace[i, w, 30] = vp[2]
    trace[i, w, 31] = omega
    trace[i, w, 32] = length
    trace[i, w, 33] = rate
    trace[i, w, 34] = l0dot
    trace[i, w, 35] = a0dot
    trace[i, w, 36] = kldot
    trace[i, w, 37] = kadot
    trace[i, w, 38] = force
    trace[i, w, 39] = torque
    trace[i, w, 40] = raw_l
    trace[i, w, 41] = raw_a
    trace[i, w, 42] = 0.5 * kl * el * el
    trace[i, w, 43] = 0.5 * ka * ea * ea
    trace[i, w, 44] = force * rate
    trace[i, w, 45] = torque * omega
    trace[i, w, 46] = spring_rate_l
    trace[i, w, 47] = spring_rate_a
    trace[i, w, 48] = limit_l
    trace[i, w, 49] = limit_a
    trace[i, w, 50] = float(wp.abs(force - raw_l) > 1.0e-3)
    trace[i, w, 51] = float(wp.abs(torque - raw_a) > 1.0e-3)
    trace[i, w, 52] = shoe_power[w]
    trace[i, w, 53] = shoe_moment[w][1] - wp.cross(x, shoe_force[w])[1]
    trace[i, w, 54] = touchdown[w]
    trace[i, w, 55] = float(status)
    if prm.response_mode != 0:
        trace[i, w, 56] = movement[0]
        trace[i, w, 57] = movement[1]
        trace[i, w, 58] = inverse_l
        trace[i, w, 59] = inverse_a
        trace[i, w, 60] = nominal_l
        trace[i, w, 61] = nominal_a
        trace[i, w, 62] = feedback_l
        trace[i, w, 63] = feedback_a
        trace[i, w, 64] = push_x
        trace[i, w, 65] = push_z
        trace[i, w, 66] = push_x * vp[0] + push_z * vp[2]
        trace[i, w, 67] = prm.ground_height
        trace[i, w, 68] = nominal_l * rate
        trace[i, w, 69] = nominal_a * omega
        trace[i, w, 70] = feedback_l * l0dot
        trace[i, w, 71] = feedback_a * a0dot
        trace[i, w, 72] = 0.5 * kldot * el * el
        trace[i, w, 73] = 0.5 * kadot * ea * ea


@wp.kernel
def _observe(
    prm: _Parameters,
    index: wp.array[int],
    knots: wp.array[float],
    ref: wp.array2d[float],
    q: wp.array[wp.transform],
    qd: wp.array[wp.spatial_vector],
    force: wp.array[wp.vec3],
    log_k: wp.array2d[float],
    touchdown: wp.array[float],
    scales: wp.array[float],
    hull: wp.array[wp.vec2],
    flags: wp.array[int],
    obs: wp.array2d[float],
):
    w = wp.tid()
    t = float(index[0]) * prm.dt
    x, p = wp.transform_get_translation(q[2 * w]), wp.transform_get_translation(q[2 * w + 1])
    vf, vp = wp.spatial_top(qd[2 * w]), wp.spatial_top(qd[2 * w + 1])
    angle, omega = _pitch(q[2 * w]), wp.spatial_bottom(qd[2 * w])[1]
    zr, ar = _curve(t, knots, ref, 0), _curve(t, knots, ref, 2)
    lr, eq = _curve(t, knots, ref, 4), _curve(t, knots, ref, 6)
    if prm.response_mode == 2:
        lr, eq = _curve(t, knots, ref, 10), ar
    length = wp.length(p - x)
    obs[w, 0] = p[2] - prm.ground_height
    obs[w, 1] = x[2] - prm.ground_height
    obs[w, 2] = p[0] - x[0]
    obs[w, 3] = length
    obs[w, 4] = vp[0]
    obs[w, 5] = vp[2]
    obs[w, 6] = vf[0]
    obs[w, 7] = vf[2]
    obs[w, 8] = angle
    obs[w, 9] = omega
    obs[w, 10] = force[w][0]
    obs[w, 11] = force[w][2]
    obs[w, 12] = lr[0]
    obs[w, 13] = eq[0]
    obs[w, 14] = zr[0] - prm.ground_height
    obs[w, 15] = ar[0]
    obs[w, 16] = log_k[w, 0]
    obs[w, 17] = log_k[w, 1]
    obs[w, 18] = (t - prm.contact_start) / prm.contact_duration
    obs[w, 19] = float(touchdown[w] >= 0.0)
    obs[w, 20] = (p[2] - zr[0]) / prm.z_scale
    obs[w, 21] = _wrap(angle - ar[0]) / prm.pitch_scale
    status = flags[w]
    for col in range(obs.shape[1]):
        value = obs[w, col] / scales[col]
        if not wp.isfinite(value):
            status = status | 1
            value = 0.0
        obs[w, col] = value
    clearance = float(1.0e6)
    for vertex in range(hull.shape[0]):
        v = hull[vertex]
        clearance = wp.min(clearance, x[2] - wp.sin(angle) * v[0] + wp.cos(angle) * v[1])
    clearance -= prm.ground_height
    if clearance < prm.clearance_min:
        status = status | 2
    if p[2] - prm.ground_height < prm.pelvis_min or p[2] - prm.ground_height > prm.pelvis_max:
        status = status | 4
    if wp.abs(angle) > prm.pitch_max:
        status = status | 8
    if length < prm.leg_min or length > prm.leg_max:
        status = status | 16
    if wp.length(vf) > prm.speed_max or wp.length(vp) > prm.speed_max:
        status = status | 32
    flags[w] = status


class Rig:
    """Simulate independent copies of the same two-stiffness mechanical rig.

    Args:
        reference: Frozen optical targets, inverse-dynamics equilibria and initial state.
        artifact_path: Identified shoe artifact. Geometry must match the frozen reference.
            A same-geometry material override changes only the foundation material. It
            never recomputes equilibria, registration, masses or initial conditions.
        config: Frozen settings; construction fields must agree with the reference.
            If omitted, construction fields are restored from reference provenance.
        num_worlds: Independent copies of the identical geometry and reference.
        device: Warp device such as ``"cpu"`` or ``"cuda:0"``.
    """

    _trace_names = TRACE_NAMES
    _ground_height_m = 0.0
    action_dim = 2
    observation_names = (
        "pelvis_z_m",
        "foot_z_m",
        "upper_minus_foot_x_m",
        "leg_length_m",
        "pelvis_vx_m_s",
        "pelvis_vz_m_s",
        "foot_vx_m_s",
        "foot_vz_m_s",
        "pitch_rad",
        "pitch_rate_rad_s",
        "shoe_fx_n",
        "shoe_fz_n",
        "leg_equilibrium_m",
        "ankle_equilibrium_rad",
        "reference_pelvis_z_m",
        "reference_pitch_rad",
        "log_leg_stiffness",
        "log_ankle_stiffness",
        "nominal_contact_phase",
        "touchdown_detected",
        "normalized_pelvis_error",
        "normalized_pitch_error",
    )
    observation_dim = len(observation_names)
    action_contract: ClassVar[dict] = {
        "version": "two_absolute_log_stiffness_v1",
        "names": ["log_leg_stiffness", "log_ankle_stiffness"],
        "bounds": [-1.0, 1.0],
        "mapping": "absolute log-space target; quintic frame ramp; instantaneous log-rate bounded",
        "fixed": ["leg_equilibrium", "ankle_equilibrium", "leg_damping_ratio", "ankle_damping_ratio"],
        "clock": "fixed nominal physical time for both schedules and full-episode motion reward",
    }
    _construction_keys: ClassVar[dict[str, str]] = {
        "foot_mass_kg": "foot_mass_kg",
        "pitch_inertia_kg_m2": "pitch_inertia_kg_m2",
        "ankle_mount_m": "ankle_mount_local_m",
        "source_shoe_side": "source_side",
        "shoe_side": "target_side",
        "leg_damping_ratio": "leg_damping_ratio",
        "ankle_damping_ratio": "ankle_damping_ratio",
    }

    def __init__(
        self,
        reference: Reference,
        artifact_path: str | Path,
        config: RigConfig | None = None,
        num_worlds: int = 1,
        device: str | None = None,
    ):
        self.reference = reference
        self.artifact_path = Path(artifact_path).resolve()
        construction = reference.provenance.get("config", {})
        if config is None:
            config = RigConfig(
                **{key: construction[value] for key, value in self._construction_keys.items() if value in construction}
            )
        self.config = config
        for key, value in self._construction_keys.items():
            if value not in construction:
                raise ValueError(f"Reference must freeze construction setting {value}")
            actual, expected = getattr(config, key), construction[value]
            same = (
                actual == expected if isinstance(actual, str) else np.allclose(actual, expected, rtol=0, atol=1.0e-12)
            )
            if not same:
                raise ValueError(f"Rig {key} differs from the frozen reference construction")
        if reference.mass_kg <= config.foot_mass_kg:
            raise ValueError("Total mass must exceed foot mass")
        if int(num_worlds) != num_worlds or num_worlds < 1:
            raise ValueError("num_worlds must be a positive integer")
        self.num_worlds = int(num_worlds)
        self.device = wp.get_device(device)
        self.shoe, self.shoe_orientation = orient_shoe(
            load_artifact(self.artifact_path), target_side=config.shoe_side, source_side=config.source_shoe_side
        )
        geometry_hash = geometry_identity(self.shoe)
        if reference.provenance.get("geometry_identity") != geometry_hash:
            raise ValueError("Shoe geometry does not match the frozen reference; only material may change")
        self.input_fingerprints = {
            "artifact_sha256": hashlib.sha256(self.artifact_path.read_bytes()).hexdigest(),
            "geometry_identity": geometry_hash,
            "material_identity": _identity(self.shoe.raw["constitutive_model"]),
            "reference_identity": reference.identity,
            "config_identity": _identity(config.to_dict()),
            "rig_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "foundation_source_sha256": physics_source_identity(),
        }
        self.ankle_mount = np.asarray(config.ankle_mount_m, dtype=float)
        self.mass = reference.mass_kg
        self.foot_mass = config.foot_mass_kg
        self.upper_mass = self.mass - self.foot_mass
        self.gravity = reference.gravity_m_s2
        self.pitch_inertia = config.pitch_inertia_kg_m2
        self.episode_frames = math.ceil(reference.duration_s * config.frame_rate_hz)
        self.frame_dt = reference.duration_s / self.episode_frames
        self.sim_dt = self.frame_dt / config.substeps
        self.sample_count = self.episode_frames * config.substeps
        self.times = np.arange(self.sample_count, dtype=float) * self.sim_dt
        self.duration = reference.duration_s
        self._log_min = np.log([config.leg_stiffness_min_n_m, config.ankle_stiffness_min_n_m_rad])
        self._log_max = np.log([config.leg_stiffness_max_n_m, config.ankle_stiffness_max_n_m_rad])
        self._log_initial = 0.5 * (self._log_min + self._log_max)
        self._log_current = np.tile(self._log_initial, (self.num_worlds, 1))
        self._log_start = wp.array(self._log_current, dtype=wp.float32, device=self.device)
        self._log_end = wp.array(self._log_current, dtype=wp.float32, device=self.device)
        self.observation_scale = np.asarray(
            [
                1,
                0.1,
                1,
                1,
                5,
                5,
                5,
                5,
                1,
                10,
                self.mass * self.gravity,
                self.mass * self.gravity,
                1,
                1,
                1,
                1,
                10,
                10,
                1,
                1,
                1,
                1,
            ],
            np.float32,
        )
        self._observation_scale_device = wp.array(self.observation_scale, dtype=wp.float32, device=self.device)
        self._observation_device = wp.zeros(
            (self.num_worlds, self.observation_dim), dtype=wp.float32, device=self.device
        )
        self._index_device = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._touchdown_device = wp.full(self.num_worlds, -1.0, dtype=wp.float32, device=self.device)
        self._flags_device = wp.zeros(self.num_worlds, dtype=wp.int32, device=self.device)
        self._frame_loss_device = wp.zeros(self.num_worlds, dtype=wp.float32, device=self.device)
        self._trace_device = wp.zeros(
            (self.sample_count, self.num_worlds, len(self._trace_names)), dtype=wp.float32, device=self.device
        )
        ref_columns = (
            "pelvis_z_m",
            "pelvis_vz_m_s",
            "pitch_rad",
            "pitch_rate_rad_s",
            "leg_length_m",
            "leg_rate_m_s",
            "ankle_equilibrium_rad",
            "ankle_equilibrium_rate_rad_s",
            "reference_fz_n",
            "reference_fx_n",
        )
        self._reference_device = wp.array(
            np.column_stack([getattr(reference, name) for name in ref_columns]), dtype=wp.float32, device=self.device
        )
        self._knots_device = wp.array(reference.time_s, dtype=wp.float32, device=self.device)
        self._prepare_runtime()
        self._build_model()
        self._parameters = self._make_parameters()
        self.graph = None
        self.use_graph = bool(config.use_graph and self.device.is_cuda)
        self.graph_status = "pending" if self.use_graph else "eager"
        self.metadata = {
            "input_fingerprints": self.input_fingerprints,
            "config": config.to_dict(),
            "action_contract": self.action_contract,
            "observation_names": list(self.observation_names),
            "observation_scale": self.observation_scale.tolist(),
            "observations_already_scaled": True,
            "reward": "-sum_substeps(dt*0.5*(pelvis_normalized_error^2+wrapped_pitch_normalized_error^2))",
            "pelvis_mapping": "measured pelvis centroid height maps directly to upper mass; not true COM",
            "clock": "fixed nominal physical time; detected contact diagnostic only; no flight masking",
            "trace_timing": "preintegration force evaluation; no extra terminal history update",
            "stiffness_interpolation": "C2 quintic log ramp; analytic Kdot; instantaneous log slew bound",
            "source_power": "raw_force*equilibrium_rate + 0.5*Kdot*error^2 + limit_intervention_power",
            "shoe_moment_y_n_m": "contact moment about the actual ankle/body origin, not the world origin",
            "damping_power": "-B*(achieved_rate-equilibrium_rate)^2",
            "contact": "massless attached shoe; external ground pressure wrench at z=0; no normal damping",
            "surround_assumptions": {
                "attachment_n_m": 0.0,
                "carrier_bond": True,
                "retention": "one-sided carrier-relative free surround, not outer tops glued to the last",
            },
            "last_support": self.last_support,
            "orientation": self.shoe_orientation,
            "sample_count": self.sample_count,
            "frame_dt_s": self.frame_dt,
            "substep_dt_s": self.sim_dt,
            "solver": "newton.solvers.SolverSemiImplicit; dynamic x/z and foot pitch; planar guide only",
        }
        self.reset()

    def _prepare_runtime(self):
        """Allocate an unused push placeholder for the default mechanical rig."""
        self._push_device = wp.zeros((1, 2), dtype=wp.float32, device=self.device)

    def _build_model(self):
        config, bed = self.config, self.shoe.column_bed
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0, 0.0, -self.gravity))
        shape_config = newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False)
        mesh = self.shoe.visual_mesh("fullfoot_last")
        local_last = mesh.vertices_m - self.ankle_mount
        self._last_hull = wp.array(_sagittal_hull(local_last), dtype=wp.vec2, device=self.device)
        carriers = []
        for world in range(self.num_worlds):
            carrier = builder.add_body(
                mass=self.foot_mass,
                com=wp.vec3(0.0),
                inertia=wp.mat33(np.eye(3) * self.pitch_inertia),
                label=f"foot_{world}",
            )
            carriers.append(carrier)
            builder.add_body(
                mass=self.upper_mass, com=wp.vec3(0.0), inertia=wp.mat33(np.eye(3)), label=f"upper_{world}"
            )
            if self.num_worlds == 1:
                builder.add_shape_mesh(
                    carrier,
                    mesh=newton.Mesh(np.asarray(local_last, np.float32), np.asarray(mesh.triangles, np.int32).ravel()),
                    cfg=shape_config,
                    color=(0.72, 0.77, 0.82),
                    label="rigid_last",
                )
                builder.add_shape_sphere(
                    carrier, radius=0.012, cfg=shape_config, color=(1, 0.5, 0.08), label="ankle_pivot"
                )
                builder.add_shape_sphere(
                    carrier + 1, radius=0.035, cfg=shape_config, color=(0.12, 0.62, 0.95), label="pelvis_centroid_proxy"
                )
        self.model = builder.finalize(device=self.device)
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.solver = newton.solvers.SolverSemiImplicit(self.model, angular_damping=0.0, enable_tri_contact=False)
        self.carrier = carriers[0]
        self._initial_q = np.zeros((2 * self.num_worlds, 7), np.float32)
        self._initial_q[:, 6] = 1.0
        self._initial_q[0::2, :3] = self.reference.foot_position_m
        self._initial_q[1::2, :3] = self.reference.upper_position_m
        angle = float(self.reference.pitch_rad[0])
        self._initial_q[0::2, 3:7] = (0, math.sin(angle / 2), 0, math.cos(angle / 2))
        self._initial_qd = np.zeros((2 * self.num_worlds, 6), np.float32)
        self._initial_qd[0::2, :3] = self.reference.foot_velocity_m_s
        self._initial_qd[1::2, :3] = self.reference.upper_velocity_m_s
        self._initial_qd[0::2, 4] = self.reference.pitch_rate_rad_s[0]
        self.contact_config = FoundationConfig(
            stretch_floor=config.stretch_floor,
            normal_damping=0.0,
            friction_stiffness=config.contact_kt_n_m,
            friction=config.contact_kd_n_s_m,
            mu=config.friction_mu,
            friction_viscous_ratio=config.friction_viscous_ratio,
            friction_release_dwell_s=config.friction_release_dwell_s,
            ground_height_m=self._ground_height_m,
        )
        # Preserve the old physical construction: only the last footprint is rigidly
        # driven; outer columns use the identified foundation's passive relaxation.
        lookup = {tuple(np.round(point, 8)): i for i, point in enumerate(bed.anchor_bottom_m[:, :2])}
        fixture = self.shoe.instron_fixture("fullfoot_last")
        supported = [lookup[tuple(np.round(point, 8))] for point in fixture.carrier_anchor_m[:, :2]]
        driven = np.zeros(len(bed.rest_length_m), np.int32)
        driven[supported] = 1
        fixture_gap = fixture.carrier_anchor_m[:, 2] - fixture.foam_free_top_m
        self.last_support = {
            "model": "idealized rigid backing on the projected fixture footprint",
            "rigid_last_is_contact_surface": False,
            "gap_aware_seating": False,
            "driven_columns": int(driven.sum()),
            "passive_columns": int(len(driven) - driven.sum()),
            "fixture_clearance_median_m": float(np.median(fixture_gap)),
            "fixture_clearance_max_m": float(np.max(fixture_gap)),
            "limitation": "fixture clearance and rigid-last seating are not a solved upper contact interface",
        }
        self.surround_config = SurroundConfig(
            driven=driven,
            attachment_n_m=0.0,
            max_strain=config.outer_max_strain,
            coupling_scale=config.outer_coupling_scale,
            sweeps=config.outer_sweeps,
            relaxation_time_s=config.outer_relaxation_s,
            carrier_bond=True,
        )
        self.foundation = MidsoleFoundation(
            bed.anchor_bottom_m - self.ankle_mount,
            np.full(len(bed.rest_length_m), self._ground_height_m),
            bed.rest_length_m,
            bed.area_m2,
            bed.neighbors,
            bed.spacing_m,
            self.shoe.material,
            carriers,
            self.model.body_com,
            self.contact_config,
            self.device,
            self.surround_config,
            world_count=self.num_worlds,
        )

    def _make_parameters(self):
        c, r = self.config, self.reference
        p = _Parameters()
        values = {
            "dt": self.sim_dt,
            "frame_dt": self.frame_dt,
            "substeps": c.substeps,
            "mass_upper": self.upper_mass,
            "mass_total": self.mass,
            "gravity": self.gravity,
            "pitch_inertia": self.pitch_inertia,
            "zeta_leg": c.leg_damping_ratio,
            "zeta_ankle": c.ankle_damping_ratio,
            "force_limit": c.force_limit_bw * self.mass * self.gravity,
            "torque_limit": c.ankle_torque_limit_n_m,
            "contact_threshold": c.contact_threshold_n,
            "contact_start": r.contact_start_s,
            "contact_duration": r.contact_duration_s,
            "z_scale": r.pelvis_scale_m,
            "pitch_scale": r.pitch_scale_rad,
            "clearance_min": c.minimum_last_clearance_m,
            "pelvis_min": c.minimum_pelvis_height_m,
            "pelvis_max": c.maximum_pelvis_height_m,
            "pitch_max": c.maximum_pitch_rad,
            "leg_min": c.minimum_leg_length_m,
            "leg_max": c.maximum_leg_length_m,
            "speed_max": c.maximum_speed_m_s,
            "ground_height": self._ground_height_m,
        }
        for key, value in values.items():
            setattr(p, key, value)
        return p

    def reset(self) -> np.ndarray:
        """Reset bodies, foam, friction, diagnostic history and stiffness history.

        The material, reference and initial pose never change during reset.
        """
        for state in (self.state_0, self.state_1):
            state.body_q.assign(self._initial_q)
            state.body_qd.assign(self._initial_qd)
            state.clear_forces()
        self.foundation.reset()
        # The foundation's public reset clears physical memory, but not all scratch
        # or diagnostic arrays. Clear those too so reset observations are exact.
        dynamic_arrays = (
            "z_free",
            "q_state",
            "peq_prev",
            "compression",
            "base_pressure",
            "tangent_anchor",
            "tangent_stuck",
            "tangent_dwell",
            "normal_force",
            "cop_moment",
            "active",
            "resultant_force",
            "resultant_moment_origin",
            "contact_power",
            "max_compression",
            "column_force",
            "column_pressed",
            "ground_force",
            "contact_point",
            "pressed_force",
            "partial_force",
            "partial_torque",
            "partial_moment",
            "partial_cop",
            "partial_normal",
            "partial_pressed",
            "partial_power",
            "partial_max",
            "partial_active",
            "surround_compression",
            "surround_scratch",
            "surround_previous",
            "surround_rate",
        )
        for name in dynamic_arrays:
            array = getattr(self.foundation, name, None)
            if array is not None:
                array.zero_()
        self._log_current[:] = self._log_initial
        self._log_start.assign(self._log_current)
        self._log_end.assign(self._log_current)
        self._index_device.zero_()
        self._touchdown_device.fill_(-1.0)
        self._flags_device.zero_()
        self._frame_loss_device.zero_()
        self._trace_device.zero_()
        self.index = 0
        self.frame = 0
        self.sim_time = 0.0
        self._episode_loss = np.zeros(self.num_worlds, dtype=float)
        self._trace_cache = None
        return self._observe()

    def _observe(self):
        wp.launch(
            _observe,
            dim=self.num_worlds,
            inputs=[
                self._parameters,
                self._index_device,
                self._knots_device,
                self._reference_device,
                self.state_0.body_q,
                self.state_0.body_qd,
                self.foundation.resultant_force,
                self._log_end,
                self._touchdown_device,
                self._observation_scale_device,
                self._last_hull,
                self._flags_device,
                self._observation_device,
            ],
            device=self.device,
        )
        return self._observation_device.numpy()

    def _substep(self, final: bool):
        self.state_0.clear_forces()
        self.foundation.apply(self.state_0, self.sim_dt)
        wp.launch(
            _actuate_record,
            dim=self.num_worlds,
            inputs=[
                self._parameters,
                self._index_device,
                self._knots_device,
                self._reference_device,
                self._log_start,
                self._log_end,
                self._push_device,
                self._touchdown_device,
                self.state_0.body_q,
                self.state_0.body_qd,
                self.state_0.body_f,
                self.foundation.resultant_force,
                self.foundation.resultant_moment_origin,
                self.foundation.contact_power,
                self.foundation.max_compression,
                self._last_hull,
                self._flags_device,
                self._frame_loss_device,
                self._trace_device,
            ],
            device=self.device,
        )
        self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
        if final and self.config.substeps % 2:
            self.state_0.assign(self.state_1)
        else:
            self.state_0, self.state_1 = self.state_1, self.state_0
        wp.launch(_planar, dim=self.num_worlds, inputs=[self.state_0.body_q, self.state_0.body_qd], device=self.device)
        wp.launch(_advance, dim=1, inputs=[self._index_device], device=self.device)

    def _frame(self):
        for substep in range(self.config.substeps):
            self._substep(final=substep == self.config.substeps - 1)

    def _capture(self):
        state_0, state_1 = self.state_0, self.state_1
        try:
            with wp.ScopedCapture(device=self.device) as capture:
                self._frame()
            self.graph = capture.graph
            self.graph_status = "captured"
        except Exception as error:
            self.state_0, self.state_1 = state_0, state_1
            self.use_graph = False
            self.graph_status = f"eager fallback: {error}"
            warnings.warn(f"Rig graph capture failed; using eager execution: {error}", stacklevel=2)

    def _update_stiffness(self, action):
        target = self._log_min + (action + 1.0) * 0.5 * (self._log_max - self._log_min)
        max_change = self.config.log_stiffness_slew_s * self.frame_dt / 1.875
        endpoint = self._log_current + np.clip(target - self._log_current, -max_change, max_change)
        self._log_start.assign(self._log_current)
        self._log_end.assign(endpoint)
        self._log_current[:] = endpoint

    def step(self, action) -> tuple[np.ndarray, np.ndarray, bool, dict]:
        """Advance one frame with two finite bounded absolute stiffness actions.

        Observations are already divided by ``observation_scale``. A nonfinite
        observation is replaced by zero ONLY to keep policy evaluation callable;
        its safety status is false and tracking loss is infinite. This is not a
        motion-reward bonus, clipping scheme, or extra penalty tier.
        """
        if self.frame >= self.episode_frames:
            raise RuntimeError("Episode is complete; call reset before stepping again")
        action = np.asarray(action, dtype=float)
        if action.shape != (self.num_worlds, 2) or not np.isfinite(action).all():
            raise ValueError(f"Actions must be finite with shape ({self.num_worlds},2)")
        if np.any(np.abs(action) > 1.0):
            raise ValueError("Actions must lie inside [-1,1]")
        self._update_stiffness(action)
        self._frame_loss_device.zero_()
        # One eager frame compiles all kernels and refreshes foundation constants
        # before graph capture. Capture never performs a host-side material update.
        if self.use_graph and self.graph is None and self.frame > 0:
            self._capture()
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self._frame()
        self.frame += 1
        self.index = self.frame * self.config.substeps
        self.sim_time = self.frame * self.frame_dt
        self._trace_cache = None
        obs = self._observe()
        loss = self._frame_loss_device.numpy().astype(float)
        flags = self._flags_device.numpy()
        invalid = ((flags & 1) != 0) | ~np.isfinite(loss)
        loss[invalid] = np.inf
        self._episode_loss += loss
        done = self.frame == self.episode_frames
        if done:
            flags = flags.copy()
            flags[self._touchdown_device.numpy() < 0] |= 256
        reasons = [[name for bit, name in SAFETY_REASONS.items() if int(flag) & bit] for flag in flags]
        info = {
            "tracking_loss": self._episode_loss.copy(),
            "safety_ok": flags == 0,
            "safety_reasons": reasons,
            "safety_flags": flags,
            "frame_tracking_loss": loss.copy(),
            "time_s": self.sim_time,
            "graph_status": self.graph_status,
            "input_fingerprints": self.input_fingerprints,
        }
        return obs, -loss, done, info

    def trace(self, world: int = 0) -> dict[str, np.ndarray]:
        """Return named full-resolution traces and rectangular-rule actuator work.

        Work sample i is cumulative work AFTER interval i, whose power was
        evaluated at ``time_s[i]``. Sum ``power * sim_dt``; do not trapezoid away
        the first or last interval. No force-limit/release flag is a power.
        """
        if not 0 <= int(world) < self.num_worlds or int(world) != world:
            raise IndexError("world index is outside the batch")
        if self._trace_cache is None:
            self._trace_cache = self._trace_device.numpy()[: self.index]
        data = self._trace_cache[:, int(world), :]
        result = {name: data[:, i].astype(float, copy=True) for i, name in enumerate(self._trace_names)}
        result["upper_x_m"] = result["pelvis_x_m"].copy()
        result["upper_z_m"] = result["pelvis_z_m"].copy()
        result["nominal_contact_phase"] = (
            result["time_s"] - self.reference.contact_start_s
        ) / self.reference.contact_duration_s
        for actuator in ("leg", "ankle"):
            for source in ("source", "damping", "limit", "body"):
                key = f"{actuator}_{source}"
                result[f"{key}_work_j"] = np.cumsum(result[f"{key}_power_w"]) * self.sim_dt
        return result
