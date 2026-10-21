# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Batched three-dimensional bilateral runner environment on a Newton articulation.

Stage 3 of the bilateral plan. Each world is the :mod:`.model3d` articulation
integrated by :class:`newton.solvers.SolverFeatherstone` at the 1.25e-4 s
substep, with one persistent shoe foundation per foot applying its full 3D
wrench to the foot body. Ten joints (three-axis hips, knees, ankles) are driven
by the same bounded impedance law as the planar model: the policy sets
equilibrium, stiffness, and damping per joint each control step, and the torque
cap acts every substep. As in :class:`.env.EnvConfig`, the torque lag and rate
bound are off by default. The floating pelvis is unactuated.

The shoe's lean contact kernels screen flight over every column, not only the
pitch-plane candidates of the planar runner, because the 3D foot also rolls and
yaws. Observations hold only simulated state, internal phase, filtered foot
loads, and torques; measurements enter rewards, terminations, initial states,
and critic inputs.

Episodes outlast the measured segments: the reference continues into the next
segment of the same trial, or at the best-matching frame where the measurement
ends. Horizontally the runner follows a pacer, like a treadmill: a point that
starts at the initial pelvis position and moves along +x at the task speed. The
actor observes its offset from the pacer and its heading, so the same policy
can run open-ended without a reference.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import warp as wp
from scipy.spatial import ConvexHull

import newton

from ..cartesian.shoe import Shoe
from . import actuation, data3d, leg_contact, model3d, shoe_randomization
from .env import _fused_shoe, response_fraction
from .kinematics3d import JOINT_NAMES
from .mechanics import SIDES, HeadArmsTrunk

wp.set_module_options({"enable_backward": False})

_J = wp.constant(model3d.ACTUATED)
VecJ = wp.types.vector(model3d.ACTUATED, wp.float32)
_FLIGHT = wp.constant(wp.float32(10.0))

_JOINTS_R = [name for name in JOINT_NAMES if name.endswith("_r")]
OBSERVATION_NAMES = (
    "pelvis_height_m",
    "gravity_x",
    "gravity_y",
    "gravity_z",
    "forward_speed_error_m_s",
    "lateral_speed_m_s",
    "vertical_speed_m_s",
    "pelvis_rate_x",
    "pelvis_rate_y",
    "pelvis_rate_z",
    *(f"{name}_rad" for name in JOINT_NAMES),
    *(f"{name}_rate_10" for name in JOINT_NAMES),
    "load_r_bw",
    "load_l_bw",
    *(f"torque_{name}_cap" for name in JOINT_NAMES),
    "ankle_r_x_m",
    "ankle_r_y_m",
    "ankle_r_z_m",
    "ankle_l_x_m",
    "ankle_l_y_m",
    "ankle_l_z_m",
    "sin_phase",
    "cos_phase",
    "task_speed_offset",
    "pacer_x_m",
    "pacer_y_m",
    "heading_sin",
    "heading_cos",
)
ACTION_NAMES = tuple(f"{output}_{name}" for output in ("equilibrium", "stiffness", "damping") for name in JOINT_NAMES)
REWARD_NAMES = ("pose", "velocity", "position", "force", "effort", "smoothness")
EXTRA_REWARD_NAMES = ("cadence", "crossing")
"""Terms outside the six weighted ones, each with its own :class:`Env3DConfig` weight."""


@dataclass(frozen=True)
class Bounds3D:
    """Impedance and torque bounds of one leg's hip flexion, adduction, rotation, knee, and ankle.

    Sagittal values match the planar :class:`..hogan.runner.Bounds` (knee flexion
    is positive here); frontal and transverse hip values are engineering choices.
    """

    equilibrium_lower_rad: tuple[float, ...] = (-1.5, -0.6, -0.8, 0.02, -1.2)
    equilibrium_upper_rad: tuple[float, ...] = (1.5, 0.6, 0.8, 2.8, 1.2)
    stiffness_max_nm_rad: tuple[float, ...] = (1500.0, 1000.0, 400.0, 1500.0, 1000.0)
    damping_max_nms_rad: tuple[float, ...] = (80.0, 50.0, 20.0, 80.0, 50.0)
    torque_max_nm: tuple[float, ...] = (250.0, 200.0, 80.0, 300.0, 200.0)
    torque_rate_max_nm_s: tuple[float, ...] = (5000.0, 4000.0, 2000.0, 6000.0, 4000.0)


def seed_action(bounds: Bounds3D | None = None) -> np.ndarray:
    """Unfitted engineering seed logits: the planar seed impedance, with moderate frontal and axial hip terms."""
    b = bounds or Bounds3D()
    lower, upper = np.tile(b.equilibrium_lower_rad, 2), np.tile(b.equilibrium_upper_rad, 2)
    fractions = np.concatenate(
        (
            (np.tile([0.2, 0.0, 0.0, 0.5, 0.0], 2) - lower) / (upper - lower),
            np.tile([250.0, 200.0, 50.0, 300.0, 200.0], 2) / np.tile(b.stiffness_max_nm_rad, 2),
            np.tile([12.0, 8.0, 3.0, 10.0, 6.0], 2) / np.tile(b.damping_max_nms_rad, 2),
        )
    )
    return np.log(fractions / (1 - fractions))


@dataclass(frozen=True)
class Env3DConfig:
    dt_s: float = 1.25e-4
    substeps: int = 16
    response_time_s: float = 0.0
    """Joint torque response time [s]; zero, the default, applies the impedance torque
    without lag, as in :attr:`.env.EnvConfig.response_time_s`."""
    torque_slew_limit: bool = False
    """Bound the joint torque rate by :attr:`Bounds3D.torque_rate_max_nm_s`; ``False``,
    the default, removes it."""
    load_time_s: float = 0.02
    frequency_hz: float = 1.5
    phase_feedback: float = 0.2
    reference_speed_m_s: float = 3.65
    speed_scale_m_s: float = 1.0
    immediate_damping: bool = True
    maximum_force_n: float = 6000.0
    minimum_hip_height_m: float = 0.3
    compression_limit: float = 0.99
    solver: str = "featherstone"
    """Newton solver of the articulation: ``"featherstone"`` or ``"mujoco"`` (MuJoCo Warp; the
    shoes apply every ground force, and its contacts are on only for :attr:`leg_contact`
    ``"mujoco"``)."""
    mass_matrix_interval: int = 2
    """Featherstone mass-matrix refresh interval [substeps]; the matrix changes little over 0.25 ms."""
    leg_contact: str = "off"
    """Inter-leg collision profile: ``"off"``, or a profile of :data:`.leg_contact.PROFILES`.
    Capsule contact between the two legs is the only body-to-body force in the model; the
    ground still comes from the shoes alone, and ``"probe"`` records their penetration
    without applying any force. ``"mujoco"`` (needs :attr:`solver` ``"mujoco"``) makes the
    capsules MuJoCo collision shapes and turns MuJoCo's contacts on for them alone."""
    leg_contact_penalty: float = 0.0
    """Reward weight on inter-leg penetration; 0 leaves the reward untouched. The tracking terms
    are blind to a swing leg crossing inside the stance leg, so this is what makes the defect cost
    something while the contact law is too soft to stop it mechanically."""
    leg_contact_penalty_depth_m: float = 0.03
    """Penetration at which the penalty saturates [m]; deeper crossings cost the same as this."""
    shoe_dr: str = "off"
    """Per-world shoe randomization: a preset of :data:`.shoe_randomization.PRESETS`, or
    ``"off"`` for the measured shoe in every world. Theta covers the foam, the Coulomb
    coefficient, the unfitted normal damping and the stack height; it is drawn at reset
    and resampled mid-episode, which is what turns a novel shoe into a testable input
    instead of a retraining event."""
    shoe_swap_probability: float = 0.01
    """Probability per control step that a running world gets a new shoe, when
    :attr:`shoe_dr` is on. RMA resamples its environment parameters at 0.01 per step.
    Zero keeps one shoe per world per episode, which is what a paired-world evaluation
    needs."""
    shoe_dr_seed: int = 0
    """Seed of the shoe draw, separate from the environment seed so the same worlds can
    be re-shod for evaluation without moving the initial states."""
    contact_observation: bool = False
    """Append the per-foot foam channels of
    :data:`.shoe_randomization.CONTACT_OBSERVATION_NAMES` to the actor inputs. They are
    the only channels that carry foam information, so an adaptation module cannot infer
    a shoe without them."""
    theta_observation: bool = False
    """Append the normalized shoe parameters to the actor inputs: the privileged teacher
    of the design, which is later distilled into a history-conditioned student."""
    pose_group_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)
    """Relative weight of the hip, knee, and ankle groups in the pose term."""
    pose_dof_weights: tuple[float, ...] = (1.0,) * 10
    """Per-coordinate multiplier on the pose term in :data:`JOINT_NAMES` order, applied on top of
    ``pose_group_weights``. One scale over ten coordinates makes a coordinate that is small in radians
    nearly free: the hip adduction offset measured 4.3 deg and was worth 1.3 % of the reward while the
    knees, whose error is large, carried half the pose loss."""
    stance_pose_relief: float = 0.0
    """Fraction by which the knee and ankle pose weight relaxes while that foot is in
    stance. A shoe changes how a stance leg has to move, so holding stance angles as
    tightly as pelvis height pins the gait to the trained shoe."""
    cadence_weight: float = 0.0
    """Weight of the cadence-band term. It replaces the measured force trace as the term
    that keeps the gait a run: it survives a shoe change, the trace does not."""
    cadence_band_spm: float = 6.0
    """Half-width of the cadence band [steps/min] inside which the cadence term is zero."""
    actuation: str | None = None
    """Smooth substep actuation of the impedance setpoints: a profile of
    :data:`.actuation.PROFILES` or a JSON object; ``None`` keeps the zero-order hold."""
    episode_s: float = 10.0
    """Episode length [s]; the reference continues across segments, see :class:`ReferenceLibrary3D`."""
    termination_height_m: float = 0.15
    termination_orientation_rad: float = 0.6
    termination_horizontal_m: float = 0.5
    """Largest horizontal distance from the pacer [m]."""
    pacer_scale_m: float = 0.15
    """Horizontal distance from the pacer in the position reward [m]."""
    reward_weights: tuple[float, float, float, float, float, float] = (0.35, 0.1, 0.2, 0.25, 0.1, 0.1)
    """Weights of the pose, velocity, position, force, effort, and smoothness terms."""
    pose_scale_rad: float = 0.15
    velocity_scale_rad_s: float = 3.0
    position_scale_m: float = 0.05
    force_scale_bw: float = 0.3
    effort_scale: float = 0.15
    smoothness_scale: float = 1.0
    """Joint torque jump at a control step boundary, in units of the joint's torque rate
    bound (:attr:`Bounds3D.torque_rate_max_nm_s`) over one control step."""

    @property
    def control_dt_s(self) -> float:
        return self.dt_s * self.substeps


@wp.struct
class Params3D:
    dt: float
    response: float
    load_rate: float
    frequency: float
    phase_feedback: float
    immediate_damping: int
    max_force: float
    hip_floor: float
    compression_limit: float
    body_weight: float
    lower: VecJ
    upper: VecJ
    stiffness_max: VecJ
    damping_max: VecJ
    cap: VecJ
    rate: VecJ


@wp.struct
class Worlds3D:
    status: wp.array[int]
    torque: wp.array[VecJ]
    load: wp.array[VecJ]
    equilibrium: wp.array[VecJ]
    stiffness: wp.array[VecJ]
    damping: wp.array[VecJ]
    phase: wp.array[float]
    sense: wp.array[wp.vec2]
    grf: wp.array2d[wp.vec3]
    effort: wp.array[VecJ]
    power_positive: wp.array[VecJ]
    power_negative: wp.array[VecJ]
    substeps: wp.array[int]
    fraction_r: wp.array[wp.float64]
    fraction_l: wp.array[wp.float64]
    invalid_r: wp.array[int]
    invalid_l: wp.array[int]
    enabled_r: wp.array[int]
    enabled_l: wp.array[int]
    shift_r: wp.array[wp.vec3]
    shift_l: wp.array[wp.vec3]


def _params(cfg: Env3DConfig, bounds: Bounds3D, body_weight_n: float) -> Params3D:
    """Kernel settings of one configuration.

    The torque response fraction and rate bound follow the planar environment, so the
    default configuration applies the impedance command without lag or a rate limit.
    """
    params = Params3D()
    params.dt = cfg.dt_s
    params.response = response_fraction(cfg.dt_s, cfg.response_time_s)
    params.load_rate = -math.expm1(-cfg.dt_s / cfg.load_time_s)
    params.frequency = cfg.frequency_hz
    params.phase_feedback = cfg.phase_feedback
    params.immediate_damping = int(cfg.immediate_damping)
    params.max_force = cfg.maximum_force_n
    params.hip_floor = cfg.minimum_hip_height_m
    params.compression_limit = cfg.compression_limit
    params.body_weight = body_weight_n
    both = lambda values: VecJ(*values, *values)  # noqa: E731
    params.lower, params.upper = both(bounds.equilibrium_lower_rad), both(bounds.equilibrium_upper_rad)
    params.stiffness_max, params.damping_max = both(bounds.stiffness_max_nm_rad), both(bounds.damping_max_nms_rad)
    params.cap = both(bounds.torque_max_nm)
    rate = bounds.torque_rate_max_nm_s if cfg.torque_slew_limit else (1.0e12,) * len(bounds.torque_rate_max_nm_s)
    params.rate = both(rate)
    return params


@wp.kernel
def _actuate(
    p: Params3D, data: Worlds3D, joint_q: wp.array[float], joint_qd: wp.array[float], joint_f: wp.array[float]
):
    w = wp.tid()
    if data.status[w] != 0:
        for j in range(_J):
            joint_f[w * 16 + 6 + j] = 0.0
        return
    torque = data.torque[w]
    load = VecJ()
    eq = data.equilibrium[w]
    k = data.stiffness[w]
    d = data.damping[w]
    for j in range(_J):
        q = joint_q[w * 17 + 7 + j]
        v = joint_qd[w * 16 + 6 + j]
        spring = k[j] * (eq[j] - q)
        command = spring - d[j] * v
        immediate = float(0.0)
        if p.immediate_damping != 0:
            command = spring
            immediate = d[j]
        cap = p.cap[j]
        limit = p.rate[j] * p.dt
        change = wp.clamp(p.response * (wp.clamp(command, -cap, cap) - torque[j]), -limit, limit)
        torque[j] = wp.clamp(torque[j] + change, -cap, cap)
        load[j] = wp.clamp(torque[j] - immediate * v, -cap, cap)
        joint_f[w * 16 + 6 + j] = load[j]
    data.torque[w] = torque
    data.load[w] = load


@wp.func
def _carrier(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    body: int,
    shift: wp.vec3,
):
    """Shoe carrier pose and twist: the foot frame moved by the world's mount shift."""
    pose = body_q[body]
    twist = body_qd[body]
    omega = wp.spatial_bottom(twist)
    arm = wp.transform_vector(pose, shift - body_com[body])
    linear = wp.spatial_top(twist) + wp.cross(omega, arm)
    return wp.transform(wp.transform_point(pose, shift), wp.transform_get_rotation(pose)), wp.spatial_vector(
        linear, omega
    )


@wp.kernel
def _stage(
    data: Worlds3D,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    carrier_q_r: wp.array[wp.transform],
    carrier_qd_r: wp.array[wp.spatial_vector],
    carrier_q_l: wp.array[wp.transform],
    carrier_qd_l: wp.array[wp.spatial_vector],
):
    w = wp.tid()
    if data.status[w] != 0:
        lifted = wp.transform(wp.vec3(0.0, 0.0, _FLIGHT), wp.quat_identity())
        carrier_q_r[w] = lifted
        carrier_q_l[w] = lifted
        carrier_qd_r[w] = wp.spatial_vector()
        carrier_qd_l[w] = wp.spatial_vector()
        return
    pose, twist = _carrier(body_q, body_qd, body_com, w * 7 + 3, data.shift_r[w])
    carrier_q_r[w] = pose
    carrier_qd_r[w] = twist
    pose, twist = _carrier(body_q, body_qd, body_com, w * 7 + 6, data.shift_l[w])
    carrier_q_l[w] = pose
    carrier_qd_l[w] = twist


@wp.func
def _apply(
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_f: wp.array[wp.spatial_vector],
    body: int,
    shift: wp.vec3,
    wrench: wp.spatial_vector,
):
    """Apply a carrier wrench, given about the shifted carrier origin, at the body's COM."""
    force = wp.spatial_top(wrench)
    arm = wp.transform_vector(body_q[body], shift - body_com[body])
    body_f[body] = wp.spatial_vector(force, wp.spatial_bottom(wrench) + wp.cross(arm, force))


@wp.kernel
def _contact(
    p: Params3D,
    data: Worlds3D,
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_f: wp.array[wp.spatial_vector],
    wrench_r: wp.array[wp.spatial_vector],
    wrench_l: wp.array[wp.spatial_vector],
    joint_qd: wp.array[float],
):
    w = wp.tid()
    if data.status[w] != 0:
        return
    fr = wrench_r[w]
    fl = wrench_l[w]
    bad = int(0)
    for i in range(6):
        if not (wp.isfinite(fr[i]) and wp.isfinite(fl[i])):
            bad = 1
    code = int(0)
    if bad != 0 or data.invalid_r[w] != 0 or data.invalid_l[w] != 0:
        code = 4
    elif (
        fr[2] < -1.0e-6
        or fl[2] < -1.0e-6
        or wp.length(wp.spatial_top(fr)) > p.max_force
        or wp.length(wp.spatial_top(fl)) > p.max_force
    ):
        code = 5
    elif wp.max(data.fraction_r[w], data.fraction_l[w]) > wp.float64(p.compression_limit + 1.0e-6):
        code = 6
    if code != 0:
        data.status[w] = code
        data.enabled_r[w] = 0
        data.enabled_l[w] = 0
        return
    _apply(body_q, body_com, body_f, w * 7 + 3, data.shift_r[w], fr)
    _apply(body_q, body_com, body_f, w * 7 + 6, data.shift_l[w], fl)
    data.grf[w, 0] = data.grf[w, 0] + wp.spatial_top(fr)
    data.grf[w, 1] = data.grf[w, 1] + wp.spatial_top(fl)
    sense = data.sense[w]
    sense[0] = sense[0] + p.load_rate * (wp.max(fr[2], 0.0) / p.body_weight - sense[0])
    sense[1] = sense[1] + p.load_rate * (wp.max(fl[2], 0.0) / p.body_weight - sense[1])
    data.sense[w] = sense
    phase = data.phase[w]
    drive = (wp.min(sense[0], 1.0) - wp.min(sense[1], 1.0)) * wp.sin(phase)
    phase = phase + 2.0 * wp.pi * p.frequency * (1.0 - p.phase_feedback * drive) * p.dt
    data.phase[w] = phase - 2.0 * wp.pi * wp.floor(phase / (2.0 * wp.pi))
    load = data.load[w]
    effort = data.effort[w]
    positive = data.power_positive[w]
    negative = data.power_negative[w]
    for j in range(_J):
        effort[j] = effort[j] + load[j] * load[j]
        power = load[j] * joint_qd[w * 16 + 6 + j]
        positive[j] = positive[j] + wp.max(power, 0.0)
        negative[j] = negative[j] + wp.min(power, 0.0)
    data.effort[w] = effort
    data.power_positive[w] = positive
    data.power_negative[w] = negative
    data.substeps[w] = data.substeps[w] + 1


@wp.kernel
def _screen(p: Params3D, data: Worlds3D, joint_q: wp.array[float], joint_qd: wp.array[float]):
    w = wp.tid()
    if data.status[w] != 0:
        return
    bad = int(0)
    for i in range(17):
        if not wp.isfinite(joint_q[w * 17 + i]):
            bad = 1
    for i in range(16):
        if not (wp.isfinite(joint_qd[w * 16 + i]) and wp.abs(joint_qd[w * 16 + i]) < 100.0):
            bad = 1
    code = int(0)
    if bad != 0:
        code = 8
    elif joint_q[w * 17 + 2] < p.hip_floor:
        code = 9
    if code != 0:
        data.status[w] = code
        data.enabled_r[w] = 0
        data.enabled_l[w] = 0


def _quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    x, y, z, w = q.unbind(-1)
    return torch.stack(
        (
            torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)), -1),
            torch.stack((2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)), -1),
            torch.stack((2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)), -1),
        ),
        -2,
    )


def _heading(rotation: torch.Tensor) -> torch.Tensor:
    """Yaw-only rotation of the pelvis forward axis, shape [..., 3, 3]."""
    yaw = torch.atan2(rotation[..., 1, 0], rotation[..., 0, 0])
    c, s = torch.cos(yaw), torch.sin(yaw)
    zero, one = torch.zeros_like(yaw), torch.ones_like(yaw)
    return torch.stack(
        (torch.stack((c, -s, zero), -1), torch.stack((s, c, zero), -1), torch.stack((zero, zero, one), -1)), -2
    )


def _quat_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack(
        (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ),
        -1,
    )


def _successors(members, q, qd, grf, control_dt_s: float, body_weight_n: float) -> list[tuple[int, int]]:
    """Return the segment and frame that continue each segment after its last frame.

    A segment whose source window is followed directly by another loaded segment
    of the same trial continues into it. Otherwise it continues at the frame of
    the same trial whose joint angles and rates, pelvis height, tilt, and vertical
    rate, and per-foot vertical force best match its last frame, at least 0.5 s
    before the end of that frame's segment.
    """

    def features(segment: int, frames) -> np.ndarray:
        position, rate = q[segment][frames], qd[segment][frames]
        return np.column_stack(
            (
                position[:, 7:] / 0.1,
                rate[:, 7:] / 2.0,
                position[:, 2:3] / 0.02,
                position[:, 3:5] / 0.05,
                rate[:, 2:3] / 0.2,
                grf[segment][frames][:, [2, 5]] / (0.2 * body_weight_n),
            )
        )

    margin = int(round(0.5 / control_dt_s))
    result = []
    for s, member in enumerate(members):
        last = len(q[s]) - 1
        following = [
            t
            for t, other in enumerate(members)
            if other["trial"] == member["trial"] and abs(other["start_s"] - member["end_s"]) < 1e-3
        ]
        if following:
            t = following[0]
            frame = int(round((member["start_s"] + (last + 1) * control_dt_s - members[t]["start_s"]) / control_dt_s))
            result.append((t, max(frame, 0)))
            continue
        target = features(s, [last])[0]
        best = (np.inf, s, 0)
        for t, other in enumerate(members):
            if other["trial"] != member["trial"]:
                continue
            frames = np.arange(3, len(q[t]) - margin)
            distance = np.sum((features(t, frames) - target) ** 2, axis=1)
            k = int(np.argmin(distance))
            best = min(best, (float(distance[k]), t, int(frames[k])))
        result.append(best[1:])
    return result


class ReferenceLibrary3D:
    """Measured 3D segments resampled to the control clock on the GPU.

    ``successor[s]`` is the segment and frame that continue segment ``s`` after its
    last frame, so an episode can follow the reference for longer than one segment.
    """

    def __init__(self, dataset: str | Path, splits: tuple[str, ...], control_dt_s: float, device):
        root = Path(dataset).resolve()
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("schema") != data3d.DATASET_SCHEMA:
            raise ValueError("Unsupported 3D bilateral dataset schema")
        self.root, self.manifest = root, manifest
        self.members = [m for m in manifest["members"] if m["split"] in splits]
        if not self.members:
            raise ValueError(f"No members in splits {splits}")
        self.trials = sorted(manifest["calibrations"])
        self.calibrations = [data3d.calibration_from_dict(manifest["calibrations"][t]) for t in self.trials]
        self.profile = json.loads(self.asset("profile.json").read_text(encoding="utf-8"))
        self.hat = HeadArmsTrunk(**manifest["head_arms_trunk"])
        q, qd, grf, starts, lengths, flight, trial_of, references = [], [], [], [], [], [], [], []
        for member in self.members:
            reference = data3d.load(root / member["reference"])
            references.append(reference)
            time = reference["time_s"]
            count = int(math.floor(time[-1] / control_dt_s)) + 1
            clock = np.arange(count) * control_dt_s
            raw = reference["q"]
            positions = np.column_stack([np.interp(clock, time, raw[:, c]) for c in range(raw.shape[1])])
            positions[:, 3:7] /= np.linalg.norm(positions[:, 3:7], axis=1, keepdims=True)
            rates = np.gradient(raw, time, axis=0)
            velocity = np.column_stack([np.interp(clock, time, rates[:, c]) for c in range(raw.shape[1])])
            fine = np.arange(count * 8) * (control_dt_s / 8)
            force = (
                np.stack(
                    [
                        np.column_stack(
                            [
                                np.interp(fine, reference["grf_time_s"], reference["grf_side_n"][s, :, a])
                                for a in range(3)
                            ]
                        )
                        for s in range(2)
                    ],
                    axis=1,
                )
                .reshape(count, 8, 2, 3)
                .mean(axis=1)
            )
            starts.append(sum(lengths))
            lengths.append(count)
            q.append(positions)
            qd.append(velocity)
            grf.append(force.reshape(count, 6))
            flight.append(np.all(force[:, :, 2] < 1.0, axis=1))
            trial_of.append(self.trials.index(member["trial"]))
        self.references = references
        self.body_weight_n = float(manifest["subject_mass_kg"]) * 9.81
        self.speeds = torch.tensor([m["speed_m_s"] for m in self.members], dtype=torch.float32, device=device)
        self.q = torch.tensor(np.concatenate(q), dtype=torch.float32, device=device)
        self.qd = torch.tensor(np.concatenate(qd), dtype=torch.float32, device=device)
        self.grf = torch.tensor(np.concatenate(grf), dtype=torch.float32, device=device)
        self.starts = torch.tensor(starts, dtype=torch.long, device=device)
        self.lengths = torch.tensor(lengths, dtype=torch.long, device=device)
        self.control_dt_s = float(control_dt_s)
        self.trial_of = torch.tensor(trial_of, dtype=torch.long, device=device)
        self.successor = torch.tensor(
            _successors(self.members, q, qd, grf, control_dt_s, self.body_weight_n), dtype=torch.long, device=device
        )
        flight_all = np.concatenate(flight)
        minimum = int(round(0.25 / control_dt_s))
        candidates = [
            (segment, k)
            for segment, (start, length) in enumerate(zip(starts, lengths, strict=True))
            for k in range(3, length - minimum)
            if flight_all[start + k]
        ]
        self.initial_candidates = torch.tensor(candidates, dtype=torch.long, device=device)

    def __len__(self) -> int:
        return len(self.members)

    def asset(self, name: str) -> Path:
        """Resolve a shared asset relative to the dataset; older manifests stored absolute local paths."""
        path = Path(self.manifest["shared_assets"][name]["file"])
        return path if path.is_absolute() else self.root / path


def _segment_cadence(library: ReferenceLibrary3D) -> np.ndarray:
    """Measured steps per minute of each reference segment, from its own foot forces.

    The cadence term is a band around what this segment actually did, not around the
    task speed divided by an assumed stride, so a policy is never asked to run at a
    cadence the subject did not use in that trial.
    """
    grf = library.grf.detach().cpu().numpy()
    starts, lengths = library.starts.cpu().numpy(), library.lengths.cpu().numpy()
    rate = library.control_dt_s
    cadence = np.full(len(lengths), 180.0, dtype=np.float32)
    for segment in range(len(lengths)):
        start, length = int(starts[segment]), int(lengths[segment])
        loaded = (grf[start : start + length, 2] > shoe_randomization.CONTACT_FORCE_N) | (
            grf[start : start + length, 5] > shoe_randomization.CONTACT_FORCE_N
        )
        touchdowns = int(np.count_nonzero(np.diff(loaded.astype(np.int8)) == 1))
        if length > 1 and touchdowns > 1:
            cadence[segment] = 60.0 * touchdowns / (length * rate)
    return cadence


def _mount_shifts(library: ReferenceLibrary3D, world_trials: np.ndarray) -> np.ndarray:
    """Per-world, per-side shoe shifts in the foot frame, shape [worlds, 2 (right, left), 3].

    The shared shoe geometry uses the dataset's mean static ankle height. Each world
    moves its shoes so the ankle sits at its own trial and side's static height.
    Datasets without a static calibration use the shared mount unchanged.
    """
    shoe = library.manifest["shoe"]
    shifts = np.zeros((len(world_trials), len(SIDES), 3), dtype=np.float32)
    calibration = shoe.get("static_calibration")
    if calibration is None:
        return shifts
    heights = calibration["ankle_height_m"]
    for world, trial in enumerate(world_trials):
        for k, side in enumerate(SIDES):
            shifts[world, k, 2] = shoe["mount_m"][2] - heights[library.trials[trial]][side]
    return shifts


class BilateralEnv3D:
    """Vectorized 3D bilateral runner worlds with the planar environment's interface."""

    observation_size = len(OBSERVATION_NAMES)
    action_size = len(ACTION_NAMES)
    privileged_size = 40

    def __init__(
        self,
        library: ReferenceLibrary3D,
        worlds: int,
        config: Env3DConfig | None = None,
        bounds: Bounds3D | None = None,
        device: str = "cuda:0",
        seed: int = 0,
        world_trials: np.ndarray | None = None,
    ):
        self.config = cfg = config or Env3DConfig()
        self.bounds = bounds or Bounds3D()
        self.library = library
        self.worlds = int(worlds)
        self.device = wp.get_device(device)
        self.torch_device = wp.device_to_torch(self.device)
        # Warp launches on its own stream, and a torch capture records only the stream it
        # captures, so :meth:`capture` records on this one to keep the forward kinematics.
        self._stream = wp.stream_to_torch(self.device.stream)
        torch.cuda.set_stream(self._stream)
        self.generator = torch.Generator(device=self.torch_device)
        self.generator.manual_seed(seed)
        w, d = self.worlds, self.device
        if world_trials is None:
            world_trials = np.arange(w) % len(library.trials)
        self.world_trials = torch.tensor(world_trials, dtype=torch.long, device=self.torch_device)
        subject = float(library.manifest["subject_mass_kg"])
        mount, pitch = library.manifest["shoe"]["mount_m"], library.manifest["shoe"]["pitch_rad"]
        shoes = {side: Shoe(library.asset(f"digital_shoe_{side}.json"), mount, pitch) for side in SIDES}
        capsules = None
        if cfg.leg_contact != "off":
            bed = shoes["left"].shoe.column_bed
            capsules = leg_contact.shoe_config(
                bed.anchor_bottom_m, bed.rest_length_m, mount, leg_contact.profile(cfg.leg_contact)
            )
        mujoco_legs = capsules is not None and capsules.engine == "mujoco"
        if mujoco_legs and cfg.solver != "mujoco":
            raise ValueError(f"leg_contact {cfg.leg_contact!r} needs solver 'mujoco', not {cfg.solver!r}")
        self.model = model3d.build_worlds(
            library.calibrations,
            world_trials,
            library.profile,
            subject,
            library.hat,
            d,
            capsules if mujoco_legs else None,
        )
        if cfg.solver == "featherstone":
            self.solver = newton.solvers.SolverFeatherstone(
                self.model, angular_damping=0.0, update_mass_matrix_interval=cfg.mass_matrix_interval
            )
        elif cfg.solver == "mujoco":
            import mujoco  # noqa: PLC0415

            if mujoco_legs:
                # Nine right-left capsule pairs, each at most two points (parallel axes), and four
                # pyramid rows per point with friction. The contact solve needs real iterations;
                # MuJoCo's Newton solver stops early at its tolerance.
                limits = {"disable_contacts": False, "nconmax": 18, "njmax": 72, "iterations": 20, "ls_iterations": 50}
            else:
                # The model has no MuJoCo constraints, so njmax=0 takes the solver's exact
                # qacc = qacc_smooth path and drops the constraint solve and its host readbacks.
                # That path never writes efc.Ma, which Euler's implicit joint damping reads; the
                # joints have no springs or damping, so disabling those terms is exact.
                limits = {"disable_contacts": True, "njmax": 0, "iterations": 1, "ls_iterations": 1}
            self.solver = newton.solvers.SolverMuJoCo(self.model, integrator="euler", jacobian="dense", **limits)
            disabled = mujoco.mjtDisableBit
            self.solver.mjw_model.opt.disableflags |= int(
                disabled.mjDSBL_SPRING | disabled.mjDSBL_DAMPER | disabled.mjDSBL_EULERDAMP
            )
        else:
            raise ValueError(f"unknown solver {cfg.solver!r}; expected 'featherstone' or 'mujoco'")
        self.states = [self.model.state(), self.model.state()]
        self.control = self.model.control()
        self.shoe_metadata = {side: shoes[side].metadata for side in SIDES}
        self.shoes = {side: _fused_shoe(shoes[side], w, cfg.dt_s, d, cfg.compression_limit) for side in SIDES}
        for shoe in self.shoes.values():
            # The screen tests whether the lowest column is near the flat ground. The foot rolls
            # and yaws, but the lowest point of a rigid point set is always a convex-hull vertex.
            anchors = shoe.lean.anchor_local.numpy()
            if np.ptp(shoe.lean.z_free_rigid.numpy()) != 0.0:
                raise ValueError("the flight screen's hull candidates need flat ground under every column")
            hull = np.unique(ConvexHull(anchors).vertices)
            shoe.lean.flight_columns = wp.array(hull.astype(np.int32), dtype=int, device=d)
        self.leg_contact = None
        if capsules is not None:
            # With the MuJoCo engine the profile has zero stiffness, so this only records penetration.
            self.leg_contact = leg_contact.LegContact(
                capsules, library.calibrations, world_trials, self.model.body_mass.numpy(), d
            )
            # A torch view of the per-world worst penetration, taken once: the reward reads it
            # inside the captured control step, and a fresh wp.to_torch there would build a new
            # wrapper on every step of the capture.
            self.leg_contact_depth = wp.to_torch(self.leg_contact.depth)
        elif cfg.leg_contact_penalty:
            raise ValueError("leg_contact_penalty needs a leg_contact profile to measure")
        b = self.bounds
        self.body_weight_n = float(self.model.body_mass.numpy()[:7].sum()) * 9.81
        # Constants of the per-step torch maths, built once: a tensor from a Python list is
        # copied from pageable memory, which stalls the host behind the queued substeps.
        self._down = torch.tensor((0.0, 0.0, -1.0), device=self.torch_device)
        self._conjugate = torch.tensor((-1.0, -1.0, -1.0, 1.0), device=self.torch_device)
        self._reward_weights = torch.tensor(cfg.reward_weights, device=self.torch_device)[:, None]
        self.params = _params(cfg, b, self.body_weight_n)
        data = Worlds3D()
        for name, dtype in (
            ("status", int),
            ("torque", VecJ),
            ("load", VecJ),
            ("equilibrium", VecJ),
            ("stiffness", VecJ),
            ("damping", VecJ),
            ("phase", float),
            ("sense", wp.vec2),
            ("effort", VecJ),
            ("power_positive", VecJ),
            ("power_negative", VecJ),
            ("substeps", int),
            ("fraction_r", wp.float64),
            ("fraction_l", wp.float64),
            ("invalid_r", int),
            ("invalid_l", int),
        ):
            setattr(data, name, wp.zeros(w, dtype=dtype, device=d))
        data.grf = wp.zeros((w, 2), dtype=wp.vec3, device=d)
        data.enabled_r = self.shoes["right"].foundation.enabled
        data.enabled_l = self.shoes["left"].foundation.enabled
        self.mount_shift_m = _mount_shifts(library, world_trials)
        data.shift_r = wp.array(self.mount_shift_m[:, 0], dtype=wp.vec3, device=d)
        data.shift_l = wp.array(self.mount_shift_m[:, 1], dtype=wp.vec3, device=d)
        self.data = data
        # Per-world shoes. The randomizer writes the two foundations and the per-world
        # mount shift in place, so it must exist before the first substep and before the
        # observation buffers are sized.
        self.shoe_randomizer = None
        if cfg.shoe_dr != "off":
            if cfg.shoe_dr not in shoe_randomization.PRESETS:
                raise ValueError(
                    f"unknown shoe_dr {cfg.shoe_dr!r}; expected one of {sorted(shoe_randomization.PRESETS)}"
                )
            self.shoe_randomizer = shoe_randomization.ShoeRandomizer(
                self,
                shoe_randomization.PRESETS[cfg.shoe_dr],
                seed=cfg.shoe_dr_seed,
                swap_probability=cfg.shoe_swap_probability,
            )
        self.observation_names = tuple(OBSERVATION_NAMES)
        if cfg.contact_observation:
            self.observation_names += shoe_randomization.CONTACT_OBSERVATION_NAMES
        if cfg.theta_observation:
            self.observation_names += shoe_randomization.THETA_OBSERVATION_NAMES
        self.observation_size = len(self.observation_names)
        self.contact_force_n = shoe_randomization.CONTACT_FORCE_N
        self.stance_clock_steps = shoe_randomization.STANCE_CLOCK_STEPS
        # Contact-interface state the actor can see and the adaptation module can read:
        # a touchdown flag and a stance clock per foot, and one step-frequency estimate.
        self.stance = torch.zeros(w, 2, device=self.torch_device)
        self.stance_previous = torch.zeros(w, 2, device=self.torch_device)
        self.stance_clock = torch.zeros(w, 2, device=self.torch_device)
        self.cadence_spm = torch.zeros(w, device=self.torch_device)
        self.touch_step = torch.zeros(w, dtype=torch.long, device=self.torch_device)
        self.cadence_reference = torch.tensor(_segment_cadence(library), dtype=torch.float32, device=self.torch_device)
        hips, knees, ankles = cfg.pose_group_weights
        dof = torch.tensor(cfg.pose_dof_weights, device=self.torch_device, dtype=torch.float32)
        if dof.numel() != 10:
            raise ValueError(f"pose_dof_weights has {dof.numel()} entries, expected 10 ({JOINT_NAMES})")
        self.pose_group = dof.clone()
        self.pose_group[torch.tensor([0, 1, 2, 5, 6, 7], device=self.torch_device)] *= hips
        self.pose_group[torch.tensor([3, 8], device=self.torch_device)] *= knees
        self.pose_group[torch.tensor([4, 9], device=self.torch_device)] *= ankles
        self.pose_stance_joints = torch.tensor(
            [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0], device=self.torch_device
        )
        self.joint_side = torch.tensor((0,) * 5 + (1,) * 5, device=self.torch_device)
        names = [
            "status",
            "torque",
            "equilibrium",
            "stiffness",
            "damping",
            "phase",
            "sense",
            "grf",
            "effort",
            "power_positive",
            "power_negative",
            "substeps",
            "enabled_r",
            "enabled_l",
        ]
        self.t = SimpleNamespace(**{name: wp.to_torch(getattr(data, name)) for name in names})
        dev = self.torch_device
        self.lower = torch.tensor(b.equilibrium_lower_rad * 2, device=dev)
        self.upper = torch.tensor(b.equilibrium_upper_rad * 2, device=dev)
        self.stiffness_max = torch.tensor(b.stiffness_max_nm_rad * 2, device=dev)
        self.damping_max = torch.tensor(b.damping_max_nms_rad * 2, device=dev)
        self.torque_cap = torch.tensor(b.torque_max_nm * 2, device=dev)
        self.jump_unit = torch.tensor(b.torque_rate_max_nm_s * 2, device=dev) * cfg.control_dt_s
        self.jump = torch.zeros(w, len(ACTION_NAMES) // 3, device=dev)
        self.segment = torch.zeros(w, dtype=torch.long, device=dev)
        self.index = torch.zeros(w, dtype=torch.long, device=dev)
        self.origin = torch.zeros(w, 3, device=dev)
        self.steps = torch.zeros(w, dtype=torch.long, device=dev)
        self.speed = torch.zeros(w, device=dev)
        self.pacer_start = torch.zeros(w, 2, device=dev)
        self.offset = torch.zeros(w, 2, device=dev)
        """Horizontal shift [m] that :meth:`recenter` removed from each world's coordinates."""
        self.max_steps = int(round(cfg.episode_s / cfg.control_dt_s))
        self.pelvis_com = torch.tensor(self.model.body_com.numpy()[0], device=dev)
        # Padded table of each trial's airborne start frames, so ``reset`` can draw one
        # candidate per world without a Python loop over trials or a ``nonzero``.
        candidates = library.initial_candidates
        trial_of = library.trial_of[candidates[:, 0]]
        widths = [int((trial_of == trial).sum()) for trial in range(len(library.trials))]
        table = torch.zeros(len(widths), max(widths), dtype=torch.long, device=dev)
        for trial, size in enumerate(widths):
            table[trial, :size] = torch.nonzero(trial_of == trial).squeeze(1)[:size]
        self.initial_table = table
        self.initial_lengths = torch.tensor(widths, dtype=torch.long, device=dev)[self.world_trials]
        self._graph = None
        self.control_steps = 0
        # A captured control step replays into the addresses it recorded, so the action it acts
        # on, the reset draw, and everything the caller reads back are fixed buffers here.
        self._action = torch.zeros(w, self.action_size, device=dev)
        self._pick = torch.zeros(w, dtype=torch.long, device=dev)
        self._observation = torch.zeros(w, self.observation_size, device=dev)
        self._privileged = torch.zeros(w, self.privileged_size, device=dev)
        self._reward = torch.zeros(w, device=dev)
        self._terminated = torch.zeros(w, dtype=torch.bool, device=dev)
        self._truncated = torch.zeros(w, dtype=torch.bool, device=dev)
        # The cadence band and the crossing penalty are reported next to the six weighted
        # terms; each carries its own weight in Env3DConfig instead of a slot in
        # ``reward_weights``, and each is absent from the log when its weight is zero.
        self._terms = {name: torch.zeros((), device=dev) for name in (*REWARD_NAMES, *EXTRA_REWARD_NAMES)}
        self._info = {
            "terms": self._terms,
            "failed": torch.zeros((), device=dev),
            "final_observation": torch.zeros(w, self.observation_size, device=dev),
            "final_privileged": torch.zeros(w, self.privileged_size, device=dev),
        }
        # The smoothed actuator, if configured, takes over the impedance write and the
        # per-substep blend. Attached last: it reads the bounds and the buffers above.
        self.actuator = actuation.attach(self)
        self._pre_graph = self._step_graph = self._advance_graph = None

    # ----- views -----

    @property
    def joint_q(self) -> torch.Tensor:
        return wp.to_torch(self.states[0].joint_q).view(self.worlds, 17)

    @property
    def joint_qd(self) -> torch.Tensor:
        return wp.to_torch(self.states[0].joint_qd).view(self.worlds, 16)

    def body_positions(self) -> torch.Tensor:
        return wp.to_torch(self.states[0].body_q).view(self.worlds, 7, 7)

    # ----- integration -----

    def _substep(self, state_in, state_out):
        if self.actuator is not None:
            # A Warp kernel, not torch: this whole sequence is recorded in the
            # environment's substep graph, which does not capture a torch launch.
            self.actuator.begin_substep()
        d, data, p = self.device, self.data, self.params
        right, left = self.shoes["right"], self.shoes["left"]
        wp.launch(
            _actuate,
            dim=self.worlds,
            inputs=[p, data, state_in.joint_q, state_in.joint_qd, self.control.joint_f],
            device=d,
        )
        wp.launch(
            _stage,
            dim=self.worlds,
            inputs=[
                data,
                state_in.body_q,
                state_in.body_qd,
                self.model.body_com,
                right.carriers.body_q,
                right.carriers.body_qd,
                left.carriers.body_q,
                left.carriers.body_qd,
            ],
            device=d,
        )
        from ..hogan.gpu_shoe import apply_ground_shoe  # noqa: PLC0415

        apply_ground_shoe(right.foundation, right.lean, right.carriers, data.fraction_r, data.invalid_r)
        apply_ground_shoe(left.foundation, left.lean, left.carriers, data.fraction_l, data.invalid_l)
        state_in.body_f.zero_()
        wp.launch(
            _contact,
            dim=self.worlds,
            inputs=[
                p,
                data,
                state_in.body_q,
                self.model.body_com,
                state_in.body_f,
                right.carriers.body_f,
                left.carriers.body_f,
                state_in.joint_qd,
            ],
            device=d,
        )
        if self.leg_contact is not None:
            self.leg_contact.apply(data.status, state_in.body_q, state_in.body_qd, self.model.body_com, state_in.body_f)
        self.solver.step(state_in, state_out, self.control, None, self.config.dt_s)
        wp.launch(_screen, dim=self.worlds, inputs=[p, data, state_out.joint_q, state_out.joint_qd], device=d)

    def _substeps(self, count: int):
        for k in range(count):
            self._substep(self.states[k % 2], self.states[(k + 1) % 2])

    def _clear_interval(self):
        """Empty the accumulators :meth:`integrate` sums one control step of forces into."""
        for name in ("grf", "effort", "power_positive", "power_negative"):
            getattr(self.t, name).zero_()
        self.t.substeps.zero_()
        if self.leg_contact is not None:
            self.leg_contact.clear()

    def integrate(self):
        """Advance every running world by one control step (an even number of substeps)."""
        self._clear_interval()
        substeps = self.config.substeps
        if substeps % 2:
            raise ValueError("The 3D environment alternates two states and needs an even substep count")
        if self._graph is None:
            self._substeps(2)
            wp.synchronize_device(self.device)
            with wp.ScopedCapture(device=self.device) as capture:
                self._substeps(substeps)
            self._graph = capture.graph
            self._substeps(substeps - 2)
            return
        wp.capture_launch(self._graph)

    def set_impedance(self, action: torch.Tensor):
        fraction = torch.sigmoid(action.float().clamp(-40.0, 40.0))
        self.t.equilibrium.copy_(self.lower + (self.upper - self.lower) * fraction[:, :10])
        self.t.stiffness.copy_(self.stiffness_max * fraction[:, 10:20])
        self.t.damping.copy_(self.damping_max * fraction[:, 20:])

    # ----- observations and rewards -----

    def observe(self) -> torch.Tensor:
        q, qd = self.joint_q, self.joint_qd
        rotation = _quat_to_matrix(q[:, 3:7])
        heading = _heading(rotation)
        gravity = torch.einsum("nji,j->ni", rotation, self._down)
        omega = qd[:, 3:6]
        com_velocity = qd[:, :3]
        local_velocity = torch.einsum("nji,nj->ni", heading, com_velocity)
        local_velocity[:, 0] -= self.speed
        bodies = self.body_positions()
        ankles = torch.stack(
            [torch.einsum("nji,nj->ni", heading, bodies[:, b, :3] - q[:, :3]) for b in (3, 6)], dim=1
        ).reshape(-1, 6)
        pacer = torch.einsum("nji,nj->ni", heading[:, :2, :2], self.pacer() - q[:, :2])
        return torch.cat(
            (
                q[:, 2:3],
                gravity,
                local_velocity,
                torch.einsum("nji,nj->ni", rotation, omega),
                q[:, 7:],
                qd[:, 6:] / 10.0,
                self.t.sense,
                self.t.torque / self.torque_cap,
                ankles,
                torch.sin(self.t.phase)[:, None],
                torch.cos(self.t.phase)[:, None],
                ((self.speed - self.config.reference_speed_m_s) / self.config.speed_scale_m_s)[:, None],
                pacer,
                heading[:, 1, 0:1],
                heading[:, 0, 0:1],
                self._contact_channels(),
                self._theta_channels(),
            ),
            dim=1,
        )

    def _theta_channels(self) -> torch.Tensor:
        """The privileged shoe parameters, or an empty block when the actor is not privileged."""
        if not self.config.theta_observation or self.shoe_randomizer is None:
            return torch.zeros(self.worlds, 0, device=self.torch_device)
        return self.shoe_randomizer.privileged()

    def _contact_channels(self) -> torch.Tensor:
        """The per-foot foam channels of :data:`.shoe_randomization.CONTACT_OBSERVATION_NAMES`.

        Every channel is a contact-interface measurement the simulator already makes: the
        screen's peak driven-column travel, the interval-mean wrench resolved into normal
        and tangential parts, the ratio the friction cone bounds, and the two clocks. A
        policy cannot infer a novel shoe from body state alone - a soft foam and a soft
        actuator look the same to the pelvis - so these are the channels the adaptation
        module is built on. A world that has just restarted reports nothing: its
        accumulators were cleared with its history.
        """
        if not self.config.contact_observation:
            return torch.zeros(self.worlds, 0, device=self.torch_device)
        force = self._interval_force()
        normal = torch.stack((force[:, 2], force[:, 5]), dim=1).clamp(min=0.0)
        tangential = torch.stack(
            (torch.linalg.norm(force[:, 0:2], dim=1), torch.linalg.norm(force[:, 3:5], dim=1)), dim=1
        )
        cone = tangential / (normal + 0.05 * self.body_weight_n)
        travel = torch.stack((wp.to_torch(self.data.fraction_r), wp.to_torch(self.data.fraction_l)), dim=1).float()
        return torch.cat(
            (
                travel,
                normal / self.body_weight_n,
                tangential / self.body_weight_n,
                cone.clamp(max=2.0),
                self.stance,
                self.stance_clock / float(self.stance_clock_steps),
            ),
            dim=1,
        )

    def pacer(self) -> torch.Tensor:
        """Horizontal pacer position [m]: the start position moving along +x at the task speed."""
        travel = self.speed * self.steps.float() * self.config.control_dt_s
        return self.pacer_start + torch.stack((travel, torch.zeros_like(travel)), dim=1)

    def _reference(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        k = self.library.starts[self.segment] + self.index
        q_ref = self.library.q[k].clone()
        q_ref[:, :3] -= self.origin
        return q_ref, self.library.qd[k], self.library.grf[k]

    def _interval_force(self) -> torch.Tensor:
        return (self.t.grf / self.t.substeps.clamp(min=1)[:, None, None]).reshape(-1, 6)

    def _orientation_error(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Rotation vector of ``a^-1 b`` for xyzw quaternions [rad]."""
        conj = a * self._conjugate
        r = _quat_multiply(conj, b)
        r = torch.where(r[:, 3:4] < 0, -r, r)
        angle = 2 * torch.atan2(r[:, :3].norm(dim=1), r[:, 3].clamp(min=1e-9))
        axis = r[:, :3] / r[:, :3].norm(dim=1, keepdim=True).clamp(min=1e-9)
        return axis * angle[:, None]

    def privileged(self) -> torch.Tensor:
        q_ref, qd_ref, grf_ref = self._reference()
        q, qd = self.joint_q, self.joint_qd
        weight = self.body_weight_n
        remaining = (self.max_steps - self.steps).float() * self.config.control_dt_s
        return torch.cat(
            (
                q_ref[:, 7:] - q[:, 7:],
                self.pacer() - q[:, :2],
                q_ref[:, 2:3] - q[:, 2:3],
                self._orientation_error(q[:, 3:7], q_ref[:, 3:7]),
                (qd_ref[:, 7:] - qd[:, 6:]) / 10.0,
                grf_ref / weight,
                self._interval_force() / weight,
                remaining.clamp(max=2.0)[:, None],
                (self.steps.float() / self.max_steps)[:, None],
            ),
            dim=1,
        )

    def reward(self) -> tuple[torch.Tensor, dict]:
        cfg = self.config
        q_ref, qd_ref, grf_ref = self._reference()
        q, qd = self.joint_q, self.joint_qd
        orientation = self._orientation_error(q[:, 3:7], q_ref[:, 3:7]).norm(dim=1)
        # Tracking is asymmetric on purpose (design document section 7.3): pelvis height,
        # belt speed and cadence stay strict, because those are what a run has to keep,
        # while a stance knee and ankle are asked to hold the measured angle only while
        # that foot is swinging. Under a novel shoe the stance solution changes, and a
        # symmetric pose term punishes exactly the adaptation the world model predicts.
        weight = self.pose_group[None, :] * (
            1.0 - cfg.stance_pose_relief * self.pose_stance_joints[None, :] * self.stance[:, self.joint_side]
        )
        pose = (
            (torch.sum(weight * (q[:, 7:] - q_ref[:, 7:]) ** 2, dim=1) + orientation**2)
            / (1.0 + weight.sum(dim=1))
            / cfg.pose_scale_rad**2
        )
        velocity = torch.mean((qd[:, 6:] - qd_ref[:, 7:]) ** 2, dim=1) / cfg.velocity_scale_rad_s**2
        position = ((q[:, 2] - q_ref[:, 2]) / cfg.position_scale_m) ** 2 + torch.sum(
            (q[:, :2] - self.pacer()) ** 2, dim=1
        ) / cfg.pacer_scale_m**2
        force = (
            torch.mean((self._interval_force() - grf_ref) ** 2, dim=1) / (cfg.force_scale_bw * self.body_weight_n) ** 2
        )
        effort = torch.mean(self.t.effort / cfg.substeps / self.torque_cap**2, dim=1) / cfg.effort_scale
        smoothness = torch.mean(self.jump**2, dim=1) / cfg.smoothness_scale**2
        terms = torch.exp(-torch.stack((pose, velocity, position, force, effort, smoothness)))
        weights = self._reward_weights
        reward = (weights * terms).sum(0)
        logged = {name: terms[i].mean() for i, name in enumerate(REWARD_NAMES)}
        if cfg.cadence_weight:
            # A band, not a target: inside ``cadence_band_spm`` of the measured cadence of
            # this segment the term is flat, so the runner is free to reorganise its stride
            # as the shoe asks, and only a gait that stops looking like running is punished.
            drift = (self.cadence_spm - self.cadence_reference[self.segment]).abs() - cfg.cadence_band_spm
            cadence = torch.exp(-((drift.clamp(min=0.0) / cfg.cadence_band_spm) ** 2))
            reward = reward + cfg.cadence_weight * cadence
            logged["cadence"] = cadence.mean()
        if cfg.leg_contact_penalty:
            # Worst penetration of the control step, saturated, so the gradient says "stop
            # crossing" rather than "run slower and cross less deeply".
            crossing = (self.leg_contact_depth / cfg.leg_contact_penalty_depth_m).clamp(max=1.0)
            reward = reward - cfg.leg_contact_penalty * crossing
            logged["crossing"] = crossing.mean()
        return reward, logged

    def _departed(self) -> torch.Tensor:
        cfg = self.config
        q_ref, _, _ = self._reference()
        q = self.joint_q
        orientation = self._orientation_error(q[:, 3:7], q_ref[:, 3:7]).norm(dim=1)
        return (
            ((q[:, 2] - q_ref[:, 2]).abs() > cfg.termination_height_m)
            | (orientation > cfg.termination_orientation_rad)
            | ((q[:, :2] - self.pacer()).norm(dim=1) > cfg.termination_horizontal_m)
        )

    # ----- resets -----

    def _root_velocity(self, q: torch.Tensor, qd_ref: torch.Tensor) -> torch.Tensor:
        """Free-joint velocity (COM linear, world angular) from reference coordinate rates."""
        quat = q[:, 3:7]
        rate = qd_ref[:, 3:7]
        conj = quat * self._conjugate
        omega = 2 * _quat_multiply(rate, conj)[:, :3]
        offset = torch.einsum("nij,j->ni", _quat_to_matrix(quat), self.pelvis_com)
        linear = qd_ref[:, :3] + torch.cross(omega, offset, dim=1)
        return torch.cat((linear, omega, qd_ref[:, 7:]), dim=1)

    def write_state(self, mask: torch.Tensor, q: torch.Tensor, qd: torch.Tensor, phase: torch.Tensor):
        # Blending replaces boolean indexing: every ``view[mask] = ...`` is a ``nonzero``
        # plus a host round trip, and ``reset`` issued about forty of them.
        where = mask[:, None]
        for state in self.states:
            positions = wp.to_torch(state.joint_q).view(self.worlds, 17)
            rates = wp.to_torch(state.joint_qd).view(self.worlds, 16)
            positions.copy_(torch.where(where, q, positions))
            rates.copy_(torch.where(where, qd, rates))
        self.pacer_start.copy_(torch.where(where, q[:, :2], self.pacer_start))
        self.offset.masked_fill_(where, 0.0)
        self.steps.copy_(torch.where(mask, torch.zeros_like(self.steps), self.steps))
        t = self.t
        t.phase.copy_(torch.where(mask, phase, t.phase))
        t.torque.masked_fill_(where, 0.0)
        t.sense.masked_fill_(where, 0.0)
        t.status.masked_fill_(mask, 0)
        t.enabled_r.masked_fill_(mask, 1)
        t.enabled_l.masked_fill_(mask, 1)
        for shoe in self.shoes.values():
            for view in shoe.history:
                view.masked_fill_(mask.reshape((self.worlds,) + (1,) * (view.dim() - 1)), 0.0)
        # A restarted world has no contact history, so neither should the channels and the
        # interval accumulators that feed them and the critic: without this the first
        # observation and the first ``privileged`` of a new episode still carry the force
        # sums of the step that ended the last one.
        for view in (self.stance, self.stance_previous, self.stance_clock):
            view.masked_fill_(where, 0.0)
        self.cadence_spm.copy_(torch.where(mask, self.cadence_reference[self.segment], self.cadence_spm))
        self.touch_step.copy_(torch.where(mask, self.steps, self.touch_step))
        if self.actuator is not None:
            # Held actuator state belongs to the episode that ended: with a filter time
            # constant or a rate bound, a reset world would otherwise start seconds away
            # from its own command.
            self.actuator.snap(mask)
        self.t.grf.mul_(~mask[:, None, None])
        self.t.effort.mul_(~mask[:, None])
        for name in ("power_positive", "power_negative"):
            getattr(self.t, name).mul_(~mask[:, None])
        self.t.substeps.mul_(~mask)
        for side in SIDES:
            wp.to_torch(getattr(self.data, f"fraction_{side[0]}")).masked_fill_(mask, 0.0)
        # Both states hold the same coordinates after the write, so forward kinematics
        # runs once and the body arrays are copied.
        newton.eval_fk(self.model, self.states[0].joint_q, self.states[0].joint_qd, self.states[0])
        wp.copy(self.states[1].body_q, self.states[0].body_q)
        wp.copy(self.states[1].body_qd, self.states[0].body_qd)

    @staticmethod
    def kinematic_phase(q: torch.Tensor, qd: torch.Tensor) -> torch.Tensor:
        """Phase convention of the planar model from right hip flexion angle and rate."""
        return torch.remainder(torch.atan2(-qd[:, 7] / 8.0, q[:, 7]), 2 * math.pi)

    def reset(self, mask: torch.Tensor | None = None):
        self._draw()
        self._reset(mask)

    def _draw(self):
        """Pick one airborne start frame per world into the buffer the step reads."""
        # Each world keeps its trial geometry, so it draws only from that trial's pool;
        # a uniform draw modulo the pool width is uniform over the pool.
        torch.randint(0, 1 << 30, (self.worlds,), generator=self.generator, out=self._pick)

    def _reset(self, mask: torch.Tensor | None = None):
        if mask is None:
            mask = torch.ones(self.worlds, dtype=torch.bool, device=self.torch_device)
        lib = self.library
        rows = self.initial_table[self.world_trials, self._pick % self.initial_lengths]
        segment, index = lib.initial_candidates[rows].unbind(1)
        k = lib.starts[segment] + index
        q = lib.q[k].clone()
        origin = q[:, :3].clone()
        origin[:, 2] = 0.0
        q[:, :2] = 0.0
        where = mask[:, None]
        self.segment.copy_(torch.where(mask, segment, self.segment))
        self.index.copy_(torch.where(mask, index, self.index))
        self.origin.copy_(torch.where(where, origin, self.origin))
        self.speed.copy_(torch.where(mask, lib.speeds[segment], self.speed))
        self.write_state(mask, q, self._root_velocity(q, lib.qd[k]), self.kinematic_phase(lib.q[k], lib.qd[k]))

    def _command(self) -> torch.Tensor:
        """Joint torque the current impedance commands at the current state [N m]."""
        q, qd, t = self.joint_q[:, 7:], self.joint_qd[:, 6:], self.t
        torque = t.stiffness * (t.equilibrium - q) - t.damping * qd
        return torch.maximum(torch.minimum(torque, self.torque_cap), -self.torque_cap)

    def position(self) -> torch.Tensor:
        """Horizontal pelvis position [m] in the frame the world started in."""
        return self.joint_q[:, :2] + self.offset

    def recenter(self):
        """Move every pelvis back over the origin in x and y.

        SolverFeatherstone works in world coordinates in float32, so far from the origin
        rounding swamps the limb inertias: started 100 m out, every world of a policy that
        runs fine at the origin turned non-finite within 10 s. The ground is flat, so
        shifting the root, the shoes' world-frame friction anchors and contact points, the
        pacer, and the reference origin together changes nothing physical.
        """
        shift = self.joint_q[:, :2].clone()
        for state in self.states:
            wp.to_torch(state.joint_q).view(self.worlds, 17)[:, :2] -= shift
        for shoe in self.shoes.values():
            foundation = shoe.foundation
            columns = foundation.column_count
            anchors = [foundation.tangent_anchor, foundation.friction_solver.scratch_anchor]
            for array in anchors:
                wp.to_torch(array).view(self.worlds, columns, 2).sub_(shift[:, None, :])
            wp.to_torch(foundation.contact_point).view(self.worlds, columns, 3)[:, :, :2] -= shift[:, None, :]
        self.offset += shift
        self.pacer_start -= shift
        self.origin[:, :2] += shift
        newton.eval_fk(self.model, self.states[0].joint_q, self.states[0].joint_qd, self.states[0])
        wp.copy(self.states[1].body_q, self.states[0].body_q)
        wp.copy(self.states[1].body_qd, self.states[0].body_qd)

    def _stage(self, action: torch.Tensor):
        """Copy ``action`` into the buffer the recorded graphs read."""
        self._action.copy_(action)

    def _pre(self):
        """The torch work between an action and the substeps: the impedance command and the shift."""
        before = self._command()
        if self.actuator is None:
            self.set_impedance(self._action)
        else:
            self.actuator.apply(self._action)
        jump = (self._command() - before) / self.jump_unit
        self.jump.copy_(torch.where((self.steps > 0)[:, None], jump, torch.zeros_like(jump)))
        self.recenter()

    def _post(self):
        """The torch work after the substeps: walk the reference and the episode counters forward."""
        self.index += 1
        self.steps += 1
        ended = self.index >= self.library.lengths[self.segment]
        following = self.library.successor[self.segment]
        self.segment.copy_(torch.where(ended, following[:, 0], self.segment))
        self.index.copy_(torch.where(ended, following[:, 1], self.index))
        self._sense_contact()

    def _sense_contact(self):
        """Touchdown flags, a stance clock per foot, and one step-frequency estimate.

        Read off the interval-mean foot force of the control step just integrated, with
        the same 50 N threshold the gait scorer and the measured-contact penalty use, so
        the policy, the reward, and the report agree on what stance means. Cadence is the
        quantity the design keeps strict while the joint angles relax, and it is the one
        gait variable a shoe change moves only through the runner's own response.
        """
        force = self._interval_force()
        normal = torch.stack((force[:, 2], force[:, 5]), dim=1)
        stance = (normal > self.contact_force_n).float()
        touched = stance * (1.0 - self.stance_previous)
        touched_any = touched.sum(dim=1) > 0
        self.stance_clock.copy_(
            torch.where(
                touched > 0.5,
                torch.ones_like(self.stance_clock),
                self.stance_clock + 1.0,
            ).clamp(max=float(self.stance_clock_steps))
        )
        self.stance.copy_(stance)
        self.stance_previous.copy_(stance)
        interval = (self.steps - self.touch_step).clamp(min=1).float() * self.config.control_dt_s
        self.cadence_spm.copy_(
            torch.where(touched_any, 0.75 * self.cadence_spm + 0.25 * (60.0 / interval), self.cadence_spm)
        )
        self.touch_step.copy_(torch.where(touched_any, self.steps, self.touch_step))

    def _swap_shoes(self):
        """Give the worlds that resample this control step their new shoe.

        Runs outside the recorded graphs, and counts control steps on the host: reading
        :attr:`steps` back would put a synchronisation in the middle of the loop that
        exists to avoid them.
        """
        self.control_steps += 1
        if self.shoe_randomizer is not None:
            self.shoe_randomizer.advance(self.control_steps)

    def advance(self, action: torch.Tensor):
        """Apply one action for one control step and advance the reference and the pacer.

        The impedance law runs every substep, so the torque follows the state within a
        control step; a new action changes it stepwise only at the boundary. That step,
        relative to the torque rate bound over one control step, is kept in :attr:`jump`.
        """
        self._stage(action)
        if self._pre_graph is None:
            self._pre()
            self.integrate()
            self._post()
        else:
            self._pre_graph.replay()
            self.integrate()
            self._advance_graph.replay()
        self._swap_shoes()

    def _step_tail(self) -> tuple:
        """The work after the substeps that :meth:`step` adds: reward, end of episode, next reset."""
        self._post()
        reward, terms = self.reward()
        failed = self.t.status != 0
        terminated = failed | self._departed()
        truncated = ~terminated & (self.steps >= self.max_steps)
        info = self._info
        info["final_observation"].copy_(self.observe())
        info["final_privileged"].copy_(self.privileged())
        info["failed"].copy_(failed.float().mean())
        for name, value in terms.items():
            self._terms[name].copy_(value)
        self._reward.copy_(reward)
        self._terminated.copy_(terminated)
        self._truncated.copy_(truncated)
        self._reset(terminated | truncated)
        self._observation.copy_(self.observe())
        self._privileged.copy_(self.privileged())

    def step(self, action: torch.Tensor):
        """Reward the step just taken, reset the worlds that ended, and return the next observation.

        The returned tensors are the environment's own buffers, valid until the next call.
        """
        self._stage(action)
        self._draw()
        if self._pre_graph is None:
            self._pre()
            self.integrate()
            self._step_tail()
        else:
            self._pre_graph.replay()
            self.integrate()
            self._step_graph.replay()
        self._swap_shoes()
        return (self._observation, self._privileged, self._reward, self._terminated, self._truncated, self._info)

    def capture(self, warmup: int = 3):
        """Record the torch work of one control step into CUDA graphs around the substeps.

        :meth:`integrate` keeps its own Warp graph, replayed between the two torch graphs: a torch
        capture drops a Warp graph launched inside it, and drops any Warp kernel that runs on a
        stream other than the one being captured, so recording happens on :attr:`_stream`. Recording
        needs live warm-up steps, so this resets the worlds, runs ``warmup`` of them, and puts both
        RNG streams back; call it before the worlds are set up. Afterwards every step writes the
        same buffers, and a step returns views of them, valid until the next call.
        """
        if self._pre_graph is not None:
            return self
        # Neither RNG stream may move: the warm-up steps draw reset candidates from the
        # environment's generator, and every torch capture seeds a graph's philox state from the
        # default generator, so both are put back once the graphs are recorded.
        default = torch.cuda.default_generators[torch.device(self.torch_device).index]
        draw, default_draw = self.generator.get_state(), default.get_state()
        self.reset()
        for _ in range(warmup):
            self.step(self._action)
        pre, stepped, advanced = (torch.cuda.CUDAGraph() for _ in range(3))
        with torch.cuda.graph(pre, stream=self._stream):
            self._pre()
        with torch.cuda.graph(stepped, stream=self._stream):
            self._step_tail()
        with torch.cuda.graph(advanced, stream=self._stream):
            self._post()
        self._pre_graph, self._step_graph, self._advance_graph = pre, stepped, advanced
        # The warm-up's last control step left its force sums in the interval accumulators, which a
        # world that has not been integrated yet reports as zero.
        self._clear_interval()
        self.generator.set_state(draw)
        default.set_state(default_draw)
        return self

    def describe(self) -> dict:
        return {
            "config": asdict(self.config),
            "bounds": asdict(self.bounds),
            "shoes": self.shoe_metadata,
            "observations": list(OBSERVATION_NAMES),
            "actions": list(ACTION_NAMES),
            "worlds": self.worlds,
            "trials": self.library.trials,
            "solver": type(self.solver).__name__,
            "leg_contact": "off" if self.leg_contact is None else self.leg_contact.describe(),
            "actuation": None if self.actuator is None else self.actuator.describe(),
            "shoe_randomization": None if self.shoe_randomizer is None else self.shoe_randomizer.describe(),
            "shoe_swaps": None if self.shoe_randomizer is None else self.shoe_randomizer.swaps,
        }
