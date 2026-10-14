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

import newton

from ..cartesian.shoe import Shoe
from . import data3d, model3d
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
)
ACTION_NAMES = tuple(f"{output}_{name}" for output in ("equilibrium", "stiffness", "damping") for name in JOINT_NAMES)


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
    mass_matrix_interval: int = 2
    """Featherstone mass-matrix refresh interval [substeps]; the matrix changes little over 0.25 ms."""
    episode_s: float = 2.0
    termination_height_m: float = 0.15
    termination_orientation_rad: float = 0.6
    termination_horizontal_m: float = 0.35
    reward_weights: tuple[float, float, float, float, float] = (0.35, 0.1, 0.2, 0.25, 0.1)
    pose_scale_rad: float = 0.15
    velocity_scale_rad_s: float = 3.0
    position_scale_m: float = 0.05
    force_scale_bw: float = 0.3
    effort_scale: float = 0.15

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
    body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector], body_com: wp.array[wp.vec3], body: int
):
    pose = body_q[body]
    twist = body_qd[body]
    omega = wp.spatial_bottom(twist)
    offset = wp.transform_vector(pose, body_com[body])
    linear = wp.spatial_top(twist) - wp.cross(omega, offset)
    return pose, wp.spatial_vector(linear, omega)


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
    pose, twist = _carrier(body_q, body_qd, body_com, w * 7 + 3)
    carrier_q_r[w] = pose
    carrier_qd_r[w] = twist
    pose, twist = _carrier(body_q, body_qd, body_com, w * 7 + 6)
    carrier_q_l[w] = pose
    carrier_qd_l[w] = twist


@wp.func
def _apply(
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_f: wp.array[wp.spatial_vector],
    body: int,
    wrench: wp.spatial_vector,
):
    force = wp.spatial_top(wrench)
    offset = wp.transform_vector(body_q[body], body_com[body])
    body_f[body] = wp.spatial_vector(force, wp.spatial_bottom(wrench) - wp.cross(offset, force))


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
    _apply(body_q, body_com, body_f, w * 7 + 3, fr)
    _apply(body_q, body_com, body_f, w * 7 + 6, fl)
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


class ReferenceLibrary3D:
    """Measured 3D segments resampled to the control clock on the GPU."""

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
        self.trial_of = torch.tensor(trial_of, dtype=torch.long, device=device)
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
        torch.cuda.set_stream(wp.stream_to_torch(self.device.stream))
        self.generator = torch.Generator(device=self.torch_device)
        self.generator.manual_seed(seed)
        w, d = self.worlds, self.device
        if world_trials is None:
            world_trials = np.arange(w) % len(library.trials)
        self.world_trials = torch.tensor(world_trials, dtype=torch.long, device=self.torch_device)
        subject = float(library.manifest["subject_mass_kg"])
        self.model = model3d.build_worlds(library.calibrations, world_trials, library.profile, subject, library.hat, d)
        self.solver = newton.solvers.SolverFeatherstone(
            self.model, angular_damping=0.0, update_mass_matrix_interval=cfg.mass_matrix_interval
        )
        self.states = [self.model.state(), self.model.state()]
        self.control = self.model.control()
        mount, pitch = library.manifest["shoe"]["mount_m"], library.manifest["shoe"]["pitch_rad"]
        shoes = {side: Shoe(library.asset(f"digital_shoe_{side}.json"), mount, pitch) for side in SIDES}
        self.shoe_metadata = {side: shoes[side].metadata for side in SIDES}
        self.shoes = {side: _fused_shoe(shoes[side], w, cfg.dt_s, d, cfg.compression_limit) for side in SIDES}
        for shoe in self.shoes.values():
            # The 3D foot rolls and yaws, so every column is a flight candidate.
            shoe.lean.flight_columns = wp.array(np.arange(shoe.foundation.column_count), dtype=int, device=d)
        b = self.bounds
        self.body_weight_n = float(self.model.body_mass.numpy()[:7].sum()) * 9.81
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
        self.data = data
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
        self.segment = torch.zeros(w, dtype=torch.long, device=dev)
        self.index = torch.zeros(w, dtype=torch.long, device=dev)
        self.origin = torch.zeros(w, 3, device=dev)
        self.steps = torch.zeros(w, dtype=torch.long, device=dev)
        self.speed = torch.zeros(w, device=dev)
        self.max_steps = int(round(cfg.episode_s / cfg.control_dt_s))
        self.pelvis_com = torch.tensor(self.model.body_com.numpy()[0], device=dev)
        self._graph = None

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
        self.solver.step(state_in, state_out, self.control, None, self.config.dt_s)
        wp.launch(_screen, dim=self.worlds, inputs=[p, data, state_out.joint_q, state_out.joint_qd], device=d)

    def _substeps(self, count: int):
        for k in range(count):
            self._substep(self.states[k % 2], self.states[(k + 1) % 2])

    def integrate(self):
        """Advance every running world by one control step (an even number of substeps)."""
        for name in ("grf", "effort", "power_positive", "power_negative"):
            getattr(self.t, name).zero_()
        self.t.substeps.zero_()
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
        gravity = torch.einsum("nji,j->ni", rotation, torch.tensor([0.0, 0.0, -1.0], device=q.device))
        omega = qd[:, 3:6]
        com_velocity = qd[:, :3]
        local_velocity = torch.einsum("nji,nj->ni", heading, com_velocity)
        local_velocity[:, 0] -= self.speed
        bodies = self.body_positions()
        ankles = torch.stack(
            [torch.einsum("nji,nj->ni", heading, bodies[:, b, :3] - q[:, :3]) for b in (3, 6)], dim=1
        ).reshape(-1, 6)
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
            ),
            dim=1,
        )

    def _reference(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        k = self.library.starts[self.segment] + self.index
        q_ref = self.library.q[k].clone()
        q_ref[:, :3] -= self.origin
        return q_ref, self.library.qd[k], self.library.grf[k]

    def _interval_force(self) -> torch.Tensor:
        return (self.t.grf / self.t.substeps.clamp(min=1)[:, None, None]).reshape(-1, 6)

    @staticmethod
    def _orientation_error(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Rotation vector of ``a^-1 b`` for xyzw quaternions [rad]."""
        conj = a * torch.tensor([-1.0, -1.0, -1.0, 1.0], device=a.device)
        r = _quat_multiply(conj, b)
        r = torch.where(r[:, 3:4] < 0, -r, r)
        angle = 2 * torch.atan2(r[:, :3].norm(dim=1), r[:, 3].clamp(min=1e-9))
        axis = r[:, :3] / r[:, :3].norm(dim=1, keepdim=True).clamp(min=1e-9)
        return axis * angle[:, None]

    def privileged(self) -> torch.Tensor:
        q_ref, qd_ref, grf_ref = self._reference()
        q, qd = self.joint_q, self.joint_qd
        weight = self.body_weight_n
        remaining = (self.library.lengths[self.segment] - self.index).float() * self.config.control_dt_s
        return torch.cat(
            (
                q_ref[:, 7:] - q[:, 7:],
                q_ref[:, :3] - q[:, :3],
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
        pose = (torch.sum((q[:, 7:] - q_ref[:, 7:]) ** 2, dim=1) + orientation**2) / 11 / cfg.pose_scale_rad**2
        velocity = torch.mean((qd[:, 6:] - qd_ref[:, 7:]) ** 2, dim=1) / cfg.velocity_scale_rad_s**2
        position = torch.sum((q[:, :3] - q_ref[:, :3]) ** 2, dim=1) / cfg.position_scale_m**2
        force = (
            torch.mean((self._interval_force() - grf_ref) ** 2, dim=1) / (cfg.force_scale_bw * self.body_weight_n) ** 2
        )
        effort = torch.mean(self.t.effort / cfg.substeps / self.torque_cap**2, dim=1) / cfg.effort_scale
        terms = torch.stack(
            (torch.exp(-pose), torch.exp(-velocity), torch.exp(-position), torch.exp(-force), torch.exp(-effort))
        )
        weights = torch.tensor(cfg.reward_weights, device=q.device)[:, None]
        names = ("pose", "velocity", "position", "force", "effort")
        return (weights * terms).sum(0), {name: terms[i].mean().item() for i, name in enumerate(names)}

    def _departed(self) -> torch.Tensor:
        cfg = self.config
        q_ref, _, _ = self._reference()
        q = self.joint_q
        orientation = self._orientation_error(q[:, 3:7], q_ref[:, 3:7]).norm(dim=1)
        return (
            ((q[:, 2] - q_ref[:, 2]).abs() > cfg.termination_height_m)
            | (orientation > cfg.termination_orientation_rad)
            | ((q[:, :2] - q_ref[:, :2]).norm(dim=1) > cfg.termination_horizontal_m)
        )

    # ----- resets -----

    def _root_velocity(self, q: torch.Tensor, qd_ref: torch.Tensor) -> torch.Tensor:
        """Free-joint velocity (COM linear, world angular) from reference coordinate rates."""
        quat = q[:, 3:7]
        rate = qd_ref[:, 3:7]
        conj = quat * torch.tensor([-1.0, -1.0, -1.0, 1.0], device=q.device)
        omega = 2 * _quat_multiply(rate, conj)[:, :3]
        offset = torch.einsum("nij,j->ni", _quat_to_matrix(quat), self.pelvis_com)
        linear = qd_ref[:, :3] + torch.cross(omega, offset, dim=1)
        return torch.cat((linear, omega, qd_ref[:, 7:]), dim=1)

    def write_state(self, mask: torch.Tensor, q: torch.Tensor, qd: torch.Tensor, phase: torch.Tensor):
        for state in self.states:
            wp.to_torch(state.joint_q).view(self.worlds, 17)[mask] = q
            wp.to_torch(state.joint_qd).view(self.worlds, 16)[mask] = qd
        t = self.t
        t.phase[mask] = phase
        t.torque[mask] = 0.0
        t.sense[mask] = 0.0
        t.status[mask] = 0
        t.enabled_r[mask] = 1
        t.enabled_l[mask] = 1
        for shoe in self.shoes.values():
            for view in shoe.history:
                view[mask] = 0
        for state in self.states:
            newton.eval_fk(self.model, state.joint_q, state.joint_qd, state)

    @staticmethod
    def kinematic_phase(q: torch.Tensor, qd: torch.Tensor) -> torch.Tensor:
        """Phase convention of the planar model from right hip flexion angle and rate."""
        return torch.remainder(torch.atan2(-qd[:, 7] / 8.0, q[:, 7]), 2 * math.pi)

    def reset(self, mask: torch.Tensor | None = None):
        if mask is None:
            mask = torch.ones(self.worlds, dtype=torch.bool, device=self.torch_device)
        count = int(mask.sum())
        if count == 0:
            return
        lib = self.library
        trials = self.world_trials[mask]
        candidates = lib.initial_candidates
        # Each world keeps its trial geometry, so it starts only from that trial's segments.
        segment_trial = lib.trial_of[candidates[:, 0]]
        choice = torch.empty(count, dtype=torch.long, device=self.torch_device)
        for trial in trials.unique():
            rows = torch.nonzero(trials == trial).squeeze(1)
            pool = torch.nonzero(segment_trial == trial).squeeze(1)
            pick = torch.randint(len(pool), (len(rows),), generator=self.generator, device=self.torch_device)
            choice[rows] = pool[pick]
        segment, index = candidates[choice].unbind(1)
        k = lib.starts[segment] + index
        q = lib.q[k].clone()
        origin = q[:, :3].clone()
        origin[:, 2] = 0.0
        q[:, :3] -= origin
        qd = self._root_velocity(q, lib.qd[k])
        self.segment[mask] = segment
        self.index[mask] = index
        self.origin[mask] = origin
        self.steps[mask] = 0
        self.speed[mask] = lib.speeds[segment]
        self.write_state(mask, q, qd, self.kinematic_phase(lib.q[k], lib.qd[k]))

    def step(self, action: torch.Tensor):
        self.set_impedance(action)
        self.integrate()
        self.index += 1
        self.steps += 1
        reward, terms = self.reward()
        failed = self.t.status != 0
        terminated = failed | self._departed()
        truncated = ~terminated & (
            (self.steps >= self.max_steps) | (self.index >= self.library.lengths[self.segment] - 1)
        )
        info = {
            "terms": terms,
            "failed": failed.float().mean().item(),
            "final_observation": self.observe(),
            "final_privileged": self.privileged(),
        }
        self.reset(terminated | truncated)
        return self.observe(), self.privileged(), reward, terminated, truncated, info

    def describe(self) -> dict:
        return {
            "config": asdict(self.config),
            "bounds": asdict(self.bounds),
            "shoes": self.shoe_metadata,
            "observations": list(OBSERVATION_NAMES),
            "actions": list(ACTION_NAMES),
            "worlds": self.worlds,
            "trials": self.library.trials,
            "solver": "SolverFeatherstone",
        }
