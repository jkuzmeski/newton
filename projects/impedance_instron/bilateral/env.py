# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Batched CUDA environment of the bilateral impedance runner for policy training.

Every world integrates the bilateral chain with two persistent shoes at the
existing 1.25e-4 s substep. A control step holds one equilibrium, stiffness, and
damping command per joint for ``substeps`` substeps; the torque cap acts on every
substep, and no torque lag or rate bound applies unless :class:`EnvConfig` asks
for one. Observations contain only simulated state, internal oscillator phase,
filtered simulated foot loads, filtered torques, and the task speed. Measured
references enter only the rewards, terminations, reference state initialization,
and the critic's privileged inputs, which the deployed policy never sees.

Episodes start in flight from a measured training state, continue through many
contacts without resetting the shoes, and end on a screen failure, a large
departure from the reference, or the end of the measured segment.
"""

from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import warp as wp

from projects.digital_shoe.runtime import clone_params

from ..cartesian.gpu.foundation import FoundationFused
from ..cartesian.shoe import Shoe
from ..hogan.gpu_shoe import apply_ground_shoe, ground_shoe
from ..hogan.runner import Bounds
from . import data as bilateral_data
from . import gpu
from .mechanics import SIDES, BilateralChain, HeadArmsTrunk, chain_from_references

OBSERVATION_NAMES = (
    "pelvis_height_m",
    "pelvis_lean_rad",
    "forward_speed_error_m_s",
    "vertical_speed_m_s",
    "pelvis_rate_rad_s",
    *(f"{joint}_rad" for joint in ("hip_r", "knee_r", "ankle_r", "hip_l", "knee_l", "ankle_l")),
    *(f"{joint}_rate_10rad_s" for joint in ("hip_r", "knee_r", "ankle_r", "hip_l", "knee_l", "ankle_l")),
    "load_r_bw",
    "load_l_bw",
    *(f"torque_{joint}_cap" for joint in ("hip_r", "knee_r", "ankle_r", "hip_l", "knee_l", "ankle_l")),
    "ankle_r_x_m",
    "ankle_r_z_m",
    "ankle_l_x_m",
    "ankle_l_z_m",
    "sin_phase",
    "cos_phase",
    "task_speed_offset",
)
ACTION_NAMES = tuple(
    f"{output}_{joint}"
    for output in ("equilibrium", "stiffness", "damping")
    for joint in ("hip_r", "knee_r", "ankle_r", "hip_l", "knee_l", "ankle_l")
)
BODY_POINTS = ("knee_r", "ankle_r", "heel_r", "toe_r", "knee_l", "ankle_l", "heel_l", "toe_l")
"""Tracked leg points, relative to the mid-hip root."""


def response_fraction(dt_s: float, response_time_s: float) -> float:
    """Per-substep first-order torque response; a nonpositive time constant means no lag."""
    return 1.0 if response_time_s <= 0 else -math.expm1(-dt_s / response_time_s)


def observation_names(config: EnvConfig) -> tuple[str, ...]:
    """Return the policy inputs for a configuration; previous actions are optional."""
    if config.previous_action_obs:
        return OBSERVATION_NAMES + tuple(f"previous_{name}" for name in ACTION_NAMES)
    return OBSERVATION_NAMES


@dataclass(frozen=True)
class EnvConfig:
    """Physics, actuation, episode, and reward settings of the training environment."""

    dt_s: float = 1.25e-4
    """Physics substep [s]."""
    substeps: int = 16
    """Substeps per control step; 16 gives a 500 Hz policy."""
    response_time_s: float = 0.0
    """Joint torque response time [s]; zero, the default, applies the impedance torque
    without lag. The lag and slew comparison is the "Torque lag and slew" study in
    ``RESULTS.md``; the runs there recorded their own value, such as the fitted single-leg
    runner's 5.3 ms (``hogan/baselines/generative_runner_score_20261010.json``)."""
    torque_slew_limit: bool = False
    """Bound the joint torque rate by ``Bounds.torque_rate_max_nm_s``; ``False``, the
    default, removes it. The ``RESULTS.md`` runs were trained with ``True``."""
    load_time_s: float = 0.02
    frequency_hz: float = 1.5
    phase_feedback: float = 0.2
    reference_speed_m_s: float = 3.65
    speed_scale_m_s: float = 1.0
    immediate_damping: bool = True
    gravity_m_s2: float = 9.81
    maximum_force_n: float = 6000.0
    minimum_hip_height_m: float = 0.2
    maximum_speed: float = 100.0
    compression_limit: float = 0.99
    episode_s: float = 2.0
    """Longest episode [s]; segments may end earlier."""
    termination_height_m: float = 0.15
    termination_pelvis_rad: float = 0.6
    termination_forward_m: float = 0.35
    reward_weights: tuple[float, float, float, float, float] = (0.35, 0.1, 0.2, 0.25, 0.1)
    """Pose, joint velocity, pelvis position, foot force, and effort reward weights."""
    pose_scale_rad: float = 0.15
    velocity_scale_rad_s: float = 3.0
    position_scale_m: float = 0.05
    force_scale_bw: float = 0.3
    effort_scale: float = 0.15
    reward_kernels: str = "hybrid"
    """``"per_term"`` averages one kernel per joint angle, absolute foot pitch, and
    force component, so one large error does not flatten the others' gradient;
    ``"pooled"`` applies one kernel to each group's mean squared error (run 1);
    ``"hybrid"`` uses per-term pose kernels and the pooled force kernel, so the
    swing foot's easy zero force does not dilute the stance foot's impact."""
    body_position_weight: float = 0.0
    """Weight of the pooled leg-point position kernel (knee, ankle, heel, toe)."""
    body_velocity_weight: float = 0.0
    """Weight of the pooled leg-point velocity kernel."""
    body_position_scale_m: float = 0.06
    body_velocity_scale_m_s: float = 1.0
    contact_match_penalty: float = 0.0
    """Per-foot penalty when simulated and measured contact (> 50 N) disagree."""
    action_smoothness_penalty: float = 0.0
    """Penalty on the L2 change of the bounded impedance fractions between control steps."""
    previous_action_obs: bool = False
    """Append the previous bounded impedance fractions to the policy inputs."""
    force_penalty: float = 0.1
    """Weight of a non-saturating penalty on the summed absolute foot-force error
    divided by body weight. The force kernel is flat once an impact spike is
    large; this keeps a gradient toward smaller spikes."""

    @property
    def control_dt_s(self) -> float:
        return self.dt_s * self.substeps


def _fused_shoe(shoe: Shoe, worlds: int, dt: float, device, compression_limit: float):
    """Allocate one shoe's fused foundation for ``worlds`` carriers, as :mod:`..hogan.gpu_runner` does."""
    source = shoe.foundation
    bed = shoe.shoe.column_bed
    foundation = FoundationFused(
        source.anchor_local.numpy(),
        np.full(len(bed.rest_length_m), source.config.ground_height_m),
        source.rest_len.numpy(),
        source.area.numpy(),
        source.neighbors.numpy(),
        bed.spacing_m,
        shoe.shoe.material,
        np.arange(worlds),
        wp.zeros(worlds, dtype=wp.vec3, device=device),
        deepcopy(source.config),
        device,
        deepcopy(source.surround),
        world_count=worlds,
    )
    foundation.world_blocks = [clone_params(source.world_blocks[0]) for _ in range(worlds)]
    foundation.world_params.assign(foundation.world_blocks)
    foundation._materials_dirty = True
    wp.copy(foundation.friction_kt, source.friction_kt)
    wp.copy(foundation.friction_kv, source.friction_kv)
    if source.friction_solver is not None:
        adapter = foundation.friction_solver
        adapter.settings.assign(np.repeat(source.friction_solver.settings.numpy(), worlds, axis=0))
        wp.copy(adapter.base_kt, source.friction_solver.base_kt)
        wp.copy(adapter.base_kv, source.friction_solver.base_kv)
    foundation.set_world_timesteps(np.full(worlds, dt))
    lean = ground_shoe(foundation, bed.rest_length_m, exact_screen=True, compression_limit=compression_limit)
    if lean is None:
        raise ValueError("The bilateral environment requires a lean elastic-Coulomb ground shoe")
    carriers = SimpleNamespace(
        body_q=wp.zeros(worlds, dtype=wp.transform, device=device),
        body_qd=wp.zeros(worlds, dtype=wp.spatial_vector, device=device),
        body_f=wp.zeros(worlds, dtype=wp.spatial_vector, device=device),
    )
    history = [
        foundation.q_state,
        foundation.peq_prev,
        foundation.tangent_stuck,
        foundation.tangent_dwell,
        foundation.ground_force,
        foundation.contact_point,
        foundation.tangent_deflection,
        foundation.tangent_maxwell_force,
    ]
    if foundation.free_column_count:
        history += [
            foundation.surround_compression,
            foundation.surround_scratch,
            foundation.surround_previous,
            foundation.surround_rate,
        ]
    adapter = foundation.friction_solver
    if adapter is not None:
        history += [
            adapter.scratch_anchor,
            adapter.scratch_stuck,
            adapter.scratch_dwell,
            adapter.deflection,
            adapter.sliding_distance,
            adapter.maxwell_force,
            adapter.stored_energy,
            adapter.column_diagnostics,
            adapter.step_diagnostics,
        ]
    views = [wp.to_torch(array).view(worlds, foundation.column_count, -1) for array in history]
    views += [wp.to_torch(lean.phase), wp.to_torch(lean.pending)]
    return SimpleNamespace(foundation=foundation, lean=lean, carriers=carriers, history=views)


class ReferenceLibrary:
    """Measured bilateral segments resampled to the control clock, resident on the GPU.

    Positions are interpolated linearly from the motion clock; velocities are
    central differences of the measured positions. Foot forces are averaged over
    each control interval from the native force clock.
    """

    def __init__(self, dataset: str | Path, splits: tuple[str, ...], control_dt_s: float, device):
        root = Path(dataset).resolve()
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("schema") != bilateral_data.DATASET_SCHEMA:
            raise ValueError("Unsupported bilateral dataset schema")
        self.root = root
        self.manifest = manifest
        self.members = [m for m in manifest["members"] if m["split"] in splits]
        if not self.members:
            raise ValueError(f"No members in splits {splits}")
        profile = json.loads((root / manifest["shared_assets"]["profile.json"]["file"]).read_text(encoding="utf-8"))
        hat = HeadArmsTrunk(**manifest.get("head_arms_trunk", {}))
        q, qd, grf, contact, starts, lengths, flight, references = [], [], [], [], [], [], [], []
        chains = []
        for member in self.members:
            reference = bilateral_data.load(root / member["reference"])
            references.append(reference)
            per_side = {
                side: {"lengths_m": reference["lengths_m"][i], "endpoint_local_m": reference["endpoint_local_m"][i]}
                for i, side in enumerate(SIDES)
            }
            chains.append(chain_from_references(per_side, profile, float(reference["subject_mass_kg"]), hat))
            time = reference["time_s"]
            count = int(math.floor(time[-1] / control_dt_s)) + 1
            clock = np.arange(count) * control_dt_s
            positions = np.column_stack([np.interp(clock, time, reference["q"][:, c]) for c in range(9)])
            rates = np.gradient(reference["q"], time, axis=0)
            velocities = np.column_stack([np.interp(clock, time, rates[:, c]) for c in range(9)])
            fine = np.arange(count * 8) * (control_dt_s / 8)
            force = np.stack(
                [
                    np.column_stack(
                        [np.interp(fine, reference["grf_time_s"], reference["grf_side_n"][s, :, a]) for a in range(2)]
                    )
                    for s in range(2)
                ],
                axis=1,
            )
            force = force.reshape(count, 8, 2, 2).mean(axis=1)
            starts.append(sum(lengths))
            lengths.append(count)
            q.append(positions)
            qd.append(velocities)
            grf.append(force.reshape(count, 4))
            contact.append((force[:, :, 1] > 50.0).astype(np.float32))
            flight.append(np.all(force[:, :, 1] < 1.0, axis=1))
        reference_chain = chains[0]
        if any(abs(chain.total_mass_kg - reference_chain.total_mass_kg) > 1e-9 for chain in chains):
            raise ValueError("Segments must describe one subject mass")
        # Static calibrations differ slightly between trials; each segment keeps its own geometry.
        self.chain = reference_chain
        self.chains = chains
        self.segment_lengths = torch.tensor(
            np.array([chain.lengths_m for chain in chains]), dtype=torch.float64, device=device
        )
        self.references = references
        self.body_weight_n = reference_chain.total_mass_kg * 9.81
        self.speeds = torch.tensor([m["speed_m_s"] for m in self.members], dtype=torch.float64, device=device)
        self.q = torch.tensor(np.concatenate(q), dtype=torch.float64, device=device)
        self.qd = torch.tensor(np.concatenate(qd), dtype=torch.float64, device=device)
        self.grf = torch.tensor(np.concatenate(grf), dtype=torch.float64, device=device)
        self.contact = torch.tensor(np.concatenate(contact), device=device)
        self.starts = torch.tensor(starts, dtype=torch.long, device=device)
        self.lengths = torch.tensor(lengths, dtype=torch.long, device=device)
        flight_all = np.concatenate(flight)
        # Initial states must leave at least a quarter second of reference and start in flight.
        minimum = int(round(0.25 / control_dt_s))
        candidates = []
        for segment, (start, length) in enumerate(zip(starts, lengths, strict=True)):
            for k in range(3, length - minimum):
                if flight_all[start + k]:
                    candidates.append((segment, k))
        if not candidates:
            raise ValueError("No flight-phase initial states")
        self.initial_candidates = torch.tensor(candidates, dtype=torch.long, device=device)

    def __len__(self) -> int:
        return len(self.members)


class BilateralEnv:
    """Vectorized bilateral runner worlds exposed as torch tensors.

    Args:
        library: Measured training segments.
        shoe_left: Left-shoe artifact path.
        shoe_right: Right-shoe (mirrored) artifact path.
        mount_m: Ankle location in the intrinsic shoe frame [m].
        pitch_rad: Static shoe pitch [rad].
        worlds: Concurrent worlds.
        config: Environment settings.
        bounds: Per-joint impedance and torque bounds, shared by both legs.
        device: CUDA device.
        seed: Random seed for initial states.
    """

    action_size = len(ACTION_NAMES)
    privileged_size = 28

    def __init__(
        self,
        library: ReferenceLibrary,
        shoe_left: str | Path,
        shoe_right: str | Path,
        mount_m,
        pitch_rad: float,
        worlds: int,
        config: EnvConfig | None = None,
        bounds: Bounds | None = None,
        device: str = "cuda:0",
        seed: int = 0,
        mode: int = gpu.MODE_IMPEDANCE,
        candidates: int = 1,
    ):
        self.config = cfg = config or EnvConfig()
        self.bounds = bounds or Bounds()
        self.library = library
        self.worlds = int(worlds)
        self.device = wp.get_device(device)
        self.torch_device = wp.device_to_torch(self.device)
        # Torch and Warp share one stream so tensor views and kernels stay ordered.
        torch.cuda.set_stream(wp.stream_to_torch(self.device.stream))
        self.generator = torch.Generator(device=self.torch_device)
        self.generator.manual_seed(seed)
        self.chain: BilateralChain = library.chain
        self.mode = mode
        self.candidates = int(candidates)
        w, d = self.worlds, self.device
        shoes = {
            "right": Shoe(shoe_right, mount_m, pitch_rad),
            "left": Shoe(shoe_left, mount_m, pitch_rad),
        }
        self.shoes = {side: _fused_shoe(shoes[side], w, cfg.dt_s, d, cfg.compression_limit) for side in SIDES}
        self.shoe_metadata = {side: shoes[side].metadata for side in SIDES}
        self.params = wp.array([gpu.chain_params(self.chain)] * w, dtype=gpu.ChainParams, device=d)
        self.segment_params = wp.array(
            [gpu.chain_params(chain) for chain in library.chains], dtype=gpu.ChainParams, device=d
        )
        self.weights = wp.zeros((self.candidates, 9, gpu.FEATURE_COUNT), dtype=wp.float64, device=d)
        settings = gpu.Settings()
        settings.mode = mode
        settings.gravity = cfg.gravity_m_s2
        settings.dt = cfg.dt_s
        settings.load_rate = -math.expm1(-cfg.dt_s / cfg.load_time_s)
        settings.phase_feedback = cfg.phase_feedback
        settings.reference_speed = cfg.reference_speed_m_s
        settings.speed_scale = cfg.speed_scale_m_s
        settings.cadence_gain = 0.0
        settings.immediate_damping = int(cfg.immediate_damping)
        both = lambda values: gpu.Vec6(*values, *values)  # noqa: E731
        settings.lower = both(self.bounds.equilibrium_lower_rad)
        settings.upper = both(self.bounds.equilibrium_upper_rad)
        settings.stiffness_max = both(self.bounds.stiffness_max_nm_rad)
        settings.damping_max = both(self.bounds.damping_max_nms_rad)
        settings.torque_cap = both(self.bounds.torque_max_nm)
        settings.torque_rate = (
            both(self.bounds.torque_rate_max_nm_s) if cfg.torque_slew_limit else gpu.Vec6(*([1.0e12] * 6))
        )
        settings.pitch_r = shoes["right"].static_pitch_rad
        settings.pitch_l = shoes["left"].static_pitch_rad
        settings.max_force = cfg.maximum_force_n
        settings.hip_floor = cfg.minimum_hip_height_m
        settings.max_speed = cfg.maximum_speed
        settings.compression_limit = cfg.compression_limit
        self.settings = settings
        data = gpu.Worlds()
        data.q = wp.zeros(w, dtype=gpu.Vec9, device=d)
        data.v = wp.zeros(w, dtype=gpu.Vec9, device=d)
        data.torque = wp.zeros(w, dtype=gpu.Vec6, device=d)
        data.load = wp.zeros(w, dtype=gpu.Vec6, device=d)
        data.phase = wp.zeros(w, dtype=wp.float64, device=d)
        data.sense = wp.zeros(w, dtype=wp.vec2d, device=d)
        data.status = wp.zeros(w, dtype=int, device=d)
        data.speed = wp.zeros(w, dtype=wp.float64, device=d)
        data.frequency = wp.full(w, cfg.frequency_hz, dtype=wp.float64, device=d)
        data.response = wp.full(w, response_fraction(cfg.dt_s, cfg.response_time_s), dtype=wp.float64, device=d)
        data.equilibrium = wp.zeros(w, dtype=gpu.Vec6, device=d)
        data.stiffness = wp.zeros(w, dtype=gpu.Vec6, device=d)
        data.damping = wp.zeros(w, dtype=gpu.Vec6, device=d)
        data.enabled_r = self.shoes["right"].foundation.enabled
        data.enabled_l = self.shoes["left"].foundation.enabled
        data.fraction_r = wp.zeros(w, dtype=wp.float64, device=d)
        data.fraction_l = wp.zeros(w, dtype=wp.float64, device=d)
        data.invalid_r = wp.zeros(w, dtype=int, device=d)
        data.invalid_l = wp.zeros(w, dtype=int, device=d)
        data.grf_sum = wp.zeros(w, dtype=wp.vec4d, device=d)
        data.fz_square = wp.zeros(w, dtype=wp.vec2d, device=d)
        data.effort = wp.zeros(w, dtype=gpu.Vec6, device=d)
        data.power_positive = wp.zeros(w, dtype=gpu.Vec6, device=d)
        data.power_negative = wp.zeros(w, dtype=gpu.Vec6, device=d)
        data.compression_max = wp.zeros(w, dtype=wp.float64, device=d)
        data.substeps = wp.zeros(w, dtype=int, device=d)
        self.data = data
        self.t = SimpleNamespace(
            **{
                name: wp.to_torch(getattr(data, name))
                for name in (
                    "q",
                    "v",
                    "torque",
                    "phase",
                    "sense",
                    "status",
                    "speed",
                    "frequency",
                    "response",
                    "equilibrium",
                    "stiffness",
                    "damping",
                    "enabled_r",
                    "enabled_l",
                    "grf_sum",
                    "fz_square",
                    "effort",
                    "power_positive",
                    "power_negative",
                    "compression_max",
                    "substeps",
                )
            }
        )
        dev = self.torch_device
        self.lower = torch.tensor(self.bounds.equilibrium_lower_rad * 2, dtype=torch.float64, device=dev)
        self.upper = torch.tensor(self.bounds.equilibrium_upper_rad * 2, dtype=torch.float64, device=dev)
        self.stiffness_max = torch.tensor(self.bounds.stiffness_max_nm_rad * 2, dtype=torch.float64, device=dev)
        self.damping_max = torch.tensor(self.bounds.damping_max_nms_rad * 2, dtype=torch.float64, device=dev)
        self.torque_cap = torch.tensor(self.bounds.torque_max_nm * 2, dtype=torch.float64, device=dev)
        self.lengths = library.segment_lengths[torch.zeros(w, dtype=torch.long, device=dev)].clone()
        """Per-world thigh and shank lengths, right then left [m], shape [worlds, 2, 2]."""
        self.segment = torch.zeros(w, dtype=torch.long, device=dev)
        self.index = torch.zeros(w, dtype=torch.long, device=dev)
        self.origin = torch.zeros(w, 2, dtype=torch.float64, device=dev)
        """Offset of each world's measured root position from its integration frame [m]."""
        self.steps = torch.zeros(w, dtype=torch.long, device=dev)
        self.max_steps = int(round(cfg.episode_s / cfg.control_dt_s))
        self.observation_names = observation_names(cfg)
        self.observation_size = len(self.observation_names)
        self.fraction = torch.zeros(w, len(ACTION_NAMES), dtype=torch.float64, device=dev)
        self.previous_fraction = torch.zeros_like(self.fraction)
        # Sagittal heel and toe of the undeformed shoe outline, in the ankle-centred shoe frame.
        outline = np.vstack([shoes[side].outline_local for side in SIDES])
        self.foot_points = torch.tensor(
            np.array([outline[np.argmin(outline[:, 0])], outline[np.argmax(outline[:, 0])]]),
            dtype=torch.float64,
            device=dev,
        )
        self.shoe_pitch = float(shoes["right"].static_pitch_rad)
        self._graph = None

    # ----- integration -----

    def _substep(self):
        d, data, settings = self.device, self.data, self.settings
        right, left = self.shoes["right"], self.shoes["left"]
        wp.launch(
            gpu.stage,
            dim=self.worlds,
            inputs=[
                settings,
                self.params,
                self.weights,
                self.worlds // self.candidates,
                data,
                right.carriers.body_q,
                right.carriers.body_qd,
                left.carriers.body_q,
                left.carriers.body_qd,
            ],
            device=d,
        )
        wp.launch(
            gpu.park,
            dim=self.worlds,
            inputs=[data, right.carriers.body_q, right.carriers.body_qd, left.carriers.body_q, left.carriers.body_qd],
            device=d,
        )
        apply_ground_shoe(right.foundation, right.lean, right.carriers, data.fraction_r, data.invalid_r)
        apply_ground_shoe(left.foundation, left.lean, left.carriers, data.fraction_l, data.invalid_l)
        wp.launch(
            gpu.advance,
            dim=self.worlds,
            inputs=[settings, self.params, data, right.carriers.body_f, left.carriers.body_f],
            device=d,
        )

    def integrate(self):
        """Advance every running world by one control step."""
        t = self.t
        for name in ("grf_sum", "fz_square", "effort", "power_positive", "power_negative", "compression_max"):
            getattr(t, name).zero_()
        t.substeps.zero_()
        if self._graph is None:
            # Compile kernels and settle shoe caches before capturing the substep loop.
            self._substep()
            wp.synchronize_device(self.device)
            with wp.ScopedCapture(device=self.device) as capture:
                for _ in range(self.config.substeps):
                    self._substep()
            self._graph = capture.graph
            # The warm-up substep already advanced the worlds; finish this control step uncaptured.
            for _ in range(self.config.substeps - 1):
                self._substep()
            return
        wp.capture_launch(self._graph)

    def set_impedance(self, action: torch.Tensor):
        """Map unbounded actions, shape [worlds, 18], to bounded per-joint impedance."""
        fraction = torch.sigmoid(action.to(torch.float64).clamp(-40.0, 40.0))
        self.previous_fraction = self.fraction
        self.fraction = fraction
        self.t.equilibrium.copy_(self.lower + (self.upper - self.lower) * fraction[:, :6])
        self.t.stiffness.copy_(self.stiffness_max * fraction[:, 6:12])
        self.t.damping.copy_(self.damping_max * fraction[:, 12:])

    def set_linear_weights(self, weights: np.ndarray):
        """Upload shared linear Hogan weights, shape [candidates, 3, 3, 14]."""
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != (self.candidates, 3, 3, gpu.FEATURE_COUNT):
            raise ValueError(f"weights must have shape ({self.candidates}, 3, 3, {gpu.FEATURE_COUNT})")
        self.weights.assign(weights.reshape(self.candidates, 9, gpu.FEATURE_COUNT))

    # ----- state -----

    def ankles(self, q: torch.Tensor) -> torch.Tensor:
        """Return both ankle positions relative to the mid-hip [m], shape [worlds, 2, 2]."""
        result = []
        for s in range(2):
            thigh = q[:, 2] + q[:, 3 + 3 * s] + math.pi
            shank = thigh + q[:, 4 + 3 * s]
            x = self.lengths[:, s, 0] * torch.cos(thigh) + self.lengths[:, s, 1] * torch.cos(shank)
            z = self.lengths[:, s, 0] * torch.sin(thigh) + self.lengths[:, s, 1] * torch.sin(shank)
            result.append(torch.stack((x, z), dim=-1))
        return torch.stack(result, dim=1)

    def body_points(self, q: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Return knee, ankle, heel, and toe positions per leg relative to the mid-hip [m], shape [N, 8, 2]."""
        points = []
        for s in range(2):
            thigh = q[:, 2] + q[:, 3 + 3 * s] + math.pi
            shank = thigh + q[:, 4 + 3 * s]
            foot = shank + q[:, 5 + 3 * s] + 0.5 * math.pi - self.shoe_pitch
            knee = lengths[:, s, 0, None] * torch.stack((torch.cos(thigh), torch.sin(thigh)), -1)
            ankle = knee + lengths[:, s, 1, None] * torch.stack((torch.cos(shank), torch.sin(shank)), -1)
            c, d = torch.cos(foot), torch.sin(foot)
            heel_toe = [ankle + torch.stack((x * c - z * d, x * d + z * c), -1) for x, z in self.foot_points.unbind(0)]
            points += [knee, ankle, *heel_toe]
        return torch.stack(points, dim=1)

    def body_point_velocities(self, q: torch.Tensor, v: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Return leg-point velocities relative to the root [m/s] by central differences along ``v``."""
        h = 1.0e-4
        return (self.body_points(q + h * v, lengths) - self.body_points(q - h * v, lengths)) / (2 * h)

    def observe(self) -> torch.Tensor:
        """Return policy observations from simulated and internal state only, shape [worlds, 32]."""
        t = self.t
        q, v = t.q, t.v
        ankles = self.ankles(q).reshape(-1, 4)
        speed = t.speed
        observation = torch.cat(
            (
                q[:, 1:2],
                q[:, 2:3] - math.pi / 2,
                (v[:, 0:1] - speed[:, None]),
                v[:, 1:3],
                q[:, 3:9],
                v[:, 3:9] / 10.0,
                t.sense,
                t.torque / self.torque_cap,
                ankles,
                torch.sin(t.phase)[:, None],
                torch.cos(t.phase)[:, None],
                ((speed - self.config.reference_speed_m_s) / self.config.speed_scale_m_s)[:, None],
            ),
            dim=1,
        )
        if self.config.previous_action_obs:
            observation = torch.cat((observation, self.fraction), dim=1)
        return observation.float()

    def _reference_index(self) -> torch.Tensor:
        return self.library.starts[self.segment] + self.index

    def _interval_force(self) -> torch.Tensor:
        count = self.t.substeps.clamp(min=1).to(torch.float64)[:, None]
        return self.t.grf_sum / count

    def privileged(self) -> torch.Tensor:
        """Return critic-only reference context, shape [worlds, 28]."""
        lib = self.library
        k = self._reference_index()
        q_ref = lib.q[k].clone()
        q_ref[:, :2] -= self.origin
        weight = lib.body_weight_n
        remaining = (lib.lengths[self.segment] - self.index).to(torch.float64) * self.config.control_dt_s
        return torch.cat(
            (
                q_ref - self.t.q,
                (lib.qd[k] - self.t.v) / 10.0,
                lib.grf[k] / weight,
                self._interval_force() / weight,
                remaining.clamp(max=2.0)[:, None],
                (self.steps.to(torch.float64) / self.max_steps)[:, None],
            ),
            dim=1,
        ).float()

    def reward(self) -> tuple[torch.Tensor, dict]:
        """Score the completed control step against the measured reference."""
        cfg, lib, t = self.config, self.library, self.t
        k = self._reference_index()
        q_ref = lib.q[k].clone()
        q_ref[:, :2] -= self.origin
        pose_error = t.q[:, 2:] - q_ref[:, 2:]
        force_error = self._interval_force() - lib.grf[k]
        force_scale = (cfg.force_scale_bw * lib.body_weight_n) ** 2
        if cfg.reward_kernels not in ("per_term", "pooled", "hybrid"):
            raise ValueError("reward_kernels must be 'per_term', 'pooled', or 'hybrid'")
        if cfg.reward_kernels == "pooled":
            pose_kernel = torch.exp(-torch.mean(pose_error**2, dim=1) / cfg.pose_scale_rad**2)
        else:
            # Absolute foot pitch sets how the shoe lands; it is the sum of the leg's chain angles.
            feet = torch.stack([pose_error[:, 0] + pose_error[:, 1 + 3 * s : 4 + 3 * s].sum(1) for s in range(2)], 1)
            pose_kernel = torch.exp(-(torch.cat((pose_error, feet), 1) ** 2) / cfg.pose_scale_rad**2).mean(1)
        if cfg.reward_kernels == "per_term":
            force_kernel = torch.exp(-(force_error**2) / force_scale).mean(1)
        else:
            force_kernel = torch.exp(-torch.mean(force_error**2, dim=1) / force_scale)
        velocity = torch.mean((t.v[:, 3:] - lib.qd[k][:, 3:]) ** 2, dim=1) / cfg.velocity_scale_rad_s**2
        position = torch.sum((t.q[:, :2] - q_ref[:, :2]) ** 2, dim=1) / cfg.position_scale_m**2
        effort = torch.mean(t.effort / self.config.substeps / self.torque_cap**2, dim=1) / cfg.effort_scale
        terms = torch.stack((pose_kernel, torch.exp(-velocity), torch.exp(-position), force_kernel, torch.exp(-effort)))
        weights = torch.tensor(cfg.reward_weights, dtype=torch.float64, device=self.torch_device)[:, None]
        penalty = force_error.abs().sum(1) / lib.body_weight_n
        reward = (weights * terms).sum(0) - cfg.force_penalty * penalty
        info = {
            name: terms[i].mean().item() for i, name in enumerate(("pose", "velocity", "position", "force", "effort"))
        }
        info["force_error_bw"] = penalty.mean().item()
        if cfg.body_position_weight or cfg.body_velocity_weight:
            points = self.body_points(t.q, self.lengths)
            points_ref = self.body_points(q_ref, self.lengths)
            position_error = ((points - points_ref) ** 2).sum(-1).mean(1)
            body_position = torch.exp(-position_error / cfg.body_position_scale_m**2)
            velocity_error = (
                (
                    (
                        self.body_point_velocities(t.q, t.v, self.lengths)
                        - self.body_point_velocities(q_ref, lib.qd[k], self.lengths)
                    )
                    ** 2
                )
                .sum(-1)
                .mean(1)
            )
            body_velocity = torch.exp(-velocity_error / cfg.body_velocity_scale_m_s**2)
            reward = reward + cfg.body_position_weight * body_position + cfg.body_velocity_weight * body_velocity
            info["body_position"] = body_position.mean().item()
            info["body_velocity"] = body_velocity.mean().item()
        if cfg.contact_match_penalty:
            simulated = self._interval_force()[:, [1, 3]] > 50.0
            mismatch = (simulated != (lib.grf[k][:, [1, 3]] > 50.0)).to(torch.float64).sum(1)
            reward = reward - cfg.contact_match_penalty * mismatch
            info["contact_mismatch"] = mismatch.mean().item()
        if cfg.action_smoothness_penalty:
            change = (self.fraction - self.previous_fraction).norm(dim=1)
            # The first step after a reset has no previous command.
            change = torch.where(self.steps > 1, change, torch.zeros_like(change))
            reward = reward - cfg.action_smoothness_penalty * change
            info["action_change"] = change.mean().item()
        return reward.float(), info

    def _departed(self) -> torch.Tensor:
        cfg = self.config
        q_ref = self.library.q[self._reference_index()].clone()
        q_ref[:, :2] -= self.origin
        q = self.t.q
        return (
            ((q[:, 1] - q_ref[:, 1]).abs() > cfg.termination_height_m)
            | ((q[:, 2] - q_ref[:, 2]).abs() > cfg.termination_pelvis_rad)
            | ((q[:, 0] - q_ref[:, 0]).abs() > cfg.termination_forward_m)
        )

    def _write_state(self, mask: torch.Tensor, q: torch.Tensor, v: torch.Tensor, phase: torch.Tensor):
        t = self.t
        self.lengths[mask] = self.library.segment_lengths[self.segment[mask]]
        wp.launch(
            gpu.assign_params,
            dim=self.worlds,
            inputs=[
                self.segment_params,
                wp.from_torch(self.segment.to(torch.int32)),
                wp.from_torch(mask.to(torch.int32)),
                self.params,
            ],
            device=self.device,
        )
        t.q[mask] = q
        t.v[mask] = v
        t.phase[mask] = phase
        t.torque[mask] = 0.0
        t.sense[mask] = 0.0
        t.status[mask] = 0
        t.enabled_r[mask] = 1
        t.enabled_l[mask] = 1
        for shoe in self.shoes.values():
            for view in shoe.history:
                view[mask] = 0

    @staticmethod
    def kinematic_phase(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Initial oscillator phase from the right hip angle and rate, as :func:`..hogan.identify.initialize`."""
        return torch.remainder(torch.atan2(-v[:, 3] / 8.0, q[:, 3]), 2 * math.pi)

    def reset(self, mask: torch.Tensor | None = None):
        """Start masked worlds in flight from random measured training states."""
        if mask is None:
            mask = torch.ones(self.worlds, dtype=torch.bool, device=self.torch_device)
        count = int(mask.sum())
        if count == 0:
            return
        lib = self.library
        choice = torch.randint(
            len(lib.initial_candidates), (count,), generator=self.generator, device=self.torch_device
        )
        segment, index = lib.initial_candidates[choice].unbind(1)
        k = lib.starts[segment] + index
        q = lib.q[k].clone()
        origin = q[:, :2].clone()
        origin[:, 1] = 0.0
        q[:, :2] -= origin
        v = lib.qd[k]
        self.segment[mask] = segment
        self.index[mask] = index
        self.origin[mask] = origin
        self.steps[mask] = 0
        self.t.speed[mask] = lib.speeds[segment]
        self.fraction[mask] = 0.0
        self.previous_fraction[mask] = 0.0
        self._write_state(mask, q, v, self.kinematic_phase(q, v))

    def step(self, action: torch.Tensor):
        """Apply one control step, score it, and reset finished worlds.

        Returns:
            Next observations, privileged critic inputs, rewards, terminations
            (failure or departure), truncations (time or segment end), and an
            info dictionary holding the pre-reset privileged inputs of
            truncated worlds and the mean reward terms.
        """
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
        done = terminated | truncated
        self.reset(done)
        return self.observe(), self.privileged(), reward, terminated, truncated, info

    def describe(self) -> dict:
        """Return the environment settings for provenance."""
        return {
            "config": asdict(self.config),
            "bounds": asdict(self.bounds),
            "chain": self.chain.to_dict(),
            "shoes": self.shoe_metadata,
            "observations": list(self.observation_names),
            "actions": list(ACTION_NAMES),
            "worlds": self.worlds,
        }
