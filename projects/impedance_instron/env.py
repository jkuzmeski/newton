# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Vectorized stance environment for one closed-loop impedance policy.

The trajectory optimizer in :mod:`projects.impedance_instron.optimize` solves one open-loop
command per shoe. This module turns the same rig into a batch of independent stance episodes so a
single feedback policy can be trained on a known shoe, frozen, and then replayed on shoes it never
saw. Every world is a private copy of the planar impedance rig: its own ankle fixture, its own
upper mass, and its own block of the batched :class:`~projects.digital_shoe.runtime.MidsoleFoundation`
column bed, so per-world materials never interact.

**One decision per frame, not per substep.** An episode is one stance window of
``episode_frames`` frames at 120 Hz, and the policy emits one action per frame. The 64 solver
substeps inside a frame are replayed from a single captured CUDA graph, exactly as
:meth:`projects.impedance_instron.example.Example.step` does. Deciding per substep would force a
re-capture every substep and give back the graph-replay speedup.

**Residual actions.** The action is a residual on a nominal command supplied at construction:
``[dL0, dlogK, dzeta]``, i.e. a shift of the commanded equilibrium length, a multiplicative change
of the commanded stiffness, and a shift of the commanded damping ratio. Stiffness is residual in
log space so it can never reach zero, and the resolved ``L0``, ``K`` and ``zeta`` are clipped into
the same box :meth:`projects.impedance_instron.control.LegCommand.bounds` enforces. A zero action
therefore reproduces the nominal open-loop command bit for bit, which is what makes the
zero-action acceptance test meaningful.

**The observation carries mechanics only.** The shoe material parameters are deliberately absent.
A policy handed the foam constants could look the answer up instead of inferring the material from
how the shoe responds, and such a policy would not transfer to a shoe whose constants were never
identified. Every entry of :data:`OBSERVATION_LAYOUT` is a quantity a real instrumented rig could
measure: leg geometry, ground reaction force, fixture pose and rates, and the policy's own
previous action.

**Rewards.** The dense per-frame reward is the negated increment of the tier 3 work proxy of
:class:`projects.impedance_instron.objective.Objective`, ``W+ / 0.25 + abs(W-) / 1.20``. That proxy
is a time integral of the leg source power, so it is exactly additive over frames: the trapezoid
rule over the whole episode equals the sum of the per-frame trapezoids, which share their
endpoints. The dense term is therefore an exact decomposition of tier 3, not a shaping heuristic.
An optional momentum-tracking term (``shape_reward``, on by default) and a terminal penalty built
from the tier 1 violations and tier 2 excursions of the same :class:`Objective` complete the
return. The reward scales are engineering choices and are documented at
:data:`WORK_REWARD_SCALE_J`, :data:`MOMENTUM_REWARD_SCALE_M_S`, :data:`TERMINAL_VIOLATION_PENALTY`
and :data:`TERMINAL_EXCURSION_PENALTY`.

**NumPy in, NumPy out.** The environment never imports a deep-learning framework. At 64 worlds one
frame moves a few kilobytes across the bus against a frame that costs milliseconds on the GPU, so
the host round trip is not the bottleneck, and keeping the boundary at NumPy leaves the module
testable and free of an optional dependency. The learner owns the tensor library.
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import warp as wp

import newton
from projects.digital_shoe.runtime import MidsoleFoundation

from .control import LegCommand
from .example import Example, create_parser
from .objective import Objective, Tolerances
from .optimize import CHECKPOINTS, Rollout, measured_target

__all__ = ["OBSERVATION_LAYOUT", "ImpedanceEnv"]

# Per-world trace columns written every substep by :func:`_apply_leg_and_record`.
TRACE_SHOE_FZ = 0
TRACE_SHOE_FX = 1
TRACE_ANKLE_Z = 2
TRACE_ANKLE_VZ = 3
TRACE_ANKLE_VX = 4
TRACE_UPPER_VZ = 5
TRACE_UPPER_VX = 6
TRACE_LEG_FORCE = 7
TRACE_SOURCE_POWER = 8
TRACE_DAMPER_POWER = 9
TRACE_LEG_LENGTH = 10
TRACE_LEG_RATE = 11
TRACE_COMPRESSION = 12
TRACE_SATURATED = 13
TRACE_SATURATION_EXCESS = 14
TRACE_UPPER_Z = 15
TRACE_COM_Z = 16
TRACE_COLUMNS = 17

# Command columns of the per-world, per-substep device command: the five numbers
# ``_apply_leg_and_record`` needs to evaluate the Hogan law. They mirror reference columns
# 13, 14, 25, 26 and 27 of :meth:`Example._make_equilibrium_command`.
COMMAND_L0 = 0
COMMAND_L0_RATE = 1
COMMAND_STIFFNESS = 2
COMMAND_DAMPING = 3
COMMAND_STIFFNESS_RATE = 4
COMMAND_COLUMNS = 5

# Reference columns the batched kernels read; shared by every world because the pitch motor and
# the opposite-foot boundary force are properties of the captured task, not of the shoe.
_REFERENCE_PITCH = 2
_REFERENCE_PITCH_RATE = 7
_REFERENCE_OTHER_FZ = 12
_REFERENCE_OTHER_FX = 23

OBSERVATION_LAYOUT: tuple[tuple[str, str, float], ...] = (
    ("leg_length_offset", "leg length minus 1 m [m]", 0.2),
    ("leg_length_rate", "leg length rate [m/s]", 2.0),
    ("shoe_fz_bw", "shoe vertical force / body weight [-]", 1.0),
    ("shoe_fx_bw", "shoe fore-aft force / body weight [-]", 1.0),
    ("foot_pitch", "fixture pitch [rad]", 0.5),
    ("foot_pitch_rate", "fixture pitch rate [rad/s]", 10.0),
    ("ankle_height", "ankle height above the floor [m]", 0.15),
    ("ankle_vz", "ankle vertical velocity [m/s]", 2.0),
    ("com_vz", "mass-weighted COM vertical velocity [m/s]", 2.0),
    ("contact_phase", "frames since commanded touchdown / episode frames [-]", 1.0),
    ("previous_d_length", "previous commanded length residual / its own limit [-]", 1.0),
    ("previous_d_log_stiffness", "previous log-stiffness residual / its own limit [-]", 1.0),
    ("previous_d_damping_ratio", "previous damping-ratio residual / its own limit [-]", 1.0),
)
"""Observation entries as ``(name, meaning, normalizing scale)``, in index order.

The third element divides the raw quantity, so every entry lands near unit magnitude on the
reference stance. No entry names a shoe material constant; see the module docstring.
"""

# Residual half-ranges. ``tanh`` squashes the raw action into [-1, 1] and these scale it:
# 50 mm of equilibrium length, a factor of exp(0.7) ~ 2 on stiffness, and 0.3 of damping ratio.
# They are wide enough to change the stance qualitatively and narrow enough that a random policy
# still lands inside the LegCommand box.
ACTION_SCALE: tuple[float, float, float] = (0.05, 0.7, 0.3)

WORK_REWARD_SCALE_J = 30.0
"""Divisor [J] of the dense work reward.

Engineering choice. The reference command spends about 160 J of tier 3 work proxy over 45 frames,
so dividing the per-frame increment by 30 J puts a typical frame reward near -0.1.
"""

MOMENTUM_REWARD_SCALE_M_S = 0.15
"""Divisor [m/s] of the momentum-tracking reward.

Calibrated to carry the tier 2 pressure DENSELY rather than at the terminal frame. An at-tolerance
error of 0.044 m/s costs about 0.29 per frame, roughly 11.7 over a stance, which exceeds the whole
0 to 227 J work range (7.6 reward units). Task accuracy therefore cannot be bought with work even
before the terminal penalty applies.

Why dense: a large terminal penalty makes the return unpredictable from early observations, because
nothing visible at frame 5 determines whether the episode ends off task. Training with a 25 per unit
terminal penalty and a 0.5 divisor here drove explained variance to -0.36 with a value loss of 151,
against 1.000 and 0.002 when the terminal term was small. Explained variance is scale invariant, so
rescaling does not fix it; moving the pressure to where the error accrues does.
"""

TERMINAL_EXCURSION_PENALTY = 10.0
"""Terminal cost per multiple of a tier 2 tolerance.

Calibrated, not chosen for feel. :class:`~projects.impedance_instron.objective.Objective` makes an
off-task rollout LEXICOGRAPHICALLY worse than any on-task one. A reward is a weighted sum, so that
ordering only survives if one tolerance unit costs more than the largest work saving available.
The observed tier 3 range on this rig is about 0 to 227 J, which is 227 / WORK_REWARD_SCALE_J
= 7.6 reward units.

The tier 2 pressure is SPLIT between here and :data:`MOMENTUM_REWARD_SCALE_M_S`. The dense term
already contributes about 11.7 reward units over a stance for an at-tolerance error, which alone
exceeds the 7.6 work range, so this terminal term only has to preserve the ordering rather than
carry it. 10 per unit does that with margin while keeping the terminal contribution small enough
for the critic to predict.

History, because both failure modes are instructive. A penalty of 1.0 was wrong: the policy cut
the work proxy 6.7x while pushing the momentum excursion from 0.87 to 2.12 tolerances, because the
trade was profitable. Raising it to 25 with all the pressure at the terminal frame was also wrong:
it drove explained variance to -0.36 with a value loss of 151, because nothing visible early in an
episode predicts whether it ends off task. If the tier 3 range on a future shoe grows, raise the
DENSE term with it first.
"""

TERMINAL_VIOLATION_PENALTY = 250.0
"""Terminal cost per unit of normalized tier 1 violation, plus the same amount as a flat charge.

Calibrated to keep tier 1 above tier 2 for the same reason. Observed excursions reach roughly four
tolerance units, so the worst realistic tier 2 charge is about 4 * 10 terminal plus the dense
momentum term, of order 50 to 60 reward units; 250 keeps infeasibility strictly worse than any
on-task-but-imperfect episode with a wide margin.
"""


@wp.kernel
def _advance_sample_index(index: wp.array[wp.int32]):
    """Advance the shared device sample counter to the substep about to be recorded."""
    index[0] = index[0] + 1


@wp.kernel
def _prescribe_world_axes(
    index: wp.array[wp.int32],
    reference: wp.array2d[wp.float32],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    """Hold every world out of plane and drive its pitch from the shared motor schedule.

    The per-world arithmetic is the arithmetic of
    :func:`projects.impedance_instron.example._constrain_planar_axes` with the two body indices
    offset by the world, so a one-world batch reproduces the single-world rig.
    """
    world = wp.tid()
    i = index[0]
    foot = 2 * world
    upper = foot + 1
    a = wp.transform_get_translation(body_q[foot])
    c = wp.transform_get_translation(body_q[upper])
    va = wp.spatial_top(body_qd[foot])
    vc = wp.spatial_top(body_qd[upper])
    body_q[foot] = wp.transform(
        wp.vec3(a[0], 0.0, a[2]),
        wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), reference[i, 2]),
    )
    body_q[upper] = wp.transform(wp.vec3(c[0], 0.0, c[2]), wp.quat_identity())
    body_qd[foot] = wp.spatial_vector(wp.vec3(va[0], 0.0, va[2]), wp.vec3(0.0, reference[i, 7], 0.0))
    body_qd[upper] = wp.spatial_vector(wp.vec3(vc[0], 0.0, vc[2]), wp.vec3(0.0))


@wp.kernel
def _apply_leg_and_record(
    index: wp.array[wp.int32],
    reference: wp.array2d[wp.float32],
    command: wp.array3d[wp.float32],
    force_limit: float,
    unilateral: int,
    com_share: float,
    shoe_force: wp.array[wp.vec3],
    compression: wp.array[wp.float32],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_f: wp.array[wp.spatial_vector],
    trace: wp.array3d[wp.float32],
):
    """Apply each world's Hogan leg and record the substep it produced.

    The impedance law, the unilateral release, the saturation split and the source-power ledger
    are those of :func:`projects.impedance_instron.example._apply_equilibrium_leg`; only the
    command now comes from a per-world array instead of a shared reference row, and the recording
    is fused into the same launch because both are one thread per world.

    ``com_share`` is ``foot_mass / mass``, so the recorded COM height is the same mass weighting
    :meth:`ImpedanceEnv._com_vz` applies to the velocities. Height is recorded directly because
    the leg length is a 3-D distance and cannot be resolved back into two heights.
    """
    world = wp.tid()
    i = index[0]
    foot = 2 * world
    upper = foot + 1
    r = wp.transform_get_translation(body_q[upper]) - wp.transform_get_translation(body_q[foot])
    length = wp.max(wp.length(r), 1.0e-6)
    n = r / length
    rate = wp.dot(n, wp.spatial_top(body_qd[upper]) - wp.spatial_top(body_qd[foot]))
    equilibrium = command[i, world, 0]
    equilibrium_rate = command[i, world, 1]
    k = command[i, world, 2]
    b = command[i, world, 3]
    k_rate = command[i, world, 4]
    error = length - equilibrium
    slip = rate - equilibrium_rate
    raw = -k * error - b * slip
    limited = wp.clamp(raw, -force_limit, force_limit)
    force = limited
    if unilateral != 0:
        # A leg extends against the ground; it cannot pull the shoe back down.
        force = wp.max(limited, 0.0)
    f = force * n
    other = wp.vec3(reference[i, 23], 0.0, reference[i, 12])
    wp.atomic_add(body_f, foot, wp.spatial_vector(-f, wp.vec3(0.0)))
    wp.atomic_add(body_f, upper, wp.spatial_vector(f + other, wp.vec3(0.0)))
    shoe = shoe_force[world]
    va = wp.spatial_top(body_qd[foot])
    vc = wp.spatial_top(body_qd[upper])
    ankle_z = wp.transform_get_translation(body_q[foot])[2]
    upper_z = wp.transform_get_translation(body_q[upper])[2]
    trace[i, world, 0] = shoe[2]
    trace[i, world, 1] = shoe[0]
    trace[i, world, 2] = ankle_z
    trace[i, world, 3] = va[2]
    trace[i, world, 4] = va[0]
    trace[i, world, 5] = vc[2]
    trace[i, world, 6] = vc[0]
    trace[i, world, 7] = force
    # Source power closes the ledger P_body + dE/dt + D exactly, in the clamped branches too.
    trace[i, world, 8] = raw * equilibrium_rate + 0.5 * k_rate * error * error + (force - raw) * rate
    trace[i, world, 9] = -b * slip * slip
    trace[i, world, 10] = length
    trace[i, world, 11] = rate
    trace[i, world, 12] = compression[world]
    trace[i, world, 13] = float(wp.abs(limited - raw) > 1.0e-4)
    trace[i, world, 14] = wp.abs(limited - raw)
    trace[i, world, 15] = upper_z
    trace[i, world, 16] = com_share * ankle_z + (1.0 - com_share) * upper_z


def _momentum_checkpoints(times: np.ndarray, velocity: np.ndarray, loaded: np.ndarray) -> list[float]:
    """Sample the velocity change through contact at the tier 2 checkpoint fractions.

    Mirrors the private helper behind :func:`projects.impedance_instron.optimize.simulate` so a
    rollout built here is scored by the same history the trajectory optimizer was scored on.

    Args:
        times: Sample times [s], shape [sample_count].
        velocity: COM velocity component [m/s], shape [sample_count].
        loaded: Mask of samples the shoe carried load in, shape [sample_count].
    """
    if loaded.sum() < 2:
        return [0.0] * len(CHECKPOINTS)
    span, values = times[loaded], velocity[loaded]
    phase = (span - span[0]) / max(span[-1] - span[0], 1.0e-9)
    return np.interp(CHECKPOINTS, phase, values - values[0]).tolist()


class ImpedanceEnv:
    """A batch of independent impedance-rig stance episodes behind a NumPy step interface.

    Args:
        num_worlds: Number of independent stance episodes advanced per :meth:`step`.
        args: Parsed namespace from
            :func:`projects.impedance_instron.example.create_parser`; the planar pitch rig and the
            equilibrium controller are required and are selected on a private copy.
        nominal: Flat :class:`~projects.impedance_instron.control.LegCommand` parameter vector the
            residual action acts around, shape [command size].
        seed: Seed of the environment's own generator. The rig itself is deterministic given the
            actions, so the seed currently only seeds :attr:`rng` for callers that randomize
            materials between episodes.
        shape_reward: Add the optional per-frame momentum-tracking term. The dense work term is
            not optional: it is an exact decomposition of the tier 3 objective.
    """

    def __init__(
        self,
        num_worlds: int,
        args,
        nominal: np.ndarray,
        seed: int = 0,
        shape_reward: bool = True,
    ):
        if int(num_worlds) < 1:
            raise ValueError(f"num_worlds must be at least one, got {num_worlds}")
        self.num_worlds = int(num_worlds)
        self.shape_reward = bool(shape_reward)
        self.rng = np.random.default_rng(seed)
        self.seed = int(seed)

        # A private copy so selecting the equilibrium controller and stamping the nominal command
        # never mutates the caller's namespace; ``Example`` also writes back into ``args``.
        self.args = copy.deepcopy(args)
        self.args.control = "equilibrium"
        self.args.control_vector = np.asarray(nominal, dtype=float).reshape(-1)
        self.device = wp.get_device()

        # The reference construction of the single-world example is 300 lines of measured-profile
        # geometry. Building one prototype reuses it verbatim instead of restating it, and the
        # prototype's reference rows are the exact float32 the open-loop rig is driven with.
        prototype = Example(newton.viewer.ViewerNull(num_frames=self.args.num_frames), self.args)
        if prototype.command.size != self.args.control_vector.size:
            raise ValueError(f"nominal must hold {prototype.command.size} parameters")
        self.nominal_parameters = self.args.control_vector.copy()
        self.times = prototype.times.copy()
        self.sample_count = int(prototype.sample_count)
        self.sim_dt = float(prototype.sim_dt)
        self.frame_dt = float(prototype.frame_dt)
        self.substeps = int(self.args.substeps)
        if (self.sample_count - 1) % self.substeps:
            raise ValueError(f"{self.sample_count - 1} substeps do not divide into whole frames of {self.substeps}")
        self._episode_frames = (self.sample_count - 1) // self.substeps
        self.mass = float(prototype.mass)
        self.foot_mass = float(prototype.foot_mass)
        self.com_mass = float(prototype.com_mass)
        self.gravity = float(prototype.gravity)
        self.body_weight_n = self.mass * self.gravity
        self.force_limit_n = float(self.args.force_limit_bw * self.body_weight_n)
        self.touchdown_time_s = float(prototype.registration["touchdown_time_s"])
        self.minimum_last_offsets = np.asarray(prototype.minimum_last_offsets, dtype=float).copy()
        reference = np.ascontiguousarray(prototype.reference, dtype=np.float32)
        self._reference_host = reference.copy()
        self.target = measured_target(prototype.profile, self.mass, self.gravity)
        self.objective = Objective(self.target, Tolerances(), body_weight_n=self.body_weight_n)

        self._build_nominal_command(reference)
        self._build_action_bounds(prototype.command)
        self._build_momentum_reference()
        self._build_model(prototype)
        # The prototype owns a second model and a second column bed on the device; nothing below
        # reads it, so it is released as soon as its host-side products are copied out.
        del prototype

        self._graph = None
        self.use_graph = bool(getattr(self.args, "graph", True)) and self.device.is_cuda
        self.graph_status = "enabled" if self.use_graph else "disabled"
        self._frame_index = 0
        self._started = False
        self._previous_residual = np.zeros((self.num_worlds, 3), dtype=np.float64)
        self._residual_history = np.zeros((self._episode_frames, self.num_worlds, 3), dtype=np.float64)
        self._command_host = np.zeros((self.sample_count, self.num_worlds, COMMAND_COLUMNS), dtype=np.float32)
        self._touchdown_velocity = np.zeros((self.num_worlds, 2), dtype=float)

    # ------------------------------------------------------------------ construction

    def _build_nominal_command(self, reference: np.ndarray) -> None:
        """Cache the nominal leg command the residual acts around, in double precision.

        Columns 13, 14, 25, 26 and 27 of the prototype reference already hold the evaluated
        equilibrium length, its rate, the stiffness, the damping and the stiffness rate. Reusing
        them rather than re-evaluating the spline is what makes a zero residual bit-exact.

        Args:
            reference: Prototype reference rows, shape [sample_count, 28].
        """
        self._nominal = np.column_stack(
            [reference[:, 13], reference[:, 14], reference[:, 25], reference[:, 26], reference[:, 27]]
        ).astype(np.float64)
        stiffness = self._nominal[:, COMMAND_STIFFNESS]
        # b = 2 zeta sqrt(k m) inverted, so the residual can move the ratio the command was
        # written in instead of the raw damper. The inverse is only ever used as a ratio.
        self._nominal_zeta = self._nominal[:, COMMAND_DAMPING] / (2.0 * np.sqrt(stiffness * self.com_mass))

    def _build_action_bounds(self, command: LegCommand) -> None:
        """Read the resolved-command box straight out of :meth:`LegCommand.bounds`.

        Args:
            command: The prototype's leg command, whose knot bounds are also profile bounds
                because a clamped B-spline stays in the convex hull of its coefficients.
        """
        lower, upper = command.bounds()
        first = command.length_knots
        second = first + command.stiffness_knots
        self.length_bounds_m = (float(lower[0]), float(upper[0]))
        self.stiffness_bounds_n_m = (float(np.exp(lower[first])), float(np.exp(upper[first])))
        self.damping_ratio_bounds = (float(lower[second]), float(upper[second]))
        self.action_scale = np.asarray(ACTION_SCALE, dtype=float)

    def _build_momentum_reference(self) -> None:
        """Resample the measured momentum history onto the substep grid, once.

        The tier 2 history is stated at fractions of contact. The dense term needs it as a
        function of time, so it is interpolated against the measured stance duration starting at
        the commanded touchdown instant. Before touchdown the term is switched off entirely.
        """
        phase = (self.times - self.touchdown_time_s) / max(self.target.duration_s, 1.0e-9)
        self._momentum_active = phase >= 0.0
        clipped = np.clip(phase, 0.0, 1.0)
        self._momentum_vx = np.interp(clipped, CHECKPOINTS, self.target.momentum_vx_m_s)
        self._momentum_vz = np.interp(clipped, CHECKPOINTS, self.target.momentum_vz_m_s)
        self._touchdown_sample = int(np.argmin(np.abs(self.times - self.touchdown_time_s)))

    def _build_model(self, prototype: Example) -> None:
        """Replicate the prototype's two-body rig once per world and batch the column bed.

        Args:
            prototype: Single-world example supplying the geometry, the calibrated shoe and the
                initial planar state.
        """
        newton.use_coord_layout_targets = True
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0, 0.0, -self.gravity))
        carriers = []
        for world in range(self.num_worlds):
            carriers.append(
                builder.add_body(
                    mass=self.foot_mass,
                    com=wp.vec3(0.0),
                    inertia=wp.mat33(np.eye(3) * prototype.pitch_inertia),
                    label=f"robot_foot_fixture_{world}",
                )
            )
            builder.add_body(
                mass=self.com_mass,
                com=wp.vec3(0.0),
                inertia=wp.mat33(np.eye(3)),
                label=f"upper_inertial_slider_{world}",
            )
        # No shapes and no ground plane: the last mesh never collides in this rig and the solver
        # is stepped with no contacts, so the meshes would only cost memory in a training batch.
        self.model = builder.finalize(device=self.device)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.solver = newton.solvers.SolverSemiImplicit(self.model, angular_damping=0.0, enable_tri_contact=False)

        self._initial_q = np.zeros((2 * self.num_worlds, 7), dtype=np.float32)
        self._initial_q[:, 6] = 1.0
        self._initial_q[0::2, :3] = prototype.planar_initial_positions[0]
        self._initial_q[1::2, :3] = prototype.planar_initial_positions[1]
        self._initial_qd = np.zeros((2 * self.num_worlds, 6), dtype=np.float32)
        self._initial_qd[0::2] = prototype.planar_initial_velocity[0]
        self._initial_qd[1::2] = prototype.planar_initial_velocity[1]

        bed = prototype.shoe.column_bed
        scale = float(self.args.shoe_stiffness_scale)
        # The same scaling ``Example.__init__`` applies: both Ogden-Hill terms and the reported
        # Pasternak coupling move together, so the scaled foam is the same shape at a different
        # stiffness rather than a differently shaped foam.
        material = replace(
            prototype.shoe.material,
            instantaneous_shear_modulus_pa=prototype.shoe.material.instantaneous_shear_modulus_pa * scale,
            instantaneous_shear_modulus_2_pa=prototype.shoe.material.instantaneous_shear_modulus_2_pa * scale,
            pasternak_n_per_m=prototype.shoe.material.pasternak_n_per_m * scale,
        )
        # Kept so a caller can randomize the material around the shoe the policy trained on.
        self.shoe = prototype.shoe
        self.material = material
        self.foundation = MidsoleFoundation(
            bed.anchor_bottom_m - prototype.ankle_mount,
            np.zeros(len(bed.rest_length_m)),
            bed.rest_length_m,
            bed.area_m2,
            bed.neighbors,
            bed.spacing_m,
            material,
            carriers,
            self.model.body_com,
            prototype.contact_config,
            self.device,
            prototype.surround_config,
            world_count=self.num_worlds,
        )
        self.reference_device = wp.array(self._reference_host, dtype=wp.float32, device=self.device)
        self.index_device = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.command_device = wp.zeros(
            (self.sample_count, self.num_worlds, COMMAND_COLUMNS), dtype=wp.float32, device=self.device
        )
        self.trace_device = wp.zeros(
            (self.sample_count, self.num_worlds, TRACE_COLUMNS), dtype=wp.float32, device=self.device
        )

    # ------------------------------------------------------------------ public interface

    @property
    def observation_dim(self) -> int:
        """Width of one observation row; see :data:`OBSERVATION_LAYOUT`."""
        return len(OBSERVATION_LAYOUT)

    @property
    def action_dim(self) -> int:
        """Width of one action row: the residual ``[dL0, dlogK, dzeta]``."""
        return 3

    @property
    def episode_frames(self) -> int:
        """Frames of one stance episode, one policy decision each."""
        return self._episode_frames

    @property
    def reference(self) -> np.ndarray:
        """Measured reference the rig is driven against, shape [sample_count, 28], read-only.

        These are the rows :meth:`projects.impedance_instron.example.Example._make_reference`
        builds from the captured running profile, already resolved onto the substep grid, so a
        caller can put a simulated signal next to the measured one it is supposed to reproduce.
        The columns an analysis or a training dashboard needs:

        * 1: ankle height of the nominal rolling reference [m]
        * 4: height of the virtual upper mass of the reference [m]
        * 9: vertical velocity of the reference upper mass [m/s]
        * 11: measured shoe vertical force [N]
        * 22: measured shoe fore-aft force [N]
        * 2 and 7: commanded fixture pitch [rad] and pitch rate [rad/s]
        * 12 and 23: measured opposite-foot vertical and fore-aft force [N]
        * 13, 14, 25, 26, 27: the nominal equilibrium length [m], its rate [m/s], stiffness [N/m],
          damping [N·s/m] and stiffness rate [N/m/s] the residual action acts around

        A read-only NumPy view is returned rather than a copy. A copy would hide the real hazard:
        the device upload in :attr:`reference_device` was made once at construction, so writing
        into the host rows would not change what the rig is driven by and would only make the
        reported reference disagree with it. Refusing the write says so immediately, and a view
        costs nothing on a 323 kB array that callers read every evaluation.
        """
        view = self._reference_host.view()
        view.setflags(write=False)
        return view

    def reset(self) -> np.ndarray:
        """Start a fresh stance in every world and return the first observation.

        Returns the observation array, shape [num_worlds, observation_dim], dtype float32.
        """
        self.state_0.body_q.assign(self._initial_q)
        self.state_0.body_qd.assign(self._initial_qd)
        self.state_1.body_q.assign(self._initial_q)
        self.state_1.body_qd.assign(self._initial_qd)
        self.state_0.clear_forces()
        self.state_1.clear_forces()
        self.foundation.reset()
        # One relaxation sweep outside any capture refreshes the per-world Maxwell constants a
        # material change invalidated; a captured graph replays no Python and could not.
        self.foundation.relax_surround(self.state_0, self.sim_dt)
        self._clear_foundation_fields()
        self.index_device.zero_()
        self.trace_device.zero_()
        self.command_device.zero_()
        self._command_host[:] = 0.0
        self._residual_history[:] = 0.0
        self._previous_residual[:] = 0.0
        self._touchdown_velocity[:] = 0.0
        self._frame_index = 0
        self._started = True
        # Sample 0 is recorded before any decision, so it is driven by the nominal command.
        self._write_command(np.array([0]), self._previous_residual)
        self._record_sample()
        return self._observe(0)

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Advance one frame in every world. Returns obs, reward, done, info.

        The action of frame ``f`` holds over substeps ``64 f + 1`` to ``64 f + 64``: the decision
        is taken from the observation at the frame boundary and is constant inside the frame,
        which is what lets the captured CUDA graph replay the whole frame.

        Args:
            actions: Raw policy output, shape [num_worlds, 3]. It is squashed with ``tanh`` and
                scaled by :data:`ACTION_SCALE` before it is added to the nominal command.

        Returns:
            ``obs`` [num_worlds, observation_dim] float32, ``reward`` [num_worlds] float32,
            ``done`` [num_worlds] bool, and an ``info`` dict holding the reward split, the
            resolved command of the frame, and, on the final frame, one
            :class:`~projects.impedance_instron.objective.Verdict` per world.
        """
        if not self._started:
            raise RuntimeError("call reset() before step()")
        if self._frame_index >= self._episode_frames:
            raise RuntimeError("the episode is over; call reset()")
        command = np.asarray(actions, dtype=float)
        if command.shape != (self.num_worlds, 3):
            raise ValueError(f"actions must have shape {(self.num_worlds, 3)}, got {command.shape}")
        if not np.all(np.isfinite(command)):
            raise ValueError("actions must be finite")

        residual = np.tanh(command) * self.action_scale
        frame = self._frame_index
        start = frame * self.substeps
        rows = np.arange(start + 1, start + self.substeps + 1)
        self._write_command(rows, residual)
        self._advance_frame()
        self._frame_index += 1
        self._previous_residual = residual
        self._residual_history[frame] = residual

        window = self.trace_device[start : start + self.substeps + 1].numpy()
        work_reward = self._work_reward(window, start)
        momentum_reward = self._momentum_reward(window, start)
        reward = work_reward + momentum_reward
        done = self._frame_index >= self._episode_frames
        info: dict = {
            "work_reward": work_reward.astype(np.float32),
            "momentum_reward": momentum_reward.astype(np.float32),
            "terminal_reward": np.zeros(self.num_worlds, dtype=np.float32),
            "frame": frame,
        }
        if done:
            terminal, verdicts, rollouts = self._terminal()
            reward = reward + terminal
            info["terminal_reward"] = terminal.astype(np.float32)
            info["verdicts"] = verdicts
            info["rollouts"] = rollouts
            info["objective_j"] = np.array([v.objective_j for v in verdicts], dtype=np.float32)
            info["feasible"] = np.array([v.feasible for v in verdicts], dtype=bool)
            info["on_task"] = np.array([v.on_task for v in verdicts], dtype=bool)
            info["value"] = np.array([v.value for v in verdicts], dtype=np.float64)
        observation = self._observe(start + self.substeps)
        return (
            observation,
            reward.astype(np.float32),
            np.full(self.num_worlds, done, dtype=bool),
            info,
        )

    def set_world_materials(self, materials: list) -> None:
        """Give every world its own foam, in world order.

        Call it between episodes, never inside a frame: it copies to the device and the next
        :meth:`reset` refreshes the relaxation constants that depend on the material.

        Args:
            materials: One :class:`~projects.digital_shoe.runtime.ShoeMaterial` per world.
        """
        if len(materials) != self.num_worlds:
            raise ValueError(f"set_world_materials needs {self.num_worlds} materials, got {len(materials)}")
        self.foundation.set_world_materials(materials)
        self._started = False

    def realised_command(self, world: int) -> dict:
        """Return the impedance one world was actually driven with, for analysis.

        Args:
            world: World index.

        Returns:
            Sample times [s], the resolved equilibrium length [m] and its rate [m/s], stiffness
            [N/m], damping [N·s/m] and damping ratio [-] over the samples produced so far, plus
            the per-frame residual actually applied.
        """
        if not 0 <= int(world) < self.num_worlds:
            raise IndexError(f"world {world} is outside a batch of {self.num_worlds}")
        filled = min(self._frame_index * self.substeps + 1, self.sample_count)
        block = self._command_host[:filled, int(world)].astype(float)
        stiffness = block[:, COMMAND_STIFFNESS]
        return {
            "time_s": self.times[:filled].copy(),
            "length_m": block[:, COMMAND_L0],
            "length_rate_m_s": block[:, COMMAND_L0_RATE],
            "stiffness_n_m": stiffness,
            "damping_n_s_m": block[:, COMMAND_DAMPING],
            "damping_ratio": block[:, COMMAND_DAMPING] / (2.0 * np.sqrt(stiffness * self.com_mass)),
            "stiffness_rate_n_m_s": block[:, COMMAND_STIFFNESS_RATE],
            "residual": self._residual_history[: self._frame_index, int(world)].copy(),
        }

    def trace(self, world: int) -> np.ndarray:
        """Return one world's recorded substep trace so far, shape [samples, :data:`TRACE_COLUMNS`].

        Args:
            world: World index.
        """
        filled = min(self._frame_index * self.substeps + 1, self.sample_count)
        return self.trace_device.numpy()[:filled, int(world)]

    # ------------------------------------------------------------------ internals

    def _clear_foundation_fields(self) -> None:
        """Zero every foundation field a fresh stance must not inherit.

        :meth:`MidsoleFoundation.reset` clears the viscoelastic history and the bristles. The
        compression, pressure, per-column force and free-surface fields are rewritten every
        substep, but a reused environment would otherwise start its first substep from the
        previous episode's values.
        """
        self.foundation.z_free.zero_()
        self.foundation.compression.zero_()
        self.foundation.base_pressure.zero_()
        self.foundation.column_force.zero_()
        self.foundation.tangent_anchor.zero_()
        if self.foundation.free_column_count:
            self.foundation.surround_compression.zero_()
            self.foundation.surround_scratch.zero_()
            self.foundation.surround_previous.zero_()
            self.foundation.surround_rate.zero_()

    def _write_command(self, rows: np.ndarray, residual: np.ndarray) -> None:
        """Resolve the nominal command plus a residual onto the given substep rows.

        The resolved stiffness is ``k exp(dlogK)`` clipped into the LegCommand box, so it is
        positive by construction. The damping follows it through
        ``b = b_nominal (zeta / zeta_nominal) sqrt(k / k_nominal)``, which is algebraically the
        same ``b = 2 zeta sqrt(k m)`` the command was written with but leaves a zero residual
        exactly equal to the nominal float32. The commanded length rate is the nominal rate: a
        residual held constant over the frame adds no rate inside it.

        Args:
            rows: Substep indices to fill, shape [n].
            residual: Resolved residual per world, shape [num_worlds, 3].
        """
        nominal = self._nominal[rows]
        length = nominal[:, COMMAND_L0][:, None]
        length_rate = nominal[:, COMMAND_L0_RATE][:, None]
        stiffness = nominal[:, COMMAND_STIFFNESS][:, None]
        damping = nominal[:, COMMAND_DAMPING][:, None]
        stiffness_rate = nominal[:, COMMAND_STIFFNESS_RATE][:, None]
        zeta = self._nominal_zeta[rows][:, None]
        resolved_length = np.clip(length + residual[None, :, 0], *self.length_bounds_m)
        resolved_stiffness = np.clip(stiffness * np.exp(residual[None, :, 1]), *self.stiffness_bounds_n_m)
        resolved_zeta = np.clip(zeta + residual[None, :, 2], *self.damping_ratio_bounds)
        gain = resolved_stiffness / stiffness
        block = np.empty((len(rows), self.num_worlds, COMMAND_COLUMNS), dtype=np.float32)
        block[:, :, COMMAND_L0] = resolved_length
        block[:, :, COMMAND_L0_RATE] = np.broadcast_to(length_rate, resolved_length.shape)
        block[:, :, COMMAND_STIFFNESS] = resolved_stiffness
        block[:, :, COMMAND_DAMPING] = damping * (resolved_zeta / zeta) * np.sqrt(gain)
        block[:, :, COMMAND_STIFFNESS_RATE] = stiffness_rate * gain
        first, last = int(rows[0]), int(rows[-1]) + 1
        self._command_host[first:last] = block
        self.command_device[first:last].assign(block)

    def _record_sample(self) -> None:
        """Load the foundation and the legs and record one substep in every world."""
        wp.launch(
            _prescribe_world_axes,
            dim=self.num_worlds,
            inputs=[self.index_device, self.reference_device, self.state_0.body_q, self.state_0.body_qd],
            device=self.device,
        )
        self.state_0.clear_forces()
        self.foundation.apply(self.state_0, self.sim_dt)
        wp.launch(
            _apply_leg_and_record,
            dim=self.num_worlds,
            inputs=[
                self.index_device,
                self.reference_device,
                self.command_device,
                self.force_limit_n,
                int(self.args.leg_unilateral),
                self.foot_mass / self.mass,
                self.foundation.resultant_force,
                self.foundation.max_compression,
                self.state_0.body_q,
                self.state_0.body_qd,
                self.state_0.body_f,
                self.trace_device,
            ],
            device=self.device,
        )

    def _substep(self, final: bool = False) -> None:
        """Integrate one substep and record the sample that follows it.

        Args:
            final: True on the last substep of a captured frame, where an odd substep count has
                to copy instead of swap so the replayed graph keeps its recorded bindings.
        """
        self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
        if final and self.substeps % 2:
            self.state_0.assign(self.state_1)
        else:
            self.state_0, self.state_1 = self.state_1, self.state_0
        wp.launch(_advance_sample_index, dim=1, inputs=[self.index_device], device=self.device)
        self._record_sample()

    def _frame(self) -> None:
        """Run exactly one full frame of substeps, the sequence a captured graph replays."""
        for substep in range(self.substeps):
            self._substep(final=substep == self.substeps - 1)

    def _advance_frame(self) -> None:
        """Replay the captured frame graph, capturing it on first use."""
        if not self.use_graph:
            self._frame()
            return
        if self._graph is None:
            state_0, state_1 = self.state_0, self.state_1
            try:
                with wp.ScopedCapture() as capture:
                    self._frame()
                self._graph = capture.graph
            except Exception as error:
                self.state_0, self.state_1 = state_0, state_1
                self.use_graph = False
                self.graph_status = f"capture failed: {error}"
                self._frame()
                return
        wp.capture_launch(self._graph)

    def _observe(self, sample: int) -> np.ndarray:
        """Build the observation of every world at one substep index.

        Args:
            sample: Substep index at the frame boundary.
        """
        row = self.trace_device[sample].numpy().astype(np.float64)
        reference = self._reference_host[sample]
        phase = (sample / self.substeps - self.touchdown_time_s / self.frame_dt) / self._episode_frames
        observation = np.empty((self.num_worlds, self.observation_dim), dtype=np.float64)
        observation[:, 0] = (row[:, TRACE_LEG_LENGTH] - 1.0) / OBSERVATION_LAYOUT[0][2]
        observation[:, 1] = row[:, TRACE_LEG_RATE] / OBSERVATION_LAYOUT[1][2]
        observation[:, 2] = row[:, TRACE_SHOE_FZ] / self.body_weight_n
        observation[:, 3] = row[:, TRACE_SHOE_FX] / self.body_weight_n
        observation[:, 4] = reference[_REFERENCE_PITCH] / OBSERVATION_LAYOUT[4][2]
        observation[:, 5] = reference[_REFERENCE_PITCH_RATE] / OBSERVATION_LAYOUT[5][2]
        observation[:, 6] = row[:, TRACE_ANKLE_Z] / OBSERVATION_LAYOUT[6][2]
        observation[:, 7] = row[:, TRACE_ANKLE_VZ] / OBSERVATION_LAYOUT[7][2]
        observation[:, 8] = self._com_vz(row) / OBSERVATION_LAYOUT[8][2]
        observation[:, 9] = phase
        observation[:, 10:13] = self._previous_residual / self.action_scale
        return np.ascontiguousarray(observation, dtype=np.float32)

    def _com_vz(self, row: np.ndarray) -> np.ndarray:
        """Return the mass-weighted COM vertical velocity [m/s] of a recorded substep row.

        Args:
            row: One substep of the trace, shape [num_worlds, :data:`TRACE_COLUMNS`].
        """
        share = self.foot_mass / self.mass
        return share * row[:, TRACE_ANKLE_VZ] + (1.0 - share) * row[:, TRACE_UPPER_VZ]

    def _work_reward(self, window: np.ndarray, start: int) -> np.ndarray:
        """Return the negated tier 3 work increment of one frame, per world.

        The trapezoid rule is additive over contiguous subintervals that share their endpoints,
        so summing this over the episode reproduces the ``W+ / 0.25 + abs(W-) / 1.20`` proxy
        :meth:`Objective.work_proxy_j` computes from the whole trace. It is a decomposition of
        the objective, not a shaping term.

        Args:
            window: Trace rows of the frame including both endpoints, shape
                [substeps + 1, num_worlds, :data:`TRACE_COLUMNS`].
            start: Substep index of the first row of the window.
        """
        times = self.times[start : start + window.shape[0]]
        power = window[:, :, TRACE_SOURCE_POWER].astype(np.float64)
        positive = np.trapezoid(np.clip(power, 0.0, None), times, axis=0)
        negative = np.trapezoid(np.clip(power, None, 0.0), times, axis=0)
        proxy = positive / self.objective.positive_efficiency + np.abs(negative) / self.objective.negative_efficiency
        return -proxy / WORK_REWARD_SCALE_J

    def _momentum_reward(self, window: np.ndarray, start: int) -> np.ndarray:
        """Return the optional momentum-tracking reward of one frame, per world.

        The measured stance momentum history is compared at the frame boundary against the
        velocity each world has gained since the commanded touchdown instant. Before touchdown
        the term is zero, because no stance has started to track.

        Args:
            window: Trace rows of the frame including both endpoints.
            start: Substep index of the first row of the window.
        """
        if not self.shape_reward:
            return np.zeros(self.num_worlds)
        share = self.foot_mass / self.mass
        end = start + window.shape[0] - 1
        if start <= self._touchdown_sample <= end:
            local = self._touchdown_sample - start
            row = window[local].astype(np.float64)
            self._touchdown_velocity[:, 0] = share * row[:, TRACE_ANKLE_VX] + (1.0 - share) * row[:, TRACE_UPPER_VX]
            self._touchdown_velocity[:, 1] = self._com_vz(row)
        if not self._momentum_active[end]:
            return np.zeros(self.num_worlds)
        row = window[-1].astype(np.float64)
        vx = share * row[:, TRACE_ANKLE_VX] + (1.0 - share) * row[:, TRACE_UPPER_VX]
        error = np.abs(vx - self._touchdown_velocity[:, 0] - self._momentum_vx[end]) + np.abs(
            self._com_vz(row) - self._touchdown_velocity[:, 1] - self._momentum_vz[end]
        )
        return -error / MOMENTUM_REWARD_SCALE_M_S

    def _rollout(self, trace: np.ndarray, world: int) -> Rollout:
        """Reduce one finished world to the task-level outcome :class:`Objective` scores.

        The fields and their thresholds are those of
        :func:`projects.impedance_instron.optimize.simulate`, so a policy and a solved open-loop
        command are graded by exactly the same verdict.

        Args:
            trace: Full episode trace, shape [sample_count, num_worlds, :data:`TRACE_COLUMNS`].
            world: World index.
        """
        rows = trace[:, world].astype(np.float64)
        times = self.times
        share = self.foot_mass / self.mass
        com_vx = share * rows[:, TRACE_ANKLE_VX] + (1.0 - share) * rows[:, TRACE_UPPER_VX]
        com_vz = share * rows[:, TRACE_ANKLE_VZ] + (1.0 - share) * rows[:, TRACE_UPPER_VZ]
        loaded = rows[:, TRACE_SHOE_FZ] > 0.02 * self.body_weight_n
        finite = bool(np.all(np.isfinite(rows)))
        power = rows[:, TRACE_SOURCE_POWER]
        return Rollout(
            completed=finite,
            contact=bool(loaded.any()),
            contact_start_s=float(times[loaded][0]) if loaded.any() else float("nan"),
            contact_end_s=float(times[loaded][-1]) if loaded.any() else float("nan"),
            contact_duration_s=float(times[loaded][-1] - times[loaded][0]) if loaded.sum() > 1 else 0.0,
            delta_vx_m_s=float(com_vx[loaded][-1] - com_vx[loaded][0]) if loaded.sum() > 1 else 0.0,
            delta_vz_m_s=float(com_vz[loaded][-1] - com_vz[loaded][0]) if loaded.sum() > 1 else 0.0,
            effort=float(np.mean((rows[:, TRACE_LEG_FORCE] / self.body_weight_n) ** 2)) if finite else float("inf"),
            actuator_work_j=float(np.trapezoid(power, times)) if finite else float("nan"),
            peak_leg_force_n=float(np.abs(rows[:, TRACE_LEG_FORCE]).max()) if finite else float("nan"),
            peak_shoe_force_n=float(rows[:, TRACE_SHOE_FZ].max()) if finite else float("nan"),
            peak_compression_m=float(rows[:, TRACE_COMPRESSION].max()) if finite else float("nan"),
            min_last_height_m=float(np.min(rows[:, TRACE_ANKLE_Z] + self.minimum_last_offsets))
            if finite
            else float("nan"),
            residual_load_n=float(rows[-1, TRACE_SHOE_FZ]) if finite else float("nan"),
            saturation_excess_n=float(np.mean(rows[:, TRACE_SATURATION_EXCESS])) if finite else float("inf"),
            saturated=bool(np.any(rows[:, TRACE_SATURATED] != 0.0)) if finite else True,
            momentum_vx_m_s=_momentum_checkpoints(times, com_vx, loaded),
            momentum_vz_m_s=_momentum_checkpoints(times, com_vz, loaded),
            damper_dissipation_j=-float(np.trapezoid(rows[:, TRACE_DAMPER_POWER], times)) if finite else float("inf"),
            positive_work_j=float(np.trapezoid(np.clip(power, 0.0, None), times)) if finite else float("nan"),
            negative_work_j=float(np.trapezoid(np.clip(power, None, 0.0), times)) if finite else float("nan"),
        )

    def _terminal(self) -> tuple[np.ndarray, list, list]:
        """Score the finished episode of every world and return the terminal reward.

        Tier 3 is already paid out frame by frame, so the terminal term charges only what the
        dense reward cannot see: the tier 1 feasibility gate and the tier 2 task deadbands, each
        in multiples of its own limit, which is what makes summing them across units defensible.
        """
        trace = self.trace_device.numpy()
        penalty = np.zeros(self.num_worlds)
        verdicts, rollouts = [], []
        for world in range(self.num_worlds):
            rollout = self._rollout(trace, world)
            verdict = self.objective.evaluate(rollout)
            rollouts.append(rollout)
            verdicts.append(verdict)
            if verdict.violations:
                penalty[world] += TERMINAL_VIOLATION_PENALTY * (1.0 + sum(verdict.violations.values()))
            if verdict.excursions:
                penalty[world] += TERMINAL_EXCURSION_PENALTY * sum(verdict.excursions.values())
        return -penalty, verdicts, rollouts


def _create_parser():
    """Extend the example parser with the settings this demonstration needs."""
    parser = create_parser()
    parser.set_defaults(control="equilibrium", viewer="null")
    parser.add_argument(
        "--nominal",
        type=Path,
        default=Path("outputs/impedance_instron/command_j.json"),
        help="Solved command file whose 'parameters' vector the residual action acts around.",
    )
    parser.add_argument(
        "--worlds",
        type=int,
        nargs="+",
        default=(1, 16, 64),
        help="World counts to time one zero-action episode at.",
    )
    parser.add_argument(
        "--reference-trace",
        type=Path,
        default=Path("outputs/impedance_instron/eval_j/trace.csv"),
        help="Stored open-loop trace the zero-action episode is checked against, when present.",
    )
    return parser


def _reference_trace(path: Path) -> dict[str, np.ndarray] | None:
    """Read the stored shoe force and leg length of an open-loop run, or None when absent.

    Args:
        path: Trace CSV written by :meth:`projects.impedance_instron.example.Example.save`.
    """
    if not path.is_file():
        return None
    names = ("shoe_fz_n", "leg_length_m")
    with path.open() as handle:
        header = handle.readline().strip().split(",")
    columns = np.loadtxt(path, delimiter=",", skiprows=1, usecols=[header.index(name) for name in names])
    return dict(zip(names, columns.T, strict=True))


def main():
    """Run one zero-action episode, check it against the solved command, and time the batch.

    A zero residual is the open-loop command, so this both demonstrates the interface and is the
    acceptance check of the environment: the printed differences are the float-atomic
    reproducibility band of the elastic foundation, not a modelling error.
    """
    args = _create_parser().parse_args()
    nominal = np.asarray(json.loads(args.nominal.read_text())["parameters"], dtype=float)
    stored = _reference_trace(args.reference_trace)
    for worlds in args.worlds:
        env = ImpedanceEnv(worlds, args, nominal)
        actions = np.zeros((worlds, env.action_dim))
        env.reset()
        for _ in range(env.episode_frames):
            env.step(actions)  # first episode also pays the one-off CUDA graph capture
        env.reset()
        began = time.perf_counter()
        for _ in range(env.episode_frames):
            _observation, reward, _done, info = env.step(actions)
        elapsed = time.perf_counter() - began
        verdict = info["verdicts"][0]
        print(
            f"worlds {worlds:3d}  {1.0e3 * elapsed / env.episode_frames:7.3f} ms/frame  "
            f"{1.0e3 * elapsed / env.episode_frames / worlds:7.4f} ms/world-frame  "
            f"return {float(np.sum(reward)) / worlds:+.3f} on the last frame  {verdict.summary()}"
        )
        if stored is not None:
            trace = env.trace(0)
            force = np.abs(trace[:, TRACE_SHOE_FZ] - stored["shoe_fz_n"]).max()
            length = np.abs(trace[:, TRACE_LEG_LENGTH] - stored["leg_length_m"]).max()
            print(
                f"             zero-action vs {args.reference_trace}: "
                f"shoe Fz {force:.3e} N ({force / np.abs(stored['shoe_fz_n']).max():.2e} of peak), "
                f"leg length {length:.3e} m ({length / np.abs(stored['leg_length_m']).max():.2e} of peak)"
            )
        del env


if __name__ == "__main__":
    main()
