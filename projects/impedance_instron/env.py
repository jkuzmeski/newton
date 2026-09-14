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

**The ankle is optional and symmetric.** Passing ``ankle`` widens the action to
``[dL0, dlogK, dzeta, dtheta0, dlogK_theta, dzeta_theta]`` and hands foot pitch to the
equilibrium-point rotational impedance of
:func:`projects.impedance_instron.example._apply_ankle_impedance` instead of an ideal prescribed
motor. Pitch then becomes an integrated state, which is what a prescribed motor structurally
cannot give: with it, stance duration and vertical impulse are satisfiable but the measured
momentum history is not, at any training budget. The ankle residual is resolved against
:meth:`projects.impedance_instron.control.AnkleCommand.bounds` by the same algebra as the leg, and
the ankle actuator's source power is traced at :data:`TRACE_ANKLE_SOURCE_POWER` so the objective
can charge it. An uncharged pitch motor is a free resource and a policy will spend it on
everything. Leaving ``ankle`` as None keeps the three-dimensional prescribed-pitch environment
bit for bit.

**The observation carries mechanics only.** The shoe material parameters are deliberately absent.
A policy handed the foam constants could look the answer up instead of inferring the material from
how the shoe responds, and such a policy would not transfer to a shoe whose constants were never
identified. Every entry of :data:`OBSERVATION_LAYOUT` is a quantity a real instrumented rig could
measure: leg geometry, ground reaction force, fixture pose and rates, and the policy's own
previous action.

**Rewards.** The dense per-frame reward is the negated increment of the tier 3 work proxy of
:class:`projects.impedance_instron.objective.Objective`, ``W+ / 0.25 + abs(W-) / 1.20`` charged on
the leg actuator and, when it runs, on the ankle actuator too. That proxy is a time integral of
each actuator's source power, so it is exactly additive over frames: the trapezoid
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
import math
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import warp as wp

import newton
from projects.digital_shoe.runtime import MidsoleFoundation

from .control import AnkleCommand, LegCommand
from .example import (
    ANKLE_ANGLE,
    ANKLE_ANGLE_RATE,
    ANKLE_COLUMN_COUNT,
    ANKLE_DAMPING,
    ANKLE_STIFFNESS,
    ANKLE_STIFFNESS_RATE,
    Example,
    create_parser,
)
from .objective import Objective, Tolerances
from .optimize import CHECKPOINTS, Rollout, measured_target

__all__ = ["ANKLE_OBSERVATION_LAYOUT", "OBSERVATION_LAYOUT", "ImpedanceEnv", "ankle_seed"]

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
# Appended for the ankle actuator. Pitch and pitch rate are recorded in both modes: they are the
# ACHIEVED state, which equals the prescribed schedule only while the prescribed motor writes it.
TRACE_ANKLE_SOURCE_POWER = 17
TRACE_ANKLE_DAMPER_POWER = 18
TRACE_ANKLE_TORQUE = 19
TRACE_ANKLE_SATURATION_EXCESS = 20
TRACE_PITCH = 21
TRACE_PITCH_RATE = 22
TRACE_COLUMNS = 23

# Command columns of the per-world, per-substep device command: the five numbers
# ``_apply_leg_and_record`` needs to evaluate the Hogan law. They mirror reference columns
# 13, 14, 25, 26 and 27 of :meth:`Example._make_equilibrium_command`.
COMMAND_L0 = 0
COMMAND_L0_RATE = 1
COMMAND_STIFFNESS = 2
COMMAND_DAMPING = 3
COMMAND_STIFFNESS_RATE = 4
# The ankle block mirrors the leg block and mirrors reference columns
# :data:`~projects.impedance_instron.example.ANKLE_ANGLE` onward. It is allocated in both modes
# and left at zero when the ankle is prescribed, so one kernel serves both.
COMMAND_ANKLE_ANGLE = 5
COMMAND_ANKLE_ANGLE_RATE = 6
COMMAND_ANKLE_STIFFNESS = 7
COMMAND_ANKLE_DAMPING = 8
COMMAND_ANKLE_STIFFNESS_RATE = 9
COMMAND_COLUMNS = 10

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
    ("episode_phase", "frames since the commanded touchdown / episode frames [-]", 1.0),
    ("stance_phase", "time since the DETECTED touchdown / measured stance duration [-]", 1.0),
    ("in_contact", "one once this world's shoe has been loaded, else zero [-]", 1.0),
    ("previous_d_length", "previous commanded length residual / its own limit [-]", 1.0),
    ("previous_d_log_stiffness", "previous log-stiffness residual / its own limit [-]", 1.0),
    ("previous_d_damping_ratio", "previous damping-ratio residual / its own limit [-]", 1.0),
)
"""Observation entries as ``(name, meaning, normalizing scale)``, in index order.

The third element divides the raw quantity, so every entry lands near unit magnitude on the
reference stance. No entry names a shoe material constant; see the module docstring.

``foot_pitch`` and ``foot_pitch_rate`` are the ACHIEVED fixture state read back from the trace,
not the commanded schedule. While the pitch motor is prescribed the two are the same number; once
the ankle actuator drives pitch they are not, and only the achieved one is observable.

``episode_phase`` and ``stance_phase`` are different clocks and both are needed. The first is the
deterministic position in the fixed-length episode. The second is where this world is inside its
OWN stance, which is the clock the momentum reward and the tier 2 momentum excursion are both
measured on, and which a memoryless policy cannot reconstruct from the instantaneous force alone.
``in_contact`` separates "stance has not started" from "stance just started", which
``stance_phase`` alone cannot express because both read zero.
"""

ANKLE_OBSERVATION_LAYOUT: tuple[tuple[str, str, float], ...] = (
    ("previous_d_angle", "previous commanded ankle-angle residual / its own limit [-]", 1.0),
    ("previous_d_log_ankle_stiffness", "previous ankle log-stiffness residual / its own limit [-]", 1.0),
    ("previous_d_ankle_damping_ratio", "previous ankle damping-ratio residual / its own limit [-]", 1.0),
)
"""Extra observation entries appended when the ankle actuator is enabled.

The previous action is in the observation because the command is a state the policy itself sets.
Feeding back only the leg half of a six-dimensional action would hide the ankle impedance the
policy just chose, and ``k_theta`` is otherwise reachable only through the pitch dynamics it
produces, so the leg-only layout is left untouched and these three are appended instead.
"""

# Residual half-ranges. ``tanh`` squashes the raw action into [-1, 1] and these scale it.
#
# Leg, indices 0..2: 50 mm of equilibrium length, a factor of exp(0.7) ~ 2 on stiffness, and 0.3
# of damping ratio. Wide enough to change the stance qualitatively, narrow enough that a random
# policy still lands inside the LegCommand box.
#
# Ankle, indices 3..5, sized against :meth:`~projects.impedance_instron.control.AnkleCommand.bounds`
# and the measured pitch, which sweeps 1.635 rad over this stance inside a 2.4 rad angle box:
# * 0.10 rad of equilibrium angle is 6 % of that sweep and 4 % of the box. The measured nominal
#   spans -0.452 to 1.183 rad, so 0.10 rad leaves 0.35 rad of headroom at the low end and 0.42 rad
#   at the high end: the clip is never what limits this residual, the physics is.
# * exp(+-1.2) is 0.30x to 3.32x of stiffness, i.e. 1200 to 13300 N m/rad around the 4000 N m/rad
#   seed, inside the [100, 20000] N m/rad box. Wider than the leg's 0.7 because the ankle nominal
#   is normally a single seeded constant rather than a solved profile, so the residual has to
#   cover more of its own box. The box ceiling stays far below the explicit-integrator limit
#   4 I / dt^2 = 5.9e6 N m/rad at 64 substeps; do not widen it without redoing that arithmetic.
# * 0.3 of damping ratio, the same number as the leg, because the two share the [0.05, 3.0] box.
ACTION_SCALE: tuple[float, ...] = (0.05, 0.7, 0.3, 0.10, 1.2, 0.3)

ACTION_SLEW_FRAMES = 9.0
"""Frames a residual needs, at minimum, to traverse its own full range.

The residual is ramped across each frame, so its rate is bounded by the per-frame change divided
by the frame. Without a limit on that change a single decision can still command a physically
absurd rate: a full swing of ``dL0`` is 0.10 m in 8.3 ms, an equilibrium velocity of 12 m/s that
the leg damper then resists, against a nominal command whose own equilibrium never exceeds
1.425 m/s.

Nine frames is a quarter of the 295 ms measured stance, which still lets a correction reconfigure
the leg between early, mid and late stance. It is chosen because it is the value at which this
rule reproduces the independent rule "the residual may not slew a channel faster than the nominal
command already slews it" on the one channel where that rule is neither degenerate nor academic:
it caps ``dL0`` at 1.33 m/s against the nominal's 1.425 m/s. The nominal stiffness and damping
ratio splines are too flat for that rule to constrain anything (0.17/s of damping ratio, and a
seeded ankle stiffness is constant), so the same nine frames are applied to every channel instead
of inventing a separate number per channel.
"""

ACTION_RATE_LIMIT: tuple[float, ...] = tuple(2.0 * value / ACTION_SLEW_FRAMES for value in ACTION_SCALE)
"""Largest change of each residual per frame, in the units of :data:`ACTION_SCALE`.

(0.0111 m, 0.156, 0.0667, 0.0222 rad, 0.267, 0.0667) per frame, i.e. 1.33 m/s of equilibrium
length, 18.7/s of log stiffness, 8.0/s of damping ratio, 2.67 rad/s of equilibrium angle, 32/s of
ankle log stiffness and 8.0/s of ankle damping ratio.

The APPLIED residual, not the requested one, is what the previous-action observation entries
report, so the limit leaves the decision process Markov: the policy can always see the state its
own rate limit has left it in.
"""

WORK_REWARD_SCALE_J = 70.0
"""Divisor [J] of the dense work reward.

Calibrated so that tier 2 keeps dominating tier 3. The tier 3 range on this rig is about 0 to 530 J
once BOTH actuators are charged: a prescribed reference command spends about 160 J of leg proxy, and
a 20000 N m/rad ankle holding the measured pitch spends about 92 J positive and -3.7 J negative,
which is another 371 J after the 0.25 and 1.20 efficiencies. Dividing by 70 leaves 7.6 reward units
of work available, against the 11.7 units the dense momentum term contributes for an at-tolerance
error, a margin of 1.55x.

That margin is the whole point and it must be preserved. At the previous value of 30, which was
calibrated when only the leg was charged and the range was 0 to 227 J, the ankle charge pushed work
to 17.7 units, ABOVE the 11.7 units of momentum pressure. Task accuracy would have become
purchasable with work again, which is exactly the failure that made the first trained policy cut the
work proxy 6.7x while pushing its momentum excursion from 0.87 to 2.12 tolerances.

If the tier 3 range changes again, for a new shoe or a new actuator, rescale this divisor to hold
the work contribution near 7.6 units rather than adjusting the penalties.
"""

CONTACT_FORCE_FRACTION = 0.02
"""Share of body weight above which the shoe counts as loaded.

One constant for the whole module. The dense momentum reward detects touchdown with it online and
:meth:`ImpedanceEnv._rollout` gates the scored stance with it, which is also the gate
:func:`projects.impedance_instron.optimize.simulate` uses. They must be the same number: a dense
term that starts its stance clock at a different instant from the criterion is measuring a
different quantity, which is exactly the defect this constant was introduced to remove.
"""

MOMENTUM_REWARD_SCALE_M_S = 0.13
"""Divisor [m/s] of the momentum-tracking reward.

Calibrated so tier 2 keeps dominating tier 3, and re-derived after the term was re-anchored on the
DETECTED touchdown. An at-tolerance error of 0.044 m/s costs 0.338 per frame; the term is now active
only from touchdown, so it covers about 35.4 stance frames rather than the roughly 40 assumed
before, giving 12.0 reward units against the 7.6 units of the 0 to 532 J work range at
:data:`WORK_REWARD_SCALE_J`. Margin 1.58x.

Do not read the margin as the whole safeguard. The reward and the tier 2 criterion must also AGREE
IN RANK, and for a long time they did not: the dense term was anchored on the COMMANDED touchdown
while the criterion used each run's own contact interval. With prescribed pitch the two coincided
and the defect was invisible. Once the ankle let contact timing move, measured Spearman correlation
against the criterion was -0.456, and against the clipped excursion it was +0.343, the wrong sign
outright. A policy trained against it ranked best of fourteen episodes on this term and fourth worst
on the criterion, and drove the momentum excursion from 1.079 to 1.951 over 800 iterations while the
reward improved. Anchoring both the datum and the stance clock on the detected touchdown raised the
correlation to -0.912, against -0.965 for a non-causal ideal.
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


@wp.func
def _pitch_of(rotation: wp.quat) -> float:
    """Return the Y-axis rotation angle of a planar fixture pose [rad].

    The same projection :func:`projects.impedance_instron.example._pitch_of` applies: only the Y
    and W components carry a planar pitch, so reading them discards any out-of-plane drift
    instead of propagating it. It is restated here rather than imported because it is private to
    that module and this one may not modify it.
    """
    return 2.0 * wp.atan2(rotation[1], rotation[3])


@wp.kernel
def _prescribe_world_axes(
    index: wp.array[wp.int32],
    reference: wp.array2d[wp.float32],
    ankle: int,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    """Hold every world out of plane, and prescribe its pitch only when no ankle actuator drives it.

    The per-world arithmetic is the arithmetic of
    :func:`projects.impedance_instron.example._constrain_planar_axes` with the two body indices
    offset by the world, so a one-world batch reproduces the single-world rig. With ``ankle``
    nonzero it is instead
    :func:`projects.impedance_instron.example._constrain_out_of_plane_axes`: Y translation, roll
    and yaw are still eliminated, but pitch and pitch rate are integrated states.
    """
    world = wp.tid()
    i = index[0]
    foot = 2 * world
    upper = foot + 1
    a = wp.transform_get_translation(body_q[foot])
    c = wp.transform_get_translation(body_q[upper])
    va = wp.spatial_top(body_qd[foot])
    vc = wp.spatial_top(body_qd[upper])
    pitch = float(0.0)
    pitch_rate = float(0.0)
    if ankle != 0:
        pitch = _pitch_of(wp.transform_get_rotation(body_q[foot]))
        pitch_rate = wp.spatial_bottom(body_qd[foot])[1]
    else:
        pitch = reference[i, 2]
        pitch_rate = reference[i, 7]
    body_q[foot] = wp.transform(
        wp.vec3(a[0], 0.0, a[2]),
        wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), pitch),
    )
    body_q[upper] = wp.transform(wp.vec3(c[0], 0.0, c[2]), wp.quat_identity())
    body_qd[foot] = wp.spatial_vector(wp.vec3(va[0], 0.0, va[2]), wp.vec3(0.0, pitch_rate, 0.0))
    body_qd[upper] = wp.spatial_vector(wp.vec3(vc[0], 0.0, vc[2]), wp.vec3(0.0))


@wp.kernel
def _apply_leg_and_record(
    index: wp.array[wp.int32],
    reference: wp.array2d[wp.float32],
    command: wp.array3d[wp.float32],
    force_limit: float,
    unilateral: int,
    ankle: int,
    torque_limit: float,
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

    With ``ankle`` nonzero the same thread also applies the rotational equilibrium-point law of
    :func:`projects.impedance_instron.example._apply_ankle_impedance` about the fixture pitch
    axis. The leg contributes a pure force and the ankle a pure torque, so fusing them into one
    launch cannot change either result. The ankle source power is recorded separately from its
    damper dissipation and from its stored energy, which must not be added together.
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
    pitch = _pitch_of(wp.transform_get_rotation(body_q[foot]))
    pitch_rate = wp.spatial_bottom(body_qd[foot])[1]
    trace[i, world, 21] = pitch
    trace[i, world, 22] = pitch_rate
    if ankle != 0:
        angle = command[i, world, 5]
        angle_rate = command[i, world, 6]
        k_theta = command[i, world, 7]
        b_theta = command[i, world, 8]
        k_theta_rate = command[i, world, 9]
        angle_error = pitch - angle
        angle_slip = pitch_rate - angle_rate
        torque_raw = -k_theta * angle_error - b_theta * angle_slip
        torque = wp.clamp(torque_raw, -torque_limit, torque_limit)
        wp.atomic_add(body_f, foot, wp.spatial_vector(wp.vec3(0.0), wp.vec3(0.0, torque, 0.0)))
        # The equilibrium-work term carries the PRE-clamp torque, so a limited sample is not
        # credited with work it never did; the clamp shows up in the saturation term instead.
        trace[i, world, 17] = (
            torque_raw * angle_rate
            + 0.5 * k_theta_rate * angle_error * angle_error
            + (torque - torque_raw) * pitch_rate
        )
        trace[i, world, 18] = -b_theta * angle_slip * angle_slip
        trace[i, world, 19] = torque
        trace[i, world, 20] = wp.abs(torque - torque_raw)


def ankle_seed(args, times: np.ndarray, pitch_rad: np.ndarray, inertia_kg_m2: float = 0.025) -> np.ndarray:
    """Return an ankle command vector that holds a measured pitch at a constant impedance.

    A convenience over :meth:`~projects.impedance_instron.control.AnkleCommand.initial` that
    takes the knot counts, the seed stiffness and the seed damping ratio from the same parsed
    namespace the environment is built with, so a caller cannot seed one resolution and simulate
    another. The angle knots are the least-squares fit of the spline basis to ``pitch_rad``; with
    six knots that fit is not exact, which bounds how closely the commanded-angle rollout can
    ever approach the prescribed one. Pass ``args.ankle_equilibrium = "measured"`` to remove the
    fit from the comparison and leave only the finite stiffness.

    Args:
        args: Parsed namespace from
            :func:`projects.impedance_instron.example.create_parser`.
        times: Evaluation grid of the rig [s], shape [sample_count].
        pitch_rad: Measured fixture pitch to reproduce [rad], shape [sample_count].
        inertia_kg_m2: Pitch inertia the commanded damping ratio is referred to [kg·m²].
    """
    command = AnkleCommand(
        times,
        angle_knots=args.ankle_angle_knots,
        stiffness_knots=args.ankle_stiffness_knots,
        damping_knots=args.ankle_damping_knots,
        inertia_kg_m2=inertia_kg_m2,
    )
    return command.initial(np.asarray(pitch_rad, dtype=float), args.ankle_stiffness, args.ankle_damping_ratio)


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
        ankle: np.ndarray | None = None,
    ):
        if int(num_worlds) < 1:
            raise ValueError(f"num_worlds must be at least one, got {num_worlds}")
        self.num_worlds = int(num_worlds)
        self.shape_reward = bool(shape_reward)
        self.rng = np.random.default_rng(seed)
        self.seed = int(seed)
        self.ankle_enabled = ankle is not None

        # A private copy so selecting the equilibrium controller and stamping the nominal command
        # never mutates the caller's namespace; ``Example`` also writes back into ``args``.
        self.args = copy.deepcopy(args)
        self.args.control = "equilibrium"
        self.args.control_vector = np.asarray(nominal, dtype=float).reshape(-1)
        if self.ankle_enabled:
            # Supplying the vector is what switches ``Example`` to the commanded angle spline;
            # a caller that wants the prescribed rollout as the stiff limit sets
            # ``args.ankle_equilibrium = "measured"`` and keeps the angle knots as a seed only.
            self.args.ankle_control = "impedance"
            self.args.ankle_vector = np.asarray(ankle, dtype=float).reshape(-1)
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
        self.torque_limit_n_m = float(getattr(self.args, "ankle_torque_limit", 0.0))
        self.pitch_inertia = float(prototype.pitch_inertia)
        self.touchdown_time_s = float(prototype.registration["touchdown_time_s"])
        self.minimum_last_offsets = np.asarray(prototype.minimum_last_offsets, dtype=float).copy()
        reference = np.ascontiguousarray(prototype.reference, dtype=np.float32)
        self._reference_host = reference.copy()
        self.target = measured_target(prototype.profile, self.mass, self.gravity)
        self.objective = Objective(self.target, Tolerances(), body_weight_n=self.body_weight_n)

        self._build_nominal_command(reference)
        self._build_action_bounds(prototype.command, getattr(prototype, "ankle_command", None))
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
        self._previous_residual = np.zeros((self.num_worlds, self.action_dim), dtype=np.float64)
        self._residual_history = np.zeros((self._episode_frames, self.num_worlds, self.action_dim), dtype=np.float64)
        self._command_host = np.zeros((self.sample_count, self.num_worlds, COMMAND_COLUMNS), dtype=np.float32)
        # Touchdown is DETECTED per world, not assumed: -1 until this world's shoe is first loaded.
        self._contact_sample = np.full(self.num_worlds, -1, dtype=np.int64)
        self._contact_velocity = np.zeros((self.num_worlds, 2), dtype=float)

    # ------------------------------------------------------------------ construction

    def _build_nominal_command(self, reference: np.ndarray) -> None:
        """Cache the nominal leg command the residual acts around, in double precision.

        Columns 13, 14, 25, 26 and 27 of the prototype reference already hold the evaluated
        equilibrium length, its rate, the stiffness, the damping and the stiffness rate. Reusing
        them rather than re-evaluating the spline is what makes a zero residual bit-exact.

        The ankle block reads reference columns
        :data:`~projects.impedance_instron.example.ANKLE_ANGLE` onward the same way, and is left
        at zero when the pitch motor is prescribed.

        Args:
            reference: Prototype reference rows, shape [sample_count, 28] without the ankle
                actuator and [sample_count, :data:`~projects.impedance_instron.example.ANKLE_COLUMN_COUNT`]
                with it.
        """
        self._nominal = np.zeros((self.sample_count, COMMAND_COLUMNS), dtype=np.float64)
        self._nominal[:, :5] = np.column_stack(
            [reference[:, 13], reference[:, 14], reference[:, 25], reference[:, 26], reference[:, 27]]
        )
        stiffness = self._nominal[:, COMMAND_STIFFNESS]
        # b = 2 zeta sqrt(k m) inverted, so the residual can move the ratio the command was
        # written in instead of the raw damper. The inverse is only ever used as a ratio.
        self._nominal_zeta = self._nominal[:, COMMAND_DAMPING] / (2.0 * np.sqrt(stiffness * self.com_mass))
        self._nominal_ankle_zeta = np.zeros(self.sample_count)
        if not self.ankle_enabled:
            return
        if reference.shape[1] < ANKLE_COLUMN_COUNT:
            raise ValueError("The prototype reference carries no ankle block; ankle control did not engage")
        self._nominal[:, COMMAND_ANKLE_ANGLE] = reference[:, ANKLE_ANGLE]
        self._nominal[:, COMMAND_ANKLE_ANGLE_RATE] = reference[:, ANKLE_ANGLE_RATE]
        self._nominal[:, COMMAND_ANKLE_STIFFNESS] = reference[:, ANKLE_STIFFNESS]
        self._nominal[:, COMMAND_ANKLE_DAMPING] = reference[:, ANKLE_DAMPING]
        self._nominal[:, COMMAND_ANKLE_STIFFNESS_RATE] = reference[:, ANKLE_STIFFNESS_RATE]
        ankle_stiffness = self._nominal[:, COMMAND_ANKLE_STIFFNESS]
        self._nominal_ankle_zeta = self._nominal[:, COMMAND_ANKLE_DAMPING] / (
            2.0 * np.sqrt(ankle_stiffness * self.pitch_inertia)
        )

    def _build_action_bounds(self, command: LegCommand, ankle: AnkleCommand | None) -> None:
        """Read the resolved-command boxes straight out of the two command classes.

        Args:
            command: The prototype's leg command, whose knot bounds are also profile bounds
                because a clamped B-spline stays in the convex hull of its coefficients.
            ankle: The prototype's ankle command, or None when pitch is prescribed. Its bounds
                are read the same way, so the residual can never resolve an ankle stiffness,
                equilibrium angle or damping ratio outside :meth:`AnkleCommand.bounds`.
        """
        lower, upper = command.bounds()
        first = command.length_knots
        second = first + command.stiffness_knots
        self.length_bounds_m = (float(lower[0]), float(upper[0]))
        self.stiffness_bounds_n_m = (float(np.exp(lower[first])), float(np.exp(upper[first])))
        self.damping_ratio_bounds = (float(lower[second]), float(upper[second]))
        self.action_scale = np.asarray(ACTION_SCALE[: self.action_dim], dtype=float)
        self.action_rate_limit = np.asarray(ACTION_RATE_LIMIT[: self.action_dim], dtype=float)
        self.angle_bounds_rad = (0.0, 0.0)
        self.ankle_stiffness_bounds_n_m_per_rad = (0.0, 0.0)
        self.ankle_damping_ratio_bounds = (0.0, 0.0)
        if ankle is None:
            return
        lower, upper = ankle.bounds()
        first = ankle.angle_knots
        second = first + ankle.stiffness_knots
        self.angle_bounds_rad = (float(lower[0]), float(upper[0]))
        self.ankle_stiffness_bounds_n_m_per_rad = (float(np.exp(lower[first])), float(np.exp(upper[first])))
        self.ankle_damping_ratio_bounds = (float(lower[second]), float(upper[second]))

    def _build_momentum_reference(self) -> None:
        """Anchor the measured momentum history at touchdown so it can be read at any phase.

        The tier 2 history is stated at nine fractions of contact, the first at 10 %. Reading it
        below that fraction by clamping would claim the body should already have gained the
        velocity it gains in the first tenth of stance, a constant penalty in the frames right
        after touchdown. Velocity change at touchdown is zero by definition, which is also how
        :func:`projects.impedance_instron.optimize.simulate` builds the history it is compared
        against, so the zero is prepended explicitly.
        """
        self._momentum_phase = np.concatenate([[0.0], CHECKPOINTS])
        self._momentum_vx = np.concatenate([[0.0], np.asarray(self.target.momentum_vx_m_s, dtype=float)])
        self._momentum_vz = np.concatenate([[0.0], np.asarray(self.target.momentum_vz_m_s, dtype=float)])

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
        if self.ankle_enabled:
            # Nothing writes pitch during the rollout any more, so the entry angle and angular
            # rate have to be part of the declared initial state, exactly as ``Example`` declares
            # them for its single world.
            angle = float(self._reference_host[0, _REFERENCE_PITCH])
            angle_rate = float(self._reference_host[0, _REFERENCE_PITCH_RATE])
            self._initial_q[0::2, 3:7] = (0.0, math.sin(0.5 * angle), 0.0, math.cos(0.5 * angle))
            self._initial_qd[0::2, 4] = angle_rate

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
        # Share the prototype's declared ground plane and contact law in every world.
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
        """Width of one observation row; see :data:`OBSERVATION_LAYOUT`.

        The ankle actuator appends :data:`ANKLE_OBSERVATION_LAYOUT`, so the width is 13 with a
        prescribed pitch motor and 16 with a commanded ankle impedance.
        """
        return len(self.observation_layout)

    @property
    def observation_layout(self) -> tuple[tuple[str, str, float], ...]:
        """Observation entries of this environment, in index order."""
        if self.ankle_enabled:
            return OBSERVATION_LAYOUT + ANKLE_OBSERVATION_LAYOUT
        return OBSERVATION_LAYOUT

    @property
    def action_dim(self) -> int:
        """Width of one action row.

        Three with a prescribed pitch motor, ``[dL0, dlogK, dzeta]``. Six with the ankle
        actuator, ``[dL0, dlogK, dzeta, dtheta0, dlogK_theta, dzeta_theta]``.
        """
        return 6 if self.ankle_enabled else 3

    @property
    def episode_frames(self) -> int:
        """Frames of one stance episode, one policy decision each."""
        return self._episode_frames

    @property
    def reference(self) -> np.ndarray:
        """Measured reference the rig is driven against, read-only.

        Shape [sample_count, 28] with a prescribed pitch motor, and
        [sample_count, :data:`~projects.impedance_instron.example.ANKLE_COLUMN_COUNT`] with the
        ankle actuator, whose commanded angle, angle rate, stiffness, damping and stiffness rate
        occupy the appended block.

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
        self._contact_sample[:] = -1
        self._contact_velocity[:] = 0.0
        self._frame_index = 0
        self._started = True
        # Sample 0 is recorded before any decision, so it is driven by the nominal command.
        zero = np.zeros((1, self.num_worlds, self.action_dim))
        self._write_command(np.array([0]), zero, np.zeros((self.num_worlds, self.action_dim)))
        self._record_sample()
        return self._observe(0)

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Advance one frame in every world. Returns obs, reward, done, info.

        The action of frame ``f`` holds over substeps ``64 f + 1`` to ``64 f + 64``: the decision
        is taken from the observation at the frame boundary and is constant inside the frame,
        which is what lets the captured CUDA graph replay the whole frame.

        Args:
            actions: Raw policy output, shape [num_worlds, :attr:`action_dim`]. It is squashed
                with ``tanh`` and scaled by :data:`ACTION_SCALE` before it is added to the
                nominal command.

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
        if command.shape != (self.num_worlds, self.action_dim):
            raise ValueError(f"actions must have shape {(self.num_worlds, self.action_dim)}, got {command.shape}")
        if not np.all(np.isfinite(command)):
            raise ValueError("actions must be finite")

        requested = np.tanh(command) * self.action_scale
        # Rate limit first, ramp second: the limit bounds the frame-to-frame change and the ramp
        # spreads what survives across the frame, so no commanded rate depends on the substep.
        residual = np.clip(
            requested,
            self._previous_residual - self.action_rate_limit,
            self._previous_residual + self.action_rate_limit,
        )
        frame = self._frame_index
        start = frame * self.substeps
        rows = np.arange(start + 1, start + self.substeps + 1)
        ramp, rate = self._frame_ramp(residual)
        self._write_command(rows, ramp, rate)
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
            # Exposed so the ankle actuator can be charged for its work: an unpenalised pitch
            # motor is a free resource and a policy will spend it on everything.
            episode = self.trace_device.numpy()
            power = episode[:, :, TRACE_ANKLE_SOURCE_POWER].astype(np.float64)
            info["ankle_positive_work_j"] = np.trapezoid(np.clip(power, 0.0, None), self.times, axis=0)
            info["ankle_negative_work_j"] = np.trapezoid(np.clip(power, None, 0.0), self.times, axis=0)
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
            the per-frame residual actually applied. With the ankle actuator the resolved
            equilibrium angle [rad] and its rate [rad/s], ankle stiffness [N·m/rad], ankle
            damping [N·m·s/rad] and ankle damping ratio [-] are reported alongside them, so a
            dashboard can plot both actuators from one call.
        """
        if not 0 <= int(world) < self.num_worlds:
            raise IndexError(f"world {world} is outside a batch of {self.num_worlds}")
        filled = min(self._frame_index * self.substeps + 1, self.sample_count)
        block = self._command_host[:filled, int(world)].astype(float)
        stiffness = block[:, COMMAND_STIFFNESS]
        realised = {
            "time_s": self.times[:filled].copy(),
            "length_m": block[:, COMMAND_L0],
            "length_rate_m_s": block[:, COMMAND_L0_RATE],
            "stiffness_n_m": stiffness,
            "damping_n_s_m": block[:, COMMAND_DAMPING],
            "damping_ratio": block[:, COMMAND_DAMPING] / (2.0 * np.sqrt(stiffness * self.com_mass)),
            "stiffness_rate_n_m_s": block[:, COMMAND_STIFFNESS_RATE],
            "residual": self._residual_history[: self._frame_index, int(world)].copy(),
        }
        if not self.ankle_enabled:
            return realised
        ankle_stiffness = block[:, COMMAND_ANKLE_STIFFNESS]
        ankle_damping = block[:, COMMAND_ANKLE_DAMPING]
        realised.update(
            {
                "angle_rad": block[:, COMMAND_ANKLE_ANGLE],
                "angle_rate_rad_s": block[:, COMMAND_ANKLE_ANGLE_RATE],
                "ankle_stiffness_n_m_per_rad": ankle_stiffness,
                "ankle_damping_n_m_s_per_rad": ankle_damping,
                "ankle_damping_ratio": ankle_damping / (2.0 * np.sqrt(ankle_stiffness * self.pitch_inertia)),
                "ankle_stiffness_rate_n_m_per_rad_s": block[:, COMMAND_ANKLE_STIFFNESS_RATE],
            }
        )
        return realised

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

    def _frame_ramp(self, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return one frame's per-substep residual ramp and the rate that produced it.

        The policy decides once per frame, but applying that decision as a STEP makes the
        commanded impedance piecewise constant with one discontinuity per frame, which defeats
        the point of the C2 spline parameterization in
        :mod:`projects.impedance_instron.control` and, worse, turns ``kdot`` into a spike whose
        size is set by the substep rather than by any physical rate. Ramping linearly from the
        previous frame's resolved residual to this one's over the frame's substeps is causal,
        because both endpoints are known when the frame begins, and bounds every rate by
        ``delta / frame_dt``. It is a first-order hold: the policy's request is fully applied only
        at the end of its own frame.

        Args:
            target: Residual the policy asked for this frame, shape [num_worlds, action_dim].
        """
        weights = (np.arange(1, self.substeps + 1, dtype=float) / self.substeps)[:, None, None]
        previous = self._previous_residual[None]
        ramp = previous + weights * (target[None] - previous)
        return ramp, (target - self._previous_residual) / self.frame_dt

    @staticmethod
    def _clip_with_rate(
        value: np.ndarray, rate: np.ndarray, bounds: tuple[float, float]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Clip a commanded signal and return the derivative of the CLIPPED signal.

        A clipped command is constant while it is held at a bound, so reporting the unclipped
        rate there would credit the energy ledger with a stiffness change that never happened.
        The one-sided test keeps the rate that moves the command back inside the box, which is
        the true derivative at the bound rather than an approximation of it.

        Args:
            value: Unclipped command.
            rate: Time derivative of the unclipped command.
            bounds: ``(minimum, maximum)`` of the command box.
        """
        low, high = bounds
        clipped = np.clip(value, low, high)
        effective = np.where(value >= high, np.minimum(rate, 0.0), np.where(value <= low, np.maximum(rate, 0.0), rate))
        return clipped, effective

    def _write_command(self, rows: np.ndarray, residual: np.ndarray, rate: np.ndarray) -> None:
        """Resolve the nominal command plus a ramped residual onto the given substep rows.

        The resolved stiffness is ``k exp(dlogK)`` clipped into the LegCommand box, so it is
        positive by construction. The damping follows it through
        ``b = b_nominal (zeta / zeta_nominal) sqrt(k / k_nominal)``, which is algebraically the
        same ``b = 2 zeta sqrt(k m)`` the command was written with but leaves a zero residual
        exactly equal to the nominal float32.

        Every commanded RATE is now the derivative of the command actually written, not the
        nominal one. The equilibrium rate carries the ramp's own rate, and the stiffness rate is
        the product rule ``kdot_nominal exp(dlogK) + k d(dlogK)/dt``. A stepped residual made both
        of these wrong: the command moved while its reported rate claimed it had not, and the
        ``0.5 kdot e^2`` term of the source power became an artefact of the decision rate.

        The ankle block is resolved by the identical algebra against
        :meth:`AnkleCommand.bounds`, with the pitch inertia in place of the leg mass, so a zero
        residual reproduces the commanded ankle impedance bit for bit as well.

        Args:
            rows: Substep indices to fill, shape [n].
            residual: Resolved residual per substep and world, shape [n, num_worlds, action_dim].
            rate: Time derivative of that residual, shape [num_worlds, action_dim].
        """
        nominal = self._nominal[rows]
        length = nominal[:, COMMAND_L0][:, None]
        length_rate = nominal[:, COMMAND_L0_RATE][:, None]
        stiffness = nominal[:, COMMAND_STIFFNESS][:, None]
        damping = nominal[:, COMMAND_DAMPING][:, None]
        stiffness_rate = nominal[:, COMMAND_STIFFNESS_RATE][:, None]
        zeta = self._nominal_zeta[rows][:, None]
        resolved_length, resolved_length_rate = self._clip_with_rate(
            length + residual[:, :, 0], length_rate + rate[None, :, 0], self.length_bounds_m
        )
        gain = np.exp(residual[:, :, 1])
        resolved_stiffness, resolved_stiffness_rate = self._clip_with_rate(
            stiffness * gain,
            stiffness_rate * gain + stiffness * gain * rate[None, :, 1],
            self.stiffness_bounds_n_m,
        )
        resolved_zeta = np.clip(zeta + residual[:, :, 2], *self.damping_ratio_bounds)
        block = np.empty((len(rows), self.num_worlds, COMMAND_COLUMNS), dtype=np.float32)
        block[:, :, COMMAND_L0] = resolved_length
        block[:, :, COMMAND_L0_RATE] = resolved_length_rate
        block[:, :, COMMAND_STIFFNESS] = resolved_stiffness
        block[:, :, COMMAND_DAMPING] = damping * (resolved_zeta / zeta) * np.sqrt(resolved_stiffness / stiffness)
        block[:, :, COMMAND_STIFFNESS_RATE] = resolved_stiffness_rate
        if self.ankle_enabled:
            angle = nominal[:, COMMAND_ANKLE_ANGLE][:, None]
            angle_rate = nominal[:, COMMAND_ANKLE_ANGLE_RATE][:, None]
            ankle_stiffness = nominal[:, COMMAND_ANKLE_STIFFNESS][:, None]
            ankle_damping = nominal[:, COMMAND_ANKLE_DAMPING][:, None]
            ankle_stiffness_rate = nominal[:, COMMAND_ANKLE_STIFFNESS_RATE][:, None]
            ankle_zeta = self._nominal_ankle_zeta[rows][:, None]
            resolved_angle, resolved_angle_rate = self._clip_with_rate(
                angle + residual[:, :, 3], angle_rate + rate[None, :, 3], self.angle_bounds_rad
            )
            ankle_gain = np.exp(residual[:, :, 4])
            resolved_ankle_stiffness, resolved_ankle_stiffness_rate = self._clip_with_rate(
                ankle_stiffness * ankle_gain,
                ankle_stiffness_rate * ankle_gain + ankle_stiffness * ankle_gain * rate[None, :, 4],
                self.ankle_stiffness_bounds_n_m_per_rad,
            )
            resolved_ankle_zeta = np.clip(ankle_zeta + residual[:, :, 5], *self.ankle_damping_ratio_bounds)
            block[:, :, COMMAND_ANKLE_ANGLE] = resolved_angle
            block[:, :, COMMAND_ANKLE_ANGLE_RATE] = resolved_angle_rate
            block[:, :, COMMAND_ANKLE_STIFFNESS] = resolved_ankle_stiffness
            block[:, :, COMMAND_ANKLE_DAMPING] = (
                ankle_damping * (resolved_ankle_zeta / ankle_zeta) * np.sqrt(resolved_ankle_stiffness / ankle_stiffness)
            )
            block[:, :, COMMAND_ANKLE_STIFFNESS_RATE] = resolved_ankle_stiffness_rate
        else:
            block[:, :, COMMAND_ANKLE_ANGLE:] = 0.0
        first, last = int(rows[0]), int(rows[-1]) + 1
        self._command_host[first:last] = block
        self.command_device[first:last].assign(block)

    def _record_sample(self) -> None:
        """Load the foundation and the legs and record one substep in every world."""
        wp.launch(
            _prescribe_world_axes,
            dim=self.num_worlds,
            inputs=[
                self.index_device,
                self.reference_device,
                int(self.ankle_enabled),
                self.state_0.body_q,
                self.state_0.body_qd,
            ],
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
                int(self.ankle_enabled),
                self.torque_limit_n_m,
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

        Pitch and pitch rate come from the trace, not from the reference schedule. While the
        pitch motor is prescribed the two are the same number up to the float32 round trip
        through the pose quaternion, about 1e-7 rad; once the ankle actuator drives pitch they
        are different quantities and only the achieved one is observable.

        ``stance_phase`` and ``in_contact`` report this world's own detected touchdown, the clock
        both the dense momentum reward and the tier 2 momentum excursion are measured on. The
        policy is memoryless, so without them it cannot tell how far into its own stance it is
        and cannot represent a phase-dependent command at all.

        Args:
            sample: Substep index at the frame boundary.
        """
        row = self.trace_device[sample].numpy().astype(np.float64)
        episode_phase = (sample / self.substeps - self.touchdown_time_s / self.frame_dt) / self._episode_frames
        active = self._contact_sample >= 0
        elapsed = self.times[sample] - self.times[np.maximum(self._contact_sample, 0)]
        stance_phase = np.where(active, np.clip(elapsed / max(self.target.duration_s, 1.0e-9), 0.0, 1.0), 0.0)
        observation = np.empty((self.num_worlds, self.observation_dim), dtype=np.float64)
        observation[:, 0] = (row[:, TRACE_LEG_LENGTH] - 1.0) / OBSERVATION_LAYOUT[0][2]
        observation[:, 1] = row[:, TRACE_LEG_RATE] / OBSERVATION_LAYOUT[1][2]
        observation[:, 2] = row[:, TRACE_SHOE_FZ] / self.body_weight_n
        observation[:, 3] = row[:, TRACE_SHOE_FX] / self.body_weight_n
        observation[:, 4] = row[:, TRACE_PITCH] / OBSERVATION_LAYOUT[4][2]
        observation[:, 5] = row[:, TRACE_PITCH_RATE] / OBSERVATION_LAYOUT[5][2]
        observation[:, 6] = row[:, TRACE_ANKLE_Z] / OBSERVATION_LAYOUT[6][2]
        observation[:, 7] = row[:, TRACE_ANKLE_VZ] / OBSERVATION_LAYOUT[7][2]
        observation[:, 8] = self._com_vz(row) / OBSERVATION_LAYOUT[8][2]
        observation[:, 9] = episode_phase
        observation[:, 10] = stance_phase
        observation[:, 11] = active
        observation[:, 12:] = self._previous_residual / self.action_scale
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

        Both actuators are charged, because :meth:`Objective.work_proxy_j` charges both and tier 3
        is paid out here rather than at the terminal frame. Charging only the leg would leave the
        pitch actuator free in the reward even though the verdict charges it, and a free actuator
        is one a policy spends without limit. Each actuator is integrated separately before the
        efficiencies are applied, so the sum over frames is still exactly the episode proxy. With
        a prescribed pitch motor the ankle column is identically zero and the leg-only reward is
        unchanged bit for bit.

        Args:
            window: Trace rows of the frame including both endpoints, shape
                [substeps + 1, num_worlds, :data:`TRACE_COLUMNS`].
            start: Substep index of the first row of the window.
        """
        times = self.times[start : start + window.shape[0]]
        proxy = self._charge(window[:, :, TRACE_SOURCE_POWER], times)
        if self.objective.charge_ankle:
            proxy = proxy + self._charge(window[:, :, TRACE_ANKLE_SOURCE_POWER], times)
        return -proxy / WORK_REWARD_SCALE_J

    def _charge(self, power: np.ndarray, times: np.ndarray) -> np.ndarray:
        """Charge one actuator's work over a window at the objective's two efficiencies.

        Args:
            power: Source power of one actuator [W], shape [samples, num_worlds].
            times: Sample times of the window [s], shape [samples].
        """
        source = power.astype(np.float64)
        positive = np.trapezoid(np.clip(source, 0.0, None), times, axis=0)
        negative = np.trapezoid(np.clip(source, None, 0.0), times, axis=0)
        return positive / self.objective.positive_efficiency + np.abs(negative) / self.objective.negative_efficiency

    def _momentum_reward(self, window: np.ndarray, start: int) -> np.ndarray:
        """Return the optional momentum-tracking reward of one frame, per world.

        The clock and the datum are this world's OWN touchdown, detected online from the traced
        shoe force with :data:`CONTACT_FORCE_FRACTION`, never the commanded touchdown instant.
        That is the whole point of this method's current form. The tier 2 momentum excursion is
        measured on the run's own contact interval, so a dense term anchored on the commanded
        instant measures a different quantity: a policy can improve it by landing earlier while
        the excursion it is supposed to predict gets worse. Measured on a trained six-dimensional
        ankle policy, the commanded anchor ranked that policy BEST of fourteen episodes on the
        dense term and fourth WORST on the criterion, a Spearman rank correlation of +0.30
        against the criterion; anchoring on the detected touchdown raises it to +0.91.

        It stays causal. Touchdown is in the past once it is detected, the stance clock is
        normalized by the MEASURED stance duration rather than by this run's own duration, which
        is not known until the episode ends, and stance duration is separately graded by its own
        tier 2 tolerance. Before its own touchdown a world scores zero, because no stance has
        started to track.

        Args:
            window: Trace rows of the frame including both endpoints.
            start: Substep index of the first row of the window.
        """
        if not self.shape_reward:
            return np.zeros(self.num_worlds)
        self._detect_touchdown(window, start)
        active = self._contact_sample >= 0
        if not active.any():
            return np.zeros(self.num_worlds)
        share = self.foot_mass / self.mass
        row = window[-1].astype(np.float64)
        vx = share * row[:, TRACE_ANKLE_VX] + (1.0 - share) * row[:, TRACE_UPPER_VX]
        end = start + window.shape[0] - 1
        elapsed = self.times[end] - self.times[np.maximum(self._contact_sample, 0)]
        phase = np.clip(elapsed / max(self.target.duration_s, 1.0e-9), 0.0, 1.0)
        reference_vx = np.interp(phase, self._momentum_phase, self._momentum_vx)
        reference_vz = np.interp(phase, self._momentum_phase, self._momentum_vz)
        error = np.abs(vx - self._contact_velocity[:, 0] - reference_vx) + np.abs(
            self._com_vz(row) - self._contact_velocity[:, 1] - reference_vz
        )
        return -np.where(active, error, 0.0) / MOMENTUM_REWARD_SCALE_M_S

    def _detect_touchdown(self, window: np.ndarray, start: int) -> None:
        """Record each world's first loaded substep and the COM velocity it landed with.

        The scan covers the whole frame, so touchdown is resolved to the substep the force
        crossed :data:`CONTACT_FORCE_FRACTION` of body weight rather than to the frame boundary,
        which is what the tier 2 excursion resolves it to as well. A world is scanned only while
        it has no touchdown yet, so a shoe that leaves the ground and lands again keeps the datum
        of the stance it is being scored on.

        Args:
            window: Trace rows of the frame including both endpoints.
            start: Substep index of the first row of the window.
        """
        pending = self._contact_sample < 0
        if not pending.any():
            return
        loaded = window[:, :, TRACE_SHOE_FZ] > CONTACT_FORCE_FRACTION * self.body_weight_n
        found = pending & loaded.any(axis=0)
        if not found.any():
            return
        worlds = np.nonzero(found)[0]
        local = np.argmax(loaded[:, worlds], axis=0)
        rows = window[local, worlds].astype(np.float64)
        share = self.foot_mass / self.mass
        self._contact_sample[worlds] = start + local
        self._contact_velocity[worlds, 0] = share * rows[:, TRACE_ANKLE_VX] + (1.0 - share) * rows[:, TRACE_UPPER_VX]
        self._contact_velocity[worlds, 1] = share * rows[:, TRACE_ANKLE_VZ] + (1.0 - share) * rows[:, TRACE_UPPER_VZ]

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
        ankle_power = rows[:, TRACE_ANKLE_SOURCE_POWER]
        share = self.foot_mass / self.mass
        com_vx = share * rows[:, TRACE_ANKLE_VX] + (1.0 - share) * rows[:, TRACE_UPPER_VX]
        com_vz = share * rows[:, TRACE_ANKLE_VZ] + (1.0 - share) * rows[:, TRACE_UPPER_VZ]
        loaded = rows[:, TRACE_SHOE_FZ] > CONTACT_FORCE_FRACTION * self.body_weight_n
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
            # Negative work stays NEGATIVE. :meth:`Objective.ankle_work_proxy_j` takes the
            # absolute value itself and charges the two halves at different efficiencies, so
            # pre-absing here would silently overcharge nothing and undercharge the sign test.
            ankle_positive_work_j=float(np.trapezoid(np.clip(ankle_power, 0.0, None), times))
            if finite
            else float("nan"),
            ankle_negative_work_j=float(np.trapezoid(np.clip(ankle_power, None, 0.0), times))
            if finite
            else float("nan"),
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
    parser.add_argument(
        "--ankle",
        action="store_true",
        help=(
            "Also drive foot pitch with a commanded rotational impedance seeded from the measured "
            "pitch, which widens the action to six dimensions and reports how closely the stiff "
            "limit reproduces the prescribed pitch motor."
        ),
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
    acceptance check of the environment: with the prescribed pitch motor the printed differences
    are the float32 rounding of a deterministic rig against a trace stored before the foundation
    reduction was made deterministic, not a modelling error.

    With ``--ankle`` the same episode also runs with foot pitch driven by a commanded rotational
    impedance seeded to hold the measured pitch. The printed pitch difference is how far the
    ankle actuator sits from the prescribed motor it is the stiff limit of; it cannot reach zero
    from inside the ankle command box, whose ceiling is a finite 2e4 N·m/rad.
    """
    args = _create_parser().parse_args()
    nominal = np.asarray(json.loads(args.nominal.read_text())["parameters"], dtype=float)
    stored = _reference_trace(args.reference_trace)
    ankle, prescribed_pitch = None, None
    if args.ankle:
        probe = ImpedanceEnv(1, args, nominal)
        probe.reset()
        for _ in range(probe.episode_frames):
            probe.step(np.zeros((1, probe.action_dim)))
        prescribed_pitch = probe.trace(0)[:, TRACE_PITCH].copy()
        # The measured pitch is the equilibrium the prescribed motor imposed, so seeding from it
        # keeps the prescribed rollout as the stiff limit of the ankle impedance.
        args.ankle_equilibrium = "measured"
        ankle = ankle_seed(args, probe.times, probe.reference[:, _REFERENCE_PITCH].astype(float))
        del probe
    for worlds in args.worlds:
        env = ImpedanceEnv(worlds, args, nominal, ankle=ankle)
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
        if prescribed_pitch is not None:
            difference = np.abs(env.trace(0)[:, TRACE_PITCH] - prescribed_pitch).max()
            print(
                f"             ankle stiff limit at {env.realised_command(0)['ankle_stiffness_n_m_per_rad'].max():.0f} "
                f"N m/rad: pitch {difference:.3e} rad "
                f"({difference / np.ptp(prescribed_pitch):.2e} of the prescribed pitch span)"
            )
        del env


if __name__ == "__main__":
    main()
