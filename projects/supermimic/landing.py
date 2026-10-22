# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Controlled landings of the SuperMimic runner with substep logs.

A :class:`LandingRig` drives the worlds of a :class:`~projects.supermimic.sim.RunnerSim`
substep by substep with an open joint torque law

    ``tau = tau_ff + K (q_eq - q) - B qd``,

where ``q_eq`` and the feed-forward ``tau_ff`` follow 1 kHz tables.
:func:`reference_schedule` builds the tables that track a reference: ``tau_ff``
carries ``B qd_ref``, so the damping acts on the velocity error. From the
substep a foot's vertical ground reaction first exceeds
:data:`~projects.supermimic.reference.CONTACT_THRESHOLD_N`, the five joints of that leg use the touchdown gains
for :attr:`LandingSchedule.touchdown_hold_s`. The rig records every substep:
ground reaction, the stiffness, damping and feed-forward torque terms, the
joint-limit constraint torque, joint state, foot pose and velocity, the largest
held column strain and the number of columns pressing on the belt.

Worlds stay in the belt frame (no treadmill shift), so foot velocities are
velocities over the belt.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from .body import BODY_NAMES, JOINT_NAMES
from .reference import CONTACT_THRESHOLD_N
from .sim import COORDS, DOFS, FOOT_BODIES, RunnerSim

__all__ = ["LandingRig", "LandingSchedule", "reference_schedule", "touchdown_starts"]

_J = wp.constant(len(JOINT_NAMES))
_COORDS = wp.constant(COORDS)
_DOFS = wp.constant(DOFS)
_BODIES = wp.constant(len(BODY_NAMES))
_FOOT_R = wp.constant(FOOT_BODIES[0])
_FOOT_L = wp.constant(FOOT_BODIES[1])


@dataclass(frozen=True)
class LandingSchedule:
    """Torque law of every world. Joint arrays follow :data:`.body.JOINT_NAMES`."""

    equilibrium: np.ndarray
    """Equilibrium angles ``q_eq`` per 1 ms [world_count, rows, 10] [rad]; the last row holds."""
    feed_forward: np.ndarray
    """Feed-forward torques ``tau_ff`` per 1 ms [world_count, rows, 10] [N m]; the last row holds."""
    stiffness: np.ndarray
    """Joint stiffness ``K`` [world_count, 10] [N m/rad]."""
    damping: np.ndarray
    """Joint damping ``B`` [world_count, 10] [N m s/rad]."""
    touchdown_stiffness: np.ndarray
    """Stiffness of a leg's joints after its touchdown [world_count, 10] [N m/rad]."""
    touchdown_damping: np.ndarray
    """Damping of a leg's joints after its touchdown [world_count, 10] [N m s/rad]."""
    touchdown_hold_s: float
    """Time after a leg's touchdown during which it uses the touchdown gains [s]."""


@wp.kernel
def _actuate(
    k: int,
    dt: float,
    substeps_per_row: int,
    hold: float,
    equilibrium: wp.array3d[float],
    feed_forward: wp.array3d[float],
    gains: wp.array3d[float],
    touchdown: wp.array2d[float],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    joint_f: wp.array[float],
    k_term: wp.array3d[float],
    b_term: wp.array3d[float],
    ff_term: wp.array3d[float],
    q: wp.array3d[float],
    qd: wp.array3d[float],
):
    w, j = wp.tid()
    row = wp.min(k / substeps_per_row, equilibrium.shape[1] - 1)
    t = float(k) * dt
    t0 = touchdown[w, j / 5]
    g = 0
    if t0 >= 0.0 and t < t0 + hold:
        g = 2
    angle = joint_q[w * _COORDS + 7 + j]
    rate = joint_qd[w * _DOFS + 6 + j]
    spring = gains[w, j, g] * (equilibrium[w, row, j] - angle)
    damper = -gains[w, j, g + 1] * rate
    ff = feed_forward[w, row, j]
    joint_f[w * _DOFS + 6 + j] = ff + spring + damper
    k_term[w, k, j] = spring
    b_term[w, k, j] = damper
    ff_term[w, k, j] = ff
    q[w, k, j] = angle
    qd[w, k, j] = rate


@wp.kernel
def _record_bed(
    k: int,
    dt: float,
    side: int,
    threshold: float,
    worlds: wp.array[wp.int32],
    feet: wp.array[wp.int32],
    body_f: wp.array[wp.spatial_vector],
    normal_force: wp.array[wp.float32],
    held_strain: wp.array[wp.float32],
    touchdown: wp.array2d[float],
    grf: wp.array4d[float],
    strain: wp.array3d[float],
):
    i = wp.tid()
    w = worlds[i]
    f = wp.spatial_top(body_f[feet[i]])
    r = float(normal_force[i])
    grf[w, k, side, 0] = f[0]
    grf[w, k, side, 1] = f[1]
    grf[w, k, side, 2] = r
    strain[w, k, side] = float(held_strain[i])
    if r > threshold and touchdown[w, side] < 0.0:
        touchdown[w, side] = float(k) * dt


@wp.kernel
def _count_columns(
    k: int,
    side: int,
    column_count: int,
    worlds: wp.array[wp.int32],
    reaction: wp.array[wp.float32],
    columns: wp.array3d[wp.int32],
):
    i = wp.tid()
    if reaction[i] > 0.0:
        wp.atomic_add(columns, worlds[i / column_count], k, side, 1)


@wp.kernel
def _record_world(
    k: int,
    joint_qd: wp.array[float],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    root_qd: wp.array3d[float],
    foot_q: wp.array3d[wp.transform],
    foot_qd: wp.array3d[wp.spatial_vector],
):
    w = wp.tid()
    for i in range(6):
        root_qd[w, k, i] = joint_qd[w * _DOFS + i]
    foot_q[w, k, 0] = body_q[w * _BODIES + _FOOT_R]
    foot_q[w, k, 1] = body_q[w * _BODIES + _FOOT_L]
    foot_qd[w, k, 0] = body_qd[w * _BODIES + _FOOT_R]
    foot_qd[w, k, 1] = body_qd[w * _BODIES + _FOOT_L]


@wp.kernel
def _record_limits(
    k: int,
    dof_map: wp.array2d[wp.int32],
    constraint: wp.array2d[float],
    limit: wp.array3d[float],
):
    w, d = wp.tid()
    dof = dof_map[w, d] - w * _DOFS - 6
    if dof >= 0:
        limit[w, k, dof] = constraint[w, d]


class LandingRig:
    """Run and record controlled landings on a :class:`RunnerSim`.

    Args:
        sim: Simulator whose worlds, shoes and materials the landings use.
        duration_s: Simulated time per landing [s].
    """

    def __init__(self, sim: RunnerSim, duration_s: float):
        self.sim = sim
        self.steps = round(duration_s / sim.config.substep_dt_s)
        self.substeps_per_row = round(1e-3 / sim.config.substep_dt_s)
        w, t, j = sim.world_count, self.steps, len(JOINT_NAMES)
        d = sim.device
        self._touchdown = wp.zeros((w, 2), dtype=float, device=d)
        self._logs = {
            "grf": wp.zeros((w, t, 2, 3), dtype=float, device=d),
            "held_strain": wp.zeros((w, t, 2), dtype=float, device=d),
            "loaded_columns": wp.zeros((w, t, 2), dtype=wp.int32, device=d),
            "k_torque": wp.zeros((w, t, j), dtype=float, device=d),
            "b_torque": wp.zeros((w, t, j), dtype=float, device=d),
            "ff_torque": wp.zeros((w, t, j), dtype=float, device=d),
            "limit_torque": wp.zeros((w, t, j), dtype=float, device=d),
            "q": wp.zeros((w, t, j), dtype=float, device=d),
            "qd": wp.zeros((w, t, j), dtype=float, device=d),
            "root_qd": wp.zeros((w, t, 6), dtype=float, device=d),
            "foot_q": wp.zeros((w, t, 2), dtype=wp.transform, device=d),
            "foot_qd": wp.zeros((w, t, 2), dtype=wp.spatial_vector, device=d),
        }

    def run(self, joint_q: np.ndarray, joint_qd: np.ndarray, schedule: LandingSchedule) -> dict[str, np.ndarray]:
        """Reset every world to a treadmill-frame state and run one landing.

        Args:
            joint_q: Generalized coordinates [world_count, 17].
            joint_qd: Generalized velocities in the treadmill frame [world_count, 16].
            schedule: Torque law.

        Returns:
            Per-substep logs ``[world_count, steps, ...]``: ``grf`` [N] (per foot friction
            x, y and ground reaction z), ``held_strain``, ``loaded_columns``,
            ``k_torque``, ``b_torque``, ``ff_torque`` and ``limit_torque`` [N m],
            ``q`` [rad], ``qd`` [rad/s], ``root_qd`` (pelvis center-of-mass velocity
            over the belt [m/s], angular velocity [rad/s]), ``foot_q`` (position,
            quaternion xyzw) and ``foot_qd`` (center-of-mass velocity over the belt
            [m/s], angular velocity [rad/s]), all in world axes; ``touchdown_s`` [world_count, 2],
            -1 for a foot that did not touch down; ``dt`` [s].
        """
        sim = self.sim
        d = sim.device
        cfg = sim.config
        dt = cfg.substep_dt_s
        gains = np.stack(
            (schedule.stiffness, schedule.damping, schedule.touchdown_stiffness, schedule.touchdown_damping), -1
        )
        equilibrium = wp.array(np.asarray(schedule.equilibrium, np.float32), dtype=float, device=d)
        feed_forward = wp.array(np.asarray(schedule.feed_forward, np.float32), dtype=float, device=d)
        gains = wp.array(gains.astype(np.float32), dtype=float, device=d)
        mask = wp.ones(sim.world_count, dtype=bool, device=d)
        sim.reset(
            mask,
            wp.array(np.asarray(joint_q, np.float32), dtype=float, device=d),
            wp.array(np.asarray(joint_qd, np.float32), dtype=float, device=d),
        )
        self._touchdown.fill_(-1.0)
        for log in self._logs.values():
            log.zero_()
        logs = self._logs
        data = sim.solver.mjw_data
        dof_map = sim.solver.mjc_dof_to_newton_dof
        s0, s1 = sim.state_0, sim.state_1
        for k in range(self.steps):
            wp.launch(
                _actuate,
                dim=(sim.world_count, len(JOINT_NAMES)),
                inputs=[
                    k,
                    dt,
                    self.substeps_per_row,
                    schedule.touchdown_hold_s,
                    equilibrium,
                    feed_forward,
                    gains,
                    self._touchdown,
                    s0.joint_q,
                    s0.joint_qd,
                ],
                outputs=[
                    sim.control.joint_f,
                    logs["k_torque"],
                    logs["b_torque"],
                    logs["ff_torque"],
                    logs["q"],
                    logs["qd"],
                ],
                device=d,
            )
            wp.launch(
                _record_world,
                dim=sim.world_count,
                inputs=[k, s0.joint_qd, s0.body_q, s0.body_qd],
                outputs=[logs["root_qd"], logs["foot_q"], logs["foot_qd"]],
                device=d,
            )
            sim.update_foot_mobility(s0)
            for bed in sim.beds:
                bed.bed.apply(s0)
                columns = bed.bed.bed
                wp.launch(
                    _record_bed,
                    dim=len(bed.worlds),
                    inputs=[
                        k,
                        dt,
                        bed.side,
                        CONTACT_THRESHOLD_N,
                        bed.world_index,
                        bed.feet,
                        s0.body_f,
                        columns.normal_force,
                        columns.held_strain,
                    ],
                    outputs=[self._touchdown, logs["grf"], logs["held_strain"]],
                    device=d,
                )
                wp.launch(
                    _count_columns,
                    dim=len(bed.worlds) * columns.column_count,
                    inputs=[k, bed.side, columns.column_count, bed.world_index, columns.reaction],
                    outputs=[logs["loaded_columns"]],
                    device=d,
                )
            sim.solver.step(s0, s1, sim.control, None, dt)
            wp.launch(
                _record_limits,
                dim=dof_map.shape,
                inputs=[k, dof_map, data.qfrc_constraint],
                outputs=[logs["limit_torque"]],
                device=d,
            )
            s0, s1 = s1, s0
        # RunnerSim.step replays a graph bound to its own state arrays.
        if s0 is not sim.state_0:
            sim.state_0.assign(s0)
        result = {name: log.numpy() for name, log in logs.items()}
        result["touchdown_s"] = self._touchdown.numpy()
        result["dt"] = dt
        return result


def reference_schedule(
    joint_q: np.ndarray,
    joint_qd: np.ndarray,
    moment: np.ndarray,
    frame_dt: float,
    starts: np.ndarray,
    duration_s: float,
    stiffness: np.ndarray,
    damping: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, LandingSchedule]:
    """Return the start states and the reference-tracking schedule of landings from reference frames.

    From each start the equilibrium angles are the reference angles ``q_ref`` and
    the feed-forward torques are ``M_ref + B qd_ref``, both linearly interpolated
    to 1 ms rows. At the reference state the torque law then gives ``M_ref``, and
    the damping acts on the velocity error ``qd_ref - qd`` instead of braking
    the reference motion. The feed-forward holds ``B``, so a leg keeps its gains
    at touchdown.

    Args:
        joint_q: Reference generalized coordinates [windows, frames, 17].
        joint_qd: Reference generalized velocities [windows, frames, 16].
        moment: Reference inverse-dynamics joint moments [windows, frames, 10] [N m].
        frame_dt: Reference frame period [s].
        starts: Start rows ``(window, frame, ...)`` [worlds, >= 2], for example from :func:`touchdown_starts`.
        duration_s: Schedule length [s].
        stiffness: Joint stiffness ``K`` [worlds, 10] [N m/rad].
        damping: Joint damping ``B`` [worlds, 10] [N m s/rad].

    Returns:
        Start ``joint_q`` [worlds, 17], start ``joint_qd`` [worlds, 16] and the schedule.
    """
    window, frame = starts[:, 0], starts[:, 1]
    frames = np.arange(joint_q.shape[1])
    position = frame[:, None] + np.arange(round(duration_s * 1000.0) + 1) * 1e-3 / frame_dt

    def rows(x: np.ndarray) -> np.ndarray:
        return np.stack(
            [
                np.stack([np.interp(p, frames, xw[:, j]) for j in range(xw.shape[1])], -1)
                for p, xw in zip(position, x[window], strict=True)
            ]
        )

    schedule = LandingSchedule(
        equilibrium=rows(joint_q[..., 7:]),
        feed_forward=rows(moment) + damping[:, None] * rows(joint_qd[..., 6:]),
        stiffness=stiffness,
        damping=damping,
        touchdown_stiffness=stiffness,
        touchdown_damping=damping,
        touchdown_hold_s=0.0,
    )
    return joint_q[window, frame], joint_qd[window, frame], schedule


def touchdown_starts(contact: np.ndarray, flight_window: np.ndarray, flight_frame: np.ndarray) -> np.ndarray:
    """Return the last flight frame before each reference touchdown.

    Args:
        contact: Reference foot contact [windows, frames, 2].
        flight_window: Window of each episode start.
        flight_frame: Frame of each episode start; both feet are off the plate around it.

    Returns:
        Rows ``(window, frame, side)`` [touchdowns, 3]: the start frame and the
        foot (0 right, 1 left) that touches down after it.
    """
    starts = set(zip(flight_window.tolist(), flight_frame.tolist(), strict=True))
    rows = []
    for w, f in sorted(starts):
        later = contact[w, f + 1 :]
        landed = np.flatnonzero(later.any(-1))
        if len(landed) == 0 or (w, f + 1) in starts:
            continue
        rows.append((w, f, int(np.argmax(later[landed[0]]))))
    return np.asarray(rows, dtype=np.int64).reshape(-1, 3)
