# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Batched SuperMimic runner worlds on Digital Shoe column beds.

Each world is the :mod:`.body` articulation integrated by
:class:`newton.solvers.SolverMuJoCo` with one
:class:`~projects.digital_shoe.column_bed.ColumnBed` under each foot. Every
world wears one shoe pair of the dataset; one bed covers the worlds that wear a
shoe on one side, with its columns placed rigidly in the foot frame. The bed is
the only ground contact.

An action sets, per world and actuated joint, an equilibrium angle, a
stiffness ``K`` and a damping ``B``. The commands pass a first-order filter and
the joint torque ``K (q_eq - q) - B qd`` is applied at every substep.

The treadmill is simulated in the belt frame. A belt at constant speed is an
inertial frame, so a runner on it obeys the same physics as one running over
still ground at the belt speed, and the column bed's ground stays at rest. After
every control step each world moves back by the belt travel, so positions stay
in the treadmill frame; the public velocities are treadmill-frame velocities.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp

import newton
from projects.digital_shoe.artifact import DigitalShoe, load_artifact
from projects.digital_shoe.column_bed import TOP_BONDED, ColumnBed
from projects.digital_shoe.runtime import FoundationConfig, ShoeMaterial

from .body import BODY_NAMES, JOINT_NAMES, BodyParams, add_body

__all__ = ["RunnerSim", "RunnerSimConfig", "Shoe", "load_shoes"]

COORDS = 7 + len(JOINT_NAMES)
DOFS = 6 + len(JOINT_NAMES)
SIDES = ("right", "left")
FOOT_BODIES = (BODY_NAMES.index("foot_r"), BODY_NAMES.index("foot_l"))
_J = wp.constant(len(JOINT_NAMES))
_COORDS = wp.constant(COORDS)
_DOFS = wp.constant(DOFS)
_BODIES = wp.constant(len(BODY_NAMES))
_MIRROR = np.diag([1.0, -1.0, 1.0])


@dataclass(frozen=True)
class Shoe:
    """A fitted shoe pair placed on the foot as in the dataset manifest."""

    name: str
    left: DigitalShoe
    right: DigitalShoe
    mount_m: tuple[float, float, float]
    """Ankle center in the left shoe frame [m]; the right shoe is its mirror."""
    rotation: np.ndarray
    """Registration rotation of the left shoe onto the measured shoe's frame; identity for the measured shoe."""

    def anchors_in_foot(self, side: str, sole_pitch: float) -> np.ndarray:
        """Return the nominal column bottoms of ``side``'s shoe in the foot frame [m]."""
        bed = (self.right if side == "right" else self.left).column_bed
        return self._in_foot(bed.anchor_bottom_m, side, sole_pitch)

    def mass_in_foot(self, side: str, sole_pitch: float) -> tuple[np.ndarray, np.ndarray]:
        """Return the shoe mass of ``side`` as point masses in the foot frame.

        The shoe mass is spread over the columns in proportion to their volume,
        each at its column's mid-height.

        Returns:
            Point masses [kg], shape ``[column_count]``, and their positions [m], shape ``[column_count, 3]``.
        """
        artifact = self.right if side == "right" else self.left
        bed = artifact.column_bed
        volume = bed.area_m2 * bed.rest_length_m
        mid = bed.anchor_bottom_m + np.outer(0.5 * bed.rest_length_m, (0.0, 0.0, 1.0))
        return artifact.mass_kg * volume / volume.sum(), self._in_foot(mid, side, sole_pitch)

    def _in_foot(self, points: np.ndarray, side: str, sole_pitch: float) -> np.ndarray:
        mirror = _MIRROR if side == "right" else np.eye(3)
        mount = mirror @ np.asarray(self.mount_m)
        c, s = np.cos(sole_pitch), np.sin(sole_pitch)
        pitch = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
        foot_from_shoe = pitch @ (mirror @ self.rotation @ mirror).T
        return (points - mount) @ foot_from_shoe.T


def load_shoes(dataset: str | Path) -> list[Shoe]:
    """Load every shoe pair of a column-bed dataset in manifest order."""
    dataset = Path(dataset)
    records = json.loads((dataset / "manifest.json").read_text())["shoes"]
    return [
        Shoe(
            name=record["name"],
            left=load_artifact(dataset / record["left"]),
            right=load_artifact(dataset / record["right"]),
            mount_m=tuple(record["mount_m"]),
            rotation=np.asarray(record.get("registration", {}).get("rotation", np.eye(3)), dtype=float),
        )
        for record in records
    ]


@dataclass(frozen=True)
class RunnerSimConfig:
    """Integration, actuation and contact settings."""

    substep_dt_s: float = 1.25e-4
    """Physics and contact step [s]."""
    control_dt_s: float = 0.01
    """Interval between actions [s]; a multiple of two substeps."""
    command_time_constant_s: float = 0.01
    """Time constant of the first-order filter on equilibrium, stiffness and damping [s]."""
    belt_speed_m_s: float = 3.70
    """Treadmill belt speed along -x of the treadmill frame [m/s]; F01 measured speed."""
    fine_dt_s: float = 1e-3
    """Interval of the ground reaction samples in ``grf_fine`` [s], as the 1 kHz force plate."""
    friction_coefficient: float = 0.8
    """Coulomb friction between outsole and belt."""


@wp.kernel
def _actuate(
    action: wp.array3d[float],
    alpha: float,
    dt: float,
    scale: float,
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    command: wp.array3d[float],
    fresh: wp.array2d[wp.int32],
    joint_f: wp.array[float],
    torque: wp.array2d[float],
    power: wp.array2d[float],
    work_pos: wp.array2d[float],
    work_neg: wp.array2d[float],
):
    """Filter the commands, apply the impedance torque of one joint and add it to the step outputs."""
    w, j = wp.tid()
    for k in range(3):
        if fresh[w, j] != 0:
            command[w, j, k] = action[w, j, k]
        else:
            command[w, j, k] += alpha * (action[w, j, k] - command[w, j, k])
    fresh[w, j] = 0
    q = joint_q[w * _COORDS + 7 + j]
    qd = joint_qd[w * _DOFS + 6 + j]
    tau = command[w, j, 1] * (command[w, j, 0] - q) - command[w, j, 2] * qd
    joint_f[w * _DOFS + 6 + j] = tau
    p = tau * qd
    torque[w, j] += tau * scale
    power[w, j] += p * scale
    work_pos[w, j] += wp.max(p, 0.0) * dt
    work_neg[w, j] += wp.min(p, 0.0) * dt


@wp.kernel
def _add_grf(
    side: int,
    belt_offset: float,
    fine: int,
    scale: float,
    fine_scale: float,
    worlds: wp.array[wp.int32],
    feet: wp.array[wp.int32],
    body_f: wp.array[wp.spatial_vector],
    ground_reaction: wp.array[wp.float32],
    pressure_center: wp.array[wp.vec2],
    held_strain: wp.array[wp.float32],
    grf: wp.array3d[float],
    cop: wp.array3d[float],
    grf_fine: wp.array4d[float],
    strain: wp.array2d[float],
):
    """Add one bed's friction, ground reaction and pressure center to its worlds' step outputs."""
    i = wp.tid()
    w = worlds[i]
    f = wp.spatial_top(body_f[feet[i]])
    r = float(ground_reaction[i])
    c = pressure_center[i]
    force = wp.vec3(f[0], f[1], r)
    for k in range(3):
        grf[w, side, k] += force[k] * scale
        grf_fine[w, fine, side, k] += force[k] * fine_scale
    cop[w, side, 0] += r * (float(c[0]) - belt_offset) * scale
    cop[w, side, 1] += r * float(c[1]) * scale
    strain[w, side] = wp.max(strain[w, side], float(held_strain[i]))


@wp.kernel
def _finish(
    belt_travel: float,
    belt_speed: float,
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    body_q: wp.array[wp.transform],
    grf: wp.array3d[float],
    q_out: wp.array2d[float],
    qd_out: wp.array2d[float],
    cop: wp.array3d[float],
):
    """Move a world back by the belt travel, copy its state and normalize its pressure centers."""
    w = wp.tid()
    joint_q[w * _COORDS] -= belt_travel
    for b in range(_BODIES):
        pose = body_q[w * _BODIES + b]
        p = wp.transform_get_translation(pose)
        body_q[w * _BODIES + b] = wp.transform(wp.vec3(p[0] - belt_travel, p[1], p[2]), wp.transform_get_rotation(pose))
    for i in range(_COORDS):
        q_out[w, i] = joint_q[w * _COORDS + i]
    for i in range(_DOFS):
        qd_out[w, i] = joint_qd[w * _DOFS + i]
    qd_out[w, 0] -= belt_speed
    for s in range(2):
        total = grf[w, s, 2]
        if total > 0.0:
            cop[w, s, 0] = cop[w, s, 0] / total
            cop[w, s, 1] = cop[w, s, 1] / total
        else:
            cop[w, s, 0] = 0.0
            cop[w, s, 1] = 0.0


@wp.kernel
def _reset(
    mask: wp.array[bool],
    q: wp.array2d[float],
    qd: wp.array2d[float],
    belt_speed: float,
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    q_out: wp.array2d[float],
    qd_out: wp.array2d[float],
    fresh: wp.array2d[wp.int32],
):
    """Set the state of a masked world and let its next action pass the filter unchanged."""
    w = wp.tid()
    if not mask[w]:
        return
    for i in range(_COORDS):
        joint_q[w * _COORDS + i] = q[w, i]
        q_out[w, i] = q[w, i]
    for i in range(_DOFS):
        joint_qd[w * _DOFS + i] = qd[w, i]
        qd_out[w, i] = qd[w, i]
    joint_qd[w * _DOFS] += belt_speed
    for j in range(_J):
        fresh[w, j] = 1


@wp.kernel
def _gather(mask: wp.array[bool], worlds: wp.array[wp.int32], bed_mask: wp.array[bool]):
    i = wp.tid()
    bed_mask[i] = mask[worlds[i]]


class _Bed:
    """One shoe on one side over the worlds that wear it."""

    def __init__(self, sim: RunnerSim, shoe: Shoe, side: str, worlds: np.ndarray):
        artifact = shoe.right if side == "right" else shoe.left
        bed = artifact.column_bed
        count = len(bed.rest_length_m)
        d = sim.device
        self.side = SIDES.index(side)
        self.worlds = worlds
        self.world_index = wp.array(worlds, dtype=wp.int32, device=d)
        self.feet = wp.array(worlds * len(BODY_NAMES) + FOOT_BODIES[self.side], dtype=wp.int32, device=d)
        self.mask = wp.zeros(len(worlds), dtype=bool, device=d)
        self.bed = ColumnBed(
            shoe.anchors_in_foot(side, sim.body.sole_pitch),
            bed.rest_length_m,
            bed.area_m2,
            bed.neighbors,
            bed.spacing_m,
            np.full(count, TOP_BONDED, dtype=np.int32),
            artifact.material,
            FoundationConfig(ground_height_m=0.0, mu=sim.config.friction_coefficient),
            self.feet.numpy(),
            sim.model.body_com,
            device=d,
        )
        self.bed.set_timesteps(sim.config.substep_dt_s)


class RunnerSim:
    """Batched runner worlds, each wearing one shoe pair.

    Args:
        body: Runner geometry and inertia.
        shoes: Shoe pairs, from :func:`load_shoes`.
        shoe_of_world: Index into ``shoes`` per world.
        config: Integration, actuation and contact settings.
        device: CUDA device.

    After :meth:`step` the attributes below hold the results in the treadmill
    frame. Per-foot arrays are ordered right, left; joint arrays follow
    :data:`.body.JOINT_NAMES`.

    - ``joint_q`` [m, rad], ``[world_count, 17]``: root position, root
      quaternion ``xyzw``, joint angles.
    - ``joint_qd`` [m/s, rad/s], ``[world_count, 16]``: pelvis center-of-mass
      velocity and angular velocity in world axes, joint rates.
    - ``grf`` [N], ``[world_count, 2, 3]``: outsole friction ``x, y`` and ground
      reaction ``z``, mean over the step.
    - ``grf_fine`` [N], ``[world_count, steps, 2, 3]``: ``grf`` as means over
      consecutive ``fine_dt_s`` intervals of the step.
    - ``cop`` [m], ``[world_count, 2, 3]``: center of pressure on the belt,
      weighted by the ground reaction over the step; zero without contact.
    - ``held_strain``, ``[world_count, 2]``: largest column strain held during
      the step.
    - ``joint_torque`` [N m] and ``joint_power`` [W], ``[world_count, 10]``: means
      over the step.
    - ``joint_work_pos`` and ``joint_work_neg`` [J], ``[world_count, 10]``:
      positive and negative joint work over the step; ``joint_work_neg <= 0``.
    """

    def __init__(
        self,
        body: BodyParams,
        shoes: Sequence[Shoe],
        shoe_of_world: Sequence[int],
        *,
        config: RunnerSimConfig | None = None,
        device=None,
    ):
        self.body = body
        self.config = config = config or RunnerSimConfig()
        self.device = d = wp.get_device(device)
        self.shoe_of_world = np.asarray(shoe_of_world, dtype=np.int32)
        self.world_count = w = len(self.shoe_of_world)
        self.substeps = round(config.control_dt_s / config.substep_dt_s)
        self.fine_substeps = round(config.fine_dt_s / config.substep_dt_s)
        if (
            self.substeps % 2
            or self.substeps % self.fine_substeps
            or not np.isclose(self.substeps * config.substep_dt_s, config.control_dt_s)
            or not np.isclose(self.fine_substeps * config.substep_dt_s, config.fine_dt_s)
        ):
            raise ValueError("control_dt_s must be an even multiple of fine_dt_s, a multiple of substep_dt_s")

        template = newton.ModelBuilder(up_axis=newton.Axis.Z)
        add_body(template, body)
        scene = newton.ModelBuilder(up_axis=newton.Axis.Z)
        for _ in range(w):
            scene.add_world(template)
        self.model = scene.finalize(device=d)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.solver = newton.solvers.SolverMuJoCo(
            self.model, integrator="euler", jacobian="dense", disable_contacts=True
        )
        self._add_shoe_masses(shoes)
        self.beds = [
            _Bed(self, shoes[s], side, np.flatnonzero(self.shoe_of_world == s).astype(np.int32))
            for s in np.unique(self.shoe_of_world)
            for side in SIDES
        ]

        joints = len(JOINT_NAMES)
        self.action = wp.zeros((w, joints, 3), dtype=float, device=d)
        self._command = wp.zeros_like(self.action)
        self._fresh = wp.ones((w, joints), dtype=wp.int32, device=d)
        self.joint_q = wp.zeros((w, COORDS), dtype=float, device=d)
        self.joint_qd = wp.zeros((w, DOFS), dtype=float, device=d)
        self.grf = wp.zeros((w, 2, 3), dtype=float, device=d)
        self.grf_fine = wp.zeros((w, self.substeps // self.fine_substeps, 2, 3), dtype=float, device=d)
        self.cop = wp.zeros((w, 2, 3), dtype=float, device=d)
        self.held_strain = wp.zeros((w, 2), dtype=float, device=d)
        self.joint_torque = wp.zeros((w, joints), dtype=float, device=d)
        self.joint_power = wp.zeros_like(self.joint_torque)
        self.joint_work_pos = wp.zeros_like(self.joint_torque)
        self.joint_work_neg = wp.zeros_like(self.joint_torque)
        with wp.ScopedCapture(device=d) as capture:
            self._simulate()
        self._graph = capture.graph

    def _add_shoe_masses(self, shoes: Sequence[Shoe]) -> None:
        """Add each world's shoe mass to its feet; the columns stay massless."""
        model = self.model
        mass, com, inertia = (a.numpy() for a in (model.body_mass, model.body_com, model.body_inertia))
        for s, shoe in enumerate(shoes):
            for side, foot in zip(SIDES, FOOT_BODIES, strict=True):
                points, positions = shoe.mass_in_foot(side, self.body.sole_pitch)
                bodies = np.flatnonzero(self.shoe_of_world == s) * len(BODY_NAMES) + foot
                if not len(bodies):
                    continue
                m0, c0, i0 = mass[bodies[0]], com[bodies[0]].astype(float), inertia[bodies[0]].astype(float)
                total = m0 + points.sum()
                center = (m0 * c0 + points @ positions) / total
                r0 = c0 - center
                r = positions - center
                i0 = i0 + m0 * (r0 @ r0 * np.eye(3) - np.outer(r0, r0))
                i0 = i0 + np.einsum(
                    "n,nij->ij",
                    points,
                    np.einsum("nk,nk->n", r, r)[:, None, None] * np.eye(3) - r[:, :, None] * r[:, None, :],
                )
                mass[bodies], com[bodies], inertia[bodies] = total, center, i0
        model.body_mass.assign(mass)
        model.body_inv_mass.assign(1.0 / mass)
        model.body_com.assign(com)
        model.body_inertia.assign(inertia)
        model.body_inv_inertia.assign(np.linalg.inv(inertia))
        self.solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)

    def set_world_materials(self, materials: Sequence[ShoeMaterial], friction: Sequence[float]) -> None:
        """Set the foam material and the outsole friction coefficient of both shoes per world."""
        if len(materials) != self.world_count or len(friction) != self.world_count:
            raise ValueError(f"Expected {self.world_count} materials and friction coefficients")
        for bed in self.beds:
            bed.bed.set_world_materials([materials[i] for i in bed.worlds], friction=[friction[i] for i in bed.worlds])

    def reset(self, mask: wp.array[bool], joint_q: wp.array2d[float], joint_qd: wp.array2d[float]) -> None:
        """Set the masked worlds to a treadmill-frame state with unloaded shoes.

        Args:
            mask: Worlds to reset, shape ``[world_count]``.
            joint_q: Generalized coordinates [m, rad], shape ``[world_count, 17]``; only masked rows are read.
            joint_qd: Generalized velocities [m/s, rad/s], shape ``[world_count, 16]``; only masked rows are read.
        """
        state = self.state_0
        wp.launch(
            _reset,
            dim=self.world_count,
            inputs=[mask, joint_q, joint_qd, self.config.belt_speed_m_s],
            outputs=[state.joint_q, state.joint_qd, self.joint_q, self.joint_qd, self._fresh],
            device=self.device,
        )
        newton.eval_fk(self.model, state.joint_q, state.joint_qd, state, mask=mask)
        for bed in self.beds:
            wp.launch(
                _gather, dim=len(bed.worlds), inputs=[mask, bed.world_index], outputs=[bed.mask], device=self.device
            )
            bed.bed.reset(bed.mask)

    def step(self, action: wp.array3d[float]) -> None:
        """Advance every world one control step.

        Args:
            action: Per world and actuated joint the equilibrium angle [rad],
                stiffness [N m/rad] and damping [N m s/rad], shape ``[world_count, 10, 3]``.
        """
        wp.copy(self.action, action)
        wp.capture_launch(self._graph)

    def _simulate(self) -> None:
        cfg = self.config
        dt = cfg.substep_dt_s
        alpha = -float(np.expm1(-dt / cfg.command_time_constant_s))
        scale = 1.0 / self.substeps
        for array in (
            self.grf,
            self.grf_fine,
            self.cop,
            self.held_strain,
            self.joint_torque,
            self.joint_power,
            self.joint_work_pos,
            self.joint_work_neg,
        ):
            array.zero_()
        for k in range(self.substeps):
            s0, s1 = self.state_0, self.state_1
            wp.launch(
                _actuate,
                dim=(self.world_count, len(JOINT_NAMES)),
                inputs=[self.action, alpha, dt, scale, s0.joint_q, s0.joint_qd, self._command, self._fresh],
                outputs=[
                    self.control.joint_f,
                    self.joint_torque,
                    self.joint_power,
                    self.joint_work_pos,
                    self.joint_work_neg,
                ],
            )
            for bed in self.beds:
                bed.bed.apply(s0)
                wp.launch(
                    _add_grf,
                    dim=len(bed.worlds),
                    inputs=[
                        bed.side,
                        cfg.belt_speed_m_s * k * dt,
                        k // self.fine_substeps,
                        scale,
                        1.0 / self.fine_substeps,
                        bed.world_index,
                        bed.feet,
                        s0.body_f,
                        bed.bed.bed.normal_force,
                        bed.bed.bed.pressure_center,
                        bed.bed.bed.held_strain,
                    ],
                    outputs=[self.grf, self.cop, self.grf_fine, self.held_strain],
                )
            self.solver.step(s0, s1, self.control, None, dt)
            self.state_0, self.state_1 = s1, s0
        wp.launch(
            _finish,
            dim=self.world_count,
            inputs=[cfg.belt_speed_m_s * cfg.control_dt_s, cfg.belt_speed_m_s],
            outputs=[
                self.state_0.joint_q,
                self.state_0.joint_qd,
                self.state_0.body_q,
                self.grf,
                self.joint_q,
                self.joint_qd,
                self.cop,
            ],
        )
