# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Differentiable stride and attached midsole gait scenarios.

Extends the differentiable elastic-foundation midsole
(:class:`~projects.digital_instron_v2.dynamics_diff.DifferentiableMidsoleFoundation`)
from the quasi-static Instron drop to the two dynamic gait scenarios of the
shipped example (:mod:`projects.digital_instron_v2.example`), recording the
ground-reaction force (GRF) on one :class:`warp.Tape` so it can be
differentiated with respect to the foam material:

* :class:`DifferentiableStride` -- a *kinematic* heel-to-toe roll. The carrier
  pose is prescribed by :func:`~projects.digital_instron_v2.dynamics.synthetic_stride`
  at every substep (no solver), so the GRF is a pure feed-forward function of the
  foam material. Reproduces the shipped forward model's GRF exactly, passive
  surround included: the shoe last presses its own footprint and the rest of the
  midsole relaxes every substep, as in the shipped ``stride`` example.

* :class:`DifferentiableAttached` -- a *fully dynamic* foot-mounted shoe with
  mass and inertia, held to the prescribed foot trajectory by a damped PD upper
  and integrated by :class:`~newton.solvers.SolverSemiImplicit`. The GRF is
  differentiated through the whole rigid-body loop -- contact, PD actuation, and
  anchored bristle friction -- which the quasi-static
  :func:`~projects.digital_instron_v2.core.predict` cannot do.

* :class:`DifferentiableSlide` -- a *kinematic* constant-velocity lateral drag at
  fixed penetration. It records the net shear force so the Coulomb friction
  coefficient ``mu`` can be identified by gradient descent from a lateral-force
  target, the clean (continuous-contact, non-zero-crossing) counterpart to the
  normal-force scenarios above.

Both dynamic drivers also record the per-substep net lateral shear on the tape
(``self.shear``) and expose the differentiable friction coefficient
(``self.friction_params``), so friction identification composes with the foam
material fit.

The drivers reduce the foundation's write-once per-substep force history instead
of evaluating another contact law or reading an overwritten diagnostic. The
attached driver also uses a counter-free PD kernel (:func:`_attach_pd`). The
gradient is piecewise exact away from contact and bristle branch transitions.
At a transition, the tape differentiates the selected branch; a finite difference
that changes branches need not agree.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import warp as wp

import newton
from projects.digital_shoe.contact import contact_kinematics

from .dynamics import (
    FoundationConfig,
    MidsoleFoundation,
    SurroundConfig,
    attach_coupling,
    build_foundation_geometry,
    load_fitted_material,
    synthetic_stride,
)
from .dynamics_diff import (
    FRIC_MU,
    MAT_G_EQ,
    MAT_G_EQ2,
    DifferentiableMidsoleFoundation,
    _column_normal_force,
    _legacy_smooth_friction,
    _pasternak_flux,
)


@wp.kernel
def _attach_pd(
    body: wp.int32,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    target: wp.array[wp.transform],
    target_vel: wp.array[wp.spatial_vector],
    kp_lin: wp.float32,
    kd_lin: wp.float32,
    kp_ang: wp.float32,
    kd_ang: wp.float32,
    max_force: wp.float32,
    body_f: wp.array[wp.spatial_vector],
):
    """Tape-safe damped-PD "shoe upper" for one substep.

    Same control law as
    :func:`~projects.digital_instron_v2.dynamics.attach_coupling` (a damped PD
    that is slack in flight and stiff against the blocked ground in stance), but
    reads the current substep's target pose/velocity directly from length-1
    arrays instead of advancing an on-device counter, so it has no side effect and
    is safe to record on a :class:`warp.Tape`.
    """
    target_pos = wp.transform_get_translation(target[0])
    target_rot = wp.transform_get_rotation(target[0])
    pos = wp.transform_get_translation(body_q[body])
    rot = wp.transform_get_rotation(body_q[body])

    e_p = target_pos - pos
    q_err = target_rot * wp.quat_inverse(rot)
    if q_err[3] < 0.0:
        q_err = wp.quat(-q_err[0], -q_err[1], -q_err[2], -q_err[3])
    e_r = 2.0 * wp.vec3(q_err[0], q_err[1], q_err[2])

    v = wp.spatial_top(body_qd[body])
    w = wp.spatial_bottom(body_qd[body])
    tv = wp.spatial_top(target_vel[0])
    tw = wp.spatial_bottom(target_vel[0])

    force = kp_lin * e_p + kd_lin * (tv - v)
    moment = kp_ang * e_r + kd_ang * (tw - w)
    mag = wp.length(force)
    if mag > max_force and mag > 1.0e-9:
        force = force * (max_force / mag)
    wp.atomic_add(body_f, body, wp.spatial_vector(force, moment))


@wp.kernel
def _ground_reaction_force(
    carrier: wp.int32,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    anchor_local: wp.array[wp.vec3],
    area: wp.array[wp.float32],
    rest_len: wp.array[wp.float32],
    neighbors: wp.array2d[wp.int32],
    compression: wp.array[wp.float32],
    base_pressure: wp.array[wp.float32],
    material_params: wp.array[wp.float32],
    normal_damping: wp.float32,
    substep: wp.int32,
    grf_hist: wp.array[wp.float32],
):
    """Sum the vertical column ground-reaction force for one substep into ``grf_hist[substep]``.

    Re-derives the normal force from the per-substep compression/pressure history
    so the total GRF is written exactly once per substep (unlike the foundation's
    single overwritten diagnostic), keeping every value on the loss path
    un-aliased across the rollout. The mechanics are
    :func:`~projects.digital_instron_v2.dynamics_diff._column_normal_force`, the
    one differentiable transcription of the runtime contact law, so this readout
    reports the same reaction the wrench applies.
    """
    i = wp.tid()
    # The shear layer follows the series modulus, the sum of both Ogden-Hill terms.
    mu_eq = material_params[MAT_G_EQ] + material_params[MAT_G_EQ2]
    ci = compression[i]
    flux = _pasternak_flux(i, compression, rest_len, neighbors, mu_eq)

    _world, _com_world, point_vel, _gap = contact_kinematics(
        body_q[carrier], body_qd[carrier], body_com[carrier], anchor_local[i], 0.0, 0
    )
    fn = _column_normal_force(ci, base_pressure[i], area[i], flux, normal_damping, point_vel[2])
    wp.atomic_add(grf_hist, substep, fn)


@wp.kernel
def _shear_reaction_force(
    carrier: wp.int32,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    anchor_local: wp.array[wp.vec3],
    area: wp.array[wp.float32],
    rest_len: wp.array[wp.float32],
    neighbors: wp.array2d[wp.int32],
    compression: wp.array[wp.float32],
    base_pressure: wp.array[wp.float32],
    material_params: wp.array[wp.float32],
    friction_params: wp.array[wp.float32],
    friction_smoothing: wp.float32,
    normal_damping: wp.float32,
    substep: wp.int32,
    shear_hist: wp.array[wp.vec2],
):
    """Preserve the deprecated stateless shear readout signature.

    The old smooth-friction behavior remains only during deprecation. Active
    scenarios reduce the shared bristle force history with
    :func:`_record_shear_force` and never call this compatibility kernel.
    """
    i = wp.tid()
    mu_eq = material_params[MAT_G_EQ] + material_params[MAT_G_EQ2]
    ci = compression[i]
    flux = _pasternak_flux(i, compression, rest_len, neighbors, mu_eq)

    _world, _com_world, point_vel, _gap = contact_kinematics(
        body_q[carrier], body_qd[carrier], body_com[carrier], anchor_local[i], 0.0, 0
    )
    fn = _column_normal_force(ci, base_pressure[i], area[i], flux, normal_damping, point_vel[2])
    pressed = wp.max(fn, 0.0)

    f_tan = _legacy_smooth_friction(
        wp.vec2(point_vel[0], point_vel[1]), pressed, friction_params[FRIC_MU], friction_smoothing
    )
    wp.atomic_add(shear_hist, substep, f_tan)


@wp.kernel
def _record_normal_force(column_force: wp.array[wp.vec3], substep: wp.int32, grf_hist: wp.array[wp.float32]):
    """Reduce the applied force history without evaluating another contact law."""
    wp.atomic_add(grf_hist, substep, column_force[wp.tid()][2])


@wp.kernel
def _record_shear_force(column_force: wp.array[wp.vec3], substep: wp.int32, shear_hist: wp.array[wp.vec2]):
    """Reduce the applied bristle shear history on the tape."""
    force = column_force[wp.tid()]
    wp.atomic_add(shear_hist, substep, wp.vec2(force[0], force[1]))


@wp.kernel
def _drag_impulse(shear: wp.array[wp.vec2], out: wp.array[wp.float32]):
    """Accumulate the streamwise (-x) drag component of the per-substep shear into ``out[0]``."""
    wp.atomic_add(out, 0, -shear[wp.tid()][0])


@wp.kernel
def _reduce_sum(values: wp.array[wp.float32], out: wp.array[wp.float32]):
    """Accumulate ``values`` into ``out[0]`` (a differentiable GRF impulse reduction)."""
    wp.atomic_add(out, 0, values[wp.tid()])


# Passive-surround settings shared with the shipped example and the identification
# (:class:`projects.digital_instron_v2.core.Surround`): the assumed vertical bond of
# untouched foam to the shoe, its compression limit, and the sweeps per substep. The
# field is warm started from the previous substep, so a few sweeps track the converged
# quasi-static surround the fit solves with 250 sweeps from zero.
# The outer bond is booked consistently by being switched off: its reaction was
# used inside the relaxation but never reported, which made it a hidden support.
SURROUND_ATTACHMENT_N_M = 0.0
SURROUND_MAX_STRAIN = 0.9
SURROUND_SWEEPS = 32


def default_surround(driven: np.ndarray, *, carrier_bond: bool) -> SurroundConfig:
    """Return the shipped example's passive-surround settings for a driven mask.

    Args:
        driven: Nonzero where the carrier drives the column, shape ``[column_count]``.
        carrier_bond: True when the untouched column tops are glued under the rigid
            carrier (a shod runtime) instead of being a free shoe surface the
            carrier never touches (a bench indenter).

    Returns:
        The :class:`~projects.digital_instron_v2.dynamics.SurroundConfig` the
        shipped forward model uses for the same scenario.
    """
    return SurroundConfig(
        driven=driven,
        attachment_n_m=SURROUND_ATTACHMENT_N_M,
        max_strain=SURROUND_MAX_STRAIN,
        sweeps=SURROUND_SWEEPS,
        carrier_bond=carrier_bond,
    )


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.array(
        [
            a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
            a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
            a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
            a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
        ],
        np.float32,
    )


def stride_trajectory(
    geometry,
    center_z: float,
    *,
    period_s: float,
    peak_depth_m: float,
    pitch_deg: float,
    roll_fraction: float,
    frame_dt: float,
    substeps: int,
    with_velocity: bool = False,
):
    """Prescribed heel-to-toe carrier trajectory sampled at every substep.

    Args:
        geometry: Column bed from
            :func:`~projects.digital_instron_v2.dynamics.build_foundation_geometry`.
        center_z: World height of the carrier origin at zero stride depth [m].
        period_s: Stride period [s].
        peak_depth_m: Peak vertical foot penetration [m].
        pitch_deg: Peak fore-aft pitch of the rolling foot [deg].
        roll_fraction: Heel-to-toe roll length as a fraction of the footprint length.
        frame_dt: Render-frame duration [s].
        substeps: Substeps per frame.
        with_velocity: Also return the per-substep spatial velocity (finite
            differenced around the periodic trajectory) for the PD upper.

    Returns:
        ``(poses, dt)`` or ``(poses, velocities, dt)``: per-substep target poses
        ``[substep_count, 7]`` (and spatial velocities ``[substep_count, 6]``) and
        the substep duration [s].
    """
    dt = frame_dt / substeps
    substep_count = int(round(period_s / dt))
    center = geometry.uv_m.mean(axis=0)
    span = float(np.ptp(geometry.uv_m[:, 0]))
    stride = synthetic_stride(peak_depth_m, pitch_deg, roll_fraction * span, period_s)

    poses = np.empty((substep_count, 7), np.float32)
    for t in range(substep_count):
        pos, rot = stride(t * dt)
        poses[t] = [center[0] + pos[0], center[1] + pos[1], center_z + pos[2], rot[0], rot[1], rot[2], rot[3]]
    if not with_velocity:
        return poses, dt

    velocities = np.empty((substep_count, 6), np.float32)
    for t in range(substep_count):
        prev = poses[(t - 1) % substep_count]
        cur = poses[t]
        lin = (cur[:3] - prev[:3]) / dt
        q_rel = _quat_mul(cur[3:7], np.array([-prev[3], -prev[4], -prev[5], prev[6]], np.float32))
        if q_rel[3] < 0.0:
            q_rel = -q_rel
        velocities[t] = np.concatenate([lin, 2.0 * q_rel[:3] / dt])
    return poses, velocities, dt


class DifferentiableStride:
    """Differentiable kinematic heel-to-toe stride over the foam bed.

    Prescribes top-anchor indenter poses and accumulates the vertical indenter
    reaction as a differentiable function of the foam ``material_params``. The
    historical ``grf`` field retains its name for compatibility; it is not a
    per-column external ground-force field. The reaction matches the live
    :class:`~projects.digital_instron_v2.dynamics.MidsoleFoundation` adapter.

    The shoe last presses only its own footprint, so the rest of the midsole is a
    passive surround that relaxes every substep exactly as the shipped ``stride``
    example and the identification relax it (see
    :meth:`~projects.digital_instron_v2.dynamics_diff.DifferentiableMidsoleFoundation.relax_surround`).
    Dropping it would leave the shear layer acting on foam that carries no load.

    Args:
        geometry: Column bed from
            :func:`~projects.digital_instron_v2.dynamics.build_foundation_geometry`.
        material: Calibrated :class:`~projects.digital_instron_v2.core.Material`.
        period_s: Stride period [s].
        peak_depth_m: Peak vertical foot penetration [m].
        pitch_deg: Peak fore-aft pitch [deg].
        roll_fraction: Roll length as a fraction of footprint length.
        frame_dt: Render-frame duration [s].
        substeps: Substeps per frame.
        config: Dynamic :class:`~projects.digital_instron_v2.dynamics.FoundationConfig`.
        surround: Passive-surround settings for the columns the last never
            touches; defaults to the shipped example's settings on
            ``geometry.driven``.
        device: Warp device.
    """

    def __init__(
        self,
        geometry,
        material,
        *,
        period_s: float = 0.6,
        peak_depth_m: float = 0.014,
        pitch_deg: float = 5.0,
        roll_fraction: float = 0.12,
        frame_dt: float = 1.0 / 60.0,
        substeps: int = 8,
        config: FoundationConfig | None = None,
        surround: SurroundConfig | None = None,
        device=None,
    ):
        self.device = device
        self.config = config or FoundationConfig(stretch_floor=0.05)
        geo = geometry
        self.column_count = int(len(geo.slack_m))
        self.surround_config = surround or default_surround(geo.driven, carrier_bond=False)
        center = geo.uv_m.mean(axis=0)
        center_z = float(geo.surface_m.mean())
        self.anchor_local = np.column_stack(
            [geo.uv_m[:, 0] - center[0], geo.uv_m[:, 1] - center[1], geo.surface_m - center_z]
        ).astype(np.float32)
        self.area = np.full(self.column_count, geo.area_m2, np.float32)
        self.z_free = geo.z_free_m.astype(np.float32)
        self.rest_len = geo.slack_m.astype(np.float32)
        self.neighbors = geo.neighbors
        self.spacing_m = geo.spacing_m
        self._geometry = geo
        self._material = material

        self.poses_host, self.dt = stride_trajectory(
            geo,
            center_z,
            period_s=period_s,
            peak_depth_m=peak_depth_m,
            pitch_deg=pitch_deg,
            roll_fraction=roll_fraction,
            frame_dt=frame_dt,
            substeps=substeps,
        )
        self.substep_count = int(len(self.poses_host))

        self.body_com = wp.array([wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3, device=device)
        self.foundation = DifferentiableMidsoleFoundation(
            self.anchor_local,
            self.z_free,
            self.rest_len,
            self.area,
            self.neighbors,
            self.spacing_m,
            material,
            0,
            self.body_com,
            self.substep_count,
            self.config,
            device=device,
            surround=self.surround_config,
        )
        self.material_params = self.foundation.material_params
        self.friction_params = self.foundation.friction_params
        self.poses = [
            wp.array(self.poses_host[t].reshape(1, 7).copy(), dtype=wp.transform, device=device)
            for t in range(self.substep_count)
        ]
        self.body_qd = wp.zeros(1, dtype=wp.spatial_vector, device=device)
        self.body_f = [
            wp.zeros(1, dtype=wp.spatial_vector, device=device, requires_grad=True) for _ in range(self.substep_count)
        ]
        self.grf = wp.zeros(self.substep_count, dtype=wp.float32, device=device, requires_grad=True)

    def forward(self) -> wp.array:
        """Roll the prescribed stride and return the vertical indenter reaction [N]."""
        self.grf.zero_()
        for t in range(self.substep_count):
            self.body_f[t].zero_()
            state = _State(self.poses[t], self.body_qd, self.body_f[t])
            self.foundation.apply(state, t, self.dt)
            wp.launch(
                _record_normal_force,
                dim=self.column_count,
                inputs=[self.foundation.applied_force[t], t, self.grf],
                device=self.device,
            )
        return self.grf

    def zero_grad(self) -> None:
        """Zero the gradients on every differentiable buffer."""
        self.foundation.zero_grad()
        self.grf.grad.zero_()
        for buf in self.body_f:
            buf.grad.zero_()

    def reference_grf(self) -> np.ndarray:
        """Return the shipped forward model's GRF over the same stride, for validation.

        Built with the same passive surround, so the comparison is between two
        implementations of one model rather than between two models.
        """
        foundation = MidsoleFoundation(
            self.anchor_local,
            self.z_free,
            self.rest_len,
            self.area,
            self.neighbors,
            self.spacing_m,
            self._material,
            0,
            self.body_com,
            self.config,
            self.device,
            self.surround_config,
        )
        state = _State(
            wp.zeros(1, dtype=wp.transform, device=self.device),
            wp.zeros(1, dtype=wp.spatial_vector, device=self.device),
            wp.zeros(1, dtype=wp.spatial_vector, device=self.device),
        )
        grf = np.empty(self.substep_count)
        for t in range(self.substep_count):
            state.body_q.assign(self.poses_host[t].reshape(1, 7))
            state.body_qd.zero_()
            state.body_f.zero_()
            foundation.apply(state, self.dt)
            grf[t] = foundation.diagnostics()["normal_force_n"]
        return grf


class DifferentiableSlide:
    """Differentiable kinematic constant-velocity slide for friction identification.

    Presses the foam bed to a fixed penetration and drags it laterally at a
    constant speed (no solver), accumulating the per-substep net shear force as a
    differentiable function of the Coulomb friction coefficient ``mu`` (and the
    foam material). The prescribed position advances consistently with the slide
    velocity while the depth stays fixed. Drag uses the same anchored bristle
    state as the live runtime. A friction target identifies ``mu`` only when some
    bristles reach the Coulomb bound; unsaturated stick is insensitive to ``mu``.

    The indenter is the same shoe last as the stride, so the columns it misses
    relax as the passive surround too. Top-anchor forces are indenter reactions,
    not local outsole-ground forces; the historical ``grf`` field retains its
    name for compatibility.

    Args:
        geometry: Column bed from
            :func:`~projects.digital_instron_v2.dynamics.build_foundation_geometry`.
        material: Calibrated :class:`~projects.digital_instron_v2.core.Material`.
        depth_m: Vertical penetration (compression) held during the slide [m].
        slide_speed_m_s: Constant lateral slide speed [m/s].
        direction: In-plane slide direction (need not be normalized).
        substeps: Number of substeps in the slide rollout.
        frame_dt: Render-frame duration [s] (substep is ``frame_dt / 8``).
        config: Dynamic :class:`~projects.digital_instron_v2.dynamics.FoundationConfig`
            (must set ``mu > 0`` and positive bristle stiffness for friction).
        surround: Passive-surround settings for the columns the last never
            touches; defaults to the shipped example's settings on
            ``geometry.driven``.
        device: Warp device.
    """

    def __init__(
        self,
        geometry,
        material,
        *,
        depth_m: float = 0.01,
        slide_speed_m_s: float = 0.2,
        direction=(1.0, 0.0),
        substeps: int = 32,
        frame_dt: float = 1.0 / 60.0,
        config: FoundationConfig | None = None,
        surround: SurroundConfig | None = None,
        device=None,
    ):
        self.device = device
        self.config = config or FoundationConfig(
            stretch_floor=0.05, normal_damping=8.0, friction_stiffness=1.0e4, friction=8.0, mu=1.0
        )
        geo = geometry
        self.column_count = int(len(geo.slack_m))
        self.surround_config = surround or default_surround(geo.driven, carrier_bond=False)
        self.substep_count = int(substeps)
        self.dt = frame_dt / 8.0
        center = geo.uv_m.mean(axis=0)
        center_z = float(geo.surface_m.mean())
        anchor_local = np.column_stack(
            [geo.uv_m[:, 0] - center[0], geo.uv_m[:, 1] - center[1], geo.surface_m - center_z]
        ).astype(np.float32)

        self.body_com = wp.array([wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3, device=device)
        self.foundation = DifferentiableMidsoleFoundation(
            anchor_local,
            geo.z_free_m.astype(np.float32),
            geo.slack_m.astype(np.float32),
            np.full(self.column_count, geo.area_m2, np.float32),
            geo.neighbors,
            geo.spacing_m,
            material,
            0,
            self.body_com,
            self.substep_count,
            self.config,
            device=device,
            surround=self.surround_config,
        )
        self.material_params = self.foundation.material_params
        self.friction_params = self.foundation.friction_params

        pose = np.array([center[0], center[1], center_z - depth_m, 0.0, 0.0, 0.0, 1.0], np.float32)
        self.pose = wp.array(pose.reshape(1, 7), dtype=wp.transform, device=device)
        speed = np.hypot(direction[0], direction[1]) or 1.0
        vx = slide_speed_m_s * direction[0] / speed
        vy = slide_speed_m_s * direction[1] / speed
        self.body_qd = wp.array(
            np.array([[vx, vy, 0.0, 0.0, 0.0, 0.0]], np.float32), dtype=wp.spatial_vector, device=device
        )
        self.poses = [self.pose]
        for t in range(1, self.substep_count):
            translated = pose.copy()
            translated[:2] += np.asarray([vx, vy], np.float32) * (t * self.dt)
            self.poses.append(wp.array(translated.reshape(1, 7), dtype=wp.transform, device=device))
        self.body_f = [
            wp.zeros(1, dtype=wp.spatial_vector, device=device, requires_grad=True) for _ in range(self.substep_count)
        ]
        self.grf = wp.zeros(self.substep_count, dtype=wp.float32, device=device, requires_grad=True)
        self.shear = wp.zeros(self.substep_count, dtype=wp.vec2, device=device, requires_grad=True)

    def forward(self) -> wp.array:
        """Roll the constant slide and return the per-substep net shear force [N], shape ``[substeps, 2]``."""
        self.grf.zero_()
        self.shear.zero_()
        for t in range(self.substep_count):
            self.body_f[t].zero_()
            state = _State(self.poses[t], self.body_qd, self.body_f[t])
            self.foundation.apply(state, t, self.dt)
            wp.launch(
                _record_normal_force,
                dim=self.column_count,
                inputs=[self.foundation.applied_force[t], t, self.grf],
                device=self.device,
            )
            wp.launch(
                _record_shear_force,
                dim=self.column_count,
                inputs=[self.foundation.applied_force[t], t, self.shear],
                device=self.device,
            )
        return self.shear

    def zero_grad(self) -> None:
        """Zero the gradients on every differentiable buffer."""
        self.foundation.zero_grad()
        self.grf.grad.zero_()
        self.shear.grad.zero_()
        for buf in self.body_f:
            buf.grad.zero_()


class DifferentiableAttached:
    """Differentiable fully dynamic foot-mounted shoe pressing the foam into the ground.

    Builds a single rigid carrier (mass + inertia) on a ground plane, holds it to
    a prescribed per-substep foot trajectory with a damped PD upper
    (:func:`_attach_pd`), and integrates it with
    :class:`~newton.solvers.SolverSemiImplicit`, accumulating the per-substep GRF
    as a differentiable function of the foam ``material_params``. Unlike the
    quasi-static :func:`~projects.digital_instron_v2.core.predict`, the gradient
    flows through the whole rigid-body loop -- contact, PD actuation, and anchored
    bristle friction. Derivatives are exact away from contact and bristle branch
    transitions. At a transition the tape differentiates the selected branch.

    Bottom anchors contact the explicit zero-height ground plane. A supplied
    config with no plane inherits that boundary; a nonzero plane is rejected.
    The live reference uses the same config. The foam is this shoe's sole, so the
    outsole carries *every* column and there is no passive surround to relax: the whole bed is driven, exactly as in the
    shipped ``attached`` example, where an all-driven
    :class:`~projects.digital_instron_v2.dynamics.SurroundConfig` makes the
    relaxation a no-op.

    Args:
        geometry: Column bed from
            :func:`~projects.digital_instron_v2.dynamics.build_foundation_geometry`.
        material: Calibrated :class:`~projects.digital_instron_v2.core.Material`.
        targets: Per-substep target poses ``[substep_count, 7]``.
        target_velocities: Per-substep target spatial velocities ``[substep_count, 6]``.
        dt: Substep duration [s].
        initial_pose: Initial carrier pose ``[7]`` (defaults to ``targets[0]``).
        mass: Carrier mass [kg].
        inertia_diag: Carrier diagonal inertia ``[3]`` [kg m^2].
        pd_gains: ``(kp_lin, kd_lin, kp_ang, kd_ang, max_force)`` for the upper.
        config: Dynamic :class:`~projects.digital_instron_v2.dynamics.FoundationConfig`.
        device: Warp device (also the model/solver device).
    """

    DEFAULT_PD_GAINS = (1.0e5, 450.0, 300.0, 40.0, 20000.0)

    def __init__(
        self,
        geometry,
        material,
        targets: np.ndarray,
        target_velocities: np.ndarray,
        dt: float,
        *,
        initial_pose: np.ndarray | None = None,
        mass: float = 0.5,
        inertia_diag=(0.05, 0.08, 0.08),
        pd_gains=DEFAULT_PD_GAINS,
        config: FoundationConfig | None = None,
        device=None,
    ):
        self.device = device
        self.dt = float(dt)
        self.pd_gains = tuple(float(g) for g in pd_gains)
        config = config or FoundationConfig(
            stretch_floor=0.05, normal_damping=8.0, friction_stiffness=2.0e4, friction=10.0, mu=1.0
        )
        if config.ground_height_m not in (None, 0.0):
            raise ValueError("the attached scenario ground plane is fixed at zero")
        self.config = replace(config, ground_height_m=0.0)
        geo = geometry
        self._geometry = geo
        self._material = material
        self.column_count = int(len(geo.slack_m))
        targets = np.ascontiguousarray(targets, np.float32)
        self.substep_count = int(len(targets))
        self.initial_pose = np.ascontiguousarray(targets[0] if initial_pose is None else initial_pose, np.float32)

        center = geo.uv_m.mean(axis=0)
        com_z = float(geo.z_free_m.mean())
        anchor_local = np.column_stack(
            [geo.uv_m[:, 0] - center[0], geo.uv_m[:, 1] - center[1], geo.z_bottom_m - com_z]
        ).astype(np.float32)

        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        inertia = wp.mat33(inertia_diag[0], 0.0, 0.0, 0.0, inertia_diag[1], 0.0, 0.0, 0.0, inertia_diag[2])
        self.carrier = builder.add_body(mass=mass, com=wp.vec3(0.0, 0.0, 0.0), inertia=inertia)
        self.model = builder.finalize(device=device, requires_grad=True)
        self.solver = newton.solvers.SolverSemiImplicit(self.model, enable_tri_contact=False)

        self.foundation = DifferentiableMidsoleFoundation(
            anchor_local,
            np.zeros(self.column_count, np.float32),
            geo.slack_m.astype(np.float32),
            np.full(self.column_count, geo.area_m2, np.float32),
            geo.neighbors,
            geo.spacing_m,
            material,
            self.carrier,
            self.model.body_com,
            self.substep_count,
            self.config,
            device=device,
        )
        self.material_params = self.foundation.material_params
        self.friction_params = self.foundation.friction_params

        self.states = [self.model.state() for _ in range(self.substep_count + 1)]
        for state in self.states:
            newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, state)
        self.targets = [
            wp.array(targets[t].reshape(1, 7).copy(), dtype=wp.transform, device=device)
            for t in range(self.substep_count)
        ]
        target_velocities = np.ascontiguousarray(target_velocities, np.float32)
        self.target_velocities = [
            wp.array(target_velocities[t].reshape(1, 6).copy(), dtype=wp.spatial_vector, device=device)
            for t in range(self.substep_count)
        ]
        self.grf = wp.zeros(self.substep_count, dtype=wp.float32, device=device, requires_grad=True)
        self.shear = wp.zeros(self.substep_count, dtype=wp.vec2, device=device, requires_grad=True)

    def forward(self) -> wp.array:
        """Integrate the dynamic shoe and return the per-substep vertical GRF [N].

        The per-substep net lateral shear force is also recorded on the tape in
        ``self.shear`` (shape ``[substep_count, 2]``) so a friction objective can be
        differentiated with respect to ``friction_params`` over the rolling contact.
        """
        kp_lin, kd_lin, kp_ang, kd_ang, max_force = self.pd_gains
        self.states[0].body_q.assign(self.initial_pose.reshape(1, 7))
        self.states[0].body_qd.zero_()
        self.grf.zero_()
        self.shear.zero_()
        for t in range(self.substep_count):
            self.states[t].body_f.zero_()
            self.foundation.apply(self.states[t], t, self.dt)
            wp.launch(
                _attach_pd,
                dim=1,
                inputs=[
                    self.carrier,
                    self.states[t].body_q,
                    self.states[t].body_qd,
                    self.targets[t],
                    self.target_velocities[t],
                    kp_lin,
                    kd_lin,
                    kp_ang,
                    kd_ang,
                    max_force,
                    self.states[t].body_f,
                ],
                device=self.device,
            )
            wp.launch(
                _record_normal_force,
                dim=self.column_count,
                inputs=[self.foundation.applied_force[t], t, self.grf],
                device=self.device,
            )
            wp.launch(
                _record_shear_force,
                dim=self.column_count,
                inputs=[self.foundation.applied_force[t], t, self.shear],
                device=self.device,
            )
            self.solver.step(self.states[t], self.states[t + 1], None, None, self.dt)
        return self.grf

    def zero_grad(self) -> None:
        """Zero the gradients on every differentiable buffer."""
        self.foundation.zero_grad()
        self.grf.grad.zero_()
        self.shear.grad.zero_()
        for state in self.states:
            state.body_q.grad.zero_()
            state.body_qd.grad.zero_()
            state.body_f.grad.zero_()

    def reference_grf(self) -> np.ndarray:
        """Return the shipped forward model's GRF (bristle friction + PD) over the same trajectory."""
        geo = self._geometry
        foundation = MidsoleFoundation(
            self.foundation.anchor_local.numpy(),
            np.zeros(self.column_count, np.float32),
            geo.slack_m.astype(np.float32),
            np.full(self.column_count, geo.area_m2, np.float32),
            geo.neighbors,
            geo.spacing_m,
            self._material,
            self.carrier,
            self.model.body_com,
            self.config,
            self.device,
        )
        kp_lin, kd_lin, kp_ang, kd_ang, max_force = self.pd_gains
        targets = wp.array(
            np.ascontiguousarray([a.numpy()[0] for a in self.targets], np.float32),
            dtype=wp.transform,
            device=self.device,
        )
        target_vels = wp.array(
            np.ascontiguousarray([a.numpy()[0] for a in self.target_velocities], np.float32),
            dtype=wp.spatial_vector,
            device=self.device,
        )
        counter = wp.zeros(1, dtype=wp.int32, device=self.device)
        out_force = wp.zeros(1, dtype=wp.float32, device=self.device)
        s0 = self.model.state()
        s1 = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, s0)
        s0.body_q.assign(self.initial_pose.reshape(1, 7))
        s0.body_qd.zero_()
        grf = np.empty(self.substep_count)
        for t in range(self.substep_count):
            s0.body_f.zero_()
            foundation.apply(s0, self.dt)
            wp.launch(
                attach_coupling,
                dim=1,
                inputs=[
                    self.carrier,
                    s0.body_q,
                    s0.body_qd,
                    targets,
                    target_vels,
                    counter,
                    self.substep_count,
                    kp_lin,
                    kd_lin,
                    kp_ang,
                    kd_ang,
                    max_force,
                    s0.body_f,
                    out_force,
                ],
                device=self.device,
            )
            self.solver.step(s0, s1, None, None, self.dt)
            s0, s1 = s1, s0
            grf[t] = foundation.diagnostics()["normal_force_n"]
        return grf


class _State:
    """Minimal simulation-state view (carrier pose/velocity/force) for the kinematic driver."""

    __slots__ = ("body_f", "body_q", "body_qd")

    def __init__(self, body_q, body_qd, body_f):
        self.body_q = body_q
        self.body_qd = body_qd
        self.body_f = body_f


def _demo() -> None:
    """Run the differentiable stride and attached scenarios and report GRF fidelity and gradients."""
    manifest = "DigitalInstron/manifest_v2.json"
    wp.init()
    device = wp.get_device("cuda:0") if wp.get_cuda_device_count() else wp.get_device("cpu")
    geometry = build_foundation_geometry(manifest)
    material = load_fitted_material(manifest)

    print("stride (kinematic heel-to-toe roll)")
    stride = DifferentiableStride(geometry, material, device=device)
    diff = stride.forward().numpy()
    reference = stride.reference_grf()
    print(f"  substeps={stride.substep_count}  peak GRF={diff.max():.1f} N")
    print(f"  GRF vs shipped forward model: max|diff|={np.max(np.abs(diff - reference)):.4f} N")
    loss = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
    stride.zero_grad()
    tape = wp.Tape()
    with tape:
        grf = stride.forward()
        wp.launch(_reduce_sum, dim=stride.substep_count, inputs=[grf, loss], device=device)
    tape.backward(loss)
    print(f"  d(GRF impulse)/d(g_eq) = {float(stride.material_params.grad.numpy()[0]):.4f}")
    tape.zero()

    print("\nattached (fully dynamic shoe: mass + inertia + PD upper + solver)")
    com_z = float(geometry.z_free_m.mean())
    poses, velocities, dt = stride_trajectory(
        geometry,
        com_z,
        period_s=0.7,
        peak_depth_m=0.024,
        pitch_deg=7.0,
        roll_fraction=0.08,
        frame_dt=1.0 / 60.0,
        substeps=128,
        with_velocity=True,
    )
    attached = DifferentiableAttached(geometry, material, poses, velocities, dt, device=device)
    diff = attached.forward().numpy()
    reference = attached.reference_grf()
    peak = max(float(reference.max()), 1.0)
    print(f"  substeps={attached.substep_count}  peak GRF={diff.max():.0f} N  (stance->flight min={diff.min():.0f} N)")
    print(
        f"  GRF vs shipped (shared bristle friction): max|diff|={np.max(np.abs(diff - reference)):.1f} N "
        f"({np.max(np.abs(diff - reference)) / peak * 100:.2f}% of peak)"
    )

    center = geometry.uv_m.mean(axis=0)
    press = np.tile(np.array([center[0], center[1], com_z - 0.02, 0.0, 0.0, 0.0, 1.0], np.float32), (160, 1))
    press_driver = DifferentiableAttached(geometry, material, press, np.zeros((160, 6), np.float32), dt, device=device)
    loss = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
    press_driver.zero_grad()
    tape = wp.Tape()
    with tape:
        grf = press_driver.forward()
        wp.launch(_reduce_sum, dim=press_driver.substep_count, inputs=[grf, loss], device=device)
    tape.backward(loss)
    print(
        f"  continuous-press d(GRF impulse)/d(g_eq) = {float(press_driver.material_params.grad.numpy()[0]):.4f} "
        f"(exact through the full dynamic loop)"
    )

    print("\nslide (kinematic constant-velocity lateral drag: friction identification)")
    slide = DifferentiableSlide(geometry, material, depth_m=0.012, slide_speed_m_s=0.25, substeps=32, device=device)
    shear = slide.forward().numpy()
    mu0 = float(slide.friction_params.numpy()[0])
    print(
        f"  substeps={slide.substep_count}  mu={mu0:.3f}  peak shear={np.abs(shear[:, 0]).max():.0f} N  "
        f"drag impulse={-shear[:, 0].sum():.0f}"
    )
    loss = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
    slide.zero_grad()
    tape = wp.Tape()
    with tape:
        s = slide.forward()
        wp.launch(_drag_impulse, dim=slide.substep_count, inputs=[s, loss], device=device)
    tape.backward(loss)
    print(f"  d(drag impulse)/d(mu) = {float(slide.friction_params.grad.numpy()[0]):.1f}")
    tape.zero()

    target = -slide.forward().numpy()[:, 0].astype(np.float64).sum()  # measured drag at the reference mu
    slide.friction_params.assign(np.array([mu0 * 0.4], np.float32))  # start from a wrong guess
    # Re-evaluate the gradient when the bristle stick/slip active set changes.
    for _ in range(4):
        loss.zero_()
        slide.zero_grad()
        tape = wp.Tape()
        with tape:
            s = slide.forward()
            wp.launch(_drag_impulse, dim=slide.substep_count, inputs=[s, loss], device=device)
        tape.backward(loss)
        sensitivity = float(slide.friction_params.grad.numpy()[0])
        if sensitivity <= 0.0:
            raise RuntimeError("friction identification needs sliding bristles")
        recovered = float(slide.friction_params.numpy()[0]) + (target - float(loss.numpy()[0])) / sensitivity
        slide.friction_params.assign(np.array([recovered], np.float32))
        tape.zero()
    print(f"  friction ID: recovered mu={recovered:.4f} from piecewise gradients (target {mu0:.4f})")


if __name__ == "__main__":
    _demo()
