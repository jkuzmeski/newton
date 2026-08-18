# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Warp-native muscle forces and the generalized forces they produce.

Given muscle activations and a coordinate configuration (optionally with
coordinate speeds), this module evaluates the rigid-tendon De Groote-Fregly
(2016) muscle-tendon force of every muscle and projects it onto the model
coordinates through the moment arms, producing the generalized (joint) forces a
muscle set applies to the skeleton.

The muscle-tendon lengthening velocity follows directly from the moment arms and
the coordinate speeds: since the moment arm about coordinate :math:`q_i` is
:math:`r_i=-\partial L_{MT}/\partial q_i`, the path lengthening velocity is
:math:`\dot L_{MT}=\sum_i (\partial L_{MT}/\partial q_i)\,\dot q_i=-\sum_i r_i\,\dot q_i`.
The generalized force a muscle of tendon force :math:`F` contributes to
coordinate :math:`q_i` is :math:`\tau_i=r_i\,F` -- OpenSim's ``Moment = MomentArm
* force`` relation -- so the total muscle-generated generalized force is
:math:`\tau_i=\sum_m r_{m,i}\,F_m`.

The rigid-tendon force is the muscle model Moco uses for its inverse and
predictive problems (``DeGrooteFregly2016Muscle`` with the tendon compliance
ignored). An isometric elastic-tendon equilibrium force -- solving the series
fiber-tendon force balance for the fiber length -- is available through
:meth:`MuscleForces.forces_elastic_tendon`.

First-order activation dynamics (Thelen 2003) map neural excitations to
activations and are integrated on-device by :meth:`MuscleForces.integrate_activation`,
completing the excitation -> activation -> force -> generalized-force pipeline.
"""

from __future__ import annotations

import copy

import numpy as np
import warp as wp

from .kinematics import _FN_SIMMSPLINE, _bake_function, _eval_baked
from .muscle import (
    activation_dot,
    dgf_active_force_length,
    dgf_force_velocity,
    dgf_force_velocity_inverse,
    dgf_passive_force_length,
    dgf_tendon_force,
    muscle_force_equilibrium_tendon,
    muscle_force_rigid_tendon,
    pennated_fiber_length,
)
from .muscle_path import MusclePaths
from .types import OsimModel

_f64 = wp.float64
_f32 = wp.float32


@wp.kernel
def muscle_force_kernel(
    activation: wp.array2d[_f32],
    lmt: wp.array2d[_f64],
    vmt: wp.array2d[_f64],
    fmax: wp.array[_f32],
    l_opt: wp.array[_f32],
    lt_slack: wp.array[_f32],
    vmax: wp.array[_f32],
    cos_penn: wp.array[_f32],
    force: wp.array2d[_f32],
):
    """Evaluate every muscle's rigid-tendon force over a batch of configurations.

    Launched with dim ``(batch, num_muscles)``; each thread evaluates the
    De Groote-Fregly rigid-tendon muscle-tendon force for one muscle in one
    configuration.
    """
    b, m = wp.tid()
    force[b, m] = muscle_force_rigid_tendon(
        activation[b, m],
        _f32(lmt[b, m]),
        _f32(vmt[b, m]),
        fmax[m],
        l_opt[m],
        lt_slack[m],
        vmax[m],
        cos_penn[m],
    )


@wp.kernel
def muscle_affine_force_kernel(
    lmt: wp.array2d[_f64],
    vmt: wp.array2d[_f64],
    fmax: wp.array[_f32],
    l_opt: wp.array[_f32],
    lt_slack: wp.array[_f32],
    vmax: wp.array[_f32],
    cos_penn: wp.array[_f32],
    active: wp.array2d[_f32],
    passive: wp.array2d[_f32],
):
    """Compute ``force(a) = a*active + passive`` coefficients in one launch."""
    b, m = wp.tid()
    p = muscle_force_rigid_tendon(
        _f32(0.0),
        _f32(lmt[b, m]),
        _f32(vmt[b, m]),
        fmax[m],
        l_opt[m],
        lt_slack[m],
        vmax[m],
        cos_penn[m],
    )
    total = muscle_force_rigid_tendon(
        _f32(1.0),
        _f32(lmt[b, m]),
        _f32(vmt[b, m]),
        fmax[m],
        l_opt[m],
        lt_slack[m],
        vmax[m],
        cos_penn[m],
    )
    active[b, m] = total - p
    passive[b, m] = p


@wp.kernel
def equilibrium_force_kernel(
    activation: wp.array2d[_f32],
    lmt: wp.array2d[_f32],
    fmax: wp.array[_f32],
    l_opt: wp.array[_f32],
    lt_slack: wp.array[_f32],
    cos_penn: wp.array[_f32],
    kt: wp.array[_f32],
    force: wp.array2d[_f32],
):
    """Evaluate every muscle's isometric elastic-tendon equilibrium force.

    Launched with dim ``(batch, num_muscles)``; each thread solves the series
    fiber-tendon force balance for one muscle in one configuration.
    """
    b, m = wp.tid()
    force[b, m] = muscle_force_equilibrium_tendon(
        activation[b, m],
        lmt[b, m],
        fmax[m],
        l_opt[m],
        lt_slack[m],
        cos_penn[m],
        kt[m],
    )


@wp.kernel
def fiber_velocity_kernel(
    activation: wp.array2d[_f32],
    lmt: wp.array2d[_f32],
    fiber_len: wp.array2d[_f32],
    fmax: wp.array[_f32],
    l_opt: wp.array[_f32],
    lt_slack: wp.array[_f32],
    vmax: wp.array[_f32],
    cos_penn: wp.array[_f32],
    kt: wp.array[_f32],
    fiber_vel: wp.array2d[_f32],
    tendon_force: wp.array2d[_f32],
):
    """Elastic-tendon fiber velocity and tendon force from the series force balance.

    Launched with dim ``(batch, num_muscles)``. Given the fiber-length state, the
    tendon (and hence its force) is determined by geometry; the fiber must carry
    that same force, which fixes the force-velocity multiplier and thus the fiber
    velocity by inverting ``dgf_force_velocity``.
    """
    b, m = wp.tid()
    lo = l_opt[m]
    cp = cos_penn[m]
    lm = fiber_len[b, m]
    # Constant-width pennation: fiber width is fixed at the optimal geometry.
    width = lo * wp.sqrt(wp.max(_f32(1.0) - cp * cp, _f32(0.0)))
    along = wp.sqrt(wp.max(lm * lm - width * width, _f32(1.0e-12)))
    cos_cur = along / wp.max(lm, _f32(1.0e-9))
    l_norm = lm / lo
    lt_norm = (lmt[b, m] - along) / lt_slack[m]
    ft = dgf_tendon_force(lt_norm, kt[m])
    tendon_force[b, m] = fmax[m] * ft
    fal = dgf_active_force_length(l_norm)
    fpe = dgf_passive_force_length(l_norm)
    denom = activation[b, m] * fal
    if denom < _f32(1.0e-6):
        fiber_vel[b, m] = _f32(0.0)
    else:
        fv = (ft / cos_cur - fpe) / denom
        fiber_vel[b, m] = dgf_force_velocity_inverse(fv) * vmax[m] * lo


@wp.kernel
def spring_force_kernel(
    length: wp.array2d[_f64],
    ldot: wp.array2d[_f64],
    stiffness: wp.array[_f64],
    resting_length: wp.array[_f64],
    dissipation: wp.array[_f64],
    tension: wp.array2d[_f32],
):
    """Path-spring tension from stretch and lengthening rate (OpenSim ``PathSpring``).

    Launched with dim ``(batch, num_springs)``. A spring carries load only when
    stretched beyond its resting length; the tension is the linear-elastic force
    modulated by a Hunt-Crossley-style dissipation term and clamped non-negative.
    """
    b, s = wp.tid()
    stretch = length[b, s] - resting_length[s]
    t = _f64(0.0)
    if stretch > _f64(0.0):
        t = stiffness[s] * stretch * (_f64(1.0) + dissipation[s] * ldot[b, s])
        t = wp.max(t, _f64(0.0))
    tension[b, s] = _f32(t)


@wp.kernel
def ligament_force_kernel(
    length: wp.array2d[_f64],
    resting_length: wp.array[_f64],
    pcsa_force: wp.array[_f64],
    ctype: wp.array[wp.int32],
    cp0: wp.array[_f64],
    cp1: wp.array[_f64],
    koff: wp.array[wp.int32],
    kcnt: wp.array[wp.int32],
    kx: wp.array[_f64],
    ky: wp.array[_f64],
    kb: wp.array[_f64],
    kc: wp.array[_f64],
    kd: wp.array[_f64],
    force: wp.array2d[_f32],
):
    """Ligament tension: ``pcsa_force`` times the normalized force-length curve.

    Launched with dim ``(batch, num_ligaments)``. The curve is evaluated at the
    normalized length ``length / resting_length`` with the shared baked-function
    evaluator, so the curve itself encodes the slack region below resting length.
    """
    b, l = wp.tid()
    x = length[b, l] / resting_length[l]
    val = _eval_baked(x, ctype[l], cp0[l], cp1[l], koff[l], kcnt[l], kx, ky, kb, kc, kd)
    force[b, l] = _f32(pcsa_force[l] * val)


@wp.kernel
def generalized_force_kernel(
    r: wp.array3d[_f64],
    force: wp.array2d[_f32],
    num_muscles: int,
    tau: wp.array2d[_f64],
):
    r"""Project rigid-tendon muscle forces onto coordinates through the moment arms.

    Launched with dim ``(batch, num_coordinates)``; computes
    :math:`\tau_c=\sum_m r_{m,c}\,F_m` on device.
    """
    b, c = wp.tid()
    acc = _f64(0.0)
    for m in range(num_muscles):
        acc += r[b, m, c] * wp.float64(force[b, m])
    tau[b, c] = acc


@wp.kernel
def activation_integrate_kernel(
    act_in: wp.array2d[_f32],
    excitation: wp.array2d[_f32],
    tau_act: wp.array[_f32],
    tau_deact: wp.array[_f32],
    dt: _f32,
    nsub: wp.int32,
    act_out: wp.array2d[_f32],
):
    """Integrate first-order muscle activation dynamics over one interval.

    Advances each muscle's activation from ``act_in`` toward ``excitation`` over
    ``dt`` using ``nsub`` RK4 substeps of ``activation_dot``
    (Thelen 2003). Excitation is held constant across the interval.
    """
    b, m = wp.tid()
    a = act_in[b, m]
    u = excitation[b, m]
    ta = tau_act[m]
    td = tau_deact[m]
    h = dt / wp.float32(nsub)
    for _i in range(nsub):
        k1 = activation_dot(a, u, ta, td)
        k2 = activation_dot(a + _f32(0.5) * h * k1, u, ta, td)
        k3 = activation_dot(a + _f32(0.5) * h * k2, u, ta, td)
        k4 = activation_dot(a + h * k3, u, ta, td)
        a = a + (h / _f32(6.0)) * (k1 + _f32(2.0) * k2 + _f32(2.0) * k3 + k4)
    act_out[b, m] = a


@wp.kernel
def fiber_forces_kernel(
    activation: wp.array2d[_f32],
    lmt: wp.array2d[_f32],
    vmt: wp.array2d[_f32],
    fmax: wp.array[_f32],
    l_opt: wp.array[_f32],
    lt_slack: wp.array[_f32],
    vmax: wp.array[_f32],
    cos_penn: wp.array[_f32],
    out: wp.array3d[_f32],
):
    """Rigid-tendon muscle force breakdown for one (batch, muscle) sample.

    Writes ``[active_fiber_force, passive_fiber_force, fiber_force, tendon_force]``
    into ``out[b, m, 0:4]`` [N]. Mirrors the fiber math of
    ``muscle_force_rigid_tendon``, so
    ``tendon_force`` equals the total returned by :meth:`MuscleForces.forces`.
    """
    b, m = wp.tid()
    lo = l_opt[m]
    lts = lt_slack[m]
    l_norm = pennated_fiber_length(lmt[b, m], lts, lo, cos_penn[m])
    lm = l_norm * lo
    cp = wp.max((lmt[b, m] - lts) / wp.max(lm, _f32(1.0e-9)), _f32(0.0))
    v_norm = (vmt[b, m] * cp) / wp.max(lo * vmax[m], _f32(1.0e-9))
    fm = fmax[m]
    active = fm * activation[b, m] * dgf_active_force_length(l_norm) * dgf_force_velocity(v_norm)
    passive = fm * dgf_passive_force_length(l_norm)
    fiber = active + passive
    out[b, m, 0] = active
    out[b, m, 1] = passive
    out[b, m, 2] = fiber
    out[b, m, 3] = wp.max(fiber * cp, _f32(0.0))


@wp.kernel
def fiber_kinematics_kernel(
    lmt: wp.array2d[_f32],
    vmt: wp.array2d[_f32],
    l_opt: wp.array[_f32],
    lt_slack: wp.array[_f32],
    vmax: wp.array[_f32],
    cos_penn: wp.array[_f32],
    out: wp.array3d[_f32],
):
    """Rigid-tendon fiber kinematics for one (batch, muscle) sample.

    Writes ``[fiber_length_m, normalized_fiber_length, pennation_angle_rad,
    normalized_fiber_velocity]`` into ``out[b, m, 0:4]`` from the constant-width
    pennation model (``pennated_fiber_length``).
    """
    b, m = wp.tid()
    lo = l_opt[m]
    l_norm = pennated_fiber_length(lmt[b, m], lt_slack[m], lo, cos_penn[m])
    lm = l_norm * lo
    cp = wp.max((lmt[b, m] - lt_slack[m]) / wp.max(lm, _f32(1.0e-9)), _f32(0.0))
    penn = wp.acos(wp.min(cp, _f32(1.0)))
    v_norm = (vmt[b, m] * cp) / wp.max(lo * vmax[m], _f32(1.0e-9))
    out[b, m, 0] = lm
    out[b, m, 1] = l_norm
    out[b, m, 2] = penn
    out[b, m, 3] = v_norm


@wp.kernel
def muscle_analysis_kernel(
    activation: wp.array2d[_f32],
    lmt: wp.array2d[_f64],
    vmt: wp.array2d[_f64],
    fmax: wp.array[_f32],
    l_opt: wp.array[_f32],
    lt_slack: wp.array[_f32],
    vmax: wp.array[_f32],
    cos_penn: wp.array[_f32],
    out: wp.array3d[_f32],
):
    """Pack fiber kinematics and force components from shared path geometry."""
    b, m = wp.tid()
    length = _f32(lmt[b, m])
    velocity = _f32(vmt[b, m])
    lo = l_opt[m]
    l_norm = pennated_fiber_length(length, lt_slack[m], lo, cos_penn[m])
    lm = l_norm * lo
    cp = wp.max((length - lt_slack[m]) / wp.max(lm, _f32(1.0e-9)), _f32(0.0))
    penn = wp.acos(wp.min(cp, _f32(1.0)))
    v_norm = (velocity * cp) / wp.max(lo * vmax[m], _f32(1.0e-9))
    fm = fmax[m]
    active = fm * activation[b, m] * dgf_active_force_length(l_norm) * dgf_force_velocity(v_norm)
    passive = fm * dgf_passive_force_length(l_norm)
    fiber = active + passive
    out[b, m, 0] = lm
    out[b, m, 1] = l_norm
    out[b, m, 2] = penn
    out[b, m, 3] = v_norm
    out[b, m, 4] = active
    out[b, m, 5] = passive
    out[b, m, 6] = fiber
    out[b, m, 7] = wp.max(fiber * cp, _f32(0.0))


@wp.kernel
def pack_scalar_pair_kernel(first: wp.array2d[_f32], second: wp.array2d[_f32], out: wp.array3d[_f32]):
    """Pack two float32 muscle fields for one final readback."""
    b, m = wp.tid()
    out[b, m, 0] = first[b, m]
    out[b, m, 1] = second[b, m]


@wp.kernel
def pack_muscle_geometry_kernel(
    lengths: wp.array2d[_f64], moment_arms: wp.array3d[_f64], ncoord: int, out: wp.array2d[_f64]
):
    """Pack path lengths and moment arms for one final readback."""
    b, m, field = wp.tid()
    offset = m * (ncoord + 1)
    if field == 0:
        out[b, offset] = lengths[b, m]
    else:
        out[b, offset + field] = moment_arms[b, m, field - 1]


class MuscleForces:
    """Rigid-tendon muscle forces and muscle-generated generalized forces.

    Args:
        model: Parsed OpenSim model.
        device: Warp device (defaults to CPU, matching the rest of the port).
    """

    def __init__(self, model: OsimModel, device=None):
        self.paths = MusclePaths(model, device=device)
        self.device = self.paths.device
        self.muscle_names = self.paths.muscle_names
        self.coordinate_names = self.paths.coordinate_names

        def _p(mus, key, default):
            return float(mus.params.get(key, default))

        fmax, l_opt, lt_slack, vmax, cos_penn, kt = [], [], [], [], [], []
        tau_act, tau_deact = [], []
        for mus in model.muscles:
            fmax.append(_p(mus, "max_isometric_force", 1.0))
            l_opt.append(_p(mus, "optimal_fiber_length", 0.1))
            lt_slack.append(_p(mus, "tendon_slack_length", 0.1))
            vmax.append(_p(mus, "max_contraction_velocity", 10.0))
            cos_penn.append(float(np.cos(_p(mus, "pennation_angle_at_optimal", 0.0))))
            e_t = _p(mus, "tendon_strain_at_one_norm_force", 0.049)
            kt.append(float(np.log(1.250 / 0.200) / (1.0 + e_t - 0.995)))
            tau_act.append(_p(mus, "activation_time_constant", 0.015))
            tau_deact.append(_p(mus, "deactivation_time_constant", 0.050))
        self._fmax = np.asarray(fmax, np.float64)
        self._l_opt = np.asarray(l_opt, np.float64)
        self._lt_slack = np.asarray(lt_slack, np.float64)
        self._vmax = np.asarray(vmax, np.float64)
        self._cos_penn = np.asarray(cos_penn, np.float64)
        self._kt = np.asarray(kt, np.float64)
        self.d_fmax = wp.array(self._fmax, dtype=_f32, device=self.device)
        self.d_l_opt = wp.array(self._l_opt, dtype=_f32, device=self.device)
        self.d_lt_slack = wp.array(self._lt_slack, dtype=_f32, device=self.device)
        self.d_vmax = wp.array(self._vmax, dtype=_f32, device=self.device)
        self.d_cos_penn = wp.array(self._cos_penn, dtype=_f32, device=self.device)
        self.d_kt = wp.array(self._kt, dtype=_f32, device=self.device)
        self._tau_act = np.asarray(tau_act, np.float64)
        self._tau_deact = np.asarray(tau_deact, np.float64)
        self.d_tau_act = wp.array(self._tau_act, dtype=_f32, device=self.device)
        self.d_tau_deact = wp.array(self._tau_deact, dtype=_f32, device=self.device)

    @property
    def num_muscles(self) -> int:
        """Number of muscles."""
        return len(self.muscle_names)

    def integrate_activation(
        self, activations: np.ndarray, excitations: np.ndarray, dt: float, substeps: int = 8
    ) -> np.ndarray:
        """Advance muscle activations by ``dt`` under first-order activation dynamics.

        Integrates Thelen (2003) activation dynamics
        (``activation_dot``) for every muscle with
        ``substeps`` RK4 substeps, holding each excitation constant across the
        interval. Activation dynamics are pose-independent, so no coordinates are
        required.

        Args:
            activations: Current activations in [0, 1], shape ``[batch, num_muscles]``
                or ``[num_muscles]``.
            excitations: Neural excitations (controls) in [0, 1], same shape as
                ``activations`` (broadcast over the batch when 1-D).
            dt: Interval length [s].
            substeps: Number of RK4 substeps taken across ``dt``.

        Returns:
            Updated activations in [0, 1], shape ``[batch, num_muscles]``.
        """
        act = np.atleast_2d(np.asarray(activations, dtype=np.float64))
        exc = np.atleast_2d(np.asarray(excitations, dtype=np.float64))
        batch = max(act.shape[0], exc.shape[0])
        if act.shape[0] == 1 and batch > 1:
            act = np.repeat(act, batch, axis=0)
        if exc.shape[0] == 1 and batch > 1:
            exc = np.repeat(exc, batch, axis=0)
        nm = self.num_muscles
        if act.shape[1] != nm or exc.shape[1] != nm:
            raise ValueError(f"activations/excitations must have {nm} muscles, got {act.shape[1]} and {exc.shape[1]}")
        d_in = wp.array(np.ascontiguousarray(act), dtype=_f32, device=self.device)
        d_exc = wp.array(np.ascontiguousarray(exc), dtype=_f32, device=self.device)
        d_out = wp.empty((batch, nm), dtype=_f32, device=self.device)
        wp.launch(
            activation_integrate_kernel,
            dim=(batch, nm),
            inputs=[d_in, d_exc, self.d_tau_act, self.d_tau_deact, _f32(dt), int(substeps), d_out],
            device=self.device,
        )
        return d_out.numpy().astype(np.float64)

    def _prepare(self, activations, coords, speeds):
        coords = np.ascontiguousarray(np.atleast_2d(coords), dtype=np.float64)
        batch, nm = coords.shape[0], self.num_muscles
        act = np.atleast_2d(np.asarray(activations, dtype=np.float64))
        if act.shape[0] == 1 and batch > 1:
            act = np.repeat(act, batch, axis=0)
        lmt = self.paths.lengths(coords)
        if speeds is None:
            vmt = np.zeros((batch, nm))
        else:
            vmt = self.paths.velocities(coords, speeds)
        return coords, np.ascontiguousarray(act), np.ascontiguousarray(lmt), np.ascontiguousarray(vmt)

    def _device_inputs(self, activations, coords, speeds):
        """Copy one muscle-state batch to the device without intermediate round-trips."""
        coords = np.ascontiguousarray(np.atleast_2d(coords), dtype=np.float64)
        batch = coords.shape[0]
        act = np.asarray(activations, dtype=np.float32)
        if act.ndim == 0:
            act = np.full((1, self.num_muscles), float(act), dtype=np.float32)
        else:
            act = np.atleast_2d(act)
        if act.shape[0] == 1 and batch > 1:
            act = np.repeat(act, batch, axis=0)
        if act.shape != (batch, self.num_muscles):
            raise ValueError(f"activations must be [batch, {self.num_muscles}]")
        d_speeds = None
        if speeds is not None:
            speeds = np.ascontiguousarray(np.atleast_2d(speeds), dtype=np.float64)
            if speeds.shape != coords.shape:
                raise ValueError(f"speeds must have shape {coords.shape}, got {speeds.shape}")
            d_speeds = wp.array(speeds, dtype=_f64, device=self.device)
        return (
            wp.array(np.ascontiguousarray(act), dtype=_f32, device=self.device),
            wp.array(coords, dtype=_f64, device=self.device),
            d_speeds,
        )

    def forces(self, activations: np.ndarray, coords: np.ndarray, speeds: np.ndarray | None = None) -> np.ndarray:
        """Return rigid-tendon muscle-tendon forces [N], shape ``[batch, num_muscles]``.

        Args:
            activations: Muscle activations in [0, 1], shape ``[batch, num_muscles]``
                or ``[num_muscles]`` (broadcast over the batch).
            coords: Coordinate configurations [batch, num_coordinates].
            speeds: Coordinate speeds [batch, num_coordinates]; ``None`` treats the
                pose as isometric (zero muscle-tendon velocity).
        """
        d_act, d_q, d_speeds = self._device_inputs(activations, coords, speeds)
        return self._forces_device(d_act, d_q, d_speeds).numpy().astype(np.float64)

    def _affine_coefficients(
        self, coords: np.ndarray, speeds: np.ndarray | None, eps: float = 1.0e-5
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return shared moment arms and affine force coefficients from one device geometry pass."""
        coords = np.ascontiguousarray(np.atleast_2d(coords), dtype=np.float64)
        q_wp = wp.array(coords, dtype=_f64, device=self.device)
        d_r = self.paths._moment_arms_device(q_wp, eps)
        d_lengths = self.paths._lengths_qwp(q_wp)
        if speeds is None:
            d_velocities = wp.zeros((coords.shape[0], self.num_muscles), dtype=_f64, device=self.device)
        else:
            speeds = np.ascontiguousarray(np.atleast_2d(speeds), dtype=np.float64)
            if speeds.shape != coords.shape:
                raise ValueError(f"speeds must have shape {coords.shape}, got {speeds.shape}")
            speeds_wp = wp.array(speeds, dtype=_f64, device=self.device)
            d_velocities = self.paths._velocities_qwp(q_wp, speeds_wp, eps, d_r)
        active = wp.empty((coords.shape[0], self.num_muscles), dtype=_f32, device=self.device)
        passive = wp.empty((coords.shape[0], self.num_muscles), dtype=_f32, device=self.device)
        wp.launch(
            muscle_affine_force_kernel,
            dim=(coords.shape[0], self.num_muscles),
            inputs=[
                d_lengths,
                d_velocities,
                self.d_fmax,
                self.d_l_opt,
                self.d_lt_slack,
                self.d_vmax,
                self.d_cos_penn,
                active,
                passive,
            ],
            device=self.device,
        )
        packed = wp.empty((coords.shape[0], self.num_muscles, 2), dtype=_f32, device=self.device)
        wp.launch(
            pack_scalar_pair_kernel,
            dim=(coords.shape[0], self.num_muscles),
            inputs=[active, passive, packed],
            device=self.device,
        )
        coefficients = packed.numpy().astype(np.float64)
        return d_r.numpy(), coefficients[:, :, 0], coefficients[:, :, 1]

    def _analysis_quantities(
        self, activations: np.ndarray, coords: np.ndarray, speeds: np.ndarray | None, eps: float = 1.0e-5
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        """Return path geometry, fiber kinematics, and forces from one device pass."""
        d_act, q_wp, speeds_wp = self._device_inputs(activations, coords, speeds)
        batch = q_wp.shape[0]
        d_r = self.paths._moment_arms_device(q_wp, eps)
        d_lengths = self.paths._lengths_qwp(q_wp)
        if speeds_wp is None:
            d_velocities = wp.zeros((batch, self.num_muscles), dtype=_f64, device=self.device)
        else:
            d_velocities = self.paths._velocities_qwp(q_wp, speeds_wp, eps, d_r)
        report = wp.empty((batch, self.num_muscles, 8), dtype=_f32, device=self.device)
        wp.launch(
            muscle_analysis_kernel,
            dim=(batch, self.num_muscles),
            inputs=[
                d_act,
                d_lengths,
                d_velocities,
                self.d_fmax,
                self.d_l_opt,
                self.d_lt_slack,
                self.d_vmax,
                self.d_cos_penn,
                report,
            ],
            device=self.device,
        )
        data = report.numpy().astype(np.float64)
        quantities = {
            "fiber_length": data[:, :, 0],
            "normalized_fiber_length": data[:, :, 1],
            "pennation_angle": data[:, :, 2],
            "normalized_fiber_velocity": data[:, :, 3],
            "active_fiber_force": data[:, :, 4],
            "passive_fiber_force": data[:, :, 5],
            "fiber_force": data[:, :, 6],
            "tendon_force": data[:, :, 7],
        }
        ncoord = len(self.coordinate_names)
        geometry = wp.empty((batch, self.num_muscles * (ncoord + 1)), dtype=_f64, device=self.device)
        wp.launch(
            pack_muscle_geometry_kernel,
            dim=(batch, self.num_muscles, ncoord + 1),
            inputs=[d_lengths, d_r, ncoord, geometry],
            device=self.device,
        )
        geometry_data = geometry.numpy().reshape(batch, self.num_muscles, ncoord + 1)
        return geometry_data[:, :, 0], geometry_data[:, :, 1:], quantities

    def forces_elastic_tendon(self, activations: np.ndarray, coords: np.ndarray) -> np.ndarray:
        """Return isometric elastic-tendon equilibrium forces [N], shape ``[batch, num_muscles]``.

        Solves the series fiber-tendon force balance for the fiber length on-device
        (see ``muscle_force_equilibrium_tendon``),
        modeling a compliant tendon rather than the rigid tendon of :meth:`forces`.
        Assumes zero fiber velocity; the tendon stiffness follows each muscle's
        ``tendon_strain_at_one_norm_force``.

        Args:
            activations: Muscle activations in [0, 1], shape ``[batch, num_muscles]``
                or ``[num_muscles]`` (broadcast over the batch).
            coords: Coordinate configurations [batch, num_coordinates].
        """
        _, act, lmt, _ = self._prepare(activations, np.ascontiguousarray(np.atleast_2d(coords), dtype=np.float64), None)
        batch = lmt.shape[0]
        d_act = wp.array(act, dtype=_f32, device=self.device)
        d_lmt = wp.array(lmt, dtype=_f32, device=self.device)
        d_force = wp.empty((batch, self.num_muscles), dtype=_f32, device=self.device)
        wp.launch(
            equilibrium_force_kernel,
            dim=(batch, self.num_muscles),
            inputs=[d_act, d_lmt, self.d_fmax, self.d_l_opt, self.d_lt_slack, self.d_cos_penn, self.d_kt, d_force],
            device=self.device,
        )
        return d_force.numpy().astype(np.float64)

    def elastic_tendon_fiber_velocity(
        self, activations: np.ndarray, coords: np.ndarray, fiber_lengths: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Return compliant-tendon fiber velocities [m/s] and tendon forces [N].

        For a compliant tendon the fiber and tendon carry the same force in series.
        Given the fiber-length state, the tendon length (and force) follows from the
        path geometry, and the fiber velocity is the value that makes the fiber
        force match the tendon force -- obtained by inverting the force-velocity
        curve. This is the algebraic fiber-velocity relation used as the muscle
        state derivative in compliant-tendon forward dynamics; at the equilibrium
        fiber length of :meth:`forces_elastic_tendon` the fiber velocity is zero.

        Args:
            activations: Muscle activations in [0, 1], shape ``[batch, num_muscles]``
                or ``[num_muscles]`` (broadcast over the batch).
            coords: Coordinate configurations [batch, num_coordinates].
            fiber_lengths: Fiber-length state [m], shape ``[batch, num_muscles]``.

        Returns:
            Dict with ``fiber_velocity`` [m/s] and ``tendon_force`` [N], each shaped
            ``[batch, num_muscles]``.
        """
        _, act, lmt, _ = self._prepare(activations, np.ascontiguousarray(np.atleast_2d(coords), dtype=np.float64), None)
        batch = lmt.shape[0]
        fib = np.ascontiguousarray(np.atleast_2d(fiber_lengths), dtype=np.float64)
        if fib.shape != (batch, self.num_muscles):
            raise ValueError(f"fiber_lengths must be [batch, {self.num_muscles}]")
        d_vel = wp.empty((batch, self.num_muscles), dtype=_f32, device=self.device)
        d_ten = wp.empty((batch, self.num_muscles), dtype=_f32, device=self.device)
        wp.launch(
            fiber_velocity_kernel,
            dim=(batch, self.num_muscles),
            inputs=[
                wp.array(act, dtype=_f32, device=self.device),
                wp.array(lmt, dtype=_f32, device=self.device),
                wp.array(fib, dtype=_f32, device=self.device),
                self.d_fmax,
                self.d_l_opt,
                self.d_lt_slack,
                self.d_vmax,
                self.d_cos_penn,
                self.d_kt,
                d_vel,
                d_ten,
            ],
            device=self.device,
        )
        packed = wp.empty((batch, self.num_muscles, 2), dtype=_f32, device=self.device)
        wp.launch(
            pack_scalar_pair_kernel,
            dim=(batch, self.num_muscles),
            inputs=[d_vel, d_ten, packed],
            device=self.device,
        )
        data = packed.numpy().astype(np.float64)
        return {"fiber_velocity": data[:, :, 0], "tendon_force": data[:, :, 1]}

    def _forces_device(
        self,
        activations: wp.array[_f32],
        coords: wp.array[_f64],
        speeds: wp.array[_f64] | None,
        eps: float = 1.0e-5,
        moment_arms: wp.array[_f64] | None = None,
        lengths: wp.array[_f64] | None = None,
    ) -> wp.array[_f32]:
        """Evaluate rigid-tendon muscle forces without leaving the device.

        Args:
            activations: Muscle activations on device, shape [batch, num_muscles].
            coords: Coordinate configurations on device, shape [batch, num_coordinates].
            speeds: Coordinate speeds on device, or ``None`` for isometric forces.
            eps: Central-difference step for moment arms [rad or m].
            moment_arms: Optional precomputed moment arms on device.
            lengths: Optional precomputed muscle-tendon lengths on device.

        Returns:
            Muscle forces [N] on device, shape [batch, num_muscles].
        """
        batch = coords.shape[0]
        d_lmt = self.paths._lengths_qwp(coords) if lengths is None else lengths
        if speeds is None:
            d_vmt = wp.zeros((batch, self.num_muscles), dtype=_f64, device=self.device)
        else:
            d_vmt = self.paths._velocities_qwp(coords, speeds, eps, moment_arms)
        d_force = wp.empty((batch, self.num_muscles), dtype=_f32, device=self.device)
        wp.launch(
            muscle_force_kernel,
            dim=(batch, self.num_muscles),
            inputs=[
                activations,
                d_lmt,
                d_vmt,
                self.d_fmax,
                self.d_l_opt,
                self.d_lt_slack,
                self.d_vmax,
                self.d_cos_penn,
                d_force,
            ],
            device=self.device,
        )
        return d_force

    def fiber_forces(
        self, activations: np.ndarray, coords: np.ndarray, speeds: np.ndarray | None = None
    ) -> dict[str, np.ndarray]:
        """Return the rigid-tendon muscle force breakdown for a batch of configurations.

        Splits each muscle's force into the components OpenSim's muscle analysis
        reports. ``tendon_force`` (the along-path force) equals :meth:`forces`;
        the fiber-frame components ``active_fiber_force`` and ``passive_fiber_force``
        sum to ``fiber_force``.

        Args:
            activations: Muscle activations in [0, 1], shape ``[batch, num_muscles]``
                or ``[num_muscles]`` (broadcast over the batch).
            coords: Coordinate configurations [batch, num_coordinates].
            speeds: Coordinate speeds [batch, num_coordinates]; ``None`` treats the
                pose as isometric (zero muscle-tendon velocity).

        Returns:
            Dict with arrays of shape ``[batch, num_muscles]`` in newtons:
            ``active_fiber_force``, ``passive_fiber_force``, ``fiber_force`` (along the
            fiber), and ``tendon_force`` (along the path).
        """
        _, act, lmt, vmt = self._prepare(activations, coords, speeds)
        batch = lmt.shape[0]
        d_act = wp.array(act, dtype=_f32, device=self.device)
        d_lmt = wp.array(lmt, dtype=_f32, device=self.device)
        d_vmt = wp.array(vmt, dtype=_f32, device=self.device)
        d_out = wp.empty((batch, self.num_muscles, 4), dtype=_f32, device=self.device)
        wp.launch(
            fiber_forces_kernel,
            dim=(batch, self.num_muscles),
            inputs=[
                d_act,
                d_lmt,
                d_vmt,
                self.d_fmax,
                self.d_l_opt,
                self.d_lt_slack,
                self.d_vmax,
                self.d_cos_penn,
                d_out,
            ],
            device=self.device,
        )
        arr = d_out.numpy().astype(np.float64)
        return {
            "active_fiber_force": arr[:, :, 0],
            "passive_fiber_force": arr[:, :, 1],
            "fiber_force": arr[:, :, 2],
            "tendon_force": arr[:, :, 3],
        }

    def fiber_kinematics(self, coords: np.ndarray, speeds: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """Return rigid-tendon fiber kinematics for a batch of configurations.

        Evaluates the constant-width pennation model used by the rigid-tendon
        force (``pennated_fiber_length``) on-device
        and returns the fiber-state quantities OpenSim's muscle analysis reports.

        Args:
            coords: Coordinate configurations [batch, num_coordinates].
            speeds: Coordinate speeds [batch, num_coordinates]; ``None`` treats the
                pose as isometric (zero fiber velocity).

        Returns:
            Dict with arrays of shape ``[batch, num_muscles]``:
            ``fiber_length`` [m], ``normalized_fiber_length`` (fiber length / optimal
            fiber length), ``pennation_angle`` [rad], and ``normalized_fiber_velocity``
            (fiber velocities per max contraction velocity; positive = lengthening).
        """
        # ``activations`` are irrelevant to kinematics; pass a zero column of the right shape.
        coords = np.ascontiguousarray(np.atleast_2d(coords), dtype=np.float64)
        _, _, lmt, vmt = self._prepare(np.zeros(self.num_muscles), coords, speeds)
        batch = lmt.shape[0]
        d_lmt = wp.array(lmt, dtype=_f32, device=self.device)
        d_vmt = wp.array(vmt, dtype=_f32, device=self.device)
        d_out = wp.empty((batch, self.num_muscles, 4), dtype=_f32, device=self.device)
        wp.launch(
            fiber_kinematics_kernel,
            dim=(batch, self.num_muscles),
            inputs=[d_lmt, d_vmt, self.d_l_opt, self.d_lt_slack, self.d_vmax, self.d_cos_penn, d_out],
            device=self.device,
        )
        arr = d_out.numpy().astype(np.float64)
        return {
            "fiber_length": arr[:, :, 0],
            "normalized_fiber_length": arr[:, :, 1],
            "pennation_angle": arr[:, :, 2],
            "normalized_fiber_velocity": arr[:, :, 3],
        }

    def generalized_forces(
        self, activations: np.ndarray, coords: np.ndarray, speeds: np.ndarray | None = None, eps: float = 1.0e-5
    ) -> np.ndarray:
        r"""Return muscle-generated generalized forces [N or N·m], shape ``[batch, num_coordinates]``.

        The generalized force on coordinate :math:`q_i` is
        :math:`\tau_i=\sum_m r_{m,i}\,F_m`, with moment arms :math:`r_{m,i}` and
        rigid-tendon muscle forces :math:`F_m`.

        Args:
            activations: Muscle activations in [0, 1].
            coords: Coordinate configurations [batch, num_coordinates].
            speeds: Coordinate speeds [batch, num_coordinates] or ``None``.
            eps: Central-difference step for the moment arms [rad or m].
        """
        d_act, q_wp, d_speeds = self._device_inputs(activations, coords, speeds)
        batch = q_wp.shape[0]
        nc = len(self.coordinate_names)
        d_r = self.paths._moment_arms_device(q_wp, eps)
        d_force = self._forces_device(d_act, q_wp, d_speeds, eps, moment_arms=d_r)
        d_tau = wp.empty((batch, nc), dtype=_f64, device=self.device)
        wp.launch(
            generalized_force_kernel,
            dim=(batch, nc),
            inputs=[d_r, d_force, self.num_muscles, d_tau],
            device=self.device,
        )
        return d_tau.numpy()


class PathSpringForces:
    """Warp-native path-spring forces and the generalized forces they produce.

    Evaluates every OpenSim ``PathSpring`` in a model: its tension follows from
    the stretch of its geometry path beyond the resting length (with a
    dissipation term coupling to the path lengthening rate), and that tension is
    projected onto the model coordinates through the path moment arms. Reuses
    :class:`~newton.opensim.MusclePaths` for the shared path
    length, velocity, and moment-arm machinery.

    Args:
        model: Parsed OpenSim model.
        device: Warp device (defaults to CPU, matching the rest of the port).
    """

    def __init__(self, model: OsimModel, device=None):
        self.model = model
        springs = model.path_springs
        # MusclePaths only reads path_points/wraps, so a view whose "muscles" are
        # the springs reuses the shared path machinery unchanged.
        view = copy.copy(model)
        view.muscles = springs
        self.paths = MusclePaths(view, device=device)
        self.device = self.paths.device
        self.spring_names: list[str] = [s.name for s in springs]
        self.num_springs = len(springs)
        self.coordinate_names = self.paths.coordinate_names
        self.d_stiffness = wp.array([s.stiffness for s in springs], dtype=_f64, device=self.device)
        self.d_resting_length = wp.array([s.resting_length for s in springs], dtype=_f64, device=self.device)
        self.d_dissipation = wp.array([s.dissipation for s in springs], dtype=_f64, device=self.device)

    def _forces_device(
        self,
        coords: wp.array[_f64],
        speeds: wp.array[_f64] | None,
        eps: float = 1.0e-5,
        moment_arms: wp.array[_f64] | None = None,
        lengths: wp.array[_f64] | None = None,
    ) -> wp.array[_f32]:
        """Evaluate path-spring tensions without leaving the device."""
        batch = coords.shape[0]
        d_lengths = self.paths._lengths_qwp(coords) if lengths is None else lengths
        if speeds is None:
            d_ldot = wp.zeros((batch, self.num_springs), dtype=_f64, device=self.device)
        else:
            d_ldot = self.paths._velocities_qwp(coords, speeds, eps, moment_arms)
        d_ten = wp.empty((batch, self.num_springs), dtype=_f32, device=self.device)
        wp.launch(
            spring_force_kernel,
            dim=(batch, self.num_springs),
            inputs=[
                d_lengths,
                d_ldot,
                self.d_stiffness,
                self.d_resting_length,
                self.d_dissipation,
                d_ten,
            ],
            device=self.device,
        )
        return d_ten

    def forces(self, coords: np.ndarray, speeds: np.ndarray | None = None) -> np.ndarray:
        """Return path-spring tensions [N], shape ``[batch, num_springs]``.

        Args:
            coords: Coordinate configurations [batch, num_coordinates].
            speeds: Coordinate speeds [batch, num_coordinates] or ``None`` (static).
        """
        coords = np.ascontiguousarray(np.atleast_2d(coords), dtype=np.float64)
        q_wp = wp.array(coords, dtype=_f64, device=self.device)
        speeds_wp = None
        if speeds is not None:
            speeds = np.ascontiguousarray(np.atleast_2d(speeds), dtype=np.float64)
            if speeds.shape != coords.shape:
                raise ValueError(f"speeds must have shape {coords.shape}, got {speeds.shape}")
            speeds_wp = wp.array(speeds, dtype=_f64, device=self.device)
        return self._forces_device(q_wp, speeds_wp).numpy()

    def generalized_forces(
        self, coords: np.ndarray, speeds: np.ndarray | None = None, eps: float = 1.0e-5
    ) -> np.ndarray:
        r"""Return path-spring generalized forces [N or N·m], shape ``[batch, num_coordinates]``.

        The generalized force on coordinate :math:`q_i` is
        :math:`	au_i=\sum_s r_{s,i}\,F_s`, with path moment arms
        :math:`r_{s,i}=-\partial L_s/\partial q_i` and spring tensions :math:`F_s`.

        Args:
            coords: Coordinate configurations [batch, num_coordinates].
            speeds: Coordinate speeds [batch, num_coordinates] or ``None``.
            eps: Central-difference step for the moment arms [rad or m].
        """
        coords = np.ascontiguousarray(np.atleast_2d(coords), dtype=np.float64)
        batch = coords.shape[0]
        nc = len(self.coordinate_names)
        q_wp = wp.array(coords, dtype=_f64, device=self.device)
        speeds_wp = None
        if speeds is not None:
            speeds = np.ascontiguousarray(np.atleast_2d(speeds), dtype=np.float64)
            if speeds.shape != coords.shape:
                raise ValueError(f"speeds must have shape {coords.shape}, got {speeds.shape}")
            speeds_wp = wp.array(speeds, dtype=_f64, device=self.device)
        d_r = self.paths._moment_arms_device(q_wp, eps)
        d_force = self._forces_device(q_wp, speeds_wp, eps, moment_arms=d_r)
        d_tau = wp.empty((batch, nc), dtype=_f64, device=self.device)
        wp.launch(
            generalized_force_kernel,
            dim=(batch, nc),
            inputs=[d_r, d_force, self.num_springs, d_tau],
            device=self.device,
        )
        return d_tau.numpy()


class LigamentForces:
    """Warp-native ligament forces and the generalized forces they produce.

    Evaluates every OpenSim ``Ligament``: its tension is ``pcsa_force`` scaled by a
    normalized force-length curve sampled at ``length / resting_length``, and that
    tension is projected onto the model coordinates through the path moment arms.
    Reuses :class:`~newton.opensim.MusclePaths` for the shared
    path length and moment-arm machinery and the baked-function evaluator for the
    force-length curve.

    Args:
        model: Parsed OpenSim model.
        device: Warp device (defaults to CPU, matching the rest of the port).
    """

    def __init__(self, model: OsimModel, device=None):
        self.model = model
        ligaments = model.ligaments
        # MusclePaths only reads path_points/wraps, so a view whose "muscles" are
        # the ligaments reuses the shared path machinery unchanged.
        view = copy.copy(model)
        view.muscles = ligaments
        self.paths = MusclePaths(view, device=device)
        self.device = self.paths.device
        self.ligament_names: list[str] = [g.name for g in ligaments]
        self.num_ligaments = len(ligaments)
        self.coordinate_names = self.paths.coordinate_names
        ctype: list[int] = []
        cp0: list[float] = []
        cp1: list[float] = []
        koff: list[int] = []
        kcnt: list[int] = []
        kx: list[float] = []
        ky: list[float] = []
        kb: list[float] = []
        kc: list[float] = []
        kd: list[float] = []
        for g in ligaments:
            curve = g.force_length_curve
            code, p0, p1, x, y, b, c, d = _bake_function(curve.get("type"), curve)
            ctype.append(code)
            cp0.append(p0)
            cp1.append(p1)
            if len(x) > 0:
                koff.append(len(kx))
                kcnt.append(len(x))
                kx.extend(x)
                ky.extend(y)
                if code == _FN_SIMMSPLINE:
                    kb.extend(b)
                    kc.extend(c)
                    kd.extend(d)
                else:
                    kb.extend([0.0] * len(x))
                    kc.extend([0.0] * len(x))
                    kd.extend([0.0] * len(x))
            else:
                koff.append(0)
                kcnt.append(0)
        if not kx:  # ensure knot arrays are never empty for the kernel launch
            kx, ky, kb, kc, kd = [0.0], [0.0], [0.0], [0.0], [0.0]
        self.d_resting_length = wp.array([g.resting_length for g in ligaments], dtype=_f64, device=self.device)
        self.d_pcsa_force = wp.array([g.pcsa_force for g in ligaments], dtype=_f64, device=self.device)
        self.d_ctype = wp.array(ctype, dtype=wp.int32, device=self.device)
        self.d_cp0 = wp.array(cp0, dtype=_f64, device=self.device)
        self.d_cp1 = wp.array(cp1, dtype=_f64, device=self.device)
        self.d_koff = wp.array(koff, dtype=wp.int32, device=self.device)
        self.d_kcnt = wp.array(kcnt, dtype=wp.int32, device=self.device)
        self.d_kx = wp.array(kx, dtype=_f64, device=self.device)
        self.d_ky = wp.array(ky, dtype=_f64, device=self.device)
        self.d_kb = wp.array(kb, dtype=_f64, device=self.device)
        self.d_kc = wp.array(kc, dtype=_f64, device=self.device)
        self.d_kd = wp.array(kd, dtype=_f64, device=self.device)

    def _forces_device(self, coords: wp.array[_f64], lengths: wp.array[_f64] | None = None) -> wp.array[_f32]:
        """Evaluate ligament tensions without leaving the device."""
        batch = coords.shape[0]
        d_lengths = self.paths._lengths_qwp(coords) if lengths is None else lengths
        d_ten = wp.empty((batch, self.num_ligaments), dtype=_f32, device=self.device)
        wp.launch(
            ligament_force_kernel,
            dim=(batch, self.num_ligaments),
            inputs=[
                d_lengths,
                self.d_resting_length,
                self.d_pcsa_force,
                self.d_ctype,
                self.d_cp0,
                self.d_cp1,
                self.d_koff,
                self.d_kcnt,
                self.d_kx,
                self.d_ky,
                self.d_kb,
                self.d_kc,
                self.d_kd,
                d_ten,
            ],
            device=self.device,
        )
        return d_ten

    def forces(self, coords: np.ndarray) -> np.ndarray:
        """Return ligament tensions [N], shape ``[batch, num_ligaments]``.

        Args:
            coords: Coordinate configurations [batch, num_coordinates].
        """
        coords = np.ascontiguousarray(np.atleast_2d(coords), dtype=np.float64)
        q_wp = wp.array(coords, dtype=_f64, device=self.device)
        return self._forces_device(q_wp).numpy()

    def generalized_forces(self, coords: np.ndarray, eps: float = 1.0e-5) -> np.ndarray:
        r"""Return ligament generalized forces [N or N·m], shape ``[batch, num_coordinates]``.

        The generalized force on coordinate :math:`q_i` is
        :math:`	au_i=\sum_g r_{g,i}\,F_g`, with path moment arms
        :math:`r_{g,i}=-\partial L_g/\partial q_i` and ligament tensions :math:`F_g`.

        Args:
            coords: Coordinate configurations [batch, num_coordinates].
            eps: Central-difference step for the moment arms [rad or m].
        """
        coords = np.ascontiguousarray(np.atleast_2d(coords), dtype=np.float64)
        batch = coords.shape[0]
        nc = len(self.coordinate_names)
        q_wp = wp.array(coords, dtype=_f64, device=self.device)
        d_r = self.paths._moment_arms_device(q_wp, eps)
        d_force = self._forces_device(q_wp)
        d_tau = wp.empty((batch, nc), dtype=_f64, device=self.device)
        wp.launch(
            generalized_force_kernel,
            dim=(batch, nc),
            inputs=[d_r, d_force, self.num_ligaments, d_tau],
            device=self.device,
        )
        return d_tau.numpy()


def compute_muscle_generalized_forces(
    model: OsimModel,
    activations: np.ndarray,
    coords: np.ndarray,
    speeds: np.ndarray | None = None,
    device=None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Compute muscle forces and the generalized forces they produce.

    Args:
        model: Parsed OpenSim model.
        activations: Muscle activations in [0, 1].
        coords: Coordinate configurations [batch, num_coordinates].
        speeds: Coordinate speeds [batch, num_coordinates] or ``None``.
        device: Warp device (defaults to CPU).

    Returns:
        ``(forces, generalized_forces, muscle_names)`` where ``forces`` is
        ``[batch, num_muscles]`` [N] and ``generalized_forces`` is
        ``[batch, num_coordinates]`` [N or N·m].
    """
    mf = MuscleForces(model, device=device)
    coords = np.atleast_2d(coords)
    return mf.forces(activations, coords, speeds), mf.generalized_forces(activations, coords, speeds), mf.muscle_names
