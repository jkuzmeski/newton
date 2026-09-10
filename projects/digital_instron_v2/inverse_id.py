# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Gradient-based foam material identification from a force-displacement curve.

This is the flagship payoff of the differentiable elastic foundation
(:mod:`projects.digital_instron_v2.dynamics_diff`): fitting the calibrated foam
material to a measured reaction-force curve with *exact* gradients instead of a
derivative-free sweep. A digital Instron drives the column bed through a
prescribed compression cycle and records the total reaction force at every
sample; :func:`fit_material_to_force_curve` differentiates that whole cycle --
including the generalized-Maxwell loading/unloading hysteresis -- with respect to
the foam material and descends a force-matching loss.

The problem is well conditioned because the reaction force scales directly with
the constitutive parameters (unlike a free-settling height, which is only weakly
sensitive to stiffness). The parameters span very different magnitudes, so the
optimizer works in a dimensionless *scale* space ``material = scale * reference``
and takes Adam steps on the scale vector.

Run the self-identification demo (recover a perturbed material from its own
synthetic curve)::

    uv run python -m projects.digital_instron_v2.inverse_id
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp

from . import dynamics, workflow
from .core import (
    EFFECTIVE_POISSON_RATIO,
    HYSTERESIS_WEIGHT,
    MAXWELL_RELAXATION_TIME_S,
    PEAK_WEIGHT,
    SURROUND_CHECK_EVERY,
    SURROUND_OVER_RELAXATION,
    SURROUND_PASSES,
    SURROUND_SOLVE_TOLERANCE_M,
    SURROUND_TOLERANCE_M,
    Material,
    metrics,
    predict,
)
from .dynamics import FoundationParams
from .dynamics_diff import (
    MAT_ALPHA,
    MAT_G_EQ,
    MAT_OVERSTRESS,
    _column_normal_force,
    _hyperfoam_pressure_diff,
    _pasternak_coupling_diff,
    _pasternak_flux,
    _surround_balance_diff,
    foundation_pressure_diff,
)
from .geometry import build_column_grid, load_mesh


@wp.kernel
def _accumulate_normal_force(
    compression: wp.array[wp.float32],
    base_pressure: wp.array[wp.float32],
    rest_len: wp.array[wp.float32],
    neighbors: wp.array2d[wp.int32],
    area: wp.array[wp.float32],
    material_params: wp.array[wp.float32],
    step: wp.int32,
    force_hist: wp.array[wp.float32],
):
    """Sum the column normal forces for one Instron sample into ``force_hist[step]``.

    Uses the shared
    :func:`~projects.digital_instron_v2.dynamics_diff._column_normal_force`, so the
    prescribed-displacement readout clamps only the unilateral ground reaction and
    then subtracts the (signed) shear flux, exactly as the runtime and
    :func:`~projects.digital_instron_v2.core.predict` do. The platen is rigid and
    quasi-static here, so there is no normal damping term.
    """
    i = wp.tid()
    g_eq = material_params[MAT_G_EQ]
    ci = compression[i]
    flux = _pasternak_flux(i, compression, rest_len, neighbors, g_eq)
    wp.atomic_add(force_hist, step, _column_normal_force(ci, base_pressure[i], area[i], flux, 0.0, 0.0))


@wp.kernel
def _force_mse(force_hist: wp.array[wp.float32], target: wp.array[wp.float32], loss: wp.array[wp.float32]):
    """Accumulate the mean-square force-matching residual across all samples."""
    k = wp.tid()
    d = force_hist[k] - target[k]
    wp.atomic_add(loss, 0, d * d)


@dataclass
class FitResult:
    """Outcome of a force-matching material identification.

    Attributes:
        material_params: Fitted ``[g_eq, alpha, overstress]`` vector
            ([Pa], [-], [-]).
        scale: Fitted dimensionless scale of each parameter relative to the
            reference used to start the fit.
        loss_history: Mean-square force residual [N^2] at each iteration.
        force: Fitted reaction-force curve [N], one value per Instron sample.
    """

    material_params: np.ndarray
    scale: np.ndarray
    loss_history: np.ndarray
    force: np.ndarray


class InstronReplay:
    """Differentiable digital-Instron replay of the elastic-foundation column bed.

    Drives the calibrated column bed through a prescribed uniform-compression
    schedule and records the total reaction force at every sample as a
    differentiable function of the foam ``material_params``. The generalized-
    Maxwell overstress recurrence is unrolled with per-sample history so the whole
    loading/unloading cycle can be recorded on one :class:`warp.Tape`.

    Args:
        displacement_m: Prescribed platen compression at each sample [m], shape
            ``[sample_count]``.
        dt_s: Sample duration [s], scalar or per-sample array.
        material: Calibrated :class:`~projects.digital_instron_v2.core.Material`.
        geometry: Column bed from
            :func:`~projects.digital_instron_v2.dynamics.build_foundation_geometry`.
        device: Warp device.
    """

    def __init__(self, displacement_m, dt_s, material, geometry, device=None):
        self.device = device
        disp = np.ascontiguousarray(displacement_m, np.float32)
        self.nsteps = int(len(disp))
        self.dt = (
            np.full(self.nsteps, float(dt_s), np.float32)
            if np.isscalar(dt_s)
            else np.ascontiguousarray(dt_s, np.float32)
        )
        geo = geometry
        m = int(len(geo.slack_m))
        self.column_count = m

        params = FoundationParams()
        self.reference = _params_from_material(material)
        params.beta = dynamics.EFFECTIVE_POISSON_RATIO / (1.0 - 2.0 * dynamics.EFFECTIVE_POISSON_RATIO)
        params.one_minus_two_poisson = 1.0 - 2.0 * dynamics.EFFECTIVE_POISSON_RATIO
        # The relaxation time is identified with the rest of the material, so it travels
        # with it instead of sitting at the historical module default.
        params.tau_s = float(getattr(material, "maxwell_relaxation_time_s", MAXWELL_RELAXATION_TIME_S))
        params.inv_h2 = 1.0 / geo.spacing_m**2
        params.stretch_floor = 0.05
        self.params = params

        # Prescribed poses: pure downward translation by the platen displacement, so
        # every column sees a uniform compression equal to displacement[k].
        poses = np.zeros((self.nsteps, 7), np.float32)
        poses[:, 2] = -disp
        poses[:, 6] = 1.0
        self.pose = wp.array(poses, dtype=wp.transform, device=device)

        anchor = np.column_stack([geo.uv_m[:, 0], geo.uv_m[:, 1], geo.slack_m])
        self.anchor_local = wp.array(np.ascontiguousarray(anchor, np.float32), dtype=wp.vec3, device=device)
        self.z_free = wp.array(np.ascontiguousarray(geo.slack_m, np.float32), dtype=wp.float32, device=device)
        self.rest_len = wp.array(np.ascontiguousarray(geo.slack_m, np.float32), dtype=wp.float32, device=device)
        self.area = wp.array(np.full(m, geo.area_m2, np.float32), dtype=wp.float32, device=device)
        self.neighbors = wp.array(np.ascontiguousarray(geo.neighbors, np.int32), dtype=wp.int32, device=device)

        self.material_params = wp.array(self.reference.copy(), dtype=wp.float32, device=device, requires_grad=True)

        def grad_zeros():
            return wp.zeros(m, dtype=wp.float32, device=device, requires_grad=True)

        self.compression = [grad_zeros() for _ in range(self.nsteps)]
        self.base_pressure = [grad_zeros() for _ in range(self.nsteps)]
        self.q_state = [grad_zeros() for _ in range(self.nsteps)]
        self.peq_hist = [grad_zeros() for _ in range(self.nsteps)]
        self.q_init = grad_zeros()
        self.peq_init = grad_zeros()
        self.force = wp.zeros(self.nsteps, dtype=wp.float32, device=device, requires_grad=True)

    def set_material(self, material_params: np.ndarray) -> None:
        """Overwrite the differentiable material vector with host values."""
        self.material_params.assign(np.ascontiguousarray(material_params, np.float32))

    def forward(self) -> wp.array:
        """Replay the compression cycle and return the per-sample reaction force [N]."""
        self.force.zero_()
        for k in range(self.nsteps):
            q_prev = self.q_init if k == 0 else self.q_state[k - 1]
            peq_prev = self.peq_init if k == 0 else self.peq_hist[k - 1]
            wp.launch(
                foundation_pressure_diff,
                dim=self.column_count,
                inputs=[
                    k,
                    float(self.dt[k]),
                    self.pose,
                    self.anchor_local,
                    self.z_free,
                    self.rest_len,
                    self.params,
                    self.material_params,
                    q_prev,
                    peq_prev,
                    self.q_state[k],
                    self.peq_hist[k],
                    self.compression[k],
                    self.base_pressure[k],
                ],
                device=self.device,
            )
            wp.launch(
                _accumulate_normal_force,
                dim=self.column_count,
                inputs=[
                    self.compression[k],
                    self.base_pressure[k],
                    self.rest_len,
                    self.neighbors,
                    self.area,
                    self.material_params,
                    k,
                    self.force,
                ],
                device=self.device,
            )
        return self.force

    def zero_grad(self) -> None:
        """Zero the gradients on every differentiable buffer."""
        self.material_params.grad.zero_()
        self.force.grad.zero_()
        for buf in (*self.compression, *self.base_pressure, *self.q_state, *self.peq_hist):
            buf.grad.zero_()
        self.q_init.grad.zero_()
        self.peq_init.grad.zero_()


def fit_material_to_force_curve(
    target_force,
    displacement_m,
    dt_s,
    material,
    geometry,
    scale0=None,
    fit_mask=None,
    iterations: int = 300,
    learning_rate: float = 0.02,
    device=None,
) -> FitResult:
    """Fit the foam material to a measured reaction-force curve by gradient descent.

    Minimizes the mean-square force residual over the compression cycle with Adam,
    optimizing a dimensionless per-parameter scale ``material = scale * reference``
    so the very different parameter magnitudes stay comparable. Gradients come from
    a single backward pass over the differentiable Instron replay.

    Args:
        target_force: Measured reaction force at each sample [N].
        displacement_m: Prescribed platen compression at each sample [m].
        dt_s: Sample duration [s], scalar or per-sample array.
        material: Reference :class:`~projects.digital_instron_v2.core.Material`
            supplying the parameter scales and the fixed constitutive constants.
        geometry: Column bed geometry.
        scale0: Initial parameter scale (defaults to ones).
        fit_mask: Optional length-3 mask selecting which parameters to optimize;
            zeros hold a parameter fixed at its ``scale0`` value (defaults to all
            ones). Pin the Maxwell overstress ratio here when it is measured
            separately from a stress-relaxation test -- the force-displacement
            curve constrains stiffness and overstress along a strongly correlated
            direction, so freeing both makes the fit ill-conditioned.
        iterations: Number of Adam iterations.
        learning_rate: Adam step size in scale space.
        device: Warp device.

    Returns:
        The fitted material vector, scale, loss history, and fitted force curve.
    """
    replay = InstronReplay(displacement_m, dt_s, material, geometry, device=device)
    ref = replay.reference
    target = wp.array(np.ascontiguousarray(target_force, np.float32), dtype=wp.float32, device=device)
    loss = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)

    scale = np.ones(3, np.float32) if scale0 is None else np.ascontiguousarray(scale0, np.float32).copy()
    mask = np.ones(3, np.float64) if fit_mask is None else np.ascontiguousarray(fit_mask, np.float64)
    m1 = np.zeros(3, np.float64)
    m2 = np.zeros(3, np.float64)
    beta1, beta2, eps = 0.9, 0.999, 1.0e-8
    history = np.empty(iterations, np.float32)

    for it in range(iterations):
        replay.set_material(scale * ref)
        replay.zero_grad()
        loss.zero_()
        tape = wp.Tape()
        with tape:
            force = replay.forward()
            wp.launch(_force_mse, dim=replay.nsteps, inputs=[force, target, loss], device=device)
        tape.backward(loss)
        history[it] = float(loss.numpy()[0])
        grad = replay.material_params.grad.numpy() * ref  # chain rule into scale space
        tape.zero()

        g = grad.astype(np.float64) * mask  # freeze masked-out parameters
        m1 = beta1 * m1 + (1.0 - beta1) * g
        m2 = beta2 * m2 + (1.0 - beta2) * g * g
        m1_hat = m1 / (1.0 - beta1 ** (it + 1))
        m2_hat = m2 / (1.0 - beta2 ** (it + 1))
        scale = (scale - learning_rate * m1_hat / (np.sqrt(m2_hat) + eps)).astype(np.float32)

    replay.set_material(scale * ref)
    force = replay.forward().numpy().copy()
    return FitResult(material_params=scale * ref, scale=scale, loss_history=history, force=force)


# ---------------------------------------------------------------------------
# Fitting the foam material to measured Instron trials (shaped indenters).
#
# Unlike :class:`InstronReplay`, which drives a synthetic *uniform* platen, the
# real digital-Instron fixtures are shaped indenters (a spherical rearfoot punch
# and a full-foot shoe last), so each column sees its own compression history.
# :class:`DifferentiableTrial` is the autodiff sibling of
# :func:`~projects.digital_instron_v2.core.predict`: it reads the per-column
# strain history baked into a :class:`~projects.digital_instron_v2.core.Trial`
# and reproduces that forward model -- Hyperfoam equilibrium pressure, the exact
# periodic generalized-Maxwell overstress fixed point, and the precomputed
# material-pinned Pasternak coupling -- as a differentiable function of the foam
# material.
# ---------------------------------------------------------------------------


@wp.kernel
def _trial_equilibrium_pressure(
    strain: wp.array2d[wp.float32],
    frame: wp.int32,
    params: FoundationParams,
    material_params: wp.array[wp.float32],
    peq_out: wp.array[wp.float32],
):
    """Hyperfoam equilibrium pressure for every column at one measured frame."""
    i = wp.tid()
    g_eq = material_params[MAT_G_EQ]
    alpha = material_params[MAT_ALPHA]
    peq_out[i] = _hyperfoam_pressure_diff(strain[frame, i], g_eq, alpha, params)


@wp.kernel
def _trial_maxwell_step(
    peq_cur: wp.array[wp.float32],
    peq_prev: wp.array[wp.float32],
    decay: wp.float32,
    ramp: wp.float32,
    material_params: wp.array[wp.float32],
    q_prev: wp.array[wp.float32],
    q_out: wp.array[wp.float32],
):
    """One tape-safe linear-overstress recurrence step (writes ``q_out`` from ``q_prev``)."""
    i = wp.tid()
    overstress = material_params[MAT_OVERSTRESS]
    q_out[i] = decay * q_prev[i] + overstress * ramp * (peq_cur[i] - peq_prev[i])


@wp.kernel
def _scale_state(state: wp.array[wp.float32], factor: wp.float32, out: wp.array[wp.float32]):
    """Scale the transient end state into the periodic fixed-point initial state."""
    i = wp.tid()
    out[i] = state[i] * factor


@wp.kernel
def _trial_frame_force(
    peq_cur: wp.array[wp.float32],
    q_cur: wp.array[wp.float32],
    laplacian: wp.array2d[wp.float32],
    slack: wp.array[wp.float32],
    frame: wp.int32,
    area: wp.array[wp.float32],
    material_params: wp.array[wp.float32],
    force_hist: wp.array[wp.float32],
):
    """Sum the column forces for one measured frame into ``force_hist[frame]``.

    Clamps only the unilateral ground reaction ``p_eq + q`` and then subtracts the
    shear-layer flux, matching :func:`~projects.digital_instron_v2.core.predict`
    and the runtime. Clamping their sum (the previous behaviour) clipped the flux
    and invented support under columns carrying no compression.

    This is the fixture-subset path, whose lateral term is the lumped per-column
    coefficient ``mu_eq * t_i`` acting on the precomputed subset Laplacian, as in
    :func:`projects.digital_instron_v2.core._column_bed_force`. The whole-bed
    path uses the pairwise face form instead (:func:`_surround_sweep_diff`).
    """
    i = wp.tid()
    coupling = material_params[MAT_G_EQ] * slack[i]
    ground = wp.max(peq_cur[i] + q_cur[i], 0.0) - coupling * laplacian[frame, i]
    wp.atomic_add(force_hist, frame, ground * area[i])


@wp.kernel
def _weighted_force_mse(
    force_hist: wp.array[wp.float32],
    target: wp.array[wp.float32],
    weight: wp.float32,
    loss: wp.array[wp.float32],
):
    """Accumulate a per-trial-weighted mean-square force residual into ``loss``."""
    k = wp.tid()
    d = force_hist[k] - target[k]
    wp.atomic_add(loss, 0, weight * d * d)


# ---------------------------------------------------------------------------
# Whole-bed passive surround.
#
# :func:`~projects.digital_instron_v2.core.predict` models the *whole* midsole
# whenever a trial carries a :class:`~projects.digital_instron_v2.core.Surround`:
# the indenter drives its own columns and the rest of the foam relaxes against
# neighbour shear, its own unilateral ground reaction and an assumed vertical
# bond to the shoe, under a Maxwell overstress that is made self-consistent with
# the compression it produces. The kernels below are the differentiable
# transcription of that solve.
#
# The shipped runtime kernels (:mod:`projects.digital_shoe.runtime`) cannot be
# reused on a :class:`warp.Tape`: they read the equilibrium modulus, the
# Hyperfoam exponent and the overstress ratio out of a
# by-value :class:`~projects.digital_shoe.runtime.FoundationParams` struct (or
# out of host-computed launch scalars), and Warp only accumulates adjoints into
# *arrays*. A differentiable material must therefore enter as elements of the
# ``requires_grad`` ``material_params`` vector. The balance formula itself is not
# duplicated -- these kernels call the shared
# :func:`~projects.digital_instron_v2.dynamics_diff._surround_balance_diff`.
# ---------------------------------------------------------------------------


@wp.kernel
def _surround_pressure_field(
    compression: wp.array2d[wp.float32],
    slack: wp.array[wp.float32],
    params: FoundationParams,
    material_params: wp.array[wp.float32],
    peq_out: wp.array2d[wp.float32],
):
    """Hyperfoam equilibrium pressure of every bed column at every frame [Pa]."""
    frame, i = wp.tid()
    peq_out[frame, i] = _hyperfoam_pressure_diff(
        compression[frame, i] / slack[i], material_params[MAT_G_EQ], material_params[MAT_ALPHA], params
    )


@wp.kernel
def _surround_pressure_increment(peq: wp.array2d[wp.float32], increment_out: wp.array2d[wp.float32]):
    """Cyclic per-frame equilibrium-pressure increment ``p_eq[f] - p_eq[f - 1]`` [Pa].

    The first frame wraps onto the last one, which is what makes the overstress
    branch periodic (the cycle is a closed loop, as in
    :func:`projects.digital_shoe.runtime.cycle_overstress`). Pulling the
    increment out of the recurrence keeps the loop body affine in its
    loop-carried state, which is what Warp differentiates reliably.
    """
    frame, i = wp.tid()
    previous = frame - 1
    if previous < 0:
        previous = peq.shape[0] - 1
    increment_out[frame, i] = peq[frame, i] - peq[previous, i]


@wp.kernel
def _surround_cycle_overstress(
    increment: wp.array2d[wp.float32],
    decay: wp.array[wp.float32],
    ramp: wp.array[wp.float32],
    fixed_point_gain: wp.float32,
    material_params: wp.array[wp.float32],
    overstress_out: wp.array2d[wp.float32],
):
    """Periodic generalized-Maxwell overstress of every column and frame [Pa].

    Differentiable transcription of
    :func:`projects.digital_shoe.runtime.cycle_overstress`: one pass finds the
    periodic state, the second records it frame by frame. ``decay``, ``ramp`` and
    ``fixed_point_gain`` depend only on the sample times and the (fixed)
    relaxation time, so they are precomputed on the host; the overstress ratio is
    read from the differentiable material vector.
    """
    i = wp.tid()
    frames = increment.shape[0]
    fraction = material_params[MAT_OVERSTRESS]
    state = float(0.0)
    for frame in range(frames):
        state = decay[frame] * state + fraction * ramp[frame] * increment[frame, i]
    state = state * fixed_point_gain
    for frame in range(frames):
        state = decay[frame] * state + fraction * ramp[frame] * increment[frame, i]
        overstress_out[frame, i] = state


@wp.kernel
def _surround_sweep_diff(
    compression_in: wp.array2d[wp.float32],
    overstress: wp.array2d[wp.float32],
    driven: wp.array[wp.int32],
    neighbors: wp.array2d[wp.int32],
    slack: wp.array[wp.float32],
    params: FoundationParams,
    material_params: wp.array[wp.float32],
    area: wp.float32,
    attachment: wp.float32,
    max_strain: wp.float32,
    compression_out: wp.array2d[wp.float32],
):
    """One Jacobi sweep of the passive surround, differentiable in the material.

    Frame-parallel twin of :func:`projects.digital_shoe.runtime.surround_sweep`
    (driven columns pass through, ``rigid = 0``, ``overstress_gain = 0``,
    ``relaxation = 1``, ``carrier_bond = 0``), with the material-pinned face
    coefficient ``k_ij = mu_eq * (t_i + t_j) / 2`` rebuilt inside the kernel from
    the differentiable material vector. The shear layer never reaches the summed
    reaction directly -- the pairwise flux cancels over the bed -- so this sweep
    is the *only* path by which the lateral coupling influences the predicted
    load, and it has to carry a gradient.
    """
    frame, i = wp.tid()
    c = compression_in[frame, i]
    if driven[i] != 0:
        compression_out[frame, i] = c
    else:
        g_eq = material_params[MAT_G_EQ]
        pull = float(0.0)
        coupling_sum = float(0.0)
        for side in range(4):
            j = neighbors[i, side]
            if j >= 0:
                coupling = _pasternak_coupling_diff(slack[i], slack[j], g_eq)
                pull += coupling * (compression_in[frame, j] - c)
                coupling_sum += coupling
        compression_out[frame, i] = _surround_balance_diff(
            c,
            0.0,
            pull,
            coupling_sum,
            slack[i],
            overstress[frame, i],
            0.0,
            g_eq,
            material_params[MAT_ALPHA],
            params,
            area,
            attachment,
            max_strain,
            1.0,
            0,
        )


@wp.kernel
def _surround_cycle_force(
    peq: wp.array2d[wp.float32],
    overstress: wp.array2d[wp.float32],
    area: wp.float32,
    force_out: wp.array[wp.float32],
):
    """Sum the unilateral ground reaction of the whole bed into each frame [N].

    Differentiable twin of :func:`projects.digital_shoe.runtime.cycle_force`: the
    clamp keeps the reaction unilateral and the shear flux cancels internally, so
    this sum is the load the Instron measures.
    """
    frame, i = wp.tid()
    wp.atomic_add(force_out, frame, area * wp.max(peq[frame, i] + overstress[frame, i], 0.0))


@wp.kernel
def _damped_sweep(
    compression: wp.array2d[wp.float32],
    sweep_out: wp.array2d[wp.float32],
    relaxation: wp.float32,
    damped_out: wp.array2d[wp.float32],
):
    """Blend one Jacobi sweep with the compression it started from.

    The adjoint of the surround fixed point is summed as a Neumann series
    ``sum_k (A^T)^k`` (:meth:`DifferentiableTrial.accumulate_gradient`), which only
    converges when the iteration ``A`` is a contraction. One undamped pass is not:
    relaxing against an overstress and then refreshing the overstress from the
    relaxed compression has loop gain ``-(1 - eq) / eq`` times the periodic
    Maxwell gain, which is why the forward solve damps that refresh
    (:func:`projects.digital_instron_v2.core._surround_force`). Seeding the adjoint
    through this blend iterates ``A_w = (1 - w) I + w A`` instead. It needs no
    correction afterwards: the parameter Jacobian of the blend is ``w dG/dt`` and
    ``(I - A_w)^-1 = (1 / w) (I - A)^-1``, so the ``w`` cancels exactly and the
    damped series sums to the same gradient.
    """
    frame, i = wp.tid()
    damped_out[frame, i] = (1.0 - relaxation) * compression[frame, i] + relaxation * sweep_out[frame, i]


@wp.kernel
def _mask_driven_columns(driven: wp.array[wp.int32], field: wp.array2d[wp.float32], masked_out: wp.array2d[wp.float32]):
    """Copy a per-column field, zeroing the columns the indenter drives.

    The driven compression is prescribed kinematics, not an unknown of the
    surround fixed point, so its adjoint seed must not be fed back into the
    adjoint iteration (:func:`_surround_sweep_diff` passes driven columns
    straight through, which would otherwise leave an eigenvalue of exactly one in
    the iteration matrix).
    """
    frame, i = wp.tid()
    if driven[i] != 0:
        masked_out[frame, i] = 0.0
    else:
        masked_out[frame, i] = field[frame, i]


@wp.kernel
def _sum_squares(field: wp.array2d[wp.float32], total: wp.array[wp.float32]):
    """Accumulate the squared Frobenius norm of a per-frame, per-column field."""
    frame, i = wp.tid()
    wp.atomic_add(total, 0, field[frame, i] * field[frame, i])


@wp.kernel
def _trapezoid_integral(force: wp.array[wp.float32], weights: wp.array[wp.float32], total: wp.array[wp.float32]):
    """Accumulate a fixed linear functional of the force curve (a trapezoid rule) into ``total``.

    The hysteresis-loop area ``trapz(force, displacement)`` is linear in the force
    vector for a prescribed displacement history, so the quadrature weights are
    precomputed on the host and the whole term is one differentiable weighted sum.
    """
    k = wp.tid()
    wp.atomic_add(total, 0, weights[k] * force[k])


@wp.kernel
def _scaled_square_residual(
    value: wp.array[wp.float32],
    index: wp.int32,
    target: wp.float32,
    inv_scale: wp.float32,
    weight: wp.float32,
    loss: wp.array[wp.float32],
):
    """Add ``weight * ((value[index] - target) / scale)^2`` to the loss."""
    residual = (value[index] - target) * inv_scale
    wp.atomic_add(loss, 0, weight * residual * residual)


def _params_from_material(material: Material) -> np.ndarray:
    """Pack a :class:`~projects.digital_instron_v2.core.Material` into ``[g_eq, alpha, overstress]``.

    The lateral shear layer is not in the vector: its coefficient is pinned to the
    material as ``mu_eq * t_i`` per column
    (:meth:`~projects.digital_instron_v2.core.Material.coupling_n_per_m`), so it
    follows ``g_eq`` instead of being fitted on its own.
    """
    return np.array(
        [
            material.instantaneous_shear_modulus_pa * material.equilibrium_fraction,
            material.hyperfoam_exponent,
            (1.0 - material.equilibrium_fraction) / material.equilibrium_fraction,
        ],
        np.float32,
    )


def _material_from_params(
    material_params: np.ndarray, relaxation_time_s: float = MAXWELL_RELAXATION_TIME_S
) -> Material:
    """Unpack a ``[g_eq, alpha, overstress]`` vector back into a :class:`~projects.digital_instron_v2.core.Material`.

    The relaxation time is not part of the differentiable vector, so it is carried
    through from the reference material instead of falling back to the module
    default (the calibration identifies it, and the two differ by an order of
    magnitude).

    Args:
        material_params: Fitted ``[g_eq, alpha, overstress]`` vector.
        relaxation_time_s: Maxwell relaxation time of the reference material [s].
    """
    g_eq, alpha, overstress = (float(value) for value in material_params)
    equilibrium_fraction = 1.0 / (1.0 + overstress)
    return Material(
        instantaneous_shear_modulus_pa=g_eq / equilibrium_fraction,
        hyperfoam_exponent=alpha,
        equilibrium_fraction=equilibrium_fraction,
        maxwell_relaxation_time_s=relaxation_time_s,
    )


@dataclass
class TrialFitResult:
    """Outcome of fitting one foam material to measured Instron trials.

    Attributes:
        material: Fitted :class:`~projects.digital_instron_v2.core.Material`.
        material_params: Fitted ``[g_eq, alpha, overstress]`` vector
            ([Pa], [-], [-]).
        scale: Fitted dimensionless scale of each parameter relative to the
            reference material used to start the fit.
        loss_history: Weighted mean-square force residual [N^2] at each iteration.
        rms_relative: Final per-trial force RMS residual as a fraction of that
            trial's peak measured force, keyed by trial name.
    """

    material: Material
    material_params: np.ndarray
    scale: np.ndarray
    loss_history: np.ndarray
    rms_relative: dict[str, float]


# Convergence controls for the adjoint fixed point of the surround solve (see
# :meth:`DifferentiableTrial.accumulate_gradient`). The adjoint state itself
# inherits the slow modes of the Jacobi relaxation, but those modes live on
# unloaded, clamped foam that contributes nothing to the material gradient, so
# the iteration is stopped on the *gradient* increment (plus its geometric tail
# estimate) instead of on the adjoint-state norm.
# The budget is measured, not guessed, and 300 is a correctness fix rather than a
# loosened tolerance: the rearfoot fixture needs 380 real terms at its damping
# ``w = 0.695``, so the previous budget of 200 stopped it on the cap at 288 and
# made it report ``adjoint_converged False``. The sum was already right there --
# the accumulated material gradient differs by 3e-4 relative between 288 and 380
# terms -- so what the old budget broke was the flag, not the gradient. At 300
# both shipped fixtures stop on :data:`ADJOINT_TOLERANCE` and the flag means what
# it says. The adjoint *state* is still far from zero out in the unloaded foam,
# which is why the stopping test is on the gradient and not on that state.
# The count is in *undamped* terms. Damping the iteration by ``w``
# (:func:`_damped_sweep`) slows every mode by ``1 / w``, so the real cap is
# divided by ``w`` and the budget means the same thing at any damping.
ADJOINT_MAX_ITERATIONS = 300
ADJOINT_TOLERANCE = 1.0e-4
ADJOINT_CHECK_INTERVAL = 10


class DifferentiableTrial:
    """Autodiff force predictor for one measured Instron trial (shaped indenter).

    Reproduces :func:`~projects.digital_instron_v2.core.predict` as a
    differentiable function of the foam ``material_params``, on either of the two
    branches that function takes.

    **Without a surround** the per-column strain history baked into the
    :class:`~projects.digital_instron_v2.core.Trial` drives the Hyperfoam
    equilibrium pressure, the exact periodic generalized-Maxwell overstress fixed
    point (a zero-state transient pass followed by a fixed-point pass, matching
    the cyclic steady state that ``predict`` evaluates in closed form), and the
    material-pinned Pasternak coupling. Every buffer on the loss path is written
    exactly once so the whole cycle records on one :class:`warp.Tape`.

    **With a** :class:`~projects.digital_instron_v2.core.Surround` the whole
    midsole bed is modelled, exactly as
    :func:`~projects.digital_instron_v2.core._surround_force` does: the indenter
    drives its own columns, the rest of the foam relaxes against neighbour shear,
    its own unilateral ground reaction and an assumed vertical bond to the shoe,
    and the Maxwell overstress is made self-consistent with the compression that
    relaxation produces. The forward solve runs the *shared runtime kernels*
    (:func:`projects.digital_shoe.runtime.relax_surround`,
    :func:`~projects.digital_shoe.runtime.cycle_overstress` and
    :func:`~projects.digital_shoe.runtime.cycle_force`) outside any tape, so the
    predicted force is the shipped forward model by construction, and the
    gradient comes from an implicit-function-theorem adjoint of the converged
    fixed point (see :meth:`accumulate_gradient`).

    Args:
        trial: Measured :class:`~projects.digital_instron_v2.core.Trial` with
            per-column ``lengths_m`` and ``compression_laplacian_m_inv``
            histories, and optionally a whole-midsole
            :class:`~projects.digital_instron_v2.core.Surround`.
        material: Reference :class:`~projects.digital_instron_v2.core.Material`
            supplying the parameter scales and fixed constitutive constants.
        device: Warp device.
        material_params: Optional shared ``requires_grad`` material vector; when
            given (e.g. by :func:`fit_material_to_trials` to fit one material to
            several trials at once) it is used instead of a private copy.
    """

    def __init__(self, trial, material: Material, device=None, material_params: wp.array | None = None):
        self.device = device
        self.name = trial.name
        self.surround = trial.surround
        slack = np.asarray(trial.slack_m, np.float32)
        lengths = np.ascontiguousarray(trial.lengths_m, np.float32)
        self.frame_count, self.driven_count = lengths.shape

        dt = trial.dt_s
        dt = np.full(self.frame_count, float(dt), np.float64) if np.isscalar(dt) else np.asarray(dt, np.float64)
        self.dt_s = dt
        self.tau_s = float(getattr(material, "maxwell_relaxation_time_s", MAXWELL_RELAXATION_TIME_S))
        decay = np.exp(-dt / self.tau_s)
        self.decay = decay
        self.ramp = self.tau_s * (1.0 - decay) / dt
        self.fixed_point_gain = float(1.0 / (1.0 - float(np.prod(decay))))

        params = FoundationParams()
        params.beta = EFFECTIVE_POISSON_RATIO / (1.0 - 2.0 * EFFECTIVE_POISSON_RATIO)
        params.one_minus_two_poisson = 1.0 - 2.0 * EFFECTIVE_POISSON_RATIO
        params.tau_s = self.tau_s
        params.stretch_floor = 1.0e-3  # match core.predict's stretch clamp
        self.params = params

        self.reference = _params_from_material(material)
        self.owns_material = material_params is None
        self.material_params = (
            wp.array(self.reference.copy(), dtype=wp.float32, device=device, requires_grad=True)
            if self.owns_material
            else material_params
        )

        self.measured = np.ascontiguousarray(trial.force_n, np.float32)
        self.peak = max(float(np.max(np.abs(self.measured))), 1.0e-9)
        self.force = wp.zeros(self.frame_count, dtype=wp.float32, device=device, requires_grad=True)

        # Shape residuals of the shipped identification (core._trial_residual): the
        # hysteresis-loop area and the peak force, on the same per-trial scale.
        displacement = np.asarray(trial.displacement_m, np.float64)
        weights = np.zeros(self.frame_count, np.float64)
        weights[1:] += 0.5 * np.diff(displacement)
        weights[:-1] += 0.5 * np.diff(displacement)
        self.loop_weights = wp.array(np.ascontiguousarray(weights, np.float32), dtype=wp.float32, device=device)
        self.measured_loop = float(np.trapezoid(self.measured.astype(np.float64), displacement))
        self.measured_peak = float(np.max(self.measured))
        self.residual_scale = max(self.measured_peak, 1.0)
        self.loop_value = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)

        # Diagnostics of the last surround solve / adjoint solve.
        self.surround_pass_changes: list[float] = []
        self.adjoint_iterations = 0
        self.adjoint_converged = True
        self.adjoint_tail_relative = 0.0
        self.adjoint_state_decay = 0.0

        if self.surround is None:
            self._init_columns(trial, slack, lengths)
        else:
            self._init_surround(trial, slack, lengths)

    # -- construction ------------------------------------------------------
    def _init_columns(self, trial, slack: np.ndarray, lengths: np.ndarray) -> None:
        """Allocate the driven-column-only buffers of the no-surround forward model."""
        device = self.device
        self.column_count = self.driven_count
        strain = np.maximum(slack[None, :] - lengths, 0.0) / slack[None, :]
        self.strain = wp.array(np.ascontiguousarray(strain, np.float32), dtype=wp.float32, device=device)
        laplacian = trial.compression_laplacian_m_inv
        laplacian = np.zeros_like(lengths) if laplacian is None else np.ascontiguousarray(laplacian, np.float32)
        self.laplacian = wp.array(laplacian, dtype=wp.float32, device=device)
        self.slack = wp.array(slack, dtype=wp.float32, device=device)
        area = (
            np.full(self.column_count, float(trial.area_m2), np.float32)
            if np.isscalar(trial.area_m2)
            else np.ascontiguousarray(trial.area_m2, np.float32)
        )
        self.area = wp.array(area, dtype=wp.float32, device=device)

        def grad_zeros():
            return wp.zeros(self.column_count, dtype=wp.float32, device=device, requires_grad=True)

        self.peq = [grad_zeros() for _ in range(self.frame_count)]
        self.q_transient = [grad_zeros() for _ in range(self.frame_count)]
        self.q_cycle = [grad_zeros() for _ in range(self.frame_count)]
        self.q_zero = grad_zeros()  # transient-pass initial state (stays zero)
        self.q_init = grad_zeros()  # fixed-point initial state for the cycle pass

    def _init_surround(self, trial, slack: np.ndarray, lengths: np.ndarray) -> None:
        """Allocate the whole-bed buffers of the passive-surround forward model."""
        device = self.device
        surround = self.surround
        self.column_count = int(len(surround.slack_m))
        shape = (self.frame_count, self.column_count)
        self.shape = shape
        self.driven_host = np.ascontiguousarray(surround.driven, np.int32)
        self.neighbors_host = np.ascontiguousarray(surround.neighbors, np.int32)
        self.slack_host = np.ascontiguousarray(surround.slack_m, np.float32)
        self.driven_compression = np.ascontiguousarray(np.maximum(slack[None, :] - lengths, 0.0), np.float32)
        self.bed_area_m2 = float(surround.area_m2)

        self.driven = wp.array(self.driven_host, dtype=wp.int32, device=device)
        self.neighbors = wp.array(self.neighbors_host, dtype=wp.int32, device=device)
        self.slack = wp.array(self.slack_host, dtype=wp.float32, device=device)
        self.dt_device = wp.array(np.ascontiguousarray(self.dt_s, np.float32), dtype=wp.float32, device=device)
        self.decay_device = wp.array(np.ascontiguousarray(self.decay, np.float32), dtype=wp.float32, device=device)
        self.ramp_device = wp.array(np.ascontiguousarray(self.ramp, np.float32), dtype=wp.float32, device=device)

        def field(requires_grad: bool = True):
            return wp.zeros(shape, dtype=wp.float32, device=device, requires_grad=requires_grad)

        self.compression = field()  # converged bed compression c* (the tape input)
        self.peq_field = field()
        self.increment = field()
        self.q_field = field()
        self.sweep_out = field()  # one Jacobi sweep of c*, only its adjoint is used
        self.damped_sweep = field()  # that sweep blended with c*, the contracting adjoint iteration
        # Damping of the adjoint iteration, refreshed by every solve from the
        # overstress feedback gain it has to tame (see :func:`_damped_sweep`).
        self.adjoint_relaxation = 1.0
        self.tape_force = wp.zeros(self.frame_count, dtype=wp.float32, device=device, requires_grad=True)
        self.adjoint_seed = field(requires_grad=False)
        self.overstress = field(requires_grad=False)
        self.refreshed = field(requires_grad=False)
        self.norm_scratch = wp.zeros(1, dtype=wp.float32, device=device)
        self.overstress_host = np.zeros(shape, np.float32)
        self.warm_start = None  # last converged bed, reused by the next solve
        self.solved = False

    # -- forward -----------------------------------------------------------
    def set_material(self, material_params: np.ndarray) -> None:
        """Overwrite the differentiable material vector with host values."""
        self.material_params.assign(np.ascontiguousarray(material_params, np.float32))

    def forward(self, warm_start: bool = False) -> wp.array:
        """Replay the measured cycle and return the per-frame reaction force [N].

        Args:
            warm_start: Reuse the previous solve's self-consistent overstress
                field, which only moves the *outer* pass loop closer to its own
                fixed point. Ignored without a surround.
        """
        if self.surround is not None:
            self._solve_surround(warm_start=warm_start)
            return self.force
        self.force.zero_()
        for f in range(self.frame_count):
            wp.launch(
                _trial_equilibrium_pressure,
                dim=self.column_count,
                inputs=[self.strain, f, self.params, self.material_params, self.peq[f]],
                device=self.device,
            )
        # Transient pass from a zero overstress state to find the cycle end state.
        for f in range(self.frame_count):
            q_prev = self.q_zero if f == 0 else self.q_transient[f - 1]
            peq_prev = self.peq[self.frame_count - 1] if f == 0 else self.peq[f - 1]
            wp.launch(
                _trial_maxwell_step,
                dim=self.column_count,
                inputs=[
                    self.peq[f],
                    peq_prev,
                    float(self.decay[f]),
                    float(self.ramp[f]),
                    self.material_params,
                    q_prev,
                    self.q_transient[f],
                ],
                device=self.device,
            )
        wp.launch(
            _scale_state,
            dim=self.column_count,
            inputs=[self.q_transient[self.frame_count - 1], self.fixed_point_gain, self.q_init],
            device=self.device,
        )
        # Fixed-point pass: the same recurrence from the periodic initial state, summing force.
        for f in range(self.frame_count):
            q_prev = self.q_init if f == 0 else self.q_cycle[f - 1]
            peq_prev = self.peq[self.frame_count - 1] if f == 0 else self.peq[f - 1]
            wp.launch(
                _trial_maxwell_step,
                dim=self.column_count,
                inputs=[
                    self.peq[f],
                    peq_prev,
                    float(self.decay[f]),
                    float(self.ramp[f]),
                    self.material_params,
                    q_prev,
                    self.q_cycle[f],
                ],
                device=self.device,
            )
            wp.launch(
                _trial_frame_force,
                dim=self.column_count,
                inputs=[
                    self.peq[f],
                    self.q_cycle[f],
                    self.laplacian,
                    self.slack,
                    f,
                    self.area,
                    self.material_params,
                    self.force,
                ],
                device=self.device,
            )
        return self.force

    def _relax(self, shoe, initial=None) -> wp.array:
        """Relax the whole bed against the currently carried overstress.

        Calls the shared :func:`~projects.digital_shoe.runtime.relax_surround`
        unchanged, with the same solver settings
        :func:`~projects.digital_instron_v2.core._surround_force` uses, so the
        relaxation is the shipped one by construction.

        The sweep count is a cap and the solve now stops on the extrapolated
        remaining travel, so the relaxation is a genuine inner solve to
        convergence and its fixed point no longer depends on where it starts.
        That is what makes ``initial`` safe: warm starting from the previous pass
        costs a few sweeps instead of the cold-solve thousands and reaches the
        same bed. Under the previous fixed sweep budget it would not have been --
        the budget was part of the model, and starting closer to the answer moved
        the predicted peak.

        Args:
            shoe: The :mod:`projects.digital_shoe.runtime` module.
            initial: Compression field to start from [m], shape
                ``[frames, column_count]``; ``None`` starts from the rigid
                indenter field.
        """
        surround = self.surround
        return shoe.relax_surround(
            self.driven_compression,
            self.driven_host,
            self.neighbors_host,
            self.slack_host,
            self.params,
            area_m2=self.bed_area_m2,
            spacing_m=float(surround.spacing_m),
            attachment_n_m=float(surround.attachment_n_m),
            max_strain=float(surround.max_strain),
            sweeps=int(surround.sweeps),
            over_relaxation=SURROUND_OVER_RELAXATION,
            overstress=self.overstress,
            initial=initial,
            tolerance_m=SURROUND_SOLVE_TOLERANCE_M,
            check_every=SURROUND_CHECK_EVERY,
            device=self.device,
        )

    def _solve_surround(self, warm_start: bool = False) -> None:
        """Solve the self-consistent surround fixed point with the shared runtime kernels.

        Line-for-line the algorithm of
        :func:`~projects.digital_instron_v2.core._surround_force`: relax the bed
        against the carried overstress, refresh the overstress from the relaxed
        compression, blend it in with weight ``equilibrium_fraction`` (which cancels
        the ``-(1 - eq) / eq`` loop gain of the undamped repetition), and stop once
        the compression field moves less than
        :data:`~projects.digital_instron_v2.core.SURROUND_TOLERANCE_M`. The solve
        deliberately stays *off* the tape: unrolling
        ``SURROUND_PASSES x surround.sweeps`` sweeps would need thousands of
        ``[frames, columns]`` buffers.

        ``warm_start`` only carries the overstress field over from the previous
        solve. That is the state of the *outer* pass loop, whose fixed point is
        unchanged by where it starts, so a warm-started solve returns the same
        force (to float32 rounding) in fewer passes.
        """
        from projects.digital_shoe import runtime as shoe  # noqa: PLC0415  # lazy: no device needed to import

        material = self.material_params.numpy()
        self.params.g_eq = float(material[MAT_G_EQ])
        self.params.alpha = float(material[MAT_ALPHA])
        fraction = float(material[MAT_OVERSTRESS])
        blend = 1.0 / (1.0 + fraction)  # = equilibrium fraction
        # Tame the same loop gain the forward blend tames, but for the adjoint
        # iteration: one undamped pass amplifies a compression perturbation by up
        # to the overstress ratio times the periodic Maxwell gain, so the damped
        # iteration puts that worst mode at zero and leaves the rest contracting.
        self.adjoint_relaxation = 1.0 / (1.0 + fraction * self.fixed_point_gain)

        warm = bool(warm_start and self.solved)
        if not warm:
            self.overstress_host[...] = 0.0
            self.overstress.zero_()
        changes: list[float] = []
        previous = None
        compression = None
        # Warm start every pass from the previous one, and the first pass from the
        # previous solve of this trial, exactly as core._surround_force does.
        started_from = self.warm_start if warm else None
        for _ in range(SURROUND_PASSES):
            compression = self._relax(shoe, started_from)
            started_from = compression
            wp.launch(
                shoe.cycle_overstress,
                dim=self.column_count,
                inputs=[compression, self.slack, self.dt_device, self.params, fraction, self.tau_s, self.refreshed],
                device=self.device,
            )
            self.overstress_host += blend * (self.refreshed.numpy() - self.overstress_host)
            self.overstress.assign(self.overstress_host)
            relaxed = compression.numpy()
            if previous is not None:
                changes.append(float(np.max(np.abs(relaxed - previous))))
            previous = relaxed
            if changes and changes[-1] < SURROUND_TOLERANCE_M:
                break
        self.surround_pass_changes = changes
        self.warm_start = compression
        wp.copy(self.compression, compression)
        self.force.zero_()
        wp.launch(
            shoe.cycle_force,
            dim=self.shape,
            inputs=[compression, self.refreshed, self.slack, self.params, self.bed_area_m2, self.force],
            device=self.device,
        )
        self.solved = True

    # -- gradients ---------------------------------------------------------
    def _record_surround_pass(
        self, target: wp.array, weight: float, loss: wp.array, shape_residuals: bool, peak_frame: int
    ) -> None:
        """Record one residual pass of the surround fixed point plus the loss on the tape.

        The pass is ``q = M(c)`` (the exact periodic Maxwell operator), one Jacobi
        sweep ``G(c, q)`` of the balance, and the summed reaction ``F(c, q)``. At the
        converged compression ``c*`` this is the fixed-point residual map whose
        partial derivatives :meth:`accumulate_gradient` needs.
        """
        wp.launch(
            _surround_pressure_field,
            dim=self.shape,
            inputs=[self.compression, self.slack, self.params, self.material_params, self.peq_field],
            device=self.device,
        )
        wp.launch(
            _surround_pressure_increment, dim=self.shape, inputs=[self.peq_field, self.increment], device=self.device
        )
        wp.launch(
            _surround_cycle_overstress,
            dim=self.column_count,
            inputs=[
                self.increment,
                self.decay_device,
                self.ramp_device,
                self.fixed_point_gain,
                self.material_params,
                self.q_field,
            ],
            device=self.device,
        )
        wp.launch(
            _surround_sweep_diff,
            dim=self.shape,
            inputs=[
                self.compression,
                self.q_field,
                self.driven,
                self.neighbors,
                self.slack,
                self.params,
                self.material_params,
                self.bed_area_m2,
                float(self.surround.attachment_n_m),
                float(self.surround.max_strain),
                self.sweep_out,
            ],
            device=self.device,
        )
        wp.launch(
            _damped_sweep,
            dim=self.shape,
            inputs=[self.compression, self.sweep_out, float(self.adjoint_relaxation), self.damped_sweep],
            device=self.device,
        )
        wp.launch(
            _surround_cycle_force,
            dim=self.shape,
            inputs=[self.peq_field, self.q_field, self.bed_area_m2, self.tape_force],
            device=self.device,
        )
        self._record_loss(self.tape_force, target, weight, loss, shape_residuals, peak_frame)

    def _record_loss(
        self,
        force: wp.array,
        target: wp.array,
        weight: float,
        loss: wp.array,
        shape_residuals: bool,
        peak_frame: int,
    ) -> None:
        """Record this trial's loss contribution on the tape.

        The default is the peak-normalized force MSE. ``shape_residuals`` adds the
        other two residuals of the shipped identification
        (:func:`~projects.digital_instron_v2.core._trial_residual`): the
        hysteresis-loop area and the peak force, squared with the same
        ``HYSTERESIS_WEIGHT`` and ``PEAK_WEIGHT`` and the same per-trial scale, so
        the gradient fit descends the identification's own objective rather than a
        force-only surrogate of it.

        The peak residual differentiates ``max_k force[k]`` through the frame that
        attains it, i.e. the one-hot subgradient of the max. That is the exact
        gradient wherever the peak frame is unique, which it is for these
        single-peak compression cycles; ``peak_frame`` is read off the forward
        force before the tape is recorded.
        """
        wp.launch(
            _weighted_force_mse,
            dim=self.frame_count,
            inputs=[force, target, float(weight), loss],
            device=self.device,
        )
        if not shape_residuals:
            return
        wp.launch(
            _trapezoid_integral,
            dim=self.frame_count,
            inputs=[force, self.loop_weights, self.loop_value],
            device=self.device,
        )
        wp.launch(
            _scaled_square_residual,
            dim=1,
            inputs=[
                self.loop_value,
                0,
                float(self.measured_loop),
                1.0 / max(abs(self.measured_loop), 1.0e-9),
                float(HYSTERESIS_WEIGHT**2),
                loss,
            ],
            device=self.device,
        )
        wp.launch(
            _scaled_square_residual,
            dim=1,
            inputs=[
                force,
                int(peak_frame),
                float(self.measured_peak),
                1.0 / self.residual_scale,
                float(PEAK_WEIGHT**2),
                loss,
            ],
            device=self.device,
        )

    def _field_norm(self, field: wp.array2d) -> float:
        """Frobenius norm of a ``[frames, columns]`` field."""
        self.norm_scratch.zero_()
        wp.launch(_sum_squares, dim=self.shape, inputs=[field, self.norm_scratch], device=self.device)
        return float(np.sqrt(max(float(self.norm_scratch.numpy()[0]), 0.0)))

    def accumulate_gradient(
        self,
        target: wp.array,
        weight: float,
        loss: wp.array,
        warm_start: bool = False,
        shape_residuals: bool = False,
    ) -> float:
        """Add this trial's weighted force-matching gradient into ``material_params.grad``.

        Without a surround this is one ordinary tape backward over the whole
        recorded cycle.

        With a surround the forward model is an *implicit* quasi-static solve, so
        the gradient comes from the implicit function theorem instead of from
        unrolling the relaxation. Write ``q = M(c, t)`` for the exact periodic
        Maxwell overstress operator and ``G(c, t)`` for one Jacobi sweep of the
        balance evaluated at that overstress; the converged compression satisfies
        ``c* = G(c*, t)`` and the loss is ``L = F(c*, t)``. With ``A = dG/dc`` and
        ``B = dG/dt`` at ``c*``,

        ``dL/dt = dF/dt + B^T sum_k (A^T)^k (dF/dc)``

        which is evaluated without ever forming ``A``: one pass (Maxwell, one sweep
        and the force readout) is recorded on a tape at ``c*``; the first backward
        gives ``dF/dt`` in ``material_params.grad`` and ``v0 = dF/dc`` in
        ``compression.grad``; each further backward re-seeds ``sweep_out.grad = v``
        with every other gradient zeroed *except* ``material_params.grad``, which
        keeps accumulating ``B^T v``, and reads the next ``v = A^T v`` back out of
        ``compression.grad``. Driven columns are masked out of ``v`` because their
        compression is prescribed kinematics, not an unknown of the fixed point
        (they pass straight through ``G``, i.e. they are a unit eigenvalue of
        ``A``).

        The iteration is stopped on the accumulated gradient rather than on
        ``||v||``: the Jacobi sweep has modes that decay in ~1e3 sweeps out in the
        unloaded foam, but those columns are clamped, so they carry no gradient at
        all. The stopping test uses the increment of ``material_params.grad`` over
        :data:`ADJOINT_CHECK_INTERVAL` iterations plus a geometric estimate of the
        remaining tail, and :attr:`adjoint_converged` records whether
        :data:`ADJOINT_TOLERANCE` was reached before :data:`ADJOINT_MAX_ITERATIONS`.

        Args:
            target: Measured force at every frame [N].
            weight: Per-trial loss weight [1/N^2].
            loss: Single-element ``requires_grad`` loss accumulator [N^2 x weight].
            warm_start: Start the surround fixed point from the previous solve.

        Returns:
            The trial's weighted loss contribution.
        """
        loss.zero_()
        self.loop_value.zero_()
        self._zero_intermediate_grads()
        if self.surround is None:
            peak_frame = int(np.argmax(self.forward().numpy())) if shape_residuals else 0
            tape = wp.Tape()
            with tape:
                force = self.forward()
                self._record_loss(force, target, weight, loss, shape_residuals, peak_frame)
            tape.backward(loss)
            return float(loss.numpy()[0])

        self._solve_surround(warm_start=warm_start)
        peak_frame = int(np.argmax(self.force.numpy())) if shape_residuals else 0
        # Convergence is measured against *this* trial's own contribution: the shared
        # gradient may already carry another trial's (converged) contribution.
        entry = self.material_params.grad.numpy().astype(np.float64)
        self.tape_force.zero_()
        tape = wp.Tape()
        with tape:
            self._record_surround_pass(target, weight, loss, shape_residuals, peak_frame)
        tape.backward(loss)
        value = float(loss.numpy()[0])
        loss.grad.zero_()

        wp.launch(
            _mask_driven_columns,
            dim=self.shape,
            inputs=[self.driven, self.compression.grad, self.adjoint_seed],
            device=self.device,
        )
        seed_norm = self._field_norm(self.adjoint_seed)
        accumulated = self.material_params.grad.numpy().astype(np.float64)
        last_step = None
        self.adjoint_iterations = 0
        self.adjoint_converged = False
        self.adjoint_tail_relative = float("inf")
        max_iterations = int(round(ADJOINT_MAX_ITERATIONS / self.adjoint_relaxation))
        while self.adjoint_iterations < max_iterations:
            self._zero_intermediate_grads()
            tape.backward(grads={self.damped_sweep: self.adjoint_seed})
            wp.launch(
                _mask_driven_columns,
                dim=self.shape,
                inputs=[self.driven, self.compression.grad, self.adjoint_seed],
                device=self.device,
            )
            self.adjoint_iterations += 1
            if self.adjoint_iterations % ADJOINT_CHECK_INTERVAL:
                continue
            now = self.material_params.grad.numpy().astype(np.float64)
            step = float(np.linalg.norm(now - accumulated))
            scale = float(np.linalg.norm(now - entry)) + 1.0e-30
            accumulated = now
            if last_step is not None and step < last_step:
                ratio = step / last_step
                self.adjoint_tail_relative = step * ratio / (1.0 - ratio) / scale
                if self.adjoint_tail_relative <= ADJOINT_TOLERANCE:
                    self.adjoint_converged = True
                    break
            last_step = step
        self.adjoint_state_decay = self._field_norm(self.adjoint_seed) / max(seed_norm, 1.0e-30)
        return value

    def _zero_intermediate_grads(self) -> None:
        """Zero every gradient on the loss path except the (shared) material vector."""
        self.force.grad.zero_()
        self.loop_value.grad.zero_()
        if self.surround is None:
            for buf in (*self.peq, *self.q_transient, *self.q_cycle):
                buf.grad.zero_()
            self.q_zero.grad.zero_()
            self.q_init.grad.zero_()
            return
        for buf in (
            self.compression,
            self.peq_field,
            self.increment,
            self.q_field,
            self.sweep_out,
            self.damped_sweep,
            self.tape_force,
        ):
            buf.grad.zero_()

    def zero_grad(self) -> None:
        """Zero the gradients on every differentiable buffer owned by this trial."""
        if self.owns_material:
            self.material_params.grad.zero_()
        self._zero_intermediate_grads()


def fit_material_to_trials(
    trials,
    initial: Material,
    scale0=None,
    fit_mask=None,
    per_trial_weights=None,
    iterations: int = 200,
    learning_rate: float = 0.03,
    shape_residuals: bool = False,
    device=None,
) -> TrialFitResult:
    """Fit one foam material to measured Instron trials with exact gradients.

    Descends a peak-normalized joint force-matching loss with Adam, sharing one
    differentiable material vector across every trial, so every trial accumulates
    into the same three-parameter gradient. Trials that carry a
    :class:`~projects.digital_instron_v2.core.Surround` contribute through the
    implicit-function-theorem adjoint of their quasi-static bed solve
    (:meth:`DifferentiableTrial.accumulate_gradient`) and are warm-started from the
    previous iteration's compression field, so the relaxation stays cheap. By default
    each trial is weighted by ``1 / peak_force^2`` -- reproducing the per-trial
    residual normalization of the shipped scipy calibration
    (:func:`~projects.digital_instron_v2.core.fit_material`) -- so a low-force
    fixture is not swamped by a high-force one.

    Args:
        trials: Measured :class:`~projects.digital_instron_v2.core.Trial` objects.
        initial: Reference :class:`~projects.digital_instron_v2.core.Material`
            supplying the parameter scales and fixed constitutive constants.
        scale0: Initial per-parameter scale (defaults to ones).
        fit_mask: Optional length-3 mask selecting which of
            ``[g_eq, alpha, overstress]`` to optimize; zeros hold a
            parameter fixed at its ``scale0`` value.
        per_trial_weights: Optional per-trial loss weights (defaults to
            ``1 / peak_force^2`` for a peak-normalized joint fit).
        iterations: Number of Adam iterations.
        learning_rate: Adam step size in scale space.
        shape_residuals: Also score the hysteresis-loop area and the peak force,
            with the weights and scales of the shipped identification
            (:func:`~projects.digital_instron_v2.core._trial_residual`). The
            default force-only loss is a surrogate of that objective, and the two
            do not share a stationary point.
        device: Warp device.

    Returns:
        The fitted material, its scale, the loss history, and the final per-trial
        force RMS residual as a fraction of each trial's peak force.
    """
    reference = _params_from_material(initial)
    shared = wp.array(reference.copy(), dtype=wp.float32, device=device, requires_grad=True)
    predictors = [DifferentiableTrial(trial, initial, device=device, material_params=shared) for trial in trials]
    targets = [wp.array(p.measured, dtype=wp.float32, device=device) for p in predictors]
    if per_trial_weights is None:
        weights = np.array([1.0 / p.peak**2 for p in predictors], np.float64)
    else:
        weights = np.ascontiguousarray(per_trial_weights, np.float64)
    # One loss accumulator per trial: a surround trial owns an implicit solve, so its
    # gradient needs its own tape and its own adjoint fixed point.
    losses = [wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True) for _ in predictors]

    scale = np.ones(3, np.float32) if scale0 is None else np.ascontiguousarray(scale0, np.float32).copy()
    mask = np.ones(3, np.float64) if fit_mask is None else np.ascontiguousarray(fit_mask, np.float64)
    m1 = np.zeros(3, np.float64)
    m2 = np.zeros(3, np.float64)
    beta1, beta2, eps = 0.9, 0.999, 1.0e-8
    history = np.empty(iterations, np.float32)

    for it in range(iterations):
        shared.assign(scale * reference)
        shared.grad.zero_()
        total = 0.0
        for p, target, weight, loss in zip(predictors, targets, weights, losses, strict=True):
            total += p.accumulate_gradient(
                target, float(weight), loss, warm_start=it > 0, shape_residuals=shape_residuals
            )
        history[it] = total
        grad = shared.grad.numpy() * reference  # chain rule into scale space

        g = grad.astype(np.float64) * mask
        m1 = beta1 * m1 + (1.0 - beta1) * g
        m2 = beta2 * m2 + (1.0 - beta2) * g * g
        m1_hat = m1 / (1.0 - beta1 ** (it + 1))
        m2_hat = m2 / (1.0 - beta2 ** (it + 1))
        scale = (scale - learning_rate * m1_hat / (np.sqrt(m2_hat) + eps)).astype(np.float32)

    shared.assign(scale * reference)
    rms_relative = {}
    for p in predictors:
        force = p.forward(warm_start=True).numpy()
        rms_relative[p.name] = float(np.sqrt(np.mean((force - p.measured) ** 2)) / p.peak)
    fitted = scale * reference
    return TrialFitResult(
        material=_material_from_params(
            fitted, getattr(initial, "maxwell_relaxation_time_s", MAXWELL_RELAXATION_TIME_S)
        ),
        material_params=fitted,
        scale=scale,
        loss_history=history,
        rms_relative=rms_relative,
    )


def _triangular_cycle(peak_m: float, samples: int) -> np.ndarray:
    """Symmetric load/unload compression ramp from 0 to ``peak_m`` and back [m]."""
    half = samples // 2
    up = np.linspace(0.0, peak_m, half, endpoint=False)
    down = np.linspace(peak_m, 0.0, samples - half)
    return np.concatenate([up, down]).astype(np.float32)


def _demo() -> None:
    """Recover a perturbed foam material from its own synthetic Instron curve."""
    MANIFEST = "DigitalInstron/manifest_v2.json"
    wp.init()
    device = wp.get_device("cuda:0") if wp.get_cuda_device_count() else wp.get_device("cpu")
    material = dynamics.load_fitted_material(MANIFEST)
    geometry = dynamics.build_foundation_geometry(MANIFEST)

    displacement = _triangular_cycle(peak_m=0.006, samples=120)
    dt = 1.0 / 200.0  # 200 Hz Instron sampling
    names = ["g_eq", "alpha", "overstress"]

    # Synthesize the "measured" load/unload curve from the true material (scale = 1).
    truth = InstronReplay(displacement, dt, material, geometry, device=device)
    target = truth.forward().numpy().copy()
    print(f"columns={truth.column_count}  samples={truth.nsteps}  peak force={target.max():.2f} N")

    # Realistic workflow: the Maxwell overstress ratio comes from a separate
    # stress-relaxation test, so pin it and fit the elastic parameters
    # (equilibrium modulus, Hyperfoam exponent) to the curve. The shear layer is
    # not fitted at all: its coefficient is pinned to the material as mu_eq * t.
    scale0 = np.array([1.3, 0.85, 1.0], np.float32)  # 15-30% error; overstress known
    fit_mask = np.array([1.0, 1.0, 0.0], np.float32)
    print(f"\nfit (overstress pinned)  start scale = {scale0}")
    result = fit_material_to_force_curve(
        target,
        displacement,
        dt,
        material,
        geometry,
        scale0=scale0,
        fit_mask=fit_mask,
        iterations=600,
        device=device,
    )
    print(f"  final loss = {result.loss_history[-1]:.3e} N^2  (start {result.loss_history[0]:.3e})")
    print("  recovered scale (true 1.0):")
    for name, s, fit_it in zip(names, result.scale, fit_mask, strict=True):
        tag = "" if fit_it else "  (pinned)"
        print(f"    {name:10s} {s:.4f}{tag}")
    rms = float(np.sqrt(np.mean((result.force - target) ** 2)))
    print(f"  force RMS residual = {rms:.4e} N  ({rms / target.max() * 100:.4f}% of peak)")

    # Freeing all three parameters matches the force curve just as well but leaves a
    # strongly correlated stiffness/overstress direction: the curve is fit but those
    # two parameters are not separately identifiable from a single-rate cycle.
    joint = fit_material_to_force_curve(
        target,
        displacement,
        dt,
        material,
        geometry,
        scale0=np.array([1.3, 0.85, 1.2], np.float32),
        iterations=600,
        device=device,
    )
    jrms = float(np.sqrt(np.mean((joint.force - target) ** 2)))
    print(
        f"\njoint 3-parameter fit  final loss = {joint.loss_history[-1]:.3e} N^2  "
        f"(force residual {jrms / target.max() * 100:.4f}% of peak)"
    )
    print("  recovered scale (true 1.0):")
    for name, s in zip(names, joint.scale, strict=True):
        print(f"    {name:10s} {s:.4f}")
    print(
        "  -> g_eq and overstress trade off along an ill-conditioned valley; pin overstress (above) for a unique fit."
    )


def _demo_measured() -> None:
    """Fit the foam material to the two measured Instron fixtures with exact gradients.

    Loads the averaged rearfoot-punch and full-foot shoe-last trials, reproduces
    the shipped scipy calibration by a peak-normalized *joint* fit, then shows the
    lower residual each fixture reaches on its own -- the single homogeneous
    material is in genuine tension between the two loading regions.
    """
    base = Path("DigitalInstron")
    if not (base / "manifest_v2.json").exists():
        print("DigitalInstron dataset not found; skipping measured-trial demo.")
        return
    config = json.loads((base / "manifest_v2.json").read_text())
    midsole = load_mesh(str(base / config["midsole_mesh"]), 0.001)
    grid = build_column_grid(midsole, config["grid"]["coarse_spacing_m"])
    trials, _, _ = workflow.prepare_trials(base, config, grid, midsole)
    material = dynamics.load_fitted_material("DigitalInstron/manifest_v2.json")
    device = wp.get_device("cuda:0") if wp.get_cuda_device_count() else wp.get_device("cpu")

    print("\nshipped calibration (reference material):")
    for trial in trials:
        rmse = metrics(trial.force_n, predict(trial, material), trial.displacement_m)["force_rmse_relative"]
        print(f"    {trial.name:16s} force RMS = {rmse * 100:.3f}% of peak")

    joint = fit_material_to_trials(
        trials, material, iterations=40, learning_rate=0.01, shape_residuals=True, device=device
    )
    print(f"\njoint fit on the identification residual (scale, true 1.0): {np.round(joint.scale, 4)}")
    for name, rms in joint.rms_relative.items():
        print(f"    {name:16s} force RMS = {rms * 100:.3f}% of peak")
    print("  -> reproduces the shipped scipy calibration; the shared material is the joint optimum.")
    force_only = fit_material_to_trials(trials, material, iterations=40, learning_rate=0.01, device=device)
    print(f"  force residual only:  scale = {np.round(force_only.scale, 4)}")
    print(
        "  -> scoring the force alone slides down the stiffness/overstress valley: the loop-area and peak\n"
        "     residuals of core._trial_residual are what make the shipped material stationary."
    )

    for trial in trials:
        per_trial = fit_material_to_trials(
            [trial], material, iterations=40, learning_rate=0.01, shape_residuals=True, device=device
        )
        rms = per_trial.rms_relative[trial.name]
        print(
            f"  per-fixture fit {trial.name:16s} force RMS = {rms * 100:.3f}% of peak  scale = {np.round(per_trial.scale, 3)}"
        )
    print(
        "  -> each fixture reaches a lower residual alone but demands a different stiffness (single-material tension)."
    )


if __name__ == "__main__":
    _demo()
    _demo_measured()
