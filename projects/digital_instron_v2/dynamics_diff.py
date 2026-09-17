# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tape-safe adapter for the shared Hyperfoam-Maxwell-Pasternak contact model.

The constitutive expressions live in :mod:`projects.digital_shoe.material`.
Normal contact, lateral coupling, anchored bristle friction, and force-to-wrench
mechanics live in :mod:`projects.digital_shoe.contact`. The live runtime and this
adapter call those same functions. Material parameters enter as array elements
so Warp can accumulate their gradients.

This adapter owns separate pressure, Maxwell, bristle, and force buffers for each
substep. No state on the loss path is overwritten during a rollout. Integer
contact flags select the same branches as the live model; derivatives are
piecewise derivatives away from contact, stick/slip, and release transitions.
The default Maxwell shear state uses separate elastic and branch-force buffers.
``friction_model="legacy"`` retains the previous anchored-bristle kernel and public signature.

The live surround uses fixed, warm-started damped sweeps. This module differentiates
through those sweeps, giving the derivative of the map the simulation evaluates.
Identification instead solves to convergence and uses an implicit adjoint; see
:class:`projects.digital_instron_v2.inverse_id.DifferentiableTrial`.

The length-five material vector is ``[g_eq, alpha, overstress, g_eq2, alpha2]``.
Pasternak coupling follows the sum of both equilibrium moduli and adds no fitted
parameter. The length-one friction vector contains the Coulomb coefficient.
Both vectors retain their existing public layout.
"""

from __future__ import annotations

import warnings

import numpy as np
import warp as wp

from projects.digital_shoe.contact import (
    bristle_step,
    contact_kinematics,
    contact_wrench,
    normal_reaction,
    pasternak_coupling,
    pasternak_flux,
    surround_balance,
)
from projects.digital_shoe.friction_maxwell import bristle_maxwell_step
from projects.digital_shoe.material import (
    hyperfoam_pressure,
    maxwell_coefficients,
    maxwell_coefficients_numpy,
    maxwell_step,
)
from projects.digital_shoe.runtime import set_material_block

from .core import Material
from .dynamics import FoundationConfig, FoundationParams, SurroundConfig

# Indices into the differentiable ``material_params`` vector.
MAT_G_EQ = wp.constant(0)  # first-term equilibrium shear modulus G_inst * equilibrium_fraction [Pa]
MAT_ALPHA = wp.constant(1)  # first-term Hyperfoam exponent
MAT_OVERSTRESS = wp.constant(2)  # (1 - equilibrium_fraction) / equilibrium_fraction
MAT_G_EQ2 = wp.constant(3)  # second-term equilibrium shear modulus [Pa]; zero disables the term
MAT_ALPHA2 = wp.constant(4)  # second-term Hyperfoam exponent
MAT_COUNT = 5  # length of the differentiable material vector

# Index into the differentiable ``friction_params`` vector.
FRIC_MU = wp.constant(0)  # Coulomb friction coefficient (bristle cone bound)


@wp.func
def _hyperfoam_pressure_diff(
    strain: wp.float32,
    g_eq: wp.float32,
    alpha: wp.float32,
    g_eq2: wp.float32,
    alpha2: wp.float32,
    p: FoundationParams,
) -> wp.float32:
    """Adapt the shared Hyperfoam law to the legacy differentiable signature."""
    return hyperfoam_pressure(strain, g_eq, alpha, g_eq2, alpha2, p.beta, p.one_minus_two_poisson, p.stretch_floor)


@wp.kernel
def foundation_pressure_diff(
    carrier: wp.int32,
    dt: wp.float32,
    body_q: wp.array[wp.transform],
    anchor_local: wp.array[wp.vec3],
    z_free: wp.array[wp.float32],
    rest_len: wp.array[wp.float32],
    params: FoundationParams,
    material_params: wp.array[wp.float32],
    q_prev: wp.array[wp.float32],
    peq_prev: wp.array[wp.float32],
    q_out: wp.array[wp.float32],
    peq_out: wp.array[wp.float32],
    compression: wp.array[wp.float32],
    base_pressure: wp.array[wp.float32],
):
    """Compression, Hyperfoam equilibrium pressure, and a tape-safe Maxwell recurrence.

    Reads the previous substep's overstress state (``q_prev``) and equilibrium
    pressure (``peq_prev``) and writes this substep's state into the separate
    ``q_out``/``peq_out`` arrays (no in-place aliasing), so the recurrence can be
    replayed on a backward pass. ``g_eq``, ``alpha`` and the overstress ratio are
    read from the differentiable ``material_params`` vector.
    """
    i = wp.tid()
    g_eq = material_params[MAT_G_EQ]
    alpha = material_params[MAT_ALPHA]
    overstress = material_params[MAT_OVERSTRESS]
    g_eq2 = material_params[MAT_G_EQ2]
    alpha2 = material_params[MAT_ALPHA2]

    world = wp.transform_point(body_q[carrier], anchor_local[i])
    comp = z_free[i] - world[2]
    if comp < 0.0:
        comp = 0.0
    compression[i] = comp
    strain = comp / rest_len[i]
    peq = _hyperfoam_pressure_diff(strain, g_eq, alpha, g_eq2, alpha2, params)
    # The identified relaxation time travels with the material (the refit moved it far
    # from the historical 80 ms default), so read it from ``params`` as the runtime does.
    decay, ramp = maxwell_coefficients(dt, params.tau_s)
    qn = maxwell_step(q_prev[i], peq, peq_prev[i], overstress, decay, ramp)
    q_out[i] = qn
    peq_out[i] = peq
    base_pressure[i] = peq + qn


@wp.func
def _pasternak_coupling_diff(t_i: wp.float32, t_j: wp.float32, mu_eq: wp.float32) -> wp.float32:
    """Adapt the shared symmetric face coefficient to the legacy signature [N/m]."""
    return pasternak_coupling(t_i, t_j, mu_eq)


@wp.func
def _pasternak_flux(
    i: wp.int32,
    compression: wp.array[wp.float32],
    rest_len: wp.array[wp.float32],
    neighbors: wp.array2d[wp.int32],
    mu_eq: wp.float32,
) -> wp.float32:
    """Lateral shear force the neighbours pull out of one column [N].

    Mirrors the shear-layer stencil of
    :func:`projects.digital_shoe.runtime.foundation_apply`: the pairwise sum
    ``sum_j k_ij * (c_j - c_i)`` over the face coefficients of
    :func:`_pasternak_coupling_diff`. Written pairwise (rather than as a
    coefficient times a Laplacian) the bed total cancels to the last bit, so the
    layer can only move load, never create it.

    Every missing neighbour (outside the midsole *or* simply not in the active
    set) is a free, zero-gradient edge and contributes nothing. Holding one at
    zero compression instead would make it a rigid hidden support that the shear
    layer leans on, which creates load instead of only spreading it.
    """
    return pasternak_flux(i, 0, compression, rest_len, neighbors, mu_eq)


@wp.func
def _column_normal_force(
    ci: wp.float32,
    base_pressure_i: wp.float32,
    area_i: wp.float32,
    flux: wp.float32,
    normal_damping: wp.float32,
    vz: wp.float32,
) -> wp.float32:
    """Return legacy bench transfer force: shared unilateral reaction minus flux [N].

    The signed Pasternak flux redistributes load and must not be clamped. This
    quantity is not the local external ground reaction when a plane is declared.
    """
    return normal_reaction(ci, base_pressure_i, area_i, normal_damping, vz, 0.0, 0) - flux


@wp.func
def _surround_balance_diff(
    c: wp.float32,
    rigid: wp.float32,
    pull: wp.float32,
    coupling_sum: wp.float32,
    thickness: wp.float32,
    overstress_base: wp.float32,
    overstress_gain: wp.float32,
    g_eq: wp.float32,
    alpha: wp.float32,
    g_eq2: wp.float32,
    alpha2: wp.float32,
    params: FoundationParams,
    area: wp.float32,
    attachment: wp.float32,
    max_strain: wp.float32,
    relaxation: wp.float32,
    carrier_bond: wp.int32,
) -> wp.float32:
    """Adapt the shared surround balance to differentiable array-read material terms."""
    return surround_balance(
        c,
        rigid,
        pull,
        coupling_sum,
        thickness,
        overstress_base,
        overstress_gain,
        g_eq,
        alpha,
        g_eq2,
        alpha2,
        params.beta,
        params.one_minus_two_poisson,
        params.stretch_floor,
        area,
        attachment,
        max_strain,
        relaxation,
        carrier_bond,
    )


# Deprecated compatibility only. Active adapters call the shared bristle law.
@wp.func
def _legacy_smooth_friction(velocity: wp.vec2, pressed: wp.float32, mu: wp.float32, smoothing: wp.float32) -> wp.vec2:
    """Preserve the old stateless raw-kernel law until its deprecation ends.

    No active foundation or scenario calls this helper. The public kernel cannot
    carry a timestep or bristle history without changing its launch signature.
    """
    force = wp.vec2(0.0, 0.0)
    if pressed > 0.0 and mu > 0.0:
        force = -mu * pressed * wp.smooth_normalize(velocity, smoothing)
    return force


@wp.kernel
def foundation_apply_diff(
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
    params: FoundationParams,
    material_params: wp.array[wp.float32],
    friction_params: wp.array[wp.float32],
    friction_smoothing: wp.float32,
    body_f: wp.array[wp.spatial_vector],
    normal_force: wp.array[wp.float32],
    cop_moment: wp.array[wp.vec3],
    pressed_force: wp.array[wp.float32],
    active_count: wp.array[wp.int32],
):
    """Deprecated stateless kernel; use foundation_apply_bristle_diff.

    The old smooth-friction behavior is retained only for compatibility until
    this raw kernel's deprecation ends. Its signature cannot carry a timestep or
    bristle history. :class:`DifferentiableMidsoleFoundation` never calls this
    path. Normal contact, material, kinematics and wrench use the shared helpers.
    """
    i = wp.tid()
    # The shear layer follows the series modulus, which is the sum of both terms.
    mu_eq = material_params[MAT_G_EQ] + material_params[MAT_G_EQ2]

    ci = compression[i]
    flux = _pasternak_flux(i, compression, rest_len, neighbors, mu_eq)

    vel = body_qd[carrier]
    world, com_world, point_vel, _gap = contact_kinematics(
        body_q[carrier], vel, body_com[carrier], anchor_local[i], 0.0, 0
    )

    fn = _column_normal_force(ci, base_pressure[i], area[i], flux, params.normal_damping, point_vel[2])
    # A column the shear layer lifts transmits a small pull, so the friction cone and
    # the centre of pressure use the pressed part only, as the runtime does.
    pressed = wp.max(fn, 0.0)

    f_tan = _legacy_smooth_friction(
        wp.vec2(point_vel[0], point_vel[1]), pressed, friction_params[FRIC_MU], friction_smoothing
    )

    force = wp.vec3(f_tan[0], f_tan[1], fn)
    torque, _moment, _power = contact_wrench(world, force, com_world, vel)
    wp.atomic_add(body_f, carrier, wp.spatial_vector(force, torque))
    wp.atomic_add(normal_force, 0, fn)
    wp.atomic_add(cop_moment, 0, wp.vec3(world[0] * pressed, world[1] * pressed, 0.0))
    wp.atomic_add(pressed_force, 0, pressed)
    if ci > 0.0:
        wp.atomic_add(active_count, 0, 1)


@wp.kernel
def foundation_apply_bristle_diff(
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
    params: FoundationParams,
    material_params: wp.array[wp.float32],
    friction_params: wp.array[wp.float32],
    dt: wp.float32,
    ground_height: wp.float32,
    ground_plane: wp.int32,
    friction_kt: wp.array[wp.float32],
    friction_kv: wp.array[wp.float32],
    anchor_prev: wp.array[wp.vec2],
    stuck_prev: wp.array[wp.int32],
    dwell_prev: wp.array[wp.float32],
    anchor_out: wp.array[wp.vec2],
    stuck_out: wp.array[wp.int32],
    dwell_out: wp.array[wp.float32],
    force_out: wp.array[wp.vec3],
    ground_force_out: wp.array[wp.vec3],
    body_f: wp.array[wp.spatial_vector],
    normal_force: wp.array[wp.float32],
    cop_moment: wp.array[wp.vec3],
    pressed_force: wp.array[wp.float32],
    active_count: wp.array[wp.int32],
):
    """Apply the shared bristle contact law with separate per-substep history buffers.

    Integer contact flags select the same stick/slip branches as the live model.
    Tape gradients are piecewise derivatives away from those branch transitions.
    ``force_out`` records signed transfer traction; ``ground_force_out`` records
    external support. Their meanings match the live foundation in both modes.
    """
    i = wp.tid()
    mu_eq = material_params[MAT_G_EQ] + material_params[MAT_G_EQ2]
    ci = compression[i]
    flux = _pasternak_flux(i, compression, rest_len, neighbors, mu_eq)
    vel = body_qd[carrier]
    point, com_world, point_vel, gap = contact_kinematics(
        body_q[carrier], vel, body_com[carrier], anchor_local[i], ground_height, ground_plane
    )
    reaction = normal_reaction(ci, base_pressure[i], area[i], params.normal_damping, point_vel[2], gap, ground_plane)
    fn = reaction - flux
    if ground_plane != 0:
        fn = reaction
    pressed = wp.max(fn, 0.0)
    f_tan, anchor, stuck, dwell = bristle_step(
        wp.vec2(point[0], point[1]),
        wp.vec2(point_vel[0], point_vel[1]),
        dt,
        pressed,
        friction_kt[i],
        friction_kv[i],
        friction_params[FRIC_MU],
        params.friction_viscous_ratio,
        params.friction_release_dwell_s,
        anchor_prev[i],
        stuck_prev[i],
        dwell_prev[i],
    )
    anchor_out[i] = anchor
    stuck_out[i] = stuck
    dwell_out[i] = dwell
    force_out[i] = wp.vec3(f_tan[0], f_tan[1], reaction - flux)
    ground_force_out[i] = wp.vec3(f_tan[0], f_tan[1], reaction)
    force = wp.vec3(f_tan[0], f_tan[1], fn)
    torque, _moment, _power = contact_wrench(point, force, com_world, vel)
    wp.atomic_add(body_f, carrier, wp.spatial_vector(force, torque))
    wp.atomic_add(normal_force, 0, fn)
    wp.atomic_add(cop_moment, 0, wp.vec3(point[0] * pressed, point[1] * pressed, 0.0))
    wp.atomic_add(pressed_force, 0, pressed)
    if ci > 0.0:
        wp.atomic_add(active_count, 0, 1)


@wp.kernel
def foundation_apply_maxwell_diff(
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
    params: FoundationParams,
    material_params: wp.array[wp.float32],
    friction_params: wp.array[wp.float32],
    dt: wp.float32,
    ground_height: wp.float32,
    ground_plane: wp.int32,
    friction_kt: wp.array[wp.float32],
    friction_kv: wp.array[wp.float32],
    anchor_prev: wp.array[wp.vec2],
    stuck_prev: wp.array[wp.int32],
    dwell_prev: wp.array[wp.float32],
    deflection_prev: wp.array[wp.vec2],
    maxwell_prev: wp.array[wp.vec2],
    deflection_out: wp.array[wp.vec2],
    maxwell_out: wp.array[wp.vec2],
    anchor_out: wp.array[wp.vec2],
    stuck_out: wp.array[wp.int32],
    dwell_out: wp.array[wp.float32],
    force_out: wp.array[wp.vec3],
    ground_force_out: wp.array[wp.vec3],
    body_f: wp.array[wp.spatial_vector],
    normal_force: wp.array[wp.float32],
    cop_moment: wp.array[wp.vec3],
    pressed_force: wp.array[wp.float32],
    active_count: wp.array[wp.int32],
):
    """Apply canonical Maxwell shear contact with separate per-substep state.

    Integer contact flags select the same stick/slip branches as the live model.
    Tape gradients are piecewise derivatives away from those branch transitions.
    ``force_out`` records signed transfer traction; ``ground_force_out`` records
    external support. Their meanings match the live foundation in both modes.
    """
    i = wp.tid()
    mu_eq = material_params[MAT_G_EQ] + material_params[MAT_G_EQ2]
    ci = compression[i]
    flux = _pasternak_flux(i, compression, rest_len, neighbors, mu_eq)
    vel = body_qd[carrier]
    point, com_world, point_vel, gap = contact_kinematics(
        body_q[carrier], vel, body_com[carrier], anchor_local[i], ground_height, ground_plane
    )
    reaction = normal_reaction(ci, base_pressure[i], area[i], params.normal_damping, point_vel[2], gap, ground_plane)
    fn = reaction - flux
    if ground_plane != 0:
        fn = reaction
    pressed = wp.max(fn, 0.0)
    f_tan, _jac, z, q, stuck, dwell = bristle_maxwell_step(
        wp.vec2(point_vel[0], point_vel[1]),
        dt,
        pressed,
        friction_kt[i],
        friction_kv[i],
        params.friction_relaxation_time_s,
        friction_params[FRIC_MU],
        params.friction_release_dwell_s,
        deflection_prev[i],
        maxwell_prev[i],
        stuck_prev[i],
        dwell_prev[i],
    )
    deflection_out[i] = z
    maxwell_out[i] = q
    anchor = wp.vec2(point[0], point[1]) + dt * wp.vec2(point_vel[0], point_vel[1]) - z
    anchor_out[i] = anchor
    stuck_out[i] = stuck
    dwell_out[i] = dwell
    force_out[i] = wp.vec3(f_tan[0], f_tan[1], reaction - flux)
    ground_force_out[i] = wp.vec3(f_tan[0], f_tan[1], reaction)
    force = wp.vec3(f_tan[0], f_tan[1], fn)
    torque, _moment, _power = contact_wrench(point, force, com_world, vel)
    wp.atomic_add(body_f, carrier, wp.spatial_vector(force, torque))
    wp.atomic_add(normal_force, 0, fn)
    wp.atomic_add(cop_moment, 0, wp.vec3(point[0] * pressed, point[1] * pressed, 0.0))
    wp.atomic_add(pressed_force, 0, pressed)
    if ci > 0.0:
        wp.atomic_add(active_count, 0, 1)


@wp.kernel
def surround_relax_diff(
    carrier: wp.int32,
    body_q: wp.array[wp.transform],
    driven: wp.array[wp.int32],
    neighbors: wp.array2d[wp.int32],
    anchor_local: wp.array[wp.vec3],
    z_free_rigid: wp.array[wp.float32],
    rest_len: wp.array[wp.float32],
    area: wp.array[wp.float32],
    q_state: wp.array[wp.float32],
    peq_prev: wp.array[wp.float32],
    params: FoundationParams,
    material_params: wp.array[wp.float32],
    decay: wp.float32,
    ramp: wp.float32,
    coupling_scale: wp.float32,
    attachment: wp.float32,
    max_strain: wp.float32,
    relaxation: wp.float32,
    carrier_bond: wp.int32,
    compression_in: wp.array[wp.float32],
    compression_out: wp.array[wp.float32],
):
    """Sweep the column bed once toward the balance the identification relaxes.

    Tape-safe transcription of
    :func:`projects.digital_shoe.runtime.surround_relax`: it reads
    ``compression_in`` and writes a *separate* ``compression_out``, so a whole
    substep's sweeps stay un-aliased on a :class:`warp.Tape`, and it takes the
    material from the differentiable ``material_params`` vector instead of the
    by-value :class:`FoundationParams` struct.

    ``decay`` and ``ramp`` are the substep's Maxwell update ``exp(-dt / tau)`` and
    ``tau (1 - decay) / dt``; the overstress gain is ``overstress * ramp``, so the
    surround settles against the overstress
    :func:`foundation_pressure_diff` is about to write for the compression this
    sweep produces.
    """
    i = wp.tid()
    g_eq = material_params[MAT_G_EQ]
    alpha = material_params[MAT_ALPHA]
    overstress = material_params[MAT_OVERSTRESS]
    g_eq2 = material_params[MAT_G_EQ2]
    alpha2 = material_params[MAT_ALPHA2]
    mu_eq = g_eq + g_eq2

    world = wp.transform_point(body_q[carrier], anchor_local[i])
    rigid = z_free_rigid[i] - world[2]
    if driven[i] != 0:
        compression_out[i] = wp.max(rigid, 0.0)
        return
    c = compression_in[i]
    pull = float(0.0)
    coupling_sum = float(0.0)
    for side in range(4):
        j = neighbors[i, side]
        if j >= 0:
            coupling = coupling_scale * _pasternak_coupling_diff(rest_len[i], rest_len[j], mu_eq)
            pull += coupling * (compression_in[j] - c)
            coupling_sum += coupling
    gain = overstress * ramp
    compression_out[i] = _surround_balance_diff(
        c,
        rigid,
        pull,
        coupling_sum,
        rest_len[i],
        decay * q_state[i] - gain * peq_prev[i],
        gain,
        g_eq,
        alpha,
        g_eq2,
        alpha2,
        params,
        area[i],
        attachment,
        max_strain,
        relaxation,
        carrier_bond,
    )


@wp.kernel
def surround_write_free_top_diff(
    carrier: wp.int32,
    body_q: wp.array[wp.transform],
    driven: wp.array[wp.int32],
    anchor_local: wp.array[wp.vec3],
    z_free_rigid: wp.array[wp.float32],
    compression: wp.array[wp.float32],
    z_free_out: wp.array[wp.float32],
):
    """Publish the relaxed free surface this substep's pressure kernel then consumes.

    Same device as :func:`projects.digital_shoe.runtime.surround_write_free_top`:
    :func:`foundation_pressure_diff` reads ``compression = z_free - world_z``, so
    writing ``z_free = world_z + c`` hands it the relaxed compression without a
    second contact path. The output is a per-substep array so nothing on the loss
    path is overwritten during a rollout, and the rate diagnostic is dropped
    because it would alias across substeps.
    """
    i = wp.tid()
    if driven[i] != 0:
        z_free_out[i] = z_free_rigid[i]
        return
    world = wp.transform_point(body_q[carrier], anchor_local[i])
    z_free_out[i] = world[2] + compression[i]


class DifferentiableMidsoleFoundation:
    """Autodiff-ready elastic-foundation force model for a fixed-length rollout.

    Unlike :class:`~projects.digital_instron_v2.dynamics.MidsoleFoundation`, which
    ping-pongs a single set of buffers for speed, this variant keeps a separate
    compression/pressure/overstress history for each of ``num_substeps`` substeps
    so the whole rollout can be recorded on one :class:`warp.Tape` and
    differentiated. ``column_force[t]`` holds signed transfer traction [N].
    ``ground_force[t]`` holds unilateral external support plus friction [N].
    ``applied_force`` selects the former for bench-top coordinates and the latter
    for a declared ground plane. These quantities must not be interchanged.

    Drive it like the forward model, but pass the current substep index so the
    correct history slot and previous state are used::

        foundation = DifferentiableMidsoleFoundation(..., num_substeps=N)
        tape = wp.Tape()
        with tape:
            for t in range(N):
                states[t].body_f.zero_()
                foundation.apply(states[t], t, dt)
                solver.step(states[t], states[t + 1], None, None, dt)
            wp.launch(loss_kernel, ...)
        tape.backward(loss)
        # gradients w.r.t. the foam material and the friction coefficient:
        foundation.material_params.grad.numpy()
        foundation.friction_params.grad.numpy()

    Args:
        anchor_local: Column attachment points in the carrier body frame [m],
            shape ``[column_count, 3]``.
        z_free: World height of each uncompressed foam column top [m], shape
            ``[column_count]``.
        rest_len: Column rest thickness [m], shape ``[column_count]``.
        area: Tributary area per column [m^2], shape ``[column_count]``.
        neighbors: Pasternak 4-neighbour indices, shape ``[column_count, 4]``.
        spacing_m: Column grid spacing [m].
        material: Calibrated :class:`~projects.digital_instron_v2.core.Material`.
        carrier_body: Index of the rigid body carrying the foundation.
        body_com: Model center-of-mass array (``model.body_com``).
        num_substeps: Number of substeps in one differentiated rollout.
        config: Dynamic :class:`~projects.digital_instron_v2.dynamics.FoundationConfig`.
        friction_smoothing: Deprecated and ignored. The shared anchored bristle
            law replaces smooth Coulomb friction.
        device: Warp device (must match the carrier state's device).
        surround: Optional
            :class:`~projects.digital_instron_v2.dynamics.SurroundConfig` letting
            the columns the carrier does not drive relax passively every substep,
            exactly as the shipped runtime and the identification relax them. The
            rollout differentiates *through* the sweeps; see :meth:`relax_surround`.
    """

    def __init__(
        self,
        anchor_local: np.ndarray,
        z_free: np.ndarray,
        rest_len: np.ndarray,
        area: np.ndarray,
        neighbors: np.ndarray,
        spacing_m: float,
        material: Material,
        carrier_body: int,
        body_com,
        num_substeps: int,
        config: FoundationConfig | None = None,
        friction_smoothing: float = 0.05,
        device=None,
        surround: SurroundConfig | None = None,
    ) -> None:
        config = config or FoundationConfig()
        self.ground_height_m = config.ground_height_m
        if self.ground_height_m is not None:
            if not np.all(np.asarray(z_free, dtype=np.float32) == np.float32(self.ground_height_m)):
                raise ValueError("initial z_free must equal ground_height_m in ground-plane mode")
            if surround is not None and not surround.carrier_bond and np.any(surround.driven == 0):
                raise ValueError("ground-plane passive columns require surround.carrier_bond=True")
        if friction_smoothing != 0.05:
            warnings.warn(
                "friction_smoothing is deprecated and ignored; the differentiable foundation uses the shared bristle law",
                DeprecationWarning,
                stacklevel=2,
            )
        self.device = device
        self.carrier = int(carrier_body)
        self.body_com = body_com
        self.column_count = int(len(rest_len))
        self.num_substeps = int(num_substeps)
        self.friction_smoothing = float(friction_smoothing)

        params = FoundationParams()
        set_material_block(params, material)
        # Grid geometry only: the shear-layer coefficient is pinned per column to
        # ``mu_eq * t_i`` and no longer scales with the spacing.
        params.inv_h2 = 1.0 / spacing_m**2
        params.stretch_floor = config.stretch_floor
        params.normal_damping = config.normal_damping
        params.friction_kt = config.friction_stiffness
        params.friction_kv = config.friction
        params.friction_viscous_ratio = config.friction_viscous_ratio
        params.friction_release_dwell_s = config.friction_release_dwell_s
        params.mu = config.mu
        params.friction_model = 1 if config.friction_model == "maxwell" else 0
        params.friction_relaxation_time_s = float(
            config.friction_relaxation_time_s
            if config.friction_relaxation_time_s is not None
            else material.maxwell_relaxation_time_s
        )
        self.params = params

        # Differentiable constitutive vector [g_eq, alpha, overstress, g_eq2, alpha2].
        self.material_params = wp.array(
            np.array([params.g_eq, params.alpha, params.overstress, params.g_eq2, params.alpha2], np.float32),
            dtype=wp.float32,
            device=device,
            requires_grad=True,
        )
        # Differentiable friction vector [mu] for gradient-based friction identification.
        self.friction_params = wp.array(
            np.array([params.mu], np.float32),
            dtype=wp.float32,
            device=device,
            requires_grad=True,
        )

        m = self.column_count
        area_m2 = np.ascontiguousarray(area, np.float64).reshape(-1)
        mean_area = float(area_m2.mean())
        if mean_area <= 0.0:
            raise ValueError("column tributary areas must be positive")
        self.friction_stiffness_per_area_n_m3 = float(
            config.friction_stiffness_per_area or config.friction_stiffness / mean_area
        )
        self.friction_damping_per_area_n_s_m3 = float(config.friction_damping_per_area or config.friction / mean_area)
        self.friction_kt = wp.array(
            np.ascontiguousarray(self.friction_stiffness_per_area_n_m3 * area_m2, np.float32),
            dtype=wp.float32,
            device=device,
        )
        self.friction_kv = wp.array(
            np.ascontiguousarray(self.friction_damping_per_area_n_s_m3 * area_m2, np.float32),
            dtype=wp.float32,
            device=device,
        )
        self.anchor_local = wp.array(np.ascontiguousarray(anchor_local, np.float32), dtype=wp.vec3, device=device)
        self.z_free = wp.array(np.ascontiguousarray(z_free, np.float32), dtype=wp.float32, device=device)
        self.rest_len = wp.array(np.ascontiguousarray(rest_len, np.float32), dtype=wp.float32, device=device)
        self.area = wp.array(np.ascontiguousarray(area, np.float32), dtype=wp.float32, device=device)
        self.neighbors = wp.array(np.ascontiguousarray(neighbors, np.int32), dtype=wp.int32, device=device)

        def grad_zeros():
            return wp.zeros(m, dtype=wp.float32, device=device, requires_grad=True)

        # Per-substep history so nothing on the loss path is overwritten within a rollout.
        self.compression = [grad_zeros() for _ in range(self.num_substeps)]
        self.base_pressure = [grad_zeros() for _ in range(self.num_substeps)]
        self.q_state = [grad_zeros() for _ in range(self.num_substeps)]
        self.peq_prev = [grad_zeros() for _ in range(self.num_substeps)]
        # Fixed zero initial overstress state (substep 0 reads these).
        self.q_init = grad_zeros()
        self.peq_init = grad_zeros()
        self.tangent_anchor_init = wp.zeros(m, dtype=wp.vec2, device=device, requires_grad=True)
        self.tangent_stuck_init = wp.zeros(m, dtype=wp.int32, device=device)
        self.tangent_dwell_init = grad_zeros()
        self.tangent_deflection_init = wp.zeros(m, dtype=wp.vec2, device=device, requires_grad=True)
        self.tangent_maxwell_force_init = wp.zeros(m, dtype=wp.vec2, device=device, requires_grad=True)
        self.tangent_deflection = [
            wp.zeros(m, dtype=wp.vec2, device=device, requires_grad=True) for _ in range(self.num_substeps)
        ]
        self.tangent_maxwell_force = [
            wp.zeros(m, dtype=wp.vec2, device=device, requires_grad=True) for _ in range(self.num_substeps)
        ]
        self.tangent_anchor = [
            wp.zeros(m, dtype=wp.vec2, device=device, requires_grad=True) for _ in range(self.num_substeps)
        ]
        self.tangent_stuck = [wp.zeros(m, dtype=wp.int32, device=device) for _ in range(self.num_substeps)]
        self.tangent_dwell = [grad_zeros() for _ in range(self.num_substeps)]
        self.column_force = [
            wp.zeros(m, dtype=wp.vec3, device=device, requires_grad=True) for _ in range(self.num_substeps)
        ]
        self.ground_force = [
            wp.zeros(m, dtype=wp.vec3, device=device, requires_grad=True) for _ in range(self.num_substeps)
        ]
        self.applied_force = self.column_force if self.ground_height_m is None else self.ground_force

        # Passive surround: one relaxed compression field per sweep and per substep, plus
        # the free-surface height each substep's pressure kernel reads, so a whole rollout
        # of sweeps is recorded once and never aliased.
        self.surround = surround
        self.free_column_count = 0
        if surround is not None:
            if len(surround.driven) != m:
                raise ValueError("the surround mask must cover every column")
            self.free_column_count = int(m - int(surround.driven.sum()))
            self.surround_sweeps = int(surround.sweeps)
            self.driven = wp.array(surround.driven, dtype=wp.int32, device=device)
            self.z_free_rigid = wp.array(np.ascontiguousarray(z_free, np.float32), dtype=wp.float32, device=device)
            self.surround_init = grad_zeros()
            self.surround_compression = [
                [grad_zeros() for _ in range(self.surround_sweeps)] for _ in range(self.num_substeps)
            ]
            self.z_free_substep = [grad_zeros() for _ in range(self.num_substeps)]

        # Diagnostics (off the loss path); overwritten each substep.
        self.normal_force = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
        self.cop_moment = wp.zeros(1, dtype=wp.vec3, device=device, requires_grad=True)
        self.pressed_force = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
        self.active = wp.zeros(1, dtype=wp.int32, device=device)

    def relax_surround(self, state, substep: int, dt: float, q_prev, peq_prev):
        """Relax the columns the carrier does not drive for one substep.

        Sweeps :func:`surround_relax_diff` over the compression field warm started
        from the previous substep and publishes the relaxed free surface with
        :func:`surround_write_free_top_diff`, so the pressure and wrench kernels
        see one bed with one contact law -- the same arrangement as
        :meth:`projects.digital_shoe.runtime.MidsoleFoundation.relax_surround`.

        Gradients: the live surround is a *fixed, warm-started* number of damped
        sweeps per substep, not a solve run to convergence, so this rollout
        differentiates straight through the sweeps. That derivative is exact for
        the map the forward pass actually evaluates -- no implicit-function
        theorem is needed or wanted here, because a converged-solve adjoint would
        answer a different question than the one the simulation asks. The
        identification, which does relax to convergence, uses the adjoint instead
        (:class:`projects.digital_instron_v2.inverse_id.DifferentiableTrial`).
        Cost is ``num_substeps * sweeps`` recorded launches and the same number of
        column-sized buffers.

        Args:
            state: Simulation state supplying the carrier pose.
            substep: Substep index in ``[0, num_substeps)``.
            dt: Substep duration [s].
            q_prev: Previous substep's Maxwell overstress state [Pa].
            peq_prev: Previous substep's equilibrium pressure [Pa].

        Returns:
            The per-substep free-surface height array [m] that
            :func:`foundation_pressure_diff` must read for this substep.
        """
        cfg = self.surround
        sweeps = self.surround_sweeps
        sub_dt = dt / sweeps
        tau = float(cfg.relaxation_time_s)
        relaxation = 1.0 if tau <= 0.0 else 1.0 - float(np.exp(-sub_dt / tau))
        decay, ramp = maxwell_coefficients_numpy(dt, self.params.tau_s)
        decay, ramp = float(decay), float(ramp)
        inputs = [
            self.carrier,
            state.body_q,
            self.driven,
            self.neighbors,
            self.anchor_local,
            self.z_free_rigid,
            self.rest_len,
            self.area,
            q_prev,
            peq_prev,
            self.params,
            self.material_params,
            decay,
            ramp,
            float(cfg.coupling_scale),
            float(cfg.attachment_n_m),
            float(cfg.max_strain),
            relaxation,
            int(bool(cfg.carrier_bond)),
        ]
        current = self.surround_init if substep == 0 else self.surround_compression[substep - 1][-1]
        for sweep in range(sweeps):
            nxt = self.surround_compression[substep][sweep]
            wp.launch(
                surround_relax_diff,
                dim=self.column_count,
                inputs=[*inputs, current, nxt],
                device=self.device,
            )
            current = nxt
        wp.launch(
            surround_write_free_top_diff,
            dim=self.column_count,
            inputs=[
                self.carrier,
                state.body_q,
                self.driven,
                self.anchor_local,
                self.z_free_rigid,
                current,
                self.z_free_substep[substep],
            ],
            device=self.device,
        )
        return self.z_free_substep[substep]

    def apply(self, state, substep: int, dt: float) -> None:
        """Accumulate the foundation wrench into ``state.body_f`` for one substep.

        Args:
            state: Simulation state supplying the carrier pose/velocity and
                receiving the wrench; ``state.body_f`` must be cleared beforehand.
            substep: Substep index in ``[0, num_substeps)``; selects the history
                slot and the previous overstress state for the recurrence.
            dt: Substep duration [s].
        """
        t = int(substep)
        q_prev = self.q_init if t == 0 else self.q_state[t - 1]
        peq_prev = self.peq_init if t == 0 else self.peq_prev[t - 1]
        z_free = self.z_free
        if self.free_column_count:
            z_free = self.relax_surround(state, t, dt, q_prev, peq_prev)

        self.normal_force.zero_()
        self.cop_moment.zero_()
        self.pressed_force.zero_()
        self.active.zero_()

        wp.launch(
            foundation_pressure_diff,
            dim=self.column_count,
            inputs=[
                self.carrier,
                dt,
                state.body_q,
                self.anchor_local,
                z_free,
                self.rest_len,
                self.params,
                self.material_params,
                q_prev,
                peq_prev,
                self.q_state[t],
                self.peq_prev[t],
                self.compression[t],
                self.base_pressure[t],
            ],
            device=self.device,
        )
        wp.launch(
            foundation_apply_maxwell_diff if self.params.friction_model == 1 else foundation_apply_bristle_diff,
            dim=self.column_count,
            inputs=[
                self.carrier,
                state.body_q,
                state.body_qd,
                self.body_com,
                self.anchor_local,
                self.area,
                self.rest_len,
                self.neighbors,
                self.compression[t],
                self.base_pressure[t],
                self.params,
                self.material_params,
                self.friction_params,
                dt,
                float(self.ground_height_m or 0.0),
                int(self.ground_height_m is not None),
                self.friction_kt,
                self.friction_kv,
                self.tangent_anchor_init if t == 0 else self.tangent_anchor[t - 1],
                self.tangent_stuck_init if t == 0 else self.tangent_stuck[t - 1],
                self.tangent_dwell_init if t == 0 else self.tangent_dwell[t - 1],
                *(
                    [
                        self.tangent_deflection_init if t == 0 else self.tangent_deflection[t - 1],
                        self.tangent_maxwell_force_init if t == 0 else self.tangent_maxwell_force[t - 1],
                        self.tangent_deflection[t],
                        self.tangent_maxwell_force[t],
                    ]
                    if self.params.friction_model == 1
                    else []
                ),
                self.tangent_anchor[t],
                self.tangent_stuck[t],
                self.tangent_dwell[t],
                self.column_force[t],
                self.ground_force[t],
                state.body_f,
                self.normal_force,
                self.cop_moment,
                self.pressed_force,
                self.active,
            ],
            device=self.device,
        )

    def zero_grad(self) -> None:
        """Zero the accumulated gradients on the differentiable buffers."""
        self.material_params.grad.zero_()
        self.friction_params.grad.zero_()
        for buf in (*self.compression, *self.base_pressure, *self.q_state, *self.peq_prev):
            buf.grad.zero_()
        self.q_init.grad.zero_()
        self.peq_init.grad.zero_()
        self.tangent_anchor_init.grad.zero_()
        self.tangent_dwell_init.grad.zero_()
        self.tangent_deflection_init.grad.zero_()
        self.tangent_maxwell_force_init.grad.zero_()
        for buf in (*self.tangent_deflection, *self.tangent_maxwell_force):
            buf.grad.zero_()
        for buf in (*self.tangent_anchor, *self.tangent_dwell, *self.column_force, *self.ground_force):
            buf.grad.zero_()
        if self.free_column_count:
            self.surround_init.grad.zero_()
            for sweeps in self.surround_compression:
                for buf in sweeps:
                    buf.grad.zero_()
            for buf in self.z_free_substep:
                buf.grad.zero_()

    def diagnostics(self) -> dict[str, float]:
        """Return the last substep's total normal force, center of pressure, and active count.

        The centre of pressure divides by the *pressed* force, matching
        :meth:`projects.digital_shoe.runtime.MidsoleFoundation.diagnostics`: columns
        the shear layer lifts carry a small pull whose moment would otherwise drag
        the reported COP off the contact patch.
        """
        fz = float(self.normal_force.numpy()[0])
        pressed = float(self.pressed_force.numpy()[0])
        moment = self.cop_moment.numpy()[0]
        cop = (float(moment[0] / pressed), float(moment[1] / pressed)) if pressed > 1.0e-9 else (0.0, 0.0)
        return {
            "normal_force_n": fz,
            "pressed_force_n": pressed,
            "cop_x_m": cop[0],
            "cop_y_m": cop[1],
            "active_columns": int(self.active.numpy()[0]),
        }
