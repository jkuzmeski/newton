# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Differentiable elastic-foundation midsole for gradient-based simulation.

This is the autodiff-ready sibling of :mod:`projects.digital_instron_v2.dynamics`.
It exposes the same calibrated Hyperfoam-Maxwell-Pasternak column bed as a live
Warp force model, but re-shaped so the whole per-substep force integration can be
recorded on a :class:`warp.Tape` and differentiated end to end.

The *mechanics* are not re-derived here. The column bed clamps only its unilateral
ground reaction and then adds the (signed) Pasternak shear flux, treats every
missing neighbour as a free zero-gradient edge, and relaxes the columns the
carrier does not drive against the same balance the identification and the live
runtime relax them against -- see
:func:`projects.digital_shoe.runtime.foundation_apply` and
:func:`projects.digital_shoe.runtime._surround_balance`. Only three structural
changes make that model differentiable, and each one is forced by autodiff:

* **Tape-safe viscoelastic recurrence.** The forward
  :func:`~projects.digital_instron_v2.dynamics.foundation_pressure` updates the
  generalized-Maxwell overstress state *in place* (``q_state[i] = qn``), which
  aliases the same array across substeps and is unsafe to differentiate. Here
  :func:`foundation_pressure_diff` takes the previous substep's state as a
  read-only input and writes the next substep's state into a *separate* array, so
  every buffer on the loss path is written exactly once per rollout.

* **Cone-respecting smooth Coulomb friction.** The forward model's anchored
  bristle friction switches on an integer stick/slip flag, which has no useful
  gradient. :func:`foundation_apply_diff` instead uses
  :func:`warp.smooth_normalize` (the pseudo-Huber smoothed direction), giving a
  friction force ``-mu * fn * v_tan / sqrt(delta^2 + |v_tan|^2)`` that is smooth
  through zero relative velocity and never exceeds the cone ``mu * fn``.

* **Material read from an array.** Warp accumulates adjoints into arrays, so the
  fitted parameters cannot travel inside the by-value ``FoundationParams`` struct
  the runtime kernels read. :func:`_surround_balance_diff` and
  :func:`surround_relax_diff` are therefore separate transcriptions of
  :func:`projects.digital_shoe.runtime._surround_balance` and
  :func:`projects.digital_shoe.runtime.surround_relax`; the balance formula still
  lives in exactly one place per side and
  ``test_surround_balance_matches_runtime`` pins the two together. Adding the
  autodiff plumbing to the shipped runtime instead would slow every forward
  simulation down for it.

The live surround is a fixed, warm-started number of damped sweeps per substep, so
this module differentiates straight *through* the sweeps -- that derivative is
exact for the map the simulation evaluates. The identification relaxes to
convergence instead and uses an implicit-function-theorem adjoint; see
:class:`projects.digital_instron_v2.inverse_id.DifferentiableTrial`.

The constitutive parameters that a fit would vary -- both Ogden-Hill term moduli
and exponents and the Maxwell overstress ratio -- are held in a
length-5 ``requires_grad`` device array so gradients of any simulation objective
with respect to the foam material are available directly from
``material_params.grad``. The lateral shear layer is not among them: its
coefficient is pinned to the material as ``k_i = mu_eq * t_i`` per column
(:meth:`projects.digital_instron_v2.core.Material.coupling_n_per_m`), so it adds
no free parameter and its gradient flows through the series modulus
``g_eq + g_eq2``. The Coulomb friction coefficient ``mu`` is held in
a separate length-1 ``requires_grad`` ``friction_params`` array, so a lateral- or
shear-force objective can be differentiated with respect to friction as well
(friction identification), independent of the constitutive fit.

See :mod:`projects.digital_instron_v2.dynamics` for the (faster, forward-only)
production force model and the geometry/calibration helpers reused here.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from projects.digital_shoe.runtime import _hyperfoam_term, set_hyperfoam_series

from .core import EFFECTIVE_POISSON_RATIO, MAXWELL_RELAXATION_TIME_S, Material
from .dynamics import FoundationConfig, FoundationParams, SurroundConfig

# Indices into the differentiable ``material_params`` vector.
MAT_G_EQ = wp.constant(0)  # first-term equilibrium shear modulus G_inst * equilibrium_fraction [Pa]
MAT_ALPHA = wp.constant(1)  # first-term Hyperfoam exponent
MAT_OVERSTRESS = wp.constant(2)  # (1 - equilibrium_fraction) / equilibrium_fraction
MAT_G_EQ2 = wp.constant(3)  # second-term equilibrium shear modulus [Pa]; zero disables the term
MAT_ALPHA2 = wp.constant(4)  # second-term Hyperfoam exponent
MAT_COUNT = 5  # length of the differentiable material vector

# Index into the differentiable ``friction_params`` vector.
FRIC_MU = wp.constant(0)  # Coulomb friction coefficient (smooth-cone bound)


@wp.func
def _hyperfoam_pressure_diff(
    strain: wp.float32,
    g_eq: wp.float32,
    alpha: wp.float32,
    g_eq2: wp.float32,
    alpha2: wp.float32,
    p: FoundationParams,
) -> wp.float32:
    """Positive uniaxial compression pressure from the two-term Hyperfoam law.

    Identical law to :func:`~projects.digital_instron_v2.dynamics._hyperfoam_pressure`,
    reusing its :func:`~projects.digital_shoe.runtime._hyperfoam_term` so the law
    itself is written once, but with both differentiable term moduli and
    exponents passed as scalars so gradients flow into them.
    """
    stretch = 1.0 - strain
    if stretch < p.stretch_floor:
        stretch = p.stretch_floor
    volume_ratio = wp.pow(stretch, p.one_minus_two_poisson)
    return _hyperfoam_term(stretch, volume_ratio, g_eq, alpha, p.beta) + _hyperfoam_term(
        stretch, volume_ratio, g_eq2, alpha2, p.beta
    )


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
    decay = wp.exp(-dt / params.tau_s)
    ramp = params.tau_s * (1.0 - decay) / dt
    qn = decay * q_prev[i] + overstress * ramp * (peq - peq_prev[i])
    q_out[i] = qn
    peq_out[i] = peq
    base_pressure[i] = peq + qn


@wp.func
def _pasternak_coupling_diff(t_i: wp.float32, t_j: wp.float32, mu_eq: wp.float32) -> wp.float32:
    """Pasternak coefficient of the shear layer between two columns [N/m].

    Differentiable transcription of
    :func:`projects.digital_shoe.runtime._pasternak_coupling`: a shear-layer
    coefficient is ``G * t``, with the foam's own equilibrium Ogden-Hill modulus
    and the mean of the two column rest thicknesses at the shared face. It takes
    ``mu_eq`` -- the SUM of the two term moduli,
    ``material_params[MAT_G_EQ] + material_params[MAT_G_EQ2]`` -- as a scalar read
    from the ``requires_grad`` material vector instead of out of the by-value
    :class:`FoundationParams` struct, so both terms carry a coupling gradient.
    """
    return mu_eq * 0.5 * (t_i + t_j)


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
    ci = compression[i]
    flux = float(0.0)
    for side in range(4):
        j = neighbors[i, side]
        if j >= 0:
            flux += _pasternak_coupling_diff(rest_len[i], rest_len[j], mu_eq) * (compression[j] - ci)
    return flux


@wp.func
def _column_normal_force(
    ci: wp.float32,
    base_pressure_i: wp.float32,
    area_i: wp.float32,
    flux: wp.float32,
    normal_damping: wp.float32,
    vz: wp.float32,
) -> wp.float32:
    """Unilateral column ground reaction minus the (signed) shear-layer flux [N].

    One differentiable transcription of the normal-force path of
    :func:`projects.digital_shoe.runtime.foundation_apply`, so the whole
    differentiable path shares the runtime's contact mechanics:

    * The foam spring and its Kelvin-Voigt dashpot are clamped *together* and
      only against the ground: neither may pull the outsole back down.
    * The Pasternak shear flux is subtracted afterwards and stays unclamped. It
      redistributes load between columns and sums to zero over a free-edged bed,
      so clamping the combined pressure (the previous behaviour here) both clipped
      the flux and invented support under uncompressed foam.

    Args:
        ci: Column compression [m].
        base_pressure_i: Foam equilibrium pressure plus Maxwell overstress [Pa].
        area_i: Tributary area of the column [m^2].
        flux: Neighbour shear pulled out of this column [N], from
            :func:`_pasternak_flux`.
        normal_damping: Per-column Kelvin-Voigt normal damping [N.s/m].
        vz: Vertical velocity of the column anchor [m/s].
    """
    reaction = wp.max(base_pressure_i, 0.0) * area_i
    if ci > 0.0:
        reaction = reaction - normal_damping * vz
    if reaction < 0.0:
        reaction = 0.0
    return reaction - flux


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
    """Differentiable transcription of :func:`projects.digital_shoe.runtime._surround_balance`.

    One damped Newton step of an undriven column toward its local balance: its own
    unilateral ground reaction (equilibrium pressure plus the overstress the step
    itself produces), the Pasternak shear ``pull = sum_j k_ij (c_j - c)`` [N] from
    its neighbours with ``coupling_sum = sum_j k_ij`` [N/m] as the shear part of
    the local tangent, and the vertical bond to the shoe above. See the runtime
    function for the meaning of every argument and of ``carrier_bond``; this is
    the same formula, so the identification, the shipped runtime, and the
    differentiable path settle untouched foam identically.

    A separate function is unavoidable: the runtime reads the equilibrium modulus
    and the Hyperfoam exponent out of the by-value :class:`FoundationParams`
    struct, and Warp can only accumulate adjoints into arrays, so a
    ``requires_grad`` material must enter as the ``g_eq``, ``alpha``, ``g_eq2``
    and ``alpha2`` scalars of both Ogden-Hill terms, read from
    :attr:`DifferentiableMidsoleFoundation.material_params`. Putting that
    autodiff plumbing into the shipped runtime would slow every forward simulation
    down for it. The two implementations are pinned to each other by
    ``test_surround_balance_matches_runtime``; change one and change both.
    """
    peq = _hyperfoam_pressure_diff(c / thickness, g_eq, alpha, g_eq2, alpha2, params)
    reaction = area * wp.max(peq + overstress_base + overstress_gain * peq, 0.0)
    step = 1.0e-3 * thickness
    peq_ahead = _hyperfoam_pressure_diff((c + step) / thickness, g_eq, alpha, g_eq2, alpha2, params)
    ahead = area * wp.max(peq_ahead + overstress_base + overstress_gain * peq_ahead, 0.0)
    stiffness = wp.max((ahead - reaction) / step + attachment + coupling_sum, 1.0e-9)
    bond_reference = float(0.0)
    upper = max_strain * thickness
    if carrier_bond != 0:
        bond_reference = rigid
        upper = wp.clamp(rigid, 0.0, upper)
    residual = reaction + attachment * (c - bond_reference) - pull
    return wp.clamp(c - relaxation * residual / stiffness, 0.0, upper)


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
    """Pasternak coupling, per-column wrench into ``body_f``, and force diagnostics.

    Matches the normal-force path of
    :func:`~projects.digital_instron_v2.dynamics.foundation_apply` (pairwise
    Pasternak flux, pressure floor, Kelvin-Voigt normal damping) but replaces the
    non-differentiable anchored bristle friction with a smooth, cone-respecting
    Coulomb law built on :func:`warp.smooth_normalize`. The shear-layer
    coefficient is pinned to the material, so it follows the equilibrium modulus
    read from the differentiable ``material_params`` vector and the per-column
    rest thickness; the friction coefficient ``mu`` comes from the differentiable
    ``friction_params`` vector, so a lateral-force objective can be differentiated
    with respect to friction too.
    """
    i = wp.tid()
    # The shear layer follows the series modulus, which is the sum of both terms.
    mu_eq = material_params[MAT_G_EQ] + material_params[MAT_G_EQ2]

    ci = compression[i]
    flux = _pasternak_flux(i, compression, rest_len, neighbors, mu_eq)

    q_body = body_q[carrier]
    world = wp.transform_point(q_body, anchor_local[i])
    com_world = wp.transform_point(q_body, body_com[carrier])
    r = world - com_world
    vel = body_qd[carrier]
    point_vel = wp.spatial_top(vel) + wp.cross(wp.spatial_bottom(vel), r)

    fn = _column_normal_force(ci, base_pressure[i], area[i], flux, params.normal_damping, point_vel[2])
    # A column the shear layer lifts transmits a small pull, so the friction cone and
    # the centre of pressure use the pressed part only, as the runtime does.
    pressed = wp.max(fn, 0.0)

    # Cone-respecting smooth Coulomb friction: ft = -mu * fn * smooth_normalize(v_tan).
    # smooth_normalize(v, delta) = v / sqrt(delta^2 + |v|^2) has magnitude < 1, so the
    # tangential force is bounded by the cone mu * fn and is smooth through v_tan = 0.
    mu = friction_params[FRIC_MU]
    f_tan = wp.vec2(0.0, 0.0)
    if pressed > 0.0 and mu > 0.0:
        v_tan = wp.vec2(point_vel[0], point_vel[1])
        f_tan = -mu * pressed * wp.smooth_normalize(v_tan, friction_smoothing)

    force = wp.vec3(f_tan[0], f_tan[1], fn)
    wp.atomic_add(body_f, carrier, wp.spatial_vector(force, wp.cross(r, force)))
    wp.atomic_add(normal_force, 0, fn)
    wp.atomic_add(cop_moment, 0, wp.vec3(world[0] * pressed, world[1] * pressed, 0.0))
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
    differentiated. Drive it exactly like the forward model, but pass the current
    substep index so the correct history slot and the previous overstress state
    are used::

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
        friction_smoothing: Tangential velocity smoothing scale for the smooth
            Coulomb friction [m/s].
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
        self.device = device
        self.carrier = int(carrier_body)
        self.body_com = body_com
        self.column_count = int(len(rest_len))
        self.num_substeps = int(num_substeps)
        self.friction_smoothing = float(friction_smoothing)

        params = FoundationParams()
        set_hyperfoam_series(params, material)
        params.beta = EFFECTIVE_POISSON_RATIO / (1.0 - 2.0 * EFFECTIVE_POISSON_RATIO)
        params.one_minus_two_poisson = 1.0 - 2.0 * EFFECTIVE_POISSON_RATIO
        params.tau_s = float(getattr(material, "maxwell_relaxation_time_s", MAXWELL_RELAXATION_TIME_S))
        params.overstress = (1.0 - material.equilibrium_fraction) / material.equilibrium_fraction
        # Grid geometry only: the shear-layer coefficient is pinned per column to
        # ``mu_eq * t_i`` and no longer scales with the spacing.
        params.inv_h2 = 1.0 / spacing_m**2
        params.stretch_floor = config.stretch_floor
        params.normal_damping = config.normal_damping
        params.friction_kt = config.friction_stiffness
        params.friction_kv = config.friction
        params.mu = config.mu
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
        decay = float(np.exp(-dt / self.params.tau_s))
        ramp = float(self.params.tau_s * (1.0 - decay) / dt)
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
            foundation_apply_diff,
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
                self.friction_smoothing,
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
