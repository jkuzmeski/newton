# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Portable, GPU-native runtime for an identified digital shoe foundation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp


@dataclass(frozen=True)
class ShoeMaterial:
    """Effective intact-shoe constitutive parameters used by the runtime.

    ``pasternak_n_per_m`` is a *reported* quantity, kept so the artifact schema
    stays readable and backward compatible. Nothing consumes it: the runtime
    derives each column's own Pasternak coefficient from the equilibrium
    Ogden-Hill shear modulus and that column's rest thickness,
    ``k_i = mu_eq * t_i``, which is the shear-layer definition ``G * t`` and adds
    no free parameter. Exporters write the bed-mean of that rule here.
    """

    instantaneous_shear_modulus_pa: float
    hyperfoam_exponent: float
    equilibrium_fraction: float
    pasternak_n_per_m: float
    effective_poisson_ratio: float = 0.0
    maxwell_relaxation_time_s: float = 0.08

    def __post_init__(self) -> None:
        values = tuple(self.__dict__.values())
        if not np.all(np.isfinite(values)):
            raise ValueError("shoe material parameters must be finite")
        if self.instantaneous_shear_modulus_pa <= 0.0 or self.hyperfoam_exponent <= 0.0:
            raise ValueError("shear modulus and Hyperfoam exponent must be positive")
        if not 0.0 < self.equilibrium_fraction <= 1.0:
            raise ValueError("equilibrium fraction must be in (0, 1]")
        if self.pasternak_n_per_m < 0.0:
            raise ValueError("Pasternak coupling must be nonnegative")
        if not -1.0 < self.effective_poisson_ratio < 0.5:
            raise ValueError("effective Poisson ratio must be in (-1, 0.5)")
        if self.maxwell_relaxation_time_s <= 0.0:
            raise ValueError("Maxwell relaxation time must be positive")


@wp.struct
class FoundationParams:
    """Device-side constitutive and contact constants for the column bed."""

    g_eq: wp.float32  # equilibrium shear modulus G_inst * equilibrium_fraction [Pa]
    alpha: wp.float32  # Hyperfoam exponent
    beta: wp.float32  # poisson / (1 - 2 poisson)
    one_minus_two_poisson: wp.float32  # volumetric stretch exponent
    tau_s: wp.float32  # Maxwell relaxation time [s]
    overstress: wp.float32  # (1 - equilibrium_fraction) / equilibrium_fraction
    inv_h2: wp.float32  # 1 / spacing^2 [1/m^2]
    stretch_floor: wp.float32  # minimum stretch (foam densification limit)
    normal_damping: wp.float32  # per-column Kelvin-Voigt normal damping [N.s/m]
    friction_kt: wp.float32  # uniform bristle tangential stiffness [N/m per column]
    friction_kv: wp.float32  # uniform bristle tangential damping [N.s/m per column]
    friction_viscous_ratio: wp.float32  # viscous cap as a fraction of the cone mu*fn
    friction_release_dwell_s: wp.float32  # unloaded dwell before the stick point is discarded [s]
    mu: wp.float32  # Coulomb friction coefficient


@wp.func
def _hyperfoam_pressure(strain: wp.float32, p: FoundationParams) -> wp.float32:
    """Positive uniaxial compression pressure from the first-order Hyperfoam law.

    At the measured zero effective Poisson ratio ``beta`` is zero and
    ``one_minus_two_poisson`` is one, so the volumetric factor is ``pow(x, 0)``
    with ``x >= stretch_floor > 0``. That is exactly one on CPU and CUDA and
    needs no special case; the stretch floor is what keeps it away from
    ``pow(0, 0)``.
    """
    stretch = 1.0 - strain
    if stretch < p.stretch_floor:
        stretch = p.stretch_floor
    volume_ratio = wp.pow(stretch, p.one_minus_two_poisson)
    return 2.0 * p.g_eq / (p.alpha * stretch) * (wp.pow(volume_ratio, -p.alpha * p.beta) - wp.pow(stretch, p.alpha))


@wp.func
def _pasternak_coupling(t_i: wp.float32, t_j: wp.float32, p: FoundationParams) -> wp.float32:
    """Pasternak coefficient of the shear layer between two columns [N/m].

    A Pasternak layer coefficient is ``G * t``. The shear modulus is the foam's
    own equilibrium Ogden-Hill modulus ``mu_eq``, already carried as
    :attr:`FoundationParams.g_eq`, and the layer thickness at the shared face is
    the mean of the two column rest thicknesses. Averaging keeps the pair
    conductance symmetric, so the lateral flux summed over the bed is exactly
    zero and the layer can only move load, never create it.
    """
    return p.g_eq * 0.5 * (t_i + t_j)


@wp.kernel
def foundation_pressure(
    carrier: wp.int32,
    dt: wp.float32,
    body_q: wp.array[wp.transform],
    anchor_local: wp.array[wp.vec3],
    z_free: wp.array[wp.float32],
    rest_len: wp.array[wp.float32],
    params: FoundationParams,
    q_state: wp.array[wp.float32],
    peq_prev: wp.array[wp.float32],
    compression: wp.array[wp.float32],
    base_pressure: wp.array[wp.float32],
):
    """Compression, Hyperfoam equilibrium pressure, and real-time Maxwell overstress."""
    i = wp.tid()
    world = wp.transform_point(body_q[carrier], anchor_local[i])
    comp = z_free[i] - world[2]
    if comp < 0.0:
        comp = 0.0
    compression[i] = comp
    strain = comp / rest_len[i]
    peq = _hyperfoam_pressure(strain, params)
    decay = wp.exp(-dt / params.tau_s)
    ramp = params.tau_s * (1.0 - decay) / dt
    qn = decay * q_state[i] + params.overstress * ramp * (peq - peq_prev[i])
    q_state[i] = qn
    peq_prev[i] = peq
    base_pressure[i] = peq + qn


@wp.func
def _cone_viscous_scale(f_elastic: wp.vec2, f_viscous: wp.vec2, f_max: wp.float32) -> wp.float32:
    """Largest fraction of a viscous force that keeps the total tangential force inside the cone.

    The elastic part is already on or inside the cone after the radial return, so scaling
    only the viscous part keeps the Coulomb bound strict while the retained fraction still
    opposes the slip velocity (the pair stays dissipative).
    """
    a = wp.dot(f_viscous, f_viscous)
    if a <= 1.0e-18:
        return 0.0
    b = 2.0 * wp.dot(f_elastic, f_viscous)
    c = wp.dot(f_elastic, f_elastic) - f_max * f_max
    if c > 0.0:
        c = 0.0
    disc = b * b - 4.0 * a * c
    if disc <= 0.0:
        return 0.0
    return wp.clamp((-b + wp.sqrt(disc)) / (2.0 * a), 0.0, 1.0)


@wp.kernel
def foundation_apply(
    carrier: wp.int32,
    dt: wp.float32,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    anchor_local: wp.array[wp.vec3],
    area: wp.array[wp.float32],
    rest_len: wp.array[wp.float32],
    neighbors: wp.array2d[wp.int32],
    compression: wp.array[wp.float32],
    base_pressure: wp.array[wp.float32],
    tangent_anchor: wp.array[wp.vec2],
    tangent_stuck: wp.array[wp.int32],
    tangent_dwell: wp.array[wp.float32],
    friction_kt: wp.array[wp.float32],
    friction_kv: wp.array[wp.float32],
    params: FoundationParams,
    body_f: wp.array[wp.spatial_vector],
    normal_force: wp.array[wp.float32],
    cop_moment: wp.array[wp.vec3],
    active_count: wp.array[wp.int32],
    resultant_force: wp.array[wp.vec3],
    resultant_moment_origin: wp.array[wp.vec3],
    contact_power: wp.array[wp.float32],
    max_compression: wp.array[wp.float32],
    column_force: wp.array[wp.vec3],
    pressed_force: wp.array[wp.float32],
):
    """Pasternak coupling, per-column wrench into ``body_f``, and force diagnostics."""
    i = wp.tid()
    ci = compression[i]
    # Pairwise shear flux with the material-pinned coefficient of every face. On
    # the square grid the tributary area is the spacing squared, so the discrete
    # Pasternak force ``-G_p A lap(c)`` is exactly this sum of face terms, and
    # writing it pairwise makes the bed total cancel to the last bit.
    #
    # Every missing neighbour is a free (zero-gradient) edge, whether it lies
    # outside the midsole or is simply not in the active set, so it contributes
    # nothing. Holding one at zero compression instead would make it a rigid
    # hidden support: the shear layer would lean on it and create load rather
    # than only spreading it.
    flux = float(0.0)
    for side in range(4):
        j = neighbors[i, side]
        if j >= 0:
            flux += _pasternak_coupling(rest_len[i], rest_len[j], params) * (compression[j] - ci)

    # Clamp the foam pressure itself: the springs cannot pull.
    ground = base_pressure[i]
    if ground < 0.0:
        ground = 0.0

    q_body = body_q[carrier]
    world = wp.transform_point(q_body, anchor_local[i])
    com_world = wp.transform_point(q_body, body_com[carrier])
    r = world - com_world
    vel = body_qd[carrier]
    point_vel = wp.spatial_top(vel) + wp.cross(wp.spatial_bottom(vel), r)

    # Kelvin-Voigt ground reaction, kept unilateral: neither the foam spring nor its
    # dashpot may pull the outsole back down. Clamping only the spring let the dashpot
    # invert on rebound, which sucked the settling midsole back into the ground and turned
    # contact into a sticky bouncer instead of an equilibrium.
    reaction = ground * area[i]
    if ci > 0.0:
        reaction = reaction - params.normal_damping * point_vel[2]
    if reaction < 0.0:
        reaction = 0.0
    # The shear-layer flux stays unclamped: it redistributes load between columns and
    # clipping it per column would invent net support under uncompressed foam.
    fn = reaction - flux
    # Unilateral Pasternak base pressure. A column that the shear layer lifts transmits a
    # small pull, so the friction cone and the centre of pressure use the pressed part.
    pressed = fn
    if pressed < 0.0:
        pressed = 0.0

    # Anchored bristle (elastoplastic) Coulomb friction: a per-column tangential
    # spring pulls the contact patch back toward a world stick point, so a planted
    # patch holds (static regime, zero drift) and carries braking/propulsion shear
    # without needing a slip velocity. When the spring force would exceed the cone
    # mu*fn the elastic trial saturates and the anchor slides onto the cone (kinetic
    # regime). Only the elastic trial enters that return map: a dashpot inside it
    # would fire the plastic update on columns that are merely moving fast, which
    # erases the elastic memory of a still-gripping bristle.
    p_t = wp.vec2(world[0], world[1])
    v_tan = wp.vec2(point_vel[0], point_vel[1])
    kt = friction_kt[i]
    kv = friction_kv[i]
    f_max = params.mu * pressed
    f_tan = wp.vec2(0.0, 0.0)
    if pressed <= 0.0 or kt <= 0.0:
        # Hold the stick point through short normal dropouts. Perimeter columns chatter
        # in and out of contact at the substep rate; discarding the elastic state on that
        # chatter is a numerical release, not a physical one. Re-entry stays bounded
        # because the cone still scales with fn.
        dwell = tangent_dwell[i] + dt
        if kt <= 0.0 or tangent_stuck[i] == 0 or dwell > params.friction_release_dwell_s:
            tangent_anchor[i] = p_t
            tangent_stuck[i] = 0
            dwell = 0.0
        tangent_dwell[i] = dwell
    else:
        tangent_dwell[i] = 0.0
        if tangent_stuck[i] == 0:
            tangent_anchor[i] = p_t  # fresh contact: seat with no pre-stretch
            tangent_stuck[i] = 1
        # Evaluate the elastic trial at the end-of-step position, so the stiff stick mode
        # is damped like a backward-Euler step instead of ringing at the substep rate.
        p_next = p_t + v_tan * dt
        f_elastic = -kt * (p_next - tangent_anchor[i])
        mag = wp.length(f_elastic)
        if mag > f_max:
            f_elastic = f_elastic * (f_max / wp.max(mag, 1.0e-12))
            tangent_anchor[i] = p_next + f_elastic / kt  # radial return on the elastic trial
        f_tan = f_elastic
        speed = wp.length(v_tan)
        if kv > 0.0 and speed > 1.0e-12:
            # Viscous term outside the return map, capped well below the cone so it can
            # regularize presliding without setting the direction of a sliding column.
            viscous = -v_tan * (wp.min(kv * speed, params.friction_viscous_ratio * f_max) / speed)
            f_tan = f_elastic + viscous * _cone_viscous_scale(f_elastic, viscous, f_max)

    force = wp.vec3(f_tan[0], f_tan[1], fn)
    column_force[i] = force
    wp.atomic_add(body_f, carrier, wp.spatial_vector(force, wp.cross(r, force)))
    wp.atomic_add(normal_force, 0, fn)
    wp.atomic_add(cop_moment, 0, wp.vec3(world[0] * pressed, world[1] * pressed, 0.0))
    wp.atomic_add(pressed_force, 0, pressed)
    wp.atomic_add(resultant_force, 0, force)
    wp.atomic_add(resultant_moment_origin, 0, wp.cross(world, force))
    wp.atomic_add(contact_power, 0, wp.dot(force, point_vel))
    wp.atomic_max(max_compression, 0, ci)
    if ci > 0.0:
        wp.atomic_add(active_count, 0, 1)


@wp.kernel
def cycle_overstress(
    compression: wp.array2d[wp.float32],
    slack: wp.array[wp.float32],
    dt_s: wp.array[wp.float32],
    params: FoundationParams,
    fraction: wp.float32,
    tau_s: wp.float32,
    overstress_out: wp.array2d[wp.float32],
):
    """Write one column's periodic Maxwell overstress for every frame of a cycle.

    Each thread owns a column and walks the cycle twice: the first pass finds the
    periodic state, the second records it frame by frame. :func:`surround_sweep`
    and :func:`cycle_force` then share this overstress, so the relaxed surround
    balances the same load the summed reaction later reports.
    """
    i = wp.tid()
    frames = compression.shape[0]
    thickness = slack[i]
    state = float(0.0)
    decay_product = float(1.0)
    previous = _hyperfoam_pressure(compression[frames - 1, i] / thickness, params)
    for frame in range(frames):
        equilibrium = _hyperfoam_pressure(compression[frame, i] / thickness, params)
        decay = wp.exp(-dt_s[frame] / tau_s)
        ramp = tau_s * (1.0 - decay) / dt_s[frame]
        state = decay * state + fraction * ramp * (equilibrium - previous)
        decay_product *= decay
        previous = equilibrium
    state = state / (1.0 - decay_product)
    previous = _hyperfoam_pressure(compression[frames - 1, i] / thickness, params)
    for frame in range(frames):
        equilibrium = _hyperfoam_pressure(compression[frame, i] / thickness, params)
        decay = wp.exp(-dt_s[frame] / tau_s)
        ramp = tau_s * (1.0 - decay) / dt_s[frame]
        state = decay * state + fraction * ramp * (equilibrium - previous)
        previous = equilibrium
        overstress_out[frame, i] = state


@wp.kernel
def cycle_force(
    compression: wp.array2d[wp.float32],
    overstress: wp.array2d[wp.float32],
    slack: wp.array[wp.float32],
    params: FoundationParams,
    area: wp.float32,
    force_out: wp.array[wp.float32],
):
    """Sum the unilateral ground reaction of every column into each frame of a cycle.

    The overstress comes from :func:`cycle_overstress` on the same compression,
    the clamp keeps the reaction unilateral, and the shear flux cancels
    internally, so the sum is the load an Instron would measure.
    """
    frame, i = wp.tid()
    thickness = slack[i]
    equilibrium = _hyperfoam_pressure(compression[frame, i] / thickness, params)
    wp.atomic_add(force_out, frame, area * wp.max(equilibrium + overstress[frame, i], 0.0))


@wp.func
def _surround_balance(
    c: wp.float32,
    rigid: wp.float32,
    pull: wp.float32,
    coupling_sum: wp.float32,
    thickness: wp.float32,
    overstress_base: wp.float32,
    overstress_gain: wp.float32,
    params: FoundationParams,
    area: wp.float32,
    attachment: wp.float32,
    max_strain: wp.float32,
    relaxation: wp.float32,
    carrier_bond: wp.int32,
) -> wp.float32:
    """One damped Newton step of an undriven column toward its local balance.

    This is the single implementation of the passive-surround balance. The
    identification sweeps it over a whole cycle at once (:func:`surround_sweep`)
    and the live runtime sweeps it every substep (:func:`surround_relax`), so a
    fitted shoe and a simulated shoe settle their untouched foam identically.

    The column carries its own unilateral ground reaction and the Pasternak shear
    ``pull`` from its neighbours, whose per-face coefficients come from
    :func:`_pasternak_coupling`. ``attachment`` is an optional vertical bond to
    the shoe above it and defaults to zero everywhere, because its reaction never
    reached the reported force or the carrier wrench: booked that way it was an
    undeclared rigid support, not a bond. ``rigid`` is the compression the carrier
    would impose on this column if it rode along rigidly:

    * ``carrier_bond == 0``: the column top is a free shoe surface the carrier
      never touches. Callers pass ``rigid = 0``, which is the bench-fixture
      surround of the identification.
    * ``carrier_bond != 0``: the column top is glued under the rigid carrier, so
      the foam cannot compress past the carrier-imposed value (it would have to
      peel off the shoe) and any bond is unstretched there.

    The Maxwell overstress the column will carry once the step is taken is
    ``overstress_base + overstress_gain * p_eq(c)``, so a relaxation that moves
    ``c`` is balanced against the load it actually ends up under. Ignoring the
    gain understates the reaction tangent by ``1 + gain`` (about a factor of ten
    for this foam), which makes the Newton step overshoot and the compression
    ring at the substep rate. The identification holds the overstress fixed
    inside one pass and refreshes it between passes, so it passes ``gain = 0``.

    Args:
        c: Current column compression [m].
        rigid: Compression the rigid carrier imposes on this column [m].
        pull: Summed neighbour shear ``sum_j k_ij * (c_j - c)`` [N].
        coupling_sum: Summed face coefficients ``sum_j k_ij`` [N/m], the shear
            part of the local tangent.
        thickness: Column rest thickness [m].
        overstress_base: Maxwell overstress the column carries at zero
            equilibrium pressure [Pa].
        overstress_gain: Overstress produced per unit equilibrium pressure by the
            step about to be taken.
        params: Device-side constitutive constants.
        area: Tributary area of the column [m^2].
        attachment: Vertical bond stiffness to the shoe [N/m], normally zero.
        max_strain: Compression limit as a fraction of rest thickness.
        relaxation: Fraction of the Newton step taken, ``1`` for quasi-static.
        carrier_bond: Nonzero when the column top rides with the carrier.

    Returns:
        The updated column compression [m].
    """
    peq = _hyperfoam_pressure(c / thickness, params)
    reaction = area * wp.max(peq + overstress_base + overstress_gain * peq, 0.0)
    step = 1.0e-3 * thickness
    peq_ahead = _hyperfoam_pressure((c + step) / thickness, params)
    ahead = area * wp.max(peq_ahead + overstress_base + overstress_gain * peq_ahead, 0.0)
    stiffness = wp.max((ahead - reaction) / step + attachment + coupling_sum, 1.0e-9)
    # A free bench surface is bonded in the undeformed shoe; a glued top is bonded where
    # the carrier holds it and cannot be compressed past that without peeling off.
    bond_reference = float(0.0)
    upper = max_strain * thickness
    if carrier_bond != 0:
        bond_reference = rigid
        upper = wp.clamp(rigid, 0.0, upper)
    residual = reaction + attachment * (c - bond_reference) - pull
    return wp.clamp(c - relaxation * residual / stiffness, 0.0, upper)


@wp.kernel
def surround_sweep(
    compression_in: wp.array2d[wp.float32],
    overstress: wp.array2d[wp.float32],
    driven: wp.array[wp.int32],
    neighbors: wp.array2d[wp.int32],
    slack: wp.array[wp.float32],
    params: FoundationParams,
    area: wp.float32,
    coupling_scale: wp.float32,
    attachment: wp.float32,
    max_strain: wp.float32,
    relaxation: wp.float32,
    compression_out: wp.array2d[wp.float32],
):
    """Relax one untouched column toward its own quasi-static balance.

    Each thread owns one frame and one column. Driven columns pass straight
    through, so the indenter keeps its imposed compression while the surrounding
    foam settles against neighbour shear and its unilateral ground reaction.
    Frames are independent, so a whole test relaxes at once.

    The support is the same equilibrium-plus-overstress reaction :func:`cycle_force`
    later sums, and the tangent differentiates that same sum. Balancing the
    equilibrium pressure alone would relax the surround against a support the
    loading then multiplies by ``1 + q / p_eq``, which is what drove the fit
    toward a short relaxation time. The overstress is held fixed inside one
    relaxation and refreshed by :func:`cycle_overstress` between passes.

    The bench fixture never touches this foam, so ``rigid = 0`` in
    :func:`_surround_balance`. ``relaxation`` is the fraction of the local Newton
    step taken: one is plain damped Jacobi, above one is successive
    over-relaxation, which reaches the same fixed point in fewer sweeps.
    """
    frame, i = wp.tid()
    if driven[i] != 0:
        compression_out[frame, i] = compression_in[frame, i]
        return
    c = compression_in[frame, i]
    pull = float(0.0)
    coupling_sum = float(0.0)
    for side in range(4):
        j = neighbors[i, side]
        if j >= 0:
            coupling = coupling_scale * _pasternak_coupling(slack[i], slack[j], params)
            pull += coupling * (compression_in[frame, j] - c)
            coupling_sum += coupling
    compression_out[frame, i] = _surround_balance(
        c,
        0.0,
        pull,
        coupling_sum,
        slack[i],
        overstress[frame, i],
        0.0,
        params,
        area,
        attachment,
        max_strain,
        relaxation,
        0,
    )


@wp.kernel
def surround_relax(
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
    decay: wp.float32,
    overstress_gain: wp.float32,
    coupling_scale: wp.float32,
    attachment: wp.float32,
    max_strain: wp.float32,
    relaxation: wp.float32,
    carrier_bond: wp.int32,
    compression_in: wp.array[wp.float32],
    compression_out: wp.array[wp.float32],
):
    """Sweep the live column bed once toward the same balance the fit relaxes.

    Driven columns take the compression their carrier pose imposes, so the
    untouched foam reads the indenter through the shear layer exactly as
    :func:`surround_sweep` does during identification. The balance itself is
    :func:`_surround_balance`; only the source of the driven compression and the
    per-substep damping differ.

    ``decay`` and ``overstress_gain`` are the substep's Maxwell update, so the
    surround settles against the overstress :func:`foundation_pressure` is about
    to write for the compression this sweep produces, not against the previous
    substep's value.
    """
    i = wp.tid()
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
            coupling = coupling_scale * _pasternak_coupling(rest_len[i], rest_len[j], params)
            pull += coupling * (compression_in[j] - c)
            coupling_sum += coupling
    compression_out[i] = _surround_balance(
        c,
        rigid,
        pull,
        coupling_sum,
        rest_len[i],
        decay * q_state[i] - overstress_gain * peq_prev[i],
        overstress_gain,
        params,
        area[i],
        attachment,
        max_strain,
        relaxation,
        carrier_bond,
    )


@wp.kernel
def surround_write_free_top(
    carrier: wp.int32,
    inv_dt: wp.float32,
    body_q: wp.array[wp.transform],
    driven: wp.array[wp.int32],
    anchor_local: wp.array[wp.vec3],
    z_free_rigid: wp.array[wp.float32],
    compression: wp.array[wp.float32],
    previous: wp.array[wp.float32],
    z_free: wp.array[wp.float32],
    rate: wp.array[wp.float32],
):
    """Publish the relaxed free surface the shared pressure kernel then consumes.

    :func:`foundation_pressure` reads ``compression = z_free - world_z``, so
    moving ``z_free`` to ``world_z + c`` hands it the relaxed compression without
    a second contact path. Driven columns keep their rigid free top.
    """
    i = wp.tid()
    if driven[i] != 0:
        z_free[i] = z_free_rigid[i]
        rate[i] = 0.0
        return
    world = wp.transform_point(body_q[carrier], anchor_local[i])
    c = compression[i]
    z_free[i] = world[2] + c
    rate[i] = (c - previous[i]) * inv_dt
    previous[i] = c


@wp.kernel
def surround_seed_driven(
    driven_compression: wp.array2d[wp.float32],
    driven_index: wp.array[wp.int32],
    compression: wp.array2d[wp.float32],
):
    """Stamp the indenter's imposed compression onto a warm-started field.

    :func:`surround_sweep` passes driven columns straight through, so a field
    reused from an earlier solve is only valid once its driven columns carry the
    compression of the current call.
    """
    frame, k = wp.tid()
    compression[frame, driven_index[k]] = driven_compression[frame, k]


@wp.kernel
def surround_update_max(
    a: wp.array2d[wp.float32],
    b: wp.array2d[wp.float32],
    out: wp.array[wp.float32],
):
    """Reduce the largest compression change between two consecutive sweeps [m]."""
    frame, i = wp.tid()
    wp.atomic_max(out, 0, wp.abs(a[frame, i] - b[frame, i]))


def relax_surround(
    driven_compression: np.ndarray,
    driven: np.ndarray,
    neighbors: np.ndarray,
    slack_m: np.ndarray,
    params: FoundationParams,
    *,
    area_m2: float,
    spacing_m: float,
    attachment_n_m: float,
    max_strain: float,
    sweeps: int,
    coupling_scale: float = 1.0,
    over_relaxation: float = 1.0,
    overstress: wp.array2d[wp.float32] | None = None,
    initial: wp.array2d[wp.float32] | None = None,
    tolerance_m: float = 0.0,
    check_every: int = 25,
    stats: dict[str, float] | None = None,
    device=None,
) -> np.ndarray:
    """Return whole-midsole compression for every frame of a test.

    The identification and the live runtime therefore share one contact model
    and one geometry: the indenter drives its columns and the rest relax.

    The sweep count is a cap, not a schedule. ``tolerance_m`` stops the solve on
    the *extrapolated remaining error* rather than on the raw update: the damped
    Jacobi sweep contracts geometrically, so an update of ``u`` with a measured
    per-interval decay ``q`` still has about ``u q / (1 - q)`` of travel left,
    and stopping on ``u`` alone would report convergence a factor ``1 / (1 - q)``
    too early. Together with ``initial`` this is what makes a fit affordable: the
    bed barely moves between finite-difference evaluations, so a warm-started
    solve needs a few sweeps where a cold one needs thousands.

    Args:
        driven_compression: Imposed compression of the driven columns [m],
            shape ``[frames, driven_count]``.
        driven: Nonzero for columns the indenter drives, shape ``[column_count]``.
        neighbors: Four in-plane neighbour indices; negative is a free edge.
        slack_m: Rest thickness per column [m].
        params: Device-side constitutive constants.
        area_m2: Tributary area per column [m^2].
        spacing_m: Column grid spacing [m].
        attachment_n_m: Vertical bond of untouched foam to the shoe [N/m];
            zero is the booked-consistently default.
        max_strain: Compression limit as a fraction of rest thickness.
        sweeps: Hard cap on relaxation sweeps.
        coupling_scale: Fraction of the material-pinned face coefficient
            ``mu_eq * t`` that drives the passive surface.
        over_relaxation: Fraction of the local Newton step taken per sweep. One
            is plain damped Jacobi; above one it is successive over-relaxation.
        overstress: Maxwell overstress carried by every column and frame [Pa],
            shape ``[frames, column_count]``, from :func:`cycle_overstress`.
            ``None`` relaxes against the equilibrium pressure alone, which is
            only the first pass of a self-consistent solve.
        initial: Warm-start compression [m], shape ``[frames, column_count]``,
            normally the previous converged solve. Its driven columns are
            overwritten with ``driven_compression``.
        tolerance_m: Extrapolated remaining compression travel that ends the
            solve [m]. Zero always runs the full ``sweeps`` cap.
        check_every: Sweeps between convergence tests. Each test costs one
            reduction and one device synchronization.
        stats: Optional mapping that receives ``sweeps``, ``update_m`` and
            ``remaining_m`` so the convergence stays visible to the caller.

    Returns:
        Device compression for every column and frame [m], shape
        ``[frames, column_count]``, ready for :func:`cycle_force`.
    """
    driven = np.ascontiguousarray(driven, np.int32)
    frames = len(driven_compression)
    count = len(slack_m)
    driven_index = np.ascontiguousarray(np.flatnonzero(driven != 0), np.int32)
    if initial is None:
        current = wp.zeros((frames, count), dtype=wp.float32, device=device)
    else:
        current = wp.clone(initial)
    device = current.device
    wp.launch(
        surround_seed_driven,
        dim=(frames, len(driven_index)),
        inputs=[
            wp.array(np.ascontiguousarray(driven_compression, np.float32), dtype=wp.float32, device=device),
            wp.array(driven_index, dtype=wp.int32, device=device),
            current,
        ],
        device=device,
    )
    scratch = wp.zeros_like(current)
    if overstress is None:
        overstress = wp.zeros((frames, count), dtype=wp.float32, device=device)
    driven_device = wp.array(driven, dtype=wp.int32, device=device)
    neighbor_device = wp.array(np.ascontiguousarray(neighbors, np.int32), dtype=wp.int32, device=device)
    slack_device = wp.array(np.ascontiguousarray(slack_m, np.float32), dtype=wp.float32, device=device)
    update = wp.zeros(1, dtype=wp.float32, device=device)
    interval = max(int(check_every), 1)
    previous_update = float("inf")
    used = 0
    change = float("nan")
    remaining = float("nan")
    for sweep in range(max(sweeps, 0)):
        wp.launch(
            surround_sweep,
            dim=(frames, count),
            inputs=[
                current,
                overstress,
                driven_device,
                neighbor_device,
                slack_device,
                params,
                float(area_m2),
                float(coupling_scale),
                float(attachment_n_m),
                float(max_strain),
                float(over_relaxation),
                scratch,
            ],
            device=device,
        )
        used = sweep + 1
        if tolerance_m > 0.0 and used % interval == 0:
            update.zero_()
            wp.launch(surround_update_max, dim=(frames, count), inputs=[current, scratch, update], device=device)
            change = float(update.numpy()[0])
            # The first interval has nothing to measure a decay against, and an
            # iteration that is not contracting has no finite extrapolation, so
            # both keep the solve running rather than declaring success.
            decay = change / previous_update if 0.0 < previous_update < float("inf") else float("inf")
            previous_update = change
            if change == 0.0:
                remaining = 0.0
            elif decay >= 1.0:
                remaining = float("inf")
            else:
                remaining = change * decay / (1.0 - decay)
            current, scratch = scratch, current
            if remaining < tolerance_m:
                break
            continue
        current, scratch = scratch, current
    if stats is not None:
        stats.update({"sweeps": float(used), "update_m": change, "remaining_m": remaining})
    return current


@wp.kernel
def foundation_reset(
    carrier: wp.int32,
    clear_body_force: wp.int32,
    body_f: wp.array[wp.spatial_vector],
    normal_force: wp.array[wp.float32],
    cop_moment: wp.array[wp.vec3],
    active_count: wp.array[wp.int32],
    resultant_force: wp.array[wp.vec3],
    resultant_moment_origin: wp.array[wp.vec3],
    contact_power: wp.array[wp.float32],
    max_compression: wp.array[wp.float32],
    pressed_force: wp.array[wp.float32],
):
    """Zero the per-substep foundation accumulators (and optionally the carrier wrench).

    Folds the reduction resets into a single one-thread kernel launch. Each of the
    accumulator memsets is a graph node that dwarfs the actual per-column physics, so
    collapsing them to one node is the dominant cost saving for the captured attached
    loop. ``clear_body_force`` also zeros the carrier wrench so the attached loop needs
    no separate :meth:`newton.State.clear_forces` launch.
    """
    normal_force[0] = 0.0
    cop_moment[0] = wp.vec3(0.0, 0.0, 0.0)
    active_count[0] = 0
    resultant_force[0] = wp.vec3(0.0, 0.0, 0.0)
    resultant_moment_origin[0] = wp.vec3(0.0, 0.0, 0.0)
    contact_power[0] = 0.0
    max_compression[0] = 0.0
    pressed_force[0] = 0.0
    if clear_body_force != 0:
        body_f[carrier] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Foundation driver
# ---------------------------------------------------------------------------
@dataclass
class FoundationConfig:
    """Tunable dynamic parameters layered on the calibrated constitutive law.

    The Instron replay leaves ``normal_damping``, ``friction_stiffness`` and
    ``friction`` at zero and keeps ``stretch_floor`` below the calibration's peak
    strain so the collected loop reproduces the fitted force-displacement response
    exactly. The free-body scenarios add foam damping and an anchored bristle
    (elastoplastic) Coulomb friction: ``friction_stiffness`` is the per-column
    tangential spring that holds a planted contact patch (true stick), ``friction``
    is its viscous regularization, and ``mu`` bounds the tangential force at the cone
    ``mu * fn`` (slip).

    The tangential bed is a property of the contact area, not of the sampling grid, so
    the per-column values are converted to ``friction_stiffness_per_area``
    ``k'' = kt / A`` [N/m^3] and ``friction_damping_per_area`` ``c'' = kv / A``
    [N.s/m^3] and re-expanded as ``kt_i = k'' * A_i``. Set the per-area fields directly
    to declare the tangential layer independently of the column count; leaving them at
    zero derives them from the per-column values and the mean tributary area, which is
    exactly the previous behaviour on a uniform grid. The current shoe settings,
    kt = 1e4 N/m over 25 mm^2, correspond to ``k'' = 4.0e8 N/m^3``, i.e. a thin outsole
    rubber layer (G ~ 0.8 MPa over 2 mm). That is a declared assumption: it is 2-3
    orders of magnitude above the shear stiffness of the identified foam, and the
    Instron identification is compression only.

    ``friction_viscous_ratio`` caps the viscous force at ``gamma * mu * fn`` outside the
    radial return, and ``friction_release_dwell_s`` keeps the stick point alive through
    normal dropouts shorter than the dwell.
    """

    stretch_floor: float = 0.05
    normal_damping: float = 0.0
    friction_stiffness: float = 0.0
    friction: float = 0.0
    friction_stiffness_per_area: float = 0.0
    friction_damping_per_area: float = 0.0
    friction_viscous_ratio: float = 0.2
    friction_release_dwell_s: float = 0.0005
    mu: float = 0.0

    def __post_init__(self) -> None:
        negative = (
            self.friction_stiffness < 0.0
            or self.friction < 0.0
            or self.friction_stiffness_per_area < 0.0
            or self.friction_damping_per_area < 0.0
            or self.friction_viscous_ratio < 0.0
            or self.friction_release_dwell_s < 0.0
        )
        if negative:
            raise ValueError("friction stiffness, damping, viscous ratio and release dwell must be nonnegative")


@dataclass
class SurroundConfig:
    """Assumed relaxation settings for the columns the carrier does not drive.

    The identification relaxes the foam outside the indenter against neighbour
    shear and its own unilateral ground reaction (:func:`surround_sweep`). This
    config gives the live runtime the same surround, so one geometry and one
    contact model serve both.

    Args:
        driven: Nonzero where the carrier drives the column, shape
            ``[column_count]``. An all-driven mask makes the relaxation a no-op.
        attachment_n_m: Vertical bond of untouched foam to the shoe [N/m].
            Zero by default: its reaction never entered the reported force or
            the carrier wrench, so any nonzero value is an undeclared support.
        max_strain: Compression limit as a fraction of rest thickness.
        coupling_scale: Assumed lateral drive on the passive surface as a
            fraction of the material-pinned face coefficient ``mu_eq * t``.
        sweeps: Relaxation sweeps per substep. The field is warm started from the
            previous substep, so few sweeps track a converged quasi-static solve.
        relaxation_time_s: First-order lag toward the local balance [s]. Zero
            takes the full Newton step, which is the quasi-static solve the
            identification uses.
        carrier_bond: True when the untouched column tops are glued under the
            rigid carrier (a shod runtime) instead of being a free shoe surface
            the carrier never touches (the bench fixture). See
            :func:`_surround_balance`.
    """

    driven: np.ndarray
    attachment_n_m: float = 0.0
    max_strain: float = 0.9
    coupling_scale: float = 1.0
    sweeps: int = 8
    relaxation_time_s: float = 0.0
    carrier_bond: bool = False

    def __post_init__(self) -> None:
        self.driven = np.ascontiguousarray(np.asarray(self.driven) != 0, np.int32)
        if self.driven.ndim != 1 or not np.any(self.driven):
            raise ValueError("the surround needs a one-dimensional mask with at least one driven column")
        if self.attachment_n_m < 0.0 or not 0.0 < self.max_strain < 1.0:
            raise ValueError("surround bond stiffness must be nonnegative and its strain limit inside (0, 1)")
        if self.coupling_scale < 0.0 or self.sweeps < 1 or self.relaxation_time_s < 0.0:
            raise ValueError("surround coupling, sweeps, and relaxation time must be nonnegative and sweeps positive")


class MidsoleFoundation:
    """Live Warp elastic-foundation force model attached to one carrier body.

    Args:
        anchor_local: Column attachment points in the carrier body frame [m],
            shape ``[column_count, 3]``.
        z_free: World height of each uncompressed foam column top [m], shape
            ``[column_count]``.
        rest_len: Column rest thickness [m], shape ``[column_count]``.
        area: Tributary area per column [m^2], shape ``[column_count]``.
        neighbors: Pasternak 4-neighbour indices, shape ``[column_count, 4]``.
        spacing_m: Column grid spacing [m].
        material: Calibrated :class:`ShoeMaterial`.
        carrier_body: Index of the rigid body carrying the foundation.
        body_com: Model center-of-mass array (``model.body_com``).
        config: Dynamic :class:`FoundationConfig`.
        device: Warp device.
        surround: Optional :class:`SurroundConfig` letting the columns the
            carrier does not drive relax passively every substep, exactly as the
            identification relaxes them.
    """

    def __init__(
        self,
        anchor_local: np.ndarray,
        z_free: np.ndarray,
        rest_len: np.ndarray,
        area: np.ndarray,
        neighbors: np.ndarray,
        spacing_m: float,
        material: ShoeMaterial,
        carrier_body: int,
        body_com,
        config: FoundationConfig | None = None,
        device=None,
        surround: SurroundConfig | None = None,
    ) -> None:
        config = config or FoundationConfig()
        self.device = device
        self.carrier = int(carrier_body)
        self.body_com = body_com
        self.column_count = int(len(rest_len))

        params = FoundationParams()
        params.g_eq = material.instantaneous_shear_modulus_pa * material.equilibrium_fraction
        params.alpha = material.hyperfoam_exponent
        poisson = float(getattr(material, "effective_poisson_ratio", 0.0))
        params.beta = poisson / (1.0 - 2.0 * poisson)
        params.one_minus_two_poisson = 1.0 - 2.0 * poisson
        params.tau_s = float(getattr(material, "maxwell_relaxation_time_s", 0.08))
        params.overstress = (1.0 - material.equilibrium_fraction) / material.equilibrium_fraction
        params.inv_h2 = 1.0 / spacing_m**2
        params.stretch_floor = config.stretch_floor
        params.normal_damping = config.normal_damping
        params.friction_kt = config.friction_stiffness
        params.friction_kv = config.friction
        params.friction_viscous_ratio = config.friction_viscous_ratio
        params.friction_release_dwell_s = config.friction_release_dwell_s
        params.mu = config.mu
        self.params = params

        m = self.column_count
        # Tangential layer per unit area, so refining the column grid keeps the same patch
        # stiffness. Deriving from the mean area reproduces the per-column setting exactly
        # on a uniform grid.
        area_m2 = np.ascontiguousarray(area, np.float64).reshape(-1)
        mean_area = float(area_m2.mean())
        if mean_area <= 0.0:
            raise ValueError("column tributary areas must be positive")
        self.friction_stiffness_per_area_n_m3 = float(
            config.friction_stiffness_per_area or config.friction_stiffness / mean_area
        )
        self.friction_damping_per_area_n_s_m3 = float(config.friction_damping_per_area or config.friction / mean_area)
        self.friction_viscous_ratio = float(config.friction_viscous_ratio)
        self.friction_release_dwell_s = float(config.friction_release_dwell_s)
        self.anchor_local = wp.array(np.ascontiguousarray(anchor_local, np.float32), dtype=wp.vec3, device=device)
        self.z_free = wp.array(np.ascontiguousarray(z_free, np.float32), dtype=wp.float32, device=device)
        self.rest_len = wp.array(np.ascontiguousarray(rest_len, np.float32), dtype=wp.float32, device=device)
        self.area = wp.array(np.ascontiguousarray(area, np.float32), dtype=wp.float32, device=device)
        self.neighbors = wp.array(np.ascontiguousarray(neighbors, np.int32), dtype=wp.int32, device=device)
        self.q_state = wp.zeros(m, dtype=wp.float32, device=device)
        self.peq_prev = wp.zeros(m, dtype=wp.float32, device=device)
        self.compression = wp.zeros(m, dtype=wp.float32, device=device)
        self.base_pressure = wp.zeros(m, dtype=wp.float32, device=device)
        self.tangent_anchor = wp.zeros(m, dtype=wp.vec2, device=device)  # world XY stick point
        self.tangent_stuck = wp.zeros(m, dtype=wp.int32, device=device)  # 1 while the bristle grips
        self.tangent_dwell = wp.zeros(m, dtype=wp.float32, device=device)  # unloaded time held by a grip [s]
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
        self.normal_force = wp.zeros(1, dtype=wp.float32, device=device)
        self.cop_moment = wp.zeros(1, dtype=wp.vec3, device=device)
        self.active = wp.zeros(1, dtype=wp.int32, device=device)
        self.resultant_force = wp.zeros(1, dtype=wp.vec3, device=device)
        self.resultant_moment_origin = wp.zeros(1, dtype=wp.vec3, device=device)
        self.contact_power = wp.zeros(1, dtype=wp.float32, device=device)
        self.max_compression = wp.zeros(1, dtype=wp.float32, device=device)
        self.column_force = wp.zeros(m, dtype=wp.vec3, device=device)
        self.pressed_force = wp.zeros(1, dtype=wp.float32, device=device)

        self.surround = surround
        self.free_column_count = 0
        if surround is not None:
            if len(surround.driven) != m:
                raise ValueError("the surround mask must cover every column")
            self.free_column_count = int(m - int(surround.driven.sum()))
            # The rigid free top the carrier imposes. surround_write_free_top overwrites
            # z_free for the undriven columns every substep, so the rigid heights need
            # their own copy.
            self.z_free_rigid = wp.array(np.ascontiguousarray(z_free, np.float32), dtype=wp.float32, device=device)
            self.driven = wp.array(surround.driven, dtype=wp.int32, device=device)
            self.surround_compression = wp.zeros(m, dtype=wp.float32, device=device)
            self.surround_scratch = wp.zeros(m, dtype=wp.float32, device=device)
            self.surround_previous = wp.zeros(m, dtype=wp.float32, device=device)
            # Compression rate of the passive surface [m/s], a diagnostic of whether the
            # relaxation is still travelling when the substep ends.
            self.surround_rate = wp.zeros(m, dtype=wp.float32, device=device)

    def reset(self) -> None:
        """Clear the viscoelastic overstress history and release the friction bristles."""
        self.q_state.zero_()
        self.peq_prev.zero_()
        self.tangent_stuck.zero_()
        self.tangent_dwell.zero_()
        if self.free_column_count:
            self.surround_compression.zero_()
            self.surround_scratch.zero_()
            self.surround_previous.zero_()
            self.surround_rate.zero_()

    def relax_surround(self, state, dt: float) -> None:
        """Relax the columns the carrier does not drive for one substep.

        Sweeps :func:`surround_relax` over the warm-started compression field and
        publishes the result through :func:`surround_write_free_top`, so the
        pressure and wrench kernels see one bed with one contact law.
        """
        cfg = self.surround
        sweeps = int(cfg.sweeps)
        sub_dt = dt / sweeps
        tau = float(cfg.relaxation_time_s)
        relaxation = 1.0 if tau <= 0.0 else 1.0 - float(np.exp(-sub_dt / tau))
        # The Maxwell update foundation_pressure applies after this relaxation, so the
        # surround balances the load it is about to carry rather than the previous one.
        decay = float(np.exp(-dt / self.params.tau_s))
        gain = float(self.params.overstress * self.params.tau_s * (1.0 - decay) / dt)
        inputs = [
            self.carrier,
            state.body_q,
            self.driven,
            self.neighbors,
            self.anchor_local,
            self.z_free_rigid,
            self.rest_len,
            self.area,
            self.q_state,
            self.peq_prev,
            self.params,
            decay,
            gain,
            float(cfg.coupling_scale),
            float(cfg.attachment_n_m),
            float(cfg.max_strain),
            relaxation,
            int(bool(cfg.carrier_bond)),
        ]
        for _ in range(sweeps):
            wp.launch(
                surround_relax,
                dim=self.column_count,
                inputs=[*inputs, self.surround_compression, self.surround_scratch],
                device=self.device,
            )
            self.surround_compression, self.surround_scratch = self.surround_scratch, self.surround_compression
        wp.launch(
            surround_write_free_top,
            dim=self.column_count,
            inputs=[
                self.carrier,
                float(1.0 / dt),
                state.body_q,
                self.driven,
                self.anchor_local,
                self.z_free_rigid,
                self.surround_compression,
                self.surround_previous,
                self.z_free,
                self.surround_rate,
            ],
            device=self.device,
        )

    def apply(self, state, dt: float, clear_body_force: bool = False) -> None:
        """Accumulate the foundation wrench into ``state.body_f`` for one substep.

        Args:
            state: Simulation state supplying the carrier pose/velocity and receiving the wrench.
            dt: Substep duration [s].
            clear_body_force: Also zero the carrier's ``body_f`` in the fused reset launch, so a
                caller that only loads the foundation wrench can skip a separate
                :meth:`newton.State.clear_forces`. Leave False when other forces are staged into
                ``body_f`` before this call (e.g. an external probe load).
        """
        if self.free_column_count:
            self.relax_surround(state, dt)
        wp.launch(
            foundation_reset,
            dim=1,
            inputs=[
                self.carrier,
                int(clear_body_force),
                state.body_f,
                self.normal_force,
                self.cop_moment,
                self.active,
                self.resultant_force,
                self.resultant_moment_origin,
                self.contact_power,
                self.max_compression,
                self.pressed_force,
            ],
            device=self.device,
        )
        wp.launch(
            foundation_pressure,
            dim=self.column_count,
            inputs=[
                self.carrier,
                dt,
                state.body_q,
                self.anchor_local,
                self.z_free,
                self.rest_len,
                self.params,
                self.q_state,
                self.peq_prev,
                self.compression,
                self.base_pressure,
            ],
            device=self.device,
        )
        wp.launch(
            foundation_apply,
            dim=self.column_count,
            inputs=[
                self.carrier,
                dt,
                state.body_q,
                state.body_qd,
                self.body_com,
                self.anchor_local,
                self.area,
                self.rest_len,
                self.neighbors,
                self.compression,
                self.base_pressure,
                self.tangent_anchor,
                self.tangent_stuck,
                self.tangent_dwell,
                self.friction_kt,
                self.friction_kv,
                self.params,
                state.body_f,
                self.normal_force,
                self.cop_moment,
                self.active,
                self.resultant_force,
                self.resultant_moment_origin,
                self.contact_power,
                self.max_compression,
                self.column_force,
                self.pressed_force,
            ],
            device=self.device,
        )

    def diagnostics(self) -> dict[str, float]:
        """Return the last substep's total normal force, center of pressure, and active count."""
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
