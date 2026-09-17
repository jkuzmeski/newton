# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared scalar contact mechanics for forward, fitting and tape-safe adapters.

Adapters choose boundary coordinates and own state storage. These functions own
unilateral reaction, lateral coupling, friction and force-to-wrench mechanics.
Autodiff uses the same piecewise bristle law; it does not substitute smooth
Coulomb friction for the forward model's static contact.
"""

from types import SimpleNamespace
from typing import Any

import numpy as np
import warp as wp

from .material import hyperfoam_pressure


def _pasternak_coupling(t_i: Any, t_j: Any, mu_eq: Any):
    """Return symmetric material-pinned shear conductance [N/m]."""
    return mu_eq * 0.5 * (t_i + t_j)


pasternak_coupling = wp.func(_pasternak_coupling)
pasternak_coupling_numpy = _pasternak_coupling


def _normal_reaction_function(ops, decorate):
    """Bind one unilateral contact expression to scalar or vectorized operations."""

    @decorate
    def normal_reaction(
        compression: Any,
        pressure: Any,
        area: Any,
        damping: Any,
        velocity_z: Any,
        gap: Any,
        ground_plane: int,
    ):
        """Return unilateral support [N], excluding internal neighbor transfer.

        ``gap`` is nominal bottom clearance [m] for an explicitly declared plane.
        A bench adapter instead supplies its imposed top compression.
        """
        reaction = ops.max(pressure, 0.0) * area
        reaction = reaction - ops.where(compression > 0.0, damping * velocity_z, 0.0)
        reaction = ops.max(reaction, 0.0)
        if ground_plane != 0:
            reaction = ops.where(gap > 0.0, 0.0, reaction)
        return reaction

    return normal_reaction


normal_reaction = _normal_reaction_function(wp, wp.func)
normal_reaction_numpy = _normal_reaction_function(
    SimpleNamespace(max=np.maximum, where=np.where), lambda function: function
)


@wp.func
def cone_viscous_scale(f_elastic: wp.vec2, f_viscous: wp.vec2, f_max: wp.float32) -> wp.float32:
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


@wp.func
def bristle_step(
    position: wp.vec2,
    velocity: wp.vec2,
    dt: float,
    normal: float,
    kt: float,
    kv: float,
    mu: float,
    viscous_ratio: float,
    release_dwell: float,
    anchor: wp.vec2,
    stuck: int,
    dwell: float,
) -> tuple[wp.vec2, wp.vec2, int, float]:
    """Advance one anchored Coulomb bristle without mutating caller-owned state.

    Positions and anchor are [m], velocity [m/s], time and dwell [s], normal
    force [N], stiffness [N/m], damping [N·s/m]. Return tangential force [N],
    new anchor, grip flag and unloaded dwell. Float state can live in separate
    timestep buffers for autodiff; branches have the same meaning in both paths.
    """
    f_max = mu * normal
    force = wp.vec2(0.0, 0.0)
    next_anchor = anchor
    next_stuck = stuck
    next_dwell = dwell
    if normal <= 0.0 or kt <= 0.0:
        next_dwell = dwell + dt
        if kt <= 0.0 or stuck == 0 or next_dwell > release_dwell:
            next_anchor = position
            next_stuck = 0
            next_dwell = 0.0
    else:
        next_dwell = 0.0
        if stuck == 0:
            next_anchor = position
            next_stuck = 1
        # Implicit elastic trial avoids ringing in the stiff stick regime.
        p_next = position + velocity * dt
        elastic = -kt * (p_next - next_anchor)
        mag = wp.length(elastic)
        if mag > f_max:
            elastic = elastic * (f_max / wp.max(mag, 1.0e-12))
            next_anchor = p_next + elastic / kt
        force = elastic
        speed = wp.length(velocity)
        if kv > 0.0 and speed > 1.0e-12:
            viscous = -velocity * (wp.min(kv * speed, viscous_ratio * f_max) / speed)
            force = elastic + viscous * cone_viscous_scale(elastic, viscous, f_max)
    return force, next_anchor, next_stuck, next_dwell


@wp.func
def contact_wrench(
    point: wp.vec3,
    force: wp.vec3,
    com_world: wp.vec3,
    velocity: wp.spatial_vector,
) -> tuple[wp.vec3, wp.vec3, float]:
    """Return COM torque [N·m], world-origin moment [N·m], and rigid power [W]."""
    arm = point - com_world
    point_velocity = wp.spatial_top(velocity) + wp.cross(wp.spatial_bottom(velocity), arm)
    return wp.cross(arm, force), wp.cross(point, force), wp.dot(force, point_velocity)


@wp.func
def _surround_balance_pressures(
    c: float,
    rigid: float,
    pull: float,
    coupling_sum: float,
    thickness: float,
    overstress_base: float,
    overstress_gain: float,
    area: float,
    attachment: float,
    max_strain: float,
    relaxation: float,
    carrier_bond: int,
    peq: float,
    peq_ahead: float,
) -> float:
    """Apply the shared constrained balance to already evaluated pressures."""
    reaction = area * wp.max(peq + overstress_base + overstress_gain * peq, 0.0)
    step = 1.0e-3 * thickness
    ahead = area * wp.max(peq_ahead + overstress_base + overstress_gain * peq_ahead, 0.0)
    stiffness = wp.max((ahead - reaction) / step + attachment + coupling_sum, 1.0e-9)
    bond_reference = float(0.0)
    upper = max_strain * thickness
    if carrier_bond != 0:
        bond_reference = rigid
        upper = wp.clamp(rigid, 0.0, upper)
    residual = reaction + attachment * (c - bond_reference) - pull
    return wp.clamp(c - relaxation * residual / stiffness, 0.0, upper)


@wp.func
def surround_balance(
    c: float,
    rigid: float,
    pull: float,
    coupling_sum: float,
    thickness: float,
    overstress_base: float,
    overstress_gain: float,
    g_eq: float,
    alpha: float,
    g_eq2: float,
    alpha2: float,
    beta: float,
    one_minus_two_poisson: float,
    stretch_floor: float,
    area: float,
    attachment: float,
    max_strain: float,
    relaxation: float,
    carrier_bond: int,
) -> float:
    """Take one constrained passive-column Newton step [m] with the shared material.

    Material scalars can originate from a constant block or differentiable
    arrays. The balance and one-sided retention bound do not depend on storage.
    """
    peq = hyperfoam_pressure(c / thickness, g_eq, alpha, g_eq2, alpha2, beta, one_minus_two_poisson, stretch_floor)
    step = 1.0e-3 * thickness
    peq_ahead = hyperfoam_pressure(
        (c + step) / thickness, g_eq, alpha, g_eq2, alpha2, beta, one_minus_two_poisson, stretch_floor
    )
    return _surround_balance_pressures(
        c,
        rigid,
        pull,
        coupling_sum,
        thickness,
        overstress_base,
        overstress_gain,
        area,
        attachment,
        max_strain,
        relaxation,
        carrier_bond,
        peq,
        peq_ahead,
    )


@wp.func
def contact_kinematics(
    transform: wp.transform,
    velocity: wp.spatial_vector,
    com_local: wp.vec3,
    anchor_local: wp.vec3,
    ground_height: float,
    ground_plane: int,
) -> tuple[wp.vec3, wp.vec3, wp.vec3, float]:
    """Return contact point, COM, point velocity and nominal plane clearance.

    Points and clearance are [m], velocity [m/s]. A bench top anchor remains a
    top anchor; a carried outsole uses its projection onto the declared plane.
    """
    world = wp.transform_point(transform, anchor_local)
    com = wp.transform_point(transform, com_local)
    point = world
    if ground_plane != 0:
        point = wp.vec3(world[0], world[1], ground_height)
    point_velocity = wp.spatial_top(velocity) + wp.cross(wp.spatial_bottom(velocity), point - com)
    return point, com, point_velocity, world[2] - ground_height


@wp.func
def pasternak_flux(
    column: int,
    base: int,
    compression: wp.array[float],
    rest_len: wp.array[float],
    neighbors: wp.array2d[int],
    mu_eq: float,
) -> float:
    """Return neighbor shear transfer [N] inside one world's compression tile."""
    ci = compression[base + column]
    flux = float(0.0)
    for side in range(4):
        j = neighbors[column, side]
        if j >= 0:
            flux += pasternak_coupling(rest_len[column], rest_len[j], mu_eq) * (compression[base + j] - ci)
    return flux
