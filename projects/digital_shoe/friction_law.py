# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Friction laws and analytical tangent Jacobians for digital shoe contact.

Provides local Coulomb bristle friction and regularized Coulomb friction
along with their analytical derivatives with respect to tangential slip
velocity for implicit and semi-implicit friction solvers.
"""

import warp as wp

from projects.digital_shoe.contact import bristle_step, cone_viscous_scale


@wp.func
def _regularized_coulomb_scale(u_norm: float, eps_u: float) -> float:
    """Return a linearly regularized reciprocal-slip factor.

    A positive ``eps_u`` continues ``1/u`` below the threshold with matching
    value and slope; otherwise the exact reciprocal is used for positive slip.
    """
    if u_norm > eps_u and u_norm > 0.0:
        return 1.0 / u_norm
    if eps_u > 0.0:
        return (-u_norm / eps_u + 2.0) / eps_u
    return 0.0


@wp.func
def regularized_force_tangent(
    velocity: wp.vec2,
    dt: float,
    normal: float,
    mu: float,
    smoothing_speed: float,
) -> tuple[wp.vec2, wp.mat22]:
    """Evaluate regularized Coulomb friction force and its analytical velocity Jacobian.

    Uses the projected isotropic friction regularized scale matching VBD's
    formulation with slip displacement ``u = velocity * dt`` and smoothing distance
    ``eps_u = smoothing_speed * dt``.

    Args:
        velocity: Tangential relative velocity [m/s].
        dt: Timestep duration [s].
        normal: Normal contact force magnitude [N] (>= 0).
        mu: Coulomb friction coefficient (>= 0).
        smoothing_speed: Smoothing velocity threshold [m/s] (>= 0).

    Returns:
        tuple[wp.vec2, wp.mat22]: Tangential friction force [N] and Jacobian
        df/dvelocity [N·s/m].
    """
    if normal <= 0.0 or mu <= 0.0:
        return wp.vec2(0.0, 0.0), wp.mat22(0.0, 0.0, 0.0, 0.0)

    speed = wp.length(velocity)
    f_max = mu * normal

    if speed <= 0.0:
        if smoothing_speed > 0.0:
            diag = -2.0 * f_max / smoothing_speed
            return wp.vec2(0.0, 0.0), wp.mat22(diag, 0.0, 0.0, diag)
        return wp.vec2(0.0, 0.0), wp.mat22(0.0, 0.0, 0.0, 0.0)

    u_norm = speed * dt
    eps_u = smoothing_speed * dt
    scale = f_max * _regularized_coulomb_scale(u_norm, eps_u)
    force = -velocity * (scale * dt)

    u = velocity / speed
    P = wp.outer(u, u)
    I = wp.identity(2, float)

    if u_norm > eps_u:
        # Full sliding regime: scale = f_max / u_norm = f_max / (speed * dt)
        # force = -f_max * u
        # df/dv = -(f_max / speed) * (I - P)
        K = (I - P) * (-f_max / speed)
    else:
        # Regularized ramp regime:
        # g(speed) = scale * dt = f_max * (-speed / smoothing_speed + 2.0) / smoothing_speed
        # g'(speed) = -f_max / (smoothing_speed * smoothing_speed)
        # df/dv = -g(speed) * I - g'(speed) * (v outer u)
        #       = -g(speed) * I + (f_max * speed / (smoothing_speed * smoothing_speed)) * P
        g = scale * dt
        g_prime = 0.0
        if smoothing_speed > 0.0:
            g_prime = -f_max / (smoothing_speed * smoothing_speed)
        K = -I * g - P * (g_prime * speed)

    return force, K


@wp.func
def bristle_force_tangent(
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
) -> tuple[wp.vec2, wp.mat22, wp.vec2, int, float]:
    """Advance Coulomb bristle using canonical bristle_step and compute velocity Jacobian.

    Directly invokes :func:`contact.bristle_step` to preserve exact force and state
    updates. Computes the analytical tangent Jacobian df/dvelocity [N·s/m] along
    the active branch.

    Args:
        position: Current tangential position [m].
        velocity: Tangential relative velocity [m/s].
        dt: Timestep duration [s].
        normal: Normal contact force magnitude [N].
        kt: Bristle tangential stiffness [N/m].
        kv: Tangential viscous damping [N·s/m].
        mu: Coulomb friction coefficient.
        viscous_ratio: Maximum fraction of normal friction cone for viscous force.
        release_dwell: Unloaded dwell duration before anchor release [s].
        anchor: Bristle anchor position [m].
        stuck: Bristle contact state (1 if stuck/active, 0 if free).
        dwell: Elapsed unloaded dwell time [s].

    Returns:
        tuple[wp.vec2, wp.mat22, wp.vec2, int, float]:
            - Tangential friction force [N] from canonical bristle_step
            - Analytical tangent Jacobian df/dvelocity [N·s/m]
            - Next anchor position [m] from canonical bristle_step
            - Next stuck state from canonical bristle_step
            - Next dwell time [s] from canonical bristle_step
    """
    # Evaluate canonical bristle step for exact force and state
    force, next_anchor, next_stuck, next_dwell = bristle_step(
        position,
        velocity,
        dt,
        normal,
        kt,
        kv,
        mu,
        viscous_ratio,
        release_dwell,
        anchor,
        stuck,
        dwell,
    )

    if normal <= 0.0 or kt <= 0.0:
        return force, wp.mat22(0.0, 0.0, 0.0, 0.0), next_anchor, next_stuck, next_dwell

    f_max = mu * normal

    # Determine anchor used for elastic step (matching bristle_step logic)
    step_anchor = anchor
    if stuck == 0:
        step_anchor = position

    p_next = position + velocity * dt
    elastic = -kt * (p_next - step_anchor)
    mag = wp.length(elastic)

    I = wp.identity(2, float)
    dfe_dv = -kt * dt * I

    if mag > f_max:
        s = f_max / wp.max(mag, 1.0e-12)
        elastic = elastic * s
        u_e = elastic / wp.max(wp.length(elastic), 1.0e-12)
        dfe_dv = (I - wp.outer(u_e, u_e)) * (s * (-kt * dt))

    df_dv = dfe_dv

    speed = wp.length(velocity)
    if kv > 0.0 and speed > 1.0e-12:
        cap = viscous_ratio * f_max
        raw_viscous = kv * speed
        u_v = velocity / speed
        P_v = wp.outer(u_v, u_v)

        viscous = -velocity * (wp.min(raw_viscous, cap) / speed)
        dfv_dv = wp.mat22(0.0, 0.0, 0.0, 0.0)

        if raw_viscous < cap:
            dfv_dv = -I * kv
        else:
            dfv_dv = (I - P_v) * (-cap / speed)

        eta = cone_viscous_scale(elastic, viscous, f_max)

        if eta <= 0.0:
            df_dv = dfe_dv
        elif eta >= 1.0:
            df_dv = dfe_dv + dfv_dv
        else:
            # Interior cone scaling: differentiate constraint f dot df = 0
            df_trial = dfe_dv + dfv_dv * eta
            denom = wp.dot(force, viscous)
            if wp.abs(denom) > 1.0e-12:
                force_df_trial = wp.transpose(df_trial) * force
                deta = force_df_trial * (-1.0 / denom)
                df_dv = df_trial + wp.outer(viscous, deta)
            else:
                df_dv = df_trial

    return force, df_dv, next_anchor, next_stuck, next_dwell
