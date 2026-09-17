# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Friction-only consistent deflection law and tangent Jacobians for digital shoe contact.

This law tracks tangential deflection z directly, integrating velocity history
z_trial = z_old + dt * v (with z_old = 0 for a new contact), rather than tracking
an absolute world anchor point.
"""

import warp as wp

from projects.digital_shoe.contact import cone_viscous_scale


@wp.func
def bristle_deflection_step(
    velocity: wp.vec2,
    dt: float,
    normal: float,
    kt: float,
    kv: float,
    mu: float,
    viscous_ratio: float,
    release_dwell: float,
    yield_width: float,
    deflection: wp.vec2,
    stuck: int,
    dwell: float,
) -> tuple[wp.vec2, wp.mat22, wp.vec2, int, float]:
    """Advance one consistent-deflection Coulomb bristle and compute velocity Jacobian.

    Tangential deflection is tracked directly: z_trial = z_old + dt * velocity.
    For a newly active contact (stuck == 0), z_old is taken as zero.
    Elastic trial force is elastic = -kt * z_trial.
    Radial Coulomb projection caps the elastic magnitude at C = mu * normal,
    with an exact radial return when yield_width == 0.0.

    NOTE ON YIELD_WIDTH:
    Nonzero yield_width provides an experimental C1 smooth shoulder, but repeated
    evaluation at zero slip velocity (v=0) is mathematically non-idempotent:
    for z in the shoulder regime, g(r) < r causes spurious numerical relaxation/creep
    towards C*(1-w) whose physical rate depends on dt. Therefore, yield_width > 0
    is an unphysical numerical regularizer, retained only as an unsupported mathematical
    experiment. Qualified production simulation and sweeps must use yield_width=0.0.
    Next deflection satisfies next_deflection = -elastic / kt.

    Args:
        velocity: Tangential relative velocity [m/s].
        dt: Timestep duration [s].
        normal: Normal contact force magnitude [N] (>= 0).
        kt: Tangential bristle stiffness [N/m].
        kv: Tangential viscous damping [N·s/m].
        mu: Coulomb friction coefficient (>= 0).
        viscous_ratio: Maximum fraction of normal friction cone for viscous force.
        release_dwell: Unloaded dwell duration before release [s].
        yield_width: Shoulder width fraction in [0, 0.25].
        deflection: Current tangential deflection z [m].
        stuck: Bristle contact state (1 if active, 0 if free).
        dwell: Elapsed unloaded dwell time [s].

    Returns:
        tuple[wp.vec2, wp.mat22, wp.vec2, int, float]:
            - Tangential friction force [N]
            - Tangent Jacobian df/dvelocity [N·s/m]
            - Next deflection z [m]
            - Next stuck state
            - Next dwell time [s]
    """
    f_max = mu * normal
    I = wp.identity(2, float)
    zero_mat = wp.mat22(0.0, 0.0, 0.0, 0.0)
    zero_vec = wp.vec2(0.0, 0.0)

    # Normal separation / unloaded dwell logic
    if normal <= 0.0 or kt <= 0.0:
        next_dwell = dwell + dt
        if kt <= 0.0 or stuck == 0 or next_dwell > release_dwell:
            return zero_vec, zero_mat, zero_vec, 0, 0.0
        else:
            # Hold z only until release_dwell; no unloaded strain accumulation
            return zero_vec, zero_mat, deflection, stuck, next_dwell

    # Contact is active
    next_dwell = 0.0
    next_stuck = 1

    # z_old = 0 for newly active contact
    z_old = zero_vec
    if stuck != 0:
        z_old = deflection

    z_trial = z_old + velocity * dt
    trial_elastic = -kt * z_trial
    r = wp.length(trial_elastic)

    C = f_max
    w = wp.clamp(yield_width, 0.0, 0.25)

    elastic = trial_elastic
    next_z = z_trial
    dfe_dv = -kt * dt * I

    if C <= 0.0:
        elastic = zero_vec
        next_z = zero_vec
        dfe_dv = zero_mat
    elif r > 1.0e-12:
        u_e = trial_elastic / r
        P_e = wp.outer(u_e, u_e)

        if w <= 0.0:
            if r > C:
                elastic = u_e * C
                next_z = -elastic / kt
                # dfe/dv = C * (I - P_e) / r * (-kt * dt)
                dfe_dv = (I - P_e) * (C / r * (-kt * dt))
        else:
            # Yield shoulder
            r_lo = C * (1.0 - w)
            r_hi = C * (1.0 + w)
            if r <= r_lo:
                # Linear elastic regime
                pass
            elif r >= r_hi:
                # Saturated Coulomb regime
                elastic = u_e * C
                next_z = -elastic / kt
                dfe_dv = (I - P_e) * (C / r * (-kt * dt))
            else:
                # Quadratic shoulder regime:
                # g(r) = r - (r - r_lo)^2 / (4 * w * C)
                # g'(r) = 1 - (r - r_lo) / (2 * w * C)
                diff = r - r_lo
                g_val = r - (diff * diff) / (4.0 * w * C)
                g_prime = 1.0 - diff / (2.0 * w * C)
                elastic = u_e * g_val
                next_z = -elastic / kt
                # dfe/dv = [g'(r) * P_e + (g(r) / r) * (I - P_e)] * (-kt * dt)
                dfe_dv = (P_e * g_prime + (I - P_e) * (g_val / r)) * (-kt * dt)
    else:
        # r <= 1e-12: near origin
        pass

    force = elastic
    df_dv = dfe_dv

    # Viscous regularizer matching contact.bristle_step / friction_law.bristle_force_tangent
    speed = wp.length(velocity)
    if kv > 0.0 and speed > 1.0e-12:
        cap = viscous_ratio * f_max
        raw_viscous = kv * speed
        u_v = velocity / speed
        P_v = wp.outer(u_v, u_v)

        viscous = -velocity * (wp.min(raw_viscous, cap) / speed)
        dfv_dv = zero_mat

        if raw_viscous < cap:
            dfv_dv = -I * kv
        else:
            dfv_dv = (I - P_v) * (-cap / speed)

        eta = cone_viscous_scale(elastic, viscous, f_max)

        if eta <= 0.0:
            df_dv = dfe_dv
        elif eta >= 1.0:
            force = elastic + viscous
            df_dv = dfe_dv + dfv_dv
        else:
            # Interior cone scaling: force = elastic + viscous * eta
            force = elastic + viscous * eta
            df_trial = dfe_dv + dfv_dv * eta
            denom = wp.dot(force, viscous)
            if wp.abs(denom) > 1.0e-12:
                force_df_trial = wp.transpose(df_trial) * force
                deta = force_df_trial * (-1.0 / denom)
                df_dv = df_trial + wp.outer(viscous, deta)
            else:
                df_dv = df_trial

    return force, df_dv, next_z, next_stuck, next_dwell
