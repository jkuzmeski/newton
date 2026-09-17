# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Friction-only consistent deflection law with speed-dependent Stribeck Coulomb cap.

This module implements an experimental physically-motivated extension for effective
fitting of shoe contact friction. The Coulomb friction cap varies with tangential
slip velocity according to an exponential Stribeck law:
    mu(v) = mu_dynamic + (mu_static - mu_dynamic) * exp(-|v|^2 / transition_speed^2)
with mu_static >= mu_dynamic >= 0 and transition_speed > 0.

Tangential deflection is tracked directly via consistent velocity integration:
    z_trial = z_old + dt * v (with z_old = 0 for a newly active contact)
and elastic force is capped by an exact hard radial return (no C1 shoulder):
    elastic = -kt * z_trial, capped at C(v) = normal * mu(v).

The step delegates state advance, radial Coulomb return, and base velocity Jacobian
to `bristle_deflection_step(..., mu=mu_eff, yield_width=0.0)` and adds the velocity
chain-rule term:
    Jac = Jac_fixed_mu + outer(dF/dC, dC/dv)
where dC/dv = Fn * dmu/dv and dF/dC accounts for the radial cap, capped viscous damping,
and the implicit cone-limiting scale factor eta.
"""

import warp as wp

from projects.digital_shoe.contact import cone_viscous_scale
from projects.digital_shoe.friction_deflection import bristle_deflection_step


@wp.func
def stribeck_coefficient(
    velocity: wp.vec2,
    mu_static: float,
    mu_dynamic: float,
    transition_speed: float,
) -> float:
    """Compute the speed-dependent friction coefficient mu(v).

    Args:
        velocity: Tangential relative velocity [m/s].
        mu_static: Static friction coefficient (>= mu_dynamic >= 0).
        mu_dynamic: Dynamic friction coefficient (>= 0).
        transition_speed: Characteristic velocity scale for Stribeck transition [m/s] (> 0).

    Returns:
        float: Coulomb friction coefficient mu(v).
    """
    v_sq = wp.dot(velocity, velocity)
    vs_sq = transition_speed * transition_speed
    exp_factor = wp.exp(-v_sq / vs_sq)
    return mu_dynamic + (mu_static - mu_dynamic) * exp_factor


@wp.func
def bristle_stribeck_step(
    velocity: wp.vec2,
    dt: float,
    normal: float,
    kt: float,
    kv: float,
    mu_static: float,
    mu_dynamic: float,
    transition_speed: float,
    viscous_ratio: float,
    release_dwell: float,
    deflection: wp.vec2,
    stuck: int,
    dwell: float,
) -> tuple[wp.vec2, wp.mat22, wp.vec2, int, float]:
    """Advance one consistent-deflection Coulomb bristle with Stribeck cap and analytic Jacobian.

    Evaluates effective friction coefficient mu_eff = mu(velocity) and delegates state
    advance, radial return, and base tangent Jacobian to `bristle_deflection_step` with
    yield_width=0.0. Adds the analytic chain-rule term outer(dF/dC, Fn * dmu/dv) to form
    the complete velocity Jacobian.

    Args:
        velocity: Tangential relative velocity [m/s].
        dt: Timestep duration [s].
        normal: Normal contact force magnitude [N] (>= 0).
        kt: Tangential bristle stiffness [N/m].
        kv: Tangential viscous damping [N·s/m].
        mu_static: Static friction coefficient (>= mu_dynamic >= 0).
        mu_dynamic: Dynamic friction coefficient (>= 0).
        transition_speed: Stribeck transition velocity scale [m/s] (> 0).
        viscous_ratio: Maximum fraction of normal friction cone for viscous force.
        release_dwell: Unloaded dwell duration before release [s].
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
    zero_vec = wp.vec2(0.0, 0.0)

    # Inactive or separating contact delegates directly
    if normal <= 0.0 or kt <= 0.0:
        return bristle_deflection_step(
            velocity,
            dt,
            normal,
            kt,
            kv,
            mu_static,
            viscous_ratio,
            release_dwell,
            0.0,
            deflection,
            stuck,
            dwell,
        )

    # Compute effective Stribeck friction coefficient
    mu_eff = stribeck_coefficient(velocity, mu_static, mu_dynamic, transition_speed)

    # Call bristle_deflection_step with yield_width=0.0 (exact hard radial return)
    force, J_fixed_mu, next_z, next_stuck, next_dwell = bristle_deflection_step(
        velocity,
        dt,
        normal,
        kt,
        kv,
        mu_eff,
        viscous_ratio,
        release_dwell,
        0.0,
        deflection,
        stuck,
        dwell,
    )

    # Velocity derivative of Coulomb cap C(v) = normal * mu_eff:
    # dmu/dv = (mu_static - mu_dynamic) * exp(-|v|^2 / vs^2) * (-2 * v / vs^2)
    # dC/dv = normal * dmu/dv = (C - normal * mu_dynamic) * (-2 * velocity / vs^2)
    vs_sq = transition_speed * transition_speed
    C = normal * mu_eff
    dC_dv = velocity * (-2.0 * (C - normal * mu_dynamic) / vs_sq)

    # Compute partial derivative dF/dC
    z_old = zero_vec
    if stuck != 0:
        z_old = deflection
    z_trial = z_old + velocity * dt
    trial_elastic = -kt * z_trial
    r = wp.length(trial_elastic)

    delastic_dC = zero_vec
    if r > C and r > 1.0e-12 and C > 0.0:
        u_e = trial_elastic / r
        delastic_dC = u_e

    dF_dC = delastic_dC

    speed = wp.length(velocity)
    if kv > 0.0 and speed > 1.0e-12 and C > 0.0:
        gamma = viscous_ratio
        cap = gamma * C
        raw_viscous = kv * speed
        u_v = velocity / speed

        viscous = -velocity * (wp.min(raw_viscous, cap) / speed)
        dviscous_dC = zero_vec
        if raw_viscous >= cap:
            dviscous_dC = -u_v * gamma

        # Evaluate elastic force for eta regime check
        elastic = trial_elastic
        if r > C and r > 1.0e-12:
            elastic = (trial_elastic / r) * C

        eta = cone_viscous_scale(elastic, viscous, C)

        if eta <= 0.0:
            dF_dC = delastic_dC
        elif eta >= 1.0:
            dF_dC = delastic_dC + dviscous_dC
        else:
            # 0 < eta < 1 (interior cone scaling where |F|^2 = C^2)
            # 2 * F . (dF_trial/dC + viscous * deta/dC) = 2 * C
            dF_trial_dC = delastic_dC + dviscous_dC * eta
            denom = wp.dot(force, viscous)
            if wp.abs(denom) > 1.0e-12:
                deta_dC = (C - wp.dot(force, dF_trial_dC)) / denom
                dF_dC = dF_trial_dC + viscous * deta_dC
            else:
                dF_dC = dF_trial_dC

    # Full velocity Jacobian including speed-dependent cap variation
    total_jacobian = J_fixed_mu + wp.outer(dF_dC, dC_dv)

    return force, total_jacobian, next_z, next_stuck, next_dwell
