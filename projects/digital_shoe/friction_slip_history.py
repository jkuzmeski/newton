# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Provisional slip-history weakening inspired by rubber cold/hot friction.

Persson, Rubber friction and tire dynamics (arXiv:1007.2713), Eq. (1),
blends cold/hot friction using accumulated sliding distance. This reduced
shoe hypothesis uses constant branch coefficients and an explicit estimate
of loaded plastic slip; it is not a temperature solver or tire-to-shoe calibration.
There is no output-force filter and no weakening at fixed zero slip.
"""

import warp as wp

from .friction_deflection import bristle_deflection_step


@wp.func
def slip_history_coefficient(distance: float, mu_cold: float, mu_hot: float, slip_scale_m: float) -> float:
    """Return coefficient from accumulated loaded sliding distance [m]."""
    return mu_hot + (mu_cold - mu_hot) * wp.exp(-wp.max(distance, 0.0) / wp.max(slip_scale_m, 1.0e-12))


@wp.func
def bristle_slip_history_step(
    velocity: wp.vec2,
    dt: float,
    normal: float,
    kt: float,
    kv: float,
    mu_cold: float,
    mu_hot: float,
    slip_scale_m: float,
    viscous_ratio: float,
    release_dwell: float,
    deflection: wp.vec2,
    stuck: int,
    dwell: float,
    sliding_distance: float,
) -> tuple[wp.vec2, wp.vec2, int, float, float]:
    """Advance a passive bristle with slip-history-dependent traction capacity.

    Distance advances only from estimated plastic slip driven by the supplied
    relative velocity under positive normal load. The old-cap return predicts
    slip; the updated coefficient then enters the canonical force/state step.
    This explicit split must be checked under timestep refinement. It deliberately
    omits flash-temperature calculation, cooling during sustained contact and
    speed-dependent cold/hot branches. Release resets the local contact exposure.

    Args:
        velocity: Relative tangential velocity [m/s].
        dt: Timestep [s].
        normal: Prescribed normal load [N].
        kt: Tangential stiffness [N/m].
        kv: Tangential damping [N s/m].
        mu_cold: Virgin-contact coefficient.
        mu_hot: Weakened coefficient, between zero and mu_cold.
        slip_scale_m: Characteristic loaded sliding distance [m].
        viscous_ratio: Fraction of capacity available to damping.
        release_dwell: Unloaded release delay [s].
        deflection: Incoming elastic displacement [m].
        stuck: Incoming contact flag.
        dwell: Incoming unloaded dwell [s].
        sliding_distance: Incoming local sliding exposure [m].

    Returns:
        Force [N], deflection [m], contact flag, dwell [s], sliding exposure [m].
    """
    distance = wp.max(sliding_distance, 0.0)
    if stuck == 0:
        distance = 0.0
    mu_old = slip_history_coefficient(distance, mu_cold, mu_hot, slip_scale_m)
    if normal > 0.0 and kt > 0.0:
        z_old = wp.vec2(0.0)
        if stuck != 0:
            z_old = deflection
        increment = velocity * dt
        trial_length = wp.length(z_old + increment)
        plastic = wp.max(trial_length - mu_old * normal / kt, 0.0)
        # Normal-load projection alone must not generate sliding exposure.
        distance += wp.min(plastic, wp.length(increment))
    mu = slip_history_coefficient(distance, mu_cold, mu_hot, slip_scale_m)
    force, _jac, z, contact, elapsed = bristle_deflection_step(
        velocity,
        dt,
        normal,
        kt,
        kv,
        mu,
        viscous_ratio,
        release_dwell,
        0.0,
        deflection,
        stuck,
        dwell,
    )
    if contact == 0:
        distance = 0.0
    return force, z, contact, elapsed, distance
