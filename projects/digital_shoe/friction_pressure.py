# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Provisional tangential pressure-dependent friction cap hypothesis.

This module implements a provisional tangential pressure-dependent cap hypothesis
for digital shoe contact.

Physical formulation:
    p = max(Fn, 0) / area
    mu_eff = mu_low / (1 + p / pressure_scale_pa)
    C = mu_eff * Fn

Effective tangential shear-traction capacity:
    tau_max = C / area = mu_eff * p = mu_low * p / (1 + p / pressure_scale_pa)
As p -> inf, tau_max asymptotically approaches the finite saturation limit:
    tau_sat = mu_low * pressure_scale_pa

IMPORTANT SCOPE AND MODELING CONTRACT:
This law is solely an effective finite shear-traction capacity hypothesis for the
tangential contact response. It is NOT a modification to the normal pressure law
or normal contact mechanics (which remain governed by the elastic foundation /
Ogden-Hill formulation), and it is NOT a calibrated material property claim.
Normal force is held fixed during the tangential step, so mu_eff is independent
of tangential slip velocity v and requires no velocity chain derivatives.

Host parameter validation belongs in root adapters or host callers. Warp functions
use non-throwing safe clamps against invalid inputs (e.g. non-positive area or
pressure_scale_pa) to prevent NaNs during kernel execution.
"""

from __future__ import annotations

import math

import warp as wp

from projects.digital_shoe.friction_deflection import bristle_deflection_step


def validate_pressure_parameters(
    area: float,
    pressure_scale_pa: float,
    mu_low: float,
) -> None:
    """Validate pressure-dependent friction parameters on host.

    Args:
        area: Contact patch area [m²].
        pressure_scale_pa: Characteristic pressure scale [Pa].
        mu_low: Low-pressure asymptotic friction coefficient (dimensionless).

    Raises:
        ValueError: If area <= 0, pressure_scale_pa <= 0, or mu_low < 0.
    """
    if not math.isfinite(area) or area <= 0.0:
        raise ValueError(f"Contact patch area must be positive [m²], got {area}")
    if not math.isfinite(pressure_scale_pa) or pressure_scale_pa <= 0.0:
        raise ValueError(f"Pressure scale must be positive [Pa], got {pressure_scale_pa}")
    if not math.isfinite(mu_low) or mu_low < 0.0:
        raise ValueError(f"Friction coefficient mu_low must be non-negative, got {mu_low}")


@wp.func
def pressure_coefficient(
    normal: float,
    area: float,
    mu_low: float,
    pressure_scale_pa: float,
) -> float:
    """Evaluate provisional effective tangential friction coefficient under pressure-dependent cap.

    p = max(Fn, 0) / area
    mu_eff = mu_low / (1 + p / pressure_scale_pa)

    Args:
        normal: Normal contact force magnitude [N].
        area: Contact patch area [m²].
        mu_low: Low-pressure asymptotic friction coefficient (dimensionless, >= 0).
        pressure_scale_pa: Characteristic pressure scale [Pa] (> 0).

    Returns:
        Effective friction coefficient mu_eff (dimensionless, >= 0).
    """
    safe_area = wp.max(area, 1.0e-12)
    safe_pstar = wp.max(pressure_scale_pa, 1.0e-12)
    safe_mu = wp.max(mu_low, 0.0)

    p = wp.max(normal, 0.0) / safe_area
    return safe_mu / (1.0 + p / safe_pstar)


@wp.func
def bristle_pressure_step(
    velocity: wp.vec2,
    dt: float,
    normal: float,
    area: float,
    kt: float,
    kv: float,
    mu_low: float,
    pressure_scale_pa: float,
    viscous_ratio: float,
    release_dwell: float,
    deflection: wp.vec2,
    stuck: int,
    dwell: float,
) -> tuple[wp.vec2, wp.mat22, wp.vec2, int, float]:
    """Advance one consistent-deflection bristle under pressure-dependent tangential cap.

    Evaluates mu_eff = pressure_coefficient(normal, area, mu_low, pressure_scale_pa)
    and advances tangential state via canonical bristle_deflection_step with yield_width=0.0.
    With normal force held fixed during the tangential step, mu_eff is velocity-independent
    and df/dv equals the canonical bristle velocity Jacobian evaluated at mu_eff.

    Args:
        velocity: Tangential relative slip velocity [m/s].
        dt: Timestep duration [s].
        normal: Normal contact force magnitude [N].
        area: Contact patch area [m²].
        kt: Tangential bristle stiffness [N/m].
        kv: Tangential viscous damping [N·s/m].
        mu_low: Low-pressure asymptotic friction coefficient (dimensionless, >= 0).
        pressure_scale_pa: Characteristic pressure scale [Pa] (> 0).
        viscous_ratio: Maximum fraction of tangential capacity for viscous damping.
        release_dwell: Unloaded dwell duration before anchor release [s].
        deflection: Current tangential deflection z [m].
        stuck: Bristle contact state (1 if active, 0 if free).
        dwell: Elapsed unloaded dwell time [s].

    Returns:
        tuple[wp.vec2, wp.mat22, wp.vec2, int, float]:
            - Tangential friction force [N]
            - Tangent Jacobian df/dvelocity [N·s/m]
            - Next tangential deflection z [m]
            - Next stuck state (1 if active, 0 if free)
            - Next unloaded dwell time [s]
    """
    mu_eff = pressure_coefficient(normal, area, mu_low, pressure_scale_pa)
    return bristle_deflection_step(
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
