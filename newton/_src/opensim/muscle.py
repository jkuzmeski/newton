# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp-native muscle-tendon force models ported from OpenSim.

This module provides differentiable, GPU-parallel implementations of the
characteristic curves and force computations for Hill-type muscle models:

- **De Groote-Fregly (2016)** — the smooth, twice-differentiable formulation
  used by :class:`DeGrooteFregly2016Muscle`. Ideal for Newton's gradient-based
  workflows (Moco-style optimal control, differentiable simulation).
- **Thelen (2003)** — the classic formulation used by :class:`Thelen2003Muscle`.

Curves are exposed as :func:`warp.func` device functions so they compose inside
Newton kernels. A rigid-tendon force evaluation (:func:`muscle_force_rigid_tendon`)
and first-order activation dynamics (:func:`activation_dot`) are provided as the
Newton-native equivalent of ``ignore_tendon_compliance=True`` muscles.

References:
    De Groote, F., Kinney, A.L., Rao, A.V., Fregly, B.J. (2016). Evaluation of
    Direct Collaborative Optimization... *Annals of Biomedical Engineering*.
    Thelen, D.G. (2003). Adjustment of Muscle Mechanics Model Parameters...
    *Journal of Biomechanical Engineering*.
"""

from __future__ import annotations

import warp as wp

# --- De Groote-Fregly (2016) default curve coefficients (OpenSim defaults) ---

# Active force-length: sum of three Gaussians, fal(l) = sum_i b1i*exp(-0.5*(l-b2i)^2/(b3i+b4i*l)^2)
_B11 = wp.constant(0.814483478343008)
_B21 = wp.constant(1.055033428970575)
_B31 = wp.constant(0.162384573599574)
_B41 = wp.constant(0.063303448465465)
_B12 = wp.constant(0.433004984392647)
_B22 = wp.constant(0.716775413397760)
_B32 = wp.constant(-0.029947116970696)
_B42 = wp.constant(0.200356847296188)
_B13 = wp.constant(0.100)
_B23 = wp.constant(1.000)
_B33 = wp.constant(0.354)
_B43 = wp.constant(0.000)

# Force-velocity: fv(v) = d1*asinh(d2*v + d3) + d4
_D1 = wp.constant(-0.318)
_D2 = wp.constant(-8.149)
_D3 = wp.constant(-0.374)
_D4 = wp.constant(0.886)

# Passive force-length: fpe(l) = (exp(kpe*(l-1)/e0) - 1)/(exp(kpe) - 1)
_KPE = wp.constant(4.0)
_E0 = wp.constant(0.6)


@wp.func
def dgf_active_force_length(l_norm: float) -> float:
    """De Groote-Fregly active force-length multiplier at normalized fiber length ``l_norm``."""
    g1 = _B11 * wp.exp(-0.5 * (l_norm - _B21) * (l_norm - _B21) / ((_B31 + _B41 * l_norm) * (_B31 + _B41 * l_norm)))
    g2 = _B12 * wp.exp(-0.5 * (l_norm - _B22) * (l_norm - _B22) / ((_B32 + _B42 * l_norm) * (_B32 + _B42 * l_norm)))
    g3 = _B13 * wp.exp(-0.5 * (l_norm - _B23) * (l_norm - _B23) / ((_B33 + _B43 * l_norm) * (_B33 + _B43 * l_norm)))
    return g1 + g2 + g3


@wp.func
def dgf_force_velocity(v_norm: float) -> float:
    """De Groote-Fregly force-velocity multiplier at normalized fiber velocity ``v_norm``.

    ``v_norm`` is in units of optimal fiber lengths per ``max_contraction_velocity``,
    i.e. -1 at maximal shortening and +1 at maximal lengthening.
    """
    arg = _D2 * v_norm + _D3
    return _D1 * wp.log(arg + wp.sqrt(arg * arg + 1.0)) + _D4


@wp.func
def dgf_force_velocity_inverse(fv: float) -> float:
    """Normalized fiber velocity for a De Groote-Fregly force-velocity multiplier ``fv``.

    Inverts :func:`dgf_force_velocity`; the multiplier is strictly increasing so the
    inverse is single-valued. Returns velocity in optimal-fiber-lengths per
    ``max_contraction_velocity`` (-1 maximal shortening, +1 maximal lengthening).
    """
    y = (fv - _D4) / _D1
    return (wp.sinh(y) - _D3) / _D2


@wp.func
def dgf_passive_force_length(l_norm: float) -> float:
    """De Groote-Fregly passive (parallel-elastic) force-length multiplier."""
    num = wp.exp(_KPE * (l_norm - 1.0) / _E0) - 1.0
    den = wp.exp(_KPE) - 1.0
    return wp.max(num / den, 0.0)


@wp.func
def dgf_tendon_force(lt_norm: float, kt: float) -> float:
    """De Groote-Fregly normalized tendon force at normalized tendon length ``lt_norm``.

    Args:
        lt_norm: Tendon length divided by ``tendon_slack_length``.
        kt: Tendon stiffness parameter (larger = stiffer). Related to
            ``tendon_strain_at_one_norm_force`` eT via ``kt = log(5)/(eT + 0.995 - 1)``.
    """
    c1 = 0.200
    c2 = 0.995
    c3 = 0.250
    return c1 * wp.exp(kt * (lt_norm - c2)) - c3


@wp.func
def tendon_stiffness_from_strain(strain_at_one_norm_force: float) -> float:
    """Convert ``tendon_strain_at_one_norm_force`` to the De Groote tendon stiffness ``kt``."""
    # fT(1 + eT) = 1 with c1=0.2, c2=0.995, c3=0.25 -> kt = ln((1+c3)/c1) / (1+eT-c2)
    return wp.log((1.0 + 0.250) / 0.200) / (1.0 + strain_at_one_norm_force - 0.995)


# --- Thelen (2003) curves ---


@wp.func
def thelen_active_force_length(l_norm: float, k_shape_active: float) -> float:
    """Thelen (2003) active force-length: Gaussian with shape factor ``k_shape_active``."""
    x = (l_norm - 1.0) * (l_norm - 1.0)
    return wp.exp(-x / k_shape_active)


@wp.func
def thelen_passive_force_length(l_norm: float, kpe: float, e0: float) -> float:
    """Thelen (2003) passive force-length with shape ``kpe`` and strain ``e0``."""
    if l_norm <= 1.0:
        return 0.0
    return (wp.exp(kpe * (l_norm - 1.0) / e0) - 1.0) / (wp.exp(kpe) - 1.0)


@wp.func
def thelen_force_velocity(v_norm: float, a: float, af: float, flen: float) -> float:
    """Thelen (2003) force-velocity multiplier.

    Args:
        v_norm: Normalized fiber velocity (optimal fiber lengths per max contraction velocity).
        a: Activation.
        af: Force-velocity shape factor ``Af``.
        flen: Maximum normalized lengthening force ``Flen``.
    """
    if v_norm <= 0.0:
        # concentric (shortening)
        return (1.0 + v_norm) / wp.max(1.0 - v_norm / af, 1.0e-6)
    # eccentric (lengthening)
    num = (2.0 + 2.0 / af) * (flen * v_norm + v_norm)
    den = flen * v_norm + (2.0 + 2.0 / af) * v_norm - flen + 1.0
    return flen - (flen - 1.0) * num / wp.max(den, 1.0e-6)


# --- Rigid-tendon force evaluation & activation dynamics ---


@wp.func
def pennated_fiber_length(lm_tendon: float, lt_slack: float, l_opt: float, cos_penn_opt: float) -> float:
    """Normalized fiber length for a rigid tendon.

    Given total muscle-tendon length ``lm_tendon``, tendon slack length ``lt_slack``,
    optimal fiber length ``l_opt`` and the cosine of the pennation angle at optimal
    length, returns the normalized fiber length ``lM / l_opt`` assuming a constant
    fiber width (parallelogram pennation model).
    """
    # Fiber width h = l_opt * sin(pennation_opt); rigid tendon: fiber along MT minus tendon.
    h = l_opt * wp.sqrt(wp.max(1.0 - cos_penn_opt * cos_penn_opt, 0.0))
    along = lm_tendon - lt_slack
    lm = wp.sqrt(wp.max(along * along + h * h, 1.0e-12))
    return lm / l_opt


@wp.func
def muscle_force_rigid_tendon(
    activation: float,
    lmt: float,
    vmt: float,
    fmax: float,
    l_opt: float,
    lt_slack: float,
    vmax: float,
    cos_penn_opt: float,
) -> float:
    """Rigid-tendon De Groote-Fregly muscle-tendon force [N].

    Args:
        activation: Muscle activation in [0, 1].
        lmt: Muscle-tendon (path) length [m].
        vmt: Muscle-tendon (path) lengthening velocity [m/s].
        fmax: Maximum isometric force [N].
        l_opt: Optimal fiber length [m].
        lt_slack: Tendon slack length [m].
        vmax: Maximum contraction velocity [optimal fiber lengths / s].
        cos_penn_opt: Cosine of pennation angle at optimal fiber length.

    Returns:
        Tendon (path) force magnitude [N], always >= 0.
    """
    l_norm = pennated_fiber_length(lmt, lt_slack, l_opt, cos_penn_opt)
    # Pennation angle at current length via constant-width assumption.
    lm = l_norm * l_opt
    cos_penn = wp.max((lmt - lt_slack) / wp.max(lm, 1.0e-9), 0.0)
    # Normalized fiber velocity (project MT velocity onto fiber, normalize).
    v_norm = (vmt * cos_penn) / wp.max(l_opt * vmax, 1.0e-9)

    fal = dgf_active_force_length(l_norm)
    fv = dgf_force_velocity(v_norm)
    fpe = dgf_passive_force_length(l_norm)

    fiber_force = fmax * (activation * fal * fv + fpe)
    return wp.max(fiber_force * cos_penn, 0.0)


@wp.func
def muscle_force_equilibrium_tendon(
    activation: float,
    lmt: float,
    fmax: float,
    l_opt: float,
    lt_slack: float,
    cos_penn_opt: float,
    kt: float,
) -> float:
    """Isometric equilibrium elastic-tendon De Groote-Fregly muscle force [N].

    Solves the series fiber-tendon force balance ``fT(lt) = (a fAL + fPE) cos(penn)``
    for the fiber length by bisection (the residual is monotonic in fiber length),
    with the tendon length set by the constant-width pennation geometry
    ``lt = lmt - lM cos(penn)``. Assumes zero fiber velocity (force-velocity = 1).

    Args:
        activation: Muscle activation in [0, 1].
        lmt: Muscle-tendon (path) length [m].
        fmax: Maximum isometric force [N].
        l_opt: Optimal fiber length [m].
        lt_slack: Tendon slack length [m].
        cos_penn_opt: Cosine of pennation angle at optimal fiber length.
        kt: De Groote-Fregly tendon stiffness (see :func:`tendon_stiffness_from_strain`).

    Returns:
        Tendon (path) force magnitude [N], always >= 0.
    """
    sin_penn_opt = wp.sqrt(wp.max(1.0 - cos_penn_opt * cos_penn_opt, 0.0))
    lo = sin_penn_opt + 1.0e-6
    hi = wp.sqrt((lmt / l_opt) * (lmt / l_opt) + sin_penn_opt * sin_penn_opt) - 1.0e-6
    for _i in range(50):
        mid = 0.5 * (lo + hi)
        along = l_opt * wp.sqrt(wp.max(mid * mid - sin_penn_opt * sin_penn_opt, 0.0))
        lt_norm = (lmt - along) / lt_slack
        cos_penn = along / wp.max(mid * l_opt, 1.0e-9)
        fiber_t = (activation * dgf_active_force_length(mid) + dgf_passive_force_length(mid)) * cos_penn
        if dgf_tendon_force(lt_norm, kt) - fiber_t > 0.0:
            lo = mid
        else:
            hi = mid
    mid = 0.5 * (lo + hi)
    along = l_opt * wp.sqrt(wp.max(mid * mid - sin_penn_opt * sin_penn_opt, 0.0))
    return fmax * wp.max(dgf_tendon_force((lmt - along) / lt_slack, kt), 0.0)


@wp.func
def activation_dot(activation: float, excitation: float, tau_act: float, tau_deact: float) -> float:
    """First-order muscle activation dynamics d(activation)/dt.

    Uses the smooth time constant of Thelen (2003): activation ramps up with
    ``tau_act`` when ``excitation > activation`` and down with ``tau_deact``.
    """
    diff = excitation - activation
    if diff > 0.0:
        tau = tau_act * (0.5 + 1.5 * activation)
    else:
        tau = tau_deact / (0.5 + 1.5 * activation)
    return diff / tau
