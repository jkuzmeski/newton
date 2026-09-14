# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared Ogden-Hill and Maxwell laws for simulation and identification.

Each constitutive expression is defined once in :func:`_material_functions`.
Warp compiles that source for the live and differentiable kernels. NumPy runs
that same source with vectorized elementary operations, retaining float64 host
fitting without device transfers. The backend changes only ``abs``, ``log``,
``max`` and ``exp``; it does not supply another material formula.

Device functions accept scalar material arguments, so Warp Tape can trace both
material and state gradients. Names ending in ``_numpy`` accept arrays for
strain, pressure and time, with scalar term exponents. They do not initialize a
Warp device. Material dataclasses and their existing import paths stay unchanged.
"""

from types import SimpleNamespace
from typing import Any

import numpy as np
import warp as wp

# Preserve the existing removable-singularity branch, including its zero alpha
# derivative inside this narrow interval. It is a limit approximation, not an
# exactly equal continuation at the cutoff.
HYPERFOAM_ALPHA_FLOOR = 1.0e-3


def _material_functions(ops, decorate):
    """Bind one set of expressions to elementary operations and compilation."""

    @decorate
    def ogden_hill_term(stretch: Any, volume_ratio: Any, mu: Any, alpha: Any, beta: Any):
        """Return one Ogden-Hill compression term [Pa].

        Args:
            stretch: Positive remaining thickness stretch [-].
            volume_ratio: Positive volume ratio ``J`` [-].
            mu: Equilibrium term shear modulus [Pa].
            alpha: Term exponent [-]; negative exponents permit densification.
            beta: Volumetric coefficient ``nu / (1 - 2 nu)`` [-].
        """
        if ops.abs(alpha) < HYPERFOAM_ALPHA_FLOOR:
            return 2.0 * mu / stretch * (-beta * ops.log(volume_ratio) - ops.log(stretch))
        return 2.0 * mu / (alpha * stretch) * (volume_ratio ** (-alpha * beta) - stretch**alpha)

    @decorate
    def hyperfoam_pressure_from_stretch(
        stretch: Any,
        g_eq: Any,
        alpha: Any,
        g_eq2: Any,
        alpha2: Any,
        beta: Any,
        one_minus_two_poisson: Any,
    ):
        """Return two-term equilibrium pressure for a positive floored stretch [Pa].

        This entry point lets a caller keep its own unilateral boundary and
        stretch clipping without allocating a second host strain array. The
        term sum and volumetric law remain shared with :func:`hyperfoam_pressure`.

        Args:
            stretch: Positive remaining thickness stretch [-], already floored.
            g_eq: First equilibrium term shear modulus [Pa].
            alpha: First term exponent [-].
            g_eq2: Second equilibrium term shear modulus [Pa].
            alpha2: Second term exponent [-].
            beta: Volumetric coefficient ``nu / (1 - 2 nu)`` [-].
            one_minus_two_poisson: Volumetric stretch exponent ``1 - 2 nu`` [-].
        """
        volume_ratio = stretch**one_minus_two_poisson
        return ogden_hill_term(stretch, volume_ratio, g_eq, alpha, beta) + ogden_hill_term(
            stretch, volume_ratio, g_eq2, alpha2, beta
        )

    @decorate
    def hyperfoam_pressure(
        strain: Any,
        g_eq: Any,
        alpha: Any,
        g_eq2: Any,
        alpha2: Any,
        beta: Any,
        one_minus_two_poisson: Any,
        stretch_floor: Any,
    ):
        """Return the two-term equilibrium compression pressure [Pa].

        Args:
            strain: Compression divided by rest thickness [-]. Callers enforce
                their own compression-only boundary before evaluating the law.
            g_eq: First equilibrium term shear modulus [Pa].
            alpha: First term exponent [-].
            g_eq2: Second equilibrium term shear modulus [Pa]; zero disables it.
            alpha2: Second term exponent [-].
            beta: Volumetric coefficient ``nu / (1 - 2 nu)`` [-].
            one_minus_two_poisson: Volumetric stretch exponent ``1 - 2 nu`` [-].
            stretch_floor: Positive lower stretch limit [-]. There is no upper
                clamp here; unilateral contact is a caller boundary condition.
        """
        stretch = ops.max(1.0 - strain, stretch_floor)
        return hyperfoam_pressure_from_stretch(stretch, g_eq, alpha, g_eq2, alpha2, beta, one_minus_two_poisson)

    @decorate
    def maxwell_coefficients(dt: Any, tau: Any):
        """Return exact exponential decay and linear-pressure ramp weights [-].

        Args:
            dt: Positive time increment [s].
            tau: Positive Maxwell relaxation time [s].
        """
        decay = ops.exp(-dt / tau)
        ramp = tau * (1.0 - decay) / dt
        return decay, ramp

    @decorate
    def maxwell_increment_step(q: Any, pressure_increment: Any, overstress: Any, decay: Any, ramp: Any):
        """Return Maxwell overstress after a linear pressure increment [Pa].

        Args:
            q: Previous branch overstress [Pa].
            pressure_increment: Current minus previous equilibrium pressure [Pa].
            overstress: Branch modulus ratio ``(1 - fraction) / fraction`` [-].
            decay: Exponential decay from :func:`maxwell_coefficients` [-].
            ramp: Linear-pressure ramp weight from :func:`maxwell_coefficients` [-].
        """
        return decay * q + overstress * ramp * pressure_increment

    @decorate
    def maxwell_step(q: Any, peq: Any, peq_prev: Any, overstress: Any, decay: Any, ramp: Any):
        """Return Maxwell overstress from consecutive equilibrium pressures [Pa].

        Args:
            q: Previous branch overstress [Pa].
            peq: Current equilibrium pressure [Pa].
            peq_prev: Previous equilibrium pressure [Pa].
            overstress: Branch modulus ratio ``(1 - fraction) / fraction`` [-].
            decay: Exponential decay from :func:`maxwell_coefficients` [-].
            ramp: Linear-pressure ramp weight from :func:`maxwell_coefficients` [-].
        """
        return maxwell_increment_step(q, peq - peq_prev, overstress, decay, ramp)

    return (
        ogden_hill_term,
        hyperfoam_pressure_from_stretch,
        hyperfoam_pressure,
        maxwell_coefficients,
        maxwell_increment_step,
        maxwell_step,
    )


(
    ogden_hill_term,
    hyperfoam_pressure_from_stretch,
    hyperfoam_pressure,
    maxwell_coefficients,
    maxwell_increment_step,
    maxwell_step,
) = _material_functions(wp, wp.func)

(
    ogden_hill_term_numpy,
    hyperfoam_pressure_from_stretch_numpy,
    hyperfoam_pressure_numpy,
    maxwell_coefficients_numpy,
    maxwell_increment_step_numpy,
    maxwell_step_numpy,
) = _material_functions(
    SimpleNamespace(abs=np.abs, log=np.log, max=np.maximum, exp=np.exp),
    lambda function: function,
)
