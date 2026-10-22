# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Sim context ``xi`` of the SuperMimic runner: shoe foam, outsole friction and shoe geometry.

``xi`` has one entry per :data:`XI_COORDINATES` coordinate, each mapped
affinely, or log-affinely, from its physical range onto ``[-1, 1]``. The
coordinates re-express the parameters of
:class:`~projects.digital_shoe.runtime.ShoeMaterial` so that each one moves a
distinct feature of the equilibrium curve ``p(e)`` of the two-term Hyperfoam
law, ``p = sum_n 2 g_n / (a_n s) (1 - s^a_n)`` with stretch ``s = 1 - e`` and a
zero effective Poisson ratio:

* ``log_modulus``: small-strain modulus ``2 g_1`` of the first term [Pa].
* ``log_plateau``: plateau ``2 g_1 / a_1`` that ``s p`` approaches as the
  first term saturates [Pa].
* ``densification_strain``: strain at which the leading power of the second
  (negative-exponent) term, ``2 g_2 / |a_2| s^(a_2 - 1)``, reaches
  :data:`DENSIFICATION_PRESSURE_PA` [-].
* ``densification_exponent``: second-term exponent ``a_2`` [-].
* ``equilibrium_fraction``: equilibrium share of the instantaneous modulus [-].
* ``log_relaxation_time``: Maxwell relaxation time [s].
* ``friction``: outsole-belt Coulomb coefficient [-].

The last :data:`GEOMETRY_COUNT` entries one-hot the column-bed geometry, in
the order of :data:`GEOMETRY_NAMES`, the shoe order of the dataset manifest.

``g_n`` are the equilibrium term moduli, instantaneous moduli times the
equilibrium fraction. The ranges were set to span the v2 fits of the Puma
Fast-R Nitro Elite 3 (FR3) and Deviate Nitro Elite 3 (Dev3) foams by about a
quarter of their difference on every side, and now serve as the unit of each
coordinate: the v3 refits lie outside them, FR3 at about 4 on
``densification_exponent`` and Dev3 at about 1.8 on ``log_plateau``.
:class:`ShoeSpace` samples training contexts in a ball around one fitted
shoe's context, on that shoe's geometry, and evaluation contexts at the fit
itself (held-out windows only) and on a shell outside the ball (held-out shoe).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from projects.digital_shoe.runtime import ShoeMaterial

__all__ = [
    "DENSIFICATION_PRESSURE_PA",
    "GEOMETRY_COUNT",
    "GEOMETRY_NAMES",
    "XI_COORDINATES",
    "XI_DIM",
    "ShoeSpace",
    "decode",
    "encode",
]

DENSIFICATION_PRESSURE_PA = 1.0e6
"""Pressure that defines the densification strain [Pa]."""

XI_COORDINATES = (
    ("log_modulus", 4.2e5, 1.05e6, True),
    ("log_plateau", 3.5e4, 5.7e4, True),
    ("densification_strain", 0.50, 0.75, False),
    ("densification_exponent", -21.0, -14.0, False),
    ("equilibrium_fraction", 0.76, 0.84, False),
    ("log_relaxation_time", 0.018, 0.042, True),
    ("friction", 0.4, 1.2, False),
)
"""Name, physical low and high bound and log scaling of each ``xi`` coordinate, in order."""

GEOMETRY_NAMES = ("puma-fast-r-nitro-elite-3", "puma-deviate-nitro-elite-3")
"""Shoe of each column-bed geometry, in one-hot order."""

GEOMETRY_COUNT = len(GEOMETRY_NAMES)

XI_DIM = len(XI_COORDINATES) + GEOMETRY_COUNT
_FRICTION = len(XI_COORDINATES) - 1
_GEOMETRY = len(XI_COORDINATES)
_LOG = np.array([c[3] for c in XI_COORDINATES])


def _scaled(physical: np.ndarray) -> np.ndarray:
    """Take the log of the log-scaled coordinates."""
    physical = np.array(physical, dtype=np.float64)
    physical[..., _LOG] = np.log(physical[..., _LOG])
    return physical


_LOW, _HIGH = (_scaled([c[i] for c in XI_COORDINATES]) for i in (1, 2))


def encode(material: ShoeMaterial, friction: float, geometry: int) -> np.ndarray:
    """Return the context of one material, friction coefficient and geometry index, shape ``[XI_DIM]``.

    Only the Ogden-Hill series, equilibrium fraction and relaxation time enter;
    the effective Poisson ratio is taken as zero, as in every fitted shoe.
    """
    f = material.equilibrium_fraction
    g1 = material.instantaneous_shear_modulus_pa * f
    g2 = material.instantaneous_shear_modulus_2_pa * f
    a1 = material.hyperfoam_exponent
    a2 = material.hyperfoam_exponent_2
    stretch = (DENSIFICATION_PRESSURE_PA * -a2 / (2.0 * g2)) ** (1.0 / (a2 - 1.0))
    physical = _scaled([2.0 * g1, 2.0 * g1 / a1, 1.0 - stretch, a2, f, material.maxwell_relaxation_time_s, friction])
    xi = np.zeros(XI_DIM, dtype=np.float32)
    xi[:_GEOMETRY] = 2.0 * (physical - _LOW) / (_HIGH - _LOW) - 1.0
    xi[_GEOMETRY + geometry] = 1.0
    return xi


def decode(xi: np.ndarray) -> tuple[list[ShoeMaterial], np.ndarray, np.ndarray]:
    """Return the material, friction coefficient and geometry of every context.

    Args:
        xi: Contexts, shape ``[count, XI_DIM]``.

    Returns:
        One :class:`~projects.digital_shoe.runtime.ShoeMaterial` per context,
        the friction coefficients, shape ``[count]``, and the geometry indices
        into :data:`GEOMETRY_NAMES`, shape ``[count]``. ``pasternak_n_per_m``
        is reported only and is set to zero.
    """
    xi = np.asarray(xi, dtype=np.float64)
    physical = _LOW + (xi[:, :_GEOMETRY] + 1.0) * 0.5 * (_HIGH - _LOW)
    physical[:, _LOG] = np.exp(physical[:, _LOG])
    materials = []
    for modulus, plateau, strain, a2, f, tau, _friction in physical:
        g2 = 0.5 * DENSIFICATION_PRESSURE_PA * -a2 * (1.0 - strain) ** (1.0 - a2)
        materials.append(
            ShoeMaterial(
                instantaneous_shear_modulus_pa=0.5 * modulus / f,
                hyperfoam_exponent=modulus / plateau,
                equilibrium_fraction=f,
                pasternak_n_per_m=0.0,
                maxwell_relaxation_time_s=tau,
                instantaneous_shear_modulus_2_pa=g2 / f,
                hyperfoam_exponent_2=a2,
            )
        )
    return materials, physical[:, _FRICTION], np.argmax(xi[:, _GEOMETRY:], axis=1)


@dataclass(frozen=True)
class ShoeSpace:
    """Training and evaluation contexts around one fitted shoe.

    Training contexts are uniform in a ball of ``radius`` around ``center``
    in the foam and friction coordinates, on the geometry of ``center``.
    ``center`` itself lies inside that ball, so it only holds out the
    reference windows it runs on; contexts on a shell of radii ``beyond``
    lie outside the ball and also hold out the shoe.
    """

    center: np.ndarray
    """Context of the fitted shoe, its nominal friction and its geometry, shape ``[XI_DIM]``."""
    radius: float = 0.5
    """Radius of the training ball in normalized foam and friction coordinates [-].

    The v3 FR3 fit starts within 10 % of the best loss spread up to 0.51 on
    the modulus, plateau, equilibrium-fraction and relaxation-time
    coordinates; the Instron sweep does not pin the densification pair.
    """
    beyond: tuple[float, float] = (0.75, 1.0)
    """Inner and outer radius of the shell of out-of-ball contexts [-]."""

    def sample_training(self, count: int, rng: np.random.Generator) -> np.ndarray:
        """Return ``count`` training contexts, shape ``[count, XI_DIM]``."""
        return self._sample(count, rng, 0.0, self.radius)

    def sample_center(self, count: int) -> np.ndarray:
        """Return ``count`` copies of ``center``, shape ``[count, XI_DIM]``."""
        return np.repeat(np.asarray(self.center, dtype=np.float32)[None], count, axis=0)

    def sample_beyond(self, count: int, rng: np.random.Generator) -> np.ndarray:
        """Return ``count`` contexts uniform in the shell of radii ``beyond``, shape ``[count, XI_DIM]``."""
        return self._sample(count, rng, *self.beyond)

    def _sample(self, count: int, rng: np.random.Generator, inner: float, outer: float) -> np.ndarray:
        direction = rng.standard_normal((count, _GEOMETRY))
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        u = rng.uniform((inner / outer) ** _GEOMETRY, 1.0, (count, 1))
        xi = self.sample_center(count)
        xi[:, :_GEOMETRY] += outer * u ** (1.0 / _GEOMETRY) * direction
        return xi
