# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Sim context ``xi`` of the SuperMimic runner: the shoe foam and outsole friction.

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

``g_n`` are the equilibrium term moduli, instantaneous moduli times the
equilibrium fraction. The ranges span the fitted Puma Fast-R Nitro Elite 3
(FR3) and Deviate Nitro Elite 3 (Dev3) foams with margin on every side.
:class:`ShoeSpace` samples training contexts uniformly in the box minus a
held-out ball around one shoe, Dev3 in the plan, in the six foam coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from projects.digital_shoe.runtime import ShoeMaterial

__all__ = [
    "DENSIFICATION_PRESSURE_PA",
    "XI_COORDINATES",
    "XI_DIM",
    "ShoeSpace",
    "decode",
    "encode",
]

DENSIFICATION_PRESSURE_PA = 1.0e6
"""Pressure that defines the densification strain [Pa]."""

XI_COORDINATES = (
    ("log_modulus", 3.0e5, 1.5e6, True),
    ("log_plateau", 2.5e4, 8.0e4, True),
    ("densification_strain", 0.45, 0.80, False),
    ("densification_exponent", -25.0, -10.0, False),
    ("equilibrium_fraction", 0.65, 0.95, False),
    ("log_relaxation_time", 0.01, 0.08, True),
    ("friction", 0.4, 1.2, False),
)
"""Name, physical low and high bound and log scaling of each ``xi`` coordinate, in order."""

XI_DIM = len(XI_COORDINATES)
_FOAM = XI_DIM - 1
_LOG = np.array([c[3] for c in XI_COORDINATES])


def _scaled(physical: np.ndarray) -> np.ndarray:
    """Take the log of the log-scaled coordinates."""
    physical = np.array(physical, dtype=np.float64)
    physical[..., _LOG] = np.log(physical[..., _LOG])
    return physical


_LOW, _HIGH = (_scaled([c[i] for c in XI_COORDINATES]) for i in (1, 2))


def encode(material: ShoeMaterial, friction: float) -> np.ndarray:
    """Return the context of one material and friction coefficient, shape ``[XI_DIM]``.

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
    return (2.0 * (physical - _LOW) / (_HIGH - _LOW) - 1.0).astype(np.float32)


def decode(xi: np.ndarray) -> tuple[list[ShoeMaterial], np.ndarray]:
    """Return the material and friction coefficient of every context.

    Args:
        xi: Contexts, shape ``[count, XI_DIM]``.

    Returns:
        One :class:`~projects.digital_shoe.runtime.ShoeMaterial` per context,
        and the friction coefficients, shape ``[count]``. ``pasternak_n_per_m``
        is reported only and is set to zero.
    """
    xi = np.asarray(xi, dtype=np.float64)
    physical = _LOW + (xi + 1.0) * 0.5 * (_HIGH - _LOW)
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
    return materials, physical[:, _FOAM]


@dataclass(frozen=True)
class ShoeSpace:
    """Training and held-out sampling of contexts.

    Training contexts are uniform in ``[-1, 1]^XI_DIM`` outside a ball of
    ``held_out_radius`` around ``held_out`` in the foam coordinates; friction is
    sampled over its whole range in both sets.
    """

    held_out: np.ndarray
    """Context at the center of the held-out ball, shape ``[XI_DIM]``; its friction entry is unused."""
    held_out_radius: float = 0.4
    """Radius of the held-out ball in normalized foam coordinates [-]."""

    def is_held_out(self, xi: np.ndarray) -> np.ndarray:
        """Return whether each context, shape ``[count, XI_DIM]``, lies in the held-out ball."""
        offset = np.asarray(xi)[:, :_FOAM] - np.asarray(self.held_out)[:_FOAM]
        return np.linalg.norm(offset, axis=1) < self.held_out_radius

    def sample_training(self, count: int, rng: np.random.Generator) -> np.ndarray:
        """Return ``count`` training contexts, shape ``[count, XI_DIM]``."""
        xi = np.empty((0, XI_DIM), dtype=np.float32)
        while len(xi) < count:
            draw = rng.uniform(-1.0, 1.0, (count, XI_DIM)).astype(np.float32)
            xi = np.concatenate([xi, draw[~self.is_held_out(draw)]])
        return xi[:count]

    def sample_held_out(self, count: int, rng: np.random.Generator) -> np.ndarray:
        """Return ``count`` contexts uniform in the held-out ball, shape ``[count, XI_DIM]``."""
        direction = rng.standard_normal((count, _FOAM))
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        radius = self.held_out_radius * rng.uniform(0.0, 1.0, (count, 1)) ** (1.0 / _FOAM)
        foam = np.asarray(self.held_out)[:_FOAM] + radius * direction
        friction = rng.uniform(-1.0, 1.0, (count, 1))
        return np.concatenate([foam, friction], axis=1).astype(np.float32)
