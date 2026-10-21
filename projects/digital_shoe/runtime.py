# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Material and contact constants shared by the Digital Shoe column bed."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from .contact import pasternak_coupling
from .material import HYPERFOAM_ALPHA_FLOOR
from .material import ogden_hill_term as _hyperfoam_term


@dataclass(frozen=True)
class ShoeMaterial:
    """Effective intact-shoe constitutive parameters used by the runtime.

    The equilibrium network is a **two-term** Ogden-Hill (Abaqus Hyperfoam)
    series, ``p_eq = sum_n 2 mu_n / (alpha_n lambda) (J^(-alpha_n beta) -
    lambda^alpha_n)``. One first-order term cannot cover both the 0-10% secant
    the published foam tables report and the 74-90% peak strains the bench
    fixtures reach, so a second term is carried. ``instantaneous_shear_modulus_2_pa
    = 0`` disables it exactly and recovers the historical single-term artifact.

    ``pasternak_n_per_m`` is a *reported* quantity, kept so the artifact schema
    stays readable and backward compatible. Nothing consumes it: the runtime
    derives each column's own Pasternak coefficient from the equilibrium
    Ogden-Hill shear modulus and that column's rest thickness,
    ``k_i = mu_eq * t_i``, which is the shear-layer definition ``G * t`` and adds
    no free parameter. Exporters write the bed-mean of that rule here.
    """

    instantaneous_shear_modulus_pa: float
    hyperfoam_exponent: float
    equilibrium_fraction: float
    pasternak_n_per_m: float
    effective_poisson_ratio: float = 0.0
    maxwell_relaxation_time_s: float = 0.08
    instantaneous_shear_modulus_2_pa: float = 0.0
    hyperfoam_exponent_2: float = 1.0
    equilibrium_shear_modulus_override_pa: float = 0.0

    def __post_init__(self) -> None:
        values = tuple(self.__dict__.values())
        if not np.all(np.isfinite(values)):
            raise ValueError("shoe material parameters must be finite")
        if self.instantaneous_shear_modulus_pa <= 0.0 or self.hyperfoam_exponent <= 0.0:
            raise ValueError("shear modulus and Hyperfoam exponent must be positive")
        if self.instantaneous_shear_modulus_2_pa < 0.0:
            raise ValueError("second Ogden-Hill shear modulus must be nonnegative")
        if self.equilibrium_shear_modulus_override_pa < 0.0:
            raise ValueError("equilibrium shear modulus override must be nonnegative")
        if not 0.0 < self.equilibrium_fraction <= 1.0:
            raise ValueError("equilibrium fraction must be in (0, 1]")
        if self.pasternak_n_per_m < 0.0:
            raise ValueError("Pasternak coupling must be nonnegative")
        if not -1.0 < self.effective_poisson_ratio < 0.5:
            raise ValueError("effective Poisson ratio must be in (-1, 0.5)")
        if self.maxwell_relaxation_time_s <= 0.0:
            raise ValueError("Maxwell relaxation time must be positive")

    @property
    def equilibrium_shear_modulus_pa(self) -> float:
        """Equilibrium shear modulus used by shear-layer and tangential laws [Pa].

        ``mu_eq = (G_1 + G_2) * f_eq``. Every term of an Ogden-Hill series adds
        ``2 mu_n`` to the small-strain compressive tangent regardless of its
        exponent, so the default is the series sum. A positive explicit override
        decouples shear response from the pressure-fit series while leaving normal
        compression unchanged.
        """
        if self.equilibrium_shear_modulus_override_pa > 0.0:
            return self.equilibrium_shear_modulus_override_pa
        return (self.instantaneous_shear_modulus_pa + self.instantaneous_shear_modulus_2_pa) * self.equilibrium_fraction


@wp.struct
class FoundationParams:
    """Device-side constitutive and contact constants for the column bed."""

    g_eq: wp.float32  # first-term equilibrium shear modulus G_inst * equilibrium_fraction [Pa]
    alpha: wp.float32  # first-term Hyperfoam exponent
    g_eq2: wp.float32  # second-term equilibrium shear modulus [Pa]; zero disables the term
    shear_override: wp.float32  # independent equilibrium shear modulus [Pa]; zero derives it from the series
    alpha2: wp.float32  # second-term Hyperfoam exponent
    beta: wp.float32  # poisson / (1 - 2 poisson)
    one_minus_two_poisson: wp.float32  # volumetric stretch exponent
    tau_s: wp.float32  # Maxwell relaxation time [s]
    overstress: wp.float32  # (1 - equilibrium_fraction) / equilibrium_fraction
    inv_h2: wp.float32  # 1 / spacing^2 [1/m^2]
    stretch_floor: wp.float32  # minimum stretch (foam densification limit)
    normal_damping: wp.float32  # per-column Kelvin-Voigt normal damping [N.s/m]
    friction_release_dwell_s: wp.float32  # unloaded dwell before the stick point is discarded [s]
    mu: wp.float32  # Coulomb friction coefficient


@wp.func
def _hyperfoam_series_term(stretch: wp.float32, mu: wp.float32, alpha: wp.float32, p: FoundationParams) -> wp.float32:
    """Evaluate one :func:`ogden_hill_term` with its volumetric powers skipped when they are exactly one.

    ``pow(J, -alpha * beta)`` with ``beta == 0`` is ``pow(J, +/-0)``, which IEEE and
    CUDA define as exactly one for every ``J``, so ``J`` is not needed either. Other
    materials and the small-alpha limit keep the shared expression unchanged.
    """
    if p.beta == 0.0 and wp.abs(alpha) >= HYPERFOAM_ALPHA_FLOOR:
        return 2.0 * mu / (alpha * stretch) * (1.0 - stretch**alpha)
    return _hyperfoam_term(stretch, stretch**p.one_minus_two_poisson, mu, alpha, p.beta)


@wp.func
def _hyperfoam_pressure(strain: wp.float32, p: FoundationParams) -> wp.float32:
    """Positive uniaxial compression pressure from the two-term Hyperfoam law.

    Sums :func:`_hyperfoam_term` over both Ogden-Hill terms. One first-order term
    has a single shape exponent and cannot match both the 0-10% secant of the
    published foam tables and the 74-90% peak strains the bench fixtures reach;
    the second term restores that freedom without a second ``pow`` per term.

    At the measured zero effective Poisson ratio ``beta`` is zero, so the
    volumetric factor is ``pow(x, 0)`` with ``x >= stretch_floor > 0``. That is
    exactly one on CPU and CUDA, and this forward entry point skips it bitwise;
    the stretch floor is what keeps it away from ``pow(0, 0)``.
    """
    stretch = wp.max(1.0 - strain, p.stretch_floor)
    return _hyperfoam_series_term(stretch, p.g_eq, p.alpha, p) + _hyperfoam_series_term(stretch, p.g_eq2, p.alpha2, p)


def set_material_block(params: FoundationParams, material) -> None:
    """Write every constitutive constant of ``material`` into ``params``.

    The one place a host derives a whole device-side material, so every column
    bed builds the same constants from the same foam.

    Args:
        params: Device-side constants to fill in place. Contact and friction
            fields are left untouched; they belong to
            :class:`FoundationConfig`, not to the material.
        material: Any material carrying the Ogden-Hill series,
            ``equilibrium_fraction``, ``effective_poisson_ratio`` and
            ``maxwell_relaxation_time_s``.
    """
    set_hyperfoam_series(params, material)
    poisson = float(getattr(material, "effective_poisson_ratio", 0.0))
    params.beta = poisson / (1.0 - 2.0 * poisson)
    params.one_minus_two_poisson = 1.0 - 2.0 * poisson
    params.tau_s = float(getattr(material, "maxwell_relaxation_time_s", 0.08))
    params.overstress = (1.0 - material.equilibrium_fraction) / material.equilibrium_fraction


def set_hyperfoam_series(params: FoundationParams, material) -> None:
    """Write both equilibrium Ogden-Hill terms of ``material`` into ``params``.

    The only place a host builds the device-side constitutive block, so a term
    cannot be forgotten at one call site and silently drop out of one solver.

    Args:
        params: Device-side constants to fill in place.
        material: Any material carrying ``instantaneous_shear_modulus_pa``,
            ``hyperfoam_exponent``, ``instantaneous_shear_modulus_2_pa``,
            ``hyperfoam_exponent_2`` and ``equilibrium_fraction``
            (:class:`ShoeMaterial` or
            :class:`projects.digital_instron_v2.core.Material`).
    """
    fraction = float(material.equilibrium_fraction)
    params.g_eq = float(material.instantaneous_shear_modulus_pa) * fraction
    params.alpha = float(material.hyperfoam_exponent)
    params.g_eq2 = float(material.instantaneous_shear_modulus_2_pa) * fraction
    params.alpha2 = float(material.hyperfoam_exponent_2)
    params.shear_override = float(getattr(material, "equilibrium_shear_modulus_override_pa", 0.0))


@wp.func
def material_shear_modulus(p: FoundationParams) -> wp.float32:
    """Return the equilibrium shear modulus used by tangential and shear-layer laws [Pa]."""
    if p.shear_override > 0.0:
        return p.shear_override
    return p.g_eq + p.g_eq2


@wp.func
def _pasternak_coupling(t_i: wp.float32, t_j: wp.float32, p: FoundationParams) -> wp.float32:
    """Pasternak coefficient of the shear layer between two columns [N/m].

    A Pasternak layer coefficient is ``G * t``. The shear modulus is the foam's
    own equilibrium Ogden-Hill modulus ``mu_eq``, which for a two-term series is
    the *sum* :attr:`FoundationParams.g_eq` + :attr:`FoundationParams.g_eq2`, and
    the layer thickness at the shared face is the mean of the two column rest
    thicknesses. Averaging keeps the pair conductance symmetric, so the lateral
    flux summed over the bed is exactly zero and the layer can only move load,
    never create it.
    """
    return pasternak_coupling(t_i, t_j, material_shear_modulus(p))


@dataclass
class FoundationConfig:
    """Contact settings layered on the calibrated constitutive law.

    ``ground_height_m`` is the horizontal contact plane [m] the column bottoms
    land on. ``mu`` bounds the elastic Coulomb traction at ``mu * R`` for the
    unilateral ground reaction ``R``; each column's tangential stiffness is
    ``G_eq * area / rest_length``, a geometry-scaled contact assumption that
    normal compression does not identify. ``friction_release_dwell_s`` preserves
    the stick point through short normal dropouts. ``normal_damping`` is a
    per-column Kelvin-Voigt normal damping [N.s/m] and ``stretch_floor`` the
    minimum stretch (foam densification limit).
    """

    stretch_floor: float = 0.05
    normal_damping: float = 0.0
    friction_release_dwell_s: float = 0.0005
    mu: float = 0.0
    ground_height_m: float | None = None

    def __post_init__(self) -> None:
        if self.ground_height_m is not None and (
            not np.isfinite(self.ground_height_m) or abs(self.ground_height_m) > float(np.finfo(np.float32).max)
        ):
            raise ValueError("ground_height_m must be finite in float32 or None")
        if self.friction_release_dwell_s < 0.0:
            raise ValueError("friction release dwell must be nonnegative")
