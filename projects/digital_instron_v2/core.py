# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Digital Instron material model and fit objective."""

from dataclasses import dataclass

import numpy as np

from projects.digital_shoe.material import (
    SOLID_MODULUS_PA,
    densification_onset,
    hyperfoam_pressure_from_stretch_numpy,
)
from projects.digital_shoe.runtime import SHEAR_LAYER_FRACTION

# Confined and unconfined compression of racing-shoe midsole foam agree within
# scatter, so the effective Poisson ratio is zero (McCulloch, Delp and Kuhl,
# arXiv:2602.12694, sections 2.4 and 3.3). The Ogden-Hill exponent
# ``beta = nu / (1 - 2 nu)`` then vanishes and the column law reduces to the
# beta -> 0 Hyperfoam term the same paper fits, which is also self-consistent
# with modelling the midsole as independent columns.
EFFECTIVE_POISSON_RATIO = 0.0
MAXWELL_RELAXATION_TIME_S = 0.08


@dataclass(frozen=True)
class Material:
    """Two-term Hyperfoam-Maxwell foam with a material-pinned Pasternak layer.

    The equilibrium network is an Ogden-Hill (Abaqus Hyperfoam) series with two
    terms, ``p_eq = sum_n 2 mu_n / (alpha_n lambda) (J^(-alpha_n beta) -
    lambda^alpha_n)``, evaluated at the measured ``beta = 0``. One first-order
    term has a single shape exponent and cannot span the range this project
    needs: the rearfoot punch reaches about 90% strain, the full-foot last about
    74%, and the published foam secant is measured over 0-10%. With one term the
    objective is bimodal in that exponent and the two fixtures pull in opposite
    directions. Setting ``instantaneous_shear_modulus_2_pa = 0`` reproduces the
    single-term law exactly, so the comparison against the previous fit is exact.

    The two Pasternak shear layers of the column bed have the modulus
    ``shear_layer_fraction * mu_eq``, with ``mu_eq`` this material's own
    equilibrium Ogden-Hill shear modulus. The fraction sets how far a loaded
    column carries its neighbours: on the full-foot last it decides whether the
    load spreads from the midsole rims into the dished footbed early or the
    footbed engages column by column, which the loading curve of the sweep
    resolves. It is fitted with the foam; the default is the runner's
    :data:`~projects.digital_shoe.runtime.SHEAR_LAYER_FRACTION`.

    """

    instantaneous_shear_modulus_pa: float
    hyperfoam_exponent: float
    equilibrium_fraction: float
    maxwell_relaxation_time_s: float = MAXWELL_RELAXATION_TIME_S
    instantaneous_shear_modulus_2_pa: float = 0.0
    hyperfoam_exponent_2: float = 1.0
    shear_layer_fraction: float = SHEAR_LAYER_FRACTION

    def __post_init__(self) -> None:
        if not np.all(np.isfinite(tuple(self.__dict__.values()))):
            raise ValueError("material parameters must be finite")
        if self.instantaneous_shear_modulus_pa <= 0.0 or self.hyperfoam_exponent <= 0.0:
            raise ValueError("shear modulus and Hyperfoam exponent must be positive")
        # The second term is optional and its exponent may be negative (a
        # densifying term), so only its modulus is constrained.
        if self.instantaneous_shear_modulus_2_pa < 0.0:
            raise ValueError("second Ogden-Hill shear modulus must be nonnegative")
        if not 0.0 < self.equilibrium_fraction <= 1.0:
            raise ValueError("equilibrium fraction must be in (0, 1]")
        if self.maxwell_relaxation_time_s <= 0.0:
            raise ValueError("Maxwell relaxation time must be positive")
        if self.shear_layer_fraction <= 0.0:
            raise ValueError("shear layer fraction must be positive")

    @property
    def equilibrium_shear_modulus_pa(self) -> float:
        """Equilibrium Ogden-Hill shear modulus ``mu_eq``, the SUM over both terms [Pa].

        ``mu_eq = (G_1 + G_2) * f_eq``. Every term of an Ogden-Hill series
        contributes ``2 mu_n`` to the small-strain compressive tangent whatever
        its exponent, so the series modulus is the sum of the term moduli and not
        the first term alone. The per-column Pasternak rule and the reported
        small-strain modulus both use this sum.
        """
        return (self.instantaneous_shear_modulus_pa + self.instantaneous_shear_modulus_2_pa) * (
            self.equilibrium_fraction
        )


# Fitted intact-shoe parameters from the one-unknown-per-column calibration that
# preceded the two-surface column bed; the column-bed fit starts from them.
#
# Under that calibration this material passed all six declared held-out gates (rearfoot peak 2.8%,
# RMSE 5.6%, loop 7.5%; full-foot peak 5.9%, RMSE 6.4%, loop 8.8%), which the
# single-term law did not: it reached 2 of 6 and split the peak error between
# the fixtures at -15% and +19%. The reason is the strain range, not extra
# freedom to absorb a disagreement. The rearfoot punch reaches about 90% strain
# and the full-foot last about 74%, and one first-order term has a single shape
# exponent for both, so its objective was bimodal (mu_eq 52 kPa with alpha 0.22
# against mu_eq 240 kPa with alpha 11.4 at nearly equal loss). With two terms a
# multi-start from both of those basins converges to this one optimum.
CALIBRATED_MATERIAL = Material(
    instantaneous_shear_modulus_pa=267615.1977311966,
    hyperfoam_exponent=18.07697643908876,
    equilibrium_fraction=0.6586721149134733,
    maxwell_relaxation_time_s=0.0051501095220815776,
    instantaneous_shear_modulus_2_pa=19487.70413858932,
    hyperfoam_exponent_2=-0.5855868486256809,
)


def _hyperfoam_pressure(strain: np.ndarray, material: Material) -> np.ndarray:
    """Evaluate the shared two-term equilibrium compression pressure [Pa], linear past the densification onset."""
    poisson = EFFECTIVE_POISSON_RATIO
    fraction = material.equilibrium_fraction
    law = (
        material.instantaneous_shear_modulus_pa * fraction,
        material.hyperfoam_exponent,
        material.instantaneous_shear_modulus_2_pa * fraction,
        material.hyperfoam_exponent_2,
        poisson / (1.0 - 2.0 * poisson),
        1.0 - 2.0 * poisson,
    )
    stretch = 1.0 - np.asarray(strain)
    onset, onset_pressure = densification_onset(*law)
    pressure = hyperfoam_pressure_from_stretch_numpy(np.clip(stretch, max(onset, 1.0e-3), 1.0), *law)
    return np.where(stretch < onset, onset_pressure + SOLID_MODULUS_PA * (onset - stretch), pressure)


HYSTERESIS_WEIGHT = 5.0
PEAK_WEIGHT = 6.0


def force_residual(measured_n: np.ndarray, displacement_m: np.ndarray, predicted_n: np.ndarray) -> np.ndarray:
    """Return one trace's force, loop-area, and peak residuals.

    Force error alone barely notices the hysteresis loop, because the loop is
    small next to the peak load. The declared gates score peak force and loop
    area as well, so the identification objective scores all three.
    """
    scale = max(float(np.max(measured_n)), 1.0)
    measured_loop = float(np.trapezoid(measured_n, displacement_m))
    loop_error = float(np.trapezoid(predicted_n, displacement_m)) - measured_loop
    return np.concatenate(
        [
            (predicted_n - measured_n) / scale,
            [HYSTERESIS_WEIGHT * loop_error / max(abs(measured_loop), 1.0e-9)],
            [PEAK_WEIGHT * (float(np.max(predicted_n)) - float(np.max(measured_n))) / scale],
        ]
    )


# Order matches the :class:`Material` field order: first-term shear modulus,
# first-term exponent, equilibrium fraction, relaxation time, second-term shear
# modulus, second-term exponent, shear-layer fraction. There is no
# fixture-specific parameter to bound.
#
# The second modulus may reach zero, which turns the series back into the
# single-term law exactly. The second exponent may be negative: an Ogden-Hill
# series admits either sign, a positive exponent produces the soft ``1 / lambda``
# plateau and a negative one produces densification, and the published two-term
# fits of this foam family use one of each. Zero is inside the interval and is
# handled by the removable limit of the Ogden-Hill term. The first exponent
# reaches 40 because the FR3 sweep drives it onto a bound of 20. The shear-layer
# fraction spans a hundredth of the foam modulus to a layer stiffened 30-fold by
# a plate.
FIT_LOWER_BOUNDS = (1.0e3, 0.1, 0.01, 5.0e-3, 0.0, -20.0, 0.01)
FIT_UPPER_BOUNDS = (1.0e8, 40.0, 1.0, 2.0, 1.0e8, 20.0, 30.0)


def metrics(measured: np.ndarray, predicted: np.ndarray, displacement: np.ndarray) -> dict[str, float | bool]:
    """Return peak, RMSE, and hysteresis errors."""

    peak = max(float(np.max(measured)), 1.0e-9)
    peak_frame = int(np.argmax(displacement))
    values = {
        "peak_force_error": abs(float(np.max(predicted)) - peak) / peak,
        "force_rmse_relative": float(np.sqrt(np.mean((predicted - measured) ** 2)) / peak),
        "loading_rmse_relative": float(
            np.sqrt(np.mean((predicted[: peak_frame + 1] - measured[: peak_frame + 1]) ** 2)) / peak
        ),
        "unloading_rmse_relative": float(
            np.sqrt(np.mean((predicted[peak_frame:] - measured[peak_frame:]) ** 2)) / peak
        ),
        "hysteresis_error": abs(float(np.trapezoid(predicted - measured, displacement)))
        / max(abs(float(np.trapezoid(measured, displacement))), 1.0e-9),
    }
    values["passed"] = all(value < 0.1 for value in values.values())
    return values
