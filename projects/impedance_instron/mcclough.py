# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Two published racing-shoe midsole foams expressed in this project's foam model.

Source: J. A. McCulloch, S. L. Delp, E. Kuhl, "Discovering the mechanics of
ultra-low density elastomeric foams in elite-level racing shoes",
arXiv:2602.12694v1 [cs.CE], 13 Feb 2026 (CC BY 4.0). The two foams are the
ASICS FF LEAP and FF TURBO PLUS, cut out of Metaspeed Sky and Edge shoes
(Section 2, Section 2.1), tested in uniaxial tension, unconfined and confined
compression, and simple shear at a strain rate of 0.25/s (Section 2.2).

What the paper actually reports
-------------------------------
Every stiffness below is a linear regression through the origin of stress on
strain over relative deformations up to 10% only (Section 2.3, "Linear elastic
stiffness and energy return"); it is *not* a secant at large strain. The
relative energy return is the ratio of the areas under the unloading and
loading curves over the whole tested range, so for compression that is the full
stretch sweep 1.0 -> 0.4 (Section 2.3, Tables 1 and 2).

=========================  ===============  ==================
quantity                   FF LEAP          FF TURBO PLUS
=========================  ===============  ==================
E_ten  (0-10% strain)      623.65 +- 96.36  884.15 +- 68.81 kPa
E_com  (0-10% strain)      299.22 +- 29.09  267.94 +- 15.67 kPa
G_shr  (0-10% strain)      117.16 +- 23.73  219.12 +- 20.39 kPa
eta_ten                    90.7 +- 1.1%     94.3 +- 1.3%
eta_com                    89.5 +- 1.6%     84.6 +- 1.3%
eta_shr                    73.6 +- 0.7%     75.6 +- 0.7%
=========================  ===============  ==================

The abstract quotes 88.9 +- 1.8% and 83.3 +- 1.5% for the same two compressive
energy returns that Tables 1 and 2 give as 89.5 +- 1.6% and 84.6 +- 1.3%. The
tables are used here; the ~0.7 point discrepancy is an unresolved internal
inconsistency of the paper and is smaller than the reported scatter.

Confined and unconfined compression agreed, so the paper concludes an effective
Poisson's ratio of approximately zero and assumes a transverse stretch of one in
tension and compression (Sections 2.4 and 3.2). That matches this project's
``effective_poisson_ratio = 0.0`` exactly, and it makes the reported E_com a
*confined* (oedometric) modulus, which is the modulus a laterally confined foam
column carries.

The model form they discovered is NOT Ogden-Hill
------------------------------------------------
The paper trains a constitutive neural network over fourteen candidate terms
and keeps the sparse survivors (Sections 2.5, 3.3). The two three-term models
they report are, with ``I1b``, ``I2b`` the isochoric invariants and ``J`` the
volume ratio:

single-invariant + mixed-invariant (Section 3.3, R^2 = 0.98-1.00)::

    psi_leap  = 26.9 kPa [I1b - 3]
              + 670.0 kPa [exp(0.0587 ln(J)^2) - 1]
              + 79.0 kPa J^3.92 [I1b - 3]
    psi_turbo = 22.8 kPa [J^2.25 - 2.25 ln(J) - 1]
              + 139.0 kPa J^1.50 [I1b - 3]
              + 19.8 kPa J^2.58 [I2b - 3]

single-invariant + principal-stretch (Section 3.3, R^2 = 0.94-1.00)::

    psi_leap  = 9.93 kPa [I1b - 3]^2
              + 14.20 kPa [exp(0.481 ln(J)^2) - 1]
              + 0.286 kPa sum_i [lambda_i^8.40 - 8.40 ln(lambda_i) - 1]
    psi_turbo = 73.9 kPa [exp(0.147 (I1b - 3)) - 1]
              + 0.00178 kPa [J^6.64 - 6.64 ln(J) - 1]
              + 0.365 kPa sum_i [lambda_i^8.33 - 8.33 ln(lambda_i) - 1]

Only the third term of each principal-stretch model is an Ogden-Hill term, and
it is the beta -> 0 special case, so **none of the published coefficients can be
copied into** :class:`~projects.digital_shoe.runtime.ShoeMaterial`. Both models
carry their tension-compression asymmetry in terms (mixed-invariant ``J^w [I1b
- 3]``, or the ``I1b`` terms) that have no counterpart in a two-term Ogden-Hill
series. The parameters here were therefore **refitted from the paper's raw
compression data**, Tables 1 and 2, not converted from the published weights.

What was matched, and what could not be
---------------------------------------
Matched:

* the whole uniaxial compression backbone, lambda = 1.0 -> 0.4 (Tables 1 and 2),
  to 2.1 kPa RMS (FF LEAP) and 5.3 kPa RMS (FF TURBO PLUS) out of a 242 and
  305 kPa peak;
* E_com exactly, re-measured on the fitted curve with the paper's own estimator
  (regression through the origin for strains up to 10%);
* the effective Poisson's ratio, zero in both;
* the *ratio* of compressive hysteresis between the two foams (see below).

NOT matched, in decreasing order of importance:

1. **Shear stiffness.** The runtime derives each column's Pasternak coefficient
   from the equilibrium Ogden-Hill modulus, ``k_i = mu_eq * t_i``, so the shear
   response is a dependent quantity, not a free parameter. An isotropic
   Ogden-Hill solid with nu = 0 has G = E/2, which here forces G = 168.3 kPa
   (FF LEAP) and 134.7 kPa (FF TURBO PLUS) against measured 117.2 and 219.1 kPa.
   **The ordering inverts**: the paper's FF TURBO PLUS is 1.87x stiffer in shear
   than FF LEAP, this model makes it 0.80x. No choice of Ogden-Hill parameters
   fixes this, because the measured pairs imply nu = +0.28 (FF LEAP) and
   nu = -0.39 (FF TURBO PLUS) through G = E/(2(1+nu)); the foams are simply not
   isotropic-with-nu-zero. Any conclusion about lateral stability drawn from a
   rollout on these two materials is invalid.
2. **Tensile stiffness and the tension-compression asymmetry.** See below.
3. **Absolute hysteresis.** See below.
4. **Structure.** The paper tests bare midsole foam with the carbon plate and
   outsole removed (Section 2.1). This project's ``ShoeMaterial`` is an
   effective *intact-shoe* parameter set. Substituting bare-foam properties into
   it therefore conflates a material change with the structure that is missing.
   This is a limitation of the comparison, not of the fit.

Energy return -> ``equilibrium_fraction``
-----------------------------------------
The runtime carries one Maxwell branch whose overstress obeys
``q' = ((1-g)/g) p_eq' - q/tau`` with ``g = equilibrium_fraction`` and
``tau = maxwell_relaxation_time_s``. For a load-unload ramp at constant strain
rate ``e' = e_max / T`` with ``T >> tau``, the overstress settles at
``q = +-((1-g)/g) tau e' p_eq'(e)``, a constant vertical offset in *energy*:

    E_load   = W + ((1-g)/g) tau e' p_eq(e_max)
    E_unload = W - ((1-g)/g) tau e' p_eq(e_max),   W = int_0^e_max p_eq de

so with the curve-shape factor ``S = W / (e_max p_eq(e_max))``,

    eta = (1 - kappa) / (1 + kappa),   kappa = (1-g)/g * tau / (S T)

    =>  (1-g)/g = (1-eta)/(1+eta) * S * T / tau                        (*)

Verified against the runtime's own periodic overstress recursion to better than
0.001 in eta.

The paper measures eta at ``T = 0.6 / 0.25 = 2.4 s``. Reading (*) literally with
this project's ``tau = 0.00515 s`` gives ``(1-g)/g = 9.65`` and ``13.67``, i.e.
``g = 0.094`` and ``0.068``: a foam seven times stiffer dynamically than
statically. That is physically absurd for a material the paper calls
low-hysteresis, and it is six to nine times outside this shoe's ``g = 0.6587``.
The reason is that the paper's hysteresis is essentially rate-independent
(they attribute the large first-cycle loss to Mullins conditioning, Section 3.1,
and state plainly in the Discussion "Limitations" that their models cannot
predict dissipation and that rate-varying data would be needed), whereas a
Maxwell branch is rate-dependent by construction. **The paper constrains only
the product ``((1-g)/g) tau``, at one single strain rate, and cannot separate
g from tau at all.**

What is transferable is the *ratio* of the two foams' losses. Writing
``kappa_paper = (1-eta)/(1+eta)``:

    kappa_leap  = (1 - 0.895) / (1 + 0.895) = 0.055409
    kappa_turbo = (1 - 0.846) / (1 + 0.846) = 0.083424
    ratio turbo/leap                        = 1.5056
    geometric mean                          = 0.067988

Anchoring that pair on this shoe's own loss level ``(1-g)/g = 0.3413/0.6587 =
0.518142`` preserves the measured ratio exactly while keeping both foams inside
the training distribution:

    (1-g)/g |_leap  = 0.518142 * 0.055409 / 0.067988 = 0.42227 -> g = 0.70310
    (1-g)/g |_turbo = 0.518142 * 0.083424 / 0.067988 = 0.63577 -> g = 0.61133

Relation (*) is a large-``T/tau`` approximation. It is exact to 4e-5 in eta at
the paper's own 2.4 s ramp, which is where it was used above, but it reads about
1.5 points high once ``T/tau`` falls to about 20. Inverting the *exact* periodic
recursion instead, the ramp time at which these two equilibrium fractions
reproduce the published eta is ``T = 0.0856 s`` (FF LEAP) and ``T = 0.0934 s``
(FF TURBO PLUS), see :data:`STANCE_RAMP_S`. Those are not fitted, they fall out,
and both land on the loading-phase duration of a running stance. So the anchored
mapping reads: *these materials reproduce the paper's compressive energy return
at running-stance loading rate, not at the paper's bench rate.* At the paper's
own 0.25/s they return 99.5% and 99.2% instead of 89.5% and 84.6%. The absolute
level of dissipation therefore comes from this project's shoe, and only the
FF TURBO PLUS / FF LEAP loss ratio of 1.51 comes from the paper.

Tension-compression asymmetry
-----------------------------
The paper's headline asymmetry (E_ten / E_com = 2.08 and 3.30) is **irrelevant
to the normal load path** of this project: the column bed is unilateral, the
reaction is clamped at zero, and no column ever carries tension. It is a real
limitation for two other things. First, the foams' tensile stiffnesses differ by
42% while their compressive stiffnesses differ by only 10%, so a real midsole in
bending separates these two foams far more than a compression-only bed can
show. Second, a bending midsole is exactly where the carbon plate acts, and the
plate is absent from the tested samples anyway. Treat any difference this model
reports between the two foams as a compression-only difference.

Reference values used throughout
--------------------------------
This project's current shoe, the two-term refit at
``outputs/impedance_instron/refit_two_term/digital_shoe.json``:
``mu_1 = 267615.2 Pa``, ``alpha_1 = 18.077``, ``mu_2 = 19487.7 Pa``,
``alpha_2 = -0.5856``, ``g = 0.6587``, ``tau = 0.00515 s``, ``nu = 0``.
"""

from __future__ import annotations

import numpy as np

from projects.digital_shoe.material import (
    hyperfoam_pressure_numpy,
    maxwell_coefficients_numpy,
    maxwell_step_numpy,
)
from projects.digital_shoe.runtime import ShoeMaterial

__all__ = [
    "FF_LEAP",
    "FF_TURBO_PLUS",
    "PAPER_COMPRESSION_KPA",
    "PAPER_PROPERTIES",
    "RANDOMIZATION_BAND",
    "REFERENCE_SHOE",
    "STANCE_RAMP_S",
    "compressive_stiffness_pa",
    "energy_return",
    "equilibrium_pressure_pa",
    "materials",
]


# --- Paper data, transcribed verbatim -------------------------------------------------

# Tables 1 and 2, uniaxial compression columns: stretch lambda and Piola stress P11 [kPa].
# Both tables share the same stretch grid. These are already the mean of the loading and
# unloading curves across n = 5 samples with the first conditioning cycle excluded
# (Section 2.3), so they are the foams' elastic backbone, not a loading curve.
PAPER_COMPRESSION_KPA: dict[str, np.ndarray] = {
    "stretch": np.array([1.000, 0.950, 0.900, 0.850, 0.800, 0.750, 0.700, 0.650, 0.600, 0.550, 0.500, 0.450, 0.400]),
    # Table 1, FF LEAP
    "FF_LEAP": np.array([0.0, 12.79, 32.65, 45.64, 55.37, 65.47, 77.17, 91.21, 108.06, 129.08, 156.19, 191.67, 241.60]),
    # Table 2, FF TURBO PLUS
    "FF_TURBO_PLUS": np.array(
        [0.0, 9.41, 32.75, 51.40, 64.33, 76.82, 90.92, 107.86, 128.72, 155.11, 189.78, 236.96, 305.43]
    ),
}

# Tables 1 and 2, summary rows. Stiffnesses are 0-10% strain regressions through the
# origin; energy returns are unloading/loading area ratios over the full tested range
# (Section 2.3). Values are (mean, standard deviation) across n = 5 samples.
PAPER_PROPERTIES: dict[str, dict[str, tuple[float, float]]] = {
    "FF_LEAP": {
        "tensile_stiffness_pa": (623.65e3, 96.36e3),  # Table 1
        "compressive_stiffness_pa": (299.22e3, 29.09e3),  # Table 1
        "shear_stiffness_pa": (117.16e3, 23.73e3),  # Table 1
        "energy_return_tension": (0.907, 0.011),  # Table 1
        "energy_return_compression": (0.895, 0.016),  # Table 1
        "energy_return_shear": (0.736, 0.007),  # Table 1
    },
    "FF_TURBO_PLUS": {
        "tensile_stiffness_pa": (884.15e3, 68.81e3),  # Table 2
        "compressive_stiffness_pa": (267.94e3, 15.67e3),  # Table 2
        "shear_stiffness_pa": (219.12e3, 20.39e3),  # Table 2
        "energy_return_tension": (0.943, 0.013),  # Table 2
        "energy_return_compression": (0.846, 0.013),  # Table 2
        "energy_return_shear": (0.756, 0.007),  # Table 2
    },
}

# Loading half-cycle duration at which each material's ratio-anchored
# equilibrium_fraction reproduces the paper's compressive energy return exactly.
# INFERRED: obtained by inverting the exact periodic Maxwell recursion, not relation
# (*) of the module docstring. Reported because the values fall out near the 0.09 s
# loading phase of a running stance rather than being chosen there.
STANCE_RAMP_S: dict[str, float] = {"FF_LEAP": 0.0856, "FF_TURBO_PLUS": 0.0934}

# Section 2.2: every test ran at a stretch rate of 0.25/s, and compression swept
# lambda = 1.0 -> 0.4, so the compressive load-unload ramp lasted 0.6 / 0.25 = 2.4 s.
PAPER_STRAIN_RATE_PER_S = 0.25
PAPER_MAX_COMPRESSIVE_STRAIN = 0.60
PAPER_COMPRESSION_RAMP_S = PAPER_MAX_COMPRESSIVE_STRAIN / PAPER_STRAIN_RATE_PER_S


# --- This project's current shoe, the baseline both foams are measured against --------

REFERENCE_SHOE = ShoeMaterial(
    instantaneous_shear_modulus_pa=267615.2,  # outputs/impedance_instron/refit_two_term/digital_shoe.json
    hyperfoam_exponent=18.077,
    instantaneous_shear_modulus_2_pa=19487.7,
    hyperfoam_exponent_2=-0.5856,
    equilibrium_fraction=0.6587,
    maxwell_relaxation_time_s=0.00515,
    effective_poisson_ratio=0.0,
    pasternak_n_per_m=0.0,  # inert: the runtime derives k_i = mu_eq * t_i per column
)


# --- The two foams ---------------------------------------------------------------------

FF_LEAP = ShoeMaterial(
    # INFERRED. Least-squares fit of the two-term Ogden-Hill law
    # p_eq = sum_n 2 mu_n / (alpha_n lambda) (J^(-alpha_n beta) - lambda^alpha_n), beta = 0,
    # to the 13 compression points of Table 1, with the paper's own 0-10% regression
    # estimator on the fitted curve constrained to E_com = 299.22 kPa (Table 1).
    # Equilibrium moduli mu_1^eq = 144.034 kPa, mu_2^eq = 24.295 kPa; the field below is
    # the INSTANTANEOUS modulus mu_1^eq / equilibrium_fraction. RMS residual 2.07 kPa
    # over a 0-241.6 kPa range.
    instantaneous_shear_modulus_pa=204855.5,
    hyperfoam_exponent=7.75000,  # INFERRED, same fit
    instantaneous_shear_modulus_2_pa=34554.1,  # INFERRED, same fit
    # INFERRED, and NOT identifiable from the paper: their compression data stops at 60%
    # strain, where the densifying second exponent barely acts. Locked to this project's
    # own value so the term stays inside the training distribution and so the
    # extrapolation past the tested range stays numerically tame (8.7 MPa at 95% strain
    # against this shoe's 4.6 MPa; leaving it free drove the FF TURBO PLUS fit to a
    # 46 GPa wall). Freeing it improves the in-range RMS only from 2.07 to 1.79 kPa.
    hyperfoam_exponent_2=-0.5856,
    # INFERRED. Ratio-anchored from eta_com = 89.5% (Table 1); see the module docstring.
    # (1-g)/g = 0.518142 * 0.055409 / 0.067988 = 0.42227.
    equilibrium_fraction=0.70310,
    # NOT DETERMINED by the paper: they tested one strain rate only and report no
    # relaxation data. Inherited from this project's shoe so that the two foams differ
    # only in what the paper actually measured.
    maxwell_relaxation_time_s=0.00515,
    effective_poisson_ratio=0.0,  # Sections 2.4 and 3.2, nu approximately 0
    pasternak_n_per_m=0.0,  # inert
)

FF_TURBO_PLUS = ShoeMaterial(
    # INFERRED. Same fit against the 13 compression points of Table 2, with the 0-10%
    # regression estimator constrained to E_com = 267.94 kPa (Table 2). Equilibrium
    # moduli mu_1^eq = 103.808 kPa, mu_2^eq = 30.850 kPa. RMS residual 5.31 kPa over a
    # 0-305.4 kPa range; the residual is larger than FF LEAP's because Table 2 has a
    # pronounced toe (9.41 kPa at 5% strain, then 32.75 kPa at 10%) that a two-term
    # Ogden-Hill series with a single positive tangent cannot reproduce.
    instantaneous_shear_modulus_pa=169806.6,
    hyperfoam_exponent=4.48154,  # INFERRED, same fit
    instantaneous_shear_modulus_2_pa=50463.3,  # INFERRED, same fit
    hyperfoam_exponent_2=-0.5856,  # INFERRED and locked, see FF_LEAP
    # INFERRED. Ratio-anchored from eta_com = 84.6% (Table 2); see the module docstring.
    # (1-g)/g = 0.518142 * 0.083424 / 0.067988 = 0.63577.
    equilibrium_fraction=0.61133,
    maxwell_relaxation_time_s=0.00515,  # NOT DETERMINED by the paper, see FF_LEAP
    effective_poisson_ratio=0.0,  # Sections 2.4 and 3.2, nu approximately 0
    pasternak_n_per_m=0.0,  # inert
)


# Fractional half-width, around this project's current shoe, that a uniform
# domain randomization needs so that both paper foams land inside the training
# distribution. Each entry is ``max_over_foams |x_foam / x_shoe - 1|``, rounded up.
#
#   field                             shoe        FF LEAP            FF TURBO PLUS
#   instantaneous_shear_modulus_pa    267615.2    204855.5 (0.766)   169806.6 (0.635)  -> 0.37
#   hyperfoam_exponent                  18.077       7.750 (0.429)      4.482 (0.248)  -> 0.76
#   instantaneous_shear_modulus_2_pa   19487.7     34554.1 (1.773)    50463.3 (2.590)  -> 1.59
#   hyperfoam_exponent_2               -0.5856     -0.5856 (1.000)    -0.5856 (1.000)  -> 0.00
#   equilibrium_fraction                0.6587      0.70310 (1.067)    0.61133 (0.928) -> 0.08
#   maxwell_relaxation_time_s           0.00515     0.00515 (1.000)    0.00515 (1.000) -> 0.00
#
# Read those per-field numbers with care. The four Ogden-Hill parameters are strongly
# correlated, so randomizing them independently at these widths samples mostly
# nonsense; and the +-76% needed on the first exponent is not a "spread", it is a
# different curve shape. The honest summary is the last entry, which is the band in
# response space: the ratio of foam equilibrium pressure to shoe equilibrium pressure
# swept over strains 0.02 to 0.60. It runs 0.81 to 2.38, i.e. a log-uniform band of
# x/ 1.71 centred on 1.39 times this shoe's pressure. The current shoe is NOT the
# centre of the distribution these foams live in: it has a much flatter crush plateau
# (alpha_1 = 18.1 against 7.8 and 4.5), so it is 10-20% stiffer than both foams below
# 5% strain and 1.9-2.4x softer than both above 30% strain. A policy frozen on this
# shoe will be extrapolating on both foams in mid-stance unless the training
# randomization also varies the exponents, not only the moduli.
RANDOMIZATION_BAND: dict[str, float | tuple[float, float]] = {
    "instantaneous_shear_modulus_pa": 0.37,
    "hyperfoam_exponent": 0.76,
    "instantaneous_shear_modulus_2_pa": 1.59,
    "hyperfoam_exponent_2": 0.00,
    "equilibrium_fraction": 0.08,
    "maxwell_relaxation_time_s": 0.00,
    "equilibrium_shear_modulus_pa": 0.29,  # summed series modulus: 0.890 and 0.712 of the shoe
    "equilibrium_pressure_ratio": (0.81, 2.38),  # over strains 0.02 to 0.60, see above
}


def materials() -> dict[str, ShoeMaterial]:
    """Return both paper foams keyed by the paper's own product names.

    Returns:
        Mapping from foam name to its :class:`~projects.digital_shoe.runtime.ShoeMaterial`.
        The keys match those of :data:`PAPER_PROPERTIES` and :data:`PAPER_COMPRESSION_KPA`.
    """
    return {"FF_LEAP": FF_LEAP, "FF_TURBO_PLUS": FF_TURBO_PLUS}


def equilibrium_pressure_pa(material: ShoeMaterial, strain: np.ndarray | float) -> np.ndarray:
    """Evaluate the shared equilibrium Ogden-Hill compression pressure [Pa].

    This uses the NumPy backend of the same source that Warp compiles for the
    live and differentiable runtimes. Both terms use equilibrium moduli
    ``mu_n * equilibrium_fraction``. The remaining thickness is floored at
    ``1e-3``, as in the host identification path.

    Args:
        material: Material carrying the two-term Ogden-Hill series.
        strain: Compressive strain ``1 - lambda`` [-], positive in compression.

    Returns:
        Compression pressure [Pa], shaped like ``strain``.
    """
    poisson = material.effective_poisson_ratio
    fraction = material.equilibrium_fraction
    return hyperfoam_pressure_numpy(
        np.asarray(strain, dtype=float),
        material.instantaneous_shear_modulus_pa * fraction,
        material.hyperfoam_exponent,
        material.instantaneous_shear_modulus_2_pa * fraction,
        material.hyperfoam_exponent_2,
        poisson / (1.0 - 2.0 * poisson),
        1.0 - 2.0 * poisson,
        1.0e-3,
    )


def compressive_stiffness_pa(material: ShoeMaterial, max_strain: float = 0.10, samples: int = 401) -> float:
    """Re-measure a material with the paper's own linear-stiffness estimator [Pa].

    Section 2.3 defines ``E_com = eps . sigma / eps . eps``, a regression through the
    origin over relative deformations up to 10%. Applying the same estimator to the
    model curve is the only apples-to-apples comparison with the reported E_com; the
    small-strain *tangent* ``2 sum_n mu_n equilibrium_fraction`` is a different and
    larger number for a stiffening foam.

    Args:
        material: Material to measure.
        max_strain: Upper end of the regression window [-].
        samples: Number of evenly spaced strain samples in the window.

    Returns:
        Regression stiffness [Pa].
    """
    strain = np.linspace(0.0, max_strain, samples)
    stress = equilibrium_pressure_pa(material, strain)
    return float(strain @ stress / (strain @ strain))


def energy_return(
    material: ShoeMaterial,
    max_strain: float = PAPER_MAX_COMPRESSIVE_STRAIN,
    ramp_s: float = PAPER_COMPRESSION_RAMP_S,
    samples: int = 1001,
) -> float:
    """Relative energy return of one triangular load-unload compression ramp [-].

    Steps the runtime's own periodic Maxwell recursion (``cycle_overstress`` followed
    by ``cycle_force``) on the host instead of the closed form of the module docstring.
    The closed form assumes ``ramp_s >> maxwell_relaxation_time_s``; it is exact to
    4e-5 at the paper's 2.4 s bench ramp but reads 1.5 points high near a 0.1 s stance
    ramp, which is precisely where these materials are used. Agrees with the Warp
    kernels to better than 0.001.

    Args:
        material: Material to evaluate.
        max_strain: Peak compressive strain of the ramp [-].
        ramp_s: Duration of the loading half-cycle [s]; the unloading half takes the
            same time.
        samples: Number of strain samples in the loading half-cycle.

    Returns:
        Ratio of unloading to loading area [-], in (0, 1].
    """
    loading = np.linspace(0.0, max_strain, samples)
    strain = np.concatenate([loading, loading[-2::-1]])
    pressure = equilibrium_pressure_pa(material, strain)
    overstress_gain = (1.0 - material.equilibrium_fraction) / material.equilibrium_fraction
    step_s = ramp_s / (samples - 1)
    decay, ramp = maxwell_coefficients_numpy(step_s, material.maxwell_relaxation_time_s)

    # Two passes, as in ``cycle_overstress``: the first closes the periodic state, the
    # second records it. Without the first pass the loop would start from an unloaded
    # branch that the periodic cycle never visits.
    state = 0.0
    previous = float(pressure[-1])
    for value in pressure:
        state = maxwell_step_numpy(state, float(value), previous, overstress_gain, decay, ramp)
        previous = float(value)
    state /= 1.0 - decay**pressure.size
    previous = float(pressure[-1])
    stress = np.empty_like(pressure)
    for index, value in enumerate(pressure):
        state = maxwell_step_numpy(state, float(value), previous, overstress_gain, decay, ramp)
        previous = float(value)
        stress[index] = max(float(value) + state, 0.0)

    load_area = float(np.trapezoid(stress[:samples], strain[:samples]))
    unload_area = -float(np.trapezoid(stress[samples - 1 :], strain[samples - 1 :]))
    return unload_area / load_area


def _report() -> str:
    """Build the paper-versus-model comparison table as text."""
    stance_ramp_s = STANCE_RAMP_S
    lines: list[str] = []
    stretch = PAPER_COMPRESSION_KPA["stretch"]
    strain = 1.0 - stretch
    for name, material in materials().items():
        paper = PAPER_PROPERTIES[name]
        mu_eq = material.equilibrium_shear_modulus_pa
        model_curve = equilibrium_pressure_pa(material, strain) / 1.0e3
        residual = model_curve - PAPER_COMPRESSION_KPA[name]
        lines.append(f"\n{name}")
        lines.append(f"  {'quantity':<34s}{'paper':>14s}{'model':>14s}   note")
        lines.append(f"  {'-' * 34}{'-' * 13:>14s}{'-' * 13:>14s}   {'-' * 34}")
        lines.append(
            f"  {'E_com, 0-10% regression [kPa]':<34s}{paper['compressive_stiffness_pa'][0] / 1e3:>14.2f}"
            f"{compressive_stiffness_pa(material) / 1e3:>14.2f}   matched by construction"
        )
        lines.append(
            f"  {'compression curve RMS [kPa]':<34s}{'-':>14s}{float(np.sqrt(np.mean(residual**2))):>14.2f}"
            f"   over 0-60% strain, peak {PAPER_COMPRESSION_KPA[name][-1]:.0f} kPa"
        )
        lines.append(
            f"  {'G_shr [kPa]':<34s}{paper['shear_stiffness_pa'][0] / 1e3:>14.2f}{mu_eq / 1e3:>14.2f}"
            "   NOT matched, G = E/2 is forced"
        )
        lines.append(
            f"  {'E_ten [kPa]':<34s}{paper['tensile_stiffness_pa'][0] / 1e3:>14.2f}{'n/a':>14s}"
            "   bed is compression only"
        )
        lines.append(
            f"  {'eta_com at stance rate [%]':<34s}{paper['energy_return_compression'][0] * 100:>14.1f}"
            f"{energy_return(material, ramp_s=stance_ramp_s[name]) * 100:>14.1f}"
            f"   matched at T = {stance_ramp_s[name]:.3f} s"
        )
        lines.append(
            f"  {'eta_com at paper rate [%]':<34s}{paper['energy_return_compression'][0] * 100:>14.1f}"
            f"{energy_return(material) * 100:>14.1f}   NOT matched, see docstring"
        )
        lines.append(f"\n  {'strain':>8s}{'paper [kPa]':>14s}{'model [kPa]':>14s}{'error [kPa]':>14s}")
        for e, p, m in zip(strain, PAPER_COMPRESSION_KPA[name], model_curve, strict=True):
            lines.append(f"  {e:>8.2f}{p:>14.2f}{m:>14.2f}{m - p:>14.2f}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(_report())
