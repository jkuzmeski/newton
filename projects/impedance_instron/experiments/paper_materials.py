# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Paper-informed compression surrogate presets and literature references.

Provides surrogate Ogden-Hill 2-term compressible hyperelastic parameters fit to
the unconfined compression experiments of:
    McCulloch, Delp, Kuhl (2026). Discovering the mechanics of ultra-low density
    elastomeric foams in elite-level racing shoes. Engineering with Computers.
    DOI: https://doi.org/10.1007/s00366-026-02398-y
    arXiv: https://arxiv.org/abs/2602.12694v2

The paper reports mean loading and unloading response after discarding the
first cycle (strain rate 0.25 s^-1, n=5 samples) over strains [0, 0.60].

Note on surrogate nature:
- These are 2-term compressible Ogden-Hill (Hyperfoam beta=0) surrogate fits to
  unconfined uniaxial compression, NOT the authors' full CANN material model.
- The paper does not report Maxwell viscoelastic branch parameters; the baseline
  equilibrium fraction feq and relaxation time tau are held fixed as an explicit
  modeling assumption.
- The Pasternak coupling shear modulus is pinned to the surrogate equilibrium
  modulus sum (mu_eq), not the full nonlinear shear response from the paper.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from projects.digital_shoe.material import hyperfoam_pressure_numpy

PAPER_SOURCES: dict[str, str] = {
    "doi": "https://doi.org/10.1007/s00366-026-02398-y",
    "arxiv": "https://arxiv.org/abs/2602.12694v2",
    "citation": (
        "McCulloch, Delp, Kuhl (2026), Discovering the mechanics of ultra-low density "
        "elastomeric foams in elite-level racing shoes, Eng. with Computers"
    ),
    "manuscript_note": ("Verified from authors arXiv:2602.12694v2 Tables 1 and 2; publisher automated page blocked."),
}

LITERATURE_LIMIT_STRAIN: float = 0.60

PAPER_TABLE_STRAIN: list[float] = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

# Unconfined compression mean stress values [kPa] from Table 1 and Table 2 (n=5, 0.25 s^-1)
FF_LEAP_TABLE_STRESS_KPA: list[float] = [
    0.0,
    12.79,
    32.65,
    45.64,
    55.37,
    65.47,
    77.17,
    91.21,
    108.06,
    129.08,
    156.19,
    191.67,
    241.60,
]

FF_TURBO_PLUS_TABLE_STRESS_KPA: list[float] = [
    0.0,
    9.41,
    32.75,
    51.40,
    64.33,
    76.82,
    90.92,
    107.86,
    128.72,
    155.11,
    189.78,
    236.96,
    305.43,
]

# Verified surrogate equilibrium parameters [Pa and dimensionless]
# (Recovered from digital_instron_v2/core.py MULTISTART_SEEDS last 2 seeds * 0.6954)
FF_LEAP_SURROGATE_PARAMS: dict[str, float] = {
    "mu1_eq_pa": 171763.8,
    "alpha1": 8.39,
    "mu2_eq_pa": 18219.48,
    "alpha2": -1.04,
    "mu_eq_sum_pa": 189983.28,
}

FF_TURBO_PLUS_SURROGATE_PARAMS: dict[str, float] = {
    "mu1_eq_pa": 168982.2,
    "alpha1": 5.65,
    "mu2_eq_pa": 12030.42,
    "alpha2": -2.0,
    "mu_eq_sum_pa": 181012.62,
}

SURROGATE_ASSUMPTIONS: list[str] = [
    "Ogden-Hill 2-term compressible hyperelastic surrogate fit to unconfined compression mean loading/unloading response (strain rate 0.25 s^-1).",
    "Not authors full CANN material model (which used polyconvex single-invariant and principal-stretch terms with L0.5 regularization).",
    "No paper Maxwell parameters reported; baseline viscoelasticity (equilibrium fraction feq and relaxation time tau) held fixed as an assumption.",
    "Inferred Pasternak shear modulus pinned to surrogate mu_eq, not paper full nonlinear shear response.",
]


def audit_fit_metrics(
    table_strain: list[float],
    table_stress_kpa: list[float],
    mu1_eq_pa: float,
    alpha1: float,
    mu2_eq_pa: float,
    alpha2: float,
) -> dict[str, Any]:
    """Compute fit metrics using the shared hyperfoam_pressure_numpy law."""
    strains = np.asarray(table_strain, dtype=np.float64)
    measured = np.asarray(table_stress_kpa, dtype=np.float64)

    # beta=0 (effective Poisson ratio 0), one_minus_two_poisson=1.0, stretch_floor=0.01
    p_pa = hyperfoam_pressure_numpy(strains, mu1_eq_pa, alpha1, mu2_eq_pa, alpha2, 0.0, 1.0, 0.01)
    p_kpa = p_pa / 1000.0

    err_kpa = p_kpa - measured
    rmse = float(np.sqrt(np.mean(err_kpa**2)))
    abs_err = np.abs(err_kpa)
    max_idx = int(np.argmax(abs_err))
    max_err = float(abs_err[max_idx])
    max_strain = float(strains[max_idx])

    # Stress at 50% strain
    p_50_pa = hyperfoam_pressure_numpy(np.array([0.5]), mu1_eq_pa, alpha1, mu2_eq_pa, alpha2, 0.0, 1.0, 0.01)
    stress_50pct = float(p_50_pa[0] / 1000.0)

    return {
        "rmse_kpa": rmse,
        "max_err_kpa": max_err,
        "max_err_strain": max_strain,
        "stress_50pct_kpa": stress_50pct,
        "pressure_curve_kpa": p_kpa.tolist(),
        "error_curve_kpa": err_kpa.tolist(),
    }


def get_paper_material_metadata(preset_name: str, source_artifact: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return sealed paper_material metadata dictionary for a given preset.

    Args:
        preset_name: 'ff_leap_compression' or 'ff_turbo_plus_compression'.
        source_artifact: Optional baseline digital shoe artifact to supply baseline feq and tau.
            When None, instantaneous moduli, feq, and tau are left unbound (None).
            When provided, valid 'equilibrium_fraction' and 'maxwell_relaxation_time_s'
            are strictly required from source_artifact['constitutive_model']['parameters'].

    Returns:
        Structured paper_material metadata dictionary.
    """
    if preset_name in ("ff_leap_compression", "material_ff_leap_compression"):
        name = "ff_leap_compression"
        label = "FF LEAP Compression Surrogate"
        mat_name = "FF LEAP"
        table_stress = FF_LEAP_TABLE_STRESS_KPA
        params = FF_LEAP_SURROGATE_PARAMS
    elif preset_name in ("ff_turbo_plus_compression", "material_ff_turbo_plus_compression"):
        name = "ff_turbo_plus_compression"
        label = "FF TURBO PLUS Compression Surrogate"
        mat_name = "FF TURBO PLUS"
        table_stress = FF_TURBO_PLUS_TABLE_STRESS_KPA
        params = FF_TURBO_PLUS_SURROGATE_PARAMS
    else:
        raise ValueError(f"Unknown paper material preset: {preset_name!r}")

    metrics = audit_fit_metrics(
        PAPER_TABLE_STRAIN,
        table_stress,
        params["mu1_eq_pa"],
        params["alpha1"],
        params["mu2_eq_pa"],
        params["alpha2"],
    )

    if source_artifact is None:
        feq: float | None = None
        tau: float | None = None
        mu1_inst: float | None = None
        mu2_inst: float | None = None
    else:
        if not isinstance(source_artifact, dict) or "constitutive_model" not in source_artifact:
            raise ValueError("source_artifact must be a dict containing 'constitutive_model'")
        p = source_artifact["constitutive_model"].get("parameters", {})
        if "equilibrium_fraction" not in p or "maxwell_relaxation_time_s" not in p:
            raise ValueError(
                "source_artifact['constitutive_model']['parameters'] must contain "
                "'equilibrium_fraction' and 'maxwell_relaxation_time_s'"
            )
        feq = float(p["equilibrium_fraction"])
        tau = float(p["maxwell_relaxation_time_s"])
        if feq <= 0.0 or feq > 1.0:
            raise ValueError(f"Invalid equilibrium_fraction in source: {feq}")
        mu1_inst = float(params["mu1_eq_pa"] / feq)
        mu2_inst = float(params["mu2_eq_pa"] / feq)

    surrogate_params = {
        "mu1_eq_pa": params["mu1_eq_pa"],
        "alpha1": params["alpha1"],
        "mu2_eq_pa": params["mu2_eq_pa"],
        "alpha2": params["alpha2"],
        "mu_eq_sum_pa": params["mu_eq_sum_pa"],
        "mu1_inst_pa": mu1_inst,
        "mu2_inst_pa": mu2_inst,
        "feq": feq,
        "tau_s": tau,
    }

    return {
        "name": name,
        "label": label,
        "material_name": mat_name,
        "sources": copy.deepcopy(PAPER_SOURCES),
        "literature_limit_strain": LITERATURE_LIMIT_STRAIN,
        "table_strain": list(PAPER_TABLE_STRAIN),
        "table_stress_kpa": list(table_stress),
        "surrogate_params": surrogate_params,
        "fit_metrics": {
            "rmse_kpa": metrics["rmse_kpa"],
            "max_err_kpa": metrics["max_err_kpa"],
            "max_err_strain": metrics["max_err_strain"],
            "stress_50pct_kpa": metrics["stress_50pct_kpa"],
            "pressure_curve_kpa": metrics["pressure_curve_kpa"],
            "error_curve_kpa": metrics["error_curve_kpa"],
        },
        "assumptions": list(SURROGATE_ASSUMPTIONS),
    }
