# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Digital Instron material model and fit."""

import logging
from dataclasses import dataclass

import numpy as np

# Confined and unconfined compression of racing-shoe midsole foam agree within
# scatter, so the effective Poisson ratio is zero (McCulloch, Delp and Kuhl,
# arXiv:2602.12694, sections 2.4 and 3.3). The Ogden-Hill exponent
# ``beta = nu / (1 - 2 nu)`` then vanishes and the column law reduces to the
# beta -> 0 Hyperfoam term the same paper fits, which is also self-consistent
# with modelling the midsole as independent columns.
EFFECTIVE_POISSON_RATIO = 0.0
MAXWELL_RELAXATION_TIME_S = 0.08

# Maximum passes of the surround fixed point: relax against the current
# overstress, refresh the overstress from the relaxed compression, repeat. The
# support a free column feels is equilibrium pressure plus Maxwell overstress,
# and the overstress follows from the compression the relaxation produces, so
# one pass is only self-consistent when the overstress vanishes. Convergence is
# slowest when the overstress dominates, which needs about ten passes.
#
# The pinned material sits in that slow region: the blended pass converges at
# about 0.63 per pass, so the cap binds before ``SURROUND_TOLERANCE_M`` does and
# the compression carries roughly 1.2e-5 m of tail. Raising the cap to 60 moves
# both measured peaks by 0.016% and the whole force history by 0.016% of peak,
# which is far below the 10% gates and below the float32 noise of the forward
# pass, so the cap is kept for a reproducible and affordable fit. Anything that
# differentiates this solve should use its own damped iteration, not this cap.
SURROUND_PASSES = 12

# Stop the surround passes once the compression field moves less than this
# between passes; 1 um is far below the 20 mm working range and below the
# float32 noise of the forward pass.
SURROUND_TOLERANCE_M = 1.0e-6

# Per-trial record of the last surround solve: the maximum compression change
# between consecutive passes [m], so a stalled fixed point stays visible.
SURROUND_CONVERGENCE: dict[str, dict[str, object]] = {}

# Last converged compression field per trial, reused as the warm start of the
# next solve. A fit spends almost all of its time on finite-difference
# perturbations that move the bed by microns, so restarting each of them from
# zero paid the full cold-solve cost hundreds of times over.
_SURROUND_WARM_START: dict[str, object] = {}

# Extrapolated remaining compression travel that ends a relaxation [m]. This is
# a bound on the distance still to travel, not the size of one update, so it is
# directly comparable with the 1 um tolerance of the outer fixed point above and
# with the working range of the bed. See
# :func:`projects.digital_shoe.runtime.relax_surround`.
SURROUND_SOLVE_TOLERANCE_M = 1.0e-8

# Sweeps between convergence tests. Each test costs one device reduction and one
# synchronization, so it must not be paid every sweep.
SURROUND_CHECK_EVERY = 25

# Fraction of the local Newton step each sweep takes. Kept at one: the sweep is a
# Jacobi iteration, whose stable over-relaxation range is only 2 / (1 + rho) and
# so is already exhausted at one. Measured on this bed, 1.5, 1.7 and 1.9 all
# diverge (the compression leaves the physical range by a factor of about seven
# and the sweep cap binds), while 1.0 converges. Real acceleration here needs a
# red-black Gauss-Seidel colouring of the bed, not a larger Jacobi step.
SURROUND_OVER_RELAXATION = 1.0

_LOGGER = logging.getLogger(__name__)


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

    The lateral shear layer is no longer a free parameter. A Pasternak layer
    coefficient is the shear modulus of the layer times its thickness, so every
    column gets ``k_i = mu_eq * t_i`` from :meth:`coupling_n_per_m`, where
    ``mu_eq`` is this material's own equilibrium Ogden-Hill shear modulus and
    ``t_i`` is that column's rest thickness. That adds no parameter and removes
    the flat direction the single fitted coefficient ran along.

    Nothing replaces it. The fitted vector is one shared material with no
    fixture-specific freedom of any kind, so the rearfoot punch and the full-foot
    last are described by the same four numbers and any disagreement between them
    stays visible in the residual instead of being absorbed by a knob.
    """

    instantaneous_shear_modulus_pa: float
    hyperfoam_exponent: float
    equilibrium_fraction: float
    maxwell_relaxation_time_s: float = MAXWELL_RELAXATION_TIME_S
    instantaneous_shear_modulus_2_pa: float = 0.0
    hyperfoam_exponent_2: float = 1.0

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

    @property
    def equilibrium_shear_modulus_pa(self) -> float:
        """Equilibrium Ogden-Hill shear modulus ``mu_eq``, the SUM over both terms [Pa].

        ``mu_eq = (G_1 + G_2) * f_eq``. Every term of an Ogden-Hill series
        contributes ``2 mu_n`` to the small-strain compressive tangent whatever
        its exponent, so the series modulus is the sum of the term moduli and not
        the first term alone. The per-column Pasternak rule
        :meth:`coupling_n_per_m` and the reported small-strain modulus both use
        this sum.
        """
        return (self.instantaneous_shear_modulus_pa + self.instantaneous_shear_modulus_2_pa) * (
            self.equilibrium_fraction
        )

    def coupling_n_per_m(self, thickness_m: np.ndarray | float) -> np.ndarray | float:
        """Return the Pasternak coefficient of a column of this rest thickness [N/m].

        The equilibrium branch is used, not the instantaneous one: the shear
        layer carries no Maxwell branch of its own, and the measured shear
        moduli it is checked against come from a 0.16 Hz sweep, so pairing it
        with the instantaneous modulus would count the same rate effect twice.

        Args:
            thickness_m: Column rest thickness [m].
        """
        return self.equilibrium_shear_modulus_pa * np.asarray(thickness_m, dtype=float)


# Fitted intact-shoe parameters produced by the checked-in Digital Instron
# calibration workflow. Keep the fit seed in the manifest separate from this
# prediction baseline.
#
# This material passes all six declared held-out gates (rearfoot peak 2.8%,
# RMSE 5.6%, loop 7.5%; full-foot peak 5.9%, RMSE 6.4%, loop 8.8%), which the
# single-term law did not: it reached 2 of 6 and split the peak error between
# the fixtures at -15% and +19%. The reason is the strain range, not extra
# freedom to absorb a disagreement. The rearfoot punch reaches about 90% strain
# and the full-foot last about 74%, and one first-order term has a single shape
# exponent for both, so its objective was bimodal (mu_eq 52 kPa with alpha 0.22
# against mu_eq 240 kPa with alpha 11.4 at nearly equal loss). With two terms a
# multi-start from both of those basins converges to this one optimum; see
# ``MULTISTART_SEEDS`` and the fitted-vector section of the project README.
CALIBRATED_MATERIAL = Material(
    instantaneous_shear_modulus_pa=267615.1977311966,
    hyperfoam_exponent=18.07697643908876,
    equilibrium_fraction=0.6586721149134733,
    maxwell_relaxation_time_s=0.0051501095220815776,
    instantaneous_shear_modulus_2_pa=19487.70413858932,
    hyperfoam_exponent_2=-0.5855868486256809,
)


@dataclass(frozen=True)
class Surround:
    """Whole-midsole geometry shared by the identification and the runtime.

    ``driven`` marks the columns the indenter presses directly. The remaining
    columns are real foam that deforms only through the lateral shear layer and
    its bond to the shoe above, exactly as the live runtime treats them. Fitting
    against a rigid zero-deflection boundary instead would identify the coupling
    against a stiffer surround than the one that is later simulated.
    """

    driven: np.ndarray  # bool [column_count]
    neighbors: np.ndarray  # int [column_count, 4]; -1 is a free outer edge
    slack_m: np.ndarray  # rest thickness [column_count]
    area_m2: float  # tributary area per column
    spacing_m: float
    # The outer bond is off by default. It used to hold the relaxation up at
    # 200 N/m per column while contributing nothing to the reported force, an
    # undeclared rigid support worth about 244 N of a 337 N rearfoot peak. With
    # it removed the summed unilateral ground reaction is the whole applied load,
    # because the lateral flux cancels exactly over the bed.
    attachment_n_m: float = 0.0
    max_strain: float = 0.9
    # A sparse indenter needs far more sweeps than a broad one: the 62-column
    # rearfoot punch is still 8% low at 250 sweeps while the 611-column last is
    # already converged. Fitting against a half-relaxed bed lets the optimizer
    # trade solver error against material parameters.
    sweeps: int = 3000

    def __post_init__(self) -> None:
        count = len(self.slack_m)
        if self.driven.shape != (count,) or self.neighbors.shape != (count, 4):
            raise ValueError("surround geometry must describe the same column count")
        if not np.any(self.driven) or self.area_m2 <= 0.0 or self.spacing_m <= 0.0:
            raise ValueError("surround needs driven columns and positive geometry")
        if self.attachment_n_m < 0.0 or not 0.0 < self.max_strain < 1.0 or self.sweeps < 1:
            raise ValueError("surround relaxation settings are out of range")


@dataclass(frozen=True)
class Trial:
    """Measured force and matching column lengths."""

    name: str
    slack_m: np.ndarray
    area_m2: np.ndarray | float
    lengths_m: np.ndarray
    dt_s: np.ndarray
    force_n: np.ndarray
    displacement_m: np.ndarray
    compression_laplacian_m_inv: np.ndarray | None = None
    surround: Surround | None = None


# Below this exponent magnitude one Ogden-Hill term is evaluated at its removable
# ``alpha -> 0`` limit, matching
# :data:`projects.digital_shoe.runtime.HYPERFOAM_ALPHA_FLOOR`.
HYPERFOAM_ALPHA_FLOOR = 1.0e-3


def _hyperfoam_term(
    stretch: np.ndarray, volume_ratio: np.ndarray, mu_pa: float, alpha: float, beta: float
) -> np.ndarray:
    """Return one Ogden-Hill (Hyperfoam) uniaxial compression term [Pa].

    Host twin of :func:`projects.digital_shoe.runtime._hyperfoam_term`; the two
    are pinned together by ``test_hyperfoam_pressure_matches_reference``.

    Args:
        stretch: Remaining thickness stretch ``lambda`` [-], already floored.
        volume_ratio: ``J = lambda^(1 - 2 nu)`` [-].
        mu_pa: Term shear modulus [Pa].
        alpha: Term exponent [-]; may be negative and is evaluated at its
            removable ``alpha -> 0`` limit near zero.
        beta: ``nu / (1 - 2 nu)`` [-].
    """
    if abs(alpha) < HYPERFOAM_ALPHA_FLOOR:
        return 2.0 * mu_pa / stretch * (-beta * np.log(volume_ratio) - np.log(stretch))
    return 2.0 * mu_pa / (alpha * stretch) * (volume_ratio ** (-alpha * beta) - stretch**alpha)


def _hyperfoam_pressure(strain: np.ndarray, material: Material) -> np.ndarray:
    """Return positive uniaxial compression pressure from the two-term Hyperfoam law."""

    # The stretch floor keeps every exponentiation away from zero, so the
    # ``beta = 0`` case is the ordinary ``x ** 0 == 1`` and needs no special path.
    stretch = np.clip(1.0 - strain, 1.0e-3, 1.0)
    poisson = EFFECTIVE_POISSON_RATIO
    beta = poisson / (1.0 - 2.0 * poisson)
    volume_ratio = stretch ** (1.0 - 2.0 * poisson)
    fraction = material.equilibrium_fraction
    return _hyperfoam_term(
        stretch, volume_ratio, material.instantaneous_shear_modulus_pa * fraction, material.hyperfoam_exponent, beta
    ) + _hyperfoam_term(
        stretch,
        volume_ratio,
        material.instantaneous_shear_modulus_2_pa * fraction,
        material.hyperfoam_exponent_2,
        beta,
    )


def _periodic_maxwell_branch(
    equilibrium_pressure: np.ndarray,
    dt_s: np.ndarray,
    fraction: float,
    relaxation_time_s: float,
) -> np.ndarray:
    """Evaluate one linear overstress branch at its exact cycle fixed point."""

    if fraction == 0.0:
        return np.zeros_like(equilibrium_pressure)
    decay = np.exp(-dt_s / relaxation_time_s)
    ramp = relaxation_time_s * (1.0 - decay) / dt_s
    pressure_increment = equilibrium_pressure - np.roll(equilibrium_pressure, 1, axis=0)
    state = np.zeros(equilibrium_pressure.shape[1])
    for frame in range(len(equilibrium_pressure)):
        state = decay[frame] * state + fraction * ramp[frame] * pressure_increment[frame]
    state /= 1.0 - float(np.prod(decay))
    result = np.empty_like(equilibrium_pressure)
    for frame in range(len(equilibrium_pressure)):
        state = decay[frame] * state + fraction * ramp[frame] * pressure_increment[frame]
        result[frame] = state
    return result


def _surround_force(trial: Trial, material: Material) -> np.ndarray:
    """Run the whole cycle on the GPU: relax the surround self-consistently, then sum the reaction.

    The relaxation and the summed reaction share one overstress field, refreshed
    between passes until it settles, so the free columns are placed under the
    load they actually carry instead of under the equilibrium pressure alone.
    """
    import warp as wp  # noqa: PLC0415  # lazy: the identification stays importable without a device

    from projects.digital_shoe.runtime import (  # noqa: PLC0415
        FoundationParams,
        ShoeMaterial,
        cycle_force,
        cycle_overstress,
        relax_surround,
        set_hyperfoam_series,
    )

    surround = trial.surround
    shoe = ShoeMaterial(
        material.instantaneous_shear_modulus_pa,
        material.hyperfoam_exponent,
        material.equilibrium_fraction,
        float(np.mean(material.coupling_n_per_m(surround.slack_m))),
        EFFECTIVE_POISSON_RATIO,
        material.maxwell_relaxation_time_s,
        material.instantaneous_shear_modulus_2_pa,
        material.hyperfoam_exponent_2,
    )
    params = FoundationParams()
    poisson = shoe.effective_poisson_ratio
    set_hyperfoam_series(params, shoe)
    params.beta = poisson / (1.0 - 2.0 * poisson)
    params.one_minus_two_poisson = 1.0 - 2.0 * poisson
    params.stretch_floor = 1.0e-3
    driven_compression = np.ascontiguousarray(np.maximum(trial.slack_m[None, :] - trial.lengths_m, 0.0), np.float32)
    frames = len(trial.dt_s)
    count = len(surround.slack_m)
    fraction = float((1.0 - material.equilibrium_fraction) / material.equilibrium_fraction)
    tau_s = float(material.maxwell_relaxation_time_s)

    # Undamped repetition of "relax against q, then recompute q" has loop gain
    # -(1 - eq) / eq, so it oscillates whenever the overstress exceeds the
    # equilibrium pressure - exactly the long-relaxation-time region the fit
    # must stay free to visit. Blending the refreshed overstress with weight
    # ``equilibrium_fraction = 1 / (1 + overstress fraction)`` cancels that gain
    # to first order and leaves the fixed point unchanged.
    blend = float(material.equilibrium_fraction)
    slack_device = None
    dt_device = None
    overstress_host = np.zeros((frames, count), np.float32)
    carried = None
    refreshed = None
    compression = None
    previous = None
    changes: list[float] = []
    # Warm start every pass from the previous one, and the first pass of this
    # solve from the previous solve of the same trial. Both fields are already
    # close to the answer, so the residual test below ends each relaxation in a
    # few sweeps instead of the cold-solve thousands.
    warm = _SURROUND_WARM_START.get(trial.name)
    if warm is not None and tuple(warm.shape) != (frames, count):
        warm = None
    solver_stats: dict[str, float] = {}
    sweeps_used: list[int] = []
    for _ in range(SURROUND_PASSES):
        compression = relax_surround(
            driven_compression,
            surround.driven,
            surround.neighbors,
            surround.slack_m,
            params,
            area_m2=surround.area_m2,
            spacing_m=surround.spacing_m,
            attachment_n_m=surround.attachment_n_m,
            max_strain=surround.max_strain,
            sweeps=surround.sweeps,
            over_relaxation=SURROUND_OVER_RELAXATION,
            overstress=carried,
            initial=warm,
            tolerance_m=SURROUND_SOLVE_TOLERANCE_M,
            check_every=SURROUND_CHECK_EVERY,
            stats=solver_stats,
        )
        warm = compression
        sweeps_used.append(int(solver_stats["sweeps"]))
        device = compression.device
        if slack_device is None:
            slack_device = wp.array(np.ascontiguousarray(surround.slack_m, np.float32), dtype=wp.float32, device=device)
            dt_device = wp.array(np.ascontiguousarray(trial.dt_s, np.float32), dtype=wp.float32, device=device)
        refreshed = wp.zeros((frames, count), dtype=wp.float32, device=device)
        wp.launch(
            cycle_overstress,
            dim=count,
            inputs=[compression, slack_device, dt_device, params, fraction, tau_s, refreshed],
            device=device,
        )
        overstress_host += blend * (refreshed.numpy() - overstress_host)
        carried = wp.array(overstress_host, dtype=wp.float32, device=device)
        relaxed = compression.numpy()
        if previous is not None:
            changes.append(float(np.max(np.abs(relaxed - previous))))
        previous = relaxed
        if changes and changes[-1] < SURROUND_TOLERANCE_M:
            break
    _SURROUND_WARM_START[trial.name] = compression
    SURROUND_CONVERGENCE[trial.name] = {
        "pass_change_m": changes,
        "max_compression_m": float(np.max(previous)),
        "sweeps_per_pass": sweeps_used,
        "solver_remaining_m": solver_stats.get("remaining_m", float("nan")),
    }
    _LOGGER.debug(
        "%s surround self-consistency: pass changes %s m, max compression %.4f m",
        trial.name,
        [f"{value:.2e}" for value in changes],
        float(np.max(previous)),
    )

    # The reported load uses the overstress of the final compression itself, so
    # the summed reaction is exact for the geometry that was relaxed.
    force = wp.zeros(frames, dtype=wp.float32, device=compression.device)
    wp.launch(
        cycle_force,
        dim=(frames, count),
        inputs=[compression, refreshed, slack_device, params, float(surround.area_m2), force],
        device=compression.device,
    )
    return force.numpy().astype(np.float64)


def predict(trial: Trial, material: Material) -> np.ndarray:
    """Predict a trial force history.

    With a :class:`Surround` the whole midsole is modelled: the indenter drives
    its own columns and the rest relax, so the identification and the live
    runtime see the same geometry and contact. The measured load then equals the
    summed unilateral ground reaction, because the shear flux cancels internally
    and no hidden bond carries part of the load.
    """

    if trial.surround is not None:
        return _surround_force(trial, material)
    return _column_bed_force(trial, material, trial.lengths_m)


def _column_bed_force(trial: Trial, material: Material, lengths_m: np.ndarray) -> np.ndarray:
    """Sum the reaction of an uncoupled column set at a given length history.

    The fixture-subset path kept for tests and plotting. Its lateral term is the
    lumped ``mu_eq * t_i`` coefficient of :meth:`Material.coupling_n_per_m` acting
    on the subset Laplacian; unlike the whole-bed path it is only force
    conserving for a bed of uniform rest thickness.

    Args:
        trial: Trial supplying the driven-column geometry and timing.
        material: Material to evaluate.
        lengths_m: Driven-column lengths per frame [m].
    """
    slack = np.asarray(trial.slack_m, dtype=float)
    strain = np.maximum(slack[None, :] - lengths_m, 0.0) / slack[None, :]
    equilibrium = _hyperfoam_pressure(strain, material)
    pressure = np.array(equilibrium, copy=True)
    maxwell_fraction = (1.0 - material.equilibrium_fraction) / material.equilibrium_fraction
    pressure += _periodic_maxwell_branch(equilibrium, trial.dt_s, maxwell_fraction, material.maxwell_relaxation_time_s)
    # Clamp only the unilateral ground reaction, then add the shear-layer flux.
    # Clamping their sum would clip the flux and invent support under columns
    # that carry no compression, so the runtime uses this same order.
    ground = np.maximum(pressure, 0.0)
    if trial.compression_laplacian_m_inv is not None:
        ground = ground - material.coupling_n_per_m(slack)[None, :] * trial.compression_laplacian_m_inv
    return np.sum(trial.area_m2 * ground, axis=1)


HYSTERESIS_WEIGHT = 5.0
PEAK_WEIGHT = 6.0


def _trial_residual(trial: Trial, material: Material) -> np.ndarray:
    """Return one trial's force, loop-area, and peak residuals.

    Force error alone barely notices the hysteresis loop, because the loop is
    small next to the peak load. The declared gates score peak force and loop
    area as well, so the identification objective scores all three.
    """
    predicted = predict(trial, material)
    scale = max(float(np.max(trial.force_n)), 1.0)
    measured_loop = float(np.trapezoid(trial.force_n, trial.displacement_m))
    loop_error = float(np.trapezoid(predicted, trial.displacement_m)) - measured_loop
    return np.concatenate(
        [
            (predicted - trial.force_n) / scale,
            [HYSTERESIS_WEIGHT * loop_error / max(abs(measured_loop), 1.0e-9)],
            [PEAK_WEIGHT * (float(np.max(predicted)) - float(np.max(trial.force_n))) / scale],
        ]
    )


# Order matches the :class:`Material` field order: first-term shear modulus,
# first-term exponent, equilibrium fraction, relaxation time, second-term shear
# modulus, second-term exponent. There is no fixture-specific parameter to bound.
#
# The second modulus may reach zero, which turns the series back into the
# single-term law exactly. The second exponent may be negative: an Ogden-Hill
# series admits either sign, a positive exponent produces the soft ``1 / lambda``
# plateau and a negative one produces densification, and the published two-term
# fits of this foam family use one of each. Zero is inside the interval and is
# handled by the removable limit in :func:`_hyperfoam_term`.
FIT_LOWER_BOUNDS = (1.0e3, 0.1, 0.01, 5.0e-3, 0.0, -20.0)
FIT_UPPER_BOUNDS = (1.0e8, 20.0, 1.0, 2.0, 1.0e8, 20.0)

# Optional extra starting points, used only when a caller asks for them. They
# cover both basins of the SINGLE-term objective -- one near ``alpha = 0.2`` with
# ``mu_eq`` about 52 kPa and one near ``alpha = 11`` with ``mu_eq`` about 240 kPa,
# at nearly equal training loss -- give the second term both roles (a soft
# plateau partner and a densifying partner) in each, and add the two two-term
# fits recovered from the published FF LEAP and FF TURBO PLUS compression tables.
MULTISTART_SEEDS: tuple[Material, ...] = (
    # Soft-exponent basin, densifying second term.
    Material(74671.4, 0.2155, 0.6954, 0.0050, 15000.0, -1.0),
    # Soft-exponent basin, plateau second term.
    Material(74671.4, 0.2155, 0.6954, 0.0050, 150000.0, 8.0),
    # Stiff-exponent basin, densifying second term.
    Material(345000.0, 11.4, 0.6954, 0.0050, 30000.0, -1.5),
    # Stiff-exponent basin, soft second term.
    Material(345000.0, 11.4, 0.6954, 0.0050, 75000.0, 0.3),
    # Published FF LEAP two-term compression fit, divided by the equilibrium
    # fraction to become instantaneous moduli.
    Material(247000.0, 8.39, 0.6954, 0.0050, 26200.0, -1.04),
    # Published FF TURBO PLUS two-term compression fit, same conversion.
    Material(243000.0, 5.65, 0.6954, 0.0050, 17300.0, -2.00),
)


def fit_material(
    trials: list[Trial],
    initial: Material,
    evaluations: int,
    history: list[dict[str, float]] | None = None,
    starts: tuple[Material, ...] = (),
    multistart_seeds: int = 0,
    bounds: tuple[tuple[float, ...], tuple[float, ...]] | None = None,
) -> Material:
    """Fit one material to all trials against force, loop area, and peak.

    Single start by default. A bounded least-squares descent is run from every
    requested start and the lowest final loss wins.

    Multi-start is opt-in rather than the default because the bimodality that
    motivated it belongs to the *single-term* law, not to this one. With one term
    the objective had two basins of nearly equal loss that differed almost
    entirely in the shape exponent. With two terms a seven-start check found five
    of seven seeds -- from both of those basins and from the published two-term
    compression fits -- converging on the same optimum within about 4% in
    ``mu_eq``, and only the seed that starts with the second term disabled stayed
    behind, at 3.9 times the loss. Paying a sevenfold cost on every fit and every
    test run to defend against a defect the model form removed is not justified.

    That makes unimodality an assumption. It is cheap to re-test -- about 0.09 s
    per residual evaluation warm-started -- and it MUST be re-tested with
    ``multistart_seeds`` set whenever the constitutive form, the objective, the
    bounds, or the fixture set changes, before the result of that change is
    trusted.

    Args:
        trials: Measured trials to fit jointly with one shared material.
        initial: Seed material; always used as the first start.
        evaluations: Maximum residual evaluations per start.
        history: Optional list that receives one row per accepted iteration of
            every start, with a ``start`` column identifying which one.
        starts: Explicit extra starts after ``initial``.
        multistart_seeds: Number of :data:`MULTISTART_SEEDS` to append to
            ``starts``; zero (the default) leaves the fit single start.
        bounds: Lower and upper parameter bounds; defaults to
            :data:`FIT_LOWER_BOUNDS` and :data:`FIT_UPPER_BOUNDS`. Widening them
            is how a bound-proximity check is run.
    """

    from scipy.optimize import least_squares

    def residual(values: np.ndarray) -> np.ndarray:
        material = Material(*values)
        return np.concatenate([_trial_residual(trial, material) for trial in trials])

    def loss(values: np.ndarray) -> float:
        return float(np.mean(residual(values) ** 2))

    lower, upper = (FIT_LOWER_BOUNDS, FIT_UPPER_BOUNDS) if bounds is None else bounds
    if not 0 <= multistart_seeds <= len(MULTISTART_SEEDS):
        raise ValueError(f"multistart_seeds must be in [0, {len(MULTISTART_SEEDS)}]")
    seeds = [initial, *starts, *MULTISTART_SEEDS[:multistart_seeds]]
    best: np.ndarray | None = None
    best_loss = np.inf
    for index, seed in enumerate(seeds):
        x0 = np.asarray(list(seed.__dict__.values()))

        def record(values: np.ndarray, index: int = index) -> None:
            if history is None:
                return
            residuals = residual(values)
            row = {
                "iteration": float(len(history)),
                "start": float(index),
                "loss": float(np.mean(residuals**2)),
                **{name: float(value) for name, value in zip(Material.__dataclass_fields__, values, strict=True)},
            }
            offset = 0
            for trial in trials:
                count = len(trial.force_n) + 2
                row[f"loss_{trial.name}"] = float(np.mean(residuals[offset : offset + count] ** 2))
                offset += count
            history.append(row)

        record(x0)
        result = least_squares(
            residual,
            x0,
            bounds=(list(lower), list(upper)),
            x_scale="jac",
            # The GPU forward pass is float32, so the default finite-difference step
            # sits in its rounding noise and the search stalls at the seed.
            diff_step=1.0e-3,
            max_nfev=evaluations,
            callback=lambda intermediate: record(np.asarray(getattr(intermediate, "x", intermediate))),
        )
        if history is not None and not np.array_equal(
            result.x, [history[-1][name] for name in Material.__dataclass_fields__]
        ):
            record(result.x)
        final = loss(result.x)
        _LOGGER.info("start %d/%d finished at loss %.6g", index + 1, len(seeds), final)
        if final < best_loss:
            best_loss, best = final, result.x
    return Material(*best)


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
