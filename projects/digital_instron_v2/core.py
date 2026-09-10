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
    """Reduced Hyperfoam-Maxwell foam with a material-pinned Pasternak layer.

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

    def __post_init__(self) -> None:
        if not np.all(np.isfinite(tuple(self.__dict__.values()))):
            raise ValueError("material parameters must be finite")
        if self.instantaneous_shear_modulus_pa <= 0.0 or self.hyperfoam_exponent <= 0.0:
            raise ValueError("shear modulus and Hyperfoam exponent must be positive")
        if not 0.0 < self.equilibrium_fraction <= 1.0:
            raise ValueError("equilibrium fraction must be in (0, 1]")
        if self.maxwell_relaxation_time_s <= 0.0:
            raise ValueError("Maxwell relaxation time must be positive")

    @property
    def equilibrium_shear_modulus_pa(self) -> float:
        """Equilibrium Ogden-Hill shear modulus ``mu_eq`` [Pa]."""
        return self.instantaneous_shear_modulus_pa * self.equilibrium_fraction

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
# This material does NOT pass the declared held-out gates, and that is the
# reported result rather than a defect to tune away. With the Poisson ratio
# measured at zero, the shear layer pinned to the material and the outer bond
# booked honestly, one shared material has no freedom left to reconcile the two
# bench fixtures: the objective is bimodal, and this is its lower-loss branch,
# which splits the error between the fixtures (rearfoot peak -15%, full-foot
# peak +19%). The other branch (mu_eq about 240 kPa, alpha about 11.4) reproduces
# the full-foot peak to 0.3% and misses the rearfoot by 21%. Section "What the
# fitted vector contains" in the project README records both.
CALIBRATED_MATERIAL = Material(
    instantaneous_shear_modulus_pa=74671.42399113576,
    hyperfoam_exponent=0.215500448269755,
    equilibrium_fraction=0.6954171811030112,
    maxwell_relaxation_time_s=0.005000150692608603,
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


def _hyperfoam_pressure(strain: np.ndarray, material: Material) -> np.ndarray:
    """Return positive uniaxial compression pressure from first-order Hyperfoam."""

    # The stretch floor keeps every exponentiation away from zero, so the
    # ``beta = 0`` case is the ordinary ``x ** 0 == 1`` and needs no special path.
    stretch = np.clip(1.0 - strain, 1.0e-3, 1.0)
    poisson = EFFECTIVE_POISSON_RATIO
    beta = poisson / (1.0 - 2.0 * poisson)
    volume_ratio = stretch ** (1.0 - 2.0 * poisson)
    pressure = 2.0 * material.equilibrium_shear_modulus_pa / (material.hyperfoam_exponent * stretch)
    pressure *= volume_ratio ** (-material.hyperfoam_exponent * beta) - stretch**material.hyperfoam_exponent
    return pressure


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
    )

    surround = trial.surround
    shoe = ShoeMaterial(
        material.instantaneous_shear_modulus_pa,
        material.hyperfoam_exponent,
        material.equilibrium_fraction,
        float(np.mean(material.coupling_n_per_m(surround.slack_m))),
        EFFECTIVE_POISSON_RATIO,
        material.maxwell_relaxation_time_s,
    )
    params = FoundationParams()
    poisson = shoe.effective_poisson_ratio
    params.g_eq = shoe.instantaneous_shear_modulus_pa * shoe.equilibrium_fraction
    params.alpha = shoe.hyperfoam_exponent
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


def fit_material(
    trials: list[Trial],
    initial: Material,
    evaluations: int,
    history: list[dict[str, float]] | None = None,
) -> Material:
    """Fit one material to all trials against force, loop area, and peak."""

    from scipy.optimize import least_squares

    def residual(values: np.ndarray) -> np.ndarray:
        material = Material(*values)
        return np.concatenate([_trial_residual(trial, material) for trial in trials])

    def record(values: np.ndarray) -> None:
        if history is None:
            return
        residuals = residual(values)
        row = {
            "iteration": float(len(history)),
            "loss": float(np.mean(residuals**2)),
            **{name: float(value) for name, value in zip(Material.__dataclass_fields__, values, strict=True)},
        }
        offset = 0
        for trial in trials:
            count = len(trial.force_n) + 2
            row[f"loss_{trial.name}"] = float(np.mean(residuals[offset : offset + count] ** 2))
            offset += count
        history.append(row)

    x0 = np.asarray(list(initial.__dict__.values()))
    record(x0)
    # Order matches Material: shear modulus, exponent, equilibrium fraction,
    # relaxation time. There is no fixture-specific parameter to bound.
    lower = [1.0e3, 0.1, 0.01, 5.0e-3]
    upper = [1.0e8, 20.0, 1.0, 2.0]
    result = least_squares(
        residual,
        x0,
        bounds=(lower, upper),
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
    return Material(*result.x)


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
