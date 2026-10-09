# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Levenberg-Marquardt identification of shared runner parameters.

The cost is the sample part of :func:`~projects.impedance_instron.hogan.identify.score`
(coordinate and GRF mean squares, averaged over trials) plus offset
regularization, written as a residual vector. Finite-difference Jacobians and
the parallel damping ladder each run as one batched rollout; only the small
damped normal-equation solve runs on the host.

:class:`newton.ik.IKOptimizerLM` is not used: it optimizes articulation joint
coordinates against FK objectives and tiles the full Jacobian per row, whereas
here parameters drive a contact rollout and the Jacobian has ~10^4-10^5 rows.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np

from .identify import Parameterization, Trial, evaluate, observed_grf, predict_many, score, vibration_weight
from .runner import RolloutConfig, Runner

_HIP_SCALE_M = 0.02
_ANGLE_SCALE_RAD = 0.05
_FORCE_SCALE_N = 100.0


@dataclass(frozen=True)
class LMConfig:
    """Levenberg-Marquardt settings; selection uses training trials only."""

    iterations: int = 15
    step: float = 0.01
    """Finite-difference step in offset units."""
    central: bool = False
    """Use central instead of forward differences (twice the rollouts)."""
    damping: float = 1e-2
    """Initial Marquardt damping relative to diag(J^T J)."""
    ladder: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)
    """Damping multipliers evaluated in parallel each iteration."""
    bound: float = 1.5
    regularization: float = 0.01
    tolerance: float = 1e-4
    """Stop when an accepted step improves the cost by less than this fraction."""
    chunk: int = 128
    """Candidates per batched rollout."""
    fast_jacobian: bool = True
    """Integrate finite-difference rollouts, and their own reference, with fast-math
    shoe kernels on CUDA. Costs, damping-ladder proposals, and accepted steps stay
    exact; only the search direction carries float32 intrinsic rounding."""

    def __post_init__(self):
        if self.iterations < 1 or self.chunk < 1 or not self.ladder:
            raise ValueError("iterations, chunk, and ladder must be positive and nonempty")
        values = (self.step, self.damping, self.bound, self.regularization, self.tolerance, *self.ladder)
        if not np.isfinite(values).all() or min(self.step, self.damping, self.bound, *self.ladder) <= 0:
            raise ValueError("LM step, damping, bound, and ladder must be finite and positive")
        if self.regularization < 0 or self.tolerance < 0:
            raise ValueError("regularization and tolerance must be nonnegative")


def residuals(trace: dict, summary: dict, trial: Trial) -> np.ndarray | None:
    """Return weighted sample residuals whose squared sum equals the score's sample terms.

    Matches the coordinate and GRF mean-square terms of :func:`identify.score`;
    peak, impulse, contact-duration and effort terms are excluded. The GRF is
    compared through the trial's force observation, if any, followed by its
    weighted unobservable vibration. Returns ``None`` for failed rollouts.
    """
    if summary["status"] != "completed":
        return None
    time = trace["time_s"]
    observed = (trial.time_s > 0) & (trial.time_s <= time[-1] + 1e-12)
    parts = []
    count = int(observed.sum())
    if count:
        simulated = np.column_stack([np.interp(trial.time_s[observed], time, trace["state"][:, c]) for c in range(6)])
        error = simulated - trial.q[observed]
        parts.append((error[:, :2] / (_HIP_SCALE_M * math.sqrt(2 * count))).ravel())
        parts.append((error[:, 2:] / (_ANGLE_SCALE_RAD * math.sqrt(4 * count))).ravel())
    steps = len(trace["grf_n"])
    if steps:
        measured = np.column_stack([np.interp(time[:-1], trial.force_time_s, trial.grf_n[:, c]) for c in range(2)])
        grf = observed_grf(trace, trial)
        scale = _FORCE_SCALE_N * math.sqrt(2 * steps)
        parts.append(((grf - measured) / scale).ravel())
        weight = vibration_weight(trial)
        if weight > 0:
            parts.append((weight * (trace["grf_n"] - grf) / scale).ravel())
    return np.concatenate(parts) if parts else np.zeros(0)


def motion_metrics(scores: list[dict]) -> dict:
    """Average per-stance errors in physical units over trials."""
    tracking = np.array([s["tracking_rmse"] for s in scores])
    force = np.array([s["grf_rmse_n"] for s in scores])
    return {
        "hip_rmse_m": tracking[:, :2].mean(0).tolist(),
        "joint_rmse_rad": float(tracking[:, 2:].mean()),
        "grf_rmse_n": force.mean(0).tolist(),
        "peak_fz_error_n": float(np.mean([s["peak_fz_error_n"] for s in scores])),
        "failed": sum(s["status"] != "completed" for s in scores),
    }


def _describe(metrics: dict) -> str:
    hip, force = metrics["hip_rmse_m"], metrics["grf_rmse_n"]
    return (
        f"hip x/y {1e3 * hip[0]:.0f}/{1e3 * hip[1]:.0f} mm, joints {math.degrees(metrics['joint_rmse_rad']):.1f} deg,"
        f" Fx/Fz {force[0]:.0f}/{force[1]:.0f} N, peak Fz {metrics['peak_fz_error_n']:+.0f} N"
    )


class _Rollouts:
    """Evaluate stacked residual vectors on the host reference backend."""

    def __init__(self, parameters: Parameterization, trials: list[Trial], config: RolloutConfig, device: str):
        self.parameters, self.trials, self.config, self.device = parameters, trials, config, device
        self.rollouts = 0

    def _predict(self, models: list[Runner]) -> list:
        return predict_many(models, self.trials, self.config, device=self.device)

    def __call__(self, offsets: np.ndarray, chunk: int, *, metrics: bool = False) -> list[np.ndarray | None]:
        """Return one residual vector per offset row, ``None`` if any trial fails.

        With ``metrics``, :attr:`metrics` holds :func:`motion_metrics` per row.
        """
        weight = 1.0 / math.sqrt(len(self.trials))
        result = []
        self.metrics = []
        for start in range(0, len(offsets), chunk):
            block = offsets[start : start + chunk]
            models = [self.parameters.model(x) for x in block]
            rows = self._predict(models)
            self.rollouts += len(block) * len(self.trials)
            for model, row in zip(models, rows, strict=True):
                if metrics:
                    self.metrics.append(
                        motion_metrics(
                            [score(*pair, trial, model) for trial, pair in zip(self.trials, row, strict=True)]
                        )
                    )
                values = [
                    residuals(trace, summary, trial) for trial, (trace, summary) in zip(self.trials, row, strict=True)
                ]
                result.append(None if any(v is None for v in values) else weight * np.concatenate(values))
        return result


def _columns(size: int, step: float, central: bool, plus_ok, minus_ok):
    """Choose each finite-difference column's operands; failed perturbations fall back."""
    plus, minus, denominator = [], [], []
    for i in range(size):
        if plus_ok[i] and (not central or minus_ok[i]):
            plus.append(("plus", i))
            minus.append(("minus", i) if central else ("reference", i))
            denominator.append(2 * step if central else step)
        elif central and (plus_ok[i] or minus_ok[i]):
            plus.append(("plus", i) if plus_ok[i] else ("reference", i))
            minus.append(("reference", i) if plus_ok[i] else ("minus", i))
            denominator.append(step)
        else:
            plus.append(None)
            minus.append(None)
            denominator.append(0.0)
    return plus, minus, denominator


class _HostEngine:
    """Keep residual vectors and the Jacobian in host memory (CPU reference)."""

    def __init__(self, parameters: Parameterization, trials: list[Trial], config: RolloutConfig, search: LMConfig):
        self.parameters, self.search = parameters, search
        self.rollouts_runner = _Rollouts(parameters, trials, config, "cpu")

    @property
    def rollouts(self) -> int:
        return self.rollouts_runner.rollouts

    def start(self, x: np.ndarray):
        value = self.rollouts_runner(x[None], 1, metrics=True)[0]
        if value is None:
            return None
        self.r0 = value
        return float(value @ value), self.rollouts_runner.metrics[0]

    def jacobian(self, x: np.ndarray):
        size, search = self.parameters.size, self.search
        eye = np.eye(size) * search.step
        perturbed = np.vstack((x + eye, x - eye)) if search.central else x + eye
        values = self.rollouts_runner(perturbed, search.chunk)
        plus_ok = [v is not None for v in values[:size]]
        minus_ok = [v is not None for v in values[size:]] if search.central else [True] * size
        sources = {"plus": values[:size], "minus": values[size:], "reference": [self.r0] * size}
        plus, minus, denominator = _columns(size, search.step, search.central, plus_ok, minus_ok)
        jacobian = np.zeros((len(self.r0), size))
        for i, (a, b, den) in enumerate(zip(plus, minus, denominator, strict=True)):
            if den:
                jacobian[:, i] = (sources[a[0]][i] - sources[b[0]][i]) / den
        return jacobian.T @ jacobian, jacobian.T @ self.r0, sum(not den for den in denominator)

    def trial(self, proposals: np.ndarray):
        self._candidates = self.rollouts_runner(proposals, len(proposals), metrics=True)
        sums = [math.inf if v is None else float(v @ v) for v in self._candidates]
        return sums, self.rollouts_runner.metrics

    def accept(self, index: int) -> None:
        self.r0 = self._candidates[index]


class _DeviceEngine:
    """Keep residual rows, the Jacobian, and ``J^T J`` on the CUDA device."""

    def __init__(
        self, parameters: Parameterization, trials: list[Trial], config: RolloutConfig, search: LMConfig, device
    ):
        from .gpu_residuals import GpuResiduals  # noqa: PLC0415 - optional execution backend

        self.parameters, self.search = parameters, search
        size = parameters.size
        jacobian_rows = 2 * size if search.central else size
        # Row ``size`` briefly holds the reference for the fused J^T J / J^T r product.
        self.ladder_row = max(jacobian_rows, size + 1)
        self.reference_row = self.ladder_row + len(search.ladder)
        # Fast differences need a reference with the same numerics as their perturbations.
        self.fast_reference_row = self.reference_row + 1
        self.objective = GpuResiduals(
            trials, config, rows=self.fast_reference_row + 1, device=device, chunk=search.chunk
        )

    @property
    def rollouts(self) -> int:
        return self.objective.rollouts

    def _models(self, offsets):
        return [self.parameters.model(x) for x in offsets]

    def start(self, x: np.ndarray):
        completed, sums, motion = self.objective.evaluate(self._models(x[None]), [self.reference_row], metrics=True)
        if not completed[0]:
            return None
        return float(sums[0]), motion_metrics(motion[0])

    def _differences(self, x: np.ndarray, fast: bool):
        """Integrate perturbations into rows ``[0, n)``; return completion flags and the reference row."""
        size, search = self.parameters.size, self.search
        eye = np.eye(size) * search.step
        perturbed = np.vstack((x + eye, x - eye)) if search.central else x + eye
        rows = np.arange(len(perturbed))
        if not fast:
            completed, _, _ = self.objective.evaluate(self._models(perturbed), rows)
            return completed, self.reference_row
        models = self._models(np.vstack((perturbed, x[None])))
        rows = np.append(rows, self.fast_reference_row)
        completed, _, _ = self.objective.evaluate(models, rows, fast=True)
        if not completed[-1]:
            # The exact reference completed; redo the same batch, and group, exactly.
            completed, _, _ = self.objective.evaluate(models, rows)
        return completed[:-1], self.fast_reference_row

    def jacobian(self, x: np.ndarray):
        size, search = self.parameters.size, self.search
        completed, reference = self._differences(x, search.fast_jacobian)
        plus_ok = completed[:size]
        minus_ok = completed[size:] if search.central else np.ones(size, dtype=bool)
        plus, minus, denominator = _columns(size, search.step, search.central, plus_ok, minus_ok)
        rows = {"plus": lambda i: i, "minus": lambda i: size + i, "reference": lambda i: reference}
        plus_rows = [0 if a is None else rows[a[0]](a[1]) for a in plus]
        minus_rows = [0 if b is None else rows[b[0]](b[1]) for b in minus]
        # The gradient always uses the exact residual of the current iterate.
        normal, gradient = self.objective.normal(plus_rows, minus_rows, denominator, self.reference_row)
        return normal, gradient, sum(not den for den in denominator)

    def trial(self, proposals: np.ndarray):
        rows = self.ladder_row + np.arange(len(proposals))
        completed, sums, motion = self.objective.evaluate(self._models(proposals), rows, metrics=True)
        return [float(v) if ok else math.inf for ok, v in zip(completed, sums, strict=True)], [
            motion_metrics(m) for m in motion
        ]

    def accept(self, index: int) -> None:
        self.objective.copy_row(self.ladder_row + index, self.reference_row)


def fit_lm(
    baseline: Runner,
    trials: list[Trial],
    *,
    config: RolloutConfig | None = None,
    search: LMConfig | None = None,
    allow_incompatible: bool = False,
    device: str = "cpu",
) -> tuple[Runner, dict]:
    """Fit on training trials with Levenberg-Marquardt; evaluate held-out trials afterwards.

    Each iteration evaluates the Jacobian and all damping-ladder proposals as
    batched rollouts, accepts the lowest-cost completed proposal if it improves
    the cost, and otherwise increases damping. Failed perturbations fall back to
    a one-sided difference or a zero column for that iteration. On CUDA, the
    residuals, Jacobian, and ``J^T J`` stay on the device; only the small damped
    normal-equation solve runs on the host. With :attr:`LMConfig.fast_jacobian`,
    the finite differences use fast-math shoe kernels while every reported cost
    and accepted step remains an exact rollout.
    """
    cfg, search = config or RolloutConfig(), search or LMConfig()
    train = [trial for trial in trials if trial.split == "train"]
    held_out = [trial for trial in trials if trial.split == "eval"]
    if not train:
        raise ValueError("Identification requires training trials")
    incompatible = [trial.id for trial in trials if not trial.provenance["compatibility"]["passed"]]
    if incompatible and not allow_incompatible:
        raise ValueError(f"Input compatibility failed for {len(incompatible)} trials; run inspect before fitting")
    observation = trials[0].force_observation
    if any(trial.force_observation != observation for trial in trials):
        raise ValueError("All trials of one fit must share a force observation")
    parameters = Parameterization(baseline, [trial.task.speed_m_s for trial in train])
    size, reg = parameters.size, search.regularization
    engine = (
        _HostEngine(parameters, train, cfg, search)
        if device == "cpu"
        else _DeviceEngine(parameters, train, cfg, search, device)
    )
    penalty_scale = math.sqrt(reg / size)

    def penalty(offsets: np.ndarray) -> float:
        scaled = penalty_scale * offsets
        return float(scaled @ scaled)

    x = np.zeros(size)
    started = engine.start(x)
    if started is None:
        raise ValueError("The initial model fails a training trial; LM needs a completed starting point")
    sample, motion = started
    cost = sample + penalty(x)
    initial_cost = cost
    initial_motion = motion
    print(f"lm 0/{search.iterations}: cost {cost:.4g} | {_describe(motion)}", flush=True)
    damping = search.damping
    history = []
    started = perf_counter()
    for iteration in range(search.iterations):
        tick = perf_counter()
        jtj, jtr, failed_columns = engine.jacobian(x)
        # The offset penalty rows contribute penalty_scale * I to the augmented Jacobian.
        normal = jtj + penalty_scale * penalty_scale * np.eye(size)
        gradient = jtr + penalty_scale * (penalty_scale * x)
        scale = np.maximum(np.diag(normal), 1e-12 * max(np.diag(normal).max(), 1e-300))
        proposals = []
        for multiplier in search.ladder:
            delta = np.linalg.solve(normal + damping * multiplier * np.diag(scale), -gradient)
            proposals.append(np.clip(x + delta, -search.bound, search.bound))
        proposals = np.asarray(proposals)
        sums, motions = engine.trial(proposals)
        costs = [
            value + penalty(p) if math.isfinite(value) else math.inf for p, value in zip(proposals, sums, strict=True)
        ]
        best = int(np.argmin(costs))
        accepted = costs[best] < cost
        improvement = (cost - costs[best]) / cost if accepted else 0.0
        if accepted:
            x, cost = proposals[best], costs[best]
            engine.accept(best)
            motion = motions[best]
            damping = max(damping * search.ladder[best] / 3.0, 1e-12)
        else:
            damping = min(damping * max(search.ladder) * 10.0, 1e12)
        history.append(
            {
                "iteration": iteration,
                "cost": cost,
                "motion": motion,
                "accepted": accepted,
                "damping": damping,
                "ladder_costs": [c if math.isfinite(c) else None for c in costs],
                "failed_jacobian_columns": failed_columns,
                "wall_s": perf_counter() - tick,
            }
        )
        print(
            f"lm {iteration + 1}/{search.iterations}: cost {cost:.4g} ({'accepted' if accepted else 'rejected'})"
            f" | {_describe(motion)} | damping {damping:.3g}, failed columns {failed_columns},"
            f" {perf_counter() - tick:.0f} s",
            flush=True,
        )
        if accepted and improvement < search.tolerance:
            break
    learned = parameters.model(x)
    results = {
        "train": {
            "baseline": evaluate(baseline, train, cfg, device=device),
            "learned": evaluate(learned, train, cfg, device=device),
        }
    }
    if held_out:
        results["eval"] = {
            "baseline": evaluate(baseline, held_out, cfg, device=device),
            "learned": evaluate(learned, held_out, cfg, device=device),
        }
    return learned, {
        "schema": "generative_runner_identification_lm_1",
        "validated": False,
        "reference_inputs_used": False,
        "device": device,
        "selection_split": "train",
        "method": "levenberg_marquardt",
        "objective": "coordinate and GRF mean squares of identify.score plus offset regularization"
        + (
            ""
            if observation is None
            else f"; simulated GRF observed through a {observation.cutoff_hz:g} Hz target filter, "
            f"unobservable vibration weighted {observation.vibration_weight:g}"
        ),
        "force_observation": None if observation is None else asdict(observation),
        "initial_model": baseline.to_dict(),
        "search": asdict(search),
        "rollout": asdict(cfg),
        "parameters": size,
        "initial_cost": initial_cost,
        "final_cost": cost,
        "initial_motion": initial_motion,
        "final_motion": motion,
        "rollouts": engine.rollouts,
        "wall_s": perf_counter() - started,
        "history": history,
        "incompatible_trials": incompatible,
        "allow_incompatible": allow_incompatible,
        "splits": results,
        "trials": [{"id": trial.id, "split": trial.split, **trial.provenance} for trial in trials],
    }
