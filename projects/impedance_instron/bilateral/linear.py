# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Bilateral linear Hogan baseline, fitted with Levenberg-Marquardt.

Both legs share one feature-weight set of the single-leg runner
(:mod:`..hogan.runner`): 3 outputs x 3 joints x 14 features. Each leg evaluates
the features with its own filtered foot load and hip-over-ankle offset, and the
left leg runs half an oscillator cycle behind the right. With fixed task speed,
the active parameters are 117 weights plus stride frequency and torque response
time, the same 119 as the single-leg fit. This is a comparison baseline; it does
not initialize the neural policy.

LM uses forward-difference Jacobians of deterministic free rollouts from
training segment starts (:mod:`.rollout`). Residuals hold root position (0.02 m),
pelvis and joint angles (0.05 rad), and per-foot GRF (100 N) at 100 Hz. Steps
after a screen failure carry a fixed penalty. Evaluation segments never enter.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..hogan.identify import ForceObservation
from ..hogan.runner import FEATURE_NAMES, Bounds, Runner
from .env import BilateralEnv, EnvConfig, ReferenceLibrary
from .gpu import MODE_LINEAR
from .rollout import SegmentRollouts

SCHEMA = "bilateral_linear_runner_1"


@dataclass
class BilateralRunner:
    """Shared linear Hogan schedule for both legs."""

    weights: np.ndarray
    """Feature weights, shape [3 outputs, 3 joints, 14 features]."""
    frequency_hz: float = 1.5
    response_time_s: float = 0.025
    load_time_s: float = 0.02
    phase_feedback: float = 0.2
    reference_speed_m_s: float = 3.65
    speed_scale_m_s: float = 1.0
    immediate_damping: bool = True
    bounds: Bounds = field(default_factory=Bounds)

    @classmethod
    def seed(cls, reference_speed_m_s: float = 3.65) -> BilateralRunner:
        """Return the unfitted engineering seed of the single-leg runner, applied to both legs."""
        return cls(
            Runner.seed(reference_speed_m_s=reference_speed_m_s).weights.copy(), reference_speed_m_s=reference_speed_m_s
        )

    def env_config(self, base: EnvConfig | None = None) -> EnvConfig:
        base = base or EnvConfig()
        data = asdict(base)
        data.update(
            response_time_s=self.response_time_s,
            load_time_s=self.load_time_s,
            frequency_hz=self.frequency_hz,
            phase_feedback=self.phase_feedback,
            reference_speed_m_s=self.reference_speed_m_s,
            speed_scale_m_s=self.speed_scale_m_s,
            immediate_damping=self.immediate_damping,
        )
        data["reward_weights"] = tuple(data["reward_weights"])
        return EnvConfig(**data)

    def upload(self, env: BilateralEnv, candidates: list[BilateralRunner] | None = None):
        """Upload this model, or one model per candidate, to a linear-mode environment."""
        models = candidates or [self]
        if len(models) != env.candidates:
            raise ValueError("Upload one model per environment candidate")
        env.set_linear_weights(np.array([m.weights for m in models]))
        per = env.worlds // env.candidates
        dt = env.config.dt_s
        env.t.frequency.copy_(env.t.frequency.new_tensor(np.repeat([m.frequency_hz for m in models], per)))
        env.t.response.copy_(
            env.t.response.new_tensor(np.repeat([-np.expm1(-dt / m.response_time_s) for m in models], per))
        )

    def to_dict(self) -> dict:
        return {
            "schema": SCHEMA,
            "features": list(FEATURE_NAMES),
            "weights": np.asarray(self.weights).tolist(),
            "bounds": asdict(self.bounds),
            **{
                name: getattr(self, name)
                for name in (
                    "frequency_hz",
                    "response_time_s",
                    "load_time_s",
                    "phase_feedback",
                    "reference_speed_m_s",
                    "speed_scale_m_s",
                    "immediate_damping",
                )
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> BilateralRunner:
        data = dict(data)
        if data.pop("schema") != SCHEMA or data.pop("features") != list(FEATURE_NAMES):
            raise ValueError("Unsupported bilateral linear runner schema")
        data["weights"] = np.asarray(data["weights"], dtype=float)
        if data["weights"].shape != (3, 3, len(FEATURE_NAMES)):
            raise ValueError("weights must have shape (3, 3, 14)")
        data["bounds"] = Bounds(**data["bounds"])
        return cls(**data)

    def save(self, path: str | Path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> BilateralRunner:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


class Parameterization:
    """Offsets of the shared weights, log frequency, and log response time from a baseline."""

    def __init__(self, baseline: BilateralRunner):
        self.baseline = baseline
        mask = np.ones(baseline.weights.shape, dtype=bool)
        mask[:, :, FEATURE_NAMES.index("task_speed_offset")] = False
        self.indices = np.flatnonzero(mask.ravel())
        self.size = len(self.indices) + 2

    def model(self, offsets: np.ndarray) -> BilateralRunner:
        n = len(self.indices)
        weights = self.baseline.weights.copy().ravel()
        weights[self.indices] += offsets[:n]
        data = asdict(self.baseline)
        data.update(
            weights=weights.reshape(self.baseline.weights.shape),
            frequency_hz=float(np.clip(self.baseline.frequency_hz * np.exp(np.clip(offsets[n], -20, 20)), 0.5, 4.0)),
            response_time_s=float(
                np.clip(self.baseline.response_time_s * np.exp(np.clip(offsets[n + 1], -20, 20)), 0.005, 0.15)
            ),
            bounds=self.baseline.bounds,
        )
        return BilateralRunner(**data)


class Residuals:
    """Batched residual vectors of candidate models over training segments."""

    def __init__(
        self,
        dataset: Path,
        chunk: int,
        horizon_s: float,
        config: EnvConfig,
        device: str,
        failure_residual: float = 10.0,
    ):
        library = ReferenceLibrary(dataset, ("train",), config.control_dt_s, device)
        assets = library.manifest["shared_assets"]
        self.chunk = chunk
        self.env = BilateralEnv(
            library,
            library.root / assets["digital_shoe_left.json"]["file"],
            library.root / assets["digital_shoe_right.json"]["file"],
            library.manifest["shoe"]["mount_m"],
            library.manifest["shoe"]["pitch_rad"],
            len(library) * chunk,
            config,
            device=device,
            mode=MODE_LINEAR,
            candidates=chunk,
        )
        self.rollouts = SegmentRollouts(self.env, chunk, horizon_s)
        self.library = library
        self.failure_residual = failure_residual
        self.observation = ForceObservation()
        dt = config.control_dt_s
        self.dt = dt
        # Sample both motion and force at 100 Hz from the prediction origin.
        self.samples = []
        for s, (reference, start) in enumerate(zip(library.references, self.rollouts.starts, strict=True)):
            steps = int(self.rollouts.steps[s])
            k = np.arange(5, steps + 1, 5)
            t = k * dt
            q = np.column_stack(
                [np.interp(t + start.origin_s, reference["time_s"], reference["q"][:, c]) for c in range(9)]
            )
            f = np.stack(
                [
                    np.column_stack(
                        [
                            np.interp(
                                t - 0.5 * dt + start.origin_s,
                                reference["grf_time_s"],
                                reference["grf_side_n"][side, :, a],
                            )
                            for a in range(2)
                        ]
                    )
                    for side in range(2)
                ],
                axis=1,
            ).reshape(len(k), 4)
            self.samples.append((k, q, f))
        self.size = sum(len(k) * 13 for k, _, _ in self.samples)
        self.rollout_count = 0

    def __call__(self, models: list[BilateralRunner]) -> tuple[np.ndarray, np.ndarray]:
        """Return residuals [len(models), size] and failure counts per model."""
        results, failures = [], []
        for start in range(0, len(models), self.chunk):
            group = models[start : start + self.chunk]
            padded = group + [group[-1]] * (self.chunk - len(group))
            padded[0].upload(self.env, padded)
            traces = self.rollouts.run(None)
            self.rollout_count += len(group) * len(self.library)
            for c in range(len(group)):
                rows, failed = [], 0
                for s, (k, q, f) in enumerate(self.samples):
                    w = c * len(self.library) + s
                    state = traces["state"][k, w]
                    grf = traces["grf_n"][: k[-1], w].reshape(-1, 2, 2)
                    observed = np.stack([self.observation.apply(grf[:, side], self.dt) for side in range(2)], axis=1)
                    force = observed.reshape(-1, 4)[k - 1]
                    residual = np.concatenate(
                        ((state[:, :2] - q[:, :2]) / 0.02, (state[:, 2:] - q[:, 2:]) / 0.05, (force - f) / 100.0),
                        axis=1,
                    )
                    failure = int(traces["failed_step"][w])
                    if failure >= 0:
                        failed += 1
                        residual[k > failure] = self.failure_residual
                    rows.append(residual.ravel())
                results.append(np.concatenate(rows))
                failures.append(failed)
        return np.array(results), np.array(failures)


def fit(
    dataset: Path,
    output: Path,
    *,
    iterations: int = 10,
    chunk: int = 40,
    horizon_s: float = 1.2,
    step: float = 1e-3,
    regularization: float = 1e-3,
    device: str = "cuda:0",
    seed_model: BilateralRunner | None = None,
) -> BilateralRunner:
    """Fit the shared bilateral linear runner on training segments."""
    output.mkdir(parents=True, exist_ok=True)
    baseline = seed_model or BilateralRunner.seed()
    config = baseline.env_config()
    residuals = Residuals(dataset, chunk, horizon_s, config, device)
    parameterization = Parameterization(baseline)
    n = parameterization.size
    offsets = np.zeros(n)
    history = []

    def evaluate(candidates):
        r, f = residuals([parameterization.model(x) for x in candidates])
        r = np.concatenate((r, np.sqrt(regularization) * np.asarray(candidates)), axis=1)
        return r, f

    r0, f0 = evaluate([offsets])
    cost = 0.5 * float(r0[0] @ r0[0])
    damping = 1e-2
    print(f"start cost {cost:.1f} failures {f0[0]} residuals {r0.shape[1]} parameters {n}", flush=True)
    history.append({"iteration": 0, "cost": cost, "failures": int(f0[0]), "damping": damping})
    current = r0[0]
    for iteration in range(1, iterations + 1):
        tic = time.perf_counter()
        probes = [offsets + step * np.eye(n)[i] for i in range(n)]
        r, _ = evaluate(probes)
        jacobian = (r - current).T / step
        normal = jacobian.T @ jacobian
        gradient = jacobian.T @ current
        ladder = [damping * factor for factor in (0.1, 1.0, 10.0, 100.0)]
        proposals = []
        for value in ladder:
            matrix = normal + value * np.diag(np.maximum(np.diag(normal), 1e-9))
            proposals.append(offsets - np.linalg.solve(matrix, gradient))
        r_try, f_try = evaluate(proposals)
        costs = 0.5 * np.einsum("ij,ij->i", r_try, r_try)
        best = int(np.argmin(costs))
        accepted = costs[best] < cost
        if accepted:
            offsets, current, cost = proposals[best], r_try[best], float(costs[best])
            damping = ladder[best]
        else:
            damping *= 100.0
        history.append(
            {
                "iteration": iteration,
                "cost": cost,
                "accepted": bool(accepted),
                "failures": int(f_try[best]),
                "damping": damping,
                "seconds": time.perf_counter() - tic,
                "rollouts": residuals.rollout_count,
            }
        )
        print(
            f"it {iteration} cost {cost:.1f} accepted {accepted} failures {f_try[best]} damping {damping:.1e} "
            f"{time.perf_counter() - tic:.0f} s",
            flush=True,
        )
        model = parameterization.model(offsets)
        model.save(output / "runner.json")
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    model = parameterization.model(offsets)
    model.save(output / "runner.json")
    (output / "fit.json").write_text(
        json.dumps(
            {
                "dataset": str(Path(dataset).resolve()),
                "training_members": [m["id"] for m in residuals.library.members],
                "parameters": n,
                "horizon_s": horizon_s,
                "step": step,
                "regularization": regularization,
                "history": history,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return model


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--chunk", type=int, default=40)
    parser.add_argument("--horizon-s", type=float, default=1.2)
    parser.add_argument("--model", type=Path, help="Start from a saved bilateral linear runner.")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    seed = BilateralRunner.load(args.model) if args.model else None
    fit(
        args.dataset,
        args.output,
        iterations=args.iterations,
        chunk=args.chunk,
        horizon_s=args.horizon_s,
        device=args.device,
        seed_model=seed,
    )


if __name__ == "__main__":
    main()
