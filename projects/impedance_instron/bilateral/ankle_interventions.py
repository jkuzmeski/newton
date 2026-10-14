# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Intervention experiments on a frozen policy's ankle impedance.

Each experiment overrides one ankle channel of the policy's command during
deterministic free rollouts and reports how the impact peak, ankle error by
phase, swing scuffing, and failures change. The overrides are diagnostics:
``swing_reference`` aims the swing ankle at the measured angle, which a deployed
policy never sees.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .ankle_diagnosis import analyze
from .env import EnvConfig, observation_names
from .evaluate import build_env
from .gpu import MODE_IMPEDANCE
from .policy import load_policy
from .rollout import SegmentRollouts, score_world, summarize

ANKLE = {"equilibrium": (2, 5), "stiffness": (8, 11), "damping": (14, 17)}
HOLD_STEPS = 30
"""Control steps (60 ms at 500 Hz) to hold a channel after touchdown."""


class Intervention:
    """Wrap a policy and override ankle channels per world and foot."""

    def __init__(self, actor, env, experiment: str):
        self.actor, self.env, self.experiment = actor, env, experiment
        w, dev = env.worlds, env.torch_device
        self.held = torch.zeros(w, 18, device=dev)
        self.countdown = torch.zeros(w, 2, dtype=torch.long, device=dev)
        self.previous = torch.zeros(w, 18, device=dev)
        self.loaded = torch.zeros(w, 2, dtype=torch.bool, device=dev)

    def __call__(self, observation: torch.Tensor) -> torch.Tensor:
        action = self.actor(observation)
        env = self.env
        load = env.t.sense.float()
        loaded = load > 0.05
        touchdown = loaded & ~self.loaded
        self.loaded = loaded
        if self.experiment in ANKLE:
            for s in range(2):
                index = ANKLE[self.experiment][s]
                start = touchdown[:, s]
                # Hold the command issued just before the load was sensed.
                self.held[start, index] = self.previous[start, index]
                self.countdown[start, s] = HOLD_STEPS
                active = self.countdown[:, s] > 0
                action[active, index] = self.held[active, index]
            self.countdown = (self.countdown - 1).clamp(min=0)
        elif self.experiment == "swing_reference":
            last = env.library.starts[env.segment] + env.library.lengths[env.segment] - 1
            k = torch.minimum(env.library.starts[env.segment] + env.index, last)
            for s in range(2):
                swing = load[:, s] < 0.01
                target = env.library.q[k, 5 + 3 * s].float()
                fraction = ((target - env.lower[2 + 3 * s]) / (env.upper[2 + 3 * s] - env.lower[2 + 3 * s])).clamp(
                    1e-4, 1 - 1e-4
                )
                action[swing, 2 + 3 * s] = torch.logit(fraction[swing])
        self.previous = action.detach().clone()
        return action


def run(model: Path, dataset: Path, split: str, experiment: str, device: str) -> tuple[dict, dict]:
    meta = json.loads(model.with_suffix(".json").read_text(encoding="utf-8"))
    data = dict(meta["environment"]["config"])
    data["reward_weights"] = tuple(data["reward_weights"])
    config = EnvConfig(**{key: value for key, value in data.items() if key in EnvConfig.__dataclass_fields__})
    actor, _ = load_policy(model, device, observation_names(config))
    env = build_env(dataset, split, 1, config, mode=MODE_IMPEDANCE, device=device)
    rollouts = SegmentRollouts(env)
    controller = actor if experiment == "none" else Intervention(actor, env, experiment)
    traces = rollouts.run(controller)
    cap = env.torque_cap.cpu().numpy()
    rows = [score_world(traces, s, env.library, s, rollouts.starts[s], torque_cap=cap) for s in range(len(env.library))]
    return summarize(rows), traces


def _swing_contact(traces: dict, dataset: Path, split: str) -> float:
    """Fraction of reference swing control steps in which the simulated swing shoe carries > 50 N."""
    from .env import ReferenceLibrary  # noqa: PLC0415
    from .rollout import initial_state  # noqa: PLC0415

    dt = float(traces["dt_s"])
    library = ReferenceLibrary(dataset, (split,), dt, "cpu")
    hits, total = 0, 0
    for w, reference in enumerate(library.references):
        n = int(traces["steps"][w]) if traces["failed_step"][w] < 0 else int(traces["failed_step"][w])
        clock = (np.arange(n) + 0.5) * dt
        origin = initial_state(library, w).origin_s
        for s in range(2):
            force = np.interp(clock, reference["grf_time_s"] - origin, reference["grf_side_n"][s, :, 1])
            # Exclude 40 ms around each measured contact edge so timing shifts are not counted as scuffs.
            loaded = force > 1.0
            margin = int(round(0.04 / dt))
            near = np.convolve(loaded.astype(float), np.ones(2 * margin + 1), mode="same") > 0
            swing = ~near
            hits += int(np.count_nonzero(swing & (traces["grf_n"][:n, w, 2 * s + 1] > 50.0)))
            total += int(np.count_nonzero(swing))
    return hits / max(total, 1)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--experiments", nargs="+", default=["none", *ANKLE, "swing_reference"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    results = {}
    for experiment in args.experiments:
        summary, traces = run(args.model, args.dataset, args.split, experiment, args.device)
        path = args.output / f"traces_{experiment}.npz"
        np.savez_compressed(path, **traces)
        phases = analyze(path, args.dataset, args.split)["summary"]
        results[experiment] = {
            "mean_loss": summary["mean_loss"],
            "failed": summary["failed"],
            "mean_abs_peak_fz_error_n": summary["mean_abs_peak_fz_error_n"],
            "grf_rmse_n": summary["grf_rmse_n"],
            "joint_rmse_deg": summary["joint_rmse_deg"],
            "swing_contact_fraction": _swing_contact(traces, args.dataset, args.split),
            "ankle_rmse_deg": {
                phase: np.mean([values[side]["ankle_rmse_deg"] for side in values]).item()
                for phase, values in phases.items()
                if phase != "touchdown" and values
            },
            "touchdown_peak_excess_n": phases.get("touchdown", {}).get("peak_excess_n_mean"),
        }
        print(experiment, json.dumps(results[experiment]), flush=True)
    (args.output / "summary.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
