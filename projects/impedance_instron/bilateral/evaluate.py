# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Evaluate a frozen bilateral controller on measured segments without refitting.

Accepts a neural policy (``.pt`` with its ``.json`` metadata) or a bilateral
linear Hogan model (``.json``). Writes per-segment scores, split summaries,
control-rate traces, and comparison figures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .env import BilateralEnv, EnvConfig, ReferenceLibrary, observation_names
from .gpu import MODE_IMPEDANCE, MODE_LINEAR
from .linear import BilateralRunner
from .policy import load_policy
from .rollout import SegmentRollouts, initial_state, score_world, summarize


def build_env(dataset, split: str, candidates: int, config: EnvConfig, *, mode: int, device: str) -> BilateralEnv:
    """Create an environment with one world per segment and candidate."""
    library = ReferenceLibrary(dataset, (split,), config.control_dt_s, device)
    assets = library.manifest["shared_assets"]
    return BilateralEnv(
        library,
        library.root / assets["digital_shoe_left.json"]["file"],
        library.root / assets["digital_shoe_right.json"]["file"],
        library.manifest["shoe"]["mount_m"],
        library.manifest["shoe"]["pitch_rad"],
        len(library) * candidates,
        config,
        device=device,
        mode=mode,
        candidates=candidates,
    )


def evaluate_split(model_path: Path, dataset: Path, split: str, *, device: str = "cuda:0") -> tuple[dict, dict]:
    """Return per-segment scores and traces for one split."""
    if model_path.suffix == ".pt":
        meta = json.loads(model_path.with_suffix(".json").read_text(encoding="utf-8"))
        config = EnvConfig(**_config(meta["environment"]["config"]))
        actor, meta = load_policy(model_path, device, observation_names(config))
        env = build_env(dataset, split, 1, config, mode=MODE_IMPEDANCE, device=device)
        rollouts = SegmentRollouts(env)
        traces = rollouts.run(actor)
    else:
        runner = BilateralRunner.load(model_path)
        config = runner.env_config()
        env = build_env(dataset, split, 1, config, mode=MODE_LINEAR, device=device)
        runner.upload(env)
        rollouts = SegmentRollouts(env)
        traces = rollouts.run(None)
    cap = env.torque_cap.cpu().numpy()
    rows = [
        score_world(traces, s, env.library, s, rollouts.starts[s], torque_cap=cap) for s in range(rollouts.segments)
    ]
    traces["segment_ids"] = np.array([m["id"] for m in env.library.members])
    return {"summary": summarize(rows), "segments": rows}, traces


def _config(data: dict) -> dict:
    data = dict(data)
    data["reward_weights"] = tuple(data["reward_weights"])
    return data


def figure(traces: dict, scores: dict, dataset: Path, split: str, path: Path, segment: int = 0) -> Path | None:
    """Plot one segment's joint angles and per-foot forces against the measurement."""
    try:
        import matplotlib  # noqa: PLC0415

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError:
        return None
    library = ReferenceLibrary(dataset, (split,), float(traces["dt_s"]), "cpu")
    reference = library.references[segment]
    start = initial_state(library, segment)
    dt = float(traces["dt_s"])
    steps = int(traces["steps"][segment])
    time = np.arange(steps + 1) * dt
    measured_time = reference["time_s"] - start.origin_s
    fig, axes = plt.subplots(4, 2, figsize=(12, 11), sharex=True)
    names = ("hip", "knee", "ankle")
    for s, side in enumerate(("right", "left")):
        for j, name in enumerate(names):
            ax = axes[j, s]
            ax.plot(measured_time, np.degrees(reference["q"][:, 3 + 3 * s + j]), "k-", lw=1, label="measured")
            ax.plot(time, np.degrees(traces["state"][: steps + 1, segment, 3 + 3 * s + j]), "C3-", lw=1, label="model")
            ax.set_ylabel(f"{side} {name} [deg]")
        ax = axes[3, s]
        ax.plot(reference["grf_time_s"] - start.origin_s, reference["grf_side_n"][s, :, 1], "k-", lw=1)
        ax.plot((np.arange(steps) + 0.5) * dt, traces["grf_n"][:steps, segment, 2 * s + 1], "C3-", lw=0.8)
        ax.plot(reference["grf_time_s"] - start.origin_s, reference["grf_side_n"][s, :, 0], "k--", lw=1)
        ax.plot((np.arange(steps) + 0.5) * dt, traces["grf_n"][:steps, segment, 2 * s], "C0-", lw=0.8)
        ax.set_ylabel(f"{side} GRF fz/fx [N]")
        ax.set_xlabel("time [s]")
    axes[0, 0].legend()
    row = scores["segments"][segment]
    fig.suptitle(f"{row['id']} ({split}): loss {row['loss']:.2f}, {row['status']}")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "eval"])
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    result = {"model": str(args.model.resolve()), "dataset": str(args.dataset.resolve()), "splits": {}}
    for split in args.splits:
        scores, traces = evaluate_split(args.model, args.dataset, split, device=args.device)
        result["splits"][split] = scores
        np.savez_compressed(args.output / f"traces_{split}.npz", **traces)
        figure(traces, scores, args.dataset, split, args.output / f"segment_{split}_0.png")
        print(split, json.dumps(scores["summary"], indent=1), flush=True)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
