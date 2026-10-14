# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Record a deterministic 3D policy rollout beside its reference and score the gait.

Writes ``gait.npz`` (joint coordinates, reference, GRF, action jumps per control step)
and ``gait.json``: cadence and GRF peak against the reference, and the per-joint RMS
error of the motion, over the worlds that stay up.
"""

import argparse
import json
from dataclasses import replace
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch

from . import env3d
from .endurance3d import reset_once, stored
from .policy import load_policy


def record(dataset: Path, policy: Path, worlds: int, seconds: float, dt_s: float, solver: str | None) -> dict:
    """Run ``policy`` for ``seconds`` and return the per-control-step traces as arrays."""
    actor, meta = load_policy(policy, "cuda:0", env3d.OBSERVATION_NAMES, env3d.ACTION_NAMES)
    config = stored(env3d.Env3DConfig, meta["environment"]["config"])
    config = replace(config, dt_s=dt_s, substeps=round(config.control_dt_s / dt_s))
    if solver is not None:
        config = replace(config, solver=solver)
    library = env3d.ReferenceLibrary3D(dataset, ("train",), config.control_dt_s, "cuda:0")
    env = env3d.BilateralEnv3D(library, worlds, config, seed=3)
    reset_once(env)
    traces = {name: [] for name in ("q", "q_ref", "grf", "grf_ref", "alive", "jump")}
    with torch.no_grad():
        for _ in range(round(seconds / config.control_dt_s)):
            env.advance(actor(env.observe()))
            q_ref, _, grf_ref = env._reference()
            for name, value in (
                ("q", env.joint_q),
                ("q_ref", q_ref),
                ("grf", env._interval_force()),
                ("grf_ref", grf_ref),
                ("alive", env.t.status == 0),
                ("jump", env.jump),
            ):
                traces[name].append(value.clone())
    arrays = {name: torch.stack(values).cpu().numpy() for name, values in traces.items()}
    return {"control_dt_s": config.control_dt_s, **arrays}


def _touchdowns(fz: np.ndarray, threshold_n: float = 50.0) -> np.ndarray:
    return np.flatnonzero(np.diff((fz > threshold_n).astype(int)) == 1) + 1


def score(traces: dict, settle_s: float = 2.0) -> dict:
    """Gait statistics after ``settle_s`` over the worlds alive throughout."""
    dt = traces["control_dt_s"]
    start = round(settle_s / dt)
    q, q_ref, grf, grf_ref = (traces[name][start:] for name in ("q", "q_ref", "grf", "grf_ref"))
    up = traces["alive"][start:].all(0)
    joints = [name.removeprefix("equilibrium_") for name in env3d.ACTION_NAMES[: env3d.model3d.ACTUATED]]
    rms = np.degrees(np.sqrt(((q[:, up, 7:] - q_ref[:, up, 7:]) ** 2).mean((0, 1))))
    stats = {"policy": ([], []), "reference": ([], [])}
    for world in np.flatnonzero(up):
        for side in range(2):
            for key, forces in (("policy", grf), ("reference", grf_ref)):
                fz = forces[:, world, 3 * side + 2]
                touchdowns = _touchdowns(fz)
                if len(touchdowns) > 1:
                    stats[key][0].append(120.0 / (np.diff(touchdowns).mean() * dt))
                stats[key][1].extend(fz[a:b].max() for a, b in pairwise(touchdowns))
    return {
        "worlds_up": int(up.sum()),
        "worlds": int(up.size),
        "cadence_steps_per_min": round(float(np.median(stats["policy"][0])), 1),
        "cadence_reference": round(float(np.median(stats["reference"][0])), 1),
        "grf_peak_n": round(float(np.median(stats["policy"][1]))),
        "grf_peak_reference_n": round(float(np.median(stats["reference"][1]))),
        "grf_peak_p95_n": round(float(np.percentile(stats["policy"][1], 95))),
        "joint_rms_deg": dict(zip(joints, np.round(rms, 1).tolist(), strict=True)),
        "joint_rms_mean_deg": round(float(rms.mean()), 2),
        "jump_p99": round(float(np.percentile(np.abs(traces["jump"][start:, up]), 99)), 3),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=64)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--dt-s", type=float, default=1.25e-4)
    parser.add_argument("--solver", choices=("featherstone", "mujoco"), help="Override the policy's solver.")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    traces = record(args.dataset, args.policy, args.worlds, args.seconds, args.dt_s, args.solver)
    np.savez_compressed(args.output / "gait.npz", **traces)
    result = score(traces)
    (args.output / "gait.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
