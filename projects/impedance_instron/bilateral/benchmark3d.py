# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Time the 3D environment's control step per solver and world count.

Each case drives a frozen policy for warm-up steps, then times whole control steps
(policy, observation, and the CUDA-graph integration) and one eager control step's
kernels.
"""

import argparse
import collections
import json
import time
from dataclasses import replace
from pathlib import Path

import torch
import warp as wp

from . import env3d
from .endurance3d import reset_once, stored
from .policy import load_policy


def measure(dataset: Path, policy: Path, worlds: int, solver: str, dt_s: float, steps: int = 50) -> dict:
    """Time one solver at one world count; returns step rates and the top kernels."""
    device = "cuda:0"
    actor, meta = load_policy(policy, device, env3d.OBSERVATION_NAMES, env3d.ACTION_NAMES)
    config = stored(env3d.Env3DConfig, meta["environment"]["config"])
    config = replace(config, dt_s=dt_s, substeps=round(config.control_dt_s / dt_s), solver=solver)
    library = env3d.ReferenceLibrary3D(dataset, ("train",), config.control_dt_s, device)
    env = env3d.BilateralEnv3D(library, worlds, config, device=device)
    reset_once(env)
    with torch.no_grad():
        for _ in range(20):
            env.advance(actor(env.observe()))
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(steps):
            env.advance(actor(env.observe()))
        torch.cuda.synchronize()
        control = (time.perf_counter() - start) / steps
        start = time.perf_counter()
        for _ in range(steps):
            env.integrate()
        torch.cuda.synchronize()
        integrate = (time.perf_counter() - start) / steps
        wp.timing_begin(cuda_filter=wp.TIMING_ALL)
        env._substeps(config.substeps)
        results = wp.timing_end()
    total = collections.Counter()
    for result in results:
        total[result.name] += result.elapsed
    alive = int((env.t.status == 0).sum())
    return {
        "worlds": worlds,
        "solver": solver,
        "dt_s": dt_s,
        "substeps": config.substeps,
        "alive": alive,
        "control_step_ms": round(control * 1e3, 3),
        "integrate_ms": round(integrate * 1e3, 3),
        "world_steps_per_s": round(worlds / control, 1),
        "realtime_factor_total": round(worlds * config.control_dt_s / control, 2),
        "kernels_ms_per_control_step": {name: round(ms, 4) for name, ms in total.most_common(30)},
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--worlds", type=int, nargs="+", default=[1024, 4096, 16384])
    parser.add_argument("--solvers", nargs="+", default=["featherstone", "mujoco"])
    parser.add_argument("--dt-s", type=float, default=2.5e-4)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for solver in args.solvers:
        for worlds in args.worlds:
            try:
                row = measure(args.dataset, args.policy, worlds, solver, args.dt_s)
            except Exception as error:
                row = {"worlds": worlds, "solver": solver, "error": repr(error)}
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "kernels_ms_per_control_step"}), flush=True)
            (args.output / "benchmark.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
