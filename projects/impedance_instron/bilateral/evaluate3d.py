# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Evaluate a frozen 3D bilateral policy with deterministic free rollouts.

Each world starts at the end of its segment's three-frame observed prefix, in
flight, with coordinate rates from backward quadratic differentiation of that
prefix, and runs open loop to the segment's end. Only screen failures stop a
world. Scores compare joint angles, pelvis pose, and per-foot 3D forces offline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .env3d import ACTION_NAMES, OBSERVATION_NAMES, BilateralEnv3D, Env3DConfig, ReferenceLibrary3D
from .kinematics3d import JOINT_NAMES
from .policy import load_policy

PREFIX = 3


def _config(data: dict) -> Env3DConfig:
    data = dict(data)
    data["reward_weights"] = tuple(data["reward_weights"])
    return Env3DConfig(**data)


def rollout(env: BilateralEnv3D, actor) -> dict:
    """Run every evaluation segment once; world ``w`` runs segment ``w``."""
    lib, dev = env.library, env.torch_device
    dt = env.config.control_dt_s
    q0, qd0, origins, steps = [], [], [], []
    for reference in lib.references:
        time, q = reference["time_s"][:PREFIX], reference["q"][:PREFIX].copy()
        t = time - time[-1]
        coefficients = np.linalg.solve(np.column_stack((np.ones(3), t, t * t)), q)
        q0.append(q[-1])
        qd0.append(coefficients[1])
        origins.append(time[-1])
        steps.append(int(np.floor((reference["time_s"][-1] - time[-1]) / dt)))
    q = torch.tensor(np.array(q0), dtype=torch.float32, device=dev)
    rates = torch.tensor(np.array(qd0), dtype=torch.float32, device=dev)
    origin = q[:, :3].clone()
    origin[:, 2] = 0.0
    q[:, :3] -= origin
    mask = torch.ones(env.worlds, dtype=torch.bool, device=dev)
    env.segment[:] = torch.arange(env.worlds, device=dev)
    env.index[:] = torch.tensor([int(round(o / dt)) for o in origins], device=dev)
    env.origin[:] = origin
    env.steps[:] = 0
    env.speed[:] = lib.speeds
    env.write_state(mask, q, env._root_velocity(q, rates), env.kinematic_phase(q, rates))
    horizon = torch.tensor(steps, device=dev)
    failed = torch.full((env.worlds,), -1, dtype=torch.long, device=dev)
    traces = {"q": [env.joint_q.clone() + torch.nn.functional.pad(origin, (0, 14))], "grf_n": [], "effort": []}
    with torch.no_grad():
        for k in range(int(horizon.max())):
            stopped = (horizon <= k) & (env.t.status == 0)
            env.t.status[stopped] = 1
            env.advance(actor(env.observe()))
            newly = (env.t.status > 1) & (failed < 0)
            failed[newly] = k
            q = env.joint_q.clone()
            q[:, :2] = env.position()
            traces["q"].append(q + torch.nn.functional.pad(origin, (0, 14)))
            traces["grf_n"].append(env._interval_force())
            traces["effort"].append(env.t.effort / env.t.substeps.clamp(min=1)[:, None])
    result = {name: torch.stack(values).cpu().numpy() for name, values in traces.items()}
    result.update(
        failed_step=failed.cpu().numpy(),
        steps=np.array(steps),
        origins_s=np.array(origins),
        dt_s=dt,
        status=env.t.status.cpu().numpy(),
    )
    return result


def score(traces: dict, library: ReferenceLibrary3D, world: int) -> dict:
    """Score one segment's 3D free prediction."""
    reference, member = library.references[world], library.members[world]
    dt = float(traces["dt_s"])
    steps, failed = int(traces["steps"][world]), int(traces["failed_step"][world])
    completed = steps if failed < 0 else failed
    time = np.arange(completed + 1) * dt
    measured_time = reference["time_s"] - traces["origins_s"][world]
    observed = (measured_time > 0) & (measured_time <= time[-1] + 1e-12)
    q = traces["q"][: completed + 1, world]
    joints = np.zeros(10)
    root = np.zeros(3)
    if observed.any():
        simulated = np.column_stack([np.interp(measured_time[observed], time, q[:, c]) for c in range(17)])
        target = reference["q"][observed]
        joints = np.sqrt(np.mean((simulated[:, 7:] - target[:, 7:]) ** 2, axis=0))
        root = np.sqrt(np.mean((simulated[:, :3] - target[:, :3]) ** 2, axis=0))
    grf = traces["grf_n"][:completed, world]
    interval = (np.arange(completed) + 0.5) * dt
    force_time = reference["grf_time_s"] - traces["origins_s"][world]
    measured = np.concatenate(
        [
            np.column_stack([np.interp(interval, force_time, reference["grf_side_n"][s, :, a]) for a in range(3)])
            for s in range(2)
        ],
        axis=1,
    )
    force = np.sqrt(np.mean((grf - measured) ** 2, axis=0)) if completed else np.zeros(6)
    loss = float(np.mean((root / 0.02) ** 2) + np.mean((joints / 0.05) ** 2) + np.mean((force / 100.0) ** 2))
    if failed >= 0:
        loss += 1000.0 * (2.0 - completed / max(steps, 1))
    return {
        "id": member["id"],
        "split": member["split"],
        "loss": loss,
        "status": "completed" if failed < 0 else "failed",
        "failure_time_s": None if failed < 0 else completed * dt,
        "root_rmse_m": root.tolist(),
        "joint_rmse_deg": dict(zip(JOINT_NAMES, np.degrees(joints).tolist(), strict=True)),
        "grf_rmse_n": force.tolist(),
    }


def evaluate(model_path: Path, dataset: Path, split: str, device: str = "cuda:0") -> dict:
    actor, meta = load_policy(model_path, device, OBSERVATION_NAMES, ACTION_NAMES)
    config = _config(meta["environment"]["config"])
    library = ReferenceLibrary3D(dataset, (split,), config.control_dt_s, device)
    trials = np.array([library.trials.index(m["trial"]) for m in library.members])
    env = BilateralEnv3D(library, len(library), config, device=device, world_trials=trials)
    traces = rollout(env, actor)
    rows = [score(traces, library, w) for w in range(len(library))]
    completed = [
        row["failure_time_s"] if row["failure_time_s"] is not None else traces["steps"][i] * traces["dt_s"]
        for i, row in enumerate(rows)
    ]
    return {
        "summary": {
            "segments": len(rows),
            "mean_loss": float(np.mean([r["loss"] for r in rows])),
            "failed": sum(r["status"] != "completed" for r in rows),
            "mean_completed_s": float(np.mean(completed)),
        },
        "segments": rows,
    }


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
        result["splits"][split] = evaluate(args.model, args.dataset, split, args.device)
        print(split, json.dumps(result["splits"][split]["summary"]), flush=True)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
