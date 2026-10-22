# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Stage-0 transition collection for the SuperMimic world model.

A simple controller runs batched :class:`~projects.supermimic.sim.RunnerSim`
worlds: the equilibrium angles follow the F01 reference plus smooth noise, and
the stiffness ``K`` and damping ``B`` of every joint are drawn per episode,
log-uniformly over wide ranges. Each world's shoe geometry, foam and friction
come from its sim context ``xi`` of :mod:`.context`. Episodes start
at reference flight frames and end on a fall or at the end of the reference
window. A step whose outputs are not finite or whose joint speeds are not
physical is a sim failure: the world is reset, and the failed step's state is
not stored, so its last row ends the episode as a truncation. Rows are pushed to a :class:`~projects.supermimic.buffer.ReplayBuffer`
and written as a dataset folder.

State rows are root position [m], root quaternion (``xyzw``), root linear
[m/s] and angular [rad/s] velocity in world axes and the treadmill frame, joint
angles [rad] and joint velocities [rad/s]. Aux rows are, per foot
(right, left), the ground reaction [N] and center of pressure [m], then the
mean joint torques [N m] and the positive and negative joint work over the
step [J]. Action rows have the
``[10, 3]`` layout of :attr:`RunnerSim.action`: equilibrium, stiffness and
damping per joint.
"""

from __future__ import annotations

import argparse
import datetime
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import warp as wp

from .body import JOINT_NAMES
from .buffer import ReplayBuffer
from .context import GEOMETRY_NAMES, XI_COORDINATES, XI_DIM, ShoeSpace, decode, encode
from .reference import DeLevaTable, f01_body_params
from .sim import RunnerSim, load_shoes

__all__ = ["AUX_DIM", "STATE_DIM", "Stage0Config", "collect", "main", "read_episode"]

JOINT_COUNT = len(JOINT_NAMES)
STATE_DIM = 13 + 2 * JOINT_COUNT
AUX_DIM = 12 + 3 * JOINT_COUNT
SHOE_DATASET = Path("Z:/newton_grid/data/dataset_3d_column_bed_v2")
"""Column-bed shoe dataset of subject F01."""
HELD_OUT_SHOE = "puma-deviate-nitro-elite-3"
"""Shoe whose foam is the center of the held-out context ball."""


@dataclass(frozen=True)
class Stage0Config:
    """Controller and episode settings of the stage-0 collection."""

    stiffness_range: tuple[float, float] = (30.0, 1500.0)
    """Per-episode log-uniform range of each joint stiffness [N m/rad]."""
    damping_range: tuple[float, float] = (0.5, 40.0)
    """Per-episode log-uniform range of each joint damping [N m s/rad]."""
    noise_std_rad: float = 0.1
    """Stationary standard deviation of the equilibrium-angle noise [rad]."""
    noise_time_constant_s: float = 0.1
    """Correlation time of the Ornstein-Uhlenbeck equilibrium-angle noise [s]."""
    fall_height_m: float = 0.6
    """A world falls when its root is below this height [m]."""
    failure_joint_speed_rad_s: float = 100.0
    """A step fails when a joint speed exceeds this; running stays below about 30 rad/s [rad/s]."""
    touchdown_force_n: float = 50.0
    """Vertical ground reaction that counts as foot contact [N]."""


def sim_state(sim: RunnerSim) -> torch.Tensor:
    """Return the state rows [W, :data:`STATE_DIM`] of ``sim`` after its last step or reset."""
    q, qd = wp.to_torch(sim.joint_q), wp.to_torch(sim.joint_qd)
    return torch.cat((q[:, :7], qd[:, :6], q[:, 7:], qd[:, 6:]), dim=1)


def sim_aux(sim: RunnerSim) -> torch.Tensor:
    """Return the aux rows [W, :data:`AUX_DIM`] of ``sim``'s last step."""
    w = sim.world_count
    parts = (sim.grf, sim.cop, sim.joint_torque, sim.joint_work_pos, sim.joint_work_neg)
    return torch.cat([wp.to_torch(a).reshape(w, -1) for a in parts], dim=1)


def _log_uniform(low: float, high: float, shape, generator: torch.Generator, device) -> torch.Tensor:
    u = torch.rand(shape, generator=generator, device=device)
    return low * (high / low) ** u


def collect(
    reference: Path,
    world_count: int,
    steps: int,
    *,
    split: str = "train",
    seed: int = 0,
    config: Stage0Config | None = None,
) -> tuple[ReplayBuffer, dict]:
    """Run the stage-0 controller and return the filled buffer and per-world metadata.

    Args:
        reference: Folder written by :func:`projects.supermimic.reference.prepare_reference`.
        world_count: Simulated worlds.
        steps: Control steps per world; also the buffer capacity.
        split: ``"train"`` draws training contexts and train windows,
            ``"eval"`` held-out contexts and eval windows.
        seed: Seed of every random draw.
        config: Controller and episode settings.
    """
    config = config or Stage0Config()
    device = "cuda"
    rng = np.random.default_rng(seed)
    generator = torch.Generator(device).manual_seed(seed)
    report = json.loads((reference / "report.json").read_text())
    body = f01_body_params(DeLevaTable(report["body_table"]))
    with np.load(reference / "reference.npz") as f:
        ref = {k: f[k] for k in f.files}

    shoes = {s.name: s for s in load_shoes(SHOE_DATASET)}
    # The friction entry of the held-out center is unused.
    held_out = encode(shoes[HELD_OUT_SHOE].left.material, 0.8, GEOMETRY_NAMES.index(HELD_OUT_SHOE))
    space = ShoeSpace(held_out=held_out)
    xi = space.sample_training(world_count, rng) if split == "train" else space.sample_held_out(world_count, rng)
    materials, friction, geometry = decode(xi)
    sim = RunnerSim(body, [shoes[name] for name in GEOMETRY_NAMES], geometry, device=device)
    sim.set_world_materials(materials, friction)

    sim_cfg = sim.config
    stride = round(sim_cfg.control_dt_s / float(ref["frame_dt"]))
    frames = ref["joint_q"].shape[1]
    ref_q = torch.as_tensor(ref["joint_q"], dtype=torch.float32, device=device)
    ref_qd = torch.as_tensor(ref["joint_qd"], dtype=torch.float32, device=device)
    in_split = ref["split"][ref["flight_window"]] == split
    starts = torch.as_tensor(
        np.stack((ref["flight_window"][in_split], ref["flight_frame"][in_split]), 1), device=device
    )

    w = world_count
    xi_t = torch.as_tensor(xi, device=device)
    buffer = ReplayBuffer(
        w,
        steps,
        state_shape=STATE_DIM,
        action_shape=(JOINT_COUNT, 3),
        xi_shape=XI_DIM,
        aux_shape=AUX_DIM,
        device=device,
    )
    window = torch.zeros(w, dtype=torch.int64, device=device)
    frame = torch.zeros_like(window)
    stiffness = torch.zeros((w, JOINT_COUNT), device=device)
    damping = torch.zeros_like(stiffness)
    noise = torch.zeros_like(stiffness)
    decay = float(np.exp(-sim_cfg.control_dt_s / config.noise_time_constant_s))
    reset = torch.ones(w, dtype=torch.bool, device=device)
    failures = torch.zeros(w, dtype=torch.int64, device=device)

    with torch.cuda.stream(wp.stream_to_torch(sim.device)):
        for _ in range(steps):
            if reset.any():
                n = int(reset.sum())
                pick = starts[torch.randint(len(starts), (n,), generator=generator, device=device)]
                window[reset], frame[reset] = pick[:, 0], pick[:, 1]
                sim.reset(
                    wp.from_torch(reset), wp.from_torch(ref_q[window, frame]), wp.from_torch(ref_qd[window, frame])
                )
                stiffness[reset] = _log_uniform(*config.stiffness_range, (n, JOINT_COUNT), generator, device)
                damping[reset] = _log_uniform(*config.damping_range, (n, JOINT_COUNT), generator, device)
                noise[reset] = config.noise_std_rad * torch.randn((n, JOINT_COUNT), generator=generator, device=device)
            state = sim_state(sim)
            done = state[:, 2] < config.fall_height_m
            action = torch.stack((ref_q[window, frame, 7:] + noise, stiffness, damping), dim=2)
            sim.step(wp.from_torch(action.contiguous()))
            aux, after = sim_aux(sim), sim_state(sim)
            failed = ~(
                torch.isfinite(aux).all(1) & (after[:, -JOINT_COUNT:].abs().amax(1) < config.failure_joint_speed_rad_s)
            )
            failures += failed
            buffer.push(
                state, action, xi_t, aux.masked_fill(failed[:, None], 0.0), window * frames + frame, reset, done
            )
            frame += stride
            reset = done | failed | (frame >= frames)
            innovation = torch.randn((w, JOINT_COUNT), generator=generator, device=device)
            noise = decay * noise + config.noise_std_rad * float(np.sqrt(1.0 - decay**2)) * innovation

    meta = {
        "dt_s": sim_cfg.control_dt_s,
        "shoe": [GEOMETRY_NAMES[g] for g in geometry],
        "split": split,
        "seed": seed,
        "reference": str(reference),
        "reference_frames_per_window": frames,
        "xi_coordinates": [c[0] for c in XI_COORDINATES],
        "config": asdict(config),
        "failures": failures.tolist(),
        "xi": xi.tolist(),
    }
    return buffer, meta


def summarize(buffer: ReplayBuffer, meta: dict) -> dict:
    """Count rows, episodes, falls, touchdowns and sim failures of a collection.

    Failure rates are per simulated world-second, overall, per shoe geometry
    and per quarter of each ``xi`` coordinate's range.
    """
    data = buffer.data
    seconds = buffer.size * meta["dt_s"]
    failures = np.asarray(meta["failures"])
    xi = np.asarray(meta["xi"])
    shoes = np.asarray(meta["shoe"])
    quarter = np.clip(np.floor((xi[:, : len(meta["xi_coordinates"])] + 1.0) * 2.0), 0, 3).astype(int)
    by_xi = {
        name: [
            float(failures[quarter[:, i] == b].sum() / max(seconds * (quarter[:, i] == b).sum(), 1e-9))
            for b in range(4)
        ]
        for i, name in enumerate(meta["xi_coordinates"])
    }
    by_shoe = {
        name: float(failures[shoes == name].sum() / (seconds * (shoes == name).sum())) for name in np.unique(shoes)
    }
    touchdown_force_n = meta["config"]["touchdown_force_n"]
    contact = data["aux"][:, :, [2, 5]] > touchdown_force_n
    touchdown = contact[:, 1:] & ~contact[:, :-1] & ~data["reset"][:, 1:, None]
    lengths = data["episode_step"][:, :-1][data["reset"][:, 1:]] + 1
    return {
        "rows": buffer.world_count * buffer.size,
        "episodes": int(data["reset"].sum()),
        "falls": int(data["done"].sum()),
        "touchdowns": int(touchdown.sum()),
        "completed_episode_steps_mean": float(lengths.float().mean()) if len(lengths) else 0.0,
        "windows_k8": int(buffer.valid_starts(8).sum()),
        "sim_failures": int(failures.sum()),
        "sim_failures_per_world_s": float(failures.sum() / (seconds * buffer.world_count)),
        "sim_failures_per_world_s_by_shoe": by_shoe,
        "sim_failures_per_world_s_by_xi_quarter": by_xi,
    }


def read_episode(folder: str | Path, world: int, episode: int) -> dict:
    """Return one episode of a dataset folder in :class:`~projects.supermimic.sim.RunnerSim` layouts.

    Args:
        folder: Dataset folder written by :func:`main`.
        world: World index.
        episode: Episode index within the world, in order.

    Returns:
        ``joint_q`` [T, 17], ``joint_qd`` [T, 16], ``grf`` and ``cop`` [T, 2, 3],
        ``joint_torque`` and step-mean ``joint_power`` [T, 10], ``action`` [T, 10, 3],
        ``dt``, ``shoe``, and the reference window and frame of the first row,
        ``reference_window`` and ``reference_frame``.
    """
    folder = Path(folder)
    meta = json.loads((folder / "meta.json").read_text())
    with np.load(folder / "buffer.npz") as f:
        state, action, aux, reset, reference = (f[k][world] for k in ("state", "action", "aux", "reset", "reference"))
    bounds = [*np.flatnonzero(reset), len(reset)]
    rows = slice(bounds[episode], bounds[episode + 1])
    state, action, aux = state[rows], action[rows], aux[rows]
    j = JOINT_COUNT
    return {
        "joint_q": np.concatenate((state[:, :7], state[:, 13 : 13 + j]), 1),
        "joint_qd": np.concatenate((state[:, 7:13], state[:, 13 + j :]), 1),
        "grf": aux[:, :6].reshape(-1, 2, 3),
        "cop": aux[:, 6:12].reshape(-1, 2, 3),
        "joint_torque": aux[:, 12 : 12 + j],
        "joint_power": (aux[:, 12 + j : 12 + 2 * j] + aux[:, 12 + 2 * j :]) / meta["dt_s"],
        "action": action,
        "dt": meta["dt_s"],
        "shoe": meta["shoe"][world],
        "reference_window": int(reference[rows.start]) // meta["reference_frames_per_window"],
        "reference_frame": int(reference[rows.start]) % meta["reference_frames_per_window"],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Collect stage-0 SuperMimic transitions.")
    parser.add_argument("--reference", type=Path, required=True, help="F01 reference folder")
    parser.add_argument("--worlds", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=1000, help="control steps per world")
    parser.add_argument("--split", choices=("train", "eval"), default="train")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    output = args.output or Path("outputs/supermimic") / f"stage0_{args.split}_{datetime.date.today()}"
    output.mkdir(parents=True, exist_ok=False)
    buffer, meta = collect(args.reference, args.worlds, args.steps, split=args.split, seed=args.seed)
    meta["summary"] = summarize(buffer, meta)
    buffer.save(output / "buffer.npz")
    (output / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta["summary"], indent=2))


if __name__ == "__main__":
    main()
