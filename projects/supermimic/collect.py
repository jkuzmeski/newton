# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Stage-0 transition collection for the SuperMimic world model.

A simple controller runs batched :class:`~projects.supermimic.sim.RunnerSim`
worlds within human joint impedance
(:data:`~projects.supermimic.policy.STIFFNESS_LIMITS`,
:data:`~projects.supermimic.policy.DAMPING_LIMITS`). The equilibrium angles
are ``q_ref + (M_ref + B qd_ref) / K`` plus smooth noise, clipped to the joint
limits (:func:`reference_action`), where ``q_ref`` and ``qd_ref`` are the F01
reference joint angles and rates and ``M_ref`` its inverse-dynamics joint
moment. At the reference state the joint torque ``K (q_eq - q) - B qd`` is then
``M_ref``: the reference posture carries the reference moment without a stiff
joint, and the damping acts on the velocity error ``qd_ref - qd`` instead of
braking the reference motion.
A leg whose reference foot is in stance carries ``M_ref`` only once its sim
foot is loaded above :attr:`Stage0Config.touchdown_force_n`; until then its
equilibrium angles are ``q_ref + B qd_ref / K`` plus noise, so a late sim
touchdown does not drive an extended leg into the belt.
The damping ``B`` of every joint is drawn per episode, log-uniformly. The
stiffness ``K`` sits at a per-episode log-uniform quantile between
:func:`moment_stiffness_floor`, which bounds the posture lead
``(M_ref + B qd_ref) / K``, and the upper limit; where the floor exceeds the
upper limit, ``K`` is the upper limit and the torque falls short. Each world's shoe geometry, foam and friction
come from its sim context ``xi`` of :mod:`.context`: the train split draws them
from a ball around one fitted shoe, on that shoe's geometry, and the eval split
uses the fit itself on the held-out reference windows. Episodes start
at reference flight frames and end on a fall or at the end of the reference
window. A step whose outputs are not finite or whose joint speeds are not
physical is a sim failure: the world is reset, and the failed step's state is
not stored, so its last row ends the episode as a truncation. Rows are pushed
to a :class:`~projects.supermimic.buffer.ReplayBuffer` and written as a dataset
folder.

State rows are root position [m], root quaternion (``xyzw``), root linear
[m/s] and angular [rad/s] velocity in world axes and the treadmill frame, joint
angles [rad] and joint velocities [rad/s]. Aux rows are, per foot
(right, left), the ground reaction [N] and center of pressure [m], with x
and y relative to the root position at the start of the step (a world model
does not see the horizontal root position), then the
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

from .analysis import inverse_dynamics
from .body import JOINT_COUNT, JOINT_NAMES, STATE_DIM, BodyParams
from .buffer import ReplayBuffer
from .context import GEOMETRY_NAMES, XI_COORDINATES, XI_DIM, ShoeSpace, decode, encode
from .cost import GenerativeCostConfig
from .policy import DAMPING_LIMITS, STIFFNESS_LIMITS
from .reference import CONTACT_THRESHOLD_N, SHOE_DATASET, load_body_params, load_shoe, load_subject
from .sim import RunnerSim, RunnerSimConfig, Shoe, load_shoes

__all__ = [
    "AUX_DIM",
    "CENTER_SHOE",
    "Stage0Config",
    "collect",
    "episode_starts",
    "loaded_moment",
    "main",
    "make_sim",
    "moment_stiffness_floor",
    "read_episode",
    "reference_action",
    "reference_moments",
    "sim_aux",
    "sim_state",
]

AUX_DIM = 12 + 3 * JOINT_COUNT
CENTER_SHOE = "puma-fast-r-nitro-elite-3"
"""Default shoe whose fit centers the training contexts and is the held-out context."""


@dataclass(frozen=True)
class Stage0Config:
    """Controller and episode settings of the stage-0 collection."""

    noise_std_rad: float = 0.1
    """Stationary standard deviation of the equilibrium-angle noise [rad]."""
    noise_time_constant_s: float = 0.1
    """Correlation time of the Ornstein-Uhlenbeck equilibrium-angle noise [s]."""
    fall_height_m: float = GenerativeCostConfig.height_min
    """A world falls when its root is below this height [m]."""
    max_posture_lead_rad: float = 0.35
    """Largest lead of the equilibrium angle over the reference posture, ``(M_ref + B qd_ref) / K`` [rad]."""
    failure_joint_speed_rad_s: float = 100.0
    """A step fails when a joint speed exceeds this; running stays below about 30 rad/s [rad/s]."""
    touchdown_force_n: float = CONTACT_THRESHOLD_N
    """Vertical ground reaction that counts as foot contact [N]."""


def sim_state(sim: RunnerSim) -> torch.Tensor:
    """Return the state rows [W, STATE_DIM] of ``sim`` after its last step or reset."""
    q, qd = wp.to_torch(sim.joint_q), wp.to_torch(sim.joint_qd)
    return torch.cat((q[:, :7], qd[:, :6], q[:, 7:], qd[:, 6:]), dim=1)


def sim_aux(sim: RunnerSim, state: torch.Tensor) -> torch.Tensor:
    """Return the aux rows [W, :data:`AUX_DIM`] of ``sim``'s last step, taken from the state rows ``state``."""
    w = sim.world_count
    grf, cop = wp.to_torch(sim.grf), wp.to_torch(sim.cop).clone()
    cop[..., :2] -= torch.where(grf[..., 2:] > 0.0, state[:, None, :2], 0.0)
    parts = (grf, cop, *(wp.to_torch(a) for a in (sim.joint_torque, sim.joint_work_pos, sim.joint_work_neg)))
    return torch.cat([a.reshape(w, -1) for a in parts], dim=1)


def _log_uniform(limits: torch.Tensor, count: int, generator: torch.Generator) -> torch.Tensor:
    u = torch.rand((count, len(limits)), generator=generator, device=limits.device)
    return limits[:, 0] * (limits[:, 1] / limits[:, 0]) ** u


def moment_stiffness_floor(
    q_ref: torch.Tensor,
    moment: torch.Tensor,
    q_limits: torch.Tensor,
    stiffness_limits: torch.Tensor,
    max_lead_rad: float,
) -> torch.Tensor:
    """Return the lowest stiffness that carries ``moment`` with a bounded posture lead.

    The equilibrium angle ``q_ref + moment / K`` leads the reference posture by
    ``moment / K``. The floor keeps that lead within the room ``d`` between
    ``q_ref`` and the equilibrium-angle limit in the direction of ``moment``, and
    within ``max_lead_rad``: ``max(K_low, |moment| / min(d, max_lead_rad))``. A
    floor above the upper stiffness limit means the human bound cannot carry
    the moment.

    Args:
        q_ref: Reference joint angles [..., joints] [rad].
        moment: Reference joint moments [..., joints] [N m].
        q_limits: Equilibrium-angle limits (lower, upper) [joints, 2] [rad].
        stiffness_limits: Stiffness limits (lower, upper) [joints, 2] [N m/rad].
        max_lead_rad: Largest posture lead [rad].

    Returns:
        Stiffness floor [..., joints] [N m/rad], not capped by the upper stiffness limit.
    """
    room = torch.where(moment > 0, q_limits[:, 1] - q_ref, q_ref - q_limits[:, 0])
    room = room.clamp(1e-6, max_lead_rad)
    return torch.maximum(stiffness_limits[:, 0], moment.abs() / room)


def reference_action(
    q_ref: torch.Tensor,
    qd_ref: torch.Tensor,
    moment: torch.Tensor,
    damping: torch.Tensor,
    stiffness_quantile: torch.Tensor,
    noise: torch.Tensor,
    q_limits: torch.Tensor,
    stiffness_limits: torch.Tensor,
    max_lead_rad: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the impedance action that carries ``moment`` at the reference state.

    The feed-forward torque ``tau = moment + B qd_ref`` sets the equilibrium
    angles ``q_ref + tau / K + noise``, clipped to ``q_limits``, so the torque
    ``K (q_eq - q) - B qd`` at ``q = q_ref`` is ``moment + B (qd_ref - qd)``.
    ``K`` sits at ``stiffness_quantile`` between the
    :func:`moment_stiffness_floor` of ``tau``, capped at the upper limit, and
    the upper limit, log-uniformly.

    Args:
        q_ref: Reference joint angles [batch, joints] [rad].
        qd_ref: Reference joint rates [batch, joints] [rad/s].
        moment: Joint moments to carry [batch, joints] [N m].
        damping: Joint damping ``B`` [batch, joints] [N m s/rad].
        stiffness_quantile: Stiffness quantile in [0, 1] [batch, joints].
        noise: Equilibrium-angle offsets [batch, joints] [rad].
        q_limits: Equilibrium-angle limits (lower, upper) [joints, 2] [rad].
        stiffness_limits: Stiffness limits (lower, upper) [joints, 2] [N m/rad].
        max_lead_rad: Largest posture lead [rad].

    Returns:
        The action [batch, joints, 3] (equilibrium, stiffness, damping) and
        whether the stiffness falls short of carrying ``tau`` [batch, joints].
    """
    tau = moment + damping * qd_ref
    floor = moment_stiffness_floor(q_ref, tau, q_limits, stiffness_limits, max_lead_rad)
    low = torch.minimum(floor, stiffness_limits[:, 1])
    stiffness = low * (stiffness_limits[:, 1] / low) ** stiffness_quantile
    q_eq = (q_ref + tau / stiffness + noise).clamp(q_limits[:, 0], q_limits[:, 1])
    return torch.stack((q_eq, stiffness, damping), dim=-1), floor > stiffness_limits[:, 1]


def loaded_moment(moment: torch.Tensor, stance: torch.Tensor, loaded: torch.Tensor) -> torch.Tensor:
    """Return ``moment`` with the joints of each reference-stance leg whose sim foot is unloaded set to zero.

    Args:
        moment: Reference joint moments [batch, joints] [N m], the right leg's joints first.
        stance: Reference foot contact [batch, 2] (right, left).
        loaded: Sim foot loaded above the touchdown force [batch, 2] (right, left).
    """
    unloaded_stance = (stance & ~loaded).repeat_interleave(moment.shape[-1] // 2, -1)
    return moment.masked_fill(unloaded_stance, 0.0)


def reference_moments(ref: dict, body: BodyParams, shoe: Shoe) -> np.ndarray:
    """Return the inverse-dynamics joint moments of the reference windows [windows, frames, 10] [N m].

    Args:
        ref: Arrays of ``reference.npz``.
        body: Body parameters of the reference.
        shoe: Shoe pair of the reference, on the feet.
    """
    dt = float(ref["frame_dt"])
    return np.stack(
        [
            inverse_dynamics(q, dt, body, grf, np.nan_to_num(cop), np.zeros_like(grf), shoe)
            for q, grf, cop in zip(ref["joint_q"], ref["grf"], ref["cop"], strict=True)
        ]
    )


def make_sim(
    body: BodyParams,
    world_count: int,
    split: str,
    rng: np.random.Generator,
    belt_speed_m_s: float,
    shoe: str = CENTER_SHOE,
    device: str = "cuda",
) -> tuple[RunnerSim, np.ndarray]:
    """Build a :class:`RunnerSim` whose worlds wear shoes from sampled contexts.

    Args:
        body: Runner body parameters.
        world_count: Simulated worlds.
        split: ``"train"`` samples training contexts around the fit of
            ``shoe``, ``"eval"`` takes that fit.
        rng: Generator of the context draws.
        belt_speed_m_s: Treadmill belt speed of the reference [m/s].
        shoe: Shoe whose fit and geometry center the contexts, by :data:`GEOMETRY_NAMES` name.
        device: Simulation device.

    Returns:
        The simulator and the context ``xi`` [world_count, XI_DIM] of each world.
    """
    shoes = {s.name: s for s in load_shoes(SHOE_DATASET)}
    space = ShoeSpace(
        center=encode(shoes[shoe].left.material, RunnerSimConfig.friction_coefficient, GEOMETRY_NAMES.index(shoe))
    )
    xi = space.sample_training(world_count, rng) if split == "train" else space.sample_held_out(world_count)
    materials, friction, geometry = decode(xi)
    sim = RunnerSim(
        body,
        [shoes[name] for name in GEOMETRY_NAMES],
        geometry,
        config=RunnerSimConfig(belt_speed_m_s=belt_speed_m_s),
        device=device,
    )
    sim.set_world_materials(materials, friction)
    return sim, xi


def episode_starts(ref: dict, split: str, device: str = "cuda") -> torch.Tensor:
    """Return the reference flight frames of ``split`` windows as (window, frame) rows [n, 2]."""
    in_split = ref["split"][ref["flight_window"]] == split
    return torch.as_tensor(np.stack((ref["flight_window"][in_split], ref["flight_frame"][in_split]), 1), device=device)


def collect(
    reference: Path,
    world_count: int,
    steps: int,
    *,
    split: str = "train",
    shoe: str = CENTER_SHOE,
    seed: int = 0,
    config: Stage0Config | None = None,
) -> tuple[ReplayBuffer, dict]:
    """Run the stage-0 controller and return the filled buffer and per-world metadata.

    Args:
        reference: Folder written by :func:`projects.supermimic.reference.prepare_reference`.
        world_count: Simulated worlds.
        steps: Control steps per world; also the buffer capacity.
        split: ``"train"`` draws training contexts and train windows,
            ``"eval"`` takes the held-out context and eval windows.
        shoe: Shoe whose fit centers the contexts, by :data:`GEOMETRY_NAMES` name.
        seed: Seed of every random draw.
        config: Controller and episode settings.
    """
    config = config or Stage0Config()
    device = "cuda"
    rng = np.random.default_rng(seed)
    generator = torch.Generator(device).manual_seed(seed)
    worn = load_shoe(load_subject(reference / "subject.json").shoe)
    body = load_body_params(reference / "body_params.json")
    with np.load(reference / "reference.npz") as f:
        ref = {k: f[k] for k in f.files}

    sim, xi = make_sim(body, world_count, split, rng, float(ref["belt_speed_m_s"]), shoe, device)

    sim_cfg = sim.config
    stride = round(sim_cfg.control_dt_s / float(ref["frame_dt"]))
    frames = ref["joint_q"].shape[1]
    ref_q = torch.as_tensor(ref["joint_q"], dtype=torch.float32, device=device)
    ref_qd = torch.as_tensor(ref["joint_qd"], dtype=torch.float32, device=device)
    ref_moment = torch.as_tensor(reference_moments(ref, body, worn), dtype=torch.float32, device=device)
    ref_contact = torch.as_tensor(ref["contact"], device=device)
    # The left joints have the limits of the right ones.
    stiffness_limits = torch.tensor(STIFFNESS_LIMITS * 2, device=device)
    damping_limits = torch.tensor(DAMPING_LIMITS * 2, device=device)
    q_limits = torch.tensor(body.joint_limits * 2, device=device)
    starts = episode_starts(ref, split, device)

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
    stiffness_quantile = torch.zeros((w, JOINT_COUNT), device=device)
    damping = torch.zeros_like(stiffness_quantile)
    noise = torch.zeros_like(stiffness_quantile)
    decay = float(np.exp(-sim_cfg.control_dt_s / config.noise_time_constant_s))
    reset = torch.ones(w, dtype=torch.bool, device=device)
    failures = torch.zeros(w, dtype=torch.int64, device=device)
    short_steps = torch.zeros(JOINT_COUNT, dtype=torch.int64, device=device)
    loaded = torch.zeros((w, 2), dtype=torch.bool, device=device)
    unloaded_stance_steps = torch.zeros(2, dtype=torch.int64, device=device)

    with torch.cuda.stream(wp.stream_to_torch(sim.device)):
        for _ in range(steps):
            if reset.any():
                n = int(reset.sum())
                pick = starts[torch.randint(len(starts), (n,), generator=generator, device=device)]
                window[reset], frame[reset] = pick[:, 0], pick[:, 1]
                sim.reset(
                    wp.from_torch(reset), wp.from_torch(ref_q[window, frame]), wp.from_torch(ref_qd[window, frame])
                )
                stiffness_quantile[reset] = torch.rand((n, JOINT_COUNT), generator=generator, device=device)
                damping[reset] = _log_uniform(damping_limits, n, generator)
                noise[reset] = config.noise_std_rad * torch.randn((n, JOINT_COUNT), generator=generator, device=device)
                loaded[reset] = False
            state = sim_state(sim)
            done = state[:, 2] < config.fall_height_m
            stance = ref_contact[window, frame]
            unloaded_stance_steps += (stance & ~loaded).sum(0)
            action, short = reference_action(
                ref_q[window, frame, 7:],
                ref_qd[window, frame, 6:],
                loaded_moment(ref_moment[window, frame], stance, loaded),
                damping,
                stiffness_quantile,
                noise,
                q_limits,
                stiffness_limits,
                config.max_posture_lead_rad,
            )
            short_steps += short.sum(0)
            sim.step(wp.from_torch(action.contiguous()))
            aux, after = sim_aux(sim, state), sim_state(sim)
            failed = ~(
                torch.isfinite(aux).all(1) & (after[:, -JOINT_COUNT:].abs().amax(1) < config.failure_joint_speed_rad_s)
            )
            failures += failed
            loaded = aux[:, [2, 5]] > config.touchdown_force_n
            buffer.push(
                state, action, xi_t, aux.masked_fill(failed[:, None], 0.0), window * frames + frame, reset, done
            )
            frame += stride
            reset = done | failed | (frame >= frames)
            innovation = torch.randn((w, JOINT_COUNT), generator=generator, device=device)
            noise = decay * noise + config.noise_std_rad * float(np.sqrt(1.0 - decay**2)) * innovation

    meta = {
        "dt_s": sim_cfg.control_dt_s,
        "shoe": [GEOMETRY_NAMES[g] for g in decode(xi)[2]],
        "split": split,
        "center_shoe": shoe,
        "seed": seed,
        "reference": str(reference),
        "reference_frames_per_window": frames,
        "xi_coordinates": [c[0] for c in XI_COORDINATES],
        "config": asdict(config),
        "failures": failures.tolist(),
        "stiffness_short_steps": dict(zip(JOINT_NAMES, short_steps.tolist(), strict=True)),
        "unloaded_stance_steps": unloaded_stance_steps.tolist(),
        "xi": xi.tolist(),
    }
    return buffer, meta


def summarize(buffer: ReplayBuffer, meta: dict) -> dict:
    """Count rows, episodes, falls, touchdowns, short stiffness and sim failures of a collection.

    Failure rates are per simulated world-second, overall, per shoe geometry
    and per quartile of each ``xi`` coordinate over the worlds.
    """
    data = buffer.data
    seconds = buffer.size * meta["dt_s"]
    failures = np.asarray(meta["failures"])
    xi = np.asarray(meta["xi"])
    shoes = np.asarray(meta["shoe"])
    quartile = [
        np.searchsorted(np.quantile(column, (0.25, 0.5, 0.75)), column, side="right")
        for column in xi[:, : len(meta["xi_coordinates"])].T
    ]
    by_xi = {
        name: [
            float(failures[quartile[i] == b].sum() / max(seconds * (quartile[i] == b).sum(), 1e-9)) for b in range(4)
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
        "stiffness_short_fraction": {
            name: count / (buffer.size * buffer.world_count) for name, count in meta["stiffness_short_steps"].items()
        },
        "sim_failures": int(failures.sum()),
        "sim_failures_per_world_s": float(failures.sum() / (seconds * buffer.world_count)),
        "sim_failures_per_world_s_by_shoe": by_shoe,
        "sim_failures_per_world_s_by_xi_quartile": by_xi,
    }


def read_episode(folder: str | Path, world: int, episode: int) -> dict:
    """Return one episode of a dataset folder in :class:`~projects.supermimic.sim.RunnerSim` layouts.

    Args:
        folder: Dataset folder written by :func:`main`.
        world: World index.
        episode: Episode index within the world, in order.

    Returns:
        ``joint_q`` [T, 17], ``joint_qd`` [T, 16], ``grf`` and ``cop`` [T, 2, 3]
        (in the treadmill frame),
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
    grf, cop = aux[:, :6].reshape(-1, 2, 3), aux[:, 6:12].reshape(-1, 2, 3).copy()
    cop[..., :2] += np.where(grf[..., 2:] > 0.0, state[:, None, :2], 0.0)
    return {
        "joint_q": np.concatenate((state[:, :7], state[:, 13 : 13 + j]), 1),
        "joint_qd": np.concatenate((state[:, 7:13], state[:, 13 + j :]), 1),
        "grf": grf,
        "cop": cop,
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
    parser.add_argument(
        "--reference", type=Path, default=Path("outputs/supermimic/reference_F01_2026-10-22"), help="reference folder"
    )
    parser.add_argument("--worlds", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=1000, help="control steps per world")
    parser.add_argument("--split", choices=("train", "eval"), default="train")
    parser.add_argument(
        "--shoe", choices=GEOMETRY_NAMES, default=CENTER_SHOE, help="shoe whose fit centers the contexts"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    output = args.output or Path("outputs/supermimic") / f"stage0_{args.split}_{datetime.date.today()}"
    output.mkdir(parents=True, exist_ok=False)
    buffer, meta = collect(args.reference, args.worlds, args.steps, split=args.split, shoe=args.shoe, seed=args.seed)
    meta["summary"] = summarize(buffer, meta)
    buffer.save(output / "buffer.npz")
    (output / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta["summary"], indent=2))


if __name__ == "__main__":
    main()
