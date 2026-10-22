# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""F01 tracking task of the SuperMimic loop (M4).

:class:`TrackerTask` connects the :class:`~projects.supermimic.world_model.WorldModel`,
the F01 reference and the :mod:`.tracking` reward to the imagination hooks of
:mod:`.train`. The imagined state is the world model's model state, whose last
row is the previous action.

The actor sees the current sim state without its horizontal root position, the
previous action in normalized units and the reference at the next
:attr:`TrackerConfig.goal_frames` control steps without horizontal root
position. Its equilibrium angles are absolute and bounded by the joint limits;
stiffness and damping are log-scale bounded by the ranges the world model was
fit on.
"""

from __future__ import annotations

import argparse
import datetime
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import warp as wp

from .body import JOINT_COUNT, STATE_DIM, BodyParams
from .buffer import ReplayBuffer
from .collect import (
    AUX_DIM,
    Stage0Config,
    episode_starts,
    make_sim,
    reference_action,
    reference_moments,
    sim_aux,
    sim_state,
)
from .context import GEOMETRY_NAMES, XI_DIM, encode
from .policy import Actor, Critic, impedance_action_bounds
from .reference import load_body_params, load_shoe, load_subject
from .sim import RunnerSim, RunnerSimConfig, Shoe
from .tracking import (
    LegKinematics,
    joint_limit_violation,
    step_reward,
    termination,
    termination_probability,
    tracking_errors,
    tracking_reward,
)
from .train import ImaginationHooks, Trainer, TrainerConfig, cosine_lr_scale, train
from .world_model import WorldModel

__all__ = [
    "TrackerCollector",
    "TrackerConfig",
    "TrackerTask",
    "fixed_shoe_sim",
    "main",
    "reference_states",
    "world_model_fit",
]


@dataclass(frozen=True)
class TrackerConfig:
    """Goal of the tracker."""

    goal_frames: int = 10
    """Future control steps of the reference the actor sees."""


def reference_states(joint_q: torch.Tensor, joint_qd: torch.Tensor) -> torch.Tensor:
    """Return reference frames in the :data:`~projects.supermimic.body.STATE_DIM` layout.

    Args:
        joint_q: Generalized coordinates of the reference, shape [..., 17].
        joint_qd: Generalized velocities of the reference, shape [..., 16].
    """
    return torch.cat((joint_q[..., :7], joint_qd[..., :6], joint_q[..., 7:], joint_qd[..., 6:]), -1)


class TrackerTask:
    """Imagination hooks of F01 tracking through a world model.

    Reference frame indices ``k`` are flat, ``window * frames + frame``, as in
    the replay buffer's ``reference`` field.

    Args:
        world_model: Fitted world model; needs a history of at least one step.
        reference: Reference frames, shape [windows, frames, STATE_DIM].
        moment: Reference inverse-dynamics joint moments, shape [windows, frames, joints] [N m].
        frame_stride: Reference frames per control step.
        body: Body parameters, for the foot kinematics and the joint limits.
        dt: Control period [s].
        config: Goal.
    """

    def __init__(
        self,
        world_model: WorldModel,
        reference: torch.Tensor,
        moment: torch.Tensor,
        frame_stride: int,
        body: BodyParams,
        dt: float,
        config: TrackerConfig,
    ):
        if world_model.history_length < 1:
            raise ValueError("The tracker reads the previous action from the model state history")
        self.world_model = world_model
        self.frames = reference.shape[1]
        self.reference = reference.reshape(-1, STATE_DIM)
        self.moment = moment.reshape(-1, moment.shape[-1])
        self.frame_stride = frame_stride
        self.body = body
        self.dt = dt
        self.config = config
        self.kinematics = LegKinematics(body, reference.device)
        joints = world_model.joint_count
        self.action_low, self.action_high, self.action_log_scale = impedance_action_bounds(
            body.joint_limits, device=reference.device
        )
        self.action_shape = (joints, 3)
        self.observation_size = STATE_DIM - 2 + 3 * joints
        self.goal_size = (STATE_DIM - 2) * config.goal_frames

    def start_action(self, k: torch.Tensor) -> torch.Tensor:
        """Return the previous action at an episode start at reference frame ``k``, shape [batch, joints, 3].

        It carries the reference moment like the stage-0 data: :func:`~projects.supermimic.collect.reference_action`
        at the stage-0 stiffness floor, capped at its upper bound, without noise, and with the damping at the
        geometric middle of its bounds.
        """
        joints = self.world_model.joint_count
        q_ref, qd_ref = self.reference[k, 13 : 13 + joints], self.reference[k, 13 + joints : 13 + 2 * joints]
        low, high = self.action_low, self.action_high
        action, _ = reference_action(
            q_ref,
            qd_ref,
            self.moment[k],
            (low[:, 2] * high[:, 2]).sqrt().expand_as(q_ref),
            torch.zeros_like(q_ref),
            torch.zeros_like(q_ref),
            torch.stack((low[:, 0], high[:, 0]), -1),
            torch.stack((low[:, 1], high[:, 1]), -1),
            Stage0Config.max_posture_lead_rad,
        )
        return action

    @property
    def hooks(self) -> ImaginationHooks:
        """Imagination hooks for :meth:`projects.supermimic.train.Trainer.update`."""
        return ImaginationHooks(
            dynamics=self.world_model,
            observe=self.observe,
            goal=self.goal,
            reward=self.reward,
            termination=self.termination,
            frame_stride=self.frame_stride,
        )

    def actor(self, *, hidden: tuple[int, ...] = (512, 256, 128)) -> Actor:
        """Return a new actor with the tracker's inputs and action bounds."""
        return Actor(
            self.observation_size + self.goal_size,
            self.action_low,
            self.action_high,
            self.action_log_scale,
            hidden=hidden,
        ).to(self.reference.device)

    def normalized_action(self, action: torch.Tensor) -> torch.Tensor:
        """Return actions [..., joints, 3] with q_eq in rad and K, B mapped log-linearly onto [0, 1] of their bounds."""
        log_low, log_high = self.action_low[..., 1:].log(), self.action_high[..., 1:].log()
        impedance = (action[..., 1:].log() - log_low) / (log_high - log_low)
        return torch.cat((action[..., :1], impedance), -1)

    def previous_action(self, model_state: torch.Tensor) -> torch.Tensor:
        """Return the previous action held in a model state, shape [batch, joints, 3]."""
        return model_state[:, -3 * self.world_model.joint_count :].unflatten(1, self.action_shape)

    def frame_index(self, k: torch.Tensor, steps: int | torch.Tensor) -> torch.Tensor:
        """Return the flat frame ``steps`` control steps after ``k``, held at the last frame of its window."""
        window, frame = k // self.frames, k % self.frames
        return window * self.frames + torch.clamp(frame + steps * self.frame_stride, max=self.frames - 1)

    def fits_horizon(self, k: torch.Tensor, horizon: int) -> torch.Tensor:
        """Return whether ``horizon`` control steps from frame ``k`` stay inside its reference window."""
        return k % self.frames + horizon * self.frame_stride < self.frames

    def observation(self, state: torch.Tensor, previous_action: torch.Tensor) -> torch.Tensor:
        """Return the actor observation of sim states [batch, STATE_DIM] and the previous actions [batch, joints, 3]."""
        return torch.cat((state[:, 2:], self.normalized_action(previous_action).flatten(1)), -1)

    def observe(self, model_state: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        """Return the actor observation of model states."""
        return self.observation(model_state[:, :STATE_DIM], self.previous_action(model_state))

    def goal(self, k: torch.Tensor) -> torch.Tensor:
        """Return the reference at the next goal frames without horizontal root position."""
        steps = torch.arange(1, self.config.goal_frames + 1, device=k.device)
        return self.reference[self.frame_index(k[:, None], steps)][..., 2:].flatten(1)

    def reward(
        self,
        model_state: torch.Tensor,
        action: torch.Tensor,
        model_state_next: torch.Tensor,
        aux: torch.Tensor,
        k: torch.Tensor,
    ) -> torch.Tensor:
        """Return the step reward of reaching ``model_state_next`` at reference frame ``k``."""
        state = model_state_next[:, :STATE_DIM]
        task = tracking_reward(tracking_errors(state, self.reference[k], self.kinematics))
        violation = joint_limit_violation(state[:, 13:23], self.body.joint_limits)
        previous = self.normalized_action(self.previous_action(model_state))
        return step_reward(task, self.normalized_action(action), previous, violation, self.dt)

    def termination(self, model_state_next: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Return the soft termination probability at reference frame ``k``."""
        return termination_probability(model_state_next[:, :STATE_DIM], self.reference[k])

    def sample_starts(self, buffer: ReplayBuffer, batch_size: int, horizon: int) -> tuple[torch.Tensor, ...]:
        """Sample imagination starts from non-terminal buffer rows whose horizon fits in the reference window.

        At an episode's first row the history holds only that row, so its previous action
        is set to :meth:`start_action`, the one the actor sees in the sim.

        Returns:
            Model states, sim contexts and reference frame indices; at most ``batch_size`` rows.
        """
        h = self.world_model.history_length
        rows = buffer.sample(2 * batch_size, 0, history=h)
        keep = self.fits_horizon(rows["reference"][:, h], horizon) & ~rows["done"][:, h]
        index = torch.nonzero(keep).squeeze(1)[:batch_size]
        k = rows["reference"][index, h]
        action = rows["action"][index, :h]
        first = rows["episode_step"][index, h] == 0
        action[first, -1] = self.start_action(k[first])
        model_state = self.world_model.start(rows["state"][index], action)
        return model_state, rows["xi"][index, h], k


class TrackerCollector:
    """Run the actor with noise on its logits in sim worlds and push the rows to a replay buffer.

    Episodes start at reference flight frames and end on :func:`~projects.supermimic.tracking.termination`,
    on a sim failure (non-finite aux or a joint speed above ``failure_joint_speed``)
    or at the end of the reference window. A failed step's state is not stored.
    At an episode start the previous action is :meth:`TrackerTask.start_action`.

    Args:
        task: Tracker task.
        sim: Sim worlds.
        xi: Sim context per world, shape [worlds, XI_DIM].
        joint_q: Reference coordinates, shape [windows, frames, 17].
        joint_qd: Reference velocities, shape [windows, frames, 16].
        starts: Episode starts as (window, frame) rows, shape [starts, 2].
        buffer: Buffer with one row per sim world.
        noise_std: Standard deviation of the noise on the actor's logits.
        failure_joint_speed: Joint speed that marks a sim failure [rad/s].
    """

    def __init__(
        self,
        task: TrackerTask,
        sim: RunnerSim,
        xi: torch.Tensor,
        joint_q: torch.Tensor,
        joint_qd: torch.Tensor,
        starts: torch.Tensor,
        buffer: ReplayBuffer,
        *,
        noise_std: float = 0.1,
        failure_joint_speed: float = Stage0Config.failure_joint_speed_rad_s,
    ):
        self.task, self.sim, self.xi, self.buffer = task, sim, xi, buffer
        self.joint_q, self.joint_qd, self.starts = joint_q, joint_qd, starts
        self.noise_std = noise_std
        self.failure_joint_speed = failure_joint_speed
        w, device = sim.world_count, xi.device
        self.window = torch.zeros(w, dtype=torch.int64, device=device)
        self.frame = torch.zeros_like(self.window)
        self.reset = torch.ones(w, dtype=torch.bool, device=device)
        self.previous = torch.zeros((w, *task.action_shape), device=device)
        self.episode_steps = torch.full_like(self.window, -1)
        """Alive steps of each world's episode; -1 before the first episode."""
        self.episode_return = torch.zeros(w, device=device)

    @torch.no_grad()
    def __call__(self, trainer: Trainer, steps: int = 1) -> dict[str, float]:
        """Advance every world ``steps`` control steps and return the step statistics."""
        task, sim, actor = self.task, self.sim, trainer.actor
        reward = torch.zeros((), device=self.xi.device)
        tracked = falls = failures = 0
        ended_steps, ended_return = [], []
        with torch.cuda.stream(wp.stream_to_torch(sim.device)):
            for _ in range(steps):
                reset = self.reset
                if reset.any():
                    ended = reset & (self.episode_steps >= 0)
                    ended_steps.append(self.episode_steps[ended])
                    ended_return.append(self.episode_return[ended])
                    self.episode_steps[reset], self.episode_return[reset] = 0, 0.0
                    pick = self.starts[torch.randint(len(self.starts), (int(reset.sum()),), device=reset.device)]
                    self.window[reset], self.frame[reset] = pick[:, 0], pick[:, 1]
                    q, qd = self.joint_q[self.window, self.frame], self.joint_qd[self.window, self.frame]
                    sim.reset(wp.from_torch(reset), wp.from_torch(q), wp.from_torch(qd))
                    self.previous[reset] = task.start_action(self.window[reset] * task.frames + self.frame[reset])
                state = sim_state(sim)
                k = self.window * task.frames + self.frame
                done = termination(state, task.reference[k])
                logits = actor.logits(torch.cat((task.observation(state, self.previous), task.goal(k)), -1))
                action = actor.squash(logits + self.noise_std * torch.randn_like(logits))
                sim.step(wp.from_torch(action.contiguous()))
                aux, after = sim_aux(sim, state), sim_state(sim)
                failed = ~(
                    torch.isfinite(aux).all(1)
                    & (after[:, -task.world_model.joint_count :].abs().amax(1) < self.failure_joint_speed)
                )
                self.buffer.push(state, action, self.xi, aux.masked_fill(failed[:, None], 0.0), k, reset, done)
                alive = ~(done | failed)
                errors = tracking_errors(after, task.reference[task.frame_index(k, 1)], task.kinematics)
                step = torch.where(alive, tracking_reward(errors), 0.0)
                reward += step.sum()
                self.episode_return += step * sim.config.control_dt_s
                self.episode_steps += alive.long()
                tracked += int(alive.sum())
                falls += int(done.sum())
                failures += int(failed.sum())
                self.previous = action
                self.frame += task.frame_stride
                self.reset = done | failed | (self.frame >= task.frames)
        seconds = steps * sim.world_count * sim.config.control_dt_s
        metrics = {
            "sim_tracking_reward": reward.item() / max(tracked, 1),
            "sim_falls_per_s": falls / seconds,
            "sim_failures_per_s": failures / seconds,
        }
        ended_steps = torch.cat(ended_steps) if ended_steps else self.episode_steps[:0]
        if len(ended_steps):
            metrics["sim_episodes"] = len(ended_steps)
            metrics["sim_episode_s"] = ended_steps.float().mean().item() * sim.config.control_dt_s
            metrics["sim_episode_return"] = torch.cat(ended_return).mean().item()
        return metrics


def fixed_shoe_sim(
    body: BodyParams, shoe: Shoe, world_count: int, belt_speed_m_s: float
) -> tuple[RunnerSim, np.ndarray]:
    """Build a :class:`RunnerSim` whose worlds all wear ``shoe`` with its fitted foams.

    Args:
        body: Runner body parameters.
        shoe: Shoe pair from the shoe dataset; its geometry must be one of
            :data:`~projects.supermimic.context.GEOMETRY_NAMES`.
        world_count: Simulated worlds.
        belt_speed_m_s: Treadmill belt speed of the reference [m/s].

    Returns:
        The simulator and the context ``xi`` [world_count, XI_DIM] of the shoe, equal in every world.
    """
    config = RunnerSimConfig(belt_speed_m_s=belt_speed_m_s)
    sim = RunnerSim(body, [shoe], np.zeros(world_count, dtype=np.int32), config=config)
    xi = encode(shoe.left.material, config.friction_coefficient, GEOMETRY_NAMES.index(shoe.name))
    return sim, np.tile(xi, (world_count, 1))


def world_model_fit(
    world_model: WorldModel,
    buffer: ReplayBuffer,
    config: TrainerConfig,
    *,
    window: int = 8,
    batches: int = 10,
    batch_size: int = 2048,
    lr: float = 2e-4,
) -> Callable[[], dict[str, float]]:
    """Return a world-model update: ``batches`` Adam steps of the k-step loss on buffer windows (paper Table A.5).

    The learning rate follows the trainer's cosine decay per update. The world model's
    normalization is left as it is: it is fitted once on the stage-0 data, whose actions
    span the action bounds, and refitting it would shift the inputs of the trained network.
    """
    optimizer = torch.optim.Adam(world_model.parameters(), lr=lr)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda update: cosine_lr_scale(update, config))
    h = world_model.history_length

    def fit() -> dict[str, float]:
        total = 0.0
        for _ in range(batches):
            rows = buffer.sample(batch_size, window, history=h)
            loss = world_model.loss(rows["state"], rows["action"][:, :-1], rows["xi"][:, 0], rows["aux"][:, :-1])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(world_model.parameters(), 1.0)
            optimizer.step()
            total += loss.detach()
        schedule.step()
        return {"world_model_loss": (total / batches).item()}

    return fit


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the SuperMimic F01 tracker (M4).")
    parser.add_argument(
        "--reference", type=Path, default=Path("outputs/supermimic/reference_F01_2026-10-22"), help="reference folder"
    )
    parser.add_argument("--stage0", type=Path, required=True, help="stage-0 buffer.npz; its world count sets --worlds")
    parser.add_argument(
        "--shoe",
        choices=("contexts", "subject"),
        default="contexts",
        help="worlds wear shoes from sampled training contexts, or all wear the subject's fitted shoe",
    )
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--pretrain", type=int, default=200, help="world-model updates on stage-0 data before the loop")
    parser.add_argument(
        "--update-to-data", type=float, default=0.4, help="policy updates per sim control step of every world"
    )
    parser.add_argument("--batch", type=int, default=4096, help="imagined rollouts per policy update")
    parser.add_argument("--model-hidden", type=int, nargs="+", default=[2048, 2048, 2048])
    parser.add_argument(
        "--critic-target-rate", type=float, default=1.0, help="Polyak rate of the target critic per update"
    )
    parser.add_argument("--checkpoint-every", type=int, default=500, help="iterations between checkpoints")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, help="new or empty output folder")
    args = parser.parse_args(argv)
    output = args.output or Path("outputs/supermimic") / f"tracker_{datetime.date.today()}"
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise SystemExit(f"{output} is not empty")
    torch.manual_seed(args.seed)
    device = "cuda"

    with np.load(args.reference / "reference.npz") as f:
        ref = {k: f[k] for k in f.files}
    body = load_body_params(args.reference / "body_params.json")
    shoe = load_shoe(load_subject(args.reference / "subject.json").shoe)
    joint_q = torch.as_tensor(ref["joint_q"], dtype=torch.float32, device=device)
    joint_qd = torch.as_tensor(ref["joint_qd"], dtype=torch.float32, device=device)
    buffer = ReplayBuffer.load(args.stage0, device=device)
    belt_speed = float(ref["belt_speed_m_s"])
    if args.shoe == "subject":
        sim, xi = fixed_shoe_sim(body, shoe, buffer.world_count, belt_speed)
    else:
        sim, xi = make_sim(body, buffer.world_count, "train", np.random.default_rng(args.seed), belt_speed)
    stride = round(sim.config.control_dt_s / float(ref["frame_dt"]))

    config = TrainerConfig(critic_target_rate=args.critic_target_rate)
    world_model = WorldModel(JOINT_COUNT, XI_DIM, AUX_DIM, hidden=tuple(args.model_hidden)).to(device)
    h = world_model.history_length
    rows = buffer.sample(16384, 8, history=h)
    world_model.fit_normalization(rows["state"], rows["action"][:, :-1], rows["xi"][:, 0], rows["aux"][:, :-1])
    fit_world_model = world_model_fit(world_model, buffer, config)
    log = (output / "log.jsonl").open("w", encoding="utf-8")
    start = time.perf_counter()
    for step in range(args.pretrain):
        metrics = fit_world_model()
        if step % 10 == 0 or step == args.pretrain - 1:
            print(
                json.dumps({"pretrain": step, "wall_s": time.perf_counter() - start, **metrics}), file=log, flush=True
            )

    moment = reference_moments(ref, body, shoe)
    task = TrackerTask(
        world_model,
        reference_states(joint_q, joint_qd),
        torch.as_tensor(moment, dtype=torch.float32, device=device),
        stride,
        body,
        sim.config.control_dt_s,
        TrackerConfig(),
    )
    critic = Critic(world_model.model_state_dim + task.goal_size + XI_DIM).to(device)
    trainer = Trainer(task.actor(), critic, config)
    collector = TrackerCollector(
        task,
        sim,
        torch.as_tensor(xi, dtype=torch.float32, device=device),
        joint_q,
        joint_qd,
        episode_starts(ref, "train"),
        buffer,
    )

    def save(iteration: int) -> None:
        torch.save(
            {
                **trainer.state_dict(),
                "world_model": world_model.state_dict(),
                "iteration": iteration,
                "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "goal_frames": task.config.goal_frames,
            },
            output / "checkpoint.pt",
        )

    # Sim steps per iteration alternate (e.g. 2, 3, 2, 3) so that their mean is 1 / update_to_data.
    collect_steps = iter(np.diff(np.floor(np.arange(args.iterations + 1) / args.update_to_data)).astype(int).tolist())

    def write(iteration: int, metrics: dict[str, float]) -> None:
        metrics["wall_s"] = time.perf_counter() - start
        metrics["gpu_peak_gb"] = torch.cuda.max_memory_allocated() / 2**30
        free, total = torch.cuda.mem_get_info()
        metrics["gpu_used_gb"] = (total - free) / 2**30
        print(json.dumps({"iteration": iteration, **metrics}), file=log, flush=True)
        if (iteration + 1) % args.checkpoint_every == 0:
            save(iteration + 1)

    train(
        trainer,
        task.hooks,
        args.iterations,
        collect=lambda t: collector(t, next(collect_steps)),
        fit_world_model=fit_world_model,
        sample_starts=lambda: task.sample_starts(buffer, args.batch, config.horizon),
        log=write,
    )
    save(args.iterations)


if __name__ == "__main__":
    main()
