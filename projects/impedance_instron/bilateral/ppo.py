# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Train the bilateral neural impedance policy with a self-contained PPO.

Clipped-surrogate PPO with generalized advantage estimation, an asymmetric
critic, running observation normalization, KL-adaptive learning rate, value
and value bootstrapping at time-limit truncations.
Only training segments are used; evaluation segments never select a policy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from . import env as planar_env
from .env import BilateralEnv, EnvConfig, ReferenceLibrary
from .policy import ActorCritic, parameter_count, save_policy


@dataclass(frozen=True)
class PPOConfig:
    """PPO hyperparameters."""

    worlds: int = 4096
    horizon: int = 32
    """Control steps collected per world per iteration."""
    iterations: int = 1500
    epochs: int = 5
    minibatches: int = 4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    learning_rate: float = 3.0e-4
    desired_kl: float = 0.01
    value_coef: float = 1.0
    entropy_coef: float = 0.0
    max_grad_norm: float = 1.0
    log_std: float = -1.0
    hidden: tuple[int, ...] = field(default=(256, 256))
    seed: int = 0
    save_every: int = 100


def _gae(rewards, values, dones, last_value, gamma, lam):
    advantages = torch.zeros_like(rewards)
    running = torch.zeros_like(last_value)
    for k in reversed(range(len(rewards))):
        next_value = last_value if k == len(rewards) - 1 else values[k + 1]
        mask = 1.0 - dones[k]
        delta = rewards[k] + gamma * next_value * mask - values[k]
        running = delta + gamma * lam * mask * running
        advantages[k] = running
    return advantages, advantages + values


def train(
    dataset: str | Path,
    output: str | Path,
    *,
    ppo: PPOConfig | None = None,
    env_config=None,
    mount_m=None,
    pitch_rad: float | None = None,
    device: str = "cuda:0",
    kind: str = "planar",
    resume: Path | None = None,
    tensorboard: Path | None = None,
) -> Path:
    """Train a policy on the dataset's training segments and write checkpoints to ``output``.

    Args:
        kind: ``"planar"`` for the sagittal two-leg model or ``"3d"`` for the
            Newton articulation of :mod:`.env3d`.
    """
    ppo = ppo or PPOConfig()
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite a nonempty run directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(ppo.seed)
    if kind == "3d":
        from . import env3d  # noqa: PLC0415

        env_config = env_config or env3d.Env3DConfig()
        library = env3d.ReferenceLibrary3D(dataset, ("train",), env_config.control_dt_s, device)
        env = env3d.BilateralEnv3D(library, ppo.worlds, env_config, device=device, seed=ppo.seed)
        interface = env3d
        seed_logits = env3d.seed_action(env.bounds)
    elif kind == "planar":
        env_config = env_config or EnvConfig()
        library = ReferenceLibrary(dataset, ("train",), env_config.control_dt_s, device)
        manifest = library.manifest
        assets = manifest["shared_assets"]
        env = BilateralEnv(
            library,
            library.root / assets["digital_shoe_left.json"]["file"],
            library.root / assets["digital_shoe_right.json"]["file"],
            mount_m if mount_m is not None else manifest["shoe"]["mount_m"],
            pitch_rad if pitch_rad is not None else manifest["shoe"]["pitch_rad"],
            ppo.worlds,
            env_config,
            device=device,
            seed=ppo.seed,
        )
        interface = planar_env
        seed_logits = None
    else:
        raise ValueError("kind must be 'planar' or '3d'")
    root = library.root
    torch_device = env.torch_device
    model = ActorCritic(
        env.privileged_size,
        ppo.hidden,
        ppo.log_std,
        observation_names=getattr(env, "observation_names", interface.OBSERVATION_NAMES),
        action_names=interface.ACTION_NAMES,
        seed_logits=seed_logits,
    ).to(torch_device)
    if resume is not None:
        # Fine-tune: keep actor, critic, and normalizers; the critic re-fits the new reward.
        model.load_state_dict(torch.load(resume, map_location=torch_device, weights_only=True)["model"])
    optimizer = torch.optim.Adam(model.parameters(), lr=ppo.learning_rate)
    learning_rate = ppo.learning_rate
    provenance = {
        "dataset": str(root),
        "manifest_sha256": hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
        "training_members": [m["id"] for m in library.members],
        "kind": kind,
        "resumed_from": None if resume is None else str(Path(resume).resolve()),
        "ppo": asdict(ppo),
        "environment": env.describe(),
        "actor_parameters": parameter_count(model.actor),
        "critic_parameters": parameter_count(model.critic),
    }
    (output / "config.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(
        f"actor parameters {provenance['actor_parameters']}, critic {provenance['critic_parameters']}, "
        f"{len(library)} training segments, {len(library.initial_candidates)} flight initial states",
        flush=True,
    )

    env.reset()
    observation, privileged = env.observe(), env.privileged()
    W, H = ppo.worlds, ppo.horizon
    episode_return = torch.zeros(W, device=torch_device)
    episode_length = torch.zeros(W, device=torch_device)
    finished_returns, finished_lengths, finished_failures = [], [], []
    log_path = output / "progress.csv"
    writer = None
    board = None
    if tensorboard is not None:
        from torch.utils.tensorboard import SummaryWriter  # noqa: PLC0415

        board = SummaryWriter(log_dir=str(tensorboard))
    best = -float("inf")
    for iteration in range(1, ppo.iterations + 1):
        start = time.perf_counter()
        obs_buf = torch.zeros(H, W, env.observation_size, device=torch_device)
        priv_buf = torch.zeros(H, W, env.privileged_size, device=torch_device)
        act_buf = torch.zeros(H, W, env.action_size, device=torch_device)
        logp_buf = torch.zeros(H, W, device=torch_device)
        value_buf = torch.zeros(H, W, device=torch_device)
        reward_buf = torch.zeros(H, W, device=torch_device)
        done_buf = torch.zeros(H, W, device=torch_device)
        terms_sum: dict[str, float] = {}
        failed_sum = 0.0
        with torch.no_grad():
            for k in range(H):
                model.actor.normalizer.update(observation)
                model.critic_normalizer.update(torch.cat((observation, privileged), dim=-1))
                dist = model.distribution(observation)
                action = dist.sample()
                obs_buf[k], priv_buf[k], act_buf[k] = observation, privileged, action
                logp_buf[k] = dist.log_prob(action).sum(-1)
                value_buf[k] = model.value(observation, privileged)
                observation, privileged, reward, terminated, truncated, info = env.step(action)
                bootstrap = model.value(info["final_observation"], info["final_privileged"])
                reward_buf[k] = reward + ppo.gamma * bootstrap * truncated.float()
                done = (terminated | truncated).float()
                done_buf[k] = done
                episode_return += reward
                episode_length += 1
                ended = done.bool()
                if ended.any():
                    finished_returns.append(episode_return[ended].mean().item())
                    finished_lengths.append(episode_length[ended].mean().item())
                    finished_failures.append(terminated[ended].float().mean().item())
                    episode_return[ended] = 0
                    episode_length[ended] = 0
                for name, value in info["terms"].items():
                    terms_sum[name] = terms_sum.get(name, 0.0) + value / H
                failed_sum += info["failed"] / H
            last_value = model.value(observation, privileged)
            advantages, returns = _gae(reward_buf, value_buf, done_buf, last_value, ppo.gamma, ppo.lam)
        collect_s = time.perf_counter() - start

        flat = lambda x: x.reshape(H * W, *x.shape[2:])  # noqa: E731
        b_obs, b_priv, b_act = flat(obs_buf), flat(priv_buf), flat(act_buf)
        b_logp, b_adv, b_ret, b_val = flat(logp_buf), flat(advantages), flat(returns), flat(value_buf)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
        size = H * W // ppo.minibatches
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "kl": 0.0}
        updates = 0
        for _ in range(ppo.epochs):
            order = torch.randperm(H * W, device=torch_device)
            for m in range(ppo.minibatches):
                index = order[m * size : (m + 1) * size]
                dist = model.distribution(b_obs[index])
                logp = dist.log_prob(b_act[index]).sum(-1)
                ratio = torch.exp(logp - b_logp[index])
                adv = b_adv[index]
                policy_loss = -torch.min(ratio * adv, ratio.clamp(1 - ppo.clip, 1 + ppo.clip) * adv).mean()
                value = model.value(b_obs[index], b_priv[index])
                clipped = b_val[index] + (value - b_val[index]).clamp(-ppo.clip, ppo.clip)
                value_loss = torch.max((value - b_ret[index]) ** 2, (clipped - b_ret[index]) ** 2).mean()
                entropy = dist.entropy().sum(-1).mean()
                loss = policy_loss + ppo.value_coef * value_loss - ppo.entropy_coef * entropy
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), ppo.max_grad_norm)
                optimizer.step()
                with torch.no_grad():
                    kl = (b_logp[index] - logp).mean().item()
                if ppo.desired_kl > 0:
                    if kl > 2.0 * ppo.desired_kl:
                        learning_rate = max(1e-5, learning_rate / 1.5)
                    elif 0 < kl < 0.5 * ppo.desired_kl:
                        learning_rate = min(1e-3, learning_rate * 1.5)
                    for group in optimizer.param_groups:
                        group["lr"] = learning_rate
                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["kl"] += kl
                updates += 1
        stats = {name: value / updates for name, value in stats.items()}
        elapsed = time.perf_counter() - start
        recent = slice(-50, None)
        row = {
            "iteration": iteration,
            "samples": iteration * H * W,
            "mean_step_reward": reward_buf.mean().item(),
            "episode_return": float(np.mean(finished_returns[recent])) if finished_returns else 0.0,
            "episode_length_s": float(np.mean(finished_lengths[recent])) * env_config.control_dt_s
            if finished_lengths
            else 0.0,
            "termination_fraction": float(np.mean(finished_failures[recent])) if finished_failures else 0.0,
            "screen_failure_fraction": failed_sum,
            **{f"reward_{name}": value for name, value in terms_sum.items()},
            **stats,
            "learning_rate": learning_rate,
            "action_std": model.log_std.exp().mean().item(),
            "samples_per_s": H * W / elapsed,
            "collect_fraction": collect_s / elapsed,
        }
        finished_returns, finished_lengths, finished_failures = (
            finished_returns[-50:],
            finished_lengths[-50:],
            finished_failures[-50:],
        )
        with log_path.open("a", newline="", encoding="utf-8") as handle:
            if writer is None:
                writer = list(row)
                csv.DictWriter(handle, fieldnames=writer).writeheader()
            csv.DictWriter(handle, fieldnames=writer).writerow(row)
        if board is not None:
            for name, value in row.items():
                if name not in ("iteration", "samples"):
                    board.add_scalar(name, value, row["samples"])
        if iteration % 10 == 0 or iteration == 1:
            print(
                f"it {iteration:5d} reward {row['mean_step_reward']:.3f} len {row['episode_length_s']:.2f} s "
                f"term {row['termination_fraction']:.2f} pose {terms_sum.get('pose', 0):.2f} "
                f"force {terms_sum.get('force', 0):.2f} pos {terms_sum.get('position', 0):.2f} "
                f"kl {stats['kl']:.4f} lr {learning_rate:.1e} std {row['action_std']:.3f} "
                f"{row['samples_per_s']:.0f} samples/s",
                flush=True,
            )
        metadata = {
            "iteration": iteration,
            "samples": row["samples"],
            "training": {k: row[k] for k in ("mean_step_reward", "episode_return", "episode_length_s")},
            "environment": env.describe(),
            "dataset": provenance["dataset"],
            "kind": kind,
            "manifest_sha256": provenance["manifest_sha256"],
        }
        if iteration % ppo.save_every == 0 or iteration == ppo.iterations:
            save_policy(model.actor, output / f"policy_{iteration:05d}.pt", metadata)
            torch.save(
                {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": iteration},
                output / "checkpoint.pt",
            )
        # Track the best training-episode return; evaluation segments never select the policy.
        if row["episode_return"] > best and iteration > 20:
            best = row["episode_return"]
            save_policy(model.actor, output / "policy_best_train.pt", metadata)
    final = save_policy(model.actor, output / "policy.pt", metadata)
    if board is not None:
        board.close()
    return final


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=PPOConfig.worlds)
    parser.add_argument("--iterations", type=int, default=PPOConfig.iterations)
    parser.add_argument("--horizon", type=int, default=PPOConfig.horizon)
    parser.add_argument("--learning-rate", type=float, default=PPOConfig.learning_rate)
    parser.add_argument("--gamma", type=float, default=PPOConfig.gamma)
    parser.add_argument("--log-std", type=float, default=PPOConfig.log_std)
    parser.add_argument("--hidden", type=int, nargs="+", default=list(PPOConfig.hidden))
    parser.add_argument("--seed", type=int, default=PPOConfig.seed)
    parser.add_argument("--save-every", type=int, default=PPOConfig.save_every)
    parser.add_argument("--substeps", type=int, default=EnvConfig.substeps)
    parser.add_argument("--episode-s", type=float, default=EnvConfig.episode_s)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--kind", choices=("planar", "3d"), default="planar")
    parser.add_argument("--resume", type=Path, help="Initialize from a run's checkpoint.pt (fine-tuning).")
    parser.add_argument("--reward-kernels", choices=("hybrid", "per_term", "pooled"), default="hybrid")
    parser.add_argument("--force-penalty", type=float, default=EnvConfig.force_penalty)
    parser.add_argument("--response-time-s", type=float, default=EnvConfig.response_time_s)
    parser.add_argument("--body-position-weight", type=float, default=EnvConfig.body_position_weight)
    parser.add_argument("--body-velocity-weight", type=float, default=EnvConfig.body_velocity_weight)
    parser.add_argument("--contact-match-penalty", type=float, default=EnvConfig.contact_match_penalty)
    parser.add_argument("--action-smoothness-penalty", type=float, default=EnvConfig.action_smoothness_penalty)
    parser.add_argument("--previous-action-obs", action="store_true")
    parser.add_argument("--tensorboard", type=Path, help="Also log the progress scalars to this TensorBoard directory.")
    args = parser.parse_args(argv)
    ppo = PPOConfig(
        worlds=args.worlds,
        horizon=args.horizon,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        log_std=args.log_std,
        hidden=tuple(args.hidden),
        seed=args.seed,
        save_every=args.save_every,
    )
    if args.kind == "3d":
        from .env3d import Env3DConfig  # noqa: PLC0415

        env_config = Env3DConfig(substeps=args.substeps, episode_s=args.episode_s)
    else:
        env_config = EnvConfig(
            substeps=args.substeps,
            episode_s=args.episode_s,
            reward_kernels=args.reward_kernels,
            force_penalty=args.force_penalty,
            response_time_s=args.response_time_s,
            body_position_weight=args.body_position_weight,
            body_velocity_weight=args.body_velocity_weight,
            contact_match_penalty=args.contact_match_penalty,
            action_smoothness_penalty=args.action_smoothness_penalty,
            previous_action_obs=args.previous_action_obs,
        )
    print(
        train(
            args.dataset,
            args.output,
            ppo=ppo,
            env_config=env_config,
            device=args.device,
            kind=args.kind,
            resume=args.resume,
            tensorboard=args.tensorboard,
        )
    )


if __name__ == "__main__":
    main()
