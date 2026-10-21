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
    learning_rate_cap: float = 1.0e-3
    """Ceiling of the KL controller. A policy that starts from a converged checkpoint needs
    a ceiling below this: the controller climbs to it within one iteration, which overwrites
    the behaviour the checkpoint brought."""
    desired_kl: float = 0.01
    value_coef: float = 1.0
    entropy_coef: float = 0.0
    max_grad_norm: float = 1.0
    log_std: float = -1.0
    hidden: tuple[int, ...] = field(default=(256, 256))
    seed: int = 0
    save_every: int = 100


@dataclass(frozen=True)
class AdaptationConfig:
    """The history-conditioned shoe estimator of the RMA student.

    Args:
        history_steps: Control steps of state-action history the encoder reads.
        query_every: Control steps between estimator evaluations; the policy stays at
            100 Hz while the encoder runs at ``100 / query_every`` Hz, as RMA does.
        scalar_width: Width of the per-scalar projection of every history value.
        embedding: Width of the vector the policy consumes.
        auxiliary_weight: Weight of the shoe-readout loss next to the PPO loss at the
            first iteration; cosine-eased to :attr:`auxiliary_floor` by the last.
        auxiliary_worlds: Worlds whose history is copied out for the readout step. The
            estimator is a regression, so a few hundred worlds per query identify the
            encoder without holding a window per world of the batch on the GPU.
        freeze_encoder: Stop updating the encoder and train the policy against its
            errors: A-RMA's third phase, and the only arm that closes the gap between
            the estimate used to collect and the estimate used to deploy.
    """

    history_steps: int = 50
    query_every: int = 10
    scalar_width: int = 32
    embedding: int = 8
    auxiliary_weight: float = 1.0
    auxiliary_floor: float = 0.1
    auxiliary_epochs: int = 4
    auxiliary_worlds: int = 256
    learning_rate: float = 3.0e-4
    freeze_encoder: bool = False


def grow_inputs(state: dict, model: ActorCritic) -> dict:
    """Fit a checkpoint trained on a different observation list into ``model``.

    A policy interface here grows at the end: the contact channels, the shoe parameters,
    and the adaptation embedding are each appended to the environment's own channels, so a
    narrower interface is a prefix of a wider one. Copied columns keep their weights; new
    columns start at zero weight with a unit-variance normalizer, so the grown network is
    exactly the network it was until the new channels earn their weight. That is what
    makes the ablation ladder comparable: every arm starts from the same behaviour.

    Returns ``state`` itself when nothing had to change, so the caller can tell.
    """
    reference = model.state_dict()
    if all(key in reference and reference[key].shape == value.shape for key, value in state.items()):
        return state
    grown = {}
    for key, value in state.items():
        target = reference.get(key)
        if target is None or target.shape == value.shape:
            grown[key] = value
            continue
        if target.dim() != 1 and not (target.dim() == 2 and target.shape[0] == value.shape[0]):
            raise ValueError(f"cannot grow {key} from {tuple(value.shape)} to {tuple(target.shape)}")
        width = min(value.shape[-1], target.shape[-1])
        replacement = torch.zeros_like(target)
        if key.endswith("normalizer.var"):
            replacement.fill_(1.0)
        replacement[..., :width].copy_(value[..., :width])
        grown[key] = replacement
    return grown


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
    teacher: Path | None = None,
    adaptation: AdaptationConfig | None = None,
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
        # Record the step's torch work on capture's own reset, before the worlds are set up.
        env.capture()
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
    observation_names = tuple(getattr(env, "observation_names", interface.OBSERVATION_NAMES))
    encoder = encoder_optimizer = history = None
    if adaptation is not None:
        from . import adaptation as adaptive  # noqa: PLC0415
        from .shoe_randomization import N_THETA  # noqa: PLC0415

        if env.shoe_randomizer is None:
            raise ValueError("the student needs per-world shoes to infer: set --shoe-dr")
        if not getattr(env.config, "contact_observation", False):
            raise ValueError("the student needs the contact channels: set --contact-obs")
        if getattr(env.config, "theta_observation", False):
            raise ValueError("the student must not be handed the shoe parameters: unset --theta-obs")
        observation_names = observation_names + adaptive.EMBEDDING_NAMES
        encoder = adaptive.ShoeEncoder(
            env.observation_size + env.action_size,
            adaptation.history_steps,
            scalar_width=adaptation.scalar_width,
            embedding=adaptation.embedding,
            theta=N_THETA,
        ).to(torch_device)
        encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=adaptation.learning_rate)
        history = adaptive.HistoryBuffer(
            ppo.worlds, env.observation_size + env.action_size, adaptation.history_steps, torch_device
        )
    model = ActorCritic(
        env.privileged_size,
        ppo.hidden,
        ppo.log_std,
        observation_names=observation_names,
        action_names=interface.ACTION_NAMES,
        seed_logits=seed_logits,
    ).to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=ppo.learning_rate)
    learning_rate = ppo.learning_rate
    if resume is not None:
        # Fine-tune: keep actor, critic, normalizers, and the optimizer with its adapted
        # learning rate; a fresh Adam at the initial rate moves a converged, narrow policy
        # so far in one update that the KL schedule collapses the rate to its floor.
        checkpoint = torch.load(resume, map_location=torch_device, weights_only=True)
        grown = grow_inputs(checkpoint["model"], model)
        model.load_state_dict(grown)
        if grown is not checkpoint["model"]:
            # The interface changed width, so the saved Adam moments describe a network
            # that no longer exists. Keep the learning rate the schedule reached; drop
            # the moments. This is the one case where a fresh optimizer is the right call.
            learning_rate = checkpoint["optimizer"]["param_groups"][0]["lr"]
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        else:
            optimizer.load_state_dict(checkpoint["optimizer"])
            learning_rate = optimizer.param_groups[0]["lr"]
    if teacher is not None:
        # Interface surgery, not a fresh policy: the teacher's shoe parameters leave the
        # inputs and the adaptation embedding enters them, with the new columns at zero
        # weight, so the student starts exactly where the teacher stands.
        from .adaptation import EMBEDDING_NAMES, grow_actor  # noqa: PLC0415
        from .policy import Actor  # noqa: PLC0415
        from .shoe_randomization import THETA_OBSERVATION_NAMES  # noqa: PLC0415

        saved = torch.load(teacher, map_location=torch_device, weights_only=True)
        if "model" in saved:
            # A training checkpoint rather than a frozen policy file, which is what a fine-tune
            # should be handed: it carries the critic and both normalizers beside the actor and
            # the exploration width. A student with a fresh critic spends its first iterations on
            # advantage estimates that know nothing of the world the teacher was trained in, and
            # those updates move the policy off the behaviour the arm exists to keep. The channels
            # the two interfaces share are the leading columns of both, so the embedding the
            # student gains enters at zero weight with a unit-variance normalizer, exactly as it
            # does through the policy-file path below.
            model.load_state_dict(grow_inputs(saved["model"], model))
            print(
                f"teacher {teacher}: actor, critic, normalizers and exploration width carried "
                f"over from its iteration {saved.get('iteration')}",
                flush=True,
            )
        else:
            names = json.loads(Path(str(teacher).removesuffix(".pt") + ".json").read_text(encoding="utf-8"))[
                "observations"
            ]
            # The seed logits do not matter here - the checkpoint overwrites every weight on the
            # next line - but their width has to match, and the Actor's default seed is the planar
            # interface's, which is narrower than this one.
            source = Actor(
                tuple(saved["hidden"]),
                observation_names=tuple(names),
                action_names=interface.ACTION_NAMES,
                seed_logits=np.zeros(len(interface.ACTION_NAMES), dtype=np.float32),
            ).to(torch_device)
            source.load_state_dict(saved["state_dict"])
            model.actor = grow_actor(source, EMBEDDING_NAMES, drop_names=THETA_OBSERVATION_NAMES)
            # The noise is a parameter of the actor-critic rather than of the actor, so growing
            # the actor cannot carry it over and the student would otherwise explore at the
            # library default - three times a converged policy's width, which turns a policy
            # that runs into one that falls over inside two seconds. A teacher written before
            # the std was recorded says so rather than failing quietly.
            saved_std = saved.get("log_std")
            if saved_std is not None and tuple(saved_std.shape) == tuple(model.log_std.shape):
                with torch.no_grad():
                    model.log_std.copy_(saved_std.to(device=torch_device, dtype=model.log_std.dtype))
            else:
                print(
                    f"warning: teacher {teacher} carries no log_std; the student starts at "
                    f"{float(np.exp(ppo.log_std)):.3f} instead of the teacher's width",
                    flush=True,
                )
            print(f"teacher {teacher}: a frozen policy file, so its critic does not come across", flush=True)
        # An optimizer holds tensors rather than modules, so the actor grown above is invisible
        # to the one built from the model as it was: it would keep stepping a discarded network
        # while the policy in the model never moves. Recompute the parameters it owns, and start
        # under the cap - the controller only caps its own increases, so a rate above it buys one
        # oversized update before the first KL read can pull it back down.
        learning_rate = min(learning_rate, ppo.learning_rate_cap)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
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
        "adaptation": None if adaptation is None else asdict(adaptation),
        "estimator": None if encoder is None else encoder.describe(),
        "teacher": None if teacher is None else str(Path(teacher).resolve()),
    }
    (output / "config.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(
        f"actor parameters {provenance['actor_parameters']}, critic {provenance['critic_parameters']}, "
        f"{len(library)} training segments, {len(library.initial_candidates)} flight initial states",
        flush=True,
    )

    env.reset()
    observation, privileged = env.observe(), env.privileged()
    if history is not None:
        history.push(torch.cat((observation, torch.zeros(ppo.worlds, env.action_size, device=torch_device)), dim=-1))
    embedding = None
    W, H = ppo.worlds, ppo.horizon
    episode_return = torch.zeros(W, device=torch_device)
    episode_length = torch.zeros(W, device=torch_device)
    finished_returns, finished_lengths, finished_failures = [], [], []
    log_path = output / "progress.csv"
    readout_log: list[dict] = []
    writer = None
    board = None
    if tensorboard is not None:
        from torch.utils.tensorboard import SummaryWriter  # noqa: PLC0415

        board = SummaryWriter(log_dir=str(tensorboard))
    best = -float("inf")
    for iteration in range(1, ppo.iterations + 1):
        start = time.perf_counter()
        # The student's observation carries the adaptation embedding, so the buffer is
        # as wide as the actor's input rather than the environment's own output.
        obs_buf = torch.zeros(H, W, len(observation_names), device=torch_device)
        priv_buf = torch.zeros(H, W, env.privileged_size, device=torch_device)
        act_buf = torch.zeros(H, W, env.action_size, device=torch_device)
        logp_buf = torch.zeros(H, W, device=torch_device)
        value_buf = torch.zeros(H, W, device=torch_device)
        reward_buf = torch.zeros(H, W, device=torch_device)
        step_reward_buf = torch.zeros(H, W, device=torch_device)
        done_buf = torch.zeros(H, W, device=torch_device)
        terms_sum: dict[str, float] = {}
        failed_sum = 0.0
        snapshot_x: list[torch.Tensor] = []
        snapshot_y: list[torch.Tensor] = []
        swaps_before = 0 if env.shoe_randomizer is None else env.shoe_randomizer.swaps
        with torch.no_grad():
            for k in range(H):
                if encoder is not None:
                    # The estimator is queried at a fraction of the policy rate and its
                    # output is carried between queries; the policy sees eight more
                    # observation columns and nothing downstream knows where they came from.
                    if ((iteration - 1) * H + k) % adaptation.query_every == 0:
                        embedding = adaptive.encode(encoder, history.windows())
                        rows = adaptive.snapshot_rows(W, adaptation.auxiliary_worlds)
                        snapshot_x.append(history.snapshot(rows))
                        snapshot_y.append(env.shoe_randomizer.privileged()[rows])
                    policy_observation = torch.cat((observation, embedding), dim=-1)
                else:
                    policy_observation = observation
                model.actor.normalizer.update(policy_observation)
                model.critic_normalizer.update(torch.cat((policy_observation, privileged), dim=-1))
                dist = model.distribution(policy_observation)
                action = dist.sample()
                obs_buf[k], priv_buf[k], act_buf[k] = policy_observation, privileged, action
                logp_buf[k] = dist.log_prob(action).sum(-1)
                value_buf[k] = model.value(policy_observation, privileged)
                observation, privileged, reward, terminated, truncated, info = env.step(action)
                if encoder is not None:
                    # The action that produced this observation belongs to the encoder's
                    # input; the bootstrap reuses the current embedding, which is at most
                    # one control step old because that is how often it is refreshed.
                    history.push(torch.cat((observation, action), dim=-1))
                    bootstrap = model.value(
                        torch.cat((info["final_observation"], embedding), dim=-1), info["final_privileged"]
                    )
                else:
                    bootstrap = model.value(info["final_observation"], info["final_privileged"])
                reward_buf[k] = reward + ppo.gamma * bootstrap * truncated.float()
                step_reward_buf[k] = reward
                done = (terminated | truncated).float()
                done_buf[k] = done
                episode_return += reward
                episode_length += 1
                ended = done.bool()
                if history is not None and ended.any():
                    history.clear(ended)
                if ended.any():
                    finished_returns.append(episode_return[ended].mean().item())
                    finished_lengths.append(episode_length[ended].mean().item())
                    finished_failures.append(terminated[ended].float().mean().item())
                    episode_return[ended] = 0
                    episode_length[ended] = 0
                for name, value in info["terms"].items():
                    terms_sum[name] = terms_sum.get(name, 0.0) + value / H
                failed_sum += info["failed"] / H
            # The 3D environment keeps its logged terms on the device until here.
            terms_sum = {name: float(value) for name, value in terms_sum.items()}
            failed_sum = float(failed_sum)
            last_value = model.value(
                observation if encoder is None else torch.cat((observation, embedding), dim=-1), privileged
            )
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
                        learning_rate = min(ppo.learning_rate_cap, learning_rate * 1.5)
                    for group in optimizer.param_groups:
                        group["lr"] = learning_rate
                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["kl"] += kl
                updates += 1
        stats = {name: value / updates for name, value in stats.items()}
        # The estimator is trained by supervised regression on the histories that were
        # actually collected: the collection loop keeps no 50-step graph, and the policy
        # treats the embedding as an input. The weight eases off so the readout stops
        # pulling once the estimate is good enough to act on.
        aux_weight = readout_mean = readout_p90 = 0.0
        if encoder is not None and snapshot_x:
            from .adaptation import cosine_drift, readout_error  # noqa: PLC0415

            aux_weight = adaptation.auxiliary_weight * cosine_drift(
                iteration, ppo.iterations, adaptation.auxiliary_floor
            )
            window = torch.cat(snapshot_x)
            target = torch.cat(snapshot_y)
            if not adaptation.freeze_encoder:
                minibatch = max(1, window.shape[0] // 8)
                for _ in range(adaptation.auxiliary_epochs):
                    order = torch.randperm(window.shape[0], device=torch_device)
                    for begin in range(0, window.shape[0], minibatch):
                        index = order[begin : begin + minibatch]
                        _, estimate = encoder(window[index])
                        loss = torch.nn.functional.smooth_l1_loss(estimate, target[index])
                        encoder_optimizer.zero_grad()
                        (aux_weight * loss).backward()
                        torch.nn.utils.clip_grad_norm_(encoder.parameters(), ppo.max_grad_norm)
                        encoder_optimizer.step()
            with torch.no_grad():
                _, estimate = encoder(window)
                error = readout_error(estimate, target)
            readout_mean, readout_p90 = error["mean"], error["p90"]
            readout_log.append({"iteration": iteration, "samples": iteration * H * W, **error})
        elapsed = time.perf_counter() - start
        recent = slice(-50, None)
        row = {
            "iteration": iteration,
            "samples": iteration * H * W,
            # The target the GAE recursion is built on and the reward the agent is actually
            # earning are different columns: the target carries ``gamma * V`` on every step the
            # horizon truncated, which is of the order of an episode return, so its mean moves
            # by whole units when survival moves by a fraction of a per cent.
            "mean_step_reward": step_reward_buf.mean().item(),
            "bootstrap_reward": reward_buf.mean().item(),
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
            "shoe_swaps": 0 if env.shoe_randomizer is None else env.shoe_randomizer.swaps - swaps_before,
            "adaptation_weight": aux_weight,
            "theta_readout_mean": readout_mean,
            "theta_readout_p90": readout_p90,
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
            save_policy(model.actor, output / f"policy_{iteration:05d}.pt", metadata, model.log_std)
            if encoder is not None:
                # The student is a pair of networks: the frozen policy file is half of it.
                torch.save(
                    {
                        "encoder": encoder.state_dict(),
                        "config": asdict(adaptation),
                        "describe": encoder.describe(),
                        "observation_names": list(env.observation_names),
                        "channel_names": list(env.observation_names) + list(interface.ACTION_NAMES),
                        "readout": readout_log[-1] if readout_log else None,
                    },
                    output / f"encoder_{iteration:05d}.pt",
                )
            torch.save(
                {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": iteration},
                output / "checkpoint.pt",
            )
        # Track the best training-episode return; evaluation segments never select the policy.
        if row["episode_return"] > best and iteration > 20:
            best = row["episode_return"]
            save_policy(model.actor, output / "policy_best_train.pt", metadata, model.log_std)
    final = save_policy(model.actor, output / "policy.pt", metadata, model.log_std)
    if encoder is not None:
        if encoder is not None:
            # The student is a pair of networks: the frozen policy file is half of it.
            torch.save(
                {
                    "encoder": encoder.state_dict(),
                    "config": asdict(adaptation),
                    "describe": encoder.describe(),
                    "observation_names": list(env.observation_names),
                    "channel_names": list(env.observation_names) + list(interface.ACTION_NAMES),
                    "readout": readout_log[-1] if readout_log else None,
                },
                output / "encoder.pt",
            )
    (output / "readout.json").write_text(json.dumps(readout_log, indent=2) + "\n", encoding="utf-8")
    if board is not None:
        board.close()
    return final


def main(argv: list[str] | None = None) -> None:
    from .leg_contact import PROFILES as LEG_CONTACT_PROFILES  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=PPOConfig.worlds)
    parser.add_argument("--iterations", type=int, default=PPOConfig.iterations)
    parser.add_argument("--horizon", type=int, default=PPOConfig.horizon)
    parser.add_argument("--learning-rate", type=float, default=PPOConfig.learning_rate)
    parser.add_argument("--learning-rate-cap", type=float, default=PPOConfig.learning_rate_cap)
    parser.add_argument("--gamma", type=float, default=PPOConfig.gamma)
    parser.add_argument("--log-std", type=float, default=PPOConfig.log_std)
    parser.add_argument("--hidden", type=int, nargs="+", default=list(PPOConfig.hidden))
    parser.add_argument("--seed", type=int, default=PPOConfig.seed)
    parser.add_argument("--save-every", type=int, default=PPOConfig.save_every)
    parser.add_argument("--substeps", type=int, help="Substeps per control step; defaults to the environment's.")
    parser.add_argument("--dt-s", type=float, help="Substep [s]; defaults to the environment's.")
    parser.add_argument("--episode-s", type=float, help="Episode length [s]; defaults to the environment's.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--kind", choices=("planar", "3d"), default="planar")
    parser.add_argument("--resume", type=Path, help="Initialize from a run's checkpoint.pt (fine-tuning).")
    parser.add_argument("--solver", choices=("featherstone", "mujoco"), help="3D only: Newton solver.")
    parser.add_argument(
        "--leg-contact",
        choices=("off", *LEG_CONTACT_PROFILES),
        default="off",
        help="3D only: inter-leg capsule profile; ``probe`` records penetration without any force.",
    )
    parser.add_argument(
        "--leg-contact-penalty",
        type=float,
        help="3D only: reward weight on inter-leg penetration; needs a --leg-contact profile to measure it.",
    )
    parser.add_argument(
        "--reward-weights",
        type=float,
        nargs=6,
        help="3D only: weights of the pose, velocity, position, force, effort, and smoothness terms.",
    )
    parser.add_argument(
        "--shoe-dr",
        default="off",
        help="3D only: per-world shoe randomization preset (off, registration, band, wide).",
    )
    parser.add_argument(
        "--shoe-swap-p",
        type=float,
        default=None,
        help="3D only: probability per control step that a running world gets a new shoe.",
    )
    parser.add_argument("--shoe-dr-seed", type=int, default=0, help="Seed of the shoe draw, separate from --seed.")
    parser.add_argument(
        "--contact-obs",
        action="store_true",
        help="3D only: add the per-foot foam channels to the actor inputs. The estimator needs them.",
    )
    parser.add_argument(
        "--theta-obs",
        action="store_true",
        help="3D only: add the shoe parameters to the actor inputs: the privileged teacher of the design.",
    )
    parser.add_argument(
        "--pose-group-weights",
        type=float,
        nargs=3,
        help="3D only: relative hip, knee, and ankle weights of the pose term.",
    )
    parser.add_argument(
        "--pose-dof-weights",
        type=float,
        nargs=10,
        metavar="W",
        help="3D only: per-joint multiplier on the pose term, in joint order; multiplies --pose-group-weights.",
    )
    parser.add_argument(
        "--stance-pose-relief",
        type=float,
        default=0.0,
        help="3D only: fraction by which the knee and ankle pose weight relaxes in stance.",
    )
    parser.add_argument("--cadence-weight", type=float, default=0.0, help="3D only: weight of the cadence-band term.")
    parser.add_argument(
        "--cadence-band-spm", type=float, default=6.0, help="Half-width of the cadence band [steps/min]."
    )
    parser.add_argument(
        "--actuation",
        default=None,
        help="3D only: smooth the impedance setpoints below the control rate; a profile of actuation.PROFILES"
        " or a JSON object. Unset keeps the zero-order hold.",
    )
    parser.add_argument(
        "--estimator",
        action="store_true",
        help="3D only: train the RMA student, a history-conditioned shoe estimator feeding the policy.",
    )
    parser.add_argument("--teacher", type=Path, help="3D only: frozen teacher policy .pt to grow the student from.")
    parser.add_argument("--history-steps", type=int, default=AdaptationConfig.history_steps)
    parser.add_argument("--query-every", type=int, default=AdaptationConfig.query_every)
    parser.add_argument("--embedding", type=int, default=AdaptationConfig.embedding)
    parser.add_argument("--aux-weight", type=float, default=AdaptationConfig.auxiliary_weight)
    parser.add_argument("--aux-floor", type=float, default=AdaptationConfig.auxiliary_floor)
    parser.add_argument(
        "--freeze-estimator",
        action="store_true",
        help="Train only the policy against a frozen estimator: A-RMA's third phase.",
    )
    parser.add_argument("--reward-kernels", choices=("hybrid", "per_term", "pooled"), default="hybrid")
    parser.add_argument("--force-penalty", type=float, default=EnvConfig.force_penalty)
    parser.add_argument(
        "--response-time-s", type=float, default=EnvConfig.response_time_s, help="Torque lag [s]; 0 disables it."
    )
    parser.add_argument(
        "--slew-limit",
        action=argparse.BooleanOptionalAction,
        default=EnvConfig.torque_slew_limit,
        help="Bound the joint torque rate; --no-slew-limit, the default, removes the bound.",
    )
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
        learning_rate_cap=args.learning_rate_cap,
        gamma=args.gamma,
        log_std=args.log_std,
        hidden=tuple(args.hidden),
        seed=args.seed,
        save_every=args.save_every,
    )
    overrides = {"dt_s": args.dt_s, "substeps": args.substeps, "episode_s": args.episode_s}
    common = {name: value for name, value in overrides.items() if value is not None}
    if args.kind == "3d":
        from .env3d import Env3DConfig  # noqa: PLC0415

        if args.reward_weights is not None:
            common["reward_weights"] = tuple(args.reward_weights)
        if args.solver is not None:
            common["solver"] = args.solver
        if args.pose_group_weights is not None:
            common["pose_group_weights"] = tuple(args.pose_group_weights)
        if args.pose_dof_weights is not None:
            common["pose_dof_weights"] = tuple(args.pose_dof_weights)
        if args.leg_contact_penalty is not None:
            common["leg_contact_penalty"] = args.leg_contact_penalty
        if args.shoe_swap_p is not None:
            common["shoe_swap_probability"] = args.shoe_swap_p
        common.update(
            leg_contact=args.leg_contact,
            shoe_dr=args.shoe_dr,
            shoe_dr_seed=args.shoe_dr_seed,
            contact_observation=args.contact_obs,
            theta_observation=args.theta_obs,
            stance_pose_relief=args.stance_pose_relief,
            cadence_weight=args.cadence_weight,
            cadence_band_spm=args.cadence_band_spm,
            actuation=args.actuation,
        )
        env_config = Env3DConfig(
            **common,
            response_time_s=args.response_time_s,
            torque_slew_limit=args.slew_limit,
        )
    else:
        env_config = EnvConfig(
            **common,
            reward_kernels=args.reward_kernels,
            force_penalty=args.force_penalty,
            response_time_s=args.response_time_s,
            torque_slew_limit=args.slew_limit,
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
            teacher=args.teacher,
            adaptation=AdaptationConfig(
                history_steps=args.history_steps,
                query_every=args.query_every,
                embedding=args.embedding,
                auxiliary_weight=args.aux_weight,
                auxiliary_floor=args.aux_floor,
                freeze_encoder=args.freeze_estimator,
            )
            if args.estimator
            else None,
            tensorboard=args.tensorboard,
        )
    )


if __name__ == "__main__":
    main()
