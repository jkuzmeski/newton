# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Two-stiffness PPO with frozen inverse-dynamics equilibria and motion-only loss.

The rig owns the reward: the negative time integral of normalized pelvis-height
and foot-pitch squared errors. PPO uses complete episodes, gamma=1 and Monte
Carlo returns. There is no force, work, momentum, duration, or entropy reward.
Physical validity gates checkpoint selection, not the tracking objective.

Torch is optional and imported only for learning or checkpoint inference. The
Gaussian actor and clipped surrogate use the standard PPO math also used in
``projects/impedance_instron/train.py``; no old experiment code is imported.
"""

from __future__ import annotations

import csv
import hashlib
import importlib
import json
import warnings
from dataclasses import asdict, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .reference import Reference
    from .rig import RigConfig

CHECKPOINT_FORMAT = "impedance_simple_policy_1"
ACTION_NAMES = ("log_leg_stiffness", "log_ankle_stiffness")
PPO_SETTINGS = {
    "gamma": 1.0,
    "advantage_lambda": 1.0,
    "learning_rate": 3.0e-4,
    "clip": 0.2,
    "epochs": 4,
    "minibatch_size": 512,
    "value_coefficient": 0.5,
    "max_grad_norm": 0.5,
    "hidden_size": 64,
    "entropy_coefficient": 0.0,
}


def _torch():
    """Load the existing optional Torch dependency only when needed."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Simple policy training/evaluation requires the optional Torch extra.") from exc
    return torch


def _rig_types():
    """Import physics lazily so checkpoint helpers do not initialize Warp."""
    module = importlib.import_module(".rig", __package__)
    return module.Rig, module.RigConfig


def _reference_type():
    """Import the offline reference without any legacy learner dependencies."""
    return importlib.import_module(".reference", __package__).Reference


def _json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def artifact_identity(path: str | Path) -> dict:
    """Record the source hash and fixed geometry for material-only replay.

    Metadata may change with a material export. All other non-material fields
    remain fixed, including unknown future fields, so overrides fail closed.
    """
    path = Path(path).resolve()
    raw = path.read_bytes()
    data = json.loads(raw)
    geometry = {
        key: value
        for key, value in data.items()
        if key not in {"shoe", "constitutive_model", "validation", "provenance"}
    }
    if "column_bed" not in geometry or "coordinate_system" not in geometry:
        raise ValueError("Shoe artifact must contain column_bed and coordinate_system")
    material = data["constitutive_model"]
    return {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "geometry_sha256": _json_hash(geometry),
        "geometry": geometry,
        "material_sha256": _json_hash(material),
        "material": material,
    }


def create_policy(observation_dim: int, *, hidden_size: int = 64, device: str = "cpu"):
    """Build an actor with exactly two bounded stiffness outputs and a critic.

    Args:
        observation_dim: Number of already scaled rig observations.
        hidden_size: Width of each of the two hidden layers.
        device: Torch device, independent of the physics device.
    """
    torch = _torch()
    if observation_dim < 1 or hidden_size < 1:
        raise ValueError("Policy dimensions must be positive")

    class Policy(torch.nn.Module):
        """Keep the critic separate so value fitting cannot change actor features."""

        action_dim = 2

        def __init__(self):
            super().__init__()
            self.observation_dim = observation_dim
            self.hidden_size = hidden_size

            def network(outputs):
                return torch.nn.Sequential(
                    torch.nn.Linear(observation_dim, hidden_size),
                    torch.nn.Tanh(),
                    torch.nn.Linear(hidden_size, hidden_size),
                    torch.nn.Tanh(),
                    torch.nn.Linear(hidden_size, outputs),
                )

            self.actor = network(2)
            self.critic = network(1)
            torch.nn.init.zeros_(self.actor[-1].weight)
            torch.nn.init.zeros_(self.actor[-1].bias)
            self.log_std = torch.nn.Parameter(torch.full((2,), -1.0))

        def forward(self, obs):
            """Return deterministic bounded log-stiffness coordinates."""
            return torch.tanh(self.actor(obs))

        def act(self, observations, *, deterministic: bool = True) -> np.ndarray:
            """Return bounded actions from already scaled NumPy rig observations."""
            with torch.no_grad():
                obs = torch.as_tensor(observations, dtype=torch.float32, device=self.log_std.device)
                actions = self(obs) if deterministic else torch.tanh(self.distribution(obs).sample())
                return actions.cpu().numpy()

        def distribution(self, obs):
            """Return the Gaussian before the invertible tanh action transform."""
            return torch.distributions.Normal(self.actor(obs), self.log_std.clamp(-5.0, 1.0).exp())

        def value(self, obs):
            """Return the undiscounted remaining negative tracking loss."""
            return self.critic(obs).squeeze(-1)

    return Policy().to(device)


def _observation_contract(rig) -> dict:
    names = list(rig.observation_names)
    scale = np.asarray(rig.observation_scale, dtype=np.float64)
    if len(names) != rig.observation_dim or scale.shape != (rig.observation_dim,):
        raise ValueError("Rig observation layout/scale does not match observation_dim")
    if len(set(names)) != len(names) or not np.all(np.isfinite(scale) & (scale > 0.0)):
        raise ValueError("Rig observation names must be unique and scales finite/positive")
    return {"names": names, "scale": scale.tolist(), "already_scaled": True}


def _safe_observations(obs, rig) -> tuple[np.ndarray, np.ndarray]:
    obs = np.asarray(obs, dtype=np.float32)
    if obs.shape != (rig.num_worlds, rig.observation_dim):
        raise ValueError("Rig returned an unexpected observation shape")
    finite = np.isfinite(obs).all(axis=1)
    # Invalid worlds stay invalid and are excluded from PPO, never rewarded.
    return np.where(np.isfinite(obs), obs, 0.0), finite


def _summarize(rig, cumulative_reward: np.ndarray, finite: np.ndarray, info: dict) -> dict:
    loss = -np.asarray(cumulative_reward, dtype=np.float64)
    reported = np.asarray(info["tracking_loss"], dtype=np.float64)
    safety = np.asarray(info["safety_ok"], dtype=bool)
    if reported.shape != loss.shape or safety.shape != loss.shape:
        raise ValueError("Rig tracking_loss and safety_ok must have one entry per world")
    jointly_finite = np.isfinite(loss) & np.isfinite(reported)
    if not np.allclose(loss[jointly_finite], reported[jointly_finite], rtol=1.0e-5, atol=1.0e-6):
        raise ValueError("Rig tracking_loss differs from negative cumulative reward")
    finite = finite & jointly_finite
    valid = safety & finite
    reasons = info.get("safety_reasons", [[] for _ in range(rig.num_worlds)])
    reasons = [[str(reason) for reason in item] if not isinstance(item, str) else [item] for item in reasons]
    if len(reasons) != rig.num_worlds:
        raise ValueError("Rig safety_reasons must have one entry per world")
    for world in range(rig.num_worlds):
        if not finite[world]:
            reasons[world].append("nonfinite rollout")
    return {
        "tracking_loss": [float(value) if ok else None for value, ok in zip(loss, finite, strict=True)],
        "tracking_loss_mean": float(loss.mean()) if finite.all() else None,
        "cumulative_reward": [
            float(value) if ok else None for value, ok in zip(cumulative_reward, finite, strict=True)
        ],
        "safety_ok": valid.tolist(),
        "numerical_ok": finite.tolist(),
        "safety_reasons": reasons,
        "valid_world_count": int(valid.sum()),
        "num_worlds": int(rig.num_worlds),
        "eligible_for_best": bool(valid.all()),
        "status": "physically_valid" if valid.all() else "physically_invalid",
        "objective": "negative cumulative pelvis-height/foot-pitch tracking reward",
        "target_achievement": "not assessed; physical validity is not target perfection",
    }


def _rollout(rig, policy=None, *, stochastic: bool = False):
    numpy_policy = policy is not None and not hasattr(policy, "parameters")
    torch = _torch() if policy is not None and not numpy_policy else None
    device = next(policy.parameters()).device if torch is not None else None
    obs, finite = _safe_observations(rig.reset(), rig)
    total = np.zeros(rig.num_worlds, dtype=np.float64)
    buffer = {key: [] for key in ("obs", "latent", "log_prob", "value", "reward")}
    done = False
    info = {}
    frames = 0
    for _ in range(rig.episode_frames):
        frames += 1
        if policy is None:
            action = np.zeros((rig.num_worlds, 2), dtype=np.float32)
        elif numpy_policy:
            action = np.asarray(policy(obs), dtype=np.float32)
        else:
            with torch.no_grad():
                tensor = torch.as_tensor(obs, device=device)
                if stochastic:
                    distribution = policy.distribution(tensor)
                    latent = distribution.sample()
                    # Tanh Jacobians cancel in the PPO ratio for stored latents.
                    log_prob = distribution.log_prob(latent).sum(-1)
                    value = policy.value(tensor)
                    action = torch.tanh(latent).cpu().numpy()
                    buffer["obs"].append(obs.copy())
                    buffer["latent"].append(latent.cpu().numpy())
                    buffer["log_prob"].append(log_prob.cpu().numpy())
                    buffer["value"].append(value.cpu().numpy())
                else:
                    action = policy(tensor).cpu().numpy()
        if action.shape != (rig.num_worlds, 2) or not np.isfinite(action).all():
            raise ValueError("Policy must return finite [worlds, 2] actions")
        obs, reward, done, info = rig.step(action)
        obs, next_finite = _safe_observations(obs, rig)
        reward = np.asarray(reward, dtype=np.float64)
        if reward.shape != (rig.num_worlds,):
            raise ValueError("Rig reward must have one entry per world")
        finite &= next_finite & np.isfinite(reward)
        total += reward
        if stochastic:
            buffer["reward"].append(np.where(np.isfinite(reward), reward, 0.0))
        if done:
            break
    if not done:
        raise ValueError("Rig did not finish after episode_frames")
    summary = _summarize(rig, total, finite, info)
    if frames != rig.episode_frames:
        summary.update(
            eligible_for_best=False,
            safety_ok=[False] * rig.num_worlds,
            numerical_ok=[False] * rig.num_worlds,
            valid_world_count=0,
            status="physically_invalid",
        )
        for reasons in summary["safety_reasons"]:
            reasons.append("incomplete episode")
    return {key: np.asarray(value) for key, value in buffer.items()}, summary


def evaluate_policy(rig, policy=None) -> dict:
    """Evaluate deterministically using the exact sum of the training rewards.

    Args:
        rig: Fresh or reusable two-action rig; this function resets it.
        policy: Frozen Torch actor, NumPy callable, or None for a zero-action
            stiffness baseline. Baselines and NumPy callables do not need Torch.

    Returns:
        Motion loss and separate validity report. The rig retains its full trace.
    """
    if rig.action_dim != 2:
        raise ValueError("Simple policies require exactly two stiffness actions")
    if policy is not None and hasattr(policy, "eval"):
        policy.eval()
    return _rollout(rig, policy)[1]


def is_better_evaluation(candidate: dict, best_loss: float | None) -> bool:
    """Rank only complete, physically valid deterministic motion evaluations."""
    loss = candidate.get("tracking_loss_mean")
    return bool(
        candidate.get("eligible_for_best", False)
        and bool(candidate.get("safety_ok"))
        and all(candidate["safety_ok"])
        and loss is not None
        and np.isfinite(loss)
        and (best_loss is None or loss < best_loss)
    )


def _update(policy, optimizer, buffer: dict, summary: dict, rng) -> dict:
    torch = _torch()
    valid = np.asarray(summary["numerical_ok"], dtype=bool)
    if not valid.any():
        return {"update_performed": False, "training_finite_worlds": 0, "physically_valid_worlds": 0}
    rewards = buffer["reward"][:, valid]
    # Full-episode Monte Carlo returns: gamma=lambda=1, no terminal bootstrap.
    returns = np.flip(np.cumsum(np.flip(rewards, axis=0), axis=0), axis=0).copy()
    advantages = returns - buffer["value"][:, valid]
    advantages = (advantages - advantages.mean()) / max(float(advantages.std()), 1.0e-8)
    device = next(policy.parameters()).device
    tensors = {
        name: torch.as_tensor(buffer[name][:, valid].reshape((-1, *buffer[name].shape[2:])), device=device)
        for name in ("obs", "latent", "log_prob")
    }
    tensors["returns"] = torch.as_tensor(returns.reshape(-1), dtype=torch.float32, device=device)
    tensors["advantages"] = torch.as_tensor(advantages.reshape(-1), dtype=torch.float32, device=device)
    count = returns.size
    policy.train()
    last_loss = None
    for _ in range(PPO_SETTINGS["epochs"]):
        indices = rng.permutation(count)
        for start in range(0, count, PPO_SETTINGS["minibatch_size"]):
            ids = torch.as_tensor(indices[start : start + PPO_SETTINGS["minibatch_size"]], device=device)
            obs = tensors["obs"][ids]
            log_prob = policy.distribution(obs).log_prob(tensors["latent"][ids]).sum(-1)
            ratio = torch.exp(log_prob - tensors["log_prob"][ids])
            advantage = tensors["advantages"][ids]
            clipped = ratio.clamp(1.0 - PPO_SETTINGS["clip"], 1.0 + PPO_SETTINGS["clip"]) * advantage
            actor_loss = -torch.minimum(ratio * advantage, clipped).mean()
            value_loss = 0.5 * (policy.value(obs) - tensors["returns"][ids]).square().mean()
            loss = actor_loss + PPO_SETTINGS["value_coefficient"] * value_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite PPO loss; no checkpoint will be selected")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), PPO_SETTINGS["max_grad_norm"], error_if_nonfinite=True)
            optimizer.step()
            last_loss = float(loss.detach().cpu())
    return {
        "update_performed": True,
        "training_finite_worlds": int(valid.sum()),
        "physically_valid_worlds": summary["valid_world_count"],
        "ppo_loss": last_loss,
    }


def _config_dict(config) -> dict:
    return config.to_dict() if hasattr(config, "to_dict") else asdict(config)


def _checkpoint(rig, policy, artifact: dict, iteration: int, summary: dict, seed: int) -> dict:
    reference = rig.reference.to_dict()
    config = _config_dict(rig.config)
    return {
        "format": CHECKPOINT_FORMAT,
        "action_dim": 2,
        "action_names": list(ACTION_NAMES),
        "action_transform": "tanh; absolute bounded log stiffness; rig rate limit",
        "reference": reference,
        "reference_sha256": _json_hash(reference),
        "reference_identity": rig.reference.identity,
        "rig_config": config,
        "rig_config_sha256": _json_hash(config),
        "artifact": artifact,
        "input_fingerprints": dict(rig.input_fingerprints),
        "observation": _observation_contract(rig),
        "network": {"observation_dim": policy.observation_dim, "hidden_size": policy.hidden_size},
        "policy_state": {key: value.detach().cpu().clone() for key, value in policy.state_dict().items()},
        "ppo": dict(PPO_SETTINGS),
        "seed": int(seed),
        "iteration": int(iteration),
        "evaluation": summary,
    }


def _save_checkpoint(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    _torch().save(payload, temporary)
    temporary.replace(path)


def _save_traces(rig, directory: Path, iteration: int) -> list[str]:
    paths = []
    for world in range(rig.num_worlds):
        path = directory / f"eval_{iteration:05d}_world_{world:03d}.npz"
        trace = {name: np.asarray(value) for name, value in rig.trace(world).items()}
        if any(value.dtype.hasobject for value in trace.values()):
            raise ValueError("Evaluation traces must contain numeric arrays, not Python objects")
        np.savez_compressed(path, **trace)
        paths.append(str(path))
    return paths


def _write_history(output: Path, history: list[dict]) -> None:
    (output / "history.json").write_text(json.dumps(history, indent=2, allow_nan=False) + "\n")
    columns = ("iteration", "tracking_loss_mean", "eligible_for_best", "valid_world_count", "is_best")
    with (output / "history.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in columns} for row in history)


def train(
    reference: Reference,
    artifact_path,
    output: Path,
    *,
    config: RigConfig | None = None,
    iterations: int = 100,
    num_worlds: int = 32,
    seed: int = 0,
    device: str | None = None,
    eval_interval: int = 10,
) -> dict:
    """Train two stiffness controls and save the best valid deterministic actor.

    Args:
        reference: Fully frozen optical targets and inverse-dynamics schedules.
        artifact_path: Self-contained shoe geometry/material artifact.
        output: New or empty result directory; existing runs are never replaced.
        config: Complete rig settings, frozen with the checkpoint.
        iterations: PPO updates. Zero performs an initial evaluation only.
        num_worlds: Parallel stochastic rollout worlds.
        seed: NumPy and Torch random seed.
        device: Physics device. The small Torch network runs on CPU.
        eval_interval: Updates between deterministic one-world evaluations.

    Returns:
        Best/last checkpoint paths, deterministic evaluation history, and status.
        If no evaluation is physically valid, best_checkpoint is None.
    """
    if iterations < 0 or num_worlds < 1 or eval_interval < 1:
        raise ValueError("Require iterations >= 0, num_worlds >= 1, eval_interval >= 1")
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Training output must be empty to avoid replacing a previous best policy")
    artifact = artifact_identity(artifact_path)
    Rig, _ = _rig_types()
    rig = Rig(reference, artifact_path, config=config, num_worlds=num_worlds, device=device)
    evaluation_rig = Rig(reference, artifact_path, config=rig.config, num_worlds=1, device=device)
    if rig.action_dim != 2 or evaluation_rig.action_dim != 2:
        raise ValueError("Simple training requires exactly two stiffness controls")
    _observation_contract(rig)
    torch = _torch()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    policy = create_policy(rig.observation_dim, hidden_size=PPO_SETTINGS["hidden_size"])
    optimizer = torch.optim.Adam(policy.parameters(), lr=PPO_SETTINGS["learning_rate"])
    output.mkdir(parents=True, exist_ok=True)
    trace_directory = output / "traces"
    trace_directory.mkdir()
    history = []
    updates = []
    best_loss = None
    best_iteration = None
    last_path = output / "last.pt"
    best_path = output / "best.pt"
    for iteration in range(iterations + 1):
        if iteration:
            buffer, rollout_summary = _rollout(rig, policy, stochastic=True)
            update = _update(policy, optimizer, buffer, rollout_summary, rng)
            updates.append({"iteration": iteration, **update, "rollout": rollout_summary})
            (output / "updates.json").write_text(json.dumps(updates, indent=2, allow_nan=False) + "\n")
        if iteration % eval_interval and iteration != iterations:
            continue
        summary = evaluate_policy(evaluation_rig, policy)
        better = is_better_evaluation(summary, best_loss)
        traces = _save_traces(evaluation_rig, trace_directory, iteration)
        record = {"iteration": iteration, **summary, "is_best": better, "traces": traces}
        history.append(record)
        payload = _checkpoint(evaluation_rig, policy, artifact, iteration, summary, seed)
        _save_checkpoint(last_path, payload)
        if better:
            best_loss = summary["tracking_loss_mean"]
            best_iteration = iteration
            _save_checkpoint(best_path, payload)
        _write_history(output, history)
    result = {
        "best_checkpoint": str(best_path) if best_iteration is not None else None,
        "best_iteration": best_iteration,
        "best_tracking_loss": best_loss,
        "last_checkpoint": str(last_path),
        "history": history,
        "reference_identity": reference.identity,
        "artifact_sha256": artifact["sha256"],
        "status": "valid_policy_selected" if best_iteration is not None else "no_physically_valid_policy",
        "target_achievement": "not assessed; best means lowest valid measured-motion loss, not perfection",
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def restore(
    checkpoint: Path,
    *,
    artifact_path=None,
    num_worlds: int = 1,
    device: str | None = None,
    allow_physics_update: bool = False,
) -> tuple[Any, Any]:
    """Restore a live rig and frozen actor without advancing the simulation.

    Args:
        checkpoint: Simple two-output checkpoint, never an old residual policy.
        artifact_path: Explicit material-only override with identical geometry.
            If omitted, the original path must still match its full source hash.
        num_worlds: Identical deterministic replay worlds.
        device: Physics device; this does not override mechanical settings.
        allow_physics_update: Explicitly re-evaluate unchanged weights after runtime
            source changes. Geometry, material, reference and settings checks remain
            enforced. Prior scores do not validate the new physics.

    Returns:
        Live rig and Torch policy. Use ``policy.act(obs, deterministic=True)``
        with NumPy rig observations for viewer stepping. No normalization updates
        or parameter optimization occur during restoration.
    """
    if num_worlds < 1:
        raise ValueError("num_worlds must be positive")
    torch = _torch()
    saved = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    if saved.get("format") != CHECKPOINT_FORMAT or saved.get("action_dim") != 2:
        raise ValueError("Expected a simple two-stiffness checkpoint; old six-action policies are unsupported")
    if saved.get("action_names") != list(ACTION_NAMES):
        raise ValueError("Checkpoint has an incompatible stiffness action contract")
    for key in ("reference", "rig_config"):
        if _json_hash(saved[key]) != saved[f"{key}_sha256"]:
            raise ValueError(f"Checkpoint {key} content hash mismatch")
    reference = _reference_type().from_dict(saved["reference"])
    if reference.identity != saved["reference_identity"]:
        raise ValueError("Checkpoint reference identity mismatch")
    Rig, RigConfig = _rig_types()
    if set(saved["rig_config"]) != {field.name for field in fields(RigConfig)}:
        raise ValueError("Checkpoint must specify ALL current rig settings; implicit new defaults are unsupported")
    config = (
        RigConfig.from_dict(saved["rig_config"])
        if hasattr(RigConfig, "from_dict")
        else RigConfig(**saved["rig_config"])
    )
    source = saved["artifact"]
    if _json_hash(source["geometry"]) != source["geometry_sha256"]:
        raise ValueError("Checkpoint geometry content hash mismatch")
    path = Path(source["path"] if artifact_path is None else artifact_path).resolve()
    current = artifact_identity(path)
    if artifact_path is None and current["sha256"] != source["sha256"]:
        raise ValueError("Original artifact changed; pass an explicit material-only override to evaluate it")
    if current["geometry_sha256"] != source["geometry_sha256"]:
        raise ValueError("Material-only override cannot change shoe geometry or its coordinate frame")
    rig = Rig(reference, path, config=config, num_worlds=num_worlds, device=device)
    variable_keys = {"artifact_sha256", "material_identity"} if artifact_path is not None else set()
    current_fixed = {key: value for key, value in rig.input_fingerprints.items() if key not in variable_keys}
    saved_fixed = {key: value for key, value in saved["input_fingerprints"].items() if key not in variable_keys}
    source_keys = {"rig_source_sha256", "foundation_source_sha256"}
    changed = {
        key: {"checkpoint": saved_fixed.get(key), "current": current_fixed.get(key)}
        for key in sorted(current_fixed.keys() | saved_fixed.keys())
        if current_fixed.get(key) != saved_fixed.get(key)
    }
    if changed and (not allow_physics_update or not changed.keys() <= source_keys):
        raise ValueError(
            "Restored rig inputs or runtime source differ from the frozen checkpoint. "
            "For source-only changes, explicitly use allow_physics_update=True "
            "(viewer: --allow-physics-update) to re-evaluate the old weights."
        )
    if changed:
        warnings.warn(
            "Re-evaluating frozen policy weights with changed physics source. "
            "The checkpoint's old scores and physical validity do not apply to this run.",
            stacklevel=2,
        )
    if rig.action_dim != 2 or _observation_contract(rig) != saved["observation"]:
        raise ValueError("Rig action/observation contract differs from the frozen checkpoint")
    policy = create_policy(**saved["network"])
    policy.load_state_dict(saved["policy_state"], strict=True)
    policy.eval()
    policy.requires_grad_(False)
    policy.checkpoint_metadata = {
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_iteration": saved["iteration"],
        "reference_identity": reference.identity,
        "artifact_sha256": current["sha256"],
        "training_artifact_sha256": source["sha256"],
        "material_override": artifact_path is not None,
        "physics_updated": bool(changed),
        "physics_source_changes": changed,
        "checkpoint_scores_applicable": not bool(changed) and artifact_path is None,
        "physics": getattr(rig, "metadata", {}),
    }
    return rig, policy


def evaluate(
    checkpoint: Path,
    *,
    artifact_path=None,
    num_worlds: int = 1,
    device: str | None = None,
    allow_physics_update: bool = False,
) -> tuple[Any, dict]:
    """Restore a checkpoint and evaluate its deterministic motion loss.

    Args:
        checkpoint: Simple two-stiffness checkpoint.
        artifact_path: Explicit geometry-checked material-only override, or None
            to restore the original artifact with strict source-hash checking.
        num_worlds: Identical deterministic replay worlds.
        device: Physics device, not a mechanical setting override.
        allow_physics_update: Re-evaluate unchanged weights after source changes,
            without accepting any other frozen input changes.

    Returns:
        Evaluated rig with full traces and a separate motion/validity report.
    """
    rig, policy = restore(
        checkpoint,
        artifact_path=artifact_path,
        num_worlds=num_worlds,
        device=device,
        allow_physics_update=allow_physics_update,
    )
    summary = evaluate_policy(rig, policy)
    summary.update(policy.checkpoint_metadata)
    return rig, summary
