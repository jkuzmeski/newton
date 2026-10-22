# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Stage-0 go/no-go of the world model (milestone M3).

``fit`` trains a :class:`~projects.supermimic.world_model.WorldModel` on a
stage-0 buffer written by :mod:`projects.supermimic.collect`. ``evaluate`` rolls
trained models out on the logged actions of one or more buffers and reports the
vertical ground reaction force error in the 10 ms control steps after
touchdown, against the stride-to-stride spread of F01 in the same binning.

Binning: a control step's force is its mean over the step, as the sim logs it.
Touchdown is the first step whose mean vertical force exceeds
:data:`~projects.supermimic.reference.CONTACT_THRESHOLD_N`; bin ``b`` is the
``b``-th step after it. The bar is the stride-to-stride standard deviation of
the subject's 10 ms plate means from the touchdown sample: the model is scored
against the sim at the same touchdown phase, so the phase within a step is no
error source. The spread of plate means on a fixed step grid, with touchdown at
all phases, is reported next to it. A
horizon ``H`` scores the force of a step predicted from a logged state ``H - 1``
steps earlier.

Each held-out buffer gets its own verdict: *go* when the 8-step error of bins
1 to 4 is at most the touchdown-aligned standard deviation of that bin. Bin 0 is
reported apart as its error-to-SD ratio: its sim step starts before touchdown
and the subject's at touchdown, so its narrow spread does not decide alone.

From the repository root::

    uv run --extra dev --extra torch-cu12 python -m projects.supermimic.m3 fit \
        --train outputs/supermimic/stage0_train/buffer.npz --out outputs/supermimic/m3/h4
    uv run --extra dev --extra torch-cu12 python -m projects.supermimic.m3 evaluate \
        --model outputs/supermimic/m3/h4 outputs/supermimic/m3/h0 \
        --buffer train=outputs/supermimic/stage0_train/buffer.npz \
        --buffer held_out_windows=outputs/supermimic/stage0_eval/buffer.npz --held-out held_out_windows \
        --out outputs/supermimic/m3/report.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from projects.impedance_instron.cartesian.visual3d import load_visual3d_export

from .buffer import ReplayBuffer
from .reference import CONTACT_THRESHOLD_N, F01, REPOSITORY, Subject, load_subject
from .world_model import WorldModel

__all__ = ["evaluate", "fit", "load_model", "steps_since_touchdown", "touchdown_force"]

STEP_S = 0.01
"""Control step [s]."""
BINS = 5
"""Steps after touchdown that are scored (the first 50 ms)."""
HORIZONS = (1, 8, 32)
"""Prediction horizons in control steps."""
GO_HORIZON = 8
"""Horizon of the go/no-go verdict."""
_FORCE_Z = (2, 5)
"""Vertical force of the right and left foot in an aux row of :mod:`projects.supermimic.collect`."""


def steps_since_touchdown(force: torch.Tensor, reset: torch.Tensor) -> torch.Tensor:
    """Count control steps since touchdown.

    Args:
        force: Step-mean vertical force [N], shape ``[..., steps]``.
        reset: Episode starts, same shape; contact already present at an episode
            start has no touchdown.

    Returns:
        Steps since touchdown, ``-1`` out of contact or in a contact without a
        touchdown, same shape as ``force``.
    """
    contact = force > CONTACT_THRESHOLD_N
    since = torch.full(force.shape, -1, dtype=torch.int64, device=force.device)
    for t in range(1, force.shape[-1]):
        fresh = ~reset[..., t]
        touchdown = contact[..., t] & ~contact[..., t - 1] & fresh
        stance = contact[..., t] & (since[..., t - 1] >= 0) & fresh
        since[..., t] = torch.where(touchdown, 0, torch.where(stance, since[..., t - 1] + 1, -1))
    return since


def touchdown_force(subject: Subject) -> tuple[np.ndarray, np.ndarray]:
    """Return a subject's step-mean vertical plate force in the :data:`BINS` steps after each touchdown.

    Returns:
        Forces [N], shape ``[touchdowns, BINS]``, over the subject's trials,
        with steps starting at the sample of touchdown (the first sample above
        the threshold after a step without contact), and with steps on a fixed
        grid at every sample offset, where touchdown falls at all phases within
        its step.
    """
    aligned, phases = [], []
    for trial in subject.trials:
        export = load_visual3d_export(REPOSITORY / subject.data / trial)
        per_step = round(STEP_S * export.analog_rate_hz)
        force = export.force_n[:, 2]
        contact = force > CONTACT_THRESHOLD_N
        flight = np.convolve(contact, np.ones(per_step), "valid") == 0
        for i in np.nonzero(contact[per_step:] & flight[:-1])[0] + per_step:
            if i + BINS * per_step <= len(force):
                aligned.append(force[i : i + BINS * per_step].reshape(BINS, per_step).mean(1))
        steps = len(force) // per_step - 1
        mean = torch.as_tensor(
            np.stack([force[o : o + steps * per_step].reshape(steps, per_step).mean(1) for o in range(per_step)])
        )
        since = steps_since_touchdown(mean, torch.zeros(mean.shape, dtype=torch.bool))
        for offset, t in torch.nonzero(since[:, : steps - BINS + 1] == 0).tolist():
            phases.append(mean[offset, t : t + BINS].numpy())
    return np.stack(aligned), np.stack(phases)


def load_model(folder: Path, device: str = "cuda") -> WorldModel:
    """Load a model written by :func:`fit`."""
    saved = torch.load(folder / "world_model.pt", map_location=device)
    model = WorldModel(**saved["config"]).to(device)
    model.load_state_dict(saved["state_dict"])
    return model.eval()


def _window(batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
    return batch["state"], batch["action"][:, :-1], batch["xi"][:, 0], batch["aux"][:, :-1]


def fit(
    model: WorldModel,
    buffer: ReplayBuffer,
    iterations: int,
    *,
    steps: int = 8,
    batch_size: int = 1024,
    learning_rate: float = 3.0e-4,
    seed: int = 0,
) -> list[dict]:
    """Fit ``model`` on ``steps``-step windows of ``buffer`` with the k-step loss.

    One iteration (sampled windows, loss, backward, clipping and Adam) replays a
    captured CUDA graph, since launch overhead dominates eager steps. The
    learning rate decays along a cosine to 3 % over the run, and the gradient
    norm is clipped to 1 (SuperMimic, Table A.5).

    Returns:
        The training loss every 1000 iterations and at the end.
    """
    generator = torch.Generator(buffer.device).manual_seed(seed)
    h = model.history_length
    model.fit_normalization(*_window(buffer.sample(16384, steps, history=h, generator=generator)))
    rate = torch.tensor(learning_rate, device=buffer.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=rate, capturable=True)
    batch = _window(buffer.sample(batch_size, steps, history=h, generator=generator))

    def step() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=False)
        loss = model.loss(*batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        return loss.detach()

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        loss = step()

    log = []
    start = time.perf_counter()
    for i in range(iterations):
        for target, source in zip(
            batch, _window(buffer.sample(batch_size, steps, history=h, generator=generator)), strict=True
        ):
            target.copy_(source)
        rate.fill_(learning_rate * (0.03 + 0.97 * 0.5 * (1.0 + math.cos(math.pi * i / iterations))))
        graph.replay()
        if i % 1000 == 0 or i == iterations - 1:
            log.append({"iteration": i, "loss": float(loss), "seconds": time.perf_counter() - start})
            print(json.dumps(log[-1]), flush=True)
    return log


@torch.no_grad()
def evaluate(model: WorldModel, buffer: ReplayBuffer, *, chunk: int = 8192) -> dict:
    """Score the post-touchdown vertical force of ``model`` on ``buffer``.

    Returns:
        Per horizon (``"1"``, ``"8"``, ``"32"``) and bin: ``rmse_n``, ``bias_n``
        (predicted minus logged), ``mean_n`` (logged) and ``count``, both feet pooled.
    """
    data, h = buffer.data, model.history_length
    size = buffer.size
    order = (buffer.rows - size + torch.arange(size, device=buffer.device)) % buffer.capacity
    data = {key: value[:, order] for key, value in data.items()}
    force = data["aux"][..., _FORCE_Z]
    since = steps_since_touchdown(force.transpose(1, 2), data["reset"][:, None].expand(-1, 2, -1)).transpose(1, 2)
    result = {}
    for horizon in HORIZONS:
        world, row, foot = torch.nonzero((since >= 0) & (since < BINS), as_tuple=True)
        episode_step = data["episode_step"][world, row]
        # The window and its history padding must stay inside an episode whose start is held.
        keep = (episode_step >= horizon - 1) & (row >= episode_step)
        world, row, foot = world[keep], row[keep], foot[keep]
        start = row - (horizon - 1)
        predicted = []
        for i in range(0, len(world), chunk):
            w, s = world[i : i + chunk, None], start[i : i + chunk]
            offset = torch.arange(-h, horizon, device=buffer.device)
            rows = s[:, None] + torch.maximum(offset, -data["episode_step"][w[:, 0], s][:, None])
            model_state = model.start(data["state"][w, rows[:, : h + 1]], data["action"][w, rows[:, :h]])
            _, aux = model.rollout(model_state, data["action"][w, rows[:, h:]], data["xi"][w[:, 0], s])
            predicted.append(aux[:, -1])
        predicted = torch.cat(predicted) if predicted else torch.zeros(0, data["aux"].shape[-1], device=buffer.device)
        foot_z = torch.as_tensor(_FORCE_Z, device=buffer.device)[foot]
        error = predicted.gather(1, foot_z[:, None])[:, 0] - force[world, row, foot]
        bins = since[world, row, foot]
        result[str(horizon)] = [
            {
                "rmse_n": float(error[bins == b].square().mean().sqrt()),
                "bias_n": float(error[bins == b].mean()),
                "mean_n": float(force[world, row, foot][bins == b].mean()),
                "count": int((bins == b).sum()),
            }
            for b in range(BINS)
        ]
    return result


def _table(report: dict) -> str:
    lines = ["| model | buffer | horizon | " + " | ".join(f"{10 * b}-{10 * b + 10} ms" for b in range(BINS)) + " |"]
    lines.append("|" + " --- |" * (3 + BINS))
    for key, label in (("aligned", "touchdown-aligned"), ("all_phases", "all phases")):
        measured = report["measured"][key]
        for stat in ("sd", "mean"):
            values = " | ".join(f"{v:.0f}" for v in measured[f"{stat}_n"])
            lines.append(f"| measured {stat}, {label} (n={measured['touchdowns']}) | | | {values} |")
    for model, buffers in report["models"].items():
        for name, horizons in buffers.items():
            for horizon, bins in horizons.items():
                cells = " | ".join(f"{b['rmse_n']:.0f} (logged {b['mean_n']:.0f}, n={b['count']})" for b in bins)
                lines.append(f"| {model} | {name} | {horizon} | {cells} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fit and score the SuperMimic world model (M3).")
    commands = parser.add_subparsers(dest="command", required=True)
    sub = commands.add_parser("fit")
    sub.add_argument("--train", type=Path, required=True, help="Buffer .npz written by collect")
    sub.add_argument("--out", type=Path, required=True)
    sub.add_argument("--history", type=int, default=4)
    sub.add_argument("--hidden", default="1024,1024,1024", help="Comma-separated hidden widths")
    sub.add_argument("--steps", type=int, default=8, help="k of the k-step loss")
    sub.add_argument("--batch-size", type=int, default=1024)
    sub.add_argument("--iterations", type=int, default=20000)
    sub.add_argument("--learning-rate", type=float, default=3.0e-4)
    sub.add_argument("--seed", type=int, default=0)
    sub = commands.add_parser("evaluate")
    sub.add_argument("--model", type=Path, nargs="+", required=True, help="Folders written by fit")
    sub.add_argument("--buffer", action="append", required=True, help="NAME=PATH of a buffer .npz")
    sub.add_argument("--held-out", nargs="+", required=True, help="Buffer names that each get a verdict")
    sub.add_argument("--subject", type=Path, default=F01, help="Subject spec of the measured spread")
    sub.add_argument("--out", type=Path, required=True, help="Report .json; a .md table is written next to it")
    args = parser.parse_args(argv)
    torch.backends.cuda.matmul.allow_tf32 = True

    if args.command == "fit":
        args.out.mkdir(parents=True, exist_ok=True)
        buffer = ReplayBuffer.load(args.train)
        config = {
            "joint_count": (buffer.data["state"].shape[-1] - 13) // 2,
            "xi_dim": buffer.data["xi"].shape[-1],
            "aux_dim": buffer.data["aux"].shape[-1],
            "history_length": args.history,
            "hidden": tuple(int(v) for v in args.hidden.split(",")),
        }
        torch.manual_seed(args.seed)
        model = WorldModel(**config).to(buffer.device)
        log = fit(
            model,
            buffer,
            args.iterations,
            steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
        )
        torch.save({"config": config, "state_dict": model.state_dict()}, args.out / "world_model.pt")
        record = {"train": str(args.train), "config": config, "args": {k: str(v) for k, v in vars(args).items()}}
        record["iterations_per_s"] = args.iterations / log[-1]["seconds"]
        record["log"] = log
        (args.out / "fit.json").write_text(json.dumps(record, indent=1) + "\n")
        return

    aligned, phases = touchdown_force(load_subject(args.subject))
    report = {
        "measured": {
            "subject": str(args.subject),
            **{
                key: {"touchdowns": len(forces), "mean_n": forces.mean(0).tolist(), "sd_n": forces.std(0).tolist()}
                for key, forces in (("aligned", aligned), ("all_phases", phases))
            },
        },
        "models": {},
    }
    buffers = dict(item.split("=", 1) for item in args.buffer)
    for name, path in buffers.items():
        buffer = ReplayBuffer.load(path)
        for folder in args.model:
            report["models"].setdefault(folder.name, {})[name] = evaluate(load_model(folder), buffer)
    sd = report["measured"]["aligned"]["sd_n"]
    report["go"] = {
        model: {
            name: all(results[name][str(GO_HORIZON)][b]["rmse_n"] <= sd[b] for b in range(1, BINS))
            for name in args.held_out
        }
        for model, results in report["models"].items()
    }
    report["first_bin_ratio"] = {
        model: {name: results[name][str(GO_HORIZON)][0]["rmse_n"] / sd[0] for name in args.held_out}
        for model, results in report["models"].items()
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    table = _table(report)
    for name in args.held_out:
        table += f"\n\nGo on {name} (8-step error <= touchdown-aligned SD at 10-50 ms): "
        table += ", ".join(f"{m}: {'go' if g[name] else 'no-go'}" for m, g in report["go"].items())
        table += "; 0-10 ms error / SD: " + ", ".join(
            f"{m}: {r[name]:.1f}" for m, r in report["first_bin_ratio"].items()
        )
    table += "\n\nThe 0-10 ms sim step starts before touchdown, the subject's at touchdown.\n"
    args.out.with_suffix(".md").write_text(table)
    print(table)


if __name__ == "__main__":
    main()
