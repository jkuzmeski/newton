# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Queue the stage-0 recollection and the M3 go/no-go on the grid as one SLURM chain.

``submit`` queues four arrays, each waiting for the previous one to succeed:

1. ``stage0``: collect shards of the train split (distinct seeds), with
   contexts in a ball around the fit of one shoe
   (:data:`~projects.supermimic.collect.CENTER_SHOE` by default), and one
   shard of the eval split, which runs that fit on the held-out reference
   windows.
2. ``stage0_merge``: :func:`merge` the train shards into ``train/`` and copies
   the eval shard to ``held_out/``, each a ``buffer.npz`` and ``meta.json``.
3. ``m3``: ``m3 fit`` with history 4 and 0, two seeds each.
4. ``m3_evaluate``: ``m3 evaluate`` of the four models, with ``held_out``
   deciding the verdict.

The reference folder, the F01 plate data and a copy of the F01 subject spec
pointing at the staged data are inputs; the shoes come from
:data:`~projects.supermimic.reference.SHOE_DATASET` on the grid share.

From the repository root::

    uv run --no-sync python -m projects.supermimic.grid.stage0 submit --commit <sha> \
        --reference outputs/supermimic/reference_F01_<date>
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from ..buffer import ReplayBuffer
from ..collect import CENTER_SHOE, summarize
from ..reference import F01, REPOSITORY
from .launch import REMOTE_ROOT, digest, submit

__all__ = ["chain", "merge"]

HISTORIES = (4, 0)
FIT_SEEDS = (0, 1)
EVAL_SEED = 100
"""Seed of the eval shard, apart from the train seeds ``0 .. shards - 1``."""
_PER_WORLD = ("shoe", "failures", "xi")
_SHARED = ("dt_s", "split", "center_shoe", "reference", "reference_frames_per_window", "xi_coordinates", "config")


def merge(shards: list[Path], output: Path) -> dict:
    """Concatenate the worlds of collect folders into one ``buffer.npz`` and ``meta.json``.

    Args:
        shards: Folders written by :func:`projects.supermimic.collect.main`,
            with equal settings and row counts.
        output: New folder of the merged collection.

    Returns:
        The merged metadata.
    """
    metas = [json.loads((shard / "meta.json").read_text()) for shard in shards]
    for meta in metas[1:]:
        for key in _SHARED:
            if meta[key] != metas[0][key]:
                raise ValueError(f"shards differ in {key!r}")
    arrays = [dict(np.load(shard / "buffer.npz")) for shard in shards]
    output.mkdir(parents=True)
    np.savez(output / "buffer.npz", **{key: np.concatenate([a[key] for a in arrays]) for key in arrays[0]})
    meta = {key: metas[0][key] for key in _SHARED}
    meta["seed"] = [m["seed"] for m in metas]
    meta["shards"] = [str(shard) for shard in shards]
    for key in _PER_WORLD:
        meta[key] = [value for m in metas for value in m[key]]
    meta["stiffness_short_steps"] = {
        joint: sum(m["stiffness_short_steps"][joint] for m in metas) for joint in metas[0]["stiffness_short_steps"]
    }
    meta["unloaded_stance_steps"] = np.sum([m["unloaded_stance_steps"] for m in metas], axis=0).tolist()
    meta["summary"] = summarize(ReplayBuffer.load(output / "buffer.npz", device="cpu"), meta)
    (output / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def _subject_input(folder: Path, data: str) -> None:
    spec = json.loads(F01.read_text())
    spec["data"] = data
    (folder / F01.name).write_text(json.dumps(spec, indent=2) + "\n")


def chain(
    commit: str,
    reference: Path,
    *,
    shoe: str,
    shards: int,
    worlds: int,
    eval_worlds: int,
    steps: int,
    iterations: int,
    collect_time: str,
    fit_time: str,
) -> list[dict]:
    """Queue the four dependent arrays; return their submission records."""
    subject = json.loads(F01.read_text())
    plates = REPOSITORY / subject["data"]
    reference_name = f"{reference.name}_{digest(reference)[:8]}"
    plates_name = f"{plates.name}_{digest(plates)[:8]}"
    with tempfile.TemporaryDirectory() as tmp:
        subject_folder = Path(tmp)
        _subject_input(subject_folder, f"{REMOTE_ROOT}/data/{plates_name}")
        subject_name = f"subject_{subject['name']}_{digest(subject_folder)[:8]}"
        data = {reference_name: reference, plates_name: plates, subject_name: subject_folder}

        collect = ["projects.supermimic.collect", "--reference", f"{{data}}/{reference_name}", "--shoe", shoe,
                   "--steps", str(steps)]  # fmt: skip
        tasks = [
            {"name": f"train_{s}", "args": [*collect, "--split", "train", "--worlds", str(worlds), "--seed", str(s),
                                           "--output", "{out}/stage0"]}
            for s in range(shards)
        ]  # fmt: skip
        tasks.append(
            {"name": "eval", "args": [*collect, "--split", "eval", "--worlds", str(eval_worlds),
                                      "--seed", str(EVAL_SEED), "--output", "{out}/stage0"]}
        )  # fmt: skip
        records = [submit("stage0", commit, tasks, data, collect_time)]
    run = records[-1]["run_dir"]
    train_shards = [f"{run}/train_{s}/stage0" for s in range(shards)]
    merge_task = {
        "name": "merge",
        "args": ["projects.supermimic.grid.stage0", "merge", "--out", "{out}/train", *train_shards,
                 "--held-out", f"{run}/eval/stage0"],
    }  # fmt: skip
    records.append(submit("stage0_merge", commit, [merge_task], {}, "00:30:00", after=records[-1]["job"]))
    buffers = records[-1]["run_dir"] + "/merge"
    fits = [
        {"name": f"h{h}_s{seed}", "args": ["projects.supermimic.m3", "fit", "--train", f"{buffers}/train/buffer.npz",
                                           "--out", "{out}", "--history", str(h), "--iterations", str(iterations),
                                           "--seed", str(seed)]}
        for h in HISTORIES
        for seed in FIT_SEEDS
    ]  # fmt: skip
    records.append(submit("m3", commit, fits, {}, fit_time, after=records[-1]["job"]))
    models = [f"{records[-1]['run_dir']}/{fit['name']}" for fit in fits]
    evaluate = {
        "name": "evaluate",
        "args": ["projects.supermimic.m3", "evaluate", "--model", *models, "--buffer",
                 f"train={buffers}/train/buffer.npz", "--buffer", f"held_out={buffers}/held_out/buffer.npz",
                 "--held-out", "held_out", "--subject", f"{{data}}/{subject_name}/{F01.name}", "--out", "{out}/report.json"],
    }  # fmt: skip
    records.append(submit("m3_evaluate", commit, [evaluate], {}, "00:30:00", after=records[-1]["job"]))
    return records


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    sub = commands.add_parser("submit", help="queue the recollection and M3 chain")
    sub.add_argument("--commit", required=True)
    sub.add_argument("--reference", type=Path, required=True, help="folder written by projects.supermimic.reference")
    sub.add_argument("--shoe", default=CENTER_SHOE, help="shoe whose fit centers the contexts and is held out")
    sub.add_argument("--shards", type=int, default=2, help="train shards, one H100 each")
    sub.add_argument("--worlds", type=int, default=256, help="worlds per train shard")
    sub.add_argument("--eval-worlds", type=int, default=128)
    sub.add_argument("--steps", type=int, default=2000, help="control steps per world")
    sub.add_argument("--iterations", type=int, default=50000, help="m3 fit iterations")
    sub.add_argument("--collect-time", default="04:00:00")
    sub.add_argument("--fit-time", default="01:00:00")
    sub = commands.add_parser("merge", help="merge collect shards (runs as a grid task)")
    sub.add_argument("--out", type=Path, required=True, help="merged train folder")
    sub.add_argument("shards", type=Path, nargs="+")
    sub.add_argument(
        "--held-out", type=Path, required=True, help="eval shard, merged alone into a held_out folder next to --out"
    )
    args = parser.parse_args(argv)
    if args.command == "merge":
        merge(args.shards, args.out)
        merge([args.held_out], args.out.parent / "held_out")
        return
    records = chain(
        args.commit,
        args.reference,
        shoe=args.shoe,
        shards=args.shards,
        worlds=args.worlds,
        eval_worlds=args.eval_worlds,
        steps=args.steps,
        iterations=args.iterations,
        collect_time=args.collect_time,
        fit_time=args.fit_time,
    )
    for record in records:
        print(json.dumps(record))


if __name__ == "__main__":
    main()
