# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Freeze train-selected materials and save matched full FR3 evaluations."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

from .identify import ForceObservation, load_trials, predict_many, scenario, score
from .materials import rematerialize, sha256, work_metrics, write_json
from .runner import RolloutConfig, Runner


def main(run: Path) -> None:
    """Evaluate three frozen train winners and the original material."""
    run = run.resolve()
    search = json.loads((run / "train_search.json").read_text())
    ranked = sorted([*search["results"], *search["refinements"]], key=lambda item: item["weighted_work_j"])
    selected = []
    keys = set()
    for item in ranked:
        if not item["eligible"]:
            continue
        p = item["parameters"]
        key = tuple(
            p[name]
            for name in (
                "equilibrium_compression_scale",
                "equilibrium_shear_scale",
                "equilibrium_fraction",
                "maxwell_relaxation_time_s",
            )
        )
        if key in keys:
            continue
        selected.append(item)
        keys.add(key)
        if len(selected) == 3:
            break
    if len(selected) != 3:
        raise RuntimeError("Fewer than three eligible distinct train candidates")
    eval_root = run / "finalists"
    eval_root.mkdir(exist_ok=True)
    write_json(
        eval_root / "selection.json",
        {
            "frozen_at": datetime.now(timezone.utc).isoformat(),
            "selection_split": "train",
            "rule": "Lowest weighted_work_j among eligible train candidates; three distinct parameter vectors.",
            "baseline": {
                "id": "original",
                "artifact": str(
                    Path("outputs/impedance_instron/hogan_scratch_20261009/dataset/assets/digital_shoe.json").resolve()
                ),
            },
            "finalists": [
                {
                    "id": item["id"],
                    "weighted_work_j": item["weighted_work_j"],
                    "parameters": item["parameters"],
                    "artifact": item["parameters"]["artifact"],
                }
                for item in selected
            ],
        },
    )
    all_trials = load_trials(
        "outputs/impedance_instron/hogan_scratch_20261009/dataset",
        mount_m=[-0.03186147427106201, 0.0, 0.10943209684347802],
        speed_m_s=3.65,
        force_observation=ForceObservation(20.0, 1.0, 1.0),
    )
    model = Path("projects/impedance_instron/hogan/baselines/generative_runner_score_20261010.json")
    runner = Runner.load(model)
    cfg = RolloutConfig(dt_s=0.000125, compression_limit=0.99)
    source_hashes = json.loads((run / "provenance.json").read_text())["source_sha256"]
    cases = [
        {
            "id": "original",
            "artifact": str(
                Path("outputs/impedance_instron/hogan_scratch_20261009/dataset/assets/digital_shoe.json").resolve()
            ),
            "label": "Current material",
        }
    ]
    cases += [
        {"id": x["id"], "artifact": x["parameters"]["artifact"], "label": "Synthetic finalist " + x["id"]}
        for x in selected
    ]
    summary = {
        "schema": "fr3-frozen-material-evaluation-v1",
        "status": "running",
        "conditions": [],
        "source_sha256": source_hashes,
        "controller_sha256": sha256(model),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(eval_root / "summary.json", summary)
    for case in cases:
        directory = eval_root / case["id"]
        directory.mkdir(exist_ok=True)
        artifact = Path(case["artifact"])
        matched = rematerialize(all_trials, artifact)
        tick = perf_counter()
        rows = predict_many([runner], matched, cfg, device="cuda:0")[0]
        condition = {
            "id": case["id"],
            "label": case["label"],
            "artifact": str(artifact),
            "artifact_sha256": sha256(artifact),
            "fitting_performed": False,
            "device": "cuda:0",
            "rollout": cfg.__dict__,
            "source_sha256": source_hashes,
            "input_model_sha256": sha256(model),
            "splits": {},
            "trials": [],
        }
        for index, (trial, (trace, rollout_summary)) in enumerate(zip(matched, rows, strict=True)):
            metrics = score(trace, rollout_summary, trial, runner) | work_metrics(trace)
            metrics["forward_displacement_m"] = float(trace["state"][-1, 0] - trace["state"][0, 0])
            condition["trials"].append(
                {
                    "id": trial.id,
                    "split": trial.split,
                    **trial.provenance,
                    "trace": f"trace_{index:03d}.npz",
                    "scenario": f"scenario_{index:03d}.json",
                    "metrics": metrics,
                }
            )
            np.savez_compressed(directory / f"trace_{index:03d}.npz", **trace)
            write_json(directory / f"scenario_{index:03d}.json", scenario(trial, cfg))
        for split in ("train", "eval"):
            split_rows = [x["metrics"] for x in condition["trials"] if x["split"] == split]
            condition["splits"][split] = {
                "mean_loss": float(np.mean([x["loss"] for x in split_rows])),
                "failed": sum(x["status"] != "completed" for x in split_rows),
                "trials": split_rows,
            }
        condition["wall_s"] = perf_counter() - tick
        shutil.copy2(model, directory / "runner.json")
        write_json(directory / "summary.json", condition)
        primary_i = next(i for i, item in enumerate(condition["trials"]) if item["id"] == "FR3_1_eval_000")
        primary = matched[primary_i]
        fine_trace, fine_summary = predict_many(
            [runner],
            [primary],
            RolloutConfig(dt_s=cfg.dt_s / 2, compression_limit=cfg.compression_limit),
            device="cuda:0",
        )[0][0]
        np.savez_compressed(directory / "half_step_primary.npz", **fine_trace)
        write_json(
            directory / "half-step.json",
            {
                "sample_id": primary.id,
                "split": "eval",
                "dt_s": cfg.dt_s / 2,
                "metrics": score(fine_trace, fine_summary, primary, runner) | work_metrics(fine_trace),
                "interpretation": "Numerical sensitivity diagnostic; not a physiological qualification.",
            },
        )
        summary["conditions"].append(
            {
                "id": case["id"],
                "label": case["label"],
                "artifact": str(artifact),
                "summary": str(directory / "summary.json"),
                "splits": {
                    k: {j: v for j, v in value.items() if j != "trials"} for k, value in condition["splits"].items()
                },
                "wall_s": condition["wall_s"],
            }
        )
        write_json(eval_root / "summary.json", summary)
        print(case["id"], summary["conditions"][-1]["splits"], flush=True)
    summary.update(status="completed", finished_at=datetime.now(timezone.utc).isoformat())
    write_json(eval_root / "summary.json", summary)


if __name__ == "__main__":
    import sys

    main(Path(sys.argv[1]))
