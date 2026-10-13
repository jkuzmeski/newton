# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Run a predeclared fixed-FR3-geometry shoe material sensitivity sweep."""

from __future__ import annotations

import itertools
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

from .identify import ForceObservation, load_trials, predict_many, score
from .materials import rematerialize, sha256, work_metrics, write_json
from .runner import RolloutConfig, Runner

DATASET = Path("outputs/impedance_instron/hogan_scratch_20261009/dataset")
MODEL = Path("projects/impedance_instron/hogan/baselines/generative_runner_score_20261010.json")
MOUNT = [-0.03186147427106201, 0.0, 0.10943209684347802]


def build_candidate(
    source: Path, destination: Path, *, compression: float, shear: float, fraction: float, tau: float
) -> dict:
    """Scale equilibrium compression/shear and Maxwell time on copied geometry."""
    raw = json.loads(source.read_text())
    base = raw["constitutive_model"]["parameters"]
    old_fraction = float(base["equilibrium_fraction"])
    base["instantaneous_shear_modulus_pa"] *= compression * old_fraction / fraction
    base["instantaneous_shear_modulus_2_pa"] *= compression * old_fraction / fraction
    base["equilibrium_fraction"] = fraction
    base["maxwell_relaxation_time_s"] = tau
    original_shear = float(
        raw["constitutive_model"]
        .get("derived_quantities", {})
        .get("equilibrium_shear_modulus_pa", base.get("equilibrium_shear_modulus_override_pa", 0.0))
    )
    base["equilibrium_shear_modulus_override_pa"] = original_shear * shear
    thickness = np.asarray(raw["column_bed"]["rest_length_m"])
    effective_shear = original_shear * shear
    base["pasternak_n_per_m"] = float(effective_shear * thickness.mean())
    raw["shoe"]["id"] += f"__sweep_c{compression:g}_s{shear:g}_e{fraction:g}_t{tau:g}"
    raw["shoe"]["model_scope"] = (
        "Synthetic material scaling on unchanged current FR3 shoe geometry; uncalibrated sensitivity study."
    )
    raw["constitutive_model"]["derived_quantities"] = {
        "equilibrium_shear_modulus_pa": effective_shear,
        "equilibrium_compression_scale": compression,
        "pasternak_rule": "k_i = independently scaled equilibrium shear modulus * rest_length_m[i]",
    }
    raw["validation"] = {
        "status": "synthetic uncalibrated parameter sweep",
        "claim_boundary": "No material calibration, human biomechanics validation, hysteresis validation, or fatigue model.",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json(destination, raw)
    return {
        "artifact": str(destination.resolve()),
        "sha256": sha256(destination),
        "equilibrium_compression_scale": compression,
        "equilibrium_shear_scale": shear,
        "equilibrium_fraction": fraction,
        "maxwell_relaxation_time_s": tau,
        "effective_shear_modulus_pa": effective_shear,
    }


def main() -> None:
    run = Path("outputs/impedance_instron/materials_sweep_fr3_20261010")
    if run.exists():
        suffix = 2
        while run.with_name(run.name + f"_{suffix}").exists():
            suffix += 1
        run = run.with_name(run.name + f"_{suffix}")
    run.mkdir(parents=True)
    (run / "search").mkdir()
    trials = load_trials(DATASET, mount_m=MOUNT, speed_m_s=3.65, force_observation=ForceObservation(20.0, 1.0, 1.0))
    if not trials or any(not t.provenance["compatibility"]["passed"] for t in trials):
        raise RuntimeError("Input FR3 trials failed compatibility checks")
    source = trials[0].shoe.artifact_path
    cfg = RolloutConfig(dt_s=0.000125, compression_limit=0.99)
    coarse = list(
        itertools.product(
            (0.5, 1.0, 2.0), (0.5, 1.0, 2.0), (0.5, 0.6952954850747634, 1.0), (0.001, 0.005000407970309165, 0.02)
        )
    )
    # At equilibrium_fraction=1, relaxation is inactive, so identical elastic candidates are deduplicated.
    candidates = []
    seen = set()
    for compression, shear, fraction, tau in coarse:
        key = (compression, shear, fraction, 0.0 if fraction == 1.0 else tau)
        if key in seen:
            continue
        seen.add(key)
        candidates.append((compression, shear, fraction, tau))
    baseline = {"compression": 1.0, "shear": 1.0, "fraction": 0.6952954850747634, "tau_s": 0.005000407970309165}
    plan = {
        "run_id": run.name,
        "domain": "impedance-instron",
        "category": "materials-testing",
        "planned_at": datetime.now(timezone.utc).isoformat(),
        "question": "Which synthetic material scalings reduce frozen-controller FR3 marathon work proxy while retaining task performance?",
        "baseline": {
            "material": baseline,
            "geometry_sha256": sha256(source),
            "controller": str(MODEL.resolve()),
            "controller_sha256": sha256(MODEL),
        },
        "changed_variables": {
            "equilibrium_compression_scale": [0.5, 1.0, 2.0],
            "equilibrium_shear_scale": [0.5, 1.0, 2.0],
            "equilibrium_fraction": [0.5, 0.6952954850747634, 1.0],
            "relaxation_time_s": [0.001, 0.005000407970309165, 0.02],
        },
        "command": "uv run --no-sync python -m projects.impedance_instron.hogan.materials_sweep",
        "effective_configuration": {
            "rollout": cfg.__dict__,
            "candidate_count_after_deduplication": len(candidates),
            "synthetic_parameterization": "scale instantaneous terms by compression*old_eq_fraction/new_eq_fraction; independently scale equilibrium shear; change Maxwell tau; preserve shoe geometry, body, friction, state, and controller",
        },
        "seed": "Deterministic grid; no fitting.",
        "splits": {s: [t.id for t in trials if t.split == s] for s in ("train", "eval")},
        "selection_rule": "Rank completed train-only candidates by 4*positive_work + absorbed_work/1.2, subject to all predeclared task-quality guardrails. Freeze three distinct finalists before held-out evaluation. No eval metrics enter selection.",
        "guardrails": {
            "all_train_windows_completed": True,
            "mean_loss_max_ratio_to_baseline": 1.25,
            "mean_forward_displacement_min_ratio_to_baseline": 0.9,
            "contact_duration_and_impulse_ratio_range": [0.8, 1.2],
        },
        "local_refinement_rule": "If a best coarse candidate lies at any scale boundary, evaluate one inward/outward local level where physically valid, clipped to [0.25, 3] stiffness/shear, [0.25,1] equilibrium fraction and [0.0005,0.04] s relaxation; do not change held-out selection protocol.",
        "budget": {
            "search": "Serial exact CUDA evaluations on train split; then baseline plus three frozen candidates on all held-out windows; native contact replay for finalists.",
            "stopping": "Finish declared grid, refinement rule, held-out evaluation, capture, and report.",
        },
        "qualification_checks": [
            "FR3 geometry hash held fixed",
            "controller and trial initialization held fixed",
            "task-quality guardrails",
            "all failed/incomplete windows retained",
            "held-out metrics excluded from selection",
        ],
        "replay_samples": [
            {
                "condition": "baseline and each frozen finalist",
                "sample_id": "FR3_1_eval_000",
                "split": "eval",
                "selection": "first held-out sample, fixed before evaluation",
            }
        ],
        "capture_plan": {
            "native": "all train search metrics; finalists native traces and scenarios",
            "deformation": "saved-pose source-checked contact replay copied from materials_geometry_20261010",
            "report": "train selection and held-out comparison, native biomechanics, compression/contact, conditional marathon scenarios",
        },
        "limitations": [
            "Synthetic material laws are uncalibrated; marathon estimates are conditional model scenarios."
        ],
    }
    write_json(run / "experiment-plan.json", plan)
    write_json(
        run / "experiment-artifacts.json",
        {
            "schema": "instron-experiment-artifacts-v1",
            "domain": "impedance-instron",
            "run_id": run.name,
            "experiment_status": "planned",
            "fitting_performed": False,
            "approved_roots": [str(Path.cwd())],
            "replay": {
                "model_identity": str(MODEL),
                "sample_id": "FR3_1_eval_000",
                "split": "eval",
                "shoe_identity": sha256(source),
                "alignment": {"declared": True, "basis": "frozen states, same-step shoe contact"},
            },
            "artifacts": {
                key: {"status": "planned", "path": value}
                for key, value in {
                    "provenance": "provenance.json",
                    "configuration": "experiment-plan.json",
                    "selected_model": "runner.json",
                    "motion_trace": "finalists/",
                    "motion_viewer": "report",
                    "geometry": "materials/",
                    "deformation": "finalists/replay_evidence/",
                    "deformation_validation": "finalists/replay_evidence/validation.json",
                    "biomechanics": "finalists/",
                    "training_history": "train_search.json",
                    "metrics": "train_search.json",
                    "qualification": "qualification.json",
                    "report_spec": "report-spec.json",
                    "report_html": "report.html",
                }.items()
            },
        },
    )
    write_json(
        run / "commands.json", {"command": plan["command"], "effective_configuration": plan["effective_configuration"]}
    )
    source_files = [
        Path("projects/impedance_instron/hogan/materials_sweep.py"),
        Path("projects/impedance_instron/hogan/materials.py"),
        Path("projects/impedance_instron/hogan/runner.py"),
        Path("projects/impedance_instron/hogan/identify.py"),
        Path("projects/impedance_instron/cartesian/shoe.py"),
        Path("projects/digital_shoe/runtime.py"),
        Path("projects/digital_shoe/material.py"),
    ]
    source_hashes = {str(path): sha256(path) for path in source_files}
    for path in source_files:
        destination = run / "source_snapshot" / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    runner = Runner.load(MODEL)
    train = [t for t in trials if t.split == "train"]
    baseline_trials = train
    base_pred = predict_many([runner], baseline_trials, cfg, device="cuda:0")[0]
    base_rows = []
    for ti, (tr, su) in zip(train, base_pred, strict=True):
        row = score(tr, su, ti, runner) | work_metrics(tr)
        row["forward_displacement_m"] = float(tr["state"][-1, 0] - tr["state"][0, 0])
        base_rows.append(row)

    def summarize(rows):
        return {
            "n": len(rows),
            "completed": sum(r["status"] == "completed" for r in rows),
            "mean_loss": float(np.mean([r["loss"] for r in rows])),
            "mean_forward_displacement_m": float(
                np.mean([r.get("forward_displacement_m", r.get("forward_distance_m", 0.0)) for r in rows])
            ),
            "mean_contact_duration_s": float(np.mean([r["contact_duration_s"] for r in rows])),
            "mean_vertical_impulse_ns": float(np.mean([r["grf_impulse_ns"][1] for r in rows])),
            "mean_positive_work_j": float(np.mean([r["total_positive_work_j"] for r in rows])),
            "mean_absorbed_work_j": float(np.mean([r["total_absorbed_work_j"] for r in rows])),
        }

    base = summarize(base_rows)
    write_json(
        run / "baseline_train_metrics.json",
        {"summary": base, "trials": [{"id": t.id, **r} for t, r in zip(train, base_rows, strict=True)]},
    )
    t0 = perf_counter()
    results = []
    for index, (compression, shear, fraction, tau) in enumerate(candidates):
        ident = f"c{compression:g}_s{shear:g}_e{fraction:g}_t{tau:g}"
        artifact = run / "materials" / f"{ident}.json"
        params = build_candidate(source, artifact, compression=compression, shear=shear, fraction=fraction, tau=tau)
        matched = rematerialize(train, artifact)
        prediction = predict_many([runner], matched, cfg, device="cuda:0")[0]
        rows = []
        for ti, (tr, su) in zip(matched, prediction, strict=True):
            row = score(tr, su, ti, runner) | work_metrics(tr)
            row["forward_displacement_m"] = float(tr["state"][-1, 0] - tr["state"][0, 0])
            rows.append(row)
        stats = summarize(rows)
        guard = {
            "completed": stats["completed"] == stats["n"],
            "loss": stats["mean_loss"] <= 1.25 * base["mean_loss"],
            "displacement": stats["mean_forward_displacement_m"] >= 0.9 * base["mean_forward_displacement_m"],
            "contact_duration": 0.8 <= stats["mean_contact_duration_s"] / base["mean_contact_duration_s"] <= 1.2,
            "vertical_impulse": 0.8 <= stats["mean_vertical_impulse_ns"] / base["mean_vertical_impulse_ns"] <= 1.2,
        }
        work = 4.0 * stats["mean_positive_work_j"] + stats["mean_absorbed_work_j"] / 1.2
        results.append(
            {
                "id": ident,
                "parameters": params,
                "metrics": stats,
                "weighted_work_j": work,
                "guardrails": guard,
                "eligible": all(guard.values()),
                "trials": [{"id": t.id, **r} for t, r in zip(matched, rows, strict=True)],
            }
        )
        write_json(run / "search" / f"{ident}.json", results[-1])
        print(f"[{index + 1}/{len(candidates)}] {ident} work={work:.3f} eligible={results[-1]['eligible']}", flush=True)
    # Predeclared bounded, one-axis refinement around the best eligible coarse point.
    eligible = [r for r in results if r["eligible"]]
    best = min(eligible, key=lambda r: r["weighted_work_j"]) if eligible else None
    refinements = []
    if best is not None:
        p = best["parameters"]
        axes = [
            ("compression", p["equilibrium_compression_scale"], (0.25, 3.0)),
            ("shear", p["equilibrium_shear_scale"], (0.25, 3.0)),
            ("fraction", p["equilibrium_fraction"], (0.25, 1.0)),
            ("tau", p["maxwell_relaxation_time_s"], (0.0005, 0.04)),
        ]
        for axis, value, bounds in axes:
            at_edge = (
                value in (0.5, 2.0)
                if axis in ("compression", "shear")
                else (value == 0.5 or value == 1.0)
                if axis == "fraction"
                else value in (0.001, 0.02)
            )
            if not at_edge:
                continue
            for direction in (-1, 1):
                trial_value = min(bounds[1], max(bounds[0], value * (1.5 if direction > 0 else 2 / 3)))
                values = {
                    "compression": p["equilibrium_compression_scale"],
                    "shear": p["equilibrium_shear_scale"],
                    "fraction": p["equilibrium_fraction"],
                    "tau": p["maxwell_relaxation_time_s"],
                }
                values[axis] = trial_value
                combo = (values["compression"], values["shear"], values["fraction"], values["tau"])
                if combo in candidates:
                    continue
                ident = f"refine_{axis}_{trial_value:.6g}"
                artifact = run / "materials" / f"{ident}.json"
                params = build_candidate(
                    source, artifact, compression=combo[0], shear=combo[1], fraction=combo[2], tau=combo[3]
                )
                matched = rematerialize(train, artifact)
                prediction = predict_many([runner], matched, cfg, device="cuda:0")[0]
                rows = []
                for ti, (tr, su) in zip(matched, prediction, strict=True):
                    row = score(tr, su, ti, runner) | work_metrics(tr)
                    row["forward_displacement_m"] = float(tr["state"][-1, 0] - tr["state"][0, 0])
                    rows.append(row)
                stats = summarize(rows)
                guard = {
                    "completed": stats["completed"] == stats["n"],
                    "loss": stats["mean_loss"] <= 1.25 * base["mean_loss"],
                    "displacement": stats["mean_forward_displacement_m"] >= 0.9 * base["mean_forward_displacement_m"],
                    "contact_duration": 0.8
                    <= stats["mean_contact_duration_s"] / base["mean_contact_duration_s"]
                    <= 1.2,
                    "vertical_impulse": 0.8
                    <= stats["mean_vertical_impulse_ns"] / base["mean_vertical_impulse_ns"]
                    <= 1.2,
                }
                result = {
                    "id": ident,
                    "parameters": params,
                    "metrics": stats,
                    "weighted_work_j": 4 * stats["mean_positive_work_j"] + stats["mean_absorbed_work_j"] / 1.2,
                    "guardrails": guard,
                    "eligible": all(guard.values()),
                    "trials": [{"id": t.id, **r} for t, r in zip(matched, rows, strict=True)],
                }
                refinements.append(result)
                write_json(run / "search" / f"{ident}.json", result)
                print(f"{ident} work={result['weighted_work_j']:.3f} eligible={result['eligible']}", flush=True)
    write_json(
        run / "train_search.json",
        {
            "baseline": base,
            "results": results,
            "refinements": refinements,
            "wall_s": perf_counter() - t0,
            "selection_split": "train",
        },
    )
    write_json(
        run / "provenance.json",
        {
            "run_started_at": plan["planned_at"],
            "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "source_sha256": source_hashes,
            "input_manifest_sha256": sha256(DATASET / "manifest.json"),
            "source_shoe_sha256": sha256(source),
            "controller_sha256": sha256(MODEL),
            "runtime": {"device": "cuda:0"},
            "geometry_fixed": True,
        },
    )
    print("TRAIN SWEEP COMPLETE", run, flush=True)


if __name__ == "__main__":
    main()
