# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Refit equilibrium commands with fixed Maxwell friction, gains and normal mechanics."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from projects.impedance_instron.cartesian.fit import _Objective
from projects.impedance_instron.cartesian.gpu.resident import fit_resident
from projects.impedance_instron.cartesian.trajectory import Spline

from .friction_controller_objective import ControllerObjective
from .friction_dynamic import compute_source_hashes
from .friction_dynamic_gpu import FrictionDynamicGPUWorkspace, parse_candidate_dict
from .friction_leg import verify_baseline_inputs
from .friction_metrics import score_friction_trace
from .friction_report import write_friction_report


def raw_metrics(reference, trace):
    """Score each unfiltered simulation force sample after interpolating only input data."""
    t = trace["time_s"]
    f = reference["unfiltered_grf_target_n"]
    target = np.column_stack([np.interp(t, reference["grf_time_s"], f[:, a]) for a in range(2)])
    return score_friction_trace({"grf_time_s": t, "grf_target_n": target}, trace, 1)


def run_refit(
    baseline: Path,
    output: Path,
    *,
    iterations=80,
    max_wall_s=600.0,
    seed=71,
    initial_fraction=0.01,
    minimum_fraction=0.0005,
    device="cuda:0",
    initial_coefficients: Path | None = None,
    tracking_penalty=100.0,
    tracking_margin=1.0,
):
    """Optimize only the bounded 12-by-4 equilibrium commands, never friction or gains."""
    baseline = Path(baseline).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    inputs = verify_baseline_inputs(baseline)
    artifact = json.loads((baseline / "digital_shoe.json").read_text())
    tau = artifact["constitutive_model"]["parameters"]["maxwell_relaxation_time_s"]
    friction = {
        "method": "maxwell",
        "mu": 0.8,
        "kt_scale": 0.1,
        "kv_scale": 1.0,
        "viscous_ratio": 0.0,
        "release_dwell_s": 0.0005,
        "yield_width": 0.0,
        "shear_relaxation_time_s": tau,
    }
    sources = compute_source_hashes()
    for name in ("friction_controller_objective.py", "friction_controller_refit.py"):
        p = Path(__file__).with_name(name)
        sources[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    output.mkdir(parents=True)
    w = FrictionDynamicGPUWorkspace(baseline, world_count=128, device=device)
    w.adapter.set_parameters([parse_candidate_dict(friction)])
    friction_before = w.adapter.settings.numpy().copy()
    original = w.equilibrium_spline
    w.engine.evaluate(np.repeat(original.coefficients[None], 128, axis=0))
    old_trace, old_run = w.engine.trace(0)
    if old_run["status"] != "completed":
        raise RuntimeError("Same-friction baseline did not complete")
    np.savez_compressed(output / "baseline_trace.npz", **old_trace)
    (output / "baseline_run.json").write_text(json.dumps(old_run, indent=2, allow_nan=False))
    initial = original
    if initial_coefficients is not None:
        with np.load(initial_coefficients, allow_pickle=False) as d:
            initial = Spline(float(d["duration_s"]), d["coefficients"])
    objective = ControllerObjective(
        w.engine, w.adapter, old_trace, tracking_penalty=tracking_penalty, tracking_margin=tracking_margin
    )
    w.engine.objective = objective
    # Capture again so the fixed-contact rollout calls the new objective.
    w.engine.graph = None
    w.engine.tail_graph = None
    w.engine.resident_graph = None
    (output / "objective.json").write_text(json.dumps(objective.description, indent=2, allow_nan=False))
    print("Starting bounded equilibrium-only refit", flush=True)
    best, trace, run, fit = fit_resident(
        w.engine,
        initial,
        max_iterations=iterations,
        max_wall_s=max_wall_s,
        initial_step_fraction=initial_fraction,
        minimum_step_fraction=minimum_fraction,
        seed=seed,
        plateau_patience=20,
        plateau_rtol=1e-4,
    )
    if verify_baseline_inputs(baseline) != inputs:
        raise RuntimeError("Baseline inputs changed")
    np.testing.assert_array_equal(w.adapter.settings.numpy(), friction_before)
    for path, digest in sources.items():
        p = Path(path)
        if not p.is_absolute():
            p = Path(__file__).resolve().parents[2] / p
        if hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Source changed during refit: {p}")
    np.savez_compressed(output / "coefficients.npz", duration_s=best.duration_s, coefficients=best.coefficients)
    np.savez_compressed(output / "trace.npz", **trace)
    (output / "run.json").write_text(json.dumps(run, indent=2, allow_nan=False))
    old_scores = raw_metrics(w.reference, old_trace)
    new_scores = raw_metrics(w.reference, trace)
    original_objective = _Objective(w.reference, w.fit_config)
    _, old_six, _ = original_objective.evaluate(old_trace, old_run)
    _, new_six, _ = original_objective.evaluate(trace, run)
    report = {
        "schema": "digital_shoe_controller_refit_1",
        "input_hashes": inputs,
        "source_hashes": sources,
        "friction_parameters": friction,
        "gains_changed": False,
        "normal_model_changed": False,
        "initial_conditions_changed": False,
        "optimization": fit,
        "objective": objective.description,
        "old_controller_raw_metrics": old_scores,
        "refit_raw_metrics": new_scores,
        "original_six_metrics_before": old_six,
        "original_six_metrics_after": new_six,
        "qualification": "Controlled equilibrium-command diagnostic with fixed engineering friction. Independent CPU/refinement/effort validation required; no calibration or promotion inferred from search loss.",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    display = {
        "forward_sign": 1,
        "scores": {"old-controller": old_scores, "refit": new_scores},
        "steps_evaluated": len(trace["time_s"]),
        "total_source_steps": len(old_trace["time_s"]),
        "is_partial_smoke": False,
        "zoom_interval_s": [0.015, 0.10],
        "description": "UNFILTERED force; same Maxwell contact, gains, normal model and initial state. Only equilibrium commands changed.",
        "qualification": report["qualification"],
    }
    reference = {"grf_time_s": w.reference["grf_time_s"], "grf_target_n": w.reference["unfiltered_grf_target_n"]}
    write_friction_report(
        display,
        old_trace["time_s"],
        {"old-controller": old_trace["grf_n"], "refit": trace["grf_n"]},
        reference,
        output / "report.html",
    )
    print(
        json.dumps(
            {"old_raw": old_scores["trace_metrics"], "new_raw": new_scores["trace_metrics"], "six_after": new_six},
            indent=2,
        ),
        flush=True,
    )
    return report


def main():
    """Run a bounded, fixed-gain equilibrium-command refit."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--iterations", type=int, default=80)
    p.add_argument("--max-wall-s", type=float, default=600.0)
    p.add_argument("--seed", type=int, default=71)
    p.add_argument("--initial-fraction", type=float, default=0.01)
    p.add_argument("--minimum-fraction", type=float, default=0.0005)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--initial-coefficients", type=Path)
    p.add_argument("--tracking-penalty", type=float, default=100.0)
    p.add_argument("--tracking-margin", type=float, default=1.0)
    a = p.parse_args()
    run_refit(
        a.baseline,
        a.output,
        iterations=a.iterations,
        max_wall_s=a.max_wall_s,
        seed=a.seed,
        initial_fraction=a.initial_fraction,
        minimum_fraction=a.minimum_fraction,
        device=a.device,
        initial_coefficients=a.initial_coefficients,
        tracking_penalty=a.tracking_penalty,
        tracking_margin=a.tracking_margin,
    )


if __name__ == "__main__":
    main()
