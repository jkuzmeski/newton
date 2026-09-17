# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run a raw-force continuity example with a fixed leg/controller and mechanical settings."""

import argparse
import json
from pathlib import Path

import numpy as np

from .friction_dynamic_gpu import FrictionDynamicGPUWorkspace, parse_candidate_dict
from .friction_leg import verify_baseline_inputs
from .friction_metrics import score_friction_trace
from .friction_report import write_friction_report


def run_continuity(baseline: Path, output: Path, *, device="cuda:0", dt_factors=(1, 2), kt_scale=0.1) -> dict:
    """Compare raw mechanical force rates without fitting peaks or filtering output."""
    baseline, output = Path(baseline), Path(output)
    if output.exists():
        raise FileExistsError(output)
    if not np.isfinite(kt_scale) or kt_scale <= 0:
        raise ValueError("kt_scale must be positive")
    if not dt_factors or any(isinstance(f, bool) or not isinstance(f, int) or f < 1 for f in dt_factors):
        raise ValueError("dt_factors must contain positive integer divisors")
    if len(set(dt_factors)) != len(dt_factors):
        raise ValueError("dt_factors must not repeat output divisors")
    hashes = verify_baseline_inputs(baseline)
    artifact = json.loads((baseline / "digital_shoe.json").read_text())
    tau = artifact["constitutive_model"]["parameters"]["maxwell_relaxation_time_s"]
    candidate = {
        "method": "maxwell",
        "mu": 0.8,
        "kt_scale": kt_scale,
        "kv_scale": 1.0,
        "viscous_ratio": 0.0,
        "release_dwell_s": 0.0005,
        "yield_width": 0.0,
        "shear_relaxation_time_s": tau,
    }
    direct = {**candidate, "method": "deflection", "kt_scale": 1.0, "viscous_ratio": 0.2}
    settings = {"legacy": {**direct, "method": "legacy"}, "direct-damper": direct, "maxwell": candidate}
    output.mkdir(parents=True)
    rows = [parse_candidate_dict(p) for p in settings.values()]
    records = []
    report_scores = {}
    report_forces = {}
    report_times = {}
    plotted_reference = None
    for factor in dt_factors:
        w = FrictionDynamicGPUWorkspace(baseline, world_count=len(rows), device=device, dt_scale=1 / factor)
        result = w.evaluate(rows, curves=True)
        if not result["completed"].all():
            raise RuntimeError(f"A continuity rollout failed at timestep divisor {factor}")
        time = w.engine.time_s[: w.steps]
        native = w.reference["grf_time_s"]
        measured = w.reference["unfiltered_grf_target_n"]
        target = np.column_stack([np.interp(time, native, measured[:, a]) for a in range(2)])
        ref = {"grf_time_s": time, "grf_target_n": target}
        for index, name in enumerate(settings):
            force = result["curves"][index]
            jump = np.abs(np.diff(force, axis=0))
            rec = {
                "name": name,
                "dt_factor": factor,
                "dt_s": w.dt,
                "max_raw_force_step_n": jump.max(axis=0).tolist(),
                "max_raw_force_rate_n_s": (jump.max(axis=0) / w.dt).tolist(),
                "max_deflection_m": float(result["fit_scores"]["max_deflection_m"][index]),
                "positive_energy_residual_j": float(result["fit_scores"]["positive_energy_residual_j"][index]),
            }
            records.append(rec)
            np.savez_compressed(output / f"{name}_dt{factor}.npz", time_s=time, grf_n=force)
            if factor == dt_factors[0]:
                report_scores[name] = score_friction_trace(ref, {"time_s": time, "grf_n": force}, 1)
                report_forces[name] = force
                report_times[name] = time
                plotted_reference = {"grf_time_s": native, "grf_target_n": measured}
    if verify_baseline_inputs(baseline) != hashes:
        raise RuntimeError("Baseline inputs changed during continuity example")
    report = {
        "schema": "digital_shoe_friction_continuity_1",
        "input_hashes": hashes,
        "settings": settings,
        "records": records,
        "scores": report_scores,
        "forward_sign": 1,
        "zoom_interval_s": [0.045, 0.065],
        "steps_evaluated": len(next(iter(report_times.values()))),
        "total_source_steps": len(next(iter(report_times.values()))),
        "is_partial_smoke": False,
        "description": "UNFILTERED simulated forces. Fixed controller and normal law. This example tests continuity and rate, not peak-height fitting.",
        "qualification": "Mechanical demonstrator, not material calibration or six-channel experimental acceptance. Shear relaxation time is an explicit extrapolation of the existing effective material timescale.",
    }
    (output / "candidate.json").write_text(
        json.dumps({"parameters": candidate, "qualification": report["qualification"]}, indent=2)
    )
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    write_friction_report(
        report,
        next(iter(report_times.values())),
        report_forces,
        plotted_reference,
        output / "report.html",
        force_times=report_times,
    )
    return report


def main() -> None:
    """Run the reproducible continuity example and write unfiltered force plots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dt-factors", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--kt-scale", type=float, default=0.1)
    args = parser.parse_args()
    run_continuity(
        args.baseline, args.output, device=args.device, dt_factors=tuple(args.dt_factors), kt_scale=args.kt_scale
    )
    print(args.output / "report.html")


if __name__ == "__main__":
    main()
