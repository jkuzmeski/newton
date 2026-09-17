# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Fit friction parameters in free leg dynamics with the controller frozen."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .friction_dynamic_gpu import BASE_PARAMETER_NAMES as PARAMETER_NAMES
from .friction_dynamic_gpu import FrictionDynamicGPUWorkspace
from .friction_leg import verify_baseline_inputs
from .friction_sweep import SCORE_NAMES

_BOUNDS = np.array(
    [[np.log(0.05), np.log(1.2)], [np.log(0.03), np.log(8.0)], [0.0, np.log1p(8.0)], [0.0, 0.5], [0.0, 0.005]]
)


def _bounds(method, speed_min=0.005):
    bounds = _BOUNDS.copy()
    if method == 4:
        bounds[0] = [np.log(0.15), np.log(1.2)]
        bounds = np.vstack((bounds, [np.log(0.1), 0.0], [np.log(speed_min), np.log(2.0)]))
    elif method == 5:
        bounds[0] = [np.log(0.4), np.log(1.6)]
        bounds = np.vstack((bounds, [np.log(1.0e4), np.log(1.0e6)]))
    return bounds


def _decode(unit, method, bounds=None):
    bounds = _bounds(method) if bounds is None else bounds
    physical = bounds[:, 0] + unit * (bounds[:, 1] - bounds[:, 0])
    rows = np.zeros((len(unit), 10 if method == 5 else 9 if method == 4 else 7), np.float32)
    rows[:, 0] = method
    rows[:, 1] = np.exp(physical[:, 0])
    rows[:, 2] = np.exp(physical[:, 1])
    rows[:, 3] = np.expm1(physical[:, 2])
    rows[:, 4:6] = physical[:, 3:5]
    if method == 4:
        rows[:, 7] = rows[:, 1] * np.exp(physical[:, 5])
        rows[:, 8] = np.exp(physical[:, 6])
    elif method == 5:
        rows[:, 7] = rows[:, 1]
        rows[:, 8] = 0.1
        rows[:, 9] = np.exp(physical[:, 5])
    return rows


def _encode(rows, bounds=None):
    method = int(rows[0, 0])
    values = rows[:, 1:6].astype(float).copy()
    values[:, :2] = np.log(values[:, :2])
    values[:, 2] = np.log1p(values[:, 2])
    if method == 4:
        values = np.column_stack((values, np.log(rows[:, 7] / rows[:, 1]), np.log(rows[:, 8])))
    elif method == 5:
        values = np.column_stack((values, np.log(rows[:, 9])))
    bounds = _bounds(method) if bounds is None else bounds
    return (values - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])


def _rank(result, limits, force_scale, score_key="fit_scores"):
    """Prefer fit quality without hiding failed or mechanically inconsistent cases."""
    f = result[score_key]
    value = f["loss"].astype(float).copy()
    comparison_rmse = result["engine_rmse"].copy()
    if score_key == "observation_scores":
        comparison_rmse[:, 4:] = result["observation_force_rmse_n"]
    extra = np.maximum(comparison_rmse[:, [0, 1, 2, 3, 5]] / limits[[0, 1, 2, 3, 5]] - 1.0, 0.0)
    value += 2.0 * np.sum(extra**2, axis=1)
    energy = np.maximum(f["positive_energy_residual_j"] / (np.abs(f["work_j"]) + 1.0) - 0.001, 0.0)
    shear = np.maximum(f["max_deflection_m"] / 0.02 - 1.0, 0.0)
    value += 10.0 * (energy**2 + shear**2)
    raw = result["raw_force_diagnostics"]
    # Detect raw sampling aliasing, not legitimate attenuation by the observation filter.
    native_raw = result["friction_scores"] if score_key == "observation_scores" else f
    peak_extra = np.maximum(
        raw[:, :2] - np.column_stack((native_raw["braking_peak_n"], native_raw["propulsive_peak_n"])), 0.0
    )
    value += 0.1 * np.sum((peak_extra / force_scale) ** 2, axis=1)
    value[~result["completed"]] = np.inf
    value[~np.isfinite(value)] = np.inf
    return value


def run_search(
    baseline,
    output,
    *,
    method=1,
    candidates=1024,
    worlds=128,
    generations=8,
    seed=29,
    device="cuda:0",
    speed_min=0.005,
    seed_candidate=None,
    score_policy="raw",
):
    """Search effective friction parameters without changing controller or normal laws."""
    baseline, output = Path(baseline).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if method not in (0, 1, 4, 5) or candidates < 1 or worlds < 1 or generations < 0:
        raise ValueError("Invalid search settings")
    if not np.isfinite(speed_min) or not 1e-6 <= speed_min < 2.0:
        raise ValueError("speed_min must be finite, at least 1e-6 and below 2 m/s")
    if score_policy not in ("raw", "matched", "legacy-full-rate"):
        raise ValueError("score_policy must be raw, matched or legacy-full-rate")
    score_key = {"raw": "physical_scores", "matched": "observation_scores", "legacy-full-rate": "fit_scores"}[
        score_policy
    ]
    bounds = _bounds(method, speed_min)
    input_hashes = verify_baseline_inputs(baseline)
    names = (
        "friction_dynamic_search.py",
        "friction_dynamic_gpu.py",
        "friction_observation.py",
        "friction_metrics.py",
        "friction_parameter_adapter.py",
        "friction_sweep.py",
        "friction_deflection.py",
        "friction_stribeck.py",
        "friction_pressure.py",
        "friction_slip_history.py",
        "friction_maxwell.py",
        "contact.py",
        "runtime.py",
    )
    sources = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in names}
    output.mkdir(parents=True)
    w = FrictionDynamicGPUWorkspace(
        baseline, world_count=worlds, device=device, matched_observation=score_policy == "matched"
    )
    if score_policy == "raw" and w.physical_target is None:
        raise ValueError("Raw-physics scoring requires the pre-20Hz reference force")
    cfg = w.fit_config
    limits = np.array([cfg.hip_tolerance_m] * 2 + [cfg.joint_tolerance_rad] * 2 + [cfg.force_tolerance_n] * 2)
    baseline_row = np.array([[0, 0.8, 1.0, 1.0, 0.2, 0.0005, 0.0]], np.float32)
    baseline_result = w.evaluate(baseline_row)
    rng = np.random.default_rng(seed)
    dimensions = len(bounds)
    parameter_names = (
        (*PARAMETER_NAMES, "mu_dynamic", "transition_speed", "pressure_scale_pa")
        if method == 5
        else (*PARAMETER_NAMES, "mu_dynamic", "transition_speed")
        if method == 4
        else PARAMETER_NAMES
    )
    unit = (np.arange(candidates)[:, None] + rng.random((candidates, dimensions))) / candidates
    for column in range(dimensions):
        rng.shuffle(unit[:, column])
    initial = _decode(unit, method, bounds)
    reference = baseline_row.copy()
    reference[:, 0] = method
    # Include the original setting and a transferred fixed-motion fit as declared seeds.
    transfer = np.array([[method, 0.497158, 0.416209, 5.20228, 0.004689, 0.00186345, 0.0]], np.float32)
    if method == 4:
        reference = np.column_stack((reference, reference[:, 1], [0.1]))
        transfer = np.column_stack((transfer, transfer[:, 1], [0.1]))
        stress_seeds = np.array(
            [
                [4, 0.8, 1, 1, 0.2, 0.0005, 0, 0.4, 0.2],
                [4, 0.8, 0.3, 1, 0.2, 0.0005, 0, 0.35, 0.5],
                [4, 0.5393827, 0.06856058, 7.985398, 0.2588549, 0.0009924, 0, 0.5393827, 0.1],
            ],
            np.float32,
        )
        initial = np.concatenate((reference, transfer, stress_seeds, initial))
    elif method == 5:
        reference = np.column_stack((reference, reference[:, 1], [0.1], [1.0e12]))
        transfer = np.column_stack((transfer, transfer[:, 1], [0.1], [1.0e12]))
        initial = np.concatenate((reference, transfer, initial))
    else:
        initial = np.concatenate((reference, transfer, initial))
    seed_hash = None
    if seed_candidate is not None:
        payload = json.loads(Path(seed_candidate).read_text())
        row = np.asarray([[payload["parameters"][name] for name in parameter_names]], np.float32)
        if int(row[0, 0]) != method:
            raise ValueError("Seed candidate must use the selected friction method")
        initial = np.concatenate((row, initial))
        seed_hash = hashlib.sha256(Path(seed_candidate).read_bytes()).hexdigest()
    pools = []
    scores = []
    ranks = []
    rmses = []
    completes = []
    raws = []
    failures = []
    stages = []
    times = []

    def evaluate(rows, stage):
        for start in range(0, len(rows), worlds):
            batch = rows[start : start + worlds]
            result = w.evaluate(batch)
            score = np.column_stack([result[score_key][name] for name in SCORE_NAMES])
            rank = _rank(result, limits, w.force_scale, score_key)
            pools.append(batch.copy())
            scores.append(score)
            ranks.append(rank)
            rmses.append(result["engine_rmse"])
            completes.append(result["completed"])
            raws.append(result["raw_force_diagnostics"])
            failures.append(result["failure_code"])
            stages.extend([stage] * len(batch))
            times.append(
                {
                    "stage": stage,
                    "attempted": len(batch),
                    "completed": int(result["completed"].sum()),
                    "seconds": w.last_seconds,
                }
            )
        print(
            stage,
            "attempted",
            sum(len(x) for x in pools),
            "best_rank",
            float(np.min(np.concatenate(ranks))),
            flush=True,
        )

    evaluate(initial, "coverage")
    convergence = []
    for generation in range(generations):
        p = np.concatenate(pools)
        r = np.concatenate(ranks)
        good = np.flatnonzero(np.isfinite(r))
        if not len(good):
            raise RuntimeError("No complete, finite candidate to refine")
        elite = good[np.argsort(r[good])[: min(12, len(good))]]
        centers = _encode(p[elite], bounds)
        radius = 0.16 * (0.6**generation)
        proposal = np.clip(
            centers[rng.integers(0, len(centers), worlds)] + rng.normal(0.0, radius, (worlds, dimensions)), 0.0, 1.0
        )
        previous = float(r.min())
        evaluate(_decode(proposal, method, bounds), f"refine{generation}")
        best = float(np.min(np.concatenate(ranks)))
        convergence.append(
            {"generation": generation, "radius": radius, "best_rank": best, "improvement": previous - best}
        )
    p = np.concatenate(pools)
    s = np.concatenate(scores)
    r = np.concatenate(ranks)
    rmse = np.concatenate(rmses)
    complete = np.concatenate(completes)
    raw = np.concatenate(raws)
    failure = np.concatenate(failures)
    mechanics = complete & (s[:, 6] <= 0.001 * (np.abs(s[:, 9]) + 1.0)) & (s[:, 7] <= 0.02) & (s[:, 8] <= 1e-3)
    six_pass = complete & np.all(rmse < limits, axis=1)
    eligible = mechanics & six_pass
    best_rank = int(np.argmin(r))
    eligible_indices = np.flatnonzero(eligible)
    selected = best_rank if not len(eligible_indices) else int(eligible_indices[np.argmin(r[eligible_indices])])
    np.savez_compressed(
        output / "candidates.npz",
        parameters=p,
        scores=s,
        selection_loss=r,
        engine_rmse=rmse,
        completed=complete,
        mechanics_screen=mechanics,
        six_channel_pass=six_pass,
        raw_force_diagnostics=raw,
        failure_code=failure,
        stage=np.asarray(stages),
    )
    # Rerun the selected candidate cold and preserve the actual free-leg trajectory.
    selected_result = w.evaluate(p[selected : selected + 1], curves=True)
    trace, run = w.trace(0)
    np.savez_compressed(output / "trace.npz", **trace)
    (output / "run.json").write_text(json.dumps(run, indent=2))
    candidate = {
        "parameters": dict(zip(parameter_names, p[selected].astype(float).tolist(), strict=True)),
        "selection": "best measured-six-channel and mechanics screened"
        if eligible[selected]
        else "best penalized diagnostic; not accepted",
        "source_hashes": sources,
        "input_hashes": input_hashes,
        "controller": "frozen baseline12 coefficients and gains",
        "qualification": "Effective friction fit to one stance, not independently calibrated or installed as default.",
    }
    (output / "candidate.json").write_text(json.dumps(candidate, indent=2))
    baseline_summary = {
        "friction_scores": {k: float(v[0]) for k, v in baseline_result["friction_scores"].items()},
        "engine_rmse": baseline_result["engine_rmse"][0].tolist(),
    }
    report = {
        "schema": "digital_shoe_friction_dynamic_search_2",
        "score_policy": score_policy,
        "observation_metadata": selected_result.get("observation_metadata"),
        "selected_observation_force_rmse_n": selected_result["observation_force_rmse_n"][0].tolist()
        if "observation_force_rmse_n" in selected_result
        else None,
        "selected_legacy_full_rate_score": {
            name: float(value[0]) for name, value in selected_result["fit_scores"].items()
        },
        "method": method,
        "seed": seed,
        "seed_candidate_sha256": seed_hash,
        "transition_speed_min_m_s": speed_min,
        "bounds_transformed": bounds.tolist(),
        "parameter_names": parameter_names,
        "score_names": SCORE_NAMES,
        "baseline": baseline_summary,
        "controller_coefficients_sha256": input_hashes["equilibrium.npz"],
        "input_hashes": input_hashes,
        "source_hashes": sources,
        "attempted": len(p),
        "completed": int(complete.sum()),
        "mechanics_screened": int(mechanics.sum()),
        "six_channel_passed": int(six_pass.sum()),
        "eligible": int(eligible.sum()),
        "selection_index": selected,
        "selected_parameters": candidate["parameters"],
        "selected_score": dict(zip(SCORE_NAMES, s[selected].astype(float).tolist(), strict=True)),
        "selected_native_score": {name: float(value[0]) for name, value in selected_result["friction_scores"].items()},
        "fit_sampling": {
            "raw": "UNFILTERED simulated force at every simulation substep versus interpolated pre-20Hz input reference (already acquisition-cleaned/Hann-filtered). No simulation-output filtering.",
            "matched": "Secondary signed bandwidth-matched observation diagnostic; never evidence that physical force became smoother.",
            "legacy-full-rate": "Legacy full-rate comparison against interpolated filtered/clipped reference, retained for old-study reproduction.",
        }[score_policy],
        "selected_engine_rmse": rmse[selected].tolist(),
        "selected_raw_force_diagnostics": raw[selected].tolist(),
        "selected_mechanics_screen": bool(mechanics[selected]),
        "selected_six_channel_pass": bool(six_pass[selected]),
        "limits_original": limits.tolist(),
        "convergence": convergence,
        "timings": times,
        "completed_candidates_per_second": int(complete.sum()) / max(sum(x["seconds"] for x in times), 1e-12),
        "selection_objective": "Friction curve/phase loss plus soft penalties for exceeding existing non-friction channel tolerances, positive tangential energy residual, >20mm elastic shear, and raw peaks missed by native sampling. Failed rollouts cannot win.",
        "qualification": "Same measured stance used for fitting. No independent friction validation. Existing normal code, material, controller gains and equilibrium coefficients unchanged. Final CPU and halfstep checks required before any promotion.",
    }
    if verify_baseline_inputs(baseline) != input_hashes:
        raise RuntimeError("Baseline changed during search")
    for name, digest in sources.items():
        if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() != digest:
            raise RuntimeError("Friction search source changed during execution")
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(
        "selected",
        json.dumps(
            {
                "parameters": candidate["parameters"],
                "score": report["selected_score"],
                "rmse": report["selected_engine_rmse"],
                "eligible": bool(eligible[selected]),
            }
        ),
        flush=True,
    )
    return report


def main():
    """Run a bounded GPU friction search with the original leg controller frozen."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", type=int, choices=(0, 1, 4, 5), default=1)
    parser.add_argument("--candidates", type=int, default=1024)
    parser.add_argument("--worlds", type=int, default=128)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--speed-min", type=float, default=0.005)
    parser.add_argument("--seed-candidate", type=Path)
    parser.add_argument("--score-policy", choices=("raw", "matched", "legacy-full-rate"), default="raw")
    args = parser.parse_args()
    run_search(
        args.baseline,
        args.output,
        method=args.method,
        candidates=args.candidates,
        worlds=args.worlds,
        generations=args.generations,
        seed=args.seed,
        device=args.device,
        speed_min=args.speed_min,
        seed_candidate=args.seed_candidate,
        score_policy=args.score_policy,
    )


if __name__ == "__main__":
    main()
