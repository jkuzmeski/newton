# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Replay frozen leg motion to evaluate friction models under prescribed kinematics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import warp as wp

from projects.digital_shoe.friction_adapter import FrictionAdapter
from projects.digital_shoe.friction_metrics import score_friction_trace
from projects.digital_shoe.friction_report import write_friction_report
from projects.digital_shoe.provenance import physics_source_identity
from projects.impedance_instron.cartesian.gpu.contact_replay import (
    archive,
    exact_cpu_poses,
    stage,
    store,
    tick,
)
from projects.impedance_instron.cartesian.mechanics import Body
from projects.impedance_instron.cartesian.shoe import Shoe

SUPPORTED_MODES = ("bristle", "implicit_bristle", "regularized")
# These diagnostics were built against the archived initial twelve-point case.
# Explicit overrides permit evaluation of another pinned bundle without changing it.
DEFAULT_BASELINE_MANIFEST = Path(
    os.environ.get(
        "NEWTON_BASELINE12_MANIFEST",
        str(Path(__file__).parents[1] / "impedance_instron" / "baselines" / "baseline12_initial.json"),
    )
)


def sha256_file(path: Path | str) -> str:
    """Compute sha256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def verify_baseline_inputs(
    baseline_dir: Path | str,
    manifest_path: Path | str = DEFAULT_BASELINE_MANIFEST,
) -> dict[str, str]:
    """Verify hashes of baseline directory files against the pinned baseline manifest.

    Args:
        baseline_dir: Directory containing baseline files (e.g. baseline12).
        manifest_path: Pinned manifest path declaring expected sha256 hashes.

    Returns:
        Mapping of verified filenames to their sha256 digests.
    """
    baseline_dir = Path(baseline_dir).resolve()
    manifest_path = Path(manifest_path).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Pinned baseline manifest not found: {manifest_path}")

    with open(manifest_path) as f:
        manifest = json.load(f)

    expected_hashes = manifest.get("files_sha256")
    required = {"trace.npz", "reference.npz", "profile.json", "summary.json", "digital_shoe.json"}
    if not isinstance(expected_hashes, dict) or not required.issubset(expected_hashes):
        raise ValueError(f"Baseline manifest must identify every replay input: {manifest_path}")

    verified_hashes: dict[str, str] = {}
    for filename, expected_hash in expected_hashes.items():
        file_path = (baseline_dir / filename).resolve()
        if not file_path.is_relative_to(baseline_dir):
            raise ValueError("Baseline manifest filenames must remain inside its directory")
        if not file_path.exists():
            raise FileNotFoundError(f"Required baseline file missing: {file_path}")
        actual_hash = sha256_file(file_path)
        if actual_hash != expected_hash:
            raise ValueError(f"Hash mismatch for {filename}: expected {expected_hash}, got {actual_hash}")
        verified_hashes[filename] = actual_hash
    return verified_hashes


def replay_mode(
    shoe: Shoe,
    q: np.ndarray,
    qd: np.ndarray,
    dt: float,
    mode: str,
    chunk_steps: int = 32,
) -> dict[str, Any]:
    """Replay prescribed poses and velocities through Shoe and optional FrictionAdapter.

    For mode 'bristle', the foundation runs without an adapter, exercising the
    exact unchanged default physics path for honest baseline timings.
    For coupled/regularized modes, FrictionAdapter is attached with zero mobility.
    """
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"Unsupported mode {mode!r}; expected one of {SUPPORTED_MODES}")
    if chunk_steps <= 0:
        raise ValueError(f"chunk_steps must be positive, got {chunk_steps}")
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")

    n = len(q)
    device = shoe.device

    shoe.foundation.reset()
    adapter: FrictionAdapter | None = None
    if mode != "bristle":
        zero_mobility = wp.zeros(shoe.foundation.world_count, dtype=wp.spatial_matrix, device=device)
        adapter = FrictionAdapter(shoe.foundation, zero_mobility, mode=mode)

    shoe.foundation._refresh_surround_constants(dt)

    saved_q = wp.array(q, dtype=wp.transform, device=device)
    saved_qd = wp.array(qd, dtype=wp.spatial_vector, device=device)
    clock = wp.zeros(1, dtype=int, device=device)
    wrenches = wp.empty(n, dtype=wp.spatial_vector, device=device)
    compression = wp.empty((n, shoe.foundation.column_count), dtype=float, device=device)

    use_graphs = device.is_cuda and chunk_steps > 1 and n >= chunk_steps

    def step_fn():
        wp.launch(
            stage,
            dim=1,
            inputs=[clock, saved_q, saved_qd, shoe.state.body_q, shoe.state.body_qd],
            device=device,
        )
        shoe.foundation.apply(shoe.state, dt, clear_body_force=True)
        wp.launch(
            store,
            dim=shoe.foundation.column_count,
            inputs=[clock, shoe.state.body_f, shoe.foundation.compression, wrenches, compression],
            device=device,
        )
        wp.launch(tick, dim=1, inputs=[clock], device=device)

    capture_time = 0.0
    chunks = 0
    tail_steps = 0

    if use_graphs:
        t_cap0 = perf_counter()
        with wp.ScopedCapture(device=device) as capture:
            for _ in range(chunk_steps):
                step_fn()
        tail = n % chunk_steps
        tail_graph = None
        if tail > 0:
            with wp.ScopedCapture(device=device) as tail_capture:
                for _ in range(tail):
                    step_fn()
            tail_graph = tail_capture.graph
        capture_time = perf_counter() - t_cap0

        # Reset runtime state before captured graph playback
        shoe.foundation.reset()
        clock.zero_()

        t_rep0 = perf_counter()
        chunks = n // chunk_steps
        for _ in range(chunks):
            wp.capture_launch(capture.graph)
        if tail_graph is not None:
            wp.capture_launch(tail_graph)
            tail_steps = tail
    else:
        shoe.foundation.reset()
        clock.zero_()
        t_rep0 = perf_counter()
        for _ in range(n):
            step_fn()

    # .numpy() performs synchronous copy; no redundant wp.synchronize needed
    w_numpy = wrenches.numpy()
    c_numpy = compression.numpy()
    final_clock = int(clock.numpy()[0])
    eval_time = perf_counter() - t_rep0
    if adapter is not None:
        adapter.detach()

    if final_clock != n:
        raise RuntimeError(f"Replay clock {final_clock} does not match requested length {n}")

    return {
        "mode": mode,
        "uses_adapter": adapter is not None,
        "wrenches": w_numpy,
        "compression": c_numpy,
        "capture_wall_s": capture_time,
        "eval_wall_s": eval_time,
        "chunk_launches": chunks,
        "tail_steps": tail_steps,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the friction leg replay tool."""
    parser = argparse.ArgumentParser(
        description="Replay frozen leg motion to evaluate friction models under prescribed kinematics."
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="Path to sealed baseline directory (e.g. baseline12).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_BASELINE_MANIFEST,
        help="Pinned baseline manifest JSON for input verification.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to output directory (must not exist, will be created).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Warp execution device (default: cuda:0).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum steps to evaluate for smoke / partial testing.",
    )
    parser.add_argument(
        "--modes",
        type=str,
        nargs="+",
        default=list(SUPPORTED_MODES),
        choices=SUPPORTED_MODES,
        help="Friction modes to evaluate (default: all three).",
    )
    parser.add_argument(
        "--forward-sign",
        type=int,
        required=True,
        choices=[1, -1],
        help="Explicit sign factor for forward GRF: F_forward = forward_sign * grf_x.",
    )
    parser.add_argument(
        "--chunk-steps",
        type=int,
        default=32,
        help="Graph capture chunk size (default: 32).",
    )
    return parser.parse_args(argv)


def run_friction_leg_replay(
    baseline_dir: Path | str,
    output_dir: Path | str,
    manifest_path: Path | str = DEFAULT_BASELINE_MANIFEST,
    device: str = "cuda:0",
    max_steps: int | None = None,
    modes: list[str] | tuple[str, ...] = SUPPORTED_MODES,
    forward_sign: int = 1,
    chunk_steps: int = 32,
) -> dict[str, Any]:
    """Execute friction leg motion replay and return evaluation report."""
    baseline_dir = Path(baseline_dir).resolve()
    if not baseline_dir.exists():
        raise FileNotFoundError(f"Baseline directory does not exist: {baseline_dir}")

    manifest_path = Path(manifest_path).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Baseline manifest does not exist: {manifest_path}")

    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    if not modes or len(set(modes)) != len(modes):
        raise ValueError("Select a nonempty, unique set of friction modes")
    if forward_sign not in (1, -1):
        raise ValueError(f"forward_sign must be 1 or -1, got {forward_sign}")
    if chunk_steps <= 0:
        raise ValueError(f"chunk_steps must be positive, got {chunk_steps}")
    if max_steps is not None and max_steps < 2:
        raise ValueError(f"max_steps must be at least two, got {max_steps}")
    for mode in modes:
        if mode not in SUPPORTED_MODES:
            raise ValueError(f"Unsupported mode {mode!r}; expected one of {SUPPORTED_MODES}")

    # 1. Verify baseline inputs against manifest
    baseline_hashes = verify_baseline_inputs(baseline_dir, manifest_path)
    output_dir.mkdir(parents=True)

    # 2. Load inputs
    trace_archive = archive(baseline_dir / "trace.npz")
    reference_path = baseline_dir / "reference.npz"
    reference_archive = archive(reference_path)

    with open(baseline_dir / "profile.json") as f:
        profile = json.load(f)
    with open(baseline_dir / "summary.json") as f:
        summary = json.load(f)

    # Check digital shoe path from summary or local
    shoe_info = summary["shoe"]
    shoe_path = baseline_dir / "digital_shoe.json"
    actual_shoe_hash = sha256_file(shoe_path)
    if actual_shoe_hash != shoe_info["sha256"]:
        raise ValueError(f"Shoe artifact hash mismatch: expected {shoe_info['sha256']}, got {actual_shoe_hash}")

    # 3. Kinematics precomputation on host
    body = Body(
        reference_archive["lengths_m"],
        reference_archive["endpoint_local_m"],
        profile["masses_kg"],
        profile["com_local_m"],
        profile["inertias_kg_m2"],
    )
    static_pitch = float(shoe_info["static_pitch_rad"])
    q_all, qd_all = exact_cpu_poses(body, trace_archive, static_pitch)
    time_all = trace_archive["time_s"]
    dt = float(summary["run"]["actual_dt_s"])

    total_steps = len(q_all)
    is_partial = False
    if max_steps is not None and max_steps < total_steps:
        q = q_all[:max_steps]
        qd = qd_all[:max_steps]
        time_s = time_all[:max_steps]
        is_partial = True
    else:
        q = q_all
        qd = qd_all
        time_s = time_all

    evaluated_steps = len(q)

    # 4. Initialize Shoe
    shoe = Shoe(shoe_path, shoe_info["mount_m"], static_pitch, device=device, friction_model="legacy")

    # 5. Replay each requested mode
    mode_results = {}
    for mode in modes:
        res = replay_mode(shoe, q, qd, dt, mode, chunk_steps=chunk_steps)
        mode_results[mode] = res

    # 6. Extract predicted GRF and normal forces
    mode_grf = {}
    mode_fz = {}
    for mode, res in mode_results.items():
        wrenches = res["wrenches"]
        grf = wrenches[:, [0, 2]].astype(np.float64)
        mode_grf[mode] = grf
        mode_fz[mode] = grf[:, 1]

    # 7. Discrepancy analysis
    first_mode = modes[0]
    base_fz = mode_fz[first_mode]
    max_normal_discrepancies = {}
    for mode in modes[1:]:
        diff = float(np.max(np.abs(mode_fz[mode] - base_fz)))
        max_normal_discrepancies[f"{first_mode}_vs_{mode}"] = diff

    max_compression_discrepancies = {
        mode: float(np.max(np.abs(res["compression"] - mode_results[first_mode]["compression"])))
        for mode, res in mode_results.items()
    }
    normal_unchanged = all(np.array_equal(value, base_fz) for value in mode_fz.values())
    compression_unchanged = all(value == 0.0 for value in max_compression_discrepancies.values())

    bristle_parity_diff = None
    if "bristle" in mode_results and "implicit_bristle" in mode_results:
        bristle_parity_diff = float(np.max(np.abs(mode_grf["bristle"] - mode_grf["implicit_bristle"])))

    regularized_diff = None
    if "bristle" in mode_results and "regularized" in mode_results:
        regularized_diff = float(np.max(np.abs(mode_grf["bristle"][:, 0] - mode_grf["regularized"][:, 0])))

    # 8. Score traces with friction_metrics
    scores = {}
    for mode, grf in mode_grf.items():
        trace_dict = {
            "time_s": time_s,
            "grf_n": grf,
        }
        score = score_friction_trace(
            reference=reference_path,
            trace=trace_dict,
            forward_sign=forward_sign,
            summary=baseline_dir / "summary.json" if not is_partial else None,
        )
        scores[mode] = score

    # 9. Save outputs
    traces_saved = {}
    for mode, res in mode_results.items():
        mode_trace_path = output_dir / f"trace_{mode}.npz"
        np.savez_compressed(
            mode_trace_path,
            time_s=time_s,
            grf_n=mode_grf[mode],
            wrench_newton=res["wrenches"],
            compression_m=res["compression"],
            forward_sign=forward_sign,
        )
        traces_saved[mode] = {
            "path": str(mode_trace_path.name),
            "sha256": sha256_file(mode_trace_path),
        }

    # Record shared physics source digest directly
    source_digest = physics_source_identity()
    if verify_baseline_inputs(baseline_dir, manifest_path) != baseline_hashes:
        raise RuntimeError("Baseline inputs changed during replay")

    report = {
        "schema": "digital_shoe_friction_leg_replay_1",
        "description": "Leg impedance contact replay evaluating friction laws under common kinematics.",
        "baseline_dir": str(baseline_dir),
        "manifest_path": str(manifest_path),
        "baseline_hashes": baseline_hashes,
        "device": str(device),
        "forward_sign": forward_sign,
        "is_partial_smoke": is_partial,
        "steps_evaluated": evaluated_steps,
        "total_source_steps": total_steps,
        "actual_dt_s": dt,
        "modes_evaluated": list(modes),
        "performance": {
            mode: {
                "uses_adapter": res["uses_adapter"],
                "capture_wall_s": res["capture_wall_s"],
                "eval_wall_s": res["eval_wall_s"],
                "chunk_launches": res["chunk_launches"],
                "tail_steps": res["tail_steps"],
            }
            for mode, res in mode_results.items()
        },
        "discrepancies": {
            "normal_unchanged": normal_unchanged,
            "compression_unchanged": compression_unchanged,
            "max_compression_discrepancies_m": max_compression_discrepancies,
            "max_normal_discrepancies_n": max_normal_discrepancies,
            "bristle_vs_implicit_bristle_max_grf_diff_n": bristle_parity_diff,
            "bristle_vs_regularized_max_fx_diff_n": regularized_diff,
        },
        "scores": scores,
        "traces": traces_saved,
        "physics_source_identity": source_digest,
        "evaluation_sources": {
            name: sha256_file(Path(__file__).with_name(name))
            for name in ("friction_leg.py", "friction_metrics.py", "friction_report.py")
        },
        "qualification": "Prescribed-motion diagnostic, not a free-dynamics or independent physical-friction validation. Normal mechanics are unchanged; no controller refit.",
        "timing_scope": "Replay plus final wrench/compression downloads; excludes capture, setup and kinematics preparation.",
    }

    report_path = output_dir / "report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    write_friction_report(report, time_s, mode_grf, reference_archive, output_dir / "report.html")
    if not normal_unchanged or not compression_unchanged:
        raise RuntimeError(f"Normal/compression invariance failed; inspect {report_path}")
    return report


def main(argv: list[str] | None = None) -> None:
    """Run CLI entrypoint for friction leg motion replay."""
    args = parse_args(argv)
    report = run_friction_leg_replay(
        baseline_dir=args.baseline,
        output_dir=args.output,
        manifest_path=args.manifest,
        device=args.device,
        max_steps=args.max_steps,
        modes=args.modes,
        forward_sign=args.forward_sign,
        chunk_steps=args.chunk_steps,
    )

    evaluated_steps = report["steps_evaluated"]
    total_steps = report["total_source_steps"]
    is_partial = report["is_partial_smoke"]
    max_normal_discrepancies = report["discrepancies"]["max_normal_discrepancies_n"]
    bristle_parity_diff = report["discrepancies"]["bristle_vs_implicit_bristle_max_grf_diff_n"]
    regularized_diff = report["discrepancies"]["bristle_vs_regularized_max_fx_diff_n"]

    print(f"Successfully finished leg friction replay to {args.output}")
    print(f"Steps: {evaluated_steps}/{total_steps} (is_partial={is_partial})")
    print(f"Normal discrepancies: {max_normal_discrepancies}")
    if bristle_parity_diff is not None:
        print(f"Bristle vs Implicit Bristle parity max diff: {bristle_parity_diff:.6e} N")
    if regularized_diff is not None:
        print(f"Bristle vs Regularized max Fx diff: {regularized_diff:.6e} N")
    for mode, sc in report["scores"].items():
        comp = sc.get("comparison_metrics", {})
        rmse_val = comp.get("full_horizontal_force_rmse_n", comp.get("horizontal_force_rmse_n", None))
        braking_val = comp.get("braking_impulse_diff_ns", None)
        prop_val = comp.get("propulsive_impulse_diff_ns", None)

        rmse_str = f"{rmse_val:.2f} N" if isinstance(rmse_val, (int, float)) else "N/A"
        braking_str = f"{braking_val:.2f} Ns" if isinstance(braking_val, (int, float)) else "N/A"
        prop_str = f"{prop_val:.2f} Ns" if isinstance(prop_val, (int, float)) else "N/A"

        print(
            f"Mode {mode}: complete={sc.get('complete')} "
            f"RMSE={rmse_str}, "
            f"Braking diff={braking_str}, "
            f"Propulsive diff={prop_str}"
        )


if __name__ == "__main__":
    main()
