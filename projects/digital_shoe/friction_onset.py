# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Initial-state, onset kinematics, and friction-only velocity diagnostic.

Analyzes foot-ground tangential plane velocity against reference kinematics,
decomposes hip translation, leg joint rotation, and foot angular contributions,
and runs frozen-normal consistent-deflection friction diagnostics to isolate
velocity discrepancies from normal load mechanics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from projects.digital_shoe.friction_history import load_history
from projects.digital_shoe.friction_leg import verify_baseline_inputs
from projects.digital_shoe.friction_sweep import FrictionSweep
from projects.impedance_instron.cartesian.mechanics import Body

DEFAULT_BASELINE_DIR = Path("outputs/impedance_instron/baseline12")


def sha256_bytes(data: bytes) -> str:
    """Return hex sha256 digest of bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path | str) -> str:
    """Return hex sha256 digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def decompose_tangential_velocity(
    body: Body,
    q: np.ndarray,
    v: np.ndarray,
    ground_height_m: float = 0.0,
) -> dict[str, Any]:
    """Decompose plane tangential velocity into hip, leg rotation, and foot lever arm terms.

    Args:
        body: Leg mechanics body model.
        q: Generalized coordinates [hip_x, hip_z, thigh, knee, ankle], shape (5,).
        v: Generalized velocities [v_hip_x, v_hip_z, omega_thigh, omega_knee, omega_ankle], shape (5,).
        ground_height_m: Ground plane height [m].

    Returns:
        Dictionary containing position, velocity, and decomposed kinematic terms.
    """
    q_arr = np.asarray(q, dtype=np.float64)
    v_arr = np.asarray(v, dtype=np.float64)
    ankle_pos, ankle_jac, _ = body.point(q_arr, 2, np.zeros(2))
    ang_jac = body.angular_jacobian(2)
    v_ankle = ankle_jac @ v_arr
    omega_foot = float(ang_jac @ v_arr)
    v_hip = v_arr[:2]
    leg_rel_v_x = v_ankle[0] - v_hip[0]
    lever_arm_z = ankle_pos[1] - ground_height_m
    foot_rot_contrib_x = omega_foot * lever_arm_z
    v_plane_x = v_ankle[0] + foot_rot_contrib_x

    return {
        "hip_pos_m": q_arr[:2].tolist(),
        "hip_vel_m_s": v_hip.tolist(),
        "ankle_pos_m": ankle_pos.tolist(),
        "ankle_vel_m_s": v_ankle.tolist(),
        "leg_rel_vel_x_m_s": float(leg_rel_v_x),
        "omega_foot_rad_s": float(omega_foot),
        "lever_arm_z_m": float(lever_arm_z),
        "foot_rot_contrib_x_m_s": float(foot_rot_contrib_x),
        "tangential_plane_vel_x_m_s": float(v_plane_x),
    }


def compute_reference_derived_kinematics(
    reference: dict[str, Any],
    query_times_s: np.ndarray,
    body: Body,
    ground_height_m: float = 0.0,
    tolerance: float = 1e-9,
) -> dict[str, np.ndarray]:
    """Evaluate reference trajectory via CubicHermiteSpline(q, v) without extrapolation.

    Preserves exact kinematic derivative consistency v = dq/dt by constructing a Hermite
    interpolant from reference positions and velocities.

    Args:
        reference: Reference archive dictionary containing 'time_s', 'state', and 'velocity'.
        query_times_s: Strictly increasing 1-D simulation time clock [s].
        body: Leg mechanics body model.
        ground_height_m: Ground plane height [m].
        tolerance: Boundary matching tolerance [s].

    Returns:
        Dictionary containing interpolated q, v, and plane tangential velocity array.
    """
    from scipy.interpolate import CubicHermiteSpline

    ref_t = np.asarray(reference["time_s"], dtype=np.float64)
    ref_q = np.asarray(reference["state"], dtype=np.float64)
    ref_v = np.asarray(reference["velocity"], dtype=np.float64)

    t_queries = np.asarray(query_times_s, dtype=np.float64)
    if len(t_queries) < 2:
        raise ValueError("query_times_s must contain at least two timestamps")
    if np.any(np.diff(t_queries) <= 0.0) or not np.isfinite(t_queries).all():
        raise ValueError("query_times_s must be strictly increasing and finite")

    # Strict query support: ensure no extrapolation outside reference time bounds
    if t_queries[0] < ref_t[0] - tolerance:
        raise ValueError(
            f"query start {t_queries[0]} precedes reference start {ref_t[0]} by more than tolerance {tolerance}"
        )
    if t_queries[-1] > ref_t[-1] + tolerance:
        raise ValueError(
            f"query endpoint {t_queries[-1]} exceeds reference endpoint {ref_t[-1]} by more than tolerance {tolerance}"
        )

    # Clip within numeric boundary to avoid numerical out-of-bounds in spline
    t_eval = np.clip(t_queries, ref_t[0], ref_t[-1])

    spline = CubicHermiteSpline(ref_t, ref_q, ref_v)
    q_sim = spline(t_eval)
    v_sim = spline.derivative()(t_eval)

    n_steps = len(t_queries)
    v_plane_x = np.zeros(n_steps, dtype=np.float32)
    ang_jac = body.angular_jacobian(2)

    for s in range(n_steps):
        q_s = q_sim[s]
        v_s = v_sim[s]
        ankle_pos, ankle_jac, _ = body.point(q_s, 2, np.zeros(2))
        v_ankle = ankle_jac @ v_s
        omega = float(ang_jac @ v_s)
        lever_z = ankle_pos[1] - ground_height_m
        v_plane_x[s] = np.float32(v_ankle[0] + omega * lever_z)

    return {
        "q": q_sim,
        "v": v_sim,
        "plane_vel_x": v_plane_x,
    }


@dataclass
class OnsetDiagnosticResult:
    """Structured diagnostic summary."""

    initial_error_q: list[float]
    initial_error_v: list[float]
    decompositions_ms: dict[str, dict[str, Any]]
    normal_sha256: str
    scores_actual: dict[str, float]
    scores_reference: dict[str, float]
    scores_candidate_actual: dict[str, float] | None
    scores_candidate_reference: dict[str, float] | None


def run_friction_onset_diagnostic(
    history_path: Path | str,
    baseline_dir: Path | str,
    output_dir: Path | str,
    candidate_path: Path | str | None = None,
) -> dict[str, Any]:
    """Execute initial-state audit, velocity decomposition, and friction-only sweep.

    Args:
        history_path: Path to frozen history_exact.npz.
        baseline_dir: Directory containing baseline reference.npz, profile.json, and trace.npz.
        output_dir: Directory to save diagnostic json and observer forces npz (must not exist).
        candidate_path: Optional path to candidate parameters JSON (must exist if provided).

    Returns:
        Summary dictionary with all diagnostic metrics.
    """
    history_file = Path(history_path).resolve()
    base_dir = Path(baseline_dir).resolve()
    out_dir = Path(output_dir).resolve()

    # Reject existing output directory before starting work
    if out_dir.exists():
        raise FileExistsError(f"Output directory already exists: {out_dir}")

    # Verify baseline files against pinned manifest
    baseline_hashes = verify_baseline_inputs(base_dir)

    # Load frozen history and verify completion provenance
    fhist = load_history(history_file)
    if not fhist.provenance.get("complete", False):
        raise ValueError(f"History cache at {history_file} is not marked complete in provenance")

    if fhist.provenance.get("baseline_hashes") != baseline_hashes:
        raise ValueError("History cache does not identify the same sealed baseline inputs")
    history_hash_before = sha256_file(history_file)

    # Capture normal_n before execution
    normal_before = np.array(fhist.normal_n, copy=True)
    normal_before_hash = sha256_bytes(normal_before.tobytes())

    ref_file = base_dir / "reference.npz"
    prof_file = base_dir / "profile.json"
    trace_file = base_dir / "trace.npz"

    with np.load(ref_file, allow_pickle=False) as rf:
        ref_data = dict(rf)
    with np.load(trace_file, allow_pickle=False) as tr:
        trace_data = dict(tr)
    with open(prof_file) as pf:
        prof_data = json.load(pf)

    # Compare actual trace clock to history exact clock
    t_sim = np.asarray(fhist.time_s, dtype=np.float64)
    t_trace = np.asarray(trace_data["time_s"], dtype=np.float64)
    if len(t_sim) != len(t_trace) or not np.array_equal(t_sim, t_trace):
        raise ValueError(f"Clock mismatch between trace ({len(t_trace)} steps) and history ({len(t_sim)} steps)")

    dt_sim = float(t_sim[1] - t_sim[0])

    body = Body(
        ref_data["lengths_m"],
        ref_data["endpoint_local_m"],
        prof_data["masses_kg"],
        prof_data["com_local_m"],
        prof_data["inertias_kg_m2"],
    )

    # 1. Initial State Parity Check
    q0_ref = np.asarray(ref_data["state"][0], dtype=np.float64)
    v0_ref = np.asarray(ref_data["velocity"][0], dtype=np.float64)
    q0_act = np.asarray(trace_data["state"][0], dtype=np.float64)
    v0_act = np.asarray(trace_data["velocity"][0], dtype=np.float64)

    diff_q0 = q0_act - q0_ref
    diff_v0 = v0_act - v0_ref
    is_q0_exact = bool(np.array_equal(q0_act, q0_ref))
    is_v0_exact = bool(np.array_equal(v0_act, v0_ref))

    # 2. Reference Spline Evaluation & Derived Velocity
    ref_derived = compute_reference_derived_kinematics(ref_data, t_sim, body, ground_height_m=0.0)
    ref_plane_x = ref_derived["plane_vel_x"]

    # 3. Time point decompositions at 0, 20, 30, 40 ms
    check_times_ms = [0.0, 20.0, 30.0, 40.0]
    decompositions = {}

    for t_ms in check_times_ms:
        target_s = t_ms * 1e-3
        # Strict lookup on matching trace clock
        idx_act = int(np.argmin(np.abs(t_sim - target_s)))
        idx_ref = int(np.argmin(np.abs(ref_data["time_s"] - target_s)))

        act_dec = decompose_tangential_velocity(body, trace_data["state"][idx_act], trace_data["velocity"][idx_act])
        ref_dec = decompose_tangential_velocity(body, ref_data["state"][idx_ref], ref_data["velocity"][idx_ref])

        diff_v_plane = act_dec["tangential_plane_vel_x_m_s"] - ref_dec["tangential_plane_vel_x_m_s"]
        diff_v_hip = act_dec["hip_vel_m_s"][0] - ref_dec["hip_vel_m_s"][0]
        diff_v_leg_rot = act_dec["leg_rel_vel_x_m_s"] - ref_dec["leg_rel_vel_x_m_s"]
        diff_v_foot_rot = act_dec["foot_rot_contrib_x_m_s"] - ref_dec["foot_rot_contrib_x_m_s"]

        decompositions[f"{t_ms:.1f}ms"] = {
            "time_sim_s": float(t_sim[idx_act]),
            "time_ref_s": float(ref_data["time_s"][idx_ref]),
            "actual": act_dec,
            "reference": ref_dec,
            "differences": {
                "delta_tangential_plane_vel_x_m_s": float(diff_v_plane),
                "delta_hip_vel_x_m_s": float(diff_v_hip),
                "delta_leg_rot_vel_x_m_s": float(diff_v_leg_rot),
                "delta_foot_rot_contrib_x_m_s": float(diff_v_foot_rot),
            },
        }

    # 4. Consistent-Deflection Friction Replay
    arrays_actual = {
        "time_s": fhist.time_s,
        "position_xy": fhist.position_xy,
        "velocity_xy": fhist.velocity_xy,
        "nominal_velocity_xy": fhist.nominal_velocity_xy,
        "normal_n": fhist.normal_n,
        "baseline_kt_n_m": fhist.baseline_kt_n_m,
        "baseline_kv_ns_m": fhist.baseline_kv_ns_m,
        "measured_time_s": fhist.measured_time_s,
        "measured_force_n": fhist.measured_force_n,
    }

    arrays_ref = dict(arrays_actual)
    ref_vel_xy = np.zeros_like(fhist.velocity_xy)
    ref_vel_xy[:, :, 0] = ref_plane_x[:, None]
    ref_vel_xy[:, :, 1] = 0.0
    arrays_ref["velocity_xy"] = ref_vel_xy

    # Baseline original parameter row: method=1 (deflection), mu=0.8, kt=1.0, kv=1.0, etc.
    param_original = np.array([[1.0, 0.8, 1.0, 1.0, 0.2, 0.0005, 0.0]], dtype=np.float32)

    sweep_act = FrictionSweep(arrays_actual, batch_size=2, use_graph=False)
    sweep_ref = FrictionSweep(arrays_ref, batch_size=2, use_graph=False)

    scores_act, curves_act = sweep_act.evaluate(param_original, curves=True)
    scores_ref, curves_ref = sweep_ref.evaluate(param_original, curves=True)

    score_keys = [
        "loss",
        "rmse_n",
        "braking_impulse_ns",
        "propulsive_impulse_ns",
        "braking_peak_n",
        "propulsive_peak_n",
        "positive_energy_residual_j",
        "max_deflection_m",
        "cone_excess_n",
        "work_j",
    ]
    scores_actual_dict = {k: float(v) for k, v in zip(score_keys, scores_act[0], strict=True)}
    scores_ref_dict = {k: float(v) for k, v in zip(score_keys, scores_ref[0], strict=True)}

    # Optional Candidate Evaluation
    scores_cand_act_dict = None
    scores_cand_ref_dict = None
    curves_cand_act = None
    curves_cand_ref = None

    if candidate_path is not None:
        cand_p = Path(candidate_path).resolve()
        if not cand_p.exists():
            raise FileNotFoundError(f"Declared candidate path does not exist: {cand_p}")
        with open(cand_p) as cf:
            cand_info = json.load(cf)
        cp = cand_info.get("parameters", cand_info)
        method = int(cp["method"])
        yield_width = float(cp.get("yield_width", 0.0))
        if method != 1:
            raise ValueError(f"Candidate method must be 1 (consistent deflection), got {method}")
        if yield_width != 0.0:
            raise ValueError(f"Candidate yield_width must be 0.0 (no unphysical creep), got {yield_width}")

        param_cand = np.array(
            [
                [
                    float(cp["method"]),
                    float(cp["mu"]),
                    float(cp["kt_scale"]),
                    float(cp["kv_scale"]),
                    float(cp["viscous_ratio"]),
                    float(cp["release_dwell_s"]),
                    yield_width,
                ]
            ],
            dtype=np.float32,
        )
        sc_c_act, curves_cand_act = sweep_act.evaluate(param_cand, curves=True)
        sc_c_ref, curves_cand_ref = sweep_ref.evaluate(param_cand, curves=True)
        scores_cand_act_dict = {k: float(v) for k, v in zip(score_keys, sc_c_act[0], strict=True)}
        scores_cand_ref_dict = {k: float(v) for k, v in zip(score_keys, sc_c_ref[0], strict=True)}

    # Verify normal array non-mutation
    normal_after = np.array(fhist.normal_n, copy=True)
    normal_after_hash = sha256_bytes(normal_after.tobytes())
    if normal_before_hash != normal_after_hash or not np.array_equal(normal_before, normal_after):
        raise AssertionError("normal_n array was mutated during diagnostic evaluation")

    # 5. Build and Save Interface Arrays for Observation
    # Observer input: [Fx, sum(normal_n)] on simulation clock
    normal_force_z = np.sum(fhist.normal_n, axis=1).astype(np.float32)
    actual_fx = curves_act[0, :, 0].astype(np.float32)
    ref_vel_fx = curves_ref[0, :, 0].astype(np.float32)

    actual_forces = np.column_stack([actual_fx, normal_force_z])
    ref_velocity_forces = np.column_stack([ref_vel_fx, normal_force_z])

    # Compute actual early difference data over the initial stance interval (0 to 50ms)
    early_mask = t_sim <= 0.050
    early_force_diff_x = actual_fx[early_mask] - ref_vel_fx[early_mask]
    max_early_force_diff_n = float(np.max(np.abs(early_force_diff_x)))

    # Create output directory now that all computations succeeded
    out_dir.mkdir(parents=True, exist_ok=False)

    save_npz_data = {
        "time_s": t_sim,
        "actual_forces": actual_forces,
        "ref_velocity_forces": ref_velocity_forces,
        "actual_plane_vel_x": np.asarray(fhist.velocity_xy[:, 0, 0], dtype=np.float32),
        "ref_plane_vel_x": ref_plane_x,
        "normal_force_z": normal_force_z,
    }

    if curves_cand_act is not None:
        cand_act_fx = curves_cand_act[0, :, 0].astype(np.float32)
        cand_ref_fx = curves_cand_ref[0, :, 0].astype(np.float32)
        save_npz_data["candidate_actual_forces"] = np.column_stack([cand_act_fx, normal_force_z])
        save_npz_data["candidate_ref_velocity_forces"] = np.column_stack([cand_ref_fx, normal_force_z])

    if verify_baseline_inputs(base_dir) != baseline_hashes or sha256_file(history_file) != history_hash_before:
        raise RuntimeError("Baseline or frozen history changed during onset diagnostic")
    np.savez_compressed(out_dir / "curves_forces.npz", **save_npz_data)

    summary = {
        "provenance": {
            "history_path": str(history_file),
            "history_sha256": sha256_file(history_file),
            "normal_n_sha256_before": normal_before_hash,
            "normal_n_sha256_after": normal_after_hash,
            "normal_n_is_immutable": True,
            "baseline_dir": str(base_dir),
            "baseline_verified_hashes": baseline_hashes,
        },
        "clocks": {
            "simulation_step_count": len(t_sim),
            "simulation_dt_s": dt_sim,
            "simulation_time_span_s": [float(t_sim[0]), float(t_sim[-1])],
            "reference_step_count": len(ref_data["time_s"]),
            "reference_dt_s": float(ref_data["time_s"][1] - ref_data["time_s"][0]),
            "reference_time_span_s": [float(ref_data["time_s"][0]), float(ref_data["time_s"][-1])],
        },
        "initial_state_parity": {
            "error_q0": diff_q0.tolist(),
            "error_v0": diff_v0.tolist(),
            "is_q0_exact_match": is_q0_exact,
            "is_v0_exact_match": is_v0_exact,
        },
        "decompositions": decompositions,
        "early_onset_comparison": {
            "time_window_s": [0.0, 0.050],
            "max_abs_force_difference_n": max_early_force_diff_n,
            "force_difference_at_20ms_n": float(
                actual_fx[int(np.argmin(np.abs(t_sim - 0.020)))] - ref_vel_fx[int(np.argmin(np.abs(t_sim - 0.020)))]
            ),
            "force_difference_at_30ms_n": float(
                actual_fx[int(np.argmin(np.abs(t_sim - 0.030)))] - ref_vel_fx[int(np.argmin(np.abs(t_sim - 0.030)))]
            ),
            "force_difference_at_40ms_n": float(
                actual_fx[int(np.argmin(np.abs(t_sim - 0.040)))] - ref_vel_fx[int(np.argmin(np.abs(t_sim - 0.040)))]
            ),
        },
        "friction_diagnostic_scores": {
            "baseline_actual_velocity": scores_actual_dict,
            "baseline_reference_velocity": scores_ref_dict,
            "candidate_actual_velocity": scores_cand_act_dict,
            "candidate_reference_velocity": scores_cand_ref_dict,
        },
    }

    with open(out_dir / "onset_diagnostic.json", "w") as jf:
        json.dump(summary, jf, indent=2, allow_nan=False)

    return summary


def main(argv: list[str] | None = None) -> None:
    """Run CLI for onset kinematics and friction diagnostic."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--history",
        type=Path,
        default=Path("outputs/friction_identification/history_exact.npz"),
        help="Path to frozen history_exact.npz",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE_DIR,
        help="Path to sealed baseline directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save diagnostic outputs (must not already exist)",
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        default=None,
        help="Path to optional candidate parameters JSON (e.g. candidate_effective_constant.json)",
    )

    args = parser.parse_args(argv)
    run_friction_onset_diagnostic(
        history_path=args.history,
        baseline_dir=args.baseline,
        output_dir=args.output_dir,
        candidate_path=args.candidate,
    )


if __name__ == "__main__":
    main()
