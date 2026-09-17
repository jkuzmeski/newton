# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reusable read-only frozen normal and kinematic cache for friction parameter sweeps.

Preserves the original bristle foundation simulation under prescribed baseline kinematics
with zero modification to normal contact mechanics, geometry, or material properties.
Records preintegration plane kinematics, normal reactions, and baseline friction forces
in chunked GPU graph captures with a single bulk download pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import warp as wp

from projects.digital_shoe.friction_leg import DEFAULT_BASELINE_MANIFEST, verify_baseline_inputs
from projects.digital_shoe.provenance import physics_source_identity
from projects.digital_shoe.runtime import contact_kinematics
from projects.impedance_instron.cartesian.gpu.contact_replay import (
    archive,
    exact_cpu_poses,
    stage,
    tick,
)
from projects.impedance_instron.cartesian.mechanics import Body
from projects.impedance_instron.cartesian.shoe import Shoe

wp.set_module_options({"enable_backward": False})


def sha256_file(path: Path | str) -> str:
    """Compute sha256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def normal_physics_source_identity() -> str:
    """Hash only the mechanics files determining normal contact reaction and kinematics."""
    base = Path(__file__).parent
    digest = hashlib.sha256()
    for name in ("runtime.py", "material.py", "contact.py"):
        path = base / name
        if path.exists():
            digest.update(name.encode("utf-8") + b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


@wp.kernel
def record_contact_preintegration_step(
    clock: wp.array[int],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    anchor_local: wp.array[wp.vec3],
    ground_height: float,
    ground_force: wp.array[wp.vec3],
    contact_point: wp.array[wp.vec3],
    position_xy: wp.array3d[float],
    velocity_xy: wp.array3d[float],
    nominal_velocity_xy: wp.array3d[float],
    normal_n: wp.array2d[float],
    baseline_force_xy: wp.array3d[float],
    com_world: wp.array2d[float],
    out_contact_point: wp.array3d[float],
):
    """Record contact kinematics, normal reactions, and baseline forces before clock advance."""
    c = wp.tid()
    s = clock[0]

    q = body_q[0]
    qd = body_qd[0]
    com_loc = body_com[0]
    local_anchor = anchor_local[c]

    # Kinematics at the projected ground plane point (ground_plane = 1)
    _pt_plane, com_w, vel_plane, _gap_plane = contact_kinematics(q, qd, com_loc, local_anchor, ground_height, 1)

    # Kinematics at nominal undeformed anchor (ground_plane = 0)
    _pt_nom, _com_nom, vel_nom, _gap_nom = contact_kinematics(q, qd, com_loc, local_anchor, ground_height, 0)

    gf = ground_force[c]
    cp = contact_point[c]

    # Reuse the actual force-point coordinates instead of recomputing them.
    position_xy[s, c, 0] = cp[0]
    position_xy[s, c, 1] = cp[1]

    velocity_xy[s, c, 0] = vel_plane[0]
    velocity_xy[s, c, 1] = vel_plane[1]

    nominal_velocity_xy[s, c, 0] = vel_nom[0]
    nominal_velocity_xy[s, c, 1] = vel_nom[1]

    normal_n[s, c] = gf[2]

    baseline_force_xy[s, c, 0] = gf[0]
    baseline_force_xy[s, c, 1] = gf[1]

    out_contact_point[s, c, 0] = cp[0]
    out_contact_point[s, c, 1] = cp[1]
    out_contact_point[s, c, 2] = cp[2]

    if c == 0:
        com_world[s, 0] = com_w[0]
        com_world[s, 1] = com_w[1]
        com_world[s, 2] = com_w[2]


@dataclass(frozen=True)
class FrictionHistory:
    """Read-only container holding frozen normal contact kinematics and baseline forces."""

    time_s: np.ndarray  # [T]
    position_xy: np.ndarray  # [T, C, 2]
    velocity_xy: np.ndarray  # [T, C, 2]
    nominal_velocity_xy: np.ndarray  # [T, C, 2]
    normal_n: np.ndarray  # [T, C]
    baseline_force_xy: np.ndarray  # [T, C, 2]
    area_m2: np.ndarray  # [C]
    baseline_kt_n_m: np.ndarray  # [C]
    baseline_kv_ns_m: np.ndarray  # [C]
    settings: dict[str, Any]
    measured_time_s: np.ndarray  # [R]
    measured_force_n: np.ndarray  # [R, 2]
    stance_metadata: dict[str, Any]
    provenance: dict[str, Any]
    COM_world: np.ndarray | None = None  # [T, 3]
    contact_point: np.ndarray | None = None  # [T, C, 3]

    def __post_init__(self):
        """Set numpy array write flags to read-only."""
        for field_name in (
            "time_s",
            "position_xy",
            "velocity_xy",
            "nominal_velocity_xy",
            "normal_n",
            "baseline_force_xy",
            "area_m2",
            "baseline_kt_n_m",
            "baseline_kv_ns_m",
            "measured_time_s",
            "measured_force_n",
            "COM_world",
            "contact_point",
        ):
            arr = getattr(self, field_name)
            if isinstance(arr, np.ndarray):
                arr.setflags(write=False)


def build_history(
    baseline_dir: Path | str,
    output_path: Path | str,
    device: str = "cuda:0",
    chunk_steps: int = 32,
    manifest_path: Path | str = DEFAULT_BASELINE_MANIFEST,
    max_steps: int | None = None,
) -> FrictionHistory:
    """Build frozen normal and kinematic contact cache from verified baseline replay.

    Args:
        baseline_dir: Path to directory containing baseline inputs (baseline12).
        output_path: Path to target file (.npz) or directory for output cache.
        device: Warp execution device ('cuda:0' or 'cpu').
        chunk_steps: Steps per captured CUDA graph chunk (default: 32).
        manifest_path: Path to pinned baseline manifest JSON.
        max_steps: Optional step limit for partial/synthetic testing (must be None for complete cache).

    Returns:
        FrictionHistory instance with read-only arrays.
    """
    baseline_dir = Path(baseline_dir).resolve()
    if not baseline_dir.exists():
        raise FileNotFoundError(f"Baseline directory not found: {baseline_dir}")

    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise FileExistsError(f"Output path already exists: {output_path}")

    if chunk_steps <= 0:
        raise ValueError(f"chunk_steps must be positive, got {chunk_steps}")
    if max_steps is not None and max_steps <= 0:
        raise ValueError(f"max_steps must be positive, got {max_steps}")

    # 1. Verify inputs
    verified_hashes = verify_baseline_inputs(baseline_dir, manifest_path)

    # 2. Load inputs
    trace_archive = archive(baseline_dir / "trace.npz")
    ref_archive = archive(baseline_dir / "reference.npz")
    with open(baseline_dir / "profile.json", encoding="utf-8") as f:
        profile = json.load(f)
    with open(baseline_dir / "summary.json", encoding="utf-8") as f:
        summary = json.load(f)

    shoe_info = summary["shoe"]
    shoe_path = baseline_dir / "digital_shoe.json"
    actual_shoe_hash = sha256_file(shoe_path)
    if actual_shoe_hash != shoe_info["sha256"]:
        raise ValueError(f"Shoe artifact hash mismatch: expected {shoe_info['sha256']}, got {actual_shoe_hash}")

    body = Body(
        ref_archive["lengths_m"],
        ref_archive["endpoint_local_m"],
        profile["masses_kg"],
        profile["com_local_m"],
        profile["inertias_kg_m2"],
    )
    static_pitch = float(shoe_info["static_pitch_rad"])
    q_all, qd_all = exact_cpu_poses(body, trace_archive, static_pitch)
    time_all = trace_archive["time_s"]
    dt = float(summary["run"]["actual_dt_s"])

    total_steps = len(q_all)
    if max_steps is not None and max_steps < total_steps:
        q = q_all[:max_steps]
        qd = qd_all[:max_steps]
        time_s = time_all[:max_steps]
        is_complete = False
    else:
        q = q_all
        qd = qd_all
        time_s = time_all
        is_complete = True

    n = len(q)
    shoe = Shoe(shoe_path, shoe_info["mount_m"], static_pitch, device=device)
    wp_device = shoe.device
    C = shoe.foundation.column_count
    ground_height = float(shoe.foundation.ground_height_m or 0.0)

    # Refresh foundation surround constants
    shoe.foundation.reset()
    shoe.foundation._refresh_surround_constants(dt)

    # Allocate replay and history buffers on device
    saved_q = wp.array(q, dtype=wp.transform, device=wp_device)
    saved_qd = wp.array(qd, dtype=wp.spatial_vector, device=wp_device)
    clock = wp.zeros(1, dtype=int, device=wp_device)

    dev_pos_xy = wp.empty((n, C, 2), dtype=float, device=wp_device)
    dev_vel_xy = wp.empty((n, C, 2), dtype=float, device=wp_device)
    dev_nom_vel_xy = wp.empty((n, C, 2), dtype=float, device=wp_device)
    dev_normal_n = wp.empty((n, C), dtype=float, device=wp_device)
    dev_baseline_force_xy = wp.empty((n, C, 2), dtype=float, device=wp_device)
    dev_com_world = wp.empty((n, 3), dtype=float, device=wp_device)
    dev_contact_point = wp.empty((n, C, 3), dtype=float, device=wp_device)

    def step_fn():
        wp.launch(
            stage,
            dim=1,
            inputs=[clock, saved_q, saved_qd, shoe.state.body_q, shoe.state.body_qd],
            device=wp_device,
        )
        shoe.foundation.apply(shoe.state, dt, clear_body_force=True)
        wp.launch(
            record_contact_preintegration_step,
            dim=C,
            inputs=[
                clock,
                shoe.state.body_q,
                shoe.state.body_qd,
                shoe.foundation.body_com,
                shoe.foundation.anchor_local,
                ground_height,
                shoe.foundation.ground_force,
                shoe.foundation.contact_point,
                dev_pos_xy,
                dev_vel_xy,
                dev_nom_vel_xy,
                dev_normal_n,
                dev_baseline_force_xy,
                dev_com_world,
                dev_contact_point,
            ],
            device=wp_device,
        )
        wp.launch(tick, dim=1, inputs=[clock], device=wp_device)

    use_graphs = wp_device.is_cuda and chunk_steps > 1 and n >= chunk_steps
    capture_time = 0.0

    if use_graphs:
        t_cap0 = perf_counter()
        with wp.ScopedCapture(device=wp_device) as capture:
            for _ in range(chunk_steps):
                step_fn()
        tail = n % chunk_steps
        tail_graph = None
        if tail > 0:
            with wp.ScopedCapture(device=wp_device) as tail_capture:
                for _ in range(tail):
                    step_fn()
            tail_graph = tail_capture.graph
        capture_time = perf_counter() - t_cap0

        shoe.foundation.reset()
        clock.zero_()

        chunks = n // chunk_steps
        for _ in range(chunks):
            wp.capture_launch(capture.graph)
        if tail_graph is not None:
            wp.capture_launch(tail_graph)
    else:
        shoe.foundation.reset()
        clock.zero_()
        for _ in range(n):
            step_fn()

    # Bulk download once via .numpy() (synchronous copy, no redundant wp.synchronize)
    host_pos_xy = dev_pos_xy.numpy()
    host_vel_xy = dev_vel_xy.numpy()
    host_nom_vel_xy = dev_nom_vel_xy.numpy()
    host_normal_n = dev_normal_n.numpy()
    host_baseline_force_xy = dev_baseline_force_xy.numpy()
    host_com_world = dev_com_world.numpy()
    host_contact_point = dev_contact_point.numpy()
    final_clock = int(clock.numpy()[0])

    if final_clock != n:
        raise RuntimeError(f"Replay clock {final_clock} did not match expected step count {n}")

    # Extract foundation configuration constants
    foundation_params = shoe.foundation.world_params.numpy()[0]
    # FoundationParams fields: g_eq, alpha, g_eq2, alpha2, beta, one_minus_two_poisson,
    # tau_s, overstress, inv_h2, stretch_floor, normal_damping, friction_kt, friction_kv,
    # friction_viscous_ratio, friction_release_dwell_s, mu
    settings = {
        "mu": float(foundation_params[15]),
        "viscous_ratio": float(foundation_params[13]),
        "dwell_s": float(foundation_params[14]),
        "dt_s": dt,
        "ground_height_m": ground_height,
    }

    area_m2 = shoe.foundation.area.numpy().copy()
    baseline_kt_n_m = shoe.foundation.friction_kt.numpy().copy()
    baseline_kv_ns_m = shoe.foundation.friction_kv.numpy().copy()

    # Reference measured data and stance metadata
    measured_time_s = np.asarray(ref_archive["grf_time_s"], dtype=np.float64)
    measured_force_n = np.asarray(ref_archive["grf_target_n"], dtype=np.float64)

    normal_threshold_n = 50.0
    measured_up = measured_force_n[:, 1]
    normal_mask = measured_up >= normal_threshold_n
    intervals: list[list[float]] = []
    diffs = np.diff(normal_mask.astype(np.int32))
    starts = np.where(diffs == 1)[0] + 1
    if normal_mask[0]:
        starts = np.r_[0, starts]
    ends = np.where(diffs == -1)[0]
    if normal_mask[-1]:
        ends = np.r_[ends, len(normal_mask) - 1]
    for s_idx, e_idx in zip(starts, ends, strict=True):
        intervals.append([float(measured_time_s[s_idx]), float(measured_time_s[e_idx])])

    stance_metadata = {
        "normal_threshold_n": normal_threshold_n,
        "active_sample_count": int(np.count_nonzero(normal_mask)),
        "contiguous_interval_count": len(intervals),
        "stance_intervals_s": intervals,
    }

    provenance = {
        "complete": is_complete,
        "step_count": n,
        "total_source_steps": total_steps,
        "baseline_dir": str(baseline_dir),
        "baseline_hashes": verified_hashes,
        "physics_source_identity": physics_source_identity(),
        "normal_physics_source_identity": normal_physics_source_identity(),
        "manifest_path": str(manifest_path),
        "device": device,
        "chunk_steps": chunk_steps,
        "capture_wall_s": capture_time,
        "recorder_source_sha256": sha256_file(Path(__file__)),
    }

    history = FrictionHistory(
        time_s=np.asarray(time_s, dtype=np.float64),
        position_xy=host_pos_xy,
        velocity_xy=host_vel_xy,
        nominal_velocity_xy=host_nom_vel_xy,
        normal_n=host_normal_n,
        baseline_force_xy=host_baseline_force_xy,
        area_m2=area_m2,
        baseline_kt_n_m=baseline_kt_n_m,
        baseline_kv_ns_m=baseline_kv_ns_m,
        settings=settings,
        measured_time_s=measured_time_s,
        measured_force_n=measured_force_n,
        stance_metadata=stance_metadata,
        provenance=provenance,
        COM_world=host_com_world,
        contact_point=host_contact_point,
    )

    # Save cache
    if output_path.suffix == ".npz":
        npz_target = output_path
        json_target = output_path.with_suffix(".json")
    else:
        output_path.mkdir(parents=True, exist_ok=False)
        npz_target = output_path / "history.npz"
        json_target = output_path / "metadata.json"

    metadata_payload = {
        "settings": settings,
        "stance_metadata": stance_metadata,
        "provenance": provenance,
    }

    np.savez_compressed(
        npz_target,
        time_s=history.time_s,
        position_xy=history.position_xy,
        velocity_xy=history.velocity_xy,
        nominal_velocity_xy=history.nominal_velocity_xy,
        normal_n=history.normal_n,
        baseline_force_xy=history.baseline_force_xy,
        area_m2=history.area_m2,
        baseline_kt_n_m=history.baseline_kt_n_m,
        baseline_kv_ns_m=history.baseline_kv_ns_m,
        measured_time_s=history.measured_time_s,
        measured_force_n=history.measured_force_n,
        COM_world=history.COM_world,
        contact_point=history.contact_point,
        metadata_json=json.dumps(metadata_payload),
    )

    with open(json_target, "w", encoding="utf-8") as f:
        json.dump(metadata_payload, f, indent=2)

    return history


def load_history(path: Path | str) -> FrictionHistory:
    """Load frozen normal and kinematic friction cache from NPZ or directory.

    Args:
        path: Path to .npz file or directory containing history.npz.

    Returns:
        FrictionHistory instance with read-only arrays.
    """
    path = Path(path).resolve()
    if path.is_dir():
        npz_path = path / "history.npz"
        json_path = path / "metadata.json"
    elif path.suffix == ".npz":
        npz_path = path
        json_path = path.with_suffix(".json")
    else:
        raise ValueError(f"Path must be an .npz file or directory containing history.npz: {path}")

    if not npz_path.exists():
        raise FileNotFoundError(f"Cache file not found: {npz_path}")

    with np.load(npz_path, allow_pickle=False) as arch:
        data = {k: arch[k] for k in arch.files}

    if "metadata_json" in data:
        metadata = json.loads(str(data["metadata_json"]))
    elif json_path.exists():
        with open(json_path, encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        raise ValueError("Cache metadata not found in archive or companion JSON")

    return FrictionHistory(
        time_s=data["time_s"],
        position_xy=data["position_xy"],
        velocity_xy=data["velocity_xy"],
        nominal_velocity_xy=data["nominal_velocity_xy"],
        normal_n=data["normal_n"],
        baseline_force_xy=data["baseline_force_xy"],
        area_m2=data["area_m2"],
        baseline_kt_n_m=data["baseline_kt_n_m"],
        baseline_kv_ns_m=data["baseline_kv_ns_m"],
        settings=metadata["settings"],
        measured_time_s=data["measured_time_s"],
        measured_force_n=data["measured_force_n"],
        stance_metadata=metadata["stance_metadata"],
        provenance=metadata["provenance"],
        COM_world=data.get("COM_world"),
        contact_point=data.get("contact_point"),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for building friction history cache."""
    parser = argparse.ArgumentParser(
        description="Build frozen normal/kinematic friction cache from verified baseline replay."
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
        help="Output path (.npz file or new directory). Must not exist.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Warp execution device (default: cuda:0).",
    )
    parser.add_argument(
        "--chunk-steps",
        type=int,
        default=32,
        help="Graph capture chunk size (default: 32).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Step limit for partial/smoke testing.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for building friction history cache."""
    args = parse_args(argv)
    print(f"Building friction history cache from {args.baseline}...")
    t0 = perf_counter()
    history = build_history(
        baseline_dir=args.baseline,
        output_path=args.output,
        device=args.device,
        chunk_steps=args.chunk_steps,
        manifest_path=args.manifest,
        max_steps=args.max_steps,
    )
    elapsed = perf_counter() - t0
    print(
        f"Built cache with {len(history.time_s)} steps, {len(history.area_m2)} columns "
        f"in {elapsed:.2f} s. Complete: {history.provenance['complete']}."
    )


if __name__ == "__main__":
    main()
