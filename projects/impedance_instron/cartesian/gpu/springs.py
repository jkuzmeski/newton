# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""GPU saved-contact spring visualization replay without leg integration.

Replays CUDA contact for saved simulated states, staging kinematic ankle poses
directly on the GPU without evaluating the Cartesian controller or advancing
dynamics. Validates agreement against saved CPU/GPU forces, moments, compression
fractions, and passive cap counts.
"""

from __future__ import annotations

import math
import sys
import time
from typing import Any

import numpy as np
import warp as wp

from projects.digital_shoe.rendering import carried_column_segment
from projects.impedance_instron.cartesian.gpu.mechanics import (
    Params,
    Vec5,
    ankle,
    foot_angle,
)

# Pose reconstruction and replay module options
wp.set_module_options({"enable_backward": False, "fuse_fp": False})


def module_options() -> dict[str, Any]:
    """Return effective compiler and numerical options for this pose module."""
    opts = wp.get_module_options(sys.modules[__name__])
    return {key: opts.get(key) for key in ("fuse_fp", "fast_math", "mode", "optimization_level")}


@wp.kernel
def kinematic_ankle_poses(
    p: Params,
    pitch: wp.float64,
    states: wp.array[Vec5],
    velocities: wp.array[Vec5],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    """Compute carried carrier transform and spatial velocity on GPU."""
    w = wp.tid()
    q = states[w]
    v = velocities[w]
    position, jx, jz = ankle(q, p)
    angle = (foot_angle(q) - pitch) / wp.float64(2.0)
    body_q[w] = wp.transform(
        wp.vec3(wp.float32(position[0]), 0.0, wp.float32(position[1])),
        wp.quat(0.0, wp.float32(-wp.sin(angle)), 0.0, wp.float32(wp.cos(angle))),
    )
    body_qd[w] = wp.spatial_vector(
        wp.float32(wp.dot(jx, v)),
        0.0,
        wp.float32(wp.dot(jz, v)),
        0.0,
        wp.float32(-((v[2] + v[3]) + v[4])),
        0.0,
    )


@wp.kernel
def render_carried_endpoints(
    body_q: wp.array[wp.transform],
    frame_indices: wp.array[int],
    anchors: wp.array[wp.vec3],
    rest: wp.array[float],
    compression_history: wp.array2d[float],
    driven: wp.array[int],
    ground_height: float,
    bottoms_out: wp.array2d[wp.vec3],
    tops_out: wp.array2d[wp.vec3],
):
    """Reconstruct carried column endpoints for sampled display frames."""
    f, c = wp.tid()
    trace_row = frame_indices[f]
    bottom, top = carried_column_segment(
        body_q[trace_row],
        anchors[c],
        rest[c],
        compression_history[trace_row, c],
        driven[c],
        ground_height,
    )
    bottoms_out[f, c] = bottom
    tops_out[f, c] = top


def replay(reference: dict, trace: dict, summary: dict, profile: dict, identity: dict) -> tuple[dict, dict]:
    """Advance contact on GPU at saved simulated states and validate agreement.

    Args:
        reference: Reference archive containing geometry and initial state.
        trace: Saved simulated state, velocity, wrench, and compression data.
        summary: Saved execution summary containing shoe metadata and run info.
        profile: Mass, inertia, and kinematic profile.
        identity: Replay identity dictionary.

    Returns:
        Tuple of (arrays, details) matching the schema returned by springs._replay.
    """
    from projects.digital_shoe import runtime  # noqa: PLC0415
    from projects.impedance_instron.cartesian import springs  # noqa: PLC0415
    from projects.impedance_instron.cartesian.gpu import (  # noqa: PLC0415
        benchmark,
        contact_replay,
    )
    from projects.impedance_instron.cartesian.gpu.mechanics import make_params  # noqa: PLC0415
    from projects.impedance_instron.cartesian.gpu.provenance import (  # noqa: PLC0415
        validate_sources,
    )
    from projects.impedance_instron.cartesian.shoe import Shoe  # noqa: PLC0415

    started = time.perf_counter()
    times, dt = springs._check_trace(reference, trace, summary, profile)
    metadata = summary["shoe"]
    if identity["artifact_sha256"] != metadata["sha256"]:
        raise ValueError("Saved shoe artifact changed; refusing spring export")

    # Guard: verify execution identity matches saved summary if present
    current_exec_id = benchmark.execution_identity()
    saved_exec_id = summary.get("execution_identity")
    if springs._plain(current_exec_id) != springs._plain(saved_exec_id):
        raise ValueError("Execution identity differs from saved summary execution identity")

    # Guard: check runtime module options
    runtime_options = wp.get_module_options(module=runtime)
    if runtime_options.get("fuse_fp") is not True:
        raise ValueError("GPU contact replay requires shared runtime fuse_fp=True")

    # Device selection: require CUDA device
    run_meta = summary.get("run", summary)
    device_name = run_meta.get("shoe_device", "cuda:0")
    device = wp.get_device(device_name)
    if not device.is_cuda:
        raise ValueError(f"GPU contact replay requires a CUDA device, got {device_name}")

    shoe = Shoe(
        metadata["path"],
        metadata["mount_m"],
        metadata["static_pitch_rad"],
        str(device),
        friction_model=metadata.get("friction_model", "legacy"),
    )
    if springs._plain(shoe.metadata) != springs._plain(metadata):
        different = sorted(
            key for key in set(shoe.metadata) | set(metadata) if shoe.metadata.get(key) != metadata.get(key)
        )
        raise ValueError(f"Full saved shoe metadata identity differs: {different}")

    frozen_shoe = summary.get("identity", {}).get("shoe")
    actual_shoe = {
        "artifact_sha256": shoe.metadata["sha256"],
        "mount_m": shoe.metadata["mount_m"],
        "static_pitch_rad": shoe.metadata["static_pitch_rad"],
        "device": str(shoe.device),
        "friction": shoe.metadata["friction"],
    }
    if frozen_shoe is not None and springs._plain(frozen_shoe) != actual_shoe:
        raise ValueError("Frozen shoe identity differs from the GPU replay shoe")

    # Guard source identities against summary if summary defines source_sha256
    if not summary.get("source_sha256"):
        raise ValueError("Saved GPU trace has no runtime source identities")
    validate_sources(summary)

    n = len(times)
    params = make_params(reference, profile)
    pitch_rad = wp.float64(metadata["static_pitch_rad"])

    # Upload states and velocities once
    states_d = wp.array(trace["state"], dtype=Vec5, device=device)
    velocities_d = wp.array(trace["velocity"], dtype=Vec5, device=device)
    body_q_d = wp.empty(n, dtype=wp.transform, device=device)
    body_qd_d = wp.empty(n, dtype=wp.spatial_vector, device=device)

    wp.launch(
        kinematic_ankle_poses,
        dim=n,
        inputs=[params, pitch_rad, states_d, velocities_d, body_q_d, body_qd_d],
        device=device,
        block_dim=32,
    )

    q_np = body_q_d.numpy()
    qd_np = body_qd_d.numpy()

    # Replay full-rate contact using contact_replay.replay
    contact_start = time.perf_counter()
    replay_result = contact_replay.replay(shoe, q_np, qd_np, dt, 32)
    contact_wall_s = time.perf_counter() - contact_start

    # Wrenches and compression histories
    wrenches = replay_result["wrench_newton"]  # shape (n, 6)
    compressions = replay_result["compression_m"]  # shape (n, column_count)

    rest = np.asarray(shoe.shoe.column_bed.rest_length_m, dtype=np.float64)
    driven = shoe.foundation.driven.numpy().astype(bool)
    passive = ~driven
    passive_cap = float(shoe.foundation.surround.max_strain) if np.any(passive) else None
    if not np.any(driven):
        raise ValueError("Saved shoe has no driven columns")

    indices = springs._frame_indices(times)
    frames, columns = len(indices), len(rest)

    # Frame extraction on GPU using render_carried_endpoints
    render_start = time.perf_counter()
    indices_d = wp.array(indices.astype(np.int32), dtype=int, device=device)
    compression_d = wp.array(compressions, dtype=float, device=device)
    bottoms_d = wp.empty((frames, columns), dtype=wp.vec3, device=device)
    tops_d = wp.empty((frames, columns), dtype=wp.vec3, device=device)

    wp.launch(
        render_carried_endpoints,
        dim=(frames, columns),
        inputs=[
            body_q_d,
            indices_d,
            shoe.foundation.anchor_local,
            shoe.foundation.rest_len,
            compression_d,
            shoe.foundation.driven,
            0.0,
            bottoms_d,
            tops_d,
        ],
        device=device,
    )

    rendered_bottoms = bottoms_d.numpy()  # shape (frames, columns, 3)
    rendered_tops = tops_d.numpy()  # shape (frames, columns, 3)
    render_wall_s = time.perf_counter() - render_start

    if not np.isfinite(rendered_bottoms).all() or not np.isfinite(rendered_tops).all():
        raise ValueError("Nonfinite endpoints produced during carried column rendering")

    sampled_compression = compressions[indices].astype(np.float32)
    snapshot_compression_error = float(np.max(np.abs(sampled_compression - compressions[indices])))

    arrays = {
        "time_s": times[indices].copy(),
        "trace_index": indices,
        "bottom_m": rendered_bottoms,
        "top_m": rendered_tops,
        "compression_m": sampled_compression,
        "rest_length_m": rest.copy(),
        "anchor_local_m": shoe.anchor_local_m.copy(),
        "driven": driven,
    }

    errors = dict.fromkeys(
        (
            "force_x_n",
            "force_z_n",
            "moment_nm",
            "compression_fraction",
            "driven_compression_fraction",
            "passive_compression_fraction",
        ),
        0.0,
    )
    errors["snapshot_compression_m"] = snapshot_compression_error
    errors["passive_cap_count"] = 0
    cap_mismatch_rows = 0
    first_mismatch = None
    maximum_compression_m = 0.0
    maximum_strain = {"driven": 0.0, "passive": 0.0}

    # Verify per-row agreement against trace using exact float64 division
    tolerances = springs._TOLERANCES
    for row in range(n):
        wrench = wrenches[row]
        wrench_ankle = np.array([wrench[0], wrench[2], -wrench[4]], dtype=np.float64)
        compression = compressions[row]
        if not np.isfinite(wrench_ankle).all() or not np.isfinite(compression).all():
            raise ValueError(f"Nonfinite replay contact at trace row {row}")

        strain = compression.astype(np.float64) / rest
        driven_max = float(np.max(strain[driven]))
        passive_max = float(np.max(strain[passive])) if np.any(passive) else 0.0
        cap_count = int(np.count_nonzero(strain[passive] >= passive_cap - 1e-6)) if passive_cap else 0

        row_errors = {
            "force_x_n": abs(float(wrench_ankle[0]) - float(trace["grf_n"][row, 0])),
            "force_z_n": abs(float(wrench_ankle[1]) - float(trace["grf_n"][row, 1])),
            "moment_nm": abs(float(wrench_ankle[2]) - float(trace["ankle_contact_moment_nm"][row])),
            "compression_fraction": abs(max(driven_max, passive_max) - float(trace["compression_fraction"][row])),
            "driven_compression_fraction": abs(driven_max - float(trace["driven_compression_fraction"][row])),
            "passive_compression_fraction": abs(passive_max - float(trace["passive_compression_fraction"][row])),
            "passive_cap_count": abs(cap_count - int(trace["passive_cap_column_count"][row])),
        }
        for key, error in row_errors.items():
            errors[key] = max(errors[key], error)
        cap_mismatch_rows += int(row_errors["passive_cap_count"] != 0)

        if first_mismatch is None and (
            max(row_errors["force_x_n"], row_errors["force_z_n"]) > tolerances["force_n"]
            or row_errors["moment_nm"] > tolerances["moment_nm"]
            or max(row_errors[key] for key in row_errors if key.endswith("compression_fraction"))
            > tolerances["compression_fraction"]
            or row_errors["passive_cap_count"] != 0
        ):
            first_mismatch = {"trace_index": row, "time_s": float(times[row]), "errors": row_errors}

        maximum_compression_m = max(maximum_compression_m, float(np.max(compression)))
        maximum_strain["driven"] = max(maximum_strain["driven"], driven_max)
        maximum_strain["passive"] = max(maximum_strain["passive"], passive_max)

    validation = {
        "passed": first_mismatch is None and errors["snapshot_compression_m"] == 0,
        "checked_rows": n,
        "snapshot_rows": frames,
        "full_shoe_metadata_match": True,
        "frozen_shoe_identity_match": frozen_shoe is not None,
        "artifact_sha256_match": True,
        "maximum_absolute_errors": errors,
        "absolute_tolerances": tolerances,
        "passive_cap_mismatch_rows": cap_mismatch_rows,
        "first_mismatch": first_mismatch,
        "maximum_compression_m": maximum_compression_m,
        "maximum_compression_fraction": maximum_strain,
        "passive_cap_fraction": passive_cap,
        "actual_dt_s": dt,
        "wall_s": time.perf_counter() - started,
        "scope": "saved-output agreement only; not physical validation or force-fit acceptance",
    }

    details = {
        "validation": validation,
        "spacing_m": float(shoe.shoe.column_bed.spacing_m),
        "color_max_mm": float(max(5, math.ceil(maximum_compression_m * 1000 / 5) * 5)),
        "time_alignment": springs._ALIGNMENT,
        "replay_metadata": {
            "backend": "gpu",
            "endpoint_alignment": "carried_column_segment",
            "timings": {
                "total_wall_s": time.perf_counter() - started,
                "contact_replay_wall_s": contact_wall_s,
                "render_endpoints_wall_s": render_wall_s,
                "capture_wall_s": replay_result.get("capture_wall_s"),
                "replay_and_readback_wall_s": replay_result.get("replay_and_readback_wall_s"),
            },
            "device": str(device),
            "execution_identity": current_exec_id,
        },
        "provenance": {
            "operation": "contact history advanced offline from zero at every saved simulated state and velocity on GPU",
            "new_dynamics_rollout": False,
            "controller_evaluated": False,
            "reference_motion_prescribed": False,
            "force_fitting": False,
            "frame_selection": "up to 181 evenly spaced row indices plus nearest rows to 0.06, 0.18, 0.30 s",
            "endpoint_source": "projects.digital_shoe.rendering.carried_column_segment (batched kernel wrapper)",
            "geometry": "full world endpoints; existing driven/passive behavior and fixture offsets retained; no gap repair",
            "maximum_frame_interval_s": float(np.max(np.diff(arrays["time_s"]))) if frames > 1 else 0.0,
            "color_scale": "fixed for the complete full-rate run, from its global peak compression [mm]",
        },
    }
    return arrays, details
