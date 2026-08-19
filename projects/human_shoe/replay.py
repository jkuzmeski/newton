# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exact prescribed-kinematics replay of Digital Instron shoe loads."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

from newton import opensim
from projects.digital_instron_v2.core import CALIBRATED_MATERIAL
from projects.digital_instron_v2.dynamics import FoundationConfig, MidsoleFoundation

from .contracts import load_experiment
from .preparation import make_human_shoe_foundation_config, prepare_attached_sole

BASE_DIR = Path(__file__).resolve().parents[2]


@dataclass(frozen=True, slots=True)
class ReplayWindow:
    """One source-sampled contact run with unloaded bracketing times [s]."""

    stance_index: int
    start_time_s: float
    end_time_s: float
    contact_start_time_s: float
    contact_end_time_s: float
    minimum_clearance_m: float


@dataclass(frozen=True, slots=True)
class PrescribedReplayConfig:
    """Configuration for exact prescribed-motion shoe load replay."""

    dt_s: float | None = None
    ground_height_m: float = 0.0
    stance_index: int = 0
    start_time_s: float | None = None
    end_time_s: float | None = None
    foundation: FoundationConfig = field(default_factory=make_human_shoe_foundation_config)
    chunk_size: int = 4096
    record_columns: bool = False


@dataclass(frozen=True, slots=True)
class ShoeLoadReplayResult:
    """Per-substep prescribed shoe load history in Newton's Z-up world [SI]."""

    time_s: np.ndarray
    dt_s: np.ndarray
    grf_n: np.ndarray
    moment_origin_nm: np.ndarray
    cop_m: np.ndarray
    cop_valid: np.ndarray
    max_compression_m: np.ndarray
    active_columns: np.ndarray
    contact_power_w: np.ndarray
    contact_work_j: np.ndarray
    impulse_ns: np.ndarray
    window: ReplayWindow
    provenance: dict[str, Any]
    column_compression_m: np.ndarray | None = None
    column_force_n: np.ndarray | None = None
    column_bottom_local_m: np.ndarray | None = None
    column_top_local_m: np.ndarray | None = None
    column_rest_len_m: np.ndarray | None = None

    def __post_init__(self) -> None:
        sample_count = len(self.time_s)
        sample_arrays = {
            "dt_s": self.dt_s,
            "grf_n": self.grf_n,
            "moment_origin_nm": self.moment_origin_nm,
            "cop_m": self.cop_m,
            "cop_valid": self.cop_valid,
            "max_compression_m": self.max_compression_m,
            "active_columns": self.active_columns,
            "contact_power_w": self.contact_power_w,
            "contact_work_j": self.contact_work_j,
            "impulse_ns": self.impulse_ns,
        }
        for name, values in sample_arrays.items():
            if len(values) != sample_count:
                raise ValueError(f"{name} must have {sample_count} samples")
        if (self.column_compression_m is None) != (self.column_force_n is None):
            raise ValueError("column compression and force histories must be provided together")
        if self.column_compression_m is not None:
            expected = self.column_compression_m.shape
            if len(expected) != 2 or expected[0] != sample_count:
                raise ValueError("column_compression_m must have shape [sample, column]")
            if self.column_force_n.shape != (*expected, 3):
                raise ValueError("column_force_n must have shape [sample, column, 3]")
            if np.any(self.column_compression_m < 0.0) or not np.all(np.isfinite(self.column_compression_m)):
                raise ValueError("column compression history must be finite and nonnegative")
            if not np.all(np.isfinite(self.column_force_n)):
                raise ValueError("column force history must be finite")

    @property
    def peak_vertical_force_n(self) -> float:
        """Maximum vertical ground-reaction force [N]."""
        return float(np.max(self.grf_n[:, 2], initial=0.0))

    @property
    def final_vertical_impulse_ns(self) -> float:
        """Final cumulative vertical impulse [N·s]."""
        return float(self.impulse_ns[-1, 2]) if len(self.impulse_ns) else 0.0

    @property
    def final_contact_work_j(self) -> float:
        """Final signed work delivered by the foundation to the carrier [J]."""
        return float(self.contact_work_j[-1]) if len(self.contact_work_j) else 0.0

    def write_csv(self, path: str | Path, metadata_path: str | Path | None = None) -> tuple[Path, Path]:
        """Write scalar CSV/JSON output and an optional per-column NPZ archive."""
        path = Path(path)
        metadata_path = Path(metadata_path) if metadata_path is not None else path.with_suffix(".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        columns = [
            ("sample_index", "1", "Zero-based replay sample index"),
            ("time_s", "s", "Prescribed motion time"),
            ("dt_s", "s", "Foundation integration step"),
            ("grf_x_n", "N", "Ground-on-shoe force, Newton world X"),
            ("grf_y_n", "N", "Ground-on-shoe force, Newton world Y"),
            ("grf_z_n", "N", "Ground-on-shoe force, Newton world Z"),
            ("moment_x_nm", "N*m", "Ground wrench moment about world origin, X"),
            ("moment_y_nm", "N*m", "Ground wrench moment about world origin, Y"),
            ("moment_z_nm", "N*m", "Ground wrench moment about world origin, Z"),
            ("cop_x_m", "m", "Normal-force center of pressure on z=0, X"),
            ("cop_y_m", "m", "Normal-force center of pressure on z=0, Y"),
            ("cop_z_m", "m", "Center-of-pressure plane height"),
            ("cop_valid", "1", "One when vertical force exceeds 1e-6 N"),
            ("max_compression_m", "m", "Maximum column compression"),
            ("active_columns", "1", "Geometrically penetrating column count"),
            ("contact_power_w", "W", "Power delivered by the foundation to the carrier"),
            ("contact_work_j", "J", "Cumulative left-rule signed contact work"),
            ("impulse_x_ns", "N*s", "Cumulative ground impulse, X"),
            ("impulse_y_ns", "N*s", "Cumulative ground impulse, Y"),
            ("impulse_z_ns", "N*s", "Cumulative ground impulse, Z"),
        ]
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow([column[0] for column in columns])
            for i in range(len(self.time_s)):
                writer.writerow(
                    [
                        i,
                        self.time_s[i],
                        self.dt_s[i],
                        *self.grf_n[i],
                        *self.moment_origin_nm[i],
                        *self.cop_m[i],
                        int(self.cop_valid[i]),
                        self.max_compression_m[i],
                        int(self.active_columns[i]),
                        self.contact_power_w[i],
                        self.contact_work_j[i],
                        *self.impulse_ns[i],
                    ]
                )
        metadata = {
            "schema_version": "human_shoe_prescribed_replay_1",
            "csv": path.name,
            "sample_count": len(self.time_s),
            "coordinate_system": "Newton right-handed Z-up",
            "force_direction": "environment on shoe",
            "moment_reference": "fixed world origin",
            "cop_plane": "world z=0",
            "cop_validity": "vertical force > 1e-6 N",
            "sampling_phase": "after prescribed pose/twist and foundation evaluation",
            "work_sign": "positive when the foundation delivers energy to the carrier",
            "work_quadrature": "left rule matching per-substep force evaluation",
            "columns": [dict(zip(("name", "unit", "description"), column, strict=True)) for column in columns],
            "window": asdict(self.window),
            "provenance": self.provenance,
        }
        if self.column_compression_m is not None and self.column_force_n is not None:
            column_path = path.with_suffix(".columns.npz")
            np.savez_compressed(
                column_path,
                time_s=self.time_s,
                compression_m=self.column_compression_m,
                force_n=self.column_force_n,
                bottom_local_m=self.column_bottom_local_m,
                top_local_m=self.column_top_local_m,
                rest_len_m=self.column_rest_len_m,
            )
            metadata["column_data"] = {
                "file": column_path.name,
                "column_count": int(self.column_compression_m.shape[1]),
                "dtype": str(self.column_compression_m.dtype),
                "compression_m": list(self.column_compression_m.shape),
                "force_n": list(self.column_force_n.shape),
                "force_semantics": "environment on shoe, Newton world XYZ",
                "column_order": "prepared sole bottom_local_m/top_local_m/rest_len_m",
            }
        else:
            path.with_suffix(".columns.npz").unlink(missing_ok=True)
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n")
        return path, metadata_path


@dataclass(slots=True)
class _CarrierState:
    body_q: wp.array[wp.transform]
    body_qd: wp.array[wp.spatial_vector]
    body_f: wp.array[wp.spatial_vector]


@wp.kernel
def _set_prescribed_sample(
    sample: wp.int32,
    poses: wp.array[wp.transform],
    twists: wp.array[wp.spatial_vector],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    body_q[0] = poses[sample]
    body_qd[0] = twists[sample]


@wp.kernel
def _record_foundation_substep(
    time: wp.float64,
    dt: wp.float32,
    ground_height: wp.float32,
    normal_force: wp.array[wp.float32],
    cop_moment: wp.array[wp.vec3],
    active: wp.array[wp.int32],
    resultant_force: wp.array[wp.vec3],
    resultant_moment_origin: wp.array[wp.vec3],
    contact_power: wp.array[wp.float32],
    max_compression: wp.array[wp.float32],
    capacity: wp.int32,
    count: wp.array[wp.int32],
    overflow: wp.array[wp.int32],
    time_out: wp.array[wp.float64],
    dt_out: wp.array[wp.float32],
    force_out: wp.array[wp.vec3],
    moment_out: wp.array[wp.vec3],
    cop_out: wp.array[wp.vec3],
    cop_valid_out: wp.array[wp.int32],
    compression_out: wp.array[wp.float32],
    active_out: wp.array[wp.int32],
    power_out: wp.array[wp.float32],
    work_out: wp.array[wp.float32],
    impulse_out: wp.array[wp.vec3],
):
    k = count[0]
    if k >= capacity:
        overflow[0] = 1
        return
    force = resultant_force[0]
    fz = normal_force[0]
    cop = wp.vec3(0.0, 0.0, ground_height)
    valid = wp.int32(0)
    if fz > 1.0e-6:
        cop = wp.vec3(cop_moment[0][0] / fz, cop_moment[0][1] / fz, ground_height)
        valid = 1
    previous_impulse = wp.vec3(0.0, 0.0, 0.0)
    previous_work = wp.float32(0.0)
    if k > 0:
        previous_impulse = impulse_out[k - 1]
        previous_work = work_out[k - 1]
    time_out[k] = time
    dt_out[k] = dt
    force_out[k] = force
    moment_out[k] = resultant_moment_origin[0]
    cop_out[k] = cop
    cop_valid_out[k] = valid
    compression_out[k] = max_compression[0]
    active_out[k] = active[0]
    power_out[k] = contact_power[0]
    impulse_out[k] = previous_impulse + force * dt
    work_out[k] = previous_work + contact_power[0] * dt
    count[0] = k + 1


@wp.kernel
def _record_foundation_columns(
    sample: wp.int32,
    compression: wp.array[wp.float32],
    column_force: wp.array[wp.vec3],
    compression_out: wp.array2d[wp.float32],
    force_out: wp.array2d[wp.vec3],
):
    column = wp.tid()
    compression_out[sample, column] = compression[column]
    force_out[sample, column] = column_force[column]


class _FoundationReplayRecorder:
    def __init__(self, capacity: int, column_count: int, *, record_columns: bool = False, device=None) -> None:
        self.capacity = int(capacity)
        self.column_count = int(column_count)
        self.record_columns = bool(record_columns)
        self.device = device
        self.count = wp.zeros(1, dtype=wp.int32, device=device)
        self.overflow = wp.zeros(1, dtype=wp.int32, device=device)
        self.time = wp.empty(capacity, dtype=wp.float64, device=device)
        self.dt = wp.empty(capacity, dtype=wp.float32, device=device)
        self.force = wp.empty(capacity, dtype=wp.vec3, device=device)
        self.moment = wp.empty(capacity, dtype=wp.vec3, device=device)
        self.cop = wp.empty(capacity, dtype=wp.vec3, device=device)
        self.cop_valid = wp.empty(capacity, dtype=wp.int32, device=device)
        self.compression = wp.empty(capacity, dtype=wp.float32, device=device)
        self.active = wp.empty(capacity, dtype=wp.int32, device=device)
        self.power = wp.empty(capacity, dtype=wp.float32, device=device)
        self.work = wp.empty(capacity, dtype=wp.float32, device=device)
        self.impulse = wp.empty(capacity, dtype=wp.vec3, device=device)
        history_rows = capacity if self.record_columns else 1
        history_columns = self.column_count if self.record_columns else 1
        self.column_compression = wp.empty((history_rows, history_columns), dtype=wp.float32, device=device)
        self.column_force = wp.empty((history_rows, history_columns), dtype=wp.vec3, device=device)

    def record(
        self,
        sample: int,
        time: float,
        dt: float,
        ground_height: float,
        foundation: MidsoleFoundation,
    ) -> None:
        if not 0 <= sample < self.capacity:
            raise IndexError(f"sample {sample} is outside recorder capacity {self.capacity}")
        wp.launch(
            _record_foundation_substep,
            dim=1,
            inputs=[
                time,
                dt,
                ground_height,
                foundation.normal_force,
                foundation.cop_moment,
                foundation.active,
                foundation.resultant_force,
                foundation.resultant_moment_origin,
                foundation.contact_power,
                foundation.max_compression,
                self.capacity,
                self.count,
                self.overflow,
                self.time,
                self.dt,
                self.force,
                self.moment,
                self.cop,
                self.cop_valid,
                self.compression,
                self.active,
                self.power,
                self.work,
                self.impulse,
            ],
            device=self.device,
        )
        if self.record_columns:
            wp.launch(
                _record_foundation_columns,
                dim=self.column_count,
                inputs=[
                    sample,
                    foundation.compression,
                    foundation.column_force,
                    self.column_compression,
                    self.column_force,
                ],
                device=self.device,
            )

    def result(
        self,
        window: ReplayWindow,
        provenance: dict[str, Any],
        prepared: PreparedAttachedSole,
    ) -> ShoeLoadReplayResult:
        count = int(self.count.numpy()[0])
        if int(self.overflow.numpy()[0]):
            raise RuntimeError("foundation replay recorder capacity was exceeded")
        return ShoeLoadReplayResult(
            time_s=self.time.numpy()[:count],
            dt_s=self.dt.numpy()[:count],
            grf_n=self.force.numpy()[:count],
            moment_origin_nm=self.moment.numpy()[:count],
            cop_m=self.cop.numpy()[:count],
            cop_valid=self.cop_valid.numpy()[:count].astype(bool),
            max_compression_m=self.compression.numpy()[:count],
            active_columns=self.active.numpy()[:count],
            contact_power_w=self.power.numpy()[:count],
            contact_work_j=self.work.numpy()[:count],
            impulse_ns=self.impulse.numpy()[:count],
            window=window,
            provenance=provenance,
            column_compression_m=(self.column_compression.numpy()[:count] if self.record_columns else None),
            column_force_n=(self.column_force.numpy()[:count] if self.record_columns else None),
            column_bottom_local_m=(prepared.column_bottom_local.copy() if self.record_columns else None),
            column_top_local_m=(prepared.column_top_local.copy() if self.record_columns else None),
            column_rest_len_m=(prepared.column_rest_len.copy() if self.record_columns else None),
        )


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (BASE_DIR / path).resolve()


def _portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(BASE_DIR))
    except ValueError:
        return str(path.resolve())


def _load_replay_inputs(experiment_path: str | Path):
    experiment_path = _resolve_repo_path(experiment_path)
    experiment = load_experiment(experiment_path)
    if experiment.motion_path is None:
        raise ValueError("prescribed replay requires experiment.motion_path")
    model_path = _resolve_repo_path(experiment.human_model_path)
    motion_path = _resolve_repo_path(experiment.motion_path)
    manifest_path = _resolve_repo_path(experiment.shoe_manifest_path)
    model = opensim.parse_osim(model_path)
    body_index = {"ground": -1, **{body.name: index for index, body in enumerate(model.bodies)}}
    import_result = opensim.OsimImportResult(model=model, body_index=body_index)
    prepared = prepare_attached_sole(import_result, experiment.attachment, manifest_path)
    source_time, source_coordinates = opensim.read_motion(model, motion_path)
    return experiment_path, experiment, model, motion_path, manifest_path, prepared, source_time, source_coordinates


def _sample_motion_hermite(
    source_time: np.ndarray,
    source_coordinates: np.ndarray,
    output_time: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a C1 cubic-Hermite coordinate interpolant and its analytic derivative."""
    source_time = np.asarray(source_time, dtype=np.float64)
    source_coordinates = np.asarray(source_coordinates, dtype=np.float64)
    output_time = np.asarray(output_time, dtype=np.float64)
    if source_time.ndim != 1 or source_coordinates.shape[0] != len(source_time):
        raise ValueError("source coordinates must have shape [time, coordinate]")
    if len(source_time) < 2 or np.any(np.diff(source_time) <= 0.0):
        raise ValueError("source times must be strictly increasing")
    if len(output_time) > 1 and np.any(np.diff(output_time) <= 0.0):
        raise ValueError("output times must be strictly increasing")
    if len(output_time) and (output_time[0] < source_time[0] or output_time[-1] > source_time[-1]):
        raise ValueError("output times must lie inside the source motion")
    slopes = np.gradient(source_coordinates, source_time, axis=0, edge_order=1)
    interval = np.searchsorted(source_time, output_time, side="right") - 1
    interval = np.clip(interval, 0, len(source_time) - 2)
    t0 = source_time[interval]
    t1 = source_time[interval + 1]
    h = t1 - t0
    u = (output_time - t0) / h
    u2 = u * u
    u3 = u2 * u
    h00 = 2.0 * u3 - 3.0 * u2 + 1.0
    h10 = u3 - 2.0 * u2 + u
    h01 = -2.0 * u3 + 3.0 * u2
    h11 = u3 - u2
    coordinates = (
        h00[:, None] * source_coordinates[interval]
        + (h10 * h)[:, None] * slopes[interval]
        + h01[:, None] * source_coordinates[interval + 1]
        + (h11 * h)[:, None] * slopes[interval + 1]
    )
    dh00 = (6.0 * u2 - 6.0 * u) / h
    dh10 = 3.0 * u2 - 4.0 * u + 1.0
    dh01 = (-6.0 * u2 + 6.0 * u) / h
    dh11 = 3.0 * u2 - 2.0 * u
    speeds = (
        dh00[:, None] * source_coordinates[interval]
        + dh10[:, None] * slopes[interval]
        + dh01[:, None] * source_coordinates[interval + 1]
        + dh11[:, None] * slopes[interval + 1]
    )
    return coordinates, speeds


def _clearance_from_native_poses(poses: np.ndarray, bottom_local: np.ndarray, basis: np.ndarray) -> np.ndarray:
    world_native = np.einsum("tij,nj->tni", poses[:, :3, :3], bottom_local) + poses[:, None, :3, 3]
    world_newton = np.einsum("ij,tnj->tni", basis, world_native)
    return np.min(world_newton[:, :, 2], axis=1)


def find_contact_windows(
    experiment_path: str | Path,
    *,
    ground_height_m: float = 0.0,
    device=None,
) -> tuple[ReplayWindow, ...]:
    """Find complete source-sampled right-shoe penetration windows."""
    _, _, model, _, _, prepared, time, coordinates = _load_replay_inputs(experiment_path)
    fk = opensim.ForwardKinematics(model, device=device)
    poses = fk.body_transforms_batch(coordinates)
    carrier = fk.body_names.index(prepared.resolved.reference.foot_body_name)
    clearance = (
        _clearance_from_native_poses(
            poses[:, carrier], prepared.column_bottom_local, opensim.OsimFrameConverter().matrix
        )
        - ground_height_m
    )
    mask = clearance < 0.0
    changes = np.diff(np.concatenate(([False], mask, [False])).astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1) - 1
    windows = []
    for stance_index, (start, stop) in enumerate(zip(starts, stops, strict=True)):
        if start == 0 or stop == len(time) - 1:
            continue
        windows.append(
            ReplayWindow(
                stance_index=stance_index,
                start_time_s=float(time[start - 1]),
                end_time_s=float(time[stop + 1]),
                contact_start_time_s=float(time[start]),
                contact_end_time_s=float(time[stop]),
                minimum_clearance_m=float(np.min(clearance[start : stop + 1])),
            )
        )
    return tuple(windows)


def _rotation_matrices_to_quaternions(rotation: np.ndarray) -> np.ndarray:
    """Convert rotation matrices to normalized Warp-order quaternions."""
    output = np.empty((len(rotation), 4), dtype=np.float32)
    for index, matrix in enumerate(rotation):
        trace = np.trace(matrix)
        if trace > 0.0:
            scale = np.sqrt(trace + 1.0) * 2.0
            quaternion = np.array(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    0.25 * scale,
                ]
            )
        else:
            axis = int(np.argmax(np.diag(matrix)))
            if axis == 0:
                scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
                quaternion = np.array(
                    [
                        0.25 * scale,
                        (matrix[0, 1] + matrix[1, 0]) / scale,
                        (matrix[0, 2] + matrix[2, 0]) / scale,
                        (matrix[2, 1] - matrix[1, 2]) / scale,
                    ]
                )
            elif axis == 1:
                scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
                quaternion = np.array(
                    [
                        (matrix[0, 1] + matrix[1, 0]) / scale,
                        0.25 * scale,
                        (matrix[1, 2] + matrix[2, 1]) / scale,
                        (matrix[0, 2] - matrix[2, 0]) / scale,
                    ]
                )
            else:
                scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
                quaternion = np.array(
                    [
                        (matrix[0, 2] + matrix[2, 0]) / scale,
                        (matrix[1, 2] + matrix[2, 1]) / scale,
                        0.25 * scale,
                        (matrix[1, 0] - matrix[0, 1]) / scale,
                    ]
                )
        quaternion /= np.linalg.norm(quaternion)
        if index and np.dot(quaternion, output[index - 1]) < 0.0:
            quaternion = -quaternion
        output[index] = quaternion
    return output


def _exact_carrier_kinematics(
    model: opensim.OsimModel,
    coordinates: np.ndarray,
    speeds: np.ndarray,
    body_name: str,
    *,
    chunk_size: int,
    device=None,
) -> tuple[np.ndarray, np.ndarray]:
    fk = opensim.ForwardKinematics(model, device=device)
    body = fk.body_names.index(body_name)
    basis = opensim.OsimFrameConverter().matrix
    poses = []
    twists = []
    for start in range(0, len(coordinates), chunk_size):
        stop = min(start + chunk_size, len(coordinates))
        transform = fk.body_transforms_batch(coordinates[start:stop])[:, body]
        velocity = fk.body_velocities_batch(coordinates[start:stop], speeds[start:stop])
        rotation = np.einsum("ij,tjk->tik", basis, transform[:, :3, :3])
        position = transform[:, :3, 3] @ basis.T
        angular = velocity["angular_velocity"][:, body] @ basis.T
        linear = velocity["linear_velocity"][:, body] @ basis.T
        quaternion = _rotation_matrices_to_quaternions(rotation)
        poses.append(np.column_stack([position, quaternion]).astype(np.float32))
        twists.append(np.column_stack([linear, angular]).astype(np.float32))
    return np.concatenate(poses), np.concatenate(twists)


def replay_prescribed_shoe_load(
    experiment_path: str | Path,
    config: PrescribedReplayConfig | None = None,
    *,
    device=None,
) -> ShoeLoadReplayResult:
    """Replay shoe loads from exact OpenSim kinematics without state integration."""
    config = config or PrescribedReplayConfig()
    (
        resolved_experiment_path,
        experiment,
        model,
        motion_path,
        manifest_path,
        prepared,
        source_time,
        source_coordinates,
    ) = _load_replay_inputs(experiment_path)
    dt = float(experiment.time_step_s if config.dt_s is None else config.dt_s)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt_s must be finite and positive")
    windows = find_contact_windows(resolved_experiment_path, ground_height_m=config.ground_height_m, device=device)
    if (config.start_time_s is None) != (config.end_time_s is None):
        raise ValueError("start_time_s and end_time_s must be provided together")
    if config.start_time_s is None:
        if not 0 <= config.stance_index < len(windows):
            raise IndexError(f"stance_index {config.stance_index} is outside {len(windows)} complete contact windows")
        window = windows[config.stance_index]
        start_time = window.start_time_s
        end_time = window.end_time_s
    else:
        start_time = float(config.start_time_s)
        end_time = float(config.end_time_s)
        window = ReplayWindow(
            stance_index=-1,
            start_time_s=start_time,
            end_time_s=end_time,
            contact_start_time_s=start_time,
            contact_end_time_s=end_time,
            minimum_clearance_m=0.0,
        )
    if not source_time[0] <= start_time < end_time <= source_time[-1]:
        raise ValueError("replay window must lie inside the source motion")
    output_time = start_time + np.arange(max(1, int(np.ceil((end_time - start_time) / dt)))) * dt
    output_time = output_time[output_time < end_time]
    coordinates, speeds = _sample_motion_hermite(source_time, source_coordinates, output_time)
    pose, twist = _exact_carrier_kinematics(
        model,
        coordinates,
        speeds,
        prepared.resolved.reference.foot_body_name,
        chunk_size=config.chunk_size,
        device=device,
    )

    poses_wp = wp.array(pose, dtype=wp.transform, device=device)
    twists_wp = wp.array(twist, dtype=wp.spatial_vector, device=device)
    state = _CarrierState(
        body_q=wp.zeros(1, dtype=wp.transform, device=device),
        body_qd=wp.zeros(1, dtype=wp.spatial_vector, device=device),
        body_f=wp.zeros(1, dtype=wp.spatial_vector, device=device),
    )
    foundation = MidsoleFoundation(
        prepared.column_bottom_local,
        np.full(len(prepared.column_bottom_local), config.ground_height_m, dtype=np.float32),
        prepared.column_rest_len,
        prepared.column_area,
        prepared.foundation_geometry.neighbors,
        prepared.foundation_geometry.spacing_m,
        CALIBRATED_MATERIAL,
        0,
        wp.zeros(1, dtype=wp.vec3, device=device),
        config.foundation,
        device,
    )
    foundation.reset()
    recorder = _FoundationReplayRecorder(
        len(output_time),
        len(prepared.column_bottom_local),
        record_columns=config.record_columns,
        device=device,
    )
    for sample, time in enumerate(output_time):
        wp.launch(
            _set_prescribed_sample,
            dim=1,
            inputs=[sample, poses_wp, twists_wp, state.body_q, state.body_qd],
            device=device,
        )
        foundation.apply(state, dt, clear_body_force=True)
        recorder.record(sample, float(time), dt, config.ground_height_m, foundation)

    provenance = {
        "experiment_path": _portable_path(resolved_experiment_path),
        "motion_path": _portable_path(motion_path),
        "shoe_manifest_path": _portable_path(manifest_path),
        "motion_interpolation": "cubic Hermite over full source motion",
        "state_integration": False,
        "kinematics": "exact OpenSim ForwardKinematics",
        "carrier_body": prepared.resolved.reference.foot_body_name,
        "dt_s": dt,
        "ground_height_m": config.ground_height_m,
        "foundation_config": asdict(config.foundation),
        "material": asdict(CALIBRATED_MATERIAL),
    }
    return recorder.result(window, provenance, prepared)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        default="experiments/human_shoe/baseline_gait2354.json",
        help="Human-shoe experiment JSON path.",
    )
    parser.add_argument("--dt", type=float, default=None, help="Replay timestep [s]; defaults to the manifest.")
    parser.add_argument("--stance-index", type=int, default=0, help="Complete contact window to replay.")
    parser.add_argument("--start-time", type=float, default=None, help="Optional explicit replay start time [s].")
    parser.add_argument("--end-time", type=float, default=None, help="Optional explicit replay end time [s].")
    parser.add_argument("--ground-height", type=float, default=0.0, help="Newton world ground height [m].")
    parser.add_argument("--device", help="Warp device override.")
    parser.add_argument(
        "--record-columns",
        action="store_true",
        help="Export per-column compression and force history to a compressed NPZ sidecar.",
    )
    parser.add_argument(
        "--output",
        default="reports/human_shoe/prescribed_stance.csv",
        help="Output CSV path; a JSON sidecar is written alongside it.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run an exact prescribed-motion replay and export its load history."""
    args = _build_arg_parser().parse_args(argv)
    if args.device:
        wp.set_device(args.device)
    config = PrescribedReplayConfig(
        dt_s=args.dt,
        ground_height_m=args.ground_height,
        stance_index=args.stance_index,
        start_time_s=args.start_time,
        end_time_s=args.end_time,
        record_columns=args.record_columns,
    )
    result = replay_prescribed_shoe_load(args.experiment, config)
    csv_path, metadata_path = result.write_csv(args.output)
    print(
        json.dumps(
            {
                "csv": str(csv_path),
                "metadata": str(metadata_path),
                "samples": len(result.time_s),
                "peak_vertical_force_n": result.peak_vertical_force_n,
                "vertical_impulse_ns": result.final_vertical_impulse_ns,
                "contact_work_j": result.final_contact_work_j,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
