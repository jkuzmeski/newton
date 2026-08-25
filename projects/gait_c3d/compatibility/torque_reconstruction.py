# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""OFFLINE COMPATIBILITY REFERENCE ONLY. Pointwise and rollout mechanics use newton.opensim, not the production Newton solver path.

Diagnose C3D inverse-to-forward torque reconstruction.

This adapter verifies the dynamics plumbing without claiming predictive gait. It
first feeds the exact filtered inverse-dynamics state, generalized forces, and
archived external wrenches through :class:`newton.opensim.ForwardDynamics`. It
then performs a trimmed open-loop rollout driven by linearly interpolated copies
of those measured inputs.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import newton.opensim as osim

from .pipeline import sha256, write_json

ARCHITECTURE_ROLE = "compatibility_reference"

_DEFAULT_DATA = Path("/home/jo31399/newton-data/gait/processed/trial_101/latest")
_POINTWISE_ROTATIONAL_THRESHOLD = 1.0e-3
_POINTWISE_TRANSLATIONAL_THRESHOLD = 1.0e-5


@dataclass
class _InterpolatedExternalLoads:
    """Linearly interpolate the archived, sanitized external-wrench trajectory."""

    times: np.ndarray
    bodies: list[str]
    wrenches: np.ndarray

    def sample(self, output_times: np.ndarray) -> tuple[list[str], np.ndarray]:
        """Sample ground-frame ``[F P T]`` wrenches at requested times."""
        return self.bodies, _interpolate(self.times, self.wrenches, output_times)


def _interpolate(times: np.ndarray, values: np.ndarray, output_times: np.ndarray) -> np.ndarray:
    """Linearly interpolate an array along its leading time axis without extrapolation."""
    times = np.asarray(times, float)
    values = np.asarray(values, float)
    output_times = np.asarray(output_times, float)
    if times.ndim != 1 or len(times) != len(values) or len(times) < 2 or np.any(np.diff(times) <= 0.0):
        raise ValueError("source times must increase strictly and match the values")
    if output_times.ndim != 1:
        raise ValueError("output times must be one-dimensional")
    tolerance = 1.0e-10
    if np.any(output_times < times[0] - tolerance) or np.any(output_times > times[-1] + tolerance):
        raise ValueError("requested time is outside the archived trajectory")
    clipped = np.clip(output_times, times[0], times[-1])
    flat = values.reshape(len(times), -1)
    result = np.column_stack([np.interp(clipped, times, flat[:, index]) for index in range(flat.shape[1])])
    return result.reshape((len(output_times), *values.shape[1:]))


def _require_array(archive: np.lib.npyio.NpzFile, name: str, ndim: int) -> np.ndarray:
    """Load one finite numeric array with the required dimensionality."""
    if name not in archive.files:
        raise ValueError(f"analysis archive is missing {name!r}; rerun the C3D pipeline")
    value = np.asarray(archive[name])
    if value.ndim != ndim:
        raise ValueError(f"analysis field {name!r} must have {ndim} dimensions")
    if not np.issubdtype(value.dtype, np.number) or not np.all(np.isfinite(value)):
        raise ValueError(f"analysis field {name!r} must contain finite numeric values")
    return np.asarray(value, float)


def _error_summary(error: np.ndarray, motion_types: list[str]) -> dict:
    """Summarize acceleration or state errors without mixing linear and angular units."""
    rotational = np.asarray([kind == "rotational" for kind in motion_types], bool)
    groups = {}
    for name, mask in (("rotational", rotational), ("translational", ~rotational)):
        values = np.abs(error[:, mask])
        groups[name] = {
            "count": int(np.count_nonzero(mask)),
            "rms": float(np.sqrt(np.mean(values**2))) if values.size else 0.0,
            "max_abs": float(np.max(values)) if values.size else 0.0,
        }
    return groups


def run_reconstruction(
    data_dir: str | os.PathLike = _DEFAULT_DATA,
    output_dir: str | os.PathLike | None = None,
    *,
    device: str = "cpu",
    rollout_dt: float = 0.001,
    trim_frames: int = 10,
) -> Path:
    """Run pointwise and open-loop torque reconstruction diagnostics.

    Args:
        data_dir: Completed ``projects.gait_c3d`` artifact directory.
        output_dir: Diagnostic output directory. Defaults to a sibling of
            ``data_dir`` so the staged source artifact remains immutable.
        device: Warp device used by the OpenSim dynamics kernels.
        rollout_dt: Fixed RK4 integration step [s].
        trim_frames: Frames omitted from both ends of the open-loop rollout.

    Returns:
        Path to the completed diagnostic directory.
    """
    data_dir = Path(data_dir).resolve()
    output_dir = (
        Path(output_dir).resolve()
        if output_dir is not None
        else data_dir.parent / f"{data_dir.name}_torque_reconstruction"
    )
    repository_root = Path(__file__).resolve().parents[2]
    if output_dir == repository_root or output_dir.is_relative_to(repository_root):
        raise ValueError("generated human-data diagnostics must stay outside the repository")
    if output_dir == data_dir or output_dir.is_relative_to(data_dir) or data_dir.is_relative_to(output_dir):
        raise ValueError("diagnostic and source artifact directories must not overlap")
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError("diagnostic output path exists and is not a directory")
    if rollout_dt <= 0.0:
        raise ValueError("rollout_dt must be positive")

    model_path = data_dir / "S001_scaled.osim"
    analysis_path = data_dir / "analysis.npz"
    manifest_path = data_dir / "manifest.json"
    for path in (model_path, analysis_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    with np.load(analysis_path, allow_pickle=False) as archive:
        times = _require_array(archive, "times", 1)
        q = _require_array(archive, "id_coordinates", 2)
        qd = _require_array(archive, "id_speeds", 2)
        qdd_reference = _require_array(archive, "id_accelerations", 2)
        tau = _require_array(archive, "id_generalized_forces", 2)
        wrenches = _require_array(archive, "id_external_wrenches", 3)
        coordinate_names = [str(value) for value in archive["id_names"]]
        external_bodies = [str(value) for value in archive["id_external_bodies"]]

    expected_state_shape = (len(times), len(coordinate_names))
    if any(value.shape != expected_state_shape for value in (q, qd, qdd_reference, tau)):
        raise ValueError("inverse-dynamics state and force arrays do not share one coordinate shape")
    if wrenches.shape != (len(times), len(external_bodies), 9):
        raise ValueError("external wrenches must have shape [time, body, 9]")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("analysis times must increase strictly")
    if trim_frames < 0 or 2 * trim_frames >= len(times) - 1:
        raise ValueError("trim_frames leaves no rollout interval")

    model = osim.parse_osim(model_path)
    forward = osim.ForwardDynamics(model, device=device)
    if coordinate_names != list(forward.coordinate_names):
        raise ValueError("analysis coordinate order does not match ForwardDynamics")
    motion_types = list(forward.motion_types)

    qdd_reconstructed = forward.accelerations(
        q,
        qd,
        tau,
        external_bodies=external_bodies,
        external_wrenches=wrenches,
    )
    acceleration_error = qdd_reconstructed - qdd_reference
    pointwise = _error_summary(acceleration_error, motion_types)
    model_coordinates = {coordinate.name: coordinate for joint in model.joints for coordinate in joint.coordinates}
    pointwise_by_coordinate = {
        name: {
            "unit": "rad/s^2" if motion_types[index] == "rotational" else "m/s^2",
            "rms": float(np.sqrt(np.mean(acceleration_error[:, index] ** 2))),
            "max_abs": float(np.max(np.abs(acceleration_error[:, index]))),
            "fixed": bool(
                model_coordinates[name].locked
                or (
                    model_coordinates[name].clamped
                    and model_coordinates[name].range[0] == model_coordinates[name].range[1]
                )
            ),
        }
        for index, name in enumerate(coordinate_names)
    }
    pointwise_pass = (
        pointwise["rotational"]["max_abs"] <= _POINTWISE_ROTATIONAL_THRESHOLD
        and pointwise["translational"]["max_abs"] <= _POINTWISE_TRANSLATIONAL_THRESHOLD
    )

    start = trim_frames
    stop = len(times) - 1 - trim_frames
    rollout_start = float(times[start])
    rollout_stop = float(times[stop])
    duration = rollout_stop - rollout_start
    steps = max(1, int(round(duration / rollout_dt)))
    rollout_dt = duration / steps

    def controls(t: float, _q: np.ndarray, _qd: np.ndarray) -> np.ndarray:
        return _interpolate(times, tau, np.asarray([t]))[0]

    interpolated_loads = _InterpolatedExternalLoads(times, external_bodies, wrenches)
    rollout = forward.simulate(
        q[start],
        qd[start],
        duration,
        rollout_dt,
        start_time=rollout_start,
        controls=controls,
        external_loads=interpolated_loads,
        integrator="rk4",
        use_graph=False,
    )
    reference_q = _interpolate(times, q, rollout.times)
    reference_qd = _interpolate(times, qd, rollout.times)
    coordinate_error = rollout.coordinates - reference_q
    speed_error = rollout.speeds - reference_qd
    finite_rows = np.all(np.isfinite(rollout.coordinates), axis=1) & np.all(np.isfinite(rollout.speeds), axis=1)
    nonfinite = np.flatnonzero(~finite_rows)
    finite_stop = int(nonfinite[0]) if len(nonfinite) else len(rollout.times)
    if finite_stop == 0:
        raise RuntimeError("forward rollout did not retain its finite initial state")
    completed = finite_stop == len(rollout.times)
    finite_slice = slice(0, finite_stop)

    fk = osim.ForwardKinematics(model, device=device)
    rollout_markers = fk.marker_positions_batch(rollout.coordinates[finite_slice])
    reference_markers = fk.marker_positions_batch(reference_q[finite_slice])
    finite_marker_error = np.linalg.norm(rollout_markers - reference_markers, axis=2)
    marker_error = np.full((len(rollout.times), finite_marker_error.shape[1]), np.nan)
    marker_error[finite_slice] = finite_marker_error
    coordinate_summary = _error_summary(coordinate_error[finite_slice], motion_types)
    speed_summary = _error_summary(speed_error[finite_slice], motion_types)
    rollout_summary = {
        "completed": completed,
        "completed_fraction": float(finite_stop / len(rollout.times)),
        "first_nonfinite_time_s": None if completed else float(rollout.times[finite_stop]),
        "start_time_s": rollout_start,
        "stop_time_s": rollout_stop,
        "duration_s": duration,
        "dt_s": rollout_dt,
        "trim_frames": trim_frames,
        "coordinate_error": coordinate_summary,
        "speed_error": speed_summary,
        "marker_error_m": {
            "rms_over_finite_prefix": float(np.sqrt(np.mean(finite_marker_error**2))),
            "max_over_finite_prefix": float(np.max(finite_marker_error)),
            "last_finite_rms": float(np.sqrt(np.mean(finite_marker_error[-1] ** 2))),
            "last_finite_max": float(np.max(finite_marker_error[-1])),
        },
    }

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = {
        "schema_version": "gait_c3d_torque_reconstruction_1",
        "architecture_role": ARCHITECTURE_ROLE,
        "reference_only": True,
        "production_eligible": False,
        "status": (
            "engineering_pointwise_pass_open_loop_completed"
            if pointwise_pass and completed
            else "engineering_pointwise_pass_open_loop_nonfinite"
            if pointwise_pass
            else "engineering_pointwise_failed"
        ),
        "scope": "diagnostic measured-load torque reconstruction; not predictive contact dynamics",
        "input": {
            "data_dir": str(data_dir),
            "manifest_status": manifest.get("status"),
            "git_commit": manifest.get("runtime", {}).get("git_commit"),
            "analysis_sha256": sha256(analysis_path),
            "model_sha256": sha256(model_path),
        },
        "pointwise_acceleration_reconstruction": {
            "passed": pointwise_pass,
            "value": pointwise,
            "per_coordinate": pointwise_by_coordinate,
            "threshold": {
                "rotational_max_abs_rad_s2": _POINTWISE_ROTATIONAL_THRESHOLD,
                "translational_max_abs_m_s2": _POINTWISE_TRANSLATIONAL_THRESHOLD,
            },
        },
        "open_loop_rollout": rollout_summary,
        "warnings": [
            "Measured external wrenches and inverse-dynamics torques are replayed; this is not predictive contact.",
            "Open-loop drift is diagnostic and is not hidden with tracking or residual controls.",
            f"The source gait artifact status is {manifest.get('status', 'unknown')}.",
        ],
    }

    staging = output_dir.parent / f".{output_dir.name}.staging-{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        write_json(staging / "torque_reconstruction_qc.json", report)
        np.savez_compressed(
            staging / "torque_reconstruction.npz",
            times=times,
            coordinate_names=np.asarray(coordinate_names, dtype="U"),
            motion_types=np.asarray(motion_types, dtype="U"),
            reference_accelerations=qdd_reference,
            reconstructed_accelerations=qdd_reconstructed,
            acceleration_error=acceleration_error,
            rollout_times=rollout.times,
            rollout_coordinates=rollout.coordinates,
            rollout_speeds=rollout.speeds,
            rollout_reference_coordinates=reference_q,
            rollout_reference_speeds=reference_qd,
            rollout_marker_error_m=marker_error,
        )
        backup = output_dir.parent / f".{output_dir.name}.previous"
        if backup.exists():
            shutil.rmtree(backup)
        if output_dir.exists():
            os.replace(output_dir, backup)
        try:
            os.replace(staging, output_dir)
        except Exception:
            if backup.exists() and not output_dir.exists():
                os.replace(backup, output_dir)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    print(f"[gait_c3d] wrote {output_dir}")
    print(
        "[gait_c3d] pointwise FD error "
        f"{pointwise['rotational']['max_abs']:.3e} rad/s^2, "
        f"{pointwise['translational']['max_abs']:.3e} m/s^2; "
        f"open-loop finite-prefix marker RMS "
        f"{rollout_summary['marker_error_m']['rms_over_finite_prefix']:.3f} m"
    )
    return output_dir


def create_parser() -> argparse.ArgumentParser:
    """Build the torque-reconstruction command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-only",
        action="store_true",
        help="required acknowledgement: this uses newton.opensim compatibility dynamics",
    )
    parser.add_argument("--data-dir", default=str(_DEFAULT_DATA))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rollout-dt", type=float, default=0.001)
    parser.add_argument("--trim-frames", type=int, default=10)
    return parser


def main() -> None:
    """Run the command-line torque-reconstruction diagnostic."""
    parser = create_parser()
    args = parser.parse_args()
    if not args.reference_only:
        parser.error("--reference-only is required; this is not a Newton-native rollout")
    run_reconstruction(
        args.data_dir,
        args.output_dir,
        device=args.device,
        rollout_dt=args.rollout_dt,
        trim_frames=args.trim_frames,
    )


if __name__ == "__main__":
    main()
