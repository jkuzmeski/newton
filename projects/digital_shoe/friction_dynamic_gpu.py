# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""GPU-batched FREE-leg friction study using frozen Cartesian Engine and controller.

Evaluates batches of candidate friction parameter rows residently on GPU across
multiple worlds. Reuses unchanged Cartesian Engine dynamics, controller gains,
and equilibrium spline from sealed baseline. All contact histories and evaluations
remain GPU-resident.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import warp as wp

from projects.digital_shoe.friction_leg import verify_baseline_inputs
from projects.digital_shoe.friction_metrics import (
    compute_braking_propulsive_impulses,
    compute_force_peaks,
)
from projects.digital_shoe.friction_parameter_adapter import (
    FrictionParameterAdapter,
)
from projects.digital_shoe.friction_sweep import SCORE_NAMES
from projects.digital_shoe.friction_sweep import _score as _score_curves
from projects.impedance_instron.cartesian import data
from projects.impedance_instron.cartesian.fit import FitConfig
from projects.impedance_instron.cartesian.gpu.engine import Engine
from projects.impedance_instron.cartesian.profile import load as load_profile
from projects.impedance_instron.cartesian.run import Config
from projects.impedance_instron.cartesian.trajectory import Spline

wp.set_module_options({"enable_backward": False})

DEFAULT_BASELINE_DIR = Path(
    os.environ.get(
        "NEWTON_BASELINE12_DIR",
        "outputs/impedance_instron/baseline12",
    )
)

BASE_PARAMETER_NAMES = (
    "method",
    "mu",
    "kt_scale",
    "kv_scale",
    "viscous_ratio",
    "release_dwell_s",
    "yield_width",
)
PARAMETER_NAMES = (
    *BASE_PARAMETER_NAMES,
    "mu_dynamic",
    "transition_speed",
    "pressure_scale_pa",
    "slip_scale_m",
    "shear_relaxation_time_s",
)


@wp.kernel
def _transpose_engine_forces_to_curves(
    forces: wp.array2d[wp.vec2d],
    curves: wp.array2d[wp.vec2],
):
    """Transpose Engine forces [T, W] to FrictionSweep curves [W, T] as float32."""
    w, s = wp.tid()
    f = forces[s, w]
    curves[w, s] = wp.vec2(wp.float32(f[0]), wp.float32(f[1]))


@wp.kernel
def _raw_peaks(curves: wp.array2d[wp.vec2], steps: int, diagnostics: wp.array[wp.vec3]):
    w = wp.tid()
    brake = float(0.0)
    prop = float(0.0)
    jump = float(0.0)
    previous = float(0.0)
    for t in range(steps):
        f = curves[w, t][0]
        brake = wp.max(brake, -f)
        prop = wp.max(prop, f)
        if t > 0:
            jump = wp.max(jump, wp.abs(f - previous))
        previous = f
    diagnostics[w] = wp.vec3(brake, prop, jump)


def parse_candidate_dict(cand: dict[str, Any]) -> list[float]:
    """Use the same strict candidate contract as independent CPU qualification."""
    from .friction_dynamic import parse_candidate  # noqa: PLC0415

    parameters, _metadata = parse_candidate(cand)
    return [float(getattr(parameters, name)) for name in PARAMETER_NAMES]


class FrictionDynamicGPUWorkspace:
    """Callable GPU-batched free-leg friction evaluation workspace.

    Reuses Cartesian Engine with frozen controller gains and frozen equilibrium
    spline. Scales candidate tangential stiffness and damping per world using
    FrictionParameterAdapter. Evaluates candidates concurrently on GPU without host roundtrips.

    Args:
        baseline_dir: Path to sealed baseline directory (e.g. baseline12).
        world_count: Number of parallel candidate worlds (e.g. 128 or 4).
        device: CUDA device (e.g. 'cuda:0').
        dt_scale: Timestep multiplier relative to baseline dt_s (default 1.0; 0.5 for halfstep).
        chunk_steps: Engine chunk capture size.
        matched_observation: Add signed, bandwidth-matched force scores at a CPU
            reporting boundary. Raw force and mechanics diagnostics are retained.
    """

    def __init__(
        self,
        baseline_dir: Path | str = DEFAULT_BASELINE_DIR,
        world_count: int = 128,
        device: str = "cuda:0",
        dt_scale: float = 1.0,
        chunk_steps: int = 32,
        matched_observation: bool = False,
    ):
        self.device = wp.get_device(device)
        if not self.device.is_cuda:
            raise ValueError("FrictionDynamicGPUWorkspace requires a CUDA device")

        base_p = Path(baseline_dir).resolve()
        if not base_p.is_dir():
            raise FileNotFoundError(f"Baseline directory not found: {base_p}")

        self.baseline_dir = base_p
        self.world_count = int(world_count)
        self.dt_scale = float(dt_scale)
        self.matched_observation = bool(matched_observation)

        self.input_hashes = verify_baseline_inputs(base_p)
        if not np.isfinite(self.dt_scale) or self.dt_scale <= 0:
            raise ValueError("dt_scale must be finite and positive")
        if isinstance(world_count, bool) or not isinstance(world_count, int) or world_count < 1:
            raise ValueError("world_count must be a positive integer")
        # Load sealed inputs
        self.reference = data.load(base_p / "reference.npz")
        self.profile = load_profile(base_p / "profile.json")

        with open(base_p / "summary.json", encoding="utf-8") as f:
            self.summary = json.load(f)

        with np.load(base_p / "equilibrium.npz", allow_pickle=False) as arch:
            self.duration_s = float(arch["duration_s"])
            self.coefficients_12 = arch["coefficients"]
            self.equilibrium_spline = Spline(self.duration_s, self.coefficients_12)

        mount_m = self.summary["shoe"]["mount_m"]
        static_pitch_rad = float(self.summary["shoe"]["static_pitch_rad"])

        sim_raw = self.summary.get("simulation_config", {})
        actual_dt = float(self.summary["run"]["actual_dt_s"]) * self.dt_scale
        self.sim_config = Config(
            dt_s=actual_dt,
            gravity_m_s2=float(sim_raw.get("gravity_m_s2", 9.81)),
            compression_limit=float(sim_raw.get("compression_limit", 0.9)),
            maximum_force_n=float(sim_raw.get("maximum_force_n", 6000.0)),
            minimum_hip_height_m=float(sim_raw.get("minimum_hip_height_m", 0.2)),
            maximum_speed=float(sim_raw.get("maximum_speed", 100.0)),
            joint_limits_diagnostic=bool(sim_raw.get("joint_limits_diagnostic", True)),
        )

        fit_raw = self.summary.get("fit_config", {})
        self.fit_config = FitConfig(**fit_raw) if fit_raw else FitConfig()

        # Initialize Cartesian Engine
        artifact_path = base_p / "digital_shoe.json"
        self.engine = Engine(
            self.reference,
            self.profile,
            artifact_path,
            mount_m,
            static_pitch_rad,
            config=self.sim_config,
            settings=self.fit_config,
            world_count=self.world_count,
            device=str(self.device),
            chunk_steps=chunk_steps,
        )

        # Attach FrictionParameterAdapter to Engine foundation
        self.adapter = FrictionParameterAdapter(
            self.engine.foundation,
            world_count=self.world_count,
        )

        # Frozen controller coefficients across all worlds
        self.frozen_coefficients = np.tile(self.coefficients_12[None, :, :], (self.world_count, 1, 1))

        # Setup scoring structures
        self._init_scoring()

    def _init_scoring(self) -> None:
        """Preallocate scoring buffers and interpolation lookups matching FrictionSweep."""
        times = self.engine.time_s[: self.engine.steps]
        self.steps = self.engine.steps
        self.dt = self.engine.dt

        measured_t = np.asarray(
            self.reference["grf_time_s"] if "grf_time_s" in self.reference else self.reference["time_s"],
            float,
        )
        measured_f = np.asarray(
            self.reference["grf_target_n"]
            if "grf_target_n" in self.reference
            else self.reference["ground_reaction_force_n"],
            float,
        )

        included = (measured_t >= times[0] - 1e-12) & (measured_t <= times[-1] + 1e-12)
        target_t = measured_t[included]
        target_f = measured_f[included, 0]
        # Normal force threshold 50 N defines active stance
        active = measured_f[included, 1] >= 50.0

        intervals = []
        for start in np.flatnonzero(active & ~np.r_[False, active[:-1]]):
            end = start
            while end + 1 < len(active) and active[end + 1]:
                end += 1
            intervals.append((int(start), int(end)))

        braking, propulsive, _ = compute_braking_propulsive_impulses(target_t, target_f, intervals)
        peaks = compute_force_peaks(target_t, target_f, intervals)
        self.targets = wp.vec4(
            braking,
            propulsive,
            peaks["braking_peak_magnitude_n"] or 0.0,
            peaks["propulsive_peak_magnitude_n"] or 0.0,
        )
        self.force_scale = max(float(np.max(np.abs(target_f))), 1.0)

        hi = np.clip(np.searchsorted(times, target_t, side="right"), 1, self.steps - 1)
        lo = hi - 1
        fraction = (target_t - times[lo]) / (times[hi] - times[lo])

        self.sample_count = len(target_t)
        self.target = wp.array(target_f, dtype=float, device=self.device)
        self.target_time = wp.array(target_t, dtype=float, device=self.device)
        self.active = wp.array(active.astype(np.int32), dtype=int, device=self.device)
        self.lower = wp.array(lo, dtype=int, device=self.device)
        self.upper = wp.array(hi, dtype=int, device=self.device)
        self.fraction = wp.array(fraction, dtype=float, device=self.device)

        # Transposed curve buffer [W, T] for primary friction scoring
        self.curves = wp.zeros((self.world_count, self.steps), dtype=wp.vec2, device=self.device)
        self.friction_scores = wp.zeros((self.world_count, len(SCORE_NAMES)), dtype=float, device=self.device)
        self.raw_peaks = wp.zeros(self.world_count, dtype=wp.vec3, device=self.device)
        # Fit at the simulation rate so high-frequency force chatter cannot hide
        # between native reference samples. Only the measured reference is interpolated.
        raw_target = np.interp(times, measured_t, measured_f[:, 0])
        raw_active = np.interp(times, measured_t, measured_f[:, 1]) >= 50.0
        self.fit_target = wp.array(raw_target, dtype=float, device=self.device)
        self.fit_time = wp.array(times, dtype=float, device=self.device)
        self.fit_active = wp.array(raw_active.astype(np.int32), dtype=int, device=self.device)
        self.fit_index = wp.array(np.arange(self.steps), dtype=int, device=self.device)
        self.fit_fraction = wp.zeros(self.steps, dtype=float, device=self.device)
        self.fit_scores = wp.zeros((self.world_count, len(SCORE_NAMES)), dtype=float, device=self.device)
        # Physical-force objective: never smooth simulation output. The pre-20Hz
        # reference is already acquisition-cleaned/Hann-filtered, not raw sensor data.
        pre = self.reference.get("unfiltered_grf_target_n")
        self.physical_target = None
        if pre is not None:
            pre = np.asarray(pre, dtype=float)
            target_raw = np.interp(times, measured_t, pre[:, 0])
            normal_raw = np.interp(times, measured_t, pre[:, 1])
            active_raw = normal_raw >= 50.0
            intervals_raw = []
            for start in np.flatnonzero(active_raw & ~np.r_[False, active_raw[:-1]]):
                end = start
                while end + 1 < len(active_raw) and active_raw[end + 1]:
                    end += 1
                intervals_raw.append((int(start), int(end)))
            b_raw, p_raw, _ = compute_braking_propulsive_impulses(times, target_raw, intervals_raw)
            peaks_raw = compute_force_peaks(times, target_raw, intervals_raw)
            self.physical_targets = wp.vec4(
                b_raw,
                p_raw,
                peaks_raw["braking_peak_magnitude_n"] or 0.0,
                peaks_raw["propulsive_peak_magnitude_n"] or 0.0,
            )
            self.physical_force_scale = max(float(np.max(np.abs(target_raw))), 1.0)
            self.physical_target = wp.array(target_raw, dtype=float, device=self.device)
            self.physical_active = wp.array(active_raw.astype(np.int32), dtype=int, device=self.device)
            self.physical_scores = wp.zeros((self.world_count, len(SCORE_NAMES)), dtype=float, device=self.device)

    def evaluate(
        self,
        parameters: np.ndarray | list,
        *,
        curves: bool = False,
    ) -> dict[str, Any]:
        """Evaluate candidate parameter rows on GPU.

        Args:
            parameters: Candidate rows of shape (K, 7) or (K, 9).
            curves: Include per-world tangential/normal GRF curves [N], shape (K, T, 2).
                Samples after each world's recorded force support are NaN.

        Returns:
            Dictionary containing:
                - friction_scores: dict mapping score name to array of shape (K,)
                - fit_scores: full simulation-rate friction metrics
                - engine_loss, engine_rmse: original six-channel leg-fit metrics
                - failure_code, integrated_steps, recorded_steps: rollout diagnostics
                - status: list of completion status strings ("completed" / "failed")
                - completed: boolean array of shape (K,)
                - curves: optional array of shape (K, T, 2)
        """
        arr = np.asarray(parameters, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        k = len(arr)
        if k > self.world_count:
            raise ValueError(f"Batch candidate count ({k}) exceeds world count ({self.world_count})")

        # Set candidate parameters into adapter
        self.adapter.set_parameters(arr)

        # Capture once; only friction parameters change between evaluations.
        if self.engine.graph is None:
            self.engine.capture(self.frozen_coefficients)
        started = perf_counter()
        self.engine.evaluate_device()

        # Transpose Engine forces [T, W] -> curves [W, T]
        wp.launch(
            _transpose_engine_forces_to_curves,
            dim=(self.world_count, self.steps),
            inputs=[self.engine.forces, self.curves],
            device=self.device,
        )

        # Score friction curves and diagnostics
        wp.launch(
            _score_curves,
            dim=self.world_count,
            inputs=[
                self.sample_count,
                self.target,
                self.target_time,
                self.active,
                self.lower,
                self.upper,
                self.fraction,
                self.targets,
                self.force_scale,
                self.curves,
                self.adapter.totals,
                self.friction_scores,
            ],
            device=self.device,
        )

        wp.launch(
            _score_curves,
            dim=self.world_count,
            inputs=[
                self.steps,
                self.fit_target,
                self.fit_time,
                self.fit_active,
                self.fit_index,
                self.fit_index,
                self.fit_fraction,
                self.targets,
                self.force_scale,
                self.curves,
                self.adapter.totals,
                self.fit_scores,
            ],
            device=self.device,
        )
        if self.physical_target is not None:
            wp.launch(
                _score_curves,
                dim=self.world_count,
                inputs=[
                    self.steps,
                    self.physical_target,
                    self.fit_time,
                    self.physical_active,
                    self.fit_index,
                    self.fit_index,
                    self.fit_fraction,
                    self.physical_targets,
                    self.physical_force_scale,
                    self.curves,
                    self.adapter.totals,
                    self.physical_scores,
                ],
                device=self.device,
            )
        wp.launch(
            _raw_peaks, dim=self.world_count, inputs=[self.curves, self.steps, self.raw_peaks], device=self.device
        )
        # Do not score stale force tails from a failed candidate as a full stance.
        raw_friction_scores = self.friction_scores.numpy()[:k].copy()
        engine_scores = self.engine.objective.read()
        integrated = self.engine.integrated.numpy()[:k].copy()
        failure_codes = self.engine.failure.numpy()[:k].copy()
        recorded = self.engine.recorded.numpy()[:k].copy()
        completed = (integrated == self.steps) & (failure_codes == 0)
        raw_friction_scores[~completed] = np.nan
        raw_friction_scores[~completed, 0] = np.inf
        friction_dict = {name: raw_friction_scores[:, idx] for idx, name in enumerate(SCORE_NAMES)}
        raw_fit_scores = self.fit_scores.numpy()[:k].copy()
        raw_fit_scores[~completed] = np.nan
        raw_fit_scores[~completed, 0] = np.inf
        fit_dict = {name: raw_fit_scores[:, idx] for idx, name in enumerate(SCORE_NAMES)}
        raw_diagnostics = self.raw_peaks.numpy()[:k].copy()
        raw_diagnostics[~completed] = np.nan
        self.last_seconds = perf_counter() - started
        status = ["completed" if c else "failed" for c in completed]

        result: dict[str, Any] = {
            "friction_scores": friction_dict,
            "fit_scores": fit_dict,
            "engine_loss": engine_scores["loss"][:k].copy(),
            "engine_rmse": engine_scores["rmse"][:k].copy(),
            "integrated_steps": integrated.copy(),
            "recorded_steps": recorded,
            "failure_code": failure_codes.copy(),
            "completed": completed,
            "raw_force_diagnostics": raw_diagnostics,
            "status": status,
        }

        if self.physical_target is not None:
            raw_physical = self.physical_scores.numpy()[:k].copy()
            raw_physical[~completed] = np.nan
            raw_physical[~completed, 0] = np.inf
            result["physical_scores"] = {name: raw_physical[:, idx] for idx, name in enumerate(SCORE_NAMES)}

        exported_curves = None
        if curves or self.matched_observation:
            exported_curves = self.curves.numpy()[:k].copy()
            for world, count in enumerate(recorded):
                exported_curves[world, int(count) :] = np.nan
        if self.matched_observation:
            # This reporting boundary runs on CPU; physical rollout remains GPU-resident.
            from .friction_metrics import score_observed_forces  # noqa: PLC0415

            observed_scores = raw_fit_scores.copy()
            observed_rmse = np.full((k, 2), np.nan)
            metadata = None
            if np.any(completed):
                observed = score_observed_forces(
                    self.reference, self.engine.time_s[: self.steps], exported_curves[completed]
                )
                observed_scores[completed, :6] = observed["scores"]
                observed_rmse[completed] = observed["force_rmse_n"]
                metadata = observed["observation"]["metadata"]
            result["observation_scores"] = {name: observed_scores[:, idx] for idx, name in enumerate(SCORE_NAMES)}
            result["observation_force_rmse_n"] = observed_rmse
            result["observation_metadata"] = metadata
        if curves:
            result["curves"] = exported_curves
        self.last_seconds = perf_counter() - started
        return result

    def trace(self, world: int = 0) -> tuple[dict[str, Any], dict[str, Any]]:
        """Extract full detailed trace and summary for a single completed world."""
        return self.engine.trace(world)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(description="GPU-batched FREE-leg friction study using frozen Cartesian Engine.")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE_DIR,
        help=f"Path to sealed baseline directory (default: {DEFAULT_BASELINE_DIR}).",
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        default=None,
        help="Path to single candidate JSON file.",
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=None,
        help="Path to JSON file or directory containing candidate definitions.",
    )
    parser.add_argument(
        "--parameter-table",
        type=Path,
        default=None,
        help="Path to parameter table (npy, npz, or json).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to output summary JSON or directory.",
    )
    parser.add_argument(
        "--worlds",
        type=int,
        default=128,
        help="Number of worlds to allocate in GPU batch (default: 128).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="CUDA device to execute on (default: cuda:0).",
    )
    parser.add_argument(
        "--dt-scale",
        type=float,
        default=1.0,
        help="Timestep scaling multiplier (default: 1.0; 0.5 for halfstep).",
    )
    parser.add_argument(
        "--legacy-full-rate",
        action="store_true",
        help="Disable matched observation reporting; preserve historical diagnostics only.",
    )
    parser.add_argument(
        "--matched-observations",
        action="store_true",
        help="Add secondary filtered observation metrics; unfiltered physical force stays primary.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    out_p = Path(args.output).resolve()
    if out_p.exists():
        raise FileExistsError(f"Output already exists: {out_p}")
    # Collect candidate parameter rows
    rows: list[list[float]] = []
    meta: list[dict[str, Any]] = []

    if args.candidate is not None:
        with open(args.candidate, encoding="utf-8") as f:
            cand = json.load(f)
        rows.append(parse_candidate_dict(cand))
        meta.append({"source": str(args.candidate)})
    elif args.candidates is not None:
        p = Path(args.candidates)
        if p.is_dir():
            for fpath in sorted(p.glob("*.json")):
                with open(fpath, encoding="utf-8") as f:
                    c = json.load(f)
                rows.append(parse_candidate_dict(c))
                meta.append({"source": str(fpath)})
        else:
            with open(p, encoding="utf-8") as f:
                c = json.load(f)
            if isinstance(c, list):
                for item in c:
                    rows.append(parse_candidate_dict(item))
                    meta.append({"source": str(p)})
            else:
                rows.append(parse_candidate_dict(c))
                meta.append({"source": str(p)})
    elif args.parameter_table is not None:
        p = Path(args.parameter_table)
        if p.suffix == ".npy":
            arr = np.load(p)
            for row in arr:
                rows.append(row.tolist())
                meta.append({"source": str(p)})
        elif p.suffix == ".npz":
            with np.load(p) as arch:
                arr = arch["parameters"]
                for row in arr:
                    rows.append(row.tolist())
                    meta.append({"source": str(p)})
        else:
            with open(p, encoding="utf-8") as f:
                data_in = json.load(f)
            for item in data_in:
                rows.append(parse_candidate_dict(item))
                meta.append({"source": str(p)})
    else:
        # Default baseline candidate
        rows.append([0.0, 0.8, 1.0, 1.0, 0.2, 0.0005, 0.0])
        meta.append({"source": "default_baseline"})

    rows = [
        [*row, row[1], 0.1, 1.0e12, 0.001, 0.005]
        if len(row) == 7
        else [*row, 1.0e12, 0.001, 0.005]
        if len(row) == 9
        else [*row, 0.001, 0.005]
        if len(row) == 10
        else [*row, 0.005]
        if len(row) == 11
        else row
        for row in rows
    ]
    world_count = max(len(rows), args.worlds)
    workspace = FrictionDynamicGPUWorkspace(
        baseline_dir=args.baseline,
        world_count=world_count,
        device=args.device,
        dt_scale=args.dt_scale,
        matched_observation=args.matched_observations and not args.legacy_full_rate,
    )

    results = workspace.evaluate(np.array(rows, dtype=np.float32), curves=True)

    out_p = Path(args.output).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    summary_out = {
        "status": "completed" if all(results["completed"]) else "partially_failed",
        "candidate_count": len(rows),
        "force_score_policy": "raw" if not args.matched_observations else "raw-with-secondary-observations",
        "observation_metadata": results.get("observation_metadata"),
        "world_count": world_count,
        "dt_scale": args.dt_scale,
        "input_hashes": workspace.input_hashes,
        "controller": "Frozen baseline equilibrium coefficients and gains; friction parameters only.",
        "candidates": [
            {
                "index": i,
                "metadata": meta[i],
                "parameters": {name: float(rows[i][j]) for j, name in enumerate(PARAMETER_NAMES)},
                "status": results["status"][i],
                "completed": bool(results["completed"][i]),
                "failure_code": int(results["failure_code"][i]),
                "friction_scores": {
                    k: float(v[i]) if np.isfinite(v[i]) else None for k, v in results["friction_scores"].items()
                },
                "fit_scores": {k: float(v[i]) if np.isfinite(v[i]) else None for k, v in results["fit_scores"].items()},
                "engine_loss": float(results["engine_loss"][i]) if np.isfinite(results["engine_loss"][i]) else None,
                "physical_scores": {
                    k: float(v[i]) if np.isfinite(v[i]) else None for k, v in results.get("physical_scores", {}).items()
                },
                "observation_scores": {
                    k: float(v[i]) if np.isfinite(v[i]) else None
                    for k, v in results.get("observation_scores", {}).items()
                },
                "recorded_steps": int(results["recorded_steps"][i]),
                "integrated_steps": int(results["integrated_steps"][i]),
            }
            for i in range(len(rows))
        ],
    }

    if out_p.is_dir() or out_p.suffix == "":
        out_p.mkdir(parents=True, exist_ok=True)
        summary_file = out_p / "summary.json"
        np.savez_compressed(
            out_p / "curves.npz",
            curves=results["curves"],
            time_s=workspace.engine.time_s[: workspace.steps],
            recorded_steps=results["recorded_steps"],
        )
    else:
        summary_file = out_p

    if verify_baseline_inputs(workspace.baseline_dir) != workspace.input_hashes:
        raise RuntimeError("Baseline inputs changed during GPU evaluation")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_out, f, indent=2, allow_nan=False)

    print(f"Evaluated {len(rows)} candidates on GPU (world_count={world_count}). Summary: {summary_file}")


if __name__ == "__main__":
    main()
