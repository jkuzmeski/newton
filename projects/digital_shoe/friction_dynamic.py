# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Frozen-controller free-leg dynamic qualification of shoe friction candidates.

This module evaluates candidate friction models on a forward-simulated single leg
under prescribed Cartesian and joint impedance control, without refitting the
controller or altering normal contact mechanics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

from projects.digital_shoe.friction_adapter import FrictionAdapter
from projects.digital_shoe.friction_leg import DEFAULT_BASELINE_MANIFEST, verify_baseline_inputs
from projects.digital_shoe.friction_metrics import score_friction_trace, score_observed_forces
from projects.impedance_instron.cartesian import data
from projects.impedance_instron.cartesian.fit import FitConfig, _Objective
from projects.impedance_instron.cartesian.profile import load as load_profile
from projects.impedance_instron.cartesian.run import Config, simulate
from projects.impedance_instron.cartesian.shoe import Shoe
from projects.impedance_instron.cartesian.trajectory import Spline

DEFAULT_BASELINE_DIR = Path(
    os.environ.get(
        "NEWTON_BASELINE12_DIR",
        "outputs/impedance_instron/baseline12",
    )
)

TRACKED_SOURCE_FILES = (
    "projects/digital_shoe/contact.py",
    "projects/digital_shoe/friction_law.py",
    "projects/digital_shoe/friction_deflection.py",
    "projects/digital_shoe/friction_stribeck.py",
    "projects/digital_shoe/friction_pressure.py",
    "projects/digital_shoe/friction_slip_history.py",
    "projects/digital_shoe/friction_maxwell.py",
    "projects/digital_shoe/friction_parameter_adapter.py",
    "projects/digital_shoe/friction_solver.py",
    "projects/digital_shoe/friction_adapter.py",
    "projects/digital_shoe/friction_metrics.py",
    "projects/digital_shoe/friction_observation.py",
    "projects/digital_shoe/friction_dynamic.py",
    "projects/digital_shoe/runtime.py",
    "projects/digital_shoe/material.py",
    "projects/digital_shoe/artifact.py",
    "projects/impedance_instron/cartesian/run.py",
    "projects/impedance_instron/cartesian/shoe.py",
    "projects/impedance_instron/cartesian/mechanics.py",
    "projects/impedance_instron/cartesian/profile.py",
    "projects/impedance_instron/cartesian/data.py",
    "projects/impedance_instron/cartesian/trajectory.py",
    "projects/impedance_instron/cartesian/fit.py",
)


@dataclass(frozen=True)
class CandidateParameters:
    """Declared candidate friction parameters from identification sweeps.

    Args:
        method: Friction formulation: 0 legacy, 1 consistent deflection, or 4 Stribeck deflection.
        mu: Effective Coulomb friction coefficient [-].
        kt_scale: Multiplier on baseline tangential column stiffness [-].
        kv_scale: Multiplier on baseline tangential column damping [-].
        viscous_ratio: Viscous damping cap as a fraction of Coulomb cone [-].
        release_dwell_s: Unloaded dwell before releasing tangential stick point [s].
        yield_width: Experimental yield shoulder width fraction; must be exactly 0.0.
        mu_dynamic: High-speed coefficient for Stribeck; defaults to mu.
        transition_speed: Stribeck transition speed [m/s].
    """

    method: int
    mu: float
    kt_scale: float
    kv_scale: float
    viscous_ratio: float
    release_dwell_s: float
    yield_width: float = 0.0
    mu_dynamic: float | None = None
    transition_speed: float = 0.1
    pressure_scale_pa: float = 1.0e12
    slip_scale_m: float = 0.001
    shear_relaxation_time_s: float = 0.005

    def __post_init__(self) -> None:
        """Validate candidate parameters against supported numerical contracts."""
        if isinstance(self.method, bool):
            raise TypeError("method cannot be a boolean")
        if not isinstance(self.method, int):
            raise TypeError(f"method must be an integer, got {type(self.method)}")

        if self.method not in (0, 1, 4, 5, 6, 7):
            if self.method in (2, 3):
                name = "regularized" if self.method == 2 else "anchor_nominal"
                raise ValueError(
                    f"Method {self.method} ({name}) is an unsupported diagnostic formulation. "
                    "Supported methods are 0 legacy, 1 deflection, 4 Stribeck, 5 pressure, 6 slip history and 7 Maxwell."
                )
            raise ValueError(f"Unknown friction method: {self.method}; must be one of 0, 1, 4, 5, 6, 7")

        if isinstance(self.yield_width, bool):
            raise TypeError("yield_width cannot be a boolean")
        if (
            not isinstance(self.yield_width, (int, float))
            or not np.isfinite(self.yield_width)
            or float(self.yield_width) != 0.0
        ):
            raise ValueError(
                f"yield_width must be exactly 0.0, got {self.yield_width}. Nonzero yield width produces "
                "spurious numerical relaxation creep at zero slip velocity and is unphysical for qualification."
            )

        for name, value in (
            ("mu", self.mu),
            ("kt_scale", self.kt_scale),
            ("kv_scale", self.kv_scale),
            ("viscous_ratio", self.viscous_ratio),
            ("release_dwell_s", self.release_dwell_s),
        ):
            if isinstance(value, bool):
                raise TypeError(f"{name} cannot be a boolean")
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative, got {value}")
        if self.kt_scale <= 0.0:
            raise ValueError("kt_scale must be positive")
        if self.mu_dynamic is None:
            object.__setattr__(self, "mu_dynamic", self.mu)
        if not np.isfinite(self.mu_dynamic) or not 0 <= self.mu_dynamic <= self.mu:
            raise ValueError("mu_dynamic must be finite and between zero and mu")
        if not np.isfinite(self.transition_speed) or self.transition_speed < 1e-6:
            raise ValueError("transition_speed must be finite and at least 1e-6 m/s")
        if (
            isinstance(self.pressure_scale_pa, bool)
            or not np.isfinite(self.pressure_scale_pa)
            or self.pressure_scale_pa <= 0
        ):
            raise ValueError("pressure_scale_pa must be finite and positive")
        if isinstance(self.slip_scale_m, bool) or not np.isfinite(self.slip_scale_m) or self.slip_scale_m <= 0:
            raise ValueError("slip_scale_m must be finite and positive")
        if (
            isinstance(self.shear_relaxation_time_s, bool)
            or not np.isfinite(self.shear_relaxation_time_s)
            or self.shear_relaxation_time_s <= 0
        ):
            raise ValueError("shear_relaxation_time_s must be finite and positive")
        if self.method == 7 and self.viscous_ratio != 0.0:
            raise ValueError(
                "Maxwell bristles use internal viscosity, not parallel cone-limited damping; set viscous_ratio=0"
            )


def sha256_file(path: Path | str) -> str:
    """Compute sha256 hex digest of an on-disk file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_source_hashes(root: Path | None = None) -> dict[str, str]:
    """Compute sha256 hashes of all required tracked source files.

    Raises:
        FileNotFoundError: If any required source file is missing.
    """
    if root is None:
        curr = Path(__file__).resolve().parent
        while curr.parent != curr and not (curr / "projects").is_dir():
            curr = curr.parent
        root = curr

    hashes: dict[str, str] = {}
    for rel_path in TRACKED_SOURCE_FILES:
        target = root / rel_path
        if not target.exists():
            raise FileNotFoundError(f"Required source file missing: {target}")
        hashes[rel_path] = sha256_file(target)
    return hashes


def parse_candidate(
    candidate_source: dict[str, Any] | Path | str | None,
) -> tuple[CandidateParameters, dict[str, Any]]:
    """Parse and validate candidate friction parameters.

    Args:
        candidate_source: JSON file path, dictionary, or None (default baseline).

    Returns:
        Tuple of (CandidateParameters, metadata_dict).
    """
    if candidate_source is None:
        params = CandidateParameters(
            method=0,
            mu=0.8,
            kt_scale=1.0,
            kv_scale=1.0,
            viscous_ratio=0.2,
            release_dwell_s=0.0005,
            yield_width=0.0,
        )
        return params, {"source": "default_baseline", "sha256": None}

    file_meta: dict[str, Any] = {"source": "in_memory_dict", "sha256": None}
    raw: dict[str, Any]
    if isinstance(candidate_source, (str, Path)):
        p = Path(candidate_source).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Candidate JSON file not found: {p}")
        file_meta = {"source": str(p), "sha256": sha256_file(p)}
        with open(p, encoding="utf-8") as f:
            raw = json.load(f)
    elif isinstance(candidate_source, dict):
        raw = candidate_source
    else:
        raise TypeError(f"candidate_source must be dict, Path, str, or None; got {type(candidate_source)}")

    params_dict = raw.get("parameters", raw)

    raw_method = params_dict.get("method", 0)
    if isinstance(raw_method, bool):
        raise TypeError("method cannot be a boolean")
    if isinstance(raw_method, str):
        m_lower = raw_method.lower().strip()
        if m_lower == "legacy":
            method = 0
        elif m_lower == "deflection":
            method = 1
        elif m_lower in ("stribeck", "deflection_stribeck"):
            method = 4
        elif m_lower in ("pressure", "deflection_pressure"):
            method = 5
        elif m_lower in ("slip_history", "cold_hot"):
            method = 6
        elif m_lower in ("maxwell", "viscoelastic"):
            method = 7
        elif m_lower in ("regularized", "anchor_nominal"):
            raise ValueError(
                f"Method {raw_method!r} is an unsupported diagnostic formulation. "
                "Supported methods are 0 legacy, 1 deflection, 4 Stribeck, 5 pressure, 6 slip history and 7 Maxwell."
            )
        else:
            raise ValueError(f"Unknown friction method name: {raw_method!r}")
    elif isinstance(raw_method, (int, float)):
        if not float(raw_method).is_integer():
            raise ValueError(f"method must be an exact integer, got fractional value {raw_method}")
        method = int(raw_method)
    else:
        raise TypeError(f"Invalid type for method: {type(raw_method)}")

    raw_yield = params_dict.get("yield_width", 0.0)
    if isinstance(raw_yield, bool):
        raise TypeError("yield_width cannot be a boolean")
    if not isinstance(raw_yield, (int, float)) or not np.isfinite(raw_yield) or float(raw_yield) != 0.0:
        raise ValueError(f"yield_width must be exactly 0.0, got {raw_yield}")
    yield_width = float(raw_yield)

    for name in (
        "mu",
        "kt_scale",
        "kv_scale",
        "viscous_ratio",
        "release_dwell_s",
        "mu_dynamic",
        "transition_speed",
        "pressure_scale_pa",
        "slip_scale_m",
        "shear_relaxation_time_s",
    ):
        if isinstance(params_dict.get(name), bool):
            raise TypeError(f"{name} cannot be a boolean")
    mu = float(params_dict.get("mu", 0.8))
    kt_scale = float(params_dict.get("kt_scale", 1.0))
    kv_scale = float(params_dict.get("kv_scale", 1.0))
    viscous_ratio = float(params_dict.get("viscous_ratio", 0.2))
    release_dwell_s = float(params_dict.get("release_dwell_s", 0.0005))

    params = CandidateParameters(
        method=method,
        mu=mu,
        kt_scale=kt_scale,
        kv_scale=kv_scale,
        viscous_ratio=viscous_ratio,
        release_dwell_s=release_dwell_s,
        yield_width=yield_width,
        mu_dynamic=float(params_dict.get("mu_dynamic", mu)),
        transition_speed=float(params_dict.get("transition_speed", 0.1)),
        pressure_scale_pa=float(params_dict.get("pressure_scale_pa", 1.0e12)),
        slip_scale_m=float(params_dict.get("slip_scale_m", 0.001)),
        shear_relaxation_time_s=float(params_dict.get("shear_relaxation_time_s", 0.005)),
    )
    return params, file_meta


def configure_candidate_friction(shoe: Shoe, candidate: CandidateParameters) -> None:
    """Apply candidate friction parameters to a shoe foundation.

    For legacy (method 0), scales foundation friction_kt and friction_kv arrays
    and updates world_params (mu, viscous_ratio, release_dwell_s) without attaching
    an adapter.
    For deflection (method 1), applies the same scalings and updates and attaches
    a FrictionAdapter with mode='deflection', zero mobility, and yield_width=0.0.
    Normal constants, normal law, and material coefficients remain untouched.

    Args:
        shoe: Shoe instance to configure.
        candidate: CandidateParameters declaring friction settings.
    """
    f = shoe.foundation

    if not hasattr(f, "_unscaled_friction_kt"):
        f._unscaled_friction_kt = f.friction_kt.numpy().copy()
        f._unscaled_friction_kv = f.friction_kv.numpy().copy()

    base_kt = f._unscaled_friction_kt
    base_kv = f._unscaled_friction_kv

    if getattr(f, "friction_solver", None) is not None:
        f.friction_solver.detach()

    f.friction_kt.assign(base_kt * float(candidate.kt_scale))
    f.friction_kv.assign(base_kv * float(candidate.kv_scale))

    for block in f.world_blocks:
        block.mu = float(candidate.mu)
        block.friction_viscous_ratio = float(candidate.viscous_ratio)
        block.friction_release_dwell_s = float(candidate.release_dwell_s)
    f.world_params.assign(f.world_blocks)

    if candidate.method == 0:
        f.friction_solver = None
    elif candidate.method == 1:
        mobility = wp.zeros(f.world_count, dtype=wp.spatial_matrix, device=shoe.device)
        FrictionAdapter(f, mobility=mobility, mode="deflection", yield_width=0.0)
    elif candidate.method in (4, 5, 6, 7):
        from .friction_parameter_adapter import FrictionParameterAdapter  # noqa: PLC0415

        adapter = FrictionParameterAdapter(f, f.world_count, base_kt=base_kt, base_kv=base_kv)
        adapter.set_parameters(
            [
                [
                    candidate.method,
                    candidate.mu,
                    candidate.kt_scale,
                    candidate.kv_scale,
                    candidate.viscous_ratio,
                    candidate.release_dwell_s,
                    0.0,
                    candidate.mu_dynamic,
                    candidate.transition_speed,
                    candidate.pressure_scale_pa,
                    candidate.slip_scale_m,
                    candidate.shear_relaxation_time_s,
                ]
            ]
        )
    else:
        raise ValueError(f"Unsupported candidate method: {candidate.method}")


def load_baseline_bundle(
    baseline_dir: Path | str,
    manifest_path: Path | str = DEFAULT_BASELINE_MANIFEST,
    device: str = "cpu",
) -> tuple[dict[str, Any], dict[str, Any], Spline, Shoe, Config, dict[str, str], dict[str, Any], float]:
    """Verify sealed inputs and load reference, profile, spline, shoe, and simulation config.

    Args:
        baseline_dir: Directory containing baseline files (e.g. baseline12).
        manifest_path: Path to baseline manifest declaring expected sha256 digests.
        device: Device to initialize shoe on (default: cpu).

    Returns:
        Tuple of (reference, profile, spline, shoe, sim_config, input_hashes, summary_dict, base_actual_dt_s).
    """
    base_p = Path(baseline_dir).resolve()
    if not base_p.is_dir():
        raise FileNotFoundError(f"Baseline directory not found: {base_p}")

    # Byte-for-byte check against pinned sealed baseline manifest
    verified_hashes = verify_baseline_inputs(base_p, manifest_path=manifest_path)

    ref = data.load(base_p / "reference.npz")
    prof = load_profile(base_p / "profile.json")

    with np.load(base_p / "equilibrium.npz", allow_pickle=False) as arch:
        spline = Spline(float(arch["duration_s"]), arch["coefficients"])

    with open(base_p / "summary.json", encoding="utf-8") as f:
        summary_raw = json.load(f)

    mount_m = summary_raw["shoe"]["mount_m"]
    static_pitch_rad = float(summary_raw["shoe"]["static_pitch_rad"])
    shoe = Shoe(base_p / "digital_shoe.json", mount_m, static_pitch_rad, device=device)

    # Use saved actual_dt_s exactly
    run_block = summary_raw.get("run", {})
    if "actual_dt_s" in run_block:
        base_actual_dt_s = float(run_block["actual_dt_s"])
    elif "actual_dt_s" in summary_raw:
        base_actual_dt_s = float(summary_raw["actual_dt_s"])
    else:
        raise KeyError("Baseline summary.json missing required actual_dt_s in run block")

    sim_cfg_raw = summary_raw.get("simulation_config")
    if not sim_cfg_raw:
        raise KeyError("Baseline summary.json missing required simulation_config")

    sim_config = Config(
        dt_s=base_actual_dt_s,
        gravity_m_s2=float(sim_cfg_raw["gravity_m_s2"]),
        compression_limit=float(sim_cfg_raw["compression_limit"]),
        maximum_force_n=float(sim_cfg_raw["maximum_force_n"]),
        minimum_hip_height_m=float(sim_cfg_raw["minimum_hip_height_m"]),
        maximum_speed=float(sim_cfg_raw["maximum_speed"]),
        joint_limits_diagnostic=bool(sim_cfg_raw["joint_limits_diagnostic"]),
    )

    return ref, prof, spline, shoe, sim_config, verified_hashes, summary_raw, base_actual_dt_s


def _finite_report(value):
    """Represent unavailable numeric diagnostics as JSON null, not NaN or infinity."""
    if isinstance(value, dict):
        return {key: _finite_report(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_report(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def run_dynamic_qualification(
    baseline_dir: Path | str = DEFAULT_BASELINE_DIR,
    candidate_source: dict[str, Any] | Path | str | None = None,
    output_dir: Path | str | None = None,
    manifest_path: Path | str = DEFAULT_BASELINE_MANIFEST,
    device: str = "cpu",
    dt_scale: float = 1.0,
    raise_on_failure: bool = True,
    include_observations: bool = False,
) -> dict[str, Any]:
    """Execute dynamic qualification comparing a candidate against the baseline.

    Args:
        baseline_dir: Path to baseline directory containing sealed inputs.
        candidate_source: Candidate parameters (Path, dict, or None for baseline).
        output_dir: Output directory to write traces and summary.json (MUST NOT exist).
        manifest_path: Path to sealed baseline manifest JSON.
        device: Device to simulate on (default: cpu).
        dt_scale: Timestep multiplier relative to baseline actual_dt_s (e.g. 1.0 or 0.5).
        raise_on_failure: Whether to raise RuntimeError if qualification fails.
        include_observations: Include secondary filtered observation metrics. Raw
            force remains primary and physical integration is never filtered.

    Returns:
        Comprehensive qualification report dictionary.
    """
    if dt_scale <= 0.0 or not np.isfinite(dt_scale):
        raise ValueError("dt_scale must be positive and finite")

    # Output directory must not exist and is validated BEFORE simulations
    out_p: Path | None = None
    if output_dir is not None:
        out_p = Path(output_dir).resolve()
        if out_p.exists():
            raise FileExistsError(f"Output directory already exists: {out_p}")
        out_p.parent.mkdir(parents=True, exist_ok=True)

    candidate_params, candidate_meta = parse_candidate(candidate_source)

    # Recheck source files before run (must raise if any missing)
    source_hashes_before = compute_source_hashes()

    ref, prof, spline, shoe, base_cfg, input_hashes, summary_raw, base_actual_dt_s = load_baseline_bundle(
        baseline_dir, manifest_path=manifest_path, device=device
    )

    fit_cfg_raw = summary_raw.get("fit_config", {})
    fit_config = FitConfig(**fit_cfg_raw) if fit_cfg_raw else FitConfig()

    actual_dt_s = base_actual_dt_s * dt_scale
    sim_config = Config(
        dt_s=actual_dt_s,
        gravity_m_s2=base_cfg.gravity_m_s2,
        compression_limit=base_cfg.compression_limit,
        maximum_force_n=base_cfg.maximum_force_n,
        minimum_hip_height_m=base_cfg.minimum_hip_height_m,
        maximum_speed=base_cfg.maximum_speed,
        joint_limits_diagnostic=base_cfg.joint_limits_diagnostic,
    )

    # 1. Run baseline simulation fresh
    shoe.foundation.reset()
    trace_base, sum_base = simulate(ref, prof, spline, shoe, config=sim_config)

    # 2. Configure candidate on a fresh shoe instance and simulate
    mount_m = summary_raw["shoe"]["mount_m"]
    static_pitch_rad = float(summary_raw["shoe"]["static_pitch_rad"])
    shoe_cand = Shoe(Path(baseline_dir) / "digital_shoe.json", mount_m, static_pitch_rad, device=device)
    configure_candidate_friction(shoe_cand, candidate_params)
    shoe_cand.foundation.reset()
    trace_cand, sum_cand = simulate(ref, prof, spline, shoe_cand, config=sim_config)

    # 3. Score horizontal GRF metrics with friction_metrics
    score_base = score_friction_trace(ref, trace_base, forward_sign=1, normal_threshold_n=50.0, summary=sum_base)
    score_cand = score_friction_trace(ref, trace_cand, forward_sign=1, normal_threshold_n=50.0, summary=sum_cand)

    # 4. Score six-channel leg metrics with cartesian fit Objective
    obj = _Objective(ref, fit_config)
    res_base, six_base, costs_base = obj.evaluate(trace_base, sum_base)
    res_cand, six_cand, costs_cand = obj.evaluate(trace_cand, sum_cand)

    # Recheck source and input hashes AFTER simulation run
    source_hashes_after = compute_source_hashes()
    if source_hashes_before != source_hashes_after:
        raise RuntimeError("Tracked source files were modified during dynamic qualification!")

    verified_inputs_after = verify_baseline_inputs(baseline_dir, manifest_path=manifest_path)
    if input_hashes != verified_inputs_after:
        raise RuntimeError("Baseline input files were modified during dynamic qualification!")

    # Check rollout completion and completeness flags
    base_complete = bool(score_base.get("complete") and res_base is not None and sum_base.get("status") == "completed")
    cand_complete = bool(score_cand.get("complete") and res_cand is not None and sum_cand.get("status") == "completed")
    overall_complete = bool(base_complete and cand_complete)

    failure_reasons = []
    if not base_complete:
        failure_reasons.append(f"Baseline simulation failed: {sum_base.get('failure')}")
    if not cand_complete:
        failure_reasons.append(f"Candidate simulation failed: {sum_cand.get('failure')}")

    # Compute comparison metrics
    diff_six_metrics: dict[str, list[float]] = {}
    if six_base and six_cand:
        for k in six_base:
            if k in six_cand:
                diff_six_metrics[k] = (np.array(six_cand[k]) - np.array(six_base[k])).tolist()

    max_normal_diff_n = None
    mean_normal_diff_n = None
    if len(trace_base.get("grf_n", [])) > 0 and len(trace_cand.get("grf_n", [])) > 0:
        min_len = min(len(trace_base["grf_n"]), len(trace_cand["grf_n"]))
        norm_diff = trace_cand["grf_n"][:min_len, 1] - trace_base["grf_n"][:min_len, 1]
        max_normal_diff_n = float(np.max(np.abs(norm_diff)))
        mean_normal_diff_n = float(np.mean(np.abs(norm_diff)))

    report: dict[str, Any] = {
        "schema": "digital_shoe_friction_dynamic_qualification_1",
        "status": "completed" if overall_complete else "failed",
        "complete": overall_complete,
        "failure_reason": "; ".join(failure_reasons) if failure_reasons else None,
        "dt_scale": dt_scale,
        "actual_dt_s": actual_dt_s,
        "baseline_actual_dt_s": base_actual_dt_s,
        "simulation_config": asdict(sim_config),
        "source_hashes": source_hashes_after,
        "input_hashes": verified_inputs_after,
        "candidate": {
            "source": candidate_meta.get("source"),
            "sha256": candidate_meta.get("sha256"),
            "parameters": asdict(candidate_params),
        },
        "baseline": {
            "status": sum_base.get("status"),
            "complete": base_complete,
            "integrated_steps": sum_base.get("integrated_steps"),
            "integrated_duration_s": sum_base.get("integrated_duration_s"),
            "actual_dt_s": sum_base.get("actual_dt_s"),
            "summary": sum_base,
            "friction_metrics": score_base,
            "six_channel_metrics": six_base,
            "six_channel_costs": costs_base,
        },
        "candidate_run": {
            "status": sum_cand.get("status"),
            "complete": cand_complete,
            "integrated_steps": sum_cand.get("integrated_steps"),
            "integrated_duration_s": sum_cand.get("integrated_duration_s"),
            "actual_dt_s": sum_cand.get("actual_dt_s"),
            "summary": sum_cand,
            "friction_metrics": score_cand,
            "six_channel_metrics": six_cand,
            "six_channel_costs": costs_cand,
        },
        "comparison": {
            "max_normal_difference_n": max_normal_diff_n,
            "mean_normal_difference_n": mean_normal_diff_n,
            "six_channel_metrics_diff": diff_six_metrics,
        },
        "qualification": (
            "Frozen-controller free-leg dynamic qualification of shoe friction candidates. "
            "Baseline controller gains, equilibrium spline, and normal contact mechanics are unchanged. "
            "Both horizontal friction metrics and six-channel limb dynamics are scored to reveal degradation."
        ),
    }

    for label, trace, complete in (
        ("baseline", trace_base, base_complete),
        ("candidate_run", trace_cand, cand_complete),
    ):
        if complete:
            raw_time = trace["time_s"]
            pre = ref["unfiltered_grf_target_n"]
            raw_target = np.column_stack([np.interp(raw_time, ref["grf_time_s"], pre[:, axis]) for axis in range(2)])
            report[label]["physical_friction_metrics"] = score_friction_trace(
                {"grf_time_s": raw_time, "grf_target_n": raw_target}, trace, 1
            )
        else:
            report[label]["physical_friction_metrics"] = None
        if complete and include_observations:
            observed = score_observed_forces(ref, trace["time_s"], trace["grf_n"])
            report[label]["observed_friction_metrics"] = observed["metrics"][0]
            report[label]["observed_force_rmse_n"] = observed["force_rmse_n"][0].tolist()
            report[label]["observation_metadata"] = observed["observation"]["metadata"]
        else:
            report[label]["observed_friction_metrics"] = None
    report["force_score_policy"] = (
        "Raw physical force at every stored timestep is primary. Pre-20Hz cleaned input reference is interpolated. Legacy six-channel diagnostics retained; matched observations are opt-in only."
    )
    report = _finite_report(report)
    if out_p is not None:
        out_p.mkdir(parents=False, exist_ok=False)
        np.savez_compressed(out_p / "trace_baseline.npz", **trace_base)
        np.savez_compressed(out_p / "trace_candidate.npz", **trace_cand)
        with open(out_p / "summary.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, allow_nan=False)

    if raise_on_failure and not overall_complete:
        raise RuntimeError(f"Dynamic friction qualification failed: {report['failure_reason']}")

    return report


def build_parser() -> argparse.ArgumentParser:
    """Construct command-line interface parser."""
    parser = argparse.ArgumentParser(
        description="Frozen-controller free-leg dynamic qualification of shoe friction candidates."
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE_DIR,
        help=f"Path to sealed baseline directory (default: {DEFAULT_BASELINE_DIR}).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_BASELINE_MANIFEST,
        help=f"Path to baseline manifest JSON (default: {DEFAULT_BASELINE_MANIFEST}).",
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        default=None,
        help="Path to candidate JSON file (optional; defaults to baseline parameters).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory to save traces and summary.json (must not exist).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Warp device to run simulation on (default: cpu).",
    )
    parser.add_argument(
        "--dt-scale",
        type=float,
        default=1.0,
        help="Timestep scaling factor (e.g. 1.0 default, 0.5 for refinement).",
    )
    parser.add_argument(
        "--matched-observations",
        action="store_true",
        help="Add secondary filtered observation metrics; never change physical force.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        report = run_dynamic_qualification(
            baseline_dir=args.baseline,
            candidate_source=args.candidate,
            output_dir=args.output,
            manifest_path=args.manifest,
            device=args.device,
            dt_scale=args.dt_scale,
            raise_on_failure=False,
            include_observations=args.matched_observations,
        )
        print(f"Qualification status: {report['status']} (complete={report['complete']})")
        if not report["complete"]:
            print(f"Failure reason: {report['failure_reason']}", file=sys.stderr)
            sys.exit(1)
    except Exception as exc:
        print(f"Error during qualification: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
