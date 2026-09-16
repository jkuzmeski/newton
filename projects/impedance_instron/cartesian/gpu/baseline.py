# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Qualify the saved twelve-point controller with a new CPU reference rollout."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from ..fit import FitConfig, _Objective
from ..profile import load as load_profile
from ..profile import validate as validate_profile
from ..report import _plain
from ..run import Config, simulate
from ..shoe import Shoe
from ..trajectory import Spline
from .provenance import source_snapshot


def build_failure_equilibrium(
    reference: dict[str, Any],
    profile: dict[str, Any],
    controls: int,
    config: Config,
) -> tuple[Spline, dict[str, Any]]:
    """Generate an independent bounded failure equilibrium using constant upper bounds.

    The candidate is constructed entirely from the profile's declared upper bounds,
    guaranteeing it stays within position, rate, and acceleration bounds while
    failing the step-zero hip force screen under the initial kinematic conditions.
    """
    duration = float(reference["time_s"][-1])
    lower = np.asarray(profile["equilibrium_lower"], dtype=np.float64)
    upper = np.asarray(profile["equilibrium_upper"], dtype=np.float64)
    rate = np.asarray(profile["equilibrium_rate_limit"], dtype=np.float64)
    acceleration = np.asarray(profile["equilibrium_acceleration_limit"], dtype=np.float64)

    coefficients = np.tile(upper, (controls, 1))
    spline = Spline(duration, coefficients)
    if not spline.bounds(lower, upper, rate, acceleration):
        raise ValueError("Constant upper equilibrium violates declared spline bounds")

    stiffness_hip = np.asarray(profile["hip_stiffness_n_m"], dtype=np.float64)
    damping_hip = np.asarray(profile["hip_damping_ns_m"], dtype=np.float64)
    state0 = np.asarray(reference["state"])[0]
    velocity0 = np.asarray(reference["velocity"])[0]

    hip_force0 = stiffness_hip * (upper[:2] - state0[:2]) - damping_hip * velocity0[:2]
    hip_force0_norm = float(np.linalg.norm(hip_force0))

    if hip_force0_norm <= config.maximum_force_n:
        raise ValueError(
            f"Analytic initial hip force norm ({hip_force0_norm:.3f} N) does not exceed "
            f"the maximum force screen ({config.maximum_force_n:.3f} N)"
        )

    failure_info = {
        "time_s": 0.0,
        "reasons": ["Hip force screen exceeded"],
        "initial_hip_force_norm_n": hip_force0_norm,
        "screen_threshold_n": float(config.maximum_force_n),
        "mechanism": "constant_equilibrium_upper_analytic_initial_hip_force_screen",
    }
    return spline, failure_info


def build(source: Path, output: Path) -> dict[str, Any]:
    """Replay a hash-verified saved controller without fitting or reseeding it.

    The source bundle keeps the original result unchanged. Only this newly
    simulated CPU baseline receives current source identities.
    """
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(f"Baseline output directory already exists: {output}")
    manifest = json.loads((source / "baseline.json").read_text())
    for name, expected in manifest["files_sha256"].items():
        if Path(name).name != name or hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Saved baseline file changed: {name}")
    original = json.loads((source / "summary.json").read_text())
    reference_path, profile_path = source / "reference.npz", source / "profile.json"
    artifact_path = source / "digital_shoe.json"
    controls = 12
    fit_config = FitConfig(**original["fit_config"])
    if fit_config.control_count != controls:
        raise ValueError("Saved configuration must use twelve controls")
    config = Config(**original["simulation_config"])
    with np.load(source / "equilibrium.npz", allow_pickle=False) as archive:
        equilibrium = Spline(float(archive["duration_s"]), archive["coefficients"])
        frozen_identity = json.loads(str(archive["identity_json"]))
    if equilibrium.coefficients.shape != (12, 4):
        raise ValueError("The baseline must have twelve controls per channel")
    expected_identity = {
        "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        "profile_sha256": hashlib.sha256(
            json.dumps(json.loads(profile_path.read_text()), sort_keys=True).encode()
        ).hexdigest(),
        "simulation_config": original["simulation_config"],
        "shoe": {
            "artifact_sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
            "mount_m": original["shoe"]["mount_m"],
            "static_pitch_rad": original["shoe"]["static_pitch_rad"],
            "friction": original["shoe"]["friction"],
            "device": "cuda:0",
        },
    }
    if frozen_identity != expected_identity:
        raise ValueError("Saved controller identity differs from its inputs")
    sources_before = source_snapshot()
    builder_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    reference_bytes = reference_path.read_bytes()
    profile_bytes = profile_path.read_bytes()

    with np.load(reference_path, allow_pickle=False) as archive:
        reference = dict(archive)
    profile = load_profile(profile_path)
    validate_profile(profile)

    mount, pitch = original["shoe"]["mount_m"], original["shoe"]["static_pitch_rad"]
    shoe = Shoe(artifact_path, mount, pitch, device="cpu")
    shoe_identity = {
        "artifact_sha256": shoe.metadata["sha256"],
        "mount_m": shoe.metadata["mount_m"],
        "static_pitch_rad": shoe.metadata["static_pitch_rad"],
        "device": str(shoe.device),
        "friction": shoe.metadata["friction"],
    }

    trace, run_summary = simulate(reference, profile, equilibrium, shoe, config=config)

    sources_after = source_snapshot()
    if sources_after != sources_before:
        raise RuntimeError("GPU runtime sources changed during baseline simulation")

    objective = _Objective(reference, fit_config)
    residual, metrics, costs = objective.evaluate(trace, run_summary)

    if run_summary.get("status") != "completed" or residual is None:
        failure_reason = run_summary.get("failure", "Objective evaluation failed or run incomplete")
        raise RuntimeError(
            f"Saved baseline simulation did not complete successfully for {controls} controls: {failure_reason}"
        )

    loss = float(residual @ residual)

    output.mkdir(parents=True)
    (output / "reference.npz").write_bytes(reference_bytes)
    (output / "profile.json").write_bytes(profile_bytes)

    profile_for_identity = json.loads(profile_bytes.decode("utf-8"))
    identity = {
        "reference_sha256": hashlib.sha256(reference_bytes).hexdigest(),
        "profile_sha256": hashlib.sha256(json.dumps(profile_for_identity, sort_keys=True).encode()).hexdigest(),
        "simulation_config": asdict(config),
        "shoe": shoe_identity,
    }

    np.savez_compressed(
        output / "equilibrium.npz",
        duration_s=equilibrium.duration_s,
        coefficients=equilibrium.coefficients,
        identity_json=json.dumps(identity, sort_keys=True),
    )
    np.savez_compressed(output / "trace.npz", **trace)

    failure_spline, failure_info = build_failure_equilibrium(reference, profile, controls, config)
    np.savez_compressed(
        output / "failure_equilibrium.npz",
        duration_s=failure_spline.duration_s,
        coefficients=failure_spline.coefficients,
        failure_reason=failure_info["reasons"][0],
        failure_time_s=failure_info["time_s"],
        initial_hip_force_norm_n=failure_info["initial_hip_force_norm_n"],
    )

    summary: dict[str, Any] = {
        "schema": "cartesian_twelve_point_baseline_1",
        "fit_config": asdict(fit_config),
        "simulation_config": asdict(config),
        "source_sha256": sources_before,
        "builder_sha256": builder_hash,
        "shoe": shoe.metadata,
        "status": "completed",
        "complete": True,
        "accepted": False,
        "refinement": {
            "performed": False,
            "passed": False,
            "reason": "saved controller CPU qualification without optimization or post-fit refinement",
        },
        "run": run_summary,
        "loss": loss,
        "metrics": metrics,
        "components": costs,
        "objective_components": costs,
        "objective": objective.description,
        "provenance": {
            "generation": "saved_twelve_point_controller",
            "control_count": controls,
            "optimization": "none",
            "selected_baseline": manifest,
            "builder_sha256": builder_hash,
            "raw_reference": str(reference_path.resolve()),
            "raw_profile": str(profile_path.resolve()),
            "raw_artifact": str(artifact_path.resolve()),
        },
        "model_scope": (
            "3 leg bodies, 1 shoe, 4 equilibrium channels; external Cartesian hip impedance, "
            "no hip torque/upper-body mass"
        ),
        "identity": identity,
    }

    (output / "summary.json").write_text(json.dumps(_plain(summary), indent=2, allow_nan=False) + "\n")
    return summary


def main(argv: list[str] | None = None) -> None:
    """Create new numerical evidence from a saved twelve-point bundle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = build(args.source, args.output)
    print(f"Qualified twelve-point CPU baseline at {args.output}; loss={summary['loss']:.7f}")


if __name__ == "__main__":
    main()
