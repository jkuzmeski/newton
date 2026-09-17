# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Numerical and local identifiability checks for a frozen-history friction fit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .friction_history import load_history
from .friction_sweep import PARAMETER_NAMES, SCORE_NAMES, FrictionSweep, _decode, _encode


def _resample_history(arrays: dict, factor: int) -> dict:
    """Interpolate frozen input signals, not the accepted normal solver itself."""
    result = dict(arrays)
    steps = len(arrays["time_s"])
    index = np.arange((steps - 1) * factor + 1) / factor
    lo = np.minimum(index.astype(int), steps - 2)
    a = index - lo
    result["time_s"] = np.linspace(arrays["time_s"][0], arrays["time_s"][-1], len(index))
    for name in ("position_xy", "velocity_xy", "nominal_velocity_xy", "normal_n"):
        original = arrays[name]
        weights = a.reshape((-1,) + (1,) * (original.ndim - 1))
        result[name] = ((1.0 - weights) * original[lo] + weights * original[lo + 1]).astype(np.float32)
    return result


def qualify_candidate(cache: Path, candidate: Path, output: Path, *, device="cuda:0") -> dict:
    """Check a candidate without installing it or claiming independent validation."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    history = load_history(cache)
    if history.provenance.get("complete") is not True:
        raise ValueError("Qualification needs a complete original history")
    payload = json.loads(Path(candidate).read_text())
    row = np.asarray([[payload["parameters"][name] for name in PARAMETER_NAMES]], np.float32)
    arrays = vars(history)
    workspace = FrictionSweep(arrays, 16, device=device)
    scores, curves = workspace.evaluate(row, curves=True)
    baseline_curve = curves[0].copy()
    qualification = {
        "schema": "digital_shoe_friction_fit_qualification_1",
        "candidate_sha256": hashlib.sha256(Path(candidate).read_bytes()).hexdigest(),
        "cache_sha256": hashlib.sha256(Path(cache).read_bytes()).hexdigest(),
        "parameters": payload["parameters"],
        "original_score": dict(zip(SCORE_NAMES, scores[0].astype(float).tolist(), strict=True)),
        "interpretation": "Numerical consistency and local sensitivity only. No new experimental condition or parameter confidence interval.",
    }
    native = history.measured_time_s
    mask = (native >= history.time_s[0]) & (native <= history.time_s[-1])
    native = native[mask]
    scaled_jacobians = {}
    unit = _encode(row)[0]
    force_scale = workspace.force_scale
    for radius in (0.01, 0.05):
        probes = []
        widths = []
        for j in range(5):
            high, low = unit.copy(), unit.copy()
            high[j] = min(1.0, high[j] + radius)
            low[j] = max(0.0, low[j] - radius)
            probes.extend((high, low))
            widths.append(high[j] - low[j])
        parameters = _decode(np.asarray(probes), int(row[0, 0]))
        _, response = workspace.evaluate(parameters, curves=True)
        jacobian = np.empty((len(native), 5))
        for j in range(5):
            high = np.interp(native, history.time_s, response[2 * j, :, 0])
            low = np.interp(native, history.time_s, response[2 * j + 1, :, 0])
            jacobian[:, j] = (high - low) / max(widths[j], 1e-12) / force_scale
        singular = np.linalg.svd(jacobian, compute_uv=False)
        rank = int(np.count_nonzero(singular > max(singular[0], 1e-12) * 1e-5))
        scaled_jacobians[str(radius)] = {
            "parameter_names": PARAMETER_NAMES[1:6],
            "column_norms": np.linalg.norm(jacobian, axis=0).tolist(),
            "singular_values": singular.tolist(),
            "rank_at_relative_threshold_1e_5": rank,
            "condition_number": float(singular[0] / singular[-1]) if singular[-1] > 1e-12 else None,
            "note": "Bounded, scaled-coordinate finite differences. Dwell changes can be quantized by the timestep.",
        }
    qualification["local_sensitivity"] = scaled_jacobians
    refinement = []
    for factor in (2, 4):
        refined = _resample_history(arrays, factor)
        smaller = FrictionSweep(refined, 1, device=device)
        refined_score, refined_curve = smaller.evaluate(row, curves=True)
        difference = refined_curve[0, ::factor] - baseline_curve
        refinement.append(
            {
                "factor": factor,
                "dt_s": smaller.dt,
                "max_force_difference_n": float(np.max(np.abs(difference))),
                "force_difference_rmse_n": float(np.sqrt(np.mean(difference[:, 0] ** 2))),
                "score": dict(zip(SCORE_NAMES, refined_score[0].astype(float).tolist(), strict=True)),
                "input_policy": "Piecewise-linear interpolation on the original time support. No new normal solve or invented terminal force sample.",
            }
        )
    qualification["timestep_refinement"] = refinement
    divided = dict(arrays)
    for name in ("position_xy", "velocity_xy", "nominal_velocity_xy"):
        divided[name] = np.repeat(arrays[name], 2, axis=1)
    divided["normal_n"] = np.repeat(arrays["normal_n"] * 0.5, 2, axis=1)
    for name in ("baseline_kt_n_m", "baseline_kv_ns_m"):
        divided[name] = np.repeat(arrays[name] * 0.5, 2)
    split = FrictionSweep(divided, 1, device=device)
    _, split_curve = split.evaluate(row, curves=True)
    qualification["tangential_subdivision"] = {
        "max_force_difference_n": float(np.max(np.abs(split_curve[0] - baseline_curve))),
        "meaning": "Split each prescribed contact into two co-located half-area bristles. This is not a normal-geometry grid-refinement test.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(qualification, indent=2, allow_nan=False))
    return qualification


def main() -> None:
    """Run numerical qualification for one saved effective friction candidate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = qualify_candidate(args.cache, args.candidate, args.output, device=args.device)
    print(
        json.dumps(
            {
                "timestep_refinement": result["timestep_refinement"],
                "tangential_subdivision": result["tangential_subdivision"],
                "local_sensitivity": result["local_sensitivity"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
