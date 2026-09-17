# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Assemble an offline comparison of saved friction-only free-leg experiments."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from projects.impedance_instron.cartesian.data import load

from .friction_leg import verify_baseline_inputs
from .friction_metrics import score_friction_trace, score_observed_forces
from .friction_report import write_friction_report


def build_comparison(baseline: Path, runs: dict[str, Path], output: Path, *, score_policy: str = "raw") -> dict:
    """Compare saved runs without rerunning physics or modifying any input."""
    baseline, output = Path(baseline), Path(output)
    if output.exists():
        raise FileExistsError(output)
    if score_policy not in ("raw", "matched", "legacy"):
        raise ValueError("score_policy must be raw, matched or legacy")
    hashes = verify_baseline_inputs(baseline)
    reference = load(baseline / "reference.npz")
    with np.load(baseline / "trace.npz", allow_pickle=False) as data:
        baseline_trace = {name: data[name].copy() for name in data.files}
    baseline_summary = json.loads((baseline / "summary.json").read_text())
    scores = {"original": score_friction_trace(reference, baseline_trace, 1, summary=baseline_summary)}
    clocks = {"original": baseline_trace["time_s"]}
    forces = {"original": baseline_trace["grf_n"]}
    sources = {"original": hashes}
    for name, folder_path in runs.items():
        if name in ("original", "measured") or name in sources:
            raise ValueError(f"Duplicate or reserved run name: {name}")
        folder = Path(folder_path)
        if (folder / "run.json").exists():
            trace_path = folder / "trace.npz"
            summary_path = folder / "run.json"
            summary = json.loads(summary_path.read_text())
            provenance_path = folder / "report.json"
            provenance = json.loads(provenance_path.read_text())
        else:
            trace_path = folder / "trace_candidate.npz"
            summary_path = folder / "summary.json"
            provenance_path = summary_path
            provenance = json.loads(summary_path.read_text())
            summary = provenance["candidate_run"]["summary"]
        if provenance.get("input_hashes") != hashes:
            raise ValueError(f"Run {name!r} does not identify the same sealed baseline inputs")
        with np.load(trace_path, allow_pickle=False) as data:
            trace = {key: data[key].copy() for key in data.files}
        score = score_friction_trace(reference, trace, 1, summary=summary)
        scores[name] = score
        if score["complete"]:
            clocks[name] = trace["time_s"]
            forces[name] = trace["grf_n"]
        sources[name] = {
            "trace_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
            "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
            "provenance_sha256": hashlib.sha256(provenance_path.read_bytes()).hexdigest(),
            "source_hashes": provenance.get("source_hashes", {}),
        }
    raw_forces, raw_clocks = dict(forces), dict(clocks)
    legacy_scores = dict(scores)
    plotted_reference = reference
    observation_metadata = {}
    saved_signals = {}
    if score_policy == "raw":
        if "unfiltered_grf_target_n" not in reference:
            raise ValueError("Raw physical comparison requires the pre-20Hz reference force")
        pre_force = reference["unfiltered_grf_target_n"]
        plotted_reference = {"grf_time_s": reference["grf_time_s"], "grf_target_n": pre_force}
        for name, force_values in forces.items():
            clock = clocks[name]
            target = np.column_stack(
                [np.interp(clock, reference["grf_time_s"], pre_force[:, axis]) for axis in range(2)]
            )
            # Completion was checked on the original saved trace before constructing
            # a common comparison clock. Simulation values are not resampled.
            scores[name] = score_friction_trace(
                {"grf_time_s": clock, "grf_target_n": target}, {"time_s": clock, "grf_n": force_values}, 1
            )
            scores[name]["metadata"]["signal_policy"] = (
                "Raw simulation samples; only the pre-20Hz acquisition-cleaned reference is interpolated."
            )
    if score_policy == "matched":
        common_clock = None
        for index, name in enumerate(forces):
            matched = score_observed_forces(reference, clocks[name], forces[name])
            observed = matched["observation"]
            if common_clock is not None and not np.array_equal(common_clock, observed["clock"]):
                raise ValueError("Compared runs need the same supported native clock for matched observations")
            common_clock = observed["clock"]
            scores[name] = matched["metrics"][0]
            clocks[name] = observed["clock"]
            forces[name] = observed["observed_prediction"]
            observation_metadata[name] = {
                "processing": observed["metadata"],
                "force_rmse_n": matched["force_rmse_n"][0].tolist(),
            }
            saved_signals[f"run_{index}_raw_time_s"] = raw_clocks[name]
            saved_signals[f"run_{index}_raw_force_n"] = raw_forces[name]
            saved_signals[f"run_{index}_observed_force_n"] = forces[name]
            plotted_reference = {"grf_time_s": observed["clock"], "grf_target_n": observed["observed_target"]}
        saved_signals["observed_time_s"] = common_clock
        saved_signals["observed_reference_n"] = plotted_reference["grf_target_n"]
    if verify_baseline_inputs(baseline) != hashes:
        raise RuntimeError("Baseline inputs changed while building comparison")
    output.mkdir(parents=True)
    report = {
        "schema": "digital_shoe_friction_comparison_2",
        "score_policy": score_policy,
        "observation": observation_metadata,
        "legacy_scores": legacy_scores,
        "signal_run_order": list(forces),
        "forward_sign": 1,
        "description": "Saved free-leg runs identify the same sealed reference, controller profile and equilibrium inputs. Input hashes are checked; source-law invariance requires a separate audit. Motion and resulting normal forces can differ.",
        "qualification": "Effective fits to the same experimental stance, not independent friction calibration. Completion is not physical acceptance; inspect all phase, motion, energy and refinement checks.",
        "steps_evaluated": len(baseline_trace["time_s"]),
        "total_source_steps": len(baseline_trace["time_s"]),
        "is_partial_smoke": False,
        "scores": scores,
        "sources": sources,
    }
    if score_policy == "matched":
        report["description"] += (
            " Both components are signed, matched force observations; negative filtered Fz is not negative physical support. No component clipping is applied. Stance uses the pre-20Hz normal signal."
        )
        np.savez_compressed(output / "comparison_signals.npz", **saved_signals)
    elif score_policy == "raw":
        report["description"] += (
            " RAW PHYSICAL FORCE: simulation curves and peaks are unfiltered. Reference is pre-20Hz acquisition-cleaned/Hann-filtered data, not raw sensor output. Only reference interpolation is used for full simulation-rate metrics."
        )
    else:
        report["description"] += " LEGACY unmatched comparison: raw simulation versus processed/clipped reference."
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    displayed = {name: score for name, score in scores.items() if name in forces}
    write_friction_report(
        {**report, "scores": displayed},
        clocks["original"],
        forces,
        plotted_reference,
        output / "report.html",
        force_times=clocks,
    )
    if score_policy == "matched":
        raw_report = {
            **report,
            "scores": {name: legacy_scores[name] for name in raw_forces},
            "description": "Raw physical force versus legacy processed reference: intentionally unmatched diagnostic. Not the corrected fit score.",
            "qualification": "Inspect raw peaks and numerical behavior here. Physical forces were not filtered before integration.",
        }
        write_friction_report(
            raw_report,
            baseline_trace["time_s"],
            raw_forces,
            reference,
            output / "raw_report.html",
            force_times=raw_clocks,
        )
    return report


def main() -> None:
    """Build a portable HTML force comparison from complete saved runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--run", action="append", default=[], help="LABEL=RUN_DIRECTORY; repeat for each candidate")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--score-policy", choices=("raw", "matched", "legacy"), default="raw")
    args = parser.parse_args()
    runs = {}
    for item in args.run:
        label, path = item.split("=", 1)
        if label in runs:
            raise ValueError(f"Duplicate run label: {label}")
        runs[label] = Path(path)
    build_comparison(args.baseline, runs, args.output, score_policy=args.score_policy)
    print(args.output / "report.html")


if __name__ == "__main__":
    main()
