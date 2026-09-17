# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run independently qualified fixed-gain controller sweeps and fresh restarts."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np

from ..data import load as load_reference
from ..fit import FitConfig
from ..profile import load as load_profile
from ..run import Config
from ..trajectory import Spline
from .baseline import _fresh_equilibrium, build_inputs
from .benchmark import _plain
from .provenance import source_snapshot

_GAIN_FIELDS = ("hip_stiffness_n_m", "joint_stiffness_nm_rad", "hip_damping_ns_m", "joint_damping_nms_rad")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, value):
    Path(path).write_text(json.dumps(_plain(value), indent=2, allow_nan=False) + "\n")


def create_plan(bundle: Path, output: Path, subject: str, *, iterations=50):
    """Freeze input identities and a bounded common K/D scaling grid."""
    if output.exists():
        raise FileExistsError(output)
    if iterations < 1:
        raise ValueError("Iterations must be positive")
    summary = json.loads((bundle / "summary.json").read_text())
    reference = bundle / "reference.npz"
    profile = bundle / "profile.json"
    artifact = bundle / "digital_shoe.json"
    reference_data = load_reference(reference)
    _check_subject(reference_data, subject)
    load_profile(profile)
    manifest_path = bundle / "baseline.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for name, expected in manifest["files_sha256"].items():
            if Path(name).name != name or _sha(bundle / name) != expected:
                raise ValueError(f"Saved source bundle changed: {name}")
    input_quality = _check_input_quality(bundle, summary, reference_data)
    inputs = {
        name: {"path": str(path.resolve()), "sha256": _sha(path)}
        for name, path in (("reference", reference), ("profile", profile), ("artifact", artifact))
    }
    cases = [
        {"id": f"k{k:.2f}_d{d:.2f}", "stiffness_multiplier": k, "damping_multiplier": d}
        for k in (0.5, 1.0, 2.0)
        for d in (0.5, 1.0, 2.0)
    ]
    plan = {
        "schema": "cartesian_fixed_gain_sweep_1",
        "subject": subject,
        "inputs": inputs,
        "input_quality": input_quality,
        "config": summary["simulation_config"],
        "fit_config": summary["fit_config"],
        "mount_m": summary["shoe"]["mount_m"],
        "pitch_rad": summary["shoe"]["static_pitch_rad"],
        "cases": cases,
        "screen_iterations": iterations,
        "final_iterations": 200,
        "restart_seeds": [17, 42, 101],
        "gains_constant_during_solve": True,
        "scope": "Common K and D multipliers across four channels; not an exhaustive eight-parameter sweep.",
        "global_optimality_claimed": False,
    }
    output.mkdir(parents=True)
    _write(output / "plan.json", plan)
    return plan


def _check_input_quality(bundle: Path, summary: dict, reference: dict) -> dict:
    """Require input-QC evidence without confusing it with controller acceptance."""
    if (bundle / "REJECTED.md").exists():
        raise ValueError("The source bundle is explicitly rejected for fitting")
    metadata = json.loads(str(reference["metadata_json"]))
    frame = metadata.get("foot_markers", {}).get("frame", {})
    prepared = summary.get("schema") == "cartesian_subject_preparation_1" or "quality_selected_window_m" in frame
    if not prepared:
        if not (bundle / "baseline.json").is_file():
            raise ValueError("Source inputs need a frozen baseline manifest or a subject input-QC certificate")
        manifest = json.loads((bundle / "baseline.json").read_text())
        if (
            not {"reference.npz", "profile.json", "digital_shoe.json", "summary.json"}
            <= manifest.get("files_sha256", {}).keys()
        ):
            raise ValueError("Frozen source manifest does not cover the required inputs and settings")
        return {"kind": "manifest_verified_frozen_baseline"}
    quality = summary.get("input_quality", {})
    if (
        quality.get("schema") != "cartesian_subject_input_quality_1"
        or quality.get("passed") is not True
        or quality.get("combined_selection_passed") is not True
        or quality.get("rigidity_evaluation") != "raw heel cluster before reference filtering"
    ):
        raise ValueError("Subject preparation lacks a passed raw input-QC certificate")
    limits = {"frame_rms": 0.002, "point_max": 0.003}
    if quality.get("rigidity_limits_m") != limits or frame.get("quality_limits_m") != limits:
        raise ValueError("Subject rigidity limits differ from the fixed input gates")
    raw = frame.get("quality_selected_window_m", {})
    values = np.asarray([quality.get("raw_frame_rms_max_m"), quality.get("raw_point_max_m")], dtype=float)
    recorded = np.asarray([raw.get("raw_cluster_frame_rms_max"), raw.get("raw_cluster_point_max")], dtype=float)
    if (
        not np.isfinite(values).all()
        or np.any(values < 0)
        or np.any(values > [0.002, 0.003])
        or not np.array_equal(values, recorded)
    ):
        raise ValueError("Raw heel rigidity is missing, inconsistent, or outside the fixed limits")
    selection = summary.get("selection", {})
    episode = selection.get("contact_episode", {})
    if selection.get("accepted_candidate_count", 0) < 1 or episode.get("assigned_side") != summary.get("side"):
        raise ValueError("Subject selection does not identify a retained same-side support episode")
    if episode.get("finite_cop_fraction") != 1.0 or episode.get("support_complete_fraction") != 1.0:
        raise ValueError("Subject support or COP coverage is incomplete")
    for name, minimum in (
        ("median_other_distance_margin_m", 0.30),
        ("nearest_fraction", 0.95),
        ("other_foot_bulk_min_height_m", 0.05),
        ("other_foot_bulk_heel_min_height_m", 0.10),
    ):
        value = episode.get(name, float("nan"))
        if not np.isfinite(value) or value <= minimum:
            raise ValueError("Subject side or isolated-support gates failed")
    speed = episode.get("treadmill_guard", {})
    if not all(speed.get(key) is True for key in ("covered", "positive", "constant", "tied", "accepted")):
        raise ValueError("Subject treadmill guard failed")
    rigid = episode.get("heel_cluster_rigid_fit", {})
    if rigid.get("evaluation") != "raw selected-window heel-cluster rigid fit before reference filtering":
        raise ValueError("Subject selection used a filtered or unknown rigidity gate")
    selected_raw = np.asarray([rigid.get("frame_rms_max_m"), rigid.get("point_max_max_m")], dtype=float)
    if (
        rigid.get("limits_m") != limits
        or np.any(selected_raw < 0)
        or np.any(selected_raw > [0.002, 0.003])
        or not np.allclose(values, selected_raw, rtol=0, atol=1.0e-12)
    ):
        raise ValueError("Selected-episode rigidity differs from the reference window")
    expected_files = {name: _sha(bundle / name) for name in ("reference.npz", "profile.json", "digital_shoe.json")}
    if quality.get("files_sha256") != expected_files:
        raise ValueError("Subject input-QC artifact identities changed")
    return quality


def _check_subject(reference, subject):
    """Reject a cosmetic subject relabeling of measured inputs."""
    metadata = json.loads(str(reference["metadata_json"]))
    declared = metadata.get("subject", metadata.get("subject_id"))
    if isinstance(declared, dict):
        declared = declared.get("name", declared.get("id"))
    if declared is not None:
        if declared != subject:
            raise ValueError("Requested subject differs from the measured reference")
        return
    captures = metadata.get("source_preparation", {})
    files = [captures.get(kind, {}).get("file", "") for kind in ("static", "trial")]
    if not all(subject in Path(file).parts for file in files):
        raise ValueError("Measured reference must identify its subject in metadata or raw-capture provenance")


def _check_inputs(plan):
    for name, record in plan["inputs"].items():
        if _sha(record["path"]) != record["sha256"]:
            raise ValueError(f"Frozen sweep input changed: {name}")
    _check_subject(load_reference(plan["inputs"]["reference"]["path"]), plan["subject"])


def _initial_controller(reference, profile, mode, seed):
    initial, provenance = _fresh_equilibrium(reference, profile)
    if mode == "reference":
        return initial, provenance
    anchor = np.asarray(provenance["neutral_anchor"])
    coefficients = initial.coefficients.copy()
    if mode == "attenuated":
        coefficients = anchor + 0.75 * (coefficients - anchor)
    elif mode == "perturbed":
        rng = np.random.default_rng(seed)
        u = np.linspace(0.0, 1.0, 12)
        modes = np.column_stack([np.sin(np.pi * u), np.sin(2 * np.pi * u)])
        perturbation = (modes @ rng.normal(size=(2, 4))) * np.asarray([0.01, 0.01, 0.025, 0.025])
        interior = anchor + 0.85 * (coefficients - anchor)
        for halvings in range(32):
            trial = Spline(initial.duration_s, interior + perturbation)
            if trial.bounds(
                *(
                    profile[k]
                    for k in (
                        "equilibrium_lower",
                        "equilibrium_upper",
                        "equilibrium_rate_limit",
                        "equilibrium_acceleration_limit",
                    )
                )
            ):
                coefficients = trial.coefficients
                perturbation_halvings = halvings
                break
            perturbation *= 0.5
        else:
            raise ValueError("No bounded perturbed fresh seed found")
    else:
        raise ValueError("Unknown initialization mode")
    result = Spline(initial.duration_s, coefficients)
    if not result.bounds(
        *(
            profile[k]
            for k in (
                "equilibrium_lower",
                "equilibrium_upper",
                "equilibrium_rate_limit",
                "equilibrium_acceleration_limit",
            )
        )
    ):
        raise ValueError("Restart controller violates unchanged bounds")
    provenance = {
        **provenance,
        "kind": f"fresh_{mode}_seed",
        "random_seed": seed if mode == "perturbed" else None,
        "postprocessing": (
            {"deviation_multiplier": 0.75}
            if mode == "attenuated"
            else {
                "deviation_multiplier": 0.85,
                "basis": "sin(pi*u), sin(2*pi*u) on control index u",
                "channel_scales_m_m_rad_rad": [0.01, 0.01, 0.025, 0.025],
                "perturbation_halvings": perturbation_halvings,
            }
        ),
        "used_previous_controller_coefficients": False,
        "used_optimizer_history": False,
    }
    return result, provenance


def run_case(directory: Path, case_id: str, *, mode="reference", seed=17, initial_seed=2718, iterations=None):
    """Qualify and fit one fixed gain set without changing any acceptance criteria."""
    from ...pipeline import main as pipeline_main  # noqa: PLC0415
    from .__main__ import optimize  # noqa: PLC0415

    plan_path = directory / "plan.json"
    plan_hash = _sha(plan_path)
    plan = json.loads(plan_path.read_text())
    _check_inputs(plan)
    case = next(c for c in plan["cases"] if c["id"] == case_id)
    iterations = plan["screen_iterations"] if iterations is None else iterations
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("Iterations must be a positive integer")
    initialization_id = f"{mode}_i{initial_seed}" if mode == "perturbed" else mode
    run_id = f"{case_id}_{initialization_id}_s{seed}_n{iterations}"
    output = directory / "runs" / run_id
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    started = perf_counter()
    profile = copy.deepcopy(load_profile(plan["inputs"]["profile"]["path"]))
    for name in _GAIN_FIELDS:
        multiplier = case["stiffness_multiplier"] if "stiffness" in name else case["damping_multiplier"]
        profile[name] = (np.asarray(profile[name]) * multiplier).tolist()
    profile["provenance"]["impedance"] += " Fixed-gain sweep: gains are constant during the entire controller solve."
    profile["provenance"]["gain_sweep"] = {
        "subject": plan["subject"],
        "case": case,
        "source_profile": plan["inputs"]["profile"],
    }
    profile_path = output / "case_profile.json"
    _write(profile_path, profile)
    fixed_profile_hash = _sha(profile_path)
    reference = load_reference(plan["inputs"]["reference"]["path"])
    record = {
        "run_id": run_id,
        "case": case,
        "subject": plan["subject"],
        "iterations_requested": iterations,
        "seed": seed,
        "initialization": {"kind": f"fresh_{mode}_seed", "status": "not_generated"},
        "initial_seed": initial_seed if mode == "perturbed" else None,
        "runner_sha256": _sha(Path(__file__)),
        "gains": {k: profile[k] for k in _GAIN_FIELDS},
        "gains_constant_during_solve": True,
        "profile_sha256": fixed_profile_hash,
        "stage": "prepare",
        "status": "running",
        "plan_sha256": plan_hash,
    }
    _write(output / "case.json", record)
    try:
        initial, initialization = _initial_controller(reference, profile, mode, initial_seed)
        record["initialization"] = initialization
        _write(output / "case.json", record)
        build_inputs(
            Path(plan["inputs"]["reference"]["path"]),
            profile_path,
            Path(plan["inputs"]["artifact"]["path"]),
            output / "baseline",
            mount_m=plan["mount_m"],
            pitch_rad=plan["pitch_rad"],
            config=Config(**plan["config"]),
            fit_config=FitConfig(**plan["fit_config"]),
            equilibrium=initial,
            initialization=initialization,
            input_provenance={"gain_sweep_plan_sha256": plan_hash, "subject": plan["subject"]},
        )
        root = Path(__file__).resolve().parents[4]
        _write(
            output / "physical_source_identity.json",
            {str(root / path): {"baseline": digest} for path, digest in source_snapshot().items()},
        )
        record["stage"] = "qualification"
        _write(output / "case.json", record)
        pipeline_main(["--output", str(output), "--stage", "validate"])
        record["stage"] = "fit"
        _write(output / "case.json", record)
        result = optimize(
            SimpleNamespace(
                directory=output / "baseline",
                output=output / "fit",
                initial_equilibrium=None,
                single_validation=output / "single/benchmark.json",
                batch_validation=output / "mixed/benchmark.json",
                contact_rounding_evidence=output / "contact/report.json",
                iterations=iterations,
                wall_seconds=3600.0,
                step_fraction=0.05,
                minimum_step_fraction=0.005,
                seed=seed,
                plateau_patience=None,
                plateau_rtol=1.0e-4,
                mesh_only=False,
            )
        )
        if _sha(profile_path) != fixed_profile_hash or _sha(plan_path) != plan_hash:
            raise ValueError("Sweep configuration changed during solve")
        _check_inputs(plan)
        frozen_fit_profile = json.loads((output / "fit/profile.json").read_text())
        if any(frozen_fit_profile[k] != profile[k] for k in _GAIN_FIELDS):
            raise ValueError("Fitted gains differ from the fixed case gains")
        record.update(
            status="completed" if result["iterations_completed"] == iterations else "budget_incomplete",
            stage="complete",
            loss=result["loss"],
            initial_loss=result["initial_loss"],
            metrics=result["metrics"],
            accepted=result["accepted"],
            refinement_passed=result["refinement"]["passed"],
            iterations_completed=result["iterations_completed"],
            search_wall_s=result["wall_s"],
            search_seconds_per_iteration=result["wall_s"] / max(1, result["iterations_completed"]),
            termination=result["termination"],
            refinement=result["refinement"],
            qualification=result["qualification"],
            counts=result["counts"],
            report=str((output / "fit/report.html").resolve()),
        )
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        record["total_wall_s"] = perf_counter() - started
        _write(output / "case.json", record)
        summarize(directory)
    return record


def summarize(directory: Path):
    """Report fit errors, numerical checks, and real work at each iteration budget."""
    import csv  # noqa: PLC0415
    import html  # noqa: PLC0415

    records = []
    for path in sorted((directory / "runs").glob("*/case.json")):
        record = json.loads(path.read_text())
        summary_path = path.parent / "fit/summary.json"
        if record["status"] in ("completed", "budget_incomplete") and summary_path.exists():
            summary = json.loads(summary_path.read_text())
            record.update(
                refinement=summary["refinement"],
                termination=summary["termination"],
                qualification=summary["qualification"],
                fit_summary_sha256=_sha(summary_path),
            )
            tolerances = summary["fit_config"]
            metrics = record["metrics"]
            normalized = [
                *(np.asarray(metrics["hip_rmse_m"]) / tolerances["hip_tolerance_m"]),
                *(np.asarray(metrics["joint_rmse_rad"]) / tolerances["joint_tolerance_rad"]),
                *(np.asarray(metrics["force_rmse_n"]) / tolerances["force_tolerance_n"]),
            ]
            record["maximum_normalized_rms_error"] = float(max(normalized))
            record["search_seconds_per_iteration"] = record["search_wall_s"] / max(1, record["iterations_completed"])
        records.append(record)
    budgets = sorted({r["iterations_requested"] for r in records})
    rankings = {}
    for budget in budgets:
        completed = [
            r
            for r in records
            if r["status"] == "completed" and r["iterations_completed"] == budget and r["refinement_passed"]
        ]
        completed.sort(key=lambda r: r["loss"])
        accepted = [r for r in completed if r["accepted"]]
        rankings[str(budget)] = {
            "numerically_qualified_loss_order": [r["run_id"] for r in completed],
            "lowest_loss_run": completed[0]["run_id"] if completed else None,
            "lowest_loss_accepted_run": accepted[0]["run_id"] if accepted else None,
        }
    _write(
        directory / "results.json",
        {"runs": records, "rankings_by_iteration_budget": rankings, "global_optimality_claimed": False},
    )
    intro = [
        "# Fixed-gain controller sweep",
        "",
        "Each solve keeps stiffness and damping constant. Only the equilibrium controller is fitted.",
        "Compare runs with matching iteration budgets. Equal iterations can perform different amounts of physics work.",
        "Finite-budget screening, fresh restarts, and local probes do not prove convergence or global optimality.",
        "",
    ]
    rows = intro.copy()
    web_tables = []
    flat_rows = []
    for budget in budgets:
        selected = [r for r in records if r["iterations_requested"] == budget]
        rows += [
            f"## {budget}-iteration budget",
            "",
            "RMS columns use hip [mm], knee/ankle [rad], and force [N]. Acceptance includes the half-timestep check.",
            "",
            "| Case / start / search seed | Loss | Hip x / z | Knee / ankle | Force x / z | Accepted | Half-step | Status |",
            "| --- | ---: | --- | --- | --- | --- | --- | --- |",
        ]
        web_rows = []
        for r in selected:
            label = f"{r['case']['id']} / {r['initialization']['kind']} / {r['seed']}"
            metrics = r.get("metrics", {})
            hip = np.asarray(metrics.get("hip_rmse_m", [np.nan, np.nan])) * 1000
            joints = metrics.get("joint_rmse_rad", [np.nan, np.nan])
            forces = metrics.get("force_rmse_n", [np.nan, np.nan])
            loss = f"{r['loss']:.6f}" if r.get("loss") is not None else "—"

            def pair(values, precision):
                return " / ".join(f"{x:.{precision}f}" if np.isfinite(x) else "—" for x in values)

            cells = [
                label,
                loss,
                pair(hip, 2),
                pair(joints, 5),
                pair(forces, 2),
                str(r.get("accepted", False)),
                str(r.get("refinement_passed", False)),
                r["status"],
            ]
            report_link = f"runs/{r['run_id']}/fit/report.html"
            markdown_label = f"[{label}]({report_link})" if "loss" in r else label
            rows.append("| " + " | ".join([markdown_label, *cells[1:]]) + " |")
            web_label = (
                f'<a href="{html.escape(report_link)}">{html.escape(label)}</a>' if "loss" in r else html.escape(label)
            )
            web_rows.append(
                "<tr><td>" + web_label + "</td>" + "".join(f"<td>{html.escape(c)}</td>" for c in cells[1:]) + "</tr>"
            )
            counts = r.get("counts", {})
            flat_rows.append(
                {
                    "run_id": r["run_id"],
                    "subject": r["subject"],
                    "stiffness_multiplier": r["case"]["stiffness_multiplier"],
                    "damping_multiplier": r["case"]["damping_multiplier"],
                    "start": r["initialization"]["kind"],
                    "search_seed": r["seed"],
                    "initial_seed": r.get("initial_seed"),
                    "iterations_requested": budget,
                    "iterations_completed": r.get("iterations_completed", 0),
                    "loss": r.get("loss"),
                    "hip_x_rms_mm": float(hip[0]) if np.isfinite(hip[0]) else None,
                    "hip_z_rms_mm": float(hip[1]) if np.isfinite(hip[1]) else None,
                    "knee_rms_rad": joints[0] if np.isfinite(joints[0]) else None,
                    "ankle_rms_rad": joints[1] if np.isfinite(joints[1]) else None,
                    "force_x_rms_n": forces[0] if np.isfinite(forces[0]) else None,
                    "force_z_rms_n": forces[1] if np.isfinite(forces[1]) else None,
                    "accepted": r.get("accepted", False),
                    "half_step_passed": r.get("refinement_passed", False),
                    "status": r["status"],
                    "search_wall_s": r.get("search_wall_s"),
                    "search_seconds_per_iteration": r.get("search_seconds_per_iteration"),
                    "allocated_slots": counts.get("physics_worlds"),
                    "real_candidates": counts.get("real_candidates"),
                    "completed_real_candidates": counts.get("completed_real_candidates"),
                    "padding_slots": counts.get("padding_worlds"),
                    "integrated_world_steps_actual": counts.get("integrated_steps_including_padding"),
                    "unique_candidates": counts.get("unique_candidates"),
                    "termination": r.get("termination"),
                }
            )
        heading = (
            "<tr>"
            + "".join(
                f"<th>{x}</th>"
                for x in (
                    "Case / start / search seed",
                    "Loss",
                    "Hip x/z [mm]",
                    "Knee/ankle [rad]",
                    "Force x/z [N]",
                    "Accepted",
                    "Half-step",
                    "Status",
                )
            )
            + "</tr>"
        )
        web_tables.append(f"<h2>{budget}-iteration budget</h2><table>{heading}{''.join(web_rows)}</table>")
        rows += [
            "",
            "| Run | Search [s] | Complete iteration [s] | Allocated slots | Completed real candidates | Skipped padding | Actual world-steps |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for r in selected:
            c = r.get("counts", {})
            rows.append(
                f"| {r['run_id']} | {r.get('search_wall_s', 0):.2f} | {r.get('search_seconds_per_iteration', 0):.3f} | {c.get('physics_worlds', 0)} | {c.get('completed_real_candidates', 0)} | {c.get('padding_worlds', 0)} | {c.get('integrated_steps_including_padding', 0)} |"
            )
        rows += [
            "",
            "Unique controllers are not tracked by the fitter; completed candidate slots must not be described as unique controllers.",
            "",
        ]
    (directory / "results.md").write_text("\n".join(rows) + "\n")
    if flat_rows:
        with (directory / "results.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(flat_rows[0]))
            writer.writeheader()
            writer.writerows(flat_rows)
    document = (
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Fixed-gain controller sweep</title>'
        "<style>body{font:15px sans-serif;margin:2em}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:.5em;text-align:left}</style>"
        "<h1>Fixed-gain controller sweep</h1><p>Gains stay constant within each solve. Compare equal iteration budgets. "
        "Finite restarts and probes do not prove a global optimum.</p><p>Detailed work counts: "
        '<a href="results.csv">CSV</a> · <a href="results.json">JSON</a> · <a href="results.md">Markdown</a></p>'
        + "".join(web_tables)
        + "</html>"
    )
    (directory / "report.html").write_text(document)
    return records


def main(argv=None):
    """Create a sweep or execute one resumable, isolated case."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create")
    create.add_argument("--bundle", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--subject", required=True)
    create.add_argument("--iterations", type=int, default=50)
    run = sub.add_parser("run")
    run.add_argument("directory", type=Path)
    run.add_argument("--case", required=True)
    run.add_argument("--mode", choices=("reference", "attenuated", "perturbed"), default="reference")
    run.add_argument("--seed", type=int, default=17)
    run.add_argument(
        "--initial-seed", type=int, default=2718, help="Perturbed-controller seed, independent of the optimizer seed"
    )
    run.add_argument("--iterations", type=int)
    report = sub.add_parser("report")
    report.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    if args.command == "create":
        create_plan(args.bundle, args.output, args.subject, iterations=args.iterations)
    elif args.command == "run":
        run_case(
            args.directory,
            args.case,
            mode=args.mode,
            seed=args.seed,
            initial_seed=args.initial_seed,
            iterations=args.iterations,
        )
    else:
        summarize(args.directory)


if __name__ == "__main__":
    main()
