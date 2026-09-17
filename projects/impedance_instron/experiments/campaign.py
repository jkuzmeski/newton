# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Seal, qualify, and replay a shoe campaign with immutable controller inputs."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import numpy as np

from projects.digital_shoe.artifact import load_artifact
from projects.digital_shoe.runtime import FoundationConfig

from ..cartesian.fit import FitConfig, _Objective
from ..cartesian.mechanics import Body
from ..cartesian.profile import load as load_profile
from ..cartesian.report import _plain
from ..cartesian.run import Config, simulate
from ..cartesian.shoe import Shoe
from ..cartesian.trajectory import Spline
from .metrics import compare, observations, refinement
from .variants import build_variant, conditions

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BUNDLE = ROOT / "outputs/impedance_instron/baseline12_accepted"
SCHEMA = "frozen_shoe_campaign_1"
BENCH_PROTOCOL = {
    "dt_s": 6.25e-5,
    "sample_dt_s": 1.0e-3,
    "warmup_cycles": 3,
    "total_cycles": 4,
    "prescribed_depth_m": 0.010,
    "period_s": 0.20,
    "max_force_cap_n": 1.0e5,
    "max_strain_cap": 0.90,
    "periodicity_tol": 0.05,
    "surround_sweeps": 5,
    "use_measured_trace": False,
}


def digest(path: Path) -> str:
    """Hash a file without changing any saved input."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def content_digest(value: dict) -> str:
    """Hash finite canonical JSON values."""
    return hashlib.sha256(json.dumps(_plain(value), sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    """Atomically publish finite JSON so interrupted runs stay detectable."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_plain(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def source_identity() -> dict:
    """Identify physical source plus every campaign implementation module."""
    from ..cartesian.gpu.provenance import source_snapshot  # noqa: PLC0415

    sources = source_snapshot()
    sources.update({str(path.relative_to(ROOT)): digest(path) for path in Path(__file__).parent.glob("*.py")})
    return sources


def verify_bundle(bundle: Path) -> dict:
    """Require saved artifact hashes, profile, and controller identity to agree."""
    manifest = json.loads((bundle / "baseline.json").read_text())
    for name, expected in manifest["files_sha256"].items():
        if Path(name).name != name or digest(bundle / name) != expected:
            raise ValueError(f"Baseline file differs from manifest: {name}")
    original = json.loads((bundle / "summary.json").read_text())
    with np.load(bundle / "equilibrium.npz", allow_pickle=False) as archive:
        coefficients = archive["coefficients"]
        duration = float(archive["duration_s"])
        identity = json.loads(str(archive["identity_json"]))
    profile = json.loads((bundle / "profile.json").read_text())
    expected = {
        "reference_sha256": digest(bundle / "reference.npz"),
        "profile_sha256": hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest(),
        "simulation_config": original["simulation_config"],
        "shoe": {
            "artifact_sha256": digest(bundle / "digital_shoe.json"),
            "mount_m": original["shoe"]["mount_m"],
            "static_pitch_rad": original["shoe"]["static_pitch_rad"],
            "friction": original["shoe"]["friction"],
            "device": "cuda:0",
        },
    }
    if identity != expected or coefficients.shape != (12, 4) or not np.isfinite(coefficients).all():
        raise ValueError("Saved controller identity is inconsistent")
    with np.load(bundle / "reference.npz", allow_pickle=False) as archive:
        if not np.isclose(duration, float(archive["time_s"][-1]), rtol=0, atol=1e-12):
            raise ValueError("Controller duration differs from measured reference")
    controller = {
        "coefficients": coefficients.tolist(),
        "duration_s": duration,
        "channels": ["hip_x_m", "hip_z_m", "knee_rad", "ankle_rad"],
        "spline": "twelve-point uniform clamped cubic equilibrium; original wall time",
        "gains": {
            k: profile[k]
            for k in ("hip_stiffness_n_m", "hip_damping_ns_m", "joint_stiffness_nm_rad", "joint_damping_nms_rad")
        },
    }
    return {"manifest": manifest, "controller": controller, "controller_sha256": content_digest(controller)}


def create(
    output: Path,
    *,
    bundle: Path = DEFAULT_BUNDLE,
    fixtures: tuple[str, ...] = ("rearfoot_punch", "fullfoot_last"),
    suite: str = "sensitivity",
) -> dict:
    """Create the selected shoe suite and seal inputs without a fit or rollout."""
    output, bundle = Path(output).resolve(), Path(bundle).resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new campaign directory: {output}")
    if not fixtures or len(set(fixtures)) != len(fixtures):
        raise ValueError("Request distinct nonempty fixture names")
    receipt = verify_bundle(bundle)
    original = json.loads((bundle / "digital_shoe.json").read_text())
    selected_conditions = conditions(suite)
    if suite == "paper_compression":
        from .paper_materials import get_paper_material_metadata  # noqa: PLC0415

        for case in selected_conditions:
            if case["family"] != "baseline":
                case["paper_material"] = get_paper_material_metadata(case["id"], original)
    variant_values = [(case, build_variant(original, case)) for case in selected_conditions]
    output.mkdir(parents=True)
    frozen = output / "baseline"
    frozen.mkdir()
    for name in [*receipt["manifest"]["files_sha256"], "baseline.json"]:
        shutil.copyfile(bundle / name, frozen / name)
    listed = []
    for case, artifact in variant_values:
        directory = output / "conditions" / case["id"]
        directory.mkdir(parents=True)
        target = directory / "digital_shoe.json"
        if case["family"] == "baseline":
            shutil.copyfile(frozen / "digital_shoe.json", target)
        else:
            write_json(target, artifact)
        load_artifact(target)
        saved = {
            **case,
            "artifact": str(target.relative_to(output)),
            "artifact_sha256": digest(target),
            "controller_sha256": receipt["controller_sha256"],
        }
        write_json(directory / "condition.json", saved)
        listed.append({**saved, "condition_sha256": digest(directory / "condition.json")})
    plan = {
        "schema": SCHEMA,
        "suite": suite,
        "baseline_source": str(bundle),
        "baseline": "baseline",
        "baseline_manifest_sha256": digest(frozen / "baseline.json"),
        "controller": receipt["controller"],
        "controller_sha256": receipt["controller_sha256"],
        "source_sha256": source_identity(),
        "conditions": listed,
        "native_dt_s": json.loads((frozen / "summary.json").read_text())["simulation_config"]["dt_s"],
        "refinement_dt_factor": 0.5,
        "controller_optimization": "forbidden",
        "protocols": ["primary"]
        + (["clearance_matched_geometry_only"] if any(case["family"] == "geometry" for case in listed) else []),
        "initialization": "identical baseline state/velocity/time zero; independent zero contact histories",
        "hysteresis": {
            "requested_fixtures": list(fixtures),
            "protocol": BENCH_PROTOCOL,
            "waveform": "synthetic haversine; identical absolute depth and timing across conditions",
            "refinement_dt_factor": 0.5,
            "available_baseline_fixtures": list(original["instron_fixtures"]),
            "scope": "independent prescribed-displacement bench; motion failure does not suppress bench",
        },
        "comparison": {
            "contact_threshold_n": 1.0,
            "failure_event_tolerance_s": 0.001,
            "primary_reference": "unchanged simulated baseline, not old-shoe measured fit loss",
        },
        "qualification": "No experiment has run. Saved baseline acceptance is not new campaign qualification.",
    }
    write_json(output / "plan.json", plan)
    (output / "plan.sha256").write_text(digest(output / "plan.json") + "\n")
    return plan


def verify(campaign: Path, *, check_source: bool = True) -> dict:
    """Fail closed on any changed sealed input or campaign physics source."""
    campaign = Path(campaign)
    if digest(campaign / "plan.json") != (campaign / "plan.sha256").read_text().strip():
        raise ValueError("Campaign plan changed after sealing")
    plan = json.loads((campaign / "plan.json").read_text())
    if plan["schema"] != SCHEMA:
        raise ValueError("Unsupported campaign schema")
    if digest(campaign / "baseline/baseline.json") != plan["baseline_manifest_sha256"]:
        raise ValueError("Baseline manifest changed")
    receipt = verify_bundle(campaign / "baseline")
    if receipt["controller_sha256"] != plan["controller_sha256"] or receipt["controller"] != plan["controller"]:
        raise ValueError("Frozen controller changed")
    for case in plan["conditions"]:
        path = campaign / case["artifact"]
        if path.resolve().parent != (campaign / "conditions" / case["id"]).resolve():
            raise ValueError("Condition artifact escaped its directory")
        if (
            digest(path) != case["artifact_sha256"]
            or digest(path.parent / "condition.json") != case["condition_sha256"]
        ):
            raise ValueError(f"Condition inputs changed: {case['id']}")
    if check_source:
        from ..cartesian.gpu.provenance import source_snapshot  # noqa: PLC0415

        current = source_snapshot()
        if any(plan["source_sha256"].get(path) != value for path, value in current.items()):
            raise ValueError("Campaign physics source changed; create and qualify a new campaign")
    return plan


def qualify(campaign: Path) -> None:
    """Run existing CPU/GPU/contact and mixed-world qualification without fitting."""
    from .. import pipeline  # noqa: PLC0415

    campaign = Path(campaign).resolve()
    verify(campaign)
    target = campaign / "qualification"
    pipeline.main(["--baseline", str(campaign / "baseline"), "--output", str(target), "--stage", "prepare"])
    pipeline.main(["--output", str(target), "--stage", "validate"])
    _require_qualification(campaign)


def _require_qualification(campaign: Path) -> None:
    """Revalidate numerical evidence instead of trusting a passed Boolean."""
    from ..cartesian.gpu.__main__ import _validation  # noqa: PLC0415

    target = campaign / "qualification"
    _validation(
        target / "single/benchmark.json",
        target / "baseline",
        mixed=False,
        expected_controls=12,
        contact_evidence=target / "contact/report.json",
    )
    _validation(target / "mixed/benchmark.json", target / "baseline", mixed=True, expected_controls=12)


def _inputs(campaign: Path):
    bundle = campaign / "baseline"
    with np.load(bundle / "reference.npz", allow_pickle=False) as archive:
        reference = dict(archive)
    with np.load(bundle / "equilibrium.npz", allow_pickle=False) as archive:
        spline = Spline(float(archive["duration_s"]), archive["coefficients"])
    profile = load_profile(bundle / "profile.json")
    summary = json.loads((bundle / "summary.json").read_text())
    return reference, profile, spline, summary


def _initial_bottom_height(artifact: Path, reference, profile, metadata) -> float:
    """Compute nominal initial shoe clearance without advancing contact."""
    bed = load_artifact(artifact).column_bed
    body = Body(
        reference["lengths_m"],
        reference["endpoint_local_m"],
        profile["masses_kg"],
        profile["com_local_m"],
        profile["inertias_kg_m2"],
    )
    q = reference["state"][0]
    ankle, _, _ = body.point(q, 2, np.zeros(2))
    angle = body.angle(q, 2) - metadata["static_pitch_rad"]
    local = bed.anchor_bottom_m - np.asarray(metadata["mount_m"])
    return float(np.min(ankle[1] + np.sin(angle) * local[:, 0] + np.cos(angle) * local[:, 2]))


def _foundation_config(artifact: Path, baseline: Path, ground: float) -> FoundationConfig | None:
    """Keep tangential stiffness per area fixed when rectangle edge cells change."""
    original_area = load_artifact(baseline).column_bed.area_m2
    case_area = load_artifact(artifact).column_bed.area_m2
    if ground == 0.0 and np.array_equal(original_area, case_area):
        return None
    mean_area = float(np.mean(original_area))
    return FoundationConfig(
        ground_height_m=ground,
        normal_damping=0.0,
        friction_stiffness=10000.0,
        friction=10.0,
        mu=0.8,
        friction_stiffness_per_area=10000.0 / mean_area,
        friction_damping_per_area=10.0 / mean_area,
    )


def _save_rollout(
    campaign,
    case,
    protocol,
    resolution,
    trace,
    run_summary,
    shoe,
    config,
    settings,
    reference,
    profile,
    plan,
    timing,
    diagnostic,
):
    directory = campaign / "conditions" / case["id"] / protocol / resolution
    directory.mkdir(parents=True, exist_ok=False)
    from ..cartesian.gpu.provenance import source_snapshot  # noqa: PLC0415

    objective = _Objective(reference, settings)
    residual, metrics, costs = objective.evaluate(trace, run_summary)
    ground = float(shoe.foundation.ground_height_m)
    summary = {
        "schema": "frozen_shoe_rollout_1",
        "condition": case["id"],
        "protocol": protocol,
        "controller_sha256": plan["controller_sha256"],
        "optimization": "none; immutable saved controller",
        "run": run_summary,
        "shoe": shoe.metadata,
        "simulation_config": asdict(config),
        "fit_config": asdict(settings),
        "source_sha256": source_snapshot(),
        "campaign_source_sha256": source_identity(),
        "metrics": metrics,
        "components": costs,
        "loss": float(residual @ residual) if residual is not None else None,
        "observations": observations(trace, run_summary, ground_height_m=ground),
        "ground_height_m": ground,
        "timing": timing,
        "diagnostic_unqualified": diagnostic,
        "accepted": False,
        "qualification": "Synthetic frozen-controller response; old-shoe fit error is descriptive, not case acceptance.",
        "initial_nominal_clearance_m": _initial_bottom_height(
            Path(shoe.metadata["path"]), reference, profile, shoe.metadata
        )
        - ground,
    }
    if str(shoe.device).startswith("cuda"):
        from ..cartesian.gpu.benchmark import execution_identity  # noqa: PLC0415

        summary["execution_identity"] = execution_identity()
    np.savez_compressed(directory / "trace.npz", **trace)
    shutil.copyfile(campaign / "baseline/reference.npz", directory / "reference.npz")
    shutil.copyfile(campaign / "baseline/profile.json", directory / "profile.json")
    shutil.copyfile(campaign / "baseline/equilibrium.npz", directory / "frozen_parent_equilibrium.npz")
    write_json(directory / "summary.json", summary)
    write_json(
        directory / "files.json",
        {
            name: digest(directory / name)
            for name in ("trace.npz", "reference.npz", "profile.json", "frozen_parent_equilibrium.npz", "summary.json")
        },
    )
    return summary


def _existing(directory: Path) -> bool:
    """Resume only complete outputs with intact artifact hashes."""
    if not directory.exists():
        return False
    if not (directory / "files.json").exists():
        raise FileExistsError(f"Incomplete output exists; preserve it and choose a new campaign: {directory}")
    for name, expected in json.loads((directory / "files.json").read_text()).items():
        if Path(name).name != name or digest(directory / name) != expected:
            raise ValueError(f"Saved rollout changed: {directory / name}")
    return True


def run(
    campaign: Path,
    *,
    selected: list[str] | None = None,
    device: str = "cuda:0",
    clearance_matched: bool = True,
    diagnostic: bool = False,
) -> None:
    """Integrate frozen cases at native and half dt, keeping all failure outcomes."""
    campaign = Path(campaign).resolve()
    plan = verify(campaign)
    if not diagnostic:
        _require_qualification(campaign)
    available = {case["id"]: case for case in plan["conditions"]}
    if selected and set(selected) - available.keys():
        raise ValueError(f"Unknown conditions: {sorted(set(selected) - available.keys())}")
    baseline_case = next(case for case in available.values() if case["family"] == "baseline")
    cases = (
        list(available.values())
        if selected is None
        else [available[key] for key in dict.fromkeys([baseline_case["id"], *selected])]
    )
    reference, profile, spline, original = _inputs(campaign)
    settings = FitConfig(**original["fit_config"])
    native_config = Config(**original["simulation_config"])
    metadata = original["shoe"]
    baseline_artifact = campaign / "baseline/digital_shoe.json"
    groups = [[case for case in cases if case["family"] in ("baseline", "material")]]
    groups.extend([case] for case in cases if case["family"] == "geometry")
    for group in filter(None, groups):
        protocols = ["primary"]
        if clearance_matched and group[0]["family"] == "geometry":
            protocols.append("clearance_matched")
        for protocol in protocols:
            ground = 0.0
            artifact = campaign / group[0]["artifact"]
            if protocol == "clearance_matched":
                ground = _initial_bottom_height(artifact, reference, profile, metadata) - _initial_bottom_height(
                    baseline_artifact, reference, profile, metadata
                )
            foundation_config = _foundation_config(artifact, baseline_artifact, ground)
            for resolution, factor in (("native", 1.0), ("refined", 0.5)):
                pending = [
                    case
                    for case in group
                    if not _existing(campaign / "conditions" / case["id"] / protocol / resolution)
                ]
                if not pending:
                    continue
                config = replace(native_config, dt_s=native_config.dt_s * factor)
                started = perf_counter()
                if device == "cpu":
                    for case in pending:
                        shoe = Shoe(
                            campaign / case["artifact"],
                            metadata["mount_m"],
                            metadata["static_pitch_rad"],
                            device="cpu",
                            foundation_config=foundation_config,
                        )
                        trace, summary = simulate(reference, profile, spline, shoe, config=config)
                        _save_rollout(
                            campaign,
                            case,
                            protocol,
                            resolution,
                            trace,
                            summary,
                            shoe,
                            config,
                            settings,
                            reference,
                            profile,
                            plan,
                            {"elapsed_s": perf_counter() - started, "batch_worlds": 1},
                            diagnostic,
                        )
                else:
                    from ..cartesian.gpu.engine import Engine  # noqa: PLC0415

                    engine = Engine(
                        reference,
                        profile,
                        campaign / pending[0]["artifact"],
                        metadata["mount_m"],
                        metadata["static_pitch_rad"],
                        config=config,
                        settings=settings,
                        world_count=len(pending),
                        device=device,
                        foundation_config=foundation_config,
                    )
                    materials = [load_artifact(campaign / case["artifact"]).material for case in pending]
                    engine.foundation.set_world_materials(materials)
                    coefficients = np.repeat(spline.coefficients[None], len(pending), axis=0)
                    engine.evaluate(coefficients)
                    if not np.array_equal(engine.coefficients.numpy(), coefficients):
                        raise RuntimeError("Frozen coefficients changed on device")
                    timing = {
                        "batch_worlds": len(pending),
                        "batch_elapsed_s": perf_counter() - started,
                        "rollout_wall_s": engine.last_wall_s,
                        "setup_wall_s": engine.setup_wall_s,
                        "capture_wall_s": engine.capture_wall_s,
                    }
                    for world, case in enumerate(pending):
                        trace, summary = engine.trace(world)
                        shoe = Shoe(
                            campaign / case["artifact"],
                            metadata["mount_m"],
                            metadata["static_pitch_rad"],
                            device=device,
                            foundation_config=foundation_config,
                        )
                        _save_rollout(
                            campaign,
                            case,
                            protocol,
                            resolution,
                            trace,
                            summary,
                            shoe,
                            config,
                            settings,
                            reference,
                            profile,
                            plan,
                            timing,
                            diagnostic,
                        )
                    del engine
                verify(campaign)
                print(f"Saved {len(pending)} frozen {protocol}/{resolution} conditions", flush=True)
    compare_saved(campaign)


def _read_rollout(directory):
    with np.load(directory / "trace.npz", allow_pickle=False) as archive:
        trace = dict(archive)
    return trace, json.loads((directory / "summary.json").read_text())


def compare_saved(campaign: Path) -> None:
    """Publish complete or failure-aware paired comparisons for all saved cases."""
    plan = verify(campaign)
    baseline = next(case for case in plan["conditions"] if case["family"] == "baseline")
    baseline_dir = campaign / "conditions" / baseline["id"] / "primary/native"
    if not (baseline_dir / "summary.json").exists():
        return
    bt, bs = _read_rollout(baseline_dir)
    for case in plan["conditions"]:
        for protocol in ("primary", "clearance_matched"):
            directory = campaign / "conditions" / case["id"] / protocol
            if not (directory / "native/summary.json").exists():
                continue
            ct, cs = _read_rollout(directory / "native")
            result = {"baseline_delta": compare(ct, cs["run"], bt, bs["run"]), "refinement": {"performed": False}}
            if (directory / "refined/summary.json").exists():
                ft, fs = _read_rollout(directory / "refined")
                result["refinement"] = refinement(ct, cs["run"], ft, fs["run"], settings=FitConfig(**cs["fit_config"]))
            write_json(directory / "comparison.json", result)


def bench(campaign: Path, *, selected: list[str] | None = None, device: str = "cuda:0") -> None:
    """Run paired-timestep fixture cycles independently of stance termination."""
    from .hysteresis import run_hysteresis  # noqa: PLC0415

    campaign = Path(campaign).resolve()
    plan = verify(campaign)
    cases = [case for case in plan["conditions"] if selected is None or case["id"] in selected]
    if selected and set(selected) - {case["id"] for case in cases}:
        raise ValueError("Unknown bench condition")
    for case in cases:
        base = campaign / "conditions" / case["id"] / "hysteresis"
        for output, factor in ((base, 1.0), (base / "refined", 0.5)):
            if _existing(output):
                continue
            protocol = dict(plan["hysteresis"]["protocol"])
            protocol["dt_s"] *= factor
            run_hysteresis(
                campaign / case["artifact"],
                output,
                fixtures=tuple(plan["hysteresis"]["requested_fixtures"]),
                device=device,
                **protocol,
            )
            write_json(
                output / "protocol_identity.json",
                {
                    "artifact_sha256": case["artifact_sha256"],
                    "controller_applied": False,
                    "protocol": protocol,
                    "source_sha256": source_identity(),
                    "requested_fixtures": plan["hysteresis"]["requested_fixtures"],
                    "device": device,
                },
            )
            write_json(output / "files.json", {path.name: digest(path) for path in output.iterdir() if path.is_file()})
            verify(campaign)
        compare_bench(base)
        print(f"Saved native/refined hysteresis for {case['id']}", flush=True)


def compare_bench(directory: Path) -> dict:
    """Report native/half-step force and work agreement without assuming loop closure."""
    native = json.loads((directory / "summary.json").read_text())
    fine = json.loads((directory / "refined/summary.json").read_text())
    result = {
        "fixtures": {},
        "force_rtol": 0.01,
        "force_atol_n": 0.1,
        "work_rtol": 0.01,
        "work_atol_j": 1e-4,
        "qualification": "Numerical bench agreement is separate from material-state closure and physical validation.",
    }
    for fixture, coarse_entry in native["fixtures"].items():
        fine_entry = fine["fixtures"].get(fixture, {})
        comparison = {
            "performed": False,
            "passed": False,
            "native_status": coarse_entry["status"],
            "refined_status": fine_entry.get("status"),
        }
        coarse_file = coarse_entry.get("files", {}).get("npz")
        fine_file = fine_entry.get("files", {}).get("npz")
        if coarse_file and fine_file:
            with np.load(directory / coarse_file, allow_pickle=False) as archive:
                ct, cf = archive["time_s"], archive["force_n"]
            with np.load(directory / "refined" / fine_file, allow_pickle=False) as archive:
                ft, ff = archive["time_s"], archive["force_n"]
            if len(ct) and len(ft):
                upper = min(ct[-1], ft[-1])
                lower = max(ct[0], ft[0])
                time = np.unique(np.r_[ct[(ct >= lower) & (ct <= upper)], ft[(ft >= lower) & (ft <= upper)]])
                force_error = float(np.max(np.abs(np.interp(time, ct, cf) - np.interp(time, ft, ff))))
                limit = max(result["force_atol_n"], result["force_rtol"] * float(np.max(np.abs(ff))))
                cm, fm = coarse_entry.get("metrics") or {}, fine_entry.get("metrics") or {}
                work_keys = ("work_input_j", "work_returned_j", "work_net_j")
                work_errors = {
                    key: abs(cm[key] - fm[key])
                    for key in work_keys
                    if cm.get(key) is not None and fm.get(key) is not None
                }
                work_limit = max(result["work_atol_j"], result["work_rtol"] * abs(fm.get("work_input_j") or 0.0))
                complete = coarse_entry["status"] == fine_entry.get("status") == "completed"
                comparison.update(
                    performed=True,
                    common_support_s=[float(lower), float(upper)],
                    maximum_force_difference_n=force_error,
                    force_limit_n=limit,
                    force_agreement=bool(complete and force_error <= limit),
                    work_comparison_available=len(work_errors) == 3,
                    work_difference_j=work_errors,
                    work_limit_j=work_limit,
                    passed=bool(
                        complete
                        and force_error <= limit
                        and len(work_errors) == 3
                        and all(value <= work_limit for value in work_errors.values())
                    ),
                    material_state_closed_native=cm.get("is_state_closed", False),
                    material_state_closed_refined=fm.get("is_state_closed", False),
                )
        result["fixtures"][fixture] = comparison
    write_json(directory / "comparison.json", result)
    return result


def reports(campaign: Path, *, individual: bool = False) -> Path:
    """Render saved comparisons and optionally audit individual spring replays."""
    from .report import write_report  # noqa: PLC0415

    campaign = Path(campaign).resolve()
    verify(campaign)
    for receipt in (campaign / "conditions").rglob("files.json"):
        _existing(receipt.parent)
    if individual:
        from ..cartesian.report import write_report as write_rollout_report  # noqa: PLC0415

        for path in (campaign / "conditions").glob("*/*/native/summary.json"):
            trace, summary = _read_rollout(path.parent)
            with np.load(path.parent / "reference.npz", allow_pickle=False) as archive:
                reference = dict(archive)
            profile = load_profile(path.parent / "profile.json")
            write_rollout_report(path.parent, reference, trace, summary, profile=profile)
    return write_report(campaign)
