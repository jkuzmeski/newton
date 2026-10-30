# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare paper-derived foam substitutions with a frozen Hogan runner."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import subprocess
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

from projects.digital_shoe.material import hyperfoam_pressure_numpy
from projects.impedance_instron.cartesian.shoe import Shoe

from .identify import ForceObservation, load_trials, predict_many, scenario, score
from .runner import RolloutConfig, Runner

PAPER_URL = "https://arxiv.org/html/2602.12694v2"
PAPER_FOAMS = {
    "ff_leap": {
        "label": "FF LEAP — elastic surrogate",
        "table": 1,
        "compression_kpa": [0, 12.79, 32.65, 45.64, 55.37, 65.47, 77.17, 91.21, 108.06, 129.08, 156.19, 191.67, 241.60],
        "compression_sd_kpa": [0, 1.24, 3.98, 5.47, 5.84, 6.08, 6.33, 6.69, 6.98, 7.59, 8.50, 10.09, 13.73],
        "shear_modulus_pa": 117160.0,
        "reported_compressive_modulus_pa": 299220.0,
        "reported_compression_energy_return": 0.895,
    },
    "ff_turbo_plus": {
        "label": "FF TURBO PLUS — elastic surrogate",
        "table": 2,
        "compression_kpa": [0, 9.41, 32.75, 51.40, 64.33, 76.82, 90.92, 107.86, 128.72, 155.11, 189.78, 236.96, 305.43],
        "compression_sd_kpa": [0, 1.08, 1.96, 4.41, 5.62, 6.22, 6.66, 7.34, 8.32, 9.66, 11.63, 14.90, 19.56],
        "shear_modulus_pa": 219120.0,
        "reported_compressive_modulus_pa": 267940.0,
        "reported_compression_energy_return": 0.846,
    },
}


def sha256(path: str | Path) -> str:
    """Hash the exact bytes of a saved input or output."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    """Write finite machine-readable evidence."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def fit_compression(foam: dict) -> dict:
    """Fit two positive-modulus Ogden-Hill terms to published compression means.

    Search the term exponents deterministically and solve the two nonnegative
    moduli by linear least squares. This is a reduced compression approximation,
    not the paper's full three-dimensional constitutive model.
    """
    strain = np.linspace(0.0, 0.6, 13)
    target = np.asarray(foam["compression_kpa"]) * 1000.0
    best = None
    for alpha in np.geomspace(0.02, 120.0, 180):
        first = hyperfoam_pressure_numpy(strain, 1.0, alpha, 0.0, 1.0, 0.0, 1.0, 0.01)
        for alpha2 in -np.geomspace(0.02, 12.0, 140):
            second = hyperfoam_pressure_numpy(strain, 0.0, 1.0, 1.0, alpha2, 0.0, 1.0, 0.01)
            design = np.column_stack((first, second))
            moduli = np.linalg.lstsq(design[1:], target[1:], rcond=None)[0]
            if np.any(moduli < 0.0) or moduli[0] <= 0.0:
                continue
            error = design @ moduli - target
            cost = float(error @ error)
            if best is None or cost < best[0]:
                best = (cost, alpha, alpha2, moduli, error)
    if best is None:
        raise ValueError("No admissible two-term compression approximation")
    _, alpha, alpha2, moduli, error = best
    predicted = target + error
    return {
        "parameters": {
            "instantaneous_shear_modulus_pa": float(moduli[0]),
            "hyperfoam_exponent": float(alpha),
            "instantaneous_shear_modulus_2_pa": float(moduli[1]),
            "hyperfoam_exponent_2": float(alpha2),
            "equilibrium_fraction": 1.0,
            "effective_poisson_ratio": 0.0,
            "equilibrium_shear_modulus_override_pa": foam["shear_modulus_pa"],
        },
        "strain": strain.tolist(),
        "target_pressure_pa": target.tolist(),
        "predicted_pressure_pa": predicted.tolist(),
        "error_pa": error.tolist(),
        "rmse_pa": float(np.sqrt(np.mean(error**2))),
        "maximum_absolute_error_pa": float(np.max(np.abs(error))),
        "r_squared": float(1.0 - np.sum(error**2) / np.sum((target - target.mean()) ** 2)),
        "small_strain_compressive_tangent_pa": float(2.0 * moduli.sum()),
        "support": "Compression strain 0-0.6; extrapolation is flagged, not treated as validated.",
        "objective": "Unweighted squared pressure error at the 13 published mean compression points.",
        "loss_assumption": "Hyperelastic equilibrium approximation; Maxwell branch disabled. Measured energy return is not reproduced.",
    }


def work_metrics(trace: dict) -> dict:
    """Integrate signed actuator work and peak power over the saved native window."""
    dt = np.diff(trace["time_s"])
    power = trace["load"][:, 3:] * trace["velocity"][:-1, 3:]
    positive = np.sum(np.maximum(power, 0.0) * dt[:, None], axis=0)
    negative = np.sum(np.minimum(power, 0.0) * dt[:, None], axis=0)
    return {
        "joint_positive_work_j": positive.tolist(),
        "joint_negative_work_j": negative.tolist(),
        "joint_net_work_j": (positive + negative).tolist(),
        "joint_peak_positive_power_w": np.maximum(power.max(axis=0), 0.0).tolist() if len(power) else [0.0] * 3,
        "joint_peak_absorbed_power_w": np.maximum(-power.min(axis=0), 0.0).tolist() if len(power) else [0.0] * 3,
        "total_positive_work_j": float(positive.sum()),
        "total_absorbed_work_j": float(-negative.sum()),
    }


def rematerialize(trials: list, artifact: Path) -> list:
    """Replace only the shoe material artifact while preserving matched trial inputs."""
    shoes = {}
    result = []
    for trial in trials:
        original = trial.shoe
        key = (tuple(original.mount_m), original.static_pitch_rad)
        if key not in shoes:
            shoes[key] = Shoe(artifact, *key)
        provenance = trial.provenance | {"shoe_artifact": str(artifact.resolve()), "shoe_sha256": sha256(artifact)}
        result.append(replace(trial, shoe=shoes[key], provenance=provenance))
    return result


def make_conditions(artifact: Path, output: Path) -> list[dict]:
    """Clone the same geometry into original, elastic-control, and two foam conditions."""
    raw = json.loads(artifact.read_text())
    output.mkdir()
    conditions = []
    for name, label in [
        ("original", "Original calibrated shoe"),
        ("original_elastic", "Original — elasticity only"),
        *[(k, v["label"]) for k, v in PAPER_FOAMS.items()],
    ]:
        data = copy.deepcopy(raw)
        parameters = data["constitutive_model"]["parameters"]
        fit = None
        if name == "original_elastic":
            fraction = parameters["equilibrium_fraction"]
            parameters["instantaneous_shear_modulus_pa"] *= fraction
            parameters["instantaneous_shear_modulus_2_pa"] *= fraction
            parameters["equilibrium_fraction"] = 1.0
        elif name in PAPER_FOAMS:
            fit = fit_compression(PAPER_FOAMS[name])
            parameters.update(fit["parameters"])
        if name != "original":
            shear = (
                parameters.get("equilibrium_shear_modulus_override_pa")
                or (parameters["instantaneous_shear_modulus_pa"] + parameters["instantaneous_shear_modulus_2_pa"])
                * parameters["equilibrium_fraction"]
            )
            thickness = np.asarray(data["column_bed"]["rest_length_m"])
            parameters["pasternak_n_per_m"] = float(shear * thickness.mean())
            data["shoe"]["id"] = raw["shoe"]["id"] + "__" + name
            data["shoe"]["model_scope"] = (
                "Virtual homogeneous material substitution in unchanged original shoe geometry; not an ASICS shoe model."
            )
            data["constitutive_model"]["derived_quantities"] = {
                "equilibrium_shear_modulus_pa": shear,
                "small_strain_compressive_modulus_pa": 2.0
                * (parameters["instantaneous_shear_modulus_pa"] + parameters["instantaneous_shear_modulus_2_pa"])
                * parameters["equilibrium_fraction"],
                "pasternak_rule": "k_i = independently specified equilibrium shear modulus * rest_length_m[i]",
            }
            data["validation"] = {
                "status": "material substitution, not original intact-shoe calibration",
                "claim_boundary": "Compression approximation and independent shear tangent only; no hysteresis, full 3D, rate, fatigue, or whole-shoe validation.",
            }
            data["provenance"] = {
                "generator": __name__,
                "geometry_source_sha256": sha256(artifact),
                "geometry_source": str(artifact.resolve()),
                "paper": PAPER_URL if fit else None,
                "condition": name,
                "material_fit": fit,
            }
        path = output / f"{name}.json"
        if name == "original":
            shutil.copy2(artifact, path)
        else:
            write_json(path, data)
        conditions.append(
            {"id": name, "label": label, "artifact": str(path.resolve()), "artifact_sha256": sha256(path), "fit": fit}
        )
    return conditions


def main(argv=None, *, condition_builder=None, trial_builder=None, study_metadata=None):
    """Plan or run an independent frozen-controller material comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--model", type=Path, default=Path(__file__).parent / "baselines/generative_runner_score_slider_20261029.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    trials = load_trials(
        args.dataset,
        mount_m=[-0.03186147427106201, 0.0, 0.10943209684347802],
        speed_m_s=3.65,
        force_observation=ForceObservation(20.0, 1.0, 1.0),
    )
    if not trials or any(not t.provenance["compatibility"]["passed"] for t in trials):
        raise ValueError("Material comparison requires compatible matched trials")
    root = args.output.resolve()
    root.mkdir(parents=True)
    conditions = (condition_builder or make_conditions)(trials[0].shoe.artifact_path, root / "materials")
    cfg = RolloutConfig(dt_s=0.000125, compression_limit=0.99)
    plan = {
        "run_id": root.name,
        "domain": "impedance-instron",
        "category": "materials-testing",
        "planned_at": datetime.now(timezone.utc).isoformat(),
        "question": "How do paper-derived foam compression and independent shear responses change joint mechanical work with the accepted controller frozen?",
        "baseline": {
            "model": str(args.model.resolve()),
            "model_sha256": sha256(args.model),
            "shoe": str(trials[0].shoe.artifact_path),
            "shoe_sha256": sha256(trials[0].shoe.artifact_path),
        },
        "changed_variables": conditions,
        "command": {
            "module": __name__,
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        },
        "effective_configuration": asdict(cfg),
        "seed": "No stochastic sampling or fitting of controller parameters.",
        "splits": {split: [t.id for t in trials if t.split == split] for split in ("train", "eval")},
        "selection_rule": "No controller or material winner selected; all declared conditions reported. Primary performance comparison uses held-out matched completed windows; training windows are secondary.",
        "budget": {
            "evaluation_and_capture_minutes": 15,
            "reporting_minutes": 10,
            "execution": "Serial exact CUDA evaluations; native saved-pose CPU contact replay.",
        },
        "stopping_conditions": "One frozen evaluation per material and trial, with unchanged numerical screens. Failures retained; no refitting.",
        "qualification_checks": [
            "Identical controller hash and geometry across conditions",
            "Unchanged trial initial states, body, task, and friction settings",
            "Compression approximation errors and out-of-support strains recorded",
            "Verified source/shoe/state-matched per-column contact replay for first held-out stance in every condition",
            "First held-out stance at half timestep per condition",
        ],
        "replay_samples": [
            {
                "condition": c["id"],
                "sample_id": "FR3_1_eval_000",
                "split": "eval",
                "selection": "First compatible held-out sample, fixed before evaluation",
            }
            for c in conditions
        ],
        "capture_plan": {
            "native": "All states, forces, actuator schedules, scenarios and per-trial metrics saved by this driver.",
            "deformation": "Run source-checked capture_replay.py against each condition's own native states; save columns, rest lengths, driven membership, endpoints and verification.",
            "report": "Materials testing comparison: joint positive/absorbed/net work, power, GRF/contact/compression; same-sample motion per material; paper-fit diagnostics and limitations.",
        },
        "paper": {
            "url": PAPER_URL,
            "data": PAPER_FOAMS,
            "interpretation": "Tables 1-2 compression means (positive stress magnitude), independent reported shear stiffness; hyperelastic approximation with no measured hysteresis reproduction. 0.6 maximum measured compression strain. Existing Puma geometry, no ASICS plate or dual-layer arrangement.",
        },
    }
    if study_metadata:
        plan.update(study_metadata(conditions, trials, args))
    write_json(root / "experiment-plan.json", plan)
    roles = (
        "provenance",
        "configuration",
        "selected_model",
        "motion_trace",
        "motion_viewer",
        "geometry",
        "deformation",
        "deformation_validation",
        "biomechanics",
        "training_history",
        "metrics",
        "qualification",
    )
    template = {
        "schema": "instron-experiment-artifacts-v1",
        "replay": {},
        "artifacts": {
            role: {"status": "missing", "reason": "Capture and verify before report completion."} for role in roles
        },
    }
    template.update(
        domain="impedance-instron",
        run_id=root.name,
        experiment_status="planned",
        fitting_performed=False,
        approved_roots=[str(Path.cwd())],
    )
    template["artifacts"]["training_history"] = {
        "status": "not_applicable",
        "reason": "Controller frozen; material approximation diagnostics saved separately. No controller optimization performed.",
    }
    write_json(root / "experiment-artifacts.json", template)
    write_json(root / "paper-data.json", plan["paper"])
    if not args.run:
        print(root / "experiment-plan.json")
        return
    started = datetime.now(timezone.utc).isoformat()
    source_paths = [*Path("projects/impedance_instron").rglob("*.py"), *Path("projects/digital_shoe").glob("*.py")]
    source_paths = [p for p in source_paths if not p.name.endswith("_report.py")]
    sources = {str(p.relative_to("projects")): sha256(p) for p in source_paths}
    for path in source_paths:
        destination = root / "source_snapshot" / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    runner = Runner.load(args.model)
    shutil.copy2(args.model, root / "runner.json")
    write_json(
        root / "provenance.json",
        {
            "run_started_at": started,
            "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "source_sha256": sources,
            "controller_sha256": sha256(args.model),
            "manifest_sha256": sha256(args.dataset / "manifest.json"),
            "paper": PAPER_URL,
            "primary_split": "eval",
            "geometry_source_sha256": sha256(trials[0].shoe.artifact_path),
        },
    )
    state = {
        "schema": "hogan-frozen-material-comparison-v1",
        "status": "running",
        "started_at": started,
        "conditions": [],
        "controller_sha256": sha256(args.model),
    }
    write_json(root / "summary.json", state)
    overall = perf_counter()
    for condition in conditions:
        name = condition["id"]
        directory = root / name
        directory.mkdir()
        matched = (
            trial_builder(trials, condition) if trial_builder else rematerialize(trials, Path(condition["artifact"]))
        )
        tick = perf_counter()
        predictions = predict_many([runner], matched, cfg, device=args.device)[0]
        native = {
            "schema": "hogan-material-condition-v1",
            "condition": condition,
            "fitting_performed": False,
            "reference_inputs_used": False,
            "device": args.device,
            "rollout": asdict(cfg),
            "source_sha256": sources,
            "input_model_sha256": sha256(args.model),
            "splits": {},
            "trials": [],
        }
        for index, (trial, (trace, summary)) in enumerate(zip(matched, predictions, strict=True)):
            result = score(trace, summary, trial, runner) | work_metrics(trace)
            compression = trace["compression_fraction"]
            result.update(
                peak_compression_fraction=float(compression.max()) if len(compression) else 0.0,
                intervals_outside_paper_compression_support=int(np.count_nonzero(compression > 0.6))
                if condition["fit"]
                else None,
                total_intervals=len(compression),
            )
            result.update(id=trial.id, split=trial.split)
            native["trials"].append(
                {
                    "id": trial.id,
                    "split": trial.split,
                    **trial.provenance,
                    "trace": f"trace_{index:03d}.npz",
                    "scenario": f"scenario_{index:03d}.json",
                    "metrics": result,
                }
            )
            np.savez_compressed(directory / f"trace_{index:03d}.npz", **trace)
            write_json(directory / f"scenario_{index:03d}.json", scenario(trial, cfg))
        for split in ("train", "eval"):
            rows = [t["metrics"] for t in native["trials"] if t["split"] == split]
            native["splits"][split] = {
                "mean_loss": float(np.mean([r["loss"] for r in rows])),
                "failed": sum(r["status"] != "completed" for r in rows),
                "trials": rows,
            }
        native["wall_s"] = perf_counter() - tick
        shutil.copy2(args.model, directory / "runner.json")
        write_json(directory / "summary.json", native)
        primary = next(t for t in matched if t.id == "FR3_1_eval_000")
        fine_trace, fine_summary = predict_many(
            [runner], [primary], replace(cfg, dt_s=cfg.dt_s / 2), device=args.device
        )[0][0]
        np.savez_compressed(directory / "half_step_primary.npz", **fine_trace)
        fine_result = score(fine_trace, fine_summary, primary, runner) | work_metrics(fine_trace)
        write_json(
            directory / "half-step.json",
            {
                "sample_id": primary.id,
                "split": primary.split,
                "dt_s": cfg.dt_s / 2,
                "metrics": fine_result,
                "interpretation": "Numerical sensitivity diagnostic; no universal physiological acceptance threshold.",
            },
        )
        state["conditions"].append(
            {
                **condition,
                "summary": str(directory / "summary.json"),
                "model": str(directory / "runner.json"),
                "wall_s": native["wall_s"],
                "splits": {k: {j: v for j, v in s.items() if j != "trials"} for k, s in native["splits"].items()},
            }
        )
        write_json(root / "summary.json", state)
        print(name, state["conditions"][-1]["splits"], flush=True)
    state.update(
        status="completed", finished_at=datetime.now(timezone.utc).isoformat(), wall_s=perf_counter() - overall
    )
    write_json(root / "summary.json", state)
    print(root, flush=True)


if __name__ == "__main__":
    main()
