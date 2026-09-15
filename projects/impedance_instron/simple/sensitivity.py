# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run independent fixed-gain and frozen-material sensitivity experiments.

This workflow does not train a policy or alter the existing response suite.
Replay copies exact saved inputs, never re-identifies or regenerates materials.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.metadata
import json
import math
import platform
import shlex
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .policy import evaluate_policy
from .reference import Reference
from .response import _response_types, _source_fingerprints, _write_json, summarize_response

if TYPE_CHECKING:
    from .recovery import RecoveryConfig
    from .rig import RigConfig

SCHEMA = "impedance_material_sensitivity_1"
_GAIN_NAMES = (
    "leg_stiffness_n_m",
    "ankle_stiffness_n_m_rad",
    "leg_damping_n_s_m",
    "ankle_damping_n_m_s_rad",
)
_DIRECTIONS = {"forward": (1.0, 0.0), "backward": (-1.0, 0.0), "upward": (0.0, 1.0), "downward": (0.0, -1.0)}
_SEALED_KEYS = (
    "schema_version",
    "suite_config",
    "rig_config",
    "recovery_config",
    "materials",
    "controllers",
    "case_plan",
    "reference_identity",
    "reference_snapshot_sha256",
    "source_fingerprints",
    "snapshot_manifest",
)


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _decode_json(payload):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")

    return json.loads(payload, object_pairs_hook=unique, parse_constant=invalid_constant)


def _strict_json(path):
    return _decode_json(Path(path).read_bytes())


def _multipliers(values, name):
    result = []
    for raw in values:
        if isinstance(raw, (bool, np.bool_)):
            raise ValueError(f"{name} must contain numeric multipliers, not booleans")
        value = float(raw)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must contain positive finite multipliers")
        if value != 1.0 and value not in result:
            result.append(value)
    return tuple(result)


def _sensitivity_sources():
    result = _source_fingerprints()
    local = Path(__file__).resolve().parent
    root = local.parents[2]
    paths = [
        local / name
        for name in (
            "sensitivity.py",
            "material_variants.py",
            "recovery.py",
            "sensitivity_report.py",
            "sensitivity_figures.py",
            "figures.py",
            "report.py",
        )
    ]
    paths.extend(
        (
            local.parent / "orientation.py",
            root / "projects/digital_shoe/artifact.py",
            root / "projects/digital_shoe/provenance.py",
            root / "newton/_src/solvers/solver.py",
        )
    )
    paths.extend((root / "newton/_src/sim").glob("*.py"))
    for path in paths:
        result[str(path.relative_to(root))] = _sha256(path)
    return result


def _check_sources(expected, *, allow_physics_update=False):
    current = _sensitivity_sources()
    changed = {
        name: {"saved": expected.get(name), "current": current.get(name)}
        for name in expected.keys() | current.keys()
        if expected.get(name) != current.get(name)
    }
    if changed and not allow_physics_update:
        raise ValueError("Sensitivity replay sources changed; use --allow-physics-update for a new experiment")
    return changed


def _rig_config_from_dict(values):
    return importlib.import_module(".rig", __package__).RigConfig.from_dict(values)


def _resolve_rig_config(reference, config):
    module = importlib.import_module(".rig", __package__)
    if config is not None:
        return _rig_config_from_dict(config.to_dict())
    construction = reference.provenance.get("config", {})
    return module.RigConfig(
        **{key: construction[value] for key, value in module.Rig._construction_keys.items() if value in construction}
    )


def _controllers(reference, multipliers):
    _, response_type = _response_types()
    construction = reference.provenance.get("config", {})
    required = ("nominal_leg_stiffness_n_m", "nominal_ankle_stiffness_n_m_rad")
    if any(key not in construction for key in required):
        raise ValueError(
            "Reference must freeze nominal leg and ankle stiffness; action midpoints are not nominal gains"
        )
    nominal = response_type(
        leg_stiffness_n_m=float(construction[required[0]]),
        ankle_stiffness_n_m_rad=float(construction[required[1]]),
    ).resolved(reference)
    gains = {name: getattr(nominal, name) for name in _GAIN_NAMES}
    result = [{"controller_id": "nominal", "varied_gain": None, "multiplier": 1.0, "gains": gains}]
    for name in _GAIN_NAMES:
        for index, multiplier in enumerate(multipliers):
            changed = {**gains, name: gains[name] * multiplier}
            if changed == gains:
                continue  # A nominal zero damper cannot acquire damping by multiplication.
            response_type(**changed)
            result.append(
                {"controller_id": f"{name}_{index}", "varied_gain": name, "multiplier": multiplier, "gains": changed}
            )
    return result


def _plan(controllers, materials, modes, full_factorial):
    result, quiet = [], {}
    original = materials[0]["id"]
    for mode in modes:
        for controller in controllers:
            cid = controller["controller_id"]
            for material in materials:
                mid = material["id"]
                case_id = f"{len(result):04d}_{mode}_{cid}_{mid}_quiet"
                quiet[mode, cid, mid] = case_id
                result.append(
                    {
                        "case_id": case_id,
                        "controller_id": cid,
                        "material_id": mid,
                        "controller_mode": mode,
                        "comparison_type": "unperturbed_baseline" if mid == original else "material_sensitivity",
                        "direction": None,
                        "baseline_case_id": quiet[mode, cid, original],
                    }
                )
        for controller in controllers:
            cid = controller["controller_id"]
            for material in materials:
                mid = material["id"]
                if not full_factorial and cid != "nominal" and mid != original:
                    continue
                for direction in _DIRECTIONS:
                    case_id = f"{len(result):04d}_{mode}_{cid}_{mid}_{direction}"
                    result.append(
                        {
                            "case_id": case_id,
                            "controller_id": cid,
                            "material_id": mid,
                            "controller_mode": mode,
                            "comparison_type": "push_recovery",
                            "direction": direction,
                            "baseline_case_id": quiet[mode, cid, mid],
                        }
                    )
    return result


def _new_output(output):
    destination = Path(output).resolve()
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError(f"Sensitivity output must be new or empty: {destination}")
    return destination


def _snapshot_path(directory, name):
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Snapshot path must stay inside suite: {name}")
    path = (directory / relative).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError(f"Snapshot escapes suite: {name}")
    return path


def _verify_snapshot(directory, name, expected):
    path = _snapshot_path(directory, name)
    if _sha256(path) != expected:
        raise ValueError(f"Saved replay input changed: {name}; create a separate suite instead")
    return path


def _seal(record):
    return {"algorithm": "sha256", "content_sha256": _digest({key: record[key] for key in _SEALED_KEYS})}


def _verify_manifest(record):
    if record.get("schema_version") != SCHEMA:
        raise ValueError("Unsupported sensitivity suite schema")
    try:
        expected = _seal(record)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Invalid sensitivity configuration manifest") from error
    if record.get("config_seal") != expected:
        raise ValueError("Sensitivity suite configuration seal mismatch")


def _runtime():
    try:
        warp_version = importlib.metadata.version("warp-lang")
    except importlib.metadata.PackageNotFoundError:
        warp_version = "unavailable"
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "warp": warp_version,
        "platform": platform.platform(),
        "environment_archive": False,
    }


_REPLAY = '''# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Run saved sensitivity inputs from this Newton checkout into a new output."""
import argparse
import sys
from pathlib import Path
from projects.impedance_instron.simple.sensitivity import replay_sensitivity

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=Path)
parser.add_argument("--device", default=None)
parser.add_argument("--allow-physics-update", action="store_true")
args = parser.parse_args()
replay_sensitivity(Path(__file__).resolve().parent, args.output, device=args.device,
                   allow_physics_update=args.allow_physics_update, command=[sys.executable, *sys.argv])
'''


def run_sensitivity_suite(
    reference: Reference | str | Path,
    artifact_path: str | Path,
    output: str | Path,
    *,
    device: str | None = None,
    config: RigConfig | None = None,
    modes: tuple[str, ...] = ("intent",),
    gain_multipliers: tuple[float, ...] = (0.5, 2.0),
    modulus_multipliers: tuple[float, ...] = (0.75, 1.25),
    relaxation_multipliers: tuple[float, ...] = (0.5, 2.0),
    material_paths: tuple[str | Path, ...] = (),
    push_force_n: float = 150.0,
    push_duration_s: float = 0.04,
    push_phase: float = 0.25,
    recovery_config: RecoveryConfig | None = None,
    full_factorial: bool = False,
    command: list[str] | None = None,
) -> Path:
    """Run one-at-a-time fixed gains and material variants on one frozen clock.

    Args:
        reference: Frozen reference object or sealed JSON path.
        artifact_path: Original shoe artifact with the reference geometry.
        output: New or empty directory; never overwrite prior experiments.
        device: Warp device, such as ``cpu`` or ``cuda:0``.
        config: Fixed rig physics settings, independent of response gains.
        modes: Distinct intent and/or equilibrium controller laws.
        gain_multipliers: Positive factors for each gain independently. Duplicate
            factors and 1.0 are removed, retaining their first occurrence.
        modulus_multipliers: One-at-a-time material modulus factors.
        relaxation_multipliers: One-at-a-time material relaxation-time factors.
        material_paths: Additional same-geometry material-only artifacts.
        push_force_n: Common positive axial peak magnitude [N].
        push_duration_s: Full raised-cosine pulse support [s].
        push_phase: Fixed recorded-contact phase at pulse start.
        recovery_config: Explicit deviation tolerances and hold window.
        full_factorial: Push every controller/material pair instead of sparse
            coverage. Controller gain changes remain one-at-a-time.
        command: Exact invocation tokens for provenance.

    Returns:
        Offline report path, with strict JSON, numeric traces and guarded replay.
        The default is 97 cases per mode; full-factorial pushes give 225.
        A short recovery window is reported as insufficient, not rejected.
    """
    build_material_variants = importlib.import_module(".material_variants", __package__).build_material_variants
    recovery_type = importlib.import_module(".recovery", __package__).RecoveryConfig

    destination = _new_output(output)
    reference_path = Path(reference).resolve() if isinstance(reference, (str, Path)) else None
    reference_bytes = reference_path.read_bytes() if reference_path is not None else None
    reference = (
        Reference.from_dict(_decode_json(reference_bytes))
        if reference_bytes is not None
        else Reference.from_dict(reference.to_dict())
    )
    modes = tuple(modes)
    if not modes or len(set(modes)) != len(modes) or any(mode not in ("intent", "equilibrium") for mode in modes):
        raise ValueError("modes must contain distinct intent/equilibrium controller names")
    gain_multipliers = _multipliers(gain_multipliers, "gain_multipliers")
    modulus_multipliers = _multipliers(modulus_multipliers, "modulus_multipliers")
    relaxation_multipliers = _multipliers(relaxation_multipliers, "relaxation_multipliers")
    push_force_n, push_duration_s, push_phase = float(push_force_n), float(push_duration_s), float(push_phase)
    if not math.isfinite(push_force_n) or push_force_n <= 0:
        raise ValueError("push_force_n must be positive and finite")
    if not math.isfinite(push_duration_s) or push_duration_s <= 0:
        raise ValueError("push_duration_s must be positive and finite")
    if not math.isfinite(push_phase) or not 0 <= push_phase <= 1:
        raise ValueError("push_phase must be inside [0, 1]")
    push_start_s = reference.contact_start_s + push_phase * reference.contact_duration_s
    if push_start_s + push_duration_s > reference.contact_start_s + reference.contact_duration_s:
        raise ValueError("Complete push must lie inside recorded contact; reference schedules are not retimed")
    if not isinstance(full_factorial, bool):
        raise ValueError("full_factorial must be a bool")
    controllers = _controllers(reference, gain_multipliers)
    config = _resolve_rig_config(reference, config)
    recovery_config = recovery_config or recovery_type()
    recovery_values = asdict(recovery_config)
    recovery_type(**recovery_values)
    suite = {
        "modes": list(modes),
        "gain_multipliers": list(gain_multipliers),
        "modulus_multipliers": list(modulus_multipliers),
        "relaxation_multipliers": list(relaxation_multipliers),
        "push_force_n": push_force_n,
        "push_duration_s": push_duration_s,
        "push_phase": push_phase,
        "push_start_s": push_start_s,
        "full_factorial": full_factorial,
        "nominal_gains": controllers[0]["gains"],
        "multiplier_policy": "stable deduplication; omit 1.0 and unchanged zero gains",
    }
    destination.mkdir(parents=True, exist_ok=True)
    reference.save(destination / "reference.json")
    materials = build_material_variants(
        Path(artifact_path).resolve(),
        destination / "materials",
        modulus_multipliers=modulus_multipliers,
        relaxation_multipliers=relaxation_multipliers,
        material_paths=tuple(material_paths),
    )
    for material in materials:
        path = Path(material["path"])
        if not path.is_absolute():
            path = destination / "materials" / path
        path = path.resolve()
        material["path"] = str(path.relative_to(destination))
        _verify_snapshot(destination, material["path"], material["sha256"])
    if not materials or not materials[0]["baseline"] or len({item["id"] for item in materials}) != len(materials):
        raise ValueError("Material variants must start with one baseline and have unique IDs")
    plan = _plan(controllers, materials, modes, full_factorial)
    reference_hash = _sha256(destination / "reference.json")
    snapshots = {"reference.json": reference_hash, **{item["path"]: item["sha256"] for item in materials}}
    record = {
        "schema_version": SCHEMA,
        "suite_config": suite,
        "rig_config": config.to_dict(),
        "recovery_config": recovery_values,
        "materials": materials,
        "controllers": controllers,
        "case_plan": plan,
        "reference_identity": reference.identity,
        "reference_snapshot_sha256": reference_hash,
        "reference_source_path": str(reference_path) if reference_path else None,
        "reference_source_sha256": hashlib.sha256(reference_bytes).hexdigest() if reference_bytes is not None else None,
        "source_fingerprints": _sensitivity_sources(),
        "snapshot_manifest": snapshots,
        "requested_device": device,
        "num_worlds_per_case": 1,
        "contract": {
            "controller": "fixed frozen-nominal gains, one gain varied at a time; no optimizer or policy",
            "reference": "all targets, initial states, ID loads and clock frozen; no source tuning",
            "quiet_pair": "original material, same controller and mode",
            "push_pair": "same material, controller and mode, without push; direct validity, not inherited pairing",
            "material": "permanent change from t=0, reset native material history; no recovery claim",
            "push": "four signed equal-peak axial raised-cosine pulses; native interval averages",
            "terminal": "true final position and velocity after integration; no extra constitutive update",
            "reproducibility": "runtime versions recorded, not an environment archive; GPU reductions may differ",
        },
    }
    record["config_seal"] = _seal(record)
    return _run_from_manifest(destination, record, device=device, command=command)


def _load_trace(destination, case):
    with np.load(destination / case["trace_file"], allow_pickle=False) as data:
        return {name: data[name] for name in data.files if not name.startswith("terminal_")}


def _terminal_valid(terminal, arrays):
    required = (
        "time_s",
        "pelvis_z_m",
        "pitch_rad",
        "leg_length_m",
        "foot_x_m",
        "foot_z_m",
        "pelvis_x_m",
        "foot_vx_m_s",
        "foot_vz_m_s",
        "pelvis_vx_m_s",
        "pelvis_vz_m_s",
        "pitch_rate_rad_s",
        "leg_rate_m_s",
    )
    return bool(
        all(name in terminal and np.isfinite(terminal[name]).all() for name in required)
        and all(
            name in arrays and np.asarray(arrays[name]).size and np.isfinite(arrays[name]).all()
            for name in ("terminal_body_q", "terminal_body_qd")
        )
    )


def _run_from_manifest(destination, manifest, *, device, command, replay=None):
    recovery = importlib.import_module(".recovery", __package__)
    write_sensitivity_report = importlib.import_module(".sensitivity_report", __package__).write_sensitivity_report

    _verify_manifest(manifest)
    for name, expected in manifest["snapshot_manifest"].items():
        _verify_snapshot(destination, name, expected)
    rig_type, response_type = _response_types()
    reference = Reference.load(destination / "reference.json")
    config = _rig_config_from_dict(manifest["rig_config"])
    recovery_config = recovery.RecoveryConfig(**manifest["recovery_config"])
    (destination / "cases").mkdir()
    (destination / "replay.py").write_text(_REPLAY, encoding="utf-8")
    _write_json(destination / "manifest.json", manifest)
    record = copy.deepcopy(manifest)
    record.update(
        status="running",
        cases=[],
        runtime=_runtime(),
        requested_device=device,
        commands={
            "invocation_argv": command,
            "replay": shlex.join(
                ["uv", "run", "--no-sync", "python", str(destination / "replay.py"), "NEW_EMPTY_OUTPUT"]
            ),
            "replay_device_override": "Append --device cpu or --device cuda:0; run from this Newton checkout",
        },
    )
    if replay is not None:
        record["replay"] = replay
    _write_json(destination / "summary.json", record)
    controllers = {item["controller_id"]: item for item in record["controllers"]}
    materials = {item["id"]: item for item in record["materials"]}
    completed = {}
    expected_samples = math.ceil(reference.duration_s * config.frame_rate_hz) * config.substeps
    dt = reference.duration_s / expected_samples
    suite = record["suite_config"]
    for planned in record["case_plan"]:
        case = copy.deepcopy(planned)
        fx, fz = _DIRECTIONS.get(case["direction"], (0.0, 0.0))
        settings = response_type(
            controller_mode=case["controller_mode"],
            **controllers[case["controller_id"]]["gains"],
            ground_height_m=0.0,
            push_start_s=suite["push_start_s"],
            push_duration_s=suite["push_duration_s"],
            push_force_x_n=fx * suite["push_force_n"],
            push_force_z_n=fz * suite["push_force_n"],
        )
        case.update(
            response_config=asdict(settings),
            status="execution_error",
            pair_valid=False,
            dt_s=dt,
            physics_metadata={},
            terminal={},
            metrics={},
            recovery={},
            requested_push_impulse_n_s=[
                0.5 * settings.push_force_x_n * suite["push_duration_s"],
                0.5 * settings.push_force_z_n * suite["push_duration_s"],
            ],
        )
        rig, trace, terminal, arrays, evaluation = None, {}, {}, {}, {}
        try:
            _verify_snapshot(destination, "reference.json", record["reference_snapshot_sha256"])
            material = materials[case["material_id"]]
            material_path = _verify_snapshot(destination, material["path"], material["sha256"])
            rig = rig_type(
                Reference.load(destination / "reference.json"),
                material_path,
                response_config=settings,
                config=config,
                num_worlds=1,
                device=device,
            )
            case.update(
                physics_metadata=rig.metadata,
                input_fingerprints=rig.input_fingerprints,
                device=str(rig.device),
                limits={
                    "leg_force_n": config.force_limit_bw * reference.mass_kg * reference.gravity_m_s2,
                    "ankle_torque_n_m": config.ankle_torque_limit_n_m,
                },
            )
            evaluation = evaluate_policy(rig)
        except Exception as error:
            case["error"] = {"type": type(error).__name__, "message": str(error)}
        if rig is not None:
            try:
                trace = rig.trace(0)
            except Exception as error:
                case["trace_error"] = {"type": type(error).__name__, "message": str(error)}
            try:
                terminal, arrays = recovery.terminal_state(rig)
            except Exception as error:
                case["terminal_error"] = {"type": type(error).__name__, "message": str(error)}
            case["graph_status"] = rig.graph_status
        case["terminal"] = terminal
        case["terminal_state_valid"] = _terminal_valid(terminal, arrays)
        case["episode_complete"] = bool(
            np.asarray(trace.get("time_s", [])).size == expected_samples
            and "time_s" in terminal
            and math.isclose(terminal["time_s"], reference.duration_s, rel_tol=1e-12, abs_tol=1e-12)
        )
        case["trace_file"] = f"cases/{case['case_id']}.npz"
        numeric = {**trace, **arrays, "terminal_time_s": np.asarray(terminal.get("time_s", np.nan))}
        numeric.update({f"terminal_{key}": np.asarray(value) for key, value in terminal.items()})
        if any(np.asarray(value).dtype.kind not in "biufc" for value in numeric.values()):
            raise ValueError("Sensitivity NPZ must contain numeric arrays only")
        np.savez_compressed(destination / case["trace_file"], **numeric)
        case["trace_sha256"] = _sha256(destination / case["trace_file"])
        baseline = case if case["baseline_case_id"] == case["case_id"] else completed[case["baseline_case_id"]]
        baseline_trace = trace if baseline is case else _load_trace(destination, baseline)
        case["baseline_trace_file"] = baseline["trace_file"]
        case["metrics"] = summarize_response(
            trace,
            baseline_trace,
            dt,
            terminal=terminal,
            baseline_terminal=baseline["terminal"],
            evaluation=evaluation,
            contact_start_s=reference.contact_start_s,
            contact_threshold_n=config.contact_threshold_n,
        )
        if not any(key in case for key in ("error", "trace_error", "terminal_error")):
            case["status"] = case["metrics"]["physical_validity"]
            if not case["episode_complete"] or not case["terminal_state_valid"]:
                case["status"] = "invalid"
        case["baseline_status"] = baseline["status"]
        case["pair_valid"] = bool(
            case["status"] == "valid"
            and baseline["status"] == "valid"
            and case["terminal_state_valid"]
            and baseline["terminal_state_valid"]
            and case["metrics"]["paired_clock_match"]
            and case["metrics"]["terminal_clock_match"]
        )
        push_end = suite["push_start_s"] + suite["push_duration_s"] if case["direction"] is not None else None
        case["recovery"] = recovery.summarize_recovery(
            trace,
            baseline_trace,
            dt,
            terminal=terminal,
            baseline_terminal=baseline["terminal"],
            push_end_s=push_end,
            config=recovery_config,
            pair_valid=case["pair_valid"],
        )
        if case["comparison_type"] == "unperturbed_baseline" and case["recovery"]["status"] != "invalid_pair":
            case["recovery"]["status"] = "unperturbed_baseline"
            case["recovery"]["interpretation"] = (
                "Unperturbed self-comparison; no material change, push, or recovery test."
            )
        record["cases"].append(case)
        completed[case["case_id"]] = case
        _write_json(destination / "summary.json", record)
        print(
            f"[{len(record['cases'])}/{len(record['case_plan'])}] {case['case_id']}: {case['status']}; pair_valid={case['pair_valid']}",
            flush=True,
        )
        del rig
    record.update(
        status="complete",
        valid_case_count=sum(case["status"] == "valid" for case in record["cases"]),
        invalid_or_failed_case_count=sum(case["status"] != "valid" for case in record["cases"]),
        valid_pair_count=sum(case["pair_valid"] for case in record["cases"]),
    )
    _write_json(destination / "summary.json", record)
    return write_sensitivity_report(destination, record)


def replay_sensitivity(
    source: str | Path,
    output: str | Path,
    *,
    device: str | None = None,
    allow_physics_update: bool = False,
    command: list[str] | None = None,
) -> Path:
    """Validate a saved suite and evaluate exact snapshots into a new directory.

    Args:
        source: Saved suite directory containing manifest.json and snapshots.
        output: New or empty output directory.
        device: Optional execution-device override, not a mechanics change.
        allow_physics_update: Permit source changes only. All input and setting
            seals remain required; changed sources create a new experiment.
        command: Exact replay invocation tokens for provenance.

    Returns:
        Fresh report path. No scores or case-validity results are inherited.
    """
    destination = _new_output(output)
    source = Path(source).resolve()
    manifest, old_record = _strict_json(source / "manifest.json"), _strict_json(source / "summary.json")
    _verify_manifest(manifest)
    _verify_manifest(old_record)
    if manifest["config_seal"] != old_record["config_seal"]:
        raise ValueError("Sensitivity summary settings differ from sealed manifest")
    for name, expected in manifest["snapshot_manifest"].items():
        _verify_snapshot(source, name, expected)
    reference = Reference.load(source / "reference.json")
    if reference.identity != manifest["reference_identity"]:
        raise ValueError("Saved reference identity mismatch")
    changed = _check_sources(manifest["source_fingerprints"], allow_physics_update=allow_physics_update)
    if changed:
        warnings.warn("Sensitivity replay sources changed: NEW EXPERIMENT; old results do not apply", stacklevel=2)
    destination.mkdir(parents=True, exist_ok=True)
    for name, expected in manifest["snapshot_manifest"].items():
        path = _snapshot_path(destination, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_verify_snapshot(source, name, expected).read_bytes())
    fresh = copy.deepcopy(manifest)
    fresh["source_fingerprints"] = _sensitivity_sources()
    fresh["config_seal"] = _seal(fresh)
    replay = {
        "source_report": str(source),
        "source_manifest_sha256": _sha256(source / "manifest.json"),
        "source_config_seal": manifest["config_seal"],
        "changed_sources": changed,
        "allow_physics_update": allow_physics_update,
        "old_results_inherited": False,
        "source_matches": not bool(changed),
        "requested_device_override": device,
    }
    return _run_from_manifest(
        destination,
        fresh,
        device=device if device is not None else old_record["requested_device"],
        command=command,
        replay=replay,
    )
