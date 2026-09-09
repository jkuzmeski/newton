# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Export and load a sealed, measured running stance without a human runtime.

The offline exporter calls the source worktree's public C3D adapters in that
worktree's own uv environment. The portable loader needs no gait code or C3D.
Foot motion is a heel/toe marker proxy, not a shoe transform or native-body pose.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

SCHEMA = "impedance_stance_1"
ARRAYS = (
    "time_s",
    "source_time_s",
    "foot_x_m",
    "foot_z_m",
    "pitch_rad",
    "reference_fz_n",
    "reference_fx_n",
    "other_fz_n",
    "other_fx_n",
    "reference_cop_x_m",
    "com_x_m",
    "com_z_m",
    "reference_com_vz_m_s",
    "reference_com_vx_m_s",
    "total_measured_fz_n",
    "total_measured_fx_n",
    "unassigned_fz_n",
    "unassigned_fx_n",
)
COORDINATES = {
    "forward_axis": "X",
    "left_axis": "Y",
    "up_axis": "Z",
    "pitch": "right-handed +Y, positive toe-down",
    "length_unit": "m",
    "force_unit": "N",
    "time_unit": "s",
    "angle_unit": "rad",
    "origin": "selected heel marker at first exported sample",
    "frame": "steady-speed treadmill-to-overground",
}


def _canonical(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unique_object(pairs: list) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _integrate(times: np.ndarray, acceleration: np.ndarray, x0: float, v0: float):
    dt = np.diff(times)
    velocity = v0 + np.r_[0.0, np.cumsum(0.5 * (acceleration[1:] + acceleration[:-1]) * dt)]
    position = x0 + np.r_[0.0, np.cumsum(0.5 * (velocity[1:] + velocity[:-1]) * dt)]
    return position, velocity


def load_profile(path: str | Path) -> dict:
    """Verify a portable running profile and return its JSON object.

    Args:
        path: Sealed ``impedance_stance_1`` JSON file.

    Raises:
        ValueError: If schema, seal, arrays, source attribution, or physics
            bookkeeping is inconsistent. This is not independent sensor QC.
    """
    result = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    if not isinstance(result, dict):
        raise ValueError("profile must be an object")
    required = set(ARRAYS) | {
        "schema_version",
        "coordinate_system",
        "mass_kg",
        "side",
        "provenance",
        "seal",
    }
    if set(result) != required or result["schema_version"] != SCHEMA:
        raise ValueError("unsupported profile fields or schema")
    if result["coordinate_system"] != COORDINATES or result["side"] not in ("left", "right"):
        raise ValueError("invalid coordinate system or side")
    content = {key: value for key, value in result.items() if key != "seal"}
    if result["seal"] != {"algorithm": "sha256", "content_sha256": hashlib.sha256(_canonical(content)).hexdigest()}:
        raise ValueError("profile seal mismatch")
    mass = _number(result["mass_kg"], "mass_kg")
    if mass <= 0:
        raise ValueError("mass_kg must be positive")
    n = len(result["time_s"])
    if n < 3:
        raise ValueError("at least three samples are required")
    values = {}
    for name in ARRAYS:
        column = result[name]
        if not isinstance(column, list) or len(column) != n:
            raise ValueError(f"{name} must have {n} samples")
        if name == "reference_cop_x_m":
            values[name] = np.asarray([np.nan if x is None else _number(x, name) for x in column])
        else:
            values[name] = np.asarray([_number(x, name) for x in column])
    t, source_t = values["time_s"], values["source_time_s"]
    if abs(t[0]) > 1e-12 or np.any(np.diff(t) <= 0) or source_t[0] < 0:
        raise ValueError("time must start at zero and increase strictly")
    if not np.allclose(source_t - source_t[0], t, rtol=0, atol=1e-10):
        raise ValueError("source and relative timelines disagree")
    if any(abs(values[name][0]) > 1e-12 for name in ("foot_x_m", "foot_z_m")):
        raise ValueError("heel displacement must start at zero")
    meta = result["provenance"]
    if not isinstance(meta, dict):
        raise ValueError("provenance must be an object")
    for name in ("sources", "running", "registration", "kinematics", "kinetics", "com_surrogate", "rights"):
        if not isinstance(meta.get(name), dict) or not meta[name]:
            raise ValueError(f"missing provenance: {name}")
    for source in meta["sources"].values():
        if not isinstance(source, dict) or not isinstance(source.get("path"), str):
            raise ValueError("each source needs a path and SHA-256")
        digest = source.get("sha256", "")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("invalid source SHA-256")
    running = meta["running"]
    if running.get("classification") != "running" or running.get("selected_side") != result["side"]:
        raise ValueError("profile is not classified running for the selected side")
    if not 0.1 <= _number(running.get("stance_duration_s"), "stance duration") <= 0.4:
        raise ValueError("running stance duration outside supported range")
    if not 130 <= _number(running.get("cadence_steps_min"), "cadence") <= 240:
        raise ValueError("running cadence outside supported range")
    if min(_number(running.get(key), key) for key in ("flight_before_s", "flight_after_s")) < 0.02:
        raise ValueError("running stance requires measured flight before and after")
    threshold = _number(meta["kinetics"].get("load_threshold_n"), "load threshold")
    if threshold <= 0:
        raise ValueError("load threshold must be positive")
    loaded = values["reference_fz_n"] > threshold
    if not np.any(loaded) or not np.array_equal(np.isfinite(values["reference_cop_x_m"]), loaded):
        raise ValueError("COP must be finite exactly when selected support exceeds its threshold")
    for axis in ("x", "z"):
        expected = values[f"reference_f{axis}_n"] + values[f"other_f{axis}_n"]
        if not np.allclose(expected, values[f"total_measured_f{axis}_n"], rtol=0, atol=1e-9):
            raise ValueError("reference and airborne-other forces must sum to measured force")
    channels = meta["kinetics"].get("platform_channels", {})
    measured = np.asarray(channels.get("force_n"), dtype=float)
    moments = np.asarray(channels.get("moment_about_lab_origin_nm"), dtype=float)
    if (
        measured.shape != (n, 2, 3)
        or moments.shape != measured.shape
        or not np.all(np.isfinite(measured))
        or not np.all(np.isfinite(moments))
    ):
        raise ValueError("two finite source platform force/moment channels are required")
    for axis, component in (("x", 0), ("z", 2)):
        if not np.allclose(
            measured[:, :, component].sum(axis=1), values[f"total_measured_f{axis}_n"], rtol=0, atol=1e-9
        ):
            raise ValueError("source platform channels disagree with measured force sum")
    displacement = np.asarray(meta["registration"].get("virtual_origin_x_m"), dtype=float)
    origin = np.asarray(meta["registration"].get("heel_origin_newton_lab_m"), dtype=float)
    if (
        displacement.shape != (n,)
        or origin.shape != (3,)
        or not np.all(np.isfinite(displacement))
        or not np.all(np.isfinite(origin))
    ):
        raise ValueError("invalid shared heel/COP origin")
    expected_cop = (
        -moments[loaded, :, 1].sum(axis=1) / values["reference_fz_n"][loaded] + displacement[loaded] - origin[0]
    )
    if not np.allclose(expected_cop, values["reference_cop_x_m"][loaded], rtol=0, atol=1e-9):
        raise ValueError("COP disagrees with summed measured wrench and shared heel origin")
    surrogate = meta["com_surrogate"]
    if surrogate.get("kind") != "force_integrated_surrogate_not_measured_com":
        raise ValueError("COM must be labeled as a force-integrated surrogate")
    for axis in ("x", "z"):
        acceleration = values[f"total_measured_f{axis}_n"] / mass
        if axis == "z":
            acceleration = acceleration - _number(surrogate.get("gravity_m_s2"), "gravity")
        x, v = _integrate(
            t,
            acceleration,
            _number(surrogate.get(f"initial_{axis}_m"), "initial position"),
            _number(surrogate.get(f"initial_v{axis}_m_s"), "initial velocity"),
        )
        if not np.allclose(x, values[f"com_{axis}_m"], rtol=0, atol=1e-9) or not np.allclose(
            v, values[f"reference_com_v{axis}_m_s"], rtol=0, atol=1e-9
        ):
            raise ValueError("COM surrogate is not the declared measured-force integration")
    return result


def _events(mask: np.ndarray):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True))


def _extract(config: dict) -> None:
    """Run only inside the source worktree's native Python environment."""
    from projects.gait_c3d.c3d_adapter import read_c3d_markers  # noqa: PLC0415
    from projects.gait_c3d.contact_dataset import read_c3d_contact_wrenches  # noqa: PLC0415
    from projects.gait_c3d.treadmill import belt_motion, load_treadmill_log  # noqa: PLC0415

    root = Path.cwd().resolve()
    subject = root / config["subject"]
    c3d = subject / config["c3d"]
    log_path = subject / config["treadmill_log"]
    source_paths = {
        "c3d": c3d,
        "treadmill_log": log_path,
        "subject_manifest": subject / "subject.json",
        "c3d_adapter": root / "projects/gait_c3d/c3d_adapter.py",
        "contact_dataset": root / "projects/gait_c3d/contact_dataset.py",
        "treadmill_adapter": root / "projects/gait_c3d/treadmill.py",
        "exporter": Path(__file__).resolve(),
    }
    sources = {name: {"path": str(path), "sha256": _hash(path)} for name, path in source_paths.items()}
    markers = read_c3d_markers(c3d, up_axis="+Z", forward_axis="-Y")
    wrench = read_c3d_contact_wrenches(
        c3d,
        platform_sides=("unassigned", "unassigned"),
        input_force_convention="ground_on_human",
        up_axis="+Z",
        forward_axis="-Y",
        lowpass_hz=20.0,
        filter_taps=401,
        load_threshold_n=50.0,
    )
    if markers.source_sha256 != wrench.metadata["source"]["sha256"]:
        raise ValueError("force and markers must share exact C3D bytes")
    log = load_treadmill_log(log_path)
    start, end = config["window_start"], config["window_end"]
    if not 3 <= end - start <= 20 or start < 0 or end > markers.times[-1]:
        raise ValueError("choose an explicit 3-20 second classification window inside the recording")
    keep = (wrench.times >= start - 1.0) & (wrench.times <= end + 1.0)
    t = wrench.times[keep]
    force, cop, loaded = wrench.force[keep], wrench.cop[keep], wrench.loaded[keep]
    if not np.all(wrench.valid[keep]):
        raise ValueError("classification window intersects invalid/filter-edge force data")
    belt = belt_motion(log, t, offset=config["belt_offset"], side="auto")
    if not np.all(belt.covered) or np.ptp(belt.speed) > 1e-5 or np.mean(belt.speed) < 2:
        raise ValueError("running export requires fully covered, steady tied-belt speed >=2 m/s")
    log_selection = (log.t >= t[0] + config["belt_offset"]) & (log.t <= t[-1] + config["belt_offset"])
    if np.max(np.abs(np.r_[log.pitch[log_selection], log.roll[log_selection]])) > 0.005:
        raise ValueError("platform pitch/roll exceed the source flat-platform tolerance")
    if (
        np.max(np.abs(wrench.plane_normals - np.array([0.0, 0.0, 1.0]))) > 1e-9
        or np.max(np.abs(wrench.plane_points[:, 2])) > 1e-9
    ):
        raise ValueError("only horizontal platforms on the lab Z=0 plane are supported")
    marker_data = {}
    for name in ("LHEE", "LTOE", "RHEE", "RTOE"):
        index = markers.marker_names.index(name)
        bracket = (markers.times >= t[0] - 0.02) & (markers.times <= t[-1] + 0.02)
        if not np.all(markers.valid[bracket, index]):
            raise ValueError(f"missing marker data: {name}")
        marker_data[name] = np.column_stack(
            [np.interp(t, markers.times, markers.positions[:, index, axis]) for axis in range(3)]
        )
    support = np.any(loaded, axis=1)
    events = []
    for i, j in _events(support):
        if i == 0 or j == len(t) or t[j] - t[i] < 0.1:
            continue
        plates = np.flatnonzero(np.any(loaded[i:j], axis=0))
        if len(plates) != 1:
            raise ValueError("each running support event must occupy exactly one force platform")
        plate = int(plates[0])
        midpoint = (i + j) // 2
        strong = np.flatnonzero(force[i:j, plate, 2] > 0.5 * np.max(force[i:j, plate, 2])) + i
        distances = {}
        for side, prefix in (("left", "L"), ("right", "R")):
            heel, toe = marker_data[prefix + "HEE"][strong], marker_data[prefix + "TOE"][strong]
            segment = toe[:, :2] - heel[:, :2]
            length_squared = np.sum(segment * segment, axis=1)
            if np.any(np.linalg.norm(toe - heel, axis=1) < 0.1):
                raise ValueError("degenerate foot marker segment during side attribution")
            delta = cop[strong, plate, :2] - heel[:, :2]
            # A near-vertical swing foot projects to a point, not an invalid foot.
            fraction = np.clip(np.sum(delta * segment, axis=1) / np.maximum(length_squared, 1e-12), 0, 1)
            distances[side] = float(np.median(np.linalg.norm(delta - fraction[:, None] * segment, axis=1)))
        side = min(distances, key=distances.get)
        other = "right" if side == "left" else "left"
        prefix = "L" if side == "left" else "R"
        opposite = "R" if side == "left" else "L"
        support_toe = marker_data[prefix + "TOE"][strong]
        other_heel = marker_data[opposite + "HEE"][strong]
        heel_clearance = float(np.median(other_heel[:, 2] - support_toe[:, 2]))
        if distances[side] > 0.08 or distances[other] - distances[side] < 0.08 or heel_clearance < 0.1:
            raise ValueError("ambiguous foot assignment: inspect markers and COP before exporting")
        events.append(
            {
                "start_s": float(t[i]),
                "end_s": float(t[j]),
                "side": side,
                "platform_index": plate,
                "midpoint_s": float(t[midpoint]),
                "cop_to_foot_segment_xy_m": distances,
                "opposite_heel_above_support_toe_m": heel_clearance,
                "side_assignment_samples": len(strong),
                "side_assignment_policy": "median COP-to-heel/TOE segment XY distance and opposite heel height at Fz > half event peak",
                "peak_fz_n": float(np.max(force[i:j, plate, 2])),
                "indices": [int(i), int(j)],
            }
        )
    interior = [event for event in events if start <= event["start_s"] and event["end_s"] <= end]
    if len(interior) < 6 or any(a["side"] == b["side"] for a, b in itertools.pairwise(interior)):
        raise ValueError("running window needs at least six alternating marker-assigned support events")
    durations = np.array([event["end_s"] - event["start_s"] for event in interior])
    flights = np.array([b["start_s"] - a["end_s"] for a, b in itertools.pairwise(interior)])
    cadence = 60 / float(np.median(np.diff([event["start_s"] for event in interior])))
    if np.any((durations < 0.1) | (durations > 0.4)) or np.any(flights < 0.02) or not 130 <= cadence <= 240:
        raise ValueError("force timing does not qualify as the supported running pattern")
    candidates = [event for event in interior if event["side"] == config["side"]]
    if config["stance_index"] < 0 or config["stance_index"] >= len(candidates):
        raise ValueError("stance index outside the qualified same-side events")
    chosen = candidates[config["stance_index"]]
    event_index = events.index(chosen)
    if event_index == 0 or event_index == len(events) - 1:
        raise ValueError("selected event lacks preceding/following support evidence")
    flight_before = chosen["start_s"] - events[event_index - 1]["end_s"]
    flight_after = events[event_index + 1]["start_s"] - chosen["end_s"]
    padding = config["padding"]
    if not 0 <= padding < min(flight_before, flight_after):
        raise ValueError("padding must remain inside the adjacent flights")
    i, j = chosen["indices"]
    samples = int(round(padding * wrench.sample_rate_hz))
    selection = slice(i - samples, j + samples + 1)
    ts = t[selection]
    relative = ts - ts[0]
    plate = chosen["platform_index"]
    total = np.sum(force[selection], axis=1)
    reference = total.copy()
    unassigned = force[selection, 1 - plate].copy()
    platform_force = force[selection]
    platform_moment = (wrench.moment[keep] + np.cross(wrench.origins[keep], force))[selection]
    summed_moment = np.sum(platform_moment, axis=1)
    reference_loaded = reference[:, 2] > wrench.load_threshold_n
    if np.any(loaded[selection, 1 - plate]):
        raise ValueError("other platform loaded inside selected single-support export")
    prefix = "L" if config["side"] == "left" else "R"
    heel, toe = marker_data[prefix + "HEE"][selection], marker_data[prefix + "TOE"][selection]
    displacement = belt.distance[selection] - belt.distance[selection][0]
    foot_x = heel[:, 0] - heel[0, 0] + displacement
    foot_z = heel[:, 2] - heel[0, 2]
    vector = toe - heel
    if np.any(np.linalg.norm(vector[:, (0, 2)], axis=1) < 0.1):
        raise ValueError("degenerate heel-to-toe sagittal orientation")
    pitch = np.unwrap(-np.arctan2(vector[:, 2], vector[:, 0]))
    cop_x = np.zeros(len(ts))
    cop_x[reference_loaded] = -summed_moment[reference_loaded, 1] / reference[reference_loaded, 2]
    cop_x += displacement - heel[0, 0]
    mass = json.loads((subject / "subject.json").read_text())["subject"]["mass_kg"]
    gravity = 9.80665
    initial_vx = float(np.mean(belt.speed)) if config["initial_vx"] is None else config["initial_vx"]
    x, vx = _integrate(relative, total[:, 0] / mass, config["initial_x"], initial_vx)
    z, vz = _integrate(relative, total[:, 2] / mass - gravity, config["initial_z"], config["initial_vz"])
    running = {
        "classification": "running",
        "selected_side": config["side"],
        "selection_policy": "explicit window; chronological same-side stance index, no fit-score selection",
        "window_s": [start, end],
        "selected_stance_source_s": [chosen["start_s"], chosen["end_s"]],
        "stance_duration_s": chosen["end_s"] - chosen["start_s"],
        "cadence_steps_min": cadence,
        "flight_before_s": flight_before,
        "flight_after_s": flight_after,
        "window_stance_duration_range_s": [float(durations.min()), float(durations.max())],
        "window_flight_range_s": [float(flights.min()), float(flights.max())],
        "belt_speed_m_s": float(np.mean(belt.speed)),
        "events": [{key: value for key, value in event.items() if key != "indices"} for event in interior],
        "uncertainty": "50 N threshold and 20 Hz filtering affect event times; plate-to-foot identity is inferred from simultaneous markers/COP, not fixed plate labels",
    }
    provenance = {
        "sources": sources,
        "source_worktree": str(root),
        "source_branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
        "running": running,
        "recording": {
            "first_point_frame": markers.first_frame,
            "point_count": len(markers.times),
            "time_range_s": [float(markers.times[0]), float(markers.times[-1])],
        },
        "registration": {
            "lab_to_newton": markers.lab_to_newton.tolist(),
            "rigid_registration_after_rotation": np.eye(4).tolist(),
            "policy": "direct acquisition markers; no IK registration, no ground-height correction",
            "heel_origin_newton_lab_m": heel[0].tolist(),
            "virtual_origin": "add integrated tied-belt speed to BOTH marker X and COP X; re-zero at first exported sample",
            "virtual_origin_x_m": displacement.tolist(),
            "belt_offset_s": config["belt_offset"],
            "belt_clock_uncertainty_s": 0.05,
            "belt_input": "D-Flow command/reference, not measured belt velocity",
            "tied_belt_residual_m": log.tied_belt_residual,
        },
        "kinematics": {
            "kind": "measured_marker_heel_to_toe_proxy",
            "heel_marker": prefix + "HEE",
            "toe_marker": prefix + "TOE",
            "source_rate_hz": markers.rate,
            "interpolation": "piecewise linear positions to native analog times; no new independent kinematic information",
            "pitch_formula": "-atan2(toe_z-heel_z, toe_x-heel_x), unwrapped",
            "limitations": "marker vector is not a shoe sole/body frame; skin motion and unknown prior gap filling; no independently measured COM",
        },
        "kinetics": {
            "kind": "measured_calibrated_c3d_platform_wrench",
            "load_threshold_n": wrench.load_threshold_n,
            "source_rate_hz": wrench.sample_rate_hz,
            "resampling": "none; retain native analog samples",
            "selected_platform_index": plate,
            "assignment": "event-specific marker/COP side attribution; BOTH feet may use the SAME platform on alternating steps",
            "other_foot": "zero by airborne single-support inference, NOT a separately measured opposite-foot channel",
            "unassigned_force": "diagnostic unloaded platform signal; already INCLUDED in reference and total, do not add twice; not opposite-foot force",
            "platform_channels": {
                "names": list(wrench.plate_names),
                "force_n": platform_force.tolist(),
                "moment_about_lab_origin_nm": platform_moment.tolist(),
                "frame": "Newton lab axes, fixed laboratory origin; before virtual translation/heel re-zero",
                "force_processing": "same 20 Hz filtered measured forces as reference, no clipping",
            },
            "total_measured_force": "sum of both measured platform signals, including unloaded noise; used for COM integration",
            "selected_signal": "sum of BOTH measured platform forces assigned to marker-confirmed single-foot support, including unloaded noise; no clipping/renormalization",
            "cop": "sum filtered moments shifted to fixed lab origin BEFORE COP derivation; null when summed Fz<=50 N; same virtual translation/heel origin as foot",
            "decoder": wrench.metadata["source"],
            "processing": wrench.metadata["processing"],
        },
        "com_surrogate": {
            "kind": "force_integrated_surrogate_not_measured_com",
            "gravity_m_s2": gravity,
            "initial_x_m": config["initial_x"],
            "initial_z_m": config["initial_z"],
            "initial_vx_m_s": initial_vx,
            "initial_vz_m_s": config["initial_vz"],
            "integration": "cumulative trapezoid twice on total measured GRF/m - gravity; no periodicity or drift correction",
            "mass_source": "subject.json subject.mass_kg; offline anthropometric model value, not a new measurement",
            "limitations": "arbitrary initial conditions; not measured/model-FK COM; forces cannot identify absolute COM position or initial velocity",
        },
        "rights": {
            "source": "user-supplied local S001 acquisition",
            "redistribution": "not established; do not commit raw or derived participant data",
            "usage": "local engineering demonstration only; not a new independent validation or reserved holdout",
            "parameter_identification": "no physiological impedance identified by this export",
        },
        "reproduction_options": {key: value for key, value in config.items() if key != "output"},
    }
    result = {
        "schema_version": SCHEMA,
        "coordinate_system": COORDINATES,
        "mass_kg": float(mass),
        "side": config["side"],
        "time_s": relative.tolist(),
        "source_time_s": ts.tolist(),
        "foot_x_m": foot_x.tolist(),
        "foot_z_m": foot_z.tolist(),
        "pitch_rad": pitch.tolist(),
        "reference_fx_n": reference[:, 0].tolist(),
        "reference_fz_n": reference[:, 2].tolist(),
        "other_fx_n": np.zeros(len(ts)).tolist(),
        "other_fz_n": np.zeros(len(ts)).tolist(),
        "unassigned_fx_n": unassigned[:, 0].tolist(),
        "unassigned_fz_n": unassigned[:, 2].tolist(),
        "total_measured_fx_n": total[:, 0].tolist(),
        "total_measured_fz_n": total[:, 2].tolist(),
        "reference_cop_x_m": [
            float(value) if is_loaded else None for value, is_loaded in zip(cop_x, reference_loaded, strict=True)
        ],
        "com_x_m": x.tolist(),
        "com_z_m": z.tolist(),
        "reference_com_vx_m_s": vx.tolist(),
        "reference_com_vz_m_s": vz.tolist(),
        "provenance": provenance,
    }
    for name, path in source_paths.items():
        if _hash(path) != sources[name]["sha256"]:
            raise ValueError(f"source changed during export: {name}")
    result["seal"] = {"algorithm": "sha256", "content_sha256": hashlib.sha256(_canonical(result)).hexdigest()}
    output = Path(config["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    load_profile(output)
    print(json.dumps({"output": str(output), "running": running, "samples": len(ts)}, indent=2))


def main() -> None:
    """Export an explicitly selected and marker-qualified measured running stance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-worktree", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subject", default="projects/gait_c3d/subjects/S001")
    parser.add_argument("--c3d", default="Trial 101.v3d.c3d")
    parser.add_argument("--treadmill-log", default="tm0001.txt")
    parser.add_argument(
        "--window-start", type=float, required=True, help="Explicit classification-window start on source C3D clock [s]"
    )
    parser.add_argument(
        "--window-end", type=float, required=True, help="Explicit classification-window end on source C3D clock [s]"
    )
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument("--stance-index", type=int, default=0)
    parser.add_argument("--padding", type=float, default=0.02, help="Adjacent flight included at each end [s]")
    parser.add_argument("--belt-offset", type=float, default=0.0)
    parser.add_argument("--initial-x", type=float, default=0.0)
    parser.add_argument("--initial-z", type=float, default=1.0)
    parser.add_argument("--initial-vx", type=float, default=None)
    parser.add_argument("--initial-vz", type=float, default=0.0)
    args = parser.parse_args()
    source = args.source_worktree.resolve()
    if not (source / "projects/gait_c3d/contact_dataset.py").is_file():
        parser.error("source worktree must supply the public measured C3D contact_dataset adapter")
    args.output = args.output.resolve()
    config = vars(args).copy()
    del config["source_worktree"]
    config["output"] = str(args.output)
    if args.output.exists():
        parser.error("output already exists; choose a new path to preserve the previous sealed artifact")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".impedance-profile-", dir=args.output.parent) as directory:
        temporary = Path(directory) / "stance.json"
        config["output"] = str(temporary)
        helper = "import json,runpy,sys; runpy.run_path(sys.argv[1])['_extract'](json.loads(sys.argv[2]))"
        subprocess.run(
            [
                "uv",
                "run",
                "--no-sync",
                "--with",
                "ezc3d==1.7.2",
                "python",
                "-c",
                helper,
                str(Path(__file__).resolve()),
                json.dumps(config),
            ],
            cwd=source,
            env={
                **{key: value for key, value in os.environ.items() if key != "VIRTUAL_ENV"},
                "PYTHONDONTWRITEBYTECODE": "1",
            },
            check=True,
        )
        load_profile(temporary)
        temporary.replace(args.output)
    print(f"Wrote verified running profile: {args.output}")


if __name__ == "__main__":
    main()
