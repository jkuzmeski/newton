# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Prepare continuous bilateral multi-stride references from Visual3D exports.

Each segment starts and ends at mid-hip height peaks in flight and spans a fixed
number of alternating foot contacts. Both legs are prepared with
:func:`..cartesian.prepare_visual3d.prepare` over the same window, then joined
on the mid-hip center. The single treadmill plate force is attributed to one
foot per contact: running has no double support, and the COP-to-foot proximity
rule of the single-leg dataset identifies the loaded foot. Loaded samples that
neither foot clearly owns reject the segment.

Hip and knee are re-solved per leg by regularized inverse kinematics toward each
leg's measured hip-to-ankle vector from the shared mid-hip root, keeping the
measured foot angle. Near full knee extension the sagittal projection can exceed
the fixed segment lengths; the measured joint angles then keep the knee from
snapping straight, and each record reports the remaining ankle error. A planar
model attaches both thighs at one point, so pelvic rotation and obliquity
(several centimeters of sagittal hip offset) move each ankle by that offset;
each record reports its peak. No force enters this kinematic processing.

Splits are contiguous source-time blocks per trial: one evaluation block with a
one-segment guard on either side, the rest for training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from functools import lru_cache
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from ..cartesian import data as reference_data
from ..cartesian.prepare_dataset import _flight_velocity, _marker_centroid_on_analog_clock, _peak_indices
from ..cartesian.prepare_visual3d import _read_triplets, prepare
from ..cartesian.visual3d import load_visual3d_export
from .mechanics import SIDES, BilateralChain, HeadArmsTrunk, chain_from_references
from .shoes import write_mirrored

ROOT = Path(__file__).resolve().parents[3]
REFERENCE_SCHEMA = "bilateral_reference_1"
DATASET_SCHEMA = "bilateral_runner_dataset_1"
TRIALS = ("FR3_1", "FR3_2")
DEFAULT_SHOE_PITCH_RAD = -0.24446041090480894
DEFAULT_MOUNT_M = (-0.03186147427106201, 0.0, 0.10943209684347802)
SEED = 20261013
FOOT_MARKERS = ("CAL1", "CAL2", "CAL3", "MT1H", "MT5H", "TOE")


@dataclass(frozen=True)
class ContactRule:
    """Attribute single-plate force to one foot by COP proximity.

    Attributes:
        threshold_n: Vertical force above which COP is trusted [N].
        gate_n: Vertical force bounding each contact's extent [N].
        radius_m: Maximum COP distance from the owning foot's marker centroid [m].
        margin_m: Required COP distance advantage over the other foot [m].
        minimum_contact_s: Shortest accepted contact [s].
        bridge_s: Longest dropout bridged inside one contact [s].
    """

    threshold_n: float = 50.0
    gate_n: float = 1.0
    radius_m: float = 0.15
    margin_m: float = 0.04
    minimum_contact_s: float = 0.08
    bridge_s: float = 0.02


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _groups(mask: np.ndarray, bridge: int) -> list[tuple[int, int]]:
    groups: list[list[int]] = []
    for sample in np.flatnonzero(mask):
        if not groups or sample - groups[-1][1] > bridge:
            groups.append([int(sample), int(sample)])
        else:
            groups[-1][1] = int(sample)
    return [(a, b) for a, b in groups]


def contact_labels(
    force_z: np.ndarray, cop_m: np.ndarray, foot_centroids: dict[str, np.ndarray], rate_hz: float, rule: ContactRule
) -> dict[str, Any]:
    """Return per-foot contact masks, ordered contact events, and unowned loaded samples.

    Args:
        force_z: Vertical plate force on the analog clock [N].
        cop_m: Plate COP on the analog clock [m], shape [samples, 3].
        foot_centroids: Foot marker centroid per side on the analog clock [m].
        rate_hz: Analog rate [Hz].
        rule: Attribution thresholds.
    """
    loaded = (force_z > rule.threshold_n) & np.isfinite(cop_m).all(axis=1)
    distance = {side: np.linalg.norm(cop_m - foot_centroids[side], axis=1) for side in SIDES}
    finite = np.isfinite(distance["right"]) & np.isfinite(distance["left"])
    events = []
    masks = {side: np.zeros(len(force_z), dtype=bool) for side in SIDES}
    for side, other in (("right", "left"), ("left", "right")):
        own = loaded & finite & (distance[side] <= rule.radius_m) & (distance[side] + rule.margin_m < distance[other])
        for first, last in _groups(own, int(round(rule.bridge_s * rate_hz))):
            if last - first + 1 < int(rule.minimum_contact_s * rate_hz):
                continue
            start, end = first, last
            # Extend through the low-force loading and unloading the COP cannot attribute.
            while start > 0 and force_z[start - 1] > rule.gate_n:
                start -= 1
            while end + 1 < len(force_z) and force_z[end + 1] > rule.gate_n:
                end += 1
            events.append((start, end, side))
            masks[side][start : end + 1] = True
    events.sort()
    overlap = masks["right"] & masks["left"]
    unowned = (force_z > rule.threshold_n) & ~(masks["right"] | masks["left"])
    return {"masks": masks, "events": events, "ambiguous": overlap | unowned}


def _foot_centroids(trial_root: Path, trial) -> dict[str, np.ndarray]:
    labels, markers, _ = _read_triplets(trial_root / "motion_processed_targets.txt", trial.manifest)
    result = {}
    for side in SIDES:
        prefix = "R" if side == "right" else "L"
        names = [f"{prefix}{name}" for name in FOOT_MARKERS]
        missing = [name for name in names if name not in labels]
        if missing:
            raise ValueError(f"{trial_root.name}: missing foot markers {missing}")
        with np.errstate(invalid="ignore"):
            centroid = np.nanmean(markers[:, [labels.index(name) for name in names], :], axis=1)
        result[side] = _marker_centroid_on_analog_clock(trial.analog_time_s, trial.marker_time_s, centroid)
    return result


def _mid_hip(trial_root: Path, trial) -> np.ndarray:
    labels, centers, valid = _read_triplets(trial_root / "motion_joint_centers.txt", trial.manifest)
    ids = [labels.index("RHIP"), labels.index("LHIP")]
    hip = 0.5 * (centers[:, ids[0]] + centers[:, ids[1]])
    hip[~(valid[:, ids[0]] & valid[:, ids[1]])] = np.nan
    return hip


@lru_cache(maxsize=4)
def _trial_context(trial_root: Path, rule: ContactRule):
    trial = load_visual3d_export(trial_root)
    labels = contact_labels(
        trial.force_n[:, 2], trial.cop_m, _foot_centroids(trial_root, trial), trial.analog_rate_hz, rule
    )
    return trial, labels, _mid_hip(trial_root, trial)


def candidate_windows(trial_root: Path, steps: int, rule: ContactRule) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return non-overlapping peak-to-peak windows spanning ``steps`` alternating contacts."""
    trial, labels, hip = _trial_context(trial_root, rule)
    finite = np.isfinite(hip[:, 2])
    height = np.where(finite, hip[:, 2], -np.inf)
    peaks = np.asarray(_peak_indices(height, trial.marker_time_s))
    peak_times = trial.marker_time_s[peaks]
    events = labels["events"]
    times = trial.analog_time_s
    windows, k = [], 0
    while k + steps <= len(events):
        chosen = events[k : k + steps]
        start_time, end_time = times[chosen[0][0]], times[chosen[-1][1]]
        before = np.flatnonzero(peak_times < start_time)
        after = np.flatnonzero(peak_times > end_time)
        sides = [event[2] for event in chosen]
        alternating = all(a != b for a, b in pairwise(sides))
        if not alternating or len(before) == 0 or len(after) == 0:
            k += 1
            continue
        start, end = int(peaks[before[-1]]), int(peaks[after[0]])
        start_s, end_s = float(trial.marker_time_s[start]), float(trial.marker_time_s[end])
        # Exactly the chosen events lie inside, the window is fully tracked, and every load is owned.
        inside = [e for e in events if times[e[1]] >= start_s and times[e[0]] <= end_s]
        analog = (times >= start_s) & (times <= end_s)
        between = [times[b[0]] - times[a[1]] for a, b in pairwise(chosen)]  # flight gaps between contacts
        if (
            inside != chosen
            or labels["ambiguous"][analog].any()
            or not finite[start : end + 1].all()
            or min(between, default=1.0) <= 0
        ):
            k += 1
            continue
        windows.append(
            {
                "start_s": start_s,
                "end_s": end_s,
                "start_index": start,
                "end_index": end,
                "contacts": [
                    {"side": side, "start_s": float(times[a]), "end_s": float(times[b])} for a, b, side in chosen
                ],
            }
        )
        k += steps
    summary = {
        "contact_events": {side: sum(e[2] == side for e in events) for side in SIDES},
        "ambiguous_loaded_samples": int(labels["ambiguous"].sum()),
        "windows": len(windows),
    }
    return windows, summary


def _side_reference(
    trial_root: Path, window: dict, side: str, profile: Path, shoe: Path, selection: dict, pitch: float, foot: str
) -> dict:
    with tempfile.TemporaryDirectory() as scratch:
        bundle = Path(scratch) / side
        prepare(
            trial_root,
            trial_root,
            bundle,
            profile,
            shoe,
            side=side,
            start_s=window["start_s"],
            end_s=window["end_s"],
            subject_mass_kg=float(selection["subject_mass_kg"]),
            belt_speed_m_s=float(selection["belt_speed_m_s"]),
            virtual_foot_reference=foot,
            shoe_static_pitch_rad=pitch,
            # Each window's per-foot ownership is checked independently by COP proximity.
            allow_force_side_override=True,
        )
        return reference_data.load(bundle / "reference.npz")


def join_sides(references: dict[str, dict], chain: BilateralChain | None = None) -> dict[str, np.ndarray]:
    """Join per-side single-leg references on the mid-hip center.

    Returns measured bilateral coordinates ``[x, z, pelvis, hip, knee, ankle per
    side]`` [m, rad]. With ``chain``, hip and knee are re-solved by regularized
    inverse kinematics toward each leg's measured hip-to-ankle vector, see
    :meth:`.mechanics.BilateralChain.fit_leg`.
    """
    right, left = references["right"], references["left"]
    if not np.array_equal(right["time_s"], left["time_s"]):
        raise ValueError("Per-side references must share the motion clock")
    if "pelvis_target_rad" not in right:
        raise ValueError("Bilateral references require measured pelvis tilt")
    pelvis = right["pelvis_target_rad"]
    hip = 0.5 * (right["hip_target_m"] + left["hip_target_m"])
    columns = [hip, pelvis[:, None]]
    for reference in (right, left):
        state = reference["state"]
        joint = np.unwrap(np.mod(state[:, 2] - pelvis, 2 * np.pi) - np.pi)
        columns.append(np.column_stack((joint, state[:, 3], state[:, 4])))
    q = np.column_stack(columns)
    if chain is not None:
        for side, reference in (("right", right), ("left", left)):
            q = chain.fit_leg(q, leg_ankle_targets(reference, hip), side)
    return q


def leg_ankle_targets(reference: dict, mid_hip: np.ndarray) -> np.ndarray:
    """Return a leg's measured hip-to-ankle vector placed on the shared mid-hip root [m].

    Pelvic rotation and obliquity move each hip center by several centimeters
    from the mid-hip in the sagittal projection. A planar model attaches both
    thighs at one point, so each leg keeps its measured configuration and its
    ankle moves with that offset; :func:`build_segment` reports the size.
    """
    return reference["ankle_target_m"] - reference["hip_target_m"] + mid_hip


def build_segment(
    trial_root: Path,
    window: dict,
    profile: Path,
    shoe: Path,
    selection: dict,
    *,
    pitch: float,
    foot_reference: str,
    hat: HeadArmsTrunk,
    rule: ContactRule,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Prepare one bilateral reference and its manifest record."""
    references = {
        side: _side_reference(trial_root, window, side, profile, shoe, selection, pitch, foot_reference)
        for side in SIDES
    }
    profile_data = json.loads(profile.read_text(encoding="utf-8"))
    chain = chain_from_references(references, profile_data, float(selection["subject_mass_kg"]), hat)
    measured = join_sides(references)
    q = join_sides(references, chain)
    right = references["right"]
    trial, labels, hip = _trial_context(trial_root, rule)
    origin_s = float(trial.marker_time_s[window["start_index"]])
    force_time = right["grf_time_s"]
    analog = np.searchsorted(trial.analog_time_s, force_time + origin_s - 1e-7)
    if np.any(np.abs(trial.analog_time_s[analog] - origin_s - force_time) > 1e-6):
        raise ValueError("Per-side GRF clock does not align with the source analog clock")
    contact = np.stack([labels["masks"][side][analog] for side in SIDES])
    if np.any(contact[0] & contact[1]) or np.any(labels["ambiguous"][analog]):
        raise ValueError("Window contains double or unattributed support")
    total = right["grf_target_n"]
    by_side = np.stack([np.where(contact[s][:, None], total, 0.0) for s in range(2)])
    mid_hip = 0.5 * (right["hip_target_m"] + references["left"]["hip_target_m"])
    flight = _flight_velocity(
        trial,
        hip[:, [0, 2]],
        window["start_index"] + 2,
        float(selection["subject_mass_kg"]),
        float(selection["belt_speed_m_s"]),
    )
    reference = {
        "time_s": right["time_s"],
        "q": q,
        "q_measured": measured,
        "hip_target_m": mid_hip,
        "side_hip_m": np.stack([references[side]["hip_target_m"] for side in SIDES]),
        "ankle_target_m": np.stack([references[side]["ankle_target_m"] for side in SIDES]),
        "pelvis_target_rad": right["pelvis_target_rad"],
        "grf_time_s": force_time,
        "grf_target_n": total,
        "grf_side_n": by_side,
        "contact_side": contact.astype(np.int8),
        "cop_target_m": right["cop_target_m"],
        "lengths_m": chain.lengths_m,
        "endpoint_local_m": chain.endpoint_local_m,
        "static_pitch_rad": np.array([float(references[side]["static_pitch_rad"]) for side in SIDES]),
        "shoe_static_pitch_rad": np.asarray(pitch),
        "subject_mass_kg": np.asarray(float(selection["subject_mass_kg"])),
        "belt_speed_m_s": np.asarray(float(selection["belt_speed_m_s"])),
    }
    if all("foot_ground_target_rad" in references[side] for side in SIDES):
        reference["foot_ground_target_rad"] = np.stack([references[side]["foot_ground_target_rad"] for side in SIDES])
    reference["metadata_json"] = np.asarray(
        json.dumps(
            {
                "schema": REFERENCE_SCHEMA,
                "coordinates": ["x", "z", "pelvis", "hip_r", "knee_r", "ankle_r", "hip_l", "knee_l", "ankle_l"],
                "trial": trial_root.name,
                "source_window_s": [window["start_s"], window["end_s"]],
                "foot_reference": foot_reference,
                "contact_rule": asdict(rule),
                "force_attribution": "single plate force assigned to the COP-identified foot; zero for the other",
                "target_processing": "regularized hip/knee IK toward each measured hip-to-ankle vector on the mid-hip root (5 mm, 0.2 rad); foot angle kept",
            }
        )
    )
    validate(reference)
    targets = [leg_ankle_targets(references[side], mid_hip) for side in SIDES]
    fk_error = max(
        float(np.max(np.linalg.norm(np.array([chain.ankle(row, side)[0] for row in q]) - target, axis=1)))
        for side, target in zip(SIDES, targets, strict=True)
    )
    projection = max(
        float(np.max(np.linalg.norm(target - ankles, axis=1)))
        for target, ankles in zip(targets, reference["ankle_target_m"], strict=True)
    )
    record = {
        "start_s": window["start_s"],
        "end_s": window["end_s"],
        "contacts": window["contacts"],
        "flight_velocity_m_s": flight,
        "flight_velocity_frame": 2,
        "ankle_fk_error_peak_m": fk_error,
        "ankle_projection_offset_peak_m": projection,
        "knee_resolve_change_peak_rad": float(np.max(np.abs(q[:, [4, 7]] - measured[:, [4, 7]]))),
    }
    return reference, record


def validate(reference: dict[str, np.ndarray]) -> None:
    """Validate bilateral reference clocks, shapes, and metadata."""
    metadata = json.loads(str(reference["metadata_json"]))
    if metadata.get("schema") != REFERENCE_SCHEMA:
        raise ValueError("Unsupported bilateral reference schema")
    time, force_time = reference["time_s"], reference["grf_time_s"]
    n, m = len(time), len(force_time)
    if time.ndim != 1 or n < 4 or time[0] != 0 or np.any(np.diff(time) <= 0):
        raise ValueError("time_s must start at zero and increase strictly")
    if force_time[0] > 0 or force_time[-1] < time[-1] or np.any(np.diff(force_time) <= 0):
        raise ValueError("GRF clock must increase and cover the motion window")
    shapes = {
        "q": (n, 9),
        "q_measured": (n, 9),
        "hip_target_m": (n, 2),
        "side_hip_m": (2, n, 2),
        "ankle_target_m": (2, n, 2),
        "pelvis_target_rad": (n,),
        "grf_target_n": (m, 2),
        "grf_side_n": (2, m, 2),
        "contact_side": (2, m),
        "cop_target_m": (m,),
        "lengths_m": (2, 2),
        "endpoint_local_m": (2, 2),
        "static_pitch_rad": (2,),
    }
    for name, shape in shapes.items():
        if reference[name].shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
        if not np.isfinite(reference[name]).all():
            raise ValueError(f"{name} must be finite")
    if np.any(reference["contact_side"].sum(0) > 1):
        raise ValueError("A force sample cannot belong to both feet")


def load(path: str | Path) -> dict[str, np.ndarray]:
    """Load and validate a bilateral reference NPZ without pickle."""
    with np.load(Path(path), allow_pickle=False) as archive:
        reference = {name: archive[name].copy() for name in archive.files}
    validate(reference)
    return reference


def _split(count: int, rng: np.random.Generator, eval_fraction: float) -> list[str]:
    """Assign one contiguous eval block with a one-segment guard on either side."""
    size = max(1, int(round(eval_fraction * count)))
    if count < size + 3:
        raise ValueError("Too few segments for a guarded evaluation block")
    start = int(rng.integers(1, count - size))
    splits = ["train"] * count
    for index in range(max(0, start - 1), min(count, start + size + 1)):
        splits[index] = "guard"
    for index in range(start, start + size):
        splits[index] = "eval"
    return splits


def _single_leg_windows(manifest_root: Path, trial: str, limit: int | None) -> tuple[list, list, list]:
    """Return the IDs, windows, and splits of one trial in a single-leg stance dataset."""
    manifest = json.loads((Path(manifest_root) / "manifest.json").read_text(encoding="utf-8"))
    members = [m for m in manifest["members"] if m["trial"] == trial][:limit]
    windows = [
        {
            "start_s": m["start_s"],
            "end_s": m["end_s"],
            "start_index": m["start_peak_index"],
            "end_index": m["end_peak_index"],
            "contacts": [{"side": "right", "start_s": m["contact_start_s"], "end_s": m["contact_end_s"]}],
        }
        for m in members
    ]
    return [m["id"] for m in members], windows, [m["split"] for m in members]


def build_dataset(
    output: Path,
    *,
    data_root: Path = ROOT / "data/F01",
    profile: Path,
    shoe: Path,
    trials: tuple[str, ...] = TRIALS,
    steps: int = 6,
    eval_fraction: float = 0.1,
    seed: int = SEED,
    pitch: float = DEFAULT_SHOE_PITCH_RAD,
    mount_m: tuple[float, float, float] = DEFAULT_MOUNT_M,
    foot_reference: str = "sole_markers",
    hat: HeadArmsTrunk | None = None,
    rule: ContactRule | None = None,
    limit: int | None = None,
    windows_from: Path | None = None,
) -> Path:
    """Write a bilateral multi-stride dataset into a fresh directory.

    Args:
        windows_from: Reuse the windows, IDs, and splits of a single-leg
            ``peak_hip_stance_dataset_1`` instead of cutting multi-stride
            segments, so two-leg models score on the single-leg baseline's stances.
    """
    hat, rule = hat or HeadArmsTrunk(), rule or ContactRule()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {output}")
    building = output.with_name(f".{output.name}.{os.getpid()}.building")
    building.mkdir(parents=True)
    assets = building / "assets"
    assets.mkdir()
    shutil.copyfile(profile, assets / "profile.json")
    shutil.copyfile(shoe, assets / "digital_shoe_left.json")
    write_mirrored(shoe, assets / "digital_shoe_right.json")
    rng = np.random.default_rng(seed)
    members, rejected, summaries = [], [], {}
    for trial_name in trials:
        trial_root = data_root / trial_name
        selection_path = trial_root / "stance_selection.json"
        if not selection_path.is_file():
            # Subject mass and belt speed are shared; only FR3_1 carries the selection record.
            selection_path = data_root / trials[0] / "stance_selection.json"
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if windows_from is None:
            windows, summaries[trial_name] = candidate_windows(trial_root, steps, rule)
            if limit is not None:
                windows = windows[:limit]
            splits = _split(len(windows), rng, eval_fraction)
            ids = [f"{trial_name}_{index:03d}" for index in range(len(windows))]
        else:
            ids, windows, splits = _single_leg_windows(windows_from, trial_name, limit)
            summaries[trial_name] = {"windows": len(windows), "source": str(windows_from)}
        for sample_id, window, split in zip(ids, windows, splits, strict=True):
            if split == "guard":
                rejected.append({"id": sample_id, "reason": "evaluation guard"})
                continue
            try:
                reference, record = build_segment(
                    trial_root,
                    window,
                    assets / "profile.json",
                    assets / "digital_shoe_left.json",
                    selection,
                    pitch=pitch,
                    foot_reference=foot_reference,
                    hat=hat,
                    rule=rule,
                )
            except ValueError as error:
                rejected.append({"id": sample_id, "reason": str(error)})
                continue
            path = building / split / f"{sample_id}.npz"
            path.parent.mkdir(exist_ok=True)
            np.savez_compressed(path, **reference)
            members.append(
                {
                    "id": sample_id,
                    "trial": trial_name,
                    "split": split,
                    "reference": str(path.relative_to(building).as_posix()),
                    "speed_m_s": float(selection["belt_speed_m_s"]),
                    **record,
                }
            )
            print(f"{sample_id} {split} {window['end_s'] - window['start_s']:.2f} s", flush=True)
    manifest = {
        "schema": DATASET_SCHEMA,
        "seed": seed,
        "policy": {
            "trials": list(trials),
            "steps_per_segment": steps,
            "windows": "Mid-hip height peak before the first contact to the peak after the last; contacts alternate."
            if windows_from is None
            else f"Single-leg peak-to-peak stance windows of {windows_from}",
            "eval_block": "One contiguous block per trial with a one-segment guard on either side.",
            "eval_fraction": eval_fraction,
            "foot_reference": foot_reference,
            "contact_rule": asdict(rule),
        },
        "subject_id": "F01",
        "head_arms_trunk": asdict(hat),
        "shoe": {"mount_m": list(mount_m), "pitch_rad": pitch},
        "shared_assets": {
            name: {"file": f"assets/{name}", "sha256": _sha256(assets / name)}
            for name in ("profile.json", "digital_shoe_left.json", "digital_shoe_right.json")
        },
        "source_hashes": {
            str(path): _sha256(path)
            for trial in trials
            for path in sorted((data_root / trial).rglob("*"))
            if path.is_file()
        },
        "trial_summaries": summaries,
        "members": members,
        "rejected": rejected,
    }
    (building / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    building.rename(output)
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/F01")
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--shoe", type=Path, required=True, help="Left-shoe artifact; the right shoe is mirrored.")
    parser.add_argument("--trials", nargs="+", default=list(TRIALS))
    parser.add_argument("--steps", type=int, default=6, help="Alternating contacts per segment.")
    parser.add_argument("--eval-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--shoe-static-pitch-rad", type=float, default=DEFAULT_SHOE_PITCH_RAD)
    parser.add_argument("--mount", type=float, nargs=3, default=list(DEFAULT_MOUNT_M))
    parser.add_argument(
        "--virtual-foot-reference", choices=("sole_markers", "reconstructed_ground"), default="sole_markers"
    )
    parser.add_argument("--limit", type=int, help="Use at most this many windows per trial (smoke tests).")
    parser.add_argument("--windows-from", type=Path, help="Reuse a single-leg stance dataset's windows and splits.")
    args = parser.parse_args(argv)
    print(
        build_dataset(
            args.output,
            data_root=args.data_root,
            profile=args.profile,
            shoe=args.shoe,
            trials=tuple(args.trials),
            steps=args.steps,
            eval_fraction=args.eval_fraction,
            seed=args.seed,
            pitch=args.shoe_static_pitch_rad,
            mount_m=tuple(args.mount),
            foot_reference=args.virtual_foot_reference,
            limit=args.limit,
            windows_from=args.windows_from,
        )
    )


if __name__ == "__main__":
    main()
