# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Identify shared generative runner dynamics from offline measurements.

Measurement arrays belong to this module, never to the runner. CUDA executes
candidate dynamics and objective reductions; CPU remains the reference backend.
Run ``python -m projects.impedance_instron.hogan.identify --help`` for commands.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..cartesian import data as reference_data
from ..cartesian import profile as profile_data
from ..cartesian.shoe import Shoe
from .mechanics import Chain, RestOfBody, chain_from_profile, from_leg_coordinates
from .runner import RolloutConfig, Runner, State, Task, simulate


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def coordinates(reference: dict, chain: Chain | None = None, *, height_offset_m: float = 0.0) -> np.ndarray:
    """Convert measured coordinates without GRF integration or filtering.

    With ``chain`` and a measured ``ankle_target_m``, hip and knee angles are
    re-solved per frame so fixed-length FK reaches the measured hip and ankle
    centers while the foot's absolute angle is kept. This uses positions only,
    so it is causal and independent of measured force. The exported knee angle
    is otherwise inconsistent with the exported joint centers.

    The upstream prepared motion may already be filtered. Missing pelvis
    orientation is rejected rather than treating upright posture as measured.
    """
    if "pelvis_target_rad" not in reference:
        raise ValueError("Generative identification requires measured pelvis_target_rad")
    if not np.isfinite(height_offset_m):
        raise ValueError("height_offset_m must be finite")
    q, _ = from_leg_coordinates(
        reference["state"], np.zeros_like(reference["state"]), reference["pelvis_target_rad"], 0.0
    )
    q[:, 1] += height_offset_m
    if chain is not None and "ankle_target_m" in reference:
        q = chain.reach(q, reference["ankle_target_m"] + np.array([0.0, height_offset_m]))
    return q


def initialize(time_s, q, *, com_velocity_m_s=None, chain: Chain | None = None) -> State:
    """Initialize at the END of a supplied three-frame observation prefix.

    Quadratic backward differentiation uses only these three observed positions,
    not exported velocities, measured forces, a plan, or future stance events.
    Phase uses current hip angle/rate as a fixed causal engineering convention.
    Torque and load memory start at zero, assuming a relaxed shoe in flight.

    Args:
        com_velocity_m_s: Flight whole-body COM velocity [m/s] estimated from data
            up to the prefix end. When given, the hip velocity is chosen so the
            ``chain`` COM moves at this velocity with the prefix joint rates; the
            single modeled leg otherwise carries unbalanced swing momentum.
        chain: Model chain; required with ``com_velocity_m_s``.
    """
    time = np.asarray(time_s, dtype=float)
    q = np.asarray(q, dtype=float)
    if time.shape != (3,) or q.shape != (3, 6) or not np.isfinite(time).all() or not np.isfinite(q).all():
        raise ValueError("Initialization needs three finite times and three six-coordinate positions")
    if np.any(np.diff(time) <= 0):
        raise ValueError("Initialization times must increase strictly")
    t = time - time[-1]
    coefficients = np.linalg.solve(np.column_stack((np.ones(3), t, t * t)), q)
    velocity = coefficients[1]
    if com_velocity_m_s is not None:
        com_velocity = np.asarray(com_velocity_m_s, dtype=float)
        if com_velocity.shape != (2,) or not np.isfinite(com_velocity).all():
            raise ValueError("com_velocity_m_s must be two finite values")
        if chain is None:
            raise ValueError("com_velocity_m_s needs the model chain")
        velocity[:2] = com_velocity - chain.com_jacobian(q[-1])[:, 2:] @ velocity[2:]
    phase = math.atan2(-velocity[3] / 8.0, q[-1, 3]) % (2 * math.pi)
    return State(q[-1], velocity, phase_rad=phase)


def _biquad(x: np.ndarray, coefficients: tuple[float, ...]) -> np.ndarray:
    """Run one transposed direct-form-II biquad pass from zero state over rows of ``x``.

    :mod:`.gpu_residuals` evaluates the same recurrence in the same operation
    order, so both backends observe a trajectory identically.
    """
    b0, b1, b2, a1, a2 = coefficients
    y = np.empty_like(x)
    z1 = np.zeros(x.shape[1])
    z2 = np.zeros(x.shape[1])
    for k in range(len(x)):
        value = x[k]
        out = b0 * value + z1
        z1 = b1 * value - a1 * out + z2
        z2 = b2 * value - a2 * out
        y[k] = out
    return y


@dataclass(frozen=True)
class ForceObservation:
    """Measurement model of a low-passed, gated force-plate GRF target.

    Visual3D exports the plate force through a zero-lag second-order
    Butterworth (run forward and backward) and zeroes both components where the
    filtered vertical force is below a small gate. The F01 exports use a 20 Hz
    cutoff, the running convention; exports before 2026-10-09 used 6 Hz. A
    physical contact force has content above that bandwidth that such a target
    cannot contain, and the smoothing moves the target's loading earlier and its
    unloading later. Passing simulated GRF through the same processing compares
    like with like. The observation only affects scores and residuals, never the
    dynamics.

    Both passes start from zero filter state, which is exact when the simulated
    window starts and ends without contact.

    A target filtered this way says nothing about force content above its
    bandwidth, so fitting the observed force alone leaves stance vibration
    unconstrained. ``vibration_weight`` penalizes that content explicitly, as
    the physical minus observed force, on the GRF error scale.

    Args:
        cutoff_hz: Nominal cutoff of the two-pass filter [Hz]. Each pass uses
            Winter's correction so that the combined response is -3 dB here.
        gate_n: Filtered vertical force below which both components are zeroed [N].
        vibration_weight: Weight of the physical force above the observation
            bandwidth relative to the GRF error; zero fits only the observed force.
    """

    cutoff_hz: float = 20.0
    gate_n: float = 1.0
    vibration_weight: float = 1.0

    def __post_init__(self):
        if not np.isfinite(self.cutoff_hz) or self.cutoff_hz <= 0:
            raise ValueError("Force observation cutoff must be finite and positive")
        if not np.isfinite(self.gate_n) or self.gate_n < 0:
            raise ValueError("Force observation gate must be finite and nonnegative")
        if not np.isfinite(self.vibration_weight) or self.vibration_weight < 0:
            raise ValueError("Force vibration weight must be finite and nonnegative")

    def coefficients(self, dt_s: float) -> tuple[float, float, float, float, float]:
        """Return one pass's ``b0, b1, b2, a1, a2`` at timestep ``dt_s`` [s]."""
        if not np.isfinite(dt_s) or dt_s <= 0 or self.cutoff_hz * dt_s >= 0.25:
            raise ValueError("Force observation needs a positive timestep well below the cutoff period")
        # Winter's two-pass correction applied to the prewarped second-order cutoff.
        omega = math.tan(math.pi * self.cutoff_hz * dt_s) / (math.sqrt(2.0) - 1.0) ** 0.25
        norm = 1.0 / (1.0 + math.sqrt(2.0) * omega + omega * omega)
        b0 = omega * omega * norm
        return (
            b0,
            2.0 * b0,
            b0,
            2.0 * (omega * omega - 1.0) * norm,
            (1.0 - math.sqrt(2.0) * omega + omega * omega) * norm,
        )

    def apply(self, grf_n, dt_s: float) -> np.ndarray:
        """Return the observed horizontal/vertical GRF [N], shape [steps, 2]."""
        force = np.asarray(grf_n, dtype=float)
        if force.ndim != 2 or force.shape[1] != 2:
            raise ValueError("GRF must have shape [steps, 2]")
        coefficients = self.coefficients(dt_s)
        observed = _biquad(_biquad(force, coefficients)[::-1], coefficients)[::-1].copy()
        observed[observed[:, 1] < self.gate_n] = 0.0
        return observed


def observed_grf(trace: dict, trial: Trial) -> np.ndarray:
    """Return simulated GRF as the trial's target was measured [N], shape [steps, 2]."""
    grf = trace["grf_n"]
    if trial.force_observation is None or not len(grf):
        return grf
    return trial.force_observation.apply(grf, float(trace["time_s"][1] - trace["time_s"][0]))


def vibration_weight(trial: Trial) -> float:
    """Return the weight of simulated GRF content the trial's target cannot observe."""
    return 0.0 if trial.force_observation is None else trial.force_observation.vibration_weight


@dataclass
class Trial:
    """Offline trial boundary separating predictive inputs from fitting targets."""

    id: str
    split: str
    chain: Chain
    shoe: Shoe
    task: Task
    initial: State
    time_s: np.ndarray
    """Observation times relative to the end of the initialization prefix [s]."""
    q: np.ndarray
    """Observed coordinates [m, m, rad, rad, rad, rad], shape [frames, 6]."""
    force_time_s: np.ndarray
    """Native force target clock [s], covering the prediction horizon."""
    grf_n: np.ndarray
    """Measured horizontal/vertical GRF [N], shape [force_frames, 2]."""
    provenance: dict
    force_observation: ForceObservation | None = None
    """How ``grf_n`` was processed; simulated GRF is observed the same way before
    scoring. ``None`` compares the raw simulated contact force."""

    def __post_init__(self):
        self.time_s = np.asarray(self.time_s, dtype=float).copy()
        self.q = np.asarray(self.q, dtype=float).copy()
        self.force_time_s = np.asarray(self.force_time_s, dtype=float).copy()
        self.grf_n = np.asarray(self.grf_n, dtype=float).copy()
        if self.split not in ("train", "eval"):
            raise ValueError("Trial split must be train or eval")
        if self.time_s.ndim != 1 or len(self.time_s) < 2 or self.time_s[0] != 0:
            raise ValueError("Trial times must start at zero and contain at least two samples")
        if self.q.shape != (len(self.time_s), 6):
            raise ValueError("Trial coordinate shape must match its clock")
        if self.force_time_s.ndim != 1 or len(self.force_time_s) < 2:
            raise ValueError("Trial force clock must contain at least two samples")
        if self.grf_n.shape != (len(self.force_time_s), 2):
            raise ValueError("Trial GRF shape must match its clock")
        if not all(np.isfinite(a).all() for a in (self.time_s, self.q, self.force_time_s, self.grf_n)):
            raise ValueError("Trial observations must be finite")
        if np.any(np.diff(self.time_s) <= 0) or np.any(np.diff(self.force_time_s) <= 0):
            raise ValueError("Trial clocks must increase strictly")
        if self.force_time_s[0] > 0 or self.force_time_s[-1] < self.time_s[-1]:
            raise ValueError("Trial forces must cover the prediction horizon")
        if self.force_observation is not None and not isinstance(self.force_observation, ForceObservation):
            raise ValueError("force_observation must be a ForceObservation or None")

    @property
    def duration_s(self) -> float:
        return float(self.time_s[-1])


def compatibility(reference: dict, q: np.ndarray, chain: Chain, shoe: Shoe) -> dict:
    """Check observed ankle/shoe/load compatibility; never alter rollout inputs.

    Only fixed-length FK disagreement with the measured ankle (10 mm) fails the
    screen. Loaded clearance and COP footprint conflicts are reported but not
    gated: marker-based ankle and shoe registration are only good to several
    millimeters, and treadmill COP is unreliable at low force.
    """
    times = reference["time_s"]
    forces = np.column_stack(
        [np.interp(times, reference["grf_time_s"], reference["grf_target_n"][:, axis]) for axis in range(2)]
    )
    loaded = forces[:, 1] > 50.0
    gaps, model_gaps, ankle_errors, cop_excess = [], [], [], []
    cop = None
    if "cop_target_m" in reference:
        cop = np.interp(times, reference["grf_time_s"], reference["cop_target_m"])
    for i, row in enumerate(q):
        ankle = chain.point(row, 3, np.zeros(2))[0]
        # Diagnose contact against measured ankle centers even if fixed-length FK disagrees.
        if "ankle_target_m" in reference:
            measured = reference["ankle_target_m"][i].copy()
            measured[1] += row[1] - reference["state"][i, 1]
            ankle_errors.append(float(np.linalg.norm(ankle - measured)))
        else:
            measured = ankle
        outline = shoe.outline(measured, chain.angle(row, 3))
        gaps.append(float(outline[:, 1].min()))
        model_gaps.append(float(shoe.outline(ankle, chain.angle(row, 3))[:, 1].min()))
        if cop is not None:
            cop_excess.append(float(max(outline[:, 0].min() - cop[i], cop[i] - outline[:, 0].max(), 0.0)))
    gaps = np.asarray(gaps)
    conflict = loaded & (gaps > 0.002)
    model_conflict = loaded & (np.asarray(model_gaps) > 0.002)
    errors = np.asarray(ankle_errors)
    cop_bad = np.asarray(cop_excess) > 0.002 if cop is not None else np.zeros(len(q), dtype=bool)
    return {
        "passed": not np.any(errors > 0.01),
        "clearance_gating": False,
        "model_loaded_clearance_conflict_frames": int(model_conflict.sum()),
        "model_loaded_clearance_peak_m": float(np.max(np.asarray(model_gaps)[loaded])) if loaded.any() else None,
        "clearance_basis": "measured ankle and independently checked model FK; all nominal bottom points",
        "loaded_clearance_conflict_frames": int(conflict.sum()),
        "loaded_clearance_peak_m": float(np.max(gaps[loaded])) if loaded.any() else None,
        "force_during_clearance_peak_n": float(forces[conflict, 1].max()) if conflict.any() else 0.0,
        "ankle_fk_error_peak_m": float(errors.max()) if len(errors) else None,
        "loaded_cop_outside_frames": int(np.count_nonzero(loaded & cop_bad)),
        "cop_checked": cop is not None,
        "cop_gating": False,
        "clearance_tolerance_m": 0.002,
        "ankle_fk_tolerance_m": 0.01,
    }


def load_trials(
    dataset: str | Path,
    *,
    mount_m=None,
    pitch_rad: float | None = None,
    speed_m_s: float | None = None,
    height_offset_m: float = 0.0,
    friction_model: str = "elastic_coulomb",
    limit_per_split: int | None = None,
    force_observation: ForceObservation | None = None,
) -> list[Trial]:
    """Load existing or multi-condition datasets without cached tracking plans.

    A member may specify ``profile``, ``shoe_artifact``, ``mount_m``, ``pitch_rad``,
    ``speed_m_s`` and ``height_offset_m``. Paths are dataset-relative; profile/shoe
    may instead come from existing ``shared_assets``. All trials describe one
    runner; optional ``subject_id`` values must agree. Speed is explicit task
    metadata, never inferred from future movement. Existing source files are
    fingerprinted in the output; declared shared-asset hashes are checked.
    ``force_observation`` declares how the GRF targets were processed; it is
    attached to every trial and only changes scoring.
    """
    root = Path(dataset).resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") not in ("peak_hip_stance_dataset_1", "generative_runner_dataset_1"):
        raise ValueError("Unsupported dataset schema")
    members = manifest["members"]
    ids = [member["id"] for member in members]
    if len(ids) != len(set(ids)) or any(member["split"] not in ("train", "eval") for member in members):
        raise ValueError("Members need unique IDs and train/eval splits")
    subjects = {member.get("subject_id", manifest.get("subject_id", "unspecified")) for member in members}
    if len(subjects) != 1:
        raise ValueError("A shared runner fit must contain one subject")
    if limit_per_split is not None and limit_per_split < 1:
        raise ValueError("limit_per_split must be positive")
    assets = manifest.get("shared_assets", {})
    for entry in assets.values():
        if "sha256" in entry and _hash(root / entry["file"]) != entry["sha256"]:
            raise ValueError(f"Shared asset hash mismatch: {entry['file']}")
    trials, shoes, counts = [], {}, {"train": 0, "eval": 0}
    for member in members:
        split = member["split"]
        if limit_per_split is not None and counts[split] >= limit_per_split:
            continue
        reference_path = root / member["reference"]
        reference = reference_data.load(reference_path)
        profile_path = root / (member.get("profile") or assets.get("profile.json", {}).get("file", ""))
        shoe_path = root / (member.get("shoe_artifact") or assets.get("digital_shoe.json", {}).get("file", ""))
        if not profile_path.is_file() or not shoe_path.is_file():
            raise ValueError("Each member needs a profile and shoe artifact, directly or via shared_assets")
        mount = member.get("mount_m", mount_m)
        if mount is None:
            raise ValueError("Supply the fixed ankle/shoe mount; it is not learned from force")
        pitch = member.get("pitch_rad", pitch_rad)
        if pitch is None:
            pitch = float(reference.get("shoe_static_pitch_rad", reference["static_pitch_rad"]))
        speed = member.get("speed_m_s", speed_m_s)
        if speed is None:
            raise ValueError("Supply speed_m_s task metadata per member or --speed")
        offset = float(member.get("height_offset_m", height_offset_m))
        rest = RestOfBody(**manifest.get("rest_of_body", {}))
        chain = chain_from_profile(reference, profile_data.load(profile_path), rest)
        key = (str(shoe_path), tuple(mount), float(pitch), friction_model)
        if key not in shoes:
            shoes[key] = Shoe(shoe_path, mount, pitch, friction_model=friction_model)
        shoe = shoes[key]
        q = coordinates(reference, chain, height_offset_m=offset)
        if len(q) < 4:
            raise ValueError("A trial needs a three-frame prefix and at least one future observation")
        flight_velocity = member.get("flight_velocity_m_s")
        if flight_velocity is not None and member.get("flight_velocity_frame") != 2:
            raise ValueError("flight_velocity_m_s must be given at the three-frame prefix end")
        initial = initialize(reference["time_s"][:3], q[:3], com_velocity_m_s=flight_velocity, chain=chain)
        origin = float(reference["time_s"][2])
        ankle = chain.point(initial.q, 3, np.zeros(2))[0]
        clearance = float(shoe.outline(ankle, chain.angle(initial.q, 3))[:, 1].min())
        qc = compatibility(reference, q, chain, shoe)
        # The relaxed-shoe initialization assumes flight at the prediction origin.
        qc["initial_clearance_m"] = clearance
        qc["passed"] = qc["passed"] and clearance > 0
        trials.append(
            Trial(
                member["id"],
                split,
                chain,
                shoe,
                Task(float(speed)),
                initial,
                reference["time_s"][2:] - origin,
                q[2:].copy(),
                reference["grf_time_s"] - origin,
                reference["grf_target_n"].copy(),
                {
                    "reference": str(reference_path),
                    "reference_sha256": _hash(reference_path),
                    "profile": str(profile_path),
                    "profile_sha256": _hash(profile_path),
                    "shoe_artifact": str(shoe_path),
                    "shoe_sha256": _hash(shoe_path),
                    "manifest_sha256": _hash(manifest_path),
                    "subject_id": next(iter(subjects)),
                    "mount_m": list(mount),
                    "pitch_rad": pitch,
                    "height_offset_m": offset,
                    "speed_m_s": speed,
                    "initialization_prefix_frames": 3,
                    "prediction_origin_s": origin,
                    "initialization": (
                        "backward quadratic positions; model COM velocity from preceding-stride plate force"
                        if flight_velocity is not None
                        else "backward quadratic positions only"
                    )
                    + "; fixed kinematic phase; zero torque/load memory",
                    "target_processing": (
                        "hip/knee re-solved to measured hip and ankle centers when available, foot angle kept; "
                        "no Hogan filter or GRF COM correction"
                    ),
                    "rest_of_body": asdict(rest),
                    "friction_model": friction_model,
                    "force_observation": None if force_observation is None else asdict(force_observation),
                    "compatibility": qc,
                },
                force_observation,
            )
        )
        counts[split] += 1
    return trials


def predict(runner: Runner, trial: Trial, config: RolloutConfig, *, device: str = "cpu") -> tuple[dict, dict]:
    """Cross the rollout boundary without passing any observation arrays."""
    if device != "cpu":
        return predict_many([runner], [trial], config, device=device)[0][0]
    return simulate(
        runner, trial.chain, trial.shoe, trial.initial, trial.task, duration_s=trial.duration_s, config=config
    )


def predict_many(models: list[Runner], trials: list[Trial], config: RolloutConfig, *, device: str) -> list:
    """Predict concurrent candidates, streaming trial blocks to bound trace memory."""
    if device == "cpu":
        return [[predict(model, trial, config) for trial in trials] for model in models]
    from .gpu_runner import GpuBatch  # noqa: PLC0415 - keep CPU inspection independent of CUDA modules

    result = [[] for _ in models]
    block = max(1, 128 // len(models))
    for start in range(0, len(trials), block):
        group = trials[start : start + block]
        batch = GpuBatch(
            [t.chain for t in group],
            [t.shoe for t in group],
            [t.initial for t in group],
            [t.task for t in group],
            [t.duration_s for t in group],
            candidates=len(models),
            config=config,
            device=device,
        )
        values = batch.evaluate(models)
        for candidate, row in enumerate(values):
            result[candidate].extend(row)
        del batch
    return result


def scenario(trial: Trial, config: RolloutConfig) -> dict:
    """Export a predictive scenario without measurements or fitting metadata."""
    return {
        "schema": "generative_runner_scenario_1",
        "chain": {
            name: getattr(trial.chain, name).tolist()
            for name in ("lengths_m", "endpoint_local_m", "masses_kg", "com_local_m", "inertias_kg_m2")
        },
        "shoe": {
            "artifact": trial.provenance["shoe_artifact"],
            "sha256": trial.provenance["shoe_sha256"],
            "mount_m": trial.provenance["mount_m"],
            "pitch_rad": trial.provenance["pitch_rad"],
            "friction_model": trial.provenance["friction_model"],
        },
        "initial": {
            "q": trial.initial.q.tolist(),
            "v": trial.initial.v.tolist(),
            "phase_rad": trial.initial.phase_rad,
            "normal_load_bw": trial.initial.normal_load_bw,
            "torque_nm": trial.initial.torque_nm.tolist(),
        },
        "task": asdict(trial.task),
        "duration_s": trial.duration_s,
        "config": asdict(config),
    }


def score(trace: dict, summary: dict, trial: Trial, runner: Runner) -> dict:
    """Compare a completed prediction to native-clock targets, entirely offline.

    Failed predictions receive an explicit unfinished-window penalty. Selection
    additionally ranks failure count before numeric loss, so early failure cannot
    win by avoiding difficult late samples. Physical metrics remain diagnostics,
    not physiological acceptance claims. With a trial ``force_observation``, the
    GRF, peak, impulse, and contact terms use the observed simulated force, and
    the unobservable vibration (physical minus observed force) adds its weighted
    mean square; the returned summary fields keep the physical contact force.
    """
    time = trace["time_s"]
    observed = (trial.time_s > 0) & (trial.time_s <= time[-1] + 1e-12)
    tracking = np.zeros(6)
    if observed.any():
        simulated = np.column_stack([np.interp(trial.time_s[observed], time, trace["state"][:, c]) for c in range(6)])
        tracking = np.sqrt(np.mean((simulated - trial.q[observed]) ** 2, axis=0))
    grf = observed_grf(trace, trial)
    force = np.zeros(2)
    vibration = np.zeros(2)
    if len(grf):
        measured = np.column_stack([np.interp(time[:-1], trial.force_time_s, trial.grf_n[:, c]) for c in range(2)])
        force = np.sqrt(np.mean((grf - measured) ** 2, axis=0))
        vibration = np.sqrt(np.mean((trace["grf_n"] - grf) ** 2, axis=0))
    if trial.force_observation is None:
        peak, simulated_impulse, contact_duration = (
            summary["peak_grf_n"][1],
            np.asarray(summary["grf_impulse_ns"]),
            summary["contact_duration_s"],
        )
    else:
        dt = summary["dt_s"]
        peak = float(grf[:, 1].max()) if len(grf) else 0.0
        simulated_impulse = grf.sum(0) * dt
        contact_duration = float(np.count_nonzero(grf[:, 1] > summary["contact_threshold_n"]) * dt)
    # Integrate the target on a horizon-clipped native clock, retaining endpoints.
    interior = (trial.force_time_s > 0) & (trial.force_time_s < trial.duration_s)
    clock = np.concatenate(([0.0], trial.force_time_s[interior], [trial.duration_s]))
    measured = np.column_stack([np.interp(clock, trial.force_time_s, trial.grf_n[:, c]) for c in range(2)])
    impulse = np.sum(0.5 * (measured[1:] + measured[:-1]) * np.diff(clock)[:, None], axis=0)
    peak_error = float(peak - measured[:, 1].max())
    impulse_error = simulated_impulse - impulse
    contact = measured[:, 1] > summary["contact_threshold_n"]
    contact_time = float(np.sum(np.diff(clock) * contact[:-1]))
    contact_error = contact_duration - contact_time
    effort = 0.0
    if len(trace["load"]):
        effort = float(np.mean((trace["load"][:, 3:] / runner.bounds.torque_max_nm) ** 2))
    loss = (
        float(np.mean((tracking[:2] / 0.02) ** 2))
        + float(np.mean((tracking[2:] / 0.05) ** 2))
        + float(np.mean((force / 100.0) ** 2))
        + (peak_error / 100.0) ** 2
        + float(np.mean((impulse_error / 20.0) ** 2))
        + (contact_error / 0.02) ** 2
        + 0.01 * effort
    )
    if trial.force_observation is not None:
        loss += float(np.mean((vibration_weight(trial) * vibration / 100.0) ** 2))
    if summary["status"] != "completed":
        loss += 1000.0 * (2.0 - time[-1] / trial.duration_s)
    observation = {} if trial.force_observation is None else {"grf_vibration_rmse_n": vibration.tolist()}
    return {
        "loss": loss,
        "tracking_rmse": tracking.tolist(),
        "grf_rmse_n": force.tolist(),
        **observation,
        "peak_fz_error_n": peak_error,
        "impulse_error_ns": impulse_error.tolist(),
        "contact_duration_error_s": contact_error,
        "id": trial.id,
        "split": trial.split,
        **summary,
    }


def evaluate(runner: Runner, trials: list[Trial], config: RolloutConfig, *, device: str = "cpu") -> dict:
    """Evaluate independent free predictions with a frozen shared model."""
    if not trials:
        raise ValueError("Evaluation needs at least one trial")
    rows = []
    for trial, (trace, summary) in zip(trials, predict_many([runner], trials, config, device=device)[0], strict=True):
        rows.append(score(trace, summary, trial, runner))
    return {
        "mean_loss": float(np.mean([row["loss"] for row in rows])),
        "failed": sum(row["status"] != "completed" for row in rows),
        "trials": rows,
    }


class Parameterization:
    """Fit shared impedance weights, stride frequency, and torque response time.

    Task-speed weights and cadence-speed gain are frozen unless training data
    contain multiple distinct speeds. There are no stance-specific coefficients.
    """

    def __init__(self, baseline: Runner, speeds):
        speeds = np.asarray(speeds, dtype=float)
        if speeds.ndim != 1 or speeds.size == 0 or not np.isfinite(speeds).all() or np.any(speeds < 0):
            raise ValueError("Training speeds must be finite and nonempty")
        self.baseline = baseline
        if not 0.5 <= baseline.frequency_hz <= 4.0 or not 0.005 <= baseline.response_time_s <= 0.15:
            raise ValueError("Identification requires frequency in [0.5, 4] Hz and response time in [0.005, 0.15] s")
        if not -1 <= baseline.cadence_speed_gain <= 1:
            raise ValueError("Identification requires cadence_speed_gain in [-1, 1]")
        self.variable_speed = bool(np.ptp(speeds) > 1e-6)
        mask = np.ones(baseline.weights.shape, dtype=bool)
        if not self.variable_speed:
            mask[:, :, -1] = False
        self.indices = np.flatnonzero(mask.ravel())
        self.size = len(self.indices) + 2 + int(self.variable_speed)

    def model(self, offsets) -> Runner:
        offsets = np.asarray(offsets, dtype=float)
        if offsets.shape != (self.size,) or not np.isfinite(offsets).all():
            raise ValueError("Parameter offsets have incorrect shape or nonfinite values")
        n = len(self.indices)
        data = self.baseline.to_dict()
        weights = self.baseline.weights.copy().ravel()
        weights[self.indices] += offsets[:n]
        data["weights"] = weights.reshape(self.baseline.weights.shape).tolist()
        data["frequency_hz"] = float(
            np.clip(self.baseline.frequency_hz * np.exp(np.clip(offsets[n], -20, 20)), 0.5, 4.0)
        )
        data["response_time_s"] = float(
            np.clip(self.baseline.response_time_s * np.exp(np.clip(offsets[n + 1], -20, 20)), 0.005, 0.15)
        )
        if self.variable_speed:
            data["cadence_speed_gain"] = float(np.clip(self.baseline.cadence_speed_gain + offsets[-1], -1.0, 1.0))
        return Runner.from_dict(data)


def force_observation(command: dict) -> ForceObservation | None:
    """Return the force observation recorded in parsed ``identify`` arguments, if any.

    Older run summaries without the setting compare the raw simulated force.
    """
    if command.get("force_filter_hz") is None:
        return None
    return ForceObservation(
        command["force_filter_hz"], command.get("force_gate_n", 1.0), command.get("force_vibration_weight", 1.0)
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("inspect", "fit", "evaluate"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, help="Frozen model for evaluate, or initialization for fit")
    parser.add_argument("--mount", type=float, nargs=3)
    parser.add_argument("--pitch", type=float)
    parser.add_argument("--speed", type=float, help="Known task speed [m/s]; members may override")
    parser.add_argument("--height-offset", type=float, default=0.0)
    parser.add_argument(
        "--friction-model",
        choices=("elastic_coulomb", "column_maxwell", "maxwell", "legacy"),
        default="elastic_coulomb",
    )
    parser.add_argument("--dt", type=float, default=1.25e-4)
    parser.add_argument(
        "--compression-limit",
        type=float,
        default=RolloutConfig.compression_limit,
        help="Driven-shoe compression fraction that ends a rollout as failed",
    )
    parser.add_argument("--device", default="cuda:0", help="CUDA backend by default; cpu selects the reference")
    parser.add_argument("--limit-per-split", type=int, help="Explicit small subset for implementation checks")
    parser.add_argument("--method", choices=("lm",), default="lm", help="Levenberg-Marquardt fit")
    parser.add_argument("--iterations", type=int, default=15, help="LM iterations")
    parser.add_argument("--chunk", type=int, default=128, help="LM candidates per batched GPU rollout")
    parser.add_argument("--lm-step", type=float, default=0.01, help="LM finite-difference step in offset units")
    parser.add_argument("--lm-damping", type=float, default=1e-2, help="Initial LM damping")
    parser.add_argument("--lm-bound", type=float, default=1.5, help="LM parameter offset bound")
    parser.add_argument("--lm-regularization", type=float, default=0.01, help="LM offset regularization")
    parser.add_argument("--lm-objective", choices=("sample", "score"), default="sample", help="LM residual objective")
    parser.add_argument("--lm-tolerance", type=float, default=1e-4, help="Relative improvement stopping tolerance")
    parser.add_argument("--central", action="store_true", help="LM central-difference Jacobian")
    parser.add_argument(
        "--exact-jacobian",
        action="store_true",
        help="Integrate LM finite differences with exact shoe kernels instead of fast math on CUDA",
    )
    parser.add_argument(
        "--intrinsic-damping",
        type=float,
        nargs=3,
        help="Fixed lag-free hip, knee, ankle damping [N m s/rad] set on the initial model for fit",
    )
    parser.add_argument(
        "--immediate-damping",
        action="store_true",
        help="Apply the scheduled impedance damping without the torque response lag (set on the initial model for fit)",
    )
    parser.add_argument(
        "--force-filter-hz",
        type=float,
        help="Score simulated GRF through the target's zero-lag two-pass Butterworth cutoff [Hz] (F01 exports: 20)",
    )
    parser.add_argument(
        "--force-gate-n",
        type=float,
        default=1.0,
        help="With --force-filter-hz, zero observed GRF where filtered Fz is below this force [N]",
    )
    parser.add_argument(
        "--force-vibration-weight",
        type=float,
        default=1.0,
        help="With --force-filter-hz, weight of simulated GRF content above the target bandwidth (0 ignores it)",
    )
    parser.add_argument(
        "--allow-incompatible",
        action="store_true",
        help="Permit diagnostic fitting of incompatible inputs; never a validation claim",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Inspect, identify, or evaluate the shared generative runner."""
    args = _parser().parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if args.command == "evaluate" and args.model is None:
        raise ValueError("evaluate requires --model")
    trials = load_trials(
        args.dataset,
        mount_m=args.mount,
        pitch_rad=args.pitch,
        speed_m_s=args.speed,
        height_offset_m=args.height_offset,
        friction_model=args.friction_model,
        limit_per_split=args.limit_per_split,
        force_observation=force_observation(vars(args)),
    )
    if not trials:
        raise ValueError("Dataset is empty")
    cfg = RolloutConfig(dt_s=args.dt, compression_limit=args.compression_limit)
    report = {
        "validated": False,
        "trials": [{"id": trial.id, "split": trial.split, **trial.provenance} for trial in trials],
    }
    if args.command == "fit":
        training_speeds = [trial.task.speed_m_s for trial in trials if trial.split == "train"]
        if not training_speeds:
            raise ValueError("Identification requires training trials")
        baseline = (
            Runner.load(args.model) if args.model else Runner.seed(reference_speed_m_s=float(np.mean(training_speeds)))
        )
        if args.intrinsic_damping is not None:
            baseline = Runner.from_dict({**baseline.to_dict(), "intrinsic_damping_nms_rad": args.intrinsic_damping})
        if args.immediate_damping:
            baseline = Runner.from_dict({**baseline.to_dict(), "immediate_damping": True})
        from .least_squares import LMConfig, fit_lm  # noqa: PLC0415 - least_squares imports this module

        model, report = fit_lm(
            baseline,
            trials,
            config=cfg,
            search=LMConfig(
                iterations=args.iterations,
                step=args.lm_step,
                chunk=args.chunk,
                damping=args.lm_damping,
                bound=args.lm_bound,
                regularization=args.lm_regularization,
                objective=args.lm_objective,
                tolerance=args.lm_tolerance,
                central=args.central,
                fast_jacobian=not args.exact_jacobian,
            ),
            allow_incompatible=args.allow_incompatible,
            device=args.device,
        )
    elif args.command == "evaluate":
        model = Runner.load(args.model)
        report.update(
            rollout=asdict(cfg),
            splits={
                name: evaluate(model, subset, cfg, device=args.device)
                for name in ("train", "eval")
                if (subset := [trial for trial in trials if trial.split == name])
            },
        )
    report["command"] = vars(args) | {
        key: str(value.resolve()) for key, value in vars(args).items() if isinstance(value, Path)
    }
    project_root = Path(__file__).resolve().parents[1]
    sources = [
        Path(__file__),
        Path(__file__).with_name("runner.py"),
        Path(__file__).with_name("generate.py"),
        Path(__file__).with_name("mechanics.py"),
        Path(__file__).with_name("gpu_runner.py"),
        Path(__file__).with_name("gpu_residuals.py"),
        Path(__file__).with_name("gpu_shoe.py"),
        Path(__file__).with_name("gpu_mechanics.py"),
        Path(__file__).with_name("least_squares.py"),
        project_root / "cartesian/shoe.py",
        project_root / "cartesian/data.py",
        project_root / "cartesian/profile.py",
        project_root / "cartesian/gpu/foundation.py",
    ]
    sources.extend((project_root.parent / "digital_shoe").glob("*.py"))
    report["source_sha256"] = {str(path.relative_to(project_root.parent)): _hash(path) for path in sources}
    if args.model is not None:
        report["input_model_sha256"] = _hash(args.model)
    args.output.mkdir(parents=True)
    if args.command != "inspect":
        model.save(args.output / "runner.json")
        # Save every prediction without letting visualization re-run a different model.
        predictions = predict_many([model], trials, cfg, device=args.device)[0]
        for index, (trial, (trace, _)) in enumerate(zip(trials, predictions, strict=True)):
            np.savez_compressed(args.output / f"trace_{index:03d}.npz", **trace)
            report["trials"][index]["trace"] = f"trace_{index:03d}.npz"
            name = f"scenario_{index:03d}.json"
            (args.output / name).write_text(
                json.dumps(scenario(trial, cfg), indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
            report["trials"][index]["scenario"] = name
    (args.output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    bad = sum(not trial.provenance["compatibility"]["passed"] for trial in trials)
    print(f"{args.command}: {len(trials)} trials; {bad} incompatible; not validated; wrote {args.output}")
    if args.command == "fit":
        from .fit_report import write_report  # noqa: PLC0415 - report imports identification helpers

        print(f"report: {write_report(args.output, device=args.device)}")


if __name__ == "__main__":
    main()
