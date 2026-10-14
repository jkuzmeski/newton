# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Decompose a frozen policy's ankle error into command, following, and actuation terms.

For each foot and gait phase (stance, early swing, late swing, and the 30 ms
before touchdown), the ankle error ``q - q_ref`` splits into

- command error ``q_eq - q_ref``: the policy aims at the wrong angle;
- following error ``q - q_eq``: the impedance does not hold the aim.

The following error is further explained by the commanded stiffness and
damping, the delivered ankle torque against the spring and damper it was asked
for (actuator lag and slew), and the joint rates. Contact-by-contact, touchdown
ankle state (angle, rate, absolute foot pitch) is related to the observed peak
vertical force. No result here feeds back into training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .env import ReferenceLibrary
from .rollout import initial_state

PHASES = ("stance", "early_swing", "late_swing", "pre_touchdown")


def _interp(time_s, values, clock):
    return np.column_stack([np.interp(clock, time_s, values[:, c]) for c in range(values.shape[1])])


def analyze(traces_path: Path, dataset: Path, split: str) -> dict:
    """Return phase-resolved ankle statistics and per-contact touchdown records."""
    traces = dict(np.load(traces_path))
    dt = float(traces["dt_s"])
    library = ReferenceLibrary(dataset, (split,), dt, "cpu")
    if "impedance" not in traces:
        raise ValueError("The traces need commanded impedance (neural policy evaluation)")
    rows = {phase: {s: [] for s in range(2)} for phase in PHASES}
    contacts = []
    for w in range(len(library)):
        reference = library.references[w]
        start = initial_state(library, w)
        steps = int(traces["steps"][w])
        failed = int(traces["failed_step"][w])
        n = steps if failed < 0 else failed
        if n < 10:
            continue
        clock = (np.arange(n) + 1) * dt
        q = traces["state"][1 : n + 1, w]
        v = traces["velocity"][1 : n + 1, w]
        impedance = traces["impedance"][:n, w]
        torque = traces["torque_nm"][:n, w]
        q_ref = _interp(reference["time_s"] - start.origin_s, reference["q"], clock)
        force_time = reference["grf_time_s"] - start.origin_s
        loaded_ref = np.stack([np.interp(clock, force_time, reference["grf_side_n"][s, :, 1]) > 50.0 for s in range(2)])
        grf = traces["grf_n"][:n, w].reshape(n, 2, 2)
        for s in range(2):
            j = 2 + 3 * s  # ankle within the six joints
            c = 3 + j  # ankle coordinate
            q_eq, k, d = impedance[:, j], impedance[:, 6 + j], impedance[:, 12 + j]
            spring = k * (q_eq - q[:, c])
            touchdowns = np.flatnonzero(~loaded_ref[s, :-1] & loaded_ref[s, 1:]) + 1
            liftoffs = np.flatnonzero(loaded_ref[s, :-1] & ~loaded_ref[s, 1:]) + 1
            phase = np.full(n, "", dtype=object)
            phase[loaded_ref[s]] = "stance"
            for lift in liftoffs:
                following = touchdowns[touchdowns > lift]
                end = following[0] if len(following) else n
                middle = (lift + end) // 2
                phase[lift:middle] = "early_swing"
                phase[middle:end] = "late_swing"
                phase[max(lift, end - int(round(0.03 / dt))) : end] = "pre_touchdown"
            foot_ref = q_ref[:, 2] + q_ref[:, 3 + 3 * s : 6 + 3 * s].sum(1)
            foot = q[:, 2] + q[:, 3 + 3 * s : 6 + 3 * s].sum(1)
            for name in PHASES:
                mask = phase == name
                if not mask.any():
                    continue
                rows[name][s].append(
                    {
                        "error": q[mask, c] - q_ref[mask, c],
                        "command": q_eq[mask] - q_ref[mask, c],
                        "following": q[mask, c] - q_eq[mask],
                        "stiffness": k[mask],
                        "damping": d[mask],
                        "torque": torque[mask, j],
                        "spring": spring[mask],
                        "rate": v[mask, c],
                        "rate_ref": _interp(
                            reference["time_s"] - start.origin_s,
                            np.gradient(reference["q"], reference["time_s"], axis=0),
                            clock,
                        )[mask, c],
                        "foot_error": foot[mask] - foot_ref[mask],
                        "knee_error": q[mask, c - 1] - q_ref[mask, c - 1],
                        "q_eq_rate": np.gradient(q_eq, dt)[mask],
                    }
                )
            for td in touchdowns:
                window = slice(td, min(n, td + int(round(0.06 / dt))))
                before = max(0, td - 1)
                contacts.append(
                    {
                        "segment": library.members[w]["id"],
                        "side": "right" if s == 0 else "left",
                        "time_s": float(clock[td]),
                        "peak_fz_n": float(grf[window, s, 1].max()),
                        "ref_peak_fz_n": float(
                            np.interp(clock[window], force_time, reference["grf_side_n"][s, :, 1]).max()
                        ),
                        "ankle_error_deg": float(np.degrees(q[before, c] - q_ref[before, c])),
                        "foot_pitch_error_deg": float(np.degrees(foot[before] - foot_ref[before])),
                        "ankle_rate_rad_s": float(v[before, c]),
                        "ankle_rate_ref_rad_s": float(
                            _interp(
                                reference["time_s"] - start.origin_s,
                                np.gradient(reference["q"], reference["time_s"], axis=0),
                                clock[[before]],
                            )[0, c]
                        ),
                        "q_eq_deg": float(np.degrees(q_eq[before])),
                        "stiffness_nm_rad": float(k[before]),
                        "damping_nms_rad": float(d[before]),
                        "torque_nm": float(torque[before, j]),
                        "sim_touchdown_lead_s": _lead(grf[:, s, 1], loaded_ref[s], td, dt),
                    }
                )
    summary = {}
    for name in PHASES:
        summary[name] = {}
        for s, side in enumerate(("right", "left")):
            if not rows[name][s]:
                continue
            cat = {key: np.concatenate([r[key] for r in rows[name][s]]) for key in rows[name][s][0]}
            summary[name][side] = {
                "samples": len(cat["error"]),
                "ankle_rmse_deg": float(np.degrees(np.sqrt(np.mean(cat["error"] ** 2)))),
                "ankle_bias_deg": float(np.degrees(np.mean(cat["error"]))),
                "command_rmse_deg": float(np.degrees(np.sqrt(np.mean(cat["command"] ** 2)))),
                "command_bias_deg": float(np.degrees(np.mean(cat["command"]))),
                "following_rmse_deg": float(np.degrees(np.sqrt(np.mean(cat["following"] ** 2)))),
                "following_bias_deg": float(np.degrees(np.mean(cat["following"]))),
                "foot_pitch_rmse_deg": float(np.degrees(np.sqrt(np.mean(cat["foot_error"] ** 2)))),
                "knee_rmse_deg": float(np.degrees(np.sqrt(np.mean(cat["knee_error"] ** 2)))),
                "stiffness_median_nm_rad": float(np.median(cat["stiffness"])),
                "stiffness_p10_p90": np.percentile(cat["stiffness"], [10, 90]).tolist(),
                "damping_median_nms_rad": float(np.median(cat["damping"])),
                "torque_rms_nm": float(np.sqrt(np.mean(cat["torque"] ** 2))),
                "spring_rms_nm": float(np.sqrt(np.mean(cat["spring"] ** 2))),
                "torque_lag_rms_nm": float(np.sqrt(np.mean((cat["torque"] - cat["spring"]) ** 2))),
                "rate_rms_rad_s": float(np.sqrt(np.mean(cat["rate"] ** 2))),
                "rate_ref_rms_rad_s": float(np.sqrt(np.mean(cat["rate_ref"] ** 2))),
                "q_eq_rate_rms_rad_s": float(np.sqrt(np.mean(cat["q_eq_rate"] ** 2))),
                "correlation_error_vs_command": float(np.corrcoef(cat["error"], cat["command"])[0, 1]),
                "correlation_error_vs_following": float(np.corrcoef(cat["error"], cat["following"])[0, 1]),
            }
    if contacts:
        peaks = np.array([c["peak_fz_n"] - c["ref_peak_fz_n"] for c in contacts])
        features = {
            name: np.array([c[name] for c in contacts])
            for name in (
                "ankle_error_deg",
                "foot_pitch_error_deg",
                "ankle_rate_rad_s",
                "stiffness_nm_rad",
                "damping_nms_rad",
                "sim_touchdown_lead_s",
            )
        }
        summary["touchdown"] = {
            "contacts": len(contacts),
            "peak_excess_n_mean": float(peaks.mean()),
            "correlation_with_peak_excess": {
                name: float(np.corrcoef(values, peaks)[0, 1]) for name, values in features.items() if np.std(values) > 0
            },
        }
    return {"summary": summary, "contacts": contacts}


def _lead(fz: np.ndarray, loaded_ref: np.ndarray, td: int, dt: float) -> float:
    """Simulated touchdown time minus measured touchdown time [s]; negative lands early."""
    window = slice(max(0, td - int(0.1 / dt)), min(len(fz), td + int(0.1 / dt)))
    loaded = np.flatnonzero(fz[window] > 50.0)
    if not len(loaded):
        return float("nan")
    return float((loaded[0] + window.start - td) * dt)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = analyze(args.traces, args.dataset, args.split)
    print(json.dumps(result["summary"], indent=1))
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
