# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Quality of the ground force the simulator produces inside a control step.

The environment reports the mean of the substep contact forces of each control step
(:meth:`.env3d.BilateralEnv3D._interval_force`), and the reference it is scored against
is a boxcar of a plate channel low-passed at 20 Hz, so nothing that reaches a reward or a
report ever looks inside the step. These functions work on the force at the substep rate
and measure what that mean hides:

``hf20_rms_n`` / ``hf30_rms_n``
    Root-mean-square of the force left after a zero-lag Butterworth low-pass at 20 Hz -
    the cutoff of :class:`..hogan.identify.ForceObservation`, so the same band the targets
    exclude - and at 30 Hz, pooled over stance windows.
``hf30_boxcar_rms_n``
    The same roughness after the one-control-step mean the environment reports: the gap
    between what the runner produces and what anything downstream can see.
``hf30_midstance_rms_n``
    Restricted to the middle of a stance, away from touchdown and lift-off, so the number
    describes steady support rather than the impact transient.
``jerk_rms_n_s3``
    Third derivative of the force in N/s^3, band-limited first so that it measures contact
    roughness rather than the sampling edge.
``peak_raw_n`` / ``peak_reported_n``
    Stance peaks before and after that mean.

Filters are zero-phase throughout, so no metric prefers one side of a transient. The
recordings these are meant for come from
``outputs/bilateral/grf_spike_2026_10_14/smooth_force/roll.py``, whose ``raw_fz`` is
``[substeps, worlds, sides]``; :func:`trace_quality` scores a whole recording.

    uv run --no-sync python -m projects.impedance_instron.bilateral.force_quality rolls/zoh.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, decimate, filtfilt

TOUCHDOWN_N = 50.0
"""Vertical force [N] above which a foot counts as loaded."""

MIN_STANCE_S = 0.06
"""Shorter loaded windows are a clipped toe in swing, not a stance."""

MID_STANCE_EDGE_S = 0.06
"""Trim at both ends of a stance, so mid-stance excludes touchdown and lift-off."""

JERK_BAND_HZ = 200.0
"""Above this band the third derivative measures the solver, not the contact."""

JERK_SAMPLE_HZ = 1000.0
"""Differentiate the band-limited force at this rate rather than at the substep rate."""

NAN = float("nan")


def lowpass(force: np.ndarray, dt_s: float, cutoff_hz: float, order: int = 2) -> np.ndarray:
    """Zero-phase Butterworth low-pass of ``force`` along axis 0."""
    coefficients = butter(order, cutoff_hz / (0.5 / dt_s), btype="low")
    return filtfilt(*coefficients, force, axis=0)


def residue(force: np.ndarray, dt_s: float, cutoff_hz: float) -> np.ndarray:
    """The part of ``force`` above ``cutoff_hz``."""
    return force - lowpass(force, dt_s, cutoff_hz)


def stances(
    force: np.ndarray, dt_s: float, threshold_n: float = TOUCHDOWN_N, min_stance_s: float = MIN_STANCE_S
) -> list[tuple[int, int]]:
    """Frame ranges ``(touchdown, lift-off)`` of every stance of one loaded channel."""
    loaded = (force > threshold_n).astype(np.int8)
    edges = np.diff(np.concatenate(([0], loaded, [0])))
    return [
        (int(start), int(end))
        for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True)
        if (end - start) * dt_s >= min_stance_s
    ]


def reported(force: np.ndarray, substeps: int) -> np.ndarray:
    """The force the environment reports: the mean of each control step's substep forces."""
    length = (len(force) // substeps) * substeps
    return force[:length].reshape(-1, substeps, *force.shape[1:]).mean(axis=1)


def jerk(
    force: np.ndarray, dt_s: float, band_hz: float = JERK_BAND_HZ, sample_hz: float = JERK_SAMPLE_HZ
) -> np.ndarray:
    """Third derivative [N/s^3] of the band-limited force, differentiated near ``sample_hz``."""
    slow = lowpass(force, dt_s, band_hz)
    factor = max(int(round((1.0 / dt_s) / sample_hz)), 1)
    if factor > 1:
        slow = decimate(slow, factor, zero_phase=True, axis=0)
    step = dt_s * factor
    rate = np.gradient(slow, step, axis=0, edge_order=2)
    acceleration = np.gradient(rate, step, axis=0, edge_order=2)
    return np.gradient(acceleration, step, axis=0, edge_order=2)


def _pooled_rms(windows: list[np.ndarray]) -> float:
    """Root mean square over the given windows, each stance weighted once."""
    return float(np.sqrt(np.mean([float(np.mean(window**2)) for window in windows]))) if windows else NAN


def _resampled_windows(frames: np.ndarray, windows: list[tuple[int, int]], factor: float) -> list[np.ndarray]:
    """The same windows on a series sampled ``factor`` times more coarsely than the frames."""
    return [frames[int(start / factor) : min(int(end / factor) + 1, len(frames))] for start, end in windows]


def quality(force: np.ndarray, dt_s: float, substeps: int, windows: list[tuple[int, int]] | None = None) -> dict:
    """Roughness, mid-stance roughness, jerk and stance peaks of one vertical force channel.

    ``force`` is the vertical force at the substep rate, ``substeps`` the number of
    substeps per control step, and ``windows`` the stance frames to score, detected from
    the force when absent. Every RMS is pooled over those windows.
    """
    force = np.asarray(force, dtype=float)
    if force.ndim != 1:
        raise ValueError("quality scores one channel at a time; use trace_quality for a recording")
    windows = stances(force, dt_s) if windows is None else windows
    if not windows:
        return {"stances": 0}
    above20, above30 = residue(force, dt_s, 20.0), residue(force, dt_s, 30.0)
    mean = reported(force, substeps)
    above30_mean = mean - lowpass(mean, dt_s * substeps, 30.0)
    third = jerk(force, dt_s)
    factor = len(force) / len(third)
    edge = int(round(MID_STANCE_EDGE_S / dt_s))
    middle = [(start + edge, end - edge) for start, end in windows if end - start > 2 * edge]
    return {
        "hf20_rms_n": _pooled_rms([above20[start:end] for start, end in windows]),
        "hf30_rms_n": _pooled_rms([above30[start:end] for start, end in windows]),
        "hf30_boxcar_rms_n": _pooled_rms(_resampled_windows(above30_mean, windows, substeps)),
        "hf30_midstance_rms_n": _pooled_rms([above30[start:end] for start, end in middle]),
        "jerk_rms_n_s3": _pooled_rms(_resampled_windows(third, windows, factor)),
        "peak_raw_n": float(np.median([force[start:end].max() for start, end in windows])),
        "peak_reported_n": float(np.median([mean[min(start // substeps, len(mean) - 1)] for start, end in windows])),
        "stances": len(windows),
    }


def trace_quality(
    force: np.ndarray,
    dt_s: float,
    control_dt_s: float,
    alive: np.ndarray | None = None,
    settle_s: float = 1.0,
    min_stances: int = 2,
) -> dict:
    """Score a whole recording: ``force`` of shape ``[substeps, worlds, sides]``.

    Worlds that fell are scored up to the step they failed, and a world and foot must
    complete ``min_stances`` stances to count. The pooled numbers are the root mean square
    over every scored stance of every scored world and foot; the peaks are medians.
    """
    substeps = int(round(control_dt_s / dt_s))
    settle = int(round(settle_s / control_dt_s))
    steps, worlds, sides = force.shape
    scores, peaks_raw, peaks_reported, counted = [], [], [], 0
    for world in range(worlds):
        end = steps
        if alive is not None:
            dead = np.flatnonzero(~alive[settle:, world])
            if len(dead):
                end = settle + int(dead[0])
        if (end - settle) < int(round(1.5 / control_dt_s)):
            continue
        counted += 1
        for side in range(sides):
            trace = force[settle * substeps : end * substeps, world, side]
            found = stances(trace, dt_s)
            if len(found) < min_stances:
                continue
            scores.append(quality(trace, dt_s, substeps, found))
            peaks_raw.append(scores[-1]["peak_raw_n"])
            peaks_reported.append(scores[-1]["peak_reported_n"])
    if not scores:
        return {"worlds_scored": counted, "stances": 0}
    merged = {"worlds_scored": counted, "worlds": worlds, "stances": int(sum(s["stances"] for s in scores))}
    for key in ("hf20_rms_n", "hf30_rms_n", "hf30_boxcar_rms_n", "hf30_midstance_rms_n", "jerk_rms_n_s3"):
        values = [s[key] for s in scores if np.isfinite(s[key])]
        merged[key] = float(np.sqrt(np.mean([value**2 for value in values]))) if values else NAN
    merged["peak_raw_n"] = float(np.median(peaks_raw)) if peaks_raw else NAN
    merged["peak_reported_n"] = float(np.median(peaks_reported)) if peaks_reported else NAN
    return merged


def main(argv: list[str] | None = None) -> None:
    """Score ``raw_fz`` recordings, the way the reports quote them."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("rolls", nargs="+", type=Path, help="Recordings with raw_fz, substep_dt_s and control_dt_s.")
    parser.add_argument("--json", type=Path, help="Write the scores here as well as printing them.")
    args = parser.parse_args(argv)
    result = {}
    for roll in args.rolls:
        with np.load(roll) as archive:
            scores = trace_quality(
                archive["raw_fz"],
                float(archive["substep_dt_s"]),
                float(archive["control_dt_s"]),
                archive["alive"] if "alive" in archive.files else None,
            )
        result[roll.stem] = scores
        print(
            f"{roll.stem:16s} >30 Hz {scores.get('hf30_rms_n', NAN):6.1f} N  mid-stance "
            f"{scores.get('hf30_midstance_rms_n', NAN):6.1f} N  jerk "
            f"{scores.get('jerk_rms_n_s3', NAN) / 1e9:5.2f} GN/s3  raw peak "
            f"{scores.get('peak_raw_n', NAN):6.0f} N  stances {scores.get('stances', 0)}",
            flush=True,
        )
    if args.json:
        args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
