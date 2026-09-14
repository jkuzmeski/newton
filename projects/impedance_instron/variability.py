# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Measure the subject's own stride-to-stride spread of the stance the rig must reproduce.

Every tolerance the task objective applies to a tracked channel should be a measurement of
this subject, not a chosen number. This module exports one number per channel that says how
far *the subject himself* moves between two strides of the same steady run, so a tracking
error can be divided by it and read as "inside or outside his own repeatability".

The inputs are sealed stance profiles written by
:mod:`projects.impedance_instron.profile`, one per stance, all from the same side of the same
steady-belt block. Each stance is cut at its own contact, time-normalised to per cent of that
contact, and compared sample by sample against the other stances. Nothing here is hardcoded:
the normalisers, the curves, the stance count and the window all come from the exported
stances.

Build the artifact with::

    uv run -m projects.impedance_instron.variability \
        --stance outputs/impedance_instron/variability/left_w79_99_s00.json \
        --stance ... \
        --output outputs/impedance_instron/stride_variability.json

and read it back anywhere with :func:`load_variability`, which needs neither C3D nor Warp.

What the numbers do and do not mean:

* Vertical and fore-aft force are measured platform channels. Their spread is the subject's
  own spread plus platform noise, and it is strongly shaped: large at the impact transient,
  small in the middle of stance. The pointwise curve keeps that shape; it is never smoothed.
* The COM is the force-integrated surrogate of the profile, started at an arbitrary height
  with an arbitrary vertical velocity. Absolute COM height and absolute COM vertical velocity
  carry no information. Only the change since touchdown is comparable across stances, which is
  the datum this module uses for both COM channels, and even that shares one common
  integration convention rather than being an independent measurement.
* The pelvis channels are different in kind: an INDEPENDENT optical measurement, not a second
  reading of the platform. Pelvis height is an absolute height above the same laboratory plane the
  platforms define, and pelvis velocities are central differences of measured positions, so none of
  the three arbitrary initial conditions of the COM surrogate enters them. They are only available
  when every stance was exported with ``--pelvis-markers``. The pelvis is NOT the whole-body centre
  of mass: :func:`compare_com_surrogates` measures how far apart the two are, stance by stance,
  after removing the one straight line the force integral cannot determine.
* Foot pitch is a Kabsch fit of a three-marker heel triangle against a static trial. Its zero
  is the assumed flat-standing reference of that static trial, identical for every stance, so
  absolute pitch is comparable; the per-stance fit residual is exported so a stance with a
  worse triangle fit is visible instead of averaged in silently.
* Time-normalising deliberately removes the stride-to-stride difference in stance *duration*.
  Duration is a separate scalar with its own measured spread, reported under ``timing``; it
  must stay a separate term in any cost that uses these normalisers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from projects.impedance_instron.profile import load_profile

SCHEMA = "impedance_stride_variability_1"
DEFAULT_OUTPUT = "outputs/impedance_instron/stride_variability.json"

# Share of body weight above which the shoe counts as loaded. This must stay equal to
# ``projects.impedance_instron.env.CONTACT_FORCE_FRACTION``, the single constant the environment's
# touchdown detector and stance gate share. It is restated here instead of imported because ``env``
# pulls in Warp and this module stays pure NumPy; ``newton/tests/test_impedance_variability.py``
# pins this value against env.py so the two cannot drift.
CONTACT_FORCE_FRACTION = 0.02

# Percent-of-contact grid: 0 to 100 inclusive, one sample per point of stance.
NORMALIZED_SAMPLES = 101

# The measured platform baseline swings by tens of newtons while the foot is airborne, which is
# more than CONTACT_FORCE_FRACTION of body weight, so the threshold alone would mark flight noise
# as contact. Contact is therefore the one contiguous loaded run that contains the stance peak.
CONTACT_DEFINITION = (
    "contiguous run of reference_fz_n above CONTACT_FORCE_FRACTION of body weight containing the "
    "stance peak; the same threshold as env.py, restricted to one run because the padded flight "
    "baseline of the measured platform crosses that threshold as noise"
)

SCALAR_KIND = "rms_over_contact_of_the_across_stride_pointwise_sd"
SCALAR_RATIONALE = (
    "A cost term of the form mean over stance of ((run - measured)/s)^2 has expected value 1 for a "
    "second stance of this same subject exactly when s is the root-mean-square of the pointwise "
    "across-stride sd, so the RMS scalar is the normaliser; the pointwise sd is exported beside it "
    "for inspection and would divide by a near-zero midstance spread if used directly"
)


class Channel(NamedTuple):
    """One tracked channel and how its stances may be compared."""

    name: str
    """Channel name used as the key of the exported normalisers."""

    key: str
    """Array name inside the stance profile."""

    unit: str
    """SI unit of the channel and of its normaliser."""

    datum: str
    """How stances are made comparable: ``"absolute"``, ``"touchdown"`` or ``"touchdown_tangent"``.

    ``"touchdown"`` compares the change since touchdown. ``"touchdown_tangent"`` also removes the
    straight line the arbitrary touchdown velocity of the COM surrogate would draw, which leaves
    only the double integral of measured force since touchdown.
    """

    tangent_key: str | None
    """Velocity array whose touchdown value defines the removed tangent, for that datum only."""

    comparability: str
    """Why that datum is the honest one for this channel."""


CHANNELS: tuple[Channel, ...] = (
    Channel(
        "vertical_force_n",
        "reference_fz_n",
        "N",
        "absolute",
        None,
        "measured calibrated platform force; absolute values are comparable across stances",
    ),
    Channel(
        "fore_aft_force_n",
        "reference_fx_n",
        "N",
        "absolute",
        None,
        "measured calibrated platform force; absolute values are comparable across stances",
    ),
    Channel(
        "com_height_m",
        "com_z_m",
        "m",
        "touchdown_tangent",
        "reference_com_vz_m_s",
        "force-integrated surrogate started at an arbitrary height AND an arbitrary vertical "
        "velocity; forces identify neither, so only the height change since touchdown with the "
        "touchdown-velocity line removed is measured, and it is not measured COM",
    ),
    Channel(
        "com_vertical_velocity_m_s",
        "reference_com_vz_m_s",
        "m/s",
        "touchdown",
        None,
        "force-integrated surrogate started at an arbitrary vertical velocity; the change since "
        "touchdown is the measured vertical impulse per unit mass and is comparable",
    ),
    Channel(
        "foot_pitch_rad",
        "pitch_rad",
        "rad",
        "absolute",
        None,
        "heel-triangle Kabsch angle against one static trial; the zero is an assumed flat-standing "
        "reference shared by every stance, so absolute pitch is comparable between stances",
    ),
)

# Optional channels, present only when every stance carries the measured pelvis marker block of
# ``impedance_stance_3``. Unlike the COM surrogate these are optical measurements with no arbitrary
# initial condition: the height is above the same laboratory plane the platforms sit on, and the
# velocities are central differences of measured positions on the original 100 Hz optical clock.
PELVIS_CHANNELS: tuple[Channel, ...] = (
    Channel(
        "pelvis_height_m",
        "pelvis_centroid_z_m",
        "m",
        "absolute",
        None,
        "measured optical height of the pelvis marker centroid above the platform plane; absolute "
        "values are comparable across stances, but the pelvis is not the whole-body centre of mass",
    ),
    Channel(
        "pelvis_vertical_velocity_m_s",
        "pelvis_centroid_vz_m_s",
        "m/s",
        "absolute",
        None,
        "central difference of the measured centroid height on the optical clock; no initial "
        "condition is assumed, so absolute values are comparable",
    ),
    Channel(
        "pelvis_fore_aft_velocity_m_s",
        "pelvis_centroid_vx_belt_m_s",
        "m/s",
        "absolute",
        None,
        "measured fore-aft centroid velocity plus the commanded belt speed, i.e. travel relative to "
        "the belt; absolute fore-aft POSITION on a treadmill is not meaningful, and the constant "
        "belt term is a D-Flow command, so the level inherits that command's accuracy while the "
        "within-stance shape and the across-stride spread do not",
    ),
)

CHANNEL_NAMES: tuple[str, ...] = tuple(channel.name for channel in CHANNELS)
PELVIS_CHANNEL_NAMES: tuple[str, ...] = tuple(channel.name for channel in PELVIS_CHANNELS)


def _canonical(value: dict) -> bytes:
    """Serialise an artifact body for hashing, rejecting non-finite numbers."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _statistics(values: np.ndarray) -> dict[str, float]:
    """Summarise a per-stance scalar: count, mean, sample sd, coefficient of variation and range."""
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        raise ValueError("a spread needs at least two stances")
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    return {
        "count": int(values.size),
        "mean": mean,
        "sd": sd,
        "cv": sd / abs(mean) if mean != 0.0 else math.inf,
        "min": float(values.min()),
        "max": float(values.max()),
    }


def contact_bounds(force_n: np.ndarray, threshold_n: float) -> tuple[int, int]:
    """Find the loaded sample range of one padded stance.

    Args:
        force_n: Vertical shoe force of the padded profile [N].
        threshold_n: Loading threshold [N], ``CONTACT_FORCE_FRACTION`` of body weight.

    Returns:
        First and last loaded sample index, inclusive. See :data:`CONTACT_DEFINITION`.

    Raises:
        ValueError: If the loaded run reaches either end of the padded profile, which means the
            export padding is too short to contain this stance's own threshold crossings.
    """
    force_n = np.asarray(force_n, dtype=float)
    if force_n.ndim != 1 or force_n.size < 3 or not math.isfinite(threshold_n):
        raise ValueError("contact detection needs a finite threshold and a one-dimensional force")
    loaded = force_n > threshold_n
    peak = int(np.argmax(force_n))
    if not loaded[peak]:
        raise ValueError("stance peak is below the contact threshold")
    start = peak
    while start > 0 and loaded[start - 1]:
        start -= 1
    end = peak
    while end < force_n.size - 1 and loaded[end + 1]:
        end += 1
    if start == 0 or end == force_n.size - 1:
        raise ValueError("contact reaches the padded profile boundary; re-export with more padding")
    return start, end


def percent_resample(time_s: np.ndarray, values: np.ndarray, start_s: float, end_s: float, samples: int) -> np.ndarray:
    """Resample one channel onto per cent of its own contact.

    Args:
        time_s: Profile clock [s].
        values: Channel samples on that clock.
        start_s: Touchdown time [s].
        end_s: Toe-off time [s].
        samples: Grid size, normally :data:`NORMALIZED_SAMPLES`.

    Returns:
        Linearly interpolated values at ``samples`` points from touchdown to toe-off.
    """
    if end_s <= start_s or samples < 3:
        raise ValueError("contact must have positive duration and at least three grid points")
    grid = start_s + (end_s - start_s) * np.linspace(0.0, 1.0, samples)
    return np.interp(grid, np.asarray(time_s, dtype=float), np.asarray(values, dtype=float))


def sample_noise(values: np.ndarray) -> float:
    """Estimate the per-sample noise of a smooth, uniformly sampled signal.

    The fourth difference of a locally cubic signal is zero, so what survives it is noise. Its
    variance is 70 times the sample variance for independent samples, which turns the measured
    fourth difference into a noise floor without smoothing the signal itself.

    Args:
        values: Uniformly sampled channel.

    Returns:
        Estimated per-sample standard deviation, in the units of ``values``.
    """
    values = np.asarray(values, dtype=float)
    if values.size < 6:
        raise ValueError("a noise floor needs at least six samples")
    return float(np.sqrt(np.mean(np.diff(values, n=4) ** 2) / 70.0))


def pelvis_series(profile: dict[str, Any]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Derive the measured pelvis channels from a profile's optical marker block.

    The centroid is the unweighted mean of the exported pelvis markers, already verified against the
    stored positions by :func:`projects.impedance_instron.profile.load_profile`. Velocities are
    central differences on the original optical knots, which the exporter surrounds with 0.15 s of
    context at each end so the derivative at touchdown and toe-off is measured, not extrapolated.
    Nothing is smoothed or filtered here.

    Args:
        profile: Loaded ``impedance_stance_3`` profile.

    Returns:
        Channel name to ``(clock, values)``, both on the profile's own relative optical clock.

    Raises:
        ValueError: If the profile carries no pelvis block, or its knots are not uniform.
    """
    reference = profile.get("pelvis_reference")
    if not isinstance(reference, dict):
        raise ValueError("profile carries no measured pelvis marker block")
    knots = np.asarray(reference["knot_time_s"], dtype=float)
    centroid = np.asarray(reference["centroid_m"], dtype=float)
    step = float(np.mean(np.diff(knots)))
    if knots.size < 3 or not np.allclose(np.diff(knots), step, rtol=0, atol=1e-9) or step <= 0.0:
        raise ValueError("pelvis knots must be a uniform optical clock")
    belt_m_s = float(profile["provenance"]["running"]["belt_speed_m_s"])
    return {
        "pelvis_centroid_z_m": (knots, centroid[:, 2]),
        "pelvis_centroid_vz_m_s": (knots, np.gradient(centroid[:, 2], step)),
        "pelvis_centroid_vx_belt_m_s": (knots, np.gradient(centroid[:, 0], step) + belt_m_s),
    }


@dataclass(frozen=True)
class Stance:
    """One exported stance, cut at its own contact and time-normalised."""

    path: str
    """Stance profile the curves were read from."""

    curves: dict[str, np.ndarray]
    """Per-channel curves on the per-cent grid, already referred to each channel's datum."""

    raw: dict[str, np.ndarray]
    """The same curves before the datum is applied, in the profile's own units and frame."""

    contact_s: tuple[float, float]
    """Touchdown and toe-off on the profile's own clock [s]."""

    summary: dict[str, Any]
    """Timing, peak force, pitch fit residual and source attribution of this stance."""


def read_stance(path: str | Path, samples: int = NORMALIZED_SAMPLES) -> Stance:
    """Load, verify and time-normalise one sealed stance profile.

    Args:
        path: Sealed profile written by :mod:`projects.impedance_instron.profile`.
        samples: Per-cent grid size.

    Returns:
        The stance curves and the per-stance numbers that justify including it.

    Raises:
        ValueError: If the profile fails its own verification, or its padding does not contain
            the threshold crossings this module needs.
    """
    path = Path(path)
    profile = load_profile(path)
    mass_kg = float(profile["mass_kg"])
    gravity = float(profile["provenance"]["com_surrogate"]["gravity_m_s2"])
    threshold_n = CONTACT_FORCE_FRACTION * mass_kg * gravity
    time_s = np.asarray(profile["time_s"], dtype=float)
    force_n = np.asarray(profile["reference_fz_n"], dtype=float)
    start, end = contact_bounds(force_n, threshold_n)
    start_s, end_s = float(time_s[start]), float(time_s[end])
    grid_s = start_s + (end_s - start_s) * np.linspace(0.0, 1.0, samples)
    series: dict[str, tuple[np.ndarray, np.ndarray]] = {
        channel.key: (time_s, np.asarray(profile[channel.key], dtype=float)) for channel in CHANNELS
    }
    channels = list(CHANNELS)
    pelvis = profile.get("pelvis_reference")
    if isinstance(pelvis, dict):
        series.update(pelvis_series(profile))
        channels += list(PELVIS_CHANNELS)
    curves: dict[str, np.ndarray] = {}
    raw: dict[str, np.ndarray] = {}
    for channel in channels:
        clock, values = series[channel.key]
        curve = percent_resample(clock, values, start_s, end_s, samples)
        raw[channel.name] = curve
        if channel.datum in ("touchdown", "touchdown_tangent"):
            curve = curve - curve[0]
        if channel.datum == "touchdown_tangent":
            # The surrogate's touchdown velocity is the exporter's arbitrary initial condition
            # carried through the padded flight, not a measurement, so its straight line is removed
            # together with the arbitrary height. What remains is padding-independent.
            tangent_clock, tangent_values = series[channel.tangent_key]
            rate = float(np.interp(start_s, tangent_clock, tangent_values))
            curve = curve - rate * (grid_s - start_s)
        curves[channel.name] = curve
    running = profile["provenance"]["running"]
    origin = float(profile["source_time_s"][0])
    reference = profile.get("pitch_reference")
    fit = {"kind": "none; legacy marker-line pitch carries no rigid fit"}
    if isinstance(reference, dict):
        knots = np.asarray(reference["knot_time_s"], dtype=float)
        rms = np.asarray(reference["fit_rms_m"], dtype=float)
        worst = np.asarray(reference["fit_max_m"], dtype=float)
        inside = (knots >= start_s) & (knots <= end_s)
        fit = {
            "kind": reference["method"],
            "knots_in_contact": int(inside.sum()),
            "mean_rms_m": float(rms[inside].mean()),
            "max_rms_m": float(rms[inside].max()),
            "max_point_error_m": float(worst[inside].max()),
        }
    pelvis_summary = {"present": False}
    if isinstance(pelvis, dict):
        quality = pelvis["quality"]
        knots = np.asarray(pelvis["knot_time_s"], dtype=float)
        inside = (knots >= start_s) & (knots <= end_s)
        height = np.asarray(pelvis["centroid_m"], dtype=float)[:, 2]
        noise = sample_noise(height[inside])
        spacing = float(np.mean(np.diff(knots)))
        pelvis_summary = {
            "present": True,
            "method": pelvis["method"],
            "marker_names": list(pelvis["marker_names"]),
            "centroid_height_noise_m": noise,
            # A central difference divides the difference of two independent samples by two steps.
            "centroid_velocity_noise_m_s": noise * math.sqrt(2.0) / (2.0 * spacing),
            "knots_in_contact": int(
                np.sum((np.asarray(pelvis["knot_time_s"]) >= start_s) & (np.asarray(pelvis["knot_time_s"]) <= end_s))
            ),
            "max_pair_distance_sd_m": float(np.max(quality["sd_m"])),
            "max_pair_distance_range_m": float(np.max(quality["range_m"])),
        }
    summary = {
        "path": str(path),
        "side": profile["side"],
        "window_s": list(running["window_s"]),
        "stance_index": int(profile["provenance"]["reproduction_options"]["stance_index"]),
        "padding_s": float(profile["provenance"]["reproduction_options"]["padding"]),
        "contact_source_s": [origin + start_s, origin + end_s],
        "contact_duration_s": end_s - start_s,
        "event_source_s": list(running["selected_stance_source_s"]),
        "event_duration_s": float(running["stance_duration_s"]),
        "peak_fz_n": float(force_n.max()),
        "peak_fz_percent_of_contact": float(100.0 * (int(np.argmax(force_n)) - start) / (end - start)),
        "belt_speed_m_s": float(running["belt_speed_m_s"]),
        "contact_threshold_n": threshold_n,
        "pitch_fit": fit,
        "pelvis": pelvis_summary,
        "schema_version": profile["schema_version"],
        "profile_sha256": profile["seal"]["content_sha256"],
    }
    return Stance(path=str(path), curves=curves, raw=raw, contact_s=(start_s, end_s), summary=summary)


def _channel_spread(channel: Channel, matrix: np.ndarray, percent: np.ndarray) -> dict[str, Any]:
    """Reduce one channel's stance-by-percent matrix to its pointwise and scalar spread."""
    pointwise = matrix.std(axis=0, ddof=1)
    scalar = float(np.sqrt(np.mean(pointwise**2)))
    if not math.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{channel.name} has no positive measured spread")
    count = matrix.shape[0]
    # Leave-one-out: each stance against the mean of the others, in units of the scalar. A normaliser
    # that is a measurement rather than a choice returns sqrt(count / (count - 1)) here.
    total = matrix.sum(axis=0)
    others = (total - matrix) / (count - 1)
    residual = np.sqrt(np.mean(((matrix - others) / scalar) ** 2, axis=1))
    peaks = matrix[np.arange(count), np.argmax(np.abs(matrix), axis=1)]
    # Half-block scalars: a spread inflated by slow drift over the recording rather than by
    # stride-to-stride variation shows up as two halves that disagree with the whole.
    half = count // 2
    halves = [float(np.sqrt(np.mean(part.std(axis=0, ddof=1) ** 2))) for part in (matrix[:half], matrix[half:])]
    return {
        "unit": channel.unit,
        "datum": channel.datum,
        "comparability": channel.comparability,
        "source_array": channel.key,
        "scalar_sd": scalar,
        "scalar_kind": SCALAR_KIND,
        "scalar_rationale": SCALAR_RATIONALE,
        "pointwise_sd": pointwise.tolist(),
        "pointwise_mean": matrix.mean(axis=0).tolist(),
        "mean_sd": float(pointwise.mean()),
        "median_sd": float(np.median(pointwise)),
        "min_sd": float(pointwise.min()),
        "min_sd_percent": float(percent[int(np.argmin(pointwise))]),
        "max_sd": float(pointwise.max()),
        "max_sd_percent": float(percent[int(np.argmax(pointwise))]),
        "range_of_stance_mean": float(np.ptp(matrix.mean(axis=0))),
        "stance_mean_level": _statistics(matrix.mean(axis=1)),
        "scalar_sd_without_stance_offset": float(
            np.sqrt(np.mean((matrix - matrix.mean(axis=1, keepdims=True)).std(axis=0, ddof=1) ** 2))
        ),
        "scalar_sd_first_half": halves[0],
        "scalar_sd_second_half": halves[1],
        "extreme_value": _statistics(peaks),
        "leave_one_out_normalised_rms": {
            "mean": float(residual.mean()),
            "max": float(residual.max()),
            "expected": float(np.sqrt(count / (count - 1))),
        },
    }


def _rms(values: np.ndarray) -> float:
    """Return the root-mean-square of one curve."""
    return float(np.sqrt(np.mean(np.asarray(values, dtype=float) ** 2)))


def _line_removed(seconds: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Remove the best-fit line and return the residual, the offset [unit] and the slope [unit/s]."""
    slope, offset = np.polyfit(seconds, values, 1)
    return values - (offset + slope * seconds), float(offset), float(slope)


def compare_com_surrogates(stances: list[Stance], percent: np.ndarray) -> dict[str, Any] | None:
    """Compare the measured pelvis height with the force-integrated COM height, stance by stance.

    The force integral determines COM ACCELERATION exactly during single support, because the
    platforms measure every external force on the body, but it cannot determine the height or the
    vertical velocity it started from. Those two unknowns are exactly one straight line in time, so
    the line is fitted to the pelvis-minus-force difference over each contact and removed. What is
    left is measured disagreement: the pelvis moving relative to the true centre of mass, plus
    optical noise, pelvis soft-tissue motion and the 20 Hz force filter.

    Args:
        stances: Time-normalised stances that all carry the pelvis block.
        percent: Per-cent grid the curves are sampled on.

    Returns:
        The per-stance and across-stance comparison, or ``None`` when any stance lacks the pelvis.
    """
    if not stances or not all(stance.summary["pelvis"]["present"] for stance in stances):
        return None
    height_rms, height_span, offsets, slopes = [], [], [], []
    velocity_rms, velocity_span, velocity_offsets = [], [], []
    residuals, pelvis_shape, force_shape = [], [], []
    for stance in stances:
        start, end = stance.contact_s
        seconds = (end - start) * percent / 100.0
        difference = stance.raw["pelvis_height_m"] - stance.raw["com_height_m"]
        residual, offset, slope = _line_removed(seconds, difference)
        height_rms.append(_rms(residual))
        height_span.append(float(np.ptp(residual)))
        offsets.append(offset)
        slopes.append(slope)
        residuals.append(residual)
        velocity = stance.raw["pelvis_vertical_velocity_m_s"] - stance.raw["com_vertical_velocity_m_s"]
        velocity_rms.append(_rms(velocity - velocity.mean()))
        velocity_span.append(float(np.ptp(velocity - velocity.mean())))
        velocity_offsets.append(float(velocity.mean()))
        pelvis_shape.append(_line_removed(seconds, stance.raw["pelvis_height_m"])[0])
        force_shape.append(_line_removed(seconds, stance.raw["com_height_m"])[0])
    residual_matrix = np.stack(residuals)
    pelvis_absolute = np.stack([stance.curves["pelvis_height_m"] for stance in stances])
    force_absolute = np.stack([stance.curves["com_height_m"] for stance in stances])
    pelvis_shape_matrix, force_shape_matrix = np.stack(pelvis_shape), np.stack(force_shape)

    def correlation(first: np.ndarray, second: np.ndarray) -> float:
        """Correlate two channels' per-stance deviations from their own across-stride mean curve."""
        a = first - first.mean(axis=0)
        b = second - second.mean(axis=0)
        scale = math.sqrt(float(np.sum(a**2)) * float(np.sum(b**2)))
        return float(np.sum(a * b) / scale) if scale > 0.0 else math.nan

    return {
        "kind": "measured pelvis marker centroid against the force-integrated COM surrogate",
        "method": "remove the one straight line the force integral cannot determine (its initial "
        "height and initial vertical velocity) from the pelvis-minus-force difference over each "
        "contact; the remaining residual is measured disagreement",
        "height_residual_m": {
            "rms": _statistics(np.array(height_rms)),
            "peak_to_peak": _statistics(np.array(height_span)),
            "removed_offset_m": _statistics(np.array(offsets)),
            "removed_slope_m_s": _statistics(np.array(slopes)),
            "across_stride_mean_curve_m": residual_matrix.mean(axis=0).tolist(),
            "across_stride_sd_curve_m": residual_matrix.std(axis=0, ddof=1).tolist(),
        },
        "vertical_velocity_residual_m_s": {
            "rms": _statistics(np.array(velocity_rms)),
            "peak_to_peak": _statistics(np.array(velocity_span)),
            "removed_offset_m_s": _statistics(np.array(velocity_offsets)),
        },
        "shared_stride_to_stride_information": {
            "kind": "correlation of the two channels' deviations from their own across-stride mean",
            "absolute": correlation(pelvis_absolute, force_absolute),
            "after_line_removal": correlation(pelvis_shape_matrix, force_shape_matrix),
            "note": "1 means the pelvis repeats exactly what the force channel already says about "
            "this stride; 0 means it is new information",
        },
        "measurement_floor": {
            "kind": "fourth-difference noise estimate of the pelvis centroid height inside contact",
            "centroid_height_noise_m": _statistics(
                np.array([stance.summary["pelvis"]["centroid_height_noise_m"] for stance in stances])
            ),
            "centroid_velocity_noise_m_s": _statistics(
                np.array([stance.summary["pelvis"]["centroid_velocity_noise_m_s"] for stance in stances])
            ),
            "note": "a residual near this floor is instrument noise, not a disagreement between the "
            "pelvis and the centre of mass",
        },
        "limitations": "the force integral is the true COM only while every external force is on the "
        "instrumented platforms, which holds inside the exported single-support contact; the pelvis "
        "is a skin-mounted cluster and is not the whole-body COM, so this residual is the SUM of "
        "pelvis-to-COM relative motion, soft-tissue motion and both instruments' noise, and it does "
        "not attribute that sum",
    }


def build_variability(paths: list[str | Path], samples: int = NORMALIZED_SAMPLES) -> dict[str, Any]:
    """Measure the stride-to-stride spread of every channel from exported stances.

    Args:
        paths: Sealed stance profiles, all the same side of the same steady block.
        samples: Per-cent grid size.

    Returns:
        The sealed artifact body, ready for :func:`write_variability`.

    Raises:
        ValueError: If fewer than three stances are given, if the stances disagree on side,
            subject or belt speed, or if any channel has no positive measured spread.
    """
    if len(paths) < 3:
        raise ValueError("a stride-to-stride spread needs at least three stances")
    stances = sorted((read_stance(path, samples) for path in paths), key=lambda s: s.summary["contact_source_s"][0])
    profiles = [json.loads(Path(stance.path).read_text(encoding="utf-8")) for stance in stances]
    sides = {profile["side"] for profile in profiles}
    masses = {round(float(profile["mass_kg"]), 9) for profile in profiles}
    belts = {round(stance.summary["belt_speed_m_s"], 6) for stance in stances}
    sources = {json.dumps(profile["provenance"]["sources"], sort_keys=True) for profile in profiles}
    if len(sides) != 1 or len(masses) != 1 or len(belts) != 1 or len(sources) != 1:
        raise ValueError("all stances must share one side, one subject and one steady belt command")
    starts = np.array([stance.summary["contact_source_s"][0] for stance in stances])
    if np.any(np.diff(starts) <= 0):
        raise ValueError("stances must be distinct and chronological; duplicate exports were given")
    percent = np.linspace(0.0, 100.0, samples)
    measured = list(CHANNELS)
    if all(stance.summary["pelvis"]["present"] for stance in stances):
        measured += list(PELVIS_CHANNELS)
    channels = {}
    for channel in measured:
        matrix = np.stack([stance.curves[channel.name] for stance in stances])
        channels[channel.name] = _channel_spread(channel, matrix, percent)
    mass_kg = float(profiles[0]["mass_kg"])
    gravity = float(profiles[0]["provenance"]["com_surrogate"]["gravity_m_s2"])
    durations = np.array([stance.summary["contact_duration_s"] for stance in stances])
    events = np.array([stance.summary["event_duration_s"] for stance in stances])
    peaks = np.array([stance.summary["peak_fz_n"] for stance in stances])
    windows = sorted({tuple(stance.summary["window_s"]) for stance in stances})
    residuals = np.array([stance.summary["pitch_fit"].get("max_rms_m", math.nan) for stance in stances], dtype=float)
    body = {
        "schema_version": SCHEMA,
        "side": profiles[0]["side"],
        "stance_count": len(stances),
        "windows_s": [list(window) for window in windows],
        "window_span_s": [float(min(w[0] for w in windows)), float(max(w[1] for w in windows))],
        "percent_of_contact": percent.tolist(),
        "subject": {
            "mass_kg": mass_kg,
            "gravity_m_s2": gravity,
            "body_weight_n": mass_kg * gravity,
        },
        "contact": {
            "force_fraction": CONTACT_FORCE_FRACTION,
            "threshold_n": CONTACT_FORCE_FRACTION * mass_kg * gravity,
            "definition": CONTACT_DEFINITION,
            "source_of_truth": "projects.impedance_instron.env.CONTACT_FORCE_FRACTION",
        },
        "timing": {
            "note": "time-normalising removes these differences from every channel curve on purpose; "
            "duration stays a separate measured tolerance",
            "contact_duration_s": _statistics(durations),
            "event_duration_s": _statistics(events),
            "stride_period_s": _statistics(np.diff(starts)) if len(starts) > 2 else None,
            "peak_vertical_force_n": _statistics(peaks),
            "belt_speed_m_s": sorted(belts)[0],
        },
        "stationarity": {
            "kind": "least-squares trend of each per-stance scalar against its own source clock",
            "span_s": float(starts[-1] - starts[0]),
            "contact_duration_trend_s_per_s": float(np.polyfit(starts - starts[0], durations, 1)[0]),
            "peak_vertical_force_trend_n_per_s": float(np.polyfit(starts - starts[0], peaks, 1)[0]),
            "note": "a large trend would mean the block is not one steady condition and the spread "
            "below is partly drift; the per-channel half-block scalars show the same thing per channel",
        },
        "pitch_fit": {
            "kind": "per-stance heel-triangle Kabsch residual inside contact",
            "max_rms_m": _statistics(residuals) if np.all(np.isfinite(residuals)) else None,
            "worst_stance": int(np.argmax(residuals)) if np.all(np.isfinite(residuals)) else None,
        },
        "channels": channels,
        "com_comparison": compare_com_surrogates(stances, percent),
        "stances": [stance.summary for stance in stances],
        "provenance": {
            "sources": profiles[0]["provenance"]["sources"],
            "source_worktree": profiles[0]["provenance"]["source_worktree"],
            "source_commit": profiles[0]["provenance"]["source_commit"],
            "exporter_schema": profiles[0]["schema_version"],
            "selection": "every qualified same-side stance of the listed windows, no fit-score selection",
            "normalized_samples": samples,
            "smoothing": "none; the pointwise sd is reported as measured",
            "limitations": "one subject, one session, one capture shoe; the spread includes platform "
            "noise and the 20 Hz force filter, and COM channels are a force-integrated surrogate, "
            "not measured COM",
        },
    }
    return seal_variability(body)


def seal_variability(body: dict[str, Any]) -> dict[str, Any]:
    """Return the artifact body with a SHA-256 seal over its own canonical content.

    Args:
        body: Artifact body, with or without an existing seal.

    Returns:
        A new object whose ``seal`` matches its content, so :func:`load_variability` accepts it.
    """
    content = {key: value for key, value in body.items() if key != "seal"}
    digest = hashlib.sha256(_canonical(content)).hexdigest()
    return {**content, "seal": {"algorithm": "sha256", "content_sha256": digest}}


@dataclass(frozen=True)
class StrideVariability:
    """The measured stride-to-stride spread, loaded from a sealed artifact."""

    stance_count: int
    """Number of stances the spread was measured from."""

    side: str
    """Body side of every stance."""

    windows_s: list[list[float]]
    """Source-clock classification windows the stances came from [s]."""

    percent_of_contact: np.ndarray
    """Per-cent grid the pointwise curves are sampled on, shape ``[normalized_samples]``."""

    scalars: dict[str, float]
    """Per-channel scalar normaliser, in that channel's own unit."""

    pointwise: dict[str, np.ndarray]
    """Per-channel across-stride sd at each per cent of contact, unsmoothed."""

    means: dict[str, np.ndarray]
    """Per-channel across-stride mean curve, the stance the normalisers describe."""

    timing: dict[str, Any]
    """Measured duration, stride period and peak force spreads, which stay separate scalars."""

    provenance: dict[str, Any]
    """Sources, selection policy and stated limitations of the measurement."""

    body: dict[str, Any]
    """The verified artifact body, for anything the fields above do not expose."""

    def normaliser(self, channel: str) -> float:
        """Return the scalar normaliser of one channel.

        Args:
            channel: Channel name from :data:`CHANNEL_NAMES`.

        Returns:
            The divisor a tracking error of that channel should be expressed in.

        Raises:
            KeyError: If the channel was not measured.
        """
        if channel not in self.scalars:
            raise KeyError(f"no measured normaliser for {channel!r}; measured channels: {sorted(self.scalars)}")
        return self.scalars[channel]


def write_variability(body: dict[str, Any], path: str | Path) -> Path:
    """Write a sealed variability artifact.

    Args:
        body: Result of :func:`build_variability`.
        path: Destination JSON path.

    Returns:
        The written path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(seal_variability(body), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    load_variability(path)
    return path


def load_variability(path: str | Path = DEFAULT_OUTPUT) -> StrideVariability:
    """Load and verify the measured stride-to-stride spread.

    Args:
        path: Sealed artifact written by :func:`write_variability`.

    Returns:
        The measured normalisers and curves.

    Raises:
        FileNotFoundError: If the artifact does not exist. There is no fallback normaliser: a
            missing measurement must stop the caller, not become a guessed tolerance.
        ValueError: If the schema, seal, channels or curve lengths are inconsistent.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"no measured stride variability at {path}: build it with "
            "`uv run -m projects.impedance_instron.variability`; this module never substitutes a "
            "default tolerance for a measurement"
        )
    body = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(body, dict) or body.get("schema_version") != SCHEMA:
        raise ValueError(f"{path} is not a {SCHEMA} artifact")
    seal = body.get("seal")
    content = {key: value for key, value in body.items() if key != "seal"}
    if seal != {"algorithm": "sha256", "content_sha256": hashlib.sha256(_canonical(content)).hexdigest()}:
        raise ValueError("stride variability seal mismatch")
    count = int(body["stance_count"])
    percent = np.asarray(body["percent_of_contact"], dtype=float)
    if count < 3 or percent.size < 3 or len(body["stances"]) != count:
        raise ValueError("artifact must record at least three stances on a per-cent grid")
    channels = body["channels"]
    known = set(CHANNEL_NAMES) | set(PELVIS_CHANNEL_NAMES)
    if not set(CHANNEL_NAMES) <= set(channels) <= known:
        raise ValueError(f"artifact channels {sorted(channels)} are not {sorted(CHANNEL_NAMES)} plus optional pelvis")
    scalars, pointwise, means = {}, {}, {}
    for name, entry in channels.items():
        scalar = float(entry["scalar_sd"])
        curve = np.asarray(entry["pointwise_sd"], dtype=float)
        mean = np.asarray(entry["pointwise_mean"], dtype=float)
        if not math.isfinite(scalar) or scalar <= 0.0:
            raise ValueError(f"{name} normaliser must be positive and finite")
        if curve.shape != percent.shape or mean.shape != percent.shape or not np.all(np.isfinite(curve)):
            raise ValueError(f"{name} curves must be finite and match the per-cent grid")
        scalars[name], pointwise[name], means[name] = scalar, curve, mean
    return StrideVariability(
        stance_count=count,
        side=str(body["side"]),
        windows_s=[list(map(float, window)) for window in body["windows_s"]],
        percent_of_contact=percent,
        scalars=scalars,
        pointwise=pointwise,
        means=means,
        timing=body["timing"],
        provenance=body["provenance"],
        body=body,
    )


def main() -> None:
    """Measure stride-to-stride variability from exported stances and seal it to JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stance", action="append", default=[], type=Path, help="Sealed stance profile; repeatable")
    parser.add_argument("--stance-glob", action="append", default=[], help="Glob of sealed stance profiles")
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--samples", type=int, default=NORMALIZED_SAMPLES)
    args = parser.parse_args()
    paths = list(args.stance)
    for pattern in args.stance_glob:
        paths.extend(sorted(Path().glob(pattern)))
    unique = sorted({Path(path).resolve() for path in paths})
    if not unique:
        parser.error("no stance profiles given")
    body = build_variability(list(unique), samples=args.samples)
    write_variability(body, args.output)
    report = {
        "output": str(args.output),
        "stance_count": body["stance_count"],
        "windows_s": body["windows_s"],
        "normalisers": {name: entry["scalar_sd"] for name, entry in body["channels"].items()},
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
