# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the measured stride-to-stride variability: contact cut, per-cent grid, seal and loader."""

import hashlib
import json
import math
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np

from projects.impedance_instron.variability import (
    CHANNEL_NAMES,
    CHANNELS,
    CONTACT_FORCE_FRACTION,
    DEFAULT_OUTPUT,
    NORMALIZED_SAMPLES,
    PELVIS_CHANNEL_NAMES,
    PELVIS_CHANNELS,
    SCHEMA,
    Stance,
    build_variability,
    compare_com_surrogates,
    contact_bounds,
    load_variability,
    pelvis_series,
    percent_resample,
    read_stance,
    sample_noise,
    seal_variability,
    write_variability,
)

GRAVITY_M_S2 = 9.80665
MASS_KG = 80.0
RATE_HZ = 2000.0
PADDING_S = 0.04
LOAD_THRESHOLD_N = 50.0


def _integrate(time_s: np.ndarray, acceleration: np.ndarray, x0: float, v0: float):
    """Trapezoid-integrate twice, exactly as the stance exporter's COM surrogate does."""
    dt = np.diff(time_s)
    velocity = v0 + np.r_[0.0, np.cumsum(0.5 * (acceleration[1:] + acceleration[:-1]) * dt)]
    position = x0 + np.r_[0.0, np.cumsum(0.5 * (velocity[1:] + velocity[:-1]) * dt)]
    return position, velocity


def _digest() -> str:
    """Return the syntactically valid shared source SHA-256 of the synthetic capture."""
    return hashlib.sha256(b"synthetic-capture").hexdigest()


def _write_stance(
    path: Path,
    *,
    index: int,
    peak_n: float,
    pitch_offset_rad: float,
    duration_s: float = 0.30,
    start_s: float = 90.0,
    padding_s: float = PADDING_S,
) -> Path:
    """Write a synthetic sealed ``impedance_stance_1`` profile with a known injected spread.

    The arrays satisfy every physics-bookkeeping rule the exporter's own loader enforces, so the
    test exercises the real verification path rather than a relaxed one.

    Args:
        path: Destination profile.
        index: Stance number, used for the stride clock and the source digests.
        peak_n: Peak vertical force of this stance [N].
        pitch_offset_rad: Constant pitch offset of this stance [rad].
        duration_s: Pulse duration [s].
        start_s: Source-clock time of the first sample [s].
        padding_s: Flight kept at each end of the pulse [s].

    Returns:
        The written path.
    """
    samples = int(round((duration_s + 2.0 * padding_s) * RATE_HZ)) + 1
    time_s = np.arange(samples) / RATE_HZ
    phase = np.clip((time_s - padding_s) / duration_s, 0.0, 1.0)
    pulse = np.where((time_s >= padding_s) & (time_s <= padding_s + duration_s), np.sin(np.pi * phase) ** 2, 0.0)
    fz = peak_n * pulse
    fx = 0.2 * peak_n * np.gradient(pulse, time_s) / max(np.max(np.abs(np.gradient(pulse, time_s))), 1e-9)
    loaded = fz > LOAD_THRESHOLD_N
    zeros = np.zeros(samples)
    platform = np.zeros((samples, 2, 3))
    platform[:, 0, 0] = fx
    platform[:, 0, 2] = fz
    com_x, com_vx = _integrate(time_s, fx / MASS_KG, 0.0, 3.0)
    com_z, com_vz = _integrate(time_s, fz / MASS_KG - GRAVITY_M_S2, 1.0, 0.0)
    body = {
        "schema_version": "impedance_stance_1",
        "coordinate_system": {
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
        },
        "mass_kg": MASS_KG,
        "side": "left",
        "time_s": time_s.tolist(),
        "source_time_s": (time_s + start_s).tolist(),
        "foot_x_m": (3.0 * time_s).tolist(),
        "foot_z_m": (0.05 * pulse).tolist(),
        "pitch_rad": (pitch_offset_rad - 0.2 + 0.4 * phase).tolist(),
        "reference_fx_n": fx.tolist(),
        "reference_fz_n": fz.tolist(),
        "other_fx_n": zeros.tolist(),
        "other_fz_n": zeros.tolist(),
        "unassigned_fx_n": zeros.tolist(),
        "unassigned_fz_n": zeros.tolist(),
        "total_measured_fx_n": fx.tolist(),
        "total_measured_fz_n": fz.tolist(),
        "reference_cop_x_m": [0.0 if flag else None for flag in loaded],
        "com_x_m": com_x.tolist(),
        "com_z_m": com_z.tolist(),
        "reference_com_vx_m_s": com_vx.tolist(),
        "reference_com_vz_m_s": com_vz.tolist(),
        "provenance": {
            "sources": {name: {"path": f"synthetic/{name}", "sha256": _digest()} for name in ("c3d", "exporter")},
            "source_worktree": "synthetic",
            "source_commit": _digest(),
            "running": {
                "classification": "running",
                "selected_side": "left",
                "window_s": [79.0, 99.0],
                "selected_stance_source_s": [start_s + padding_s, start_s + padding_s + duration_s],
                "stance_duration_s": duration_s,
                "cadence_steps_min": 167.0,
                "flight_before_s": 0.05,
                "flight_after_s": 0.05,
                "belt_speed_m_s": 3.0,
            },
            "registration": {
                "virtual_origin_x_m": zeros.tolist(),
                "heel_origin_newton_lab_m": [0.0, 0.0, 0.0],
            },
            "kinematics": {"kind": "measured_marker_heel_to_toe_proxy"},
            "kinetics": {
                "load_threshold_n": LOAD_THRESHOLD_N,
                "platform_channels": {
                    "force_n": platform.tolist(),
                    "moment_about_lab_origin_nm": np.zeros((samples, 2, 3)).tolist(),
                },
            },
            "com_surrogate": {
                "kind": "force_integrated_surrogate_not_measured_com",
                "gravity_m_s2": GRAVITY_M_S2,
                "initial_x_m": 0.0,
                "initial_z_m": 1.0,
                "initial_vx_m_s": 3.0,
                "initial_vz_m_s": 0.0,
            },
            "rights": {"source": "synthetic"},
            "reproduction_options": {"stance_index": index, "padding": padding_s},
        },
    }
    body["seal"] = {
        "algorithm": "sha256",
        "content_sha256": hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        ).hexdigest(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body) + "\n", encoding="utf-8")
    return path


PEAKS_N = (1700.0, 1750.0, 1800.0, 1720.0, 1780.0, 1760.0)
PITCH_OFFSETS_RAD = (0.00, 0.02, -0.02, 0.01, -0.01, 0.03)


def _write_stances(directory: Path) -> list[Path]:
    """Write one synthetic stance per injected amplitude, on a running stride clock."""
    return [
        _write_stance(
            directory / f"stance_{index:02d}.json",
            index=index,
            peak_n=peak,
            pitch_offset_rad=pitch,
            start_s=90.0 + 0.72 * index,
        )
        for index, (peak, pitch) in enumerate(zip(PEAKS_N, PITCH_OFFSETS_RAD, strict=True))
    ]


class ContactDefinitionTest(unittest.TestCase):
    """Cover the one definition of when stance began and the padding it needs."""

    def test_constant_matches_the_environment(self):
        """Check the restated contact fraction equals env.CONTACT_FORCE_FRACTION, its source of truth."""
        source = Path("projects/impedance_instron/legacy/env.py").read_text(encoding="utf-8")
        found = re.search(r"^CONTACT_FORCE_FRACTION\s*=\s*([0-9.eE+-]+)", source, re.M)
        self.assertIsNotNone(found, "env.py no longer defines CONTACT_FORCE_FRACTION")
        self.assertEqual(float(found.group(1)), CONTACT_FORCE_FRACTION)

    def test_airborne_baseline_noise_is_not_contact(self):
        """Check the cut keeps the loaded run around the peak when flight noise crosses the threshold.

        The measured platform baseline swings by more than the threshold while the foot is
        airborne, so a first-crossing rule would start stance in the padding.
        """
        force = np.zeros(400)
        force[20:30] = 30.0
        force[150:250] = 1000.0 * np.sin(np.pi * np.linspace(0.0, 1.0, 100)) ** 2
        force[380:390] = -30.0
        start, end = contact_bounds(force, 16.0)
        self.assertGreater(start, 30)
        self.assertLess(end, 380)
        self.assertTrue(np.all(force[start : end + 1] > 16.0))

    def test_contact_touching_the_padding_boundary_is_rejected(self):
        """Check a stance whose loaded run reaches the exported edge raises instead of being cut short."""
        force = np.linspace(100.0, 1000.0, 200)
        with self.assertRaises(ValueError):
            contact_bounds(force, 16.0)

    def test_unloaded_profile_is_rejected(self):
        """Check a profile that never exceeds the threshold cannot become a stance."""
        with self.assertRaises(ValueError):
            contact_bounds(np.full(100, 5.0), 16.0)


class PercentGridTest(unittest.TestCase):
    """Cover time normalisation to per cent of each stance's own contact."""

    def test_grid_spans_touchdown_to_toe_off(self):
        """Check the resampled curve starts at touchdown and ends at toe-off."""
        time_s = np.linspace(0.0, 1.0, 1001)
        values = 5.0 * time_s
        curve = percent_resample(time_s, values, 0.2, 0.8, NORMALIZED_SAMPLES)
        self.assertEqual(curve.size, NORMALIZED_SAMPLES)
        self.assertAlmostEqual(curve[0], 1.0, places=6)
        self.assertAlmostEqual(curve[-1], 4.0, places=6)

    def test_normalisation_removes_a_duration_difference(self):
        """Check two stances of identical shape but different duration give the same curve."""
        time_s = np.linspace(0.0, 1.0, 2001)
        short = np.sin(np.pi * np.clip(time_s / 0.25, 0.0, 1.0))
        long = np.sin(np.pi * np.clip(time_s / 0.35, 0.0, 1.0))
        first = percent_resample(time_s, short, 0.0, 0.25, NORMALIZED_SAMPLES)
        second = percent_resample(time_s, long, 0.0, 0.35, NORMALIZED_SAMPLES)
        self.assertLess(float(np.max(np.abs(first - second))), 1e-3)

    def test_degenerate_contact_is_rejected(self):
        """Check a non-positive contact duration cannot be normalised."""
        with self.assertRaises(ValueError):
            percent_resample(np.linspace(0.0, 1.0, 10), np.zeros(10), 0.5, 0.5, NORMALIZED_SAMPLES)


class MeasuredSpreadTest(unittest.TestCase):
    """Cover the measurement itself against a known injected stride-to-stride spread."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.paths = _write_stances(Path(self.directory.name))
        self.body = build_variability(self.paths)

    def tearDown(self):
        self.directory.cleanup()

    def test_every_stance_is_read_and_counted(self):
        """Check the artifact counts the stances it was given and records each one."""
        self.assertEqual(self.body["stance_count"], len(self.paths))
        self.assertEqual(len(self.body["stances"]), len(self.paths))
        self.assertEqual({stance["side"] for stance in self.body["stances"]}, {"left"})

    def test_vertical_force_normaliser_recovers_the_injected_amplitude_spread(self):
        """Check the scalar equals the injected peak spread times the shape, within 5 per cent."""
        channel = self.body["channels"]["vertical_force_n"]
        mean_curve = np.asarray(channel["pointwise_mean"], dtype=float)
        relative = float(np.std(PEAKS_N, ddof=1) / np.mean(PEAKS_N))
        expected = relative * float(np.sqrt(np.mean(mean_curve**2)))
        self.assertAlmostEqual(channel["scalar_sd"], expected, delta=0.05 * expected)

    def test_pitch_normaliser_recovers_the_injected_offset_spread(self):
        """Check a constant per-stance pitch offset comes back as the pitch normaliser."""
        expected = float(np.std(PITCH_OFFSETS_RAD, ddof=1))
        self.assertAlmostEqual(self.body["channels"]["foot_pitch_rad"]["scalar_sd"], expected, delta=0.05 * expected)

    def test_com_channels_are_referred_to_touchdown(self):
        """Check both COM channels start at zero spread, since their absolute level is arbitrary."""
        self.assertEqual(self.body["channels"]["com_height_m"]["datum"], "touchdown_tangent")
        for name in ("com_height_m", "com_vertical_velocity_m_s"):
            curve = np.asarray(self.body["channels"][name]["pointwise_sd"], dtype=float)
            mean_curve = np.asarray(self.body["channels"][name]["pointwise_mean"], dtype=float)
            self.assertEqual(float(curve[0]), 0.0)
            self.assertEqual(float(mean_curve[0]), 0.0)

    def test_leave_one_out_error_is_one_normaliser(self):
        """Check a held-out stance lands about one normaliser from the others, as designed."""
        for name in CHANNEL_NAMES:
            check = self.body["channels"][name]["leave_one_out_normalised_rms"]
            self.assertAlmostEqual(check["mean"], check["expected"], delta=0.35)

    def test_curves_do_not_depend_on_the_exported_padding(self):
        """Check a longer flight padding leaves every channel curve unchanged.

        The COM surrogate integrates from the first padded sample, so a naive touchdown datum
        would leave the arbitrary touchdown velocity, and with it the padding, inside the COM
        height curve.
        """
        root = Path(self.directory.name)
        first = read_stance(_write_stance(root / "pad_a.json", index=0, peak_n=1750.0, pitch_offset_rad=0.0))
        second = read_stance(
            _write_stance(root / "pad_b.json", index=0, peak_n=1750.0, pitch_offset_rad=0.0, padding_s=0.07)
        )
        for name in CHANNEL_NAMES:
            np.testing.assert_allclose(first.curves[name], second.curves[name], atol=1e-9)

    def test_timing_is_kept_as_a_separate_measured_tolerance(self):
        """Check duration, stride period and peak force keep their own spreads beside the curves."""
        timing = self.body["timing"]
        for key in ("contact_duration_s", "event_duration_s", "stride_period_s", "peak_vertical_force_n"):
            self.assertGreaterEqual(timing[key]["count"], 2)
            self.assertTrue(math.isfinite(timing[key]["sd"]))
        self.assertAlmostEqual(timing["stride_period_s"]["mean"], 0.72, places=6)

    def test_provenance_and_window_are_recorded(self):
        """Check the artifact carries the window, the sources and the stated limitations."""
        self.assertEqual(self.body["windows_s"], [[79.0, 99.0]])
        self.assertIn("c3d", self.body["provenance"]["sources"])
        self.assertIn("limitations", self.body["provenance"])
        self.assertEqual(self.body["provenance"]["smoothing"], "none; the pointwise sd is reported as measured")
        self.assertEqual(
            self.body["contact"]["source_of_truth"], "projects.impedance_instron.env.CONTACT_FORCE_FRACTION"
        )
        self.assertAlmostEqual(self.body["contact"]["threshold_n"], CONTACT_FORCE_FRACTION * MASS_KG * GRAVITY_M_S2)

    def test_too_few_stances_is_rejected(self):
        """Check a spread cannot be claimed from fewer than three stances."""
        with self.assertRaises(ValueError):
            build_variability(self.paths[:2])

    def test_duplicate_stances_are_rejected(self):
        """Check the same stance twice cannot inflate the stance count."""
        with self.assertRaises(ValueError):
            build_variability([self.paths[0], self.paths[0], self.paths[1]])

    def test_one_stance_read_alone_reports_its_own_contact(self):
        """Check a single stance exposes its measured contact, peak and source attribution."""
        stance = read_stance(self.paths[0])
        self.assertEqual(set(stance.curves), set(CHANNEL_NAMES))
        self.assertAlmostEqual(stance.summary["contact_duration_s"], 0.282, delta=0.01)
        self.assertAlmostEqual(stance.summary["peak_fz_n"], PEAKS_N[0], places=6)
        self.assertAlmostEqual(stance.summary["peak_fz_percent_of_contact"], 50.0, delta=1.0)


def _pelvis_profile(height_m: np.ndarray, forward_m: np.ndarray, rate_hz: float = 100.0, belt_m_s: float = 3.0) -> dict:
    """Return the smallest profile object :func:`pelvis_series` reads, with a known centroid."""
    knots = np.arange(height_m.size) / rate_hz
    markers = np.zeros((height_m.size, 4, 3))
    markers[:, :, 2] = height_m[:, None]
    markers[:, :, 0] = forward_m[:, None]
    markers[:, 0, 1] = 0.1
    markers[:, 1, 1] = -0.1
    return {
        "pelvis_reference": {
            "marker_names": ["LASI", "RASI", "LPSI", "RPSI"],
            "knot_time_s": knots.tolist(),
            "centroid_m": markers.mean(axis=1).tolist(),
        },
        "provenance": {"running": {"belt_speed_m_s": belt_m_s}},
    }


class PelvisSeriesTest(unittest.TestCase):
    """Cover the measured pelvis channels derived from the optical marker block."""

    def test_velocities_are_central_differences_of_the_measured_centroid(self):
        """Check pelvis velocity is the derivative of the measured height, with no filtering."""
        rate = 100.0
        knots = np.arange(200) / rate
        height = 1.0 + 0.03 * np.sin(2.0 * np.pi * 1.5 * knots)
        forward = -3.0 * knots
        series = pelvis_series(_pelvis_profile(height, forward))
        self.assertEqual(set(series), {channel.key for channel in PELVIS_CHANNELS})
        clock, values = series["pelvis_centroid_z_m"]
        np.testing.assert_allclose(values, height, atol=1e-12)
        np.testing.assert_allclose(clock, knots, atol=1e-12)
        expected = 0.03 * 2.0 * np.pi * 1.5 * np.cos(2.0 * np.pi * 1.5 * knots)
        _, measured = series["pelvis_centroid_vz_m_s"]
        self.assertLess(float(np.max(np.abs(measured[2:-2] - expected[2:-2]))), 2e-3)

    def test_fore_aft_velocity_is_measured_relative_to_the_belt(self):
        """Check a pelvis held in place on a 3 m/s belt travels at the belt speed, not at zero."""
        knots = np.arange(200) / 100.0
        series = pelvis_series(_pelvis_profile(np.full(knots.size, 1.0), np.zeros(knots.size)))
        _, forward = series["pelvis_centroid_vx_belt_m_s"]
        np.testing.assert_allclose(forward, 3.0, atol=1e-9)

    def test_a_profile_without_the_block_raises(self):
        """Check a stance exported without pelvis markers cannot silently produce a pelvis channel."""
        with self.assertRaises(ValueError):
            pelvis_series({"provenance": {"running": {"belt_speed_m_s": 3.0}}})


class ComComparisonTest(unittest.TestCase):
    """Cover the pelvis-against-force-integral comparison, the reason the pelvis was exported."""

    def _stance(
        self,
        residual: np.ndarray,
        offset: float,
        slope: float,
        amplitude: float = 0.05,
        samples: int = NORMALIZED_SAMPLES,
    ) -> Stance:
        """Build a stance whose pelvis height differs from the force COM by a known line plus residual."""
        percent = np.linspace(0.0, 100.0, samples)
        seconds = 0.3 * percent / 100.0
        force = 1.0 - amplitude * np.sin(np.pi * percent / 100.0)
        pelvis = force + offset + slope * seconds + residual
        raw = {
            "com_height_m": force,
            "pelvis_height_m": pelvis,
            "com_vertical_velocity_m_s": np.gradient(force, seconds),
            "pelvis_vertical_velocity_m_s": np.gradient(pelvis, seconds),
        }
        curves = {"com_height_m": force - force[0], "pelvis_height_m": pelvis}
        pelvis_summary = {"present": True, "centroid_height_noise_m": 0.0002, "centroid_velocity_noise_m_s": 0.014}
        return Stance(
            path="synthetic", curves=curves, raw=raw, contact_s=(0.0, 0.3), summary={"pelvis": pelvis_summary}
        )

    def test_the_unknowable_line_is_removed_and_the_rest_is_reported(self):
        """Check an arbitrary initial height and velocity vanish while a real difference survives."""
        percent = np.linspace(0.0, 100.0, NORMALIZED_SAMPLES)
        # A cosine over the contact is orthogonal to both removed degrees of freedom, so the whole
        # of it must survive the line removal.
        residual = 0.004 * np.cos(2.0 * np.pi * percent / 100.0)
        stances = [self._stance(residual, offset=0.4 + 0.01 * index, slope=-1.5 * index) for index in range(4)]
        result = compare_com_surrogates(stances, percent)
        self.assertAlmostEqual(
            result["height_residual_m"]["rms"]["mean"], float(np.sqrt(np.mean(residual**2))), places=5
        )
        self.assertAlmostEqual(result["height_residual_m"]["rms"]["sd"], 0.0, places=9)
        # The grid samples both ends of the contact, so the discrete cosine leaks 4e-5 m into the offset.
        self.assertAlmostEqual(result["height_residual_m"]["removed_offset_m"]["mean"], 0.415, places=4)
        self.assertEqual(len(result["height_residual_m"]["across_stride_mean_curve_m"]), NORMALIZED_SAMPLES)

    def test_a_pelvis_that_repeats_the_force_channel_correlates_with_it(self):
        """Check the shared-information measure is one when the pelvis adds nothing new."""
        percent = np.linspace(0.0, 100.0, NORMALIZED_SAMPLES)
        stances = [
            self._stance(np.zeros(NORMALIZED_SAMPLES), offset=0.4, slope=0.1 * index, amplitude=0.05 + 0.002 * index)
            for index in range(5)
        ]
        shared = compare_com_surrogates(stances, percent)["shared_stride_to_stride_information"]
        self.assertAlmostEqual(shared["after_line_removal"], 1.0, places=6)

    def test_stances_without_the_pelvis_block_are_not_compared(self):
        """Check the comparison is absent rather than guessed when a stance has no pelvis."""
        percent = np.linspace(0.0, 100.0, NORMALIZED_SAMPLES)
        stance = self._stance(np.zeros(NORMALIZED_SAMPLES), offset=0.0, slope=0.0)
        stance.summary["pelvis"] = {"present": False}
        self.assertIsNone(compare_com_surrogates([stance], percent))


class SampleNoiseTest(unittest.TestCase):
    """Cover the noise floor that decides whether a pelvis residual means anything."""

    def test_white_noise_on_a_smooth_signal_is_recovered(self):
        """Check the fourth-difference estimate returns the injected per-sample noise."""
        generator = np.random.default_rng(7)
        clock = np.arange(2000) / 100.0
        smooth = 1.0 + 0.05 * np.sin(2.0 * np.pi * 1.5 * clock)
        injected = 0.0004
        estimate = sample_noise(smooth + generator.normal(0.0, injected, clock.size))
        self.assertAlmostEqual(estimate, injected, delta=0.1 * injected)

    def test_a_noise_free_signal_has_a_negligible_floor(self):
        """Check a clean signal does not invent a noise floor."""
        clock = np.arange(500) / 100.0
        self.assertLess(sample_noise(1.0 + 0.05 * np.sin(2.0 * np.pi * clock)), 1e-6)

    def test_a_short_signal_is_rejected(self):
        """Check the estimate refuses a window too short to support it."""
        with self.assertRaises(ValueError):
            sample_noise(np.zeros(4))


class ArtifactTest(unittest.TestCase):
    """Cover the sealed artifact and the loader the cost will use."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.paths = _write_stances(self.root)
        self.artifact = self.root / "stride_variability.json"
        write_variability(build_variability(self.paths), self.artifact)

    def tearDown(self):
        self.directory.cleanup()

    def test_artifact_round_trips(self):
        """Check every normaliser, curve and count survives a write and a read."""
        built = build_variability(self.paths)
        loaded = load_variability(self.artifact)
        self.assertEqual(loaded.stance_count, built["stance_count"])
        self.assertEqual(loaded.side, built["side"])
        self.assertEqual(loaded.windows_s, [list(window) for window in built["windows_s"]])
        self.assertEqual(loaded.body["schema_version"], SCHEMA)
        for name in CHANNEL_NAMES:
            self.assertAlmostEqual(loaded.normaliser(name), built["channels"][name]["scalar_sd"], places=12)
            np.testing.assert_allclose(loaded.pointwise[name], built["channels"][name]["pointwise_sd"])
            np.testing.assert_allclose(loaded.means[name], built["channels"][name]["pointwise_mean"])

    def test_every_normaliser_is_positive_and_finite(self):
        """Check no channel offers a zero, negative or non-finite divisor."""
        loaded = load_variability(self.artifact)
        self.assertEqual(set(loaded.scalars), set(CHANNEL_NAMES))
        for name, scalar in loaded.scalars.items():
            self.assertTrue(math.isfinite(scalar), name)
            self.assertGreater(scalar, 0.0, name)

    def test_pointwise_curves_have_the_documented_length(self):
        """Check every curve is sampled on the documented per-cent grid, ends included."""
        loaded = load_variability(self.artifact)
        self.assertEqual(loaded.percent_of_contact.size, NORMALIZED_SAMPLES)
        self.assertEqual(float(loaded.percent_of_contact[0]), 0.0)
        self.assertEqual(float(loaded.percent_of_contact[-1]), 100.0)
        for name in CHANNEL_NAMES:
            self.assertEqual(loaded.pointwise[name].size, NORMALIZED_SAMPLES)
            self.assertEqual(loaded.means[name].size, NORMALIZED_SAMPLES)
            self.assertTrue(np.all(np.isfinite(loaded.pointwise[name])))

    def test_units_and_comparability_are_declared_per_channel(self):
        """Check each channel names its unit, its datum and why that datum is honest."""
        body = load_variability(self.artifact).body
        for channel in CHANNELS:
            entry = body["channels"][channel.name]
            self.assertEqual(entry["unit"], channel.unit)
            self.assertEqual(entry["datum"], channel.datum)
            self.assertEqual(entry["source_array"], channel.key)
            self.assertTrue(entry["comparability"].strip())

    def test_missing_artifact_raises_a_clear_error(self):
        """Check a missing measurement stops the caller instead of becoming a guessed tolerance."""
        with self.assertRaises(FileNotFoundError) as raised:
            load_variability(self.root / "absent.json")
        message = str(raised.exception)
        self.assertIn("absent.json", message)
        self.assertIn("variability", message)

    def test_tampered_artifact_is_rejected(self):
        """Check editing a normaliser without resealing fails verification."""
        body = json.loads(self.artifact.read_text(encoding="utf-8"))
        body["channels"]["vertical_force_n"]["scalar_sd"] *= 0.5
        self.artifact.write_text(json.dumps(body), encoding="utf-8")
        with self.assertRaises(ValueError):
            load_variability(self.artifact)

    def test_nonpositive_normaliser_is_rejected(self):
        """Check a resealed artifact with a zero divisor is still refused."""
        body = json.loads(self.artifact.read_text(encoding="utf-8"))
        body["channels"]["foot_pitch_rad"]["scalar_sd"] = 0.0
        self.artifact.write_text(json.dumps(seal_variability(body)), encoding="utf-8")
        with self.assertRaises(ValueError):
            load_variability(self.artifact)

    def test_unknown_channel_has_no_guessed_normaliser(self):
        """Check asking for an unmeasured channel raises instead of returning a default."""
        loaded = load_variability(self.artifact)
        with self.assertRaises(KeyError):
            loaded.normaliser("knee_moment_n_m")


class MeasuredArtifactTest(unittest.TestCase):
    """Check the artifact built from the participant's own stances, when it is present."""

    @unittest.skipUnless(Path(DEFAULT_OUTPUT).is_file(), "measured stride variability artifact is absent")
    def test_measured_artifact_is_verified_and_usable(self):
        """Check the exported measurement loads, is positive everywhere and names its stances."""
        loaded = load_variability(DEFAULT_OUTPUT)
        self.assertGreaterEqual(loaded.stance_count, 3)
        self.assertEqual(len(loaded.body["stances"]), loaded.stance_count)
        self.assertTrue(loaded.windows_s)
        for name in loaded.scalars:
            self.assertGreater(loaded.normaliser(name), 0.0, name)
            self.assertEqual(loaded.pointwise[name].size, loaded.percent_of_contact.size)

    @unittest.skipUnless(Path(DEFAULT_OUTPUT).is_file(), "measured stride variability artifact is absent")
    def test_measured_pelvis_channels_come_with_the_comparison(self):
        """Check a pelvis-carrying artifact also reports how far the pelvis is from the force COM."""
        loaded = load_variability(DEFAULT_OUTPUT)
        if not set(PELVIS_CHANNEL_NAMES) <= set(loaded.scalars):
            self.skipTest("artifact was built from stances without the measured pelvis block")
        comparison = loaded.body["com_comparison"]
        self.assertIsNotNone(comparison)
        self.assertGreater(comparison["height_residual_m"]["rms"]["mean"], 0.0)
        self.assertEqual(
            len(comparison["height_residual_m"]["across_stride_mean_curve_m"]), loaded.percent_of_contact.size
        )
        for stance in loaded.body["stances"]:
            self.assertTrue(stance["pelvis"]["present"])
            self.assertEqual(stance["schema_version"], "impedance_stance_3")


if __name__ == "__main__":
    unittest.main()
