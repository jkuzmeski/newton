# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test finite-window recovery with synthetic channels and native-state fakes."""

import copy
import json
import unittest
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import numpy as np

from projects.impedance_instron.simple.recovery import RecoveryConfig, summarize_recovery, terminal_state

_CHANNELS = (
    "pelvis_x_m",
    "pelvis_z_m",
    "foot_x_m",
    "foot_z_m",
    "leg_length_m",
    "pitch_rad",
    "pelvis_vx_m_s",
    "pelvis_vz_m_s",
    "foot_vx_m_s",
    "foot_vz_m_s",
    "leg_rate_m_s",
    "pitch_rate_rad_s",
)


def _pair(count=30, dt=0.01):
    trace = {name: np.zeros(count) for name in _CHANNELS}
    trace["time_s"] = np.arange(count, dtype=float) * dt
    terminal = dict.fromkeys(_CHANNELS, 0.0)
    terminal["time_s"] = count * dt
    return trace, copy.deepcopy(trace), terminal, terminal.copy()


def _summary(pair, *, dt=0.01, push_end_s=0.1, **kwargs):
    trace, baseline, terminal, baseline_terminal = pair
    return summarize_recovery(
        trace,
        baseline,
        dt,
        terminal=terminal,
        baseline_terminal=baseline_terminal,
        push_end_s=push_end_s,
        **kwargs,
    )


class TestRecoveryConfig(unittest.TestCase):
    def test_config_roundtrip(self):
        """Preserve frozen engineering defaults through strict JSON serialization."""
        config = RecoveryConfig()
        self.assertEqual(config.minimum_window_s, 0.15)
        self.assertEqual(config.dwell_s, 0.05)
        self.assertEqual(config.position_tolerance_m, 0.001)
        self.assertEqual(config.angle_tolerance_rad, 0.001)
        self.assertEqual(config.velocity_tolerance_m_s, 0.01)
        self.assertEqual(config.angular_velocity_tolerance_rad_s, 0.01)
        self.assertEqual(RecoveryConfig.from_dict(json.loads(json.dumps(config.to_dict(), allow_nan=False))), config)
        with self.assertRaises(FrozenInstanceError):
            config.dwell_s = 0.1
        self.assertEqual(RecoveryConfig.from_dict({"dwell_s": 0.07}).dwell_s, 0.07)

    def test_reject_invalid_settings(self):
        """Reject nonfinite, nonpositive, boolean, unknown, and nonnumeric settings."""
        for name in RecoveryConfig().to_dict():
            for value in (0, -1, float("nan"), float("inf"), True, np.bool_(False), "0.1", None, []):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    RecoveryConfig.from_dict({name: value})
        for value in ({"settling_time": 0.1}, {1: 0.1}, [], None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                RecoveryConfig.from_dict(value)


class TestRecoverySummary(unittest.TestCase):
    def test_return_with_positions_and_rates(self):
        """Accept a full observed post-pulse window with positions and rates in band."""
        pair = _pair()
        pair[0]["pelvis_z_m"][10:15] = 0.02
        result = _summary(pair)
        self.assertEqual(result["status"], "returned_within_window")
        self.assertTrue(result["engineering_screen"])
        self.assertEqual(result["state_scope"], "planar_body_positions_and_rates_only")
        self.assertIn("foam, Maxwell, and friction-history states are not screened", result["interpretation"])
        self.assertTrue(result["window"]["window_adequate"])
        self.assertAlmostEqual(result["window"]["observed_suffix_entry_time_s"], 0.15)
        self.assertAlmostEqual(result["window"]["observed_suffix_entry_after_pulse_s"], 0.05)
        self.assertAlmostEqual(result["window"]["observed_suffix_duration_s"], 0.15)
        json.dumps(result, allow_nan=False)

    def test_velocity_prevents_false_return(self):
        """Reject each out-of-band rate even when all position deviations are zero."""
        for name in (
            "pelvis_vx_m_s",
            "pelvis_vz_m_s",
            "foot_vx_m_s",
            "foot_vz_m_s",
            "leg_rate_m_s",
            "pitch_rate_rad_s",
        ):
            pair = _pair()
            pair[0][name][-4:] = 0.02
            with self.subTest(name=name):
                result = _summary(pair)
                self.assertEqual(result["status"], "not_returned_within_window")
                self.assertFalse(result["window"]["final_dwell_within_tolerances"])
                self.assertTrue(result["window"]["terminal_within_tolerances"])
                self.assertIsNone(result["window"]["observed_suffix_entry_time_s"])

    def test_all_position_channels_required(self):
        """Reject each displaced position channel independently of the other channels."""
        for name in ("pelvis_x_m", "pelvis_z_m", "foot_x_m", "foot_z_m", "leg_length_m", "pitch_rad"):
            pair = _pair()
            pair[2][name] = 0.002
            with self.subTest(name=name):
                self.assertEqual(_summary(pair)["status"], "not_returned_within_window")

    def test_short_window(self):
        """Reject a short post-pulse window despite apparently returned endpoints."""
        result = _summary(_pair(20))
        self.assertEqual(result["status"], "insufficient_window")
        self.assertFalse(result["returned_within_window"])
        self.assertAlmostEqual(result["window"]["available_post_pulse_s"], 0.1)
        self.assertIsNone(result["window"]["observed_suffix_entry_time_s"])
        self.assertEqual(_summary(_pair(), push_end_s=0.5)["status"], "insufficient_window")
        config = replace(RecoveryConfig(), dwell_s=0.25)
        self.assertEqual(_summary(_pair(), config=config)["status"], "insufficient_window")

    def test_exact_minimum_window(self):
        """Accept exact configured duration boundaries without losing float roundoff."""
        result = _summary(_pair(25))
        self.assertEqual(result["status"], "returned_within_window")
        self.assertAlmostEqual(result["window"]["observed_post_pulse_s"], 0.15)

    def test_missing_required_channels(self):
        """Invalidate missing position and rate fields in either trajectory endpoint."""
        for index, label in enumerate(("trace", "baseline_trace", "terminal", "baseline_terminal")):
            for name in (*_CHANNELS, "time_s"):
                pair = _pair()
                del pair[index][name]
                with self.subTest(source=label, name=name):
                    result = _summary(pair)
                    self.assertEqual(result["status"], "invalid_pair")
                    self.assertIn(name, result["missing_fields"][label])
                    json.dumps(result, allow_nan=False)

    def test_nonfinite_both_sides(self):
        """Invalidate nonfinite required or extra numeric fields on either side."""
        for index, label in enumerate(("trace", "baseline_trace", "terminal", "baseline_terminal")):
            for name in ("pelvis_vx_m_s", "pitch_rad", "time_s", "extra_diagnostic"):
                for value in (float("nan"), float("inf")):
                    pair = _pair()
                    pair[index][name] = np.full(30, value) if index < 2 else value
                    with self.subTest(source=label, name=name, value=value):
                        result = _summary(pair)
                        self.assertEqual(result["status"], "invalid_pair")
                        self.assertIn(name, result["nonfinite_fields"][label])
                        self.assertFalse(result["returned_within_window"])
                        json.dumps(result, allow_nan=False)

    def test_invalid_or_malformed_pair(self):
        """Fail closed on external invalidity, malformed records, and wrong shapes."""
        for value in (False, 1, "valid", None):
            with self.subTest(value=value):
                self.assertEqual(_summary(_pair(), pair_valid=value)["status"], "invalid_pair")
        for index in range(4):
            for bad in ([], [[0.0]], "bad", {"bad": 1}):
                pair = _pair()
                pair[index]["leg_rate_m_s"] = bad
                with self.subTest(index=index, bad=bad):
                    self.assertEqual(_summary(pair)["status"], "invalid_pair")
            pair = list(_pair())
            pair[index] = None
            self.assertEqual(_summary(pair)["status"], "invalid_pair")

    def test_wrapped_pitch(self):
        """Compare pitch modulo two pi without wrapping the angular velocity."""
        pair = _pair()
        pair[0]["pitch_rad"][:] = np.pi - 0.0002
        pair[1]["pitch_rad"][:] = -np.pi + 0.0002
        pair[2]["pitch_rad"] = np.pi - 0.0003
        pair[3]["pitch_rad"] = -np.pi + 0.0003
        result = _summary(pair)
        self.assertEqual(result["status"], "returned_within_window")
        self.assertAlmostEqual(result["deviations"]["pitch_rad"]["peak_abs_deviation"], 0.0004)
        self.assertAlmostEqual(result["deviations"]["pitch_rad"]["final_deviation"], -0.0006)
        pair[2]["pitch_rate_rad_s"] = 2 * np.pi
        self.assertEqual(_summary(pair)["status"], "not_returned_within_window")

    def test_terminal_not_last_sample(self):
        """Use actual terminal position and velocity instead of the last trace sample."""
        for name in ("pelvis_z_m", "pelvis_vz_m_s"):
            pair = _pair()
            pair[2][name] = 0.1
            with self.subTest(name=name):
                result = _summary(pair)
                self.assertEqual(result["status"], "not_returned_within_window")
                self.assertEqual(result["deviations"][name]["peak_abs_deviation"], 0)
                self.assertEqual(result["deviations"][name]["last_trace_deviation"], 0)
                self.assertEqual(result["deviations"][name]["final_deviation"], 0.1)
                self.assertEqual(result["deviations"][name]["peak_abs_including_terminal_deviation"], 0.1)
                self.assertEqual(result["deviations"][name]["peak_abs_post_pulse_deviation"], 0.1)
                self.assertAlmostEqual(result["last_trace_time_s"], 0.29)
                self.assertAlmostEqual(result["final_state_time_s"], 0.3)

    def test_dwell_break_and_regain(self):
        """Require a full final dwell after the last band violation, not early entry."""
        pair = _pair()
        pair[0]["leg_rate_m_s"][26] = 0.02
        result = _summary(pair)
        self.assertEqual(result["status"], "not_returned_within_window")
        self.assertAlmostEqual(result["window"]["observed_suffix_duration_s"], 0.03)
        pair[0]["leg_rate_m_s"][26] = 0
        pair[0]["leg_rate_m_s"][24] = 0.02
        result = _summary(pair)
        self.assertEqual(result["status"], "returned_within_window")
        self.assertAlmostEqual(result["window"]["observed_suffix_entry_time_s"], 0.25)
        self.assertAlmostEqual(result["window"]["observed_suffix_duration_s"], 0.05)

    def test_persistent_material_change(self):
        """Retain displacement evidence but never claim recovery for a persistent material."""
        pair = _pair()
        pair[0]["foot_z_m"][5] = 0.02
        result = _summary(pair, push_end_s=None)
        self.assertEqual(result["status"], "persistent_material_change")
        self.assertFalse(result["returned_within_window"])
        self.assertIsNone(result["window"]["observed_suffix_entry_time_s"])
        self.assertIsNone(result["window"]["available_post_pulse_s"])
        self.assertEqual(result["deviations"]["foot_z_m"]["peak_abs_deviation"], 0.02)
        self.assertEqual(_summary(pair, push_end_s=None, pair_valid=False)["status"], "invalid_pair")

    def test_clocks_match_exactly(self):
        """Reject even tiny baseline time shifts and mismatched terminal endpoints."""
        pair = _pair()
        pair[1]["time_s"][5] += 1e-12
        self.assertFalse(_summary(pair)["paired_clock_match"])
        self.assertEqual(_summary(pair)["status"], "invalid_pair")
        pair = _pair()
        pair[3]["time_s"] += 1e-12
        result = _summary(pair)
        self.assertEqual(result["status"], "invalid_pair")
        self.assertFalse(result["terminal_clock_match"])
        self.assertIsNone(result["deviations"]["pelvis_z_m"]["final_deviation"])

    def test_sampling_clock(self):
        """Reject duplicate, reversed, nonuniform, wrong-dt, and false-terminal clocks."""
        for kind in (
            "duplicate",
            "reversed",
            "nonuniform",
            "terminal_gap",
            "terminal_is_last",
            "negative",
            "scalar",
            "empty",
        ):
            pair = _pair()
            for index in (0, 1):
                if kind == "duplicate":
                    pair[index]["time_s"][5] = pair[index]["time_s"][4]
                elif kind == "reversed":
                    pair[index]["time_s"] = pair[index]["time_s"][::-1].copy()
                elif kind == "nonuniform":
                    pair[index]["time_s"][5] += 0.001
                elif kind == "negative":
                    pair[index]["time_s"] -= 0.1
                elif kind == "scalar":
                    pair[index]["time_s"] = np.array(0.0)
                elif kind == "empty":
                    pair[index]["time_s"] = np.array([])
            if kind.startswith("terminal"):
                pair[2]["time_s"] = pair[3]["time_s"] = 0.4 if kind == "terminal_gap" else 0.29
            with self.subTest(kind=kind):
                result = _summary(pair)
                self.assertEqual(result["status"], "invalid_pair")
                self.assertFalse(result["sampling_clock_valid"])
        self.assertEqual(_summary(_pair(), dt=0.005)["status"], "invalid_pair")

    def test_native_float32_clock(self):
        """Allow native float32 storage roundoff without allowing paired retiming."""
        dt = 1 / 7680
        pair = _pair(2304, dt)
        for trace in pair[:2]:
            trace["time_s"] = (np.arange(2304, dtype=np.float32) * np.float32(dt)).astype(float)
        result = _summary(pair, dt=dt)
        self.assertTrue(result["sampling_clock_valid"])
        self.assertEqual(result["status"], "returned_within_window")

    def test_do_not_extend_observation(self):
        """Count only actually observed post-pulse time without reference extension."""
        pair = _pair(10)
        for trace in pair[:2]:
            trace["time_s"] += 0.2
        for terminal in pair[2:]:
            terminal["time_s"] = 0.3
        result = _summary(pair, push_end_s=0.05)
        self.assertAlmostEqual(result["window"]["available_post_pulse_s"], 0.25)
        self.assertAlmostEqual(result["window"]["observed_post_pulse_s"], 0.1)
        self.assertEqual(result["status"], "insufficient_window")

    def test_nonfinite_subtraction(self):
        """Invalidate arithmetic overflow even when each supplied position is finite."""
        pair = _pair()
        pair[0]["pelvis_x_m"][:] = np.finfo(float).max
        pair[1]["pelvis_x_m"][:] = -np.finfo(float).max
        pair[2]["pelvis_x_m"] = np.finfo(float).max
        pair[3]["pelvis_x_m"] = -np.finfo(float).max
        result = _summary(pair)
        self.assertEqual(result["status"], "invalid_pair")
        self.assertIn("nonfinite_difference:trace:pelvis_x_m", result["invalid_reasons"])
        self.assertIn("nonfinite_difference:terminal:pelvis_x_m", result["invalid_reasons"])
        json.dumps(result, allow_nan=False)

    def test_roundoff_cannot_replace_dwell(self):
        """Reject terminal-only entry even when a very large clock inflates roundoff."""
        pair = _pair(2, dt=1e6)
        pair[0]["pelvis_z_m"][:] = 0.1
        result = _summary(pair, dt=1e6)
        self.assertEqual(result["status"], "not_returned_within_window")
        self.assertEqual(result["window"]["observed_suffix_duration_s"], 0)
        pair = _pair()
        for trace in pair[:2]:
            trace["time_s"] += 1e6
        for terminal in pair[2:]:
            terminal["time_s"] += 1e6 + 0.01
        self.assertEqual(_summary(pair)["status"], "invalid_pair")

    def test_bands_and_post_pulse_peaks(self):
        """Apply inclusive custom bands and keep post-pulse peaks separate."""
        pair = _pair()
        pair[0]["pelvis_x_m"][:10] = 0.1
        pair[0]["pelvis_x_m"][10:] = 0.002
        pair[2]["pelvis_x_m"] = 0.002
        config = replace(RecoveryConfig(), position_tolerance_m=0.002)
        result = _summary(pair, config=config)
        self.assertEqual(result["status"], "returned_within_window")
        self.assertEqual(result["deviations"]["pelvis_x_m"]["peak_abs_deviation"], 0.1)
        self.assertEqual(result["deviations"]["pelvis_x_m"]["peak_abs_post_pulse_deviation"], 0.002)
        self.assertEqual(result["deviations"]["pelvis_x_m"]["tolerance"], 0.002)

    def test_invalid_call_settings(self):
        """Reject invalid dt, pulse-end, and config values rather than silently default."""
        for dt in (0, -1, float("nan"), float("inf"), True, "0.01"):
            with self.subTest(dt=dt), self.assertRaises(ValueError):
                _summary(_pair(), dt=dt)
        for push_end_s in (-1, float("nan"), float("inf"), True, "0.1"):
            with self.subTest(push_end_s=push_end_s), self.assertRaises(ValueError):
                _summary(_pair(), push_end_s=push_end_s)
        with self.assertRaises(ValueError):
            _summary(_pair(), config={})


class _Array:
    def __init__(self, data):
        self.data = np.asarray(data)

    def numpy(self):
        return self.data


class TestTerminalState(unittest.TestCase):
    def test_native_positions_rates_and_arrays(self):
        """Extract actual native body positions and spatial rates with the leg-axis projection."""
        angle = 0.4
        q = np.array([[1, 0, 2, 0, np.sin(angle / 2), 0, np.cos(angle / 2)], [4, 0, 6, 0, 0, 0, 1]])
        qd = np.array([[1, 0, 2, 0, 7, 0], [6, 0, 12, 0, 0, 0]], dtype=float)
        rig = SimpleNamespace(state_0=SimpleNamespace(body_q=_Array(q), body_qd=_Array(qd)), sim_time=0.3)
        terminal, arrays = terminal_state(rig)
        expected = dict(zip(_CHANNELS, (4, 6, 1, 2, 5, angle, 6, 12, 1, 2, 11, 7), strict=True))
        for name, value in expected.items():
            with self.subTest(name=name):
                self.assertAlmostEqual(terminal[name], value)
        self.assertEqual(terminal["time_s"], 0.3)
        np.testing.assert_array_equal(arrays["terminal_body_q"], q)
        np.testing.assert_array_equal(arrays["terminal_body_qd"], qd)
        self.assertEqual(float(arrays["terminal_time_s"]), 0.3)
        q[:] = -100
        qd[:] = -100
        self.assertEqual(arrays["terminal_body_q"][0, 0], 1)
        self.assertEqual(arrays["terminal_body_qd"][0, 0], 1)

    def test_degenerate_leg_is_not_a_zero_rate(self):
        """Mark undefined zero-length leg rate nonfinite instead of inventing return."""
        q = np.array([[0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1]], dtype=float)
        rig = SimpleNamespace(state_0=SimpleNamespace(body_q=_Array(q), body_qd=_Array(np.zeros((2, 6)))), sim_time=0.3)
        terminal, _ = terminal_state(rig)
        self.assertTrue(np.isnan(terminal["leg_rate_m_s"]))
        pair = list(_pair())
        pair[2] = terminal
        self.assertEqual(_summary(pair)["status"], "invalid_pair")

    def test_state_shape_validation(self):
        """Reject arrays that cannot describe the reduced two-body native state."""
        for q, qd in (
            (np.zeros((1, 7)), np.zeros((1, 6))),
            (np.zeros((2, 6)), np.zeros((2, 6))),
            (np.zeros((2, 7)), np.zeros((3, 6))),
        ):
            rig = SimpleNamespace(state_0=SimpleNamespace(body_q=_Array(q), body_qd=_Array(qd)), sim_time=0.3)
            with self.subTest(q=q.shape, qd=qd.shape), self.assertRaises(ValueError):
                terminal_state(rig)


if __name__ == "__main__":
    unittest.main()
