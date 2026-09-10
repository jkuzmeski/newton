# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the angle-only mechanical ankle after the supplied-data example runs."""

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import warp as wp

from projects.impedance_instron.example import Example, _apply_leg, _prescribe_axes, create_parser
from projects.impedance_instron.trajectory import TrajectoryCubic


def _reference_fixture():
    time = np.linspace(0.0, 0.3, 601)
    knots = np.linspace(-0.2, 0.5, 71)
    example = Example.__new__(Example)
    example.args = SimpleNamespace(
        reference_mode="pitch",
        pitch_cutoff=12.0,
        ankle_x=0.15,
        track_speed=0.05,
        initial_clearance=0.0,
        kinematic_rate_hz=100.0,
        unload_duration=0.04,
        unload_acceleration=25.0,
    )
    example.mass, example.foot_mass, example.com_mass = 80.0, 2.0, 78.0
    example.gravity = 9.80665
    example.times = time
    example.duration = time[-1]
    example.ankle_mount = np.array([-0.075, 0.0, 0.105])
    example.shoe = SimpleNamespace(
        column_bed=SimpleNamespace(anchor_bottom_m=np.array([[-0.15, 0.0, 0.0], [0.15, 0.0, 0.0]]))
    )
    example.profile = {
        "time_s": time,
        "source_time_s": time + 90.0,
        "foot_x_m": np.sin(time),
        "foot_z_m": np.cos(time),
        "pitch_rad": np.zeros_like(time),
        "com_x_m": 3 * time,
        "com_z_m": 1 + 0.1 * time**2,
        "reference_com_vx_m_s": np.full_like(time, 3.0),
        "reference_com_vz_m_s": 0.2 * time,
        "reference_fz_n": np.full_like(time, 80.0 * (9.80665 + 0.2)),
        "reference_fx_n": np.zeros_like(time),
        "total_measured_fx_n": np.zeros_like(time),
        "other_fz_n": np.zeros_like(time),
        "reference_cop_x_m": np.zeros_like(time),
        "provenance": {"running": {"selected_stance_source_s": [90.02, 90.28]}},
        "pitch_reference": {"knot_time_s": knots, "pitch_rad": 0.1 * np.sin(5 * knots), "source_rate_hz": 100.0},
    }
    return example


class TestPitchTrajectory(unittest.TestCase):
    def test_linear_angle_has_exact_derivatives(self):
        """Preserve a linear command with zero acceleration under smoothing and interpolation."""
        knots = np.linspace(-0.2, 0.5, 71)
        target = np.linspace(0.0, 0.3, 777)
        curve = TrajectoryCubic.fit(knots, 2 * knots + 0.1)
        angle, speed, acceleration = curve.evaluate(target)
        np.testing.assert_allclose(angle, 2 * target + 0.1, atol=1e-12)
        np.testing.assert_allclose(speed, 2.0, atol=1e-10)
        np.testing.assert_allclose(acceleration, 0.0, atol=1e-8)

    def test_acceleration_is_continuous_at_optical_knots(self):
        """Avoid the acceleration jumps of piecewise C1 marker interpolation."""
        knots = np.linspace(-0.2, 0.5, 71)
        curve = TrajectoryCubic.fit(knots, 0.3 * np.sin(12 * knots), cutoff_hz=0.0)
        points = knots[2:-2]
        left = curve.evaluate(points - 1e-9)
        right = curve.evaluate(points + 1e-9)
        for a, b in zip(left, right, strict=True):
            np.testing.assert_allclose(a, b, atol=1e-5)

    def test_reject_bad_clocks_and_extrapolation(self):
        """Reject irregular source knots, unsupported cutoffs, and extrapolated commands."""
        knots = np.linspace(-0.2, 0.5, 71)
        for cutoff in (-1.0, 50.0, float("nan")):
            with self.assertRaises(ValueError):
                TrajectoryCubic.fit(knots, knots, cutoff_hz=cutoff)
        bad = knots.copy()
        bad[5] += 0.001
        with self.assertRaisesRegex(ValueError, "uniform"):
            TrajectoryCubic.fit(bad, knots)
        curve = TrajectoryCubic.fit(knots, knots)
        with self.assertRaisesRegex(ValueError, "extrapolate"):
            curve.evaluate([-0.3])


class TestPitchReference(unittest.TestCase):
    def test_marker_translations_cannot_drive_ankle_reference(self):
        """Ignore all measured foot translations and the old marker-line pitch array."""
        example = _reference_fixture()
        example._make_reference()
        reference = example.reference.copy()
        example.profile["foot_x_m"] *= 1000
        example.profile["foot_z_m"] *= -1000
        example.profile["pitch_rad"] += 100.0
        example._make_reference()
        np.testing.assert_array_equal(example.reference, reference)
        np.testing.assert_allclose(reference[:, 0], 0.15 + 0.05 * example.times, atol=1e-8)

    def test_constant_gap_reference_is_mass_consistent(self):
        """Map the force-integrated centroid to two masses with one fixed vertical gap."""
        example = _reference_fixture()
        example._make_reference()
        ref = example.reference
        np.testing.assert_array_equal(ref[:, 13], np.full(len(ref), ref[0, 13], dtype=np.float32))
        np.testing.assert_array_equal(ref[:, 14], 0.0)
        center = (2.0 * ref[:, 1] + 78.0 * ref[:, 4]) / 80.0
        np.testing.assert_allclose(center, example.profile["com_z_m"], atol=2e-7)
        acceleration = (2.0 * ref[:, 10] + 78.0 * ref[:, 16]) / 80.0
        np.testing.assert_allclose(acceleration, 0.2, atol=2e-6)

    def test_scheduled_impedance_accounts_for_spring_energy_change(self):
        """Include gain-schedule work rather than crediting released virtual spring energy to the shoe."""
        reference = np.zeros((1, 22), np.float32)
        reference[0, [10, 11, 12, 13, 14, 19, 20, 21]] = [2.0, 1000.0, 20.0, 1.0, -0.2, 0.4, -5.0, 60.0]
        q = wp.array([[0, 0, 0, 0, 0, 0, 1], [0.5, 0, 1.1, 0, 0, 0, 1]], dtype=wp.transform, device="cpu")
        velocity = np.zeros((2, 6), np.float32)
        velocity[:, 2] = [0.2, -0.1]
        qd = wp.array(velocity, dtype=wp.spatial_vector, device="cpu")
        force = wp.zeros(2, dtype=wp.spatial_vector, device="cpu")
        diagnostics = wp.zeros(5, dtype=wp.float32, device="cpu")
        wp.launch(
            _apply_leg,
            dim=1,
            inputs=[
                0,
                wp.array(reference, device="cpu"),
                2.0,
                10000.0,
                200.0,
                5000.0,
                9.80665,
                q,
                qd,
                force,
                diagnostics,
            ],
            device="cpu",
        )
        d = diagnostics.numpy()
        rate, error, desired_rate = -0.3, 0.1, -0.2
        spring_power = 10000 * 0.4 * error * (rate - desired_rate) + 0.5 * 10000 * (-5.0) * error**2
        self.assertAlmostEqual(float(d[1] + d[2]), float(d[0]) * rate + spring_power, delta=0.005)
        self.assertAlmostEqual(float(d[3]), 0.5 * 10000 * 0.4 * error**2, delta=0.001)

    def test_pitch_projection_keeps_vertical_states_free(self):
        """Apply angle and track commands without overwriting either vertical state."""
        reference = np.zeros((1, 19), np.float32)
        reference[0, 2] = 0.7
        reference[0, 7] = 2.0
        q = wp.array([[0, 0, 0.21, 0, 0, 0, 1], [0, 0, 1.05, 0, 0, 0, 1]], dtype=wp.transform, device="cpu")
        velocity = np.zeros((2, 6), np.float32)
        velocity[:, 2] = [0.3, -0.4]
        qd = wp.array(velocity, dtype=wp.spatial_vector, device="cpu")
        wp.launch(_prescribe_axes, dim=1, inputs=[0, wp.array(reference, device="cpu"), 0, q, qd], device="cpu")
        np.testing.assert_allclose(q.numpy()[:, 2], [0.21, 1.05], atol=1e-7)
        np.testing.assert_allclose(qd.numpy()[:, 2], [0.3, -0.4], atol=1e-7)


@unittest.skipUnless(
    Path("outputs/impedance_instron/stance_pitch.json").is_file()
    and Path("DigitalInstron/digital_shoe_showcase/digital_shoe.json").is_file(),
    "Generate supplied-data inputs first",
)
class TestSuppliedPitchAnkle(unittest.TestCase):
    def test_complete_native_stance_with_fixed_ankle_mount(self):
        """Finish the measured running example without marker replay or ground penetration."""
        args = create_parser().parse_args(["--viewer", "null"])
        with wp.ScopedDevice("cpu"):
            example = Example(MagicMock(), args)
            for _ in range(120):
                example.step()
            example.test_final()
            rows = example.rows()
            self.assertEqual(example.index, example.sample_count)
            self.assertTrue(all(row["last_min_height_m"] >= -0.001 for row in rows))
            self.assertTrue(all(row["ankle_x_m"] == 0.0 for row in rows))
            self.assertLess(rows[-1]["shoe_fz_n"], 0.1 * example.mass * example.gravity)
            self.assertEqual(rows[-1]["impedance_gain"], 0.0)
            np.testing.assert_allclose(
                example.foundation.anchor_local.numpy(),
                example.shoe.column_bed.anchor_bottom_m - example.ankle_mount,
                atol=2e-8,
            )
            q_history = example.foundation.q_state.numpy().copy()
            example.render()
            example.step()
            example.render()
            np.testing.assert_array_equal(example.foundation.q_state.numpy(), q_history)
            dt = example.sim_dt
            impulse = (
                sum(row["shoe_fz_n"] + row["other_fz_n"] - example.mass * example.gravity for row in rows[:-1]) * dt
            )
            delta_momentum = example.mass * (rows[-1]["com_vz_m_s"] - rows[0]["com_vz_m_s"])
            self.assertAlmostEqual(impulse, delta_momentum, delta=0.01)
            self.assertFalse(example.metadata["shoe_orientation"]["anatomical_side_validated"])


if __name__ == "__main__":
    unittest.main()
