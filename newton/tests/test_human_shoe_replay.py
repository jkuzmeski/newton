# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for exact prescribed-kinematics human-shoe load replay."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import warp as wp

from projects.digital_instron_v2.core import CALIBRATED_MATERIAL
from projects.digital_instron_v2.dynamics import FoundationConfig, MidsoleFoundation
from projects.human_shoe.replay import (
    PrescribedReplayConfig,
    _CarrierState,
    _sample_motion_hermite,
    find_contact_windows,
    replay_prescribed_shoe_load,
)

EXPERIMENT = "experiments/human_shoe/baseline_gait2354.json"


class TestPrescribedMotionSampling(unittest.TestCase):
    def test_hermite_sampling_reproduces_knots_and_linear_derivative(self):
        """Reproduce source knots and the analytic derivative of a linear motion."""
        source_time = np.array([0.0, 0.2, 0.5, 1.0])
        source = np.column_stack([2.0 * source_time + 1.0, -3.0 * source_time + 4.0])
        coordinates, speeds = _sample_motion_hermite(source_time, source, source_time)

        np.testing.assert_allclose(coordinates, source, atol=1.0e-12)
        np.testing.assert_allclose(speeds, [[2.0, -3.0]] * len(source_time), atol=1.0e-12)

    def test_checked_motion_has_three_complete_right_stances(self):
        """Find the three complete right-shoe penetration runs in Gait2354."""
        windows = find_contact_windows(EXPERIMENT, device="cpu")

        self.assertEqual(len(windows), 3)
        np.testing.assert_allclose(
            [[window.start_time_s, window.end_time_s] for window in windows],
            [[0.65, 1.23333333], [1.88333333, 2.5], [3.11666667, 3.7]],
            atol=1.0e-8,
        )
        self.assertTrue(all(window.minimum_clearance_m < -0.01 for window in windows))


class TestFoundationReplayObservables(unittest.TestCase):
    def _foundation(self, body_height: float, velocity_z: float):
        state = _CarrierState(
            body_q=wp.array([[0.0, 0.0, body_height, 0.0, 0.0, 0.0, 1.0]], dtype=wp.transform),
            body_qd=wp.array([[0.0, 0.0, velocity_z, 0.0, 0.0, 0.0]], dtype=wp.spatial_vector),
            body_f=wp.zeros(1, dtype=wp.spatial_vector),
        )
        foundation = MidsoleFoundation(
            np.array([[0.03, -0.02, 0.0]], dtype=np.float32),
            np.zeros(1, dtype=np.float32),
            np.array([0.1], dtype=np.float32),
            np.array([1.0e-3], dtype=np.float32),
            np.array([[-1, -1, -1, -1]], dtype=np.int32),
            0.01,
            CALIBRATED_MATERIAL,
            0,
            wp.zeros(1, dtype=wp.vec3),
            FoundationConfig(),
        )
        foundation.apply(state, 1.0e-3, clear_body_force=True)
        return state, foundation

    def test_reductions_match_applied_one_column_wrench_and_power(self):
        """Record force, world-origin moment, COP, compression, and contact power."""
        state, foundation = self._foundation(body_height=-0.01, velocity_z=-0.2)
        force = foundation.resultant_force.numpy()[0]
        moment = foundation.resultant_moment_origin.numpy()[0]
        body_wrench = state.body_f.numpy()[0]
        fz = float(force[2])

        self.assertGreater(fz, 0.0)
        np.testing.assert_allclose(force, body_wrench[:3], rtol=1.0e-6)
        np.testing.assert_allclose(moment, body_wrench[3:], rtol=1.0e-6)
        np.testing.assert_allclose(moment, [-0.02 * fz, -0.03 * fz, 0.0], rtol=1.0e-5, atol=1.0e-6)
        np.testing.assert_allclose(foundation.cop_moment.numpy()[0, :2] / fz, [0.03, -0.02], atol=1.0e-6)
        self.assertAlmostEqual(float(foundation.max_compression.numpy()[0]), 0.01, places=6)
        self.assertEqual(int(foundation.active.numpy()[0]), 1)
        self.assertAlmostEqual(float(foundation.contact_power.numpy()[0]), -0.2 * fz, delta=1.0e-5 * fz)

    def test_no_contact_has_zero_observables(self):
        """Keep all reductions zero while the prescribed sole is above ground."""
        state, foundation = self._foundation(body_height=0.01, velocity_z=0.0)

        np.testing.assert_allclose(foundation.resultant_force.numpy(), 0.0)
        np.testing.assert_allclose(foundation.resultant_moment_origin.numpy(), 0.0)
        np.testing.assert_allclose(state.body_f.numpy(), 0.0)
        self.assertEqual(float(foundation.contact_power.numpy()[0]), 0.0)
        self.assertEqual(float(foundation.max_compression.numpy()[0]), 0.0)
        self.assertEqual(int(foundation.active.numpy()[0]), 0)


class TestPrescribedShoeReplay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Run one coarse exact replay shared by result and export tests."""
        cls.result = replay_prescribed_shoe_load(EXPERIMENT, PrescribedReplayConfig(dt_s=0.005))

    def test_first_stance_replay_is_finite_and_unloaded_at_brackets(self):
        """Replay exact OpenSim kinematics with a finite, bounded shoe response."""
        result = self.result
        self.assertGreater(len(result.time_s), 100)
        self.assertTrue(np.all(np.diff(result.time_s) > 0.0))
        self.assertTrue(np.all(np.isfinite(result.grf_n)))
        self.assertGreater(result.peak_vertical_force_n, 1000.0)
        self.assertLess(result.peak_vertical_force_n, 2000.0)
        self.assertGreater(result.final_vertical_impulse_ns, 250.0)
        self.assertLess(result.final_vertical_impulse_ns, 400.0)
        self.assertEqual(float(result.grf_n[0, 2]), 0.0)
        self.assertEqual(float(result.grf_n[-1, 2]), 0.0)
        self.assertGreater(float(np.max(result.max_compression_m)), 0.01)
        self.assertTrue(np.any(result.cop_valid))
        np.testing.assert_allclose(
            result.impulse_ns[-1], np.sum(result.grf_n * result.dt_s[:, None], axis=0), rtol=2.0e-5
        )
        self.assertAlmostEqual(
            result.final_contact_work_j,
            float(np.sum(result.contact_power_w * result.dt_s)),
            delta=1.0e-4 * abs(result.final_contact_work_j),
        )
        self.assertTrue(np.all(np.abs(result.cop_m[result.cop_valid, :2]) < 1.0))

    def test_export_writes_units_and_provenance_sidecar(self):
        """Round-trip the replay trace through CSV plus finite JSON metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path, metadata_path = self.result.write_csv(Path(tmpdir) / "replay.csv")
            metadata = json.loads(metadata_path.read_text())
            rows = csv_path.read_text().splitlines()

        self.assertEqual(len(rows), len(self.result.time_s) + 1)
        self.assertEqual(metadata["sample_count"], len(self.result.time_s))
        self.assertEqual(metadata["coordinate_system"], "Newton right-handed Z-up")
        self.assertEqual(metadata["moment_reference"], "fixed world origin")
        self.assertEqual(
            metadata["provenance"]["experiment_path"],
            "experiments/human_shoe/baseline_gait2354.json",
        )
        self.assertFalse(Path(metadata["provenance"]["motion_path"]).is_absolute())
        self.assertEqual(len(metadata["columns"]), len(rows[0].split(",")))

    def test_timestep_refinement_preserves_impulse_and_work(self):
        """Keep integral shoe loads stable when halving the replay timestep."""
        medium = replay_prescribed_shoe_load(EXPERIMENT, PrescribedReplayConfig(dt_s=0.004))
        fine = replay_prescribed_shoe_load(EXPERIMENT, PrescribedReplayConfig(dt_s=0.002))

        self.assertLess(abs(medium.final_vertical_impulse_ns / fine.final_vertical_impulse_ns - 1.0), 0.01)
        self.assertLess(abs(medium.final_contact_work_j / fine.final_contact_work_j - 1.0), 0.02)
        self.assertLess(abs(medium.peak_vertical_force_n / fine.peak_vertical_force_n - 1.0), 0.02)
        self.assertLess(abs(np.max(medium.max_compression_m) - np.max(fine.max_compression_m)), 1.0e-4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
