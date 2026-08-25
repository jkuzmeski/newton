# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for preliminary gait C3D residual sensitivity."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import newton.opensim as osim
from projects.gait_c3d.compatibility import residual_sensitivity


class TestGaitC3DResidualSensitivity(unittest.TestCase):
    """Verify lag alignment, residual axes, roots, and preliminary publication."""

    @staticmethod
    def _model() -> osim.OsimModel:
        """Build a model whose structural root names do not contain pelvis."""
        root_coordinates = [
            osim.OsimCoordinate("turn_a", motion_type="rotational"),
            osim.OsimCoordinate("slide_a", motion_type="translational"),
            osim.OsimCoordinate("turn_b", motion_type="rotational"),
            osim.OsimCoordinate("slide_b", motion_type="translational"),
            osim.OsimCoordinate("turn_c", motion_type="rotational"),
            osim.OsimCoordinate("slide_c", motion_type="translational"),
        ]
        body = osim.OsimBody(
            "floating_body",
            mass=10.0,
            mass_center=(0.1, -0.2, 0.3),
            inertia=(1.0, 2.0, 3.0, 0.1, 0.2, 0.3),
        )
        return osim.OsimModel(
            name="structural_root",
            bodies=[body],
            joints=[
                osim.OsimJoint(
                    "world_attachment",
                    type="CustomJoint",
                    parent_body="ground",
                    child_body="floating_body",
                    coordinates=root_coordinates,
                )
            ],
        )

    @staticmethod
    def _write_schema3(data_dir: Path, model_name: str = "model.osim") -> None:
        """Write the strict synthetic source artifacts used by publication tests."""
        data_dir.mkdir()
        times = np.linspace(0.0, 0.1, 101)
        names = np.asarray(["turn_a", "slide_a", "turn_b", "slide_b", "turn_c", "slide_c"])
        wrenches = np.zeros((len(times), 1, 9))
        wrenches[:, 0, 0] = times
        np.savez_compressed(
            data_dir / "analysis.npz",
            schema_version=np.asarray("gait_c3d_analysis_3"),
            times=times,
            id_coordinates=np.zeros((len(times), 6)),
            id_speeds=np.zeros((len(times), 6)),
            id_accelerations=np.zeros((len(times), 6)),
            id_names=names,
            id_external_bodies=np.asarray(["floating_body"]),
            id_external_wrenches=wrenches,
        )
        qc = {
            "schema_version": "gait_c3d_analysis_3",
            "pelvis_residuals": {"normalization": {"body_weight_N": 100.0, "marker_height_m": 2.0}},
            "subject_mass": {"kg": 10.0},
            "artifacts": {"model": model_name},
        }
        (data_dir / "qc_summary.json").write_text(json.dumps(qc))
        (data_dir / "manifest.json").write_text(json.dumps({"schema_version": "gait_c3d_analysis_3"}))
        (data_dir / model_name).write_text("synthetic")

    def test_interpolate_lag_sign_and_values(self):
        """Interpret positive lag as a delayed measured wrench."""
        times = np.linspace(0.0, 0.1, 11)
        wrenches = np.zeros((11, 1, 9))
        wrenches[:, 0, 0] = 100.0 * times
        target_times, shifted = residual_sensitivity.interpolate_wrench_lags(times, wrenches, np.asarray([-5, 0, 5]))
        np.testing.assert_allclose(target_times, times[1:-1])
        np.testing.assert_allclose(shifted[0, :, 0, 0], 100.0 * (target_times - 0.005))
        np.testing.assert_allclose(shifted[1, :, 0, 0], 100.0 * target_times)
        np.testing.assert_allclose(shifted[2, :, 0, 0], 100.0 * (target_times + 0.005))

        delayed = wrenches.copy()
        delayed[:, 0, 0] = 100.0 * (times - 0.005)
        delayed_times, aligned = residual_sensitivity.interpolate_wrench_lags(times, delayed, np.asarray([5]))
        np.testing.assert_allclose(aligned[0, :, 0, 0], 100.0 * delayed_times)

    def test_interpolation_rejects_extrapolation(self):
        """Reject an explicit grid that would extend a wrench endpoint."""
        times = np.linspace(0.0, 0.1, 11)
        wrenches = np.zeros((11, 1, 9))
        with self.assertRaisesRegex(ValueError, "extrapolate"):
            residual_sensitivity.interpolate_wrench_lags(
                times,
                wrenches,
                np.asarray([5]),
                sample_indices=np.arange(len(times)),
            )

    def test_resultant_metrics_preserve_lag_time_coordinate_axes(self):
        """Reduce vector components last and time second for every lag."""
        values = np.zeros((2, 3, 7))
        translation_indices = [1, 3, 6]
        rotation_indices = [0, 2, 5]
        for index in translation_indices:
            values[0, :, index] = [3.0, 4.0, 0.0]
            values[1, :, index] = [0.0, 0.0, 6.0]
        for index in rotation_indices:
            values[0, :, index] = 2.0
            values[1, :, index] = [1.0, 2.0, 3.0]
        metrics = residual_sensitivity.resultant_metrics(
            values, translation_indices, rotation_indices, body_weight_N=10.0, subject_height_m=2.0
        )
        expected_force_rms = np.asarray([np.sqrt((27.0 + 48.0) / 3.0), 6.0])
        expected_force_peak = np.asarray([4.0 * np.sqrt(3.0), 6.0 * np.sqrt(3.0)])
        expected_moment_rms = np.asarray([2.0 * np.sqrt(3.0), np.sqrt(14.0)])
        np.testing.assert_allclose(metrics["translation_rms_N"], expected_force_rms)
        np.testing.assert_allclose(metrics["translation_peak_N"], expected_force_peak)
        np.testing.assert_allclose(metrics["rotation_rms_Nm"], expected_moment_rms)
        np.testing.assert_allclose(metrics["translation_rms_fraction_BW"], expected_force_rms / 10.0)
        np.testing.assert_allclose(metrics["rotation_rms_fraction_BW_height"], expected_moment_rms / 20.0)

    def test_structural_roots_use_joint_topology_and_exact_names(self):
        """Find arbitrary root names from the ground-attached joint."""
        names = ["turn_a", "slide_a", "turn_b", "slide_b", "turn_c", "slide_c"]
        roots = residual_sensitivity.structural_root_groups(self._model(), names)
        self.assertEqual(roots["joint_names"], ["world_attachment"])
        self.assertEqual(roots["translation_names"], ["slide_a", "slide_b", "slide_c"])
        self.assertEqual(roots["translation_indices"], [1, 3, 5])
        self.assertEqual(roots["rotation_names"], ["turn_a", "turn_b", "turn_c"])
        self.assertEqual(roots["rotation_indices"], [0, 2, 4])

    def test_model_audit_archives_mass_com_and_inertia(self):
        """Archive inertial properties without applying a model adjustment."""
        audit = residual_sensitivity.audit_model_inertia(self._model(), measured_mass_kg=10.0)
        self.assertEqual(audit["model_total_mass_kg"], 10.0)
        self.assertTrue(audit["within_one_percent_measured_mass"])
        self.assertFalse(audit["adjustments_applied"])
        self.assertEqual(audit["segments"][0]["com_body_m"], [0.1, -0.2, 0.3])
        tensor = np.asarray(audit["segments"][0]["inertia_about_com_kg_m2"])
        np.testing.assert_allclose(tensor, [[1.0, 0.1, 0.2], [0.1, 2.0, 0.3], [0.2, 0.3, 3.0]])

    def test_loader_rejects_non_schema3_archive(self):
        """Reject an analysis archive that is not exact schema 3."""
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            self._write_schema3(data_dir)
            with np.load(data_dir / "analysis.npz", allow_pickle=False) as source:
                arrays = {name: source[name] for name in source.files}
            arrays["schema_version"] = np.asarray("gait_c3d_analysis_2")
            np.savez_compressed(data_dir / "analysis.npz", **arrays)
            with self.assertRaisesRegex(ValueError, "schema_version"):
                residual_sensitivity.load_schema3(data_dir)

    def test_publication_remains_preliminary_and_non_overwriting(self):
        """Archive all lags but accept no timing correction or overwrite."""
        model = self._model()
        case = self

        class Solver:
            """Return small finite root loads and count batched calls."""

            coordinate_names = ("turn_a", "slide_a", "turn_b", "slide_b", "turn_c", "slide_c")
            calls = 0

            def __init__(self, parsed_model, device=None):
                self.parsed_model = parsed_model
                self.device = device

            def solve(self, q, qd, qdd, external_bodies=None, external_wrenches=None):
                Solver.calls += 1
                case.assertEqual(q.shape, (41 * 61, 6))
                case.assertEqual(external_wrenches.shape, (41 * 61, 1, 9))
                common_times = np.linspace(0.02, 0.08, 61)
                expected = common_times[None, :] + np.arange(-20, 21)[:, None] * 0.001
                np.testing.assert_allclose(external_wrenches[:, 0, 0].reshape(41, 61), expected)
                row_code = np.arange(len(q), dtype=float)[:, None]
                return np.repeat(row_code, q.shape[1], axis=1)

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            output_dir = Path(tmp) / "published"
            self._write_schema3(data_dir)
            with (
                mock.patch.object(residual_sensitivity.osim, "parse_osim", return_value=model),
                mock.patch.object(residual_sensitivity.osim, "InverseDynamics", Solver),
            ):
                result = residual_sensitivity.run(data_dir, output_dir, device="cpu")
            self.assertEqual(result, output_dir.resolve())
            self.assertEqual(Solver.calls, 1)
            summary = json.loads((output_dir / "residual_sensitivity.json").read_text())
            self.assertEqual(summary["status"], "preliminary_sensitivity_only_no_timing_accepted")
            self.assertIsNone(summary["accepted_timing_lag_ms"])
            self.assertFalse(summary["timing_adjustment_applied"])
            self.assertEqual(summary["scope"]["inverse_dynamics_calls"], 1)
            self.assertTrue((output_dir / "residual_sensitivity_source.py").is_file())
            self.assertIn("residual_sensitivity_source.py", summary["source_hashes"])
            with np.load(output_dir / "residual_sensitivity.npz", allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive["lag_ms"], np.arange(-20, 21))
                self.assertEqual(archive["generalized_forces"].shape, (41, 61, 6))
                self.assertEqual(archive["lagged_external_wrenches"].shape, (41, 61, 1, 9))
                self.assertEqual(archive["generalized_forces"][2, 3, 4], 2 * 61 + 3)
            with self.assertRaises(FileExistsError):
                residual_sensitivity.run(data_dir, output_dir, device="cpu")


if __name__ == "__main__":
    unittest.main()
