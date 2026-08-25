# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test exact 12-sphere RRA-adjusted contact calibration."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from projects.gait_c3d.compatibility import moco_contact_calibration as calibration
from projects.gait_c3d.oracles import opensim_moco_contact_reference as reference


class TestMocoContactCalibration(unittest.TestCase):
    """Verify the calibrated topology, parameterization, and objective contract."""

    def test_default_builder_bytes_remain_pinned(self):
        """Preserve byte-identical default official contact files."""
        root = Path("/home/jo31399/newton-data/gait/processed/trial_101/opensim_moco_contact_reference")
        if not root.is_dir():
            self.skipTest("private Trial 101 reference artifact is unavailable")
        geometry = reference.xml_bytes(reference.build_contact_geometry_xml())
        forces = reference.xml_bytes(reference.build_force_xml())
        self.assertEqual(geometry, (root / "S001_ContactGeometrySet.xml").read_bytes())
        self.assertEqual(forces, (root / "S001_ContactForceSet.xml").read_bytes())

    def test_parameterization_preserves_exact_topology(self):
        """Change only bounded geometry and material values, not topology identity."""
        parameterization = calibration.ContactParameterization()
        encoded = parameterization.x0
        encoded[:6] = np.linspace(-0.006, 0.006, 6)
        encoded[6] = -0.004
        encoded[7:10] = (1.1, 0.9, 0.003)
        encoded[10:] = (5.8, 0.7, 0.2, 0.1)
        candidate = parameterization.decode(encoded)
        defaults = reference.sphere_specs()
        self.assertEqual(len(candidate.spheres), 12)
        for sphere, default in zip(candidate.spheres, defaults, strict=True):
            self.assertEqual(
                (sphere.side, sphere.role, sphere.name, sphere.force_name, sphere.body, sphere.radius_m),
                (default.side, default.role, default.name, default.force_name, default.body, default.radius_m),
            )
            self.assertAlmostEqual(sphere.location_m[0], 1.1 * default.location_m[0] + 0.003)
            self.assertAlmostEqual(sphere.location_m[2], 0.9 * default.location_m[2])
        self.assertAlmostEqual(candidate.material["stiffness"], 10**5.8)
        self.assertEqual(candidate.material["static_friction"], candidate.material["dynamic_friction"])

    def test_subject_seed_uses_scaled_asymmetric_landmarks(self):
        """Retarget sphere positions to S001 anatomy and actual toe frames."""
        root = Path("/home/jo31399/newton-data/gait/processed/trial_101/rra_adjusted_contact_input")
        if not root.is_dir():
            self.skipTest("private Trial 101 RRA input is unavailable")
        inputs = calibration.load_calibration_inputs(root)
        model = calibration.osim.parse_osim(inputs.model_path)
        prepared, repairs = calibration.prepare_contact_model(model, inputs.coordinate_names, inputs.coordinates)
        spheres = calibration.subject_sphere_specs(prepared)
        self.assertEqual(set(repairs), {"mtp_angle_l", "mtp_angle_r"})
        self.assertEqual(len(spheres), 12)
        left = {sphere.role: sphere for sphere in spheres if sphere.side == "left"}
        right = {sphere.role: sphere for sphere in spheres if sphere.side == "right"}
        self.assertNotEqual(left["heel"].location_m[0], right["heel"].location_m[0])
        self.assertEqual(left["lateralToe"].body, "toes_l")
        self.assertEqual(right["medialToe"].body, "toes_r")
        self.assertTrue(all(sphere.radius_m == 0.035 for sphere in spheres))

    def test_parameterization_rejects_bounds_and_shape(self):
        """Reject malformed and out-of-range calibration vectors."""
        parameterization = calibration.ContactParameterization()
        with self.assertRaises(ValueError):
            parameterization.decode(np.zeros(2))
        encoded = parameterization.x0
        encoded[0] = parameterization.upper_bounds[0] + 1.0e-6
        with self.assertRaises(ValueError):
            parameterization.decode(encoded)

    def test_custom_builders_reject_topology_mutation(self):
        """Reject calibrated specs that rename, reorder, remove, or reattach a sphere."""
        spheres = list(reference.sphere_specs())
        spheres[0] = reference.SphereSpec(
            spheres[0].side,
            spheres[0].role,
            "renamed",
            spheres[0].force_name,
            spheres[0].body,
            spheres[0].location_m,
            spheres[0].radius_m,
        )
        with self.assertRaises(ValueError):
            reference.build_contact_geometry_xml(spheres=spheres)
        with self.assertRaises(ValueError):
            reference.newton_augmentation_spec(spheres=spheres)

    def test_objective_has_fixed_finite_length_at_low_predicted_load(self):
        """Keep COP residual dimensions finite when a candidate predicts no contact."""
        times = np.linspace(0.0, 0.1, 11)
        measured = np.zeros((len(times), 2, 9))
        measured[..., 1] = 500.0
        measured[..., 3] = 0.1
        contact = np.ones((len(times), 2), dtype=bool)
        parameterization = calibration.ContactParameterization()

        class Evaluator:
            def __call__(self, candidate):
                del candidate
                return calibration.ContactEvaluation(np.zeros_like(measured), np.zeros((len(times), 12)))

        objective = calibration.ContactObjective(times, measured, contact, parameterization, Evaluator())
        first = objective(parameterization.x0)
        second = objective(parameterization.x0)
        self.assertEqual(first.shape, second.shape)
        self.assertTrue(np.all(np.isfinite(first)))
        self.assertGreater(np.linalg.norm(first), 0.0)

    def test_diagnostic_report_writes_figures_and_log(self):
        """Publish viewable evidence with an explicit overall QC status."""
        times = np.linspace(0.0, 0.1, 11)
        measured = np.zeros((len(times), 2, 9))
        measured[..., 1] = 500.0
        measured[..., 3] = 0.1
        predicted = measured.copy()
        side_metrics = {
            "vertical_force": {"peak_relative_error": 0.0, "impulse_relative_error": 0.0},
            "horizontal_force": {"ap_rms_N": 0.0, "ml_rms_N": 0.0},
            "timing": {"onset_error_s": 0.0, "release_error_s": 0.0},
            "cop": {"rms_m": 0.0},
            "gates": {"synthetic_gate": True},
        }
        qc = {
            "passed": True,
            "sides": {"left": side_metrics, "right": side_metrics},
            "global_gates": {"synthetic_global_gate": True},
            "maximum_sphere_penetration_m": 0.0,
        }
        trace = [
            {
                "residual_sumsq": 1.0,
                "term_sumsq": {"force_waveform": 1.0, "penetration_above_0_020_m": 0.0},
            }
        ]
        candidate = calibration.ContactParameterization().decode(calibration.ContactParameterization().x0)
        with tempfile.TemporaryDirectory() as temporary:
            artifacts = calibration.write_diagnostic_report(
                temporary,
                times,
                predicted,
                measured,
                np.zeros((len(times), 12)),
                trace,
                qc,
                candidate,
            )
            self.assertIn("calibration_report.md", artifacts)
            self.assertIn("figures/grf_tracking.png", artifacts)
            self.assertTrue((Path(temporary) / "figures/grf_tracking.png").read_bytes().startswith(b"\x89PNG"))
            self.assertIn("overall_qc=PASS", (Path(temporary) / "run.log").read_text())

    def test_load_inputs_rejects_unaccepted_artifact(self):
        """Reject a missing or non-production RRA calibration source."""
        with self.assertRaises(FileNotFoundError):
            calibration.load_calibration_inputs("/path/that/does/not/exist")


if __name__ == "__main__":
    unittest.main()
