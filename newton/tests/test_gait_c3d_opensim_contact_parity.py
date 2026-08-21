# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for official/Newton SmoothSphereHalfSpace contact parity."""

from __future__ import annotations

import importlib
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from projects.gait_c3d import opensim_contact_parity as parity
from projects.gait_c3d import predictive_contact

_REAL_SIDECAR = Path(
    "/home/jo31399/newton-data/gait/processed/trial_101/"
    "stage2_prescribed_contact_calibrated_clean_v2/contact_sidecar.json"
)
_HAS_OFFICIAL = importlib.util.find_spec("opensim") is not None


class TestGaitC3DOpenSimContactParity(unittest.TestCase):
    """Verify deterministic XML, ordering, math, and optional official loading."""

    @staticmethod
    def _sidecar() -> predictive_contact.PredictiveContactSidecar:
        material = predictive_contact.MaterialConfig(
            law="SmoothSphereHalfSpaceForce",
            stiffness=1.25e6,
            dissipation=0.25,
            static_friction=0.8,
            dynamic_friction=0.6,
            viscous_friction=0.05,
            transition_velocity=0.1,
            constant_contact_force=1.0e-5,
            hertz_smoothing=300.0,
            hunt_crossley_smoothing=50.0,
            bounds=dict(predictive_contact._MATERIAL_BOUNDS),
        )
        spheres = []
        for side, body, prefix in (("left", "calcn_l", "l"), ("right", "calcn_r", "r")):
            for index, role in enumerate(predictive_contact._ROLES):
                spheres.append(
                    predictive_contact.SphereConfig(
                        name=f"sphere_{prefix}_{role}",
                        force_name=f"force_{prefix}_{role}",
                        side=side,
                        role=role,
                        body=body,
                        marker=predictive_contact._MARKERS[(side, role)],
                        marker_landmark_m=(0.0, 0.0, 0.0),
                        geometry_seed_method="mean_inverse_vertical_projection_to_ground_tangent",
                        geometry_seed_m=(0.1 * index, 0.02, -0.01),
                        seed_radius_m=0.03,
                        phase_frame_indices=(0,),
                        tangent_residual_rms_m=0.0,
                        tangent_residual_max_abs_m=0.0,
                        center_m=(0.1 * index, 0.02, -0.01),
                        radius_m=0.03,
                        center_displacement_bounds_m=(-0.03, 0.03),
                        radius_bounds_m=(0.01, 0.06),
                    )
                )
        return predictive_contact.PredictiveContactSidecar(
            schema_version="gait_c3d_predictive_contact_sidecar_2",
            source_model_path="model.osim",
            source_model_sha256="0" * 64,
            source_analysis_path="analysis.npz",
            source_analysis_sha256="1" * 64,
            frame="opensim_x_forward_y_up_z_right",
            units={"length": "m", "force": "N", "moment": "N*m", "time": "s"},
            ground=predictive_contact.GroundConfig("contact_ground", -0.003, 0.0, (-0.02, 0.02)),
            material=material,
            spheres=tuple(spheres),
            calibration=predictive_contact.CalibrationConfig("left", "right", 50.0, 200.0, 0.001, {"unused": 1.0}),
            normalization=predictive_contact.NormalizationConfig(700.0, 1.75),
        )

    def test_builds_deterministic_official_sets_in_sidecar_order(self):
        sidecar = self._sidecar()
        geometry_one = parity._xml_bytes(parity.build_contact_geometry_xml(sidecar))
        geometry_two = parity._xml_bytes(parity.build_contact_geometry_xml(sidecar))
        forces_one = parity._xml_bytes(parity.build_force_xml(sidecar))
        forces_two = parity._xml_bytes(parity.build_force_xml(sidecar))
        self.assertEqual(geometry_one, geometry_two)
        self.assertEqual(forces_one, forces_two)

        geometry_root = ET.fromstring(geometry_one)
        geometry_objects = geometry_root.find("./ContactGeometrySet/objects")
        self.assertIsNotNone(geometry_objects)
        self.assertEqual(
            [item.get("name") for item in geometry_objects],
            [sidecar.ground.name, *[sphere.name for sphere in sidecar.spheres]],
        )
        self.assertEqual(geometry_objects[0].findtext("socket_frame"), "/ground")
        self.assertEqual(geometry_objects[1].findtext("socket_frame"), "/bodyset/calcn_l")
        self.assertEqual(geometry_objects[0].findtext("location"), "0 -0.0030000000000000001 0")

        force_root = ET.fromstring(forces_one)
        force_objects = force_root.find("./ForceSet/objects")
        self.assertIsNotNone(force_objects)
        self.assertEqual(
            [item.get("name") for item in force_objects],
            [sphere.force_name for sphere in sidecar.spheres],
        )
        self.assertEqual(
            force_objects[0].findtext("socket_sphere"),
            f"/contactgeometryset/{sidecar.spheres[0].name}",
        )
        self.assertEqual(force_objects[0].findtext("stiffness"), "1250000")

    def test_record_order_and_anatomical_aggregation_are_explicit(self):
        names = ("left_heel", "right_heel", "left_toe")
        sides = ("left", "right", "left")
        elements = []
        for index, name in enumerate(names):
            labels = parity.expected_record_labels(name)
            values = np.arange(12, dtype=float) + 100.0 * index
            elements.append(parity.sphere_record_wrench(name, labels, values))
        values = np.asarray([elements])
        aggregate = parity.aggregate_element_wrenches(values, sides)
        np.testing.assert_array_equal(aggregate[0, 0], values[0, 0] + values[0, 2])
        np.testing.assert_array_equal(aggregate[0, 1], values[0, 1])
        with self.assertRaisesRegex(ValueError, "record labels"):
            parity.sphere_record_wrench(names[0], reversed(parity.expected_record_labels(names[0])), np.arange(12))

    def test_frame_selection_is_deterministic_and_strict(self):
        np.testing.assert_array_equal(parity.select_frame_indices(107), [0, 26, 53, 80, 106])
        np.testing.assert_array_equal(parity.select_frame_indices(4, full_frames=True), [0, 1, 2, 3])
        np.testing.assert_array_equal(parity.select_frame_indices(107, [3, 8, 99]), [3, 8, 99])
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            parity.select_frame_indices(10, [1], full_frames=True)
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            parity.select_frame_indices(10, [2, 2])
        with self.assertRaises(IndexError):
            parity.select_frame_indices(10, [10])

    def test_reports_max_rms_and_predeclared_force_torque_gates(self):
        official = np.zeros((2, 2, 6))
        official[0, 0, 0] = 100.0
        official[1, 1, 5] = 10.0
        newton = official.copy()
        newton[0, 0, 0] += 0.009
        newton[1, 1, 5] -= 0.0009
        difference, metrics = parity.comparison_metrics(official, newton)
        self.assertTrue(metrics["passed"])
        self.assertEqual(metrics["force"]["atol"], 0.001)
        self.assertEqual(metrics["torque"]["atol"], 0.0001)
        self.assertEqual(metrics["force"]["rtol"], 0.0001)
        self.assertAlmostEqual(metrics["force"]["max_abs"], 0.009)
        self.assertLessEqual(metrics["force"]["max_normalized_error"], 1.0)

        newton[1, 0, 2] = 0.0011
        _, failed = parity.comparison_metrics(official, newton)
        self.assertFalse(failed["passed"])
        self.assertFalse(failed["force"]["passed"])

    @unittest.skipUnless(_HAS_OFFICIAL and _REAL_SIDECAR.is_file(), "official OpenSim or Trial 101 fixture unavailable")
    def test_optional_official_sets_load_on_the_original_scaled_model(self):
        opensim = importlib.import_module("opensim")

        sidecar = predictive_contact.load_contact_sidecar(_REAL_SIDECAR)
        model_path = Path(sidecar.source_model_path)
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            geometry_path = tmpdir / "geometry.xml"
            force_path = tmpdir / "forces.xml"
            geometry_path.write_bytes(parity._xml_bytes(parity.build_contact_geometry_xml(sidecar)))
            force_path.write_bytes(parity._xml_bytes(parity.build_force_xml(sidecar)))
            model = parity._load_official_augmented_model(opensim, model_path, geometry_path, force_path)
            state = model.initSystem()
            model.realizeDynamics(state)
            self.assertEqual(
                [
                    model.getContactGeometrySet().get(index).getName()
                    for index in range(model.getContactGeometrySet().getSize())
                ],
                [sidecar.ground.name, *[sphere.name for sphere in sidecar.spheres]],
            )
            for sphere in sidecar.spheres:
                force = opensim.SmoothSphereHalfSpaceForce.safeDownCast(model.getForceSet().get(sphere.force_name))
                self.assertIsNotNone(force)
                labels = force.getRecordLabels()
                self.assertEqual(
                    [labels.get(index) for index in range(labels.getSize())],
                    list(parity.expected_record_labels(sphere.force_name)),
                )


if __name__ == "__main__":
    unittest.main()
