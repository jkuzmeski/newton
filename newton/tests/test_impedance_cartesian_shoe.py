# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the contact bridge without motion files or a calibrated shoe download."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from newton.tests.test_digital_shoe import _tiny_artifact
from projects.digital_shoe.runtime import FoundationConfig
from projects.impedance_instron.cartesian.shoe import Shoe


class TestCartesianShoe(unittest.TestCase):
    """Keep contact a force response rather than prescribed foot motion."""

    def setUp(self):
        """Write a small standalone two-column shoe and fixture."""
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        raw = _tiny_artifact()
        raw["visual_meshes"] = {
            "fullfoot_last": {
                "vertices_m": [[-0.02, -0.01, 0.02], [0.02, -0.01, 0.02], [0.02, 0.01, 0.02], [-0.02, 0.01, 0.02]],
                "triangles": [[0, 1, 2], [0, 2, 3]],
            }
        }
        raw["instron_fixtures"] = {
            "fullfoot_last": {
                "carrier_anchor_m": [[-0.01, 0.0, 0.02], [0.01, 0.0, 0.02]],
                "foam_free_top_m": [0.02, 0.02],
                "foam_bottom_m": [0.0, 0.0],
                "rest_length_m": [0.02, 0.02],
                "area_m2": [0.0001, 0.0001],
                "neighbors": [[1, -1, -1, -1], [0, -1, -1, -1]],
                "spacing_m": 0.01,
            }
        }
        path = Path(self.directory.name) / "shoe.json"
        path.write_text(json.dumps(raw))
        self.shoe = Shoe(path, [0, 0, 0.1], 0.0)

    def test_flight_and_compression(self):
        """Produce zero flight force and an upward response to compression."""
        force, compression = self.shoe.apply([0, 0.2], [0, 0], 0, 0, 0.0001)
        np.testing.assert_array_equal(force, 0)
        self.assertEqual(compression, 0)
        force, compression = self.shoe.apply([0, 0.099], [0, 0], 0, 0, 0.0001)
        self.assertGreater(force[1], 0)
        self.assertAlmostEqual(force[2], 0, places=6)
        self.assertAlmostEqual(compression, 0.001, places=6)

    def test_elastic_coulomb_is_the_leg_default_and_scales_with_column_geometry(self):
        """Use area-scaled equilibrium shear stiffness without a Maxwell branch."""
        self.assertEqual(self.shoe.foundation.config.friction_model, "elastic_coulomb")
        self.assertEqual(int(self.shoe.foundation.friction_solver.settings.numpy()[0, 0]), 9)
        self.assertEqual(FoundationConfig().friction_model, "elastic_coulomb")
        cfg = FoundationConfig(friction_model="column_maxwell")
        self.assertEqual(cfg.friction_model, "column_maxwell")
        material = self.shoe.shoe.material
        g_eq = material.equilibrium_shear_modulus_pa
        area = np.asarray(self.shoe.shoe.column_bed.area_m2)
        rest = np.asarray(self.shoe.shoe.column_bed.rest_length_m)
        kt = g_eq * area / rest
        dt, speed = 0.001, 0.01
        expected = -np.sum(kt * dt * speed)
        actual, _compression = self.shoe.apply([0, 0.099], [speed, 0], 0, 0, dt)
        np.testing.assert_allclose(actual[0], expected, rtol=1e-5, atol=1e-7)

        explicit_maxwell = Shoe(self.shoe.artifact_path, [0, 0, 0.1], 0.0, friction_model="maxwell")
        prior_force, _compression = explicit_maxwell.apply([0, 0.099], [speed, 0], 0, 0, dt)
        self.assertGreater(abs(prior_force[0]), 10.0 * abs(actual[0]))

        explicit_column_maxwell = Shoe(self.shoe.artifact_path, [0, 0, 0.1], 0.0, friction_model="column_maxwell")
        self.assertEqual(explicit_column_maxwell.foundation.config.friction_model, "column_maxwell")

    def test_pitch_sign_and_virtual_power(self):
        """Map Newton wrench signs into mathematical planar angular power."""
        force, _ = self.shoe.apply([0, 0.1], [0.2, -0.1], 0.1, 0.3, 0.0001)
        self.assertAlmostEqual(force[2], -float(self.shoe.state.body_f.numpy()[0, 4]))
        expected = np.dot(force, [0.2, -0.1, 0.3])
        reported = float(self.shoe.foundation.contact_power.numpy()[0])
        self.assertAlmostEqual(expected, reported, places=5)

    def test_translation_invariance(self):
        """Preserve normal loading under a stationary-ground horizontal translation."""
        a, _ = self.shoe.apply([0, 0.099], [0, 0], 0, 0, 0.0001)
        self.shoe.foundation.reset()
        b, _ = self.shoe.apply([3, 0.099], [0, 0], 0, 0, 0.0001)
        np.testing.assert_allclose(a, b, atol=1e-5)

    def test_reject_ambiguous_footprint(self):
        """Reject duplicate fixture coordinates instead of silently dropping support."""
        path = self.shoe.artifact_path
        raw = json.loads(path.read_text())
        fixture = raw["instron_fixtures"]["fullfoot_last"]
        fixture["carrier_anchor_m"][1] = fixture["carrier_anchor_m"][0]
        path.write_text(json.dumps(raw))
        with self.assertRaisesRegex(ValueError, "unique planar"):
            Shoe(path, [0, 0, 0.1], 0)

    def test_carrier_has_no_visual_or_collision_shapes(self):
        """Keep contact mesh-free while retaining artifact geometry for the fit report."""
        model = self.shoe.model
        self.assertEqual(model.shape_count, 0)
        self.assertAlmostEqual(float(model.body_mass.numpy()[0]), 1.0)
        raw = json.loads(self.shoe.artifact_path.read_text())
        vertices = np.asarray(raw["visual_meshes"]["fullfoot_last"]["vertices_m"])
        np.testing.assert_allclose(self.shoe.last_vertices_local_m + self.shoe.mount_m, vertices)

    def test_distributed_pressure_and_moment(self):
        """Recover the ankle wrench from distributed ground forces rather than a point load."""
        wrench, _ = self.shoe.apply([0, 0.099], [0, 0], 0, 0, 0.0001)
        self.assertTrue(np.all(self.shoe.foundation.ground_force.numpy()[:, 2] > 0))
        wrench, _ = self.shoe.apply([0, 0.099], [0.1, 0], 0.05, 0.1, 0.0001)
        force = self.shoe.foundation.ground_force.numpy().astype(float)
        self.assertNotAlmostEqual(float(force[0, 2]), float(force[1, 2]))
        point = self.shoe.foundation.contact_point.numpy().astype(float)
        total = force.sum(axis=0)
        moment = np.cross(point - [0, 0, 0.099], force).sum(axis=0)
        np.testing.assert_allclose(wrench, [total[0], total[2], -moment[1]], atol=1e-6)

    def test_keep_coupled_passive_margin(self):
        """Retain outer foam columns without rigidly attaching their tops to the last."""
        path = self.shoe.artifact_path
        raw = json.loads(path.read_text())
        bed = raw["column_bed"]
        bed["anchor_bottom_m"].append([0.03, 0, 0])
        bed["rest_length_m"].append(0.02)
        bed["area_m2"].append(0.0001)
        bed["neighbors"] = [[1, -1, -1, -1], [0, 2, -1, -1], [1, -1, -1, -1]]
        path.write_text(json.dumps(raw))
        shoe = Shoe(path, [0, 0, 0.1], 0)
        self.assertEqual(shoe.foundation.free_column_count, 1)
        np.testing.assert_array_equal(shoe.foundation.driven.numpy(), [1, 1, 0])
        shoe.apply([0, 0.098], [0, 0], 0, 0, 0.0001)
        compression = shoe.foundation.compression.numpy()
        self.assertEqual(compression.shape, (3,))
        self.assertGreater(compression[2], 0)
        self.assertLessEqual(compression[2], 0.002 + 1e-7)

    def test_fixed_attachment_offsets_preserve_rest_geometry(self):
        """Keep fixture-to-foam offsets rigid without changing calibrated spring rest lengths."""
        path = self.shoe.artifact_path
        raw = json.loads(path.read_text())
        for point in raw["instron_fixtures"]["fullfoot_last"]["carrier_anchor_m"]:
            point[2] += 0.005
        for vertex in raw["visual_meshes"]["fullfoot_last"]["vertices_m"]:
            vertex[2] += 0.005
        path.write_text(json.dumps(raw))
        shoe = Shoe(path, [0, 0, 0.1], 0)
        np.testing.assert_array_equal(shoe.shoe.column_bed.rest_length_m, self.shoe.shoe.column_bed.rest_length_m)
        np.testing.assert_array_equal(shoe.anchor_local_m, self.shoe.anchor_local_m)
        nominal_top = shoe.anchor_local_m.copy()
        nominal_top[:, 2] += shoe.shoe.column_bed.rest_length_m
        np.testing.assert_allclose(shoe.attachment_local_m - nominal_top, [[0, 0, 0.005]] * 2, atol=1e-12)
        expected = self.shoe.apply([0, 0.099], [0, 0], 0, 0, 0.0001)[0]
        actual = shoe.apply([0, 0.099], [0, 0], 0, 0, 0.0001)[0]
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
