# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the contact bridge without motion files or a calibrated shoe download."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import warp as wp

from newton.tests.test_digital_shoe import _tiny_artifact
from projects.digital_shoe.column_bed import TOP_BONDED, TOP_FREE
from projects.impedance_instron.cartesian.shoe import Shoe


def _tiny_shoe_file(directory: str) -> Path:
    """Write a small standalone two-column shoe and fixture."""
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
            "bed_bottom_above_platen_m": [0.0, 0.0],
        }
    }
    path = Path(directory) / "shoe.json"
    path.write_text(json.dumps(raw))
    return path


class TestShoeDevice(unittest.TestCase):
    def test_cpu_device_is_rejected(self):
        """Refuse a CPU shoe because the column bed runs only on CUDA."""
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                Shoe(_tiny_shoe_file(directory), [0, 0, 0.1], 0.0, device="cpu")


@unittest.skipUnless(wp.is_cuda_available(), "The column bed requires CUDA")
class TestShoe(unittest.TestCase):
    """Keep contact a force response rather than prescribed foot motion."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.shoe = Shoe(_tiny_shoe_file(self.directory.name), [0, 0, 0.1], 0.0)

    def test_flight_and_compression(self):
        """Produce zero flight force and an upward response to compression."""
        force, strain = self.shoe.apply([0, 0.2], [0, 0], 0, 0, 0.0001)
        np.testing.assert_array_equal(force, 0)
        self.assertEqual(strain, 0)
        force, strain = self.shoe.apply([0, 0.099], [0, 0], 0, 0, 0.0001)
        self.assertGreater(force[1], 0)
        self.assertAlmostEqual(force[2], 0, places=6)
        self.assertAlmostEqual(strain, 0.001 / 0.02, places=5)

    def test_moving_sole_friction_opposes_motion_within_the_cone(self):
        """Resist a sliding foot with a drag no larger than mu times the ground reaction."""
        actual, _strain = self.shoe.apply([0, 0.099], [0.01, 0], 0, 0, 0.001)
        normal = float(self.shoe.bed.bed.normal_force.numpy()[0])
        self.assertGreater(normal, 0.0)
        self.assertLess(actual[0], 0.0)
        self.assertLessEqual(-actual[0], self.shoe.config.mu * normal * (1.0 + 1e-3))

    def test_pitch_sign(self):
        """Map the Newton +Y torque into the mathematical planar angle convention."""
        force, _ = self.shoe.apply([0, 0.1], [0.2, -0.1], 0.1, 0.3, 0.0001)
        self.assertNotEqual(force[2], 0)
        self.assertAlmostEqual(force[2], -float(self.shoe.state.body_f.numpy()[0, 4]))

    def test_translation_invariance(self):
        """Preserve normal loading under a stationary-ground horizontal translation."""
        a, _ = self.shoe.apply([0, 0.099], [0, 0], 0, 0, 0.0001)
        self.shoe.reset()
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
        """Recover the ankle wrench from the carried column forces rather than a point load."""
        self.shoe.apply([0, 0.099], [0, 0], 0, 0, 0.0001)
        self.assertTrue(np.all(self.shoe.bed.column_state("reaction") > 0))
        self.shoe.reset()
        pitch = 0.05
        wrench, _ = self.shoe.apply([0, 0.099], [0, 0], pitch, 0, 0.0001)
        carried = self.shoe.bed.column_state("carried")[0].astype(float)
        self.assertNotAlmostEqual(carried[0], carried[1])
        anchors = self.shoe.anchor_local_m
        c, s = np.cos(pitch), np.sin(pitch)
        # Carried forces act at the plane point under each nominal bottom.
        x = c * anchors[:, 0] - s * anchors[:, 2]
        moment_y = -np.sum(x * carried)
        np.testing.assert_allclose(wrench, [0, carried.sum(), -moment_y], rtol=1e-5, atol=1e-5)

    def test_bond_footprint_and_free_margin(self):
        """Bond the fixture footprint to the last and leave outer column tops free."""
        path = self.shoe.artifact_path
        raw = json.loads(path.read_text())
        bed = raw["column_bed"]
        bed["anchor_bottom_m"].append([0.03, 0, 0])
        bed["rest_length_m"].append(0.02)
        bed["area_m2"].append(0.0001)
        bed["neighbors"] = [[1, -1, -1, -1], [0, 2, -1, -1], [1, -1, -1, -1]]
        bed["top_bonded"].append(False)
        raw["instron_fixtures"]["fullfoot_last"]["bed_bottom_above_platen_m"].append(0.0)
        path.write_text(json.dumps(raw))
        shoe = Shoe(path, [0, 0, 0.1], 0)
        np.testing.assert_array_equal(shoe.connection, [TOP_BONDED, TOP_BONDED, TOP_FREE])
        self.assertEqual((shoe.metadata["bonded_columns"], shoe.metadata["free_columns"]), (2, 1))
        _, strain = shoe.apply([0, 0.098], [0, 0], 0, 0, 0.0001)
        compression = shoe.bed.column_state("compression")[0]
        self.assertEqual(compression.shape, (3,))
        self.assertGreater(compression[2], 0)
        self.assertLessEqual(compression[2], 0.002 + 1e-7)
        self.assertAlmostEqual(strain, compression[:2].max() / 0.02, places=6)

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
