# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check full-bed portable Instron replay without measured assets."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from newton.tests.test_digital_shoe import _tiny_artifact
from projects.digital_shoe.artifact import load_artifact
from projects.digital_shoe.column_bed import TOP_FREE, TOP_PRESSED
from projects.digital_shoe.showcase import Example


def _fixture_artifact():
    """Add a one-column indenter above the existing two-column synthetic bed."""
    data = _tiny_artifact()
    data["visual_meshes"] = {
        "fullfoot_last": {
            "vertices_m": [[-0.02, -0.01, 0.03], [0.02, -0.01, 0.03], [0.02, 0.01, 0.03], [-0.02, 0.01, 0.03]],
            "triangles": [[0, 1, 2], [0, 2, 3]],
        }
    }
    data["instron_fixtures"] = {
        "fullfoot_last": {
            "carrier_anchor_m": [[-0.01, 0.0, 0.02]],
            "foam_free_top_m": [0.02],
            "foam_bottom_m": [0.0],
            "rest_length_m": [0.02],
            "area_m2": [0.0001],
            "neighbors": [[-1, -1, -1, -1]],
            "spacing_m": 0.01,
        }
    }
    curve = data["validation"]["curves"][0]
    curve["fixture"] = "fullfoot_last"
    curve["metrics"]["simulated_peak_force_n"] = 90.0
    return data


class TestDigitalShoeConsumers(unittest.TestCase):
    """Press the indenter footprint of the whole column bed in the Virtual Instron."""

    def setUp(self):
        """Create an artifact whose passive column is absent from the indenter mapping."""
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "shoe.json"
        self.path.write_text(json.dumps(_fixture_artifact()))
        self.example = Example(
            MagicMock(), SimpleNamespace(artifact=self.path, mode="instron", fixture="fullfoot_last")
        )

    def test_instron_keeps_whole_bed_and_free_neighbor(self):
        """Keep the unpressed column instead of truncating at the fixture footprint."""
        bed = self.example.bed.bed
        self.assertEqual(self.example.bed.column_count, 2)
        self.assertEqual(self.example.foundation_config.ground_height_m, 0.0)
        np.testing.assert_array_equal(bed.connection.numpy(), [TOP_PRESSED, TOP_FREE])
        np.testing.assert_array_equal(bed.neighbors.numpy(), [[1, -1, -1, -1], [0, -1, -1, -1]])
        np.testing.assert_allclose(bed.rest_len.numpy(), [0.02, 0.02])
        np.testing.assert_allclose(bed.area.numpy(), [0.0001, 0.0001])

    def test_both_fixture_adapters_keep_the_same_neighbor_bed(self):
        """Keep the full bed for both full-foot and rearfoot fixture adapters."""
        data = _fixture_artifact()
        data["instron_fixtures"]["rearfoot_punch"] = data["instron_fixtures"]["fullfoot_last"].copy()
        curve = data["validation"]["curves"][0].copy()
        curve["fixture"] = "rearfoot_punch"
        data["validation"]["curves"].append(curve)
        self.path.write_text(json.dumps(data))
        shoe = load_artifact(self.path)
        for name in ("fullfoot_last", "rearfoot_punch"):
            with self.subTest(fixture=name):
                scene = SimpleNamespace(shoe=shoe, fixture_name=name, _add_instron_indenter_visual=MagicMock())
                anchor, rest, area, neighbors, _, connection, _ = Example._build_instron(scene, MagicMock())
                self.assertEqual(len(rest), 2)
                np.testing.assert_array_equal(connection, [TOP_PRESSED, TOP_FREE])
                np.testing.assert_array_equal(anchor, shoe.column_bed.anchor_bottom_m)
                np.testing.assert_array_equal(neighbors, shoe.column_bed.neighbors)
                np.testing.assert_array_equal(area, shoe.column_bed.area_m2)

    def test_export_curve_gate_uses_force_evaluation_time(self):
        """Compare warm force samples at their pre-integration phase and reject drift."""
        scene = SimpleNamespace(
            sim_time=3.0,
            _period=0.5,
            sim_dt=0.01,
            _expected_peak_force_n=1000.0,
            _cycle_time=np.array([0.0, 0.25, 0.5]),
            _predicted_force=np.array([0.0, 1000.0, 0.0]),
            _peak_force=MagicMock(),
            shoe=SimpleNamespace(shoe_id="test"),
        )
        scene._peak_force.numpy.return_value = np.array([1000.0])
        scene.history = [{"time_s": float(t)} for t in np.arange(2.5, 3.001, 0.05)]
        phase = np.array([row["time_s"] - scene.sim_dt for row in scene.history]) % scene._period
        force = np.interp(phase, scene._cycle_time, scene._predicted_force)
        Example._test_instron(scene, force)
        with self.assertRaisesRegex(AssertionError, "runtime curve differs"):
            Example._test_instron(scene, force + 100.0)

    def test_instron_free_top_uses_solved_compression(self):
        """Render the shear-loaded free top rather than rigid indenter translation."""
        example = self.example
        example.sim_time = 0.5
        example.step()
        example.render()
        compression = example.bed.column_state("compression")[0]
        self.assertGreater(float(compression[1]), 0.0)
        self.assertLess(float(compression[1]), float(compression[0]))
        bottoms = example._bottoms.numpy()
        tops = example._tops.numpy()
        np.testing.assert_allclose(bottoms[:, 2], 0.0, atol=1.0e-7)
        np.testing.assert_allclose(tops[:, 2], bottoms[:, 2] + 0.02 - compression, atol=1.0e-7)
        self.assertGreater(float(tops[1, 2]), float(tops[0, 2]))


if __name__ == "__main__":
    unittest.main()
