# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check full-bed portable Instron replay without measured assets."""

import copy
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
            "bed_bottom_above_platen_m": [0.0, 0.002],
        }
    }
    curve = data["validation"]["curves"][0]
    curve["fixture"] = "fullfoot_last"
    curve["metrics"]["peak_force_n"] = 90.0
    # The exported replay drives the last from first contact: preload plus start offset deeper than the stroke.
    curve["indenter_depth_m"] = [0.003, 0.013, 0.003]
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

    def test_instron_stands_the_bed_in_its_bench_pose(self):
        """Stand every column on the platen at its fitted bench height, not at its shoe-frame bottom."""
        bottoms = self.example.bed.bed.anchor_local.numpy()
        np.testing.assert_allclose(bottoms[:, 2], [0.0, 0.002], atol=1e-9)
        np.testing.assert_allclose(bottoms[:, :2], [[-0.01, 0.0], [0.01, 0.0]], atol=1e-9)

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
                scene = SimpleNamespace(
                    shoe=shoe, fixture_name=name, curve_name=None, _add_instron_indenter_visual=MagicMock()
                )
                anchor, rest, area, neighbors, _, connection, _ = Example._build_instron(scene, MagicMock())
                self.assertEqual(len(rest), 2)
                np.testing.assert_array_equal(connection, [TOP_PRESSED, TOP_FREE])
                np.testing.assert_array_equal(anchor[:, :2], shoe.column_bed.anchor_bottom_m[:, :2])
                np.testing.assert_array_equal(anchor[:, 2], [0.0, 0.002])
                np.testing.assert_array_equal(neighbors, shoe.column_bed.neighbors)
                np.testing.assert_array_equal(area, shoe.column_bed.area_m2)

    def test_instron_drives_the_exported_indenter_depth(self):
        """Drive the last along the depth the exporter replayed, not the raw crosshead stroke."""
        np.testing.assert_allclose(self.example._instron_pose(0.0)[2], -0.003, atol=1e-7)
        np.testing.assert_allclose(self.example._instron_pose(0.5)[2], -0.013, atol=1e-7)

    def test_instron_drives_the_named_curve(self):
        """Select a validation curve by name instead of always the fixture's first curve."""
        data = _fixture_artifact()
        curve = copy.deepcopy(data["validation"]["curves"][0])
        curve["name"] = "deeper"
        curve["indenter_depth_m"] = [0.004, 0.024, 0.004]
        curve["predicted_force_n"] = [0.0, 180.0, 0.0]
        data["validation"]["curves"].append(curve)
        self.path.write_text(json.dumps(data))
        example = Example(
            MagicMock(),
            SimpleNamespace(artifact=self.path, mode="instron", fixture="fullfoot_last", curve="deeper"),
        )
        np.testing.assert_allclose(example._instron_pose(0.5)[2], -0.024, atol=1e-7)
        self.assertEqual(example._expected_peak_force_n, 180.0)

    def _gate_scene(self, peak_n):
        """Return a warmed Instron scene whose runtime curve equals its exported prediction."""
        scene = SimpleNamespace(
            sim_time=3.0,
            _period=0.5,
            sim_dt=0.01,
            _expected_peak_force_n=peak_n,
            _cycle_time=np.array([0.0, 0.25, 0.5]),
            _predicted_force=np.array([0.0, peak_n, 0.0]),
            _peak_force=MagicMock(),
            shoe=SimpleNamespace(shoe_id="test"),
        )
        scene._peak_force.numpy.return_value = np.array([peak_n])
        scene.history = [{"time_s": float(t)} for t in np.arange(2.5, 3.001, 0.05)]
        phase = np.array([row["time_s"] - scene.sim_dt for row in scene.history]) % scene._period
        return scene, np.interp(phase, scene._cycle_time, scene._predicted_force)

    def test_export_curve_gate_uses_force_evaluation_time(self):
        """Compare warm force samples at their pre-integration phase and reject drift."""
        scene, force = self._gate_scene(1000.0)
        Example._test_instron(scene, force)
        with self.assertRaisesRegex(AssertionError, "runtime curve differs"):
            Example._test_instron(scene, force + 100.0)

    def test_instron_gate_accepts_a_light_curve_that_matches_its_prediction(self):
        """Pass an 800 N sweep curve: the gate is the exported prediction, not a fixed force range."""
        scene, force = self._gate_scene(790.0)
        Example._test_instron(scene, force)

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
