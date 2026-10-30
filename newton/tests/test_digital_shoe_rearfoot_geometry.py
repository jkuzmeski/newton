# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Keep one physical shoe bed for both Instron fixture datums."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from newton.tests.test_digital_shoe_consumers import _fixture_artifact
from projects.digital_shoe.artifact import load_artifact
from projects.digital_shoe.column_bed import TOP_FREE, TOP_PRESSED
from projects.digital_shoe.showcase import Example


def _curved_artifact():
    """Describe a nonflat outsole and a legacy zero-bottom rearfoot fixture."""
    data = _fixture_artifact()
    bed = data["column_bed"]
    bed["anchor_bottom_m"][0][2] = 0.006
    bed["anchor_bottom_m"][1][2] = 0.014
    full = data["instron_fixtures"]["fullfoot_last"]
    full["foam_bottom_m"] = [0.006]
    full["foam_free_top_m"] = [0.026]
    full["carrier_anchor_m"] = [[-0.01, 0.0, 0.029]]
    full["bed_bottom_above_platen_m"] = [0.0, 0.008]
    rear = copy.deepcopy(full)
    rear.update(
        foam_bottom_m=[0.0],
        foam_free_top_m=[0.02],
        carrier_anchor_m=[[-0.01, 0.0, 0.02]],
        indenter={"type": "circular_punch", "radius_m": 0.006},
    )
    data["instron_fixtures"]["rearfoot_punch"] = rear
    curve = copy.deepcopy(data["validation"]["curves"][0])
    curve["fixture"] = "rearfoot_punch"
    data["validation"]["curves"].append(curve)
    return data


class TestRearfootShoeGeometry(unittest.TestCase):
    """Verify datum conversion in geometry, rendering and force evaluation."""

    def setUp(self):
        """Load both fixtures against an explicitly curved two-column bed."""
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "shoe.json"
        self.path.write_text(json.dumps(_curved_artifact()))
        self.shoe = load_artifact(self.path)

    def scene(self, fixture):
        """Build fixture inputs without using a viewer or a physical collision mesh."""
        scene = SimpleNamespace(
            shoe=self.shoe, fixture_name=fixture, curve_name=None, _add_instron_indenter_visual=MagicMock()
        )
        return scene, Example._build_instron(scene, MagicMock())

    def test_both_fixtures_preserve_intrinsic_bottom_and_top(self):
        """Use the same nonflat outsole and rest surface for either fixture."""
        bed = self.shoe.column_bed
        for fixture in ("fullfoot_last", "rearfoot_punch"):
            with self.subTest(fixture=fixture):
                scene, (anchor, rest, area, neighbors, _, connection, top_gap) = self.scene(fixture)
                np.testing.assert_array_equal(anchor[:, :2], bed.anchor_bottom_m[:, :2])
                np.testing.assert_array_equal(anchor[:, 2], [0.0, 0.008])
                np.testing.assert_array_equal(rest, bed.rest_length_m)
                np.testing.assert_array_equal(area, bed.area_m2)
                np.testing.assert_array_equal(neighbors, bed.neighbors)
                np.testing.assert_array_equal(connection, [TOP_PRESSED, TOP_FREE])
                self.assertEqual(scene.foundation_config.ground_height_m, 0.0)
                source = self.shoe.instron_fixture(fixture)
                gap = source.carrier_anchor_m[:, 2] - source.foam_free_top_m
                np.testing.assert_allclose(top_gap, [gap[0], 0.0], atol=1e-15)

    def test_legacy_and_physical_fixture_datums_are_equivalent(self):
        """Read old flattened and new physical fixture artifacts without double shifting."""
        old_scene, old_inputs = self.scene("rearfoot_punch")
        data = _curved_artifact()
        rear = data["instron_fixtures"]["rearfoot_punch"]
        rear.update(foam_bottom_m=[0.006], foam_free_top_m=[0.026], carrier_anchor_m=[[-0.01, 0.0, 0.026]])
        self.path.write_text(json.dumps(data))
        self.shoe = load_artifact(self.path)
        new_scene, new_inputs = self.scene("rearfoot_punch")
        for before, after in zip(old_inputs, new_inputs, strict=True):
            np.testing.assert_allclose(after, before, atol=1e-15)
        self.assertEqual(old_scene.foundation_config, new_scene.foundation_config)

    def test_punch_visual_is_centered_over_the_rearfoot_patch(self):
        """Place the punch above the driven heel sites rather than at the shoe origin."""
        builder = MagicMock()
        example = Example(builder, SimpleNamespace(mode="instron", fixture="rearfoot_punch", artifact=self.path))
        points = self.shoe.instron_fixture("rearfoot_punch").carrier_anchor_m[:, :2]
        expected_xy = 0.5 * (points.min(axis=0) + points.max(axis=0))
        shape = example.model.shape_transform.numpy()[1]
        np.testing.assert_allclose(shape[:2], expected_xy, atol=1e-7)
        np.testing.assert_allclose(shape[3:], [0.0, 0.0, 0.0, 1.0], atol=1e-7)

    def test_rest_columns_stand_on_the_curved_outsole(self):
        """Draw unloaded columns from the artifact outsole, floating where it curves off the platen."""
        example = Example(
            MagicMock(),
            SimpleNamespace(mode="instron", fixture="rearfoot_punch", artifact=self.path),
        )
        example.render()
        bed = self.shoe.column_bed
        bench = np.column_stack((bed.anchor_bottom_m[:, :2], [0.0, 0.008])) - [0.0, 0.0, 0.003]
        np.testing.assert_allclose(example._bottoms.numpy(), bench, atol=1e-7)
        np.testing.assert_allclose(example._tops.numpy()[:, 2], bench[:, 2] + bed.rest_length_m, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
