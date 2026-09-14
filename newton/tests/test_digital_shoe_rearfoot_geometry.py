# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Keep one physical shoe bed for both Instron fixture datums."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from newton.tests.test_digital_shoe_consumers import _fixture_artifact
from projects.digital_instron_v2.dynamics import _build_rearfoot_geometry
from projects.digital_shoe.artifact import load_artifact
from projects.digital_shoe.runtime import MidsoleFoundation
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
        scene = SimpleNamespace(shoe=self.shoe, fixture_name=fixture, _add_instron_indenter_visual=MagicMock())
        return scene, Example._build_instron(scene, MagicMock())

    def test_both_fixtures_preserve_intrinsic_bottom_and_top(self):
        """Use the same nonflat outsole and rest surface for either fixture."""
        bed = self.shoe.column_bed
        for fixture in ("fullfoot_last", "rearfoot_punch"):
            with self.subTest(fixture=fixture):
                _, (anchor, free, rest, area, neighbors, _) = self.scene(fixture)
                np.testing.assert_allclose(free - rest, bed.anchor_bottom_m[:, 2], atol=1e-15)
                np.testing.assert_allclose(free, bed.anchor_bottom_m[:, 2] + bed.rest_length_m, atol=1e-15)
                np.testing.assert_array_equal(rest, bed.rest_length_m)
                np.testing.assert_array_equal(area, bed.area_m2)
                np.testing.assert_array_equal(neighbors, bed.neighbors)
                source = self.shoe.instron_fixture(fixture)
                gap = source.carrier_anchor_m[:, 2] - source.foam_free_top_m
                np.testing.assert_allclose(anchor[:1, 2] - free[:1], gap, atol=1e-15)

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
        np.testing.assert_array_equal(old_scene.surround_config.driven, new_scene.surround_config.driven)

    def test_native_rearfoot_geometry_preserves_mesh_elevations(self):
        """Keep the same mesh-derived vertical datum when exporting rearfoot fixtures."""
        grid = SimpleNamespace(
            uv_m=np.array([[-0.01, 0.0], [0.01, 0.0]]),
            bottom_m=np.array([0.011, 0.019]),
            top_m=np.array([0.031, 0.039]),
            slack_m=np.array([0.02, 0.02]),
            area_m2=0.0001,
            spacing_m=0.02,
            thickness_axis=2,
        )
        config = {"midsole_mesh": "shoe.obj", "grid": {"rearfoot_length_fraction": 0.2}}
        source = {"indenter": {"radius_m": 0.006}}
        with (
            patch("projects.digital_instron_v2.dynamics.load_mesh"),
            patch("projects.digital_instron_v2.dynamics.rearfoot_center", return_value=np.array([-0.01, 0.0])),
        ):
            geometry = _build_rearfoot_geometry(config, Path("."), grid, source, "shoe.obj")
        np.testing.assert_allclose(geometry.z_bottom_m, [0.0, 0.008], atol=1e-15)
        np.testing.assert_allclose(geometry.z_free_m, [0.02, 0.028], atol=1e-15)
        np.testing.assert_allclose(geometry.surface_m, geometry.z_free_m, atol=1e-15)
        self.assertAlmostEqual(geometry.z_shift_m, 0.011)
        np.testing.assert_array_equal(geometry.driven, [True, False])

    def test_punch_visual_is_centered_over_the_rearfoot_patch(self):
        """Place the punch above the driven heel sites rather than at the shoe origin."""
        builder = MagicMock()
        example = Example(builder, SimpleNamespace(mode="instron", fixture="rearfoot_punch", artifact=self.path))
        points = self.shoe.instron_fixture("rearfoot_punch").carrier_anchor_m[:, :2]
        expected_xy = 0.5 * (points.min(axis=0) + points.max(axis=0))
        shape = example.model.shape_transform.numpy()[1]
        np.testing.assert_allclose(shape[:2], expected_xy, atol=1e-7)
        np.testing.assert_allclose(shape[3:], [0.0, 0.0, 0.0, 1.0], atol=1e-7)

    def test_datum_shift_preserves_vertical_compression_and_force_history(self):
        """Preserve the calibrated vertical bench response when restoring outsole shape."""
        example = Example(MagicMock(), SimpleNamespace(mode="instron", fixture="rearfoot_punch", artifact=self.path))
        actual = example.foundation
        bottom = self.shoe.column_bed.anchor_bottom_m[:, 2]
        flat_anchor = actual.anchor_local.numpy().copy()
        flat_anchor[:, 2] -= bottom
        flat = MidsoleFoundation(
            flat_anchor,
            actual.z_free.numpy() - bottom,
            actual.rest_len.numpy(),
            actual.area.numpy(),
            actual.neighbors.numpy(),
            self.shoe.column_bed.spacing_m,
            self.shoe.material,
            example.carrier,
            example.model.body_com,
            example.foundation_config,
            example.device,
            example.surround_config,
        )
        reference_state = example.model.state()
        for depth in (0.0, 0.002, 0.006, 0.009, 0.003, 0.0):
            pose = np.array([[0.0, 0.0, -depth, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
            for state in (example.state_0, reference_state):
                state.body_q.assign(pose)
                state.body_qd.zero_()
            for _ in range(4):
                actual.apply(example.state_0, 0.001, clear_body_force=True)
                flat.apply(reference_state, 0.001, clear_body_force=True)
            np.testing.assert_allclose(actual.compression.numpy(), flat.compression.numpy(), atol=1e-8, rtol=2e-5)
            np.testing.assert_allclose(
                actual.resultant_force.numpy(), flat.resultant_force.numpy(), atol=2e-5, rtol=2e-5
            )
        example.render()
        np.testing.assert_allclose(example._fixed_bottom.numpy(), self.shoe.column_bed.anchor_bottom_m, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
