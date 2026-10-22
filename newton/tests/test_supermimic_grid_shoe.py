# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check how a new shoe inherits the measured shoe's mount through the shared last."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from newton.tests.test_digital_shoe_consumers import _fixture_artifact
from projects.supermimic.grid.shoe import register_to_measured


class TestShoeRegistration(unittest.TestCase):
    def test_mount_and_pitch_follow_the_last(self):
        """Move the ankle and the static pitch with the rigid motion of the last."""
        angle = 0.1
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])
        translation = np.array([0.01, -0.002, 0.004])
        measured = _fixture_artifact()
        mesh = measured["visual_meshes"]["fullfoot_last"]
        mesh["vertices_m"] = [[-0.02, -0.01, 0.03], [0.02, -0.01, 0.03], [0.02, 0.01, 0.035], [-0.02, 0.01, 0.04]]
        other = json.loads(json.dumps(measured))
        other["visual_meshes"]["fullfoot_last"]["vertices_m"] = (
            np.asarray(mesh["vertices_m"]) @ rotation.T + translation
        ).tolist()
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ("measured.json", "other.json")]
            for path, data in zip(paths, (measured, other), strict=True):
                path.write_text(json.dumps(data))
            mount, pitch, record = register_to_measured(*paths, [-0.03, 0.0, 0.09], -0.2)
        np.testing.assert_allclose(mount, rotation @ [-0.03, 0.0, 0.09] + translation, atol=1e-12)
        self.assertAlmostEqual(pitch, -0.2 + angle, places=12)
        self.assertLess(record["residual_max_m"], 1e-12)


if __name__ == "__main__":
    unittest.main()
