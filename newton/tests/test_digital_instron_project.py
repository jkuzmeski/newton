# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test geometry used by the Digital Instron pipeline."""

import unittest
from pathlib import Path

import numpy as np

from projects.digital_instron_v2.geometry import build_column_grid


class TestDigitalInstronProject(unittest.TestCase):
    def test_raycast_box_thickness(self):
        """Recover a box's uniform thickness from vertical rays."""
        import trimesh

        grid = build_column_grid(trimesh.creation.box(extents=(0.02, 0.02, 0.01)), 0.005)
        np.testing.assert_allclose(grid.slack_m, 0.01)

    def test_prepared_shoe_assets_exist(self):
        """Keep the prepared shoe grids and source scans available for the pipeline."""
        for slug in ("puma-fast-r-nitro-elite-3", "puma-deviate-nitro-elite-3"):
            shoe = Path("DigitalInstron") / slug
            self.assertTrue((shoe / "source.glb").exists())
            for side in ("left", "right"):
                self.assertTrue((shoe / side / "column_grid.npz").exists())

    def test_sweep_fit_inputs_exist(self):
        """Keep raw sweep sources and prepared material fits available."""
        for path in Path("DigitalInstron/FoamSweep").glob("*/*steps.tracking.csv"):
            self.assertTrue(path.exists())
        for slug in ("puma-fast-r-nitro-elite-3", "puma-deviate-nitro-elite-3"):
            self.assertTrue((Path("DigitalInstron/FoamSweep/fit") / slug / "material_fit.json").exists())


if __name__ == "__main__":
    unittest.main()
