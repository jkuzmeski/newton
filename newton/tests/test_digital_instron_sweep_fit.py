# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for replaying Instron last traces through the two-surface column bed."""

import unittest

import numpy as np
import warp as wp

from projects.digital_instron_v2.core import CALIBRATED_MATERIAL, Material
from projects.digital_instron_v2.last_replay import LastBed, LastReplay, physical
from projects.digital_shoe.column_bed import BOTTOM_LANDED, TOP_HELD


class TestDigitalInstronSweepFit(unittest.TestCase):
    @unittest.skipUnless(wp.is_cuda_available(), "ColumnBed needs CUDA")
    def test_floating_free_and_conserved_columns(self):
        """Shear holds up floating columns, isolated ends carry nothing, and the load is conserved."""
        h = 0.005
        # 0: tread on the platen, 1: tread floating beside it, 2: outside the tread beside 0,
        # 3: outside the tread alone, 4: tread floating alone.
        bed = LastBed(
            uv_m=np.array([[0.0, 0.0], [h, 0.0], [-h, 0.0], [10 * h, 0.0], [20 * h, 0.0]]),
            bottom_m=np.array([0.0, 0.005, 0.0, 0.0, 0.005]),
            gap_m=np.zeros(5),
            rest_m=np.full(5, 0.02),
            area_m2=h * h,
            spacing_m=h,
            tread=np.array([True, True, False, False, True]),
        )
        depth = 0.003 * np.sin(np.linspace(0.0, np.pi, 101))[:, None]
        force, columns = LastReplay(bed, depth, 1.0e-3, 1, cycles=1).run(
            [CALIBRATED_MATERIAL], snapshot_frames=np.array([50])
        )
        carried, reaction = columns["carried"][0], columns["reaction"][0]
        compression, support = columns["compression"][0], columns["support"][0]

        self.assertEqual(support[0], TOP_HELD | BOTTOM_LANDED)
        self.assertEqual(support[1], TOP_HELD)
        self.assertGreater(carried[1], 0.0)
        self.assertEqual(reaction[1], 0.0)
        self.assertEqual(carried[2], 0.0)
        self.assertGreater(reaction[2], 0.0)
        for alone in (3, 4):
            self.assertEqual(carried[alone], 0.0)
            self.assertEqual(reaction[alone], 0.0)
            self.assertLess(compression[alone], 1.0e-9)

        total = carried.sum()
        self.assertAlmostEqual(reaction.sum(), total, delta=1.0e-4 * total)
        self.assertAlmostEqual(columns["carrier"][0], total, delta=1.0e-5 * total)
        self.assertAlmostEqual(force[0, 0, 50], total, delta=1.0e-5 * total)

    @unittest.skipUnless(wp.is_cuda_available(), "ColumnBed needs CUDA")
    def test_cold_start_jump_stays_physical(self):
        """A last starting already pressed into a stiff bed with high overhangs keeps the bed balanced."""
        h = 0.005
        u, v = np.meshgrid(np.arange(16) * h, np.arange(16) * h)
        uv = np.column_stack((u.ravel(), v.ravel()))
        tread = np.max(np.abs(uv - uv.mean(axis=0)), axis=1) < 3 * h
        bed = LastBed(
            uv_m=uv,
            bottom_m=np.where(tread, 0.0, 0.012),
            gap_m=np.zeros(len(uv)),
            rest_m=np.full(len(uv), 0.008),
            area_m2=h * h,
            spacing_m=h,
            tread=tread,
        )
        stiff = Material(
            instantaneous_shear_modulus_pa=299469,
            hyperfoam_exponent=9.32164,
            equilibrium_fraction=0.825796,
            maxwell_relaxation_time_s=0.0362688,
            instantaneous_shear_modulus_2_pa=6.64752,
            hyperfoam_exponent_2=-15.8274,
        )
        force = LastReplay(bed, np.full((40, 1), 0.003), 1.0e-3, 1, cycles=1).run([stiff])

        self.assertTrue(np.all(np.isfinite(force)))

    def test_physical_flags_balanced_divergence(self):
        """A diverged bed whose last and platen forces stay balanced is still flagged by its columns."""
        force = np.array([[0.0, 1000.0, 0.0]] * 3)
        reaction = force.copy()
        strain = np.array([0.6, 3.0, 0.6])
        lowest = np.array([-0.5, -0.5, -100.0])

        np.testing.assert_array_equal(physical(force, reaction, strain, lowest), [True, False, False])


if __name__ == "__main__":
    unittest.main(verbosity=2)
