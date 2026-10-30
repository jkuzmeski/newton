# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for replaying Instron last traces through the two-surface column bed."""

import unittest
from dataclasses import replace

import numpy as np
import warp as wp

from projects.digital_instron_v2.core import CALIBRATED_MATERIAL, Material
from projects.digital_instron_v2.last_replay import LastBed, LastReplay, physical
from projects.digital_instron_v2.sweep_fit import fit_material
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
            area_m2=np.full(5, h * h),
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
            area_m2=np.full(len(uv), h * h),
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

    @unittest.skipUnless(wp.is_cuda_available(), "ColumnBed needs CUDA")
    def test_start_offset_deepens_only_its_candidate(self):
        """A candidate's start offset replays as the same stroke made that much deeper."""
        h = 0.005
        bed = LastBed(
            uv_m=np.array([[0.0, 0.0], [h, 0.0]]),
            bottom_m=np.zeros(2),
            gap_m=np.array([0.0, 0.002]),
            rest_m=np.full(2, 0.02),
            area_m2=np.full(2, h * h),
            spacing_m=h,
            tread=np.ones(2, dtype=bool),
        )
        depth = 0.006 * np.sin(np.linspace(0.0, np.pi, 21))[:, None]
        offset = 0.002
        both = LastReplay(bed, depth, 1.0e-3, 2).run([CALIBRATED_MATERIAL] * 2, offsets_m=[0.0, offset])[:, 0]
        plain = LastReplay(bed, depth, 1.0e-3, 1).run([CALIBRATED_MATERIAL])[0, 0]
        deeper = LastReplay(bed, depth + offset, 1.0e-3, 1).run([CALIBRATED_MATERIAL])[0, 0]

        np.testing.assert_allclose(both[0], plain, rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(both[1], deeper, rtol=1.0e-6, atol=1.0e-6)
        self.assertGreater(deeper[0], 1.5 * plain[0])

    @unittest.skipUnless(wp.is_cuda_available(), "ColumnBed needs CUDA")
    def test_slider_dissipates_beyond_the_branch_off_case(self):
        """A cyclic replay with a slider material dissipates about 2*sigma_y*area*stroke beyond the branch-off law."""
        h = 0.005
        bed = LastBed(
            uv_m=np.array([[0.0, 0.0], [h, 0.0]]),
            bottom_m=np.zeros(2),
            gap_m=np.zeros(2),
            rest_m=np.full(2, 0.02),
            area_m2=np.full(2, h * h),
            spacing_m=h,
            tread=np.ones(2, dtype=bool),
        )
        stroke = 0.012
        depth = stroke * np.sin(np.linspace(0.0, np.pi, 81))[:, None]
        # The slider modulus is far above the yield over the stroke, so each leg rides
        # at the yield stress after a tiny clip transient and the slider adds exactly
        # the rate-independent Jenkin dissipation 2*sigma_y*area*stroke.
        branch_off = replace(CALIBRATED_MATERIAL, slider_modulus_pa=0.0, slider_yield_pa=0.0)
        slider = replace(CALIBRATED_MATERIAL, slider_modulus_pa=5.0e7, slider_yield_pa=1.0e4)
        loop = [
            abs(np.trapezoid(LastReplay(bed, depth, 1.0e-3, 1).run([material])[0, 0], depth[:, 0]))
            for material in (branch_off, slider)
        ]

        self.assertGreater(loop[0], 0.0)
        expected = 2.0 * slider.slider_yield_pa * bed.area_m2.sum() * stroke
        self.assertAlmostEqual(loop[1] - loop[0], expected, delta=0.2 * expected)

    @unittest.skipUnless(wp.is_cuda_available(), "ColumnBed needs CUDA")
    def test_disabled_slider_reproduces_the_branch_off_force(self):
        """A material with explicit zero slider fields replays the branch-off force exactly."""
        h = 0.005
        bed = LastBed(
            uv_m=np.array([[0.0, 0.0], [h, 0.0]]),
            bottom_m=np.zeros(2),
            gap_m=np.zeros(2),
            rest_m=np.full(2, 0.02),
            area_m2=np.full(2, h * h),
            spacing_m=h,
            tread=np.ones(2, dtype=bool),
        )
        depth = 0.006 * np.sin(np.linspace(0.0, np.pi, 21))[:, None]
        zeroed = replace(CALIBRATED_MATERIAL, slider_modulus_pa=0.0, slider_yield_pa=0.0)
        force = LastReplay(bed, depth, 1.0e-3, 2).run([CALIBRATED_MATERIAL, zeroed])[:, 0]

        np.testing.assert_array_equal(force[0], force[1])

    @unittest.skipUnless(wp.is_cuda_available(), "ColumnBed needs CUDA")
    def test_fit_replays_in_substeps(self):
        """The fit scores its start on the substepped replay, the runner's contact step."""
        h = 0.005
        bed = LastBed(
            uv_m=np.array([[0.0, 0.0], [h, 0.0]]),
            bottom_m=np.zeros(2),
            gap_m=np.zeros(2),
            rest_m=np.full(2, 0.02),
            area_m2=np.full(2, h * h),
            spacing_m=h,
            tread=np.ones(2, dtype=bool),
        )
        frames = 11
        depth = 0.010 * np.sin(np.linspace(0.0, np.pi, frames))[:, None]
        displacement = depth[:, 0]
        # A slow material lags the 1 ms frame most, so one step and eight steps differ.
        material = replace(CALIBRATED_MATERIAL, maxwell_relaxation_time_s=6.0e-3)
        substepped = LastReplay(bed, depth, 1.0e-3, 1, substeps=8).run([material])[0, 0]
        single = LastReplay(bed, depth, 1.0e-3, 1).run([material])[0, 0]
        self.assertGreater(np.max(np.abs(substepped - single)), 1.0e-3 * np.max(single))
        trace = {"time_s": np.arange(frames) * 1.0e-3, "displacement_m": displacement, "force_n": substepped}

        _, _, records = fit_material(
            bed, depth, 1.0e-3, [trace], [material], evaluations=1, substeps=8, offset_bounds_m=(-1.0e-3, 1.0e-3)
        )

        self.assertAlmostEqual(records[0]["loss"], 0.0, places=10)

    def test_physical_flags_balanced_divergence(self):
        """A diverged bed whose last and platen forces stay balanced is still flagged by its columns."""
        force = np.array([[0.0, 1000.0, 0.0]] * 3)
        reaction = force.copy()
        strain = np.array([0.6, 3.0, 0.6])
        lowest = np.array([-0.5, -0.5, -100.0])

        np.testing.assert_array_equal(physical(force, reaction, strain, lowest), [True, False, False])


if __name__ == "__main__":
    unittest.main(verbosity=2)
