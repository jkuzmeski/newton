# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the post-densification (bottoming-out) branch of the shoe column law.

The identified Ogden-Hill series is evaluated at a floored stretch, so on its own
the runtime holds the column pressure constant while a foot drives the column
crushed further. :func:`projects.digital_shoe.material.densification_multiplier`
adds the missing branch. These tests pin the two properties that make the branch
usable: it changes nothing inside the range the identification measured, and it
bounds a crushed column with a continuous, finite, rising force.
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import get_test_devices
from projects.digital_shoe import material, runtime
from projects.digital_shoe.runtime import FoundationConfig, FoundationParams, MidsoleFoundation, ShoeMaterial

# Onset at the stretch floor, solid-like cap at a tenth of it, as in the runner.
ONSET = 0.05
LIMIT = 0.005
GAIN = 3.0
FOAM = ShoeMaterial(
    instantaneous_shear_modulus_pa=75475.0,
    hyperfoam_exponent=0.2449,
    equilibrium_fraction=0.6953,
    pasternak_n_per_m=1563.0,
    maxwell_relaxation_time_s=0.005,
    instantaneous_shear_modulus_2_pa=16.6,
    hyperfoam_exponent_2=1.7883,
)
STRAIN = np.linspace(-0.2, 1.4, 3201, dtype=np.float32)


@wp.kernel
def _forward(strain: wp.array[wp.float32], p: FoundationParams, out: wp.array[wp.float32]):
    """Evaluate the runtime's forward pressure entry point."""
    i = wp.tid()
    out[i] = runtime._hyperfoam_pressure(strain[i], p)


def _params(gain: float) -> FoundationParams:
    """Fill a device-side block with the measured foam and a chosen branch gain."""
    params = FoundationParams()
    runtime.set_material_block(params, FOAM)
    params.inv_h2 = 1.0
    params.stretch_floor = ONSET
    params.densify_onset = ONSET
    params.densify_limit = LIMIT
    params.densify_gain = gain
    return params


def _device_pressure(params: FoundationParams, strain: np.ndarray, device) -> np.ndarray:
    values = wp.array(strain, dtype=wp.float32, device=device)
    out = wp.zeros_like(values)
    wp.launch(_forward, dim=len(strain), inputs=[values, params, out], device=device)
    return out.numpy()


def _column_foundation(device, gain: float, rest: float, area: float) -> MidsoleFoundation:
    """Build a one-column ground-plane foundation whose carrier can be driven down."""
    builder = newton.ModelBuilder()
    body = builder.add_body(mass=1.0, com=wp.vec3(0.0), inertia=wp.mat33(np.eye(3)))
    model = builder.finalize(device=device)
    foundation = MidsoleFoundation(
        np.zeros((1, 3), np.float32),
        np.zeros(1, np.float32),
        np.array([rest], np.float32),
        np.array([area], np.float32),
        np.full((1, 4), -1, np.int32),
        0.005,
        FOAM,
        body,
        model.body_com,
        FoundationConfig(ground_height_m=0.0, stretch_floor=ONSET, densify_gain=gain),
        device,
    )
    foundation.reset()
    return foundation, model


class TestShoeDensification(unittest.TestCase):
    """Verify the branch is inert where the law is measured and stiff where it is not."""

    def test_branch_is_exactly_one_below_the_onset(self):
        """Below the clip the branch must not move a single bit of the identified law."""
        multiplier = material.densification_multiplier_numpy(1.0 - STRAIN.astype(np.float64), ONSET, LIMIT, GAIN)
        below = STRAIN <= 1.0 - ONSET
        np.testing.assert_array_equal(multiplier[below], np.ones(below.sum()))
        for device in get_test_devices():
            clipped = _device_pressure(_params(0.0), STRAIN, device)
            branched = _device_pressure(_params(GAIN), STRAIN, device)
            with self.subTest(device=str(device), region="identified range is bitwise equal"):
                self.assertTrue(np.array_equal(clipped[below], branched[below]))
            with self.subTest(device=str(device), region="crushed range is stiffer"):
                self.assertTrue(np.all(branched[~below] >= clipped[~below]))

    def test_crushed_column_hardens_toward_a_solid_like_cap(self):
        """Replace the flat plateau of the clipped law with a rising, bounded force."""
        strain = np.linspace(0.0, 1.4, 1401, dtype=np.float64)
        pressure = material.hyperfoam_pressure_numpy(
            strain,
            float(FOAM.equilibrium_shear_modulus_pa),
            FOAM.hyperfoam_exponent,
            0.0,
            1.0,
            0.0,
            1.0,
            ONSET,
        )
        multiplier = material.densification_multiplier_numpy(1.0 - strain, ONSET, LIMIT, GAIN)
        crushed = strain >= 1.0 - ONSET
        with self.subTest("plateau of the clipped law"):
            self.assertEqual(np.ptp(pressure[crushed & (strain < 1.0)]), 0.0)
        branch = pressure * multiplier
        with self.subTest("branch rises monotonically"):
            self.assertTrue(np.all(np.diff(branch) >= 0.0))
            rising = (strain > 1.0 - ONSET) & (strain < 1.0 - LIMIT)
            self.assertTrue(np.all(np.diff(branch[rising]) > 0.0))
        index = int(np.argmin(np.abs(strain - (1.0 - LIMIT))))
        with self.subTest("cap is reached at the pack limit"):
            self.assertAlmostEqual(multiplier[index], 1.0 + GAIN, delta=1.0e-9)
            self.assertAlmostEqual(pressure[index] * multiplier[index], (1.0 + GAIN) * pressure[index])
        with self.subTest("finite past full compression"):
            self.assertTrue(np.all(np.isfinite((pressure * multiplier)[strain >= 1.0])))
            np.testing.assert_array_equal(multiplier[strain >= 1.0], np.full((strain >= 1.0).sum(), 1.0 + GAIN))

    def test_branch_keeps_force_and_tangent_continuous(self):
        """A stiffness step at either end would ring the column at every footfall."""
        remaining = np.linspace(0.0, 2.0 * ONSET, 4001, dtype=np.float64)
        multiplier = material.densification_multiplier_numpy(remaining, ONSET, LIMIT, GAIN)
        slope = np.gradient(multiplier, remaining)
        peak = float(np.abs(slope).max())
        self.assertGreater(peak, 0.0)
        self.assertLessEqual(multiplier.max(), 1.0 + GAIN)
        self.assertGreaterEqual(multiplier.min(), 1.0)
        for edge in (ONSET, LIMIT):
            with self.subTest(stretch=edge):
                i = int(np.argmin(np.abs(remaining - edge)))
                self.assertLess(float(abs(slope[i])), 0.02 * peak)
                self.assertLess(float(abs(slope[i + 1] - slope[i - 1])), 0.05 * peak)
        self.assertLess(float(np.abs(np.diff(slope)).max()), 0.05 * peak)

    def test_config_keeps_the_branch_below_the_clip(self):
        """The default must reproduce the published artifact, and an onset above the
        stretch floor would double-count what the Hyperfoam series already carries."""
        self.assertEqual(FoundationConfig().densify_gain, 0.0)
        for kwargs in (
            {"densify_gain": -1.0},
            {"densify_gain": 3.0, "densify_limit": 0.0},
            {"densify_gain": 3.0, "densify_limit": 0.06},
            {"densify_gain": 3.0, "densify_onset": 1.0},
            {"densify_gain": 3.0, "densify_onset": 0.05, "densify_limit": 0.05},
            {"densify_gain": 3.0, "densify_onset": 0.04, "densify_limit": 0.05},
        ):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    FoundationConfig(**kwargs)
        # An onset above the clip is legal: it hardens the tail of the fitted series.
        FoundationConfig(densify_onset=0.12, densify_limit=0.03, densify_gain=8.0)

    def test_foundation_keeps_the_branch_through_a_material_swap(self):
        """Deep crushes get the branch, shallow ones stay bitwise equal, and swapping
        the foam for a randomized batch must not drop the branch."""
        device = wp.get_device("cpu")
        rest, area = 0.032, 2.5e-5
        softer = ShoeMaterial(
            instantaneous_shear_modulus_pa=FOAM.instantaneous_shear_modulus_pa * 0.7,
            hyperfoam_exponent=FOAM.hyperfoam_exponent,
            equilibrium_fraction=FOAM.equilibrium_fraction,
            pasternak_n_per_m=FOAM.pasternak_n_per_m * 0.7,
            maxwell_relaxation_time_s=FOAM.maxwell_relaxation_time_s,
            instantaneous_shear_modulus_2_pa=FOAM.instantaneous_shear_modulus_2_pa * 0.7,
            hyperfoam_exponent_2=FOAM.hyperfoam_exponent_2,
        )
        forces = {}
        for gain in (0.0, GAIN):
            foundation, model = _column_foundation(device, gain, rest, area)
            foundation.set_world_materials([softer])
            block = foundation.world_blocks[0]
            self.assertEqual(float(block.densify_gain), gain)
            self.assertEqual(float(block.densify_onset), ONSET)
            self.assertAlmostEqual(float(block.g_eq), 0.7 * FOAM.instantaneous_shear_modulus_pa * FOAM.equilibrium_fraction, delta=1.0e-3)
            state = model.state()
            for fraction in (0.5, 0.98):
                state.body_q.assign(np.array([[0.0, 0.0, -fraction * rest, 0.0, 0.0, 0.0, 1.0]], np.float32))
                state.body_qd.zero_()
                state.clear_forces()
                foundation.apply(state, 1.0e-4)
                forces[(gain, fraction)] = float(foundation.resultant_force.numpy()[0, 2])
        with self.subTest("shallow crush bitwise unchanged by the branch"):
            self.assertEqual(forces[(0.0, 0.5)], forces[(GAIN, 0.5)])
        with self.subTest("deep crush resisted"):
            self.assertGreater(forces[(GAIN, 0.98)], 3.0 * forces[(0.0, 0.98)])
            self.assertTrue(np.isfinite(forces[(GAIN, 0.98)]))


if __name__ == "__main__":
    unittest.main()
