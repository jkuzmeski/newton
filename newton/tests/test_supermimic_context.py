# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import itertools
import unittest

import numpy as np

from projects.digital_shoe.material import hyperfoam_pressure_numpy
from projects.digital_shoe.runtime import ShoeMaterial
from projects.supermimic.context import GEOMETRY_COUNT, XI_DIM, ShoeSpace, decode, encode

# Fitted foams of dataset_3d_column_bed_v2 (digital_shoe_1 artifacts).
FR3 = ShoeMaterial(
    instantaneous_shear_modulus_pa=571550.6629016441,
    hyperfoam_exponent=23.14278679300506,
    equilibrium_fraction=0.7749540943836701,
    pasternak_n_per_m=0.0,
    maxwell_relaxation_time_s=0.020887948695105442,
    instantaneous_shear_modulus_2_pa=0.02072370272219707,
    hyperfoam_exponent_2=-15.190542009243998,
)
DEV3 = ShoeMaterial(
    instantaneous_shear_modulus_pa=300032.4620302825,
    hyperfoam_exponent=9.404489065909589,
    equilibrium_fraction=0.8247197315825366,
    pasternak_n_per_m=0.0,
    maxwell_relaxation_time_s=0.03659245705535089,
    instantaneous_shear_modulus_2_pa=0.7327244193363989,
    hyperfoam_exponent_2=-19.99829416304294,
)


def _pressure(material: ShoeMaterial, strain: np.ndarray) -> np.ndarray:
    f = material.equilibrium_fraction
    return hyperfoam_pressure_numpy(
        strain,
        material.instantaneous_shear_modulus_pa * f,
        material.hyperfoam_exponent,
        material.instantaneous_shear_modulus_2_pa * f,
        material.hyperfoam_exponent_2,
        0.0,
        1.0,
        0.05,
    )


class TestSupermimicContext(unittest.TestCase):
    def test_encode_decode_round_trip(self):
        """Recover each fitted foam, its pressure curve, friction and geometry from its context."""
        strain = np.linspace(0.0, 0.85, 50)
        for geometry, material in enumerate((FR3, DEV3)):
            xi = encode(material, 0.8, geometry)
            self.assertTrue(np.all(np.abs(xi[:-GEOMETRY_COUNT]) < 0.8))
            (decoded,), friction, (decoded_geometry,) = decode(xi[None])
            self.assertEqual(decoded_geometry, geometry)
            for field in dataclasses.fields(ShoeMaterial):
                np.testing.assert_allclose(
                    getattr(decoded, field.name), getattr(material, field.name), rtol=1e-5, err_msg=field.name
                )
            np.testing.assert_allclose(_pressure(decoded, strain), _pressure(material, strain), rtol=1e-4)
            np.testing.assert_allclose(friction, [0.8], rtol=1e-6)

    def test_box_corners_decode_to_valid_materials(self):
        """Decode every corner of the context box to a valid material."""
        corners = np.array(list(itertools.product((-1.0, 1.0), repeat=XI_DIM - GEOMETRY_COUNT)))
        corners = np.concatenate([corners, np.ones((len(corners), 1)), np.zeros((len(corners), 1))], axis=1)
        materials, friction, _geometry = decode(corners)
        self.assertEqual(len(materials), len(corners))
        self.assertTrue(np.all((friction >= 0.4 - 1e-9) & (friction <= 1.2 + 1e-9)))

    def test_training_excludes_held_out_ball(self):
        """Sample training contexts in the box and outside the ball on both geometries, held-out ones inside it."""
        space = ShoeSpace(held_out=encode(DEV3, 0.8, 1))
        rng = np.random.default_rng(0)
        train = space.sample_training(4096, rng)
        held_out = space.sample_held_out(512, rng)
        self.assertEqual(train.shape, (4096, XI_DIM))
        self.assertEqual(train.dtype, np.float32)
        self.assertTrue(np.all(np.abs(train) <= 1.0))
        self.assertTrue(np.all(np.abs(held_out) <= 1.0))
        self.assertFalse(space.is_held_out(train).any())
        self.assertTrue(space.is_held_out(held_out).all())
        self.assertFalse(space.is_held_out(encode(FR3, 0.8, 0)[None]).any())
        np.testing.assert_allclose(decode(train)[2].mean(), 0.5, atol=0.05)
        self.assertTrue(np.all(decode(held_out)[2] == 1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
