# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import itertools
import unittest

import numpy as np

from projects.digital_shoe.material import hyperfoam_pressure_numpy
from projects.digital_shoe.runtime import ShoeMaterial
from projects.supermimic.context import GEOMETRY_COUNT, XI_DIM, ShoeSpace, decode, encode

# Fitted foams of dataset_3d_column_bed_v3 (refit C).
FR3 = ShoeMaterial(
    instantaneous_shear_modulus_pa=638128.2882609903,
    hyperfoam_exponent=18.656078293186706,
    equilibrium_fraction=0.8321094191331598,
    pasternak_n_per_m=0.0,
    maxwell_relaxation_time_s=0.023136749731871742,
    instantaneous_shear_modulus_2_pa=7028.750144838635,
    hyperfoam_exponent_2=-3.0640556924791653,
)
DEV3 = ShoeMaterial(
    instantaneous_shear_modulus_pa=312211.69995435723,
    hyperfoam_exponent=7.608389725640643,
    equilibrium_fraction=0.8433826667363614,
    pasternak_n_per_m=0.0,
    maxwell_relaxation_time_s=0.03615985968807929,
    instantaneous_shear_modulus_2_pa=12.08748525186517,
    hyperfoam_exponent_2=-14.237670395643558,
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

    def test_training_contexts_lie_in_a_ball_around_the_fit(self):
        """Sample valid training foams and frictions in the ball on the fit's geometry, and out-of-ball contexts on a shell."""
        center = encode(FR3, 0.8, 0)
        space = ShoeSpace(center=center)
        train = space.sample_training(4096, np.random.default_rng(0))
        self.assertEqual(train.shape, (4096, XI_DIM))
        self.assertEqual(train.dtype, np.float32)
        offset = np.linalg.norm(train[:, :-GEOMETRY_COUNT] - center[:-GEOMETRY_COUNT], axis=1)
        self.assertLessEqual(offset.max(), space.radius + 1e-6)
        self.assertGreater(offset.max(), 0.9 * space.radius)
        materials, friction, geometry = decode(train)
        self.assertTrue(all(0.0 < m.equilibrium_fraction < 1.0 and m.hyperfoam_exponent_2 < 0.0 for m in materials))
        self.assertTrue(np.all(friction > 0.0))
        self.assertTrue(np.all(geometry == 0))
        np.testing.assert_array_equal(space.sample_center(3), np.repeat(center[None], 3, axis=0))
        beyond = space.sample_beyond(1024, np.random.default_rng(1))
        offset = np.linalg.norm(beyond[:, :-GEOMETRY_COUNT] - center[:-GEOMETRY_COUNT], axis=1)
        self.assertGreaterEqual(offset.min(), space.beyond[0] - 1e-6)
        self.assertLessEqual(offset.max(), space.beyond[1] + 1e-6)
        np.testing.assert_array_equal(
            beyond[:, -GEOMETRY_COUNT:], np.repeat(center[None, -GEOMETRY_COUNT:], 1024, axis=0)
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
