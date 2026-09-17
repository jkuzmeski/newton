# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check canonical contact, deterministic cycle reduction and source provenance."""

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import warp as wp

from projects.digital_instron_v2.workflow import compression_laplacian
from projects.digital_shoe.contact import normal_reaction, normal_reaction_numpy
from projects.digital_shoe.provenance import physics_source_identity
from projects.digital_shoe.runtime import FoundationParams, cycle_force


@wp.kernel
def _reaction_samples(
    compression: wp.array[float],
    pressure: wp.array[float],
    area: wp.array[float],
    velocity: wp.array[float],
    gap: wp.array[float],
    plane: int,
    force: wp.array[float],
):
    i = wp.tid()
    force[i] = normal_reaction(compression[i], pressure[i], area[i], 3.0, velocity[i], gap[i], plane)


class TestSharedContact(unittest.TestCase):
    """Validate shared laws rather than another independent implementation."""

    def test_host_and_device_use_same_contact_source(self):
        """Require host and device normal reactions to share one function body."""
        self.assertIs(normal_reaction.func.__code__, normal_reaction_numpy.__code__)

    def test_normal_reaction_matches_host_on_cpu_and_cuda(self):
        """Match pressure, damping, lift-off and unilateral clipping across adapters."""
        arrays = [
            np.array(x, dtype=np.float32)
            for x in (
                [0, 0.002, 0.002, 0.002, 0.0],
                [100, 100, -100, 100, 100],
                [0.001] * 5,
                [0, 1, -1, -1, 0],
                [-0.001, -0.001, -0.001, 0.01, 0.01],
            )
        ]
        for device in [wp.get_device("cpu"), *wp.get_cuda_devices()]:
            for plane in (0, 1):
                with self.subTest(device=str(device), plane=plane):
                    values = [wp.array(x, dtype=float, device=device) for x in arrays]
                    out = wp.zeros(5, dtype=float, device=device)
                    wp.launch(_reaction_samples, dim=5, inputs=[*values, plane, out], device=device)
                    expected = normal_reaction_numpy(*arrays[:3], 3.0, arrays[3], arrays[4], plane)
                    np.testing.assert_allclose(out.numpy(), expected, rtol=1.0e-6, atol=1.0e-7)
                    self.assertTrue(np.all(out.numpy() >= 0))

    def test_cycle_force_is_repeatable_and_retains_additive_output(self):
        """Sum each fitted frame in a fixed order while preserving the kernel contract."""
        rng = np.random.default_rng(1227)
        frames, count = 5, 1025
        compression = np.zeros((frames, count), np.float32)
        pressure = np.exp(rng.uniform(-8.0, 12.0, (frames, count))).astype(np.float32)
        slack = np.full(count, 0.02, np.float32)
        for device in [wp.get_device("cpu"), *wp.get_cuda_devices()]:
            with self.subTest(device=str(device)):
                p = FoundationParams()
                p.stretch_floor = 0.05
                p.one_minus_two_poisson = 1.0
                p.alpha = p.alpha2 = 1.0
                inputs = [
                    wp.array(compression, dtype=float, device=device),
                    wp.array(pressure, dtype=float, device=device),
                    wp.array(slack, dtype=float, device=device),
                    p,
                    1.0,
                ]
                out = wp.zeros(frames, dtype=float, device=device)
                reference = None
                for _ in range(10):
                    out.zero_()
                    wp.launch(cycle_force, dim=(frames, count), inputs=[*inputs, out], device=device)
                    actual = out.numpy()
                    if reference is None:
                        reference = actual.copy()
                    else:
                        np.testing.assert_array_equal(actual, reference)
                expected = np.zeros(frames, np.float32)
                for column in range(count):
                    expected += pressure[:, column]
                np.testing.assert_array_equal(reference, expected)
                out.fill_(128.0)
                wp.launch(cycle_force, dim=(frames, count), inputs=[*inputs, out], device=device)
                np.testing.assert_array_equal(out.numpy(), reference + np.float32(128.0))

    def test_physics_fingerprint_covers_shared_dependencies(self):
        """Invalidate checkpoint physics identity when any shared law source changes."""
        baseline = physics_source_identity()
        read = Path.read_bytes
        for filename in (
            "runtime.py",
            "material.py",
            "contact.py",
            "friction_law.py",
            "friction_deflection.py",
            "friction_stribeck.py",
            "friction_pressure.py",
            "friction_slip_history.py",
            "friction_maxwell.py",
            "friction_parameter_adapter.py",
            "friction_solver.py",
            "friction_adapter.py",
        ):
            with self.subTest(filename=filename):

                def changed(path, target=filename):
                    data = read(path)
                    return data + b"\n# changed law" if path.name == target else data

                with patch.object(Path, "read_bytes", changed):
                    self.assertNotEqual(physics_source_identity(), baseline)
        self.assertEqual(physics_source_identity(), baseline)

    def test_missing_subset_neighbor_is_not_rigid_support(self):
        """Give every absent neighbor a free edge in the legacy geometry utility."""
        points = np.array([[0.0, 0.0], [0.005, 0.0]])
        whole = np.array([[0.0, 0.0], [0.005, 0.0], [0.010, 0.0]])
        value = compression_laplacian(np.array([[0.002, 0.002]]), points, whole, 0.005)
        np.testing.assert_array_equal(value, np.zeros((1, 2)))


if __name__ == "__main__":
    unittest.main()
