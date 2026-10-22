# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check that a column bed pressed by a pitched rigid sole loads every penetrating column."""

import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

from projects.digital_shoe.column_bed import TOP_BONDED, ColumnBed
from projects.digital_shoe.runtime import FoundationConfig, ShoeMaterial

FR3 = ShoeMaterial(
    571550.66,
    23.142787,
    0.77495,
    0.0,
    maxwell_relaxation_time_s=0.020888,
    instantaneous_shear_modulus_2_pa=0.020724,
    hyperfoam_exponent_2=-15.190542,
)
DEV3 = ShoeMaterial(
    300032.46,
    9.404489,
    0.82472,
    0.0,
    maxwell_relaxation_time_s=0.036592,
    instantaneous_shear_modulus_2_pa=0.73272,
    hyperfoam_exponent_2=-19.998294,
)


def _flat_bed(rows: int, cols: int, spacing: float):
    x, y = np.meshgrid(np.arange(rows) * spacing, np.arange(cols) * spacing, indexing="ij")
    anchors = np.stack((x.ravel(), y.ravel() - y.mean(), np.zeros(x.size)), 1)
    index = np.arange(x.size).reshape(rows, cols)
    pad = np.pad(index, 1, constant_values=-1)
    neighbors = np.stack((pad[:-2, 1:-1], pad[2:, 1:-1], pad[1:-1, :-2], pad[1:-1, 2:]), -1).reshape(-1, 4)
    return anchors, neighbors


@unittest.skipUnless(wp.is_cuda_available(), "ColumnBed needs CUDA")
class TestColumnBedPitchedSole(unittest.TestCase):
    def test_deep_columns_carry_ground_reaction(self):
        """A rigid flat sole pitched 15 deg and pressed 6 mm in loads the ground under every deep column."""
        spacing, thickness, depth, pitch = 0.005, 0.03, 0.006, np.radians(15.0)
        anchors, neighbors = _flat_bed(40, 16, spacing)
        n = len(anchors)
        rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), float(pitch))
        height = anchors[:, 0] * np.sin(-pitch)
        for material in (FR3, DEV3):
            with self.subTest(alpha=material.hyperfoam_exponent):
                bed = ColumnBed(
                    anchors,
                    np.full(n, thickness),
                    spacing**2,
                    neighbors,
                    spacing,
                    np.full(n, TOP_BONDED, np.int32),
                    material,
                    FoundationConfig(ground_height_m=0.0),
                    np.array([0]),
                    wp.zeros(1, dtype=wp.vec3),
                )
                bed.set_timesteps(1.25e-4)
                state = SimpleNamespace(
                    body_q=wp.array([[0.0, 0.0, -depth - height.min(), *rotation]], dtype=wp.transform),
                    body_qd=wp.zeros(1, dtype=wp.spatial_vector),
                    body_f=wp.zeros(1, dtype=wp.spatial_vector),
                )
                for _ in range(400):
                    bed.apply(state)
                penetration = -(height - height.min() - depth)
                # Columns pressed in deeper than 2.5 mm must reach the ground; the
                # shallowest rows may hang in their neighbours' shear layer.
                deep = penetration > 2.5e-3
                reaction = bed.bed.reaction.numpy()[deep]
                self.assertGreater(deep.sum(), 40)
                np.testing.assert_array_less(penetration[deep] - 1.0e-6, bed.bed.compression.numpy()[deep])
                self.assertTrue(
                    np.all(reaction > 0.0),
                    f"{np.sum(reaction <= 0.0)} of {deep.sum()} deep columns carry no ground reaction",
                )

    def test_implicit_carrier_gains_no_energy_on_the_wall(self):
        """A heavy block sliding at 2 m/s and dropped at 4 m/s on its edge gains no energy on the wall."""
        spacing, thickness, mass, dt = 0.005, 0.03, 20.0, 1.25e-4
        anchors, neighbors = _flat_bed(12, 6, spacing)
        n = len(anchors)
        com = np.array([anchors[:, 0].mean(), 0.0, 0.03])
        inertia = mass * np.diag([0.002, 0.01, 0.01])
        mobility = wp.zeros(1, dtype=wp.spatial_matrix)
        bed = ColumnBed(
            anchors,
            np.full(n, thickness),
            spacing**2,
            neighbors,
            spacing,
            np.full(n, TOP_BONDED, np.int32),
            FR3,
            FoundationConfig(ground_height_m=0.0),
            np.array([0]),
            wp.array([com], dtype=wp.vec3),
            carrier_mobility=mobility,
        )
        bed.set_timesteps(dt)
        rotation = np.array(wp.quat_to_matrix(wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.4))).reshape(3, 3)
        center = np.array([0.0, 0.0, -(anchors @ rotation.T)[:, 2].min()]) + rotation @ com
        velocity, omega = np.array([2.0, 0.0, -4.0]), np.zeros(3)
        speed = np.linalg.norm(velocity)
        state = SimpleNamespace(
            body_q=wp.zeros(1, dtype=wp.transform),
            body_qd=wp.zeros(1, dtype=wp.spatial_vector),
            body_f=wp.zeros(1, dtype=wp.spatial_vector),
        )
        strain = 0.0
        for _ in range(400):
            world_inertia = rotation @ inertia @ rotation.T
            w = np.zeros((6, 6))
            w[:3, :3] = np.eye(3) / mass
            w[3:, 3:] = np.linalg.inv(world_inertia)
            mobility.assign(w[None].astype(np.float32))
            quat = wp.quat_from_matrix(wp.mat33(*rotation.ravel()))
            state.body_q.assign([[*(center - rotation @ com), *quat]])
            state.body_qd.assign([[*velocity, *omega]])
            bed.apply(state)
            wrench = state.body_f.numpy()[0].astype(np.float64)
            velocity = velocity + dt * (wrench[:3] / mass - np.array([0.0, 0.0, 9.81]))
            omega = omega + dt * np.linalg.solve(world_inertia, wrench[3:])
            center = center + dt * velocity
            angle = np.linalg.norm(omega) * dt
            if angle > 0.0:
                axis = omega / np.linalg.norm(omega)
                k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
                rotation = (np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * k @ k) @ rotation
            strain = max(strain, float(bed.bed.held_strain.numpy()[0]))
            self.assertTrue(np.isfinite(wrench).all())
        self.assertLess(strain, 0.95)
        self.assertLess(abs(velocity[2]), 4.0)
        self.assertLess(np.linalg.norm(velocity), speed)


if __name__ == "__main__":
    unittest.main(verbosity=2)
