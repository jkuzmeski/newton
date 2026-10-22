# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check that a column bed pressed by a pitched rigid sole loads every penetrating column."""

import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

from projects.digital_shoe.column_bed import TOP_BONDED, TOP_FREE, ColumnBed
from projects.digital_shoe.material import SOLID_MODULUS_PA
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

FR3_V3 = ShoeMaterial(
    638128.29,
    18.656078,
    0.83210942,
    0.0,
    maxwell_relaxation_time_s=0.023136750,
    instantaneous_shear_modulus_2_pa=7028.7501,
    hyperfoam_exponent_2=-3.0640557,
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
                # Columns pressed in deeper than 3.5 mm must reach the ground; the
                # shallowest loaded rows may hang in their neighbours' shear layer.
                deep = penetration > 3.5e-3
                reaction = bed.bed.reaction.numpy()[deep]
                self.assertGreaterEqual(deep.sum(), 32)
                np.testing.assert_array_less(penetration[deep] - 1.0e-6, bed.bed.compression.numpy()[deep])
                self.assertTrue(
                    np.all(reaction > 0.0),
                    f"{np.sum(reaction <= 0.0)} of {deep.sum()} deep columns carry no ground reaction",
                )

    def _block_on_bed(self, mu: float, normal_damping: float = 0.0):
        """Return a 20 kg block glued on a 12 x 6 bed with its implicit mobility array."""
        spacing, thickness, mass = 0.005, 0.03, 20.0
        anchors, neighbors = _flat_bed(12, 6, spacing)
        n = len(anchors)
        com = np.array([anchors[:, 0].mean(), 0.0, 0.03])
        mobility = wp.zeros(1, dtype=wp.spatial_matrix)
        bed = ColumnBed(
            anchors,
            np.full(n, thickness),
            spacing**2,
            neighbors,
            spacing,
            np.full(n, TOP_BONDED, np.int32),
            FR3,
            FoundationConfig(ground_height_m=0.0, mu=mu, normal_damping=normal_damping),
            np.array([0]),
            wp.array([com], dtype=wp.vec3),
            carrier_mobility=mobility,
        )
        bed.set_timesteps(1.25e-4)
        return bed, mobility, anchors, com, mass, mass * np.diag([0.002, 0.01, 0.01])

    @staticmethod
    def _run(bed, mobility, anchors, com, mass, inertia, rotation, velocity, steps, push=lambda step: np.zeros(6)):
        """Integrate the block on the bed under gravity and a ``push(step)`` wrench; yield its velocity and the bed wrench."""
        dt = 1.25e-4
        center = np.array([0.0, 0.0, -(anchors @ rotation.T)[:, 2].min()]) + rotation @ com
        omega = np.zeros(3)
        state = SimpleNamespace(
            body_q=wp.zeros(1, dtype=wp.transform),
            body_qd=wp.zeros(1, dtype=wp.spatial_vector),
            body_f=wp.zeros(1, dtype=wp.spatial_vector),
        )
        for step in range(steps):
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
            load = wrench + push(step)
            velocity = velocity + dt * (load[:3] / mass - np.array([0.0, 0.0, 9.81]))
            omega = omega + dt * np.linalg.solve(world_inertia, load[3:])
            center = center + dt * velocity
            angle = np.linalg.norm(omega) * dt
            if angle > 0.0:
                axis = omega / np.linalg.norm(omega)
                k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
                rotation = (np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * k @ k) @ rotation
            yield velocity, wrench

    def test_implicit_carrier_gains_no_energy_on_the_wall(self):
        """A heavy block sliding at 2 m/s and dropped at 4 m/s on its edge gains no energy on the wall."""
        bed, mobility, anchors, com, mass, inertia = self._block_on_bed(0.0)
        rotation = np.array(wp.quat_to_matrix(wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.4))).reshape(3, 3)
        start = np.array([2.0, 0.0, -4.0])
        strain = 0.0
        for velocity, wrench in self._run(bed, mobility, anchors, com, mass, inertia, rotation, start, 400):
            strain = max(strain, float(bed.bed.held_strain.numpy()[0]))
            self.assertTrue(np.isfinite(wrench).all())
            end = velocity
        self.assertLess(strain, 0.95)
        self.assertLess(abs(end[2]), 4.0)
        self.assertLess(np.linalg.norm(end), np.linalg.norm(start))

    def test_bottoms_stick_below_and_slide_at_the_friction_limit(self):
        """A block settled on the bed stands still under a push below mu W and slides with friction mu W above it."""
        mu, weight, settle = 0.8, 20.0 * 9.81, 3200
        for ratio in (0.5, 1.5):
            with self.subTest(ratio=ratio):
                bed, mobility, anchors, com, mass, _inertia = self._block_on_bed(mu, normal_damping=10.0)
                # A block that cannot turn, pushed along x: its center moves with its bottoms.
                force = ratio * mu * weight
                wrench = np.array([force, 0.0, 0.0, 0.0, 0.0, 0.0])
                history = list(
                    self._run(
                        bed, mobility, anchors, com, mass, 1.0e6 * np.eye(3), np.eye(3), np.zeros(3), settle + 800,
                        lambda step, wrench=wrench: wrench if step >= settle else np.zeros(6),
                    )
                )  # fmt: skip
                velocity = np.array([v for v, _w in history[settle + 400 :]])
                friction = np.array([w[0] for _v, w in history[settle + 400 :]])
                normal = np.array([w[2] for _v, w in history[settle + 400 :]])
                stuck = bed.bed.stuck.numpy()
                if ratio < 1.0:
                    self.assertLess(np.abs(velocity[:, 0]).max(), 1.0e-3)
                    self.assertTrue(stuck.all())
                    np.testing.assert_allclose(friction, -force, rtol=1e-2)
                else:
                    self.assertFalse(stuck.any())
                    np.testing.assert_allclose(friction, -mu * normal, rtol=1e-3)
                    self.assertGreater(velocity[-1, 0] - velocity[0, 0], 0.5 * 0.05 * (ratio - 1.0) * mu * 9.81)

    def test_sliding_friction_opposes_the_slip_of_each_bottom(self):
        """A settled block sliding and spinning over the bed gets friction mu R against the slip of every bottom."""
        mu, dt = 0.8, 1.25e-4
        bed, mobility, anchors, com, mass, inertia = self._block_on_bed(mu, normal_damping=10.0)
        mobility.assign(np.diag([1.0 / mass] * 3 + list(1.0 / np.diag(inertia)))[None].astype(np.float32))
        center = np.array([0.0, 0.0, com[2]])
        state = SimpleNamespace(
            body_q=wp.array([[*(center - com), 0.0, 0.0, 0.0, 1.0]], dtype=wp.transform),
            body_qd=wp.zeros(1, dtype=wp.spatial_vector),
            body_f=wp.zeros(1, dtype=wp.spatial_vector),
        )
        speed_z = 0.0
        for _ in range(3200):
            state.body_q.assign([[*(center - com), 0.0, 0.0, 0.0, 1.0]])
            state.body_qd.assign([[0.0, 0.0, speed_z, 0.0, 0.0, 0.0]])
            bed.apply(state)
            speed_z += dt * (state.body_f.numpy()[0, 2] / mass - 9.81)
            center[2] += dt * speed_z
        state.body_q.assign([[*(center - com), 0.0, 0.0, 0.0, 1.0]])
        slide, spin = np.array([0.3, 0.1]), 15.0
        state.body_qd.assign([[*slide, speed_z, 0.0, 0.0, spin]])
        bed.apply(state)
        reaction = bed.bed.reaction.numpy()
        friction = bed.bed.tangential_load.numpy()
        lever = anchors[:, :2] - com[:2]
        slip = slide + spin * np.stack((-lever[:, 1], lever[:, 0]), 1)
        loaded = reaction > 0.0
        self.assertGreater(loaded.sum(), 40)
        self.assertGreater(np.linalg.norm(slip, axis=1)[loaded].min(), 0.05)
        direction = -slip / np.linalg.norm(slip, axis=1, keepdims=True)
        np.testing.assert_allclose(np.sum(friction * direction, 1)[loaded], mu * reaction[loaded], rtol=1.0e-2)
        self.assertFalse(bed.bed.stuck.numpy().any())


def _rimmed_bed(worlds: int, mobility=None):
    """Return a v3 FR3 bed of 40 x 16 columns, one per world, whose two thin outer rings have free tops."""
    spacing = 0.005
    anchors, neighbors = _flat_bed(40, 16, spacing)
    rim = np.zeros((40, 16), bool)
    rim[:2] = rim[-2:] = rim[:, :2] = rim[:, -2:] = True
    connection = np.where(rim.ravel(), TOP_FREE, TOP_BONDED).astype(np.int32)
    bed = ColumnBed(
        anchors,
        np.where(rim.ravel(), 0.012, 0.03),
        spacing**2,
        neighbors,
        spacing,
        connection,
        FR3_V3,
        FoundationConfig(ground_height_m=0.0, mu=0.8),
        np.arange(worlds),
        wp.zeros(worlds, dtype=wp.vec3),
        carrier_mobility=mobility,
        max_strain=0.9,
    )
    bed.set_timesteps(1.25e-4)
    return bed, anchors


def _press(bed, loads: np.ndarray, mass: float, steps: int):
    """Press 1 kg-class rigid carriers, one per world, with constant ``loads`` [N]; return the bed force and strain per step."""
    dt = 1.25e-4
    worlds = len(loads)
    thickness = bed.bed.rest_len.numpy()
    state = SimpleNamespace(body_f=wp.zeros(worlds, dtype=wp.spatial_vector))
    height = np.zeros(worlds)
    speed = np.zeros(worlds)
    applied, strain = [], []
    for _ in range(steps):
        q = np.zeros((worlds, 7), np.float32)
        q[:, 2] = height
        q[:, 6] = 1.0
        qd = np.zeros((worlds, 6), np.float32)
        qd[:, 2] = speed
        state.body_q = wp.array(q, dtype=wp.transform)
        state.body_qd = wp.array(qd, dtype=wp.spatial_vector)
        bed.apply(state)
        force = state.body_f.numpy()[:, 2]
        speed += dt * (force - loads) / mass - dt * 9.81
        height += dt * speed
        applied.append(force)
        strain.append((bed.column_state("compression") / thickness).max(1))
    return np.array(applied), np.array(strain), speed


@unittest.skipUnless(wp.is_cuda_available(), "ColumnBed needs CUDA")
class TestColumnBedWall(unittest.TestCase):
    """Shear layers that follow compression onto the densification wall must not create energy."""

    def test_light_carrier_settles_on_the_wall(self):
        """A 1 kg rigid carrier pressed by stance loads onto a bed with free rim tops settles at the load."""
        loads = np.array([1500.0, 2000.0, 3000.0])
        mass, dt = 1.0, 1.25e-4
        rows = np.diag([1.0 / mass] * 3 + [1.0e-6] * 3)
        mobility = wp.array(np.tile(rows, (len(loads), 1, 1)), dtype=wp.spatial_matrix)
        bed, _anchors = _rimmed_bed(len(loads), mobility)
        thickness = bed.bed.rest_len.numpy()
        state = SimpleNamespace(body_f=wp.zeros(len(loads), dtype=wp.spatial_vector))
        height = np.zeros(len(loads))
        speed = np.zeros(len(loads))
        applied, strain = [], []
        for _ in range(3200):
            q = np.zeros((len(loads), 7), np.float32)
            q[:, 2] = height
            q[:, 6] = 1.0
            qd = np.zeros((len(loads), 6), np.float32)
            qd[:, 2] = speed
            state.body_q = wp.array(q, dtype=wp.transform)
            state.body_qd = wp.array(qd, dtype=wp.spatial_vector)
            bed.apply(state)
            force = state.body_f.numpy()[:, 2]
            speed += dt * (force - loads) / mass - dt * 9.81
            height += dt * speed
            applied.append(force)
            strain.append((bed.column_state("compression") / thickness).max())
        applied = np.array(applied)
        self.assertTrue(np.isfinite(applied).all())
        self.assertLess(np.max(strain), 0.92)
        np.testing.assert_allclose(applied[-800:].mean(0), loads + mass * 9.81, rtol=0.03)

    def test_carrier_pressed_hard_never_passes_the_compression_stop(self):
        """A 1 kg carrier pressed with 4 kN stays bounded and never closes a column past the stop.

        The 3 x 3 bed cannot carry 4 kN below the stop, so its columns end up on it.
        """
        mass, load = 1.0, 4000.0
        rows = np.diag([1.0 / mass] * 3 + [1.0e-6] * 3)
        mobility = wp.array(rows[None], dtype=wp.spatial_matrix)
        rimmed, _anchors = _rimmed_bed(1, mobility)
        spacing, thickness = 0.005, 0.03
        anchors, neighbors = _flat_bed(3, 3, spacing)
        n = len(anchors)
        small = ColumnBed(
            anchors,
            np.full(n, thickness),
            spacing**2,
            neighbors,
            spacing,
            np.full(n, TOP_BONDED, np.int32),
            FR3_V3,
            FoundationConfig(ground_height_m=0.0, mu=0.8),
            np.array([0]),
            wp.zeros(1, dtype=wp.vec3),
            carrier_mobility=mobility,
        )
        small.set_timesteps(1.25e-4)
        for name, bed in (("rimmed", rimmed), ("small", small)):
            with self.subTest(bed=name):
                applied, strain, speed = _press(bed, np.array([load]), mass, 3200)
                stop = bed.bed.max_strain
                self.assertTrue(np.isfinite(applied).all())
                self.assertLess(np.abs(applied).max(), 1.0e6)
                # The load acts after the carrier solve, so the stop is held to one step of its speed gain.
                self.assertLessEqual(strain.max(), stop + 3.0e-3)
                np.testing.assert_allclose(applied[-800:].mean(), load + mass * 9.81, rtol=0.03)
                if name == "small":
                    self.assertGreater(strain.max(), stop - 0.01)
                    self.assertLess(abs(speed[0]), 0.05)

    def test_carrier_let_go_past_the_stop_leaves_with_the_foam_energy(self):
        """A carrier held past the stop and let go gains no more than the foam stores beyond the stop.

        Pushing the carrier back to the stop in one step would launch it at the
        overshoot over the step, far more than the foam between the two strains stores.
        """
        mass, dt = 1.0, 1.25e-4
        rows = np.diag([1.0 / mass] * 3 + [1.0e-6] * 3)
        spacing, thickness = 0.005, 0.03
        anchors, neighbors = _flat_bed(3, 3, spacing)
        n = len(anchors)
        bed = ColumnBed(
            anchors,
            np.full(n, thickness),
            spacing**2,
            neighbors,
            spacing,
            np.full(n, TOP_BONDED, np.int32),
            FR3_V3,
            FoundationConfig(ground_height_m=0.0, mu=0.8),
            np.arange(2),
            wp.zeros(2, dtype=wp.vec3),
            carrier_mobility=wp.array(np.tile(rows, (2, 1, 1)), dtype=wp.spatial_matrix),
        )
        bed.set_timesteps(dt)
        below, past = bed.bed.max_strain - 0.01, bed.bed.max_strain + 0.02
        height = -np.array([below, past]) * thickness
        speed = np.zeros(2)
        state = SimpleNamespace(body_f=wp.zeros(2, dtype=wp.spatial_vector))
        # Hold both carriers still until the overstress has relaxed, then let go.
        for step in range(round(0.4 / dt)):
            q = np.zeros((2, 7), np.float32)
            q[:, 2] = height
            q[:, 6] = 1.0
            qd = np.zeros((2, 6), np.float32)
            qd[:, 2] = speed
            state.body_q = wp.array(q, dtype=wp.transform)
            state.body_qd = wp.array(qd, dtype=wp.spatial_vector)
            bed.apply(state)
            force = state.body_f.numpy()[:, 2]
            if step == round(0.2 / dt) - 1:
                held = force[0]
            if step >= round(0.2 / dt):
                speed += dt * force / mass
                height += dt * speed
        self.assertTrue((height > 0.0).all())
        gained = 0.5 * mass * (speed[1] ** 2 - speed[0] ** 2)
        # The foam between the two strains, with its tangent at most the solid modulus.
        room = (past - below) * thickness
        stored = room * (held + 0.5 * n * spacing**2 * SOLID_MODULUS_PA * room / thickness)
        self.assertGreater(gained, 0.0)
        self.assertLessEqual(gained, stored)

    def test_bed_does_no_work_over_a_closed_press_and_tilt(self):
        """A carrier driven around a closed loop of depth and pitch gets no work out of the bed, either way round."""
        bed, anchors = _rimmed_bed(2)
        dt, period, depth, tilt = 1.25e-4, 0.2, 0.024, 0.05
        center = anchors.mean(0)
        state = SimpleNamespace(body_f=wp.zeros(2, dtype=wp.spatial_vector))
        work = np.zeros(2)
        steps = round(period / dt)
        direction = np.array([1.0, -1.0])
        previous = None
        for step in range(steps + 1):
            phase = 2.0 * np.pi * step / steps
            z = -0.5 * depth * (1.0 - np.cos(phase)) * np.ones(2)
            pitch = tilt * np.sin(direction * phase)
            q = np.zeros((2, 7), np.float32)
            qd = np.zeros((2, 6), np.float32)
            for w in range(2):
                rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), float(pitch[w]))
                # Pitch about the bed center at the anchor plane.
                offset = np.array(wp.quat_rotate(rotation, wp.vec3(*(-center))))
                q[w, :3] = center + offset + np.array([0.0, 0.0, z[w]])
                q[w, 3:] = np.array(rotation)
            state.body_q = wp.array(q, dtype=wp.transform)
            state.body_qd = wp.array(qd, dtype=wp.spatial_vector)
            bed.apply(state)
            wrench = state.body_f.numpy()
            # The bed's force acts at the carrier origin, its torque about y turns the pitch.
            pose = np.concatenate((q[:, :3], pitch[:, None]), 1)
            load = np.concatenate((wrench[:, :3], wrench[:, 4:5]), 1)
            if previous is not None:
                work += (0.5 * (load + previous[1]) * (pose - previous[0])).sum(1)
            previous = (pose, load)
        self.assertTrue(np.isfinite(work).all())
        self.assertLessEqual(work.max(), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
