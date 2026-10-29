# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the explicit column bed: normal support, bristle friction and passivity under a free carrier."""

import unittest
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import warp as wp

from projects.digital_shoe.column_bed import SHEAR_LAYER_FRACTION, TOP_BONDED, TOP_FREE, TOP_PRESSED, ColumnBed
from projects.digital_shoe.material import hyperfoam_pressure_numpy
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

    def _block_on_bed(self, mu: float, normal_damping: float = 0.0, mass: float = 20.0):
        """Return a block glued on a 12 x 6 bed, with its anchors, center of mass, mass and inertia."""
        spacing, thickness = 0.005, 0.03
        anchors, neighbors = _flat_bed(12, 6, spacing)
        n = len(anchors)
        com = np.array([anchors[:, 0].mean(), 0.0, 0.03])
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
        )
        return bed, anchors, com, mass, mass * np.diag([0.002, 0.01, 0.01])

    @staticmethod
    def _run(bed, anchors, com, mass, inertia, rotation, velocity, steps, push=lambda step: np.zeros(6), dt=1.25e-4):
        """Integrate the block on the bed under gravity and a ``push(step)`` wrench by semi-implicit Euler.

        Yield its velocity, the bed wrench and its kinetic plus potential energy gain since the start [J].
        """
        bed.set_timesteps(dt)
        center = np.array([0.0, 0.0, -(anchors @ rotation.T)[:, 2].min()]) + rotation @ com
        omega = np.zeros(3)

        def energy():
            return (
                0.5 * mass * velocity @ velocity
                + 0.5 * omega @ (rotation @ inertia @ rotation.T @ omega)
                + 9.81 * mass * center[2]
            )

        start = energy()
        state = SimpleNamespace(
            body_q=wp.zeros(1, dtype=wp.transform),
            body_qd=wp.zeros(1, dtype=wp.spatial_vector),
            body_f=wp.zeros(1, dtype=wp.spatial_vector),
        )
        for step in range(steps):
            world_inertia = rotation @ inertia @ rotation.T
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
            yield velocity, wrench, energy() - start

    def _assert_inside_the_cone(self, bed, mu):
        """Check that no column carries more friction than ``mu`` times its ground reaction."""
        friction = np.linalg.norm(bed.bed.tangential_load.numpy(), axis=1)
        reaction = bed.bed.reaction.numpy()
        np.testing.assert_array_less(friction, mu * np.maximum(reaction, 0.0) * (1.0 + 1.0e-5) + 1.0e-6)

    @staticmethod
    def _bristle_stiffness(bed):
        """Return the summed bristle stiffness of the bed's columns at the default shear fraction [N/m]."""
        g_eq = FR3.equilibrium_fraction * (FR3.instantaneous_shear_modulus_pa + FR3.instantaneous_shear_modulus_2_pa)
        return float(np.sum(g_eq * bed.bed.area.numpy() / bed.bed.rest_len.numpy()))

    def test_bottoms_stick_below_and_slide_at_the_friction_limit(self):
        """A settled block stays on its bristles under a push below mu W and slides with friction mu W above it."""
        mu, weight, settle, dt = 0.8, 20.0 * 9.81, 3200, 1.25e-4
        for ratio in (0.3, 1.5):
            with self.subTest(ratio=ratio):
                bed, anchors, com, mass, _inertia = self._block_on_bed(mu, normal_damping=10.0)
                # A block that cannot turn, pushed along x: its center moves with its bottoms.
                force = ratio * mu * weight
                wrench = np.array([force, 0.0, 0.0, 0.0, 0.0, 0.0])
                travel, velocity, friction, normal, stuck = 0.0, [], [], [], []
                for step, (v, w, _e) in enumerate(
                    self._run(
                        bed, anchors, com, mass, 1.0e6 * np.eye(3), np.eye(3), np.zeros(3), settle + 800,
                        lambda step, wrench=wrench: wrench if step >= settle else np.zeros(6),
                    )
                ):  # fmt: skip
                    if step < settle:
                        continue
                    self._assert_inside_the_cone(bed, mu)
                    travel += dt * v[0]
                    velocity.append(v[0])
                    friction.append(w[0])
                    normal.append(w[2])
                    stuck.append(bed.bed.stuck.numpy().all())
                friction, normal = np.array(friction), np.array(normal)
                if ratio < 1.0:
                    # A step load stretches the bristles to at most twice their static deflection.
                    self.assertTrue(all(stuck))
                    self.assertLess(abs(travel), 2.0 * force / self._bristle_stiffness(bed))
                else:
                    self.assertFalse(any(stuck[400:]))
                    np.testing.assert_allclose(friction[400:], -mu * normal[400:], rtol=1e-3)
                    self.assertGreater(velocity[-1] - velocity[400], 0.5 * 0.05 * (ratio - 1.0) * mu * 9.81)

    def test_friction_is_continuous_when_a_sliding_block_is_caught(self):
        """A settled block launched at 0.5 m/s slides at mu W, then its bristles catch it without a friction jump."""
        mu, weight, settle = 0.8, 20.0 * 9.81, 3200
        bed, anchors, com, mass, _inertia = self._block_on_bed(mu, normal_damping=10.0)
        launch = 0.5 * mass / 1.25e-4
        friction, stuck = [], []
        for step, (_v, w, _e) in enumerate(
            self._run(
                bed, anchors, com, mass, 1.0e6 * np.eye(3), np.eye(3), np.zeros(3), settle + 1600,
                lambda step: np.array([launch, 0.0, 0.0, 0.0, 0.0, 0.0]) if step == settle else np.zeros(6),
            )
        ):  # fmt: skip
            if step > settle:
                self._assert_inside_the_cone(bed, mu)
                friction.append(w[0])
                stuck.append(bed.bed.stuck.numpy().all())
        friction = np.array(friction)
        self.assertLess(friction.min(), -0.98 * mu * weight)
        sliding = int(np.argmin(friction))
        self.assertFalse(stuck[sliding])
        self.assertTrue(any(stuck[sliding:]))
        jump = np.abs(np.diff(friction)).max()
        self.assertLess(jump, 0.05 * mu * weight, f"largest friction step {jump:.1f} N")

    def test_heel_drop_gains_no_energy_and_holds_at_running_timesteps(self):
        """A 3 kg block landing on its heel at 2 m/s while sliding at 0.5 m/s stays passive and smooth at 1e-4 and 5e-4 s."""
        mu, duration = 0.8, 0.08
        rotation = np.array(wp.quat_to_matrix(wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -0.15))).reshape(3, 3)
        start = np.array([0.5, 0.0, -2.0])
        peaks = {}
        for dt in (1.0e-4, 5.0e-4):
            with self.subTest(dt=dt):
                bed, anchors, com, mass, inertia = self._block_on_bed(mu, mass=3.0)
                gain, normal, friction = -np.inf, [], []
                for _v, wrench, energy in self._run(
                    bed, anchors, com, mass, inertia, rotation, start, round(duration / dt), dt=dt
                ):
                    self.assertTrue(np.isfinite(wrench).all())
                    self._assert_inside_the_cone(bed, mu)
                    gain = max(gain, energy)
                    normal.append(wrench[2])
                    friction.append(wrench[0])
                # The bed's stored energy is left out of the sum, so the sum can only fall.
                self.assertLessEqual(gain, 1.0e-3 * 0.5 * mass * start @ start)
                peaks[dt] = max(normal)
                jump = np.abs(np.diff(friction)).max()
                self.assertLess(jump, 0.1 * mu * peaks[dt], f"largest friction step {jump:.1f} N")
        np.testing.assert_allclose(peaks[5.0e-4], peaks[1.0e-4], rtol=0.05)


def _rimmed_bed(worlds: int):
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
    """A light free carrier on the capped densification wall settles, and the bed never does work."""

    def test_light_carrier_settles_on_the_wall(self):
        """A free 1 kg carrier pressed by stance loads onto a bed with free rim tops settles at the load."""
        loads = np.array([1500.0, 2000.0, 3000.0])
        mass = 1.0
        bed, _anchors = _rimmed_bed(len(loads))
        applied, strain, _speed = _press(bed, loads, mass, 3200)
        self.assertTrue(np.isfinite(applied).all())
        self.assertLess(np.max(strain), 0.92)
        np.testing.assert_allclose(applied[-800:].mean(0), loads + mass * 9.81, rtol=0.03)

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


class TestColumnBedWorlds(unittest.TestCase):
    """Check what a world keeps or gets independently of its neighbors and of the ground."""

    @staticmethod
    def _bed(worlds: int, material=FR3):
        spacing, thickness = 0.005, 0.03
        anchors, neighbors = _flat_bed(6, 4, spacing)
        n = len(anchors)
        return ColumnBed(
            anchors,
            np.full(n, thickness),
            spacing**2,
            neighbors,
            spacing,
            np.full(n, TOP_BONDED, np.int32),
            material,
            FoundationConfig(ground_height_m=0.0, mu=0.5),
            np.arange(worlds),
            wp.zeros(worlds, dtype=wp.vec3),
        )

    @staticmethod
    def _state(heights, wrench=0.0, slide=0.0):
        count = len(heights)
        q = np.zeros((count, 7), np.float32)
        q[:, 2] = heights
        q[:, 6] = 1.0
        qd = np.zeros((count, 6), np.float32)
        qd[:, 0] = slide
        return SimpleNamespace(
            body_q=wp.array(q, dtype=wp.transform),
            body_qd=wp.array(qd, dtype=wp.spatial_vector),
            body_f=wp.array(np.full((count, 6), wrench, np.float32), dtype=wp.spatial_vector),
        )

    def test_masked_reset_returns_only_the_selected_worlds_to_rest(self):
        """Reset the masked world to the unloaded shape without disturbing the other world's history."""
        bed = self._bed(2, replace(FR3, slider_modulus_pa=5.0e7, slider_yield_pa=2.0e4))
        bed.set_timesteps(1.25e-4)
        state = self._state([-0.003, -0.003], slide=0.01)
        for _ in range(20):
            bed.apply(state)
        names = ("top", "bottom", "q_state", "peq_prev", "slider_state", "strain_prev", "deflection")
        before = {name: bed.column_state(name).copy() for name in names}
        self.assertTrue(all(np.any(before[name][0] != 0.0) for name in names[2:]))
        bed.reset(wp.array([True, False], dtype=bool))
        for name in names:
            after = bed.column_state(name)
            np.testing.assert_array_equal(after[0], 0.0)
            np.testing.assert_array_equal(after[1], before[name][1])

    def test_carrier_without_carried_load_gets_no_wrench(self):
        """Overwrite the wrench with zero for a carrier held up off the ground, and load it once it presses in."""
        bed = self._bed(1)
        bed.set_timesteps(1.25e-4)
        floating = self._state([0.05], wrench=1.0)
        bed.apply(floating)
        np.testing.assert_array_equal(floating.body_f.numpy(), 0.0)
        np.testing.assert_array_equal(bed.bed.carried_force.numpy(), 0.0)
        pressed = self._state([-0.003], wrench=1.0)
        for _ in range(5):
            bed.apply(pressed)
        self.assertGreater(pressed.body_f.numpy()[0, 2], 0.0)

    def test_unloaded_bottom_loses_its_bristle_deflection(self):
        """A bottom dragged inside the cone stores its slip, drops it when lifted off and lands again with no friction."""
        dt, slide, steps = 1.25e-4, 0.01, 200
        bed = self._bed(1)
        bed.set_timesteps(dt)
        pressed = self._state([-0.003], slide=slide)
        for _ in range(steps):
            bed.apply(pressed)
        self.assertTrue(bed.column_state("stuck").all())
        np.testing.assert_allclose(bed.column_state("deflection")[0, :, 0], slide * dt * steps, rtol=1.0e-3)
        lifted = self._state([0.01], slide=slide)
        bed.apply(lifted)
        np.testing.assert_array_equal(bed.column_state("deflection"), 0.0)
        np.testing.assert_array_equal(bed.column_state("tangential_load"), 0.0)
        landed = self._state([-0.003])
        bed.apply(landed)
        self.assertGreater(bed.column_state("reaction").min(), 0.0)
        np.testing.assert_array_equal(bed.column_state("tangential_load"), 0.0)
        np.testing.assert_array_equal(landed.body_f.numpy()[0, :2], 0.0)

    def test_bed_rejects_a_max_strain_below_the_stretch_floor(self):
        """Refuse a compression limit that closes columns past the stretch floor, where the foam force is flat."""
        spacing = 0.005
        anchors, neighbors = _flat_bed(2, 2, spacing)
        config = FoundationConfig(ground_height_m=0.0, stretch_floor=0.1)
        args = (anchors, np.full(4, 0.03), spacing**2, neighbors, spacing, np.full(4, TOP_BONDED, np.int32))
        with self.assertRaisesRegex(ValueError, "stretch_floor"):
            ColumnBed(*args, FR3, config, np.array([0]), wp.zeros(1, dtype=wp.vec3), max_strain=0.95)
        ColumnBed(*args, FR3, config, np.array([0]), wp.zeros(1, dtype=wp.vec3), max_strain=0.9)

    def test_shear_layer_joins_a_loaded_and_a_free_column_at_a_fraction_of_the_foam_shear_modulus(self):
        """A free column beside a pressed one is held down by the series layer of modulus ``shear_layer_fraction * G_eq``."""
        self.assertAlmostEqual(SHEAR_LAYER_FRACTION, 0.1)
        self.assertEqual(FR3.shear_layer_fraction, SHEAR_LAYER_FRACTION)
        for material in (FR3, replace(FR3, shear_layer_fraction=0.3)):
            with self.subTest(fraction=material.shear_layer_fraction):
                self._check_shear_layer(material)

    def _check_shear_layer(self, material):
        spacing, depth = 0.005, 0.012
        thickness = np.array([0.03, 0.02])
        anchors, neighbors = _flat_bed(2, 1, spacing)
        bed = ColumnBed(
            anchors,
            thickness,
            spacing**2,
            neighbors,
            spacing,
            np.array([TOP_BONDED, TOP_FREE], np.int32),
            material,
            FoundationConfig(ground_height_m=0.0),
            np.array([0]),
            wp.zeros(1, dtype=wp.vec3),
        )
        bed.set_timesteps(1.25e-4)
        state = self._state([-depth])
        for _ in range(2400):
            bed.apply(state)
        top, compression, reaction = (bed.column_state(name)[0] for name in ("top", "compression", "reaction"))
        g_eq = material.equilibrium_fraction * (
            material.instantaneous_shear_modulus_pa + material.instantaneous_shear_modulus_2_pa
        )
        coupling = 2.0 * (material.shear_layer_fraction * g_eq) * np.prod(thickness) / thickness.sum()
        foam = spacing**2 * hyperfoam_pressure_numpy(
            compression[1] / thickness[1],
            material.equilibrium_fraction * material.instantaneous_shear_modulus_pa,
            material.hyperfoam_exponent,
            material.equilibrium_fraction * material.instantaneous_shear_modulus_2_pa,
            material.hyperfoam_exponent_2,
            0.0,
            1.0,
            0.05,
        )
        self.assertGreater(compression[1], 1.0e-3)
        np.testing.assert_allclose(reaction[1], coupling * (top[0] - top[1]), rtol=1.0e-2)
        np.testing.assert_allclose(foam, reaction[1], rtol=2.0e-2)

    def test_pressed_tops_stay_finite_when_the_carrier_passes_the_rest_thickness(self):
        """Keep pressed-top columns finite when an explicit carrier is driven deeper than one thickness into the plane."""
        spacing, thickness = 0.005, 0.03
        anchors, neighbors = _flat_bed(6, 4, spacing)
        n = len(anchors)
        bed = ColumnBed(
            anchors,
            np.full(n, thickness),
            spacing**2,
            neighbors,
            spacing,
            np.full(n, TOP_PRESSED, np.int32),
            FR3,
            FoundationConfig(ground_height_m=0.0),
            np.array([0]),
            wp.zeros(1, dtype=wp.vec3),
        )
        bed.set_timesteps(1.25e-4)
        state = self._state([-1.5 * thickness])
        for _ in range(10):
            bed.apply(state)
        for name in ("top", "bottom", "compression", "reaction"):
            self.assertTrue(np.isfinite(bed.column_state(name)).all(), name)
        self.assertTrue(np.isfinite(state.body_f.numpy()).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
