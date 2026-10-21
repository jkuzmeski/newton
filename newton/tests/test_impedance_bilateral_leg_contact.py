# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the leg capsule geometry, the inter-leg penalty kernel, and its opt-in default."""

import unittest

import numpy as np
import warp as wp

from projects.impedance_instron.bilateral import leg_contact as lc
from projects.impedance_instron.bilateral.kinematics3d import Calibration

MASS_KG = (11.7, 8.2, 3.3, 1.4, 8.2, 3.3, 1.4)
CONFIG = lc.LegContactConfig()
"""The shipped profile; the tests below quote its radii and stiffness in their numbers."""


def _calibration() -> Calibration:
    """A static calibration with the two legs mirrored about the sagittal plane."""
    return Calibration(
        frames={},
        origins={},
        templates={},
        offsets={name: np.eye(3) for name in ("hip_r", "hip_l", "knee_r", "knee_l", "ankle_r", "ankle_l")},
        hip_m={"right": np.array([0.0, -0.0945, 0.0]), "left": np.array([0.0, 0.0945, 0.0])},
        knee_m={"right": np.array([0.0, 0.0, -0.38]), "left": np.array([0.0, 0.0, -0.38])},
        ankle_m={"right": np.array([0.0, 0.0, -0.39]), "left": np.array([0.0, 0.0, -0.39])},
    )


def _bed():
    """A 0.10 m wide, 0.20 m long shoe bed of 5 mm columns with 60 mm of foam."""
    xs, ys = np.arange(-0.10, 0.1001, 0.005), np.arange(-0.05, 0.0501, 0.005)
    points = np.array([[x, y, 0.0] for x in xs for y in ys])
    return points, np.full(len(points), 0.06)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _apart(worlds: int = 1) -> np.ndarray:
    """Body origins two metres along +x, so only the pair a test moves can touch."""
    one = np.stack((np.arange(7.0) * 2.0, np.zeros(7), np.zeros(7)), axis=1)
    return np.tile(one, (worlds, 1, 1)).astype(np.float64)


def _rotation(axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Rodrigues rotation matrices about ``axis`` by ``angle``, batched over leading axes."""
    kx, ky, kz = (axis[..., i] for i in range(3))
    zero = np.zeros_like(kx)
    hat = np.stack(
        (np.stack((zero, -kz, ky), -1), np.stack((kz, zero, -kx), -1), np.stack((-ky, kx, zero), -1)),
        -2,
    )
    cosine, sine = np.cos(angle)[..., None, None], np.sin(angle)[..., None, None]
    identity = np.eye(3) * np.ones_like(cosine)
    return identity + sine * hat + (1.0 - cosine) * (hat @ hat)


def _poses(rng, worlds: int, distance: float = 1.0):
    """Random body poses of ``worlds`` worlds as arrays plus the matching rotations."""
    position = rng.normal(0.0, distance, (worlds, 7, 3))
    axis = _unit(rng.normal(size=(worlds, 7, 3)))
    angle = rng.uniform(-np.pi, np.pi, (worlds, 7))
    quat = np.concatenate((axis * np.sin(angle / 2)[..., None], np.cos(angle / 2)[..., None]), axis=-1)
    return position, quat, _rotation(axis, angle)


def _identity(worlds: int = 1) -> np.ndarray:
    quat = np.zeros((worlds, 7, 4))
    quat[..., 3] = 1.0
    return quat


def _contact(config: lc.LegContactConfig, worlds: int) -> lc.LegContact:
    return lc.LegContact(config, [_calibration()], np.zeros(worlds, dtype=int), np.tile(MASS_KG, (worlds, 1)), "cpu")


def _state(worlds: int, position, quat, com=(0.0, 0.0, 0.0), twist=None):
    """The four body arrays the kernel reads and writes."""
    return {
        "body_q": wp.array(
            np.concatenate((position.reshape(-1, 3), quat.reshape(-1, 4)), axis=1).astype(np.float32),
            dtype=wp.transform,
            device="cpu",
        ),
        "body_qd": wp.array(
            (np.zeros((worlds * 7, 6)) if twist is None else np.asarray(twist)).reshape(-1, 6).astype(np.float32),
            dtype=wp.spatial_vector,
            device="cpu",
        ),
        "body_com": wp.array(np.tile(com, (worlds * 7, 1)).astype(np.float32), dtype=wp.vec3, device="cpu"),
        "body_f": wp.zeros(worlds * 7, dtype=wp.spatial_vector, device="cpu"),
    }


class TestCapsuleGeometry(unittest.TestCase):
    def test_segment_capsules_lie_on_the_joint_lines(self):
        """Thigh and shank capsules span the configured fractions of their joint-centre lines."""
        ends = lc.local_capsules(lc.LegContactConfig(span=(0.1, 0.9)), _calibration())
        for body, axis in (
            (1, (0.0, 0.0, -0.38)),
            (2, (0.0, 0.0, -0.39)),
            (4, (0.0, 0.0, -0.38)),
            (5, (0.0, 0.0, -0.39)),
        ):
            np.testing.assert_allclose(ends[body], np.outer((0.1, 0.9), axis))
        np.testing.assert_array_equal(ends[0], np.zeros((2, 3)))

    def test_foot_capsule_is_measured_from_the_shoe_bed(self):
        """The foot capsule takes its span, half-width, and midsole height from the column bed."""
        points, rest = _bed()
        (heel, toe), half_width, height = lc.shoe_envelope(points, rest, (0.0, 0.0, 0.09))
        # The capsule ends sit at 5 % and 85 % of the 200 mm heel-to-toe extent, so it is
        # 10 mm inside the heel and 30 mm short of the toe rather than centred.
        self.assertAlmostEqual(heel, -0.09, places=9)
        self.assertAlmostEqual(toe, 0.07, places=9)
        self.assertAlmostEqual(half_width, 0.05, places=9)
        self.assertAlmostEqual(height, 0.03 - 0.09, places=9)

    def test_shoe_config_never_thickens_the_foot_beyond_the_bed(self):
        """The assumed foot radius survives only where the shoe is at least that wide."""
        points, rest = _bed()
        mount = (0.0, 0.0, 0.09)
        self.assertAlmostEqual(lc.shoe_config(points, rest, mount).radius_m[2], 0.035, places=9)
        narrow = points.copy()
        narrow[:, 1] *= 0.4
        self.assertAlmostEqual(lc.shoe_config(narrow, rest, mount).radius_m[2], 0.02, places=9)
        self.assertEqual(lc.shoe_config(points, rest, mount).radius_m[:2], CONFIG.radius_m[:2])
        self.assertEqual(lc.shoe_config(points, rest, mount).stiffness_n_m, CONFIG.stiffness_n_m)


class TestSegmentGap(unittest.TestCase):
    def test_matches_a_dense_point_sampling(self):
        """Closest-point distances agree with sampling both axes finely."""
        rng = np.random.default_rng(0)
        start = rng.normal(size=(40, 2, 3))
        step = rng.normal(scale=0.4, size=(40, 2, 3))
        for k in range(20):
            step[2 * k + 1] = step[2 * k] + rng.normal(scale=0.02, size=3)
        p0, p1 = start[:, 0], start[:, 0] + step[:, 0]
        q0, q1 = start[:, 1], start[:, 1] + step[:, 1]
        exact = lc.segment_gap(p0, p1, q0, q1)
        sample = np.linspace(0.0, 1.0, 300)
        a = p0[:, None, :] + sample[None, :, None] * (p1 - p0)[:, None, :]
        b = q0[:, None, :] + sample[None, :, None] * (q1 - q0)[:, None, :]
        dense = np.linalg.norm(a[:, :, None, :] - b[:, None, :, :], axis=-1).min((1, 2))
        # A point sampling can only overestimate the true distance, and by no more than
        # about its own spacing along the longer axis.
        self.assertTrue(np.all(exact <= dense + 1e-9))
        self.assertTrue(np.all(exact >= dense - 2e-3))

    def test_parallel_axes_take_the_perpendicular_distance(self):
        """Two parallel axes a fixed distance apart report that distance, not their offset."""
        rng = np.random.default_rng(3)
        axis = _unit(rng.normal(size=(6, 3)))
        start = rng.normal(size=(6, 3))[:, None, :] + rng.uniform(0.4, 1.0, (6, 2, 1)) * axis[:, None, :]
        across = _unit(rng.normal(size=(6, 3)))
        across = _unit(across - np.einsum("...i,...i->...", across, axis)[:, None] * axis)
        offset = rng.uniform(0.05, 0.5, 6)
        lifted = start + offset[:, None, None] * across[:, None, :]
        gap = lc.segment_gap(start[:, 0], start[:, 1], lifted[:, 0], lifted[:, 1])
        np.testing.assert_allclose(gap, offset, atol=1e-9)

    def test_a_shared_point_has_zero_gap(self):
        rng = np.random.default_rng(1)
        points = rng.normal(size=(4, 3))
        points[1] = points[3]
        self.assertAlmostEqual(
            float(lc.segment_gap(points[0:1], points[1:2], points[2:3], points[3:4])[0]), 0.0, places=9
        )


class TestInterLegKernel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()

    def _run(self, config, worlds, position, quat, com=(0.0, 0.0, 0.0), twist=None):
        contact = _contact(config, worlds)
        state = _state(worlds, position, quat, com, twist)
        contact.apply(wp.zeros(worlds, dtype=int, device="cpu"), **state)
        return contact, state["body_f"]

    def test_penetration_matches_the_numpy_reference(self):
        """The kernel's per-world depth equals the NumPy gaps that calibrated the radii."""
        rng = np.random.default_rng(2)
        worlds = 12
        position, quat, rotation = _poses(rng, worlds)
        ends = lc.local_capsules(CONFIG, _calibration())
        axes = position[..., :, None, :] + np.einsum("wfij,fej->wfei", rotation, ends, optimize=True)
        expected = np.maximum(0.0, -lc.capsule_gaps(axes, CONFIG).min(axis=1))
        contact, _ = self._run(CONFIG, worlds, position, quat)
        np.testing.assert_allclose(contact.depth.numpy(), expected, atol=2e-5, rtol=1e-4)

    def test_clearance_applies_no_force_and_records_no_depth(self):
        """Legs a body width apart are untouched, so an upright stance sees no contact."""
        position, quat = _apart(), _identity()
        position[0, 1], position[0, 4] = (0.0, -0.20, 0.9), (0.0, 0.20, 0.9)
        contact, body_f = self._run(CONFIG, 1, position, quat)
        self.assertEqual(float(np.abs(body_f.numpy()).max()), 0.0)
        self.assertEqual(float(contact.depth.numpy()[0]), 0.0)

    def test_penetration_pushes_the_legs_apart_and_balances(self):
        """A 20 mm overlap gives k*depth along the line of centres, equal and opposite."""
        position, quat = _apart(), _identity()
        # Both thigh axes vertical 100 mm apart: the 60 mm radii overlap by 20 mm. The COMs
        # sit at the contact point, so the recorded torque is zero and the force is direct.
        position[0, 1], position[0, 4] = (0.0, -0.05, 0.9), (0.0, 0.05, 0.9)
        contact, body_f = self._run(CONFIG, 1, position, quat, com=(0.0, 0.0, -0.0228))
        force, torque = body_f.numpy()[:, :3], body_f.numpy()[:, 3:]
        expected = CONFIG.stiffness_n_m * 0.02
        self.assertAlmostEqual(float(contact.depth.numpy()[0]), 0.02, places=5)
        np.testing.assert_allclose(force[1], (0.0, -expected, 0.0), atol=2e-2)
        np.testing.assert_allclose(force[4], (0.0, expected, 0.0), atol=2e-2)
        self.assertEqual(float(np.abs(force[[2, 3, 5, 6]]).max()), 0.0)
        self.assertEqual(float(np.abs(force[[0, 1, 2, 3, 4, 5, 6]].sum(0)).max()), 0.0)
        self.assertEqual(float(np.abs(torque[[1, 4]]).max()), 0.0)

    def test_the_pair_force_is_capped(self):
        """A deep overlap from a bad initial state cannot apply more than the cap."""
        position, quat = _apart(), _identity()
        position[0, 1], position[0, 4] = (0.0, -0.01, 0.9), (0.0, 0.01, 0.9)  # 100 mm deep
        contact, body_f = self._run(
            lc.LegContactConfig(maximum_force_n=250.0), 1, position, quat, com=(0.0, 0.0, -0.0228)
        )
        self.assertAlmostEqual(float(contact.depth.numpy()[0]), 2 * CONFIG.radius_m[0] - 0.02, places=5)
        self.assertAlmostEqual(float(abs(body_f.numpy()[1, 1])), 250.0, places=3)

    def test_probe_profile_records_without_touching_body_f(self):
        """``probe`` measures the same penetration as the contact profile and applies nothing."""
        position, quat = _apart(), _identity()
        position[0, 2], position[0, 5] = (0.0, -0.042, 0.5), (0.0, 0.042, 0.5)
        depth = 2 * CONFIG.radius_m[1] - 0.084  # 4 mm, so the elastic force stays under the cap
        contact, body_f = self._run(lc.profile("probe"), 1, position, quat)
        self.assertEqual(float(np.abs(body_f.numpy()).max()), 0.0)
        self.assertAlmostEqual(float(contact.depth.numpy()[0]), depth, places=5)
        contact, body_f = self._run(lc.profile("capsule_v1"), 1, position, quat)
        self.assertAlmostEqual(float(contact.depth.numpy()[0]), depth, places=5)
        self.assertAlmostEqual(float(abs(body_f.numpy()[2, 1])), CONFIG.stiffness_n_m * depth, delta=1.0)

    def test_parallel_axes_touch_along_the_shortest_line(self):
        """Two vertical shanks at different heights collide on their perpendicular distance."""
        position, quat = _apart(), _identity()
        position[0, 2], position[0, 5] = (0.0, -0.042, 0.9), (0.0, 0.042, 0.7)
        contact, body_f = self._run(CONFIG, 1, position, quat)
        self.assertAlmostEqual(float(contact.depth.numpy()[0]), 2 * CONFIG.radius_m[1] - 0.084, places=5)
        self.assertAlmostEqual(float(abs(body_f.numpy()[2, 1])), CONFIG.stiffness_n_m * 0.004, delta=1.0)

    def test_pair_force_stays_in_the_transverse_plane(self):
        """Two axes that only reach each other end to end are still pushed apart sideways."""
        position, quat = _apart(), _identity()
        # The right shank axis ends 50 mm below the left one and 60 mm to its inside, so the
        # shortest line runs at 45 degrees; the 10 mm overlap must still push along y only.
        position[0, 2], position[0, 5] = (0.0, -0.03, 0.9), (0.0, 0.03, 1.317)
        contact, body_f = self._run(CONFIG, 1, position, quat)
        overlap = 2 * CONFIG.radius_m[1] - float(np.sqrt(0.06**2 + 0.0504**2))
        self.assertAlmostEqual(float(contact.depth.numpy()[0]), overlap, places=4)
        np.testing.assert_allclose(body_f.numpy()[2, :3], (0.0, -CONFIG.stiffness_n_m * overlap, 0.0), atol=1.0)

    def test_normal_damping_opposes_closing_without_adhesion(self):
        """Closing adds the damping force, opening subtracts it, and separation never pulls."""
        position, quat = _apart(), _identity()
        position[0, 1], position[0, 4] = (0.0, -0.05, 0.9), (0.0, 0.05, 0.9)
        twist = np.zeros((1, 7, 6))
        twist[0, 1, 1], twist[0, 4, 1] = 0.05, -0.05  # the legs close at 0.1 m/s
        closing, force_closing = self._run(CONFIG, 1, position, quat, com=(0.0, 0.0, -0.0228), twist=twist)
        elastic = CONFIG.stiffness_n_m * 0.02
        damping = float(closing.damping.numpy()[0])
        self.assertAlmostEqual(float(force_closing.numpy()[1, 1]), -(elastic + 0.1 * damping), delta=1.0)
        twist *= -1.0  # the legs open at the same speed
        _, force_opening = self._run(CONFIG, 1, position, quat, com=(0.0, 0.0, -0.0228), twist=twist)
        self.assertAlmostEqual(float(force_opening.numpy()[1, 1]), -(elastic - 0.1 * damping), delta=1.0)
        twist *= 40.0  # the legs separate four times faster than the foam can push
        _, force_separating = self._run(CONFIG, 1, position, quat, com=(0.0, 0.0, -0.0228), twist=twist)
        self.assertEqual(float(np.abs(force_separating.numpy()).max()), 0.0)

    def test_dead_worlds_are_skipped(self):
        """A world the environment has retired gets no force and no depth."""
        rng = np.random.default_rng(4)
        position, quat, _ = _poses(rng, 3)
        contact = _contact(CONFIG, 3)
        state = _state(3, position, quat)
        contact.apply(wp.array([0, 5, 0], dtype=int, device="cpu"), **state)
        self.assertEqual(float(np.abs(state["body_f"].numpy()[14:21]).max()), 0.0)
        self.assertEqual(float(contact.depth.numpy()[1]), 0.0)

    def test_the_recorded_pair_is_the_deepest_one(self):
        position, quat = _apart(), _identity()
        position[0, 1], position[0, 4] = (0.0, -0.05, 0.9), (0.0, 0.05, 0.9)  # 20 mm
        position[0, 2], position[0, 5] = (2.0, -0.01, 0.5), (10.0, 0.01, 0.5)
        contact, _ = self._run(CONFIG, 1, position, quat, com=(0.0, 0.0, -0.0228))
        self.assertEqual(int(contact.pair.numpy()[0]), lc.PAIR_LABELS.index("thigh_r/thigh_l"))
        self.assertAlmostEqual(float(contact.depth.numpy()[0]), 0.02, places=5)


class TestProfiles(unittest.TestCase):
    def test_the_ramp_climbs_to_the_target_and_stops_there(self):
        """A ramped profile starts below its target, climbs with the control steps, and settles."""
        ramp = lc.LegContactConfig(stiffness_n_m=4.0e3, maximum_force_n=120.0, ramp_steps=4)
        contact = _contact(ramp, 1)
        self.assertAlmostEqual(float(contact.gains.numpy()[0]), ramp.ramp_from[0], places=6)
        self.assertAlmostEqual(float(contact.gains.numpy()[1]), ramp.ramp_from[1], places=6)
        contact.clear()
        contact.clear()
        self.assertAlmostEqual(
            float(contact.gains.numpy()[0]), 0.5 * (ramp.ramp_from[0] + ramp.stiffness_n_m), places=3
        )
        for _ in range(4):
            contact.clear()
        self.assertAlmostEqual(float(contact.gains.numpy()[0]), ramp.stiffness_n_m, places=3)
        self.assertAlmostEqual(float(contact.gains.numpy()[1]), ramp.maximum_force_n, places=3)
        self.assertEqual(contact.control_steps, 4)

    def test_a_fixed_profile_never_moves_its_gains(self):
        contact = _contact(lc.profile("capsule_v1"), 1)
        for _ in range(3):
            contact.clear()
        self.assertEqual(contact.control_steps, 0)
        self.assertAlmostEqual(float(contact.gains.numpy()[0]), lc.profile("capsule_v1").stiffness_n_m, places=3)

    def test_only_the_named_profiles_apply_force(self):
        self.assertTrue(lc.profile("probe").measures_only)
        self.assertFalse(lc.profile("capsule_v1").measures_only)
        with self.assertRaises(KeyError):
            lc.profile("capsule_v2")

    def test_the_environment_leaves_leg_contact_off_by_default(self):
        """Existing runs and stored policies must not gain contact they were not trained with."""
        from projects.impedance_instron.bilateral.env3d import Env3DConfig  # noqa: PLC0415

        self.assertEqual(Env3DConfig().leg_contact, "off")
