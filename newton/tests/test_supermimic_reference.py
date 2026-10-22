# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np

import newton
from projects.supermimic.body import BODY_NAMES, BodyParams, add_body
from projects.supermimic.reference import (
    DeLevaTable,
    _hinge_angle,
    _hip_angles,
    _intervals,
    _matrix_quaternion,
    _rotvec_rotation,
    f01_body_params,
    fit_coordinates,
    forward_kinematics,
    load_body_params,
)


def _random_q(rng: np.random.Generator, count: int) -> np.ndarray:
    rotation = _rotvec_rotation(rng.uniform(-0.3, 0.3, (count, 3)))
    position = rng.uniform(-0.2, 0.2, (count, 3)) + np.array((0.0, 0.0, 0.9))
    return np.concatenate((position, _matrix_quaternion(rotation), rng.uniform(-0.6, 0.6, (count, 10))), axis=1)


class TestSupermimicReference(unittest.TestCase):
    def test_forward_kinematics_matches_newton(self):
        """Match Newton's body poses of the runner articulation with the reference forward kinematics."""
        params = f01_body_params(DeLevaTable.MALE)
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        add_body(builder, params)
        model = builder.finalize(device="cpu")
        state = model.state()
        for q in _random_q(np.random.default_rng(0), 4):
            state.joint_q.assign(q.astype(np.float32))
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)
            body_q = state.body_q.numpy()
            frames = forward_kinematics(q, params)
            for i, name in enumerate(BODY_NAMES):
                rotation, origin = frames[name]
                np.testing.assert_allclose(body_q[i, :3], origin, atol=1e-5)
                expected = _matrix_quaternion(rotation)
                self.assertGreater(abs(float(np.dot(body_q[i, 3:], expected))), 1.0 - 1e-6)

    def test_joint_angles_decompose(self):
        """Recover hip, knee and ankle angles from the relative body rotations."""
        params = BodyParams()
        q = _random_q(np.random.default_rng(1), 50)
        frames = forward_kinematics(q, params)
        for i, side in enumerate(("r", "l")):
            pelvis, thigh, shank, foot = (
                frames[n][0] for n in ("pelvis", f"thigh_{side}", f"shank_{side}", f"foot_{side}")
            )
            angles = q[:, 7 + 5 * i : 12 + 5 * i]
            np.testing.assert_allclose(
                _hip_angles(np.swapaxes(pelvis, -1, -2) @ thigh, side), angles[:, :3], atol=1e-12
            )
            knee = _hinge_angle(np.swapaxes(thigh, -1, -2) @ shank, np.array(params.knee_axis))
            ankle = _hinge_angle(np.swapaxes(shank, -1, -2) @ foot, np.array(params.ankle_axis))
            np.testing.assert_allclose(knee, angles[:, 3], atol=1e-12)
            np.testing.assert_allclose(ankle, angles[:, 4], atol=1e-12)

    def test_fit_coordinates_recovers_pose(self):
        """Fit the coordinates that generated exact marker positions from a perturbed start."""
        params = BodyParams()
        rng = np.random.default_rng(2)
        local = {name: rng.uniform(-0.1, 0.1, (3, 3)) for name in BODY_NAMES}
        q = _random_q(rng, 20)
        frames = forward_kinematics(q, params)
        targets = np.concatenate(
            [frames[s][1][:, None] + np.einsum("fij,mj->fmi", frames[s][0], local[s]) for s in local], axis=1
        )
        start = q.copy()
        start[:, :3] += 0.02
        start[:, 7:] += 0.1
        fitted = fit_coordinates(targets, local, start, params)
        np.testing.assert_allclose(fitted[:, :3], q[:, :3], atol=1e-6)
        np.testing.assert_allclose(fitted[:, 7:], q[:, 7:], atol=1e-6)
        np.testing.assert_allclose(np.abs(np.sum(fitted[:, 3:7] * q[:, 3:7], axis=1)), 1.0, atol=1e-9)

    def test_f01_body_params(self):
        """Build F01 parameters with the measured mass and leg lengths, per de Leva table."""
        male = f01_body_params(DeLevaTable.MALE)
        female = f01_body_params(DeLevaTable.FEMALE)
        for params in (male, female):
            self.assertAlmostEqual(params.mass, 66.5, places=9)
            self.assertGreater(-params.knee_offset[2], 0.35)
            self.assertLess(-params.knee_offset[2], 0.42)
            self.assertLess(params.hip_offset[1], -0.07)
            self.assertGreater(params.pelvis.com[2], 0.2)
        self.assertAlmostEqual(male.thigh.mass, 0.1416 * 66.5)
        self.assertAlmostEqual(female.thigh.mass, 0.1478 * 66.5)

    def test_load_body_params(self):
        """Read back body parameters written as JSON."""
        params = f01_body_params(DeLevaTable.MALE)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "body_params.json"
            path.write_text(json.dumps(asdict(params)), encoding="utf-8")
            self.assertEqual(load_body_params(path), params)

    def test_intervals(self):
        """Return start and end indices of each true run."""
        mask = np.array([1, 1, 0, 0, 1, 0, 1, 1, 1], dtype=bool)
        np.testing.assert_array_equal(_intervals(mask), [[0, 2], [4, 5], [6, 9]])


if __name__ == "__main__":
    unittest.main(verbosity=2)
