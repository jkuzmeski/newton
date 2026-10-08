# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check pelvis-leg mechanics with shared synthetic runner fixtures."""

import json
import math
import unittest
from pathlib import Path

import numpy as np

from newton.tests.test_digital_shoe import _tiny_artifact
from projects.impedance_instron.cartesian.shoe import Shoe
from projects.impedance_instron.hogan.mechanics import Chain, chain_from_profile, from_leg_coordinates

_PROFILE = {
    "masses_kg": [8.66, 3.45, 1.46],
    "com_local_m": [[0.17, 0.0], [0.17, 0.0], [0.05, 0.0]],
    "inertias_kg_m2": [0.15, 0.05, 0.008],
}


def _reference(hip_z=0.899, duration_s=0.02, frames=21, grf_z=0.0, cop_x=None):
    """Build a static standing reference with a vertical leg and level foot."""
    time = np.linspace(0.0, duration_s, frames)
    state = np.tile([0.0, hip_z, -0.5 * math.pi, 0.0, 0.0], (frames, 1))
    reference = {
        "time_s": time,
        "state": state,
        "velocity": np.zeros_like(state),
        "hip_target_m": state[:, :2].copy(),
        "joint_target_rad": state[:, 3:5].copy(),
        "lengths_m": np.array([0.4, 0.4]),
        "endpoint_local_m": np.array([0.15, -0.05]),
        "static_pitch_rad": np.asarray(0.0),
        "static_ankle_m": np.zeros(2),
        "static_heel_m": np.zeros(2),
        "subject_mass_kg": np.asarray(70.0),
        "grf_time_s": time.copy(),
        "grf_target_n": np.column_stack((np.zeros(frames), np.full(frames, grf_z))),
        "metadata_json": np.asarray(json.dumps({"schema": "cartesian_single_leg_1", "side": "right"})),
    }
    if cop_x is not None:
        reference["cop_target_m"] = np.full(frames, cop_x)
    return reference


def _tiny_shoe(directory: str) -> Shoe:
    """Write the two-column test shoe with its ankle 0.1 m above the column bottoms."""
    raw = _tiny_artifact()
    raw["visual_meshes"] = {
        "fullfoot_last": {
            "vertices_m": [[-0.02, -0.01, 0.02], [0.02, -0.01, 0.02], [0.02, 0.01, 0.02], [-0.02, 0.01, 0.02]],
            "triangles": [[0, 1, 2], [0, 2, 3]],
        }
    }
    raw["instron_fixtures"] = {
        "fullfoot_last": {
            "carrier_anchor_m": [[-0.01, 0.0, 0.02], [0.01, 0.0, 0.02]],
            "foam_free_top_m": [0.02, 0.02],
            "foam_bottom_m": [0.0, 0.0],
            "rest_length_m": [0.02, 0.02],
            "area_m2": [0.0001, 0.0001],
            "neighbors": [[1, -1, -1, -1], [0, -1, -1, -1]],
            "spacing_m": 0.01,
        }
    }
    path = Path(directory) / "shoe.json"
    path.write_text(json.dumps(raw))
    return Shoe(path, [0, 0, 0.1], 0.0)


def _chain() -> Chain:
    """Build a synthetic pelvis-leg chain with fixed inertial properties."""
    return chain_from_profile(_reference(), _PROFILE)


class TestHoganMechanics(unittest.TestCase):
    """Verify active pelvis-leg kinematics and dynamics."""

    def test_free_fall_from_rest(self):
        """Accelerate every body at gravity with no rotation when unloaded at rest."""
        chain = _chain()
        q = np.array([0.1, 0.9, 1.4, 0.3, -0.5, 0.2])
        mass, bias = chain.dynamics(q, np.zeros(6))
        np.testing.assert_allclose(np.linalg.solve(mass, -bias), [0, -9.81, 0, 0, 0, 0], atol=1e-10)

    def test_point_jacobian_matches_finite_differences(self):
        """Differentiate a foot point position consistently with its analytic Jacobian."""
        chain = _chain()
        q = np.array([0.1, 0.9, 1.4, 0.3, -0.5, 0.2])
        local = np.array([0.08, -0.03])
        _, jacobian, _ = chain.point(q, 3, local)
        numeric = np.empty((2, 6))
        for i in range(6):
            step = np.zeros(6)
            step[i] = 1e-6
            numeric[:, i] = (chain.point(q + step, 3, local)[0] - chain.point(q - step, 3, local)[0]) / 2e-6
        np.testing.assert_allclose(jacobian, numeric, atol=1e-8)

    def test_reach_pins_ankle_and_keeps_foot_angle(self):
        """Re-solve hip and knee so the ankle lands on a target without turning the foot."""
        chain = chain_from_profile(_reference(), _PROFILE)
        q, _ = from_leg_coordinates(
            np.array([[0.0, 0.9, -1.3, -0.6, 0.25]]), np.zeros((1, 5)), np.array([1.45]), np.array([0.0])
        )
        target = np.array([[0.05, 0.2]])
        solved = chain.reach(q, target)
        np.testing.assert_allclose(chain.kinematics(solved[0])[2], target[0], atol=1e-12)
        self.assertAlmostEqual(chain.angle(solved[0], 3), chain.angle(q[0], 3), places=12)
        self.assertLess(solved[0, 4], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
