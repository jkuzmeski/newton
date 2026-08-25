# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test the neutral Newton-core C3D contact path."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

import numpy as np

from projects.gait_c3d import newton_contact_calibration as native


class TestNewtonNativeContact(unittest.TestCase):
    """Guard the adapter boundary and Newton core contact semantics."""

    def test_runtime_has_no_opensim_import_or_contact_wrapper(self):
        """Keep OpenSim and OpenSimContact outside the native runtime module."""
        source = Path(native.__file__).read_text()
        tree = ast.parse(source)
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        self.assertFalse(any("opensim" in name.lower() for name in imports))
        self.assertNotIn("OpenSimContact", source)

    def test_native_parameterization_preserves_12_sphere_identity(self):
        """Fit geometry/material values without changing neutral topology identity."""
        topology = {
            "spheres": [
                {
                    "name": f"sphere_{index}",
                    "side": "left" if index < 6 else "right",
                    "role": native._ROLE_ORDER[index % 6],
                    "body": native._BODY_ORDER[(index // 2) % 4],
                    "location_m": [0.01 * index, 0.0, 0.0],
                    "radius_m": 0.035,
                }
                for index in range(12)
            ]
        }
        parameterization = native.NativeParameterization(topology)
        encoded = parameterization.x0.copy()
        encoded[6] = -0.01
        encoded[10:] = (4.2, 12.0, 30.0, 0.4)
        candidate = parameterization.decode(encoded)
        self.assertEqual([item["name"] for item in candidate.spheres], [item["name"] for item in topology["spheres"]])
        self.assertAlmostEqual(candidate.ke, 10**4.2)
        self.assertEqual((candidate.kd, candidate.kf, candidate.mu), (12.0, 30.0, 0.4))

    def test_core_newton_linear_penalty_force(self):
        """Recover the analytic Newton spring plus closing-damping force."""
        poses = np.zeros((1, 4, 7), dtype=float)
        poses[..., 2] = 0.08
        poses[..., 6] = 1.0
        velocities = np.zeros((1, 4, 6), dtype=float)
        velocities[0, 0, 2] = -1.0
        spheres = []
        for index in range(12):
            spheres.append(
                {
                    "name": f"sphere_{index}",
                    "side": "left" if index < 6 else "right",
                    "role": native._ROLE_ORDER[index % 6],
                    "body": native._BODY_ORDER[min(index // 3, 3)],
                    "location_m": [0.0, 0.0, 0.0 if index == 0 else 1.0],
                    "radius_m": 0.1,
                }
            )
        inputs = native.NativeContactInput(
            Path("/synthetic"),
            np.asarray([0.0]),
            poses,
            velocities,
            np.zeros((1, 2, 9)),
            np.zeros((1, 2), dtype=bool),
            {"spheres": spheres},
            {},
        )
        candidate = native.NativeCandidate(tuple(spheres), ke=1000.0, kd=10.0, kf=0.0, mu=0.0)
        evaluation = native.NewtonContactEvaluator(inputs, device="cpu")(candidate)
        self.assertAlmostEqual(evaluation.penetrations_m[0, 0], 0.02, places=6)
        self.assertAlmostEqual(evaluation.foot_wrenches[0, 0, 2], 30.0, places=3)
        self.assertAlmostEqual(evaluation.foot_wrenches[0, 1, 2], 0.0, places=6)

    def test_private_converted_input_is_hash_current(self):
        """Load the real neutral artifact when private Trial 101 data are available."""
        root = Path("/home/jo31399/newton-data/gait/processed/trial_101/newton_contact_input_v1")
        if not root.is_dir():
            self.skipTest("private neutral contact input is unavailable")
        inputs = native.load_native_contact_input(root)
        self.assertEqual(inputs.body_pose.shape, (1077, 4, 7))
        self.assertEqual(inputs.body_velocity.shape, (1077, 4, 6))
        self.assertFalse(inputs.manifest["architecture"]["opensim_used_after_this_boundary"])


if __name__ == "__main__":
    unittest.main()
