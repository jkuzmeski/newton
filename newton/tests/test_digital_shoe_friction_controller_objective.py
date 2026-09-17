# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test phase objectives and raw controller scoring primitives."""

import os
import unittest
from pathlib import Path

import numpy as np
import warp as wp

from projects.digital_shoe.friction_controller_objective import _phase_segment, phase_targets


@wp.kernel
def _segments(a: wp.array[wp.float64], b: wp.array[wp.float64], out: wp.array[wp.vec4d]):
    i = wp.tid()
    out[i] = _phase_segment(a[i], b[i], wp.float64(0.0), wp.float64(2.0))


class TestControllerObjective(unittest.TestCase):
    """Check exact zero-crossing integrals and phase time moments."""

    def test_resident_objective_matches_residual_norm_and_invalidates_failures(self):
        """Keep composite loss consistent and reject stale or nonfinite force histories."""
        from projects.digital_shoe.friction_controller_objective import ControllerObjective  # noqa: PLC0415
        from projects.digital_shoe.friction_dynamic_gpu import (  # noqa: PLC0415
            FrictionDynamicGPUWorkspace,
            parse_candidate_dict,
        )

        baseline = Path(os.environ.get("NEWTON_BASELINE12_DIR", "outputs/impedance_instron/baseline12"))
        if not baseline.exists() or not wp.is_cuda_available():
            self.skipTest("Local sealed baseline and CUDA required")
        w = FrictionDynamicGPUWorkspace(baseline, world_count=2)
        candidate = {
            "method": "maxwell",
            "mu": 0.8,
            "kt_scale": 0.1,
            "kv_scale": 1.0,
            "viscous_ratio": 0.0,
            "release_dwell_s": 0.0005,
            "shear_relaxation_time_s": 0.005150109522,
        }
        w.adapter.set_parameters([parse_candidate_dict(candidate)])
        w.engine.evaluate(w.frozen_coefficients)
        trace, run = w.engine.trace(0)
        self.assertEqual(run["status"], "completed")
        objective = ControllerObjective(w.engine, w.adapter, trace)
        objective.launch(w.engine.states, w.engine.forces, w.engine.integrated, w.engine.failure)
        scores = objective.read()
        np.testing.assert_allclose(scores["loss"], np.sum(scores["residual"] ** 2, axis=0), rtol=1e-10)
        changed = w.engine.forces.numpy()
        changed[3, 1, 0] = np.nan
        w.engine.forces.assign(changed)
        objective.launch(w.engine.states, w.engine.forces, w.engine.integrated, w.engine.failure)
        rejected = objective.read()
        self.assertTrue(np.isfinite(rejected["loss"][0]))
        self.assertTrue(np.isinf(rejected["loss"][1]))
        self.assertTrue(np.isinf(rejected["residual"][:, 1]).all())

    def test_linear_segments_and_centroids(self):
        """Match analytical triangular areas and their force-weighted times."""
        for device in [wp.get_device("cpu"), *wp.get_cuda_devices()]:
            a = wp.array([-2.0, 2.0, -2.0, 2.0], dtype=wp.float64, device=device)
            b = wp.array([2.0, -2.0, -2.0, 2.0], dtype=wp.float64, device=device)
            out = wp.empty(4, dtype=wp.vec4d, device=device)
            wp.launch(_segments, dim=4, inputs=[a, b, out], device=device)
            expected = np.array(
                [[1.0, 1.0, 1 / 3, 5 / 3], [1.0, 1.0, 5 / 3, 1 / 3], [4.0, 0.0, 4.0, 0.0], [0.0, 4.0, 0.0, 4.0]]
            )
            np.testing.assert_allclose(out.numpy(), expected, rtol=1e-12, atol=1e-12)

    def test_host_targets_keep_disconnected_contact_intervals(self):
        """Exclude gaps instead of integrating force through unsupported contact."""
        t = np.arange(5.0) * 0.02
        f = np.array([-10.0, -10.0, 999.0, 20.0, 20.0])
        active = np.array([1, 1, 0, 1, 1], bool)
        result = phase_targets(t, f, active, early_end=0.03)
        np.testing.assert_allclose(result[:4], [0.2, 0.4, 10.0, 20.0], atol=1e-12)
        np.testing.assert_allclose(result[4:], [0.01, 0.07, 10.0, 0.2], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
