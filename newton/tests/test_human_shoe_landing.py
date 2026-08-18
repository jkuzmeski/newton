# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the direct-coupled human-shoe landing example."""

from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np
import warp as wp

import newton
import newton.viewer
from projects.digital_instron_v2.core import CALIBRATED_MATERIAL
from projects.digital_instron_v2.dynamics import MidsoleFoundation
from projects.human_shoe import load_experiment
from projects.human_shoe.landing import (
    DEFAULT_DROP_CLEARANCE_M,
    EXPERIMENT_PATH,
    Example,
    resolve_landing_config,
)


def _outsole_world_z(example: Example) -> np.ndarray:
    """Return the unclamped outsole-column heights in the current state [m]."""
    body = example.state_0.body_q.numpy()[example.carrier]
    body_xform = wp.transform(wp.vec3(*body[:3]), wp.quat(*body[3:]))
    return np.asarray(
        [float(wp.transform_point(body_xform, wp.vec3(*point))[2]) for point in example.column_bottom_local]
    )


class TestHumanShoeLanding(unittest.TestCase):
    def test_import_and_attachment_contract(self):
        """Keep the direct landing on the expected OpenSim layout and Newton Z-up attachment."""
        example = Example(newton.viewer.ViewerNull(num_frames=1), Example.create_parser().parse_args([]))

        self.assertEqual(example.model.body_count, 12)
        self.assertEqual(example.model.joint_count, 12)
        self.assertEqual(len(example.model.joint_q), 27)
        self.assertEqual(len(example.model.joint_qd), 27)
        self.assertEqual(example.model.joint_label[0], "ground_pelvis")
        self.assertEqual(
            {name: example.import_result.coordinate_dof[name] for name in ("pelvis_tx", "pelvis_ty", "pelvis_tz")},
            {"pelvis_tx": 0, "pelvis_ty": 1, "pelvis_tz": 2},
        )
        self.assertEqual(
            {
                name: example.import_result.coordinate_dof[name]
                for name in ("pelvis_tilt", "pelvis_list", "pelvis_rotation")
            },
            {"pelvis_tilt": 3, "pelvis_list": 4, "pelvis_rotation": 5},
        )
        self.assertEqual(example.model.body_label[example.carrier], "calcn_r")
        self.assertAlmostEqual(example.initial_motion_time_s, 0.5)
        self.assertGreater(float(np.linalg.norm(example.state_0.joint_q.numpy()[3:])), 0.1)
        self.assertGreater(float(np.linalg.norm(example.state_0.joint_qd.numpy())), 0.01)
        self.assertEqual(example.resolved.shoe_carrier_body_index, example.carrier)
        self.assertEqual(example.column_area.shape, example.column_rest_len.shape)
        self.assertTrue(np.all(example.column_area > 0.0))
        self.assertLess(example.attachment_alignment_rms_m, 1.0e-7)
        self.assertLess(example.attachment_alignment_max_m, 1.0e-7)
        self.assertEqual(float(example.model.joint_target_ke.numpy()[1]), 0.0)
        self.assertEqual(float(example.model.joint_target_kd.numpy()[1]), 0.0)
        self.assertGreater(float(example.model.joint_target_ke.numpy()[0]), 0.0)
        np.testing.assert_allclose(_outsole_world_z(example).min(), DEFAULT_DROP_CLEARANCE_M, atol=1.0e-6)
        np.testing.assert_allclose(example.model.gravity.numpy()[0], [0.0, 0.0, -9.80665], atol=1.0e-5)
        self.assertIsInstance(example.foundation, MidsoleFoundation)
        self.assertAlmostEqual(float(example.foundation.params.pasternak), CALIBRATED_MATERIAL.pasternak_n_per_m)
        self.assertGreater(float(example.foundation.params.normal_damping), 0.0)
        self.assertGreater(float(example.foundation.params.friction_kt), 0.0)
        self.assertGreater(float(example.foundation.params.mu), 0.0)

    def test_resolve_landing_config_uses_manifest_and_override(self):
        """Resolve controller, timestep, and seed before constructing the GPU example."""
        experiment = load_experiment(EXPERIMENT_PATH)
        runtime = resolve_landing_config(experiment)
        override = resolve_landing_config(experiment, dt_override=1.0e-4)

        self.assertEqual(runtime.dt, experiment.time_step_s)
        self.assertEqual(runtime.random_seed, experiment.random_seed)
        self.assertEqual(override.dt, 1.0e-4)
        first = np.random.default_rng(runtime.random_seed).standard_normal(4)
        second = np.random.default_rng(runtime.random_seed).standard_normal(4)
        np.testing.assert_array_equal(first, second)

    def test_resolve_landing_config_rejects_unknown_controller(self):
        """Reject an experiment whose versioned controller is unavailable."""
        experiment = replace(load_experiment(EXPERIMENT_PATH), controller_id="missing")
        with self.assertRaisesRegex(ValueError, "unknown controller_id"):
            resolve_landing_config(experiment)

    def test_resolve_landing_config_rejects_bad_timestep_override(self):
        """Reject nonpositive and nonfinite timestep overrides."""
        experiment = load_experiment(EXPERIMENT_PATH)
        for dt in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(dt=dt), self.assertRaisesRegex(ValueError, "finite and positive"):
                resolve_landing_config(experiment, dt_override=dt)

    def test_foundation_force_targets_calcn_r(self):
        """Apply a Z-up foundation wrench only to the attached calcn_r body."""
        example = Example(newton.viewer.ViewerNull(num_frames=1), Example.create_parser().parse_args([]))
        q = example.state_0.joint_q.numpy()
        q[1] -= DEFAULT_DROP_CLEARANCE_M + 0.002
        example.state_0.joint_q.assign(wp.array(q, dtype=wp.float32, device=example.device))
        example.state_0.joint_qd.zero_()
        newton.eval_fk(example.model, example.state_0.joint_q, example.state_0.joint_qd, example.state_0)

        example._apply_foundation()

        forces = example.state_0.body_f.numpy()
        self.assertGreater(float(example.foundation.normal_force.numpy()[0]), 0.0)
        self.assertGreater(float(forces[example.carrier, 2]), 0.0)
        np.testing.assert_allclose(np.delete(forces, example.carrier, axis=0), 0.0, atol=1.0e-7)

    def test_drop_contact_smoke(self):
        """Drop onto the calibrated sole and keep the direct dynamics finite."""
        viewer = newton.viewer.ViewerNull(num_frames=1)
        args = Example.create_parser().parse_args(["--duration", "0.08"])
        example = Example(viewer, args)
        initial_pelvis_z = float(example.state_0.body_q.numpy()[0, 2])

        while example.sim_time < example.duration:
            example.step()
        example.render()
        example.test_final()

        diagnostics = example.diagnostics()
        self.assertTrue(diagnostics.finite)
        self.assertGreater(diagnostics.peak_normal_force_n, 100.0)
        self.assertLess(diagnostics.peak_normal_force_n, 800.0)
        self.assertGreaterEqual(diagnostics.peak_active_columns, 20)
        self.assertLessEqual(diagnostics.peak_active_columns, 100)
        self.assertGreater(diagnostics.peak_compression_m, 1.0e-3)
        self.assertLess(diagnostics.peak_compression_m, 4.0e-3)
        self.assertLess(diagnostics.peak_compression_m, float(example.column_rest_len.max()))
        self.assertLess(diagnostics.pelvis_height_m, initial_pelvis_z)
        self.assertEqual(viewer.frame_count, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
