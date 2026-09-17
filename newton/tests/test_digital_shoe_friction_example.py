# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit and regression tests for digital shoe friction example and multi-mode comparison."""

import json
import os
import tempfile
import unittest

import numpy as np

import newton.examples
from newton.viewer import ViewerNull
from projects.digital_shoe.friction_example import (
    Example,
    FrictionSimEngine,
    prescribed_normal_profile,
    prescribed_shear_force,
)


class TestDigitalShoeFrictionExample(unittest.TestCase):
    """Verify digital shoe friction example runs cleanly, respects frozen normal, and tests final state."""

    def test_modes_share_identical_patch_geometry(self):
        """Use identical deterministic contact samples across all friction models."""
        engines = [
            FrictionSimEngine(mode=mode, column_count=64, device="cpu")
            for mode in ("bristle", "implicit_bristle", "regularized")
        ]
        for engine in engines[1:]:
            np.testing.assert_array_equal(engine.points_initial, engines[0].points_initial)

    def test_example_runs_headless_and_passes_test_final(self):
        """Verify the friction example executes headlessly in test mode and validates test_final."""
        parser = Example.create_parser()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            cli_args = [
                "--viewer",
                "null",
                "--num-frames",
                "3",
                "--test",
                "--mode",
                "implicit_bristle",
                "--columns",
                "32",
                "--output",
                tmp_path,
            ]
            args = parser.parse_args(cli_args)
            viewer = ViewerNull(num_frames=args.num_frames)
            example = Example(viewer, args)
            newton.examples.run(example, args)

            self.assertTrue(os.path.exists(tmp_path))
            with open(tmp_path) as f:
                data = json.load(f)
            self.assertIn("implicit_bristle", data)
            self.assertGreater(len(data["implicit_bristle"]["times"]), 0)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_compare_all_three_modes_identical_normal(self):
        """Verify all three friction modes execute under identical prescribed normal reactions."""
        parser = Example.create_parser()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            cli_args = [
                "--viewer",
                "null",
                "--num-frames",
                "2",
                "--test",
                "--compare",
                "--columns",
                "32",
                "--output",
                tmp_path,
            ]
            args = parser.parse_args(cli_args)
            viewer = ViewerNull(num_frames=args.num_frames)
            example = Example(viewer, args)
            newton.examples.run(example, args)

            with open(tmp_path) as f:
                data = json.load(f)

            for m in ("bristle", "implicit_bristle", "regularized"):
                self.assertIn(m, data)
                self.assertEqual(len(data[m]["times"]), len(data["bristle"]["times"]))
                # Verify prescribed normal force history was identical across all modes
                np.testing.assert_allclose(
                    data[m]["prescribed_normal"],
                    data["bristle"]["prescribed_normal"],
                    rtol=1e-5,
                    atol=1e-5,
                    err_msg=f"Prescribed normal reactions differed between mode {m} and bristle",
                )
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_normal_dropout_produces_zero_friction(self):
        """Verify that zero prescribed normal load yields zero tangential friction across all modes."""
        cols = 32
        dt = 0.005
        dropout_normal = prescribed_normal_profile(
            sim_time=0.55,
            total_columns=cols,
            dropout_start=0.5,
            dropout_end=0.6,
        )
        self.assertEqual(float(np.sum(dropout_normal)), 0.0)

        f_shear, t_yaw = prescribed_shear_force(0.55)

        for mode in ("bristle", "implicit_bristle", "regularized"):
            engine = FrictionSimEngine(mode=mode, column_count=cols)
            res, _ = engine.advance(
                dt=dt,
                normal_profile=dropout_normal,
                prescribed_shear=f_shear,
                prescribed_yaw_torque=t_yaw,
            )
            wrench = res.wrench.numpy()[0]
            self.assertAlmostEqual(float(wrench[0]), 0.0, places=4)
            self.assertAlmostEqual(float(wrench[1]), 0.0, places=4)

    def test_dynamic_velocity_advance_matches_impulse_formula(self):
        """Verify dynamic state strictly advances via v_next = v_free + dt * mobility * wrench."""
        cols = 32
        dt = 0.01
        normal_profile = prescribed_normal_profile(sim_time=0.1, total_columns=cols)
        f_shear = np.array([200.0, 50.0], dtype=np.float32)

        engine = FrictionSimEngine(mode="bristle", column_count=cols)
        v_initial = engine.vel.copy()

        res, _ = engine.advance(
            dt=dt,
            normal_profile=normal_profile,
            prescribed_shear=f_shear,
            prescribed_yaw_torque=0.0,
        )
        wrench_np = res.wrench.numpy()[0]

        f_ext = np.array([200.0, 50.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        v_free = v_initial + dt * (engine.mobility_np @ f_ext)
        expected_v_next = v_free + dt * (engine.mobility_np @ wrench_np)

        np.testing.assert_allclose(
            engine.vel,
            expected_v_next,
            rtol=1e-5,
            atol=1e-5,
            err_msg="Carrier velocity did not match the required impulse integration formula",
        )
