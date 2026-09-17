# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for digital shoe friction leg motion replay."""

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import warp as wp

from projects.digital_shoe.friction_leg import (
    run_friction_leg_replay,
    verify_baseline_inputs,
)

BASELINE_DIR = Path(os.environ.get("NEWTON_BASELINE12_DIR", "outputs/impedance_instron/baseline12"))


class TestDigitalShoeFrictionLeg(unittest.TestCase):
    """Test suite for friction leg replay tool and parity metrics."""

    def test_verify_baseline_hashes(self) -> None:
        """Verify baseline inputs match the sealed baseline manifest hashes."""
        if not BASELINE_DIR.exists():
            self.skipTest(f"Baseline directory not found: {BASELINE_DIR}")
        hashes = verify_baseline_inputs(BASELINE_DIR)
        self.assertIn("trace.npz", hashes)
        self.assertIn("reference.npz", hashes)
        self.assertIn("digital_shoe.json", hashes)
        self.assertIn("summary.json", hashes)

    def test_replay_smoke_parity_and_untouched_normals(self) -> None:
        """Verify bristle and implicit bristle agree and normals are identical across modes."""
        if not BASELINE_DIR.exists():
            self.skipTest(f"Baseline directory not found: {BASELINE_DIR}")
        device = "cuda:0" if wp.is_cuda_available() else "cpu"

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "smoke_out"
            report = run_friction_leg_replay(
                baseline_dir=BASELINE_DIR,
                output_dir=out_dir,
                device=device,
                max_steps=64,
                modes=("bristle", "implicit_bristle", "regularized"),
                forward_sign=1,
                chunk_steps=32,
            )

            self.assertTrue(report["is_partial_smoke"])
            self.assertEqual(report["steps_evaluated"], 64)

            # Normal force discrepancies across modes must be 0
            normal_diffs = report["discrepancies"]["max_normal_discrepancies_n"]
            for pair, diff in normal_diffs.items():
                self.assertAlmostEqual(diff, 0.0, places=5, msg=f"Normal force mismatch for {pair}")

            # Bristle vs implicit_bristle must match under zero mobility prescribed motion
            parity_diff = report["discrepancies"]["bristle_vs_implicit_bristle_max_grf_diff_n"]
            self.assertIsNotNone(parity_diff)
            self.assertAlmostEqual(parity_diff, 0.0, places=5, msg="Bristle and implicit_bristle GRF mismatch")

            # Check that files were created
            self.assertTrue((out_dir / "report.json").exists())
            self.assertTrue((out_dir / "trace_bristle.npz").exists())
            self.assertTrue((out_dir / "trace_implicit_bristle.npz").exists())
            self.assertTrue((out_dir / "trace_regularized.npz").exists())

            # Load and verify trace structure
            with np.load(out_dir / "trace_bristle.npz") as t:
                self.assertEqual(len(t["time_s"]), 64)
                self.assertEqual(t["grf_n"].shape, (64, 2))
                self.assertEqual(t["wrench_newton"].shape, (64, 6))

    def test_prevent_output_overwrite(self) -> None:
        """Verify error is raised if output directory already exists."""
        if not BASELINE_DIR.exists():
            self.skipTest(f"Baseline directory not found: {BASELINE_DIR}")
        device = "cuda:0" if wp.is_cuda_available() else "cpu"

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "existing_dir"
            out_dir.mkdir()
            with self.assertRaises(FileExistsError):
                run_friction_leg_replay(
                    baseline_dir=BASELINE_DIR,
                    output_dir=out_dir,
                    device=device,
                    max_steps=10,
                )


if __name__ == "__main__":
    unittest.main()
