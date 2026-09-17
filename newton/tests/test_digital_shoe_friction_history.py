# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for digital shoe friction history cache."""

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import warp as wp

from projects.digital_shoe.friction_history import (
    build_history,
    load_history,
    normal_physics_source_identity,
    verify_baseline_inputs,
)
from projects.digital_shoe.friction_leg import replay_mode
from projects.digital_shoe.provenance import physics_source_identity
from projects.impedance_instron.cartesian.gpu.contact_replay import archive, exact_cpu_poses
from projects.impedance_instron.cartesian.mechanics import Body
from projects.impedance_instron.cartesian.shoe import Shoe

BASELINE_DIR = Path(os.environ.get("NEWTON_BASELINE12_DIR", "outputs/impedance_instron/baseline12"))


class TestDigitalShoeFrictionHistory(unittest.TestCase):
    """Test suite for digital shoe friction history cache generation and validation."""

    def test_verify_manifest_and_source_fingerprints(self) -> None:
        """Verify baseline manifest verification and source identity hash calculation."""
        if not BASELINE_DIR.exists():
            self.skipTest(f"Baseline directory not found: {BASELINE_DIR}")

        verified_hashes = verify_baseline_inputs(BASELINE_DIR)
        self.assertIn("trace.npz", verified_hashes)
        self.assertIn("reference.npz", verified_hashes)
        self.assertIn("digital_shoe.json", verified_hashes)

        norm_id = normal_physics_source_identity()
        phys_id = physics_source_identity()
        self.assertIsInstance(norm_id, str)
        self.assertEqual(len(norm_id), 64)
        self.assertIsInstance(phys_id, str)
        self.assertEqual(len(phys_id), 64)
        # Normal identity only includes normal mechanics; distinct from full friction law set
        self.assertNotEqual(norm_id, phys_id)

    def test_prevent_output_overwrite(self) -> None:
        """Verify build_history raises FileExistsError if target already exists."""
        if not BASELINE_DIR.exists():
            self.skipTest(f"Baseline directory not found: {BASELINE_DIR}")

        with tempfile.TemporaryDirectory() as tmp_dir:
            existing = Path(tmp_dir) / "already_exists.npz"
            existing.touch()
            with self.assertRaises(FileExistsError):
                build_history(BASELINE_DIR, existing, max_steps=5)

    def test_synthetic_steps_and_field_shapes(self) -> None:
        """Verify 5-step synthetic cache construction, fields, shapes, and read-only flags."""
        if not BASELINE_DIR.exists():
            self.skipTest(f"Baseline directory not found: {BASELINE_DIR}")
        device = "cuda:0" if wp.is_cuda_available() else "cpu"

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_npz = Path(tmp_dir) / "test_cache.npz"
            hist = build_history(
                BASELINE_DIR,
                out_npz,
                device=device,
                chunk_steps=2,
                max_steps=5,
            )

            self.assertEqual(len(hist.time_s), 5)
            # Verify increasing uniform clock
            dt = np.diff(hist.time_s)
            self.assertTrue(np.all(dt > 0.0))
            self.assertTrue(np.allclose(dt, dt[0]))

            # Verify shapes and dimensions
            c_count = len(hist.area_m2)
            self.assertGreater(c_count, 0)
            self.assertEqual(hist.position_xy.shape, (5, c_count, 2))
            self.assertEqual(hist.velocity_xy.shape, (5, c_count, 2))
            self.assertEqual(hist.nominal_velocity_xy.shape, (5, c_count, 2))
            self.assertEqual(hist.normal_n.shape, (5, c_count))
            self.assertEqual(hist.baseline_force_xy.shape, (5, c_count, 2))
            self.assertEqual(hist.baseline_kt_n_m.shape, (c_count,))
            self.assertEqual(hist.baseline_kv_ns_m.shape, (c_count,))

            self.assertFalse(hist.provenance["complete"])
            self.assertEqual(hist.provenance["step_count"], 5)

            # Test reload
            loaded = load_history(out_npz)
            self.assertEqual(loaded.normal_n.shape, (5, c_count))
            self.assertTrue(np.array_equal(loaded.normal_n, hist.normal_n))

            # Test read-only enforcement
            with self.assertRaises(ValueError):
                loaded.normal_n[0, 0] = 999.0

            with self.assertRaises(ValueError):
                loaded.position_xy[0, 0, 0] = 999.0

    def test_baseline_force_and_normal_reduction_parity(self) -> None:
        """Verify summed cache baseline force and normal match full replay within float32 reduction."""
        if not BASELINE_DIR.exists():
            self.skipTest(f"Baseline directory not found: {BASELINE_DIR}")
        device = "cuda:0" if wp.is_cuda_available() else "cpu"

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_npz = Path(tmp_dir) / "parity_cache.npz"
            eval_steps = 64
            hist = build_history(
                BASELINE_DIR,
                out_npz,
                device=device,
                chunk_steps=32,
                max_steps=eval_steps,
            )

            # Replay original bristle foundation
            trace_archive = archive(BASELINE_DIR / "trace.npz")
            ref_archive = archive(BASELINE_DIR / "reference.npz")
            with open(BASELINE_DIR / "profile.json", encoding="utf-8") as f:
                profile = json.load(f)
            with open(BASELINE_DIR / "summary.json", encoding="utf-8") as f:
                summary = json.load(f)

            shoe_info = summary["shoe"]
            body = Body(
                ref_archive["lengths_m"],
                ref_archive["endpoint_local_m"],
                profile["masses_kg"],
                profile["com_local_m"],
                profile["inertias_kg_m2"],
            )
            static_pitch = float(shoe_info["static_pitch_rad"])
            q_all, qd_all = exact_cpu_poses(body, trace_archive, static_pitch)
            dt = float(summary["run"]["actual_dt_s"])

            shoe = Shoe(BASELINE_DIR / "digital_shoe.json", shoe_info["mount_m"], static_pitch, device=device)
            res = replay_mode(shoe, q_all[:eval_steps], qd_all[:eval_steps], dt, "bristle", chunk_steps=32)
            replay_wrenches = res["wrenches"]

            # Sum of per-column baseline_force_xy matches body_f xy
            sum_f_xy = np.sum(hist.baseline_force_xy, axis=1)
            replay_f_xy = replay_wrenches[:, :2]
            max_diff_f = float(np.max(np.abs(sum_f_xy - replay_f_xy)))
            self.assertLess(max_diff_f, 1e-4)

            # Sum of per-column normal_n matches body_f z
            sum_normal = np.sum(hist.normal_n, axis=1)
            replay_normal = replay_wrenches[:, 2]
            max_diff_normal = float(np.max(np.abs(sum_normal - replay_normal)))
            self.assertLess(max_diff_normal, 1e-4)

    def test_complete_run_steps_and_clock(self) -> None:
        """Verify full 5760-step execution produces complete cache with valid stance metadata."""
        if not BASELINE_DIR.exists():
            self.skipTest(f"Baseline directory not found: {BASELINE_DIR}")
        device = "cuda:0" if wp.is_cuda_available() else "cpu"

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_npz = Path(tmp_dir) / "full_cache.npz"
            hist = build_history(
                BASELINE_DIR,
                out_npz,
                device=device,
                chunk_steps=32,
            )

            self.assertTrue(hist.provenance["complete"])
            self.assertEqual(len(hist.time_s), 5760)
            self.assertEqual(hist.provenance["step_count"], 5760)
            self.assertAlmostEqual(hist.time_s[-1] - hist.time_s[0], 0.36 - (hist.time_s[1] - hist.time_s[0]), places=4)

            # Check stance metadata
            self.assertGreater(hist.stance_metadata["active_sample_count"], 0)
            self.assertGreater(hist.stance_metadata["contiguous_interval_count"], 0)


if __name__ == "__main__":
    unittest.main()
