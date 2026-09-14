# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test the small active CLI and offline diagnostic report."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from projects.impedance_instron.__main__ import create_parser
from projects.impedance_instron.simple.report import summarize_trace, write_report


def trace_fixture():
    time = np.arange(10) * 0.01
    zeros = np.zeros(10)
    trace = {
        name: zeros.copy()
        for name in (
            "pitch_rad",
            "reference_pitch_rad",
            "ankle_equilibrium_rad",
            "shoe_fx_n",
            "reference_fx_n",
            "leg_source_power_w",
            "ankle_source_power_w",
            "leg_damping_power_w",
            "ankle_damping_power_w",
            "tracking_error",
            "compression_m",
            "last_clearance_m",
            "leg_stiffness_n_m",
            "ankle_stiffness_n_m_rad",
            "leg_damping_n_s_m",
            "ankle_damping_n_m_s_rad",
        )
    }
    trace.update(
        time_s=time,
        pelvis_z_m=np.ones(10),
        reference_pelvis_z_m=np.ones(10),
        shoe_fz_n=np.ones(10) * 100,
        reference_fz_n=np.ones(10) * 120,
    )
    return trace


class TestSimpleReport(unittest.TestCase):
    def test_work_is_diagnostic_only(self):
        """Change work and force diagnostics without changing the tracking score."""
        trace = trace_fixture()
        baseline = summarize_trace(trace, 0.01)
        trace["leg_source_power_w"][:] = 100
        trace["ankle_source_power_w"][:] = -20
        trace["shoe_fz_n"][:] = 1000
        changed = summarize_trace(trace, 0.01)
        self.assertEqual(baseline["tracking_loss"], changed["tracking_loss"])
        self.assertAlmostEqual(changed["leg_positive_work_j"], 10)
        self.assertAlmostEqual(changed["ankle_negative_work_j"], -2)
        self.assertAlmostEqual(changed["vertical_impulse_n_s"], 100)

    def test_report_contains_separate_stiffness_axes(self):
        """Save one offline report with separate leg and ankle stiffness panels."""
        trace = trace_fixture()
        rig = SimpleNamespace(
            trace=lambda world: trace, frame_dt=0.02, config=SimpleNamespace(substeps=2), num_worlds=1
        )
        with tempfile.TemporaryDirectory() as directory:
            path = write_report(rig, directory, {"safety_ok": [True]})
            page = path.read_text()
            self.assertIn("Leg stiffness", page)
            self.assertIn("Ankle rotational stiffness", page)
            self.assertIn("N/m", page)
            self.assertIn("N·m/rad", page)
            self.assertIn("evaluation only", page)
            self.assertNotIn("<script", page)
            self.assertNotIn("https://", page)
            self.assertTrue((Path(directory) / "summary.json").is_file())
            with np.load(Path(directory) / "trace_world_0.npz") as saved:
                np.testing.assert_array_equal(saved["time_s"], trace["time_s"])

    def test_cli_has_only_explicit_material_override(self):
        """Keep frozen evaluation free of accidental rig or reference overrides."""
        parser = create_parser()
        args = parser.parse_args(["evaluate", "best.pt", "--material", "foam.json"])
        self.assertEqual(args.material, Path("foam.json"))
        self.assertFalse(hasattr(args, "reference"))
        self.assertFalse(hasattr(args, "stiffness"))
        self.assertFalse(hasattr(args, "momentum"))

    def test_cli_requires_explicit_physics_source_update(self):
        """Keep source migration opt-in and separate from frozen physical settings."""
        parser = create_parser()
        default = parser.parse_args(["evaluate", "best.pt"])
        updated = parser.parse_args(["evaluate", "best.pt", "--allow-physics-update"])
        self.assertFalse(default.allow_physics_update)
        self.assertTrue(updated.allow_physics_update)
        self.assertFalse(hasattr(updated, "reference"))
        self.assertFalse(hasattr(updated, "rig_config"))

    def test_prepare_uses_new_inputs(self):
        """Default to the retained optical and two-term inputs, not legacy commands."""
        args = create_parser().parse_args(["prepare"])
        self.assertEqual(args.artifact, Path("outputs/impedance_instron/inputs/digital_shoe.json"))
        self.assertFalse(hasattr(args, "nominal"))


if __name__ == "__main__":
    unittest.main()
