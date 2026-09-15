# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify the comparisons and units behind the figure-first sensitivity story."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from projects.impedance_instron.simple.sensitivity_figures import overview_figures


class TestSensitivityFigures(unittest.TestCase):
    """Check data pairing independently of the SVG coordinate implementation."""

    def setUp(self):
        """Create distinct quiet baselines that expose incorrect subtraction."""
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name)
        self.traces = {}
        self.lines = []
        self.scatters = []
        self.cases = []
        specs = (
            ("quiet", "nominal", "baseline", None, "quiet", 1.0),
            ("soft", "nominal", "soft", None, "quiet", 1.002),
            ("gain_quiet", "gain", "baseline", None, "gain_quiet", 1.2),
            ("push", "nominal", "baseline", "forward", "quiet", 1.001),
            ("gain_push", "gain", "baseline", "forward", "gain_quiet", 1.203),
        )
        for cid, controller, material, direction, baseline, height in specs:
            path = f"cases/{cid}.npz"
            trace = {
                "time_s": np.array([0.0, 0.1, 0.2]),
                "pelvis_z_m": np.full(3, height),
                "pelvis_x_m": np.full(3, height),
                "pelvis_vx_m_s": np.full(3, height),
                "pelvis_vz_m_s": np.full(3, height),
                "compression_m": np.full(3, 0.01),
                "shoe_fz_n": np.full(3, 100.0),
            }
            self.traces[path] = trace
            self.cases.append(
                {
                    "case_id": cid,
                    "controller_id": controller,
                    "material_id": material,
                    "controller_mode": "intent",
                    "direction": direction,
                    "comparison_type": "push_recovery"
                    if direction
                    else ("material_sensitivity" if material == "soft" else "unperturbed_baseline"),
                    "baseline_case_id": baseline,
                    "trace_file": path,
                    "baseline_trace_file": f"cases/{baseline}.npz",
                    "status": "valid",
                    "pair_valid": True,
                    "response_config": {"push_start_s": 0.12, "push_duration_s": 0.04},
                    "recovery": {
                        "status": "not_returned_within_window" if direction else "unperturbed_baseline",
                        "terminal_clock_match": True,
                        "final_state_time_s": 0.3,
                        "window": {"required_dwell_s": 0.05},
                        "deviations": {
                            name: {"final_deviation": 0.009, "tolerance": 0.001}
                            for name in ("pelvis_z_m", "pelvis_x_m", "pelvis_vx_m_s", "pelvis_vz_m_s")
                        },
                    },
                }
            )
        self.record = {
            "cases": self.cases,
            "recovery_config": {"position_tolerance_m": 0.001, "velocity_tolerance_m_s": 0.01},
            "controllers": [
                {"controller_id": "nominal", "varied_gain": None, "multiplier": 1.0},
                {"controller_id": "gain", "varied_gain": "leg_damping_n_s_m", "multiplier": 2.0},
            ],
            "materials": [
                {"id": "baseline", "baseline": True, "type": "baseline"},
                {"id": "soft", "type": "modulus", "factors": {"modulus_multiplier": 0.75}},
            ],
        }

    def render(self):
        """Capture curves and points while retaining normal grouping and export."""

        def line(curves, **kwargs):
            self.lines.append((list(curves), kwargs))
            return '<figure><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"></svg></figure>'

        def scatter(points, **kwargs):
            self.scatters.append((copy.deepcopy(points), kwargs))
            return '<figure><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"></svg></figure>'

        with (
            patch("projects.impedance_instron.simple.sensitivity_figures.line_figure", side_effect=line),
            patch("projects.impedance_instron.simple.sensitivity_figures.scatter_figure", side_effect=scatter),
        ):
            return overview_figures(self.output, self.record, lambda destination, path: self.traces.get(path, {}))

    def test_material_difference_uses_original_at_same_controller(self):
        """Subtract quiet original material and express small motion in millimeters."""
        self.render()
        curves, options = next(
            item for item in self.lines if item[1]["title"] == "How does the upper mass move differently?"
        )
        self.assertEqual([c.label for c in curves], ["Original material", "Modulus 0.75x"])
        self.assertEqual(len({c.color for c in curves}), len(curves))
        np.testing.assert_allclose(curves[0].value, 0)
        np.testing.assert_allclose(curves[1].value, 2)
        self.assertIn("[mm]", options["y_label"])
        self.assertEqual(options["spans"], ())

    def test_gain_difference_uses_own_quiet_and_actual_terminal(self):
        """Keep controller baselines distinct and mark the post-integration endpoint."""
        self.render()
        curves, options = next(
            item for item in self.lines if item[1]["title"] == "Vertical displacement caused by the push"
        )
        np.testing.assert_allclose(curves[0].value, 1)
        np.testing.assert_allclose(curves[1].value, 3)
        self.assertEqual(len({c.color for c in curves}), len(curves))
        self.assertEqual(curves[1].terminal_time, 300)
        self.assertAlmostEqual(curves[1].terminal_value, 9)
        self.assertNotEqual(curves[1].terminal_value, curves[1].value[-1])
        self.assertEqual(options["spans"][0]["start"], 120)
        self.assertEqual(options["spans"][0]["end"], 160)

    def test_wrong_baseline_identity_is_not_a_valid_curve(self):
        """Keep a mismatched baseline out of causal gain comparisons."""
        self.cases[-1]["baseline_case_id"] = "quiet"
        page = self.render()
        curves, _ = next(item for item in self.lines if item[1]["title"] == "Vertical displacement caused by the push")
        self.assertEqual(len(curves), 1)
        self.assertIn("baseline identity", page)
        self.assertIn("gain_push", page)

    def test_endpoint_projection_discloses_scope_and_missing_endpoints(self):
        """Keep terminal projections distinct from the full recovery decision."""
        self.cases[-1]["recovery"]["terminal_clock_match"] = False
        page = self.render()
        points, options = self.scatters[0]
        self.assertIsNone(points[-1]["x"])
        self.assertIsNone(points[-1]["y"])
        self.assertEqual(points[0]["legend_label"], "Forward")
        self.assertEqual(options["x_band"], (-1.0, 1.0))
        self.assertEqual(options["y_band"], (-10.0, 10.0))
        self.assertIn("NOT sufficient for full recovery", options["caption"])
        self.assertIn("Physical safety checks", page)


if __name__ == "__main__":
    unittest.main()
