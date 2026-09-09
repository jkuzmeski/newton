# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify the reduced impedance rig after the measured-running example works."""

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import warp as wp

import newton
from projects.impedance_instron.example import Example, _apply_leg, _curve
from projects.impedance_instron.profile import load_profile
from projects.impedance_instron.report import _AUDIT_FIELDS, _REQUIRED, _audit, _summarize, write_report


def _reference_example():
    time = np.linspace(0.0, 0.3, 601)
    total_mass, foot_mass, gravity = 80.0, 2.0, 9.80665
    example = Example.__new__(Example)
    example.args = SimpleNamespace(kinematic_rate_hz=100.0, initial_clearance=0.0)
    example.mass, example.foot_mass = total_mass, foot_mass
    example.com_mass = total_mass - foot_mass
    example.gravity = gravity
    example.times = time
    example.duration = time[-1]
    example.shoe = SimpleNamespace(
        column_bed=SimpleNamespace(anchor_bottom_m=np.array([[-0.15, 0.0, 0.0], [0.15, 0.0, 0.0]]))
    )
    example.profile = {
        "time_s": time,
        "source_time_s": 90.0 + time,
        "foot_x_m": 0.03 * np.sin(10.0 * time),
        "foot_z_m": 0.003 * np.sin(10.0 * time),
        "pitch_rad": 0.1 * np.sin(10.0 * time),
        "com_x_m": 3.0 * time,
        "com_z_m": 1.0 + 0.1 * time**2,
        "reference_com_vz_m_s": 0.2 * time,
        "reference_fz_n": np.full(len(time), total_mass * (gravity + 0.2)),
        "reference_fx_n": np.zeros(len(time)),
        "other_fz_n": np.zeros(len(time)),
        "reference_cop_x_m": np.zeros(len(time)),
        "provenance": {"running": {"selected_stance_source_s": [90.02, 90.28]}},
    }
    example._make_reference()
    return example


def _report_inputs():
    rows = []
    for t in (0.0, 0.1, 0.2):
        row = dict.fromkeys(_REQUIRED, 0.0)
        row.update(
            time_s=t,
            shoe_fz_n=800.0,
            reference_fz_n=800.0,
            com_z_m=1.0,
            active_power_w=10.0 - 100.0 * t,
            damping_power_w=-2.0,
            max_compression_m=0.01,
        )
        rows.append(row)
    metadata = dict.fromkeys(_AUDIT_FIELDS, "fixed")
    metadata.update(
        expected_duration_s=0.2,
        mass_kg=80.0,
        foot_mass_kg=2.0,
        shoe_stiffness_scale=1.0,
        engineering_qualification={"passed": True, "reasons": []},
        shoe_identification_passed=False,
    )
    return rows, metadata


class TestImpedanceReference(unittest.TestCase):
    def test_mass_weighted_reference_matches_total_centroid(self):
        """Account for fixture inertia when mapping the total COM to the upper slider."""
        example = _reference_example()
        ref = example.reference
        centroid_z = (example.foot_mass * ref[:, 1] + example.com_mass * ref[:, 4]) / example.mass
        np.testing.assert_allclose(centroid_z, example.profile["com_z_m"], rtol=0, atol=2e-7)
        centroid_vz = (example.foot_mass * ref[:, 6] + example.com_mass * ref[:, 9]) / example.mass
        np.testing.assert_allclose(centroid_vz, example.profile["reference_com_vz_m_s"], rtol=0, atol=2e-7)
        acceleration = (example.foot_mass * ref[:, 10] + example.com_mass * ref[:, 16]) / example.mass
        np.testing.assert_allclose(acceleration, 0.2, rtol=0, atol=2e-6)

    def test_preserve_heel_origin_for_cop_comparison(self):
        """Keep the measured heel and simulated shoe in the same fore-aft origin."""
        example = _reference_example()
        angle = example.reference[0, 2]
        heel_x = example.reference[0, 0] - example.registration["heel_to_center_m"] * np.cos(angle)
        self.assertAlmostEqual(float(heel_x), 0.0, places=7)

    def test_hermite_derivatives_do_not_grow_with_solver_rate(self):
        """Recover quadratic velocity and acceleration independently of sample spacing."""
        knots = np.linspace(0.0, 0.3, 31)
        for count in (301, 1201):
            time = np.linspace(0.0, 0.3, count)
            position, velocity, acceleration = _curve(knots**2 + 2 * knots, knots, time)
            np.testing.assert_allclose(position, time**2 + 2 * time, atol=1e-12)
            np.testing.assert_allclose(velocity, 2 * time + 2, atol=1e-11)
            np.testing.assert_allclose(acceleration, 2.0, atol=1e-9)


class TestImpedanceNativeControl(unittest.TestCase):
    def test_reciprocal_forces_and_active_source_power(self):
        """Cancel internal forces and close mass-plus-spring power even at saturation."""
        reference = np.zeros((1, 19), np.float32)
        reference[0, [10, 11, 12, 13, 14]] = [2.0, 1000.0, 20.0, 1.0, -0.2]
        q = wp.array(
            np.array([[0, 0, 0, 0, 0, 0, 1], [0.5, 0, 1.1, 0, 0, 0, 1]], np.float32), dtype=wp.transform, device="cpu"
        )
        velocity = np.zeros((2, 6), np.float32)
        velocity[:, 2] = [0.2, -0.1]
        qd = wp.array(velocity, dtype=wp.spatial_vector, device="cpu")
        for force_limit in (1.0, 5000.0):
            forces = wp.zeros(2, dtype=wp.spatial_vector, device="cpu")
            diag = wp.zeros(5, dtype=wp.float32, device="cpu")
            wp.launch(
                _apply_leg,
                dim=1,
                inputs=[
                    0,
                    wp.array(reference, device="cpu"),
                    2.0,
                    10000.0,
                    200.0,
                    force_limit,
                    9.80665,
                    q,
                    qd,
                    forces,
                    diag,
                ],
                device="cpu",
            )
            values = diag.numpy()
            wrench = forces.numpy()
            self.assertAlmostEqual(float(wrench[:, 2].sum()), 20.0, places=4)
            rate, error, desired_rate = -0.3, 0.1, -0.2
            expected = float(values[0]) * rate + 10000.0 * error * (rate - desired_rate)
            self.assertAlmostEqual(float(values[1] + values[2]), expected, delta=0.002)
            self.assertLessEqual(float(values[2]), 0.0)
            self.assertEqual(bool(values[4]), force_limit == 1.0)

    def test_native_solver_preserves_total_vertical_balance(self):
        """Advance two native bodies with a measured-support surrogate and reciprocal leg forces."""
        gravity, foot_mass, upper_mass = 9.80665, 2.0, 78.0
        newton.use_coord_layout_targets = True
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0, 0.0, -gravity))
        builder.add_body(mass=foot_mass, inertia=wp.mat33(np.eye(3)))
        builder.add_body(mass=upper_mass, inertia=wp.mat33(np.eye(3)))
        model = builder.finalize(device="cpu")
        state, out = model.state(), model.state()
        forces = np.zeros((2, 6), np.float32)
        forces[:, 2] = [1000.0 - 750.0, 750.0 + 20.0]
        state.body_f.assign(forces)
        solver = newton.solvers.SolverSemiImplicit(model, angular_damping=0.0, enable_tri_contact=False)
        dt = 0.0001
        solver.step(state, out, model.control(), None, dt)
        acceleration = out.body_qd.numpy()[:, 2] / dt
        self.assertAlmostEqual(
            float(foot_mass * acceleration[0] + upper_mass * acceleration[1]), 1020.0 - 80.0 * gravity, delta=0.002
        )


class TestImpedanceReport(unittest.TestCase):
    def test_signed_work_and_visible_qualification(self):
        """Split power zero crossings and keep model-validation limits visible."""
        rows, metadata = _report_inputs()
        with tempfile.TemporaryDirectory() as directory:
            report = write_report(Path(directory), rows, metadata)
            summary = json.loads((Path(directory) / "summary.json").read_text())
            self.assertAlmostEqual(summary["metrics"]["positive_active_leg_work_j"], 0.5)
            self.assertAlmostEqual(summary["metrics"]["negative_active_leg_work_j"], -0.5)
            self.assertIn("not passed all declared validation gates", report.read_text())

    def test_only_same_artifact_scale_scenarios_pass_audit(self):
        """Reject changed rig settings, arbitrary artifacts, and failed engineering runs."""
        rows, metadata = _report_inputs()
        columns = {key: np.array([row[key] for row in rows]) for key in rows[0]}
        baseline = _summarize(columns, metadata)
        candidate = copy.deepcopy(baseline)
        candidate["metadata"].update(shoe_stiffness_scale=1.3, scenario_changes=["shoe_stiffness_scale"])
        self.assertTrue(_audit(candidate, baseline)["eligible"])
        for key in ("foot_mass_kg", "force_limit_n", "registration", "artifact_hash", "processed_reference_hash"):
            altered = copy.deepcopy(candidate)
            altered["metadata"][key] = "changed"
            self.assertFalse(_audit(altered, baseline)["eligible"], key)
        failed = copy.deepcopy(candidate)
        failed["metadata"]["engineering_qualification"]["passed"] = False
        self.assertFalse(_audit(failed, baseline)["eligible"])

    def test_shifted_trace_is_not_complete(self):
        """Reject a late-starting trace even when its duration matches the requested window."""
        rows, metadata = _report_inputs()
        for row in rows:
            row["time_s"] += 1.0
        columns = {key: np.array([row[key] for row in rows]) for key in rows[0]}
        self.assertFalse(_summarize(columns, metadata)["window"]["complete"])


@unittest.skipUnless(
    Path("outputs/impedance_instron/stance.json").is_file(), "Generate the local running profile first"
)
class TestLocalRunningProfile(unittest.TestCase):
    def setUp(self):
        """Load the local sealed profile without copying participant data into tests."""
        self.load_profile = load_profile
        self.source = Path("outputs/impedance_instron/stance.json")
        self.data = json.loads(self.source.read_text())
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "profile.json"

    def write(self, *, reseal=False):
        """Write an isolated corruption, optionally retaining a valid content checksum."""
        if reseal:
            self.data.pop("seal", None)
            content = json.dumps(self.data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            self.data["seal"] = {"algorithm": "sha256", "content_sha256": hashlib.sha256(content).hexdigest()}
        self.path.write_text(json.dumps(self.data))

    def test_default_is_late_running_not_walking_lead_in(self):
        """Require flight-qualified running from the later source recording."""
        data = self.load_profile(self.source)
        self.assertGreater(data["source_time_s"][0], 70.0)
        self.assertEqual(data["provenance"]["running"]["classification"], "running")
        self.assertEqual(data["provenance"]["kinematics"]["source_rate_hz"], 100)
        self.assertGreater(data["provenance"]["running"]["flight_before_s"], 0.02)
        self.assertEqual(data["reference_fz_n"], data["total_measured_fz_n"])
        self.assertTrue(all(value == 0.0 for value in data["other_fz_n"]))

    def test_reject_tampered_force_seal(self):
        """Reject a changed measured force before using it as controller input."""
        self.data["reference_fz_n"][0] += 1.0
        self.write()
        with self.assertRaisesRegex(ValueError, "seal"):
            self.load_profile(self.path)

    def test_reject_resealed_walking_and_bad_clock(self):
        """Reject walking classification and nonmonotone time even after resealing."""
        for corruption in ("walking", "clock"):
            with self.subTest(corruption=corruption):
                self.data = json.loads(self.source.read_text())
                if corruption == "walking":
                    self.data["provenance"]["running"]["classification"] = "walking"
                else:
                    self.data["time_s"][2] = self.data["time_s"][1]
                self.write(reseal=True)
                with self.assertRaises(ValueError):
                    self.load_profile(self.path)

    def test_reject_inconsistent_com_cop_and_force_channels(self):
        """Reject altered COM integration, COP origins, and measured platform sums."""
        for corruption in ("com", "cop", "platform", "flight_cop"):
            with self.subTest(corruption=corruption):
                self.data = json.loads(self.source.read_text())
                if corruption == "com":
                    self.data["com_z_m"][50] += 0.01
                elif corruption == "platform":
                    self.data["provenance"]["kinetics"]["platform_channels"]["force_n"][50][1][2] += 1.0
                elif corruption == "cop":
                    index = next(i for i, x in enumerate(self.data["reference_cop_x_m"]) if x is not None)
                    self.data["reference_cop_x_m"][index] += 0.1
                else:
                    index = next(i for i, x in enumerate(self.data["reference_cop_x_m"]) if x is None)
                    self.data["reference_cop_x_m"][index] = 0.0
                self.write(reseal=True)
                with self.assertRaises(ValueError):
                    self.load_profile(self.path)


if __name__ == "__main__":
    unittest.main()
