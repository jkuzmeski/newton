# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test the sealed optical reference and offline reduced-rig inverse dynamics."""

import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from projects.impedance_instron.simple.reference import (
    Reference,
    _equilibrium_error,
    geometry_identity,
    prepare_reference,
    tracking_error,
)


def _fixture():
    time = np.linspace(0.0, 0.4, 401)
    knots = np.linspace(-0.15, 0.55, 71)
    heel = np.column_stack((0.01 * knots, np.zeros_like(knots), 0.06 + 0.01 * knots))
    pelvis = np.column_stack((-0.1 + 0.03 * knots, np.zeros_like(knots), 1.05 + 0.002 * np.sin(15 * knots)))
    force_z = np.where((time >= 0.04) & (time <= 0.35), 900.0 + 200.0 * np.sin(12 * time), 0.0)
    force_x = -0.1 * force_z
    forces = np.column_stack((force_x, np.zeros_like(time), force_z))
    moments = np.column_stack((np.zeros_like(time), -0.02 * force_z + 3.0, np.zeros_like(time)))
    profile = {
        "schema_version": "impedance_stance_3",
        "side": "left",
        "mass_kg": 80.0,
        "time_s": time.tolist(),
        "source_time_s": (time + 90).tolist(),
        "reference_fx_n": force_x.tolist(),
        "reference_fz_n": force_z.tolist(),
        "com_z_m": [float("nan")],
        "reference_com_vz_m_s": [float("nan")],
        "pitch_reference": {
            "knot_time_s": knots.tolist(),
            "pitch_rad": (0.03 * np.sin(10 * knots)).tolist(),
            "heel_marker_xyz_m": np.repeat(heel[:, None, :], 3, axis=1).tolist(),
            "static_template": {"position_m": [[0, 0, 0.06]] * 3},
        },
        "pelvis_reference": {"knot_time_s": knots.tolist(), "centroid_m": pelvis.tolist()},
        "provenance": {
            "registration": {
                "heel_origin_newton_lab_m": [0, 0, 0.06],
                "virtual_origin_x_m": (3 * time).tolist(),
                "lab_to_newton": np.eye(3).tolist(),
            },
            "running": {"belt_speed_m_s": 3.0, "selected_stance_source_s": [90.04, 90.35]},
            "kinetics": {
                "platform_channels": {
                    "moment_about_lab_origin_nm": moments[:, None, :].tolist(),
                    "force_n": forces[:, None, :].tolist(),
                }
            },
            "sources": {},
            "reproduction_options": {},
            "rights": {},
        },
    }
    raw = {
        "column_bed": {"anchor_bottom_m": [[-0.12, 0, 0], [0.12, 0, 0]]},
        "visual_meshes": {"fullfoot_last": {"vertices_m": [[-0.12, 0, 0.04], [0.12, 0, 0.08]]}},
        "instron_fixtures": {},
        "coordinate_system": {"up_axis": "+Z"},
        "constitutive_model": {"modulus": 1},
        "provenance": {},
    }
    shoe = SimpleNamespace(
        raw=raw,
        column_bed=SimpleNamespace(anchor_bottom_m=np.array(raw["column_bed"]["anchor_bottom_m"])),
        visual_meshes={
            "fullfoot_last": SimpleNamespace(vertices_m=np.array(raw["visual_meshes"]["fullfoot_last"]["vertices_m"]))
        },
    )
    variability = SimpleNamespace(
        side="left",
        stance_count=83,
        windows_s=[[79, 139]],
        body={"subject": {"mass_kg": 80}},
        normaliser=lambda name: {"pelvis_height_m": 0.003, "foot_pitch_rad": 0.02}[name],
    )
    return profile, variability, shoe


class TestImpedanceSimpleReference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile, cls.variability, cls.shoe = _fixture()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.json"
            source.write_text("{}", encoding="utf-8")
            with (
                patch("projects.impedance_instron.profile.load_profile", return_value=cls.profile),
                patch("projects.impedance_instron.variability.load_variability", return_value=cls.variability),
                patch("projects.digital_shoe.load_artifact", return_value=cls.shoe),
                patch("projects.impedance_instron.orientation.orient_shoe", return_value=(cls.shoe, {})),
            ):
                cls.reference = prepare_reference(source, source, source)

    def test_pure_tracking(self):
        """Verify the loss contains only scaled height and wrapped pitch error."""
        result = tracking_error([1.003, 1.006], [2 * np.pi + 0.02, -2 * np.pi], 1, 0, 0.003, 0.02)
        np.testing.assert_allclose(result, [1, 2], atol=1e-12)
        self.assertEqual(tracking_error(1001, 0, 1, 0, 1, 1), 500000)
        for scale in (0, -1, np.inf, np.nan):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                tracking_error(1, 0, 1, 0, scale, 1)
        with self.assertRaises(ValueError):
            tracking_error(np.nan, 0, 1, 0, 1, 1)

    def test_sealed_portability(self):
        """Load embedded schedules after all source files disappear and reject tampering."""
        reference = self.reference
        body = reference.to_dict()
        restored = Reference.from_dict(json.loads(json.dumps(body)))
        self.assertEqual(reference.identity, restored.identity)
        with tempfile.TemporaryDirectory() as directory:
            path = reference.save(Path(directory) / "reference.json")
            self.assertEqual(Reference.load(path).identity, reference.identity)
            text = path.read_text(encoding="utf-8")
            path.write_text(
                text.replace('"schema_version":', '"schema_version":"duplicate", "schema_version":'), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                Reference.load(path)
        body["pelvis_z_m"][0] += 1
        with self.assertRaisesRegex(ValueError, "seal"):
            Reference.from_dict(body)
        with self.assertRaises(ValueError):
            reference.pelvis_z_m[0] = 0

    def test_consistent_sample_derivatives(self):
        """Verify Hermite derivatives, scalar broadcasting, and clamped zero rates."""
        reference = self.reference
        times = np.array([0.05317, 0.20271, 0.33337])
        epsilon = 1e-7
        sample = reference.sample(times)
        plus, minus = reference.sample(times + epsilon), reference.sample(times - epsilon)
        pairs = {
            "pelvis_z_m": "pelvis_vz_m_s",
            "pitch_rad": "pitch_rate_rad_s",
            "leg_length_m": "leg_rate_m_s",
            "ankle_equilibrium_rad": "ankle_equilibrium_rate_rad_s",
        }
        for position, velocity in pairs.items():
            np.testing.assert_allclose((plus[position] - minus[position]) / (2 * epsilon), sample[velocity], atol=3e-8)
            outside = reference.sample([-1, reference.duration_s + 1])
            np.testing.assert_array_equal(outside[velocity], [0, 0])
            np.testing.assert_allclose(outside[position], getattr(reference, position)[[0, -1]])
        self.assertEqual(np.shape(reference.sample(0.1)["pitch_rad"]), ())
        with self.assertRaises(ValueError):
            reference.sample(np.nan)

    def test_linear_forcing_ode(self):
        """Reject the quasi-static F/K shortcut by checking the exact moving-damper solution."""
        time = np.linspace(0, 0.4, 401)
        stiffness, damping = 12000, 500
        force = 300 + 1500 * time
        tau = damping / stiffness
        expected = 300 / stiffness + 1500 / stiffness * (time - tau + tau * np.exp(-time / tau))
        error, rate = _equilibrium_error(time, force, stiffness, damping)
        np.testing.assert_allclose(error, expected, atol=1e-14)
        np.testing.assert_allclose(stiffness * error + damping * rate, force, atol=1e-12)
        self.assertGreater(np.max(np.abs(error - force / stiffness)), 0.005)

    def test_reduced_inverse_dynamics(self):
        """Verify both impedance equations and expose unactuated force residuals."""
        reference = self.reference
        inverse = reference.provenance["inverse_dynamics"]
        config = reference.provenance["config"]
        leg_error = reference.leg_length_m - inverse["leg_measured_length_m"]
        leg_rate = reference.leg_rate_m_s - inverse["leg_measured_rate_m_s"]
        np.testing.assert_allclose(
            config["nominal_leg_stiffness_n_m"] * leg_error + inverse["leg_damping_n_s_m"] * leg_rate,
            reference.inverse_leg_force_n,
            atol=3e-12,
        )
        np.testing.assert_allclose(
            config["nominal_ankle_stiffness_n_m_rad"] * (reference.ankle_equilibrium_rad - reference.pitch_rad)
            + inverse["ankle_damping_n_m_s_rad"]
            * (reference.ankle_equilibrium_rate_rad_s - reference.pitch_rate_rad_s),
            reference.inverse_ankle_torque_n_m,
            atol=1e-12,
        )
        delta = np.array(inverse["upper_position_m"]) - inverse["foot_position_m"]
        direction = delta / np.linalg.norm(delta, axis=1)[:, None]
        residual = np.array(inverse["upper_orthogonal_force_n"])
        np.testing.assert_allclose(np.sum(direction * residual, axis=1), 0, atol=4e-13)
        gravity = np.array([0, 0, -reference.gravity_m_s2])
        axial = direction * reference.inverse_leg_force_n[:, None]
        np.testing.assert_allclose(
            axial + residual,
            inverse["upper_mass_kg"] * (np.array(inverse["upper_acceleration_m_s2"]) - gravity),
            atol=1e-12,
        )
        measured = np.column_stack(
            (reference.reference_fx_n, np.zeros_like(reference.time_s), reference.reference_fz_n)
        )
        np.testing.assert_allclose(
            np.array(inverse["foot_balance_force_n"]),
            config["foot_mass_kg"] * (np.array(inverse["foot_acceleration_m_s2"]) - gravity) - (measured - axial),
            atol=1e-12,
        )
        start = np.searchsorted(reference.time_s, reference.contact_start_s)
        bottom = self.shoe.column_bed.anchor_bottom_m - config["ankle_mount_local_m"]
        height = (
            np.array(inverse["foot_position_m"])[start, 2]
            - np.sin(reference.pitch_rad[start]) * bottom[:, 0]
            + np.cos(reference.pitch_rad[start]) * bottom[:, 2]
        )
        self.assertAlmostEqual(float(height.min()), 0, places=13)
        self.assertGreater(np.max(np.abs(residual)), 20)
        self.assertFalse(inverse["exact_compatibility_claimed"])
        # The poisoned force-integrated COM channels in the fixture must be irrelevant.
        self.assertTrue(np.all(np.isfinite(reference.pelvis_z_m)))
        self.assertAlmostEqual(reference.upper_position_m[2], reference.pelvis_z_m[0])
        self.assertGreater(reference.upper_velocity_m_s[0], 3)

    def test_contact_moment_frame(self):
        """Verify the measured moment shifts from the lab origin to the mapped ankle."""
        reference = self.reference
        inverse = reference.provenance["inverse_dynamics"]
        foot = np.array(inverse["foot_position_m"])
        ankle_lab_x = foot[:, 0] - 3 * reference.time_s
        expected = (
            -0.02 * reference.reference_fz_n
            + 3
            - (foot[:, 2] * reference.reference_fx_n - ankle_lab_x * reference.reference_fz_n)
        )
        np.testing.assert_allclose(inverse["contact_moment_n_m"], expected, atol=2e-12)
        np.testing.assert_allclose(
            reference.inverse_ankle_torque_n_m, 0.025 * np.array(inverse["pitch_acceleration_rad_s2"]) - expected
        )
        self.assertEqual(inverse["contact_moment_n_m"][0], 3)

    def test_geometry_fingerprint(self):
        """Accept material-only changes and reject a changed oriented geometry hash."""
        shoe = copy.deepcopy(self.shoe)
        before = geometry_identity(shoe)
        shoe.raw["constitutive_model"]["modulus"] = 1000
        shoe.raw["provenance"]["extra"] = "new material"
        self.assertEqual(geometry_identity(shoe), before)
        shoe.raw["column_bed"]["anchor_bottom_m"][0][0] += 0.001
        self.assertNotEqual(geometry_identity(shoe), before)

    def test_invalid_reference(self):
        """Reject invalid scales, array shapes, and reference clocks after resealing."""
        for update in (
            {"pelvis_scale_m": 0},
            {"pitch_scale_rad": np.nan},
            {"time_s": self.reference.time_s[::-1]},
            {"foot_position_m": [0, 0]},
            {"pitch_rad": self.reference.pitch_rad[:-1]},
        ):
            with self.subTest(update=list(update)), self.assertRaises(ValueError):
                replace(self.reference, **update)


if __name__ == "__main__":
    unittest.main()
