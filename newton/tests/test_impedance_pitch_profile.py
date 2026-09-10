# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check heel-cluster pitch reconstruction and the sealed optical-knot contract."""

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from projects.impedance_instron.profile import (
    ACQUISITION_FACTS,
    ARRAYS,
    COORDINATES,
    LEGACY_SCHEMA,
    SCHEMA,
    _build_pitch_reference,
    _canonical,
    _fit_heel_cluster,
    _integrate,
    _pitch_from_rotations,
    _pitch_kinematics,
    load_profile,
)


def _rotation_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _markers(*, angle=0.0):
    times = np.arange(251) / 100.0
    heel = np.array([[0.1, 0.15, 0.1], [0.1, 0.12, 0.04], [0.1, 0.18, 0.04]])
    # The upper-mounted metatarsal marker is below the high HEE marker.
    static = np.vstack((heel, [0.35, 0.17, 0.06], [0.43, 0.15, 0.04], [0.17, 0.12, 0.16]))
    positions = np.broadcast_to(static @ _rotation_y(angle).T, (len(times), 6, 3)).copy()
    return SimpleNamespace(
        times=times,
        rate=100.0,
        marker_names=("LHEE", "LHEE2", "LHEE3", "LTOE", "LHLX", "LANK"),
        positions=positions,
        valid=np.ones((len(times), 6), dtype=bool),
        lab_to_newton=np.eye(3),
    )


def _synthetic_profile():
    source_times = 0.8015 + np.arange(601) / 2000.0
    time = source_times - source_times[0]
    reference = _build_pitch_reference(_markers(angle=0.3), _markers(), source_times, "left")
    force = np.zeros((len(time), 2, 3))
    force[:, 0, 2] = 800.0
    mass, gravity = 80.0, 9.80665
    x, vx = _integrate(time, np.zeros(len(time)), 0.0, 3.0)
    z, vz = _integrate(time, force[:, 0, 2] / mass - gravity, 1.0, 0.0)
    result = {key: np.zeros(len(time)).tolist() for key in ARRAYS}
    result.update(
        schema_version=SCHEMA,
        coordinate_system=COORDINATES,
        mass_kg=mass,
        side="left",
        time_s=time.tolist(),
        source_time_s=source_times.tolist(),
        pitch_rad=np.interp(time, reference["knot_time_s"], reference["pitch_rad"]).tolist(),
        pitch_reference=reference,
        reference_fz_n=force[:, 0, 2].tolist(),
        total_measured_fz_n=force[:, 0, 2].tolist(),
        com_x_m=x.tolist(),
        com_z_m=z.tolist(),
        reference_com_vx_m_s=vx.tolist(),
        reference_com_vz_m_s=vz.tolist(),
        provenance={
            "sources": {
                key: {"path": "synthetic-test/" + key, "sha256": "0" * 64}
                for key in (
                    "c3d",
                    "calibration",
                    "treadmill_log",
                    "subject_manifest",
                    "c3d_adapter",
                    "contact_dataset",
                    "treadmill_adapter",
                    "exporter",
                )
            },
            "running": {
                "classification": "running",
                "selected_side": "left",
                "stance_duration_s": 0.3,
                "cadence_steps_min": 180.0,
                "flight_before_s": 0.03,
                "flight_after_s": 0.03,
            },
            "registration": {
                "heel_origin_newton_lab_m": [0.0, 0.0, 0.0],
                "virtual_origin_x_m": np.zeros(len(time)).tolist(),
                "lab_to_newton": np.eye(3).tolist(),
                "rigid_registration_after_rotation": np.eye(4).tolist(),
            },
            "kinematics": _pitch_kinematics("L"),
            "kinetics": {
                "load_threshold_n": 50.0,
                "source_rate_hz": 2000.0,
                "platform_channels": {
                    "force_n": force.tolist(),
                    "moment_about_lab_origin_nm": np.zeros_like(force).tolist(),
                },
            },
            "com_surrogate": {
                "kind": "force_integrated_surrogate_not_measured_com",
                "gravity_m_s2": gravity,
                "initial_x_m": 0.0,
                "initial_vx_m_s": 3.0,
                "initial_z_m": 1.0,
                "initial_vz_m_s": 0.0,
            },
            "recording": {"point_count": 251, "time_range_s": [0.0, 2.5], "first_point_frame": 0},
            "acquisition": ACQUISITION_FACTS.copy(),
            "rights": {"source": "synthetic regression fixture; no participant data"},
        },
    )
    return result


class TestHeelPitchMath(unittest.TestCase):
    def test_kabsch_recovers_proper_rotation_and_translation(self):
        """Recover a no-scale world rotation from the heel triangle only."""
        template = _markers().positions[0, :3]
        yaw = np.array([[np.cos(0.2), -np.sin(0.2), 0.0], [np.sin(0.2), np.cos(0.2), 0.0], [0.0, 0.0, 1.0]])
        roll = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(-0.3), -np.sin(-0.3)], [0.0, np.sin(-0.3), np.cos(-0.3)]])
        expected = yaw @ _rotation_y(0.7) @ roll
        shift = np.array([0.8, -0.3, 0.2])
        observed = (template @ expected.T + shift)[None]
        rotation, translation, rms, maximum = _fit_heel_cluster(template, observed)
        np.testing.assert_allclose(rotation[0], expected, atol=1e-14)
        np.testing.assert_allclose(translation[0], shift, atol=1e-14)
        np.testing.assert_allclose(np.linalg.det(rotation), 1.0, atol=1e-14)
        self.assertLess(float(maximum.max()), 1e-14)
        self.assertLess(float(rms.max()), 1e-14)
        forward = expected @ np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(
            _pitch_from_rotations(rotation, np.array([1.0, 0.0, 0.0])), [-np.arctan2(forward[2], forward[0])]
        )
        np.testing.assert_allclose(_pitch_from_rotations(_rotation_y(0.7)[None], np.array([1.0, 0.0, 0.0])), [0.7])

    def test_static_ground_heading_removes_marker_height_offset(self):
        """Define flat mechanical zero without treating the HEE-to-TOE line as the sole."""
        markers = _markers()
        times = np.array([0.8015, 1.1015])
        ref = _build_pitch_reference(markers, markers, times, "left")
        np.testing.assert_allclose(ref["pitch_rad"], 0.0, atol=1e-14)
        delta = markers.positions[0, 3] - markers.positions[0, 0]
        self.assertGreater(-np.arctan2(delta[2], delta[0]), 0.1)
        self.assertFalse(ref["neutral_reference"]["independently_measured_sole_axis"])
        self.assertEqual(ref["neutral_reference"]["kind"], "mechanical_flat_reference_assumption")

    def test_original_clock_keeps_context_outside_padded_profile(self):
        """Retain 100 Hz source knots with at least 150 ms on both outer sides."""
        times = np.array([0.8015, 1.1015])
        ref = _build_pitch_reference(_markers(), _markers(), times, "left")
        knots = np.array(ref["knot_time_s"])
        self.assertLessEqual(knots[0], -0.15)
        self.assertGreaterEqual(knots[-1], 0.45)
        np.testing.assert_allclose(np.diff(knots), 0.01, atol=1e-14)
        np.testing.assert_allclose((knots + times[0]) * 100.0, np.rint((knots + times[0]) * 100.0), atol=1e-12)

    def test_dynamic_toe_hallux_and_ankle_do_not_change_pitch(self):
        """Exclude moving metatarsal, hallux, and ankle markers from the rigid heel fit."""
        times = np.array([0.8015, 1.1015])
        trial, cal = _markers(angle=0.4), _markers()
        ref = _build_pitch_reference(trial, cal, times, "left")
        trial.positions[:, 3:] += np.arange(len(trial.times))[:, None, None] * 0.01
        trial.valid[:, 3:] = False
        changed = _build_pitch_reference(trial, cal, times, "left")
        self.assertEqual(ref, changed)

    def test_unwrap_crosses_pi_without_angle_jump(self):
        """Unwrap the raw heel pitch before interpolating optical knots."""
        angle = np.linspace(2.8, 3.5, 15)
        rotation = np.array([_rotation_y(value) for value in angle])
        np.testing.assert_allclose(_pitch_from_rotations(rotation, np.array([1.0, 0.0, 0.0])), angle, atol=1e-14)

    def test_reject_degenerate_missing_and_nonrigid_heel_data(self):
        """Reject invalid triangles, missing context markers, and excessive rigid-fit residuals."""
        for corruption in ("degenerate", "missing", "nonrigid", "context", "rate"):
            with self.subTest(corruption=corruption):
                trial, cal = _markers(), _markers()
                times = np.array([0.8015, 1.1015])
                if corruption == "degenerate":
                    trial.positions[:, 2] = trial.positions[:, 1]
                elif corruption == "missing":
                    trial.valid[70, 1] = False
                elif corruption == "nonrigid":
                    trial.positions[90, 1, 1] += 0.02
                elif corruption == "context":
                    times[0] = 0.1
                else:
                    trial.rate = 200.0
                with self.assertRaises(ValueError):
                    _build_pitch_reference(trial, cal, times, "left")


class TestPitchProfileLoader(unittest.TestCase):
    def setUp(self):
        """Create an isolated sealed synthetic stance for portable-loader checks."""
        self.data = _synthetic_profile()
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "profile.json"

    def write(self):
        """Reseal changed data to test semantics beyond the content hash."""
        self.data.pop("seal", None)
        self.data["seal"] = {"algorithm": "sha256", "content_sha256": hashlib.sha256(_canonical(self.data)).hexdigest()}
        self.path.write_text(json.dumps(self.data))

    def test_accept_v2_and_preserve_legacy_v1(self):
        """Load both schemas without requiring new pitch metadata from legacy artifacts."""
        self.write()
        self.assertEqual(load_profile(self.path), self.data)
        self.data["schema_version"] = LEGACY_SCHEMA
        del self.data["pitch_reference"]
        del self.data["provenance"]["acquisition"]
        del self.data["provenance"]["sources"]["calibration"]
        self.write()
        self.assertEqual(load_profile(self.path), self.data)

    def test_reject_resealed_bad_pitch_contract(self):
        """Reject changed optical clocks, rotation geometry, rates, assumptions, and sources."""
        pristine = copy.deepcopy(self.data)
        for corruption in (
            "clock",
            "context",
            "rate",
            "analog_rate",
            "bool",
            "length",
            "reflection",
            "wrong_fit",
            "pitch",
            "sampled_pitch",
            "residual",
            "template",
            "forward",
            "sole_claim",
            "cross_shoe",
            "missing_cal",
            "unknown_source",
            "registration",
            "recording",
            "quality",
            "unknown_field",
        ):
            with self.subTest(corruption=corruption):
                self.data = copy.deepcopy(pristine)
                ref = self.data["pitch_reference"]
                meta = self.data["provenance"]
                if corruption == "clock":
                    ref["knot_time_s"] = [x + 0.001 for x in ref["knot_time_s"]]
                elif corruption == "context":
                    ref["knot_time_s"][0] = -0.1
                elif corruption == "rate":
                    ref["source_rate_hz"] = 200.0
                elif corruption == "analog_rate":
                    meta["kinetics"]["source_rate_hz"] = 1000.0
                elif corruption == "bool":
                    ref["pitch_rad"][0] = True
                elif corruption == "length":
                    ref["fit_rms_m"].pop()
                elif corruption == "reflection":
                    ref["rotation_matrix"][0][0] = [-x for x in ref["rotation_matrix"][0][0]]
                elif corruption == "wrong_fit":
                    ref["rotation_matrix"][0] = np.eye(3).tolist()
                elif corruption == "pitch":
                    ref["pitch_rad"][0] += 0.1
                elif corruption == "sampled_pitch":
                    self.data["pitch_rad"][1] += 0.1
                elif corruption == "residual":
                    ref["fit_rms_m"][0] += 0.001
                elif corruption == "template":
                    ref["static_template"]["marker_names"][2] = "LANK"
                elif corruption == "forward":
                    ref["static_template"]["forward_unit_vector"][2] = 0.1
                elif corruption == "sole_claim":
                    ref["neutral_reference"]["independently_measured_sole_axis"] = True
                elif corruption == "cross_shoe":
                    meta["acquisition"]["capture_shoe_matches_modeled_puma"] = True
                elif corruption == "missing_cal":
                    del meta["sources"]["calibration"]
                elif corruption == "unknown_source":
                    meta["sources"]["unknown"] = {"path": "unknown", "sha256": "0" * 64}
                elif corruption == "registration":
                    meta["registration"]["lab_to_newton"][0][0] = -1.0
                elif corruption == "recording":
                    meta["recording"]["point_count"] = 100
                elif corruption == "quality":
                    ref["quality"]["max_allowed_rms_m"] = 1.0
                else:
                    ref["secret_offset"] = 0.1
                self.write()
                with self.assertRaises(ValueError):
                    load_profile(self.path)

    def test_retain_measured_force_cop_and_com_validators(self):
        """Reject inconsistent measured bookkeeping even in the new pitch schema."""
        pristine = copy.deepcopy(self.data)
        for corruption in ("force", "cop", "com", "running"):
            with self.subTest(corruption=corruption):
                self.data = copy.deepcopy(pristine)
                if corruption == "force":
                    self.data["reference_fz_n"][0] += 1.0
                elif corruption == "cop":
                    self.data["reference_cop_x_m"][0] += 0.1
                elif corruption == "com":
                    self.data["com_z_m"][1] += 0.01
                else:
                    self.data["provenance"]["running"]["classification"] = "walking"
                self.write()
                with self.assertRaises(ValueError):
                    load_profile(self.path)


@unittest.skipUnless(
    Path("outputs/impedance_instron/stance_pitch.json").is_file(), "Export the local heel-cluster pitch profile first"
)
class TestLocalHeelPitchProfile(unittest.TestCase):
    def test_measured_running_profile_and_legacy_force_identity(self):
        """Keep measured forces and legacy context unchanged in the supplied cross-shoe stance."""
        profile = load_profile("outputs/impedance_instron/stance_pitch.json")
        self.assertEqual(profile["schema_version"], SCHEMA)
        ref = profile["pitch_reference"]
        self.assertEqual(ref["static_template"]["marker_names"], ["LHEE", "LHEE2", "LHEE3"])
        self.assertLess(ref["quality"]["max_fit_rms_m"], 0.002)
        self.assertFalse(profile["provenance"]["acquisition"]["capture_shoe_matches_modeled_puma"])
        legacy_path = Path("outputs/impedance_instron/stance.json")
        if legacy_path.is_file():
            legacy = load_profile(legacy_path)
            for name in ARRAYS:
                if name != "pitch_rad":
                    self.assertEqual(profile[name], legacy[name], name)
            self.assertEqual(profile["provenance"]["kinetics"], legacy["provenance"]["kinetics"])
            self.assertEqual(profile["provenance"]["running"], legacy["provenance"]["running"])
            self.assertNotEqual(profile["pitch_rad"], legacy["pitch_rad"])


if __name__ == "__main__":
    unittest.main()
