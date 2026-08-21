# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for preliminary prescribed-motion C3D predictive contact."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import newton.opensim as osim
from projects.gait_c3d import predictive_contact


class TestGaitC3DPredictiveContact(unittest.TestCase):
    """Verify strict augmentation, wrench reconstruction, QC, and publication."""

    @staticmethod
    def _model() -> osim.OsimModel:
        """Build the bilateral model subset required by the contact sidecar."""
        marker_locations = {
            "L.Heel": (-0.034, 0.027, -0.014),
            "L.Toe.Med": (0.198, 0.001, 0.046),
            "L.Toe.Lat": (0.175, -0.009, -0.068),
            "L.Toe.Tip": (0.208, 0.018, -0.003),
            "R.Heel": (-0.031, 0.021, 0.013),
            "R.Toe.Med": (0.200, -0.011, -0.044),
            "R.Toe.Lat": (0.177, -0.010, 0.072),
            "R.Toe.Tip": (0.212, 0.011, -0.003),
        }
        markers = []
        for name, location in marker_locations.items():
            body = "calcn_l" if name.startswith("L.") else "calcn_r"
            markers.append(osim.OsimMarker(name=name, body=body, location=location))
        return osim.OsimModel(
            name="synthetic_scaled",
            bodies=[osim.OsimBody("calcn_l", mass=35.0), osim.OsimBody("calcn_r", mass=35.0)],
            markers=markers,
        )

    @staticmethod
    def _write_sidecar(
        tmpdir: Path,
        model: osim.OsimModel,
        model_path: Path,
        analysis_path: Path | None = None,
    ) -> Path:
        """Write a prescribed-motion geometry sidecar with synthetic transforms."""
        sidecar_path = tmpdir / "contact_sidecar.json"
        if analysis_path is None:
            analysis_path = tmpdir / "analysis.npz"
            contact = np.zeros((10, 2), dtype=bool)
            contact[1:9] = True
            np.savez_compressed(
                analysis_path,
                schema_version=np.asarray("gait_c3d_analysis_3"),
                id_coordinates=np.zeros((10, 2)),
                id_names=np.asarray(["q0", "q1"]),
                contact=contact,
                foot_names=np.asarray(["left", "right"]),
            )

        class InitialKinematics:
            """Return a frozen pose with landmarks above the ground plane."""

            coordinate_names = ("q0", "q1")
            body_names = ("ground", "calcn_l", "calcn_r")

            def __init__(self, parsed_model, device=None):
                self.parsed_model = parsed_model
                self.device = device

            def body_transforms_batch(self, coordinates):
                transforms = np.repeat(np.eye(4)[None, None], len(coordinates), axis=0)
                transforms = np.repeat(transforms, 3, axis=1)
                transforms[:, 1:, 1, 3] = 0.10
                return transforms

        with (
            mock.patch.object(predictive_contact.osim, "parse_osim", return_value=model),
            mock.patch.object(predictive_contact.osim, "ForwardKinematics", InitialKinematics),
        ):
            predictive_contact.write_initial_contact_sidecar(
                model_path,
                analysis_path,
                sidecar_path,
                platform_height_m=0.0,
                body_height_m=1.75,
                train_side="left",
            )
        return sidecar_path

    def test_freeze_role_phases_from_the_longest_contiguous_stance(self):
        """Ignore a wrapped boundary fragment when freezing heel and toe phase frames."""
        wrapped = np.r_[np.arange(12), np.arange(51, 107)]

        heel = predictive_contact._role_phase_frames(wrapped, "heel")
        forefoot = predictive_contact._role_phase_frames(wrapped, "medial_forefoot")
        toe = predictive_contact._role_phase_frames(wrapped, "toe")

        np.testing.assert_array_equal(heel, np.arange(51, 63))
        np.testing.assert_array_equal(forefoot, np.arange(73, 85))
        np.testing.assert_array_equal(toe, np.arange(95, 107))

    def test_write_and_load_a_strict_bilateral_sidecar(self):
        """Freeze eight marker-seeded spheres and reject unrecognized sidecar data."""
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            model_path = tmpdir / "S001_scaled.osim"
            model_path.write_text("synthetic", encoding="utf-8")
            sidecar_path = self._write_sidecar(tmpdir, self._model(), model_path)

            sidecar = predictive_contact.load_contact_sidecar(sidecar_path)

            self.assertEqual(len(sidecar.spheres), 8)
            self.assertEqual({sphere.side for sphere in sidecar.spheres}, {"left", "right"})
            self.assertEqual({sphere.role for sphere in sidecar.spheres}, set(predictive_contact._ROLES))
            self.assertEqual(sidecar.material.bounds["stiffness"], (1.0e5, 5.0e7))
            self.assertEqual(sidecar.calibration.load_threshold_n, 50.0)
            self.assertEqual(sidecar.calibration.cop_load_threshold_n, 200.0)
            self.assertEqual(sidecar.source_model_sha256, hashlib.sha256(b"synthetic").hexdigest())
            for sphere in sidecar.spheres:
                naive = np.asarray(sphere.marker_landmark_m) + np.array([0.0, sphere.radius_m, 0.0])
                self.assertFalse(np.allclose(sphere.geometry_seed_m, naive))
                np.testing.assert_allclose(sphere.center_m, sphere.geometry_seed_m)
                self.assertGreater(len(sphere.phase_frame_indices), 0)
                self.assertEqual(sphere.geometry_seed_method, "mean_inverse_vertical_projection_to_ground_tangent")
                self.assertAlmostEqual(sphere.tangent_residual_rms_m, 0.0, places=12)

            malformed = json.loads(sidecar_path.read_text(encoding="utf-8"))
            malformed["unreviewed_override"] = True
            sidecar_path.write_text(json.dumps(malformed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown fields"):
                predictive_contact.load_contact_sidecar(sidecar_path)

    def test_reject_nonshared_initial_radii_and_out_of_bounds_material(self):
        """Enforce bilateral initialization and the predeclared material bounds."""
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            model_path = tmpdir / "S001_scaled.osim"
            model_path.write_text("synthetic", encoding="utf-8")
            sidecar_path = self._write_sidecar(tmpdir, self._model(), model_path)
            data = json.loads(sidecar_path.read_text(encoding="utf-8"))
            right_heel = next(
                sphere for sphere in data["spheres"] if sphere["side"] == "right" and sphere["role"] == "heel"
            )
            right_heel["radius_m"] = 0.04
            sidecar_path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "shared bilaterally"):
                predictive_contact.load_contact_sidecar(sidecar_path)

            sidecar_path.unlink()
            data = json.loads(self._write_sidecar(tmpdir, self._model(), model_path).read_text(encoding="utf-8"))
            data["material"]["stiffness"] = 1.0e9
            sidecar_path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "outside its frozen bounds"):
                predictive_contact.load_contact_sidecar(sidecar_path)

    def test_augment_only_a_deep_copy_with_smooth_contact(self):
        """Leave the scaled model unchanged and append one ground plus eight sphere laws."""
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            model_path = tmpdir / "S001_scaled.osim"
            model_path.write_text("synthetic", encoding="utf-8")
            model = self._model()
            sidecar = predictive_contact.load_contact_sidecar(self._write_sidecar(tmpdir, model, model_path))
            before = copy.deepcopy(model)

            augmented = predictive_contact.augment_contact_model(model, sidecar)

            self.assertEqual(model.contact_geometry, before.contact_geometry)
            self.assertEqual(model.contact_forces, before.contact_forces)
            self.assertIsNot(augmented, model)
            self.assertEqual(len(augmented.contact_geometry), 9)
            self.assertEqual(len(augmented.contact_forces), 8)
            ground = augmented.contact_geometry[0]
            self.assertEqual(ground.type, "ContactHalfSpace")
            self.assertEqual(ground.body, "ground")
            np.testing.assert_allclose(ground.orientation, [0.0, 0.0, -np.pi / 2.0])
            for force in augmented.contact_forces:
                self.assertEqual(force.type, "SmoothSphereHalfSpaceForce")
                self.assertEqual(force.half_space, ground.name)
                self.assertEqual(force.params["stiffness"], 1.0e6)

    def test_real_smooth_contact_pushes_the_feet_upward(self):
        """Evaluate the real contact kernel and require positive OpenSim-Y foot force."""
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            model_path = tmpdir / "S001_scaled.osim"
            model_path.write_text("synthetic", encoding="utf-8")
            model = self._model()
            sidecar = predictive_contact.load_contact_sidecar(self._write_sidecar(tmpdir, model, model_path))
            contact = osim.OpenSimContact(predictive_contact.augment_contact_model(model, sidecar), device="cpu")
            state = np.zeros((1, len(contact.coordinate_names)))

            body_names, wrenches = contact.body_wrenches(
                state,
                np.zeros_like(state),
                h=predictive_contact._VELOCITY_STENCIL_H_S,
                frame="opensim",
            )

            self.assertEqual(body_names, ["calcn_l", "calcn_r"])
            self.assertTrue(np.all(np.isfinite(wrenches)))
            self.assertTrue(np.all(wrenches[0, :, 1] > 0.0))

    def test_reconstruct_cop_and_vertical_free_moment_from_ground_wrench(self):
        """Recover a known plane COP and free moment from force, point, and couple."""
        force = np.array([80.0, 600.0, -30.0])
        cop_expected = np.array([0.25, 0.10, -0.12])
        free_expected = 14.0
        point = np.array([-0.4, 0.8, 0.3])
        moment_origin = np.cross(cop_expected, force) + np.array([0.0, free_expected, 0.0])
        couple_at_point = moment_origin - np.cross(point, force)
        wrench = np.concatenate((force, point, couple_at_point))

        cop, free_moment = predictive_contact.ground_wrench_to_cop_free_moment(
            wrench[None], 0.10, load_threshold_n=50.0
        )

        np.testing.assert_allclose(cop[0], cop_expected, atol=1.0e-12)
        self.assertAlmostEqual(free_moment[0], free_expected)
        unloaded = wrench.copy()
        unloaded[1] = 49.9
        cop, free_moment = predictive_contact.ground_wrench_to_cop_free_moment(
            unloaded[None], 0.10, load_threshold_n=50.0
        )
        self.assertTrue(np.all(np.isnan(cop)))
        self.assertTrue(np.all(np.isnan(free_moment)))

    def test_compute_penetration_in_the_opensim_y_up_plane(self):
        """Measure only sphere depth below the stationary ground half-space."""
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            model_path = tmpdir / "S001_scaled.osim"
            model_path.write_text("synthetic", encoding="utf-8")
            model = self._model()
            sidecar = predictive_contact.load_contact_sidecar(self._write_sidecar(tmpdir, model, model_path))
            transforms = np.repeat(np.eye(4)[None, None], 2, axis=0)
            transforms = np.repeat(transforms, 3, axis=1)
            transforms[0, 1, 1, 3] = 0.10
            transforms[0, 2, 1, 3] = 0.10
            transforms[1, 1, 1, 3] = -0.05
            transforms[1, 2, 1, 3] = -0.05

            penetration = predictive_contact.sphere_penetrations(transforms, ["ground", "calcn_l", "calcn_r"], sidecar)

            np.testing.assert_allclose(penetration[0], 0.0, atol=1.0e-15)
            self.assertTrue(np.all(penetration[1] >= 0.0))
            self.assertAlmostEqual(penetration[1, 0], 0.15, places=12)

    def test_perfect_prescribed_targets_pass_qc_with_the_frozen_masks(self):
        """Pass every declared contact gate for exact finite target reconstruction."""
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            model_path = tmpdir / "S001_scaled.osim"
            model_path.write_text("synthetic", encoding="utf-8")
            sidecar = predictive_contact.load_contact_sidecar(self._write_sidecar(tmpdir, self._model(), model_path))
            times = np.arange(8, dtype=float) * 0.01
            force = np.zeros((8, 2, 3))
            force[2:6, :, 1] = 500.0
            cop = np.full((8, 2, 3), np.nan)
            cop[2:6, 0] = [0.2, 0.0, -0.1]
            cop[2:6, 1] = [0.2, 0.0, 0.1]
            free = np.zeros_like(force)
            free[2:6, :, 1] = 5.0
            contact = force[..., 1] >= 50.0
            penetration = np.zeros((8, 8))

            qc = predictive_contact.compute_contact_qc(
                times,
                force,
                cop,
                free,
                contact,
                force,
                cop,
                free[..., 1],
                penetration,
                sidecar,
            )

            self.assertIs(qc["passed"], True)
            self.assertEqual(qc["sides"]["left"]["split"], "fit")
            self.assertEqual(qc["sides"]["right"]["split"], "held_out")
            self.assertEqual(qc["sides"]["left"]["cop"]["loaded_frame_count"], 4)
            self.assertIs(qc["global_gates"]["no_measured_load_passed_to_contact_evaluator"], True)

    def test_publish_an_atomic_artifact_without_measured_contact_inputs(self):
        """Call OpenSimContact with state only and publish explicit preliminary failure/pass data."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            data_dir.mkdir()
            model_path = data_dir / "S001_scaled.osim"
            model_path.write_text("synthetic", encoding="utf-8")
            (data_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": "gait_c3d_analysis_3",
                        "status": "stage0_complete",
                        "runtime": {"git_commit": "a" * 40, "git_dirty": False},
                    }
                ),
                encoding="utf-8",
            )
            model = self._model()
            times = np.arange(8, dtype=float) * 0.01
            contact_mask = np.zeros((8, 2), dtype=bool)
            contact_mask[2:6] = True
            grf = np.zeros((8, 2, 3))
            grf[..., 1][contact_mask] = 500.0
            cop = np.full((8, 2, 3), np.nan)
            cop[2:6, 0] = [0.2, 0.0, -0.1]
            cop[2:6, 1] = [0.2, 0.0, 0.1]
            free_torque = np.zeros((8, 2, 3))
            free_torque[2:6, :, 1] = 5.0
            np.savez_compressed(
                data_dir / "analysis.npz",
                schema_version=np.asarray("gait_c3d_analysis_3"),
                times=times,
                id_coordinates=np.zeros((8, 2)),
                id_speeds=np.zeros((8, 2)),
                id_names=np.asarray(["q0", "q1"]),
                grf=grf,
                cop=cop,
                free_torque=free_torque,
                contact=contact_mask,
                foot_names=np.asarray(["left", "right"]),
            )
            sidecar_path = self._write_sidecar(root, model, model_path, data_dir / "analysis.npz")
            calibrated_sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            calibrated_sidecar["ground"]["height_m"] = 0.005
            for sphere in calibrated_sidecar["spheres"]:
                sphere["radius_m"] = 0.04
            sidecar_path.write_text(json.dumps(calibrated_sidecar), encoding="utf-8")
            calls = []

            class FakeContact:
                """Record evaluator arguments and return exact synthetic foot targets."""

                coordinate_names = ("q0", "q1")

                def __init__(self, augmented, device=None):
                    self.augmented = augmented
                    self.device = device
                    self.coordinate_names = ("q0", "q1")
                    self.assert_contact_count = len(augmented.contact_forces)

                def body_wrenches(self, coordinates, speeds, **kwargs):
                    calls.append((coordinates, speeds, kwargs))
                    sample_times = np.linspace(times[0], times[-1], len(coordinates))
                    sampled_grf = predictive_contact._interpolate_numeric(times, grf, sample_times)
                    sampled_cop = predictive_contact._interpolate_optional_numeric(times, cop, sample_times)
                    sampled_free = predictive_contact._interpolate_numeric(times, free_torque, sample_times)
                    wrenches = np.zeros((len(coordinates), 2, 9))
                    wrenches[..., :3] = sampled_grf
                    loaded = sampled_grf[..., 1] >= 50.0
                    wrenches[..., 3:6][loaded] = np.nan_to_num(sampled_cop[loaded])
                    wrenches[..., 7] = sampled_free[..., 1]
                    return ["calcn_l", "calcn_r"], wrenches

            class FakeKinematics:
                """Return separated feet and homogeneous body transforms."""

                coordinate_names = ("q0", "q1")
                body_names = ("ground", "calcn_l", "calcn_r")

                def __init__(self, augmented, device=None):
                    self.augmented = augmented
                    self.device = device

                def body_transforms_batch(self, coordinates):
                    transforms = np.repeat(np.eye(4)[None, None], len(coordinates), axis=0)
                    transforms = np.repeat(transforms, 3, axis=1)
                    transforms[:, 1:, 1, 3] = 0.10
                    return transforms

            output_dir = root / "prescribed_contact"
            with (
                mock.patch.object(predictive_contact.osim, "parse_osim", return_value=model),
                mock.patch.object(predictive_contact.osim, "OpenSimContact", FakeContact),
                mock.patch.object(predictive_contact.osim, "ForwardKinematics", FakeKinematics),
            ):
                completed = predictive_contact.run_prescribed_contact(data_dir, sidecar_path, output_dir, device="cpu")

            self.assertEqual(completed, output_dir)
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0][0].shape, (71, 2))
            self.assertEqual(calls[1][0].shape, (141, 2))
            for coordinates, speeds, kwargs in calls:
                self.assertEqual(coordinates.shape, speeds.shape)
                self.assertEqual(set(kwargs), {"h", "frame"})
                self.assertEqual(kwargs["frame"], "opensim")
                self.assertEqual(kwargs["h"], predictive_contact._VELOCITY_STENCIL_H_S)
            self.assertEqual(
                {path.name for path in output_dir.iterdir()},
                {"contact_sidecar.json", "contact_analysis.npz", "qc_summary.json", "manifest.json"},
            )
            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertIs(manifest["information_set"]["measured_load_input"], False)
            self.assertIs(manifest["settings"]["optimization_performed"], False)
            self.assertEqual(manifest["settings"]["held_out_side"], "right")
            self.assertEqual(manifest["settings"]["velocity_stencil_h_s"], 1.0e-6)
            self.assertIsInstance(manifest["runtime"]["git_dirty"], bool)
            self.assertIn("packages", manifest["runtime"])
            self.assertEqual(manifest["comparison_provenance"]["nominal_grid"]["frame_count"], 71)
            self.assertEqual(manifest["comparison_provenance"]["half_step_grid"]["frame_count"], 141)
            for artifact in manifest["artifacts"].values():
                artifact_path = output_dir / artifact["path"]
                self.assertEqual(hashlib.sha256(artifact_path.read_bytes()).hexdigest(), artifact["sha256"])
            qc = json.loads((output_dir / "qc_summary.json").read_text(encoding="utf-8"))
            self.assertIs(qc["passed"], True, qc)
            with np.load(output_dir / "contact_analysis.npz", allow_pickle=False) as archive:
                self.assertEqual(archive["predicted_grf"].shape, (71, 2, 3))
                self.assertEqual(archive["smaller_step_body_wrenches"].shape, (141, 2, 9))
                self.assertEqual(archive["sphere_penetration"].shape, (71, 8))

    def test_reorder_native_bodies_and_reject_invalid_sets_or_downward_force(self):
        """Reorder native feet, reject invalid sets, and fail the upward-force gate."""
        self.assertEqual(predictive_contact._contact_body_indices(["calcn_r", "calcn_l"], "test"), [1, 0])
        with self.assertRaisesRegex(ValueError, "exactly"):
            predictive_contact._contact_body_indices(["calcn_r", "pelvis"], "test")

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            model_path = tmpdir / "S001_scaled.osim"
            model_path.write_text("synthetic", encoding="utf-8")
            sidecar = predictive_contact.load_contact_sidecar(self._write_sidecar(tmpdir, self._model(), model_path))
            times = np.arange(8, dtype=float) * 0.01
            measured = np.zeros((8, 2, 3))
            measured[2:6, :, 1] = 500.0
            predicted = measured.copy()
            predicted[2:6, :, 1] *= -1.0
            cop = np.full((8, 2, 3), np.nan)
            free = np.zeros_like(measured)
            qc = predictive_contact.compute_contact_qc(
                times,
                measured,
                cop,
                free,
                measured[..., 1] >= 50.0,
                predicted,
                cop,
                np.zeros((8, 2)),
                np.zeros((8, 8)),
                sidecar,
            )

            self.assertIs(qc["passed"], False)
            self.assertIs(qc["sides"]["left"]["gates"]["opensim_y_normal_force_is_upward"], False)
            self.assertLess(qc["sides"]["right"]["force_direction"]["minimum_normal_force_N"], 0.0)

    def test_validate_source_schema_frame_and_immutable_seed_radius(self):
        """Reject mismatched source contracts and changes to the frozen seed radius."""
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            manifest_path = tmpdir / "manifest.json"
            manifest = {
                "schema_version": "wrong",
                "status": "complete",
                "runtime": {"git_dirty": False},
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "schema_version"):
                predictive_contact._load_source_manifest(manifest_path)
            manifest["schema_version"] = "gait_c3d_analysis_3"
            manifest["frame"] = "wrong_frame"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "frame"):
                predictive_contact._load_source_manifest(manifest_path)

            model_path = tmpdir / "S001_scaled.osim"
            model_path.write_text("synthetic", encoding="utf-8")
            sidecar_path = self._write_sidecar(tmpdir, self._model(), model_path)
            sidecar_data = json.loads(sidecar_path.read_text(encoding="utf-8"))
            sidecar_data["spheres"][0]["seed_radius_m"] = 0.031
            sidecar_path.write_text(json.dumps(sidecar_data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "immutable initial seed radius"):
                predictive_contact.load_contact_sidecar(sidecar_path)

            sidecar_data["spheres"][0]["seed_radius_m"] = 0.03
            sidecar_data["spheres"][0]["center_m"][0] = sidecar_data["spheres"][0]["geometry_seed_m"][0] + 0.031
            sidecar_path.write_text(json.dumps(sidecar_data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "geometry-seed bounds"):
                predictive_contact.load_contact_sidecar(sidecar_path)

            sidecar_data["spheres"][0]["center_m"] = sidecar_data["spheres"][0]["geometry_seed_m"]
            sidecar_data["calibration"]["prescribed_time_step_s"] = 0.002
            sidecar_path.write_text(json.dumps(sidecar_data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "frozen 1 ms"):
                predictive_contact.load_contact_sidecar(sidecar_path)

    def test_refuse_to_replace_an_existing_artifact(self):
        """Fail before evaluation rather than mix a new run into an old directory."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            output_dir = root / "result"
            data_dir.mkdir()
            output_dir.mkdir()
            with self.assertRaises(FileExistsError):
                predictive_contact.run_prescribed_contact(data_dir, root / "missing.json", output_dir)


if __name__ == "__main__":
    unittest.main()
