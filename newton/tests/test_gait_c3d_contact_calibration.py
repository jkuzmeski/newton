# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for bounded preliminary C3D normal-contact calibration."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from projects.gait_c3d.compatibility import contact_calibration, predictive_contact


class TestGaitC3DContactCalibration(unittest.TestCase):
    """Verify fit bounds, split isolation, penalties, traces, and publication."""

    @staticmethod
    def _write_sidecar(root: Path, *, train_side: str = "left") -> Path:
        """Write a strict synthetic predictive-contact sidecar."""
        source_model = root / "source.osim"
        source_analysis = root / "analysis.npz"
        source_model.write_bytes(b"model")
        source_analysis.write_bytes(b"analysis")
        held_out = "right" if train_side == "left" else "left"
        roles = ("heel", "medial_forefoot", "lateral_forefoot", "toe")
        marker_names = {
            ("left", "heel"): "L.Heel",
            ("left", "medial_forefoot"): "L.Toe.Med",
            ("left", "lateral_forefoot"): "L.Toe.Lat",
            ("left", "toe"): "L.Toe.Tip",
            ("right", "heel"): "R.Heel",
            ("right", "medial_forefoot"): "R.Toe.Med",
            ("right", "lateral_forefoot"): "R.Toe.Lat",
            ("right", "toe"): "R.Toe.Tip",
        }
        spheres = []
        for side_index, side in enumerate(("left", "right")):
            for role_index, role in enumerate(roles):
                seed = [0.05 * role_index, 0.08 + 0.002 * side_index, 0.02 * (2 * side_index - 1)]
                spheres.append(
                    {
                        "name": f"contact_{side[0]}_{role}",
                        "force_name": f"force_{side[0]}_{role}",
                        "side": side,
                        "role": role,
                        "body": "calcn_l" if side == "left" else "calcn_r",
                        "marker": marker_names[(side, role)],
                        "marker_landmark_m": seed,
                        "geometry_seed_method": "mean_inverse_vertical_projection_to_ground_tangent",
                        "geometry_seed_m": seed,
                        "seed_radius_m": 0.03,
                        "phase_frame_indices": [1, 2, 3],
                        "tangent_residual_rms_m": 0.0,
                        "tangent_residual_max_abs_m": 0.0,
                        "center_m": seed,
                        "radius_m": 0.03,
                        "center_displacement_bounds_m": [-0.03, 0.03],
                        "radius_bounds_m": [0.01, 0.06],
                    }
                )
        data = {
            "schema_version": "gait_c3d_predictive_contact_sidecar_2",
            "source_model_path": str(source_model),
            "source_model_sha256": hashlib.sha256(b"model").hexdigest(),
            "source_analysis_path": str(source_analysis),
            "source_analysis_sha256": hashlib.sha256(b"analysis").hexdigest(),
            "frame": "opensim_x_forward_y_up_z_right",
            "units": {"length": "m", "force": "N", "moment": "N*m", "time": "s"},
            "ground": {
                "name": "contact_ground",
                "height_m": 0.005,
                "platform_height_m": 0.005,
                "height_bounds_m": [-0.02, 0.02],
            },
            "material": {
                "law": "SmoothSphereHalfSpaceForce",
                "stiffness": 1.0e6,
                "dissipation": 1.0,
                "static_friction": 0.8,
                "dynamic_friction": 0.6,
                "viscous_friction": 0.0,
                "transition_velocity": 0.1,
                "constant_contact_force": 1.0e-5,
                "hertz_smoothing": 300.0,
                "hunt_crossley_smoothing": 50.0,
                "bounds": {
                    "stiffness": [1.0e5, 5.0e7],
                    "dissipation": [0.0, 5.0],
                    "static_friction": [0.2, 1.5],
                    "dynamic_friction": [0.1, 1.5],
                    "viscous_friction": [0.0, 1.0],
                    "transition_velocity": [0.01, 0.5],
                },
            },
            "spheres": spheres,
            "calibration": {
                "train_side": train_side,
                "held_out_side": held_out,
                "load_threshold_n": 50.0,
                "cop_load_threshold_n": 200.0,
                "prescribed_time_step_s": 0.001,
                "objective_weights": {
                    "vertical_force": 1.0,
                    "horizontal_force": 1.0,
                    "impulse": 1.0,
                    "cop": 1.0,
                    "free_moment": 1.0,
                    "regularization": 0.1,
                    "bilateral": 1.0,
                },
            },
            "normalization": {"body_weight_n": 700.0, "body_height_m": 1.75},
        }
        path = root / "contact_sidecar.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        predictive_contact.load_contact_sidecar(path)
        return path

    @staticmethod
    def _target() -> tuple[np.ndarray, np.ndarray]:
        """Return a small bilateral vertical-force target."""
        times = np.arange(5, dtype=float) * 0.01
        force = np.asarray(
            [
                [0.0, 20.0],
                [300.0, 250.0],
                [600.0, 500.0],
                [300.0, 250.0],
                [0.0, 20.0],
            ]
        )
        return times, force

    @staticmethod
    def _fake_evaluator(held_out_scale: float = 1.0, train_penetration_m: float = 0.0):
        """Build a deterministic prediction-only evaluator."""
        times, target = TestGaitC3DContactCalibration._target()

        def evaluate(sidecar):
            stiffness_scale = sidecar.material.stiffness / 1.0e6
            heel = next(sphere for sphere in sidecar.spheres if sphere.side == "left" and sphere.role == "heel")
            vertical_offset = heel.center_m[1] - heel.geometry_seed_m[1]
            train_scale = stiffness_scale * (1.0 - 4.0 * vertical_offset)
            predicted = target.copy()
            predicted[:, 0] = target[:, 0] * train_scale
            predicted[:, 1] = target[:, 1] * held_out_scale
            penetration = np.zeros((len(times), len(sidecar.spheres)))
            for index, sphere in enumerate(sidecar.spheres):
                penetration[:, index] = train_penetration_m if sphere.side == "left" else 0.5 * held_out_scale
            return contact_calibration.NormalContactEvaluation(predicted, penetration)

        return evaluate

    def test_builtin_evaluator_provenance_hashes_frozen_q_and_qd(self):
        """Identify trusted built-in evaluation by immutable prescribed-state hashes."""
        q = np.arange(12, dtype=float).reshape(4, 3)
        qd = -q
        evaluator = contact_calibration._PrescribedNormalEvaluator(None, q, qd, ("a", "b", "c"), "cpu")
        expected_q = contact_calibration._array_sha256(q)
        expected_qd = contact_calibration._array_sha256(qd)
        q[:] = 0.0
        qd[:] = 0.0

        provenance = evaluator.provenance()

        self.assertEqual(provenance["kind"], "built_in_prescribed_normal_evaluator")
        self.assertEqual(provenance["measured_input_isolation"], "verified")
        self.assertEqual(provenance["q_sha256"], expected_q)
        self.assertEqual(provenance["qd_sha256"], expected_qd)

    def test_encode_exact_frozen_bounds_and_shared_vertical_offsets(self):
        """Encode six parameters and decode only role-shared vertical displacement."""
        with tempfile.TemporaryDirectory() as tmp:
            sidecar = predictive_contact.load_contact_sidecar(self._write_sidecar(Path(tmp)))
            parameters = contact_calibration.NormalContactParameterization(sidecar)

            self.assertEqual(parameters.names[0], "ground_height_m")
            self.assertEqual(parameters.names[-1], "log10_stiffness")
            np.testing.assert_allclose(parameters.lower_bounds, [-0.015, -0.03, -0.03, -0.03, -0.03, 5.0])
            np.testing.assert_allclose(
                parameters.upper_bounds,
                [0.025, 0.03, 0.03, 0.03, 0.03, np.log10(5.0e7)],
            )
            encoded = np.asarray([0.01, 0.012, -0.004, 0.003, -0.002, 7.0])
            candidate = parameters.decode(encoded)

            self.assertAlmostEqual(candidate.material.stiffness, 1.0e7)
            self.assertAlmostEqual(candidate.ground.height_m, 0.01)
            for role_index, role in enumerate(("heel", "medial_forefoot", "lateral_forefoot", "toe")):
                role_spheres = [sphere for sphere in candidate.spheres if sphere.role == role]
                self.assertEqual(len(role_spheres), 2)
                for sphere in role_spheres:
                    displacement = np.asarray(sphere.center_m) - np.asarray(sphere.geometry_seed_m)
                    np.testing.assert_allclose(displacement, [0.0, encoded[role_index + 1], 0.0])
            for original, candidate_sphere in zip(sidecar.spheres, candidate.spheres, strict=True):
                self.assertEqual(candidate_sphere.geometry_seed_m, original.geometry_seed_m)
                self.assertEqual(candidate_sphere.radius_m, original.radius_m)

    def test_never_use_held_out_force_or_penetration_in_the_fit(self):
        """Keep fitted parameters invariant to arbitrary held-out evaluator outputs."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sidecar_path = self._write_sidecar(root)
            times, target = self._target()
            first = contact_calibration.calibrate_normal_contact(
                sidecar_path,
                root / "first",
                times,
                target,
                self._fake_evaluator(held_out_scale=1.0),
                max_nfev=2,
            )
            second = contact_calibration.calibrate_normal_contact(
                sidecar_path,
                root / "second",
                times,
                target,
                self._fake_evaluator(held_out_scale=100.0),
                max_nfev=2,
            )

            first_result = json.loads((first / "optimizer_result.json").read_text(encoding="utf-8"))
            second_result = json.loads((second / "optimizer_result.json").read_text(encoding="utf-8"))
            self.assertEqual(first_result["final_parameters"], second_result["final_parameters"])
            self.assertIs(first_result["held_out_side_used_in_objective"], False)
            trace = json.loads((second / "evaluation_trace.json").read_text(encoding="utf-8"))
            self.assertTrue(all(entry["objective_side"] == "left" for entry in trace))
            self.assertTrue(all(entry["held_out_side_used_in_objective"] is False for entry in trace))

    def test_nan_held_out_values_are_isolated_until_frozen_reporting(self):
        """Ignore nonfinite held-out targets and predictions until post-fit reporting."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sidecar_path = self._write_sidecar(root)
            times, target = self._target()
            target[:, 1] = np.nan
            base_evaluator = self._fake_evaluator()

            def evaluate(sidecar):
                evaluation = base_evaluator(sidecar)
                force = evaluation.vertical_force_n.copy()
                penetration = evaluation.sphere_penetration_m.copy()
                force[:, 1] = np.nan
                for index, sphere in enumerate(sidecar.spheres):
                    if sphere.side == "right":
                        penetration[:, index] = np.nan
                return contact_calibration.NormalContactEvaluation(force, penetration)

            output = contact_calibration.calibrate_normal_contact(
                sidecar_path,
                root / "fit",
                times,
                target,
                evaluate,
                max_nfev=2,
            )

            fit_metrics = json.loads((output / "fit_metrics.json").read_text(encoding="utf-8"))
            held_out = json.loads((output / "held_out_metrics.json").read_text(encoding="utf-8"))
            trace = json.loads((output / "evaluation_trace.json").read_text(encoding="utf-8"))
            self.assertIs(fit_metrics["valid"], True)
            self.assertIs(held_out["valid"], False)
            self.assertEqual(held_out["status"], "invalid_frozen_evaluation")
            self.assertIn("measured_vertical_force_finite", held_out["invalid_reasons"])
            self.assertIn("predicted_vertical_force_finite", held_out["invalid_reasons"])
            self.assertIn("sphere_penetration_finite", held_out["invalid_reasons"])
            self.assertTrue(all(np.isfinite(entry["residual_sumsq"]) for entry in trace))

    def test_normalize_samplewise_residuals_for_grid_invariance(self):
        """Keep objective term sums of squares invariant to uniform sample count."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sidecar_path = self._write_sidecar(root)
            outputs = []
            for sample_count in (5, 41):
                times = np.linspace(0.0, 0.04, sample_count)
                target = np.tile([350.0, 200.0], (sample_count, 1))

                def evaluate(sidecar, target=target):
                    predicted = target.copy()
                    predicted[:, 0] = 280.0 * sidecar.material.stiffness / 1.0e6
                    penetration = np.zeros((len(target), len(sidecar.spheres)))
                    for index, sphere in enumerate(sidecar.spheres):
                        if sphere.side == "left":
                            penetration[:, index] = 0.025
                    return contact_calibration.NormalContactEvaluation(predicted, penetration)

                outputs.append(
                    contact_calibration.calibrate_normal_contact(
                        sidecar_path,
                        root / f"fit_{sample_count}",
                        times,
                        target,
                        evaluate,
                        max_nfev=1,
                    )
                )

            traces = [json.loads((output / "evaluation_trace.json").read_text(encoding="utf-8")) for output in outputs]
            first_terms = [trace[0]["residual_term_sumsq"] for trace in traces]
            self.assertEqual(first_terms[0].keys(), first_terms[1].keys())
            for name in first_terms[0]:
                self.assertAlmostEqual(first_terms[0][name], first_terms[1][name], places=12, msg=name)
            self.assertAlmostEqual(traces[0][0]["residual_sumsq"], traces[1][0]["residual_sumsq"], places=12)

    def test_penalize_training_penetration_above_twenty_millimeters(self):
        """Record a positive objective penalty only above the frozen depth limit."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sidecar_path = self._write_sidecar(root)
            times, target = self._target()
            output = contact_calibration.calibrate_normal_contact(
                sidecar_path,
                root / "fit",
                times,
                target,
                self._fake_evaluator(train_penetration_m=0.025),
                max_nfev=1,
            )

            trace = json.loads((output / "evaluation_trace.json").read_text(encoding="utf-8"))
            penalties = [entry["residual_term_sumsq"]["training_side_penetration_above_0_020_m"] for entry in trace]
            self.assertTrue(all(value > 0.0 for value in penalties))
            held_out = json.loads((output / "held_out_metrics.json").read_text(encoding="utf-8"))
            self.assertIs(held_out["used_by_optimizer"], False)

    def test_publish_trace_hashes_and_strict_preliminary_scope(self):
        """Publish every fit record atomically without claiming Stage 2 or FD acceptance."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sidecar_path = self._write_sidecar(root)
            times, target = self._target()
            output = contact_calibration.calibrate_normal_contact(
                sidecar_path,
                root / "fit",
                times,
                target,
                self._fake_evaluator(),
                max_nfev=1,
            )

            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            optimizer = json.loads((output / "optimizer_result.json").read_text(encoding="utf-8"))
            trace = json.loads((output / "evaluation_trace.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["scope"], "preliminary_stage_2_normal_contact_only")
            self.assertIs(manifest["claims"]["complete_stage_2_calibration"], False)
            self.assertIs(manifest["claims"]["forward_dynamics"], False)
            self.assertEqual(manifest["evaluator_provenance"]["kind"], "injected_callable")
            self.assertEqual(manifest["evaluator_provenance"]["measured_input_isolation"], "unverifiable")
            self.assertEqual(manifest["information_set"]["measured_input_isolation"], "unverifiable")
            self.assertIn("git", manifest["runtime_provenance"])
            self.assertEqual(
                manifest["runtime_provenance"]["code"]["sha256"],
                hashlib.sha256(Path(contact_calibration.__file__).read_bytes()).hexdigest(),
            )
            self.assertIs(manifest["claims"]["optimizer_success"], optimizer["success"])
            self.assertEqual(
                manifest["status"],
                (
                    "preliminary_normal_contact_fit_succeeded"
                    if optimizer["success"]
                    else "preliminary_normal_contact_fit_optimizer_unsuccessful"
                ),
            )
            self.assertGreaterEqual(len(trace), 1)
            self.assertEqual(optimizer["traced_evaluator_call_count"], len(trace))
            for record in manifest["artifacts"].values():
                artifact = output / record["path"]
                self.assertEqual(hashlib.sha256(artifact.read_bytes()).hexdigest(), record["sha256"])
            calibrated = predictive_contact.load_contact_sidecar(output / "calibrated_contact_sidecar.json")
            self.assertEqual(calibrated.calibration.held_out_side, "right")

    def test_reject_repository_and_overlapping_output_paths(self):
        """Reject unsafe destinations before evaluating or creating an artifact."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            sidecar_path = self._write_sidecar(root)
            times, target = self._target()
            repository_output = Path(contact_calibration.__file__).resolve().parents[2] / "unsafe_calibration_output"

            with self.assertRaisesRegex(ValueError, "outside the repository"):
                contact_calibration.calibrate_normal_contact(
                    sidecar_path,
                    repository_output,
                    times,
                    target,
                    self._fake_evaluator(),
                    max_nfev=1,
                )
            with self.assertRaisesRegex(ValueError, "must not overlap"):
                contact_calibration.calibrate_normal_contact(
                    sidecar_path,
                    source / "nested",
                    times,
                    target,
                    self._fake_evaluator(),
                    max_nfev=1,
                    source_dir=source,
                )
            self.assertFalse(repository_output.exists())
            self.assertFalse((source / "nested").exists())


if __name__ == "__main__":
    unittest.main()
