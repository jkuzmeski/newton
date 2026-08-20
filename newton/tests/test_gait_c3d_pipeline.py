# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the project-local C3D gait pipeline helpers."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from projects.gait_c3d import pipeline, torque_reconstruction


class TestGaitC3DPipeline(unittest.TestCase):
    """Verify deterministic parsing, transforms, events, and artifact contracts."""

    @staticmethod
    def _write_metric(path: Path, values: np.ndarray) -> None:
        """Write a privacy-safe Visual3D-like belt metric fixture."""
        rows = ["SPEEDCHANGE", "METRIC", "PROCESSED", "synthetic", "ITEM VALUE"]
        rows.extend(f"{index} {value:.6f}" for index, value in enumerate(values, start=1))
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    @staticmethod
    def _external_load_labels() -> list[str]:
        """Build the required external-load column order."""
        labels: list[str] = []
        for side in ("l", "r"):
            labels.extend(f"ground_force_{side}_v{axis}" for axis in "xyz")
            labels.extend(f"ground_force_{side}_p{axis}" for axis in "xyz")
            labels.extend(f"ground_torque_{side}_{axis}" for axis in "xyz")
        return labels

    def test_read_visual3d_metrics_strictly(self):
        """Preserve metric samples and reject malformed or nonsequential rows."""
        with tempfile.TemporaryDirectory() as tmpdir:
            metric = Path(tmpdir) / "BeltSynthetic.txt"
            self._write_metric(metric, np.array([0.0, 1.5, 3.0]))

            items, speed = pipeline.read_visual3d_metric(metric)

            np.testing.assert_array_equal(items, [1.0, 2.0, 3.0])
            np.testing.assert_array_equal(speed, [0.0, 1.5, 3.0])

            metric.write_text(
                "one\ntwo\nthree\nfour\nfive\n1 0.0\n3 3.0\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "non-sequential"):
                pipeline.read_visual3d_metric(metric)

            metric.write_text(
                "one\ntwo\nthree\nfour\nfive\n1 0.0 extra\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "two numeric fields"):
                pipeline.read_visual3d_metric(metric)

    def test_require_identical_tied_belt_exports(self):
        """Accept identical tied belts and reject a differing right-belt sample."""
        with tempfile.TemporaryDirectory() as tmpdir:
            incoming = Path(tmpdir)
            values = np.ones(53224)
            self._write_metric(incoming / "LeftBelt101.txt", values)
            self._write_metric(incoming / "RightBelt101.txt", values)

            speed, displacement, registration = pipeline._load_belt(
                incoming,
                np.array([0.0, 1.0, 2.0]),
            )

            np.testing.assert_array_equal(speed, 1.0)
            np.testing.assert_allclose(displacement, [0.0, 1.0, 2.0])
            self.assertIs(registration["left_right_identical"], True)
            self.assertEqual(registration["units"], "m/s")

            values[-2] = 0.5
            self._write_metric(incoming / "RightBelt101.txt", values)
            with self.assertRaisesRegex(ValueError, "tied, identical belt profiles"):
                pipeline._load_belt(incoming, np.array([0.0, 1.0, 2.0]))

    def test_invalidate_cache_when_any_belt_source_changes(self):
        """Include every belt export and extraction setting in cache provenance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            incoming = Path(tmpdir)
            for name in (
                "Trial 101.v3d.c3d",
                "LeftBelt101.txt",
                "RightBelt101.txt",
                "Speedchange101.txt",
            ):
                (incoming / name).write_bytes(name.encode())
            before = pipeline._cache_provenance(incoming)
            (incoming / "LeftBelt101.txt").write_bytes(b"changed belt")
            after = pipeline._cache_provenance(incoming)

            self.assertNotEqual(
                before["source_hashes"]["LeftBelt101.txt"],
                after["source_hashes"]["LeftBelt101.txt"],
            )
            self.assertEqual(before["extraction_config"], after["extraction_config"])

    def test_anchor_metric_clock_to_c3d_seconds(self):
        """Reproduce every documented item-to-C3D clock anchor exactly."""
        anchor_indices = np.array([0, 1356, 22244, 43139, 52098, 53223])
        anchor_seconds = np.array([0.0, 4.68, 74.69, 144.70, 174.71, 178.46])

        clock = pipeline.register_belt_clock(53224)

        np.testing.assert_array_equal(clock[anchor_indices], anchor_seconds)
        self.assertTrue(np.all(np.diff(clock) > 0.0))
        self.assertNotAlmostEqual(clock[43139], 43139.0 / 300.0)
        with self.assertRaisesRegex(ValueError, "cover the complete metric"):
            pipeline.register_belt_clock(53223)
        with self.assertRaisesRegex(ValueError, "increase strictly"):
            pipeline.register_belt_clock(
                3,
                anchor_indices=np.array([0, 1, 2]),
                anchor_times=np.array([0.0, 1.0, 1.0]),
            )

    def test_integrate_belt_speed_with_trapezoids(self):
        """Integrate nonuniform registered speed without losing the zero origin."""
        displacement = pipeline.integrate_speed(
            np.array([0.0, 1.0, 3.0]),
            np.array([0.0, 2.0, 2.0]),
        )

        np.testing.assert_allclose(displacement, [0.0, 1.0, 5.0])
        with self.assertRaisesRegex(ValueError, "increase strictly"):
            pipeline.integrate_speed([0.0, 1.0, 1.0], [1.0, 1.0, 1.0])

    def test_make_a_treadmill_stance_foot_stationary_overground(self):
        """Apply xOG=xTM+s so a stance foot moving with the belt is stationary."""
        times = np.linspace(0.0, 1.0, 11)
        speed = np.full(len(times), 2.0)
        displacement = pipeline.integrate_speed(times, speed)
        treadmill = np.zeros((len(times), 2, 3))
        treadmill[:, :, 0] = -displacement[:, None]

        overground = pipeline.treadmill_to_overground(treadmill, displacement)
        qc = pipeline._stance_speed_qc(
            times,
            ["L.Heel", "R.Heel"],
            treadmill,
            overground,
            np.ones((len(times), 2), dtype=bool),
        )

        np.testing.assert_allclose(overground[..., 0], 0.0, atol=1.0e-15)
        self.assertAlmostEqual(qc["left"]["treadmill_mean_mps"], -2.0)
        self.assertAlmostEqual(qc["right"]["treadmill_mean_mps"], -2.0)
        self.assertAlmostEqual(qc["left"]["overground_rms_mps"], 0.0)
        self.assertAlmostEqual(qc["right"]["overground_rms_mps"], 0.0)

    def test_reference_overground_displacement_to_the_stride_start(self):
        """Retain absolute belt travel while translating by relative stride travel."""
        points = np.zeros((3, 1, 3))
        absolute_displacement = np.array([10.0, 10.5, 11.25])

        mapped = pipeline.treadmill_to_overground(
            points,
            absolute_displacement,
            reference_index=1,
        )

        np.testing.assert_allclose(mapped[:, 0, 0], [-0.5, 0.0, 0.75])
        np.testing.assert_array_equal(absolute_displacement, [10.0, 10.5, 11.25])
        with self.assertRaisesRegex(IndexError, "reference_index"):
            pipeline.treadmill_to_overground(points, absolute_displacement, reference_index=3)

    def test_rotate_and_shift_force_platform_arrays(self):
        """Rotate lab axes, shift only COP x, and preserve force/free-torque vectors."""
        rotation = np.array(
            [
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
            ]
        )
        force_lab = np.zeros((3, 2, 3))
        force_lab[:, :, 2] = 100.0
        cop_lab_mm = np.zeros_like(force_lab)
        cop_lab_mm[:, :, :] = [250.0, 200.0, 0.0]
        torque_lab_nmm = np.zeros_like(force_lab)
        torque_lab_nmm[:, :, 2] = 5000.0
        displacement = np.array([4.0, 4.5, 5.25])
        contact = np.ones((3, 2), dtype=bool)

        force, cop, torque = pipeline.transform_force_platform_arrays(
            force_lab,
            cop_lab_mm,
            torque_lab_nmm,
            rotation,
            displacement,
            contact,
            reference_index=0,
        )

        expected_force = np.zeros_like(force)
        expected_force[:, :, 1] = 100.0
        expected_torque = np.zeros_like(torque)
        expected_torque[:, :, 1] = 5.0
        np.testing.assert_allclose(force, expected_force, atol=1.0e-15)
        np.testing.assert_allclose(torque, expected_torque, atol=1.0e-15)
        expected_cop = np.empty_like(cop)
        expected_cop[:, :, 0] = np.array([-0.2, 0.3, 1.05])[:, None]
        expected_cop[:, :, 1] = 0.0
        expected_cop[:, :, 2] = -0.25
        np.testing.assert_allclose(cop, expected_cop, atol=1.0e-15)
        self.assertAlmostEqual(np.linalg.det(rotation), 1.0)

        contact[1, 1] = False
        force, cop, torque = pipeline.transform_force_platform_arrays(
            force_lab,
            cop_lab_mm,
            torque_lab_nmm,
            rotation,
            displacement,
            contact,
        )
        np.testing.assert_array_equal(force[1, 1], 0.0)
        np.testing.assert_array_equal(torque[1, 1], 0.0)
        self.assertTrue(np.all(np.isnan(cop[1, 1])))

    def test_filter_force_and_moment_before_deriving_cop_and_free_torque(self):
        """Derive a varying filtered wrench that preserves its moment identity."""
        butter, sosfiltfilt = pipeline._signal_tools()
        sos = butter(4, 20.0, btype="low", fs=1000.0, output="sos")
        time = np.arange(1000, dtype=float) / 1000.0
        loaded = np.zeros(1000, dtype=bool)
        loaded[250:750] = True
        force = np.zeros((1000, 3))
        force[loaded, 0] = 80.0 + 30.0 * np.sin(2.0 * np.pi * 13.0 * time[loaded])
        force[loaded, 1] = -40.0 + 20.0 * np.cos(2.0 * np.pi * 11.0 * time[loaded])
        force[loaded, 2] = 800.0 + 200.0 * np.sin(2.0 * np.pi * 17.0 * time[loaded])
        corners = np.array(
            [
                [0.0, 500.0, 500.0, 0.0],
                [850.0, 850.0, -850.0, -850.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        platform_origin = np.mean(corners, axis=1)
        application_point = np.repeat(platform_origin[None], len(time), axis=0)
        application_point[loaded, 0] += 120.0 + 80.0 * np.sin(2.0 * np.pi * 19.0 * time[loaded])
        application_point[loaded, 1] += -200.0 + 60.0 * np.cos(2.0 * np.pi * 17.0 * time[loaded])
        free_torque = np.zeros_like(force)
        free_torque[loaded, 2] = 15000.0 + 2000.0 * np.sin(2.0 * np.pi * 7.0 * time[loaded])
        moment = np.cross(application_point - platform_origin, force) + free_torque

        filtered_force, cop, torque, contact, filtered_moment, identity_error = pipeline.filter_force_platform_wrench(
            force,
            moment,
            corners,
            sos,
            contact_threshold_n=50.0,
        )

        reconstructed_moment = np.cross(cop[contact] - platform_origin, filtered_force[contact]) + torque[contact]
        np.testing.assert_allclose(reconstructed_moment, filtered_moment[contact], atol=1.0e-8)
        self.assertLess(identity_error, 1.0e-8)
        legacy_cop = np.full_like(application_point, np.nan)
        legacy_cop[loaded] = sosfiltfilt(sos, application_point[loaded], axis=0)
        legacy_torque = sosfiltfilt(sos, free_torque, axis=0)
        legacy_valid = contact & np.all(np.isfinite(legacy_cop), axis=1)
        legacy_moment = (
            np.cross(legacy_cop[legacy_valid] - platform_origin, filtered_force[legacy_valid])
            + legacy_torque[legacy_valid]
        )
        self.assertGreater(np.max(np.abs(legacy_moment - filtered_moment[legacy_valid])), 1000.0)
        np.testing.assert_array_equal(filtered_force[~contact], 0.0)
        np.testing.assert_array_equal(torque[~contact], 0.0)
        self.assertTrue(np.all(np.isnan(cop[~contact])))
        invalid_moment = moment.copy()
        invalid_moment[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "must be finite"):
            pipeline.filter_force_platform_wrench(
                force,
                invalid_moment,
                corners,
                sos,
                contact_threshold_n=50.0,
            )

    def test_gate_cop_proximity_to_the_assigned_foot(self):
        """Accept anatomically associated COPs and reject a bilateral side swap."""
        marker_names = ["L.Heel", "L.Toe.Tip", "R.Heel", "R.Toe.Tip"]
        frame = np.array(
            [
                [0.0, 0.08, -0.1],
                [0.25, 0.03, -0.1],
                [0.5, 0.08, 0.1],
                [0.75, 0.03, 0.1],
            ]
        )
        markers = np.repeat(frame[None], 3, axis=0)
        cop = np.repeat(np.array([[[0.12, 0.0, -0.1], [0.62, 0.0, 0.1]]]), 3, axis=0)
        grf = np.zeros_like(cop)
        grf[:, :, 1] = 300.0
        contact = np.ones((3, 2), dtype=bool)

        result = pipeline.cop_foot_proximity_qc(marker_names, markers, cop, grf, contact)

        self.assertIs(result["passed"], True)
        self.assertEqual(result["sides"]["left"]["ipsilateral_closer_fraction"], 1.0)
        self.assertEqual(result["sides"]["right"]["ipsilateral_closer_fraction"], 1.0)
        swapped = pipeline.cop_foot_proximity_qc(marker_names, markers, cop[:, ::-1], grf, contact)
        self.assertIs(swapped["passed"], False)

    def test_detect_contact_runs_and_select_a_complete_stride(self):
        """Detect sustained contact bouts and select the first eligible stride."""
        times = np.arange(12, dtype=float) * 0.1
        contact = np.array([False, True, True, False, False, True, True, False, True, True, False, False])

        runs = pipeline.contact_runs(contact, min_frames=2)
        stride = pipeline.select_stride(times, contact, search_time=0.4)

        self.assertEqual(runs, [(1, 3), (5, 7), (8, 10)])
        self.assertEqual(stride, (5, 9))
        with self.assertRaisesRegex(ValueError, "fewer than two left contacts"):
            pipeline.select_stride(times, contact, search_time=0.6)

    def test_use_actual_vertical_force_in_friction_ratio(self):
        """Compute friction ratios from actual active vertical force without a hidden floor."""
        grf = np.zeros((20, 2, 3))
        grf[:, :, 0] = 50.0
        grf[:, :, 1] = 60.0
        contact = np.ones((20, 2), dtype=bool)

        _signs, friction = pipeline._force_qc(grf, contact)

        self.assertAlmostEqual(friction["left"]["peak_horizontal_over_vertical"], 5.0 / 6.0)
        self.assertAlmostEqual(friction["right"]["minimum_vertical_force_N"], 60.0)

    def test_validate_external_load_schema_order(self):
        """Accept exact load triplets and reject missing or misordered columns."""
        labels = self._external_load_labels()
        pipeline.validate_external_load_schema(labels)

        missing = labels[:-1]
        with self.assertRaisesRegex(ValueError, "required schema"):
            pipeline.validate_external_load_schema(missing)

        misordered = labels.copy()
        misordered[0], misordered[1] = misordered[1], misordered[0]
        with self.assertRaisesRegex(ValueError, "required schema"):
            pipeline.validate_external_load_schema(misordered)

    def test_sample_only_archived_external_load_frames(self):
        """Reuse sanitized load frames exactly and reject interpolated requests."""
        times = np.array([1.0, 1.1, 1.2, 1.3])
        wrenches = np.arange(4 * 2 * 9, dtype=float).reshape(4, 2, 9)
        loads = pipeline._SampledExternalLoads(times, ["left", "right"], wrenches)

        bodies, subset = loads.sample(times[[0, 2, 3]])

        self.assertEqual(bodies, ["left", "right"])
        np.testing.assert_array_equal(subset, wrenches[[0, 2, 3]])
        with self.assertRaisesRegex(ValueError, "outside the sampled time grid"):
            loads.sample(np.array([1.05]))

    def test_interpolate_torque_reconstruction_inputs_without_extrapolation(self):
        """Interpolate all trailing components and reject times outside the archive."""
        times = np.array([0.0, 1.0, 2.0])
        values = np.zeros((3, 2, 2))
        values[:, 0, 0] = [0.0, 2.0, 4.0]
        values[:, 1, 1] = [1.0, 3.0, 5.0]

        sampled = torque_reconstruction._interpolate(times, values, np.array([0.5, 1.5]))

        self.assertEqual(sampled.shape, (2, 2, 2))
        np.testing.assert_allclose(sampled[:, 0, 0], [1.0, 3.0])
        np.testing.assert_allclose(sampled[:, 1, 1], [2.0, 4.0])
        with self.assertRaisesRegex(ValueError, "outside the archived trajectory"):
            torque_reconstruction._interpolate(times, values, np.array([2.1]))

    def test_reject_overlapping_torque_reconstruction_paths(self):
        """Keep diagnostic replacement paths disjoint from immutable source artifacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_dir = root / "source" / "latest"
            data_dir.mkdir(parents=True)

            with self.assertRaisesRegex(ValueError, "must not overlap"):
                torque_reconstruction.run_reconstruction(data_dir, data_dir / "diagnostic")
            with self.assertRaisesRegex(ValueError, "must not overlap"):
                torque_reconstruction.run_reconstruction(data_dir, root / "source")

            output_file = root / "diagnostic"
            output_file.write_text("not a directory", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not a directory"):
                torque_reconstruction.run_reconstruction(data_dir, output_file)

    def test_separate_rotational_and_translational_reconstruction_errors(self):
        """Report dynamics errors without mixing angular and linear units."""
        error = np.array([[3.0, 4.0], [0.0, 0.0]])

        summary = torque_reconstruction._error_summary(error, ["rotational", "translational"])

        self.assertEqual(summary["rotational"]["count"], 1)
        self.assertAlmostEqual(summary["rotational"]["max_abs"], 3.0)
        self.assertEqual(summary["translational"]["count"], 1)
        self.assertAlmostEqual(summary["translational"]["max_abs"], 4.0)

    def test_preserve_marker_units_as_metres(self):
        """Label C3D marker values as metres after numerical conversion."""
        markers = pipeline._marker_data_meters(
            np.array([0.0, 0.01]),
            ["LHEE"],
            np.array([[[0.1, 0.2, 0.3]], [[0.2, 0.2, 0.3]]]),
            100.0,
        )

        self.assertEqual(markers.units, "m")
        np.testing.assert_allclose(markers.data[0, 0], [0.1, 0.2, 0.3])

    def test_load_synthetic_artifacts_through_the_viewer(self):
        """Load a privacy-safe analysis contract and complete viewer semantic checks."""
        import newton.examples  # noqa: PLC0415
        import newton.opensim as opensim  # noqa: PLC0415
        import newton.viewer  # noqa: PLC0415
        from projects.gait_c3d.viewer import Example  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            model_source = Path(newton.examples.get_asset("gait2354_subject01.osim"))
            model_path = directory / "S001_scaled.osim"
            shutil.copy2(model_source, model_path)
            model = opensim.parse_osim(model_path)
            fk = opensim.ForwardKinematics(model, device="cpu")
            coordinate_names = list(fk.coordinate_names)
            defaults = np.array(
                [coordinate.default_value for joint in model.joints for coordinate in joint.coordinates], float
            )
            frame_count = 4
            coords = np.repeat(defaults[None], frame_count, axis=0)
            coords[:, coordinate_names.index("pelvis_tx")] += np.linspace(0.0, 0.3, frame_count)
            markers = fk.marker_positions_batch(coords)
            times = np.arange(frame_count, dtype=float) * 0.01
            grf = np.zeros((frame_count, 2, 3))
            grf[:, :, 1] = 400.0
            cop = np.zeros_like(grf)
            contact = np.ones((frame_count, 2), dtype=bool)
            np.savez_compressed(
                directory / "analysis.npz",
                times=times,
                coords=coords,
                coordinate_names=np.asarray(coordinate_names, dtype="U"),
                target_markers=markers,
                predicted_markers=markers,
                marker_names=np.asarray(fk.marker_names, dtype="U"),
                grf=grf,
                cop=cop,
                free_torque=np.zeros_like(grf),
                contact=contact,
                belt_speed=np.full(frame_count, 1.0),
                belt_displacement_relative=np.linspace(0.0, 0.3, frame_count),
                belt_displacement_absolute=np.linspace(10.0, 10.3, frame_count),
                com=fk.center_of_mass_batch(coords),
                activations=np.empty((frame_count, 0)),
                muscle_names=np.empty(0, dtype="U"),
            )
            pipeline.write_json(
                directory / "qc_summary.json",
                {
                    "status": "synthetic_test",
                    "warnings": [],
                    "gates": {},
                    "stance_heel_speeds": {
                        "left": {"treadmill_rms_mps": 1.0, "overground_rms_mps": 0.1},
                        "right": {"treadmill_rms_mps": 1.0, "overground_rms_mps": 0.1},
                    },
                },
            )
            args = SimpleNamespace(
                data_dir=str(directory),
                geometry=None,
                download_geometry=False,
                residual_scale=8.0,
                grf_scale=5.0e-4,
                com_trail_frames=10,
                show_treadmill_ghost=True,
                ghost_lane_offset=-0.9,
            )
            viewer = newton.viewer.ViewerNull(num_frames=frame_count)
            example = Example(viewer, args)
            for _frame in range(frame_count):
                example.render()
                example.step()
            example.test_final()

    def test_serialize_provenance_and_qc_deterministically(self):
        """Hash synthetic provenance and serialize nested NumPy QC values to JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            source = directory / "anonymous-input.bin"
            source.write_bytes(b"privacy-safe synthetic gait input\n")
            expected = hashlib.sha256(source.read_bytes()).hexdigest()
            self.assertEqual(pipeline.sha256(source), expected)

            qc_path = directory / "qc_summary.json"
            qc = {
                "passed": np.bool_(True),
                "count": np.int64(3),
                "rms": np.float64(0.012),
                "vector": np.array([1.0, 2.0]),
                "nested": (np.int32(4), {"finite": np.float32(5.0)}),
            }
            pipeline.write_json(qc_path, qc)
            encoded = qc_path.read_text(encoding="utf-8")
            decoded = json.loads(encoded)

            self.assertTrue(encoded.endswith("\n"))
            self.assertLess(encoded.index('"count"'), encoded.index('"passed"'))
            self.assertIs(decoded["passed"], True)
            self.assertEqual(decoded["count"], 3)
            self.assertEqual(decoded["vector"], [1.0, 2.0])
            self.assertEqual(decoded["nested"], [4, {"finite": 5.0}])


if __name__ == "__main__":
    unittest.main()
