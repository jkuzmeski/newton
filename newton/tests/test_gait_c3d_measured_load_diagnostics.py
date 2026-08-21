# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Stage 1 C3D measured-load diagnostic harness."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from projects.gait_c3d import measured_load_diagnostics as diagnostics


class _FakeForward:
    """Provide deterministic finite trajectories and diagonal mass matrices."""

    def __init__(self, coordinate_count: int):
        self.coordinate_count = coordinate_count

    def simulate(
        self,
        initial_coordinates,
        initial_speeds,
        duration,
        dt,
        *,
        start_time,
        controls,
        external_loads,
        integrator,
        use_graph,
    ):
        """Return a stationary trajectory while evaluating callback endpoints."""
        del integrator, use_graph
        steps = int(round(duration / dt))
        times = start_time + np.arange(steps + 1) * dt
        coordinates = np.repeat(np.asarray(initial_coordinates)[None], steps + 1, axis=0)
        speeds = np.repeat(np.asarray(initial_speeds)[None], steps + 1, axis=0)
        for time, q, qd in zip(times, coordinates, speeds, strict=True):
            controls(float(time), q, qd)
            external_loads.sample(np.asarray([time]))
        return SimpleNamespace(times=times, coordinates=coordinates, speeds=speeds)

    def mass_matrix(self, coordinates):
        """Return an identity matrix for every input state."""
        values = np.asarray(coordinates)
        return np.repeat(np.eye(self.coordinate_count)[None], len(values), axis=0)


class _FakeKinematics:
    """Provide simple marker, COM, body-pose, and velocity results."""

    marker_names = ("marker",)
    body_names = ("foot_l", "foot_r")

    def marker_positions_batch(self, coordinates):
        """Map the first three coordinates to one synthetic marker."""
        values = np.asarray(coordinates)
        return values[:, None, :3]

    def center_of_mass_batch(self, coordinates):
        """Return a fixed COM one metre above ground."""
        return np.tile([0.0, 1.0, 0.0], (len(coordinates), 1))

    def body_velocities_batch(self, coordinates, speeds):
        """Return zero body velocities."""
        del speeds
        shape = (len(coordinates), 2, 3)
        return {"angular_velocity": np.zeros(shape), "linear_velocity": np.zeros(shape)}

    def body_transforms_batch(self, coordinates):
        """Return identity body transforms."""
        return np.repeat(np.eye(4)[None, None], len(coordinates) * 2, axis=0).reshape(len(coordinates), 2, 4, 4)


class TestMeasuredLoadDiagnostics(unittest.TestCase):
    """Verify sampling, controller, scheduling, metrics, and publication contracts."""

    @staticmethod
    def _fixture():
        """Build a three-coordinate bilateral measured-load fixture."""
        coordinates = [
            SimpleNamespace(name="not_named_pelvis", range=(-10.0, 10.0)),
            SimpleNamespace(name="root_translation", range=(-10.0, 10.0)),
        ]
        internal = SimpleNamespace(name="pelvis_named_but_internal", range=(-10.0, 10.0))
        model = SimpleNamespace(
            joints=[
                SimpleNamespace(parent_body="ground", coordinates=coordinates),
                SimpleNamespace(parent_body="torso", coordinates=[internal]),
            ],
            bodies=[SimpleNamespace(mass=2.0), SimpleNamespace(mass=3.0)],
            gravity=(0.0, -10.0, 0.0),
        )
        times = np.arange(5, dtype=float) * 0.01
        q = np.zeros((5, 3))
        qd = np.zeros_like(q)
        tau = np.tile([100.0, -100.0, 20.0], (5, 1))
        wrenches = np.zeros((5, 2, 9))
        wrenches[:, 0, 0] = 1.0
        wrenches[:, 1, 0] = 2.0
        trajectory = diagnostics.MeasuredLoadTrajectory(
            times=times,
            coordinates=q,
            speeds=qd,
            generalized_forces=tau,
            coordinate_names=("not_named_pelvis", "root_translation", "pelvis_named_but_internal"),
            motion_types=("rotational", "translational", "rotational"),
            external_bodies=("foot_l", "foot_r"),
            external_wrenches=wrenches,
        )
        return model, trajectory

    def test_sample_linear_and_causal_zoh_without_extrapolation(self):
        """Hit both endpoints and reject lower and upper extrapolation."""
        times = np.array([1.0, 2.0, 4.0])
        values = np.array([[10.0], [20.0], [40.0]])
        linear = diagnostics.StrictSampler(times, values, "linear")
        zoh = diagnostics.StrictSampler(times, values, "zoh")

        np.testing.assert_allclose(linear.sample([1.0, 1.5, 4.0])[:, 0], [10.0, 15.0, 40.0])
        np.testing.assert_array_equal(zoh.sample([1.0, 1.999, 2.0, 4.0])[:, 0], [10.0, 10.0, 20.0, 40.0])
        with self.assertRaisesRegex(ValueError, "outside"):
            linear.sample([0.9])
        with self.assertRaisesRegex(ValueError, "outside"):
            zoh.sample([4.1])

    def test_reject_nonfinite_or_mutable_sampler_sources(self):
        """Reject nonfinite values and isolate accepted data from later mutation."""
        times = np.array([0.0, 1.0])
        values = np.array([[1.0], [2.0]])
        sampler = diagnostics.StrictSampler(times, values)
        values[:] = 99.0
        np.testing.assert_array_equal(sampler.sample([0.0, 1.0])[:, 0], [1.0, 2.0])
        with self.assertRaisesRegex(ValueError, "finite"):
            diagnostics.StrictSampler(times, np.array([[1.0], [np.nan]]))

    def test_identify_root_structurally_instead_of_by_name(self):
        """Select ground-joint coordinates even when names are misleading."""
        model, trajectory = self._fixture()
        mask = diagnostics.structural_root_mask(model, trajectory.coordinate_names)
        np.testing.assert_array_equal(mask, [True, True, False])

    def test_identify_all_six_actual_root_coordinate_roles(self):
        """Identify three rotational and three translational ground coordinates."""
        names = ("pelvis_tilt", "pelvis_list", "pelvis_rotation", "pelvis_tx", "pelvis_ty", "pelvis_tz")
        root_coordinates = [SimpleNamespace(name=name) for name in names]
        model = SimpleNamespace(
            joints=[
                SimpleNamespace(parent_body="ground", coordinates=root_coordinates),
                SimpleNamespace(parent_body="pelvis", coordinates=[SimpleNamespace(name="hip_flexion_l")]),
            ]
        )

        mask = diagnostics.structural_root_mask(model, (*names, "hip_flexion_l"))

        np.testing.assert_array_equal(mask, [True, True, True, True, True, True, False])

    def test_bound_nonroot_feedforward_and_feedback_with_exact_zero_root(self):
        """Enforce total headroom and exact-zero root force in every component."""
        model, trajectory = self._fixture()
        root = diagnostics.structural_root_mask(model, trajectory.coordinate_names)
        config = diagnostics.ControllerConfig(
            rotational_kp=10.0,
            rotational_kd=1.0,
            rotational_effort_limit=5.0,
            translational_kp=10.0,
            translational_kd=1.0,
            translational_effort_limit=5.0,
        )
        controller = diagnostics.BoundedNonRootController(trajectory, root, config)

        value = controller.evaluate(0.01, np.array([2.0, 2.0, -1.0]), np.zeros(3))

        for component in (
            value.raw_feedforward,
            value.feedforward,
            value.raw_feedback,
            value.feedback,
            value.total,
        ):
            np.testing.assert_array_equal(component[root], 0.0)
        self.assertLessEqual(abs(value.total[2]), 5.0)
        np.testing.assert_allclose(value.total, value.feedforward + value.feedback)
        self.assertTrue(value.saturated[2])

    def test_select_bilateral_load_variants_without_mutating_source(self):
        """Preserve body order and zero only the unselected bilateral load."""
        _, trajectory = self._fixture()
        source = trajectory.external_wrenches.copy()
        bodies, left = diagnostics.select_load_variant(trajectory.external_bodies, trajectory.external_wrenches, "left")
        _, right = diagnostics.select_load_variant(trajectory.external_bodies, trajectory.external_wrenches, "right")
        _, none = diagnostics.select_load_variant(trajectory.external_bodies, trajectory.external_wrenches, "none")

        self.assertEqual(bodies, ["foot_l", "foot_r"])
        np.testing.assert_array_equal(left[:, 0], source[:, 0])
        np.testing.assert_array_equal(left[:, 1], 0.0)
        np.testing.assert_array_equal(right[:, 0], 0.0)
        np.testing.assert_array_equal(right[:, 1], source[:, 1])
        np.testing.assert_array_equal(none, 0.0)
        np.testing.assert_array_equal(trajectory.external_wrenches, source)

    def test_schedule_every_restart_and_mark_unavailable_edge_cells(self):
        """Keep all 107 by 3 cells without shortening boundary windows."""
        times = 20.6 + np.arange(107) * 0.01
        schedule = diagnostics.restart_schedule(times, (0.025, 0.05, 0.1))

        self.assertEqual(len(schedule), 107 * 3)
        counts = {
            horizon: sum(cell.status == "scheduled" and cell.horizon_s == horizon for cell in schedule)
            for horizon in (0.025, 0.05, 0.1)
        }
        self.assertEqual(counts, {0.025: 104, 0.05: 102, 0.1: 97})
        self.assertEqual(schedule[-1].status, "unavailable_source_boundary")

    def test_recover_known_absolute_error_growth_slopes(self):
        """Recover per-coordinate least-squares slopes on a synthetic window."""
        times = np.array([2.0, 2.1, 2.2, 2.3])
        elapsed = times - times[0]
        error = np.column_stack((2.0 * elapsed, -3.0 * elapsed))
        slopes = diagnostics._growth_slopes(times, error)
        np.testing.assert_allclose(slopes, [2.0, 3.0], atol=1.0e-14)

    def test_apply_predeclared_refinement_trigger(self):
        """Use direct common-grid errors rather than equal aggregate norms."""
        times = np.array([0.0, 0.5, 1.0])

        def record(values, dt):
            array = np.asarray(values, dtype=float)[:, None]
            return SimpleNamespace(
                actual_dt_s=dt,
                metrics={"completed": True, "finite_sample_count": len(times)},
                times=times,
                coordinates=array,
                speeds=array,
                marker_error_m=array,
                total_energy_j=array[:, 0],
            )

        fine = record([0.0, 1.0, 0.0], 0.00025)
        close = record([0.0, 1.01, 0.0], 0.0005)
        opposite_with_equal_rms = record([0.0, -1.0, 0.0], 0.0005)

        self.assertFalse(diagnostics.should_run_refinement([fine, close], 0.05, times)[0])
        self.assertTrue(diagnostics.should_run_refinement([fine, opposite_with_equal_rms], 0.05, times)[0])
        fine.metrics["completed"] = False
        self.assertTrue(diagnostics.should_run_refinement([fine, close], 0.05, times)[0])

    def test_compare_convergence_on_unequal_grids_and_recover_order(self):
        """Interpolate unequal output grids and recover second-order convergence."""
        source_times = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        records = []
        for timestep in (0.4, 0.2, 0.1):
            times = np.append(np.arange(0.0, 1.0, timestep), 1.0)
            error = timestep**2 * times
            records.append(
                SimpleNamespace(
                    actual_dt_s=timestep,
                    metrics={"finite_sample_count": len(times)},
                    times=times,
                    coordinates=error[:, None],
                    speeds=(2.0 * error)[:, None],
                    marker_error_m=(3.0 * error)[:, None],
                    total_energy_j=4.0 * error,
                )
            )

        summary = diagnostics.summarize_convergence(records, source_times)

        comparisons = summary["successive_common_grid_comparisons"]
        self.assertEqual(len(comparisons), 2)
        self.assertEqual(comparisons[0]["common_grid_sample_count"], len(source_times))
        self.assertAlmostEqual(summary["observed_order"]["coordinate"], 2.0)
        self.assertAlmostEqual(summary["observed_order"]["speed"], 2.0)

    def test_compute_analytic_energy_condition_spd_and_nonzero_external_power(self):
        """Match analytic energy and transported nonzero wrench power."""

        class DiagonalForward:
            def mass_matrix(self, coordinates):
                """Return a fixed positive-definite diagonal mass matrix."""
                return np.repeat(np.diag([2.0, 4.0, 6.0])[None], len(coordinates), axis=0)

        class MovingKinematics(_FakeKinematics):
            def body_velocities_batch(self, coordinates, speeds):
                """Return a moving left foot origin with angular velocity."""
                del speeds
                angular = np.zeros((len(coordinates), 2, 3))
                linear = np.zeros_like(angular)
                angular[:, 0, 2] = 2.0
                linear[:, 0, 0] = 1.0
                return {"angular_velocity": angular, "linear_velocity": linear}

        model, _ = self._fixture()
        times = np.array([0.0, 0.1])
        coordinates = np.zeros((2, 3))
        speeds = np.tile([1.0, 2.0, 3.0], (2, 1))
        zeros = np.zeros((2, 3))
        wrenches = np.zeros((2, 2, 9))
        wrenches[:, 0, :3] = [3.0, 4.0, 0.0]
        wrenches[:, 0, 3:6] = [0.0, 1.0, 0.0]
        wrenches[:, 0, 6:9] = [0.0, 0.0, 5.0]

        (
            kinetic,
            potential,
            energy,
            condition,
            symmetry_error,
            minimum_eigenvalue,
            cholesky_success,
            external,
            feedforward,
            feedback,
        ) = diagnostics._energy_and_power(
            DiagonalForward(),
            MovingKinematics(),
            model,
            times,
            coordinates,
            speeds,
            zeros,
            zeros,
            ["foot_l", "foot_r"],
            wrenches,
            1,
            1.0e-10,
        )

        np.testing.assert_allclose(kinetic, 0.5 * (2.0 + 16.0 + 54.0))
        np.testing.assert_allclose(potential, 50.0)
        np.testing.assert_allclose(energy, kinetic + potential)
        np.testing.assert_allclose(condition, 3.0)
        np.testing.assert_array_equal(symmetry_error, 0.0)
        np.testing.assert_allclose(minimum_eigenvalue, 2.0)
        np.testing.assert_array_equal(cholesky_success, True)
        np.testing.assert_allclose(external, 7.0)
        np.testing.assert_array_equal(feedforward, 0.0)
        np.testing.assert_array_equal(feedback, 0.0)

    def test_diagnose_asymmetric_and_indefinite_mass_matrices(self):
        """Reject symmetry and positive-definiteness independently of condition."""

        class InvalidForward:
            def mass_matrix(self, coordinates):
                """Return one asymmetric and one indefinite matrix."""
                del coordinates
                return np.asarray([[[1.0, 1.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, -1.0]]])

        model = SimpleNamespace(bodies=[SimpleNamespace(mass=1.0)], gravity=(0.0, -10.0, 0.0))
        fk = SimpleNamespace(center_of_mass_batch=lambda q: np.zeros((len(q), 3)), body_names=())
        result = diagnostics._energy_and_power(
            InvalidForward(),
            fk,
            model,
            np.array([0.0, 0.1]),
            np.zeros((2, 2)),
            np.zeros((2, 2)),
            np.zeros((2, 2)),
            np.zeros((2, 2)),
            [],
            np.zeros((2, 0, 9)),
            2,
            1.0e-10,
        )

        self.assertGreater(result[4][0], 0.0)
        self.assertLess(result[5][1], 0.0)
        np.testing.assert_array_equal(result[6], False)

    def test_run_window_archives_controller_components_and_metrics(self):
        """Archive stable-shape bounded control, work, energy, and condition arrays."""
        model, trajectory = self._fixture()
        root = diagnostics.structural_root_mask(model, trajectory.coordinate_names)
        config = diagnostics.Stage1Config(
            controller=diagnostics.ControllerConfig(
                rotational_effort_limit=25.0,
                translational_effort_limit=25.0,
            )
        )

        record = diagnostics.run_window(
            _FakeForward(3),
            _FakeKinematics(),
            model,
            trajectory,
            root,
            start_index=0,
            duration_s=0.04,
            requested_dt_s=0.01,
            variant="bounded_nonroot_tracking_linear",
            config=config,
        )

        self.assertEqual(record.status, "completed")
        self.assertEqual(record.total_control.shape, (5, 3))
        np.testing.assert_array_equal(record.feedforward[:, root], 0.0)
        np.testing.assert_array_equal(record.feedback[:, root], 0.0)
        np.testing.assert_array_equal(record.total_control[:, root], 0.0)
        self.assertTrue(record.metrics["root_feedforward_feedback_exact_zero"])
        self.assertEqual(record.metrics["mass_condition_number_max"], 1.0)
        self.assertIn("external_work_j", record.metrics)
        self.assertEqual(record.marker_error_m.shape, (5, 1))

    def test_reject_a_finite_but_truncated_solver_time_grid(self):
        """Do not call a finite trajectory complete when it ends before the requested window."""

        class TruncatedForward(_FakeForward):
            def simulate(self, *args, **kwargs):
                """Drop the final solver sample without introducing a nonfinite state."""
                rollout = super().simulate(*args, **kwargs)
                rollout.times = rollout.times[:-1]
                rollout.coordinates = rollout.coordinates[:-1]
                rollout.speeds = rollout.speeds[:-1]
                return rollout

        model, trajectory = self._fixture()
        root = diagnostics.structural_root_mask(model, trajectory.coordinate_names)
        record = diagnostics.run_window(
            TruncatedForward(3),
            _FakeKinematics(),
            model,
            trajectory,
            root,
            start_index=0,
            duration_s=0.04,
            requested_dt_s=0.01,
            variant="bounded_nonroot_tracking_linear",
            config=diagnostics.Stage1Config(),
        )

        self.assertEqual(record.status, "incomplete_time_grid")
        self.assertIs(record.metrics["completed"], False)
        self.assertIs(record.metrics["time_grid_complete"], False)

    def test_preserve_nonfinite_prefix_and_do_not_misattribute_event_coordinate(self):
        """Keep the finite prefix and leave non-coordinate failure attribution empty."""

        class NonfiniteForward(_FakeForward):
            def simulate(self, *args, **kwargs):
                """Append a nonfinite final coordinate to a normal result."""
                rollout = super().simulate(*args, **kwargs)
                rollout.coordinates[-1, 0] = np.nan
                return rollout

        model, trajectory = self._fixture()
        root = diagnostics.structural_root_mask(model, trajectory.coordinate_names)
        config = diagnostics.Stage1Config()
        record = diagnostics.run_window(
            NonfiniteForward(3),
            _FakeKinematics(),
            model,
            trajectory,
            root,
            start_index=0,
            duration_s=0.04,
            requested_dt_s=0.01,
            variant="bounded_nonroot_tracking_linear",
            config=config,
        )

        self.assertEqual(record.status, "nonfinite")
        self.assertEqual(record.metrics["finite_sample_count"], 4)
        self.assertEqual(record.marker_error_m.shape, (5, 1))
        restart = diagnostics._restart_metrics(record, trajectory, config)
        self.assertEqual(restart["event_type"], "nonfinite")
        self.assertIsNone(restart["event_coordinate"])
        self.assertIsNotNone(restart["largest_error_coordinate"])
        self.assertFalse(restart["metrics_acceptable"])

    def test_archive_numerical_solver_exception_but_propagate_schema_error(self):
        """Archive numerical solver failure and propagate programming or schema faults."""

        class FailingForward(_FakeForward):
            exception = RuntimeError("numerical failure")

            def simulate(self, *args, **kwargs):
                """Raise the configured exception."""
                del args, kwargs
                raise self.exception

        model, trajectory = self._fixture()
        root = diagnostics.structural_root_mask(model, trajectory.coordinate_names)
        config = diagnostics.Stage1Config()
        record = diagnostics.run_window(
            FailingForward(3),
            _FakeKinematics(),
            model,
            trajectory,
            root,
            start_index=0,
            duration_s=0.04,
            requested_dt_s=0.01,
            variant="full_id_all_linear",
            config=config,
        )
        self.assertEqual(record.status, "solver_exception")
        self.assertFalse(record.metrics["completed"])

        FailingForward.exception = ValueError("schema failure")
        with self.assertRaisesRegex(ValueError, "schema failure"):
            diagnostics.run_window(
                FailingForward(3),
                _FakeKinematics(),
                model,
                trajectory,
                root,
                start_index=0,
                duration_s=0.04,
                requested_dt_s=0.01,
                variant="full_id_all_linear",
                config=config,
            )

    def test_reject_overlapping_output_paths_before_execution(self):
        """Reject repository and source overlaps before any heavy simulation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source"
            source.mkdir()
            repository = Path(tmpdir) / "repo"
            repository.mkdir()
            with self.assertRaisesRegex(ValueError, "overlap"):
                diagnostics._validate_paths(source, source / "child", repository)
            with self.assertRaisesRegex(ValueError, "outside"):
                diagnostics._validate_paths(source, repository / "artifact", repository)

    def test_reject_npz_timestep_key_collisions(self):
        """Reject distinct timesteps that round to the same NPZ run key."""
        config = diagnostics.Stage1Config(
            timesteps_s=(0.0010000001, 0.0010000002),
            refinement_timestep_s=0.0005,
        )
        with self.assertRaisesRegex(ValueError, "collide"):
            config.validate()

    def test_write_strict_json_with_nonfinite_values_mapped_to_null(self):
        """Emit standard JSON null instead of NaN or Infinity tokens."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "strict.json"
            diagnostics._write_json(
                path,
                {"python_nan": float("nan"), "numpy_inf": np.float64(np.inf), "array": np.array([1.0, np.nan])},
            )
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("NaN", text)
            self.assertNotIn("Infinity", text)
            self.assertEqual(json.loads(text), {"array": [1.0, None], "numpy_inf": None, "python_nan": None})

    def test_publish_staged_artifact_and_recover_interrupted_backup(self):
        """Restore caught failures and recover a backup left by process interruption."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "stage1"
            output.mkdir()
            (output / "old.txt").write_text("accepted", encoding="utf-8")
            manifest = {"schema_version": diagnostics._SCHEMA, "scope": diagnostics._SCOPE}
            qc = {"scope": diagnostics._SCOPE}
            real_replace = os.replace
            call_count = 0

            def fail_publication(source, destination):
                nonlocal call_count
                call_count += 1
                if call_count == 2:
                    raise OSError("synthetic publication interruption")
                return real_replace(source, destination)

            with mock.patch.object(diagnostics.os, "replace", side_effect=fail_publication):
                with self.assertRaisesRegex(OSError, "interruption"):
                    diagnostics.publish_artifacts(
                        output,
                        manifest,
                        qc,
                        diagnostics.ControllerConfig(),
                        {},
                        [],
                        {},
                        3,
                    )
            self.assertEqual((output / "old.txt").read_text(encoding="utf-8"), "accepted")

            backup = output.parent / f".{output.name}.previous-crash"
            staging = output.parent / f".{output.name}.staging-crash"
            os.replace(output, backup)
            staging.mkdir()
            (staging / "partial.txt").write_text("not accepted", encoding="utf-8")
            self.assertEqual(diagnostics._recover_staged_publication(output), "restored_previous_output")
            self.assertEqual((output / "old.txt").read_text(encoding="utf-8"), "accepted")
            self.assertFalse(staging.exists())

            diagnostics.publish_artifacts(
                output,
                manifest,
                qc,
                diagnostics.ControllerConfig(),
                {},
                [],
                {},
                3,
            )
            self.assertFalse((output / "old.txt").exists())
            self.assertEqual(json.loads((output / "manifest.json").read_text())["scope"], diagnostics._SCOPE)
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {
                    "manifest.json",
                    "qc_summary.json",
                    "controller_config.json",
                    "convergence.npz",
                    "restart_map.npz",
                    "input_decomposition.npz",
                },
            )

    def test_gate_requires_convergence_and_actual_restart_decomposition_completion(self):
        """Fail on direct convergence, conditioning, restart, or decomposition defects."""
        times = np.array([0.0, 0.01])

        def record(requested_dt, *, completed=True, condition=1.0):
            values = np.zeros((2, 3))
            metrics = {
                "completed": completed,
                "metric_error": None,
                "finite_sample_count": 2,
                "external_work_j": 0.0,
                "feedforward_work_j": 0.0,
                "feedback_work_j": 0.0,
                "kinetic_energy_min_j": 0.0,
                "potential_energy_min_j": 0.0,
                "mass_condition_number_max": condition,
                "mass_symmetry_relative_error_max": 0.0,
                "mass_min_eigenvalue_min": 1.0,
                "mass_cholesky_all_success": True,
                "coordinate_rms": 0.0,
                "marker_rms_m": 0.0,
                "marker_max_m": 0.0,
                "root_feedforward_feedback_exact_zero": True,
                "range_violation_count": 0,
                "saturation_fraction_nonroot": 0.0,
            }
            return SimpleNamespace(
                requested_dt_s=requested_dt,
                actual_dt_s=requested_dt,
                metrics=metrics,
                times=times,
                coordinates=values,
                speeds=values,
                marker_error_m=np.zeros((2, 1)),
                total_energy_j=np.zeros(2),
            )

        config = diagnostics.Stage1Config()
        nominal = record(0.001)
        fine = record(0.0005)
        convergence = {"bounded_nonroot_tracking_linear": [nominal, fine]}
        restarts = [{"status": "completed", "metrics_acceptable": True}]
        decomposition = {name: record(0.001) for name in diagnostics._VARIANTS}
        sections = ("convergence", "restarts", "decomposition")

        gate = diagnostics._tracking_gate(convergence, restarts, decomposition, sections, config, times)
        self.assertTrue(gate["passed"])

        failed_restarts = [{"status": "solver_exception", "metrics_acceptable": False}]
        gate = diagnostics._tracking_gate(convergence, failed_restarts, decomposition, sections, config, times)
        self.assertFalse(gate["gates"]["all_scheduled_restarts_completed_with_acceptable_metrics"])

        failed_decomposition = dict(decomposition)
        failed_decomposition[next(iter(failed_decomposition))] = record(0.001, completed=False)
        gate = diagnostics._tracking_gate(convergence, restarts, failed_decomposition, sections, config, times)
        self.assertFalse(gate["gates"]["all_decomposition_variants_completed_with_acceptable_metrics"])

        ill_conditioned = {"bounded_nonroot_tracking_linear": [record(0.001, condition=1.0e13), fine]}
        gate = diagnostics._tracking_gate(ill_conditioned, restarts, decomposition, sections, config, times)
        self.assertFalse(gate["gates"]["mass_condition_number_within_limit"])
        self.assertEqual(diagnostics._SCOPE, "engineering_measured_load_tracking")
        self.assertIn("generalized_force_only_no_external_load_linear", diagnostics._VARIANTS)
        self.assertNotIn("feedforward_torque_only_linear", diagnostics._VARIANTS)

        parser = diagnostics.create_parser()
        args = parser.parse_args(["--section", "restarts", "--restart-start-limit", "2"])
        self.assertEqual(args.sections, ["restarts"])
        self.assertEqual(args.restart_start_limit, 2)


if __name__ == "__main__":
    unittest.main()
