# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test treadmill belt logs and the treadmill-to-overground virtual origin."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

import newton
from projects.gait_c3d.native_motion_fit import (
    NativeC3DMarkers,
    fit_c3d_marker_motion,
    load_native_motion_artifact,
    marker_attachments_from_model,
    marker_positions_from_joint_q,
    write_native_motion_artifact,
)
from projects.gait_c3d.treadmill import (
    COLUMNS,
    BeltMotion,
    belt_motion,
    belt_motion_for_frames,
    find_treadmill_log,
    load_treadmill_log,
)


def _write_log(directory: Path, speed: np.ndarray, rate: float = 300.0, right: np.ndarray | None = None) -> Path:
    """Write a synthetic D-Flow log whose distance column is a right-rectangle sum."""
    right = speed if right is None else right
    time = 2532.802615 + np.arange(len(speed)) / rate
    step = 1.0 / rate

    def accumulate(values: np.ndarray) -> np.ndarray:
        return np.concatenate([[0.0], np.cumsum(values[1:] * step)])

    rows = np.column_stack(
        [time, speed, accumulate(speed), right, accumulate(right), np.zeros(len(speed)), np.zeros(len(speed))]
    )
    path = directory / "tm0001.txt"
    with path.open("w") as stream:
        stream.write("\t".join(COLUMNS) + "\n")
        for row in rows:
            stream.write("\t".join(f"{value:.6f}" for value in row) + "\n")
    return path


def _ramp_hold(rate: float = 300.0) -> np.ndarray:
    """Return a 1 s ramp to 1.5 m/s followed by a 1 s hold."""
    ramp = np.linspace(0.0, 1.5, int(rate), endpoint=False)
    return np.concatenate([ramp, np.full(int(rate) + 1, 1.5)])


class TestGaitTreadmillLog(unittest.TestCase):
    """Test parsing and integration of a D-Flow treadmill log."""

    def test_parses_channels_and_seals_source(self):
        """Parse every logged channel and record the source name and hash."""
        with tempfile.TemporaryDirectory() as directory:
            path = _write_log(Path(directory), np.full(600, 1.5))
            log = load_treadmill_log(path)
        self.assertEqual(log.source_file, "tm0001.txt")
        self.assertEqual(len(log.source_sha256), 64)
        self.assertEqual(len(log.time), 600)
        # The logged clock keeps six decimals, so the median interval rounds to 3333 us.
        self.assertAlmostEqual(log.rate, 300.0, delta=0.05)
        self.assertAlmostEqual(log.duration, 599.0 / 300.0, places=5)
        self.assertAlmostEqual(log.tied_belt_residual, 0.0)
        np.testing.assert_allclose(log.speed("mean"), 1.5)
        self.assertAlmostEqual(float(log.t[0]), 0.0)

    def test_rejects_an_unexpected_header(self):
        """Reject a log whose column layout is not the D-Flow layout."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tm0001.txt"
            path.write_text("Time\tspeed\n0.0\t1.0\n")
            with self.assertRaisesRegex(ValueError, "unexpected treadmill log header"):
                load_treadmill_log(path)

    def test_rejects_an_unknown_belt_side(self):
        """Reject a belt side that is not left, right, or mean."""
        with tempfile.TemporaryDirectory() as directory:
            log = load_treadmill_log(_write_log(Path(directory), np.full(600, 1.5)))
        with self.assertRaisesRegex(ValueError, "unknown belt side"):
            log.speed("both")

    def test_drops_every_timestamp_that_breaks_monotonic_order(self):
        """Drop stale rows until the controller clock exceeds its prior maximum."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tm0001.txt"
            rows = np.zeros((4, len(COLUMNS)), dtype=np.float64)
            rows[:, 0] = (0.0, 2.0, 1.5, 1.75)
            with path.open("w") as stream:
                stream.write("\t".join(COLUMNS) + "\n")
                for row in rows:
                    stream.write("\t".join(str(value) for value in row) + "\n")
            log = load_treadmill_log(path)
        np.testing.assert_array_equal(log.time, (0.0, 2.0))

    def test_ignores_constant_distance_counter_offsets_between_belts(self):
        """Compare left and right travel after removing each counter's origin."""
        with tempfile.TemporaryDirectory() as directory:
            log = load_treadmill_log(_write_log(Path(directory), np.full(600, 1.5)))
        shifted = replace(log, right_distance=log.right_distance + 4.0)
        self.assertAlmostEqual(shifted.tied_belt_residual, 0.0)

    def test_integrates_piecewise_linear_speed_exactly(self):
        """Recover belt travel of a ramp and a hold to micrometre accuracy."""
        with tempfile.TemporaryDirectory() as directory:
            log = load_treadmill_log(_write_log(Path(directory), _ramp_hold()))
        query = np.asarray([0.0, 0.5, 1.0, 1.5, 2.0])
        expected = np.asarray([0.0, 0.1875, 0.75, 1.5, 2.25])
        np.testing.assert_allclose(log.travel(query, "left"), expected, atol=1.0e-6)

    def test_clamps_queries_outside_the_logged_interval(self):
        """Hold belt travel constant before the first and after the last sample."""
        with tempfile.TemporaryDirectory() as directory:
            log = load_treadmill_log(_write_log(Path(directory), np.full(600, 1.5)))
        travel = log.travel(np.asarray([-10.0, 0.0, 100.0]), "left")
        self.assertAlmostEqual(travel[0], 0.0)
        self.assertAlmostEqual(travel[2], log.travel(np.asarray([log.duration]), "left")[0])


class TestGaitTreadmillBeltMotion(unittest.TestCase):
    """Test the overground virtual origin built from a treadmill log."""

    @staticmethod
    def _log(directory: str, right: np.ndarray | None = None) -> object:
        return load_treadmill_log(_write_log(Path(directory), _ramp_hold(), right=right))

    def test_maps_belt_travel_to_the_newton_forward_axis(self):
        """Point the overground offset along Newton +X for a +Z up, -Y forward lab."""
        with tempfile.TemporaryDirectory() as directory:
            motion = belt_motion_for_frames(self._log(directory), 100, rate=100.0)
        np.testing.assert_allclose(motion.axis, (1.0, 0.0, 0.0), atol=1.0e-12)
        np.testing.assert_allclose(motion.offsets()[:, 1:], 0.0, atol=1.0e-12)
        np.testing.assert_allclose(motion.offsets()[:, 0], motion.distance, atol=0.0)
        np.testing.assert_allclose(motion.velocities()[:, 0], motion.speed, atol=0.0)

    def test_resamples_a_frame_window_consistently(self):
        """Match a fitted sub-window against the same frames of the full timeline."""
        with tempfile.TemporaryDirectory() as directory:
            log = self._log(directory)
            full = belt_motion_for_frames(log, 200, rate=100.0)
            window = belt_motion_for_frames(log, 50, rate=100.0, first_frame=120)
        np.testing.assert_allclose(window.distance, full.distance[120:170] - full.distance[120], atol=1.0e-12)
        np.testing.assert_allclose(window.speed, full.speed[120:170], atol=1.0e-12)
        self.assertAlmostEqual(window.times[0], 1.2)

    def test_honors_frame_stride_and_rate(self):
        """Sample belt travel on strided capture frame times, not on log samples."""
        with tempfile.TemporaryDirectory() as directory:
            motion = belt_motion_for_frames(self._log(directory), 10, rate=100.0, stride=5)
        np.testing.assert_allclose(motion.times, np.arange(10) * 0.05, atol=1.0e-12)

    def test_shifts_the_log_by_a_sync_offset(self):
        """Move the sampled belt profile when a log-to-capture offset is given."""
        with tempfile.TemporaryDirectory() as directory:
            log = self._log(directory)
            zero = belt_motion_for_frames(log, 100, rate=100.0)
            late = belt_motion_for_frames(log, 100, rate=100.0, offset=0.5)
        self.assertGreater(late.travel, zero.travel)
        self.assertAlmostEqual(late.offset, 0.5)

    def test_rejects_a_split_belt_trial(self):
        """Refuse one virtual origin when the two belts do not travel together."""
        with tempfile.TemporaryDirectory() as directory:
            log = self._log(directory, right=_ramp_hold() * 0.5)
            with self.assertRaisesRegex(ValueError, "split-belt"):
                belt_motion_for_frames(log, 100, rate=100.0)
            motion = belt_motion_for_frames(log, 100, rate=100.0, side="right")
        self.assertEqual(motion.side, "right")
        self.assertGreater(motion.tied_belt_residual, 0.1)

    def test_marks_frames_outside_the_logged_interval(self):
        """Flag capture frames outside the log and stop their virtual-origin velocity."""
        with tempfile.TemporaryDirectory() as directory:
            motion = belt_motion_for_frames(self._log(directory), 400, rate=100.0)
        self.assertTrue(bool(motion.covered[0]))
        self.assertFalse(bool(motion.covered[-1]))
        self.assertAlmostEqual(float(motion.distance[-1]), float(motion.distance[200]), places=6)
        self.assertEqual(float(motion.speed[-1]), 0.0)

    def test_rejects_nonmonotonic_frame_times(self):
        """Reject a capture timeline that cannot define ordered motion."""
        with tempfile.TemporaryDirectory() as directory:
            log = self._log(directory)
        with self.assertRaisesRegex(ValueError, "increase strictly"):
            belt_motion(log, np.asarray((0.0, 0.02, 0.01)))

    def test_publishes_a_sealable_manifest_block(self):
        """Describe the applied transform with JSON-safe manifest values."""
        with tempfile.TemporaryDirectory() as directory:
            motion = belt_motion_for_frames(self._log(directory), 100, rate=100.0)
        block = motion.manifest_block()
        self.assertEqual(block["side"], "left")
        self.assertEqual(block["applied_stage"], "post_ik_root_translation")
        self.assertEqual(block["axis"], [1.0, 0.0, 0.0])
        self.assertEqual(block["covered_frames"], 100)
        self.assertAlmostEqual(block["distance_m"], motion.travel)
        self.assertEqual(block["source"]["file"], "tm0001.txt")
        self.assertAlmostEqual(block["source_rate_hz"], motion.source_rate)

    def test_rejects_an_invalid_belt_axis(self):
        """Reject a belt motion whose offset direction is not a unit vector."""
        with self.assertRaisesRegex(ValueError, "unit vector"):
            BeltMotion(
                times=np.zeros(2),
                speed=np.zeros(2),
                distance=np.zeros(2),
                covered=np.ones(2, dtype=bool),
                axis=np.asarray((2.0, 0.0, 0.0)),
                side="left",
                offset=0.0,
                source_file="tm0001.txt",
                source_sha256="0" * 64,
                source_rate=300.0,
                tied_belt_residual=0.0,
            )

    def test_rejects_an_empty_frame_range(self):
        """Reject a resample request that selects no capture frames."""
        with tempfile.TemporaryDirectory() as directory:
            log = self._log(directory)
        with self.assertRaisesRegex(ValueError, "must be positive"):
            belt_motion_for_frames(log, 0, rate=100.0)
        with self.assertRaisesRegex(ValueError, "nonempty"):
            belt_motion(log, np.zeros(0))

    def test_finds_a_subject_bundle_log(self):
        """Return the treadmill log of a subject bundle only when it exists."""
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(find_treadmill_log(directory))
            _write_log(Path(directory), np.full(600, 1.5))
            self.assertIsNotNone(find_treadmill_log(directory))


class TestGaitTreadmillOvergroundFit(unittest.TestCase):
    """Test the treadmill-to-overground transform applied to a fitted motion."""

    @classmethod
    def setUpClass(cls):
        """Build the free-root calibrated S001 test model and its markers."""
        cls.previous_target_layout = newton.use_coord_layout_targets
        newton.use_coord_layout_targets = True
        base = Path(__file__).parents[2] / "projects" / "gait_c3d" / "assets" / "s001_calibrated"
        builder = newton.ModelBuilder()
        builder.add_mjcf(str(base / "model" / "subject.xml"), floating=True, parse_sites=True)
        cls.model = builder.finalize(device="cpu")
        cls.attachments = marker_attachments_from_model(cls.model)
        cls.seed = cls.model.joint_q.numpy().copy()
        cls.markers = cls._markers(8)

    @classmethod
    def tearDownClass(cls):
        """Restore the global coordinate-target option."""
        newton.use_coord_layout_targets = cls.previous_target_layout

    @classmethod
    def _markers(cls, frame_count: int) -> NativeC3DMarkers:
        """Create a walking-like synthetic marker trial in native site order."""
        positions = []
        for frame in range(frame_count):
            target = cls.seed.copy()
            phase = 0.4 * frame
            target[:3] += np.asarray((0.02 * np.sin(phase), 0.01 * np.cos(phase), 0.0), dtype=np.float32)
            target[7:] += np.asarray(
                [0.05 * np.sin(phase + index * 0.2) for index in range(cls.model.joint_coord_count - 7)],
                dtype=np.float32,
            )
            positions.append(marker_positions_from_joint_q(cls.model, cls.attachments, target))
        return NativeC3DMarkers(
            times=np.arange(frame_count, dtype=np.float64) / 100.0,
            positions=np.stack(positions).astype(np.float32),
            valid=np.ones((frame_count, len(cls.attachments)), dtype=bool),
            marker_names=tuple(attachment.name for attachment in cls.attachments),
            source_file="trial.c3d",
            source_sha256="0" * 64,
        )

    def _fit(self, belt: BeltMotion | None, registration: np.ndarray | None = None):
        """Fit the synthetic trial with an optional treadmill transform and registration."""
        return fit_c3d_marker_motion(
            self.model,
            self.attachments,
            self.markers,
            self.seed,
            belt=belt,
            registration=registration,
            iterations=10,
            batch_size=1,
        )

    def _belt(self, directory: str) -> BeltMotion:
        """Create a constant 1.5 m/s belt covering the synthetic trial."""
        log = load_treadmill_log(_write_log(Path(directory), np.full(600, 1.5)))
        return belt_motion_for_frames(log, len(self.markers.times), rate=100.0)

    def test_moves_only_the_free_root_into_the_overground_frame(self):
        """Translate the root by the belt travel and leave every joint angle unchanged."""
        with tempfile.TemporaryDirectory() as directory:
            belt = self._belt(directory)
        lab = self._fit(None)
        overground = self._fit(belt)
        self.assertEqual(overground.joint_q.dtype, np.float64)
        np.testing.assert_array_equal(overground.joint_q[:, 3:], lab.joint_q[:, 3:])
        np.testing.assert_allclose(
            overground.joint_q[:, :3].astype(np.float64) - lab.joint_q[:, :3].astype(np.float64),
            belt.offsets(),
            atol=1.0e-6,
        )

    def test_rotates_belt_motion_with_marker_registration(self):
        """Apply belt travel in the same registered frame as the C3D markers."""
        with tempfile.TemporaryDirectory() as directory:
            belt = self._belt(directory)
        registration = np.eye(4)
        registration[:2, :2] = ((0.0, -1.0), (1.0 + 1.0e-8, 0.0))
        lab = self._fit(None, registration)
        overground = self._fit(belt, registration)
        registered_axis = registration[:3, :3] @ belt.axis
        registered_axis /= np.linalg.norm(registered_axis)
        expected = belt.distance[:, None] * registered_axis
        np.testing.assert_allclose(
            overground.joint_q[:, :3] - lab.joint_q[:, :3],
            expected,
            atol=1.0e-6,
        )
        np.testing.assert_allclose(overground.treadmill["axis"], registered_axis, atol=1.0e-12)

    def test_keeps_marker_diagnostics_bit_identical(self):
        """Leave every published marker residual unchanged by the transform."""
        with tempfile.TemporaryDirectory() as directory:
            belt = self._belt(directory)
        lab = self._fit(None)
        overground = self._fit(belt)
        np.testing.assert_array_equal(overground.frame_rms, lab.frame_rms)
        np.testing.assert_array_equal(overground.frame_max, lab.frame_max)
        np.testing.assert_array_equal(overground.marker_rms, lab.marker_rms)
        np.testing.assert_array_equal(overground.body_rms, lab.body_rms)
        np.testing.assert_allclose(
            overground.targets.astype(np.float64) - lab.targets.astype(np.float64),
            np.broadcast_to(belt.offsets()[:, None, :], lab.targets.shape),
            atol=1.0e-6,
        )

    def test_adds_the_belt_speed_to_the_root_velocity(self):
        """Recover the belt speed in the fitted root linear velocity."""
        with tempfile.TemporaryDirectory() as directory:
            belt = self._belt(directory)
        lab = self._fit(None)
        overground = self._fit(belt)
        difference = overground.joint_qd[:, :3].astype(np.float64) - lab.joint_qd[:, :3].astype(np.float64)
        np.testing.assert_allclose(difference, belt.velocities(), atol=1.0e-4)

    def test_publishes_and_reloads_the_treadmill_block(self):
        """Seal the applied transform in the artifact manifest and read it back."""
        with tempfile.TemporaryDirectory() as directory:
            belt = self._belt(directory)
            overground = self._fit(belt)
            artifact = write_native_motion_artifact(overground, Path(directory) / "motion")
            manifest = json.loads((artifact / "manifest.json").read_text())
            loaded = load_native_motion_artifact(artifact)
        self.assertEqual(manifest["schema_version"], "gait_native_motion_artifact_3")
        self.assertEqual(loaded.joint_q.dtype, np.float64)
        self.assertEqual(loaded.treadmill, overground.treadmill)
        self.assertEqual(loaded.treadmill["applied_stage"], "post_ik_root_translation")
        self.assertAlmostEqual(loaded.treadmill["distance_m"], belt.travel)
        self.assertIsNone(self._fit(None).treadmill)

    def test_rejects_a_belt_that_misses_source_frames(self):
        """Refuse a belt motion that does not cover the selected C3D frames."""
        with tempfile.TemporaryDirectory() as directory:
            log = load_treadmill_log(_write_log(Path(directory), np.full(600, 1.5)))
            short = belt_motion_for_frames(log, len(self.markers.times) - 1, rate=100.0)
            full = belt_motion_for_frames(log, len(self.markers.times), rate=100.0)
        with self.assertRaisesRegex(ValueError, "every source C3D frame"):
            self._fit(short)
        uncovered = full.covered.copy()
        uncovered[-1] = False
        with self.assertRaisesRegex(ValueError, "selected C3D frames"):
            self._fit(replace(full, covered=uncovered))

    def test_rejects_a_belt_on_a_different_timeline(self):
        """Refuse same-length belt samples that do not align with C3D frame times."""
        with tempfile.TemporaryDirectory() as directory:
            belt = self._belt(directory)
        shifted = replace(belt, times=belt.times + 0.001)
        with self.assertRaisesRegex(ValueError, "frame times"):
            self._fit(shifted)


if __name__ == "__main__":
    unittest.main()
