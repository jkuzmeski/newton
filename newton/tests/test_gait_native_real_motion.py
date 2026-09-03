# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test real-C3D marker mapping and native motion artifact publication."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import newton
from newton.examples.opensim.example_native_motion_fit import (
    Example as NativeMotionExample,
)
from newton.examples.opensim.example_native_motion_fit import (
    _default_motion_output,
    _resolve_subject_marker_map,
    _strip_c3d_prefix,
    _warmup_frame_count,
    create_parser,
)
from projects.gait_c3d.c3d_adapter import C3DMarkerTrajectory
from projects.gait_c3d.marker_clusters import TRACKING_CLUSTER_C3D_SOURCES
from projects.gait_c3d.marker_map import C3DMarkerMap, apply_c3d_marker_map, save_c3d_marker_map
from projects.gait_c3d.native_motion_fit import (
    NativeC3DMarkers,
    fit_c3d_marker_motion,
    load_native_motion_artifact,
    map_c3d_markers_to_native,
    marker_attachments_from_model,
    marker_positions_from_joint_q,
    write_native_motion_artifact,
)

_SOURCE_FOR_NATIVE = {
    "Sternum": "STRN",
    "R.Acromium": "RSHO",
    "L.Acromium": "LSHO",
    "R.ASIS": "RASI",
    "L.ASIS": "LASI",
    "R.Thigh.Centroid": ("RTH2", "RTH3", "RTH4"),
    "R.Knee.Lat": "RKNE",
    "R.Knee.Med": "RMKNE",
    "R.Shank.Centroid": ("RTIB2", "RTIB3", "RTIB4"),
    "R.Ankle.Lat": "RANK",
    "R.Ankle.Med": "RMANK",
    "R.Heel": "RHEE",
    "R.Toe.Lat": "RMTH5",
    "R.Toe.Med": "RMTH1",
    "R.Toe.Tip": "RHLX",
    "L.Thigh.Centroid": ("LTH2", "LTH3", "LTH4"),
    "L.Knee.Lat": "LKNE",
    "L.Knee.Med": "LMKNE",
    "L.Shank.Centroid": ("LTIB2", "LTIB3", "LTIB4"),
    "L.Ankle.Lat": "LANK",
    "L.Ankle.Med": "LMANK",
    "L.Heel": "LHEE",
    "L.Toe.Lat": "LMTH5",
    "L.Toe.Med": "LMTH1",
    "L.Toe.Tip": "LHLX",
}


class TestNativeRealMotion(unittest.TestCase):
    """Test name-joined C3D targets and saved native motion output."""

    @classmethod
    def setUpClass(cls):
        """Build the calibrated free-root model."""
        cls.previous_target_layout = newton.use_coord_layout_targets
        newton.use_coord_layout_targets = True
        base = Path(__file__).parents[2] / "projects" / "gait_c3d" / "assets" / "s001_calibrated"
        builder = newton.ModelBuilder()
        builder.add_mjcf(str(base / "model" / "subject.xml"), floating=True, parse_sites=True)
        cls.model = builder.finalize(device="cpu")
        cls.attachments = marker_attachments_from_model(cls.model)

    @classmethod
    def tearDownClass(cls):
        """Restore the global coordinate-target option."""
        newton.use_coord_layout_targets = cls.previous_target_layout

    def test_maps_hallux_and_virtual_markers_by_name(self):
        """Map hallux toe labels and virtual sacrum/head markers without order assumptions."""
        names = ("LHLX", "RHLX", "LPSI", "RPSI", "LFHD", "RFHD", "LBHD", "RBHD")
        positions = np.asarray(
            [[[index + 0.1, index + 0.2, index + 0.3] for index in range(len(names))]], dtype=np.float32
        )
        trajectory = C3DMarkerTrajectory(
            times=np.asarray((0.0,)),
            positions=positions,
            valid=np.ones((1, len(names)), dtype=bool),
            marker_names=names,
            rate=100.0,
            first_frame=0,
            lab_to_newton=np.eye(3),
            source_file="trial.c3d",
            source_sha256="0" * 64,
        )
        attachments = tuple(
            type(self.attachments[0])(name, 0, (0.0, 0.0, 0.0))
            for name in ("L.Toe.Tip", "R.Toe.Tip", "V.Sacral", "Top.Head")
        )
        mapped = map_c3d_markers_to_native(trajectory, attachments)
        self.assertEqual(mapped.marker_names, tuple(attachment.name for attachment in attachments))
        np.testing.assert_array_equal(mapped.valid, np.ones((1, 4), dtype=bool))
        np.testing.assert_allclose(mapped.positions[0, 0], positions[0, 0])
        np.testing.assert_allclose(mapped.positions[0, 2], 0.5 * (positions[0, 2] + positions[0, 3]))
        np.testing.assert_allclose(mapped.positions[0, 3], positions[0, 4:].mean(axis=0))

    def test_centroids_use_remaining_sources_when_one_label_is_absent(self):
        """Average available markers for every tracking cluster."""
        for cluster_name, sources in TRACKING_CLUSTER_C3D_SOURCES.items():
            with self.subTest(cluster_name=cluster_name):
                positions = np.asarray(
                    [
                        [[0.0, 0.0, 0.0], [0.03, 0.06, 0.09]],
                        [[0.1, 0.2, 0.3], [0.04, 0.08, 0.12]],
                    ],
                    dtype=np.float32,
                )
                trajectory = C3DMarkerTrajectory(
                    times=np.asarray((0.0, 0.01)),
                    positions=positions,
                    valid=np.ones((2, 2), dtype=bool),
                    marker_names=sources[1:],
                    rate=100.0,
                    first_frame=0,
                    lab_to_newton=np.eye(3),
                    source_file="trial.c3d",
                    source_sha256="4" * 64,
                )
                attachment = type(self.attachments[0])(cluster_name, 0, (0.0, 0.0, 0.0))

                mapped = map_c3d_markers_to_native(trajectory, (attachment,))

                self.assertTrue(np.all(mapped.valid))
                np.testing.assert_allclose(mapped.positions[:, 0], positions.mean(axis=1))

    def test_centroids_use_framewise_valid_sources(self):
        """Average only valid cluster markers on each frame."""
        positions = np.asarray(
            [
                [[0.0, 0.0, 0.0], [0.03, 0.06, 0.09]],
                [[0.1, 0.2, 0.3], [0.04, 0.08, 0.12]],
            ],
            dtype=np.float32,
        )
        trajectory = C3DMarkerTrajectory(
            times=np.asarray((0.0, 0.01)),
            positions=positions,
            valid=np.asarray([[True, False], [False, False]]),
            marker_names=("LTH3", "LTH4"),
            rate=100.0,
            first_frame=0,
            lab_to_newton=np.eye(3),
            source_file="trial.c3d",
            source_sha256="5" * 64,
        )
        attachment = type(self.attachments[0])("L.Thigh.Centroid", 0, (0.0, 0.0, 0.0))

        mapped = map_c3d_markers_to_native(trajectory, (attachment,))

        np.testing.assert_array_equal(mapped.valid[:, 0], np.asarray([True, False]))
        np.testing.assert_allclose(mapped.positions[0, 0], positions[0, 0])
        np.testing.assert_array_equal(mapped.positions[1, 0], np.zeros(3, dtype=np.float32))

    def test_maps_thigh_and_shank_clusters_to_centroids(self):
        """Average each complete thigh and shank source cluster into one target."""
        names = ("RTH2", "RTH3", "RTH4", "RTIB2", "RTIB3", "RTIB4")
        positions = np.asarray(
            [
                [
                    [0.0, 0.0, 0.0],
                    [0.03, 0.06, 0.09],
                    [0.06, 0.12, 0.18],
                    [0.1, 0.0, 0.0],
                    [0.1, 0.03, 0.06],
                    [0.1, 0.06, 0.12],
                ]
            ],
            dtype=np.float32,
        )
        trajectory = C3DMarkerTrajectory(
            times=np.asarray((0.0,)),
            positions=positions,
            valid=np.ones((1, len(names)), dtype=bool),
            marker_names=names,
            rate=100.0,
            first_frame=0,
            lab_to_newton=np.eye(3),
            source_file="trial.c3d",
            source_sha256="3" * 64,
        )
        attachments = tuple(
            type(self.attachments[0])(name, 0, (0.0, 0.0, 0.0)) for name in ("R.Thigh.Centroid", "R.Shank.Centroid")
        )
        mapped = map_c3d_markers_to_native(trajectory, attachments)
        np.testing.assert_allclose(mapped.positions[0], positions[0].reshape(2, 3, 3).mean(axis=1))
        self.assertTrue(np.all(mapped.valid))

    def test_warmup_frame_count_primes_the_selected_batch_path(self):
        """Choose a small warmup that matches the selected CUDA batch path."""
        common = {
            "frame_count": 100,
            "start_frame": 0,
            "end_frame": None,
            "stride": 1,
            "max_frames": 0,
        }
        self.assertEqual(_warmup_frame_count(**common, batch_size=0), 2)
        self.assertEqual(_warmup_frame_count(**common, batch_size=8), 9)
        self.assertEqual(_warmup_frame_count(**common, batch_size=100), 2)
        self.assertEqual(_warmup_frame_count(**{**common, "max_frames": 4}, batch_size=8), 2)

    def test_parser_exposes_motion_load_and_overwrite_flags(self):
        """Parse motion replay, overwrite, and solve-batch options."""
        args = create_parser().parse_args(["--motion", "/tmp/motion", "--overwrite", "--batch-size", "64"])
        self.assertEqual(args.motion, "/tmp/motion")
        self.assertTrue(args.overwrite)
        self.assertEqual(args.batch_size, 64)

    def test_parser_exposes_marker_mapping_and_overwrite_flags(self):
        """Parse real C3D marker mapping and full-solve overwrite options."""
        args = create_parser().parse_args(
            ["--c3d", "/tmp/trial.c3d", "--marker-map", "/tmp/labels.json", "--keep-c3d-prefix", "--overwrite"]
        )
        self.assertEqual(args.c3d, "/tmp/trial.c3d")
        self.assertEqual(args.marker_map, "/tmp/labels.json")
        self.assertTrue(args.keep_c3d_prefix)
        self.assertTrue(args.overwrite)

    def test_rejects_marker_map_without_c3d_input(self):
        """Reject mapping controls in synthetic and saved-motion modes."""
        for values in (
            ("--synthetic", "--marker-map", "/tmp/map.json"),
            ("--motion", "/tmp/motion", "--keep-c3d-prefix"),
        ):
            with self.subTest(values=values):
                with self.assertRaisesRegex(ValueError, "require --c3d"):
                    NativeMotionExample(None, create_parser().parse_args(values))

    def test_reuses_bundled_prefix_policy_unless_cli_overrides_it(self):
        """Decode dynamic labels with the preprocessing stored beside the map."""
        metadata = {"strip_prefix": False}
        self.assertFalse(_strip_c3d_prefix(None, False, metadata))
        self.assertFalse(_strip_c3d_prefix(None, True, {"strip_prefix": True}))
        self.assertTrue(_strip_c3d_prefix("/tmp/explicit.json", False, metadata))

    def test_default_motion_output_stays_inside_subject_bundle(self):
        """Derive a stable subject-local output path from the trial filename."""
        subject = Path("/subjects/S001_calibrated")
        output = _default_motion_output(subject, "/incoming/Trial 101.v3d.c3d")
        self.assertEqual(output, subject / "motions" / "trial_101_native_motion")

    def test_loads_subject_marker_map_and_allows_explicit_override(self):
        """Resolve and verify a bundled marker map unless an explicit map overrides it."""
        with tempfile.TemporaryDirectory() as directory:
            subject = Path(directory) / "subject"
            subject.mkdir()
            bundled = save_c3d_marker_map(C3DMarkerMap({"LASI": "BUNDLED_LASI"}), subject / "labels.json")
            explicit = save_c3d_marker_map(C3DMarkerMap({"LASI": "EXPLICIT_LASI"}), Path(directory) / "explicit.json")
            (subject / "subject.json").write_text(
                json.dumps(
                    {
                        "artifacts": {"marker_map": bundled.name},
                        "marker_mapping": {
                            "file": bundled.name,
                            "sha256": hashlib.sha256(bundled.read_bytes()).hexdigest(),
                            "strip_prefix": True,
                        },
                    }
                ),
                encoding="utf-8",
            )
            marker_map, path, metadata = _resolve_subject_marker_map(subject, None)
            override, override_path, override_metadata = _resolve_subject_marker_map(subject, str(explicit))
            bundled.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                _resolve_subject_marker_map(subject, None)
        self.assertEqual(marker_map.source_for("LASI"), "BUNDLED_LASI")
        self.assertEqual(path, bundled)
        self.assertTrue(metadata["strip_prefix"])
        self.assertEqual(override.source_for("LASI"), "EXPLICIT_LASI")
        self.assertEqual(override_path, explicit)
        self.assertIsNone(override_metadata)

    def test_fits_named_c3d_targets_and_publishes_artifact(self):
        """Fit finite name-joined targets and write a sealed motion artifact."""
        q0 = self.model.joint_q.numpy().copy()
        q1 = q0.copy()
        q1[0] += 0.01
        q1[5] += 0.01
        q1[3:7] /= np.linalg.norm(q1[3:7])
        q1[7] = 0.03
        q1[13] = 0.12
        q1[18] = 0.14
        target_q = np.asarray((q0, q1), dtype=np.float32)
        target_native = np.asarray(
            [marker_positions_from_joint_q(self.model, self.attachments, value) for value in target_q]
        )
        source_names = list(
            dict.fromkeys(
                source
                for value in _SOURCE_FOR_NATIVE.values()
                for source in (value if isinstance(value, tuple) else (value,))
            )
        )
        source_names.extend(
            name for name in ("LPSI", "RPSI", "LFHD", "RFHD", "LBHD", "RBHD") if name not in source_names
        )
        source_index = {name: index for index, name in enumerate(source_names)}
        source_positions = np.zeros((2, len(source_names), 3), dtype=np.float32)
        for marker_index, attachment in enumerate(self.attachments):
            if attachment.name in _SOURCE_FOR_NATIVE:
                source = _SOURCE_FOR_NATIVE[attachment.name]
                sources = source if isinstance(source, tuple) else (source,)
            elif attachment.name == "V.Sacral":
                sources = ("LPSI", "RPSI")
            elif attachment.name == "Top.Head":
                sources = ("LFHD", "RFHD", "LBHD", "RBHD")
            else:
                raise AssertionError(attachment.name)
            for source in sources:
                source_positions[:, source_index[source]] = target_native[:, marker_index]
        trajectory = C3DMarkerTrajectory(
            times=np.asarray((0.0, 0.01)),
            positions=source_positions,
            valid=np.ones((2, len(source_names)), dtype=bool),
            marker_names=tuple(source_names),
            rate=100.0,
            first_frame=0,
            lab_to_newton=np.eye(3),
            source_file="trial.c3d",
            source_sha256="1" * 64,
        )
        mapped = map_c3d_markers_to_native(trajectory, self.attachments)
        aliases = {name: f"LAB_{name}" for name in trajectory.marker_names}
        custom = C3DMarkerTrajectory(
            times=trajectory.times.copy(),
            positions=trajectory.positions.copy(),
            valid=trajectory.valid.copy(),
            marker_names=tuple(aliases[name] for name in trajectory.marker_names),
            rate=trajectory.rate,
            first_frame=trajectory.first_frame,
            lab_to_newton=trajectory.lab_to_newton.copy(),
            source_file=trajectory.source_file,
            source_sha256=trajectory.source_sha256,
        )
        canonicalized = apply_c3d_marker_map(custom, C3DMarkerMap(aliases), required=trajectory.marker_names)
        custom_mapped = map_c3d_markers_to_native(canonicalized, self.attachments)
        np.testing.assert_array_equal(custom_mapped.positions, mapped.positions)
        np.testing.assert_array_equal(custom_mapped.valid, mapped.valid)
        motion = fit_c3d_marker_motion(self.model, self.attachments, mapped, q0, iterations=40)
        self.assertEqual(motion.joint_q.shape, (2, self.model.joint_coord_count))
        self.assertTrue(np.all(np.isfinite(motion.joint_qd)))
        quaternion = motion.joint_q[:, 3:7]
        vector, scalar = quaternion[:, :3], quaternion[:, 3, None]
        com = self.model.body_com.numpy()[int(self.model.joint_child.numpy()[0])]
        rotated_com = com + 2.0 * np.cross(vector, scalar * com + np.cross(vector, com))
        np.testing.assert_allclose(
            motion.joint_qd[:, :3],
            np.repeat(np.diff(motion.joint_q[:, :3] + rotated_com, axis=0) / 0.01, 2, axis=0),
            atol=1.0e-5,
        )
        qdot = np.diff(quaternion, axis=0)[0] / 0.01
        expected_angular = 2.0 * (scalar * qdot[:3] - qdot[3] * vector + np.cross(vector, qdot[:3]))
        np.testing.assert_allclose(motion.joint_qd[:, 3:6], expected_angular, atol=1.0e-5)
        self.assertLess(float(np.max(motion.frame_rms)), 1.0e-3)
        with tempfile.TemporaryDirectory() as directory:
            output = write_native_motion_artifact(
                motion,
                Path(directory) / "motion",
                model_path=Path("projects/gait_c3d/assets/s001_calibrated/model/subject.xml"),
                marker_mapping={
                    "schema_version": "gait_c3d_marker_map_1",
                    "file_sha256": "a" * 64,
                    "canonical_to_source": {"LASI": "LAB_LASI"},
                },
            )
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertTrue((output / "motion.npz").is_file())
            self.assertEqual(manifest["schema_version"], "gait_native_motion_artifact_1")
            self.assertEqual(manifest["frames"]["count"], 2)
            self.assertEqual(manifest["markers"]["valid_count"], 2 * len(self.attachments))
            self.assertEqual(
                manifest["marker_mapping"]["centroids"]["R.Thigh.Centroid"],
                ["RTH2", "RTH3", "RTH4"],
            )
            self.assertEqual(
                manifest["marker_mapping"]["acquisition"]["canonical_to_source"],
                {"LASI": "LAB_LASI"},
            )
            loaded = load_native_motion_artifact(output / "motion.npz")
            self.assertEqual(loaded.marker_names, motion.marker_names)
            np.testing.assert_array_equal(loaded.joint_q, motion.joint_q)
            write_native_motion_artifact(motion, output, overwrite=True)
            self.assertEqual(load_native_motion_artifact(output).times.shape, (2,))
            unsafe = Path(directory) / "subject"
            unsafe.mkdir()
            (unsafe / "subject.xml").write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exact native motion artifact"):
                write_native_motion_artifact(motion, unsafe, overwrite=True)

    def test_fit_preserves_predictions_for_occluded_markers(self):
        """Keep full marker predictions while fitting only visible markers."""
        q0 = self.model.joint_q.numpy().copy()
        target = np.asarray([marker_positions_from_joint_q(self.model, self.attachments, q0)] * 3)
        valid = np.ones((3, len(self.attachments)), dtype=bool)
        valid[:, ::4] = False
        target[~valid] = 0.0
        mapped = NativeC3DMarkers(
            np.asarray((0.0, 0.01, 0.02)),
            target,
            valid,
            tuple(attachment.name for attachment in self.attachments),
            "trial.c3d",
            "2" * 64,
        )
        motion = fit_c3d_marker_motion(self.model, self.attachments, mapped, q0, iterations=10, batch_size=2)
        self.assertEqual(motion.predictions.shape, target.shape)
        self.assertTrue(np.all(np.isfinite(motion.predictions)))
        self.assertGreater(float(np.linalg.norm(motion.predictions[:, ~valid[0]])), 0.0)


if __name__ == "__main__":
    unittest.main()
