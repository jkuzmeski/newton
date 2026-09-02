# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test real-C3D marker mapping and native motion artifact publication."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import newton
from projects.gait_c3d.c3d_adapter import C3DMarkerTrajectory
from projects.gait_c3d.native_motion_fit import (
    NativeC3DMarkers,
    fit_c3d_marker_motion,
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
    "R.Thigh.Upper": "RTH2",
    "R.Thigh.Front": "RTH3",
    "R.Thigh.Rear": "RTH4",
    "R.Knee.Lat": "RKNE",
    "R.Knee.Med": "RMKNE",
    "R.Shank.Upper": "RTIB2",
    "R.Shank.Front": "RTIB3",
    "R.Shank.Rear": "RTIB4",
    "R.Ankle.Lat": "RANK",
    "R.Ankle.Med": "RMANK",
    "R.Heel": "RHEE",
    "R.Toe.Lat": "RMTH5",
    "R.Toe.Med": "RMTH1",
    "R.Toe.Tip": "RHLX",
    "L.Thigh.Upper": "LTH2",
    "L.Thigh.Front": "LTH3",
    "L.Thigh.Rear": "LTH4",
    "L.Knee.Lat": "LKNE",
    "L.Knee.Med": "LMKNE",
    "L.Shank.Upper": "LTIB2",
    "L.Shank.Front": "LTIB3",
    "L.Shank.Rear": "LTIB4",
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

    def test_fits_named_c3d_targets_and_publishes_artifact(self):
        """Fit finite name-joined targets and write a sealed motion artifact."""
        q0 = self.model.joint_q.numpy().copy()
        q1 = q0.copy()
        q1[0] += 0.01
        q1[7] = 0.03
        q1[13] = 0.12
        q1[18] = 0.14
        target_q = np.asarray((q0, q1), dtype=np.float32)
        target_native = np.asarray(
            [marker_positions_from_joint_q(self.model, self.attachments, value) for value in target_q]
        )
        source_names = list(
            dict.fromkeys((*_SOURCE_FOR_NATIVE.values(), "LPSI", "RPSI", "LFHD", "RFHD", "LBHD", "RBHD"))
        )
        source_index = {name: index for index, name in enumerate(source_names)}
        source_positions = np.zeros((2, len(source_names), 3), dtype=np.float32)
        for marker_index, attachment in enumerate(self.attachments):
            if attachment.name in _SOURCE_FOR_NATIVE:
                sources = (_SOURCE_FOR_NATIVE[attachment.name],)
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
        motion = fit_c3d_marker_motion(self.model, self.attachments, mapped, q0, iterations=40)
        self.assertEqual(motion.joint_q.shape, (2, self.model.joint_coord_count))
        self.assertTrue(np.all(np.isfinite(motion.joint_qd)))
        self.assertLess(float(np.max(motion.frame_rms)), 1.0e-3)
        with tempfile.TemporaryDirectory() as directory:
            output = write_native_motion_artifact(
                motion,
                Path(directory) / "motion",
                model_path=Path("projects/gait_c3d/assets/s001_calibrated/model/subject.xml"),
            )
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertTrue((output / "motion.npz").is_file())
            self.assertEqual(manifest["schema_version"], "gait_native_motion_artifact_1")
        self.assertEqual(manifest["frames"]["count"], 2)

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
