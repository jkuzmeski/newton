# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test direct C3D-to-NPZ/Warp marker conversion."""

import importlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from projects.gait_c3d.c3d_adapter import (
    C3DMarkerTrajectory,
    c3d_to_marker_artifact,
    lab_to_newton_rotation,
    load_marker_artifact,
    read_c3d_markers,
    write_marker_artifact,
)


class TestGaitC3DAdapter(unittest.TestCase):
    """Test finite neutral marker artifacts and Warp upload."""

    @staticmethod
    def _markers() -> C3DMarkerTrajectory:
        return C3DMarkerTrajectory(
            times=np.asarray([0.0, 0.01], dtype=np.float64),
            positions=np.asarray(
                [
                    [[0.0, 0.0, 1.0], [0.0, 0.1, 1.0]],
                    [[0.01, 0.0, 1.0], [0.01, 0.1, 1.0]],
                ],
                dtype=np.float32,
            ),
            valid=np.asarray([[True, True], [True, False]]),
            marker_names=("LASI", "RASI"),
            rate=100.0,
            first_frame=0,
            lab_to_newton=lab_to_newton_rotation("+Z", "-Y"),
            source_file="static.c3d",
            source_sha256="0" * 64,
        )

    def test_maps_lab_axes_to_newton(self):
        """Map lab forward, left, and up to Newton +X, +Y, and +Z."""
        rotation = lab_to_newton_rotation("+Z", "-Y")
        lab = np.asarray(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
        np.testing.assert_allclose(lab @ rotation.T, np.eye(3), atol=1.0e-12)

    def test_round_trips_sealed_npz_and_uploads_warp(self):
        """Preserve finite arrays and visibility through NPZ and Warp upload."""
        markers = self._markers()
        with tempfile.TemporaryDirectory() as directory:
            artifact = write_marker_artifact(markers, Path(directory) / "artifact")
            loaded = load_marker_artifact(artifact)
        np.testing.assert_array_equal(loaded.times, markers.times)
        np.testing.assert_array_equal(loaded.positions, markers.positions)
        np.testing.assert_array_equal(loaded.valid, markers.valid)
        self.assertEqual(loaded.marker_names, markers.marker_names)
        device = loaded.to_warp("cpu")
        self.assertEqual(device.positions.shape, (2, 2))
        np.testing.assert_array_equal(device.positions.numpy(), markers.positions)
        np.testing.assert_array_equal(device.valid.numpy(), markers.valid.astype(np.uint8))

    def test_rejects_tampered_npz(self):
        """Reject a marker payload whose bytes no longer match its manifest."""
        with tempfile.TemporaryDirectory() as directory:
            artifact = write_marker_artifact(self._markers(), Path(directory) / "artifact")
            payload = artifact / "markers.npz"
            payload.write_bytes(payload.read_bytes() + b"tamper")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_marker_artifact(artifact)

    def test_rejects_tampered_manifest(self):
        """Reject semantic metadata changes that break the manifest seal."""
        with tempfile.TemporaryDirectory() as directory:
            artifact = write_marker_artifact(self._markers(), Path(directory) / "artifact")
            path = artifact / "manifest.json"
            manifest = json.loads(path.read_text())
            manifest["markers"]["names"].reverse()
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "manifest seal mismatch"):
                load_marker_artifact(artifact)

    def test_decodes_synthetic_c3d_without_intermediate_files(self):
        """Decode an ezc3d file directly into Newton-frame meters and NPZ."""
        if importlib.util.find_spec("ezc3d") is None:
            self.skipTest("ezc3d is not installed")
        ezc3d = importlib.import_module("ezc3d")
        c3d = ezc3d.c3d()
        c3d["parameters"]["POINT"]["RATE"]["value"] = [100.0]
        c3d["parameters"]["POINT"]["LABELS"]["value"] = ["SUBJECT:LASI", "SUBJECT:RASI"]
        points = np.zeros((4, 2, 2), dtype=float)
        points[:3, 0, :] = np.asarray(((0.0, 0.0), (0.0, -10.0), (1000.0, 1000.0)))
        points[:3, 1, :] = np.asarray(((100.0, 100.0), (0.0, -10.0), (1000.0, 1000.0)))
        points[3] = 1.0
        c3d["data"]["points"] = points
        c3d["header"]["points"]["first_frame"] = 10
        c3d["header"]["points"]["last_frame"] = 11
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "static.c3d"
            c3d.write(str(source))
            markers = read_c3d_markers(source)
            artifact = c3d_to_marker_artifact(source, Path(directory) / "artifact")
            loaded = load_marker_artifact(artifact)
        self.assertEqual(markers.marker_names, ("LASI", "RASI"))
        self.assertEqual(markers.first_frame, 10)
        np.testing.assert_allclose(markers.times, (0.0, 0.01))
        np.testing.assert_allclose(markers.positions[0, 0], (0.0, 0.0, 1.0))
        np.testing.assert_allclose(markers.positions[0, 1], (0.0, 0.1, 1.0))
        np.testing.assert_array_equal(loaded.positions, markers.positions)


if __name__ == "__main__":
    unittest.main()
