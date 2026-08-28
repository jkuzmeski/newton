# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test static marker segment calibration."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from projects.gait_c3d.c3d_adapter import C3DMarkerTrajectory
from projects.gait_c3d.segment_calibration import (
    build_static_segment_calibration,
    load_static_segment_calibration,
)


class TestStaticSegmentCalibration(unittest.TestCase):
    """Test Visual3D-style CODA and segment landmark construction."""

    @staticmethod
    def _markers() -> C3DMarkerTrajectory:
        """Create a finite bilateral static marker pose."""
        values = {
            "LASI": (0.0, 0.12, 1.0),
            "RASI": (0.0, -0.12, 1.0),
            "LPSI": (-0.20, 0.08, 1.05),
            "RPSI": (-0.20, -0.08, 1.05),
            "LKNE": (-0.02, 0.17, 0.50),
            "LMKNE": (-0.02, 0.05, 0.50),
            "RKNE": (-0.02, -0.17, 0.50),
            "RMKNE": (-0.02, -0.05, 0.50),
            "LANK": (-0.01, 0.16, 0.08),
            "LMANK": (-0.01, 0.07, 0.08),
            "RANK": (-0.01, -0.16, 0.08),
            "RMANK": (-0.01, -0.07, 0.08),
            "LHEE": (-0.12, 0.12, 0.04),
            "LTOE": (0.13, 0.12, 0.04),
            "RHEE": (-0.12, -0.12, 0.04),
            "RTOE": (0.13, -0.12, 0.04),
            "LMTH1": (0.10, 0.08, 0.04),
            "LMTH5": (0.10, 0.16, 0.04),
            "RMTH1": (0.10, -0.08, 0.04),
            "RMTH5": (0.10, -0.16, 0.04),
        }
        names = tuple(values)
        positions = np.asarray([[values[name] for name in names]], dtype=np.float32)
        return C3DMarkerTrajectory(
            times=np.asarray((0.5,), dtype=np.float64),
            positions=positions,
            valid=np.ones((1, len(names)), dtype=bool),
            marker_names=names,
            rate=100.0,
            first_frame=0,
            lab_to_newton=np.eye(3),
            source_file="static.c3d",
            source_sha256="0" * 64,
        )

    def test_builds_coda_hips_and_per_side_segments(self):
        """Build CODA hip centers, knee/ankle centers, and segment lengths."""
        with tempfile.TemporaryDirectory() as directory:
            calibration = build_static_segment_calibration(
                self._markers(), Path(directory) / "calibration.json", marker_radius=0.006
            )
        self.assertAlmostEqual(calibration.pelvis["asis_distance_m"], 0.24, places=6)
        np.testing.assert_allclose(
            np.asarray(calibration.pelvis["hip_centers_m"]["left"])
            - np.asarray(calibration.pelvis["hip_centers_m"]["right"]),
            (0.0, 0.72 * 0.24, 0.0),
            atol=1.0e-6,
        )
        self.assertAlmostEqual(calibration.segments["thigh_left"]["length_m"], calibration.segments["thigh_right"]["length_m"])
        self.assertAlmostEqual(calibration.segments["thigh_left"]["width_m"], 0.12, places=6)
        self.assertAlmostEqual(calibration.segments["shank_left"]["width_m"], 0.09, places=6)
        self.assertTrue(calibration.segments["foot_left"]["flat_ground"])
        np.testing.assert_allclose(
            np.asarray(calibration.segments["foot_left"]["basis_forward_left_up"])[:, 2],
            (0.0, 0.0, 1.0),
            atol=1.0e-12,
        )

    def test_round_trips_sealed_calibration(self):
        """Seal and reload the static calibration without changing values."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            calibration = build_static_segment_calibration(self._markers(), path, marker_radius=0.006)
            loaded = load_static_segment_calibration(path)
            manifest = json.loads(path.read_text(encoding="utf-8"))
            seal = manifest.pop("seal")
            expected = hashlib.sha256(
                json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            ).hexdigest()
        self.assertEqual(seal, {"algorithm": "sha256", "content_sha256": expected})
        np.testing.assert_allclose(
            loaded.marker_positions["LASI"], calibration.marker_positions["LASI"], atol=1.0e-12
        )

    def test_rejects_tampered_calibration(self):
        """Reject calibration payloads whose values no longer match their seal."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            build_static_segment_calibration(self._markers(), path, marker_radius=0.006)
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest["markers"]["LASI"][0] += 0.1
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "seal mismatch"):
                load_static_segment_calibration(path)


if __name__ == "__main__":
    unittest.main()
