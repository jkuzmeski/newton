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
from projects.gait_c3d.marker_map import C3DMarkerMap, apply_c3d_marker_map
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
            "LHLX": (0.13, 0.12, 0.04),
            "RHEE": (-0.12, -0.12, 0.04),
            "RHLX": (0.13, -0.12, 0.04),
            "LMTH1": (0.10, 0.08, 0.04),
            "LMTH5": (0.10, 0.16, 0.04),
            "RMTH1": (0.10, -0.08, 0.04),
            "RMTH5": (0.10, -0.16, 0.04),
            "LTH2": (-0.08, 0.10, 0.70),
            "LTH3": (-0.07, 0.11, 0.65),
            "LTH4": (-0.06, 0.12, 0.68),
            "RTH2": (-0.08, -0.10, 0.70),
            "RTH3": (-0.07, -0.11, 0.65),
            "RTH4": (-0.06, -0.12, 0.68),
            "LTIB2": (-0.03, 0.10, 0.35),
            "LTIB3": (-0.02, 0.11, 0.30),
            "LTIB4": (-0.01, 0.12, 0.33),
            "RTIB2": (-0.03, -0.10, 0.35),
            "RTIB3": (-0.02, -0.11, 0.30),
            "RTIB4": (-0.01, -0.12, 0.33),
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
        self.assertAlmostEqual(
            calibration.segments["thigh_left"]["length_m"], calibration.segments["thigh_right"]["length_m"]
        )
        self.assertAlmostEqual(calibration.segments["thigh_left"]["width_m"], 0.12, places=6)
        self.assertAlmostEqual(calibration.segments["shank_left"]["width_m"], 0.09, places=6)
        self.assertTrue(calibration.segments["foot_left"]["flat_ground"])
        np.testing.assert_allclose(
            calibration.marker_positions["L.Thigh.Centroid"],
            np.mean([calibration.marker_positions[name] for name in ("LTH2", "LTH3", "LTH4")], axis=0),
        )
        np.testing.assert_allclose(
            np.asarray(calibration.segments["foot_left"]["basis_forward_left_up"])[:, 2],
            (0.0, 0.0, 1.0),
            atol=1.0e-12,
        )

    def test_custom_label_map_preserves_static_calibration(self):
        """Build identical segment geometry from canonical and custom C3D labels."""
        canonical = self._markers()
        aliases = {name: f"LAB_{name}" for name in canonical.marker_names}
        custom = C3DMarkerTrajectory(
            times=canonical.times.copy(),
            positions=canonical.positions.copy(),
            valid=canonical.valid.copy(),
            marker_names=tuple(aliases[name] for name in canonical.marker_names),
            rate=canonical.rate,
            first_frame=canonical.first_frame,
            lab_to_newton=canonical.lab_to_newton.copy(),
            source_file=canonical.source_file,
            source_sha256=canonical.source_sha256,
        )
        mapped = apply_c3d_marker_map(
            custom,
            C3DMarkerMap(aliases),
            required=canonical.marker_names,
        )
        with tempfile.TemporaryDirectory() as directory:
            first = build_static_segment_calibration(canonical, Path(directory) / "canonical.json", marker_radius=0.006)
            second = build_static_segment_calibration(mapped, Path(directory) / "custom.json", marker_radius=0.006)
        self.assertEqual(first.pelvis, second.pelvis)
        self.assertEqual(first.segments, second.segments)
        self.assertEqual(set(first.marker_positions), set(second.marker_positions))
        for name in first.marker_positions:
            np.testing.assert_array_equal(first.marker_positions[name], second.marker_positions[name])

    def test_preserves_raw_psis_slope_in_pelvis_frame(self):
        """Preserve the bilateral PSIS slope in the calibrated pelvis frame."""
        markers = self._markers()
        positions = markers.positions.copy()
        positions[0, markers.marker_names.index("LPSI"), 2] += 0.02
        positions[0, markers.marker_names.index("RPSI"), 2] -= 0.02
        tilted = C3DMarkerTrajectory(
            times=markers.times,
            positions=positions,
            valid=markers.valid,
            marker_names=markers.marker_names,
            rate=markers.rate,
            first_frame=markers.first_frame,
            lab_to_newton=markers.lab_to_newton,
            source_file=markers.source_file,
            source_sha256=markers.source_sha256,
        )
        with tempfile.TemporaryDirectory() as directory:
            calibration = build_static_segment_calibration(
                tilted, Path(directory) / "calibration.json", marker_radius=0.006
            )
        pelvis = calibration.pelvis
        right_axis = np.asarray(pelvis["basis_right_anterior_up"])[:, 0]
        self.assertGreater(abs(float(right_axis[2])), 0.01)
        np.testing.assert_allclose(
            pelvis["posterior_markers"]["left"],
            calibration.marker_positions["LPSI"],
            atol=1.0e-7,
        )
        np.testing.assert_allclose(
            pelvis["posterior_markers"]["right"],
            calibration.marker_positions["RPSI"],
            atol=1.0e-7,
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
        np.testing.assert_allclose(loaded.marker_positions["LASI"], calibration.marker_positions["LASI"], atol=1.0e-12)

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
