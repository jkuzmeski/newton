# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test exact versioned C3D marker-label alias maps."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from projects.gait_c3d.c3d_adapter import C3DMarkerTrajectory
from projects.gait_c3d.marker_map import (
    CANONICAL_C3D_MARKERS,
    MARKER_MAP_SCHEMA_VERSION,
    NATIVE_MARKER_SOURCES,
    S001_MARKER_MAP,
    C3DMarkerMap,
    MarkerMapError,
    apply_c3d_marker_map,
    load_c3d_marker_map,
    required_c3d_sources,
    save_c3d_marker_map,
    validate_c3d_marker_map,
)


class TestGaitMarkerMap(unittest.TestCase):
    """Test immutable marker maps, validation, and trajectory application."""

    @staticmethod
    def _trajectory(names=("L_ASIS", "RASI", "EXTRA")) -> C3DMarkerTrajectory:
        frame_count = 2
        marker_count = len(names)
        positions = np.arange(frame_count * marker_count * 3, dtype=np.float32).reshape(frame_count, marker_count, 3)
        return C3DMarkerTrajectory(
            times=np.asarray([0.0, 0.01], dtype=np.float64),
            positions=positions,
            valid=np.ones((frame_count, marker_count), dtype=bool),
            marker_names=tuple(names),
            rate=100.0,
            first_frame=17,
            lab_to_newton=np.eye(3, dtype=np.float64),
            source_file="trial.c3d",
            source_sha256="a" * 64,
        )

    def test_exposes_s001_identity_profile_and_native_recipes(self):
        """Keep the S001 profile exact and flatten native recipes in request order."""
        self.assertEqual(S001_MARKER_MAP.schema_version, MARKER_MAP_SCHEMA_VERSION)
        self.assertEqual(dict(S001_MARKER_MAP.markers), {})
        self.assertEqual(NATIVE_MARKER_SOURCES["V.Sacral"], ("LPSI", "RPSI"))
        self.assertEqual(NATIVE_MARKER_SOURCES["L.Toe.Tip"], ("LHLX",))
        self.assertNotIn("LTOE", CANONICAL_C3D_MARKERS)
        self.assertTrue({"C7", "CLAV", "T10"}.issubset(CANONICAL_C3D_MARKERS))
        self.assertEqual(
            required_c3d_sources(("V.Sacral", "L.ASIS", "V.Sacral", "Top.Head")),
            ("LPSI", "RPSI", "LASI", "LFHD", "RFHD", "LBHD", "RBHD"),
        )
        with self.assertRaises(TypeError):
            NATIVE_MARKER_SOURCES["Other"] = ("OTHER",)

    def test_rejects_unknown_native_attachment_names_together(self):
        """Report all unknown native attachment names in one error."""
        with self.assertRaisesRegex(ValueError, "Missing.*Other"):
            required_c3d_sources(("Missing", "L.ASIS", "Other"))

    def test_round_trips_partial_alias_map_as_deterministic_json(self):
        """Round-trip a partial immutable map while omitted labels stay identity."""
        mutable = {"LASI": "L_ASIS", "RASI": "R_ASIS"}
        marker_map = C3DMarkerMap(mutable)
        mutable["LASI"] = "changed"
        self.assertEqual(marker_map.source_for("LASI"), "L_ASIS")
        self.assertEqual(marker_map.source_for("LPSI"), "LPSI")
        with self.assertRaises(TypeError):
            marker_map.markers["LASI"] = "changed"
        with tempfile.TemporaryDirectory() as directory:
            path = save_c3d_marker_map(marker_map, Path(directory) / "nested" / "map.json")
            first = path.read_bytes()
            loaded = load_c3d_marker_map(path)
            save_c3d_marker_map(loaded, path)
            second = path.read_bytes()
        self.assertEqual(first, second)
        self.assertEqual(loaded, marker_map)
        self.assertEqual(json.loads(first)["markers"], dict(marker_map.markers))

    def test_detects_duplicate_json_keys(self):
        """Reject duplicate top-level and marker JSON keys instead of overwriting."""
        documents = (
            '{"schema_version":"gait_c3d_marker_map_1","markers":{},"markers":{}}',
            '{"schema_version":"gait_c3d_marker_map_1","markers":{"LASI":"A","LASI":"B"}}',
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "map.json"
            for document in documents:
                with self.subTest(document=document):
                    path.write_text(document, encoding="utf-8")
                    with self.assertRaisesRegex(MarkerMapError, "duplicate JSON key"):
                        load_c3d_marker_map(path)

    def test_groups_unknown_fields_keys_and_bad_version(self):
        """Collect independent schema and canonical-vocabulary issues together."""
        value = {
            "schema_version": "future",
            "markers": {"NOT_A_GAIT_MARKER": "SOURCE"},
            "axis": "+Z",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "map.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(MarkerMapError) as raised:
                load_c3d_marker_map(path)
        self.assertEqual(len(raised.exception.issues), 3)
        self.assertEqual(
            {issue.code for issue in raised.exception.issues},
            {"unknown_field", "unsupported_schema_version", "unknown_canonical_marker"},
        )

    def test_rejects_duplicate_effective_sources(self):
        """Reject explicit source reuse and aliases that collide with identity."""
        for aliases in (
            {"LASI": "LEFT", "RASI": "LEFT"},
            {"LASI": "RASI"},
        ):
            with self.subTest(aliases=aliases):
                with self.assertRaisesRegex(MarkerMapError, "both resolve to source"):
                    C3DMarkerMap(aliases)

    def test_groups_missing_and_invalid_required_labels(self):
        """Collect every missing, repeated, and unknown caller requirement."""
        trajectory = self._trajectory()
        marker_map = C3DMarkerMap({"LASI": "MISSING_LEFT", "RASI": "MISSING_RIGHT"})
        validation = validate_c3d_marker_map(
            trajectory,
            marker_map,
            required=("LASI", "RASI", "RASI", "UNKNOWN"),
        )
        self.assertFalse(validation.is_valid)
        self.assertEqual(
            [issue.code for issue in validation.issues],
            [
                "duplicate_required_marker",
                "unknown_required_marker",
                "missing_source_label",
                "missing_source_label",
            ],
        )
        with self.assertRaises(MarkerMapError) as raised:
            validation.raise_for_errors()
        self.assertIn("4 issues", str(raised.exception))
        self.assertIn("MISSING_LEFT", str(raised.exception))
        self.assertIn("MISSING_RIGHT", str(raised.exception))

    def test_applies_exact_aliases_without_mutating_raw_trajectory(self):
        """Canonicalize exact columns and preserve independent arrays and provenance."""
        trajectory = self._trajectory()
        raw_times = trajectory.times.copy()
        raw_positions = trajectory.positions.copy()
        raw_valid = trajectory.valid.copy()
        raw_rotation = trajectory.lab_to_newton.copy()
        marker_map = C3DMarkerMap({"LASI": "L_ASIS"})
        canonical = apply_c3d_marker_map(trajectory, marker_map, required=("LASI", "RASI"))
        self.assertEqual(canonical.marker_names, ("LASI", "RASI", "EXTRA"))
        np.testing.assert_array_equal(canonical.positions, raw_positions)
        self.assertEqual(canonical.rate, trajectory.rate)
        self.assertEqual(canonical.first_frame, trajectory.first_frame)
        self.assertEqual(canonical.source_file, trajectory.source_file)
        self.assertEqual(canonical.source_sha256, trajectory.source_sha256)
        self.assertFalse(np.shares_memory(canonical.times, trajectory.times))
        self.assertFalse(np.shares_memory(canonical.positions, trajectory.positions))
        self.assertFalse(np.shares_memory(canonical.valid, trajectory.valid))
        self.assertFalse(np.shares_memory(canonical.lab_to_newton, trajectory.lab_to_newton))
        canonical.times[0] = 10.0
        canonical.positions[0, 0] = -1.0
        canonical.valid[0, 0] = False
        canonical.lab_to_newton[0, 0] = -1.0
        np.testing.assert_array_equal(trajectory.times, raw_times)
        np.testing.assert_array_equal(trajectory.positions, raw_positions)
        np.testing.assert_array_equal(trajectory.valid, raw_valid)
        np.testing.assert_array_equal(trajectory.lab_to_newton, raw_rotation)

    def test_uses_explicit_source_and_drops_same_named_shadow(self):
        """Use an explicit source authoritatively when a canonical raw label also exists."""
        trajectory = self._trajectory(("LASI", "L_ASIS", "RASI"))
        canonical = apply_c3d_marker_map(
            trajectory,
            C3DMarkerMap({"LASI": "L_ASIS"}),
            required=("LASI",),
        )
        self.assertEqual(canonical.marker_names, ("LASI", "RASI"))
        np.testing.assert_array_equal(canonical.positions[:, 0], trajectory.positions[:, 1])

    def test_never_falls_back_or_normalizes_source_labels(self):
        """Reject absent exact aliases without fallback, case folding, or punctuation changes."""
        trajectory = self._trajectory(("LASI", "left_asis"))
        marker_map = C3DMarkerMap({"LASI": "LEFT_ASIS"})
        with self.assertRaises(MarkerMapError) as raised:
            apply_c3d_marker_map(trajectory, marker_map, required=("LASI",))
        issue = raised.exception.issues[0]
        self.assertEqual(issue.canonical, "LASI")
        self.assertEqual(issue.source, "LEFT_ASIS")


if __name__ == "__main__":
    unittest.main()
