# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test the interactive C3D-to-MJCF marker-mapping example."""

import unittest

import numpy as np

from newton.examples.opensim.example_marker_mapper import (
    _DEFAULT_SOURCE,
    Example,
    _fit_display_registration,
    _unique_name_suggestions,
    create_parser,
)
from projects.gait_c3d.c3d_adapter import C3DMarkerTrajectory


class TestGaitMarkerMapper(unittest.TestCase):
    """Test pure visual-mapper behavior without opening a viewer."""

    def test_fits_display_only_rigid_registration(self):
        """Align corresponding source and MJCF marker points rigidly."""
        source = np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.2), (0.2, 0.3, 1.0)))
        rotation = np.asarray(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
        translation = np.asarray((0.5, -0.25, 1.2))
        target = source @ rotation + translation
        fitted_rotation, fitted_translation = _fit_display_registration(source, target)
        np.testing.assert_allclose(source @ fitted_rotation + fitted_translation, target, atol=1.0e-12)
        self.assertAlmostEqual(float(np.linalg.det(fitted_rotation)), 1.0)

    def test_suggests_only_unique_normalized_label_matches(self):
        """Offer quick name hints without resolving ambiguous labels automatically."""
        suggestions = _unique_name_suggestions(
            ("LASI", "RASI", "LANK"),
            ("LAB_L_ASIS", "RASI_A", "RASI_B", "session-LANK"),
        )
        self.assertEqual(suggestions, {"LASI": "LAB_L_ASIS", "LANK": "session-LANK"})

    def test_identity_reversion_matches_saved_map_semantics(self):
        """Keep identity assigned and reject stealing another effective source."""
        editor = object.__new__(Example)
        editor.roles = ("LASI", "RASI")
        editor.selected_target = 0
        editor.aliases = {}
        editor.edited_roles = set()
        editor.assignments = {"LASI": "LASI", "RASI": "RASI"}
        editor.raw_markers = C3DMarkerTrajectory(
            times=np.asarray((0.0,)),
            positions=np.zeros((1, 2, 3), dtype=np.float32),
            valid=np.ones((1, 2), dtype=bool),
            marker_names=("LASI", "RASI"),
            rate=100.0,
            first_frame=0,
            lab_to_newton=np.eye(3),
            source_file="static.c3d",
            source_sha256="0" * 64,
        )
        editor.dirty = False
        editor.status = ""
        editor._update_visuals = lambda: None
        editor._assign(_DEFAULT_SOURCE)
        self.assertEqual(editor.assignments["LASI"], "LASI")
        editor._assign("RASI")
        self.assertEqual(editor.assignments, {"LASI": "LASI", "RASI": "RASI"})
        self.assertIn("already assigned", editor.status)

    def test_preserves_unresolved_loaded_aliases_until_edited(self):
        """Keep configured aliases when the current C3D lacks their source labels."""
        editor = object.__new__(Example)
        editor.aliases = {"LASI": "LAB_LASI"}
        editor.edited_roles = set()
        editor.assignments = {"RASI": "RASI"}
        marker_map = editor._current_map()
        self.assertEqual(marker_map.source_for("LASI"), "LAB_LASI")

    def test_parses_real_marker_mapping_inputs(self):
        """Expose the subject, C3D, map, frame, and exact-prefix controls."""
        args = create_parser().parse_args(
            (
                "--subject",
                "/tmp/subject",
                "--c3d",
                "/tmp/static.c3d",
                "--marker-map",
                "/tmp/map.json",
                "--frame",
                "12",
                "--keep-c3d-prefix",
            )
        )
        self.assertEqual(args.subject, "/tmp/subject")
        self.assertEqual(args.c3d, "/tmp/static.c3d")
        self.assertEqual(args.marker_map, "/tmp/map.json")
        self.assertEqual(args.frame, 12)
        self.assertTrue(args.keep_c3d_prefix)


if __name__ == "__main__":
    unittest.main()
