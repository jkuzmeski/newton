# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for the human-shoe attachment viewer."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton.viewer
from projects.human_shoe.viewer import Example


class _StrictColorViewer(newton.viewer.ViewerNull):
    """Match the GL viewer's array requirement for point-instance colors."""

    def log_points(self, name, points, radii=None, colors=None, hidden=False):
        if colors is not None and not isinstance(colors, wp.array):
            raise TypeError("point colors must be a Warp array")
        super().log_points(name, points, radii, colors, hidden)


class TestHumanShoeViewer(unittest.TestCase):
    def test_smoke_builds_attachment_scene(self):
        """Build the viewer scene and verify the imported shoe attachment is finite."""
        viewer = newton.viewer.ViewerNull(num_frames=1)
        example = Example(viewer, Example.create_parser().parse_args([]))
        example.test_final()
        np.testing.assert_allclose(example.model.gravity.numpy()[0], [0.0, 0.0, -9.80665], atol=1.0e-5)
        self.assertGreater(example.model.shape_count, 0)
        self.assertEqual(example.model.shape_label[example.midsole_shape], "digital_instron_midsole")
        self.assertEqual(
            int(example.model.shape_body.numpy()[example.midsole_shape]),
            example.resolved.shoe_carrier_body_index,
        )
        self.assertTrue(np.all(np.isfinite(example.column_top_local)))
        self.assertTrue(np.all(np.isfinite(example.column_bottom_local)))
        self.assertGreater(np.count_nonzero(example.column_rest_len > 0.0), 0)
        self.assertLess(example.attachment_alignment_rms_m, 1.0e-7)
        self.assertLess(example.attachment_alignment_max_m, 1.0e-7)
        np.testing.assert_allclose(
            example.resolved.shoe_to_foot[:3, 3],
            [0.13274929, -0.01622024, 0.010912963333333333],
            atol=1.0e-12,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            example.column_top_local.mean(axis=0),
            example.resolved.shoe_to_foot[:3, 3],
            atol=1.0e-6,
        )
        for contact_name in example.resolved.reference.contact_geometry_names:
            self.assertEqual(example.model.shape_label.count(contact_name), 1)
        for contact in example.osim_model.contact_geometry:
            if contact.body != "calcn_r" or contact.type != "ContactSphere":
                continue
            support = np.asarray(contact.location, dtype=np.float64) - np.array([0.0, float(contact.radius), 0.0])
            nearest_column = np.linalg.norm(example.column_top_local - support, axis=1).min()
            self.assertLess(nearest_column, 1.0e-7)

    def test_render_headless(self):
        """Render one frame with GL-compatible Warp color arrays."""
        viewer = _StrictColorViewer(num_frames=1)
        args = Example.create_parser().parse_args(["--show-columns", "--show-column-lines"])
        example = Example(viewer, args)
        compressions = []
        for frame in np.linspace(0, example.num_frames - 1, 12, dtype=int):
            example.frame = int(frame)
            wp.copy(example.state.body_q, example.body_q_frames[example.frame])
            example._update_column_deformation()
            compressions.append((example.frame, example._column_compression.numpy()))
        example.render()
        self.assertIsNotNone(example.state.body_q)
        self.assertEqual(example.state.body_q.device, wp.get_device())
        self.assertEqual(viewer.frame_count, 1)
        peak_frame, peak_compression = max(compressions, key=lambda item: float(np.max(item[1])))
        peak = float(np.max(peak_compression))
        self.assertGreater(peak, 1.0e-4)
        self.assertGreater(float(np.ptp([float(np.max(c)) for _, c in compressions])), 1.0e-4)

        example.frame = peak_frame
        wp.copy(example.state.body_q, example.body_q_frames[example.frame])
        example._update_column_deformation()
        bottom = example._column_bottom_world.numpy()
        top = example._column_top_world.numpy()
        self.assertGreaterEqual(float(bottom[:, 2].min()), -1.0e-7)
        self.assertTrue(np.any(np.isclose(bottom[:, 2], 0.0, atol=1.0e-7)))
        shortening = example.column_rest_len - np.linalg.norm(top - bottom, axis=1)
        self.assertGreater(float(shortening.max()), 1.0e-4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
