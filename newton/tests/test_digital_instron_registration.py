# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Digital Instron last-to-midsole registration."""

import tempfile
import unittest

import numpy as np

from projects.digital_instron_v2.geometry import ColumnGrid
from projects.digital_instron_v2.registration import LastModel, load_last, register_last

LENGTH = 0.24


def _profile(x):
    """Plantar height [m] along the last: 5 mm heel seat, ball tread at 68 %, toe spring."""
    t = np.clip(x / LENGTH, 0.0, 1.0)
    return 0.005 * (np.maximum(0.68 - t, 0.0) / 0.68) ** 1.5 + 0.010 * (np.maximum(t - 0.68, 0.0) / 0.32) ** 2


def _half_width(x):
    """Tread half width [m]: narrow heel, broad ball."""
    t = np.clip(x / LENGTH, 0.0, 1.0)
    return 0.026 + 0.014 * np.sin(np.pi * np.minimum(t / 0.68, 1.0) / 2.0) - 0.020 * np.maximum(t - 0.85, 0.0) / 0.15


def _plantar(x, y):
    """Plantar height [m] with a 3 mm/40 mm cross-cup."""
    return _profile(x) + 0.003 * (y / 0.04) ** 2


def _synthetic_last():
    xs = np.arange(0.0, LENGTH + 1e-9, 0.001)
    ys = np.arange(-0.040, 0.040 + 1e-9, 0.001)
    x, y = np.meshgrid(xs, ys)
    z = np.where(np.abs(y) <= _half_width(x), _plantar(x, y), np.nan)
    return LastModel(xs, ys, z - np.nanmin(z), np.eye(4), {})


class TestDigitalInstronRegistration(unittest.TestCase):
    def test_last_up_axis_is_the_attachment_plate_normal(self):
        """Use the plate normal, not the body's extents, and find the narrow heel end."""
        import trimesh

        sections = []
        for x in np.linspace(0.0, LENGTH, 49):
            for y in np.linspace(-1.0, 1.0, 9) * _half_width(x):
                sections += [[x, y, _plantar(x, y)], [x, y, 0.045]]
        body = trimesh.convex.convex_hull(np.asarray(sections))
        stud = trimesh.creation.cylinder(radius=0.01, segment=[[0.12, 0.0, 0.045], [0.12, 0.0, 0.110]])
        plate = trimesh.creation.box(extents=[0.095, 0.095, 0.006])
        plate.apply_translation([0.12, 0.0, 0.113])
        attachment = trimesh.util.concatenate([stud, plate])
        tilt = trimesh.transformations.rotation_matrix(np.deg2rad(1.0), [0.0, 1.0, 0.0], [0.12, 0.0, 0.045])
        attachment.apply_transform(tilt)
        assembly = trimesh.util.concatenate([body, attachment])
        # Toe toward -source y, up toward +source x, millimeters.
        source = trimesh.transformations.rotation_matrix(
            np.pi / 2, [0.0, 0.0, 1.0]
        ) @ trimesh.transformations.rotation_matrix(-np.pi / 2, [0.0, 1.0, 0.0])
        assembly.apply_transform(source)
        assembly.apply_scale(1000.0)
        path = f"{self.enterContext(tempfile.TemporaryDirectory())}/last.stl"
        assembly.export(path)
        last = load_last(path)
        up_source = source[:3, :3] @ tilt[:3, :3] @ [0.0, 0.0, 1.0]
        self.assertAlmostEqual(abs(float(np.dot(last.evidence["up_in_source"], up_source))), 1.0, places=5)
        self.assertGreater(float(np.dot(last.evidence["up_in_source"], up_source)), 0.0)
        toe_source = source[:3, :3] @ [1.0, 0.0, 0.0]
        self.assertGreater(float(last.source_to_last[0, :3] @ toe_source), 0.0)
        self.assertAlmostEqual(last.evidence["length_m"], LENGTH, delta=0.004)

    def test_register_recovers_pose_and_rocker_pitch(self):
        """A conforming footbed on a sole pitched 3 deg rests flat and the last touches everywhere."""
        last = _synthetic_last()
        pitch, pose = np.deg2rad(3.0), np.array([-0.125, 0.004, np.deg2rad(2.0)])
        u = np.arange(-0.145, 0.1451, 0.005)
        v = np.arange(-0.055, 0.0551, 0.005)
        uu, vv = (a.ravel() for a in np.meshgrid(u, v))
        local = (np.column_stack((uu, vv)) - pose[:2]) @ np.array(
            [[np.cos(pose[2]), -np.sin(pose[2])], [np.sin(pose[2]), np.cos(pose[2])]]
        )
        heel = (uu + 0.145) / 0.29
        keep = np.abs(vv) <= 0.040 + 0.016 * np.sin(np.pi * np.clip(heel / 0.7, 0.0, 1.0) / 2.0)
        # Flat sole pitched by 3 deg with a rocker toe curling up over the last 40 %.
        bottom = pitch * uu + 0.030 * (np.maximum(heel - 0.6, 0.0) / 0.4) ** 2
        top = pitch * uu + 0.030 + _plantar(local[:, 0], local[:, 1])
        grid = ColumnGrid(np.column_stack((uu, vv))[keep], bottom[keep], top[keep], 0.005**2, 0.005, 2)

        reg = register_last(grid, last, modulus_pa=2.4e5)
        self.assertAlmostEqual(reg.qc["rest_pitch_deg"], 3.0, delta=0.1)
        self.assertAlmostEqual(reg.qc["rest_roll_deg"], 0.0, delta=0.1)
        self.assertAlmostEqual(reg.qc["yaw_deg"], 2.0, delta=0.5)
        self.assertAlmostEqual(reg.qc["heel_offset_m"], 0.0225, delta=0.002)
        self.assertLess(reg.qc["gap_percentiles_mm"]["p95"], 0.3)
        under = np.isfinite(reg.surface_m)
        np.testing.assert_allclose(reg.surface_m[under], grid.top_m[under], atol=5e-4)

        turned = ColumnGrid(-grid.uv_m, grid.bottom_m, grid.top_m, grid.area_m2, grid.spacing_m, 2)
        self.assertTrue(register_last(turned, last, modulus_pa=2.4e5).qc["shoe_heel_at_input_plus_x"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
