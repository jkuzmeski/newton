# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for turning midsole scans into canonical column beds."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from projects.digital_instron_v2.shoe_prep import CELL_RAYS, SPACING_M, ShoePrepError, prepare_shoe, sample_columns

try:
    import trimesh
except ImportError:
    trimesh = None


def _midsole(medial_indent_m: float = 0.018, lateral_indent_m: float = 0.006, side: str = "right"):
    """Return a right (or mirrored left) midsole in canonical axes [m].

    A subdivided unit box is bent into a 280 mm sole: narrow thick heel at
    x = 0, wide forefoot, 20 mm toe spring, and a deeper arch on the medial
    (+y for a right shoe) outline.
    """
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    vertices, faces = trimesh.remesh.subdivide_to_size(box.vertices + 0.5, box.faces, max_edge=0.02)
    u, w, h = vertices.T
    half_width = 0.038 + 0.014 * u
    arch = np.exp(-(((u - 0.5) / 0.1) ** 2))
    medial = half_width - medial_indent_m * arch
    lateral = -(half_width - lateral_indent_m * arch)
    bottom = 0.020 * np.clip((u - 0.8) / 0.2, 0.0, None) ** 2
    points = np.column_stack((0.28 * u, lateral + w * (medial - lateral), bottom + h * (0.035 - 0.008 * u)))
    if side == "left":
        points[:, 1] *= -1.0
        faces = faces[:, ::-1]
    return trimesh.Trimesh(points, faces, process=False)


def _to_source(mesh, rotation: np.ndarray):
    """Pose a canonical mesh like an RTINGS export: rotated, offset, millimeters."""
    source = mesh.copy()
    source.vertices = (mesh.vertices @ rotation.T + [0.01, -0.12, 0.03]) * 1000.0
    return source


# Heel toward source +y, width along source x, up along source z.
_RTINGS_POSE = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def _thickness_at(grid: dict, x_fraction: float) -> float:
    x = grid["uv_m"][:, 0]
    near = np.abs((x - x.min()) / np.ptp(x) - x_fraction) < 0.05
    return float(np.median(grid["top_m"][near] - grid["bottom_m"][near]))


@unittest.skipIf(trimesh is None, "trimesh is required")
class TestDigitalInstronShoePrep(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _prepare(self, mesh, name: str, **kwargs) -> dict:
        path = self.root / f"{name}.glb"
        mesh.export(path)
        return prepare_shoe(path, self.root / kwargs.pop("out", "out"), **kwargs)

    def _grid(self, slug: str, side: str, out: str = "out") -> dict:
        with np.load(self.root / out / slug / side / "column_grid.npz") as data:
            return {key: data[key] for key in data.files}

    def test_prepare_rotated_right_shoe_deterministically(self):
        """Recover heel-to-toe, the right side, and a byte-identical mirror pair."""
        source = _to_source(_midsole(), _RTINGS_POSE)
        shoe = self._prepare(source, "synthetic")
        self._prepare(source, "synthetic", out="again")
        self.assertEqual(shoe["geometry_status"], "ready")
        self.assertEqual(shoe["source_side"], "right")
        self.assertFalse(shoe["frame_evidence"]["up_flipped_from_source"])
        right, left = self._grid("synthetic", "right"), self._grid("synthetic", "left")
        self.assertGreater(_thickness_at(right, 0.12) - _thickness_at(right, 0.75), 0.004)
        np.testing.assert_allclose(np.sort(left["uv_m"][:, 1]), np.sort(-right["uv_m"][:, 1]))
        for relative in ("shoe.json", "right/column_grid.npz", "left/preflight.json", "qc_top.png"):
            self.assertEqual(
                (self.root / "out/synthetic" / relative).read_bytes(),
                (self.root / "again/synthetic" / relative).read_bytes(),
                relative,
            )
        # Rerunning in place from <slug>/source.glb keeps the slug and leaves no staging folder.
        rerun = prepare_shoe(self.root / "again/synthetic/source.glb", self.root / "again")
        self.assertEqual(rerun["slug"], "synthetic")
        self.assertEqual(sorted(path.name for path in (self.root / "again").iterdir()), ["synthetic"])
        self.assertEqual(
            (self.root / "out/synthetic/right/column_grid.npz").read_bytes(),
            (self.root / "again/synthetic/right/column_grid.npz").read_bytes(),
        )

    def test_detect_upside_down_left_shoe(self):
        """Flip an upside-down left scan upright and mirror it back to the right bed."""
        flip = np.diag([1.0, -1.0, -1.0]) @ _RTINGS_POSE
        shoe = self._prepare(_to_source(_midsole(side="left"), flip), "left_shoe")
        self.assertEqual(shoe["source_side"], "left")
        self.assertTrue(shoe["frame_evidence"]["up_flipped_from_source"])
        self.assertTrue(shoe["right"]["mirrored_from_source"])
        self._prepare(_to_source(_midsole(), _RTINGS_POSE), "right_shoe")
        mirrored, direct = self._grid("left_shoe", "right"), self._grid("right_shoe", "right")
        self.assertLess(abs(len(mirrored["uv_m"]) - len(direct["uv_m"])), 0.02 * len(direct["uv_m"]))
        for fraction in (0.12, 0.5, 0.75):
            self.assertAlmostEqual(_thickness_at(mirrored, fraction), _thickness_at(direct, fraction), delta=0.001)

    def test_fill_columns_lost_to_a_bottom_hole(self):
        """Interpolate columns behind an open bottom patch and report the boundary."""
        intact = self._prepare(_to_source(_midsole(), _RTINGS_POSE), "intact")
        mesh = _midsole()
        centers = mesh.triangles_center
        hole = (np.linalg.norm(centers[:, :2] - [0.14, 0.0], axis=1) < 0.012) & (mesh.face_normals[:, 2] < -0.9)
        mesh.update_faces(~hole)
        shoe = self._prepare(_to_source(mesh, _RTINGS_POSE), "holed")
        self.assertGreater(shoe["mesh_qc"]["boundary_after_fill"]["open_edges"], 0)
        self.assertGreater(shoe["column_qc"]["hole_gaps_filled"], 0)
        self.assertLess(shoe["column_qc"]["build_column_grid_columns"], shoe["column_qc"]["columns"])
        self.assertEqual(shoe["column_qc"]["columns"], intact["column_qc"]["columns"])
        self.assertAlmostEqual(shoe["right"]["median_thickness_mm"], intact["right"]["median_thickness_mm"], delta=0.2)

    def test_reject_ambiguous_side_and_extra_bodies(self):
        """Reject a symmetric outline and a scan with a second body, with a reason on disk."""
        with self.assertRaisesRegex(ShoePrepError, "medial side not established"):
            self._prepare(_to_source(_midsole(0.010, 0.010), _RTINGS_POSE), "symmetric")
        status = json.loads((self.root / "out/symmetric/shoe.json").read_text())
        self.assertEqual(status["geometry_status"], "rejected")
        self.assertFalse((self.root / "out/symmetric/right/column_grid.npz").exists())
        extra = _midsole()
        extra.apply_translation([0.0, 0.0, 0.1])
        with self.assertRaisesRegex(ShoePrepError, "bodies"):
            self._prepare(_to_source(trimesh.util.concatenate([_midsole(), extra]), _RTINGS_POSE), "two_bodies")

    def test_outline_cells_average_their_covered_area(self):
        """An outline cell keeps the mean thickness and covered area of its cell, not its center ray."""
        # Cross-section in (y, z): 20 mm thick, tapering to 6 mm over the outer 4 mm of each side.
        section = np.array([[-21.0, 0.0], [21.0, 0.0], [21.0, 6.0], [17.0, 20.0], [-17.0, 20.0], [-21.0, 6.0]]) * 1e-3
        fan = np.array([[0, i, i + 1] for i in range(1, len(section) - 1)])
        prism = trimesh.creation.extrude_triangulation(section, fan, height=0.1)
        prism.vertices = prism.vertices[:, [2, 0, 1]]
        grid, qc = sample_columns(prism)

        def thickness(y):
            return np.clip(20.0 - 3.5 * (np.abs(y) * 1e3 - 17.0), None, 20.0) * 1e-3

        offsets = ((np.arange(CELL_RAYS) + 0.5) / CELL_RAYS - 0.5) * SPACING_M
        for y in (-0.001, 0.019):  # interior cell, and the outline cell centered 2 mm inside the edge
            column = np.flatnonzero(np.all(np.isclose(grid.uv_m, [0.05, y]), axis=1))[0]
            inside = y + offsets[y + offsets < 0.021]
            np.testing.assert_allclose(grid.slack_m[column], thickness(inside).mean(), rtol=1e-6)
            np.testing.assert_allclose(grid.area_m2[column], SPACING_M**2 * len(inside) / CELL_RAYS, rtol=1e-9)
        self.assertGreater(grid.slack_m[column], thickness(0.019) + 5e-4)
        self.assertGreater(qc["partial_cells"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
