# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check explicit engineering shoe-side transforms without private assets."""

import copy
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from projects.digital_shoe import FoundationConfig, MidsoleFoundation, ShoeMaterial, load_artifact
from projects.impedance_instron.orientation import orient_shoe


def _synthetic_artifact():
    """Build a scalene tetrahedron above a small asymmetric column bed."""
    points = [[-0.01, 0.005, 0.0], [0.0, 0.005, 0.001], [-0.01, 0.015, 0.002], [0.0, 0.015, 0.0]]
    neighbors = [[-1, 1, -1, 2], [0, -2, -1, 3], [-1, 3, 0, -2], [2, -1, 1, -1]]
    bed = {
        "anchor_bottom_m": points,
        "rest_length_m": [0.021, 0.023, 0.024, 0.022],
        "area_m2": [0.00010, 0.00012, 0.00011, 0.00009],
        "neighbors": neighbors,
        "spacing_m": 0.01,
    }
    fixture = {key: copy.deepcopy(value) for key, value in bed.items() if key != "anchor_bottom_m"}
    fixture.update(
        carrier_anchor_m=[[x, y, z + 0.03] for x, y, z in points],
        foam_free_top_m=[0.025, 0.026, 0.025, 0.024],
        foam_bottom_m=[0.0, 0.001, 0.002, 0.0],
    )
    return {
        "schema_version": "digital_shoe_1",
        "shoe": {"id": "synthetic_filename_left", "model_scope": "synthetic only"},
        "coordinate_system": {"length_unit": "m", "up_axis": "+Z", "handedness": "right"},
        "constitutive_model": {
            "type": "effective_hyperfoam_maxwell_pasternak_foundation",
            "parameters": asdict(ShoeMaterial(19000.0, 5.1, 0.11, 900.0)),
        },
        "column_bed": bed,
        "visual_meshes": {
            "fullfoot_last": {
                "vertices_m": [[0.0, 0.005, 0.03], [0.027, 0.005, 0.03], [0.0, 0.018, 0.03], [0.0, 0.005, 0.049]],
                "triangles": [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
            }
        },
        "instron_fixtures": {"fullfoot_last": fixture},
        "validation": {"claim_boundary": "synthetic only", "curves": []},
        "provenance": {"generator": "synthetic", "source_files": []},
    }


def _load(data):
    """Load through the public artifact API and discard the temporary file."""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "shoe.json"
        path.write_text(json.dumps(data))
        return load_artifact(path)


def _face_orientation(mesh):
    """Return outward-positive face tests for the closed synthetic tetrahedron."""
    faces = mesh.vertices_m[mesh.triangles]
    normals = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
    return np.einsum("ij,ij->i", normals, faces.mean(axis=1) - mesh.vertices_m.mean(axis=0))


class TestImpedanceOrientation(unittest.TestCase):
    def test_explicit_side_and_copy_preserve_original(self):
        """Reflect the chosen side without inferring it from a left filename."""
        source = _load(_synthetic_artifact())
        before = copy.deepcopy(source)
        target, metadata = orient_shoe(source, "left", source_side="right")
        self.assertEqual(metadata["source_shoe_side"], "right")
        self.assertEqual(metadata["target_shoe_side"], "left")
        self.assertEqual(metadata["side_interpretation"], "explicit_engineering_convention")
        self.assertFalse(metadata["anatomical_side_validated"])
        self.assertTrue(metadata["reflection_applied"])
        self.assertEqual(target.shoe_id, source.shoe_id)
        self.assertEqual(target.material, source.material)
        self.assertEqual(target.validation, source.validation)
        self.assertEqual(source.raw, before.raw)
        np.testing.assert_array_equal(source.column_bed.anchor_bottom_m, before.column_bed.anchor_bottom_m)
        np.testing.assert_array_equal(
            source.visual_mesh("fullfoot_last").triangles, before.visual_mesh("fullfoot_last").triangles
        )
        np.testing.assert_array_equal(
            target.column_bed.anchor_bottom_m, source.column_bed.anchor_bottom_m * [1.0, -1.0, 1.0]
        )
        self.assertEqual(target.raw["provenance"]["impedance_instron_orientation"], metadata)
        reloaded = _load(target.raw)
        np.testing.assert_array_equal(reloaded.column_bed.anchor_bottom_m, target.column_bed.anchor_bottom_m)
        np.testing.assert_array_equal(
            reloaded.visual_mesh("fullfoot_last").triangles, target.visual_mesh("fullfoot_last").triangles
        )
        target.column_bed.rest_length_m[0] = 100.0
        self.assertEqual(source.column_bed.rest_length_m[0], before.column_bed.rest_length_m[0])
        target.raw["validation"]["curves"].append("changed")
        self.assertEqual(source.validation, before.validation)

    def test_reflect_fixture_and_neighbor_directions(self):
        """Reflect fixture anchors and swap Y neighbor slots without reindexing columns."""
        source = _load(_synthetic_artifact())
        target, _ = orient_shoe(source, "left", source_side="right")
        for old, new in ((source.column_bed, target.column_bed), (source.instron_fixture(), target.instron_fixture())):
            np.testing.assert_array_equal(new.neighbors, old.neighbors[:, [0, 1, 3, 2]])
            np.testing.assert_array_equal(new.rest_length_m, old.rest_length_m)
            np.testing.assert_array_equal(new.area_m2, old.area_m2)
            self.assertEqual(new.spacing_m, old.spacing_m)
        old, new = source.instron_fixture(), target.instron_fixture()
        np.testing.assert_array_equal(new.carrier_anchor_m, old.carrier_anchor_m * [1.0, -1.0, 1.0])
        np.testing.assert_array_equal(new.foam_free_top_m, old.foam_free_top_m)
        np.testing.assert_array_equal(new.foam_bottom_m, old.foam_bottom_m)
        points = target.column_bed.anchor_bottom_m
        for i, row in enumerate(target.column_bed.neighbors):
            for slot, neighbor in enumerate(row):
                if neighbor >= 0:
                    direction = points[neighbor, :2] - points[i, :2]
                    expected = [[-0.01, 0], [0.01, 0], [0, -0.01], [0, 0.01]][slot]
                    np.testing.assert_allclose(direction, expected, atol=1e-15)
        reloaded = _load(target.raw)
        np.testing.assert_array_equal(reloaded.instron_fixture().carrier_anchor_m, new.carrier_anchor_m)
        np.testing.assert_array_equal(reloaded.instron_fixture().neighbors, new.neighbors)

    def test_synthetic_chiral_mesh_winding_and_round_trip(self):
        """Keep outward normals on a chiral mesh and recover geometry after two mirrors."""
        source = _load(_synthetic_artifact())
        target, metadata = orient_shoe(source, "left", source_side="right")
        old, new = source.visual_mesh("fullfoot_last"), target.visual_mesh("fullfoot_last")
        np.testing.assert_array_equal(new.vertices_m, old.vertices_m * [1.0, -1.0, 1.0])
        np.testing.assert_array_equal(new.triangles, old.triangles[:, [0, 2, 1]])
        self.assertTrue(np.all(_face_orientation(old) > 0.0))
        self.assertTrue(np.all(_face_orientation(new) > 0.0))
        self.assertEqual(metadata["winding_repairs"], [])
        restored, _ = orient_shoe(target, "right", source_side="left")
        np.testing.assert_array_equal(restored.visual_mesh("fullfoot_last").vertices_m, old.vertices_m)
        np.testing.assert_array_equal(restored.visual_mesh("fullfoot_last").triangles, old.triangles)
        np.testing.assert_array_equal(restored.column_bed.neighbors, source.column_bed.neighbors)

    def test_same_side_and_unknown_inward_mesh(self):
        """Keep unknown input winding rather than treating a mesh name as evidence."""
        data = _synthetic_artifact()
        data["provenance"]["generator"] = "projects.digital_instron_v2.export_digital_shoe"
        data["visual_meshes"]["fullfoot_last"]["triangles"] = [
            [a, c, b] for a, b, c in data["visual_meshes"]["fullfoot_last"]["triangles"]
        ]
        source = _load(data)
        for side in ("right", "left"):
            target, metadata = orient_shoe(source, side, source_side="right")
            self.assertEqual(metadata["winding_repairs"], [])
            self.assertFalse(metadata["audited_last_winding_contract_matched"])
            self.assertTrue(np.all(_face_orientation(target.visual_mesh("fullfoot_last")) < 0.0))
            if side == "right":
                self.assertFalse(metadata["reflection_applied"])
                np.testing.assert_array_equal(
                    target.visual_mesh("fullfoot_last").triangles, source.visual_mesh("fullfoot_last").triangles
                )
                np.testing.assert_array_equal(target.column_bed.neighbors, source.column_bed.neighbors)

    def test_exact_audit_contract_repairs_without_reflection(self):
        """Require both pinned geometry and provenance before repairing known winding."""
        data = _synthetic_artifact()
        mesh = data["visual_meshes"]["fullfoot_last"]
        mesh["triangles"] = [[a, c, b] for a, b, c in mesh["triangles"]]
        data["provenance"] = {
            "generator": "projects.digital_instron_v2.export_digital_shoe",
            "source_files": [{"role": "test_source", "sha256": "a" * 64}],
        }
        source = _load(data)
        _, unknown = orient_shoe(source, "right", source_side="right")
        digest = unknown["mesh_winding"]["fullfoot_last"]["input_geometry_sha256"]
        # A synthetic audit exercises the same strict gate without redistributing footwear.
        with (
            patch("projects.impedance_instron.orientation._AUDITED_LAST_MESH_SHA256", digest),
            patch("projects.impedance_instron.orientation._AUDITED_SOURCE_HASHES", {"test_source": "a" * 64}),
        ):
            for side in ("right", "left"):
                target, metadata = orient_shoe(source, side, source_side="right")
                self.assertTrue(metadata["audited_last_winding_contract_matched"])
                self.assertEqual(metadata["winding_repairs"], ["fullfoot_last"])
                self.assertTrue(np.all(_face_orientation(target.visual_mesh("fullfoot_last")) > 0.0))
            target, _ = orient_shoe(source, "right", source_side="right")
            again, metadata = orient_shoe(target, "right", source_side="right")
            self.assertEqual(metadata["winding_repairs"], [])
            np.testing.assert_array_equal(
                again.visual_mesh("fullfoot_last").triangles, target.visual_mesh("fullfoot_last").triangles
            )
            wrong_source = copy.deepcopy(source)
            wrong_source.provenance["source_files"][0]["sha256"] = "b" * 64
            _, metadata = orient_shoe(wrong_source, "right", source_side="right")
            self.assertFalse(metadata["audited_last_winding_contract_matched"])
            changed = copy.deepcopy(source)
            changed.visual_mesh("fullfoot_last").vertices_m[0, 0] += 1e-9
            _, metadata = orient_shoe(changed, "right", source_side="right")
            self.assertFalse(metadata["audited_last_winding_contract_matched"])

    def test_reject_implicit_side_and_unsupported_frame(self):
        """Reject absent side interpretation and incompatible coordinate declarations."""
        source = _load(_synthetic_artifact())
        with self.assertRaises(TypeError):
            orient_shoe(source)
        for side in ("auto", "LEFT", "", None):
            with self.assertRaises(ValueError):
                orient_shoe(source, side, source_side="right")
            with self.assertRaises(ValueError):
                orient_shoe(source, "left", source_side=side)
        source.raw["coordinate_system"]["handedness"] = "left"
        with self.assertRaisesRegex(ValueError, "right-handed"):
            orient_shoe(source, "left", source_side="right")

    def test_native_foundation_sagittal_mirror_invariance(self):
        """Keep compression, Fz and COP X under reflection at nonzero sagittal pitch."""
        source = _load(_synthetic_artifact())
        target, _ = orient_shoe(source, "left", source_side="right")
        diagnostics = []
        compressions = []
        wrenches = []
        with wp.ScopedDevice("cpu"):
            for shoe in (source, target):
                builder = newton.ModelBuilder()
                carrier = builder.add_body(
                    xform=wp.transform(
                        wp.vec3(0.018, 0.0, -0.006), wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.22)
                    ),
                    mass=2.0,
                    inertia=wp.mat33(np.eye(3) * 0.025),
                )
                model = builder.finalize(device="cpu")
                state = model.state()
                state.body_qd.assign(np.array([[0.02, 0.0, -0.08, 0.0, 0.6, 0.0]], dtype=np.float32))
                bed = shoe.column_bed
                foundation = MidsoleFoundation(
                    bed.anchor_bottom_m,
                    np.zeros(len(bed.rest_length_m)),
                    bed.rest_length_m,
                    bed.area_m2,
                    bed.neighbors,
                    bed.spacing_m,
                    shoe.material,
                    carrier,
                    model.body_com,
                    FoundationConfig(normal_damping=0.5),
                    device="cpu",
                )
                for _ in range(3):
                    foundation.apply(state, 0.001, clear_body_force=True)
                diagnostics.append(foundation.diagnostics())
                compressions.append(foundation.compression.numpy())
                wrenches.append(state.body_f.numpy()[carrier])
        first, second = diagnostics
        self.assertGreater(first["normal_force_n"], 1.0)
        self.assertGreater(abs(first["cop_y_m"]), 0.001)
        self.assertGreater(float(np.ptp(compressions[0])), 0.001)
        np.testing.assert_allclose(compressions[0], compressions[1], atol=1e-8, rtol=1e-6)
        for name in ("normal_force_n", "cop_x_m", "active_columns"):
            self.assertAlmostEqual(first[name], second[name], delta=1e-6 * max(1.0, abs(first[name])))
        self.assertAlmostEqual(first["cop_y_m"], -second["cop_y_m"], delta=1e-8)
        np.testing.assert_allclose(wrenches[1], wrenches[0] * [1, -1, 1, -1, 1, -1], atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
