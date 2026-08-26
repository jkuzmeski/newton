# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test scaled VTP conversion into neutral subject MJCF assets."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import newton
from projects.gait_c3d.native_model import SimpleGaitConfig
from projects.gait_c3d.subject_mjcf import write_subject_mjcf
from projects.gait_c3d.vtp_adapter import (
    compile_scaled_vtp_visuals,
    read_scaled_display_geometry,
    read_vtp,
    simple_config_from_scaled_gait2354,
)

_VTP = """<?xml version="1.0"?>
<VTKFile type="PolyData" version="0.1">
  <PolyData><Piece NumberOfPoints="4" NumberOfPolys="1">
    <Points><DataArray type="Float32" NumberOfComponents="3" format="ascii">
      0 0 0  0 1 0  1 1 0  1 0 0
    </DataArray></Points>
    <Polys>
      <DataArray type="Int32" Name="connectivity" format="ascii">0 1 2 3</DataArray>
      <DataArray type="Int32" Name="offsets" format="ascii">4</DataArray>
    </Polys>
  </Piece></PolyData>
</VTKFile>
"""
_SOURCE_BODIES = ("pelvis", "torso", "femur_l", "femur_r", "tibia_l", "tibia_r")
_MASS_CENTERS = {
    "torso": "0 0.30 0",
    "femur_l": "0 -0.20 0",
    "femur_r": "0 -0.20 0",
    "tibia_l": "0 -0.18 0",
    "tibia_r": "0 -0.18 0",
}


def _scaled_osim() -> str:
    bodies = []
    for body in _SOURCE_BODIES:
        bodies.append(
            f"""<Body name="{body}">
              <mass_center>{_MASS_CENTERS.get(body, "0 0 0")}</mass_center>
              <VisibleObject><scale_factors>1 1 1</scale_factors><transform>0 0 0 0 0 0</transform>
                <GeometrySet><objects><DisplayGeometry>
                  <geometry_file>mesh.vtp</geometry_file>
                  <scale_factors>1 1 1</scale_factors><transform>0 0 0 0 0 0</transform>
                </DisplayGeometry></objects></GeometrySet>
              </VisibleObject>
            </Body>"""
        )
    return (
        '<OpenSimDocument Version="20302"><Model><BodySet><objects>'
        + "".join(bodies)
        + "</objects></BodySet></Model></OpenSimDocument>"
    )


def _modern_scaled_osim() -> str:
    masses = {
        "pelvis": (13.0, "0 0 0"),
        "torso": (37.0, "0 0.30 0"),
        "femur_l": (10.0, "0 -0.20 0"),
        "femur_r": (10.0, "0 -0.20 0"),
        "tibia_l": (4.0, "0 -0.18 0"),
        "tibia_r": (4.0, "0 -0.18 0"),
        "talus_l": (0.1, "0 0 0"),
        "talus_r": (0.1, "0 0 0"),
        "calcn_l": (1.35, "0 0 0"),
        "calcn_r": (1.35, "0 0 0"),
        "toes_l": (0.25, "0 0 0"),
        "toes_r": (0.25, "0 0 0"),
    }
    mapped = {"pelvis", "torso", "femur_l", "femur_r", "tibia_l", "tibia_r"}
    bodies = []
    for name, (mass, com) in masses.items():
        geometry = (
            f"<attached_geometry><Mesh name='{name}_mesh'><socket_frame>..</socket_frame>"
            "<scale_factors>1 1 1</scale_factors><mesh_file>mesh.vtp</mesh_file></Mesh></attached_geometry>"
            if name in mapped
            else ""
        )
        bodies.append(f"<Body name='{name}'><mass>{mass}</mass><mass_center>{com}</mass_center>{geometry}</Body>")

    def frames(parent_name: str, parent_body: str, parent_xyz: str, child_name: str, child_body: str) -> str:
        return (
            f"<frames><PhysicalOffsetFrame name='{parent_name}'><socket_parent>/bodyset/{parent_body}</socket_parent>"
            f"<translation>{parent_xyz}</translation><orientation>0 0 0</orientation></PhysicalOffsetFrame>"
            f"<PhysicalOffsetFrame name='{child_name}'><socket_parent>/bodyset/{child_body}</socket_parent>"
            "<translation>0 0 0</translation><orientation>0 0 0</orientation></PhysicalOffsetFrame></frames>"
        )

    joints = []
    for side, lateral in (("l", -0.076), ("r", 0.076)):
        joints.append(
            f"<CustomJoint name='hip_{side}'><socket_parent_frame>pelvis_{side}</socket_parent_frame>"
            f"<socket_child_frame>femur_{side}_frame</socket_child_frame>"
            + frames(f"pelvis_{side}", "pelvis", f"0 -0.06 {lateral}", f"femur_{side}_frame", f"femur_{side}")
            + "<SpatialTransform /></CustomJoint>"
        )
        joints.append(
            f"<CustomJoint name='knee_{side}'><socket_parent_frame>femur_{side}_knee</socket_parent_frame>"
            f"<socket_child_frame>tibia_{side}_frame</socket_child_frame>"
            + frames(f"femur_{side}_knee", f"femur_{side}", "0 0 0", f"tibia_{side}_frame", f"tibia_{side}")
            + "<SpatialTransform><TransformAxis name='translation2'><axis>0 1 0</axis>"
            "<MultiplierFunction><function><SimmSpline><x>-1 1</x><y>-0.45 -0.45</y></SimmSpline></function>"
            "<scale>1</scale></MultiplierFunction></TransformAxis></SpatialTransform></CustomJoint>"
        )
        joints.append(
            f"<CustomJoint name='ankle_{side}'><socket_parent_frame>tibia_{side}_ankle</socket_parent_frame>"
            f"<socket_child_frame>talus_{side}_frame</socket_child_frame>"
            + frames(f"tibia_{side}_ankle", f"tibia_{side}", "0 -0.4 0", f"talus_{side}_frame", f"talus_{side}")
            + "<SpatialTransform /></CustomJoint>"
        )
    joints.append(
        "<CustomJoint name='back'><socket_parent_frame>pelvis_back</socket_parent_frame>"
        "<socket_child_frame>torso_frame</socket_child_frame>"
        + frames("pelvis_back", "pelvis", "0 0.08 0", "torso_frame", "torso")
        + "<SpatialTransform /></CustomJoint>"
    )
    return (
        '<OpenSimDocument Version="40600"><Model><BodySet><objects>'
        + "".join(bodies)
        + "</objects></BodySet><JointSet><objects>"
        + "".join(joints)
        + "</objects></JointSet></Model></OpenSimDocument>"
    )


class TestGaitVTPAdapter(unittest.TestCase):
    """Test deterministic visual-only geometry conversion."""

    def test_reads_and_triangulates_ascii_vtp(self):
        """Triangulate an ASCII VTP polygon deterministically."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mesh.vtp"
            path.write_text(_VTP)
            vertices, triangles = read_vtp(path)
        self.assertEqual(vertices.shape, (4, 3))
        np.testing.assert_array_equal(triangles, ((0, 1, 2), (0, 2, 3)))

    def test_compiles_modern_opensim_scale_tool_output(self):
        """Read modern Mesh and PhysicalOffsetFrame layout from official ScaleTool."""
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            (source / "mesh.vtp").write_text(_VTP)
            model_path = source / "scaled.osim"
            model_path.write_text(_modern_scaled_osim())
            config = simple_config_from_scaled_gait2354(model_path, body_height=1.70)
            bundle = compile_scaled_vtp_visuals(
                model_path,
                source,
                Path(directory) / "bundle",
                config,
            )
        self.assertEqual(len(bundle.meshes), 6)
        self.assertAlmostEqual(config.hip_half_width, 0.076)
        self.assertAlmostEqual(config.pelvis_hip_drop, 0.06)
        self.assertAlmostEqual(config.thigh_length, 0.45)
        self.assertAlmostEqual(config.shank_length, 0.40)
        self.assertAlmostEqual(config.torso_center_offset, 0.38)

    def test_rejects_two_nonidentity_legacy_scale_levels(self):
        """Reject stale models that would apply subject geometry scaling twice."""
        stale = (
            _scaled_osim()
            .replace(
                "<scale_factors>1 1 1</scale_factors><transform>",
                "<scale_factors>0.9 0.9 0.9</scale_factors><transform>",
                1,
            )
            .replace(
                "<scale_factors>1 1 1</scale_factors><transform>",
                "<scale_factors>0.8 0.8 0.8</scale_factors><transform>",
                1,
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stale.osim"
            path.write_text(stale)
            with self.assertRaisesRegex(ValueError, "two nonidentity legacy scale levels"):
                read_scaled_display_geometry(path)

    def test_compiles_scaled_visuals_and_loads_one_call_mjcf(self):
        """Bake body-local meshes and load the saved subject with add_mjcf."""
        previous_layout = newton.use_coord_layout_targets
        newton.use_coord_layout_targets = True
        try:
            with tempfile.TemporaryDirectory() as directory:
                source = Path(directory) / "source"
                source.mkdir()
                (source / "mesh.vtp").write_text(_VTP)
                model_path = source / "scaled.osim"
                model_path.write_text(_scaled_osim())
                bundle = compile_scaled_vtp_visuals(model_path, source, Path(directory) / "bundle", SimpleGaitConfig())
                mjcf = write_subject_mjcf(
                    SimpleGaitConfig(),
                    bundle.root / "subject.xml",
                    visual_meshes=bundle.meshes,
                    include_fallback_geometry=False,
                )
                builder = newton.ModelBuilder()
                builder.add_mjcf(str(mjcf))
                model = builder.finalize(device="cpu")
                manifest = json.loads(bundle.manifest_path.read_text())
                first_obj = (bundle.root / bundle.meshes[0].file).read_text().splitlines()
        finally:
            newton.use_coord_layout_targets = previous_layout
        self.assertEqual(len(bundle.meshes), 6)
        self.assertEqual(len(manifest["meshes"]), 6)
        femur_record = next(record for record in manifest["meshes"] if record["mesh"]["body"] == "femur_left")
        np.testing.assert_allclose(femur_record["source"]["source_proximal_newton"], (0.0, 0.0, 0.2))
        np.testing.assert_allclose(femur_record["source"]["target_proximal_newton"], (0.0, 0.0, 0.225))
        self.assertEqual(model.body_count, 8)
        self.assertEqual(model.joint_dof_count, 16)
        self.assertEqual(model.shape_count, 16)
        connector = next(
            index for index, label in enumerate(model.shape_label) if label.endswith("/geometry_abdomen_connector")
        )
        shape_types = model.shape_type.numpy()
        shape_flags = model.shape_flags.numpy()
        self.assertEqual(shape_types[connector], newton.GeoType.BOX)
        self.assertFalse(shape_flags[connector] & newton.ShapeFlags.COLLIDE_SHAPES)
        mesh_indices = np.flatnonzero(shape_types == newton.GeoType.MESH)
        self.assertEqual(len(mesh_indices), 6)
        for shape in mesh_indices:
            self.assertFalse(shape_flags[shape] & newton.ShapeFlags.COLLIDE_SHAPES)
        self.assertEqual(first_obj[1], "v 0 0 1")


if __name__ == "__main__":
    unittest.main()
