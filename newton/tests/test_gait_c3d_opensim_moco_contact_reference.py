# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the pinned S001 OpenSim 3-D-walking contact reference."""

from __future__ import annotations

import importlib
import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from projects.gait_c3d import opensim_moco_contact_reference as reference

_HAS_OFFICIAL = importlib.util.find_spec("opensim") is not None
_HAS_S001_MODEL = reference._DEFAULT_RRA_MODEL.is_file()


class TestGaitC3DOpenSimMocoContactReference(unittest.TestCase):
    """Check topology, generated formats, grouping, and wrench math."""

    @staticmethod
    def _write_external_loads(directory: Path) -> Path:
        data = directory / "measured.mot"
        data.write_text(
            "synthetic\nendheader\ntime\tground_force_l_vy\tground_force_r_vy\n0\t100\t0\n",
            encoding="utf-8",
        )
        document = ET.Element("OpenSimDocument", {"Version": "40000"})
        loads = ET.SubElement(document, "ExternalLoads", {"name": "measured_reference"})
        objects = ET.SubElement(loads, "objects")
        for side, suffix in (("left", "l"), ("right", "r")):
            force = ET.SubElement(objects, "ExternalForce", {"name": side})
            ET.SubElement(force, "applied_to_body").text = f"calcn_{suffix}"
            ET.SubElement(force, "force_expressed_in_body").text = "ground"
            ET.SubElement(force, "point_expressed_in_body").text = "ground"
            ET.SubElement(force, "force_identifier").text = f"ground_force_{suffix}_v"
            ET.SubElement(force, "point_identifier").text = f"ground_force_{suffix}_p"
            ET.SubElement(force, "torque_identifier").text = f"ground_torque_{suffix}_"
        ET.SubElement(loads, "datafile").text = data.name
        path = directory / "measured.xml"
        ET.indent(document, space="  ")
        path.write_bytes(ET.tostring(document, encoding="utf-8", xml_declaration=True) + b"\n")
        return path

    def test_pinned_topology_is_six_per_foot_mirrored_and_body_split(self):
        spheres = reference.sphere_specs()
        self.assertEqual(len(spheres), 12)
        for side in reference._SIDE_ORDER:
            selected = [sphere for sphere in spheres if sphere.side == side]
            self.assertEqual([sphere.role for sphere in selected], list(reference._ROLE_ORDER))
            self.assertEqual([sphere.body for sphere in selected[:4]], [f"calcn_{reference._SUFFIX[side]}"] * 4)
            self.assertEqual([sphere.body for sphere in selected[4:]], [f"toes_{reference._SUFFIX[side]}"] * 2)
            self.assertEqual({sphere.radius_m for sphere in selected}, {0.035})
            self.assertEqual({sphere.location_m[1] for sphere in selected}, {reference.S001_ALIGNMENT.offset_m})
        left = {sphere.role: sphere for sphere in spheres if sphere.side == "left"}
        right = {sphere.role: sphere for sphere in spheres if sphere.side == "right"}
        for role in reference._ROLE_ORDER:
            self.assertEqual(left[role].location_m[0], right[role].location_m[0])
            self.assertEqual(left[role].location_m[1], right[role].location_m[1])
            self.assertEqual(left[role].location_m[2], -right[role].location_m[2])

    def test_alignment_is_bounded_archived_and_derivation_is_reproducible(self):
        self.assertNotEqual(reference.S001_ALIGNMENT.offset_m, reference._OFFICIAL_EXAMPLE_OFFSET_M)
        self.assertLess(reference.S001_ALIGNMENT.offset_m, 0.0)
        self.assertLessEqual(reference._ALIGNMENT_BOUNDS_M[0], reference.S001_ALIGNMENT.offset_m)
        self.assertLessEqual(reference.S001_ALIGNMENT.offset_m, reference._ALIGNMENT_BOUNDS_M[1])
        center = np.ones((2, 2, 6)) * 0.045
        gain = np.ones_like(center)
        center[..., 0] = 0.055
        center[..., 1] = 0.04  # lowest surface clearance is 5 mm.
        alignment = reference.derive_vertical_alignment(center, gain, np.array([[True, False], [True, True]]))
        self.assertAlmostEqual(alignment.offset_m, -0.005)
        self.assertEqual(alignment.stance_observation_count, 3)
        self.assertAlmostEqual(alignment.rms_clearance_after_m, 0.0, places=14)
        clipped = reference.derive_vertical_alignment(
            np.ones((1, 2, 6)) * 0.2,
            np.ones((1, 2, 6)),
            np.ones((1, 2), dtype=bool),
        )
        self.assertEqual(clipped.offset_m, reference._ALIGNMENT_BOUNDS_M[0])
        self.assertLess(clipped.unconstrained_offset_m, clipped.offset_m)

    def test_geometry_and_force_xml_are_deterministic_official_sets(self):
        geometry = reference.xml_bytes(reference.build_contact_geometry_xml())
        forces = reference.xml_bytes(reference.build_force_xml())
        self.assertEqual(geometry, reference.xml_bytes(reference.build_contact_geometry_xml()))
        self.assertEqual(forces, reference.xml_bytes(reference.build_force_xml()))
        geometry_root = ET.fromstring(geometry)
        geometry_objects = geometry_root.find("./ContactGeometrySet/objects")
        self.assertIsNotNone(geometry_objects)
        assert geometry_objects is not None
        self.assertEqual(geometry_objects[0].tag, "ContactHalfSpace")
        self.assertEqual(geometry_objects[0].get("name"), "floor")
        self.assertEqual(len(geometry_objects.findall("ContactSphere")), 12)
        by_name = {item.get("name"): item for item in geometry_objects.findall("ContactSphere")}
        self.assertEqual(by_name["medialMidfoot_l"].findtext("socket_frame"), "/bodyset/calcn_l")
        self.assertEqual(by_name["lateralToe_r"].findtext("socket_frame"), "/bodyset/toes_r")
        left_z = float(by_name["lateralRearfoot_l"].findtext("location", "").split()[2])
        right_z = float(by_name["lateralRearfoot_r"].findtext("location", "").split()[2])
        self.assertEqual(left_z, -right_z)

        force_root = ET.fromstring(forces)
        force_objects = force_root.find("./ForceSet/objects")
        self.assertIsNotNone(force_objects)
        assert force_objects is not None
        self.assertEqual(len(force_objects.findall("SmoothSphereHalfSpaceForce")), 12)
        first = force_objects[0]
        self.assertEqual(first.get("name"), "contactHeel_l")
        self.assertEqual(first.findtext("socket_sphere"), "/contactgeometryset/heel_l")
        self.assertEqual(first.findtext("socket_half_space"), "/contactgeometryset/floor")
        self.assertEqual(float(first.findtext("stiffness", "nan")), 1.0e6)
        # The pinned XML relies on this OpenSim default; Newton states it explicitly.
        self.assertIsNone(first.find("constant_contact_force"))

    def test_newton_spec_matches_xml_and_has_four_contact_bodies(self):
        spec = reference.newton_augmentation_spec()
        self.assertEqual(len(spec.contact_geometry), 13)
        self.assertEqual(len(spec.contact_forces), 12)
        self.assertEqual(spec.contact_geometry[0].type, "ContactHalfSpace")
        spheres = {item.name: item for item in spec.contact_geometry[1:]}
        self.assertEqual(spheres["medialToe_l"].body, "toes_l")
        self.assertEqual(spheres["medialMidfoot_r"].body, "calcn_r")
        self.assertEqual({item.body for item in spheres.values()}, set(reference._BODY_ORDER))
        force = spec.contact_forces[0]
        self.assertEqual(force.type, "SmoothSphereHalfSpaceForce")
        self.assertEqual(force.geometries, [force.sphere, "floor"])
        self.assertEqual(force.params["constant_contact_force"], 1.0e-5)

    def test_moco_groups_use_toe_alternatives_and_reference_only_loads(self):
        groups = reference.moco_contact_groups()
        self.assertEqual([group.side for group in groups], ["left", "right"])
        for group in groups:
            suffix = reference._SUFFIX[group.side]
            self.assertEqual(len(group.contact_force_paths), 6)
            self.assertEqual(group.external_force_name, group.side)
            self.assertEqual(group.applied_to_body, f"calcn_{suffix}")
            self.assertEqual(group.alternative_frame_paths, (f"/bodyset/toes_{suffix}",))
            self.assertTrue(all(path.startswith("/contact") for path in group.contact_force_paths))
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            external = self._write_external_loads(directory)
            self.assertEqual(reference.validate_external_loads_reference(external), directory / "measured.mot")
            clean_model = directory / "clean.osim"
            clean_model.write_text("<OpenSimDocument><Model /></OpenSimDocument>\n", encoding="utf-8")
            reference.assert_model_has_no_external_loads(clean_model)
            loaded_model = directory / "loaded.osim"
            loaded_model.write_text(
                "<OpenSimDocument><Model><ForceSet><objects><ExternalForce name='measured' />"
                "</objects></ForceSet></Model></OpenSimDocument>\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "must not contain"):
                reference.assert_model_has_no_external_loads(loaded_model)

    def test_aggregation_translates_calcn_and_toe_moments_to_ground_origin(self):
        names = ("toes_r", "calcn_l", "calcn_r", "toes_l")
        values = np.zeros((1, 4, 9))
        # Left calcn: F=(1,2,3), P=(2,0,0), T=(0,1,0).
        values[0, names.index("calcn_l")] = [1, 2, 3, 2, 0, 0, 0, 1, 0]
        # Left toes: F=(4,5,6), P=(0,0,3), T=(1,0,2).
        values[0, names.index("toes_l")] = [4, 5, 6, 0, 0, 3, 1, 0, 2]
        aggregate = reference.aggregate_body_wrenches(names, values)
        np.testing.assert_array_equal(aggregate[0, 0, :3], [5, 7, 9])
        expected_moment = (
            np.array([0, 1, 0]) + np.cross([2, 0, 0], [1, 2, 3]) + np.array([1, 0, 2]) + np.cross([0, 0, 3], [4, 5, 6])
        )
        np.testing.assert_array_equal(aggregate[0, 0, 3:6], [0, 0, 0])
        np.testing.assert_array_equal(aggregate[0, 0, 6:9], expected_moment)
        with self.assertRaisesRegex(ValueError, "exactly"):
            reference.aggregate_body_wrenches(names[:-1], values)

    def test_cop_and_free_moment_are_independent_from_force_tracking(self):
        values = np.zeros((1, 2, 9))
        force = np.array([10.0, 100.0, -5.0])
        cop = np.array([0.2, 0.0, -0.1])
        free = 3.0
        moment = np.cross(cop, force) + np.array([0.0, free, 0.0])
        values[0, 0, :3] = force
        values[0, 0, 6:9] = moment
        calculated_cop, calculated_free = reference.cop_and_free_moment(values)
        np.testing.assert_allclose(calculated_cop[0, 0], cop)
        self.assertAlmostEqual(calculated_free[0, 0], free)
        self.assertTrue(np.all(np.isnan(calculated_cop[0, 1])))
        validation = reference.independent_load_validation(values, values.copy())
        self.assertEqual(validation["loaded_comparison_count"], 1)
        self.assertEqual(validation["cop_rms_m"], 0.0)
        self.assertEqual(validation["free_moment_rms_Nm"], 0.0)

    def test_provenance_and_written_files_pin_sources_hashes_frame_and_units(self):
        provenance = reference.provenance()
        self.assertEqual(provenance["pinned_upstream"]["commit"], reference._PINNED_COMMIT)
        self.assertEqual(provenance["frame"], "opensim_x_forward_y_up_z_right")
        self.assertEqual(provenance["units"]["moment"], "N*m")
        self.assertFalse(provenance["contact_tracking_contract"]["external_loads_added_to_predictive_model"])
        self.assertEqual(len(provenance["pinned_upstream"]["files"]), 6)
        for metadata in provenance["pinned_upstream"]["files"].values():
            self.assertEqual(len(metadata["sha256"]), 64)
            self.assertIn(reference._PINNED_COMMIT, metadata["url"])
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            external = self._write_external_loads(directory)
            output = directory / "output"
            first = reference.write_reference_files(output, external_loads_path=external)
            first_bytes = {path.name: path.read_bytes() for path in output.iterdir()}
            second = reference.write_reference_files(output, external_loads_path=external)
            second_bytes = {path.name: path.read_bytes() for path in output.iterdir()}
            self.assertEqual(first, second)
            self.assertEqual(first_bytes, second_bytes)
            manifest = json.loads(first.read_text(encoding="utf-8"))
            groups = json.loads((output / "S001_MocoContactTrackingGoal_groups.json").read_text(encoding="utf-8"))
            self.assertFalse(groups["model_added_external_loads"])
            self.assertEqual(manifest["measured_reference"]["data_path"], str(directory / "measured.mot"))

    @unittest.skipUnless(_HAS_OFFICIAL and _HAS_S001_MODEL, "official OpenSim or accepted S001 RRA model unavailable")
    def test_optional_official_xml_loads_with_root_force_paths_and_no_external_loads(self):
        opensim = importlib.import_module("opensim")
        model = reference._load_official_augmented_model(opensim, reference._DEFAULT_RRA_MODEL)
        state = model.initSystem()
        model.realizeDynamics(state)
        reference.assert_model_has_no_external_loads(model)
        self.assertEqual(model.getContactGeometrySet().getSize(), 13)
        for sphere in reference.sphere_specs():
            component = model.getComponent(f"/{sphere.force_name}")
            force = opensim.SmoothSphereHalfSpaceForce.safeDownCast(component)
            self.assertIsNotNone(force)
            labels = force.getRecordLabels()
            self.assertEqual(
                [labels.get(index) for index in range(labels.getSize())],
                list(reference.expected_record_labels(sphere.force_name)),
            )


if __name__ == "__main__":
    unittest.main()
