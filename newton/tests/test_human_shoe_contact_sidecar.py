# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the human shoe contact sidecar runtime."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import newton
from newton import opensim
from newton.opensim import OsimImportResult
from projects.human_shoe.adapter import resolve_attachment
from projects.human_shoe.contact_sidecar import (
    ContactGeometrySidecarContract,
    HumanShoeContactSidecarContract,
    inject_contact_sidecar,
    load_contact_sidecar,
    write_contact_augmented_osim,
)
from projects.human_shoe.contracts import FootShoeAttachmentContract, load_experiment


class TestHumanShoeContactSidecar(unittest.TestCase):
    """Verify sidecar loading, injection, and write/reparse behavior."""

    def test_loads_valid_direct_and_marker_spheres(self):
        """Load direct and marker-seeded sphere contacts from JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sidecar.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "human_shoe_contact_sidecar_1",
                        "source_model_path": "model.osim",
                        "generated_model_path": "out.osim",
                        "generated_model_name": "generated",
                        "contacts": [
                            {
                                "name": "heel",
                                "type": "ContactSphere",
                                "body_name": "calcn_r",
                                "location_m": [0.0, -0.1, 0.0],
                                "support_offset_m": [0.0, 0.0, 0.0],
                                "orientation_deg": [0.0, 0.0, 0.0],
                                "radius_m": 0.02,
                            },
                            {
                                "name": "toe",
                                "type": "ContactSphere",
                                "body_name": "calcn_r",
                                "support_marker_name": "R.Toe",
                                "support_offset_m": [0.0, 0.0, 0.0],
                                "orientation_deg": [0.0, 0.0, 0.0],
                                "radius_m": 0.02,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            contract = load_contact_sidecar(path)

            self.assertEqual(contract.schema_version, "human_shoe_contact_sidecar_1")
            self.assertEqual(contract.generated_model_name, "generated")
            self.assertEqual(contract.contacts[0].name, "heel")
            self.assertEqual(contract.contacts[1].support_marker_name, "R.Toe")

    def test_checked_in_sidecar_generates_baseline_contacts(self):
        """Generate the baseline model with three direct shoe-top support contacts."""
        sidecar_path = Path("experiments/human_shoe/gait2354_subject01_contacts.json")
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "derived.osim"
            write_contact_augmented_osim(sidecar_path, output_path=output_path)
            checked_in = Path("experiments/human_shoe/generated/gait2354_subject01_with_shoe_contacts.osim")
            self.assertEqual(output_path.read_text(), checked_in.read_text())
            model = opensim.parse_osim(output_path)

        contacts = [contact for contact in model.contact_geometry if contact.body == "calcn_r"]
        self.assertEqual(
            [contact.name for contact in contacts],
            ["hs_calcn_r_heel", "hs_calcn_r_forefoot_lat", "hs_calcn_r_forefoot_med"],
        )
        np.testing.assert_allclose(
            [
                np.asarray(contact.location, dtype=np.float64) - np.array([0.0, float(contact.radius), 0.0])
                for contact in contacts
            ],
            [
                [0.03745469, -0.01977471, 0.00982458],
                [0.19745469, -0.01333833, 0.02482458],
                [0.19745469, -0.01414376, -0.00517542],
            ],
        )
        self.assertEqual(model.version, 40000)
        self.assertEqual(model.contact_forces, [])

        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        import_result = opensim.add_osim(builder, model, parse_muscles=False, parse_contacts=True)
        experiment = load_experiment("experiments/human_shoe/baseline_gait2354.json")
        resolved = resolve_attachment(import_result, experiment.attachment)
        self.assertEqual(builder.shape_count, 3)
        self.assertEqual(resolved.foot_body_index, import_result.body_index["calcn_r"])
        np.testing.assert_allclose(
            resolved.shoe_to_foot[:3, 3],
            [0.13274929, -0.01622024, 0.010912963333333333],
            atol=1.0e-12,
            rtol=0.0,
        )

    def test_rejects_unknown_keys_duplicate_names_and_forces(self):
        """Reject strict schema violations and duplicate contact names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sidecar.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "human_shoe_contact_sidecar_1",
                        "source_model_path": "model.osim",
                        "generated_model_path": "out.osim",
                        "contacts": [
                            {
                                "name": "heel",
                                "type": "ContactSphere",
                                "body_name": "calcn_r",
                                "location_m": [0.0, -0.1, 0.0],
                                "support_offset_m": [0.0, 0.0, 0.0],
                                "orientation_deg": [0.0, 0.0, 0.0],
                                "radius_m": 0.02,
                            },
                            {
                                "name": "heel",
                                "type": "ContactSphere",
                                "body_name": "calcn_r",
                                "location_m": [0.0, -0.1, 0.0],
                                "support_offset_m": [0.0, 0.0, 0.0],
                                "orientation_deg": [0.0, 0.0, 0.0],
                                "radius_m": 0.02,
                            },
                        ],
                        "contact_forces": [],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "unknown fields"):
                load_contact_sidecar(path)

        with self.assertRaisesRegex(ValueError, "contacts must have unique names"):
            HumanShoeContactSidecarContract(
                schema_version="human_shoe_contact_sidecar_1",
                source_model_path="model.osim",
                generated_model_path="out.osim",
                generated_model_name=None,
                contacts=(
                    ContactGeometrySidecarContract(
                        name="heel",
                        type="ContactSphere",
                        body_name="calcn_r",
                        location_m=np.array([0.0, -0.1, 0.0]),
                        support_marker_name=None,
                        support_offset_m=np.array([0.0, 0.0, 0.0]),
                        orientation_deg=np.array([0.0, 0.0, 0.0]),
                        radius_m=0.02,
                        mesh_file=None,
                    ),
                    ContactGeometrySidecarContract(
                        name="heel",
                        type="ContactSphere",
                        body_name="calcn_r",
                        location_m=np.array([0.0, -0.1, 0.0]),
                        support_marker_name=None,
                        support_offset_m=np.array([0.0, 0.0, 0.0]),
                        orientation_deg=np.array([0.0, 0.0, 0.0]),
                        radius_m=0.02,
                        mesh_file=None,
                    ),
                ),
            )

    def test_rejects_bad_body_and_marker_mismatch(self):
        """Reject unknown bodies and marker/body mismatches during injection."""
        model = opensim.OsimModel(bodies=[opensim.OsimBody(name="calcn_r")], markers=[])
        sidecar = HumanShoeContactSidecarContract(
            schema_version="human_shoe_contact_sidecar_1",
            source_model_path="model.osim",
            generated_model_path="out.osim",
            generated_model_name=None,
            contacts=(
                ContactGeometrySidecarContract(
                    name="heel",
                    type="ContactSphere",
                    body_name="missing",
                    location_m=np.array([0.0, -0.1, 0.0]),
                    support_marker_name=None,
                    support_offset_m=np.array([0.0, 0.0, 0.0]),
                    orientation_deg=np.array([0.0, 0.0, 0.0]),
                    radius_m=0.02,
                    mesh_file=None,
                ),
            ),
        )
        with self.assertRaisesRegex(KeyError, "body 'missing' not found"):
            inject_contact_sidecar(model, sidecar)

        model = opensim.OsimModel(
            bodies=[opensim.OsimBody(name="calcn_r")],
            markers=[opensim.OsimMarker(name="R.Toe", body="tibia_r", location=(0.0, 0.0, 0.0))],
        )
        sidecar = HumanShoeContactSidecarContract(
            schema_version="human_shoe_contact_sidecar_1",
            source_model_path="model.osim",
            generated_model_path="out.osim",
            generated_model_name=None,
            contacts=(
                ContactGeometrySidecarContract(
                    name="toe",
                    type="ContactSphere",
                    body_name="calcn_r",
                    location_m=None,
                    support_marker_name="R.Toe",
                    support_offset_m=np.array([0.0, 0.0, 0.0]),
                    orientation_deg=np.array([0.0, 0.0, 0.0]),
                    radius_m=0.02,
                    mesh_file=None,
                ),
            ),
        )
        with self.assertRaisesRegex(ValueError, "attached to body 'tibia_r'"):
            inject_contact_sidecar(model, sidecar)

    def test_collision_and_replace_idempotence(self):
        """Reject collisions by default and replace same-name contacts deterministically."""
        model = opensim.OsimModel(
            bodies=[opensim.OsimBody(name="calcn_r")],
            contact_geometry=[
                opensim.OsimContactGeometry(name="heel", type="ContactSphere", body="calcn_r", radius=0.01),
            ],
        )
        sidecar = HumanShoeContactSidecarContract(
            schema_version="human_shoe_contact_sidecar_1",
            source_model_path="model.osim",
            generated_model_path="out.osim",
            generated_model_name="generated",
            contacts=(
                ContactGeometrySidecarContract(
                    name="heel",
                    type="ContactSphere",
                    body_name="calcn_r",
                    location_m=np.array([0.0, -0.1, 0.0]),
                    support_marker_name=None,
                    support_offset_m=np.array([0.0, 0.0, 0.0]),
                    orientation_deg=np.array([90.0, 0.0, 0.0]),
                    radius_m=0.02,
                    mesh_file=None,
                ),
            ),
        )
        with self.assertRaisesRegex(ValueError, "already exists"):
            inject_contact_sidecar(model, sidecar)

        injected = inject_contact_sidecar(model, sidecar, replace_existing=True)
        self.assertEqual(injected.name, "generated")
        self.assertEqual(len(injected.contact_geometry), 1)
        np.testing.assert_allclose(injected.contact_geometry[0].orientation, np.deg2rad([90.0, 0.0, 0.0]))
        self.assertEqual(injected.contact_geometry[0].radius, 0.02)

    def test_rejects_contact_mesh_validation(self):
        """Reject invalid contact mesh definitions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sidecar.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "human_shoe_contact_sidecar_1",
                        "source_model_path": "model.osim",
                        "generated_model_path": "out.osim",
                        "contacts": [
                            {
                                "name": "sole",
                                "type": "ContactMesh",
                                "body_name": "calcn_r",
                                "orientation_deg": [0.0, 0.0, 0.0],
                                "mesh_file": "sole.stl",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "ContactMesh must define location_m"):
                load_contact_sidecar(path)

    def test_writes_and_reparses_augmented_model(self):
        """Write an augmented model and verify a reparsed document keeps the contact set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = Path(tmpdir)
            sidecar_path = temp_path / "sidecar.json"
            output_path = temp_path / "out.osim"
            source_model_path = Path("newton/examples/assets/gait2354_subject01.osim").resolve()
            sidecar_path.write_text(
                json.dumps(
                    {
                        "schema_version": "human_shoe_contact_sidecar_1",
                        "source_model_path": str(source_model_path),
                        "generated_model_path": "generated.osim",
                        "generated_model_name": "gait2354_contact",
                        "contacts": [
                            {
                                "name": "heel",
                                "type": "ContactSphere",
                                "body_name": "calcn_r",
                                "support_marker_name": "R.Heel",
                                "support_offset_m": [0.0, 0.0, 0.0],
                                "orientation_deg": [0.0, 0.0, 0.0],
                                "radius_m": 0.02,
                            },
                            {
                                "name": "toe",
                                "type": "ContactSphere",
                                "body_name": "calcn_r",
                                "location_m": [0.21, 0.0, 0.0],
                                "support_offset_m": [0.0, 0.0, 0.0],
                                "orientation_deg": [0.0, 0.0, 0.0],
                                "radius_m": 0.02,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            written = write_contact_augmented_osim(sidecar_path, output_path=output_path)
            self.assertEqual(written, output_path)
            reparsed = opensim.parse_osim(written)
            self.assertEqual(reparsed.name, "gait2354_contact")
            self.assertEqual({cg.name for cg in reparsed.contact_geometry if cg.body == "calcn_r"}, {"heel", "toe"})

            import_result = OsimImportResult(model=reparsed, body_index={"ground": -1, "calcn_r": 0})
            resolve_attachment(
                import_result,
                FootShoeAttachmentContract(
                    foot_body_name="calcn_r",
                    shoe_carrier_body_name="calcn_r",
                    translation_m=[0.0, 0.0, 0.0],
                    rotation_deg=[0.0, 0.0, 0.0],
                ),
            )

    def test_injects_gait_marker_seeded_contacts_and_resolves_attachment(self):
        """Inject marker-seeded contacts into gait2354 and resolve the foot attachment."""
        model = opensim.parse_osim(Path("newton/examples/assets/gait2354_subject01.osim").resolve())
        model.markers.append(opensim.OsimMarker(name="R.Toe", body="calcn_r", location=(0.21, 0.0, 0.0)))
        sidecar = HumanShoeContactSidecarContract(
            schema_version="human_shoe_contact_sidecar_1",
            source_model_path="gait2354_subject01.osim",
            generated_model_path="out.osim",
            generated_model_name=None,
            contacts=(
                ContactGeometrySidecarContract(
                    name="heel",
                    type="ContactSphere",
                    body_name="calcn_r",
                    location_m=None,
                    support_marker_name="R.Heel",
                    support_offset_m=np.array([0.0, 0.0, 0.0]),
                    orientation_deg=np.array([0.0, 0.0, 0.0]),
                    radius_m=0.02,
                    mesh_file=None,
                ),
                ContactGeometrySidecarContract(
                    name="toe",
                    type="ContactSphere",
                    body_name="calcn_r",
                    location_m=None,
                    support_marker_name="R.Toe",
                    support_offset_m=np.array([0.0, 0.0, 0.0]),
                    orientation_deg=np.array([0.0, 0.0, 0.0]),
                    radius_m=0.02,
                    mesh_file=None,
                ),
            ),
        )
        injected = inject_contact_sidecar(model, sidecar)
        import_result = OsimImportResult(model=injected, body_index={"ground": -1, "calcn_r": 0})

        resolved = resolve_attachment(
            import_result,
            FootShoeAttachmentContract(
                foot_body_name="calcn_r",
                shoe_carrier_body_name="calcn_r",
                translation_m=[0.0, 0.0, 0.0],
                rotation_deg=[0.0, 0.0, 0.0],
            ),
        )

        self.assertEqual(resolved.reference.contact_geometry_names, ("heel", "toe"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
