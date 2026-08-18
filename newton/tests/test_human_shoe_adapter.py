# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the human shoe OpenSim contact attachment adapter."""

import unittest

import numpy as np

from newton.opensim import OsimBody, OsimContactGeometry, OsimImportResult, OsimModel
from projects.human_shoe import (
    OSIM_LOCAL_TO_Z_UP_JUMP_BASIS,
    FootShoeAttachmentContract,
    attach_sole_geometry,
    resolve_attachment,
)


def _make_import_result(contact_geometry: list[OsimContactGeometry]) -> OsimImportResult:
    model = OsimModel(
        bodies=[OsimBody(name="foot"), OsimBody(name="shoe_carrier")],
        contact_geometry=contact_geometry,
    )
    return OsimImportResult(model=model, body_index={"ground": -1, "foot": 0, "shoe_carrier": 1})


class TestHumanShoeAdapter(unittest.TestCase):
    def test_resolve_attachment_uses_contact_spheres(self):
        """Resolve heel and toe spheres into a foot-frame support centroid."""
        import_result = _make_import_result(
            [
                OsimContactGeometry(
                    name="heel", type="ContactSphere", body="foot", location=(-0.05, -0.07, 0.0), radius=0.03
                ),
                OsimContactGeometry(
                    name="toe", type="ContactSphere", body="foot", location=(0.16, -0.07, 0.0), radius=0.03
                ),
            ]
        )
        contract = FootShoeAttachmentContract(
            foot_body_name="foot",
            shoe_carrier_body_name="foot",
            translation_m=[0.01, 0.02, 0.03],
            rotation_deg=[0.0, 0.0, 0.0],
        )

        resolved = resolve_attachment(import_result, contract)

        np.testing.assert_allclose(resolved.reference.origin_in_foot_m, [0.055, -0.1, 0.0])
        np.testing.assert_allclose(
            resolved.reference.support_points_in_foot_m,
            [[-0.05, -0.1, 0.0], [0.16, -0.1, 0.0]],
        )
        self.assertEqual(resolved.reference.contact_geometry_names, ("heel", "toe"))
        self.assertEqual(resolved.foot_body_index, 0)
        self.assertEqual(resolved.shoe_carrier_body_index, 0)
        np.testing.assert_allclose(resolved.shoe_to_foot[:3, 3], [0.065, -0.08, 0.03])
        np.testing.assert_allclose(resolved.shoe_to_foot[:3, :3] @ [0.0, 0.0, 1.0], [0.0, 1.0, 0.0])

    def test_resolve_attachment_rejects_distinct_carrier_frame(self):
        """Reject a carrier whose local frame differs from the resolved foot frame."""
        import_result = _make_import_result(
            [OsimContactGeometry(name="heel", type="ContactSphere", body="foot", location=(0.0, 0.0, 0.0), radius=0.03)]
        )
        contract = FootShoeAttachmentContract(
            foot_body_name="foot",
            shoe_carrier_body_name="shoe_carrier",
            translation_m=[0.0, 0.0, 0.0],
            rotation_deg=[0.0, 0.0, 0.0],
        )

        with self.assertRaisesRegex(ValueError, "requires foot_body_name and shoe_carrier_body_name"):
            resolve_attachment(import_result, contract)

    def test_resolve_attachment_rejects_missing_foot(self):
        """Reject contracts whose foot body is absent from the import result."""
        import_result = _make_import_result([])
        contract = FootShoeAttachmentContract(
            foot_body_name="missing",
            shoe_carrier_body_name="foot",
            translation_m=[0.0, 0.0, 0.0],
            rotation_deg=[0.0, 0.0, 0.0],
        )

        with self.assertRaisesRegex(KeyError, "foot body 'missing' not found"):
            resolve_attachment(import_result, contract)

    def test_resolve_attachment_rejects_missing_carrier(self):
        """Reject contracts whose shoe carrier is absent from the import result."""
        import_result = _make_import_result([])
        contract = FootShoeAttachmentContract(
            foot_body_name="foot",
            shoe_carrier_body_name="missing",
            translation_m=[0.0, 0.0, 0.0],
            rotation_deg=[0.0, 0.0, 0.0],
        )

        with self.assertRaisesRegex(KeyError, "shoe carrier body 'missing' not found"):
            resolve_attachment(import_result, contract)

    def test_resolve_attachment_rejects_missing_contact_geometry(self):
        """Reject foot bodies without any attached contact geometry."""
        import_result = _make_import_result([])
        contract = FootShoeAttachmentContract(
            foot_body_name="foot",
            shoe_carrier_body_name="foot",
            translation_m=[0.0, 0.0, 0.0],
            rotation_deg=[0.0, 0.0, 0.0],
        )

        with self.assertRaisesRegex(ValueError, "no contact geometry attached to foot body 'foot'"):
            resolve_attachment(import_result, contract)

    def test_resolve_attachment_rejects_unsupported_geometry(self):
        """Reject unsupported non-ground contact geometry types."""
        import_result = _make_import_result(
            [OsimContactGeometry(name="pad", type="ContactHalfSpace", body="foot", location=(0.0, 0.0, 0.0))]
        )
        contract = FootShoeAttachmentContract(
            foot_body_name="foot",
            shoe_carrier_body_name="foot",
            translation_m=[0.0, 0.0, 0.0],
            rotation_deg=[0.0, 0.0, 0.0],
        )

        with self.assertRaisesRegex(ValueError, "unsupported contact geometry type 'ContactHalfSpace'"):
            resolve_attachment(import_result, contract)

    def test_resolve_attachment_uses_contact_mesh_origin(self):
        """Use a contact mesh frame origin as its explicit attachment reference."""
        import_result = _make_import_result(
            [OsimContactGeometry(name="sole_mesh", type="ContactMesh", body="foot", location=(0.1, -0.05, 0.02))]
        )
        contract = FootShoeAttachmentContract(
            foot_body_name="foot",
            shoe_carrier_body_name="foot",
            translation_m=[0.0, 0.0, 0.0],
            rotation_deg=[0.0, 0.0, 0.0],
        )

        resolved = resolve_attachment(import_result, contract)

        self.assertEqual(resolved.reference.contact_geometry_names, ("sole_mesh",))
        np.testing.assert_allclose(resolved.reference.origin_in_foot_m, [0.1, -0.05, 0.02])

    def test_attach_sole_geometry_validates_shapes(self):
        """Reject sole arrays that are not [N, 3] or do not match in length."""
        contract = FootShoeAttachmentContract(
            foot_body_name="foot",
            shoe_carrier_body_name="foot",
            translation_m=[0.0, 0.0, 0.0],
            rotation_deg=[0.0, 0.0, 0.0],
        )
        resolved = resolve_attachment(
            _make_import_result(
                [
                    OsimContactGeometry(
                        name="heel", type="ContactSphere", body="foot", location=(0.0, 0.0, 0.0), radius=0.01
                    )
                ]
            ),
            contract,
        )

        with self.assertRaisesRegex(ValueError, "bottom_local must have shape \\[N, 3\\]"):
            attach_sole_geometry(resolved, np.zeros((3,)), np.zeros((1, 3)))
        with self.assertRaisesRegex(ValueError, "bottom_local and top_local must have matching shapes"):
            attach_sole_geometry(resolved, np.zeros((2, 3)), np.zeros((1, 3)))

    def test_attach_sole_geometry_zero_offset_centers_top(self):
        """Apply the zero-offset translation at the top-interface centroid."""
        import_result = _make_import_result(
            [OsimContactGeometry(name="heel", type="ContactSphere", body="foot", location=(0.0, 0.0, 0.0), radius=0.02)]
        )
        contract = FootShoeAttachmentContract(
            foot_body_name="foot",
            shoe_carrier_body_name="foot",
            translation_m=[0.0, 0.0, 0.0],
            rotation_deg=[0.0, 0.0, 0.0],
        )
        resolved = resolve_attachment(import_result, contract)

        bottom = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        top = np.array([[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]])
        attached = attach_sole_geometry(resolved, bottom, top)

        np.testing.assert_allclose(attached.top_local.mean(axis=0), resolved.shoe_to_foot[:3, 3])
        np.testing.assert_allclose(
            attached.rest_len, np.linalg.norm(attached.bottom_local - attached.top_local, axis=1)
        )

    def test_attach_sole_geometry_round_trips_z_up_basis(self):
        """Map OpenSim-local sole coordinates into a Z-up jump fixture basis."""
        import_result = _make_import_result(
            [OsimContactGeometry(name="heel", type="ContactSphere", body="foot", location=(0.0, 0.0, 0.0), radius=0.02)]
        )
        contract = FootShoeAttachmentContract(
            foot_body_name="foot",
            shoe_carrier_body_name="foot",
            translation_m=[0.0, 0.0, 0.0],
            rotation_deg=[0.0, 0.0, 0.0],
        )
        resolved = resolve_attachment(import_result, contract)
        bottom = np.array([[1.0, 0.0, 0.0]])
        top = np.array([[0.0, 0.0, 0.0]])

        attached = attach_sole_geometry(resolved, bottom, top, output_basis=OSIM_LOCAL_TO_Z_UP_JUMP_BASIS)
        np.testing.assert_allclose(
            attached.top_local,
            (resolved.shoe_to_foot[:3, 3] @ OSIM_LOCAL_TO_Z_UP_JUMP_BASIS.T).reshape(1, 3),
        )
        np.testing.assert_allclose(attached.bottom_local - attached.top_local, [[1.0, 0.0, 0.0]])


if __name__ == "__main__":
    unittest.main(verbosity=2)
