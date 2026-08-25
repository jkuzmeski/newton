# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test OpenSim-referenced gait2354 measurement scaling."""

import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from projects.gait_c3d.c3d_adapter import C3DMarkerTrajectory
from projects.gait_c3d.subject_scaling import (
    _ALIASES,
    _MEASUREMENTS,
    _VIRTUAL,
    _measurement_factors,
    _scale_inertia,
    _scale_osim_document,
)


class TestGaitSubjectScaling(unittest.TestCase):
    """Test measurement semantics against the pinned OpenSim marker fixture."""

    def test_recovers_uniform_scale_from_reference_markers(self):
        """Recover a known scale from every gait2354 marker-pair measurement."""
        reference_path = Path(__file__).parents[2] / "projects/gait_c3d/assets/gait2354_scale_reference.json"
        reference = json.loads(reference_path.read_text())["marker_positions"]
        target_to_source = {target: source for source, target in _ALIASES.items()}
        required_targets = {marker for _, pairs, _ in _MEASUREMENTS for pair in pairs for marker in pair}
        names = []
        positions = []
        scale = 1.25
        for target in sorted(required_targets):
            source = target_to_source[target]
            osim = scale * np.asarray(reference[target])
            newton_position = np.asarray((osim[0], -osim[2], osim[1]))
            names.append(source)
            positions.append(newton_position)
        for target, sources in _VIRTUAL.items():
            osim = scale * np.asarray(reference[target])
            newton_position = np.asarray((osim[0], -osim[2], osim[1]))
            for source in sources:
                if source not in names:
                    names.append(source)
                    positions.append(newton_position)
        frame = np.asarray(positions, dtype=np.float32)
        trajectory = C3DMarkerTrajectory(
            times=np.asarray((0.0, 0.01)),
            positions=np.stack((frame, frame)),
            valid=np.ones((2, len(names)), dtype=bool),
            marker_names=tuple(names),
            rate=100.0,
            first_frame=0,
            lab_to_newton=np.eye(3),
            source_file="synthetic.c3d",
            source_sha256="0" * 64,
        )
        factors, diagnostics = _measurement_factors(trajectory, None)
        self.assertEqual(set(diagnostics), {"pelvis", "torso", "thigh", "shank", "foot"})
        for value in factors.values():
            np.testing.assert_allclose(value, (scale, scale, scale), atol=1.0e-7)

    def test_scales_inertia_like_a_stretched_solid_box(self):
        """Match analytic box inertia after anisotropic OpenSim body scaling."""
        mass = 4.0
        dimensions = np.asarray((0.3, 0.5, 0.7))
        factors = np.asarray((1.3, 0.8, 1.7))

        def inertia(values):
            x, y, z = values
            return mass / 12.0 * np.asarray((y * y + z * z, x * x + z * z, x * x + y * y))

        source = inertia(dimensions)
        result = _scale_inertia((*source, 0.0, 0.0, 0.0), tuple(factors), 1.0)
        np.testing.assert_allclose(result[:3], inertia(dimensions * factors), rtol=1.0e-12)

    def test_scales_legacy_geometry_joint_functions_and_mass_once(self):
        """Preserve corrected ModelScaler XML semantics on a compact fixture."""
        model = """<OpenSimDocument Version="20302"><Model><BodySet><objects>
        <Body name="thigh"><mass>5</mass><mass_center>0 -0.2 0</mass_center>
          <inertia_xx>1</inertia_xx><inertia_yy>1</inertia_yy><inertia_zz>1</inertia_zz>
          <inertia_xy>0</inertia_xy><inertia_xz>0</inertia_xz><inertia_yz>0</inertia_yz>
          <VisibleObject><scale_factors>1.2 1.2 1.2</scale_factors><GeometrySet><objects>
            <DisplayGeometry><geometry_file>thigh.vtp</geometry_file><scale_factors>1.1 1.1 1.1</scale_factors></DisplayGeometry>
          </objects></GeometrySet></VisibleObject>
        </Body>
        <Body name="shank"><mass>3</mass><mass_center>0 -0.1 0</mass_center>
          <inertia_xx>1</inertia_xx><inertia_yy>1</inertia_yy><inertia_zz>1</inertia_zz>
          <inertia_xy>0</inertia_xy><inertia_xz>0</inertia_xz><inertia_yz>0</inertia_yz>
          <Joint><CustomJoint name="knee"><parent_body>thigh</parent_body><location_in_parent>0 0 0</location_in_parent><location>0 0 0</location>
            <SpatialTransform><TransformAxis name="translation2"><axis>0 1 0</axis><function>
              <NaturalCubicSpline><x>0 1</x><y>-0.4 -0.5</y></NaturalCubicSpline>
            </function></TransformAxis></SpatialTransform>
          </CustomJoint></Joint>
        </Body></objects></BodySet></Model></OpenSimDocument>"""
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.osim"
            output = Path(directory) / "scaled.osim"
            source.write_text(model)
            _scale_osim_document(
                str(source),
                {"thigh": (1.5, 2.0, 2.5), "shank": (1.0, 1.0, 1.0)},
                str(output),
                preserve_mass_distribution=True,
                subject_mass=16.0,
            )
            root = ET.parse(output).getroot()
        thigh = next(body for body in root.iter("Body") if body.get("name") == "thigh")
        np.testing.assert_allclose(
            [float(value) for value in thigh.find("VisibleObject/scale_factors").text.split()],
            (1.8, 2.4, 3.0),
        )
        np.testing.assert_allclose(
            [
                float(value)
                for value in thigh.find("VisibleObject/GeometrySet/objects/DisplayGeometry/scale_factors").text.split()
            ],
            (1.1, 1.1, 1.1),
        )
        spline = next(root.iter("NaturalCubicSpline"))
        np.testing.assert_allclose([float(value) for value in spline.find("y").text.split()], (-0.8, -1.0))
        self.assertAlmostEqual(sum(float(body.findtext("mass")) for body in root.iter("Body")), 16.0)


if __name__ == "__main__":
    unittest.main()
