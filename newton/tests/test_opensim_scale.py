# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the OpenSim-port subject-scaling workflow (ScaleTool)."""

import os
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np

import newton.examples
import newton.opensim as osim
from newton._src.opensim.kinematics import ForwardKinematics
from newton._src.opensim.mocap import MarkerData
from newton._src.opensim.scale import _scale_inertia, measurement_scale_value, synthesize_markers

# A self-contained legacy (Version < 30000) leg model with a MarkerSet.
LEG_OSIM = """<?xml version="1.0" encoding="UTF-8" ?>
<OpenSimDocument Version="20302">
    <Model name="leg">
        <gravity> 0 -9.80665 0</gravity>
        <BodySet>
            <objects>
                <Body name="ground">
                    <mass>0</mass><mass_center> 0 0 0</mass_center>
                    <inertia_xx>0</inertia_xx><inertia_yy>0</inertia_yy><inertia_zz>0</inertia_zz>
                    <inertia_xy>0</inertia_xy><inertia_xz>0</inertia_xz><inertia_yz>0</inertia_yz>
                    <Joint/>
                </Body>
                <Body name="thigh">
                    <mass>5.0</mass><mass_center> 0 -0.2 0</mass_center>
                    <inertia_xx>0.1</inertia_xx><inertia_yy>0.02</inertia_yy><inertia_zz>0.1</inertia_zz>
                    <inertia_xy>0</inertia_xy><inertia_xz>0</inertia_xz><inertia_yz>0</inertia_yz>
                    <Joint>
                        <CustomJoint name="hip">
                            <SpatialTransform>
                                <TransformAxis name="rotation1"><coordinates>hip_angle</coordinates><axis>0 0 1</axis>
                                    <function><LinearFunction><coefficients> 1 0</coefficients></LinearFunction></function></TransformAxis>
                                <TransformAxis name="rotation2"><coordinates></coordinates><axis>0 1 0</axis>
                                    <function><Constant><value>0</value></Constant></function></TransformAxis>
                                <TransformAxis name="rotation3"><coordinates></coordinates><axis>1 0 0</axis>
                                    <function><Constant><value>0</value></Constant></function></TransformAxis>
                                <TransformAxis name="translation1"><coordinates></coordinates><axis>1 0 0</axis>
                                    <function><Constant><value>0</value></Constant></function></TransformAxis>
                                <TransformAxis name="translation2"><coordinates></coordinates><axis>0 1 0</axis>
                                    <function><Constant><value>0</value></Constant></function></TransformAxis>
                                <TransformAxis name="translation3"><coordinates></coordinates><axis>0 0 1</axis>
                                    <function><Constant><value>0</value></Constant></function></TransformAxis>
                            </SpatialTransform>
                            <parent_body>ground</parent_body>
                            <location_in_parent> 0 1.0 0</location_in_parent>
                            <orientation_in_parent> 0 0 0</orientation_in_parent>
                            <location> 0 0 0</location><orientation> 0 0 0</orientation>
                            <CoordinateSet><objects>
                                <Coordinate name="hip_angle"><motion_type>rotational</motion_type>
                                    <default_value>0</default_value><range>-2 2</range></Coordinate>
                            </objects></CoordinateSet>
                        </CustomJoint>
                    </Joint>
                </Body>
                <Body name="shank">
                    <mass>3.0</mass><mass_center> 0 -0.2 0</mass_center>
                    <inertia_xx>0.05</inertia_xx><inertia_yy>0.01</inertia_yy><inertia_zz>0.05</inertia_zz>
                    <inertia_xy>0</inertia_xy><inertia_xz>0</inertia_xz><inertia_yz>0</inertia_yz>
                    <Joint>
                        <CustomJoint name="knee">
                            <SpatialTransform>
                                <TransformAxis name="rotation1"><coordinates>knee_angle</coordinates><axis>0 0 1</axis>
                                    <function><LinearFunction><coefficients> 1 0</coefficients></LinearFunction></function></TransformAxis>
                                <TransformAxis name="rotation2"><coordinates></coordinates><axis>0 1 0</axis>
                                    <function><Constant><value>0</value></Constant></function></TransformAxis>
                                <TransformAxis name="rotation3"><coordinates></coordinates><axis>1 0 0</axis>
                                    <function><Constant><value>0</value></Constant></function></TransformAxis>
                                <TransformAxis name="translation1"><coordinates></coordinates><axis>1 0 0</axis>
                                    <function><Constant><value>0</value></Constant></function></TransformAxis>
                                <TransformAxis name="translation2"><coordinates></coordinates><axis>0 1 0</axis>
                                    <function><Constant><value>0</value></Constant></function></TransformAxis>
                                <TransformAxis name="translation3"><coordinates></coordinates><axis>0 0 1</axis>
                                    <function><Constant><value>0</value></Constant></function></TransformAxis>
                            </SpatialTransform>
                            <parent_body>thigh</parent_body>
                            <location_in_parent> 0 -0.4 0</location_in_parent>
                            <orientation_in_parent> 0 0 0</orientation_in_parent>
                            <location> 0 0 0</location><orientation> 0 0 0</orientation>
                            <CoordinateSet><objects>
                                <Coordinate name="knee_angle"><motion_type>rotational</motion_type>
                                    <default_value>0</default_value><range>-2 1</range></Coordinate>
                            </objects></CoordinateSet>
                        </CustomJoint>
                    </Joint>
                </Body>
            </objects>
        </BodySet>
        <MarkerSet>
            <objects>
                <Marker name="thigh_top"><body>thigh</body><location> 0.06 -0.05 0.02</location></Marker>
                <Marker name="thigh_bot"><body>thigh</body><location> -0.05 -0.35 -0.03</location></Marker>
                <Marker name="shank_top"><body>shank</body><location> 0.05 -0.05 0.02</location></Marker>
                <Marker name="shank_mid"><body>shank</body><location> -0.04 -0.25 -0.02</location></Marker>
                <Marker name="ankle"><body>shank</body><location> 0 -0.4 0.03</location></Marker>
            </objects>
        </MarkerSet>
    </Model>
</OpenSimDocument>
"""


def _write_leg() -> str:
    fd, path = tempfile.mkstemp(suffix=".osim")
    os.close(fd)
    with open(path, "w") as f:
        f.write(LEG_OSIM)
    return path


class TestMarkerAssignment(unittest.TestCase):
    def test_apply_drops_unmapped_and_renames(self):
        """apply_marker_assignment renames to model markers and discards unmapped labels."""
        md = MarkerData(
            times=np.array([0.0, 1.0]),
            marker_names=["Sub:RASI", "Sub:LASI", "Sub:JUNK"],
            data=np.arange(2 * 3 * 3, dtype=float).reshape(2, 3, 3),
            rate=1.0,
        )
        out = osim.apply_marker_assignment(md, {"Sub:RASI": "R.ASIS", "Sub:LASI": "L.ASIS", "Sub:JUNK": None})
        self.assertEqual(out.marker_names, ["R.ASIS", "L.ASIS"])
        self.assertEqual(out.data.shape, (2, 2, 3))
        np.testing.assert_array_equal(out.data[:, 0, :], md.data[:, 0, :])

    def test_suggest_uses_alias_exact_and_fuzzy(self):
        """suggest_marker_assignment prefers aliases, then exact, then fuzzy matches."""
        model = ["R.ASIS", "L.ASIS", "Sternum"]
        got = osim.suggest_marker_assignment(["RASI", "Sternum", "ZZZ"], model, aliases={"RASI": "R.ASIS"})
        self.assertEqual(got["RASI"], "R.ASIS")
        self.assertEqual(got["Sternum"], "Sternum")
        self.assertIsNone(got["ZZZ"])


class TestScaleMath(unittest.TestCase):
    def test_measurement_scale_value_is_distance_ratio(self):
        """A measurement's scale value is the mean experimental/model marker-pair distance ratio."""
        exp = {"a": np.array([0.0, 0.0, 0.0]), "b": np.array([2.0, 0.0, 0.0])}
        mdl = {"a": np.array([0.0, 0.0, 0.0]), "b": np.array([1.0, 0.0, 0.0])}
        meas = osim.Measurement("m", [osim.MarkerPair("a", "b")], {"body": "XYZ"})
        self.assertAlmostEqual(measurement_scale_value(meas, exp, mdl), 2.0)

    def test_inertia_scaling_matches_solid_box(self):
        """_scale_inertia reproduces the analytic inertia of a stretched solid box.

        A uniform box has I = (m/12) diag(y^2+z^2, x^2+z^2, x^2+y^2). Stretching the
        box edges by (sx, sy, sz) with mass held constant must transform the tensor
        exactly the way _scale_inertia does (mass_scale = 1).
        """
        m, lx, ly, lz = 4.0, 0.3, 0.5, 0.7
        s = (1.3, 0.8, 1.7)

        def box_I(ex, ey, ez):
            return (m / 12.0) * np.array([ey**2 + ez**2, ex**2 + ez**2, ex**2 + ey**2])

        base = box_I(lx, ly, lz)
        stretched = box_I(lx * s[0], ly * s[1], lz * s[2])
        scaled = _scale_inertia((base[0], base[1], base[2], 0.0, 0.0, 0.0), s, 1.0)
        np.testing.assert_allclose(scaled[:3], stretched, rtol=1e-12)
        np.testing.assert_allclose(scaled[3:], (0.0, 0.0, 0.0), atol=1e-12)


class TestGait2354Measurements(unittest.TestCase):
    def test_share_bilateral_length_scales(self):
        """Use bilateral segment lengths without mixing joint breadth into uniform scale."""
        measurements = {measurement.name: measurement for measurement in osim.gait2354_measurement_set()}

        self.assertEqual(
            [(pair.marker1, pair.marker2) for pair in measurements["thigh"].marker_pairs],
            [("R.ASIS", "R.Knee.Lat"), ("L.ASIS", "L.Knee.Lat")],
        )
        self.assertEqual(
            set(measurements["thigh"].body_scales),
            {"femur_r", "femur_l", "patella_r", "patella_l"},
        )
        self.assertEqual(
            set(measurements["shank"].body_scales),
            {"tibia_r", "tibia_l", "talus_r", "talus_l"},
        )
        self.assertNotIn(
            ("R.Knee.Lat", "R.Knee.Med"), [(p.marker1, p.marker2) for p in measurements["thigh"].marker_pairs]
        )


class TestModelScalerDocument(unittest.TestCase):
    def test_manual_scale_geometry(self):
        """ModelScaler scales markers, joint frames, mass center, and mass by the body factors."""
        path = _write_leg()
        try:
            model = osim.parse_osim(path)
            scaler = osim.ModelScaler(model, path)
            out = path + ".scaled.osim"
            scaler.scale(
                {"thigh": (1.5, 2.0, 1.5), "shank": (2.0, 2.0, 2.0)},
                out,
                preserve_mass_distribution=False,
            )
            m = osim.parse_osim(out)
            markers = {mk.name: np.array(mk.location) for mk in m.markers}
            np.testing.assert_allclose(markers["thigh_top"], [0.06 * 1.5, -0.05 * 2.0, 0.02 * 1.5], rtol=1e-6)
            # knee joint location_in_parent is in the (parent) thigh frame -> thigh factors.
            knee = next(j for j in m.joints if j.name == "knee")
            np.testing.assert_allclose(knee.parent_transform.translation, [0.0, -0.4 * 2.0, 0.0], atol=1e-9)
            thigh = m.body("thigh")
            np.testing.assert_allclose(thigh.mass_center, [0.0, -0.2 * 2.0, 0.0], atol=1e-9)
            # preserve_mass_distribution=False scales mass by the volume factor.
            self.assertAlmostEqual(thigh.mass, 5.0 * 1.5 * 2.0 * 1.5, places=6)
        finally:
            os.remove(path)
            if os.path.exists(path + ".scaled.osim"):
                os.remove(path + ".scaled.osim")

    def test_scale_display_geometry_once(self):
        """Scale the combined visible/display geometry exactly once per body."""
        path = _write_leg()
        out = path + ".scaled.osim"
        try:
            tree = ET.parse(path)
            thigh = next(body for body in tree.getroot().iter("Body") if body.get("name") == "thigh")
            visible = ET.SubElement(thigh, "VisibleObject")
            ET.SubElement(visible, "scale_factors").text = " 1.2 1.2 1.2 "
            geometry_set = ET.SubElement(visible, "GeometrySet")
            objects = ET.SubElement(geometry_set, "objects")
            display = ET.SubElement(objects, "DisplayGeometry")
            ET.SubElement(display, "geometry_file").text = " thigh.vtp "
            ET.SubElement(display, "scale_factors").text = " 1.1 1.1 1.1 "
            tree.write(path)

            model = osim.parse_osim(path)
            osim.ModelScaler(model, path).scale({"thigh": (1.5, 2.0, 2.5)}, out)
            entries = osim.read_display_geometry(out)["thigh"]

            self.assertEqual(len(entries), 1)
            np.testing.assert_allclose(entries[0][2], 1.2 * 1.1 * np.array([1.5, 2.0, 2.5]))
        finally:
            for candidate in (path, out):
                if os.path.exists(candidate):
                    os.remove(candidate)

    def test_scale_custom_joint_translation_function(self):
        """Scale CustomJoint translation-function output in its parent body frame."""
        path = _write_leg()
        out = path + ".scaled.osim"
        try:
            tree = ET.parse(path)
            knee = next(joint for joint in tree.getroot().iter("CustomJoint") if joint.get("name") == "knee")
            knee.find("location_in_parent").text = " 0 0 0 "
            translation = next(
                axis
                for axis in knee.find("SpatialTransform").findall("TransformAxis")
                if axis.get("name") == "translation2"
            )
            function = translation.find("function")
            function.clear()
            spline = ET.SubElement(function, "NaturalCubicSpline")
            ET.SubElement(spline, "x").text = " 0 1 "
            ET.SubElement(spline, "y").text = " -0.4 -0.5 "
            free_translation = next(
                axis
                for axis in knee.find("SpatialTransform").findall("TransformAxis")
                if axis.get("name") == "translation1"
            )
            free_function = free_translation.find("function")
            free_function.clear()
            linear = ET.SubElement(free_function, "LinearFunction")
            ET.SubElement(linear, "coefficients").text = " 1 0 "
            tree.write(path)

            model = osim.parse_osim(path)
            osim.ModelScaler(model, path).scale({"thigh": (1.5, 2.0, 1.5), "shank": (1.0, 1.0, 1.0)}, out)
            scaled_tree = ET.parse(out)
            scaled_knee = next(
                joint for joint in scaled_tree.getroot().iter("CustomJoint") if joint.get("name") == "knee"
            )
            scaled_translation = next(
                axis
                for axis in scaled_knee.find("SpatialTransform").findall("TransformAxis")
                if axis.get("name") == "translation2"
            )

            np.testing.assert_allclose(
                [float(value) for value in scaled_translation.find("function/NaturalCubicSpline/y").text.split()],
                [-0.8, -1.0],
            )
            np.testing.assert_allclose(
                [float(value) for value in scaled_translation.find("axis").text.split()],
                [0.0, 1.0, 0.0],
            )
            scaled_free_translation = next(
                axis
                for axis in scaled_knee.find("SpatialTransform").findall("TransformAxis")
                if axis.get("name") == "translation1"
            )
            np.testing.assert_allclose(
                [
                    float(value)
                    for value in scaled_free_translation.find("function/LinearFunction/coefficients").text.split()
                ],
                [1.0, 0.0],
            )
        finally:
            for candidate in (path, out):
                if os.path.exists(candidate):
                    os.remove(candidate)

    def test_scale_gait2354_knee_translation_and_display_once(self):
        """Scale gait2354 knee translation and nested bone geometry by one body factor."""
        path = newton.examples.get_asset("gait2354_subject01.osim")
        fd, out = tempfile.mkstemp(suffix=".scaled.osim")
        os.close(fd)
        try:
            original_geometry = osim.read_display_geometry(path)
            original_tree = ET.parse(path)
            original_knee = next(
                joint for joint in original_tree.getroot().iter("CustomJoint") if joint.get("name") == "knee_l"
            )
            original_translation = next(
                axis
                for axis in original_knee.find("SpatialTransform").findall("TransformAxis")
                if axis.get("name") == "translation2"
            )
            original_y = np.array(
                [float(value) for value in original_translation.find("function/NaturalCubicSpline/y").text.split()]
            )
            original_moving_point = next(
                point
                for point in original_tree.getroot().iter("MovingPathPoint")
                if point.get("name") == "rect_fem_l-P3"
            )
            original_moving_x = np.array(
                [float(value) for value in original_moving_point.find("x_location/NaturalCubicSpline/y").text.split()]
            )

            model = osim.parse_osim(path)
            osim.ModelScaler(model, path).scale({"femur_l": (1.1, 1.1, 1.1), "tibia_l": (0.9, 0.9, 0.9)}, out)
            scaled_geometry = osim.read_display_geometry(out)
            scaled_tree = ET.parse(out)
            scaled_knee = next(
                joint for joint in scaled_tree.getroot().iter("CustomJoint") if joint.get("name") == "knee_l"
            )
            scaled_translation = next(
                axis
                for axis in scaled_knee.find("SpatialTransform").findall("TransformAxis")
                if axis.get("name") == "translation2"
            )
            scaled_y = np.array(
                [float(value) for value in scaled_translation.find("function/NaturalCubicSpline/y").text.split()]
            )

            np.testing.assert_allclose(scaled_y, 1.1 * original_y)
            np.testing.assert_allclose(scaled_geometry["femur_l"][0][2], 1.1 * original_geometry["femur_l"][0][2])
            np.testing.assert_allclose(scaled_geometry["tibia_l"][0][2], 0.9 * original_geometry["tibia_l"][0][2])
            scaled_moving_point = next(
                point for point in scaled_tree.getroot().iter("MovingPathPoint") if point.get("name") == "rect_fem_l-P3"
            )
            scaled_moving_x = np.array(
                [float(value) for value in scaled_moving_point.find("x_location/NaturalCubicSpline/y").text.split()]
            )
            np.testing.assert_allclose(scaled_moving_x, 0.9 * original_moving_x, atol=5.0e-9)
        finally:
            if os.path.exists(out):
                os.remove(out)

    def test_preserve_mass_distribution_hits_target(self):
        """With preserve_mass_distribution the total mass equals the subject mass, keeping the ratio."""
        path = _write_leg()
        try:
            model = osim.parse_osim(path)
            out = path + ".scaled.osim"
            osim.ModelScaler(model, path).scale(
                {"thigh": (1.2, 1.2, 1.2), "shank": (1.2, 1.2, 1.2)},
                out,
                preserve_mass_distribution=True,
                subject_mass=16.0,
            )
            m = osim.parse_osim(out)
            self.assertAlmostEqual(sum(b.mass for b in m.bodies), 16.0, places=5)
            # 5:3 distribution preserved -> 10:6.
            self.assertAlmostEqual(m.body("thigh").mass, 10.0, places=5)
            self.assertAlmostEqual(m.body("shank").mass, 6.0, places=5)
        finally:
            os.remove(path)
            if os.path.exists(out):
                os.remove(out)


class TestScaleToolRecovery(unittest.TestCase):
    def test_recovers_known_stretch(self):
        """ScaleTool recovers a known segment stretch and places markers to sub-mm error.

        A synthetic "subject" is the leg model stretched by known isotropic factors.
        Static markers sampled from that subject at the default pose must let
        ScaleTool recover the factors and fit the placed model to near-zero RMS.
        """
        path = _write_leg()
        subj = path + ".subject.osim"
        out = path + ".fit.osim"
        try:
            truth = {"thigh": 1.3, "shank": 0.8}
            osim.ModelScaler(osim.parse_osim(path), path).scale(
                {"thigh": (1.3, 1.3, 1.3), "shank": (0.8, 0.8, 0.8)}, subj
            )
            fk = ForwardKinematics(osim.parse_osim(subj))
            pos = fk.marker_positions(np.zeros(fk.ncoord))
            names = list(pos.keys())
            data = np.stack([np.array([pos[n] for n in names])], axis=0)
            static = MarkerData(times=np.array([0.0]), marker_names=names, data=data, rate=1.0)

            measurements = [
                osim.Measurement("thigh", [osim.MarkerPair("thigh_top", "thigh_bot")], {"thigh": "XYZ"}),
                osim.Measurement("shank", [osim.MarkerPair("shank_top", "ankle")], {"shank": "XYZ"}),
            ]
            res = osim.ScaleTool(path, measurements).run(static, out)
            self.assertAlmostEqual(res.scale_factors["thigh"][0], truth["thigh"], places=4)
            self.assertAlmostEqual(res.scale_factors["shank"][1], truth["shank"], places=4)
            self.assertLess(res.static_rms, 1e-3)
        finally:
            for p in (path, subj, out, out.rsplit(".", 1)[0] + "_static.trc"):
                if os.path.exists(p):
                    os.remove(p)


@unittest.skipUnless(
    os.environ.get("NEWTON_OPENSIM_C3D", "") and os.path.exists(os.environ.get("NEWTON_OPENSIM_C3D", "")),
    "set NEWTON_OPENSIM_C3D to a static-trial .c3d to test read_c3d (requires ezc3d)",
)
class TestReadC3D(unittest.TestCase):
    def test_read_c3d_static_trial(self):
        """read_c3d loads a static trial into meters in the OpenSim ground frame."""
        md = osim.read_c3d(os.environ["NEWTON_OPENSIM_C3D"])
        self.assertGreater(len(md.marker_names), 0)
        self.assertEqual(md.data.shape[2], 3)
        finite = md.data[np.isfinite(md.data)]
        self.assertLess(np.nanmax(np.abs(finite)), 10.0)  # meters, not millimeters


class TestSynthesizeMarkers(unittest.TestCase):
    def test_appends_centroid_markers(self):
        """synthesize_markers appends the per-frame centroid of the named source labels."""
        data = np.arange(1 * 4 * 3, dtype=float).reshape(1, 4, 3)
        md = MarkerData(times=np.array([0.0]), marker_names=["LPSI", "RPSI", "A", "B"], data=data, rate=1.0)
        out = synthesize_markers(md, {"V.Sacral": ("LPSI", "RPSI")})
        self.assertIn("V.Sacral", out.marker_names)
        expected = 0.5 * (md.data[:, 0, :] + md.data[:, 1, :])
        np.testing.assert_allclose(out.data[:, out.marker_names.index("V.Sacral"), :], expected)

    def test_ignores_occlusions_and_skips_insufficient(self):
        """A virtual marker averages only finite sources and is skipped below ``min_present``."""
        data = np.array([[[0.0, 0.0, 0.0], [np.nan, np.nan, np.nan], [2.0, 0.0, 0.0]]])
        md = MarkerData(times=np.array([0.0]), marker_names=["a", "b", "c"], data=data, rate=1.0)
        out = synthesize_markers(md, {"mid": ("a", "b", "c")}, min_present=2)
        # NaN 'b' is ignored; centroid of finite a,c.
        np.testing.assert_allclose(out.data[0, out.marker_names.index("mid"), :], [1.0, 0.0, 0.0])
        # Too few present -> not created; existing labels are never overwritten.
        self.assertEqual(synthesize_markers(md, {"z": ("a", "missing")}, min_present=2).marker_names, md.marker_names)
        self.assertEqual(synthesize_markers(md, {"a": ("b", "c")}).marker_names, md.marker_names)


if __name__ == "__main__":
    unittest.main()
