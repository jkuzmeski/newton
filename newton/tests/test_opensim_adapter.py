# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test the public OpenSim motion and frame adapters."""

import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

import newton.opensim as opensim


class TestOpenSimAdapter(unittest.TestCase):
    """Test frame, marker, and numeric storage conversion."""

    def test_load_namespace_lazily(self):
        """Keep core Newton imports isolated while preserving public access."""
        code = """
import sys
import newton
assert 'newton.opensim' not in sys.modules
from newton import opensim
assert opensim is newton.opensim
assert 'newton.opensim' in sys.modules
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_convert_world_vectors_at_the_boundary(self):
        """Map OpenSim Y-up vectors to Newton Z-up and back exactly."""
        converter = opensim.OpenSimFrameConverter()
        source = np.asarray([[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0]])
        converted = converter.transform_vectors(source)
        np.testing.assert_allclose(converted, [[1.0, -3.0, 2.0], [-4.0, 6.0, 5.0]])
        np.testing.assert_allclose(converter.inverse_vectors(converted), source)

    def test_round_trip_marker_data(self):
        """Preserve marker trajectories through TRC conversion."""
        data = np.arange(18, dtype=float).reshape(3, 2, 3) / 10.0
        markers = opensim.OpenSimMarkerData(
            times=np.arange(3) * 0.01,
            marker_names=["heel", "toe"],
            data=data,
            rate=100.0,
            units="m",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "markers.trc")
            opensim.write_trc(path, markers, units="mm")
            result = opensim.read_trc(path)
        self.assertEqual(result.marker_names, markers.marker_names)
        np.testing.assert_allclose(result.times, markers.times)
        np.testing.assert_allclose(result.data, markers.data, atol=1.0e-6)

    def test_round_trip_storage(self):
        """Preserve coordinate trajectories through MOT conversion."""
        times = np.linspace(0.0, 1.0, 5)
        data = np.column_stack((np.sin(times), np.cos(times)))
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "coordinates.mot")
            opensim.write_storage(path, times, ["q0", "q1"], data, name="coordinates")
            result = opensim.read_storage(path)
        self.assertIsInstance(result, opensim.OpenSimStorage)
        self.assertEqual(result.labels, ["q0", "q1"])
        np.testing.assert_allclose(result.times, times, atol=1.0e-7)
        np.testing.assert_allclose(result.data, data, atol=1.0e-7)

    def test_convert_centimeter_marker_data(self):
        """Convert centimeter TRC values to and from meters."""
        markers = opensim.OpenSimMarkerData(
            times=np.asarray([0.0, 0.01]),
            marker_names=["toe"],
            data=np.asarray([[[0.01, 0.02, 0.03]], [[0.04, 0.05, 0.06]]]),
            rate=100.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "markers.trc")
            opensim.write_trc(path, markers, units="cm")
            result = opensim.read_trc(path)
        self.assertEqual(result.units, "cm")
        np.testing.assert_allclose(result.data, markers.data)

    def test_reject_invalid_marker_data(self):
        """Reject unsupported units and inconsistent marker array shapes."""
        markers = opensim.OpenSimMarkerData(
            times=np.asarray([0.0]),
            marker_names=["toe"],
            data=np.zeros((1, 1, 2)),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "markers.trc")
            with self.assertRaisesRegex(ValueError, "marker arrays"):
                opensim.write_trc(path, markers)
            markers.data = np.zeros((1, 1, 3))
            with self.assertRaisesRegex(ValueError, "unsupported TRC"):
                opensim.write_trc(path, markers, units="inch")
            markers.times = np.asarray([0.1, 0.0])
            markers.data = np.zeros((2, 1, 3))
            markers.rate = 100.0
            with self.assertRaisesRegex(ValueError, "increase strictly"):
                opensim.write_trc(path, markers)

    def test_reject_corrupt_trc_rows(self):
        """Reject malformed values, row widths, and declared counts in TRC data."""
        valid = (
            "PathFileType\t4\t(X/Y/Z)\tmarkers.trc\n"
            "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n"
            "100\t100\t1\t1\tmm\t100\t1\t1\n"
            "Frame#\tTime\ttoe\t\t\n"
            "\t\tX1\tY1\tZ1\n"
            "1\t0.0\t1.0\t2.0\t3.0\n"
        )
        with self.assertRaisesRegex(ValueError, "marker value is invalid"):
            opensim.read_trc(valid.replace("3.0", "bad"))
        with self.assertRaisesRegex(ValueError, "has 4 fields; expected 5"):
            opensim.read_trc(valid.replace("1\t0.0\t1.0\t2.0\t3.0", "1\t0.0\t1.0\t2.0"))
        with self.assertRaisesRegex(ValueError, "declares 2 frames but contains 1"):
            opensim.read_trc(valid.replace("100\t100\t1\t1", "100\t100\t2\t1"))

    def test_reject_invalid_storage(self):
        """Reject malformed rows, nonmonotonic time, and shape mismatches."""
        malformed = """table
version=1
endheader
time q0
0.0 1.0
0.1
"""
        with self.assertRaisesRegex(ValueError, "has 1 values; expected 2"):
            opensim.read_storage(malformed)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "coordinates.mot")
            with self.assertRaisesRegex(ValueError, "storage times"):
                opensim.write_storage(path, np.asarray([0.1, 0.0]), ["q0"], np.zeros((2, 1)))
            with self.assertRaisesRegex(ValueError, "storage arrays"):
                opensim.write_storage(path, np.asarray([0.0]), ["q0"], np.zeros((1, 2)))
            with self.assertRaisesRegex(ValueError, "exclude time"):
                opensim.write_storage(path, np.asarray([0.0]), ["Time"], np.zeros((1, 1)))

    def test_default_storage_units_are_not_degrees(self):
        """Do not infer degree units when a storage header omits inDegrees."""
        storage = """table
version=1
endheader
time q0
0.0 1.0
"""
        result = opensim.read_storage(storage)
        self.assertFalse(result.in_degrees)

    def test_reject_invalid_vector_shape(self):
        """Reject frame conversion input without three-vector rows."""
        with self.assertRaisesRegex(ValueError, "trailing dimension of 3"):
            opensim.OpenSimFrameConverter().transform_vectors(np.zeros((2, 2)))


if __name__ == "__main__":
    unittest.main()
