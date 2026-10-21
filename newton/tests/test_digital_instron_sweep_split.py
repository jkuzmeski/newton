# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for splitting Digital Instron foam sweeps."""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from projects.digital_instron_v2.frame_qc import read_numeric_csv
from projects.digital_instron_v2.sweep_split import _condition_rows, _measure_condition


class TestDigitalInstronSweepSplit(unittest.TestCase):
    def _write_trace(self, path: Path, loop_value: int) -> None:
        """Write synthetic cycles with a known force target and counter label."""
        columns = [
            "Total Time (s)",
            "Cycle Elapsed Time (s)",
            "Total Cycles",
            "Elapsed Cycles",
            "Step",
            "160ms Loop(1)",
            "200ms Loop(2)",
            "240ms Loop(3)",
            "280ms Loop(4)",
            "320ms Loop(5)",
            "Position (mm)",
            "Force (N)",
        ]
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(columns)
            for cycle in (1, *range(90, 101)):
                for sample in range(500):
                    t = sample / 1000.0
                    pulse = max(0.0, 1.0 - abs(t - 0.25) / 0.08)
                    compression_n = 20.0 + 780.0 * pulse
                    position = 10.0 - 2.0 * pulse
                    writer.writerow(
                        [cycle * 0.5 + t, t, cycle, cycle, 3, loop_value, 0, 0, 0, 0, position, -compression_n]
                    )

    def test_decode_counters_and_reject_a_mislabelled_force(self):
        """Decode a loop target and reject a counter whose signal peak disagrees."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.csv"
            self._write_trace(path, 1)
            columns = read_numeric_csv(path)
            rows = _condition_rows(columns, 160, 800)
            self.assertEqual(np.count_nonzero(rows), 12 * 500)
            measured = _measure_condition(columns, rows, 160, 800)
            self.assertAlmostEqual(measured["measured_peak_n"], 800.0)

            mislabelled_columns = {name: values.copy() for name, values in columns.items()}
            mislabelled_columns["160ms Loop(1)"][:] = 2
            mislabelled = _condition_rows(mislabelled_columns, 160, 1200)
            with self.assertRaisesRegex(ValueError, "Signal/counter mismatch"):
                _measure_condition(mislabelled_columns, mislabelled, 160, 1200)


if __name__ == "__main__":
    unittest.main()
