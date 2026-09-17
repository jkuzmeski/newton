# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Keep timing-only search inputs explicit and protected from accidental overwrite."""

import argparse
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from projects.impedance_instron.cartesian.gpu.profile_search import _load_bundle, _positive, profile_search


class TestSearchProfileInputs(unittest.TestCase):
    """Reject incomplete manifests and invalid profiling budgets before GPU setup."""

    def test_positive_counts(self):
        """Reject nonpositive CLI counts."""
        self.assertEqual(_positive("3"), 3)
        for value in ("0", "-1"):
            with self.assertRaises(argparse.ArgumentTypeError):
                _positive(value)

    def test_bad_budgets_and_existing_output(self):
        """Reject invalid budgets and preserve existing output directories."""
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            for value in (0, -1, True, 1.5):
                with self.assertRaises(ValueError):
                    profile_search(directory, directory / "new", iterations=value)
                with self.assertRaises(ValueError):
                    profile_search(directory, directory / "new", repeats=value)
            with self.assertRaises(FileExistsError):
                profile_search(directory, directory)

    def test_incomplete_or_changed_bundle(self):
        """Verify required manifest coverage and reject changed input bytes."""
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            manifest = directory / "baseline.json"
            manifest.write_text(json.dumps({"files_sha256": {}}))
            with self.assertRaisesRegex(ValueError, "required inputs"):
                _load_bundle(directory)
            names = ("reference.npz", "profile.json", "equilibrium.npz", "summary.json", "digital_shoe.json")
            hashes = {}
            for name in names:
                (directory / name).write_bytes(b"original")
                hashes[name] = hashlib.sha256(b"original").hexdigest()
            manifest.write_text(json.dumps({"files_sha256": hashes}))
            (directory / "reference.npz").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Saved baseline file changed"):
                _load_bundle(directory)


if __name__ == "__main__":
    unittest.main()
