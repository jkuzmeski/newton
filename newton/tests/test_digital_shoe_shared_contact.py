# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the physics source provenance of the shared shoe mechanics."""

import unittest
from pathlib import Path
from unittest.mock import patch

from projects.digital_shoe.provenance import physics_source_identity


class TestSharedContact(unittest.TestCase):
    def test_physics_fingerprint_covers_shared_dependencies(self):
        """Invalidate checkpoint physics identity when any shared law source changes."""
        baseline = physics_source_identity()
        read = Path.read_bytes
        for filename in (
            "column_bed.py",
            "runtime.py",
            "material.py",
            "contact.py",
            "friction_maxwell.py",
        ):
            with self.subTest(filename=filename):

                def changed(path, target=filename):
                    data = read(path)
                    return data + b"\n# changed law" if path.name == target else data

                with patch.object(Path, "read_bytes", changed):
                    self.assertNotEqual(physics_source_identity(), baseline)
        self.assertEqual(physics_source_identity(), baseline)


if __name__ == "__main__":
    unittest.main()
