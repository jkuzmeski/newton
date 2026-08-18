# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the human shoe manifest contract scaffold."""

import json
import tempfile
import unittest
from pathlib import Path

from projects.human_shoe import contracts


class TestHumanShoeContracts(unittest.TestCase):
    def test_load_manifest(self):
        """Load the Digital Instron manifest into validated contract objects."""
        manifest = contracts.load_manifest(Path("DigitalInstron/manifest_v2.json"))

        self.assertEqual(manifest.midsole_mesh, "puma-fast-r-nitro-elite-3-3d-internal-wt-LR.obj")
        self.assertEqual(manifest.cache_dir, "processed/v2_cache")
        self.assertEqual(manifest.grid.coarse_spacing_m, 0.005)
        self.assertEqual(manifest.fit.initial_equilibrium_fraction, 0.5)
        self.assertEqual(len(manifest.trials), 2)
        self.assertEqual(manifest.trials[0].indenter.radius_m, 0.022)
        self.assertEqual(
            manifest.trials[1].indenter.path, "Instron Shoe Last Size 9 6drop merged attachment 1 left.STL"
        )

    def test_rejects_invalid_grid_spacing(self):
        """Reject nonpositive grid spacings."""
        with self.assertRaisesRegex(ValueError, "coarse_spacing_m must be positive"):
            contracts.DigitalInstronGridContract(coarse_spacing_m=0.0, rearfoot_length_fraction=0.2)

    def test_rejects_invalid_trial_indenter_shape(self):
        """Reject indenter vectors that do not have three components."""
        with self.assertRaisesRegex(ValueError, "rotation_deg must have shape \\(3,\\)"):
            contracts.DigitalInstronIndenterContract(radius_m=0.02, rotation_deg=[90.0, 0.0])

    def test_rejects_missing_trials(self):
        """Reject manifests without any trials."""
        manifest_data = contracts.load_manifest(Path("DigitalInstron/manifest_v2.json"))
        with self.assertRaisesRegex(ValueError, "trials must not be empty"):
            contracts.DigitalInstronManifestContract(
                midsole_mesh=manifest_data.midsole_mesh,
                cache_dir=manifest_data.cache_dir,
                grid=manifest_data.grid,
                fit=manifest_data.fit,
                trials=(),
            )

    def test_loads_baseline_experiment(self):
        """Load the versioned human-shoe integration experiment template."""
        experiment = contracts.load_experiment(Path("experiments/human_shoe/baseline_gait2354.json"))

        self.assertEqual(experiment.schema_version, "human_shoe_experiment_1")
        self.assertEqual(experiment.attachment.foot_body_name, "calcn_r")
        self.assertEqual(experiment.attachment.translation_m.shape, (3,))
        self.assertEqual(
            experiment.contact_sidecar_path,
            "experiments/human_shoe/gait2354_subject01_contacts.json",
        )
        self.assertTrue(Path(experiment.human_model_path).is_file())
        self.assertEqual(experiment.controller_id, "gait2354_drop_pd_v1")
        self.assertEqual(experiment.time_step_s, 5.0e-5)
        self.assertEqual(experiment.random_seed, 0)
        self.assertEqual(experiment.motion_path, "newton/examples/assets/gait2354_subject01_walk.mot")
        self.assertEqual(experiment.initial_motion_frame, 0)

    def test_rejects_unknown_experiment_fields(self):
        """Reject misspelled top-level and attachment fields in experiment JSON."""
        source = json.loads(Path("experiments/human_shoe/baseline_gait2354.json").read_text())
        cases = []
        top_level = dict(source)
        top_level["time_steps_s"] = top_level["time_step_s"]
        cases.append(top_level)
        attachment = json.loads(json.dumps(source))
        attachment["attachment"]["translation_mm"] = [0.0, 0.0, 0.0]
        cases.append(attachment)

        for data in cases:
            with self.subTest(fields=sorted(data)), tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "experiment.json"
                path.write_text(json.dumps(data))
                with self.assertRaisesRegex(ValueError, "unknown fields"):
                    contracts.load_experiment(path)

    def test_rejects_unknown_experiment_schema(self):
        """Reject integration manifests with an unsupported schema version."""
        attachment = contracts.FootShoeAttachmentContract(
            foot_body_name="calcn_r",
            shoe_carrier_body_name="calcn_r",
            translation_m=[0.0, 0.0, 0.0],
            rotation_deg=[0.0, 0.0, 0.0],
        )
        with self.assertRaisesRegex(ValueError, "schema_version must be human_shoe_experiment_1"):
            contracts.HumanShoeExperimentContract(
                schema_version="human_shoe_experiment_0",
                human_model_path="human.osim",
                shoe_manifest_path="shoe.json",
                controller_id="test",
                attachment=attachment,
                time_step_s=0.001,
                random_seed=0,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
