# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test neutral motion-capture marker layout compilation."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import newton
from projects.gait_c3d.marker_layout import (
    compile_subject_marker_layout,
    load_subject_marker_layout,
    scale_subject_marker_layout_from_base,
)
from projects.gait_c3d.native_model import SimpleGaitConfig
from projects.gait_c3d.subject_mjcf import write_subject_mjcf
from projects.gait_c3d.vtp_adapter import simple_gait_body_transforms

_OPENSIM_TO_NEWTON = np.asarray(
    ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
    dtype=np.float64,
)
_SOURCE_TO_TARGET = {
    "pelvis": "pelvis",
    "torso": "torso",
    "femur_l": "femur_left",
    "femur_r": "femur_right",
    "tibia_l": "tibia_left",
    "tibia_r": "tibia_right",
    "calcn_l": "foot_left",
    "calcn_r": "foot_right",
}


def _reseal(manifest: dict) -> dict:
    """Replace a test manifest's content seal after a deliberate mutation."""
    manifest.pop("seal", None)
    content = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    manifest["seal"] = {"algorithm": "sha256", "content_sha256": hashlib.sha256(content).hexdigest()}
    return manifest


def _write_synthetic_sources(root: Path, config: SimpleGaitConfig, offset: float):
    """Write source transforms whose converted body frames match the native frames."""
    target_transforms = simple_gait_body_transforms(config)
    source_transforms = {}
    marker_lines = []
    expected = {}
    for index, (source_body, target_body) in enumerate(_SOURCE_TO_TARGET.items()):
        target = target_transforms[target_body]
        source = np.eye(4)
        source[:3, :3] = _OPENSIM_TO_NEWTON.T @ target[:3, :3]
        target_translation = target[:3, 3] - np.asarray((0.0, 0.0, offset))
        source[:3, 3] = _OPENSIM_TO_NEWTON.T @ target_translation
        source_transforms[source_body] = source.tolist()
        name = f"Marker.{index}"
        local = np.asarray((0.01 * (index + 1), -0.004 * index, 0.006 * (index - 2)))
        expected[name] = (target_body, local)
        marker_lines.append(
            f'<Marker name="{name}"><socket_parent_frame>/bodyset/{source_body}</socket_parent_frame>'
            f"<location>{local[0]} {local[1]} {local[2]}</location></Marker>"
        )
    marker_path = root / "adjusted_markers.xml"
    marker_path.write_text(
        "<OpenSimDocument><MarkerSet><objects>" + "".join(marker_lines) + "</objects></MarkerSet></OpenSimDocument>",
        encoding="utf-8",
    )
    transforms_path = root / "body_transforms.json"
    transforms_path.write_text(json.dumps(source_transforms), encoding="utf-8")
    return marker_path, transforms_path, expected


class TestSubjectMarkerLayout(unittest.TestCase):
    """Test sealed marker conversion and MJCF site import."""

    def test_converts_row_vector_frames_and_seals_layout(self):
        """Convert source markers into exact native body-local positions."""
        config = SimpleGaitConfig()
        offset = 0.037
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker_path, transforms_path, expected = _write_synthetic_sources(root, config, offset)
            layout = compile_subject_marker_layout(
                marker_path,
                transforms_path,
                config,
                root / "marker_layout.json",
                source_ground_offset_z=offset,
            )
            loaded = load_subject_marker_layout(layout.path)

        self.assertEqual(len(loaded.markers), len(expected))
        self.assertEqual(loaded.source_ground_offset_z, offset)
        for marker in loaded.markers:
            body, position = expected[marker.name]
            self.assertEqual(marker.body, body)
            np.testing.assert_allclose(marker.position, position, atol=1.0e-12)
            self.assertEqual(marker.site_name, f"marker_{marker.name}")

    def test_scales_marker_layout_from_base(self):
        """Scale S001-style marker positions and neutral frames together."""
        config = SimpleGaitConfig()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker_path, transforms_path, _ = _write_synthetic_sources(root, config, 0.02)
            base_path = root / "base_layout.json"
            base = compile_subject_marker_layout(
                marker_path,
                transforms_path,
                config,
                base_path,
                source_ground_offset_z=0.02,
            )
            scaled = scale_subject_marker_layout_from_base(
                base.path,
                root / "scaled_layout.json",
                length_scale=1.5,
            )
            base_manifest = json.loads(base.path.read_text(encoding="utf-8"))
            scaled_manifest = json.loads(scaled.path.read_text(encoding="utf-8"))
            base_layout_hash = hashlib.sha256(base.path.read_bytes()).hexdigest()
        self.assertEqual(len(scaled.markers), len(base.markers))
        self.assertAlmostEqual(scaled.source_ground_offset_z, 0.03)
        for marker, original in zip(scaled.markers, base.markers, strict=True):
            np.testing.assert_allclose(marker.position, 1.5 * np.asarray(original.position), atol=1.0e-12)
        for name, transform in scaled.target_body_transforms.items():
            expected_transform = base.target_body_transforms[name].copy()
            expected_transform[:3, 3] *= 1.5
            np.testing.assert_allclose(transform, expected_transform, atol=1.0e-12)
        self.assertEqual(scaled_manifest["derived_from"]["layout_file"], base.path.name)
        self.assertEqual(scaled_manifest["derived_from"]["layout_sha256"], base_layout_hash)
        self.assertEqual(
            [(marker["name"], marker["site_name"]) for marker in base_manifest["markers"]],
            [(marker["name"], marker["site_name"]) for marker in scaled_manifest["markers"]],
        )

    def test_rejects_tampered_layout(self):
        """Reject marker data changed without updating the content seal."""
        config = SimpleGaitConfig()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker_path, transforms_path, _ = _write_synthetic_sources(root, config, 0.0)
            layout = compile_subject_marker_layout(
                marker_path,
                transforms_path,
                config,
                root / "marker_layout.json",
                source_ground_offset_z=0.0,
            )
            manifest = json.loads(layout.path.read_text(encoding="utf-8"))
            manifest["markers"][0]["position_m"][0] += 1.0
            layout.path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "seal mismatch"):
                load_subject_marker_layout(layout.path)

    def test_hashes_inline_transforms_and_validates_provenance(self):
        """Hash inline transforms and reject self-sealed missing provenance."""
        config = SimpleGaitConfig()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker_path, transforms_path, _ = _write_synthetic_sources(root, config, 0.0)
            transforms = json.loads(transforms_path.read_text(encoding="utf-8"))
            layout = compile_subject_marker_layout(
                marker_path,
                transforms,
                config,
                root / "marker_layout.json",
                source_ground_offset_z=0.0,
            )
            manifest = json.loads(layout.path.read_text(encoding="utf-8"))
            self.assertIsNone(manifest["source"]["body_transforms_file"])
            self.assertRegex(manifest["source"]["body_transforms_sha256"], r"^[0-9a-f]{64}$")

            missing = json.loads(layout.path.read_text(encoding="utf-8"))
            del missing["source"]["marker_set_sha256"]
            layout.path.write_text(json.dumps(_reseal(missing)), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source provenance"):
                load_subject_marker_layout(layout.path)

            invalid = json.loads(json.dumps(manifest))
            invalid["source"]["body_transforms_sha256"] = "not-a-hash"
            layout.path.write_text(json.dumps(_reseal(invalid)), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source provenance"):
                load_subject_marker_layout(layout.path)

            missing_target = json.loads(json.dumps(manifest))
            del missing_target["target"]
            layout.path.write_text(json.dumps(_reseal(missing_target)), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "target transforms"):
                load_subject_marker_layout(layout.path)

    def test_imports_hidden_sites_with_native_bodies_and_offsets(self):
        """Import every marker as a hidden non-colliding MJCF site."""
        config = SimpleGaitConfig()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker_path, transforms_path, expected = _write_synthetic_sources(root, config, 0.0)
            layout = compile_subject_marker_layout(
                marker_path,
                transforms_path,
                config,
                root / "marker_layout.json",
                source_ground_offset_z=0.0,
            )
            subject_path = write_subject_mjcf(
                config,
                root / "subject.xml",
                marker_sites=layout.marker_sites,
            )
            builder = newton.ModelBuilder()
            builder.add_mjcf(str(subject_path), floating=True, parse_sites=True)
            model = builder.finalize(device="cpu")

        body_by_suffix = {label.rsplit("/", 1)[-1]: index for index, label in enumerate(model.body_label)}
        shape_flags = model.shape_flags.numpy()
        shape_body = model.shape_body.numpy()
        shape_transform = model.shape_transform.numpy()
        site_by_name = {
            label.rsplit("/", 1)[-1]: index
            for index, label in enumerate(model.shape_label)
            if shape_flags[index] & newton.ShapeFlags.SITE
        }
        self.assertEqual(set(site_by_name), {marker.site_name for marker in layout.markers})
        for marker in layout.markers:
            site = site_by_name[marker.site_name]
            self.assertEqual(shape_body[site], body_by_suffix[marker.body])
            np.testing.assert_allclose(shape_transform[site, :3], marker.position, atol=1.0e-7)
            self.assertFalse(shape_flags[site] & newton.ShapeFlags.VISIBLE)
            self.assertFalse(shape_flags[site] & newton.ShapeFlags.COLLIDE_SHAPES)
            expected_body, expected_position = expected[marker.name]
            self.assertEqual(marker.body, expected_body)
            np.testing.assert_allclose(marker.position, expected_position, atol=1.0e-12)

    def test_rejects_invalid_marker_definitions(self):
        """Reject duplicate, missing, nonfinite, unknown, and unmapped markers."""
        config = SimpleGaitConfig()
        cases = (
            (
                "duplicate",
                "<MarkerSet><Marker name='a'><body>pelvis</body><location>0 0 0</location></Marker>"
                "<Marker name='a'><body>pelvis</body><location>0 0 0</location></Marker></MarkerSet>",
                {"pelvis": np.eye(4).tolist()},
                "duplicate source marker",
            ),
            (
                "missing_location",
                "<MarkerSet><Marker name='a'><body>pelvis</body></Marker></MarkerSet>",
                {"pelvis": np.eye(4).tolist()},
                "three finite values",
            ),
            (
                "nonfinite",
                "<MarkerSet><Marker name='a'><body>pelvis</body><location>nan 0 0</location></Marker></MarkerSet>",
                {"pelvis": np.eye(4).tolist()},
                "three finite values",
            ),
            (
                "unsupported_body",
                "<MarkerSet><Marker name='a'><body>radius_r</body><location>0 0 0</location></Marker></MarkerSet>",
                {"radius_r": np.eye(4).tolist()},
                "unsupported source body",
            ),
            (
                "missing_transform",
                "<MarkerSet><Marker name='a'><body>torso</body><location>0 0 0</location></Marker></MarkerSet>",
                {"pelvis": np.eye(4).tolist()},
                "missing source transform",
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, marker_xml, transforms, message in cases:
                with self.subTest(name=name):
                    marker_path = root / f"{name}.xml"
                    marker_path.write_text(marker_xml, encoding="utf-8")
                    transforms_path = root / f"{name}.json"
                    transforms_path.write_text(json.dumps(transforms), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, message):
                        compile_subject_marker_layout(
                            marker_path,
                            transforms_path,
                            config,
                            root / f"{name}_layout.json",
                            source_ground_offset_z=0.0,
                        )


if __name__ == "__main__":
    unittest.main()
