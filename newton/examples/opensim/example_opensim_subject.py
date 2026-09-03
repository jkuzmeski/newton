# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example OpenSim Subject
#
# Proves the adapter boundary from optional C3D/VTP source data to a saved
# subject MJCF, then runs the result with Newton-native contact and Featherstone.
#
# Command: python -m newton.examples opensim_subject
#
###########################################################################

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from projects.gait_c3d.c3d_adapter import c3d_to_marker_artifact, load_marker_artifact
from projects.gait_c3d.calibrated_subject import write_calibrated_subject_mjcf
from projects.gait_c3d.marker_layout import (
    compile_subject_marker_layout,
    load_subject_marker_layout,
    scale_subject_marker_layout_from_base,
)
from projects.gait_c3d.marker_map import (
    NATIVE_MARKER_SOURCES,
    apply_c3d_marker_map,
    load_c3d_marker_map,
    load_subject_c3d_marker_map,
    required_c3d_sources,
    save_c3d_marker_map,
)
from projects.gait_c3d.native_model import SimpleGaitConfig
from projects.gait_c3d.segment_calibration import build_static_segment_calibration, load_static_segment_calibration
from projects.gait_c3d.subject_mjcf import scale_subject_mjcf_from_base, write_subject_mjcf
from projects.gait_c3d.subject_scaling import (
    build_subject_with_official_opensim,
    place_markers_with_official_opensim,
    scale_gait2354_from_markers,
)
from projects.gait_c3d.vtp_adapter import (
    compile_scaled_vtp_visuals,
    joint_centers_from_official_transforms,
    simple_config_from_scaled_gait2354,
    subject_inertials_from_scaled_gait2354,
)

_SUBJECT_BUNDLE_SCHEMA = "gait_subject_bundle_1"
_NATIVE_MARKER_REQUIREMENTS = required_c3d_sources(NATIVE_MARKER_SOURCES)
_PARITY_MARKER_REQUIREMENTS = (
    "LASI",
    "RASI",
    "LSHO",
    "RSHO",
    "STRN",
    "LKNE",
    "RKNE",
    "LANK",
    "RANK",
    "LHEE",
    "LHLX",
    "RHEE",
    "RHLX",
    "LPSI",
    "RPSI",
    "LFHD",
    "RFHD",
    "LBHD",
    "RBHD",
)


def _resolve_subject_artifact(subject_dir: Path, manifest: dict, name: str, *, required: bool = False) -> Path | None:
    """Resolve one declared bundle artifact without allowing path escape."""
    artifacts = manifest.get("artifacts")
    artifact = artifacts.get(name) if isinstance(artifacts, dict) else None
    if artifact is None and not required:
        return None
    if not isinstance(artifact, str) or not artifact or Path(artifact).is_absolute() or ".." in Path(artifact).parts:
        raise ValueError(f"subject bundle artifact {name!r} must be a safe relative path")
    path = (subject_dir / artifact).resolve()
    try:
        path.relative_to(subject_dir.resolve())
    except ValueError as error:
        raise ValueError(f"subject bundle artifact {name!r} escapes the bundle") from error
    if not path.exists():
        raise FileNotFoundError(f"subject bundle artifact {name!r} is missing: {path}")
    return path


def _clear_subject_directory_preserving_c3d(subject_dir: Path) -> None:
    """Clear generated bundle files without deleting subject-local C3D sources."""
    preserved = {path.resolve() for path in subject_dir.rglob("*") if path.is_file() and path.suffix.lower() == ".c3d"}

    def contains_preserved_file(directory: Path) -> bool:
        """Return whether a directory contains one of the preserved C3D files."""
        return any(path == directory or directory in path.parents for path in preserved)

    def clear(directory: Path) -> None:
        """Remove generated children while retaining preserved source files."""
        for child in directory.iterdir():
            resolved = child.resolve()
            if resolved in preserved:
                continue
            if child.is_dir():
                if contains_preserved_file(resolved):
                    clear(child)
                    try:
                        child.rmdir()
                    except OSError:
                        pass
                else:
                    shutil.rmtree(child)
            else:
                child.unlink()

    clear(subject_dir)


def _read_subject_bundle(subject_dir: Path) -> tuple[Path, dict]:
    """Read one compiled subject bundle and return its MJCF path and metadata."""
    manifest_path = subject_dir / "subject.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"subject bundle is missing {manifest_path}; rebuild it with source inputs or choose another subject"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != _SUBJECT_BUNDLE_SCHEMA:
        raise ValueError(f"unsupported subject bundle schema in {manifest_path}")
    model_path = _resolve_subject_artifact(subject_dir, manifest, "model", required=True)
    if model_path is None or not model_path.is_file():
        raise FileNotFoundError(f"subject bundle model is not a file: {model_path}")
    return model_path, manifest


def _write_subject_bundle_manifest(
    subject_dir: Path,
    *,
    args,
    config: SimpleGaitConfig,
    visual_mesh_count: int,
) -> None:
    """Publish the metadata needed to reopen one compiled subject folder."""
    artifacts = {"model": "model/subject.xml"}
    for name in ("markers", "opensim_subject"):
        path = subject_dir / name
        if path.is_dir() or path.is_file():
            artifacts[name] = name
    if (subject_dir / "marker_map.json").is_file():
        artifacts["marker_map"] = "marker_map.json"
    calibration_path = subject_dir / "calibration" / "segment_calibration.json"
    if calibration_path.is_file():
        artifacts["calibration"] = "calibration/segment_calibration.json"
    if (subject_dir / "model" / "manifest.json").is_file():
        artifacts["model_manifest"] = "model/manifest.json"
    if (subject_dir / "model" / "marker_layout.json").is_file():
        artifacts["marker_layout"] = "model/marker_layout.json"
    manifest = {
        "schema_version": _SUBJECT_BUNDLE_SCHEMA,
        "base_marker_set": getattr(args, "base_marker_set", None),
        "subject": {
            "name": args.subject_name,
            "mass_kg": float(
                sum(
                    (
                        config.pelvis_mass,
                        config.torso_mass,
                        2.0 * config.thigh_mass,
                        2.0 * config.shank_mass,
                        2.0 * config.foot_mass,
                    )
                )
            ),
            "height_m": float(args.body_height),
            "hip_width_m": float(2.0 * config.hip_half_width),
            "visual_mesh_count": int(visual_mesh_count),
        },
        "artifacts": artifacts,
        "sources": {
            "base_marker_set": getattr(args, "base_marker_set", None),
            "base_subject": Path(args.base_subject).name if args.base_subject else None,
            "static_calibration": Path(args.static_calibration).name if args.static_calibration else None,
            "c3d": Path(args.c3d).name if args.c3d else None,
            "marker_map": Path(args.marker_map).name if args.marker_map else None,
            "template": Path(args.template_osim).name if args.template_osim else None,
            "scaled_model": Path(args.scaled_osim).name if args.scaled_osim else None,
        },
    }
    if calibration_path.is_file():
        manifest["calibration"] = {
            "file": "calibration/segment_calibration.json",
            "sha256": hashlib.sha256(calibration_path.read_bytes()).hexdigest(),
        }
    marker_map_path = subject_dir / "marker_map.json"
    if marker_map_path.is_file():
        manifest["marker_mapping"] = {
            "file": marker_map_path.name,
            "sha256": hashlib.sha256(marker_map_path.read_bytes()).hexdigest(),
            "strip_prefix": not args.keep_c3d_prefix,
        }
    (subject_dir / "subject.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _publish_marker_map(marker_map_path: str | Path, subject_dir: Path):
    """Copy a validated normalized marker map into a subject bundle."""
    marker_map = load_c3d_marker_map(marker_map_path)
    saved = save_c3d_marker_map(marker_map, subject_dir / "marker_map.json")
    print(f"Marker map: {len(marker_map.markers)} aliases -> {saved}")
    return marker_map


def _apply_subject_marker_map(
    markers,
    marker_map_path: str | Path | None,
    subject_dir: Path,
    *,
    required: tuple[str, ...] = (),
):
    """Canonicalize decoded labels and publish the map with the subject."""
    if marker_map_path is None:
        return markers
    marker_map = load_c3d_marker_map(marker_map_path)
    canonical = apply_c3d_marker_map(markers, marker_map, required=required)
    save_c3d_marker_map(marker_map, subject_dir / "marker_map.json")
    print(f"Marker map: {len(marker_map.markers)} aliases -> {subject_dir / 'marker_map.json'}")
    return canonical


class Example:
    """Compile, load, and simulate one reusable subject model."""

    def _init_runtime(self, args):
        """Load the saved MJCF and initialize the common Newton runtime."""
        newton.use_coord_layout_targets = True
        self.free_root = args.free_root
        self.show_self_collision = args.show_self_collision
        self.show_markers = args.show_markers
        self.show_calibration = args.show_calibration
        builder = newton.ModelBuilder()
        builder.add_mjcf(
            str(self.subject_xml),
            floating=True if self.free_root else False,
            parse_sites=True,
            enable_self_collisions=True,
            force_show_colliders=self.show_self_collision,
        )
        self.model = builder.finalize(device=args.device)
        self.torso_dof_count = (
            3 if any(label.rsplit("/", 1)[-1].startswith("torso_flexion") for label in self.model.joint_label) else 0
        )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.neutral_body_q = self.state_0.body_q.numpy().copy()
        shape_flags = self.model.shape_flags.numpy()
        self.marker_site_indices = tuple(
            index
            for index, label in enumerate(self.model.shape_label)
            if shape_flags[index] & newton.ShapeFlags.SITE and label.rsplit("/", 1)[-1].startswith("marker_")
        )
        self.marker_points = None
        self.marker_radii = None
        self.marker_colors = None
        if self.marker_site_indices:
            marker_count = len(self.marker_site_indices)
            self.marker_points = wp.zeros(marker_count, dtype=wp.vec3, device=self.model.device)
            self.marker_radii = wp.full(marker_count, 0.012, dtype=wp.float32, device=self.model.device)
            self.marker_colors = wp.full(
                marker_count, wp.vec3(0.15, 0.95, 0.25), dtype=wp.vec3, device=self.model.device
            )
            self._update_marker_points()
        elif self.show_markers:
            raise ValueError(
                "--show-markers requires compiled marker sites; rebuild with --marker-demo, --base-subject, or source inputs"
            )
        self.calibration_point_array = None
        self.calibration_point_radii = None
        self.calibration_point_colors = None
        if self.show_calibration:
            if self.calibration_points is None:
                raise ValueError("--show-calibration requires --static-cal")
            points = np.asarray(self.calibration_points, dtype=np.float32)
            self.calibration_point_array = wp.array(points, dtype=wp.vec3, device=self.model.device)
            self.calibration_point_radii = wp.full(len(points), 0.016, dtype=wp.float32, device=self.model.device)
            colors = np.tile(np.asarray((0.95, 0.15, 0.10), dtype=np.float32), (len(points), 1))
            self.calibration_point_colors = wp.array(colors, dtype=wp.vec3, device=self.model.device)
        self.pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.pipeline.contacts()
        self.solver = newton.solvers.SolverFeatherstone(self.model, angular_damping=0.01)
        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(3.2, -3.2, 1.7), pitch=-5.0, yaw=135.0)
        print(
            f"Subject: {self.model.body_count} bodies, {self.model.joint_dof_count} DOFs, "
            f"{self.model.shape_count} shapes, root {'free' if self.free_root else 'fixed for standing inspection'} "
            f"-> {self.subject_xml}"
        )

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.frame_dt = 1.0 / 60.0
        self.sim_substeps = args.subject_substeps
        if self.sim_substeps <= 0:
            raise ValueError("--substeps must be positive")
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.calibration = None
        self.calibration_points = None
        default_subject_dir = (
            Path(__file__).resolve().parents[3] / "projects" / "gait_c3d" / "assets" / "s001_calibrated"
        )
        default_base_output_dir = (
            Path(__file__).resolve().parents[3] / "projects" / "gait_c3d" / "subjects" / "S001_scaled"
        )
        default_calibrated_output_dir = (
            Path(__file__).resolve().parents[3] / "projects" / "gait_c3d" / "subjects" / "calibrated_subject"
        )
        use_project_subject_dir = (
            args.subject_dir is None and not args.test and not args.base_subject and not args.static_calibration
        )
        self.subject_dir = (
            Path(args.subject_dir).expanduser().resolve()
            if args.subject_dir
            else default_base_output_dir
            if args.base_subject and not args.test
            else default_calibrated_output_dir
            if args.static_calibration and not args.test
            else default_subject_dir
            if use_project_subject_dir
            else Path(tempfile.mkdtemp(prefix="newton-opensim-subject-"))
        )
        compile_requested = (
            args.base_subject
            or args.marker_demo
            or args.static_calibration
            or any((args.c3d, args.template_osim, args.scaled_osim, args.geometry_dir))
        )
        if args.marker_map and not (args.static_calibration or args.c3d or args.base_subject):
            raise ValueError("--marker-map requires --static-cal, --c3d, or --base-subject")
        if args.keep_c3d_prefix and not (
            args.static_calibration or args.c3d or (args.base_subject and args.marker_map)
        ):
            raise ValueError("--keep-c3d-prefix requires C3D input or --base-subject with --marker-map")
        if args.static_calibration and args.c3d:
            raise ValueError("--static-cal cannot be combined with --c3d")
        if args.static_calibration and args.marker_demo:
            raise ValueError("--static-cal cannot be combined with --marker-demo")
        if args.static_calibration and any((args.template_osim, args.scaled_osim, args.geometry_dir)):
            raise ValueError("--static-cal cannot be combined with OpenSim or geometry source inputs")
        if args.base_subject and args.marker_demo:
            raise ValueError("--base-subject cannot be combined with --marker-demo")
        if args.base_subject and any((args.c3d, args.template_osim, args.scaled_osim, args.geometry_dir)):
            raise ValueError("--base-subject cannot be combined with C3D, OpenSim, or geometry source inputs")
        if args.marker_demo and any((args.c3d, args.template_osim, args.scaled_osim, args.geometry_dir)):
            raise ValueError("--marker-demo cannot be combined with subject source inputs")
        if (
            (args.subject_dir or use_project_subject_dir)
            and self.subject_dir.is_dir()
            and not compile_requested
            and not args.overwrite_subject_dir
        ):
            self.subject_xml, bundle_manifest = _read_subject_bundle(self.subject_dir)
            self.model_dir = self.subject_xml.parent
            subject_metadata = bundle_manifest.get("subject", {})
            self.visual_mesh_count = int(subject_metadata.get("visual_mesh_count", 0))
            self.inertial_data = None
            self.joint_centers = None
            self.marker_placement = None
            self.marker_artifact = None
            self.device_markers = None
            marker_layout_path = _resolve_subject_artifact(self.subject_dir, bundle_manifest, "marker_layout")
            self.marker_layout = (
                load_subject_marker_layout(marker_layout_path) if marker_layout_path is not None else None
            )
            marker_dir = _resolve_subject_artifact(self.subject_dir, bundle_manifest, "markers")
            if marker_dir is not None and (marker_dir / "manifest.json").is_file():
                self.marker_artifact = marker_dir
                self.device_markers = load_marker_artifact(marker_dir).to_warp(args.device)
            calibration_path = _resolve_subject_artifact(self.subject_dir, bundle_manifest, "calibration")
            if calibration_path is not None:
                if calibration_path.is_dir():
                    calibration_path = calibration_path / "segment_calibration.json"
                self.calibration = load_static_segment_calibration(calibration_path)
                model_manifest_path = _resolve_subject_artifact(self.subject_dir, bundle_manifest, "model_manifest")
                if model_manifest_path is not None:
                    model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
                    ground_offset = np.asarray(model_manifest.get("ground", {}).get("global_offset_m", (0.0, 0.0, 0.0)))
                    self.calibration_points = (
                        np.asarray(
                            [
                                self.calibration.pelvis["hip_centers_m"]["left"],
                                self.calibration.pelvis["hip_centers_m"]["right"],
                                self.calibration.segments["shank_left"]["proximal_m"],
                                self.calibration.segments["shank_right"]["proximal_m"],
                                self.calibration.segments["shank_left"]["distal_m"],
                                self.calibration.segments["shank_right"]["distal_m"],
                            ],
                            dtype=np.float64,
                        )
                        + ground_offset
                    )
            self._init_runtime(args)
            return
        if args.base_subject and Path(args.base_subject).expanduser().resolve() == self.subject_dir:
            raise ValueError("--base-subject and --subject must refer to different bundles")
        if self.subject_dir.exists() and any(self.subject_dir.iterdir()):
            if args.overwrite_subject_dir:
                _clear_subject_directory_preserving_c3d(self.subject_dir)
            else:
                raise FileExistsError(
                    f"subject directory is not empty: {self.subject_dir}; pass --overwrite to rebuild"
                )
        self.subject_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir = self.subject_dir / "model"

        if args.static_calibration:
            base_subject = (
                Path(args.base_subject).expanduser().resolve()
                if args.base_subject
                else (Path(__file__).resolve().parents[3] / "projects" / "gait_c3d" / "assets" / "s001_base")
            )
            base_manifest_path = base_subject / "subject.json"
            if not base_manifest_path.is_file():
                raise FileNotFoundError(f"base subject bundle is missing: {base_manifest_path}")
            base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
            base_metadata = base_manifest.get("subject", {})
            _, inherited_map, inherited_metadata = load_subject_c3d_marker_map(base_subject, base_manifest)
            marker_map_path = args.marker_map or inherited_map
            if args.marker_map is None and inherited_metadata is not None:
                args.keep_c3d_prefix = not inherited_metadata.get("strip_prefix", True)
            args.base_subject = str(base_subject)
            args.base_marker_set = base_manifest.get("base_marker_set") or base_manifest.get("sources", {}).get(
                "base_marker_set"
            )
            args.body_mass = float(args.body_mass) if args.body_mass is not None else float(base_metadata["mass_kg"])
            args.body_height = (
                float(args.body_height) if args.body_height is not None else float(base_metadata["height_m"])
            )
            marker_artifact = c3d_to_marker_artifact(
                args.static_calibration,
                self.subject_dir / "markers",
                up_axis=args.c3d_up_axis,
                forward_axis=args.c3d_forward_axis,
                strip_prefix=not args.keep_c3d_prefix,
            )
            markers = _apply_subject_marker_map(
                load_marker_artifact(marker_artifact),
                marker_map_path,
                self.subject_dir,
                required=_NATIVE_MARKER_REQUIREMENTS,
            )
            calibration_path = self.subject_dir / "calibration" / "segment_calibration.json"
            self.calibration = build_static_segment_calibration(
                markers,
                calibration_path,
                marker_radius=args.marker_radius,
                time_range=(args.calibration_start, args.calibration_end),
            )
            base_calibration = base_subject / "model" / "segment_calibration.json"
            if not base_calibration.is_file():
                raise FileNotFoundError(f"base segment calibration is missing: {base_calibration}")
            calibrated = write_calibrated_subject_mjcf(
                base_subject,
                self.calibration,
                self.model_dir / "subject.xml",
                body_mass=args.body_mass,
                model_name=args.subject_name,
                base_calibration=base_calibration,
            )
            calibrated_manifest = json.loads((self.model_dir / "manifest.json").read_text(encoding="utf-8"))
            ground_offset = np.asarray(calibrated_manifest["ground"]["global_offset_m"], dtype=np.float64)
            self.calibration_points = (
                np.asarray(
                    [
                        self.calibration.pelvis["hip_centers_m"]["left"],
                        self.calibration.pelvis["hip_centers_m"]["right"],
                        self.calibration.segments["shank_left"]["proximal_m"],
                        self.calibration.segments["shank_right"]["proximal_m"],
                        self.calibration.segments["shank_left"]["distal_m"],
                        self.calibration.segments["shank_right"]["distal_m"],
                    ],
                    dtype=np.float64,
                )
                + ground_offset
            )
            self.subject_xml = calibrated.path
            self.visual_mesh_count = int(base_metadata.get("visual_mesh_count", 0))
            self.marker_layout = None
            self.inertial_data = None
            self.joint_centers = None
            self.marker_placement = None
            self.marker_artifact = marker_artifact
            self.device_markers = markers.to_warp(args.device)
            print(
                f"Static calibration: {len(self.calibration.marker_positions)} markers, "
                f"CODA/Bell-Brand hips, flat feet -> {calibration_path}"
            )
            print(f"Calibrated MJCF: {self.subject_xml}")
            _write_subject_bundle_manifest(
                self.subject_dir,
                args=args,
                config=SimpleGaitConfig.for_subject(
                    body_mass=args.body_mass,
                    body_height=args.body_height,
                    hip_width=args.hip_width,
                ),
                visual_mesh_count=self.visual_mesh_count,
            )
            self._init_runtime(args)
            return

        if args.base_subject:
            base_subject = Path(args.base_subject).expanduser().resolve()
            base_manifest_path = base_subject / "subject.json"
            if not base_manifest_path.is_file():
                raise FileNotFoundError(f"base subject bundle is missing: {base_manifest_path}")
            base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
            base_metadata = base_manifest.get("subject", {})
            _, inherited_map, inherited_metadata = load_subject_c3d_marker_map(base_subject, base_manifest)
            if args.marker_map:
                _publish_marker_map(args.marker_map, self.subject_dir)
            elif inherited_map is not None:
                _publish_marker_map(inherited_map, self.subject_dir)
                if inherited_metadata is not None:
                    args.keep_c3d_prefix = not inherited_metadata.get("strip_prefix", True)
            args.base_marker_set = base_manifest.get("base_marker_set") or base_manifest.get("sources", {}).get(
                "base_marker_set"
            )
            args.body_mass = float(args.body_mass) if args.body_mass is not None else float(base_metadata["mass_kg"])
            args.body_height = (
                float(args.body_height) if args.body_height is not None else float(base_metadata["height_m"])
            )
            args.hip_width = float(args.hip_width) if args.hip_width is not None else None
            scaled = scale_subject_mjcf_from_base(
                base_subject,
                self.model_dir / "subject.xml",
                body_mass=args.body_mass,
                body_height=args.body_height,
                hip_width=args.hip_width,
                model_name=args.subject_name,
            )
            base_layout = base_subject / "model" / "marker_layout.json"
            if not base_layout.is_file():
                raise FileNotFoundError(f"base subject marker layout is missing: {base_layout}")
            self.marker_layout = scale_subject_marker_layout_from_base(
                base_layout,
                self.model_dir / "marker_layout.json",
                length_scale=scaled.length_scale,
                hip_width=2.0 * scaled.config.hip_half_width,
            )
            self.subject_xml = scaled.path
            self.visual_mesh_count = int(base_metadata.get("visual_mesh_count", 0))
            self.inertial_data = None
            self.joint_centers = None
            self.marker_placement = None
            self.marker_artifact = None
            self.device_markers = None
            print(
                f"Base subject: {base_subject} -> {self.subject_xml} "
                f"(length x{scaled.length_scale:.6g}, mass x{scaled.mass_scale:.6g})"
            )
            print(f"Markers: {len(self.marker_layout.markers)} S001 sites -> {self.marker_layout.path}")
            _write_subject_bundle_manifest(
                self.subject_dir,
                args=args,
                config=scaled.config,
                visual_mesh_count=self.visual_mesh_count,
            )
            self._init_runtime(args)
            return

        args.body_mass = float(args.body_mass) if args.body_mass is not None else 81.4
        args.body_height = float(args.body_height) if args.body_height is not None else 1.695898298375747
        config = SimpleGaitConfig.for_subject(
            body_mass=args.body_mass,
            body_height=args.body_height,
            hip_width=args.hip_width,
        )
        if args.template_osim and args.scaled_osim:
            raise ValueError("provide either --template or --scaled-osim, not both")
        self.marker_artifact = None
        self.device_markers = None
        markers = None
        if args.c3d:
            self.marker_artifact = c3d_to_marker_artifact(
                args.c3d,
                self.subject_dir / "markers",
                up_axis=args.c3d_up_axis,
                forward_axis=args.c3d_forward_axis,
                strip_prefix=not args.keep_c3d_prefix,
            )
            required_markers = (
                _NATIVE_MARKER_REQUIREMENTS
                if (args.scaling_backend == "official" and args.template_osim) or args.official_marker_placement
                else _PARITY_MARKER_REQUIREMENTS
                if args.template_osim
                else ()
            )
            markers = _apply_subject_marker_map(
                load_marker_artifact(self.marker_artifact),
                args.marker_map,
                self.subject_dir,
                required=required_markers,
            )
            self.device_markers = markers.to_warp(args.device)
            print(
                f"C3D: {len(markers.times)} frames x {len(markers.marker_names)} markers "
                f"-> {self.marker_artifact / 'markers.npz'}"
            )

        scaled_osim = args.scaled_osim
        source_body_transforms = None
        source_ground_offset_z = 0.0
        inertial_data = None
        joint_centers = None
        self.marker_placement = None
        if args.template_osim:
            if markers is None:
                raise ValueError("--template requires --c3d")
            if args.scaling_backend == "official":
                official = build_subject_with_official_opensim(
                    markers,
                    args.template_osim,
                    self.subject_dir / "opensim_subject",
                    subject_mass=args.body_mass,
                    subject_height=args.body_height,
                    time_range=(args.scale_start, args.scale_end),
                )
                scaled_osim = str(official.scaled_model_path)
                source_body_transforms = official.body_transforms_path
                self.marker_placement = official
                print(
                    f"Official ScaleTool: {len(official.scale_factors)} body factors, "
                    f"MarkerPlacer RMS {official.marker_rms:.4f} m, max {official.marker_max:.4f} m "
                    f"-> {official.placed_model_path}"
                )
            else:
                scaling = scale_gait2354_from_markers(
                    markers,
                    args.template_osim,
                    self.subject_dir / "scaling_parity",
                    subject_mass=args.body_mass,
                    time_range=(args.scale_start, args.scale_end),
                )
                scaled_osim = str(scaling.model_path)
                print(f"Parity scaler: {len(scaling.scale_factors)} body factors -> {scaling.model_path}")
        if scaled_osim:
            config = simple_config_from_scaled_gait2354(scaled_osim, body_height=args.body_height)
            print(
                f"Scale: {config.pelvis_mass + config.torso_mass + 2.0 * (config.thigh_mass + config.shank_mass + config.foot_mass):.3f} kg, "
                f"thigh {config.thigh_length:.3f} m, shank {config.shank_length:.3f} m"
            )
            if source_body_transforms is not None:
                inertial_data = subject_inertials_from_scaled_gait2354(
                    scaled_osim,
                    config,
                    source_body_transforms,
                )
                print("Inertia: official OpenSim COM/full tensors mapped to all 8 Newton bodies")

        if args.official_marker_placement and self.marker_placement is None:
            if markers is None or not scaled_osim:
                raise ValueError("--official-marker-placement requires --c3d and a scaled model")
            self.marker_placement = place_markers_with_official_opensim(
                markers,
                scaled_osim,
                self.subject_dir / "opensim_marker_placement",
                subject_mass=args.body_mass,
                subject_height=args.body_height,
                time_range=(args.scale_start, args.scale_end),
            )
            print(
                f"MarkerPlacer: RMS {self.marker_placement.marker_rms:.4f} m, "
                f"max {self.marker_placement.marker_max:.4f} m -> {self.marker_placement.model_path}"
            )

        visual_meshes = ()
        contact_layout = None
        if bool(scaled_osim) != bool(args.geometry_dir):
            raise ValueError("a scaled/template model path and --geometry must be provided together")
        if scaled_osim:
            visuals = compile_scaled_vtp_visuals(
                scaled_osim,
                args.geometry_dir,
                self.model_dir,
                config,
                source_body_transforms=source_body_transforms,
            )
            visual_meshes = visuals.meshes
            contact_layout = visuals.contact_layout
            print(f"VTP: {len(visual_meshes)} scaled visual meshes -> {visuals.root}")
            if contact_layout is not None:
                source_ground_offset_z = contact_layout.root_height_offset_z
                config = replace(
                    config,
                    pelvis_height=config.pelvis_height + source_ground_offset_z,
                )
                if source_body_transforms is not None:
                    joint_centers = joint_centers_from_official_transforms(
                        config,
                        source_body_transforms,
                        source_ground_offset_z=contact_layout.root_height_offset_z,
                    )
                print(
                    f"Contact: radius {contact_layout.radius:.4f} m, root height offset "
                    f"{contact_layout.root_height_offset_z:.4f} m"
                )

        self.marker_layout = None
        if self.marker_placement is not None and source_body_transforms is not None:
            marker_set_path = self.marker_placement.marker_set_path
            marker_source_transforms = source_body_transforms
        elif args.marker_demo:
            reference = SimpleGaitConfig()
            if not np.isclose(config.pelvis_height, reference.pelvis_height) or not np.isclose(
                config.hip_half_width, reference.hip_half_width
            ):
                raise ValueError("--marker-demo requires the default subject height and hip width")
            demo_root = Path(__file__).resolve().parents[3] / "projects" / "gait_c3d" / "assets" / "marker_layout_demo"
            marker_set_path = demo_root / "adjusted_markers.xml"
            marker_source_transforms = demo_root / "body_transforms.json"
        else:
            marker_set_path = None
            marker_source_transforms = None
        if marker_set_path is not None:
            self.marker_layout = compile_subject_marker_layout(
                marker_set_path,
                marker_source_transforms,
                config,
                self.model_dir / "marker_layout.json",
                source_ground_offset_z=source_ground_offset_z,
            )
            print(f"Markers: {len(self.marker_layout.markers)} native MJCF sites -> {self.marker_layout.path}")

        self.visual_mesh_count = len(visual_meshes)
        self.inertial_data = inertial_data
        self.joint_centers = joint_centers
        self.subject_xml = write_subject_mjcf(
            config,
            self.model_dir / "subject.xml",
            model_name=args.subject_name,
            visual_meshes=visual_meshes,
            include_fallback_geometry=not visual_meshes,
            contact_centers=contact_layout.centers if contact_layout is not None else None,
            contact_radius=contact_layout.radius if contact_layout is not None else None,
            inertial_data=inertial_data,
            joint_centers=joint_centers,
            marker_sites=self.marker_layout.marker_sites if self.marker_layout is not None else (),
        )
        _write_subject_bundle_manifest(
            self.subject_dir,
            args=args,
            config=config,
            visual_mesh_count=len(visual_meshes),
        )
        self._init_runtime(args)

    def _update_marker_points(self):
        """Update the viewer overlay from imported body-local MJCF sites."""
        if self.marker_points is None:
            return
        body_q = self.state_0.body_q.numpy()
        shape_body = self.model.shape_body.numpy()
        shape_transform = self.model.shape_transform.numpy()
        points = np.empty((len(self.marker_site_indices), 3), dtype=np.float32)
        for output_index, shape_index in enumerate(self.marker_site_indices):
            body_transform = wp.transform(*body_q[shape_body[shape_index]])
            site_transform = wp.transform(*shape_transform[shape_index])
            position = wp.transform_get_translation(body_transform * site_transform)
            points[output_index] = (position[0], position[1], position[2])
        self.marker_points.assign(points)

    def simulate(self):
        """Advance one display frame through native Newton APIs."""
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        """Advance one example frame."""
        self.simulate()
        self.sim_time += self.frame_dt

    def test_final(self):
        """Verify model structure, root policy, artifacts, and finite state."""
        if not self.subject_xml.is_file():
            raise ValueError("subject MJCF was not published")
        expected_dofs = (16 if self.free_root else 10) + self.torso_dof_count
        if self.model.body_count != 8 or self.model.joint_dof_count != expected_dofs:
            raise ValueError("subject model has an unexpected topology")
        shape_types = self.model.shape_type.numpy()
        shape_flags = self.model.shape_flags.numpy()
        if self.show_self_collision:
            self_collision_proxies = [
                index
                for index, label in enumerate(self.model.shape_label)
                if label.rsplit("/", 1)[-1].startswith("collision_")
            ]
            if len(self_collision_proxies) != 6 or any(
                not shape_flags[index] & newton.ShapeFlags.VISIBLE for index in self_collision_proxies
            ):
                raise ValueError("self-collision proxies were not made visible")
        contact_spheres = [
            index
            for index, label in enumerate(self.model.shape_label)
            if "/contact_left_" in label or "/contact_right_" in label
        ]
        if len(contact_spheres) != 8 or any(shape_types[index] != newton.GeoType.SPHERE for index in contact_spheres):
            raise ValueError("subject model must contain eight foot contact spheres")
        visible_feet = [index for index, label in enumerate(self.model.shape_label) if "/visual_foot_" in label]
        if len(visible_feet) != 8 or any(shape_types[index] != newton.GeoType.SPHERE for index in visible_feet):
            raise ValueError("subject viewer must contain eight visible foot spheres")
        ground = [index for index, label in enumerate(self.model.shape_label) if label.endswith("/ground")]
        if len(ground) != 1 or shape_types[ground[0]] != newton.GeoType.PLANE:
            raise ValueError("subject model must contain one ground plane")
        visual_ground = [
            index for index, label in enumerate(self.model.shape_label) if label.endswith("/visual_ground")
        ]
        if len(visual_ground) != 1 or shape_types[visual_ground[0]] != newton.GeoType.PLANE:
            raise ValueError("subject viewer must contain one visible ground plane")
        if self.visual_mesh_count:
            if any(label.endswith("/geometry_abdomen_connector") for label in self.model.shape_label):
                raise ValueError("scaled visual model must not contain an abdomen connector")
        if self.inertial_data is not None:
            body_mass = self.model.body_mass.numpy()
            body_com = self.model.body_com.numpy()
            for name, expected in self.inertial_data.items():
                body = next(index for index, label in enumerate(self.model.body_label) if label.endswith(f"/{name}"))
                if abs(float(body_mass[body]) - expected.mass) > 2.0e-5:
                    raise ValueError(f"OpenSim-derived mass changed for {name}")
                if not np.allclose(body_com[body], expected.position, atol=1.0e-6):
                    raise ValueError(f"OpenSim-derived COM changed for {name}")
        if self.joint_centers is not None:
            joint_xform_child = self.model.joint_X_c.numpy()
            for name, expected in self.joint_centers.items():
                if name.startswith("hip_"):
                    side = name.removeprefix("hip_")
                    joint = next(
                        index for index, label in enumerate(self.model.joint_label) if f"/hip_flexion_{side}_" in label
                    )
                else:
                    joint = next(
                        index for index, label in enumerate(self.model.joint_label) if label.endswith(f"/{name}")
                    )
                if not np.allclose(joint_xform_child[joint, :3], expected, atol=1.0e-6):
                    raise ValueError(f"official joint center changed for {name}")
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()
        if not np.all(np.isfinite(body_q)) or not np.all(np.isfinite(body_qd)):
            raise ValueError("subject rollout produced nonfinite body state")
        if not self.free_root:
            if np.any(np.abs(body_q[:, :2]) > 1.0) or np.any(body_q[:, 2] < -0.05) or np.any(body_q[:, 2] > 2.5):
                raise ValueError("standing inspection moved one or more bodies outside the visible subject bounds")
        if self.free_root:
            root_force = self.control.joint_f.numpy()[:6]
            if not np.array_equal(root_force, np.zeros(6, dtype=root_force.dtype)):
                raise ValueError("free pelvis controls must remain exactly zero")
        elif body_q[0, 2] < 0.9:
            raise ValueError("fixed-root standing inspection lost pelvis height")
        if self.marker_placement is not None:
            placement_model = getattr(
                self.marker_placement,
                "placed_model_path",
                getattr(self.marker_placement, "model_path", None),
            )
            if (
                placement_model is None
                or not placement_model.is_file()
                or not self.marker_placement.manifest_path.is_file()
            ):
                raise ValueError("official OpenSim MarkerPlacer artifacts are missing")
            if not np.isfinite(self.marker_placement.marker_rms) or not np.isfinite(self.marker_placement.marker_max):
                raise ValueError("official OpenSim MarkerPlacer metrics are nonfinite")
            if hasattr(self.marker_placement, "marker_rms_limit") and (
                self.marker_placement.marker_rms > self.marker_placement.marker_rms_limit
                or self.marker_placement.marker_max > self.marker_placement.marker_max_limit
            ):
                raise ValueError("official OpenSim MarkerPlacer failed its engineering QC gate")
        if self.marker_artifact is not None:
            if not (self.marker_artifact / "manifest.json").is_file() or self.device_markers is None:
                raise ValueError("C3D marker artifact or Warp upload is missing")
            if self.device_markers.positions.shape[0] == 0:
                raise ValueError("C3D Warp marker array is empty")
        if self.marker_layout is not None:
            site_by_name = {
                self.model.shape_label[index].rsplit("/", 1)[-1]: index for index in self.marker_site_indices
            }
            if set(site_by_name) != {marker.site_name for marker in self.marker_layout.markers}:
                raise ValueError("subject marker layout does not match the imported MJCF sites")
            body_by_name = {label.rsplit("/", 1)[-1]: index for index, label in enumerate(self.model.body_label)}
            for body_name, expected_transform in self.marker_layout.target_body_transforms.items():
                body = body_by_name.get(body_name)
                if body is None:
                    raise ValueError(f"subject marker layout target body is missing: {body_name!r}")
                body_q = self.neutral_body_q[body]
                actual_rotation = np.asarray(wp.quat_to_matrix(wp.quat(*body_q[3:]))).reshape(3, 3)
                if not np.allclose(body_q[:3], expected_transform[:3, 3], atol=1.0e-6) or not np.allclose(
                    actual_rotation, expected_transform[:3, :3], atol=1.0e-6
                ):
                    raise ValueError(f"neutral target transform changed for marker body {body_name!r}")
            shape_body = self.model.shape_body.numpy()
            shape_transform = self.model.shape_transform.numpy()
            for marker in self.marker_layout.markers:
                site = site_by_name[marker.site_name]
                body_name = self.model.body_label[shape_body[site]].rsplit("/", 1)[-1]
                if body_name != marker.body or not np.allclose(shape_transform[site, :3], marker.position, atol=1.0e-7):
                    raise ValueError(f"imported MJCF site changed for marker {marker.name!r}")
            if self.show_markers and self.marker_points is None:
                raise ValueError("subject marker overlay was not initialized")

    def render(self):
        """Render current body and contact state."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        if self.show_markers and self.marker_points is not None:
            self._update_marker_points()
            self.viewer.log_points(
                "subject/neutral_marker_sites",
                self.marker_points,
                self.marker_radii,
                self.marker_colors,
            )
        if self.show_calibration and self.calibration_point_array is not None:
            self.viewer.log_points(
                "subject/calibration_joint_centers",
                self.calibration_point_array,
                self.calibration_point_radii,
                self.calibration_point_colors,
            )
        self.viewer.end_frame()


def create_parser():
    """Create the concise command-line interface for the subject example."""
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--subject",
        dest="subject_dir",
        help="Compiled subject folder to load or rebuild",
    )
    parser.add_argument(
        "--subject-dir",
        dest="subject_dir",
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--overwrite",
        dest="overwrite_subject_dir",
        action="store_true",
        help="Replace an existing subject output directory",
    )
    parser.add_argument(
        "--overwrite-subject-dir",
        dest="overwrite_subject_dir",
        action="store_true",
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--free-root",
        action="store_true",
        help="Run the unassisted six-DOF pelvis; default fixes the pelvis for standing inspection",
    )
    parser.add_argument(
        "--show-collision",
        dest="show_self_collision",
        action="store_true",
        help="Show the orange self-collision proxies (hidden by default)",
    )
    parser.add_argument(
        "--show-markers",
        action="store_true",
        help="Show compiled neutral motion-capture marker sites",
    )
    parser.add_argument(
        "--marker-demo",
        action="store_true",
        help="Build the tracked compact neutral-marker demonstration subject",
    )
    parser.add_argument(
        "--base-subject",
        help="Scale a compiled subject bundle, normally S001, with its real bone meshes and marker layout",
    )
    parser.add_argument(
        "--static-cal",
        dest="static_calibration",
        help="Build a calibrated native subject from a static calibration C3D",
    )
    parser.add_argument(
        "--show-calibration",
        action="store_true",
        help="Show calibrated hip, knee, and ankle centers (requires --static-cal)",
    )
    parser.add_argument(
        "--show-self-collision",
        dest="show_self_collision",
        action="store_true",
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--substeps",
        dest="subject_substeps",
        type=int,
        default=50,
        help="Physics substeps per 60 Hz display frame",
    )
    parser.add_argument(
        "--subject-substeps",
        dest="subject_substeps",
        type=int,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--mass", dest="body_mass", type=float, default=None, help="Subject body mass [kg]")
    parser.add_argument(
        "--body-mass",
        dest="body_mass",
        type=float,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--height",
        dest="body_height",
        type=float,
        default=None,
        help="Subject standing height [m]",
    )
    parser.add_argument(
        "--body-height",
        dest="body_height",
        type=float,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--c3d", help="Optional calibration or dynamic C3D file")
    parser.add_argument(
        "--marker-map",
        help="Exact C3D label map created by the marker_mapper example",
    )
    parser.add_argument(
        "--keep-c3d-prefix",
        action="store_true",
        help="Keep subject prefixes such as Person01:LASI when applying the marker map",
    )
    parser.add_argument("--template", dest="template_osim", help="OpenSim template to scale from --c3d")
    parser.add_argument(
        "--template-osim",
        dest="template_osim",
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--geometry", dest="geometry_dir", help="OpenSim VTP geometry directory")
    parser.add_argument(
        "--geometry-dir",
        dest="geometry_dir",
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )

    # Keep reproducibility controls accepted without crowding the normal help.
    parser.add_argument("--subject-name", default="example_subject", help=argparse.SUPPRESS)
    parser.add_argument("--hip-width", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--marker-radius", type=float, default=0.006, help="Static marker radius [m]")
    parser.add_argument("--calibration-start", type=float, default=0.5, help=argparse.SUPPRESS)
    parser.add_argument("--calibration-end", type=float, default=1.0, help=argparse.SUPPRESS)
    parser.add_argument("--c3d-up-axis", default="+Z", help=argparse.SUPPRESS)
    parser.add_argument("--c3d-forward-axis", default="-Y", help=argparse.SUPPRESS)
    parser.add_argument(
        "--scaling-backend",
        choices=("official", "parity"),
        default="official",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--scaled-osim", help=argparse.SUPPRESS)
    parser.add_argument("--scale-start", type=float, default=0.5, help=argparse.SUPPRESS)
    parser.add_argument("--scale-end", type=float, default=1.0, help=argparse.SUPPRESS)
    parser.add_argument("--official-marker-placement", action="store_true", help=argparse.SUPPRESS)
    return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(create_parser())
    newton.examples.run(Example(viewer, args), args)
