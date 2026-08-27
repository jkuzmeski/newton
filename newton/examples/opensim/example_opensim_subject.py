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

import shutil
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from projects.gait_c3d.c3d_adapter import c3d_to_marker_artifact, load_marker_artifact
from projects.gait_c3d.native_model import SimpleGaitConfig
from projects.gait_c3d.subject_mjcf import write_subject_mjcf
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


class Example:
    """Compile, load, and simulate one reusable subject model."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.frame_dt = 1.0 / 60.0
        self.sim_substeps = args.subject_substeps
        if self.sim_substeps <= 0:
            raise ValueError("--subject-substeps must be positive")
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.subject_dir = (
            Path(args.subject_dir).expanduser().resolve()
            if args.subject_dir
            else Path(tempfile.mkdtemp(prefix="newton-opensim-subject-"))
        )
        if self.subject_dir.exists() and any(self.subject_dir.iterdir()):
            if args.overwrite_subject_dir:
                shutil.rmtree(self.subject_dir)
            else:
                raise FileExistsError(
                    f"subject directory is not empty: {self.subject_dir}; pass --overwrite-subject-dir to rebuild"
                )
        self.subject_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir = self.subject_dir / "model"

        config = SimpleGaitConfig.for_subject(
            body_mass=args.body_mass,
            body_height=args.body_height,
            hip_width=args.hip_width,
        )
        if args.template_osim and args.scaled_osim:
            raise ValueError("provide either --template-osim or --scaled-osim, not both")
        self.marker_artifact = None
        self.device_markers = None
        markers = None
        if args.c3d:
            self.marker_artifact = c3d_to_marker_artifact(
                args.c3d,
                self.subject_dir / "markers",
                up_axis=args.c3d_up_axis,
                forward_axis=args.c3d_forward_axis,
            )
            markers = load_marker_artifact(self.marker_artifact)
            self.device_markers = markers.to_warp(args.device)
            print(
                f"C3D: {len(markers.times)} frames x {len(markers.marker_names)} markers "
                f"-> {self.marker_artifact / 'markers.npz'}"
            )

        scaled_osim = args.scaled_osim
        source_body_transforms = None
        inertial_data = None
        joint_centers = None
        self.marker_placement = None
        if args.template_osim:
            if markers is None:
                raise ValueError("--template-osim requires --c3d")
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
            raise ValueError("a scaled/template model path and --geometry-dir must be provided together")
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
                config = replace(
                    config,
                    pelvis_height=config.pelvis_height + contact_layout.root_height_offset_z,
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
        )
        newton.use_coord_layout_targets = True
        self.free_root = args.free_root
        self.show_self_collision = args.show_self_collision
        builder = newton.ModelBuilder()
        builder.add_mjcf(
            str(self.subject_xml),
            floating=True if self.free_root else False,
            enable_self_collisions=True,
            force_show_colliders=args.show_self_collision,
        )
        self.model = builder.finalize(device=args.device)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
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
        expected_dofs = 16 if self.free_root else 10
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
            connector = [
                index
                for index, label in enumerate(self.model.shape_label)
                if label.endswith("/geometry_abdomen_connector")
            ]
            if len(connector) != 1 or shape_types[connector[0]] != newton.GeoType.BOX:
                raise ValueError("scaled visual model must contain an abdomen connector")
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

    def render(self):
        """Render current body and contact state."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


def create_parser():
    """Create command-line arguments for the subject compiler example."""
    parser = newton.examples.create_parser()
    parser.add_argument("--subject-dir", help="Directory for MJCF, NPZ, manifest, and converted mesh outputs")
    parser.add_argument(
        "--overwrite-subject-dir",
        action="store_true",
        help="Delete a nonempty subject output directory before rebuilding",
    )
    parser.add_argument("--subject-name", default="example_subject", help="Saved MJCF model name")
    parser.add_argument(
        "--free-root",
        action="store_true",
        help="Run the unassisted six-DOF pelvis; default fixes the pelvis for standing model inspection",
    )
    parser.add_argument(
        "--show-self-collision",
        action="store_true",
        help="Show invisible segment self-collision proxies in the viewer",
    )
    parser.add_argument(
        "--subject-substeps",
        type=int,
        default=50,
        help="Featherstone/contact substeps per 60 Hz display frame",
    )
    parser.add_argument("--body-mass", type=float, default=81.4, help="Subject body mass [kg]")
    parser.add_argument("--body-height", type=float, default=1.695898298375747, help="Subject standing height [m]")
    parser.add_argument("--hip-width", type=float, default=0.152, help="Hip-joint center spacing [m]")
    parser.add_argument("--c3d", help="Optional calibration or dynamic C3D to decode directly")
    parser.add_argument("--c3d-up-axis", default="+Z", help="C3D lab axis pointing upward")
    parser.add_argument("--c3d-forward-axis", default="-Y", help="C3D lab axis pointing subject-forward")
    parser.add_argument("--template-osim", help="Pinned generic gait2354 model to scale directly from --c3d")
    parser.add_argument(
        "--scaling-backend",
        choices=("official", "parity"),
        default="official",
        help="Subject scaler: official OpenSim ScaleTool (default) or project parity implementation",
    )
    parser.add_argument("--scaled-osim", help="Optional accepted pre-scaled gait2354 model for VTP visuals")
    parser.add_argument("--scale-start", type=float, default=0.5, help="Static scaling window start [s]")
    parser.add_argument("--scale-end", type=float, default=1.0, help="Static scaling window end [s]")
    parser.add_argument(
        "--official-marker-placement",
        action="store_true",
        help="Run official OpenSim 4.6 MarkerPlacer as an offline oracle stage",
    )
    parser.add_argument("--geometry-dir", help="Geometry directory containing referenced VTP files")
    return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(create_parser())
    newton.examples.run(Example(viewer, args), args)
