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
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from projects.gait_c3d.c3d_adapter import c3d_to_marker_artifact, load_marker_artifact
from projects.gait_c3d.native_model import SimpleGaitConfig
from projects.gait_c3d.subject_mjcf import write_subject_mjcf
from projects.gait_c3d.vtp_adapter import compile_scaled_vtp_visuals, simple_config_from_scaled_gait2354


class Example:
    """Compile, load, and simulate one reusable subject model."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.frame_dt = 1.0 / 60.0
        self.sim_substeps = 10
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
        if args.scaled_osim:
            config = simple_config_from_scaled_gait2354(args.scaled_osim, body_height=args.body_height)
            print(
                f"Scale: {config.pelvis_mass + config.torso_mass + 2.0 * (config.thigh_mass + config.shank_mass + config.foot_mass):.3f} kg, "
                f"thigh {config.thigh_length:.3f} m, shank {config.shank_length:.3f} m"
            )
        self.marker_artifact = None
        self.device_markers = None
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

        visual_meshes = ()
        if bool(args.scaled_osim) != bool(args.geometry_dir):
            raise ValueError("--scaled-osim and --geometry-dir must be provided together")
        if args.scaled_osim:
            visuals = compile_scaled_vtp_visuals(
                args.scaled_osim,
                args.geometry_dir,
                self.model_dir,
                config,
            )
            visual_meshes = visuals.meshes
            print(f"VTP: {len(visual_meshes)} scaled visual meshes -> {visuals.root}")

        self.subject_xml = write_subject_mjcf(
            config,
            self.model_dir / "subject.xml",
            model_name=args.subject_name,
            visual_meshes=visual_meshes,
            include_fallback_geometry=not visual_meshes,
        )
        newton.use_coord_layout_targets = True
        builder = newton.ModelBuilder()
        builder.add_mjcf(str(self.subject_xml))
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
            f"{self.model.shape_count} shapes -> {self.subject_xml}"
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
        if self.model.body_count != 8 or self.model.joint_dof_count != 16:
            raise ValueError("subject model has an unexpected topology")
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()
        if not np.all(np.isfinite(body_q)) or not np.all(np.isfinite(body_qd)):
            raise ValueError("subject rollout produced nonfinite body state")
        root_force = self.control.joint_f.numpy()[:6]
        if not np.array_equal(root_force, np.zeros(6, dtype=root_force.dtype)):
            raise ValueError("free pelvis controls must remain exactly zero")
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
    parser.add_argument("--body-mass", type=float, default=81.4, help="Subject body mass [kg]")
    parser.add_argument("--body-height", type=float, default=1.695898298375747, help="Subject standing height [m]")
    parser.add_argument("--hip-width", type=float, default=0.152, help="Hip-joint center spacing [m]")
    parser.add_argument("--c3d", help="Optional calibration or dynamic C3D to decode directly")
    parser.add_argument("--c3d-up-axis", default="+Z", help="C3D lab axis pointing upward")
    parser.add_argument("--c3d-forward-axis", default="-Y", help="C3D lab axis pointing subject-forward")
    parser.add_argument("--scaled-osim", help="Optional accepted scaled gait2354 model for VTP visuals")
    parser.add_argument("--geometry-dir", help="Geometry directory containing referenced VTP files")
    return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(create_parser())
    newton.examples.run(Example(viewer, args), args)
