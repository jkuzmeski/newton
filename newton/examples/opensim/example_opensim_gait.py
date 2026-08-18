# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example OpenSim Gait
#
# Visualizes a measured walking motion on the OpenSim "gait2354" 23-DOF, 54-
# muscle musculoskeletal model. The coordinate trajectory (an inverse-kinematics
# result) is played back through the Newton-native, OpenSim-exact forward
# kinematics; the skeleton bones and muscle-tendon paths are rebuilt every frame
# entirely in Warp kernels (see :class:`newton.opensim.MotionVisualizer`), with
# muscles colored by their normalized muscle-tendon length so lengthening muscles
# light up over the gait cycle.
#
# This is a kinematic playback (no dynamics): each body's transform is set from
# the OpenSim-exact FK, which reproduces the CustomJoint SpatialTransform coupling
# (e.g. the SimmSpline knee translation) that Newton's generic joints do not.
#
# Pass --download-geometry (or --geometry <dir>) to render the actual OpenSim bone
# meshes (subject-scaled .vtp display geometry, skinned in Warp) instead of the
# stick figure.
#
# Command: python -m newton.examples opensim_gait
#          python -m newton.examples opensim_gait --download-geometry
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.opensim as osim


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args

        device = wp.get_device()

        # Parse the OpenSim model and load the walking coordinate trajectory.
        model_path = newton.examples.get_asset("gait2354_subject01.osim")
        motion_path = newton.examples.get_asset("gait2354_subject01_walk.mot")
        self.osim_model = osim.parse_osim(model_path)
        time, coords = osim.read_motion(self.osim_model, motion_path)

        # Precompute all per-frame renderables (body transforms, bones, muscle
        # paths + colors) in Warp.
        self.viz = osim.MotionVisualizer(self.osim_model, coords, time=time, device=device)
        self.num_frames = self.viz.num_frames

        # Optionally load the actual OpenSim bone meshes (.vtp display geometry) so
        # the skeleton renders as solid, subject-scaled bones instead of a stick
        # figure. Off by default so the example runs offline.
        geometry_dir = getattr(self.args, "geometry", None)
        if geometry_dir is None and getattr(self.args, "download_geometry", False):
            geometry_dir = osim.fetch_opensim_geometry()
        self.use_meshes = False
        if geometry_dir:
            loaded = self.viz.load_meshes(model_path, geometry_dir)
            self.use_meshes = loaded > 0
            print(
                f"[opensim_gait] loaded {loaded} bone meshes from {geometry_dir}"
                if loaded
                else f"[opensim_gait] no display meshes found under {geometry_dir}; showing stick figure"
            )

        dt = float(np.mean(np.diff(time))) if len(time) > 1 else 1.0 / 60.0
        self.frame_dt = dt
        self.fps = 1.0 / dt
        self.sim_time = float(time[0]) if len(time) else 0.0
        self.frame = 0

        # Build a Newton model as a render container: OpenSim bodies/joints (used
        # only to hold shapes) plus a visual site sphere at every body origin and
        # a ground plane. The playback drives body_q directly, so no solver runs.
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        osim.add_osim(builder, self.osim_model, parse_muscles=False, parse_contacts=False)
        for body in range(builder.body_count):
            builder.add_shape_sphere(body, radius=0.014, as_site=True, color=(0.9, 0.85, 0.55))
        builder.add_ground_plane()
        self.model = builder.finalize(device=device)

        self.state = self.model.state()
        # Per-frame body transforms aligned to this model's body order.
        self.body_q_frames = self.viz.body_transforms(self.model.body_label)
        wp.copy(self.state.body_q, self.body_q_frames[self.frame])

        self.viewer.set_model(self.model)

    def step(self):
        self.frame = (self.frame + 1) % self.num_frames
        wp.copy(self.state.body_q, self.body_q_frames[self.frame])
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        if self.use_meshes:
            self.viz.render_meshes(self.viewer, self.frame)
        else:
            self.viz.render_skeleton(self.viewer, self.frame)
        self.viz.render_muscles(self.viewer, self.frame)
        self.viewer.end_frame()

    def test_final(self):
        # The playback must produce finite body poses, keep the pelvis upright at
        # a plausible walking height, and advance it forward over the stride.
        body = {name: i for i, name in enumerate(self.model.body_label)}
        frames = self.body_q_frames.numpy()
        if not np.all(np.isfinite(frames)):
            raise AssertionError("non-finite body transforms in gait playback")

        pelvis = body["pelvis"]
        heights = frames[:, pelvis, 2]
        if not (0.7 < heights.min() and heights.max() < 1.3):
            raise AssertionError(f"pelvis height out of range: {heights.min()}..{heights.max()}")

        # A real stride must swing the legs: the feet clear the ground during
        # swing, so their vertical excursion over the cycle is substantial.
        foot_lift = max(float(np.ptp(frames[:, body[name], 1])) for name in ("calcn_r", "calcn_l") if name in body)
        if foot_lift < 0.05:
            raise AssertionError(f"feet do not lift over the stride: {foot_lift:.3f} m")

        # Muscle paths should have non-zero extent (they were rebuilt in Warp).
        starts = self.viz.muscle_starts.numpy()[0]
        ends = self.viz.muscle_ends.numpy()[0]
        if np.linalg.norm(starts - ends, axis=1).max() <= 0.0:
            raise AssertionError("muscle segments have zero length")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--geometry",
            type=str,
            default=None,
            help="Directory of OpenSim .vtp bone meshes; renders solid bones instead of the stick figure.",
        )
        parser.add_argument(
            "--download-geometry",
            action="store_true",
            help="Download the standard OpenSim bone meshes (opensim-models) and render solid bones.",
        )
        return parser


if __name__ == "__main__":
    # Parse arguments (including the mesh options) and initialize the viewer.
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)

    # Create viewer and run
    newton.examples.run(Example(viewer, args), args)
