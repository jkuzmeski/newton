# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example OpenSim Muscle Activity
#
# Plays the measured "gait2354" walking motion (23-DOF, 54-muscle model) and
# colors every muscle-tendon path by its Static-Optimization activation, so the
# muscles light up (blue -> yellow -> red) as they are recruited over the gait
# cycle.
#
# The activations are computed with the Newton-native, differentiable
# Static-Optimization solver (:func:`newton.opensim.solve_static_optimization`),
# which resolves the muscle-redundancy problem frame by frame by minimizing the
# sum of squared activations subject to the inverse-dynamics moment balance. To
# keep startup fast the optimization is run on a handful of frames spread across
# the stride (``--so-frames``) and the result is resampled onto the smooth
# playback frames by :meth:`newton.opensim.MotionVisualizer.color_muscles_by`.
#
# No ground-reaction data ships with the model, so during stance the balance is
# carried by reserve actuators and the muscle activity is only illustrative;
# the point of the example is the analysis-driven visualization overlay.
#
# Command: python -m newton.examples opensim_muscle_activity
#          python -m newton.examples opensim_muscle_activity --so-frames 12
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
        # paths + colors) in Warp for the full, smooth trajectory.
        self.viz = osim.MotionVisualizer(self.osim_model, coords, time=time, device=device)
        self.num_frames = self.viz.num_frames

        # Resolve muscle activations with Static Optimization on a coarse set of
        # frames spanning the stride, then recolor the muscle paths by them.
        self.activations = self._solve_activations(model_path, motion_path)

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

    def _solve_activations(self, model_path, motion_path):
        """Run Static Optimization on a decimated stride and color the muscles by it."""
        storage = osim.read_storage(motion_path)
        n = len(storage.times)
        so_frames = int(max(2, min(getattr(self.args, "so_frames", 10), n)))
        idx = np.linspace(0, n - 1, so_frames).round().astype(int)
        decimated = osim.Storage(
            times=np.asarray(storage.times)[idx],
            labels=storage.labels,
            data=storage.data[idx],
            in_degrees=storage.in_degrees,
            name="gait_decimated",
        )
        print(f"[opensim_muscle_activity] running Static Optimization on {so_frames} frames...")
        result = osim.solve_static_optimization(self.osim_model, decimated)
        # Reorder the activation columns to the visualizer's muscle order.
        col = {name: i for i, name in enumerate(result.muscle_names)}
        order = [col[name] for name in self.viz.muscle_names]
        activations = result.activations[:, order]
        self.viz.color_muscles_by(activations, times=result.times, vmin=0.0, vmax=1.0)
        return activations

    def step(self):
        self.frame = (self.frame + 1) % self.num_frames
        wp.copy(self.state.body_q, self.body_q_frames[self.frame])
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        self.viz.render_skeleton(self.viewer, self.frame)
        self.viz.render_muscles(self.viewer, self.frame)
        self.viewer.end_frame()

    def test_final(self):
        # Static Optimization must return valid activations in [0, 1] and the
        # recoloring must span the colormap (some muscles quiet, some active).
        a = self.activations
        if not np.all(np.isfinite(a)):
            raise AssertionError("non-finite Static-Optimization activations")
        if a.min() < -1e-6 or a.max() > 1.0 + 1e-6:
            raise AssertionError(f"activations out of [0, 1]: {a.min()}..{a.max()}")
        if a.max() < 0.1:
            raise AssertionError("no muscle was recruited by Static Optimization")

        colors = self.viz.muscle_colors.numpy()
        if not np.all(np.isfinite(colors)):
            raise AssertionError("non-finite muscle colors")

        # The playback itself must stay physically plausible.
        frames = self.body_q_frames.numpy()
        if not np.all(np.isfinite(frames)):
            raise AssertionError("non-finite body transforms in gait playback")
        body = {name: i for i, name in enumerate(self.model.body_label)}
        heights = frames[:, body["pelvis"], 2]
        if not (0.7 < heights.min() and heights.max() < 1.3):
            raise AssertionError(f"pelvis height out of range: {heights.min()}..{heights.max()}")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--so-frames",
            type=int,
            default=10,
            help="Number of stride frames to run Static Optimization on (more = smoother, slower).",
        )
        return parser


if __name__ == "__main__":
    # Parse arguments and initialize the viewer.
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)

    # Create viewer and run
    newton.examples.run(Example(viewer, args), args)
