# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example OpenSim Marker IK
#
# Reconstructs a walking gait with Newton-native marker inverse kinematics.
# Synthetic motion-capture targets are generated from a decimated slice of the
# bundled gait2354 motion, perturbed by deterministic millimeter-scale noise,
# and optionally occluded. The example solves each frame from the available
# marker observations, then plays the reconstructed skeleton and muscle paths
# with target markers, predicted markers, and visually magnified residual lines
# overlaid. The reported residual statistics always use the true, unscaled error.
#
# Command: python -m newton.examples opensim_marker_ik
#          python -m newton.examples opensim_marker_ik --marker-occlusion 0.25
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.opensim as osim


class Example:
    """Reconstruct a noisy, partially occluded gait marker trajectory."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.device = wp.get_device()

        marker_noise = float(args.marker_noise)
        marker_occlusion = float(args.marker_occlusion)
        residual_scale = float(args.residual_scale)
        if marker_noise < 0.0:
            raise ValueError("--marker-noise must be non-negative")
        if not 0.0 <= marker_occlusion < 0.5:
            raise ValueError("--marker-occlusion must be in [0, 0.5)")
        if residual_scale < 0.0:
            raise ValueError("--residual-scale must be non-negative")

        model_path = newton.examples.get_asset("gait2354_subject01.osim")
        motion_path = newton.examples.get_asset("gait2354_subject01_walk.mot")
        self.osim_model = osim.parse_osim(model_path)
        motion_time, motion_coords = osim.read_motion(self.osim_model, motion_path)

        requested_frames = max(2, int(args.ik_frames))
        slice_start = len(motion_time) // 8
        slice_stop = len(motion_time) - slice_start
        num_frames = min(requested_frames, slice_stop - slice_start)
        sample_indices = np.linspace(slice_start, slice_stop - 1, num_frames).round().astype(int)
        self.times = motion_time[sample_indices]
        self.true_coords = motion_coords[sample_indices]
        self.num_frames = num_frames

        # Generate the mock motion-capture trial in OpenSim's native Y-up frame.
        self.fk = osim.ForwardKinematics(self.osim_model, device=self.device)
        clean_targets = self.fk.marker_positions_batch(self.true_coords)
        rng = np.random.default_rng(2025)
        target_data = clean_targets + rng.normal(scale=marker_noise, size=clean_targets.shape)

        missing = rng.random(target_data.shape[:2]) < marker_occlusion
        # Preserve a distributed pelvis/bilateral-leg set so every IK frame remains
        # anatomically constrained while the remaining markers may be occluded.
        required_markers = (
            "Sternum",
            "R.ASIS",
            "L.ASIS",
            "V.Sacral",
            "R.Knee.Lat",
            "L.Knee.Lat",
            "R.Ankle.Lat",
            "L.Ankle.Lat",
            "R.Heel",
            "L.Heel",
            "R.Toe.Tip",
            "L.Toe.Tip",
        )
        marker_index = {name: index for index, name in enumerate(self.fk.marker_names)}
        for name in required_markers:
            if name in marker_index:
                missing[:, marker_index[name]] = False
        target_data[missing] = np.nan
        rate = 1.0 / float(np.mean(np.diff(self.times))) if num_frames > 1 else 1.0
        self.targets = osim.MarkerData(
            times=self.times,
            marker_names=list(self.fk.marker_names),
            data=target_data,
            rate=rate,
            units="m",
        )
        self.occluded_count = int(np.count_nonzero(missing))

        self.ik = osim.InverseKinematics(self.osim_model, device=self.device)
        self.solved_coords = np.empty_like(self.true_coords)
        self.marker_rms = np.empty(num_frames)
        self.marker_max = np.empty(num_frames)
        self.initial_rms = np.empty(num_frames)
        default_coords = np.array(
            [coordinate.default_value for joint in self.osim_model.joints for coordinate in joint.coordinates]
        )
        default_markers = self.fk.marker_positions(default_coords)

        q_guess = None
        for frame in range(num_frames):
            observations = self.targets.frame(frame)
            q_guess, rms, maximum = self.ik.solve_frame(observations, q0=q_guess)
            self.solved_coords[frame] = q_guess
            self.marker_rms[frame] = rms
            self.marker_max[frame] = maximum
            initial_distances = [np.linalg.norm(default_markers[name] - point) for name, point in observations.items()]
            self.initial_rms[frame] = float(np.sqrt(np.mean(np.square(initial_distances))))

        print(
            f"[opensim_marker_ik] solved {num_frames} frames with "
            f"{self.occluded_count}/{missing.size} observations occluded; "
            f"median marker RMS={1000.0 * np.median(self.marker_rms):.2f} mm"
        )

        self.viz = osim.MotionVisualizer(
            self.osim_model,
            self.solved_coords,
            time=self.times,
            device=self.device,
        )
        predicted = self.fk.marker_positions_batch(self.solved_coords)
        converter = osim.OsimFrameConverter(newton.Axis.Z)
        target_world = converter.transform_vectors(target_data)
        predicted_world = converter.transform_vectors(predicted)

        # ViewerGL requires overlay arrays to live on the viewer/model device.
        self.target_points = []
        self.target_radii = []
        self.target_colors = []
        self.predicted_points = []
        self.predicted_radii = []
        self.predicted_colors = []
        self.residual_starts = []
        self.residual_ends = []
        for frame in range(num_frames):
            visible = ~missing[frame]
            target_frame = np.ascontiguousarray(target_world[frame, visible], dtype=np.float32)
            predicted_frame = np.ascontiguousarray(predicted_world[frame], dtype=np.float32)
            self.target_points.append(wp.array(target_frame, dtype=wp.vec3, device=self.device))
            self.target_radii.append(
                wp.array(np.full(len(target_frame), 0.012, dtype=np.float32), dtype=wp.float32, device=self.device)
            )
            self.target_colors.append(
                wp.array(
                    np.tile(np.array((0.10, 0.95, 0.25), dtype=np.float32), (len(target_frame), 1)),
                    dtype=wp.vec3,
                    device=self.device,
                )
            )
            self.predicted_points.append(wp.array(predicted_frame, dtype=wp.vec3, device=self.device))
            self.predicted_radii.append(
                wp.array(np.full(len(predicted_frame), 0.008, dtype=np.float32), dtype=wp.float32, device=self.device)
            )
            self.predicted_colors.append(
                wp.array(
                    np.tile(np.array((0.15, 0.45, 1.0), dtype=np.float32), (len(predicted_frame), 1)),
                    dtype=wp.vec3,
                    device=self.device,
                )
            )
            self.residual_starts.append(wp.array(target_frame, dtype=wp.vec3, device=self.device))
            residual_end = target_world[frame, visible] + residual_scale * (
                predicted_world[frame, visible] - target_world[frame, visible]
            )
            self.residual_ends.append(wp.array(np.ascontiguousarray(residual_end), dtype=wp.vec3, device=self.device))

        dt = float(np.mean(np.diff(self.times))) if num_frames > 1 else 1.0 / 30.0
        self.frame_dt = dt
        self.fps = 1.0 / dt
        self.sim_time = float(self.times[0])
        self.frame = 0

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        osim.add_osim(builder, self.osim_model, parse_muscles=False, parse_contacts=False)
        for body in range(builder.body_count):
            builder.add_shape_sphere(body, radius=0.012, as_site=True, color=(0.86, 0.83, 0.62))
        builder.add_ground_plane()
        self.model = builder.finalize(device=self.device)
        self.state = self.model.state()
        self.body_q_frames = self.viz.body_transforms(self.model.body_label)
        wp.copy(self.state.body_q, self.body_q_frames[self.frame])

        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(pos=wp.vec3(0.6, -2.3, 1.15), pitch=-4.0, yaw=90.0)

    def step(self):
        """Advance to the next reconstructed marker-IK frame."""
        self.frame = (self.frame + 1) % self.num_frames
        wp.copy(self.state.body_q, self.body_q_frames[self.frame])
        self.sim_time += self.frame_dt

    def render(self):
        """Render the reconstruction and its marker residuals."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        self.viz.render_skeleton(self.viewer, self.frame)
        self.viz.render_muscles(self.viewer, self.frame)
        self.viewer.log_points(
            "markers/target",
            self.target_points[self.frame],
            self.target_radii[self.frame],
            self.target_colors[self.frame],
        )
        self.viewer.log_points(
            "markers/predicted",
            self.predicted_points[self.frame],
            self.predicted_radii[self.frame],
            self.predicted_colors[self.frame],
        )
        self.viewer.log_lines(
            "markers/residual",
            self.residual_starts[self.frame],
            self.residual_ends[self.frame],
            (1.0, 0.35, 0.10),
        )
        self.viewer.end_frame()

    def test_final(self):
        """Verify marker IK improves the fit and reconstructs a plausible gait."""
        if not np.all(np.isfinite(self.solved_coords)):
            raise AssertionError("marker IK produced non-finite coordinates")
        if not np.all(np.isfinite(self.marker_rms)) or not np.all(np.isfinite(self.marker_max)):
            raise AssertionError("marker IK produced non-finite residual statistics")

        median_rms = float(np.median(self.marker_rms))
        median_initial = float(np.median(self.initial_rms))
        if median_rms >= 0.1 * median_initial:
            raise AssertionError(
                f"marker IK did not substantially improve the fit: {median_rms:.4f} vs {median_initial:.4f} m"
            )
        expected_scale = max(float(self.args.marker_noise), 1.0e-5)
        if median_rms > 3.0 * expected_scale + 5.0e-4:
            raise AssertionError(f"marker residual is too large: {median_rms:.4f} m")

        coordinate_error = self.solved_coords - self.true_coords
        rotational = np.array([motion_type == "rotational" for motion_type in self.ik.motion_types])
        rotational_rms = float(np.sqrt(np.mean(coordinate_error[:, rotational] ** 2)))
        translational_rms = float(np.sqrt(np.mean(coordinate_error[:, ~rotational] ** 2)))
        if rotational_rms > np.deg2rad(3.0):
            raise AssertionError(f"rotational coordinate RMS is too large: {np.rad2deg(rotational_rms):.2f} deg")
        if translational_rms > 0.005:
            raise AssertionError(f"translational coordinate RMS is too large: {1000.0 * translational_rms:.2f} mm")

        body = {name: index for index, name in enumerate(self.model.body_label)}
        body_frames = self.body_q_frames.numpy()
        pelvis_height = body_frames[:, body["pelvis"], 2]
        if not (0.7 < pelvis_height.min() and pelvis_height.max() < 1.3):
            raise AssertionError(
                f"reconstructed pelvis height is implausible: {pelvis_height.min()}..{pelvis_height.max()}"
            )

        if float(self.args.marker_occlusion) > 0.0 and self.occluded_count == 0:
            raise AssertionError("requested marker occlusion did not hide any observations")
        if min(len(points) for points in self.target_points) < 8:
            raise AssertionError("too few visible markers to constrain marker IK")

    @staticmethod
    def create_parser():
        """Create the example command-line parser."""
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--ik-frames",
            type=int,
            default=32,
            help="Number of decimated gait frames to reconstruct.",
        )
        parser.add_argument(
            "--marker-noise",
            type=float,
            default=0.002,
            metavar="METERS",
            help="Standard deviation of deterministic Gaussian marker noise [m].",
        )
        parser.add_argument(
            "--marker-occlusion",
            type=float,
            default=0.1,
            metavar="FRACTION",
            help="Probability that a non-essential marker observation is hidden in each frame (must be below 0.5).",
        )
        parser.add_argument(
            "--residual-scale",
            type=float,
            default=20.0,
            help="Visual magnification applied to marker residual lines.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
