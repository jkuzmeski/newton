# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example OpenSim Gait 2D
#
# A planar (2D) muscle-driven leg performing a gait swing phase. The two-segment
# leg (thigh + shank) hangs from a hip pin joint and is spanned by four
# rigid-tendon Thelen muscles: a hip flexor/extensor pair and a knee
# flexor/extensor pair. Only the muscle *excitations* are prescribed -- the hip
# and knee angles are a pure *result* of the closed-loop muscle dynamics
# (:func:`newton.opensim.simulate_muscle_driven`).
#
# Physiological joint stops are added through the forward-dynamics
# ``coordinate_controls`` hook (the same idea as OpenSim's CoordinateLimitForce):
# a stiff one-sided spring-damper keeps each coordinate inside its range, so the
# knee cannot hyper-extend as the shank swings.
#
# The muscle paths are colored by their activation over the motion
# (:meth:`newton.opensim.MotionVisualizer.color_muscles_by`), so each muscle
# lights up (blue -> yellow -> red) exactly when it is recruited.
#
# Command: python -m newton.examples opensim_gait2d
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.opensim as osim

# Planar leg: thigh + shank on hip/knee pin joints, spanned by a hip flexor
# (HFL) / extensor (HEX) pair and a knee extensor (KEX) / flexor (KFL) pair. The
# tendon slack lengths place every fiber near its optimum in the neutral pose so
# the muscles have the authority to swing the limb.
_LEG2D_OSIM = """<?xml version="1.0" encoding="UTF-8" ?>
<OpenSimDocument Version="40000">
	<Model name="leg2d">
		<gravity>0 -9.80665 0</gravity>
		<BodySet><objects>
			<Body name="thigh"><mass>6.0</mass><mass_center>0 -0.17 0</mass_center><inertia>0.10 0.02 0.10 0 0 0</inertia></Body>
			<Body name="shank"><mass>2.5</mass><mass_center>0 -0.18 0</mass_center><inertia>0.04 0.006 0.04 0 0 0</inertia></Body>
		</objects></BodySet>
		<JointSet><objects>
			<PinJoint name="hip">
				<socket_parent_frame>hip_parent_offset</socket_parent_frame>
				<socket_child_frame>hip_child_offset</socket_child_frame>
				<coordinates><objects>
					<Coordinate name="hip_flex"><motion_type>rotational</motion_type><default_value>0</default_value><range>-0.70 0.90</range></Coordinate>
				</objects></coordinates>
				<frames>
					<PhysicalOffsetFrame name="hip_parent_offset"><socket_parent>/ground</socket_parent><translation>0 1.0 0</translation><orientation>0 0 0</orientation></PhysicalOffsetFrame>
					<PhysicalOffsetFrame name="hip_child_offset"><socket_parent>/bodyset/thigh</socket_parent><translation>0 0 0</translation><orientation>0 0 0</orientation></PhysicalOffsetFrame>
				</frames>
			</PinJoint>
			<PinJoint name="knee">
				<socket_parent_frame>knee_parent_offset</socket_parent_frame>
				<socket_child_frame>knee_child_offset</socket_child_frame>
				<coordinates><objects>
					<Coordinate name="knee_flex"><motion_type>rotational</motion_type><default_value>0</default_value><range>-2.10 0.02</range></Coordinate>
				</objects></coordinates>
				<frames>
					<PhysicalOffsetFrame name="knee_parent_offset"><socket_parent>/bodyset/thigh</socket_parent><translation>0 -0.40 0</translation><orientation>0 0 0</orientation></PhysicalOffsetFrame>
					<PhysicalOffsetFrame name="knee_child_offset"><socket_parent>/bodyset/shank</socket_parent><translation>0 0 0</translation><orientation>0 0 0</orientation></PhysicalOffsetFrame>
				</frames>
			</PinJoint>
		</objects></JointSet>
		<ForceSet><objects>
			<Thelen2003Muscle name="HFL">
				<GeometryPath><PathPointSet><objects>
					<PathPoint name="HFL-0"><socket_parent_frame>/ground</socket_parent_frame><location>0.05 1.07 0</location></PathPoint>
					<PathPoint name="HFL-1"><socket_parent_frame>/bodyset/thigh</socket_parent_frame><location>0.05 -0.05 0</location></PathPoint>
				</objects></PathPointSet></GeometryPath>
				<max_isometric_force>1200</max_isometric_force>
				<optimal_fiber_length>0.09</optimal_fiber_length>
				<tendon_slack_length>0.03</tendon_slack_length>
				<pennation_angle_at_optimal>0</pennation_angle_at_optimal>
				<ignore_tendon_compliance>true</ignore_tendon_compliance>
			</Thelen2003Muscle>
			<Thelen2003Muscle name="HEX">
				<GeometryPath><PathPointSet><objects>
					<PathPoint name="HEX-0"><socket_parent_frame>/ground</socket_parent_frame><location>-0.07 1.06 0</location></PathPoint>
					<PathPoint name="HEX-1"><socket_parent_frame>/bodyset/thigh</socket_parent_frame><location>-0.06 -0.07 0</location></PathPoint>
				</objects></PathPointSet></GeometryPath>
				<max_isometric_force>1200</max_isometric_force>
				<optimal_fiber_length>0.09</optimal_fiber_length>
				<tendon_slack_length>0.0404</tendon_slack_length>
				<pennation_angle_at_optimal>0</pennation_angle_at_optimal>
				<ignore_tendon_compliance>true</ignore_tendon_compliance>
			</Thelen2003Muscle>
			<Thelen2003Muscle name="KEX">
				<GeometryPath><PathPointSet><objects>
					<PathPoint name="KEX-0"><socket_parent_frame>/bodyset/thigh</socket_parent_frame><location>0.06 -0.3 0</location></PathPoint>
					<PathPoint name="KEX-1"><socket_parent_frame>/bodyset/thigh</socket_parent_frame><location>0.055 -0.4 0</location></PathPoint>
					<PathPoint name="KEX-2"><socket_parent_frame>/bodyset/shank</socket_parent_frame><location>0.045 -0.05 0</location></PathPoint>
				</objects></PathPointSet></GeometryPath>
				<max_isometric_force>500</max_isometric_force>
				<optimal_fiber_length>0.09</optimal_fiber_length>
				<tendon_slack_length>0.0611</tendon_slack_length>
				<pennation_angle_at_optimal>0</pennation_angle_at_optimal>
				<ignore_tendon_compliance>true</ignore_tendon_compliance>
			</Thelen2003Muscle>
			<Thelen2003Muscle name="KFL">
				<GeometryPath><PathPointSet><objects>
					<PathPoint name="KFL-0"><socket_parent_frame>/bodyset/thigh</socket_parent_frame><location>-0.05 -0.3 0</location></PathPoint>
					<PathPoint name="KFL-1"><socket_parent_frame>/bodyset/shank</socket_parent_frame><location>-0.05 -0.07 0</location></PathPoint>
				</objects></PathPointSet></GeometryPath>
				<max_isometric_force>700</max_isometric_force>
				<optimal_fiber_length>0.09</optimal_fiber_length>
				<tendon_slack_length>0.08</tendon_slack_length>
				<pennation_angle_at_optimal>0</pennation_angle_at_optimal>
				<ignore_tendon_compliance>true</ignore_tendon_compliance>
			</Thelen2003Muscle>
		</objects></ForceSet>
	</Model>
</OpenSimDocument>
"""

# Per-muscle excitation timing (start, end, peak) of a raised-cosine pulse; the
# baseline co-contraction keeps every muscle slightly active.
_EXCITATION = {
    "HFL": (0.05, 0.55, 0.32),  # hip flexor drives the thigh forward
    "HEX": (0.60, 1.15, 0.24),  # hip extensor returns the thigh
    "KEX": (0.55, 1.05, 0.22),  # knee extensor straightens the shank late in swing
    "KFL": (0.05, 0.50, 0.30),  # knee flexor picks up the foot early in swing
}
_BASELINE = 0.02


def build_leg2d_model() -> osim.OsimModel:
    """Parse the embedded planar two-segment leg with four Thelen muscles."""
    return osim.parse_osim(_LEG2D_OSIM)


def coordinate_limit_forces(model, *, stiffness=400.0, damping=8.0, transition=0.12):
    """Build a ``coordinate_controls`` callable that enforces joint ranges.

    Returns ``tau(t, q, qd) -> [num_coordinates]`` applying a one-sided
    spring-damper as each coordinate nears its range limit -- the differentiable
    analog of OpenSim's ``CoordinateLimitForce`` -- so the passive leg stays
    physiological instead of hyper-extending.

    Args:
        model: Parsed OpenSim model whose coordinate ranges define the stops.
        stiffness: Limit spring stiffness [N·m/rad].
        damping: Limit damping while penetrating the stop [N·m·s/rad].
        transition: Angular band [rad] over which the stop engages.
    """
    ranges = [c.range for j in model.joints for c in j.coordinates]

    def tau(t, q, qd):
        out = np.zeros(len(q))
        for i, rng in enumerate(ranges):
            if rng is None:
                continue
            lo, hi = rng
            if q[i] > hi - transition:
                x = q[i] - (hi - transition)
                out[i] -= stiffness * x * x / transition + damping * max(qd[i], 0.0)
            if q[i] < lo + transition:
                x = (lo + transition) - q[i]
                out[i] += stiffness * x * x / transition + damping * max(-qd[i], 0.0)
        return out

    return tau


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        device = wp.get_device()

        self.osim_model = build_leg2d_model()
        muscle_names = [m.name for m in self.osim_model.muscles]

        def excitation(t):
            values = dict.fromkeys(muscle_names, _BASELINE)
            for name, (t0, t1, amp) in _EXCITATION.items():
                if t0 < t < t1:
                    values[name] = _BASELINE + amp * np.sin(np.pi * (t - t0) / (t1 - t0)) ** 2
            return np.array([values[name] for name in muscle_names])

        self.controller = osim.PrescribedController(muscle_names, excitation)
        self.limit_forces = coordinate_limit_forces(self.osim_model)

        duration = float(getattr(args, "duration", 1.4))
        self.result = osim.simulate_muscle_driven(
            self.osim_model,
            self.controller,
            initial_coordinates=np.array([0.0, 0.0]),
            initial_speeds=np.array([0.0, 0.0]),
            duration=duration,
            dt=0.001,
            integrator="rk4",
            coordinate_controls=self.limit_forces,
            device=device,
        )

        # Subsample to ~60 fps for playback.
        times = self.result.times
        coords = self.result.coordinates
        stride = max(1, int(round((1.0 / 60.0) / (times[1] - times[0])))) if len(times) > 1 else 1
        self.play_times = times[::stride]
        play_coords = coords[::stride]

        self.viz = osim.MotionVisualizer(self.osim_model, play_coords, time=self.play_times, device=device)
        self.num_frames = self.viz.num_frames

        # Color the muscle paths by their activation over the motion.
        col = {name: i for i, name in enumerate(self.result.muscle_names)}
        order = [col[name] for name in self.viz.muscle_names]
        self.activations = self.result.activations[::stride][:, order]
        self.viz.color_muscles_by(self.activations, times=self.play_times, vmin=0.0, vmax=1.0)

        self.frame_dt = 1.0 / 60.0
        self.fps = 60.0
        self.sim_time = float(self.play_times[0]) if len(self.play_times) else 0.0
        self.frame = 0

        # Render container: OpenSim bodies as shape holders plus a ground plane.
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        osim.add_osim(builder, self.osim_model, parse_muscles=False, parse_contacts=False)
        for body in range(builder.body_count):
            builder.add_shape_sphere(body, radius=0.02, as_site=True, color=(0.9, 0.85, 0.55))
        builder.add_ground_plane()
        self.model = builder.finalize(device=device)

        self.state = self.model.state()
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
        self.viz.render_skeleton(self.viewer, self.frame)
        self.viz.render_muscles(self.viewer, self.frame)
        self.viewer.end_frame()

    def test_final(self):
        # The muscles alone must swing the leg through a plausible gait range and
        # the coordinate limit forces must keep both joints inside their bounds.
        coords = self.result.coordinates
        if not np.all(np.isfinite(coords)):
            raise AssertionError("non-finite coordinates in muscle-driven leg swing")

        names = self.result.coordinate_names
        ranges = {c.name: c.range for j in self.osim_model.joints for c in j.coordinates}
        for i, name in enumerate(names):
            lo, hi = ranges[name]
            col = coords[:, i]
            if col.min() < lo - 0.05 or col.max() > hi + 0.05:
                raise AssertionError(f"{name} left its range: {col.min():.3f}..{col.max():.3f} vs [{lo}, {hi}]")

        hip = coords[:, names.index("hip_flex")]
        knee = coords[:, names.index("knee_flex")]
        if np.ptp(hip) < np.deg2rad(20.0):
            raise AssertionError(f"hip barely moved: {np.rad2deg(np.ptp(hip)):.1f} deg")
        if np.ptp(knee) < np.deg2rad(20.0):
            raise AssertionError(f"knee barely moved: {np.rad2deg(np.ptp(knee)):.1f} deg")

        a = self.result.activations
        if a.min() < -1e-6 or a.max() > 1.0 + 1e-6:
            raise AssertionError(f"activations out of [0, 1]: {a.min()}..{a.max()}")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--duration", type=float, default=1.4, help="Simulated duration [s].")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
