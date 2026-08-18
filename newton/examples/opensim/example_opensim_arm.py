# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# Example OpenSim Arm (muscle-driven forward dynamics)
#
# A planar "arm26"-style elbow: a forearm hangs from a pin joint under gravity,
# spanned by an antagonist muscle pair (a flexor "BIC" and an extensor "TRI").
# A :class:`newton.opensim.PrescribedController` ramps the flexor excitation, and
# :func:`newton.opensim.simulate_muscle_driven` closes the loop -- muscle
# activation dynamics, De Groote-Fregly muscle-tendon force, and the
# OpenSim-exact multibody dynamics -- so the forearm lifts itself entirely from
# the muscle it is told to excite (no joint torques are prescribed).
#
# The resulting coordinate trajectory is played back through the Newton-native
# OpenSim forward kinematics, rebuilding the skeleton and muscle-tendon paths in
# Warp every frame (see :class:`newton.opensim.MotionVisualizer`); muscles are
# colored by their normalized length so the flexor visibly shortens as the elbow
# closes.
#
# Command: python -m newton.examples opensim_arm
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.opensim as osim

# A planar single-DOF elbow: a forearm hangs from a pin joint 0.9 m above the
# origin, spanned by a rigid-tendon flexor (BIC) and extensor (TRI). The muscles
# are sized so the flexor operates near its optimal fiber length in the hanging
# pose, giving it the authority to lift the limb against gravity.
_ARM26_OSIM = """<?xml version="1.0" encoding="UTF-8" ?>
<OpenSimDocument Version="40000">
\t<Model name="arm26_planar">
\t\t<gravity>0 -9.80665 0</gravity>
\t\t<BodySet><objects>
\t\t\t<Body name="forearm">
\t\t\t\t<mass>0.8</mass>
\t\t\t\t<mass_center>0 -0.15 0</mass_center>
\t\t\t\t<inertia>0.006 0.001 0.006 0 0 0</inertia>
\t\t\t</Body>
\t\t</objects></BodySet>
\t\t<JointSet><objects>
\t\t\t<PinJoint name="elbow">
\t\t\t\t<socket_parent_frame>elbow_parent_offset</socket_parent_frame>
\t\t\t\t<socket_child_frame>elbow_child_offset</socket_child_frame>
\t\t\t\t<coordinates><objects>
\t\t\t\t\t<Coordinate name="elbow_flex">
\t\t\t\t\t\t<motion_type>rotational</motion_type>
\t\t\t\t\t\t<default_value>0</default_value>
\t\t\t\t\t\t<range>-0.2 2.6</range>
\t\t\t\t\t</Coordinate>
\t\t\t\t</objects></coordinates>
\t\t\t\t<frames>
\t\t\t\t\t<PhysicalOffsetFrame name="elbow_parent_offset">
\t\t\t\t\t\t<socket_parent>/ground</socket_parent>
\t\t\t\t\t\t<translation>0 0.9 0</translation>
\t\t\t\t\t\t<orientation>0 0 0</orientation>
\t\t\t\t\t</PhysicalOffsetFrame>
\t\t\t\t\t<PhysicalOffsetFrame name="elbow_child_offset">
\t\t\t\t\t\t<socket_parent>/bodyset/forearm</socket_parent>
\t\t\t\t\t\t<translation>0 0 0</translation>
\t\t\t\t\t\t<orientation>0 0 0</orientation>
\t\t\t\t\t</PhysicalOffsetFrame>
\t\t\t\t</frames>
\t\t\t</PinJoint>
\t\t</objects></JointSet>
\t\t<ForceSet><objects>
\t\t\t<Thelen2003Muscle name="BIC">
\t\t\t\t<GeometryPath><PathPointSet><objects>
\t\t\t\t\t<PathPoint name="BIC-origin"><socket_parent_frame>/ground</socket_parent_frame><location>0.025 1.12 0</location></PathPoint>
\t\t\t\t\t<PathPoint name="BIC-insertion"><socket_parent_frame>/bodyset/forearm</socket_parent_frame><location>0.025 -0.04 0</location></PathPoint>
\t\t\t\t</objects></PathPointSet></GeometryPath>
\t\t\t\t<max_isometric_force>300</max_isometric_force>
\t\t\t\t<optimal_fiber_length>0.12</optimal_fiber_length>
\t\t\t\t<tendon_slack_length>0.14</tendon_slack_length>
\t\t\t\t<pennation_angle_at_optimal>0</pennation_angle_at_optimal>
\t\t\t\t<ignore_tendon_compliance>true</ignore_tendon_compliance>
\t\t\t</Thelen2003Muscle>
\t\t\t<Thelen2003Muscle name="TRI">
\t\t\t\t<GeometryPath><PathPointSet><objects>
\t\t\t\t\t<PathPoint name="TRI-origin"><socket_parent_frame>/ground</socket_parent_frame><location>-0.03 1.05 0</location></PathPoint>
\t\t\t\t\t<PathPoint name="TRI-insertion"><socket_parent_frame>/bodyset/forearm</socket_parent_frame><location>-0.03 -0.05 0</location></PathPoint>
\t\t\t\t</objects></PathPointSet></GeometryPath>
\t\t\t\t<max_isometric_force>300</max_isometric_force>
\t\t\t\t<optimal_fiber_length>0.12</optimal_fiber_length>
\t\t\t\t<tendon_slack_length>0.14</tendon_slack_length>
\t\t\t\t<pennation_angle_at_optimal>0</pennation_angle_at_optimal>
\t\t\t\t<ignore_tendon_compliance>true</ignore_tendon_compliance>
\t\t\t</Thelen2003Muscle>
\t\t</objects></ForceSet>
\t</Model>
</OpenSimDocument>
"""


def build_arm26_model() -> osim.OsimModel:
    """Parse the embedded planar single-DOF elbow with a flexor/extensor pair.

    The forearm hangs from a pin joint 0.9 m above the origin; the rigid-tendon
    muscles are sized so the flexor operates near its optimal fiber length in the
    hanging pose and can lift the limb against gravity.
    """
    return osim.parse_osim(_ARM26_OSIM)


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        device = wp.get_device()

        self.osim_model = build_arm26_model()

        # Muscle excitations: ramp the flexor (BIC) from rest to a hold while the
        # extensor (TRI) stays at a small co-contraction baseline. This is the
        # only input -- the elbow angle is a *result* of the muscle dynamics.
        muscle_names = [m.name for m in self.osim_model.muscles]
        ramp_start, ramp_end, hold = 0.1, 0.6, 0.5

        def excitation(t):
            frac = min(max((t - ramp_start) / (ramp_end - ramp_start), 0.0), 1.0)
            values = dict.fromkeys(muscle_names, 0.02)
            values["BIC"] = 0.02 + hold * frac
            return np.array([values[name] for name in muscle_names])

        self.controller = osim.PrescribedController(muscle_names, excitation)

        duration = float(getattr(args, "duration", 1.5))
        self.result = osim.simulate_muscle_driven(
            self.osim_model,
            self.controller,
            initial_coordinates=np.array([0.0]),
            initial_speeds=np.array([0.0]),
            duration=duration,
            dt=0.002,
            integrator="rk4",
            device=device,
        )

        # Subsample to a renderable frame rate (~60 fps playback).
        times = self.result.times
        coords = self.result.coordinates
        stride = max(1, int(round((1.0 / 60.0) / (times[1] - times[0])))) if len(times) > 1 else 1
        self.play_times = times[::stride]
        play_coords = coords[::stride]

        self.viz = osim.MotionVisualizer(self.osim_model, play_coords, time=self.play_times, device=device)
        self.num_frames = self.viz.num_frames

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
        # The forearm must lift itself from the muscle excitation alone: finite
        # poses, a substantial net elbow flexion, and non-degenerate muscle paths.
        frames = self.body_q_frames.numpy()
        if not np.all(np.isfinite(frames)):
            raise AssertionError("non-finite body transforms in arm playback")

        angle = self.result.coordinates[:, 0]
        if not np.all(np.isfinite(angle)):
            raise AssertionError("non-finite elbow trajectory")
        net_flexion = float(np.degrees(angle.max() - angle[0]))
        if net_flexion < 45.0:
            raise AssertionError(f"muscle failed to flex the elbow: {net_flexion:.1f} deg")

        # The flexor activation must actually rise in response to its excitation.
        bic = self.result.muscle_names.index("BIC")
        if float(self.result.activations[:, bic].max()) < 0.2:
            raise AssertionError("flexor activation did not rise")

        starts = self.viz.muscle_starts.numpy()[0]
        ends = self.viz.muscle_ends.numpy()[0]
        if np.linalg.norm(starts - ends, axis=1).max() <= 0.0:
            raise AssertionError("muscle segments have zero length")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--duration",
            type=float,
            default=1.5,
            help="Simulated duration [s] of the muscle-driven forward dynamics.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
