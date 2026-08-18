# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example OpenSim Contact Hop
#
# A planar leg hops in place, driven purely by foot-ground contact fed back
# into the forward dynamics. The pelvis rides a vertical slider and the hip,
# knee, and ankle are pin joints; a foot with heel and toe contact spheres
# meets a ground half-space through OpenSim ``SmoothSphereHalfSpaceForce``
# elements. Nothing holds the leg up externally -- at every integration substep
# :class:`~newton.opensim.OpenSimContact` evaluates the ground reaction from the
# current state and :func:`~newton.opensim.simulate_muscle_driven` adds it to the
# joint moments, closing the contact loop (``contact=True``).
#
# A rhythmic joint controller (crouch -> explosive push-off -> flight) pumps
# energy through the contact so the leg leaves the ground and lands repeatedly,
# reaching a peak vertical ground-reaction force of several times body weight.
#
# Command: python -m newton.examples opensim_contact_hop
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.opensim as osim

# Planar leg: pelvis on a vertical slider, hip/knee/ankle pins, and a foot whose
# heel/toe spheres contact a ground half-space. Straight-leg standing height is
# thigh + shank + ankle + sphere radius = 0.90 m.
_STANCE_LEG_OSIM = """<?xml version="1.0" encoding="UTF-8" ?>
<OpenSimDocument Version="40000">
  <Model name="stance_leg">
    <gravity>0 -9.80665 0</gravity>
    <BodySet name="bodyset"><objects>
      <Body name="pelvis"><mass>6.0</mass><mass_center>0 0 0</mass_center><inertia>0.05 0.05 0.05 0 0 0</inertia></Body>
      <Body name="thigh"><mass>6.0</mass><mass_center>0 -0.2 0</mass_center><inertia>0.10 0.02 0.10 0 0 0</inertia></Body>
      <Body name="shank"><mass>3.0</mass><mass_center>0 -0.2 0</mass_center><inertia>0.05 0.01 0.05 0 0 0</inertia></Body>
      <Body name="foot"><mass>1.0</mass><mass_center>0.06 -0.035 0</mass_center><inertia>0.01 0.01 0.01 0 0 0</inertia></Body>
    </objects></BodySet>
    <JointSet name="jointset"><objects>
      <SliderJoint name="base">
        <socket_parent_frame>base_g</socket_parent_frame><socket_child_frame>base_c</socket_child_frame>
        <coordinates><objects>
          <Coordinate name="base_ty"><motion_type>translational</motion_type><default_value>0.9</default_value><range>-1 3</range></Coordinate>
        </objects></coordinates>
        <frames>
          <PhysicalOffsetFrame name="base_g"><socket_parent>/ground</socket_parent><translation>0 0 0</translation><orientation>0 0 1.5707963267948966</orientation></PhysicalOffsetFrame>
          <PhysicalOffsetFrame name="base_c"><socket_parent>/bodyset/pelvis</socket_parent><translation>0 0 0</translation><orientation>0 0 1.5707963267948966</orientation></PhysicalOffsetFrame>
        </frames>
      </SliderJoint>
      <PinJoint name="hip">
        <socket_parent_frame>hip_p</socket_parent_frame><socket_child_frame>hip_c</socket_child_frame>
        <coordinates><objects>
          <Coordinate name="hip_flex"><motion_type>rotational</motion_type><default_value>0</default_value><range>-1.5 1.5</range></Coordinate>
        </objects></coordinates>
        <frames>
          <PhysicalOffsetFrame name="hip_p"><socket_parent>/bodyset/pelvis</socket_parent><translation>0 0 0</translation><orientation>0 0 0</orientation></PhysicalOffsetFrame>
          <PhysicalOffsetFrame name="hip_c"><socket_parent>/bodyset/thigh</socket_parent><translation>0 0 0</translation><orientation>0 0 0</orientation></PhysicalOffsetFrame>
        </frames>
      </PinJoint>
      <PinJoint name="knee">
        <socket_parent_frame>knee_p</socket_parent_frame><socket_child_frame>knee_c</socket_child_frame>
        <coordinates><objects>
          <Coordinate name="knee_flex"><motion_type>rotational</motion_type><default_value>0</default_value><range>-2.2 0.1</range></Coordinate>
        </objects></coordinates>
        <frames>
          <PhysicalOffsetFrame name="knee_p"><socket_parent>/bodyset/thigh</socket_parent><translation>0 -0.4 0</translation><orientation>0 0 0</orientation></PhysicalOffsetFrame>
          <PhysicalOffsetFrame name="knee_c"><socket_parent>/bodyset/shank</socket_parent><translation>0 0 0</translation><orientation>0 0 0</orientation></PhysicalOffsetFrame>
        </frames>
      </PinJoint>
      <PinJoint name="ankle">
        <socket_parent_frame>ankle_p</socket_parent_frame><socket_child_frame>ankle_c</socket_child_frame>
        <coordinates><objects>
          <Coordinate name="ankle_flex"><motion_type>rotational</motion_type><default_value>0</default_value><range>-1.0 1.0</range></Coordinate>
        </objects></coordinates>
        <frames>
          <PhysicalOffsetFrame name="ankle_p"><socket_parent>/bodyset/shank</socket_parent><translation>0 -0.4 0</translation><orientation>0 0 0</orientation></PhysicalOffsetFrame>
          <PhysicalOffsetFrame name="ankle_c"><socket_parent>/bodyset/foot</socket_parent><translation>0 0 0</translation><orientation>0 0 0</orientation></PhysicalOffsetFrame>
        </frames>
      </PinJoint>
    </objects></JointSet>
    <ContactGeometrySet name="contactgeometryset"><objects>
      <ContactHalfSpace name="floor"><socket_frame>/ground</socket_frame><location>0 0 0</location><orientation>0 0 -1.5707963267948966</orientation></ContactHalfSpace>
      <ContactSphere name="heel"><socket_frame>/bodyset/foot</socket_frame><location>-0.05 -0.07 0</location><radius>0.03</radius></ContactSphere>
      <ContactSphere name="toe"><socket_frame>/bodyset/foot</socket_frame><location>0.16 -0.07 0</location><radius>0.03</radius></ContactSphere>
    </objects></ContactGeometrySet>
    <ForceSet name="forceset"><objects>
      <SmoothSphereHalfSpaceForce name="heel_c"><socket_sphere>/contactgeometryset/heel</socket_sphere><socket_half_space>/contactgeometryset/floor</socket_half_space>
        <stiffness>500000</stiffness><dissipation>1.0</dissipation><static_friction>0.9</static_friction><dynamic_friction>0.8</dynamic_friction><viscous_friction>0.5</viscous_friction><transition_velocity>0.1</transition_velocity></SmoothSphereHalfSpaceForce>
      <SmoothSphereHalfSpaceForce name="toe_c"><socket_sphere>/contactgeometryset/toe</socket_sphere><socket_half_space>/contactgeometryset/floor</socket_half_space>
        <stiffness>500000</stiffness><dissipation>1.0</dissipation><static_friction>0.9</static_friction><dynamic_friction>0.8</dynamic_friction><viscous_friction>0.5</viscous_friction><transition_velocity>0.1</transition_velocity></SmoothSphereHalfSpaceForce>
    </objects></ForceSet>
  </Model>
</OpenSimDocument>"""

_STAND_HEIGHT = 0.90  # hip height with the leg straight and the sole on the floor
_HOP_PERIOD = 0.70  # [s] one crouch->push->flight->land cycle


def build_stance_leg_model() -> osim.OsimModel:
    """Parse the embedded planar hopping leg with foot-ground contact."""
    return osim.parse_osim(_STANCE_LEG_OSIM)


def hop_controller(model, *, period: float = _HOP_PERIOD):
    """Build a ``coordinate_controls`` callable that makes the leg hop.

    Returns ``tau(t, q, qd) -> [num_coordinates]``: a PD servo that tracks a
    rhythmic knee/ankle target (deep crouch, explosive extension, straight
    flight) while keeping the trunk upright. The vertical base coordinate is left
    free -- it is supported only by the closed contact loop.

    Args:
        model: Parsed leg model whose coordinate order sets the output layout.
        period: Duration [s] of one hop cycle.
    """
    names = [c.name for j in model.joints for c in j.coordinates]
    idx = {n: i for i, n in enumerate(names)}
    kp = {"hip_flex": 600.0, "knee_flex": 900.0, "ankle_flex": 350.0}
    kd = {"hip_flex": 25.0, "knee_flex": 30.0, "ankle_flex": 14.0}

    def targets(t):
        phase = (t % period) / period
        if phase < 0.35:  # crouch to load the leg
            s = phase / 0.35
            return {
                "hip_flex": 0.0,
                "knee_flex": -0.85 * np.sin(0.5 * np.pi * s),
                "ankle_flex": 0.45 * np.sin(0.5 * np.pi * s),
            }
        if phase < 0.50:  # explosive extension: push off the ground
            s = (phase - 0.35) / 0.15
            return {
                "hip_flex": 0.0,
                "knee_flex": -0.85 * (1.0 - s) + 0.10 * s,
                "ankle_flex": 0.45 * (1.0 - s) - 0.25 * s,
            }
        return {"hip_flex": 0.0, "knee_flex": 0.05, "ankle_flex": -0.25}  # straight flight

    def tau(t, q, qd):
        out = np.zeros(len(q))
        for name, target in targets(t).items():
            i = idx[name]
            out[i] = -kp[name] * (q[i] - target) - kd[name] * qd[i]
        return out

    return tau


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        device = wp.get_device()

        self.osim_model = build_stance_leg_model()
        self.names = [c.name for j in self.osim_model.joints for c in j.coordinates]
        self.controller = hop_controller(self.osim_model, period=float(getattr(args, "hop_period", _HOP_PERIOD)))

        # Close the contact loop: foot-ground reaction drives the forward dynamics.
        q0 = np.zeros(len(self.names))
        q0[self.names.index("base_ty")] = _STAND_HEIGHT
        duration = float(getattr(args, "duration", 2.8))
        self.result = osim.simulate_muscle_driven(
            self.osim_model,
            np.zeros(0),  # no muscles: pure joint actuators + contact
            initial_coordinates=q0,
            initial_speeds=np.zeros(len(self.names)),
            duration=duration,
            dt=1.5e-4,
            integrator="rk4",
            coordinate_controls=self.controller,
            contact=True,
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

        self.frame_dt = 1.0 / 60.0
        self.fps = 60.0
        self.sim_time = float(self.play_times[0]) if len(self.play_times) else 0.0
        self.frame = 0

        # Render container: leg bodies, foot contact spheres, and a ground plane.
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        osim.add_osim(builder, self.osim_model, parse_muscles=False, parse_contacts=True)
        for body in range(builder.body_count):
            builder.add_shape_sphere(body, radius=0.025, as_site=True, color=(0.9, 0.85, 0.55))
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
        self.viewer.end_frame()

    def test_final(self):
        # The leg must hop: bounded, in-range motion that leaves the ground and
        # lands under closed-loop contact, with realistic push-off forces.
        coords = self.result.coordinates
        if not np.all(np.isfinite(coords)):
            raise AssertionError("non-finite coordinates in the hopping leg")

        ranges = {c.name: c.range for j in self.osim_model.joints for c in j.coordinates}
        for i, name in enumerate(self.names):
            lo, hi = ranges[name]
            col = coords[:, i]
            if col.min() < lo - 0.05 or col.max() > hi + 0.05:
                raise AssertionError(f"{name} left its range: {col.min():.3f}..{col.max():.3f}")

        ty = coords[:, self.names.index("base_ty")]
        if ty.min() < 0.6:
            raise AssertionError(f"leg collapsed: base_ty min {ty.min():.3f}")
        if np.ptp(ty) < 0.05:
            raise AssertionError(f"leg did not hop: base_ty amplitude {np.ptp(ty) * 100:.1f} cm")
        if ty.max() <= _STAND_HEIGHT + 0.005:
            raise AssertionError("leg never rose above standing height (no flight)")

        # Contact must produce a flight phase (GRF -> 0) and a push-off > body weight.
        contact = osim.OpenSimContact(self.osim_model)
        sub = slice(0, len(ty), max(1, len(ty) // 200))
        grf_z = contact.forces(coords[sub], self.result.speeds[sub])[:, :, 2].sum(axis=1)
        body_weight = sum(b.mass for b in self.osim_model.bodies) * 9.80665
        if grf_z.min() > 1.0:
            raise AssertionError("no flight phase: vertical GRF never reached zero")
        if grf_z.max() < body_weight:
            raise AssertionError(f"push-off GRF {grf_z.max():.0f} N below body weight {body_weight:.0f} N")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--duration", type=float, default=2.8, help="Simulated duration [s].")
        parser.add_argument("--hop-period", type=float, default=_HOP_PERIOD, help="Hop cycle period [s].")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
