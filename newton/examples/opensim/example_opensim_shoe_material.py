# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example OpenSim Shoe Material
#
# A shod planar leg is dropped onto the ground and lands on a *3D shoe sole*: a
# triangulated box mesh under the foot that meets the floor through an OpenSim
# ``ElasticFoundationForce`` -- a compliant, per-face spring foundation whose
# material model is (stiffness, dissipation, friction). The foot-ground reaction
# is fed back into the forward dynamics at every substep, closing the contact
# loop (see ``opensim_contact_hop``).
#
# The example then perturbs the shoe material *slightly* away from its
# nominal baseline (+/- 10-50% on stiffness, dissipation, and friction) and
# measures how far the resulting motion drifts: kinematic deviations (center-of-
# mass height and joint-angle RMSE) and kinetic deviations (peak ground-reaction
# force, loading rate, and impulse). Run ``--material-sweep`` to print the table.
#
# Command: python -m newton.examples opensim_shoe_material
#          python -m newton.examples opensim_shoe_material --material-sweep
#
###########################################################################

import sys

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.opensim as osim


# --------------------------------------------------------------------------- #
# 3D shoe sole: a triangulated box slab under the foot. Every triangle face
# becomes an ElasticFoundationForce spring; the bottom face carries the load.
# --------------------------------------------------------------------------- #
def _box_mesh(size, center, sub):
    """Return ``(vertices[N,3], faces[M,3])`` for a subdivided closed box."""
    lx, ly, lz = size
    cx, cy, cz = center
    nx, ny, nz = sub
    verts: list[list[float]] = []
    faces: list[list[int]] = []
    index: dict[tuple[float, float, float], int] = {}

    def vid(p):
        key = (round(float(p[0]), 6), round(float(p[1]), 6), round(float(p[2]), 6))
        if key not in index:
            index[key] = len(verts)
            verts.append(list(key))
        return index[key]

    def grid(origin, u, v, nu, nv, flip):
        for i in range(nu):
            for j in range(nv):
                p00 = origin + u * (i / nu) + v * (j / nv)
                p10 = origin + u * ((i + 1) / nu) + v * (j / nv)
                p11 = origin + u * ((i + 1) / nu) + v * ((j + 1) / nv)
                p01 = origin + u * (i / nu) + v * ((j + 1) / nv)
                a, b, c, d = vid(p00), vid(p10), vid(p11), vid(p01)
                if flip:
                    faces.append([a, c, b])
                    faces.append([a, d, c])
                else:
                    faces.append([a, b, c])
                    faces.append([a, c, d])

    o = np.array([cx - lx / 2, cy - ly / 2, cz - lz / 2])
    ex = np.array([lx, 0.0, 0.0])
    ey = np.array([0.0, ly, 0.0])
    ez = np.array([0.0, 0.0, lz])
    grid(o, ex, ez, nx, nz, flip=True)
    grid(o + ey, ex, ez, nx, nz, flip=False)
    grid(o, ex, ey, nx, ny, flip=False)
    grid(o + ez, ex, ey, nx, ny, flip=True)
    grid(o, ey, ez, ny, nz, flip=True)
    grid(o + ex, ey, ez, ny, nz, flip=False)
    return np.asarray(verts, np.float64), np.asarray(faces, np.int64)


# Sole footprint under the foot: 24 cm long, 9 cm wide, 2 cm thick, its bottom
# 10 cm below the foot origin so it meets the floor at standing height.
_SOLE_LEN, _SOLE_W, _SOLE_THK = 0.24, 0.09, 0.02
_SOLE_CENTER = (0.06, -0.10 + _SOLE_THK / 2.0, 0.0)


def _sole_mesh():
    return _box_mesh((_SOLE_LEN, _SOLE_THK, _SOLE_W), _SOLE_CENTER, sub=(10, 1, 4))


# --------------------------------------------------------------------------- #
# Shod leg: pelvis on a vertical slider, hip/knee/ankle pins, and a foot wearing
# the sole mesh. The material properties are formatted into the contact force.
# --------------------------------------------------------------------------- #
_SHOD_LEG_OSIM = """<?xml version="1.0" encoding="UTF-8" ?>
<OpenSimDocument Version="40000">
  <Model name="shod_leg">
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
      <ContactMesh name="sole"><socket_frame>/bodyset/foot</socket_frame><location>0 0 0</location><orientation>0 0 0</orientation><filename></filename></ContactMesh>
    </objects></ContactGeometrySet>
    <ForceSet name="forceset"><objects>
      <ElasticFoundationForce name="sole_ef">
        <contact_parameters><objects>
          <ElasticFoundationForce::ContactParameters>
            <geometry>sole floor</geometry>
            <stiffness>{stiffness}</stiffness><dissipation>{dissipation}</dissipation>
            <static_friction>{static_friction}</static_friction><dynamic_friction>{dynamic_friction}</dynamic_friction>
            <viscous_friction>0.0</viscous_friction>
          </ElasticFoundationForce::ContactParameters>
        </objects></contact_parameters>
        <transition_velocity>0.1</transition_velocity>
      </ElasticFoundationForce>
    </objects></ForceSet>
  </Model>
</OpenSimDocument>"""

# Illustrative nominal parameters chosen to produce about 1.5 mm static
# compression under body weight in this synthetic model. They are not fit to
# measured footwear. "stiffness" is per-area [Pa/m] in the SimTK force law.
_NOMINAL = {"stiffness": 5.0e6, "dissipation": 1.0, "static_friction": 0.9, "dynamic_friction": 0.8}
_STAND = 0.90  # hip height with the leg straight and the sole on the floor
_DROP = 0.06  # release height above standing


def build_shod_leg(material=None):
    """Parse the shod-leg model with the given shoe ``material`` (defaults to the nominal baseline)."""
    mat = dict(_NOMINAL)
    if material:
        mat.update(material)
    return osim.parse_osim(_SHOD_LEG_OSIM.format(**mat))


def _posture_controller(names):
    """Moderate joint PD for a compliant landing; the vertical base is left free."""
    kp = {"hip_flex": 300.0, "knee_flex": 400.0, "ankle_flex": 180.0}
    kd = {"hip_flex": 12.0, "knee_flex": 16.0, "ankle_flex": 8.0}
    tgt = {"hip_flex": 0.0, "knee_flex": -0.10, "ankle_flex": 0.0}
    idx = {n: names.index(n) for n in kp}

    def tau(t, q, qd):
        out = np.zeros(len(q))
        for n, kpn in kp.items():
            i = idx[n]
            out[i] = kpn * (tgt[n] - q[i]) - kd[n] * qd[i]
        return out

    return tau


def simulate_landing(material=None, duration=1.2, dt=1.0e-3, device=None):
    """Drop the shod leg and let the shoe contact absorb the landing.

    Returns ``(result, contact, names)``. ``contact`` is the mesh-backed
    :class:`~newton.opensim.OpenSimContact`; recompute ground reaction from
    ``result.coordinates`` / ``result.speeds`` with ``contact.forces``.
    """
    model = build_shod_leg(material)
    contact = osim.OpenSimContact(model, meshes={"sole": _sole_mesh()}, device=device)
    names = list(contact.coordinate_names)
    q0 = np.zeros(len(names))
    q0[names.index("base_ty")] = _STAND + _DROP
    q0[names.index("knee_flex")] = -0.10
    result = osim.simulate_muscle_driven(
        model,
        np.zeros(0),
        initial_coordinates=q0,
        initial_speeds=np.zeros(len(names)),
        duration=duration,
        dt=dt,
        integrator="rk4",
        coordinate_controls=_posture_controller(names),
        contact=contact,
        device=device,
    )
    return result, contact, names


def _landing_signals(result, contact, names):
    """Return time, base height, joint-angle matrix, and vertical GRF trajectory."""
    t = np.asarray(result.times)
    q = np.asarray(result.coordinates)
    qd = np.asarray(result.speeds)
    grf_z = contact.forces(q, qd)[:, :, 2].sum(axis=1)
    base = q[:, names.index("base_ty")]
    joints = np.column_stack([q[:, names.index(n)] for n in ("hip_flex", "knee_flex", "ankle_flex")])
    return t, base, joints, grf_z


def material_sweep(duration=1.2, dt=1.0e-3, device=None):
    """Perturb the nominal shoe material and print kinematic/kinetic deviations.

    Returns a dict mapping each perturbation label to its deviation metrics.
    """
    k0 = _NOMINAL["stiffness"]
    c0 = _NOMINAL["dissipation"]
    perturbations = {
        "nominal": {},
        "stiffness -20%": {"stiffness": 0.80 * k0},
        "stiffness +20%": {"stiffness": 1.20 * k0},
        "dissipation -50%": {"dissipation": 0.50 * c0},
        "dissipation +50%": {"dissipation": 1.50 * c0},
        "friction -20%": {"static_friction": 0.72, "dynamic_friction": 0.64},
        "friction +20%": {"static_friction": 1.08, "dynamic_friction": 0.96},
        "all +10%": {"stiffness": 1.1 * k0, "dissipation": 1.1 * c0, "static_friction": 0.99, "dynamic_friction": 0.88},
    }
    ref = None
    out: dict[str, dict] = {}
    deg = 180.0 / np.pi
    print(
        f"3D shoe material sensitivity (drop-landing) -- deviations vs nominal baseline "
        f"(k={k0:.1e} Pa/m, c={c0:.1f} s/m, mu_s=0.9, mu_d=0.8)\n"
    )
    header = f"{'perturbation':<17}{'baseRMSE':>9}{'jointRMSE':>10}{'peakGRF':>9}{'loadRate':>10}{'impulse':>9}"
    print(header)
    print(f"{'':17}{'[mm]':>9}{'[deg]':>10}{'[%]':>9}{'[%]':>10}{'[%]':>9}")
    for label, material in perturbations.items():
        result, contact, names = simulate_landing(material, duration=duration, dt=dt, device=device)
        t, base, joints, grf_z = _landing_signals(result, contact, names)
        peak = float(grf_z.max())
        impulse = float(np.trapezoid(grf_z, t))
        rate = float(np.max(np.diff(grf_z[: int(grf_z.argmax()) + 1]) / (t[1] - t[0]))) if grf_z.argmax() else 0.0
        if ref is None:
            ref = {"base": base, "joints": joints, "peak": peak, "rate": rate, "impulse": impulse}
        base_rmse = float(np.sqrt(np.mean((base - ref["base"]) ** 2)) * 1000.0)
        joint_rmse = float(np.sqrt(np.mean((joints - ref["joints"]) ** 2)) * deg)
        d_peak = 100.0 * (peak - ref["peak"]) / ref["peak"]
        d_rate = 100.0 * (rate - ref["rate"]) / ref["rate"] if ref["rate"] else 0.0
        d_imp = 100.0 * (impulse - ref["impulse"]) / ref["impulse"]
        out[label] = {
            "base_rmse_mm": base_rmse,
            "joint_rmse_deg": joint_rmse,
            "peak_grf_pct": d_peak,
            "loading_rate_pct": d_rate,
            "impulse_pct": d_imp,
            "peak_grf_N": peak,
        }
        if label == "nominal":
            print(
                f"{label:<17}{0.0:>9.3f}{0.0:>10.3f}{0.0:>9.2f}{0.0:>10.1f}{0.0:>9.2f}"
                f"   (peak GRF {peak:.0f} N = {peak / (16.0 * 9.80665):.2f}x body weight)"
            )
        else:
            print(f"{label:<17}{base_rmse:>9.3f}{joint_rmse:>10.3f}{d_peak:>+9.2f}{d_rate:>+10.1f}{d_imp:>+9.2f}")
    return out


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        device = wp.get_device()

        self.osim_model = build_shod_leg()
        self.result, self.contact, self.names = simulate_landing(
            duration=float(getattr(args, "duration", 1.2)), device=device
        )

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

        # Render container: leg bones, joint markers, and the shoe sole as a box.
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        osim.add_osim(builder, self.osim_model, parse_muscles=False, parse_contacts=False)
        foot = [b.name for b in self.osim_model.bodies].index("foot")
        for body in range(builder.body_count):
            builder.add_shape_sphere(body, radius=0.022, as_site=True, color=(0.9, 0.85, 0.55))
        builder.add_shape_box(
            foot,
            xform=wp.transform(wp.vec3(*_SOLE_CENTER), wp.quat_identity()),
            hx=_SOLE_LEN / 2.0,
            hy=_SOLE_THK / 2.0,
            hz=_SOLE_W / 2.0,
            color=(0.25, 0.35, 0.75),
        )
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
        # The nominal landing must be finite, in range, land on the shoe, and
        # settle at body-weight equilibrium; perturbing the material must produce
        # a measurable but bounded change in the peak ground reaction.
        _times, base, _joints, grf_z = _landing_signals(self.result, self.contact, self.names)
        if not np.all(np.isfinite(self.result.coordinates)):
            raise AssertionError("non-finite coordinates in the shod-leg landing")

        ranges = {c.name: c.range for j in self.osim_model.joints for c in j.coordinates}
        for i, name in enumerate(self.names):
            lo, hi = ranges[name]
            col = self.result.coordinates[:, i]
            if col.min() < lo - 0.05 or col.max() > hi + 0.05:
                raise AssertionError(f"{name} left its range: {col.min():.3f}..{col.max():.3f}")

        body_weight = sum(b.mass for b in self.osim_model.bodies) * 9.80665
        if base.min() > _STAND + _DROP - 0.02:
            raise AssertionError("leg never dropped onto the shoe")
        if grf_z.max() < body_weight:
            raise AssertionError(f"impact GRF {grf_z.max():.0f} N below body weight {body_weight:.0f} N")
        if abs(grf_z[-1] - body_weight) > 0.25 * body_weight:
            raise AssertionError(f"landing did not settle at body weight: final GRF {grf_z[-1]:.0f} N")

        # A slight material change (+20% stiffness) must produce a measurable but
        # bounded change in the peak ground reaction. The impact peak (~0.13 s) is
        # captured by any short run, so compare against the nominal peak above.
        stiffer, contact2, names2 = simulate_landing(
            {"stiffness": 1.20 * _NOMINAL["stiffness"]}, duration=0.5, device=wp.get_device()
        )
        _, _, _, grf2 = _landing_signals(stiffer, contact2, names2)
        rel = abs(grf2.max() - grf_z.max()) / grf_z.max()
        if not (1e-4 < rel < 0.25):
            raise AssertionError(f"+20% stiffness peak-GRF deviation {rel * 100:.2f}% out of expected band")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--duration", type=float, default=1.2, help="Simulated duration [s].")
        parser.add_argument(
            "--material-sweep",
            action="store_true",
            help="Print kinematic/kinetic deviations for perturbed shoe materials and exit.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    if "--material-sweep" in sys.argv:
        parser.parse_args()
        material_sweep()
    else:
        viewer, args = newton.examples.init(parser)
        newton.examples.run(Example(viewer, args), args)
