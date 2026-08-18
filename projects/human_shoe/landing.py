# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Direct-coupled human-shoe landing in Newton's standard Z-up world.

This is a harnessed prototype: the imported OpenSim human keeps the current
derived 12-joint / 27-DOF layout, the Digital Instron sole is attached in the
``calcn_r`` frame, and the shared calibrated foundation applies its wrench
directly to ``state.body_f[calcn_r]``. The shoe is not a separate rigid body and
no Newton collision pair is created.

The pelvis vertical DOF is left unactuated and the remaining DOFs are lightly
posture-stabilized. Contact uses the same Hyperfoam-Maxwell-Pasternak, damping,
and stick-slip friction implementation as the Digital Instron jump scenario.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.opensim as opensim
from projects.digital_instron_v2.core import CALIBRATED_MATERIAL
from projects.digital_instron_v2.dynamics import (
    FoundationConfig,
    MidsoleFoundation,
    build_foundation_geometry,
    column_colors,
)
from projects.digital_instron_v2.geometry import load_mesh
from projects.human_shoe import (
    HumanShoeExperimentContract,
    attach_sole_geometry,
    load_experiment,
    load_manifest,
    resolve_attachment,
)

BASE_DIR = Path(__file__).resolve().parents[2]
EXPERIMENT_PATH = BASE_DIR / "experiments/human_shoe/baseline_gait2354.json"
DEFAULT_DURATION_S = 0.2
DEFAULT_DROP_CLEARANCE_M = 0.012
FRAME_DT_S = 1.0 / 240.0
DEFAULT_CONTROLLER_ID = "gait2354_drop_pd_v1"


@dataclass(frozen=True, slots=True)
class LandingControllerConfig:
    """Versioned posture-controller gains for the landing prototype [SI]."""

    root_horizontal_ke: float
    root_horizontal_kd: float
    root_rotation_ke: float
    root_rotation_kd: float
    joint_ke: float
    joint_kd: float
    armature: float
    damping: float


_CONTROLLER_CONFIGS = {
    DEFAULT_CONTROLLER_ID: LandingControllerConfig(
        root_horizontal_ke=150.0,
        root_horizontal_kd=20.0,
        root_rotation_ke=60.0,
        root_rotation_kd=10.0,
        joint_ke=10.0,
        joint_kd=2.0,
        armature=0.01,
        damping=1.0,
    )
}


@dataclass(frozen=True, slots=True)
class LandingRuntimeConfig:
    """Resolved deterministic runtime settings for one landing experiment."""

    controller: LandingControllerConfig
    dt: float
    random_seed: int


def resolve_landing_config(
    experiment: HumanShoeExperimentContract,
    *,
    dt_override: float | None = None,
) -> LandingRuntimeConfig:
    """Resolve and validate the manifest-backed landing runtime configuration."""
    try:
        controller = _CONTROLLER_CONFIGS[experiment.controller_id]
    except KeyError as exc:
        supported = ", ".join(sorted(_CONTROLLER_CONFIGS))
        raise ValueError(f"unknown controller_id {experiment.controller_id!r}; supported values: {supported}") from exc
    dt = float(experiment.time_step_s if dt_override is None else dt_override)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    return LandingRuntimeConfig(controller=controller, dt=dt, random_seed=experiment.random_seed)


@wp.kernel
def _deform_columns_z_up(
    carrier: wp.int32,
    body_q: wp.array[wp.transform],
    bottom_local: wp.array[wp.vec3],
    top_local: wp.array[wp.vec3],
    bottom_world: wp.array[wp.vec3],
    top_world: wp.array[wp.vec3],
    compression: wp.array[wp.float32],
):
    """Transform the attached sole columns into the Z-up Newton world."""
    i = wp.tid()
    b = wp.transform_point(body_q[carrier], bottom_local[i])
    t = wp.transform_point(body_q[carrier], top_local[i])
    bottom_world[i] = wp.vec3(b[0], b[1], wp.max(b[2], 0.0))
    top_world[i] = wp.vec3(t[0], t[1], wp.max(t[2], 0.0))
    compression[i] = wp.max(-b[2], 0.0)


@dataclass(frozen=True)
class LandingDiagnostics:
    """Current and peak landing metrics sampled from the sole bed [SI]."""

    normal_force_n: float
    active_columns: int
    compression_m: float
    peak_normal_force_n: float
    peak_active_columns: int
    peak_compression_m: float
    pelvis_height_m: float
    finite: bool


def _resolve_repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (BASE_DIR / path).resolve()


def _standing_root_height(state: newton.State, carrier: int, bottom_local: np.ndarray, clearance: float) -> float:
    """Place the pelvis so the lowest outsole point starts above ground."""
    q = state.body_q.numpy()[carrier]
    body_xform = wp.transform(wp.vec3(*q[:3]), wp.quat(*q[3:]))
    lowest = min(float(wp.transform_point(body_xform, wp.vec3(*pt))[2]) for pt in bottom_local)
    root_height = float(state.joint_q.numpy()[1])
    return root_height + clearance - lowest


class Example:
    """Run a controlled landing with the derived human shoe model."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.device = wp.get_device()
        self.duration = float(getattr(args, "duration", DEFAULT_DURATION_S))
        drop_clearance = float(getattr(args, "drop_height", DEFAULT_DROP_CLEARANCE_M))
        self.experiment = load_experiment(_resolve_repo_path(getattr(args, "experiment", EXPERIMENT_PATH)))
        runtime = resolve_landing_config(self.experiment, dt_override=getattr(args, "dt", None))
        self.controller_config = runtime.controller
        self.dt = runtime.dt
        self.rng = np.random.default_rng(runtime.random_seed)
        if not np.isfinite(self.duration) or self.duration <= 0.0:
            raise ValueError("duration must be finite and positive")
        if not np.isfinite(drop_clearance) or drop_clearance < 0.0:
            raise ValueError("drop height must be finite and nonnegative")

        self.human_model_path = _resolve_repo_path(self.experiment.human_model_path)
        self.manifest_path = _resolve_repo_path(self.experiment.shoe_manifest_path)
        self.motion_path = _resolve_repo_path(self.experiment.motion_path) if self.experiment.motion_path else None

        self.osim_model = opensim.parse_osim(self.human_model_path)
        self.builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        self.import_result = opensim.add_osim(
            self.builder,
            self.osim_model,
            parse_contacts=False,
            parse_muscles=False,
        )
        self.resolved = resolve_attachment(self.import_result, self.experiment.attachment)
        self.carrier = self.resolved.shoe_carrier_body_index

        geo = build_foundation_geometry(self.manifest_path)
        self.foundation_geometry = geo
        manifest = load_manifest(self.manifest_path)
        midsole = load_mesh(self.manifest_path.parent / manifest.midsole_mesh, 0.001)
        midsole_vertices = np.asarray(midsole.vertices, dtype=np.float64).copy()
        midsole_vertices[:, 2] -= geo.z_shift_m
        top_interface = np.column_stack([geo.uv_m, geo.z_free_m])
        top_reference = np.broadcast_to(top_interface.mean(axis=0), midsole_vertices.shape)
        attached_mesh = attach_sole_geometry(self.resolved, midsole_vertices, top_reference)
        mesh = newton.Mesh(
            np.asarray(attached_mesh.bottom_local, dtype=np.float32).copy(),
            np.asarray(midsole.faces, dtype=np.int32).reshape(-1).copy(),
            compute_inertia=False,
        )
        self.builder.add_shape_mesh(
            self.carrier,
            mesh=mesh,
            cfg=newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False),
            color=(0.22, 0.34, 0.72),
            label="digital_instron_midsole",
        )

        column = attach_sole_geometry(self.resolved, np.column_stack([geo.uv_m, geo.z_bottom_m]), top_interface)
        if column.alignment_max_m > 0.5 * geo.spacing_m + 1.0e-9:
            raise ValueError(
                f"shoe-top contact alignment residual {column.alignment_max_m:.6f} m "
                f"exceeds half the {geo.spacing_m:.6f} m column spacing"
            )
        self.attachment_alignment_rms_m = column.alignment_rms_m
        self.attachment_alignment_max_m = column.alignment_max_m
        self.column_bottom_local = np.asarray(column.bottom_local, dtype=np.float32)
        self.column_top_local = np.asarray(column.top_local, dtype=np.float32)
        self.column_rest_len = np.asarray(column.rest_len, dtype=np.float32)
        self.column_area = np.full(len(self.column_bottom_local), geo.area_m2, dtype=np.float32)
        self._bottom_local = wp.array(self.column_bottom_local, dtype=wp.vec3, device=self.device)
        self._top_local = wp.array(self.column_top_local, dtype=wp.vec3, device=self.device)
        self._render_compression = wp.zeros(len(self.column_bottom_local), dtype=wp.float32, device=self.device)
        self._bottom_world = wp.zeros(len(self.column_bottom_local), dtype=wp.vec3, device=self.device)
        self._top_world = wp.zeros(len(self.column_bottom_local), dtype=wp.vec3, device=self.device)
        self._colors = wp.zeros(len(self.column_bottom_local), dtype=wp.vec3, device=self.device)
        self._peak_force_n = 0.0
        self._peak_compression_m = 0.0
        self._peak_active_columns = 0

        for contact in self.import_result.model.contact_geometry:
            if contact.type != "ContactSphere":
                continue
            body = self.import_result.body_index[contact.body]
            self.builder.add_shape_sphere(
                body,
                xform=wp.transform(wp.vec3(*contact.location), wp.quat_identity()),
                radius=float(contact.radius),
                as_site=True,
                color=(0.25, 0.55, 0.95),
                label=contact.name,
            )
        for body in range(self.builder.body_count):
            self.builder.add_shape_sphere(body, radius=0.014, as_site=True, color=(0.9, 0.85, 0.55))
        self.builder.add_ground_plane()
        self.model = self.builder.finalize(device=self.device)
        self._validate_import_contract()
        self.foundation_config = FoundationConfig(
            stretch_floor=0.05,
            normal_damping=40.0,
            friction_stiffness=2.0e4,
            friction=20.0,
            mu=1.0,
        )
        self.foundation = MidsoleFoundation(
            self.column_bottom_local,
            np.zeros(len(self.column_bottom_local), dtype=np.float32),
            self.column_rest_len,
            self.column_area,
            geo.neighbors,
            geo.spacing_m,
            CALIBRATED_MATERIAL,
            self.carrier,
            self.model.body_com,
            self.foundation_config,
            self.device,
        )
        self.solver = newton.solvers.SolverFeatherstone(self.model)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self._configure_control()
        self._initialize_state(drop_clearance)
        self.viewer.set_model(self.model)

        self.sim_time = 0.0
        self.frame_dt = FRAME_DT_S
        self.sim_substeps = max(1, int(round(self.frame_dt / self.dt)))
        self.frame = 0
        self.n_frames = max(1, int(np.ceil(self.duration / self.frame_dt)))

    def _validate_import_contract(self) -> None:
        """Reject importer drift that would invalidate the hard-coded root control layout."""
        if self.builder.up_axis != newton.Axis.Z:
            raise ValueError(f"expected a Z-up Newton builder, got {self.builder.up_axis}")
        gravity = self.model.gravity.numpy()
        if not np.allclose(gravity[:, :2], 0.0, atol=1.0e-6) or not np.all(gravity[:, 2] < 0.0):
            raise ValueError(f"expected gravity along Newton -Z, got {gravity.tolist()}")
        if self.model.body_count != 12 or self.model.joint_count != 12:
            raise ValueError(
                f"expected 12 bodies and 12 joints, got {self.model.body_count} bodies and {self.model.joint_count} joints"
            )
        if len(self.model.joint_q) != 27 or len(self.model.joint_qd) != 27:
            raise ValueError(
                f"expected 27 coordinates and DOFs, got {len(self.model.joint_q)} and {len(self.model.joint_qd)}"
            )
        if self.model.joint_label[0] != "ground_pelvis":
            raise ValueError(f"expected root joint 'ground_pelvis', got {self.model.joint_label[0]!r}")
        if int(self.model.joint_q_start.numpy()[0]) != 0 or int(self.model.joint_qd_start.numpy()[0]) != 0:
            raise ValueError("expected ground_pelvis coordinates and DOFs to start at index 0")

    def _configure_control(self) -> None:
        """Apply the versioned posture controller while leaving vertical pelvis motion free."""
        controller = self.controller_config
        q0 = np.asarray(self.model.joint_q.numpy(), dtype=np.float32)
        qd0 = np.asarray(self.model.joint_qd.numpy(), dtype=np.float32)
        self.control.joint_target_q.assign(wp.array(q0, dtype=wp.float32, device=self.device))
        self.control.joint_target_qd.assign(wp.array(qd0, dtype=wp.float32, device=self.device))
        joint_qd_start = np.asarray(self.model.joint_qd_start.numpy(), dtype=np.int32)
        joint_target_ke = np.zeros(len(self.model.joint_target_ke), dtype=np.float32)
        joint_target_kd = np.zeros(len(self.model.joint_target_kd), dtype=np.float32)
        joint_armature = np.full(len(self.model.joint_armature), controller.armature, dtype=np.float32)
        joint_damping = np.full(len(self.model.joint_damping), controller.damping, dtype=np.float32)
        for joint_idx in range(self.model.joint_count):
            dof_start = int(joint_qd_start[joint_idx])
            dof_end = int(joint_qd_start[joint_idx + 1]) if joint_idx + 1 < self.model.joint_count else len(qd0)
            is_root = joint_idx == 0
            for axis in range(dof_end - dof_start):
                idx = dof_start + axis
                if is_root and axis == 1:
                    joint_target_ke[idx] = 0.0
                    joint_target_kd[idx] = 0.0
                elif is_root and axis in (0, 2):
                    joint_target_ke[idx] = controller.root_horizontal_ke
                    joint_target_kd[idx] = controller.root_horizontal_kd
                elif is_root:
                    joint_target_ke[idx] = controller.root_rotation_ke
                    joint_target_kd[idx] = controller.root_rotation_kd
                else:
                    joint_target_ke[idx] = controller.joint_ke
                    joint_target_kd[idx] = controller.joint_kd
        self.model.joint_target_ke.assign(wp.array(joint_target_ke, dtype=wp.float32, device=self.device))
        self.model.joint_target_kd.assign(wp.array(joint_target_kd, dtype=wp.float32, device=self.device))
        self.model.joint_armature.assign(wp.array(joint_armature, dtype=wp.float32, device=self.device))
        self.model.joint_damping.assign(wp.array(joint_damping, dtype=wp.float32, device=self.device))

    def _initial_joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Map the configured OpenSim motion frame into Newton's imported D6 layout."""
        joint_q = np.asarray(self.model.joint_q.numpy(), dtype=np.float32)
        joint_qd = np.asarray(self.model.joint_qd.numpy(), dtype=np.float32)
        if self.motion_path is None:
            return joint_q, joint_qd

        times, coordinates = opensim.read_motion(self.osim_model, self.motion_path)
        frame = self.experiment.initial_motion_frame
        if frame >= len(times):
            raise ValueError(f"initial_motion_frame {frame} is outside motion with {len(times)} frames")
        coordinate_names = [coordinate.name for joint in self.osim_model.joints for coordinate in joint.coordinates]
        coordinate_speeds = np.gradient(coordinates, times, axis=0, edge_order=1)
        joint_qd.fill(0.0)
        missing = []
        for column, name in enumerate(coordinate_names):
            target = self.import_result.coordinate_dof.get(name)
            if target is None:
                if abs(coordinates[frame, column]) > 1.0e-10 or abs(coordinate_speeds[frame, column]) > 1.0e-10:
                    missing.append(name)
                continue
            joint_q[target] = coordinates[frame, column]
            if name == "pelvis_ty":
                # Use the measured vertical root speed, but start the controlled
                # drop without gait-cycle angular velocities fighting the pose hold.
                joint_qd[target] = coordinate_speeds[frame, column]
        if missing:
            raise ValueError(f"motion has nonzero coordinates without a Newton scalar mapping: {', '.join(missing)}")
        self.initial_motion_time_s = float(times[frame])
        return joint_q, joint_qd

    def _initialize_state(self, clearance: float) -> None:
        """Initialize the landing from the configured motion frame above the ground."""
        joint_q, joint_qd = self._initial_joint_state()
        self.state_0.joint_q.assign(wp.array(joint_q, dtype=wp.float32, device=self.device))
        self.state_0.joint_qd.assign(wp.array(joint_qd, dtype=wp.float32, device=self.device))
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        q = np.asarray(self.state_0.joint_q.numpy(), dtype=np.float32)
        q[1] = _standing_root_height(self.state_0, self.carrier, self.column_bottom_local, clearance)
        self.state_0.joint_q.assign(wp.array(q, dtype=wp.float32, device=self.device))
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        self.state_1.body_q.assign(self.state_0.body_q)
        self.state_1.body_qd.assign(self.state_0.body_qd)
        self.control.joint_target_q.assign(self.state_0.joint_q)
        self.control.joint_target_qd.zero_()
        self.foundation.reset()

    def _update_columns(self) -> None:
        wp.launch(
            _deform_columns_z_up,
            dim=len(self.column_bottom_local),
            inputs=[
                self.carrier,
                self.state_0.body_q,
                self._bottom_local,
                self._top_local,
                self._bottom_world,
                self._top_world,
                self._render_compression,
            ],
            device=self.device,
        )
        wp.launch(
            column_colors,
            dim=len(self.column_bottom_local),
            inputs=[self._render_compression, 0.012, self._colors],
            device=self.device,
        )

    def _apply_foundation(self) -> None:
        """Apply the shared calibrated Z-up foundation to the OpenSim foot."""
        self.state_0.clear_forces()
        self.foundation.apply(self.state_0, self.dt)

    def _advance_substep(self) -> None:
        self._apply_foundation()
        self.solver.step(self.state_0, self.state_1, self.control, None, self.dt)
        self.state_0, self.state_1 = self.state_1, self.state_0
        self.sim_time += self.dt

    def step(self) -> None:
        remaining = self.duration - self.sim_time
        if remaining <= 0.0:
            return
        substeps = min(self.sim_substeps, max(1, int(np.ceil(remaining / self.dt))))
        for _ in range(substeps):
            self._advance_substep()
        compression = self.foundation.compression.numpy()
        foundation_diag = self.foundation.diagnostics()
        self._peak_force_n = max(self._peak_force_n, foundation_diag["normal_force_n"])
        self._peak_compression_m = max(
            self._peak_compression_m,
            float(np.max(compression) if len(compression) else 0.0),
        )
        self._peak_active_columns = max(self._peak_active_columns, int(foundation_diag["active_columns"]))
        self.frame += 1

    def render(self) -> None:
        self._update_columns()
        self.viewer.begin_frame(self.sim_time)
        # Imported OpenSim bodies are visible as dynamic body sites, while the
        # calibrated mesh and column bed make the foot attachment explicit.
        self.viewer.log_state(self.state_0)
        self.viewer.log_points("/human_shoe/contact_points", self._bottom_world, radii=0.0025, colors=self._colors)
        self.viewer.log_lines(
            "/human_shoe/columns", self._bottom_world, self._top_world, colors=self._colors, width=0.002
        )
        self.viewer.end_frame()

    def diagnostics(self) -> LandingDiagnostics:
        """Return the current landing diagnostics."""
        compression = self.foundation.compression.numpy()
        foundation_diag = self.foundation.diagnostics()
        body_q = self.state_0.body_q.numpy()
        body_f = self.state_0.body_f.numpy()
        joint_q = self.state_0.joint_q.numpy()
        joint_qd = self.state_0.joint_qd.numpy()
        return LandingDiagnostics(
            normal_force_n=float(foundation_diag["normal_force_n"]),
            active_columns=int(foundation_diag["active_columns"]),
            compression_m=float(np.max(compression) if len(compression) else 0.0),
            peak_normal_force_n=float(self._peak_force_n),
            peak_active_columns=int(self._peak_active_columns),
            peak_compression_m=float(self._peak_compression_m),
            pelvis_height_m=float(body_q[0, 2]),
            finite=bool(
                np.all(np.isfinite(body_q))
                and np.all(np.isfinite(body_f))
                and np.all(np.isfinite(joint_q))
                and np.all(np.isfinite(joint_qd))
            ),
        )

    def test_final(self):
        """Verify the landing stayed finite and activated the sole bed."""
        diag = self.diagnostics()
        if self.frame <= 0:
            raise AssertionError("no landing frames were simulated")
        if not diag.finite:
            raise AssertionError("non-finite landing state")
        if diag.peak_normal_force_n <= 0.0:
            raise AssertionError("sole bed never generated a normal force")
        if diag.peak_active_columns <= 0:
            raise AssertionError("no sole columns activated during landing")
        if diag.peak_compression_m <= 0.0:
            raise AssertionError("no column compression was measured")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--experiment", type=str, default=str(EXPERIMENT_PATH), help="Human shoe experiment JSON.")
        parser.add_argument("--duration", type=float, default=DEFAULT_DURATION_S, help="Landing duration [s].")
        parser.add_argument(
            "--dt",
            type=float,
            default=None,
            help="Dynamics timestep [s]. Defaults to the experiment manifest value.",
        )
        parser.add_argument(
            "--drop-height",
            type=float,
            default=DEFAULT_DROP_CLEARANCE_M,
            help="Initial clearance between the lowest outsole point and the ground [m].",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
