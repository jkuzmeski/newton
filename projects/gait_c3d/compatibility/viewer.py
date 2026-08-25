# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""REFERENCE VIEWER ONLY. It imports OpenSim-compatible assets for visualization and is not a production mechanics entrypoint.

Visualize the analyzed C3D treadmill trial as exact overground gait."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton import opensim

DEFAULT_DATA_DIR = Path("/home/jo31399/newton-data/gait/processed/trial_101/latest")
ARCHITECTURE_ROLE = "compatibility_reference"

_CORE_FIELDS = (
    "times",
    "coords",
    "coordinate_names",
    "target_markers",
    "predicted_markers",
    "marker_names",
)
_SIDE_COLORS = ((0.05, 0.85, 1.0), (1.0, 0.15, 0.55))


def _strings(values: np.ndarray) -> list[str]:
    """Convert an NPZ Unicode vector to ordinary Python strings."""
    return [str(value) for value in np.asarray(values).reshape(-1)]


def _require_shape(name: str, value: np.ndarray, shape: tuple[int | None, ...]) -> np.ndarray:
    """Validate an analysis array shape while allowing named variable axes."""
    value = np.asarray(value)
    if value.ndim != len(shape) or any(
        expected is not None and actual != expected for actual, expected in zip(value.shape, shape, strict=True)
    ):
        expected_text = "[" + ", ".join("*" if item is None else str(item) for item in shape) + "]"
        raise ValueError(f"analysis field {name!r} has shape {value.shape}, expected {expected_text}")
    return value


def _numeric_leaves(value) -> np.ndarray:
    """Collect finite numeric leaves from a nested JSON value."""
    leaves: list[float] = []

    def visit(item) -> None:
        if isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        elif isinstance(item, (int, float)) and not isinstance(item, bool) and np.isfinite(item):
            leaves.append(float(item))

    visit(value)
    return np.asarray(leaves, dtype=np.float64)


def _stance_speed_rms(qc: dict) -> tuple[float, float]:
    """Read treadmill and overground stance-heel speed metrics from QC."""
    metrics = qc.get("stance_heel_speeds")
    if not isinstance(metrics, dict):
        raise AssertionError("QC does not contain stance_heel_speeds")

    def select(frame: str) -> np.ndarray:
        direct = metrics.get(frame)
        if direct is not None:
            return _numeric_leaves(direct)
        rms_values: list[float] = []
        all_values: list[float] = []

        def visit(value, path: tuple[str, ...]) -> None:
            if isinstance(value, dict):
                for key, child in value.items():
                    visit(child, (*path, str(key).lower()))
            elif isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value):
                joined = ".".join(path)
                if frame in joined:
                    all_values.append(float(value))
                    if "rms" in joined:
                        rms_values.append(float(value))

        visit(metrics, ())
        return np.asarray(rms_values or all_values, dtype=np.float64)

    treadmill = select("treadmill")
    overground = select("overground")
    if treadmill.size == 0 or overground.size == 0:
        raise AssertionError("QC stance_heel_speeds lacks treadmill or overground values")
    return float(np.sqrt(np.mean(treadmill * treadmill))), float(np.sqrt(np.mean(overground * overground)))


class Example:
    """Render exact-FK overground gait and its measured analysis overlays."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.device = wp.get_device()
        self.data_dir = Path(args.data_dir).expanduser().resolve()
        analysis_path = self.data_dir / "analysis.npz"
        model_path = self.data_dir / "S001_scaled.osim"
        qc_path = self.data_dir / "qc_summary.json"
        for path in (analysis_path, model_path, qc_path):
            if not path.is_file():
                raise FileNotFoundError(f"required gait artifact is missing: {path}")

        with np.load(analysis_path, allow_pickle=False) as archive:
            missing = [name for name in _CORE_FIELDS if name not in archive]
            if missing:
                raise ValueError(f"analysis.npz is missing required fields: {', '.join(missing)}")
            analysis = {name: np.asarray(archive[name]) for name in archive.files}
        with qc_path.open(encoding="utf-8") as stream:
            self.qc = json.load(stream)

        self.times = _require_shape("times", analysis["times"], (None,)).astype(np.float64, copy=False)
        if len(self.times) < 2 or not np.all(np.isfinite(self.times)) or not np.all(np.diff(self.times) > 0.0):
            raise ValueError("analysis times must contain at least two finite, strictly increasing samples")
        self.num_frames = len(self.times)
        stored_coordinate_names = _strings(analysis["coordinate_names"])
        coords = _require_shape("coords", analysis["coords"], (self.num_frames, len(stored_coordinate_names)))

        self.osim_model = opensim.parse_osim(str(model_path))
        model_coordinate_names = [
            coordinate.name for joint in self.osim_model.joints for coordinate in joint.coordinates
        ]
        stored_index = {name: index for index, name in enumerate(stored_coordinate_names)}
        missing_coordinates = [name for name in model_coordinate_names if name not in stored_index]
        if missing_coordinates:
            raise ValueError(f"analysis coordinates do not match scaled model: missing {missing_coordinates}")
        self.coords = np.ascontiguousarray(
            coords[:, [stored_index[name] for name in model_coordinate_names]], dtype=np.float64
        )
        self.coordinate_names = model_coordinate_names

        marker_names = _strings(analysis["marker_names"])
        self.target_markers = _require_shape(
            "target_markers", analysis["target_markers"], (self.num_frames, len(marker_names), 3)
        ).astype(np.float64, copy=False)
        self.predicted_markers = _require_shape(
            "predicted_markers", analysis["predicted_markers"], (self.num_frames, len(marker_names), 3)
        ).astype(np.float64, copy=False)
        self.marker_names = marker_names
        load_shape = (self.num_frames, 2, 3)
        self.has_loads = all(name in analysis for name in ("grf", "cop", "contact"))
        self.grf = _require_shape("grf", analysis.get("grf", np.zeros(load_shape)), load_shape).astype(
            np.float64, copy=False
        )
        self.cop = _require_shape("cop", analysis.get("cop", np.full(load_shape, np.nan)), load_shape).astype(
            np.float64, copy=False
        )
        self.free_torque = _require_shape(
            "free_torque", analysis.get("free_torque", np.zeros(load_shape)), load_shape
        ).astype(np.float64, copy=False)
        self.contact = _require_shape(
            "contact", analysis.get("contact", np.zeros((self.num_frames, 2), dtype=bool)), (self.num_frames, 2)
        ).astype(bool, copy=False)

        if "pelvis_tx" in stored_index:
            fallback_displacement = coords[:, stored_index["pelvis_tx"]] - coords[0, stored_index["pelvis_tx"]]
        else:
            fallback_displacement = np.zeros(self.num_frames)
        fallback_speed = np.gradient(fallback_displacement, self.times)
        self.has_belt_data = "belt_displacement_relative" in analysis
        self.belt_speed = _require_shape(
            "belt_speed", analysis.get("belt_speed", fallback_speed), (self.num_frames,)
        ).astype(np.float64, copy=False)
        self.belt_displacement_relative = _require_shape(
            "belt_displacement_relative",
            analysis.get("belt_displacement_relative", fallback_displacement),
            (self.num_frames,),
        ).astype(np.float64, copy=False)
        self.belt_displacement_absolute = _require_shape(
            "belt_displacement_absolute",
            analysis.get("belt_displacement_absolute", self.belt_displacement_relative),
            (self.num_frames,),
        ).astype(np.float64, copy=False)
        fallback_com = np.nanmean(self.target_markers, axis=1)
        self.com = _require_shape("com", analysis.get("com", fallback_com), (self.num_frames, 3)).astype(
            np.float64, copy=False
        )

        self.visualizer = opensim.MotionVisualizer(
            self.osim_model,
            self.coords,
            time=self.times,
            device=self.device,
        )
        geometry_dir = args.geometry
        if geometry_dir is None and args.download_geometry:
            geometry_dir = opensim.fetch_opensim_geometry()
        self.use_meshes = False
        if geometry_dir:
            self.use_meshes = self.visualizer.load_meshes(str(model_path), geometry_dir) > 0
        activation_names = _strings(analysis.get("muscle_names", np.empty(0, dtype="U1")))
        self.activations = _require_shape(
            "activations",
            analysis.get("activations", np.empty((self.num_frames, 0))),
            (self.num_frames, len(activation_names)),
        ).astype(np.float64, copy=False)
        if activation_names:
            activation_index = {name: index for index, name in enumerate(activation_names)}
            missing_muscles = [name for name in self.visualizer.muscle_names if name not in activation_index]
            if missing_muscles:
                raise ValueError(f"analysis activations do not match scaled model: missing {missing_muscles}")
            ordered = self.activations[:, [activation_index[name] for name in self.visualizer.muscle_names]]
            self.visualizer.color_muscles_by(ordered, vmin=0.0, vmax=1.0)
            reserve_gate = self.qc.get("gates", {}).get("static_optimization_reserves", {})
            if reserve_gate and not reserve_gate.get("passed", False):
                print(
                    "[gait_c3d] WARNING: Static Optimization reserve QC failed; "
                    "muscle activation colors are illustrative only."
                )
        warnings = self.qc.get("warnings", [])
        if warnings:
            print("[gait_c3d] QC WARNINGS:")
            for warning in warnings:
                print(f"  - {warning}")
        print(
            "[gait_c3d] legend: measured markers=green, predicted=blue, residuals=orange, "
            "left GRF=cyan, right GRF=magenta, COM=yellow, treadmill ghost=gray"
        )

        self.ghost_visualizer = None
        if args.show_treadmill_ghost:
            ghost_coords = self.coords.copy()
            name_to_index = {name: index for index, name in enumerate(self.coordinate_names)}
            if "pelvis_tx" not in name_to_index:
                raise ValueError("--show-treadmill-ghost requires the pelvis_tx coordinate")
            ghost_coords[:, name_to_index["pelvis_tx"]] -= self.belt_displacement_relative
            if "pelvis_tz" in name_to_index:
                ghost_coords[:, name_to_index["pelvis_tz"]] += float(args.ghost_lane_offset)
            self.ghost_visualizer = opensim.MotionVisualizer(
                self.osim_model,
                ghost_coords,
                time=self.times,
                device=self.device,
            )

        converter = opensim.OsimFrameConverter(newton.Axis.Z)
        target_world = converter.transform_vectors(self.target_markers)
        predicted_world = converter.transform_vectors(self.predicted_markers)
        self.grf_world = converter.transform_vectors(self.grf)
        self.cop_world = converter.transform_vectors(self.cop)
        self.com_world = converter.transform_vectors(self.com)

        residual_scale = float(args.residual_scale)
        if not np.isfinite(residual_scale) or residual_scale <= 0.0:
            raise ValueError("--residual-scale must be finite and positive")
        self._target_points: list[wp.array[wp.vec3]] = []
        self._target_radii: list[wp.array[wp.float32]] = []
        self._target_colors: list[wp.array[wp.vec3]] = []
        self._predicted_points: list[wp.array[wp.vec3]] = []
        self._predicted_radii: list[wp.array[wp.float32]] = []
        self._predicted_colors: list[wp.array[wp.vec3]] = []
        self._residual_starts: list[wp.array[wp.vec3]] = []
        self._residual_ends: list[wp.array[wp.vec3]] = []
        self.marker_rms = np.empty(self.num_frames, dtype=np.float64)
        for frame in range(self.num_frames):
            visible = np.all(np.isfinite(target_world[frame]), axis=1)
            target = np.ascontiguousarray(target_world[frame, visible], dtype=np.float32)
            predicted = np.ascontiguousarray(predicted_world[frame, visible], dtype=np.float32)
            residual = predicted - target
            self.marker_rms[frame] = (
                float(np.sqrt(np.mean(np.sum(residual * residual, axis=1)))) if len(target) else np.nan
            )
            self._target_points.append(wp.array(target, dtype=wp.vec3, device=self.device))
            self._target_radii.append(
                wp.array(np.full(len(target), 0.012, dtype=np.float32), dtype=wp.float32, device=self.device)
            )
            self._target_colors.append(
                wp.array(
                    np.tile(np.asarray((0.10, 0.95, 0.25), dtype=np.float32), (len(target), 1)),
                    dtype=wp.vec3,
                    device=self.device,
                )
            )
            predicted_all = np.ascontiguousarray(predicted_world[frame], dtype=np.float32)
            self._predicted_points.append(wp.array(predicted_all, dtype=wp.vec3, device=self.device))
            self._predicted_radii.append(
                wp.array(np.full(len(predicted_all), 0.008, dtype=np.float32), dtype=wp.float32, device=self.device)
            )
            self._predicted_colors.append(
                wp.array(
                    np.tile(np.asarray((0.15, 0.45, 1.0), dtype=np.float32), (len(predicted_all), 1)),
                    dtype=wp.vec3,
                    device=self.device,
                )
            )
            self._residual_starts.append(wp.array(target, dtype=wp.vec3, device=self.device))
            self._residual_ends.append(
                wp.array(np.ascontiguousarray(target + residual_scale * residual), dtype=wp.vec3, device=self.device)
            )

        grf_scale = float(args.grf_scale)
        if not np.isfinite(grf_scale) or grf_scale <= 0.0:
            raise ValueError("--grf-scale must be finite and positive")
        self._load_valid = (
            self.contact & np.all(np.isfinite(self.cop_world), axis=2) & np.all(np.isfinite(self.grf_world), axis=2)
        )
        self._cop_side: list[wp.array[wp.vec3]] = []
        self._cop_radii_side: list[wp.array[wp.float32]] = []
        self._cop_colors_side: list[wp.array[wp.vec3]] = []
        self._grf_end_side: list[wp.array[wp.vec3]] = []
        for side in range(2):
            starts = np.zeros((self.num_frames, 3), dtype=np.float32)
            valid = self._load_valid[:, side]
            starts[valid] = self.cop_world[valid, side].astype(np.float32)
            ends = starts.copy()
            ends[valid] += (grf_scale * self.grf_world[valid, side]).astype(np.float32)
            self._cop_side.append(wp.array(starts, dtype=wp.vec3, device=self.device))
            self._cop_radii_side.append(
                wp.array(np.full(self.num_frames, 0.018, dtype=np.float32), dtype=wp.float32, device=self.device)
            )
            self._cop_colors_side.append(
                wp.array(
                    np.tile(np.asarray(_SIDE_COLORS[side], dtype=np.float32), (self.num_frames, 1)),
                    dtype=wp.vec3,
                    device=self.device,
                )
            )
            self._grf_end_side.append(wp.array(ends, dtype=wp.vec3, device=self.device))
        self._com_points = wp.array(
            np.ascontiguousarray(self.com_world, dtype=np.float32), dtype=wp.vec3, device=self.device
        )
        self._com_radii = wp.array(
            np.full(self.num_frames, 0.022, dtype=np.float32), dtype=wp.float32, device=self.device
        )
        self._com_colors = wp.array(
            np.tile(np.asarray((1.0, 0.86, 0.12), dtype=np.float32), (self.num_frames, 1)),
            dtype=wp.vec3,
            device=self.device,
        )

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        opensim.add_osim(builder, self.osim_model, parse_muscles=False, parse_contacts=False)
        pelvis_tx = (
            self.coords[:, self.coordinate_names.index("pelvis_tx")]
            if "pelvis_tx" in self.coordinate_names
            else self.com[:, 0]
        )
        runway_start = float(np.min(pelvis_tx) - 1.0)
        runway_end = float(np.max(pelvis_tx) + 2.5)
        builder.add_ground_plane(color=(0.055, 0.065, 0.075), label="gait_ground")
        builder.add_shape_box(
            -1,
            xform=wp.transform(wp.vec3(0.5 * (runway_start + runway_end), 0.0, -0.012), wp.quat_identity()),
            hx=0.5 * (runway_end - runway_start),
            hy=max(0.8, abs(float(args.ghost_lane_offset)) + 0.55 if args.show_treadmill_ghost else 0.8),
            hz=0.01,
            as_site=True,
            color=(0.22, 0.24, 0.27),
            label="forward_runway",
        )
        self.model = builder.finalize(device=self.device)
        self.state = self.model.state()
        self.body_q_frames = self.visualizer.body_transforms(self.model.body_label)
        self._body_frames_numpy = self.body_q_frames.numpy()
        self._pelvis_body = self.model.body_label.index("pelvis") if "pelvis" in self.model.body_label else None

        self.frame = 0
        self.frame_dt = float(np.mean(np.diff(self.times)))
        self.fps = 1.0 / self.frame_dt
        self.sim_time = 0.0
        wp.copy(self.state.body_q, self.body_q_frames[0])
        self.viewer.set_model(self.model)
        self._update_camera()

    def _update_camera(self) -> None:
        """Track pelvis progression with a stable screenshot-friendly view."""
        if not hasattr(self.viewer, "set_camera"):
            return
        if self._pelvis_body is not None:
            pelvis = self._body_frames_numpy[self.frame, self._pelvis_body, :3]
            x = float(pelvis[0])
            z = float(pelvis[2])
        else:
            x = float(self.com_world[self.frame, 0])
            z = float(self.com_world[self.frame, 2])
        self.viewer.set_camera(pos=wp.vec3(x, -3.2, max(1.15, z + 0.15)), pitch=-6.0, yaw=90.0)

    def step(self) -> None:
        """Advance exact kinematic playback by one analysis frame."""
        self.frame = (self.frame + 1) % self.num_frames
        self.sim_time += self.frame_dt
        wp.copy(self.state.body_q, self.body_q_frames[self.frame])
        self._update_camera()

    def render(self) -> None:
        """Render skeleton, muscles, measurements, loads, and QC scalars."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        if self.use_meshes:
            self.visualizer.render_meshes(self.viewer, self.frame)
        else:
            self.visualizer.render_skeleton(self.viewer, self.frame, name="/gait/skeleton")
        self.visualizer.render_muscles(self.viewer, self.frame, name="/gait/muscles")
        if self.ghost_visualizer is not None:
            self.ghost_visualizer.render_skeleton(
                self.viewer,
                self.frame,
                name="/gait/treadmill_ghost",
                color=(0.42, 0.44, 0.47),
            )
        self.viewer.log_points(
            "/gait/markers/measured",
            self._target_points[self.frame],
            radii=self._target_radii[self.frame],
            colors=self._target_colors[self.frame],
        )
        self.viewer.log_points(
            "/gait/markers/predicted",
            self._predicted_points[self.frame],
            radii=self._predicted_radii[self.frame],
            colors=self._predicted_colors[self.frame],
        )
        self.viewer.log_lines(
            "/gait/markers/residual_magnified",
            self._residual_starts[self.frame],
            self._residual_ends[self.frame],
            (1.0, 0.35, 0.08),
            width=0.004,
        )
        for side, name in enumerate(("left", "right")):
            valid = bool(self._load_valid[self.frame, side])
            self.viewer.log_points(
                f"/gait/cop/{name}",
                self._cop_side[side][self.frame : self.frame + 1],
                radii=self._cop_radii_side[side][self.frame : self.frame + 1],
                colors=self._cop_colors_side[side][self.frame : self.frame + 1],
                hidden=not valid,
            )
            self.viewer.log_arrows(
                f"/gait/grf/{name}",
                self._cop_side[side][self.frame : self.frame + 1],
                self._grf_end_side[side][self.frame : self.frame + 1],
                colors=_SIDE_COLORS[side],
                width=0.009,
                hidden=not valid,
            )
        trail_start = max(0, self.frame - int(self.args.com_trail_frames))
        self.viewer.log_points(
            "/gait/com/current",
            self._com_points[self.frame : self.frame + 1],
            radii=self._com_radii[self.frame : self.frame + 1],
            colors=self._com_colors[self.frame : self.frame + 1],
        )
        if self.frame > trail_start:
            self.viewer.log_lines(
                "/gait/com/trail",
                self._com_points[trail_start : self.frame],
                self._com_points[trail_start + 1 : self.frame + 1],
                (1.0, 0.65, 0.08),
                width=0.004,
            )
        self.viewer.log_scalar("/gait/belt_speed_mps", self.belt_speed[self.frame])
        self.viewer.log_scalar("/gait/marker_rms_m", self.marker_rms[self.frame])
        self.viewer.log_scalar("/gait/grf/left_fz_n", self.grf_world[self.frame, 0, 2])
        self.viewer.log_scalar("/gait/grf/right_fz_n", self.grf_world[self.frame, 1, 2])
        self.viewer.log_scalar("/gait/overground_displacement_m", self.belt_displacement_relative[self.frame])
        self.viewer.end_frame()

    def test_final(self) -> None:
        """Verify finite exact FK, overground progression, stance correction, and bilateral contact."""
        poses = self._body_frames_numpy
        if not np.all(np.isfinite(poses)) or not np.all(np.isfinite(self.coords)):
            raise AssertionError("exact-FK playback contains non-finite poses or coordinates")
        if not np.all(np.isfinite(self.predicted_markers)):
            raise AssertionError("predicted marker positions are non-finite")
        target_finite = np.isfinite(self.target_markers)
        complete_target = np.all(target_finite, axis=2) | ~np.any(target_finite, axis=2)
        if not np.all(complete_target) or np.any(np.sum(np.all(target_finite, axis=2), axis=1) == 0):
            raise AssertionError("measured marker positions contain partial vectors or an empty frame")
        if self.has_loads:
            if not np.all(np.isfinite(self.grf)) or not np.all(np.isfinite(self.free_torque)):
                raise AssertionError("ground-reaction loads contain non-finite values")
            if not np.all(np.isfinite(self.cop[self.contact])):
                raise AssertionError("contact-active COP positions contain non-finite values")
        if not np.all(np.isfinite(self.com)) or not np.all(np.isfinite(self.marker_rms)):
            raise AssertionError("COM or marker residual output contains non-finite values")
        if self.activations.shape[1] and (
            not np.all(np.isfinite(self.activations))
            or float(np.min(self.activations)) < -1.0e-6
            or float(np.max(self.activations)) > 1.0 + 1.0e-6
        ):
            raise AssertionError("muscle activations are non-finite or outside [0, 1]")

        if "pelvis_tx" not in self.coordinate_names:
            raise AssertionError("scaled model has no pelvis_tx progression coordinate")
        pelvis_tx = self.coords[:, self.coordinate_names.index("pelvis_tx")]
        progression = float(pelvis_tx[-1] - pelvis_tx[0])
        if progression <= 0.05:
            raise AssertionError(f"overground pelvis progression is not positive: {progression:.3f} m")
        if (
            self.has_belt_data
            and float(self.belt_displacement_relative[-1] - self.belt_displacement_relative[0]) <= 0.05
        ):
            raise AssertionError("relative belt displacement does not advance overground")

        treadmill_rms, overground_rms = _stance_speed_rms(self.qc)
        if not overground_rms < treadmill_rms:
            raise AssertionError(
                "overground correction did not reduce stance-heel speed: "
                f"{overground_rms:.4f} vs treadmill {treadmill_rms:.4f} m/s"
            )
        if self.has_loads:
            if not np.all(np.any(self.contact, axis=0)):
                raise AssertionError("the selected stride does not contain contact for both feet")
            peak_fz = np.max(self.grf_world[:, :, 2], axis=0)
            if np.any(peak_fz <= 50.0):
                raise AssertionError(f"bilateral vertical GRF is missing or implausible: {peak_fz} N")
            if not np.any(self.contact[:, 0] & self.contact[:, 1]):
                raise AssertionError("the selected gait stride contains no double-support contact")

    @staticmethod
    def create_parser():
        """Create the standard Newton example command-line parser."""
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--reference-only",
            action="store_true",
            help="required acknowledgement: this visualizes OpenSim-compatible reference artifacts",
        )
        parser.add_argument(
            "--data-dir",
            type=str,
            default=str(DEFAULT_DATA_DIR),
            help="Pipeline output directory containing analysis.npz, S001_scaled.osim, and qc_summary.json.",
        )
        parser.add_argument(
            "--geometry",
            type=str,
            default=None,
            help="Directory containing OpenSim .vtp bone geometry.",
        )
        parser.add_argument(
            "--download-geometry",
            action="store_true",
            help="Download the commit-pinned OpenSim geometry collection.",
        )
        parser.add_argument(
            "--residual-scale",
            type=float,
            default=8.0,
            help="Visual magnification applied to marker residual lines.",
        )
        parser.add_argument(
            "--grf-scale",
            type=float,
            default=5.0e-4,
            help="GRF arrow length scale [m/N].",
        )
        parser.add_argument(
            "--com-trail-frames",
            type=int,
            default=120,
            help="Maximum number of preceding COM samples in the trail.",
        )
        parser.add_argument(
            "--show-treadmill-ghost",
            action="store_true",
            help="Render a gray treadmill-frame skeleton in a laterally offset lane.",
        )
        parser.add_argument(
            "--ghost-lane-offset",
            type=float,
            default=-0.9,
            metavar="METERS",
            help="Treadmill ghost offset along OpenSim's lateral pelvis_tz coordinate [m].",
        )
        return parser


def main() -> None:
    """Run the analyzed overground gait visualizer."""
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    if not args.reference_only:
        parser.error("--reference-only is required; this viewer consumes compatibility artifacts")
    newton.examples.run(Example(viewer, args), args)


if __name__ == "__main__":
    main()
