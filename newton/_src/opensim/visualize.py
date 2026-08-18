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

r"""Warp-native visualization of OpenSim motions.

Turns a coordinate trajectory (an OpenSim ``.mot``/``.sto`` motion, an inverse-
kinematics result, or any batch of generalized coordinates) into per-frame
renderables for a Newton :class:`~newton.viewer.ViewerBase`:

- **Body transforms** driving each imported body's :attr:`~newton.State.body_q`,
  computed with the OpenSim-exact forward kinematics
  (:class:`~newton.opensim.ForwardKinematics`). This reproduces
  ``CustomJoint`` ``SpatialTransform`` coupling (e.g. the gait2354 ``SimmSpline``
  knee translation) that Newton's generic joints do not, so the rendered skeleton
  matches OpenSim frame-for-frame.
- **Bone segments** connecting each joint's parent- and child-body origins.
- **Muscle-tendon paths** as poly-lines through their active ``GeometryPath``
  points (fixed / conditional / moving), colored by normalized muscle-tendon
  length so lengthening muscles light up over the gait cycle.

All per-frame geometry (body transforms, bone endpoints, world-space muscle path
points, segment assembly, and length coloring) is precomputed once on the Warp
device with :mod:`warp` kernels, so playback is a per-frame array slice plus
:meth:`~newton.viewer.ViewerBase.log_lines` calls.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import numpy as np
import warp as wp

from ..core import Axis
from ..utils.download_assets import download_git_folder
from .frame import OsimFrameConverter
from .kinematics import ForwardKinematics, euler_xyz_to_matrix
from .mocap import Storage, read_storage
from .muscle_path import MusclePaths
from .types import OsimModel

# opensim-org/opensim-models Geometry folder, pinned for reproducible downloads.
_OPENSIM_MODELS_URL = "https://github.com/opensim-org/opensim-models.git"
_OPENSIM_MODELS_REF = "d9b05d470b1a481c222372c85b75772faf8f7792"

_f64 = wp.float64
_mat44d = wp.mat44d
_vec3d = wp.vec3d


def _transform_to_mat44d(xform: wp.transform) -> wp.mat44d:
    """Convert a Warp transform to a double-precision homogeneous matrix."""
    values = np.asarray(xform, dtype=np.float64)
    rotation = np.asarray(wp.quat_to_matrix(wp.quat(*values[3:])), dtype=np.float64).reshape(3, 3)
    translation = values[:3]
    return wp.mat44d(
        rotation[0, 0],
        rotation[0, 1],
        rotation[0, 2],
        translation[0],
        rotation[1, 0],
        rotation[1, 1],
        rotation[1, 2],
        translation[1],
        rotation[2, 0],
        rotation[2, 1],
        rotation[2, 2],
        translation[2],
        0.0,
        0.0,
        0.0,
        1.0,
    )


@wp.kernel
def _apply_world_xform_kernel(world_xform: _mat44d, poses: wp.array2d[_mat44d]):
    """Left-compose an OpenSim body pose with the target Newton world transform."""
    frame, body = wp.tid()
    poses[frame, body] = world_xform * poses[frame, body]


@wp.kernel
def _body_xform_gather_kernel(
    poses: wp.array2d[_mat44d],
    gather: wp.array[wp.int32],
    out: wp.array2d[wp.transformf],
):
    """Convert gathered ``float64`` body pose matrices to ``transformf`` per frame.

    Launched with dim ``(num_frames, num_targets)``. ``gather[i]`` selects the
    source body column so the output columns follow a caller-supplied body order.
    """
    b, i = wp.tid()
    x = poses[b, gather[i]]
    r = wp.mat33(
        wp.float32(x[0, 0]),
        wp.float32(x[0, 1]),
        wp.float32(x[0, 2]),
        wp.float32(x[1, 0]),
        wp.float32(x[1, 1]),
        wp.float32(x[1, 2]),
        wp.float32(x[2, 0]),
        wp.float32(x[2, 1]),
        wp.float32(x[2, 2]),
    )
    q = wp.quat_from_matrix(r)
    p = wp.vec3(wp.float32(x[0, 3]), wp.float32(x[1, 3]), wp.float32(x[2, 3]))
    out[b, i] = wp.transformf(p, q)


@wp.kernel
def _bone_kernel(
    poses: wp.array2d[_mat44d],
    bone_parent: wp.array[wp.int32],
    bone_child: wp.array[wp.int32],
    starts: wp.array2d[wp.vec3],
    ends: wp.array2d[wp.vec3],
):
    """Set each bone segment to span its parent- and child-body origins."""
    b, s = wp.tid()
    xp = poses[b, bone_parent[s]]
    xc = poses[b, bone_child[s]]
    starts[b, s] = wp.vec3(wp.float32(xp[0, 3]), wp.float32(xp[1, 3]), wp.float32(xp[2, 3]))
    ends[b, s] = wp.vec3(wp.float32(xc[0, 3]), wp.float32(xc[1, 3]), wp.float32(xc[2, 3]))


@wp.kernel
def _world_points_kernel(
    poses: wp.array2d[_mat44d],
    point_body: wp.array[wp.int32],
    point_loc: wp.array2d[_vec3d],
    out: wp.array2d[wp.vec3],
):
    """Transform each muscle path point into ground and downcast to ``float32``."""
    b, p = wp.tid()
    x = poses[b, point_body[p]]
    loc = point_loc[b, p]
    out[b, p] = wp.vec3(
        wp.float32(x[0, 0] * loc[0] + x[0, 1] * loc[1] + x[0, 2] * loc[2] + x[0, 3]),
        wp.float32(x[1, 0] * loc[0] + x[1, 1] * loc[1] + x[1, 2] * loc[2] + x[1, 3]),
        wp.float32(x[2, 0] * loc[0] + x[2, 1] * loc[1] + x[2, 2] * loc[2] + x[2, 3]),
    )


@wp.kernel
def _segments_kernel(
    points: wp.array2d[wp.vec3],
    point_active: wp.array2d[wp.int32],
    seg_a: wp.array[wp.int32],
    seg_b: wp.array[wp.int32],
    starts: wp.array2d[wp.vec3],
    ends: wp.array2d[wp.vec3],
):
    """Build muscle line segments; collapse a segment when either point is gated off."""
    b, s = wp.tid()
    point_a = seg_a[s]
    point_b = seg_b[s]
    pa = points[b, point_a]
    pc = points[b, point_b]
    if point_active[b, point_a] == 0 or point_active[b, point_b] == 0:
        pc = pa
    starts[b, s] = pa
    ends[b, s] = pc


@wp.kernel
def _length_range_kernel(
    lengths: wp.array2d[_f64], nframe: int, length_min: wp.array[_f64], length_max: wp.array[_f64]
):
    """Reduce each muscle's trajectory length range on device."""
    muscle = wp.tid()
    minimum = lengths[0, muscle]
    maximum = minimum
    for frame in range(1, nframe):
        minimum = wp.min(minimum, lengths[frame, muscle])
        maximum = wp.max(maximum, lengths[frame, muscle])
    length_min[muscle] = minimum
    length_max[muscle] = maximum


@wp.kernel
def _muscle_color_kernel(
    lengths: wp.array2d[_f64],
    length_min: wp.array[_f64],
    length_max: wp.array[_f64],
    seg_muscle: wp.array[wp.int32],
    color_lo: wp.vec3,
    color_hi: wp.vec3,
    out: wp.array2d[wp.vec3],
):
    """Color each muscle segment by its normalized muscle-tendon length in ``[0, 1]``."""
    b, s = wp.tid()
    m = seg_muscle[s]
    denom = length_max[m] - length_min[m]
    t = float(0.0)
    if denom > _f64(1.0e-9):
        t = wp.float32((lengths[b, m] - length_min[m]) / denom)
    t = wp.clamp(t, 0.0, 1.0)
    out[b, s] = color_lo * (1.0 - t) + color_hi * t


@wp.kernel
def _skin_kernel(
    poses: wp.array2d[_mat44d],
    frame: wp.int32,
    body: wp.int32,
    local: wp.array[wp.vec3],
    out: wp.array[wp.vec3],
):
    """Rigidly transform a mesh's body-frame vertices into ground for one frame."""
    v = wp.tid()
    x = poses[frame, body]
    p = local[v]
    out[v] = wp.vec3(
        wp.float32(x[0, 0]) * p[0] + wp.float32(x[0, 1]) * p[1] + wp.float32(x[0, 2]) * p[2] + wp.float32(x[0, 3]),
        wp.float32(x[1, 0]) * p[0] + wp.float32(x[1, 1]) * p[1] + wp.float32(x[1, 2]) * p[2] + wp.float32(x[1, 3]),
        wp.float32(x[2, 0]) * p[0] + wp.float32(x[2, 1]) * p[1] + wp.float32(x[2, 2]) * p[2] + wp.float32(x[2, 3]),
    )


class MotionVisualizer:
    """Precompute Warp renderables for an OpenSim coordinate trajectory.

    Args:
        model: Parsed OpenSim model (see :func:`~newton.opensim.parse_osim`).
        coords: Coordinate trajectory, shape ``[num_frames, num_coordinates]`` in
            native units (radians / meters), column order
            :attr:`~newton.opensim.ForwardKinematics.coordinate_names`.
        time: Optional frame times [s], shape ``[num_frames]``. Defaults to frame
            indices.
        device: Warp device for the kernels (defaults to the CPU, matching the
            rest of the port).
        up_axis: Target Newton world up axis. Defaults to :attr:`newton.Axis.Z`.
        world_xform: Optional explicit transform from OpenSim's Y-up world into
            the target Newton world. When provided, this overrides ``up_axis``.
        muscle_color: ``(short_rgb, long_rgb)`` endpoints of the per-muscle
            length colormap. Short muscles use ``short_rgb``; fully lengthened
            muscles use ``long_rgb``.

    Attributes:
        num_frames: Number of trajectory frames.
        time: Frame times [s], shape ``[num_frames]``.
        body_names: Forward-kinematics body order (``"ground"`` first).
        muscle_names: Muscle names in path order.
    """

    def __init__(
        self,
        model: OsimModel,
        coords: np.ndarray,
        *,
        time: np.ndarray | None = None,
        device=None,
        up_axis: Axis = Axis.Z,
        world_xform: wp.transform | None = None,
        muscle_color: tuple[tuple[float, float, float], tuple[float, float, float]] = (
            (0.30, 0.32, 0.65),
            (0.95, 0.10, 0.12),
        ),
    ):
        self.model = model
        self.paths = MusclePaths(model, device=device)
        self.fk: ForwardKinematics = self.paths.fk
        self.device = self.fk.device
        if world_xform is None:
            world_xform = OsimFrameConverter(up_axis).world_xform
        self.world_xform = world_xform

        coords = np.ascontiguousarray(np.atleast_2d(coords), dtype=np.float64)
        if coords.shape[1] != self.fk.ncoord:
            raise ValueError(
                f"coords has {coords.shape[1]} columns, expected {self.fk.ncoord} (one per model coordinate)"
            )
        self.num_frames = coords.shape[0]
        self.time = (
            np.asarray(time, dtype=float).reshape(-1) if time is not None else np.arange(self.num_frames, dtype=float)
        )
        self.body_names = self.fk.body_names
        self.muscle_names = self.paths.muscle_names
        self._muscle_color_lo = wp.vec3(*[float(c) for c in muscle_color[0]])
        self._muscle_color_hi = wp.vec3(*[float(c) for c in muscle_color[1]])

        # OpenSim-exact body pose matrices for every frame (float64, on device).
        self._q_wp = wp.array(coords, dtype=_f64, device=self.device)
        self._poses = self.fk._launch_body_transforms(self._q_wp)  # [num_frames, nbody]
        wp.launch(
            _apply_world_xform_kernel,
            dim=self._poses.shape,
            inputs=[_transform_to_mat44d(self.world_xform), self._poses],
            device=self.device,
        )

        self._build_bones()
        self._build_muscles()

        # Optional skinned display meshes (populated by ``load_meshes``).
        self._meshes: list[dict] = []
        self.num_meshes = 0

    # -- setup ---------------------------------------------------------------
    def _build_bones(self) -> None:
        bidx = {n: i for i, n in enumerate(self.body_names)}
        parent, child = [], []
        for j in self.model.joints:
            if j.parent_body == "ground":
                continue
            if j.parent_body in bidx and j.child_body in bidx:
                parent.append(bidx[j.parent_body])
                child.append(bidx[j.child_body])
        self.num_bones = len(parent)
        self.bone_starts = wp.zeros((self.num_frames, self.num_bones), dtype=wp.vec3, device=self.device)
        self.bone_ends = wp.zeros((self.num_frames, self.num_bones), dtype=wp.vec3, device=self.device)
        if self.num_bones:
            d_parent = wp.array(np.asarray(parent, np.int32), dtype=wp.int32, device=self.device)
            d_child = wp.array(np.asarray(child, np.int32), dtype=wp.int32, device=self.device)
            wp.launch(
                _bone_kernel,
                dim=(self.num_frames, self.num_bones),
                inputs=[self._poses, d_parent, d_child, self.bone_starts, self.bone_ends],
                device=self.device,
            )

    def _build_muscles(self) -> None:
        paths = self.paths
        npoint = paths.npoint
        # Per-frame path-point locations (moving points) and gating (conditional
        # points), evaluated on device by the shared muscle-path kernel.
        d_loc, d_active = paths._sample_points(self._q_wp)

        world = wp.zeros((self.num_frames, npoint), dtype=wp.vec3, device=self.device)
        if npoint:
            wp.launch(
                _world_points_kernel,
                dim=(self.num_frames, npoint),
                inputs=[self._poses, paths.d_point_body, d_loc, world],
                device=self.device,
            )

        # Segments: consecutive point pairs within each muscle path.
        off = paths._musc_off
        seg_a, seg_b, seg_m = [], [], []
        for m in range(len(paths.muscle_names)):
            for p in range(int(off[m]), int(off[m + 1]) - 1):
                seg_a.append(p)
                seg_b.append(p + 1)
                seg_m.append(m)
        self.num_segments = len(seg_a)
        self._seg_muscle = np.asarray(seg_m, np.int32)
        self.muscle_starts = wp.zeros((self.num_frames, self.num_segments), dtype=wp.vec3, device=self.device)
        self.muscle_ends = wp.zeros((self.num_frames, self.num_segments), dtype=wp.vec3, device=self.device)
        self.muscle_colors = wp.zeros((self.num_frames, self.num_segments), dtype=wp.vec3, device=self.device)
        if not self.num_segments:
            return

        d_seg_a = wp.array(np.asarray(seg_a, np.int32), dtype=wp.int32, device=self.device)
        d_seg_b = wp.array(np.asarray(seg_b, np.int32), dtype=wp.int32, device=self.device)
        wp.launch(
            _segments_kernel,
            dim=(self.num_frames, self.num_segments),
            inputs=[world, d_active, d_seg_a, d_seg_b, self.muscle_starts, self.muscle_ends],
            device=self.device,
        )

        # Color by normalized muscle-tendon length over the trajectory.
        d_len = paths._lengths_qwp(self._q_wp)
        d_lmin = wp.empty(len(paths.muscle_names), dtype=_f64, device=self.device)
        d_lmax = wp.empty(len(paths.muscle_names), dtype=_f64, device=self.device)
        wp.launch(
            _length_range_kernel,
            dim=len(paths.muscle_names),
            inputs=[d_len, self.num_frames, d_lmin, d_lmax],
            device=self.device,
        )
        d_seg_m = wp.array(np.asarray(seg_m, np.int32), dtype=wp.int32, device=self.device)
        lo, hi = self._muscle_color_lo, self._muscle_color_hi
        wp.launch(
            _muscle_color_kernel,
            dim=(self.num_frames, self.num_segments),
            inputs=[d_len, d_lmin, d_lmax, d_seg_m, lo, hi, self.muscle_colors],
            device=self.device,
        )

    # -- rendering -----------------------------------------------------------
    def render_skeleton(
        self,
        viewer,
        frame: int,
        *,
        name: str = "skeleton",
        color: tuple[float, float, float] = (0.85, 0.86, 0.92),
        hidden: bool = False,
    ) -> None:
        """Log the bone segments for ``frame`` to ``viewer`` as lines."""
        if self.num_bones:
            viewer.log_lines(name, self.bone_starts[frame], self.bone_ends[frame], color, hidden=hidden)

    def render_muscles(
        self,
        viewer,
        frame: int,
        *,
        name: str = "muscles",
        hidden: bool = False,
    ) -> None:
        """Log the muscle path segments for ``frame``, colored by muscle length."""
        if self.num_segments:
            viewer.log_lines(
                name,
                self.muscle_starts[frame],
                self.muscle_ends[frame],
                self.muscle_colors[frame],
                hidden=hidden,
            )

    def color_muscles_by(
        self,
        values: np.ndarray,
        *,
        times: np.ndarray | None = None,
        vmin: float = 0.0,
        vmax: float = 1.0,
    ) -> None:
        """Recolor the muscle paths by a per-frame, per-muscle scalar field.

        Overrides the default length-based coloring so an analysis result (for
        example Static-Optimization activations) can be shown directly on the
        muscle geometry: low values render blue, mid values yellow, and high
        values red. Call once after construction; :meth:`render_muscles` then
        uses the new colors.

        Args:
            values: Per-frame per-muscle scalars in :attr:`muscle_names` order,
                shape ``[len(times) or num_frames, num_muscles]``.
            times: Sample times [s] for ``values``; when given, the field is
                linearly resampled onto the visualizer's frame times so a coarse
                analysis can drive a finer playback. Defaults to the frames.
            vmin: Value mapped to the low (blue) end of the colormap.
            vmax: Value mapped to the high (red) end of the colormap.
        """
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.muscle_names):
            raise ValueError(f"values must have shape [frames, {len(self.muscle_names)}] (one column per muscle)")
        if times is not None:
            times = np.asarray(times, dtype=float).reshape(-1)
            values = np.stack([np.interp(self.time, times, values[:, m]) for m in range(values.shape[1])], axis=1)
        if values.shape[0] != self.num_frames:
            raise ValueError(f"values has {values.shape[0]} frames, expected {self.num_frames}; pass times to resample")
        if not self.num_segments:
            return
        t = np.clip((values - vmin) / max(vmax - vmin, 1e-12), 0.0, 1.0)
        stops = np.array([[0.30, 0.32, 0.65], [0.95, 0.85, 0.20], [0.95, 0.10, 0.12]])
        xs = np.array([0.0, 0.5, 1.0])
        seg_t = t[:, self._seg_muscle]  # [num_frames, num_segments]
        colors = np.empty((self.num_frames, self.num_segments, 3), np.float32)
        for ch in range(3):
            colors[:, :, ch] = np.interp(seg_t, xs, stops[:, ch])
        self.muscle_colors = wp.array(colors, dtype=wp.vec3, device=self.device)

    # -- display meshes ------------------------------------------------------
    def load_meshes(
        self,
        source,
        geometry_dir: str,
        *,
        include_ground: bool = False,
        color: tuple[float, float, float] = (0.87, 0.85, 0.80),
    ) -> int:
        """Load per-body OpenSim display geometry so the skeleton can be shown as solid bones.

        The model's ``.vtp`` display-geometry references (with their subject-specific
        scale factors and body-frame offsets) are read from ``source`` and the
        meshes loaded from ``geometry_dir``. Each mesh's vertices are baked into
        its body frame; :meth:`render_meshes` then rigidly transforms them into
        ground per frame with a Warp kernel, so the bones track the same
        OpenSim-exact body poses as the stick-figure skeleton.

        Missing files are skipped (with a side-aware fallback, e.g. ``femur.vtp``
        -> ``femur_r.vtp`` for body ``femur_r``); only ASCII VTK ``.vtp`` geometry
        is supported. Use :func:`fetch_opensim_geometry` to obtain the standard
        OpenSim bone meshes.

        Args:
            source: The ``.osim`` path (or XML text) carrying the display geometry.
            geometry_dir: Directory containing the referenced ``.vtp`` mesh files.
            include_ground: Also load geometry attached to ``ground`` (e.g. a
                treadmill). Off by default so the viewer's own ground plane shows.
            color: Default bone color used when logging the meshes.

        Returns:
            The number of meshes loaded.
        """
        geom = read_display_geometry(source)
        bidx = {n: i for i, n in enumerate(self.body_names)}
        self._meshes = []
        for body, entries in geom.items():
            if body not in bidx:
                continue
            if body == "ground" and not include_ground:
                continue
            for k, (fname, xform, scale) in enumerate(entries):
                path = _resolve_geometry_file(fname, body, geometry_dir)
                if path is None:
                    continue
                pts, tris = _read_vtp(path)
                if len(tris) == 0:
                    continue
                local = (pts.astype(np.float64) * scale) @ xform[:3, :3].T + xform[:3, 3]
                stem = os.path.splitext(os.path.basename(path))[0]
                self._meshes.append(
                    {
                        "name": f"bone/{body}/{k}_{stem}",
                        "body": bidx[body],
                        "local": wp.array(local.astype(np.float32), dtype=wp.vec3, device=self.device),
                        "indices": wp.array(tris.reshape(-1), dtype=wp.int32, device=self.device),
                        "world": wp.zeros(len(local), dtype=wp.vec3, device=self.device),
                        "color": color,
                    }
                )
        self.num_meshes = len(self._meshes)
        return self.num_meshes

    def render_meshes(self, viewer, frame: int, *, color: tuple[float, float, float] | None = None) -> None:
        """Skin and log the loaded display meshes for ``frame`` (call :meth:`load_meshes` first)."""
        for mesh in self._meshes:
            wp.launch(
                _skin_kernel,
                dim=len(mesh["local"]),
                inputs=[self._poses, int(frame), mesh["body"], mesh["local"], mesh["world"]],
                device=self.device,
            )
            viewer.log_mesh(
                mesh["name"],
                mesh["world"],
                mesh["indices"],
                color=color or mesh["color"],
                backface_culling=False,
            )

    # -- body-transform alignment --------------------------------------------
    def body_transforms(self, body_names: list[str]) -> wp.array:
        """Return per-frame body transforms aligned to ``body_names``.

        Args:
            body_names: Target body order (e.g. a finalized Newton model's
                ``body_label``). Each name must be a body of the source model.

        Returns:
            A ``wp.array2d[wp.transformf]`` of shape ``[num_frames, len(body_names)]``
            whose columns follow ``body_names``; row ``f`` can be copied directly
            into a :attr:`~newton.State.body_q` for frame ``f``.
        """
        index = {n: i for i, n in enumerate(self.body_names)}
        try:
            gather = np.asarray([index[n] for n in body_names], np.int32)
        except KeyError as exc:  # pragma: no cover - defensive
            raise ValueError(f"body {exc} is not part of the OpenSim model") from exc
        out = wp.zeros((self.num_frames, len(body_names)), dtype=wp.transformf, device=self.device)
        d_gather = wp.array(gather, dtype=wp.int32, device=self.device)
        wp.launch(
            _body_xform_gather_kernel,
            dim=(self.num_frames, len(body_names)),
            inputs=[self._poses, d_gather, out],
            device=self.device,
        )
        return out


def read_motion(
    model: OsimModel, source, *, coordinate_names: list[str] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Read an OpenSim ``.mot``/``.sto`` motion into a coordinate trajectory.

    Columns are matched to the model's coordinates by name and rotational
    coordinates are converted to radians when the storage is in degrees, so the
    result is ready for :class:`MotionVisualizer` (and the rest of the port).

    Args:
        model: Parsed OpenSim model whose coordinate order defines the columns.
        source: Path to a ``.mot``/``.sto`` file (or a :class:`~newton.opensim.Storage`).
        coordinate_names: Optional explicit coordinate order; defaults to the
            model's coordinate order.

    Returns:
        A ``(time, coords)`` tuple: frame times [s], shape ``[num_frames]``, and
        the coordinate trajectory [radians / meters], shape
        ``[num_frames, num_coordinates]``.
    """
    storage = source if isinstance(source, Storage) else read_storage(source)
    if coordinate_names is None:
        coordinate_names = [c.name for j in model.joints for c in j.coordinates]
    motion = {c.name: c.motion_type for j in model.joints for c in j.coordinates}
    label_index = {lab: i for i, lab in enumerate(storage.labels)}
    coords = np.zeros((storage.data.shape[0], len(coordinate_names)))
    for i, name in enumerate(coordinate_names):
        j = label_index.get(name)
        if j is None:
            continue
        col = storage.data[:, j]
        if storage.in_degrees and motion.get(name) == "rotational":
            col = np.deg2rad(col)
        coords[:, i] = col
    return np.asarray(storage.times, dtype=float), coords


def _vtp_floats(text: str | None) -> list[float]:
    return [float(x) for x in (text or "").replace(",", " ").split()]


def _read_vtp(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read an ASCII VTK XML PolyData ``.vtp`` into ``(points[N, 3], triangles[M, 3])``.

    Args:
        path: Path to an ASCII ``.vtp`` file (OpenSim display geometry).

    Returns:
        ``(points, triangles)`` as ``float32``/``int32`` arrays. ``Polys`` and
        ``Strips`` cells are triangulated (fan / strip).

    Raises:
        NotImplementedError: A DataArray uses a non-ASCII (binary/appended) format.
        ValueError: The file is not VTK PolyData.
    """
    piece = ET.parse(path).getroot().find(".//PolyData/Piece")
    if piece is None:
        raise ValueError(f"{path}: not a VTK PolyData (.vtp) file")

    def _array(parent, name=None):
        if parent is None:
            return None
        for da in parent.findall("DataArray"):
            if name is None or da.get("Name") == name:
                if (da.get("format") or "ascii").lower() != "ascii":
                    raise NotImplementedError(f"{path}: only ASCII .vtp geometry is supported")
                return _vtp_floats(da.text)
        return None

    points = np.asarray(_array(piece.find("Points")), dtype=np.float32).reshape(-1, 3)
    tris: list[tuple[int, int, int]] = []
    polys = piece.find("Polys")
    if polys is not None:
        conn = np.asarray(_array(polys, "connectivity"), dtype=np.int64)
        start = 0
        for end in np.asarray(_array(polys, "offsets"), dtype=np.int64):
            face = conn[start : int(end)]
            for k in range(1, len(face) - 1):  # fan triangulation
                tris.append((int(face[0]), int(face[k]), int(face[k + 1])))
            start = int(end)
    strips = piece.find("Strips")
    if strips is not None:
        conn = np.asarray(_array(strips, "connectivity"), dtype=np.int64)
        start = 0
        for end in np.asarray(_array(strips, "offsets"), dtype=np.int64):
            s = conn[start : int(end)]
            for k in range(len(s) - 2):
                a, b, c = int(s[k]), int(s[k + 1]), int(s[k + 2])
                tris.append((a, b, c) if k % 2 == 0 else (b, a, c))
            start = int(end)
    return points, np.asarray(tris, dtype=np.int32).reshape(-1, 3)


def read_display_geometry(source) -> dict[str, list[tuple[str, np.ndarray, np.ndarray]]]:
    """Read each body's display geometry from an ``.osim`` model.

    Handles the legacy ``<VisibleObject>/<GeometrySet>/<DisplayGeometry>`` layout
    (OpenSim 3.x models such as gait2354) and, best-effort, the newer ``<Mesh>``
    components. A ``DisplayGeometry`` transform (``rX rY rZ tx ty tz``) is combined
    with the enclosing ``VisibleObject`` transform, and their scale factors are
    multiplied.

    Args:
        source: Path to an ``.osim`` file, or the model XML text.

    Returns:
        Mapping ``body_name -> [(geometry_file, transform_4x4, scale_3), ...]``
        where ``transform_4x4`` places the geometry in the body frame.
    """
    text = source
    if isinstance(source, str) and len(source) < 4096 and os.path.exists(source):
        text = open(source).read()
    root = ET.fromstring(text)

    def _xform(vals):
        m = np.eye(4)
        if len(vals) == 6:
            m[:3, :3] = euler_xyz_to_matrix(*vals[:3])
            m[:3, 3] = vals[3:]
        return m

    out: dict[str, list[tuple[str, np.ndarray, np.ndarray]]] = {}
    for body in root.iter("Body"):
        name = body.get("name")
        if not name:
            continue
        entries: list[tuple[str, np.ndarray, np.ndarray]] = []
        for vo in body.iter("VisibleObject"):
            vt = vo.find("transform")
            vs = vo.find("scale_factors")
            vo_x = _xform(_vtp_floats(vt.text)) if vt is not None else np.eye(4)
            vo_s = np.asarray(_vtp_floats(vs.text)) if vs is not None and (vs.text or "").split() else np.ones(3)
            for dg in vo.iter("DisplayGeometry"):
                gf = dg.find("geometry_file")
                if gf is None or not (gf.text or "").strip():
                    continue
                dt = dg.find("transform")
                ds = dg.find("scale_factors")
                dg_x = _xform(_vtp_floats(dt.text)) if dt is not None else np.eye(4)
                dg_s = np.asarray(_vtp_floats(ds.text)) if ds is not None and (ds.text or "").split() else np.ones(3)
                entries.append((gf.text.strip(), vo_x @ dg_x, vo_s * dg_s))
        for mesh in body.iter("Mesh"):  # newer format, best effort
            mf = mesh.find("mesh_file")
            if mf is None or not (mf.text or "").strip():
                continue
            ms = mesh.find("scale_factors")
            m_s = np.asarray(_vtp_floats(ms.text)) if ms is not None and (ms.text or "").split() else np.ones(3)
            entries.append((mf.text.strip(), np.eye(4), m_s))
        if entries:
            out[name] = entries
    return out


def _resolve_geometry_file(filename: str, body_name: str, geometry_dir: str) -> str | None:
    """Resolve a referenced ``.vtp`` against ``geometry_dir`` with side-aware fallbacks."""
    candidates = [filename]
    base, ext = os.path.splitext(filename)
    for side in ("r", "l"):
        if body_name.endswith("_" + side):
            candidates += [f"{base}_{side}{ext}", f"{side}_{base}{ext}"]
    for c in candidates:
        p = os.path.join(geometry_dir, c)
        if os.path.isfile(p):
            return p
    return None


def fetch_opensim_geometry(
    cache_dir: str | None = None, *, ref: str = _OPENSIM_MODELS_REF, force_refresh: bool = False
) -> str:
    """Download the standard OpenSim bone geometry (``.vtp`` files) and return its path.

    Fetches the ``Geometry`` folder of the ``opensim-org/opensim-models`` repository
    (pinned by commit for reproducibility) into Newton's asset cache, for use as the
    ``geometry_dir`` of :meth:`MotionVisualizer.load_meshes`.

    Args:
        cache_dir: Cache directory (defaults to Newton's asset cache).
        ref: Git ref (branch/tag/SHA) to download; defaults to a pinned commit.
        force_refresh: Re-verify the cached copy against the remote.

    Returns:
        Path to the local ``Geometry`` directory.
    """
    return str(
        download_git_folder(_OPENSIM_MODELS_URL, "Geometry", cache_dir=cache_dir, ref=ref, force_refresh=force_refresh)
    )
