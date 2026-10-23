# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Replay Instron last trials through the runtime's two-surface column bed.

The midsole is the runtime column bed over the platen, the runtime ground plane
at ``z = 0``. Every column has a top and a bottom: the bottom rests ``h`` above
the platen in the registered pose and lands on it, and the top under the last
tread is pressed by the last once its clearance ``g`` closes. Tops outside the
tread are free. Shear between neighbouring columns carries floating bottoms and
unpressed tops (:class:`~projects.digital_shoe.column_bed.ColumnBed`), so a
fitted shoe and a simulated runner shoe share one material and one contact law.

The carrier is the Instron last, held at its registered x, y, pitch, roll and
yaw and driven down by the measured crosshead stroke. Each world replays one
trace with one material; worlds are tiled as ``candidate * trace_count + trace``.
One replayed cycle is a captured CUDA graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import warp as wp

from projects.digital_shoe.column_bed import TOP_FREE, TOP_PRESSED, ColumnBed
from projects.digital_shoe.runtime import FoundationConfig

from .core import CALIBRATED_MATERIAL, Material
from .geometry import ColumnGrid, neighbor_indices
from .registration import LastRegistration

# Contact settings of the runner shoe (projects/impedance_instron/cartesian/shoe.py).
# Friction is inert here because the last never moves sideways.
CONTACT = FoundationConfig(ground_height_m=0.0, normal_damping=0.0, mu=0.8)


@dataclass(frozen=True)
class LastBed:
    """Midsole columns in the platen frame at first contact."""

    uv_m: np.ndarray  # column footprint [column_count, 2]
    bottom_m: np.ndarray  # rest bottom clearance over the platen h [column_count]
    gap_m: np.ndarray  # rest clearance from the top to the last surface g, zero off the tread [column_count]
    rest_m: np.ndarray  # rest thickness [column_count]
    area_m2: np.ndarray  # midsole footprint area [column_count]
    spacing_m: float  # grid spacing
    tread: np.ndarray  # bool, the last tread is over the column [column_count]


def last_bed(grid: ColumnGrid, registration: LastRegistration) -> LastBed:
    """Place the registered midsole under the last."""
    tread = np.isfinite(registration.surface_m)
    return LastBed(
        uv_m=np.asarray(grid.uv_m, dtype=float),
        bottom_m=np.asarray(registration.bottom_above_platen_m, dtype=float),
        gap_m=np.where(tread, registration.gap_m, 0.0),
        rest_m=np.asarray(grid.slack_m, dtype=float),
        area_m2=np.asarray(grid.area_m2, dtype=float),
        spacing_m=float(grid.spacing_m),
        tread=tread,
    )


@wp.kernel
def _drive(
    step: int,
    trace_count: int,
    depth: wp.array2d[float],
    rate: wp.array2d[float],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    """Pose each world's last at its trace's stroke for one substep."""
    w = wp.tid()
    trace = w % trace_count
    body_q[w] = wp.transform(wp.vec3(0.0, 0.0, -depth[step, trace]), wp.quat_identity())
    body_qd[w] = wp.spatial_vector(wp.vec3(0.0, 0.0, -rate[step, trace]), wp.vec3(0.0))


@wp.kernel
def _record(
    frame: int,
    body_f: wp.array[wp.spatial_vector],
    platen: wp.array[wp.float32],
    force: wp.array2d[float],
    reaction: wp.array2d[float],
):
    """Store the carrier's vertical wrench, the load the last puts into the shoe, and the platen reaction."""
    w = wp.tid()
    force[frame, w] = wp.spatial_top(body_f[w])[2]
    reaction[frame, w] = platen[w]


BALANCE_TOLERANCE = 0.02
"""Allowed gap between the last's load and the platen reaction, as a fraction of the peak load."""


COLUMN_TOLERANCE = 5.0e-3
"""Allowed pull of one column end on the last or the platen, as a fraction of the peak load."""


def physical(force: np.ndarray, reaction: np.ndarray, strain: np.ndarray, lowest: np.ndarray) -> np.ndarray:
    """Return which replayed cycles are physical.

    A cycle is physical when its last load ``T`` and platen reaction ``R`` are
    finite, the platen only pushes, and ``T`` and ``R`` agree within
    :data:`BALANCE_TOLERANCE` of the peak load every frame, and when no column
    is compressed past its thickness or pulls on the last or the platen by more
    than :data:`COLUMN_TOLERANCE` of the peak load. A diverged column solve
    breaks one of these.

    Args:
        force: ``T`` per frame [N], frames on the trailing axis.
        reaction: ``R`` per frame [N], shaped like ``force``.
        strain: Largest column strain over the cycle, ``force`` without its frame axis.
        lowest: Lowest column ``T`` or ``R`` over the cycle [N], shaped like ``strain``.
    """
    peak = np.maximum(np.max(np.abs(force), axis=-1), 1.0)
    balanced = np.max(np.abs(force - reaction), axis=-1) <= BALANCE_TOLERANCE * peak
    finite = np.all(np.isfinite(force) & np.isfinite(reaction), axis=-1)
    pushing = np.min(reaction, axis=-1) >= -BALANCE_TOLERANCE * peak
    columns = (strain <= 1.0) & (lowest >= -COLUMN_TOLERANCE * peak)
    return finite & balanced & pushing & columns


@wp.kernel
def _column_extremes(
    column_count: int,
    rest_len: wp.array[wp.float32],
    compression: wp.array[wp.float32],
    platen: wp.array[wp.float32],
    carried: wp.array[wp.float32],
    strain: wp.array[float],
    lowest: wp.array[float],
):
    """Track each world's largest column strain and lowest column end force; non-finite values count as extreme."""
    i = wp.tid()
    w = i / column_count
    s = compression[i] / rest_len[i % column_count]
    if not wp.isfinite(s):
        s = 1.0e30
    low = wp.min(platen[i], carried[i])
    if not wp.isfinite(low):
        low = -1.0e30
    wp.atomic_max(strain, w, s)
    wp.atomic_min(lowest, w, low)


class LastReplay:
    """Replay a batch of strokes for a batch of materials in one launch sequence.

    Args:
        bed: Column bed from :func:`last_bed`.
        depth_m: Last descent from first contact [m], shape ``[frames, trace_count]``.
            Each trace is one periodic cycle.
        dt_s: Frame interval shared by all traces [s].
        candidates: Materials evaluated together.
        cycles: Cycles replayed; the last is reported, the others settle the
            Maxwell history.
        substeps: Contact substeps per frame; the stroke is linearly interpolated.
        device: CUDA device.
    """

    def __init__(
        self,
        bed: LastBed,
        depth_m: np.ndarray,
        dt_s: float,
        candidates: int,
        *,
        cycles: int = 2,
        substeps: int = 1,
        device=None,
    ):
        depth_m = np.asarray(depth_m, dtype=float)
        self.frames, self.trace_count = depth_m.shape
        self.candidates = int(candidates)
        self.cycles = int(cycles)
        self.substeps = int(substeps)
        self.world_count = self.candidates * self.trace_count
        self.dt_s = float(dt_s) / self.substeps
        self.device = wp.get_device(device)
        self.bed = ColumnBed(
            np.column_stack((bed.uv_m, bed.bottom_m)),
            bed.rest_m,
            bed.area_m2,
            neighbor_indices(bed.uv_m, bed.uv_m, bed.spacing_m),
            bed.spacing_m,
            np.where(bed.tread, TOP_PRESSED, TOP_FREE),
            CALIBRATED_MATERIAL,
            CONTACT,
            np.arange(self.world_count),
            wp.zeros(self.world_count, dtype=wp.vec3, device=self.device),
            top_gap_m=bed.gap_m,
            device=self.device,
        )
        self.bed.set_timesteps(self.dt_s)
        self.carriers = SimpleNamespace(
            body_q=wp.zeros(self.world_count, dtype=wp.transform, device=self.device),
            body_qd=wp.zeros(self.world_count, dtype=wp.spatial_vector, device=self.device),
            body_f=wp.zeros(self.world_count, dtype=wp.spatial_vector, device=self.device),
        )
        # Substep k of frame f sits at fraction (k + 1) / substeps between frames f - 1 and f.
        frame = np.arange(self.frames * self.substeps) / self.substeps + 1.0 / self.substeps - 1.0
        depth = np.stack([np.interp(frame, np.arange(self.frames), d) for d in depth_m.T], axis=1)
        rate = np.gradient(depth, self.dt_s, axis=0)
        self.depth = wp.array(depth, dtype=float, device=self.device)
        self.rate = wp.array(rate, dtype=float, device=self.device)
        self.force = wp.zeros((self.frames, self.world_count), dtype=float, device=self.device)
        self.reaction = wp.zeros((self.frames, self.world_count), dtype=float, device=self.device)
        self.strain = wp.zeros(self.world_count, dtype=float, device=self.device)
        self.lowest = wp.zeros(self.world_count, dtype=float, device=self.device)
        self.launches = 0
        self._cycle = None

    def run(self, materials: list[Material], *, snapshot_frames: np.ndarray | None = None):
        """Return the last's force [N], shape ``[candidates, trace_count, frames]``.

        A world whose reported cycle fails :func:`physical` returns NaN forces.

        With ``snapshot_frames`` (one frame per trace) the per-column state of
        the reported cycle at those frames is returned as well, each indexed by
        world: ``compression`` [m], ``reaction`` (platen reaction ``R``) [N],
        ``carried`` (last reaction ``T``) [N], ``support`` (the
        ``TOP_HELD | BOTTOM_LANDED`` bits) and ``carrier`` (the last's vertical
        force) [N].
        """
        if len(materials) != self.candidates:
            raise ValueError("run needs one material per candidate")
        self.launches += 1
        self.bed.set_world_materials([m for m in materials for _ in range(self.trace_count)])
        self.bed.reset()
        if self._cycle is None:
            with wp.ScopedCapture(device=self.device) as capture:
                self._replay_cycle()
            self._cycle = capture.graph
        snapshots = {}
        for cycle in range(self.cycles - (snapshot_frames is not None)):
            if cycle == self.cycles - 1:
                self.strain.zero_()
                self.lowest.zero_()
            wp.capture_launch(self._cycle)
        if snapshot_frames is not None:
            self.strain.zero_()
            self.lowest.zero_()
            self._replay_cycle(snapshot_frames, snapshots)
        force = self.force.numpy().T.reshape(self.candidates, self.trace_count, self.frames)
        reaction = self.reaction.numpy().T.reshape(force.shape)
        strain = self.strain.numpy().reshape(force.shape[:2])
        lowest = self.lowest.numpy().reshape(force.shape[:2])
        force[~physical(force, reaction, strain, lowest)] = np.nan
        if snapshot_frames is None:
            return force
        columns = {
            key: np.stack([snapshots[f][key][w] for w, f in enumerate(np.tile(snapshot_frames, self.candidates))])
            for key in ("compression", "reaction", "carried", "support", "carrier")
        }
        return force, columns

    def _replay_cycle(self, snapshot_frames=None, snapshots=None) -> None:
        """Replay one cycle and record the last's force, reading column snapshots on the way."""
        worlds = self.world_count
        for frame in range(self.frames):
            for sub in range(self.substeps):
                wp.launch(
                    _drive,
                    dim=worlds,
                    inputs=[frame * self.substeps + sub, self.trace_count, self.depth, self.rate],
                    outputs=[self.carriers.body_q, self.carriers.body_qd],
                    device=self.device,
                )
                self.bed.apply(self.carriers)
            wp.launch(
                _record,
                dim=worlds,
                inputs=[frame, self.carriers.body_f, self.bed.bed.normal_force],
                outputs=[self.force, self.reaction],
                device=self.device,
            )
            bed = self.bed.bed
            wp.launch(
                _column_extremes,
                dim=worlds * self.bed.column_count,
                inputs=[self.bed.column_count, bed.rest_len, bed.compression, bed.reaction, bed.carried],
                outputs=[self.strain, self.lowest],
                device=self.device,
            )
            if snapshot_frames is not None and frame in snapshot_frames:
                snapshots[frame] = {
                    key: self.bed.column_state(key) for key in ("compression", "reaction", "carried", "support")
                }
                snapshots[frame]["carrier"] = self.carriers.body_f.numpy()[:, 2].copy()
