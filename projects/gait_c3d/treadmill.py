# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
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

"""Treadmill belt logs and the treadmill-to-overground virtual origin.

A treadmill capture holds the subject near the laboratory origin while the belt
carries the ground backward. Overground position is recovered by subtracting a
virtual origin that travels with the belt, following Jung and Lee, *Sensors*
2021, 21(3), 786. That reference measures belt travel optically because a
consumer treadmill exposes no speed signal; a Motek D-Flow ``tm0001.txt`` log
reports belt speed and travel directly, so the belt marker chain, its
re-indexing and its sag projection are not needed here.

The map is a pure time-varying translation, so joint angles are unchanged and
only the free-root translation and its fore-aft velocity move. See
``TREADMILL_TO_OVERGROUND_PLAN.md`` for the measured synchronization and
protocol numbers this module relies on.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .c3d_adapter import lab_to_newton_rotation

COLUMNS = (
    "Time",
    "leftbelt_speed",
    "leftbelt_distance",
    "rightbelt_speed",
    "rightbelt_distance",
    "platform_pitch",
    "platform_roll",
)
"""Column order of a Motek D-Flow treadmill log."""

BELT_TRAVEL_AXIS_LAB = (0.0, 1.0, 0.0)
"""Laboratory direction the belt surface travels in, subject-backward."""

TIED_BELT_TOLERANCE = 1.0e-3
"""Largest left-right belt travel difference accepted as a tied belt [m]."""


@dataclass(frozen=True, slots=True)
class TreadmillLog:
    """Belt and platform channels of one parsed treadmill log.

    ``Time`` is the D-Flow controller uptime clock, not trial time, so
    :attr:`t` is the time base that aligns with the C3D frame timeline.
    """

    time: np.ndarray
    """Controller clock [s], shape [sample_count]."""

    left_speed: np.ndarray
    """Left belt speed [m/s], shape [sample_count]."""

    left_distance: np.ndarray
    """Left belt travel [m], shape [sample_count]."""

    right_speed: np.ndarray
    """Right belt speed [m/s], shape [sample_count]."""

    right_distance: np.ndarray
    """Right belt travel [m], shape [sample_count]."""

    pitch: np.ndarray
    """Platform pitch, shape [sample_count]."""

    roll: np.ndarray
    """Platform roll, shape [sample_count]."""

    source_file: str
    """Logical source log basename."""

    source_sha256: str
    """SHA-256 of the source log bytes."""

    def __post_init__(self) -> None:
        sample_count = len(self.time)
        for name in ("left_speed", "left_distance", "right_speed", "right_distance", "pitch", "roll"):
            channel = getattr(self, name)
            if channel.shape != (sample_count,):
                raise ValueError(f"treadmill channel {name!r} has an invalid shape")
            if not np.all(np.isfinite(channel)):
                raise ValueError(f"treadmill channel {name!r} must be finite")
        if sample_count < 2:
            raise ValueError("a treadmill log needs at least two samples")
        if np.any(np.diff(self.time) <= 0.0):
            raise ValueError("treadmill log times must increase strictly")

    @property
    def t(self) -> np.ndarray:
        """Log-relative time, zero at the first sample [s], shape [sample_count]."""
        return self.time - self.time[0]

    @property
    def duration(self) -> float:
        """Span between the first and last logged sample [s]."""
        return float(self.time[-1] - self.time[0])

    @property
    def rate(self) -> float:
        """Sample rate from the median sample interval [Hz]."""
        return 1.0 / float(np.median(np.diff(self.time)))

    @property
    def tied_belt_residual(self) -> float:
        """Largest absolute left-right belt travel difference [m]."""
        return float(np.max(np.abs(self.left_distance - self.right_distance)))

    def speed(self, side: str = "left") -> np.ndarray:
        """Return one belt speed channel.

        Args:
            side: ``"left"``, ``"right"`` or ``"mean"``.
        """
        return self._select(side, self.left_speed, self.right_speed)

    def distance(self, side: str = "left") -> np.ndarray:
        """Return one logged belt travel channel.

        Args:
            side: ``"left"``, ``"right"`` or ``"mean"``.
        """
        return self._select(side, self.left_distance, self.right_distance)

    @staticmethod
    def _select(side: str, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        if side == "left":
            return left
        if side == "right":
            return right
        if side == "mean":
            return 0.5 * (left + right)
        raise ValueError(f"unknown belt side {side!r}")

    def travel(self, times: np.ndarray, side: str = "left") -> np.ndarray:
        """Return belt travel since the first sample at arbitrary times [m].

        The D-Flow speed reference is piecewise linear, so integrating it as
        such is exact between samples and across dropped samples. The logged
        travel column instead accumulates a right-rectangle sum and lags this
        result by up to about 5 mm inside a ramp.

        Args:
            times: Query times on the log-relative base [s], shape [query_count].
            side: ``"left"``, ``"right"`` or ``"mean"``.
        """
        t = self.t
        v = self.speed(side)
        cumulative = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(t))])
        query = np.clip(np.asarray(times, dtype=np.float64), t[0], t[-1])
        i = np.clip(np.searchsorted(t, query) - 1, 0, len(t) - 2)
        fraction = (query - t[i]) / (t[i + 1] - t[i])
        speed_at = v[i] + (v[i + 1] - v[i]) * fraction
        return cumulative[i] + 0.5 * (v[i] + speed_at) * (query - t[i])


@dataclass(frozen=True, slots=True)
class BeltMotion:
    """Belt travel resampled onto a capture frame timeline.

    ``distance`` is zero at the first frame, so it is the virtual origin travel
    that maps that frame's laboratory pose onto itself.
    """

    times: np.ndarray
    """Capture frame times on the C3D time base [s], shape [frame_count]."""

    speed: np.ndarray
    """Belt speed at each frame [m/s], shape [frame_count]."""

    distance: np.ndarray
    """Belt travel since the first frame [m], shape [frame_count]."""

    covered: np.ndarray
    """True where a frame falls inside the logged interval, shape [frame_count]."""

    axis: np.ndarray
    """Newton-frame unit direction of the overground offset, shape [3]."""

    side: str
    """Belt channel the travel came from."""

    offset: float
    """Log-relative time of frame zero [s]."""

    source_file: str
    """Logical source log basename."""

    source_sha256: str
    """SHA-256 of the source log bytes."""

    tied_belt_residual: float
    """Largest absolute left-right belt travel difference in the log [m]."""

    def __post_init__(self) -> None:
        frame_count = len(self.times)
        for name in ("speed", "distance", "covered"):
            if getattr(self, name).shape != (frame_count,):
                raise ValueError(f"belt channel {name!r} has an invalid shape")
        if self.axis.shape != (3,) or not np.isclose(np.linalg.norm(self.axis), 1.0, atol=1.0e-12):
            raise ValueError("belt axis must be a unit vector")
        if not np.all(np.isfinite(self.distance)) or not np.all(np.isfinite(self.speed)):
            raise ValueError("belt motion arrays must be finite")

    @property
    def travel(self) -> float:
        """Total belt travel over the frame range [m]."""
        return float(self.distance[-1] - self.distance[0])

    def offsets(self) -> np.ndarray:
        """Return the per-frame overground offset [m], shape [frame_count, 3].

        Add this to a Newton-frame laboratory position to place it in the
        overground frame.
        """
        return self.distance[:, None] * self.axis[None, :]

    def velocities(self) -> np.ndarray:
        """Return the per-frame overground velocity offset [m/s], shape [frame_count, 3]."""
        return self.speed[:, None] * self.axis[None, :]

    def select(self, indices: np.ndarray) -> BeltMotion:
        """Return the belt motion of a frame subset, re-zeroed at its first frame.

        Args:
            indices: Frame indices into this motion, shape [selected_count].
        """
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1 or len(indices) == 0:
            raise ValueError("frame selection must be a nonempty one-dimensional index array")
        distance = self.distance[indices]
        return BeltMotion(
            times=self.times[indices],
            speed=self.speed[indices],
            distance=distance - distance[0],
            covered=self.covered[indices],
            axis=self.axis,
            side=self.side,
            offset=self.offset,
            source_file=self.source_file,
            source_sha256=self.source_sha256,
            tied_belt_residual=self.tied_belt_residual,
        )

    def manifest_block(self) -> dict[str, Any]:
        """Return the sealed-manifest description of this transform."""
        return {
            "source": {"file": self.source_file, "sha256": self.source_sha256},
            "side": self.side,
            "offset_s": self.offset,
            "axis": [float(value) for value in self.axis],
            "distance_m": self.travel,
            "speed_max_m_s": float(np.max(np.abs(self.speed))),
            "tied_belt_residual_m": self.tied_belt_residual,
            "covered_frames": int(np.count_nonzero(self.covered)),
            "applied_stage": "post_ik_root_translation",
        }


def load_treadmill_log(path: str | Path) -> TreadmillLog:
    """Read a Motek D-Flow ``tm0001.txt`` treadmill log.

    Non-finite rows and non-increasing timestamps are dropped, so the result is
    always safe to interpolate on.

    Args:
        path: Tab-separated log file with the columns in :data:`COLUMNS`.
    """
    source = Path(path).resolve()
    with source.open() as stream:
        header = tuple(stream.readline().strip().split("\t"))
    if header != COLUMNS:
        raise ValueError(f"{source}: unexpected treadmill log header {header}")
    raw = np.loadtxt(source, skiprows=1, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != len(COLUMNS):
        raise ValueError(f"{source}: treadmill log must have {len(COLUMNS)} columns")
    raw = raw[np.isfinite(raw).all(axis=1)]
    raw = raw[np.concatenate([[True], np.diff(raw[:, 0]) > 0.0])]
    return TreadmillLog(
        time=raw[:, 0],
        left_speed=raw[:, 1],
        left_distance=raw[:, 2],
        right_speed=raw[:, 3],
        right_distance=raw[:, 4],
        pitch=raw[:, 5],
        roll=raw[:, 6],
        source_file=source.name,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    )


def belt_motion(
    log: TreadmillLog,
    times: np.ndarray,
    *,
    side: str = "auto",
    offset: float = 0.0,
    up_axis: str = "+Z",
    forward_axis: str = "-Y",
    travel_axis_lab: tuple[float, float, float] = BELT_TRAVEL_AXIS_LAB,
    tied_tolerance: float = TIED_BELT_TOLERANCE,
) -> BeltMotion:
    """Resample a treadmill log onto capture frame times.

    Args:
        log: Parsed treadmill log.
        times: Capture frame times on the C3D time base, where frame ``i`` is at
            ``i / rate`` seconds [s], shape [frame_count].
        side: Belt channel to use, or ``"auto"`` to require a tied belt and use
            the left channel.
        offset: Log-relative time of frame zero [s]. Positive means the log
            started before the capture.
        up_axis: Laboratory axis that points upward.
        forward_axis: Laboratory axis that points subject-forward.
        travel_axis_lab: Laboratory direction the belt surface travels in.
        tied_tolerance: Largest left-right travel difference accepted by
            ``side="auto"`` [m].

    Raises:
        ValueError: If ``side="auto"`` and the two belts differ by more than
            ``tied_tolerance``. A split-belt trial needs one virtual origin per
            foot, which this transform does not model.
    """
    residual = log.tied_belt_residual
    if side == "auto":
        if residual > tied_tolerance:
            raise ValueError(
                f"split-belt log: left and right travel differ by {residual:.4f} m; "
                "pass an explicit belt side or use a per-foot transform"
            )
        side = "left"
    times = np.asarray(times, dtype=np.float64)
    if times.ndim != 1 or len(times) == 0:
        raise ValueError("frame times must be a nonempty one-dimensional array")
    query = times + offset
    t = log.t
    travel = log.travel(query, side)
    rotation = lab_to_newton_rotation(up_axis, forward_axis)
    axis = -(rotation @ np.asarray(travel_axis_lab, dtype=np.float64))
    axis = axis / np.linalg.norm(axis)
    return BeltMotion(
        times=times,
        speed=np.interp(np.clip(query, t[0], t[-1]), t, log.speed(side)),
        distance=travel - travel[0],
        covered=(query >= t[0]) & (query <= t[-1]),
        axis=axis,
        side=side,
        offset=float(offset),
        source_file=log.source_file,
        source_sha256=log.source_sha256,
        tied_belt_residual=residual,
    )


def belt_motion_for_frames(
    log: TreadmillLog,
    frame_count: int,
    *,
    rate: float = 100.0,
    first_frame: int = 0,
    stride: int = 1,
    **kwargs: Any,
) -> BeltMotion:
    """Resample a treadmill log onto a C3D point timeline.

    Frame times come from the capture, never from the log, because the log runs
    at its own rate and spans slightly more than the capture.

    Args:
        log: Parsed treadmill log.
        frame_count: Number of fitted frames.
        rate: C3D point rate [Hz].
        first_frame: Index of the first fitted frame in the source C3D.
        stride: Frame stride of the fitted range.
        **kwargs: Forwarded to :func:`belt_motion`.
    """
    if frame_count < 1 or stride < 1 or rate <= 0.0:
        raise ValueError("frame count, stride and rate must be positive")
    times = (first_frame + stride * np.arange(frame_count, dtype=np.float64)) / rate
    return belt_motion(log, times, **kwargs)


def find_treadmill_log(subject_dir: str | Path, name: str = "tm0001.txt") -> Path | None:
    """Return the treadmill log inside a subject bundle, if it is present.

    Args:
        subject_dir: Compiled subject bundle root.
        name: Log basename written by the acquisition software.
    """
    candidate = Path(subject_dir) / name
    return candidate if candidate.is_file() else None
