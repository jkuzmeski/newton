# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Motion-capture and motion I/O for the OpenSim port.

Readers/writers for the file formats used by OpenSim inverse-kinematics
workflows:

- ``.trc`` marker trajectories (:class:`OpenSimMarkerData`, :func:`read_trc`,
  :func:`write_trc`).
- ``.mot``/``.sto`` storage of coordinate/time-series data (:class:`OpenSimStorage`,
  :func:`read_storage`, :func:`write_storage`).

Marker positions are converted to meters from the units declared in the
``.trc`` header. :class:`OpenSimStorage` values are returned verbatim because
rotational columns can be degrees when ``inDegrees=yes``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np


def _read_text(source: str | os.PathLike) -> str:
    if isinstance(source, os.PathLike) or (isinstance(source, str) and os.path.exists(source)):
        with open(source, encoding="utf-8") as stream:
            return stream.read()
    return str(source)


def _position_scale(units: str) -> float:
    scales = {"m": 1.0, "cm": 0.01, "mm": 0.001}
    try:
        return scales[units.strip().lower()]
    except KeyError as error:
        raise ValueError(f"unsupported TRC position units: {units!r}") from error


def _validate_marker_data(markers: OpenSimMarkerData) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(markers.times)
    data = np.asarray(markers.data)
    expected = (len(times), len(markers.marker_names), 3)
    if times.ndim != 1 or data.shape != expected:
        raise ValueError(
            f"marker arrays must have shapes [frame_count] and {expected}, got {times.shape} and {data.shape}"
        )
    if not np.all(np.isfinite(times)):
        raise ValueError("marker times must be finite")
    if len(times) > 1 and np.any(np.diff(times) <= 0.0):
        raise ValueError("marker times must increase strictly")
    if len(set(markers.marker_names)) != len(markers.marker_names) or any(not name for name in markers.marker_names):
        raise ValueError("marker names must be nonempty and unique")
    return times, data


@dataclass
class OpenSimMarkerData:
    """Marker trajectories from a ``.trc`` file."""

    times: np.ndarray
    """Frame times [s], shape ``[num_frames]``."""

    marker_names: list[str]
    """Marker labels in column order."""

    data: np.ndarray
    """Marker positions in ground [m], shape ``[num_frames, num_markers, 3]``."""

    rate: float = 0.0
    """Data rate [Hz]."""

    units: str = "m"
    """Original position units string from the file header."""

    def index(self, name: str) -> int:
        """Return the column index of marker ``name``."""
        return self.marker_names.index(name)

    def frame(self, i: int) -> dict[str, np.ndarray]:
        """Return a name->position dict for frame ``i`` (skipping NaN markers)."""
        out: dict[str, np.ndarray] = {}
        for k, nm in enumerate(self.marker_names):
            p = self.data[i, k]
            if not np.any(np.isnan(p)):
                out[nm] = p
        return out


def read_trc(source: str | os.PathLike) -> OpenSimMarkerData:
    """Read a ``.trc`` marker file into a :class:`OpenSimMarkerData` (positions in meters)."""
    text = _read_text(source)
    lines = text.splitlines()
    if len(lines) < 5:
        raise ValueError("TRC data must contain file, metadata, marker, and component headers")
    meta = lines[2].split("\t")
    if len(meta) < 5:
        raise ValueError("TRC metadata must declare frame count, marker count, and units")
    rate = float(meta[0]) if meta[0].strip() else 0.0
    if not np.isfinite(rate) or rate < 0.0:
        raise ValueError("TRC data rate must be finite and nonnegative")
    declared_frames = int(meta[2])
    declared_markers = int(meta[3])
    units = meta[4].strip()
    header = lines[3].split("\t")
    names = [field.strip() for field in header[2:] if field.strip()]
    n_markers = len(names)
    if n_markers != declared_markers:
        raise ValueError(f"TRC header declares {declared_markers} markers but names {n_markers}")
    times: list[float] = []
    rows: list[list[str]] = []
    expected_width = 2 + 3 * n_markers
    for line_index, line in enumerate(lines[5:], start=6):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != expected_width:
            raise ValueError(f"TRC row {line_index} has {len(parts)} fields; expected {expected_width}")
        try:
            frame_number = float(parts[0])
            time = float(parts[1])
        except ValueError as error:
            raise ValueError(f"TRC row {line_index} has an invalid frame number or time") from error
        if not np.isfinite(frame_number) or not frame_number.is_integer():
            raise ValueError(f"TRC row {line_index} has an invalid frame number")
        times.append(time)
        rows.append(parts[2:])
    if len(rows) != declared_frames:
        raise ValueError(f"TRC metadata declares {declared_frames} frames but contains {len(rows)}")
    data = np.full((len(rows), n_markers, 3), np.nan)
    for frame_index, values in enumerate(rows):
        for marker_index in range(n_markers):
            for component in range(3):
                value = values[3 * marker_index + component].strip()
                if not value or value.lower() == "nan":
                    continue
                try:
                    parsed = float(value)
                except ValueError as error:
                    raise ValueError(f"TRC marker value is invalid at frame {frame_index}") from error
                if not np.isfinite(parsed):
                    raise ValueError(f"TRC marker value is nonfinite at frame {frame_index}")
                data[frame_index, marker_index, component] = parsed
    scale = _position_scale(units)
    times_array = np.asarray(times)
    if not np.all(np.isfinite(times_array)) or (len(times_array) > 1 and np.any(np.diff(times_array) <= 0.0)):
        raise ValueError("TRC frame times must be finite and increase strictly")
    if len(set(names)) != len(names) or any(not name for name in names):
        raise ValueError("TRC marker names must be nonempty and unique")
    return OpenSimMarkerData(times=times_array, marker_names=names, data=data * scale, rate=rate, units=units)


def write_trc(path: str | os.PathLike, markers: OpenSimMarkerData, units: str = "mm") -> None:
    """Write a :class:`OpenSimMarkerData` to a ``.trc`` file.

    Args:
        path: Output file path.
        markers: Marker trajectories (positions in meters).
        units: Position units to write (``"mm"``, ``"cm"``, or ``"m"``).
    """
    times, data = _validate_marker_data(markers)
    scale = 1.0 / _position_scale(units)
    n_frames = len(times)
    n_markers = len(markers.marker_names)
    duration = times[-1] - times[0] if n_frames > 1 else 0.0
    if not markers.rate and n_frames > 1 and duration <= 0.0:
        raise ValueError("marker times must increase when the data rate is inferred")
    rate = markers.rate or ((n_frames - 1) / duration if n_frames > 1 else 1.0)
    if not np.isfinite(rate) or rate <= 0.0:
        raise ValueError("marker data rate must be finite and positive")
    fname = os.path.basename(str(path))
    lines = [
        f"PathFileType\t4\t(X/Y/Z)\t{fname}",
        "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames",
        f"{rate:g}\t{rate:g}\t{n_frames}\t{n_markers}\t{units}\t{rate:g}\t1\t{n_frames}",
    ]
    hdr = ["Frame#", "Time"]
    for nm in markers.marker_names:
        hdr += [nm, "", ""]
    lines.append("\t".join(hdr))
    sub = ["", ""]
    for i in range(n_markers):
        sub += [f"X{i + 1}", f"Y{i + 1}", f"Z{i + 1}"]
    lines.append("\t".join(sub))
    lines.append("")
    for fi in range(n_frames):
        row = [str(fi + 1), f"{times[fi]:g}"]
        for mi in range(n_markers):
            p = data[fi, mi] * scale
            row += [f"{p[0]:.5f}", f"{p[1]:.5f}", f"{p[2]:.5f}"]
        lines.append("\t".join(row))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


@dataclass
class OpenSimStorage:
    """Time-series storage from a ``.mot``/``.sto`` file."""

    times: np.ndarray
    """Independent time column [s], shape ``[num_rows]``."""

    labels: list[str]
    """Dependent column labels, excluding ``time``."""

    data: np.ndarray
    """Dependent values, shape ``[num_rows, num_labels]``."""

    in_degrees: bool = False
    """Whether rotational columns are in degrees."""

    name: str = "table"
    """Table name from the header."""

    metadata: dict[str, str] = field(default_factory=dict)
    """Remaining header key/value pairs."""

    def column(self, name: str) -> np.ndarray:
        """Return the values of dependent column ``name``."""
        return self.data[:, self.labels.index(name)]


def read_storage(source: str | os.PathLike) -> OpenSimStorage:
    """Read an OpenSim ``.mot``/``.sto`` storage file into a :class:`OpenSimStorage`."""
    text = _read_text(source)
    lines = text.splitlines()
    try:
        hi = next(i for i, line in enumerate(lines) if line.strip().lower() == "endheader")
    except StopIteration as error:
        raise ValueError("OpenSim storage data must contain an endheader line") from error
    metadata: dict[str, str] = {}
    name = "table"
    in_degrees = False
    for l in lines[:hi]:
        s = l.strip()
        if not s:
            continue
        if "=" in s:
            k, v = s.split("=", 1)
            metadata[k.strip()] = v.strip()
            if k.strip().lower() == "indegrees":
                in_degrees = v.strip().lower() == "yes"
        elif name == "table":
            name = s
    if hi + 1 >= len(lines):
        raise ValueError("OpenSim storage data must contain a column header")
    cols = [column.strip() for column in lines[hi + 1].replace("\t", " ").split() if column.strip()]
    if not cols or cols[0].lower() != "time" or len(set(cols)) != len(cols):
        raise ValueError("OpenSim storage columns must start with time and have unique labels")
    rows = []
    for row_index, line in enumerate(lines[hi + 2 :], start=hi + 3):
        if not line.strip():
            continue
        parts = line.replace("\t", " ").split()
        if len(parts) != len(cols):
            raise ValueError(f"OpenSim storage row {row_index} has {len(parts)} values; expected {len(cols)}")
        rows.append([float(part) for part in parts])
    arr = np.asarray(rows, dtype=float).reshape(-1, len(cols))
    if not np.all(np.isfinite(arr)):
        raise ValueError("OpenSim storage values must be finite")
    times = arr[:, 0]
    if len(times) > 1 and np.any(np.diff(times) <= 0.0):
        raise ValueError("OpenSim storage times must increase strictly")
    return OpenSimStorage(
        times=times,
        labels=cols[1:],
        data=arr[:, 1:],
        in_degrees=in_degrees,
        name=name,
        metadata=metadata,
    )


def write_storage(
    path: str | os.PathLike,
    times: np.ndarray,
    labels: list[str],
    data: np.ndarray,
    name: str = "results",
    in_degrees: bool = True,
) -> None:
    """Write coordinate/time-series data to an OpenSim ``.mot`` file.

    Args:
        path: Output file path.
        times: Time column [s], shape ``[num_rows]``.
        labels: Dependent column labels (excluding ``time``).
        data: Dependent values, shape ``[num_rows, num_labels]``.
        name: Table name written to the header.
        in_degrees: Value of the ``inDegrees`` header flag.
    """
    times = np.asarray(times)
    data = np.asarray(data)
    if times.ndim != 1 or data.shape != (len(times), len(labels)):
        raise ValueError(
            f"storage arrays must have shapes [row_count] and {(len(times), len(labels))}, "
            f"got {times.shape} and {data.shape}"
        )
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(data)):
        raise ValueError("storage values must be finite")
    if len(times) > 1 and np.any(np.diff(times) <= 0.0):
        raise ValueError("storage times must increase strictly")
    normalized_labels = [label.lower() for label in labels]
    if (
        len(set(normalized_labels)) != len(labels)
        or "time" in normalized_labels
        or any(not label or any(char.isspace() for char in label) for label in labels)
    ):
        raise ValueError("storage labels must be nonempty, unique, exclude time, and contain no whitespace")
    n_rows = len(times)
    n_cols = len(labels) + 1
    header = [
        name,
        "version=1",
        f"nRows={n_rows}",
        f"nColumns={n_cols}",
        f"inDegrees={'yes' if in_degrees else 'no'}",
        "endheader",
    ]
    out = ["\t".join(["time", *labels])]
    for i in range(n_rows):
        row = [f"{times[i]:.8f}"] + [f"{data[i, j]:.8f}" for j in range(len(labels))]
        out.append("\t".join(row))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(header) + "\n" + "\n".join(out) + "\n")
