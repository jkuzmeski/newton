# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Motion-capture and motion I/O for the OpenSim port.

Readers/writers for the file formats used by OpenSim inverse-kinematics
workflows:

- ``.trc`` marker trajectories (:class:`MarkerData`, :func:`read_trc`,
  :func:`write_trc`).
- ``.mot``/``.sto`` storage of coordinate/time-series data (:class:`Storage`,
  :func:`read_storage`, :func:`write_storage`).

All quantities are returned in SI units: marker positions are converted to
meters (from millimeters when the ``.trc`` header says so). Storage values are
returned verbatim (typically degrees for rotational coordinates, matching
OpenSim's ``inDegrees=yes`` convention).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np


def _read_text(source: str | os.PathLike) -> str:
    if isinstance(source, (str, os.PathLike)) and os.path.exists(str(source)):
        with open(source, encoding="utf-8") as f:
            return f.read()
    return str(source)


@dataclass
class MarkerData:
    """Marker trajectories from a ``.trc`` file.

    Attributes:
        times: Frame times [s], shape ``[num_frames]``.
        marker_names: Marker labels in column order.
        data: Marker positions in ground [m], shape ``[num_frames, num_markers, 3]``.
            Missing observations are ``NaN``.
        rate: Data rate [Hz].
        units: Original position units string from the file header.
    """

    times: np.ndarray
    marker_names: list[str]
    data: np.ndarray
    rate: float = 0.0
    units: str = "m"

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


def read_trc(source: str | os.PathLike) -> MarkerData:
    """Read a ``.trc`` marker file into a :class:`MarkerData` (positions in meters)."""
    text = _read_text(source)
    lines = text.splitlines()
    meta = lines[2].split("\t")
    rate = float(meta[0]) if meta and meta[0].strip() else 0.0
    units = meta[4].strip() if len(meta) > 4 else "m"
    header = lines[3].split("\t")
    names = [h.strip() for h in header[2:] if h.strip()]
    times: list[float] = []
    rows: list[list[str]] = []
    for ln in lines[5:]:
        if not ln.strip():
            continue
        parts = ln.split("\t")
        if len(parts) < 3:
            continue
        times.append(float(parts[1]))
        rows.append(parts[2:])
    n_markers = len(names)
    data = np.full((len(rows), n_markers, 3), np.nan)
    for fi, vals in enumerate(rows):
        for mi in range(n_markers):
            for c in range(3):
                idx = 3 * mi + c
                if idx < len(vals):
                    s = vals[idx].strip()
                    if s and s.lower() not in ("nan",):
                        try:
                            data[fi, mi, c] = float(s)
                        except ValueError:
                            pass
    scale = 0.001 if units.lower() == "mm" else 1.0
    return MarkerData(times=np.asarray(times), marker_names=names, data=data * scale, rate=rate, units=units)


def write_trc(path: str | os.PathLike, markers: MarkerData, units: str = "mm") -> None:
    """Write a :class:`MarkerData` to a ``.trc`` file.

    Args:
        path: Output file path.
        markers: Marker trajectories (positions in meters).
        units: Position units to write (``"mm"`` or ``"m"``).
    """
    scale = 1000.0 if units.lower() == "mm" else 1.0
    n_frames = len(markers.times)
    n_markers = len(markers.marker_names)
    rate = markers.rate or ((n_frames - 1) / (markers.times[-1] - markers.times[0]) if n_frames > 1 else 1.0)
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
        row = [str(fi + 1), f"{markers.times[fi]:g}"]
        for mi in range(n_markers):
            p = markers.data[fi, mi] * scale
            row += [f"{p[0]:.5f}", f"{p[1]:.5f}", f"{p[2]:.5f}"]
        lines.append("\t".join(row))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


@dataclass
class Storage:
    """Time-series storage from a ``.mot``/``.sto`` file.

    Attributes:
        times: Independent (time) column [s], shape ``[num_rows]``.
        labels: Dependent column labels (excluding ``time``).
        data: Dependent values, shape ``[num_rows, num_labels]``.
        in_degrees: Whether rotational columns are in degrees.
        name: Table name from the header.
        metadata: Remaining header key/value pairs.
    """

    times: np.ndarray
    labels: list[str]
    data: np.ndarray
    in_degrees: bool = True
    name: str = "table"
    metadata: dict[str, str] = field(default_factory=dict)

    def column(self, name: str) -> np.ndarray:
        """Return the values of dependent column ``name``."""
        return self.data[:, self.labels.index(name)]


def read_storage(source: str | os.PathLike) -> Storage:
    """Read an OpenSim ``.mot``/``.sto`` storage file into a :class:`Storage`."""
    text = _read_text(source)
    lines = text.splitlines()
    hi = next(i for i, l in enumerate(lines) if l.strip().lower() == "endheader")
    metadata: dict[str, str] = {}
    name = "table"
    in_degrees = True
    for l in lines[:hi]:
        s = l.strip()
        if not s:
            continue
        if "=" in s:
            k, v = s.split("=", 1)
            metadata[k.strip()] = v.strip()
            if k.strip() == "inDegrees":
                in_degrees = v.strip().lower() == "yes"
        elif "\t" not in s and " " not in s.split(" ", 1)[0]:
            name = s
    cols = [c.strip() for c in lines[hi + 1].replace("\t", " ").split() if c.strip()]
    rows = []
    for ln in lines[hi + 2 :]:
        if not ln.strip():
            continue
        parts = [p for p in ln.replace("\t", " ").split() if p != ""]
        if len(parts) < len(cols):
            continue
        rows.append([float(p) for p in parts[: len(cols)]])
    arr = np.asarray(rows)
    times = arr[:, 0]
    return Storage(
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
