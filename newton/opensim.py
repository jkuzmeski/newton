# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""OpenSim motion-file and coordinate-frame adapters.

This module reads and writes marker trajectories and numeric storage files. It
does not provide an OpenSim model importer or an OpenSim-shaped simulation
runtime. Convert source data at this boundary, then use Newton models, states,
contacts, and solvers for simulation.
"""

from ._src.opensim.frame import OpenSimFrameConverter
from ._src.opensim.mocap import (
    OpenSimMarkerData,
    OpenSimStorage,
    read_storage,
    read_trc,
    write_storage,
    write_trc,
)

__all__ = [
    "OpenSimFrameConverter",
    "OpenSimMarkerData",
    "OpenSimStorage",
    "read_storage",
    "read_trc",
    "write_storage",
    "write_trc",
]
