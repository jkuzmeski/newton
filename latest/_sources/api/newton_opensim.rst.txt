.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

newton.opensim
==============

OpenSim motion-file and coordinate-frame adapters.

This module reads and writes marker trajectories and numeric storage files. It
does not provide an OpenSim model importer or an OpenSim-shaped simulation
runtime. Convert source data at this boundary, then use Newton models, states,
contacts, and solvers for simulation.

.. py:module:: newton.opensim
.. currentmodule:: newton.opensim

.. rubric:: Classes

.. autosummary::
   :toctree: _generated
   :nosignatures:

   OpenSimFrameConverter
   OpenSimMarkerData
   OpenSimStorage

.. rubric:: Functions

.. autosummary::
   :toctree: _generated
   :signatures: long

   read_storage
   read_trc
   write_storage
   write_trc
