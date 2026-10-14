# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run the bilateral runner experiments on the LLSC TX-Green grid.

:mod:`.experiments` lists every experiment as a flat, indexed run list.
:mod:`.run` executes one index inside a SLURM array task. :mod:`.client` runs
on the workstation: it snapshots the code (committed or not), pushes it to a
bare repository on the grid, syncs prepared datasets through the ``Z:`` share,
and submits throttled array jobs.
"""
