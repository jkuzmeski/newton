# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compatibility alias for :mod:`projects.impedance_instron.legacy.cmaes`."""

import sys

from .legacy import cmaes as _implementation

if __name__ != "__main__":
    sys.modules[__name__] = _implementation
