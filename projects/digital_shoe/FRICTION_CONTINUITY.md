<!--
SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
SPDX-License-Identifier: Apache-2.0
-->

# Constitutive contact-force continuity

The old fixed-leg/controller continuity comparison has been retired. The shared
friction laws and their constitutive continuity tests remain available; no
simulation-output filter is part of these laws. The column bed does not use
them; it uses plain Coulomb friction, described in
[FRICTION_COLUMN.md](FRICTION_COLUMN.md).

## What the discontinuity tests show

The exact-return bristle force is C0 across stick/slip yield, velocity reversal,
unloading and re-entry for continuous normal inputs and admissible histories.
Its tangent has a kink at yield; it is not globally C1. A history reset under finite
load or a discontinuity imposed in normal force is outside that continuity claim.
The retained tests are in
[`test_digital_shoe_friction_continuity.py`](../../newton/tests/test_digital_shoe_friction_continuity.py).
