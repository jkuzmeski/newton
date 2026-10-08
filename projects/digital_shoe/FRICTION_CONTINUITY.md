<!--
SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
SPDX-License-Identifier: Apache-2.0
-->

# Constitutive contact-force continuity

The old fixed-leg/controller continuity comparison has been retired. The shared
friction laws and their constitutive continuity tests remain available; no
simulation-output filter is part of these laws. The runtime default is
area-scaled `elastic_coulomb`, with Maxwell shear available explicitly; see
[FRICTION_COLUMN.md](FRICTION_COLUMN.md).

## What the discontinuity tests show

The exact-return bristle force is C0 across stick/slip yield, velocity reversal,
unloading and re-entry for continuous normal inputs and admissible histories.
Its tangent has a kink at yield; it is not globally C1. A history reset under finite
load or a discontinuity imposed in normal force is outside that continuity claim.
The retained tests are in
[`test_digital_shoe_friction_continuity.py`](../../newton/tests/test_digital_shoe_friction_continuity.py).

## Internal viscoelastic shear stress instead of direct velocity damping

An ideal parallel dashpot applies a term proportional to relative velocity. It can
therefore produce a force jump if velocity jumps, and a steep force change during
rapid deceleration. Simply softening the elastic spring does not remove this term.

The explicit Maxwell-bristle model uses an equilibrium spring in parallel with a
Maxwell branch (another spring in series with a dashpot). This entire element is
in series with the Coulomb slider. The branch force `q` is a mechanical state:

```text
branch_k = viscosity / relaxation_time
q_dot = branch_k * elastic_velocity - q / relaxation_time
traction = equilibrium_k * elastic_deflection + q
|traction| <= mu * normal_force
```

Both elastic and internal branch states are advanced together by backward Euler
and a single radial return. The total stored energy is

```text
E = 0.5 * equilibrium_k * |z|^2 + |q|^2 / (2 * branch_k)
```

The return dissipates plastic work, the dashpot dissipates `|q|^2 / viscosity`, and
backward Euler adds nonnegative numerical dissipation. Tests check this balance,
not just force appearance. There is no external smoothing of the computed force.

At fixed displacement, nonzero branch stress relaxes on its physical timescale.
That is intended viscoelastic behavior, unlike the rejected yield-shoulder scheme
whose static relaxation depended on the number of timesteps.

The implementation is retained in [`friction_maxwell.py`](friction_maxwell.py),
with tests in
[`test_digital_shoe_friction_maxwell.py`](../../newton/tests/test_digital_shoe_friction_maxwell.py).
Its shear assumptions do not constitute independent tangential calibration.

## Optional constitutive hypotheses

Pressure-dependent and loaded-slip weakening laws remain optional research
hypotheses, not calibrated continuity fixes. Their shared parameter-adapter codes
and row fields are documented in
[FRICTION_IDENTIFICATION.md](FRICTION_IDENTIFICATION.md).
