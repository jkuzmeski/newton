<!--
SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
SPDX-License-Identifier: Apache-2.0
-->

# Friction parameter adapters and calibration limits

The Cartesian-controller friction replay, parameter-search, and qualification
experiments have been retired. Their fitted candidates are not installed as
defaults. The shared constitutive laws, solver adapters, standalone
[friction example](FRICTION.md), and supplied-signal observation and report
utilities remain available.

`FoundationConfig` defaults to area-scaled `elastic_coulomb`; see
[FRICTION_COLUMN.md](FRICTION_COLUMN.md). This is a contact assumption derived from
the fitted normal material, not independent outsole friction calibration.

## Shared adapter scope

[`FrictionParameterAdapter`](friction_parameter_adapter.py) attaches through
`MidsoleFoundation.friction_solver`. It leaves the normal solve, contact points,
compression state, and material laws unchanged, then applies per-world tangential
parameters. Friction can still affect a freely moving body's subsequent motion
and thus later normal loads.

The adapter is reusable without a leg controller. It is not an optimizer and
does not certify supplied parameters.

## Separate mode namespaces

Integer mode codes differ between
[`FrictionSolver`](friction_solver.py) and `FrictionParameterAdapter`:

| Constitutive mode | Solver code | Parameter-adapter code |
|---|---:|---:|
| Legacy anchored bristle | 0 | 0 |
| Implicit anchored bristle | 1 | Not supported |
| Regularized Coulomb | 2 | Not supported |
| Consistent deflection | 3 | 1 |
| Implicit consistent deflection | 4 | Not supported |
| Stribeck deflection | Not supported | 4 |
| Pressure-dependent deflection | Not supported | 5 |
| Loaded-slip-history deflection | Not supported | 6 |
| Configured Maxwell bristle | Not supported | 7 |
| Material-derived column Maxwell | Not supported | 8 |
| Area-scaled elastic Coulomb | Not supported | 9 |

Diagnostic parameter-adapter codes 2 and 3 are rejected.

## Per-world parameter rows

`FrictionParameterAdapter.set_parameters()` accepts row lengths 7, 9, 10, 11,
or 12. The first seven fields are:

```python
PARAMETER_NAMES = (
    "method",           # Exact integer adapter code
    "mu",               # Nonnegative Coulomb coefficient
    "kt_scale",         # Positive stiffness multiplier
    "kv_scale",         # Nonnegative viscosity/damping multiplier
    "viscous_ratio",    # Nonnegative direct damping cap fraction
    "release_dwell_s",  # Nonnegative unloaded history dwell [s]
    "yield_width",      # Must be exactly 0.0
)
```

Nine-field rows append `mu_dynamic` and `transition_speed` [m/s]. The dynamic
coefficient must lie between zero and `mu`, and the transition speed must be at
least 1e-6 m/s. Longer rows append positive `pressure_scale_pa` [Pa],
`slip_scale_m` [m], and `shear_relaxation_time_s` [s], in that order.
Shorter supported rows receive the defaults defined in `set_parameters()`.

Maxwell methods 7 and 8 require `viscous_ratio=0`: viscosity is internal to the
Maxwell branch, not an additional direct velocity damper. Column-derived methods
8 and 9 require positive finite column areas and rest lengths.

## Qualification limits

Constitutive tests check force bounds, history updates, energy balance, and
numerical consistency. They do not identify outsole friction from normal
compression tests or establish that a model predicts measured running forces.
No historical controller-fit loss or tracking gate is part of the shared runtime.
