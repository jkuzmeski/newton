<!--
SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
SPDX-License-Identifier: Apache-2.0
-->

# Friction-only Digital Shoe mechanics

> **Current default:** Digital Shoe now uses Maxwell shear friction automatically.
> The leg-shoe baseline uses mu 0.8, equilibrium stiffness 1000 N/m per nominal
> 25 mm² column, internal viscosity 10 N s/m, and the material relaxation time.
> `friction_model="legacy"` explicitly selects the previous law. Historical
> studies below retain their original settings and do not define current defaults.

This investigation evaluates tangential friction models while leaving the
accepted normal compression, material, and contact mechanics unchanged. It
does not refit the two-term foam model, alter the passive surround, replace
normal support, or enable another collision response. The shared runtime now selects Maxwell shear friction by default.
Choose `FoundationConfig(friction_model="legacy")` for the previous anchored-bristle
behavior. Explicit solver adapters remain available for diagnostic comparisons.

## Five tangential solver modes

`FrictionSolver` provides five selectable tangential modes:

| Mode | Integer code | Force model | Time stepping | History state |
|---|---|---|---|---|
| `bristle` | 0 | Legacy anchored Coulomb bristle and cone-limited damping | Explicit current-velocity predictor | World XY anchor `a`, `stuck`, `dwell` |
| `implicit_bristle` | 1 | Same `contact.bristle_step` force law and anchor history | Coupled tangential velocity solve for all carrier contacts | World XY anchor `a`, `stuck`, `dwell` |
| `regularized` | 2 | VBD-inspired regularized Coulomb law without elastic anchor memory | Coupled tangential velocity solve for all carrier contacts | None (stateless tangential response) |
| `deflection` | 3 | Consistent velocity-integrated deflection bristle | Explicit current-velocity predictor | Deflection vector `z`, `stuck`, `dwell` |
| `implicit_deflection` | 4 | Consistent velocity-integrated deflection bristle | Coupled tangential velocity solve for all carrier contacts | Deflection vector `z`, `stuck`, `dwell` |

### Consistent deflection vs legacy anchor tracking

The legacy bristle law (`contact.bristle_step`) tracks an absolute anchor point
in the world horizontal plane:

```text
trial_anchor = anchor_old (or contact_point_xy if newly stuck)
elastic = -kt * (contact_point_xy + dt * velocity_xy - trial_anchor)
```

The XY position comes from the nominal outsole point, but rigid velocity is
evaluated at its projection onto the ground plane. Their time derivatives can
disagree under pitch. In the tested history this produces positive tangential
energy residuals. The normal law is not changed to address this inconsistency.

The **consistent deflection** formulation (`friction_deflection.bristle_deflection_step`)
directly integrates contact slip velocity:

```text
z_trial = z_old + dt * velocity_xy (with z_old = 0 for a new contact)
elastic = -kt * z_trial
```

Radial Coulomb projection caps the elastic magnitude at $C = \mu \cdot F_n$
with an exact hard return when `yield_width == 0.0`. The next deflection is
strictly work-conjugate with velocity:

```text
next_deflection = -elastic / kt
```

With fixed stiffness and exact radial return, tests verify:
1. Zero-slip idempotence for an admissible loaded state at fixed normal load.
   Changing the cap or releasing contact can change the stored deflection.
2. Non-positive discrete tangential energy residual, within floating-point precision.
3. Independence from world-position offsets and consistent use of contact slip velocity.
These tests concern the reduced flat-plane friction law, not material calibration.

### Unsupported nonzero yield width

`bristle_deflection_step` accepts an optional `yield_width` argument ($w \in [0, 0.25]$)
providing a smooth $C^1$ transition shoulder.

**Nonzero yield width is unsupported for physical simulation and sweeps.**
Repeated evaluation at zero slip velocity ($v = 0$) in the shoulder regime is
mathematically non-idempotent: $g(r) < r$ causes spurious numerical relaxation/creep
toward $C \cdot (1 - w)$ at a rate dictated by the simulation time step $\Delta t$.
Therefore:
- `yield_width` must be strictly `0.0` in all production simulations, sweeps, and fits.
- `FrictionParameterAdapter`, `FrictionDynamicGPUWorkspace`, and `friction_dynamic` CLI
  explicitly enforce `yield_width == 0.0` and reject nonzero values.
- The mathematical formulation is retained solely as an unsupported reference.

### Coupled solve mechanics

During a coupled friction substep (`implicit_bristle`, `regularized`, or `implicit_deflection`),
contact points, normal reactions, and world-space inverse spatial inertia are read-only.
In world XY, each contact has velocity $J_i v$, where $J_i$ includes its COM lever arm.
The coupled equation is:

```text
v = v_free + dt * mobility * sum(J_i.T * f_i(J_i v))
||f_i|| <= mu * normal_i
```

The consumer supplies `mobility` mapping world-space COM wrench to acceleration.
Newton iterations evaluate trial velocities using analytic, branch-dependent
Jacobians and a pivoted $6 \times 6$ linear solve with backtracking line search.
History is updated once after the fixed iteration budget. Inspect the returned residual;
convergence is not guaranteed for every input.

## Separate dynamic Stribeck experiment

`friction_stribeck.py` implements an experimental velocity-dependent Coulomb cap
for exploratory parameter fitting:

$$\mu(v) = \mu_{\text{dynamic}} + (\mu_{\text{static}} - \mu_{\text{dynamic}}) \exp\left(-\frac{\|v\|^2}{v_s^2}\right)$$

where $\mu_{\text{static}} \ge \mu_{\text{dynamic}} \ge 0$ and $v_s > 0$ is
the transition velocity. The law delegates state advance and radial return to
`bristle_deflection_step` with $\mu = \mu_{\text{eff}}$ and $w = 0.0$, adding the
analytic velocity chain-rule derivative $\frac{\partial F}{\partial C} \frac{\partial C}{\partial v}$.

**Experimental status and negative finding:**
- The dynamic Stribeck law was evaluated across 9,233 parameter candidates in free-leg
  simulations (seeds 43, 67, and 89).
- When evaluated against full simulation-rate reference forces (seed 89), the
  optimizer converged to $\mu_{\text{dynamic}} = \mu_{\text{static}} = 0.5360$,
  collapsing the velocity-dependent term entirely.
- Apparent improvements under native-rate scoring (seed 43) were confirmed to be
  sub-sample aliasing artifacts that produced severe inter-sample force oscillations
  (91.94 N $F_x$ difference under time step halving).
- This selected run alone does not justify extra Stribeck parameters over constant-coefficient
  consistent deflection. It does not establish that speed dependence is absent in real shoes.

## Integration with MidsoleFoundation

```python
from projects.digital_shoe import FrictionAdapter, MidsoleFoundation

# Attach the adapter to an existing MidsoleFoundation instance
friction = FrictionAdapter(
    foundation,
    mobility=mobility,             # wp.array[wp.spatial_matrix], shape [world_count]
    mode="implicit_deflection",   # or "deflection", "implicit_bristle", "regularized"
    iterations=8,
)

# Supply the actual constrained/articulated COM mobility, not a visual carrier mass.
# Zero mobility denotes prescribed motion. Do not apply both result velocity and wrench.
# Advance foundation normally: normal mechanics execute first, followed by friction dispatch
foundation.apply(state, dt)

# Access forces, predicted velocities, and residuals
result = friction.result

# Detach adapter to restore default legacy bristle path
friction.detach()
```

The adapter redirects legacy tangential anchor arrays to zero stiffness and
scratch buffers, runs the unmodified normal kernels, and executes the tangential
solve before wrench reduction. Stored vertical compression, Pasternak neighbor
shear, Maxwell relaxation states, and contact points are preserved untouched.

## Runnable CLI examples

### Synthetic multi-mode demonstration

Compare tangential modes under identical prescribed normal loading and lateral shear:

```bash
# Run the repaired coupled mode headlessly
uv run --no-sync -m projects.digital_shoe friction   --mode implicit_deflection --viewer null --num-frames 120 --test   --output outputs/friction_identification/synthetic.json

# Interactive visual comparison
uv run --no-sync -m projects.digital_shoe friction --mode deflection --viewer gl
```

### Prescribed leg replay comparison

Replay saved Cartesian leg kinematics to evaluate friction force curves:

```bash
uv run --no-sync -m projects.digital_shoe friction-leg   --baseline outputs/impedance_instron/baseline12   --output outputs/friction_identification/leg_run   --forward-sign 1 --device cuda:0
```

For the complete parameter sweep, dynamic optimization, and independent qualification
workflow, see [FRICTION_IDENTIFICATION.md](FRICTION_IDENTIFICATION.md).
