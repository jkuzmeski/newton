<!--
SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
SPDX-License-Identifier: Apache-2.0
-->

# Digital Instron v2

> **Standalone presentation:** Use `projects.digital_instron_v2.export_digital_shoe`
> to produce a portable `digital_shoe.json`, then run the artifact-only Virtual
> Instron, free-body drop, and rocker examples in `projects/digital_shoe`. See
> `projects/digital_shoe/README.md`. These examples do not use gait or a human model.

Identify one shoe-level effective viscoelastic midsole model from intact Digital
Instron bench tests, then exercise that calibrated model in live Newton
rigid-body physics.

## Calibration (`workflow.py`)

Fit the shared two-term Hyperfoam-Maxwell-Pasternak column model to the rearfoot
and fullfoot bench cycles:

```bash
uv run -m projects.digital_instron_v2.workflow --manifest DigitalInstron/manifest_v2.json
```

The fitted parameters are cached at
`DigitalInstron/processed/v2_cache/digital_instron_material.json` and consumed by
the dynamic example below.

## Dynamic midsole example (`example.py`)

`dynamics.py` turns the calibrated column bed into a live Warp force model: each
substep every column reads its carrier-body pose, computes its through-thickness
compression, evaluates the two-term Hyperfoam equilibrium pressure with a real-time
generalized-Maxwell overstress branch and Pasternak lateral coupling, adds an
anchored bristle (elastoplastic) Coulomb friction that holds a planted contact
patch and saturates at `mu * fn`, and accumulates the full six-component
ground-reaction wrench (normal, tangential shear, and the resultant moment that
carries the center of pressure) into `newton.State.body_f`. Four scenarios share
the same foundation:

```bash
# Displacement-controlled digital Instron: squish the midsole between a
# shoe-last crosshead and the ground plane; record the hysteresis loop.
uv run -m projects.digital_instron_v2.example --mode instron

# Free, massive midsole resting in stable equilibrium on the foundation;
# a sub-cone lateral load is held by the anchored stick-slip foam friction.
uv run -m projects.digital_instron_v2.example --mode settle

# Synthetic running stride that rolls a foot heel-to-toe over the foundation,
# producing a ground-reaction force profile and a migrating center of pressure.
uv run -m projects.digital_instron_v2.example --mode stride

# Fully dynamic, foot-mounted shoe with mass and inertia. A damped bilateral
# "upper" keeps the midsole coupled to the foot for the whole stride, so the
# shoe presses the foam into the ground in stance and the entire bed lifts clear
# with the foot in flight; the stance/flight ground reaction is recorded.
uv run -m projects.digital_instron_v2.example --mode attached
```

Add `--viewer null --num-frames N --test` to run headlessly and audit the
recorded response, or `--viewer gl` for the interactive viewer (the midsole
renders as a live bed of compression-coloured foam columns/springs that sink and
redden under load).

The `attached` mode is launch-overhead-bound (dozens of tiny per-substep kernel
launches for only ~600 columns, not compute), so it is optimised two ways. The foot
trajectory is precomputed once into device arrays and the per-substep force resets
are fused into a single kernel launch (each 1-element memset was otherwise a graph
node costing far more than the actual physics), leaving the whole 128-substep frame
fully on the GPU; that frame is then captured into a single CUDA graph and replayed
once per frame. Together these run the mode about 17x faster than eager launches
(~5 ms/frame on an A6000, several times faster than real time). Pass `--eager` to
disable graph capture for debugging.

## Scope on main

The standalone shoe and calibration tools do not depend on an OpenSim runtime.
The legacy jumping-leg crossover and human-shoe experiments remain on
`jkuzmeski/digital-shoe-standalone`; they are not part of this integration.

## Tests

```bash
uv run --extra dev -m unittest newton.tests.test_digital_instron_core
uv run --extra dev -m unittest newton.tests.test_digital_instron_project
uv run --extra dev -m unittest newton.tests.test_digital_instron_dynamics
```

`test_digital_instron_dynamics` verifies that the live per-substep force
integration reproduces the calibrated `core.predict` model to float precision,
that the Pasternak neighbour table matches the calibration Laplacian operator,
that the lateral flux cancels over the whole bed, that the zero-Poisson Hyperfoam
law agrees between CPU and GPU, and that each example scenario passes its
physical audit.

## What the fitted vector contains, and what it does not

Three quantities that used to be fitted or hidden are now pinned or declared:

* **Effective Poisson ratio 0.** Confined and unconfined compression of
  racing-shoe midsole foam agree within scatter (McCulloch, Delp and Kuhl,
  arXiv:2602.12694), so `EFFECTIVE_POISSON_RATIO = 0.0` and the Ogden-Hill
  exponent `beta` vanishes. This also makes the independent-column assumption
  self-consistent with the constitutive law.
* **Pasternak coefficient pinned to the material.** A shear layer coefficient is
  `G * t`, so each column gets `k_i = mu_eq * t_i` from
  `Material.coupling_n_per_m`, with `mu_eq` the equilibrium Ogden-Hill shear
  modulus, which for the two-term series is the **sum** of the term moduli. Nothing is fitted for it, the face coefficient is symmetric, and the
  lateral flux therefore cancels exactly over the bed. The artifact still reports
  `pasternak_n_per_m`, now as the bed mean of that rule.
* **The outer bond of the passive surround defaults to zero.** It used to hold
  the relaxation up at 200 N/m per column while contributing nothing to the
  reported force or the carrier wrench, which made it an undeclared rigid
  support worth a large part of the rearfoot peak. With it removed the summed
  unilateral ground reaction is the whole applied load.

Nothing replaced the fitted coefficient. The fit is one shared material with **no
fixture-specific freedom**: the rearfoot punch and the full-foot last are
described by the same six numbers -- two Ogden-Hill terms plus the Maxwell
branch -- so any disagreement between the two fixtures stays visible in the
residual instead of being absorbed by a knob.

## The shipped material passes its own gates, with two Hyperfoam terms

The equilibrium network is a two-term Ogden-Hill series. One first-order term has
a single shape exponent, and the two bench fixtures do not share a strain range:
the rearfoot punch drives its thinnest column to 89.6% strain and the full-foot
last reaches about 74%, while the published foam secant is measured over 0-10%.
One term could not span that, its objective was bimodal in the exponent, and it
failed four of six held-out gates.

Held-out gates, same protocol, same cycles, nothing weakened:

| fixture | peak force | force RMSE | loop area | one term | two terms |
|---|---|---|---|---|---|
| rearfoot punch | threshold 10% | | | 15.1% / 8.9% / 14.8% | **2.8% / 5.6% / 7.5%** |
| full-foot last | threshold 10% | | | 19.4% / 12.1% / 9.8% | **5.9% / 6.4% / 8.8%** |

2 of 6 gates passed with one term. **6 of 6 pass with two.** Training loss falls
from 0.006414 to 0.001811, a factor of 3.5.

### Fitted vector

| | term 1 | term 2 |
|---|---|---|
| instantaneous shear modulus | 267.6 kPa | 19.5 kPa |
| equilibrium shear modulus | 176.3 kPa | 12.8 kPa |
| exponent `alpha` | 18.077 | -0.586 |

with `equilibrium_fraction` 0.6587 and `tau` 5.15 ms. Either sign of an Ogden-Hill
exponent is admissible: a large positive exponent produces a soft `1/lambda`
plateau and a negative one produces densification, and the fit uses one of each.
`mu_eq` is the **sum**, 189.1 kPa, giving a small-strain compressive modulus of
378 kPa and a per-column Pasternak coefficient of 952-8277 N/m.

### Each term earns its place

Term 2's share of the equilibrium pressure rises monotonically with strain:
10.2% at 5% strain, 29.3% at 25%, 53.0% at 50%, 73.0% at the full-foot peak and
86.1% at the rearfoot peak (`outputs/impedance_instron/refit_two_term/term_contribution.png`).
The two terms really do divide the strain range, which is the hypothesis the
second term was added to test.

Profiling the objective in each second-term parameter, refitting everything else
at each point, gives interior minima rather than flat valleys:

| `mu_2` [kPa, instantaneous] | 0 | 4.9 | 9.7 | **19.5** | 29.2 | 39.0 | 78.0 |
|---|---|---|---|---|---|---|---|
| loss | 0.00711 | 0.00220 | 0.00197 | **0.00180** | 0.00186 | 0.00212 | 0.00407 |

| `alpha_2` | -4.0 | -2.0 | -1.0 | **-0.586** | -0.25 | +0.25 | +2.0 |
|---|---|---|---|---|---|---|---|
| loss | 0.00468 | 0.00271 | 0.00193 | **0.00180** | 0.00183 | 0.00215 | 0.00392 |

### What is still soft

* **`alpha_1` is weakly identified.** It sits at 18.08 against a 20.0 bound.
  Refitting with the exponent bounds widened to 40 moves it to 21.17 -- interior,
  not against the new bound -- for a 1.2% lower loss, a 5.9% higher `mu_eq`
  (200.3 kPa) and gates that still all pass (2.6/3.3/9.3% and 5.4/3.9/8.4%). The
  conclusion does not depend on the bound, but the exponent's exact value does.
* **`equilibrium_fraction` and `tau` are not separable.** Their correlation at the
  optimum is 1.000 (column-scaled Jacobian condition number 225). Both bench
  cycles run at one rate, so how much relaxes and how fast cannot be told apart.
  This replaces the old `corr(G, pasternak) = 1.000` degeneracy, which is gone
  because the Pasternak coefficient is now pinned rather than fitted.
* **The fixtures still disagree.** Fitted alone, each is reproduced almost
  exactly, and they still ask for different foam:

  | fit | `mu_eq` | own fixture (held out) |
  |---|---|---|
  | rearfoot punch alone | 81.1 kPa | peak 0.04%, RMSE 0.5%, loop 0.07% |
  | full-foot last alone | 308.1 kPa | peak 0.4%, RMSE 0.6%, loop 0.2% |

  That ratio is **3.80x**, up from 2.54x with one term, and both single-fixture
  fits push `alpha_1` onto its 20.0 bound. The per-fixture adjoints still point in
  nearly opposite directions, with their sum 6.6% of either. The two-term form did
  not remove the tension between the fixtures; it moved the shared compromise to a
  place where both fixtures are inside their gates.

### Multi-start is now opt-in

`fit_material` runs a single start by default. The bimodality that justified a
multi-start belonged to the one-term law. A seven-start check on the two-term law
found five of seven seeds -- from both old basins and from the published two-term
compression fits -- converging on the same optimum within about 4% in `mu_eq`,
one converging on the same optimum with the two terms relabelled, and only the
seed that starts with the second term disabled staying behind at 3.9x the loss.
That makes unimodality an **assumption**, not a proof. Re-test it with
`--multistart-seeds 6` whenever the constitutive form, objective, bounds or
fixture set change; at about 0.09 s per residual evaluation the check is cheap.

### Against the published foam tables

McCulloch, Delp and Kuhl (arXiv:2602.12694) report 268-299 kPa compressive
stiffness for FF LEAP and FF TURBO PLUS, and our own one-term fits to their
compression tables give a like-for-like 205-216 kPa. Our `E = 378 kPa` sits
**above** both, where the single-term fit at 104 kPa sat below. Two-term fits to
those same tables give 362-380 kPa, so the like-for-like two-term comparison is
close. Treat all of this as context, not a target: those are cut-cube specimens of
different foams from a different manufacturer, and foam bonded in a shoe with
curvature, skin, glue and a plate can legitimately differ from a cube. See
`outputs/impedance_instron/refit_two_term/stress_stretch_two_term.png`, whose
right panel shows that the published tables stop at 40% compression while the
fixtures reach 74-90%.
