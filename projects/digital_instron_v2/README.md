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

Fit the shared Hyperfoam-Maxwell-Pasternak column model to the rearfoot and
fullfoot bench cycles:

```bash
uv run -m projects.digital_instron_v2.workflow --manifest DigitalInstron/manifest_v2.json
```

The fitted parameters are cached at
`DigitalInstron/processed/v2_cache/digital_instron_material.json` and consumed by
the dynamic example below.

## Dynamic midsole example (`example.py`)

`dynamics.py` turns the calibrated column bed into a live Warp force model: each
substep every column reads its carrier-body pose, computes its through-thickness
compression, evaluates the Hyperfoam equilibrium pressure with a real-time
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
  modulus. Nothing is fitted for it, the face coefficient is symmetric, and the
  lateral flux therefore cancels exactly over the bed. The artifact still reports
  `pasternak_n_per_m`, now as the bed mean of that rule.
* **The outer bond of the passive surround defaults to zero.** It used to hold
  the relaxation up at 200 N/m per column while contributing nothing to the
  reported force or the carrier wrench, which made it an undeclared rigid
  support worth a large part of the rearfoot peak. With it removed the summed
  unilateral ground reaction is the whole applied load.

Nothing replaced the fitted coefficient. The fit is one shared material with **no
fixture-specific freedom**: the rearfoot punch and the full-foot last are
described by the same four numbers, so any disagreement between the two fixtures
stays visible in the residual instead of being absorbed by a knob.

## The shipped material does not pass its own gates

That is the reported result, not a defect left unfixed. Fitted alone, each
fixture is reproduced almost exactly, but they ask for different foam:

| fit | small-strain `E_eq` | own fixture (held out) | other fixture |
|---|---|---|---|
| rearfoot punch alone | 0.134 MPa | peak 0.1%, RMSE 0.6%, hysteresis 0.1% | full-foot peak +44.0% |
| full-foot last alone | 0.341 MPa | peak 1.7%, RMSE 1.6%, hysteresis 0.2% | rearfoot peak -27.0% |

The full-foot fixture demands a **2.54x stiffer** foam. With one shared material
and nothing fixture-specific left to absorb that, the joint fit has to split the
difference, and no branch of it passes: the shipped material misses the rearfoot
peak by 15.1% and the full-foot peak by 19.4%.

The joint objective is also **bimodal**. A five-seed multi-start finds a soft
branch (`mu_eq` 51.9 kPa, `alpha` 0.216, training loss 0.00641, shipped) that
splits the error, and a stiff branch (`mu_eq` about 240 kPa, `alpha` about 11.4,
training loss 0.00672) that reproduces the full-foot peak to 0.3% and misses the
rearfoot by 21%. The lower training loss ships, because the objective chooses,
not the reader.

The leading candidate for the cause is the constitutive form rather than any
fixture: the two fixtures cover very different strain ranges, and a single
first-order Hyperfoam term cannot span them. The multi-start result is direct
evidence, because the two branches differ almost entirely in `alpha`, the shape
parameter. A second Hyperfoam term is the next thing to try.
