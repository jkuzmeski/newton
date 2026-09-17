# Controller search with reverse-mode differentiation

## Decision and current status

Optimize the **existing 48 spline coefficients**, not a neural network or a new
physical controller. Use reverse-mode differentiation of the **actual discrete
leg/shoe rollout**, with a constrained quasi-Newton search in scaled coefficient
space. Preserve one shared controller and the current six-channel measured loss.

This document is the implementation framework. An experimental full-contact
`EngineAdjoint` now supports retained histories, the original full-stance measured
loss, and forward/backward CUDA graphs. Short coupled-window checks pass, including
loaded contact. The full-stance baseline loss and primal histories match exactly,
but the current full-stance finite-difference audit does not pass every direction.
**The full-gradient backend is not yet qualified for optimization, and the new
optimizer is not implemented.** The forward search remains the reference/default.

See [Newton internals reuse](NEWTON_REUSE.md) for existing tiled L-BFGS, cached
solve-adjoint, and workspace code that should guide the next implementation.

The problem is locally, piecewise smooth. Contact activation, the passive cap,
friction return, and dwell/release branches are not globally smooth. Good local
derivatives should be useful; they do not guarantee a globally smooth landscape
or a wall-time speedup over an already batched finite-difference search.

## 1. Preserve the actual optimization problem

Let `theta` contain the 12 control points for each of the four equilibrium
channels, in the existing row-major order. Optimize dimensionless coordinates:

```text
theta = theta_start + S z
S = diag(tile([0.2 m, 0.2 m, 0.5 rad, 0.5 rad], 12))
equilibrium(t) = B(t) theta
```

Read the scales from `FitConfig.parameter_scale`; do not hard-code different
gains or physical bounds. The spline basis and recorded sample maps are fixed.

For the saved baseline there are 5,760 semi-implicit steps over a 0.36 s stance.
The nominal timestep remains 62.5 microseconds. The refined replay remains at
half that timestep. Neither recorded motion after the initial state nor measured
forces become applied motion or forces during optimization.

For each of the hip, joint, and force blocks, the current residual is

```text
r[b,s,c] = sqrt(w[b,s] / 2) * (prediction[b,s,c] - target[b,s,c]) / tolerance[b]
L = sum(r * r)                    # NOT another 0.5 * sum(r * r)
grad_theta L = 2 * J.T @ r
```

Retain the native motion/force grids, interpolation, trapezoidal weights,
normalization, failure screens, and acceptance limits from
[`MeasuredObjective`](cartesian/gpu/objective.py) and
[`_Objective`](cartesian/fit.py). Do not add a new smoothing, regularization,
safety-penalty, or terminal-velocity loss silently.

## 2. Discrete backpropagation, not an equilibrium adjoint

Write the recurrent state and contact output as

```text
x[t+1] = F_t(x[t], theta)
wrench[t] = G_t(x[t], theta)
```

`x` includes the leg coordinates/velocities **and the shoe's internal memory**.
Material, friction, geometry, gains, initial physical state, and target data stay
fixed. Do not compute or optimize material/friction parameter gradients merely
because the existing identification adapter enables them by default.

A backward step consumes the outgoing state cotangent, the direct state-loss
seed, and the wrench-loss seed. It returns the incoming state cotangent and a
contribution to all 48 coefficient gradients.

Differentiate the eight executed passive-relaxation sweeps. They are a fixed,
warm-started transient update, not a solve taken to equilibrium. A converged
implicit-function adjoint would differentiate a different forward map. Likewise,
do not substitute a continuous-time adjoint for the implemented timestep map.

The mass solve can use an exact local implicit reverse rule because it really
solves a linear system. If `a = M_eff^-1 b` and `y = M_eff^-T a_bar`, then
`b_bar = y`. The existing Cholesky routine reads the lower triangle of `M`:

```text
M_bar[i,i] = -y[i] * a[i]
M_bar[i,j] = -y[i] * a[j] - y[j] * a[i]   for i > j
M_bar[i,j] = 0                           for i < j
```

This rule is now registered on the existing
[`solve`](cartesian/gpu/mechanics.py). It preserves the original solve, including
failure on invalid factors; it adds no regularization or projection of physics.

## 3. State and storage contract

Every rollout/replay boundary must carry at least:

| State | Precision | Reason |
| --- | --- | --- |
| Leg `q`, `v` | float64 | Dynamics and damping |
| Maxwell `q_state`, `peq_prev` | float32 | Viscoelastic memory |
| Bristle anchor and dwell | float32 | Tangential contact memory |
| Bristle grip flag | int32 | Select the actual contact branch |
| Last surround-compression field | float32 | Warm-start the next sweep |

Also retain the clock/index and validity/range bookkeeping needed to reproduce
failure and diagnostic histories. Preserve `surround_previous` if replaying its
rate diagnostics. Derive other scratch fields only after checking their consumers.
Do not detach continuous state at a checkpoint, reset the shoe between segments,
or treat integer branch flags as differentiable parameters.

Use distinct primal buffers for every step and every sweep **within a recorded
tape**. The current mutable forward buffers, device clock, and fused dynamic
sweep loop cannot simply be recorded once and reversed. A reverse launch must
see the primal inputs and time/basis index of its own step, not the final clock.

### Full tape and checkpointed tape share an interface

Start with an out-of-place full tape as the correctness reference. Benchmark it
against a reusable segment workspace with `K = 32, 64, 128` steps. `K = T` is the
full-tape case. Select the fastest validated option within a memory budget.

The existing `DifferentiableMidsoleFoundation` allocates approximately 22 float32
values plus one int32 flag per column per step, including eight surround sweeps.
At 910 columns and 5,760 steps, primals plus float gradients are about **0.88 GiB
per trajectory**. This is an allocation estimate, not a measured peak. It excludes
CUDA graphs, temporary workspaces, leg buffers, and allocator overhead. It makes
a full tape plausible for one trajectory on the 48 GiB A6000, but not for all
128 old search worlds (about 112 GiB for that storage alone).

Checkpointing reduces retained state approximately to

```text
B * (ceil(T / K) * checkpoint_bytes + K * step_workspace_bytes)
```

plus gradient buffers, objective history, graphs, and scratch. It adds one forward
recomputation of each segment. Use it when memory or graph/workspace limits justify
that cost. It is **full-horizon backprop**, not truncated BPTT.

### Reverse segment schedule

1. Forward through the complete stance; save boundary state and objective outputs.
2. Evaluate the original scalar loss. For a failed trajectory, return `valid=False`;
   do not differentiate infinity or give a truncated trajectory a good score.
3. Form loss seeds on GPU. For an interpolated prediction,
   `dL/dprediction = w * error / tolerance^2`; scatter through its lower/upper
   interpolation weights to state and preintegration-wrench histories.
4. For segments in reverse order, restore the exact checkpoint and replay the
   segment into its non-aliased local workspace.
5. Clear local adjoints, inject the incoming terminal cotangent and direct loss
   seeds, then launch the segment backward graph.
6. Accumulate local coefficient gradients into a separate persistent 48-vector.
   Copy the start-state cotangent into the next segment's carry buffers.

Give each direct state seed one segment owner. Interpolation may straddle a
boundary; scatter its two contributions globally before processing segments.
Do not double-count terminal/boundary samples or erase accumulated coefficient
gradients with `Tape.zero()`. Capture the shorter tail separately; do not round
up the simulated duration to a whole segment.

## 4. Reuse the shared laws, not an incompatible adapter

The existing
[`DifferentiableMidsoleFoundation`](../digital_instron_v2/dynamics_diff.py)
already calls the shared Maxwell, passive-balance, unilateral-contact, anchored
bristle, and ground-wrench expressions. Reuse those expressions. Do not revive
its deprecated smooth-friction kernel or create another material implementation.

Its current adapter is not a drop-in replacement:

- It owns a single carrier/world and allocates a full list of step/sweep arrays.
- It needs an explicit checkpoint import/export contract for recurrent state.
- Its bristle kernel atomically accumulates the carrier wrench. The qualified
  forward path uses fixed-order partial/final reductions. Add a tape-safe,
  deterministic reduction to match that path; do not accept extra forward noise
  as an unavoidable property of autodiff.
- Reused mutable diagnostics are not time histories for the measured force loss.
- Preserve float64 leg/objective, float32 shoe, and the current cast/FMA boundaries.

Keep the carried-ground force as the measured external wrench. Do not replace it
with internal column transfer traction. There is no new ground body or ground
reaction dynamics requirement in this fixed-plane problem.

## 5. Constraints belong in the optimizer

The existing control-polygon bounds form a convex polytope `A theta <= b` with
**264 linear inequalities in 48 variables**:

- 96 upper/lower position constraints.
- 88 upper/lower first-derivative control-polygon constraints.
- 80 upper/lower second-derivative control-polygon constraints.

Use the exact knot scales from
[`canonical_knot_scales`](cartesian/gpu/resident.py) and the duration factors in
[`Spline.bounds`](cartesian/trajectory.py). These are the exact constraints of
the *existing bound checker*. They are sufficient, generally conservative bounds
on the continuous spline; they are not necessary conditions for every bounded
spline curve.

In scaled coordinates, use `A_z = A S` and `b_z = b - A theta_start`.
Position boxes alone, coefficient clipping, or a `tanh` parameterization do not
enforce the rate and acceleration constraints.

Start with a projected-gradient direction as a transparent check. Then use a
small constrained quasi-Newton subproblem, with an SPD damped BFGS/L-BFGS model:

```text
min_d  g.T d + 0.5 d.T H d
s.t.   A_z (z + d) <= b_z
       ||d||_infinity <= trust_radius
```

The state dimension of this QP is only 48. Prefer a dense damped BFGS model here:
its 48x48 float64 matrix occupies just 18 KiB. L-BFGS is an optional alternative,
not a requirement for such a small parameter vector. Keep matrices/history,
solves, and selection on GPU; no new SciPy/PyTorch dependency is required. Warp
currently ships Adam/SGD, not this general-polytope constrained quasi-Newton
solver. The latter must be implemented and tested, not treated as an existing API.

For a feasible descent direction, evaluate a small batch of Armijo trial lengths
with the original forward engine. Reject failed rollouts and recheck exact spline
bounds before admission. Keep curvature pairs only with adequate positive
`y.T s`; reset/damp otherwise. If the quasi-Newton direction is not a descent
direction, use the projected-gradient fallback.

Ray clipping alone can stall at an active bound because an outward direction has
zero feasible length. It does not replace projection or a constrained subproblem.
An ordinary L-BFGS-B implementation also handles only boxes, not this polytope.
Use projected-gradient/KKT residuals to describe local stationarity; a loss
plateau is not proof of convergence or measured-fit acceptance.

## 6. GPU work allocation and performance criteria

Use **one incumbent trajectory for the scalar VJP**, then perhaps 4–8 useful
line-search candidates in parallel. Do not retain 128 taped worlds merely to
preserve the old coordinate-search allocation. The old 128-world search remains
an unchanged comparison backend; a new backend reports its own actual counts.

After eager correctness is established, preallocate arrays and capture forward,
reverse, seed, zeroing, and optimizer operations. A captured forward graph does
not supply a backward graph automatically. Compile/capture outside timing. Avoid
Python-per-substep work and per-step host transfers in the final implementation.

Reverse mode computes all 48 scalar-loss derivatives together. It removes the
need for 96 coordinate-probe trajectories, **not 96 serial GPU calls**: the old
probes already run together. Therefore neither a 48x/96x speedup nor sub-second
updates can be promised from rollout counts alone. Checkpoint replay and backward
kernels have real costs; a small single-world workload can also underuse the GPU.

Measure:

- Wall time for a complete forward/gradient/proposal/accepted-update cycle.
- Forward, reverse, recomputation, trial-search, and setup times separately.
- Peak VRAM, actual simulated world-steps, failed trials, and gradient evaluations.
- Best unchanged measured loss versus elapsed wall time, and time to fixed loss
  thresholds, from the same saved controller and several equal-time runs.

The current reference is about **1.39 s per full poll-plus-trial iteration** in
three short fixed-work repeats. See the [forward timing evidence](cartesian/gpu/README.md#measured-same-results-comparison).
Packed adjoint storage reduced a captured value/gradient evaluation to about
2.14 s, but that is a different unit of work and the full gradient is not qualified.
An adjoint optimizer must win on useful progress per second, not only evaluations
or GPU busy percentage.

If quasi-Newton curvature is insufficient, consider a later matrix-free
Gauss–Newton/LM method using validated JVP/VJP products. A scalar VJP gives
`J.T @ r`, not the entire residual Jacobian. Neither JVP support for the complete
runtime nor a cheap full Jacobian is assumed in phase one.

## 7. Implementation order and acceptance

1. **Local derivative foundation (started):** explicit mass-solve adjoint,
   overwrite-safe accumulations, and CPU/CUDA local gradient checks.
2. **Tape-safe contact step:** shared laws, deterministic wrench reduction,
   exact recurrent-state containers, no in-tape aliases or mutable clock inputs.
3. **Short coupled rollout:** compare all state/contact fields and failure branches
   with the qualified engine; check coefficient directional derivatives over
   1, 8, 32, and 128 steps.
4. **Full horizon and checkpointing:** test full-tape versus segmented gradients,
   terminal and cross-boundary samples, odd/even sweep histories, and reset replay.
5. **Constrained optimizer:** projected-gradient descent first, then quasi-Newton
   directions and GPU trial batches. Keep the default optimizer unchanged initially.
6. **Equal-wall-time experiment:** compare loss curves, gradient accuracy, memory,
   and complete update latency; only then choose the backend default and fuse more.
7. **Winner validation:** normal CPU/GPU qualification, fixed half-timestep replay,
   unchanged acceptance tests, and the existing spring-enabled HTML report.

Gradient checks must include several directions and adjacent finite-difference
scales. For contact use scaled perturbations large enough to resolve float32
rounding; record contact/cap/friction mode changes. Check branch-stable cases
strictly and label nonsmooth cases, rather than hiding them with a loose universal
tolerance. Checkpoints must reproduce branch/state histories. Monitor finite
gradients and long-horizon adjoint norms. Stop and diagnose invalid gradients
instead of quietly substituting zeros or changing the contact law.

The full-contact gradient, checkpoint equivalence, constrained optimizer,
wall-time advantage, and sub-second target are **not yet validated**.

## 8. Runnable first-stage audit

```bash
uv run --no-sync -m projects.impedance_instron.cartesian.gpu.gradient_audit \
  --device cuda:0 \
  --output outputs/impedance_instron/gradient_audit.json
```

Use `--device cpu` for the CPU check and a fresh output path. This loads the
hash-verified saved baseline, samples a stance pose, and tests sensitivities to
`q`, `v`, all spline coefficients through a direction, and an independent contact
wrench. Its scalar is a diagnostic seed on the next state/velocity, not a new fit
loss. **It does not backpropagate through the shoe or optimize a gait.**

Initial experiments produced NaNs with naive backprop. Adding only a solve
adjoint removed NaNs but left incorrect gradients from repeated vector-component
assignment. Explicit additive accumulation fixed those derivatives without
enabling the experimental global component-overwrite option. At epsilon 1e-4,
the corrected local probe's directional relative errors were below 7e-9 on the
sampled CPU/CUDA cases. This is local evidence, not a full-rollout gradient claim.
Raw experiments are retained under `outputs/impedance_instron/autodiff_design/`.

## Primary sources and checked implementation references

- [Warp differentiability](https://nvidia.github.io/warp/stable/user_guide/differentiability.html):
  Tape, memory overwrites, vector-component assignment, dynamic-loop limitations,
  and custom gradients. Checked against installed Warp `1.17.0.dev20260807` code.
- [Warp checkpoint example](https://github.com/NVIDIA/warp/blob/main/warp/examples/optim/example_fluid_checkpoint.py):
  explicit restore/recompute/adjoint carry and forward/backward CUDA graph capture.
  The installed copy was inspected too. Its duration-rounding shortcut is **not**
  appropriate for this fixed-timestep qualification protocol.
- [Shared contact functions](../digital_shoe/contact.py) and
  [tape-safe shoe adapter](../digital_instron_v2/dynamics_diff.py).
- [Current engine](cartesian/gpu/engine.py),
  [fused forward foundation](cartesian/gpu/foundation.py),
  [measured objective](cartesian/gpu/objective.py), and
  [resident finite-difference optimizer](cartesian/gpu/resident.py).
