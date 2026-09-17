# Newton internals that can support controller autodiff

This is a source/API audit, not a claim that the existing Newton IK solver is a
drop-in trajectory optimizer. No Newton-core implementation was changed by this
audit. The controller still needs float64 coefficient optimization, its 264 hard
spline inequalities, and the original measured loss.

## Highest-value reuse

| Component | Useful part | Required adaptation |
| --- | --- | --- |
| `IKOptimizerLBFGS` | Tiled two-loop recursion, circular curvature history, slopes, parallel candidate layout/selection | Decouple FK/joint coordinates, support float64, add hard polytope constraints, use value-only trial evaluation |
| `IKOptimizerLM` | Tiled normal-equation assembly and damped solves | Needs a residual Jacobian, not just our scalar VJP; float32/FK assumptions remain |
| Featherstone dense solve | Cached Cholesky/solution and explicit solve VJP | Preserve our float64 lower-triangle solve contract; do not replace the leg physics or add regularization |
| Kamino dense factorization | Blocked LLT, factor reuse, offset-based batched storage, capturable iteration patterns | Inspected dense kernels are float32; SPD solves are not an indefinite KKT or inequality solver |
| Newton's Warp FEM scratch-store usage | Reuse allocated workspaces | Safe only when lifetimes preserve all tape primals; does not automatically solve fragmented gradient clearing |
| Public rigid-contact kinematics | Caller-owned differentiable point/distance outputs | Frozen contact set/normals only; not our Maxwell/bristle/passive-shoe law |

## 1. Reuse the IK optimizer's math, not its entire frontend

Public exports exist in [`newton.ik`](../../newton/ik.py):
`IKOptimizerLBFGS` and `IKOptimizerLM`.

The useful implementation is
[`ik_lbfgs_optimizer.py`](../../newton/_src/sim/ik/ik_lbfgs_optimizer.py):

- `_build_specialized()` defines tiled search-direction, history-update,
  directional-slope, and candidate-selection kernels.
- `_compute_search_direction_template()` applies the L-BFGS two-loop recursion.
- `_update_history_template()` manages `(s,y,rho)` in a ring buffer and rejects
  nonpositive curvature pairs.
- `_line_search()` lays out and evaluates several candidate lengths together.

The public classes require an articulation `Model`, float32 joint coordinates,
IK objectives, FK, and joint-space update integration. Our spline knots must not
be disguised as 48 dummy joints. An extraction/generalization of the math kernels
is cleaner than a dummy model or private-import hack.

**Do not reuse its trial policy unchanged.** It evaluates gradients at every
candidate to check strong-Wolfe curvature. Full-stance backprop is expensive;
our first implementation should use value-only feasible Armijo trials, then
compute a new gradient only at the accepted point.

Also preserve normalization explicitly. The IK code's autodiff seed is the
residual itself (`J.T @ r`). Our unchanged measured loss is `sum(r*r)`, so our VJP
must use `2*J.T @ r`. Do not transplant seeds, absolute curvature thresholds, or
cost conventions without checking them in our scaled coordinates.

The current IK joint-limit objective is a soft residual. It is not a substitute
for the 264 position/rate/acceleration control-polygon inequalities. Keep strict
canonical bound checks before admitting any candidate.

Recommendation: first generalize the tiled optimizer core behind an appropriate
public API, retaining the existing IK API/defaults. Keep the controller's value/
gradient oracle and constraint layer separate. Do not switch to penalty-fitting
or change the measured objective just to accommodate the IK frontend.

## 2. Featherstone already uses the right solve-adjoint pattern

In [`featherstone/kernels.py`](../../newton/_src/solvers/featherstone/kernels.py):

- `dense_cholesky()` forms a factor.
- Its custom gradient is deliberately a no-op.
- `dense_solve()` uses that saved factor and writes the solution.
- `adj_dense_solve()` solves the transposed system using the saved factor and
  forms the matrix VJP from the saved solution.

This supports our decision to differentiate the linear solve rather than
blindly backpropagate through factorization loops. It also suggests an efficiency
improvement: save the small factor and solution per step instead of recomputing
all of them during reverse mode.

The implementation is float32, uses flat offsets and scratch arrays, and has
specific gradient-buffer requirements. It is not a direct replacement for our
5x5 float64 function. Our solver uses the lower triangle as a symmetric matrix;
its tested lower-triangle adjoint must remain correct. Reuse the strategy and
factor-caching mechanism, not incompatible storage or a different solve contract.

[`solver_featherstone.py`](../../newton/_src/solvers/featherstone/solver_featherstone.py)
also separates reusable model workspace from per-state differentiable buffers.
That is useful guidance for keeping tape inputs alive while reusing untaped work.

## 3. Kamino linear algebra helps with a constrained step

Relevant code:

- [`linear.py`](../../newton/_src/solvers/kamino/_src/linalg/linear.py)
- [`llt_blocked.py`](../../newton/_src/solvers/kamino/_src/linalg/factorize/llt_blocked.py)
- [`llt_sequential.py`](../../newton/_src/solvers/kamino/_src/linalg/factorize/llt_sequential.py)
- [`conjugate.py`](../../newton/_src/solvers/kamino/_src/linalg/conjugate.py)
- [`blas.py`](../../newton/_src/solvers/kamino/_src/linalg/blas.py)

The blocked LLT implementation provides reusable factor/solve workspaces and tile
padding. With 32-wide tiles, a 48-variable matrix is padded to 64. The sequential
alternative assigns one thread per matrix; that is not the first choice for GPU
parallelism in one coefficient problem.

Although some wrappers advertise a generic `dtype`, the inspected LLT and dense
GEMV kernels explicitly accept float32 arrays. A dtype argument alone does not
make that dense path float64. A generalized kernel and numerical tests are needed.
Custom-operator/generic Krylov paths need their own audit before claiming float64
support. These helpers are not exported as public Newton linear-solver APIs.

Cholesky can handle an SPD Hessian or suitable SPD Schur/reduced system. It cannot
be applied directly to an indefinite saddle-point KKT matrix. None of these
linear solves alone supplies the inequality algorithm.

Kamino also has
[`PADMMSolver`](../../newton/_src/solvers/kamino/_src/solvers/padmm/solver.py),
but its current interface and storage are tied to a Kamino dynamics model and its
constraint problem. It is another implementation reference, not a verified
plug-in for this coefficient polytope.

The capturable-loop pattern in `conjugate.py` is useful: keep convergence masks
and iteration counts on device, with a conditional CUDA graph loop or a bounded
unrolled fallback. Its noncaptured path reads convergence state on the host; do
not describe every execution mode as host-transfer-free.

## 4. Workspace reuse: useful, but no free tape memory reuse

The implicit MPM solver uses public `warp.fem.TemporaryStore` and
`warp.fem.borrow_temporary`:
[`solver_implicit_mpm.py`](../../newton/_src/solvers/implicit_mpm/solver_implicit_mpm.py).
This needs no new dependency beyond Warp.

Use pooling for scratch with non-overlapping lifetimes, or a checkpoint workspace
whose previous backward pass has finished. Reusing and overwriting the same
history buffer at every step would break BPTT.

A follow-up measurement split the current complete-stance graph:

| Warm operation | Measured wall time |
| --- | ---: |
| Forward | 0.714–0.743 s |
| Clear gradients | 3.219–3.409 s |
| Reverse kernels | 2.849–2.860 s |

The tape tracks **172,818 gradient arrays**. These are three warm measurements
following one cold graph replay on the A6000. They are not optimizer-update
benchmarks. Evidence is in
`outputs/impedance_instron/adjoint_contact/backward_breakdown.json`.

This makes packed per-field storage and bulk gradient clearing a concrete next
experiment. Preserve distinct timestep views, then clear the owning gradient
buffers rather than issuing a clear for every view. No packing speedup has yet
been measured. A temporary pool alone is not a bulk-zeroing solution.

## 5. Contact-gradient conventions, not another shoe model

Public `newton.eval_rigid_contact_kinematics()` is implemented in
[`contact_kinematics.py`](../../newton/_src/sim/contact_kinematics.py), with helpers
in [`differentiable_contacts.py`](../../newton/_src/geometry/differentiable_contacts.py).
It differentiates caller-owned contact point/distance outputs with respect to
body transforms, while freezing normals and the discrete contact set.

This is a useful API/lifetime pattern and a clear statement of the derivative
boundary. It does not provide the shoe's material, anchored bristles, passive
relaxation, or derivatives of contact-set changes. Keep our shared shoe law.
Neither this helper nor switching to another Newton solver resolves the remaining
full-stance finite-difference/branch sensitivity automatically.

## Practical order

1. Profile gradient clearing versus actual reverse arithmetic.
2. Reuse persistent/packed storage and cached-solve patterns where measurements
   justify them; preserve the tested primal and gradient contracts.
3. Generalize Newton's existing tiled L-BFGS math instead of writing that math again.
4. Add float64 hard-constraint handling and value-only trial rollouts.
5. Keep the default optimizer unchanged until full-gradient validation and
   equal-wall-time loss comparisons pass.

This audit did not benchmark the proposed substitutions or change Newton-core
APIs. Internal source can be inspected and generalized, but runnable project
examples should use supported public exports rather than private imports.
