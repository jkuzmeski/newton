# Twelve-point controller baseline

This worktree has one controller pipeline: one leg, one shoe, one stance,
with **12 cubic control points per equilibrium channel** (48 coefficients).
One shared controller uses a fixed batch of **128 CUDA worlds**.
There is no trunk, opposite leg, hip-angle motor, or added upper-body load.

## Selected baseline

`outputs/impedance_instron/baseline12_accepted/` holds the byte-identical selected
fit from the K2/D2 gain campaign, plus its portable shoe artifact.
[`baseline.json`](baseline.json) versions its 48 controller coefficients and
frozen identity, and records its source, fixed gains, acceptance, and file hashes. Open `outputs/impedance_instron/baseline12_accepted/report.html`
to inspect the saved replay without running Python or fitting again.

- Loss: **1.53892290**, after 200 iterations from a fresh perturbed controller.
- Initialization seed: **2718**. Search seed: **101**. No prior fitted coefficients
  or optimizer history were used to obtain this selected result.
- Fixed stiffness: hip **[8000, 12000] N/m**, joints **[240, 180] N·m/rad**.
- Fixed damping: hip **[80, 80] N·s/m**, joints **[12, 8] N·m·s/rad**.
  These gains are **2× the previous nominal stiffness and damping**.
- Measured RMS: hip **7.053141 / 19.988332 mm**, knee/ankle
  **0.02118672 / 0.02920087 rad**, force **89.852197 / 79.162898 N**.
- Native and half-step measured RMS, half-step agreement, and spring-contact
  replay: **passed**. Passive compression caps remain active and visible.
- Hip-Z RMS margin: only **0.011668 mm** below the unchanged 20 mm limit.
  This is numerical acceptance, not a claim of robustness or physical validity.

The default pipeline now starts from this accepted 12-point controller.
Further optimization still selects by the unchanged scalar loss; a lower-loss
result can fail an individual RMS limit. Always inspect its acceptance result.

The former baseline is preserved at `outputs/impedance_instron/baseline12/` and
in [its archived manifest](baselines/baseline12_initial.json). Use `--baseline`
with that directory to repeat historical runs. Original fit evidence and source
hashes are never rewritten. The [gain workflow](cartesian/gpu/README.md#fixed-gain-screens-and-fresh-restarts)
provides the reusable screen, restart, probe, and replay commands.

## Run the complete pipeline

Run from the repository root with its existing `uv` environment and CUDA.
Choose a new output directory. The local baseline bundle is required; restricted
motion and shoe data are not supplied by a source-only checkout.

```bash
uv run --no-sync -m projects.impedance_instron \
  --output outputs/impedance_instron/run12
```

This command:

1. Checks bundle hashes and replays the selected controller on CPU.
2. Runs same-step CPU/GPU, common-pose contact, and 128-world
   permutation/reset/isolation checks, including a failed candidate.
3. Fits one shared controller on GPU with the original six-channel measured loss.
4. Runs a frozen half-timestep check and writes `fit/report.html`, including
   verified spring and deformation views.

Defaults are 200 iterations, a 3,600-second soft search cap, seed 17,
plateau patience 20, and relative plateau improvement 0.0001. A plateau is not
proof of convergence. Setup, numerical validation, refinement, and reporting
are outside the search cap. `--iterations 1` is a short end-to-end smoke run.
The timestep stays 62.5 microseconds, with 31.25 microseconds for refinement.

### Fit a new controller from scratch

The default command above is a warm start. To generate new, unfitted controller
coefficients instead, use:

```bash
uv run --no-sync -m projects.impedance_instron --from-scratch   --output outputs/impedance_instron/fresh12 --iterations 200 --wall-seconds 3600
```

This mode does not use a saved controller or previous optimizer history.
It samples `q_reference + (D/K) * velocity_reference` at twelve cubic-spline
Greville abscissae, then contracts the channels toward their initial neutral
points until the original strict control-polygon bounds hold. No simulation
loss or measured GRF is used to choose the seed. This is deterministic,
measurement-based initialization, not random coefficients or prescribed motion.
The recorded data, calibrated shoe, physical model, gains, and limits stay fixed.

The prepared baseline and fit summary record the initialization formula,
contraction factors, starting coefficients, and `used_previous_controller_coefficients: false`.
Use `--from-scratch` with separate stages to require matching fresh provenance.

Stages can also run separately:

```bash
uv run --no-sync -m projects.impedance_instron --output outputs/impedance_instron/run12 --stage prepare
uv run --no-sync -m projects.impedance_instron --output outputs/impedance_instron/run12 --stage validate
uv run --no-sync -m projects.impedance_instron --output outputs/impedance_instron/run12 --stage fit
uv run --no-sync -m projects.impedance_instron --output outputs/impedance_instron/run12 --stage report
```

`--baseline DIRECTORY` selects a complete saved bundle with the same manifest
format. Existing stage outputs are not overwritten, except an explicit report
rebuild. If qualification fails, inspect the failed flags; do not enlarge the
limits or substitute evidence from another input or source version.

## Differentiable search framework

See [the reverse-mode search framework](AUTODIFF_SEARCH.md) for full-horizon
backprop, memory/checkpoint policy, exact spline constraints, and validation.
The shared mass-solve adjoint, tape-safe full-contact rollout, measured objective,
and runnable gradient audits are implemented as experimental diagnostics. Short
coupled-window checks pass, but the full-stance gradient audit remains unqualified.
There is no integrated adjoint optimizer; the current forward search stays the default.

## Search performance

Use the [complete-search profiler](cartesian/gpu/README.md#profile-complete-search)
to compare equal-work GPU searches without rebuilding an HTML report on every
repeat. It reports full iteration time and useful candidate throughput, not
kernel enqueue time. Profiling does not replace numerical qualification.

## What remains

- `pipeline.py`: the single entry point and stage order.
- `cartesian/`: reference validation, spline algebra, fixed physical model,
  measured objective, CPU reference rollout, shoe attachment, and replay.
- `cartesian/gpu/`: GPU dynamics/objective, shared resident search, numerical
  qualification, and GPU spring export.
- Newton and `projects/digital_shoe/`: shared framework and material/contact laws.

Retired bilateral, paper, two-stiffness and learned-controller rigs, old
preparation chains, serial optimizers, multi-island search, control-count
comparisons, compatibility aliases, and their tests/reports are removed.
The default pipeline starts with the frozen filtered measured reference.
Subject-specific C3D preparation is a separate, fail-closed workflow described
in the GPU README; it does not replace frozen inputs during a controller fit.

Apart from the explicitly selected 2× stiffness and damping, the inherited
foundation interface, material, friction, masses, bounds, initial physical state,
loss, and acceptance limits are unchanged. Recorded
motion after the initial state and measured GRF are targets, never applied
motion or extra forces. The reference retains its original 20 Hz filtering
metadata. This cleanup does not certify biological validity or a new shoe
interface. General Newton APIs and the separate calibration/gait projects are
not retired by this controller cleanup.

## Historical cleanup verification

The controller project shrank from **97 Python files / 43,084 source lines** to
**28 files / 7,094 lines**: a net deletion of **35,990 lines (83.5%)**.
These counts exclude tests, generated outputs, and compiler caches.
Another 48 obsolete test modules were removed. Newton public source is unchanged.

The retained pipeline passed 90 targeted tests on CPU/CUDA, all pre-commit
checks, and a fresh one-iteration end-to-end CUDA smoke run. Same-step parity,
128-world isolation/reset/permutation, frozen half-step refinement, and spring
replay passed. Spring-history errors were zero. The smoke run is verification,
not a replacement for the selected 200-iteration baseline or a convergence claim.
Its evidence is in `outputs/impedance_instron/cleanup_validation/verification.json`
and its replay is `outputs/impedance_instron/cleanup_validation/fit/report.html`.
That earlier baseline and cleanup smoke result were outside measured-fit acceptance.
They are historical evidence, not the newly selected accepted baseline above.
