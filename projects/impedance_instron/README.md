# Twelve-point controller baseline

This worktree has one controller pipeline: one leg, one shoe, one stance,
with **12 cubic control points per equilibrium channel** (48 coefficients).
One shared controller uses a fixed batch of **128 CUDA worlds**.
There is no trunk, opposite leg, hip-angle motor, or added upper-body load.

## Selected baseline

`outputs/impedance_instron/baseline12/` holds the byte-identical winner from
`cartesian/fixed200_retest/fit12`, plus its portable shoe artifact.
[`baseline.json`](baseline.json) records the original result and file hashes.
Open `outputs/impedance_instron/baseline12/report.html` to inspect it immediately.
No Python process or refit is needed for this saved HTML replay.

- Loss: **2.18563891**, after 200 iterations from the original fresh seed.
- Frozen half-step check: passed; maximum GRF difference **14.50 N** (limit 25 N).
- Measured fit: **not accepted**. Hip-up RMS is 20.63 mm (limit 20 mm);
  vertical-force RMS is 104.04 N (limit 100 N).
- Passive compression cap: active in coarse and refined winners.

The new pipeline **starts from this optimized 12-point controller**, not a
fresh six-point seed, a knot-insertion conversion, or another experiment.
The original result is evidence, not a claim that the changed checkout has
already passed qualification. Historical source hashes are never rewritten.

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
The pipeline starts with the frozen filtered measured reference; raw C3D
preparation is not part of this baseline-only worktree.

The inherited foundation interface, material, friction, gains, masses, bounds,
initial physical state, loss, and acceptance limits are unchanged. Recorded
motion after the initial state and measured GRF are targets, never applied
motion or extra forces. The reference retains its original 20 Hz filtering
metadata. This cleanup does not certify biological validity or a new shoe
interface. General Newton APIs and the separate calibration/gait projects are
not retired by this controller cleanup.

## Cleanup verification

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
The selected baseline and smoke result both remain outside measured-fit acceptance.
