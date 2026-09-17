# Twelve-point CUDA pipeline

The [project command](../../README.md) is the supported end-to-end path.
It builds new evidence and runs one shared 12-point controller on 128 fixed
worlds. Proposals, residual Jacobians, damped Gauss–Newton solves, bounds,
incumbent selection and fit history stay on CUDA during search.
The CPU submits graphs and checks the soft wall budget. Plateau stopping copies
only one status integer per completed iteration; it does not copy fit arrays.

Leg dynamics and the objective use float64. The shared shoe remains float32.
Physics, failure screens, six measured-loss channels, and acceptance limits
are unchanged. No CPU optimizer, multi-island mode, knot-insertion
continuation, count-comparison harness, or scaling sweep is retained.

For an already prepared and qualified run, the lower-level command is:

```bash
uv run --no-sync -m projects.impedance_instron.cartesian.gpu \
  outputs/impedance_instron/run12/baseline \
  --single-validation outputs/impedance_instron/run12/single/benchmark.json \
  --batch-validation outputs/impedance_instron/run12/mixed/benchmark.json \
  --contact-rounding-evidence outputs/impedance_instron/run12/contact/report.json \
  --output outputs/impedance_instron/run12/fit
```

Use a new fit directory. All input, source and runtime identities must match.
Source changes require new qualification; old source-hash exceptions have been
removed. The specific full-trajectory ankle-moment rounding warning can still
be qualified only by the unchanged common-pose contact limits. Other failed
numerical checks block fitting. This is separate from measured-fit acceptance.

Each completed fit writes a frozen half-step check and a spring-enabled HTML
report. Passive-cap activation stays visible even when numerical checks pass.
Use a separate `WARP_CACHE_PATH` when testing while another process compiles.

## Profile complete search

Measure full poll-plus-trial iterations from the saved controller:

```bash
uv run --no-sync -m projects.impedance_instron.cartesian.gpu.profile_search \
  --output outputs/impedance_instron/search_profile \
  --iterations 10 --repeats 3
```

This timing-only command checks the saved bundle's file hashes. It keeps the
128 worlds, 48 coefficients, physical timestep, objective, proposal seed, and
both search phases. It warms graph replay before timing and starts every repeat
from the same saved controller. Plateau stopping is disabled for fixed-work
comparisons. Use a fresh output directory.

`profile.json` records complete iteration wall time, completed real candidates
per second, candidate/padding counts, source and runtime identities, and losses.
`search_*.json` retain each repeat's history. Setup, compilation, and final
unloading remain separate from search timing. Completed candidate counts include
baseline reevaluations, not unique controllers. Compare equal iteration budgets
and loss histories, not GPU utilization alone.

The command deliberately produces **no numerical qualification or accepted fit**.
Historical hashes identify the input bundle; they do not certify changed source.
Use the [complete pipeline](../../README.md#run-the-complete-pipeline) for fresh
CPU/GPU qualification, refinement, and an interactive winner replay.

### Fused rollout kernels

For beds of up to 1,024 columns, the CUDA engine keeps the shoe update within
one block per independent world. Two columns per thread fit the 910-column
baseline into a 512-thread block. Each Jacobi sweep still reads the previous
sweep, including the previous driven-column values. The shared balance,
float32 FMA policy, and odd/even scratch-buffer results are unchanged.

The block also runs the shared pressure, friction, and ordered wrench reductions.
Exact zero-compression constitutive values are cached on device and refreshed
on reset or material changes. Compression diagnostics retain float64 divisions,
nonfinite checks, and integer cap counts. CPU execution and unsupported layouts
retain the shared fallback.

The float64 leg kernel advances the state and stages the next carrier. The
resident baseline loop therefore uses two kernels per timestep, plus reset,
initial staging, chunk-loop control, and objective work. The leg and shoe retain
their separate FMA policies. This scheduling follows the block-local approach
used by Newton's Kamino kernels; it does not replace either physical solver.

Bound-rejected padding slots are no longer integrated. Proposal generation,
random state, Jacobians, selection, and hard bounds are unchanged. The legacy
`physics_worlds` counter counts allocated launch slots; completed-world and
integrated-step counters report actual work. No unique-controller count is claimed.

### Measured same-results comparison

On the RTX A6000, three five-iteration repeats from the saved controller gave:

| Metric | Previous cooperative path | Fused path with padding skip and cache |
| --- | ---: | ---: |
| Median complete iteration | 2.757 s | 1.387 s |
| Completed real candidates / search second | 36.89 | 73.08 |
| Final loss after five iterations | 2.1830351003 | 2.1830351003 |
| Completed real candidates per repeat | 491 | 491 |
| Integrated world-steps per repeat | 7,372,800 | 2,828,160 |

The median time fell 49.7%. Every retained coefficient, tracking metric, and
batch loss matched exactly. Before/after runs were sequential, not interleaved.
The 789 padding slots per repeat remain allocated but do no integration work.
**Sub-second iterations were not reached.** Larger blocks, smaller blocks, and
a lower-matrix-only experiment were slower and were not retained.

One instrumented five-iteration search attributed about 98% of its wall time
to the rollout graphs. Optimizer and snapshot intervals totaled about 26 ms per
iteration, including about 22 ms for the serial damped solves. Small eager-kernel
intervals include host enqueue gaps; these are not precision microbenchmarks.

Evidence is under `outputs/impedance_instron/speed_round2/` when the local
benchmark bundle is available. The source-only checkout does not include these
generated files. Historical first-pass timings remain under `gpu_search_speed/`.

### Longer fitted run

Fresh CPU/GPU and mixed-world qualification passed for the retained code. A
200-iteration fit completed in 286.1 search seconds (1.43 s per iteration),
evaluating 25,473 real candidate slots. Loss fell from 2.18564 to 2.05824.
Both force RMS errors fell below 100 N, and the frozen half-timestep check passed.
The result is **not accepted**: vertical hip RMS error remains 20.68 mm against
the unchanged 20 mm limit. This is a longer run of the existing optimizer, not
an autodiff-fitting result or a changed acceptance criterion.

The controller, traces, qualification evidence, and verified spring replay are
under `outputs/impedance_instron/speed_round2_qualified/`. Open
`fit/report.html` to inspect the saved result.
