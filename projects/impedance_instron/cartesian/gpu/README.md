# Twelve-point CUDA pipeline

The [project command](../../README.md) is the supported end-to-end path.
It builds new evidence and runs one shared 12-point controller on 128 fixed
worlds. Proposals, residual Jacobians, damped Gauss–Newton solves, bounds,
incumbent selection and fit history stay on CUDA during search.
The CPU submits graphs and checks the soft wall budget. Plateau stopping copies
only one status integer per completed iteration; it does not copy fit arrays.

Leg dynamics and the objective use float64. The shared shoe remains float32.
Physics, failure screens, six measured-loss channels, and acceptance limits
are unchanged. No serial optimizer, multi-island mode, knot-insertion
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
