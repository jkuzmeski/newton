# Impedance Instron

A runnable **running-stance** experiment with a shoe, a foot fixture, and a virtual
center of mass. There is no human skeleton, muscle model, or human multibody
rollout. Newton integrates two inertial bodies; only their vertical motion is
free. The robot track and foot pitch follow measured motion.

This is an engineering example, not a validated prediction of human performance.
It does not implement a physical robot or online learning yet.

## Run the supplied local example

From this worktree:

```bash
uv run --extra examples -m newton.examples impedance_instron --viewer gl

# Complete the same stance without a window and check the engineering state.
uv run --extra examples -m newton.examples impedance_instron \
  --viewer null --num-frames 120 --test
```

The defaults load `outputs/impedance_instron/stance.json` and
`DigitalInstron/digital_shoe_showcase/digital_shoe.json`. These generated inputs
are intentionally ignored. Missing inputs fail explicitly; there is no synthetic
motion or proxy-shoe fallback. The blue point is the **combined COM**, not just
the upper inertial slider. The exposed colored columns are the actual calibrated
Digital Shoe. One stance plays and then holds; no fabricated swing is appended.

## Rebuild the inputs

Reuse the existing calibrated shoe artifact, or generate it from the supplied
measurements as described in [Digital Shoe](../digital_shoe/README.md):

```bash
uv run -m projects.digital_instron_v2.export_digital_shoe \
  --manifest DigitalInstron/manifest_v2.json \
  --output DigitalInstron/digital_shoe_showcase

uv run -m projects.impedance_instron.profile \
  --source-worktree /home/jo31399/newton-worktrees/foot-contact \
  --window-start 90 --window-end 95 --side left --stance-index 0 \
  --output outputs/impedance_instron/stance.json
```

The exporter refuses to overwrite an existing profile. Use a new output filename
for another selection and pass it with `--profile` when running the example.
The source worktree must contain the local S001 acquisition and public
`projects.gait_c3d` adapters. Its own `uv` environment performs the offline C3D
extraction, with `ezc3d==1.7.2` as an isolated optional dependency. The example
runtime needs only the two portable JSON inputs; it never imports the gait
pipeline, opens a C3D, or loads a human model.

### Why this is running, not the walking lead-in

The source is S001 `Trial 101.v3d.c3d` in branch `jkuzmeski/foot-contact`.
The full recording is about 178 seconds long. The earlier 8-14 second reference
on that branch is **walking and is not used** here.

The explicit 90-95 second window has:

- A steady 3 m/s belt command/reference, not independently measured belt speed.
- Alternating marker-assigned support events and force-free flight.
- Approximately 0.30 second contacts and 167-168 steps/min cadence.
- Selected left contact at **90.0115-90.3065 s**, with 0.0535 s preceding flight
  and 0.0645 s following flight.
- Exported interval **89.9915-90.3265 s**, including 20 ms flight padding at each end.

Both feet use **the same physical force platform** during this running section.
The exporter assigns each event from simultaneous heel/toe markers, COP distance,
and opposite-foot height. It does not carry over the walking plate-to-foot labels.
It sums the two measured platform signals, preserves both original force/moment
channels, and includes the unloaded platform noise without clipping. Opposite-foot
support is zero by an explicitly recorded airborne inference, not a separate
measurement. The `unassigned_*` noise channels are already included in the sum.

The profile retains 100 Hz marker information and 2000 Hz measured forces. Motion
upsampling does not create additional measured bandwidth. Source hashes, processing,
axis transforms, event evidence, and the shared heel/COP origin are inside the
sealed JSON. Its loader verifies force sums, COP bookkeeping, and COM integration.

## Mechanical model

Let `M` be total mass, `mf` the fixture mass, and `mu = M - mf` the upper mass.
The default fixture mass is 2 kg; the source total mass is approximately 81.93 kg.
The force-integrated reference `Zref` represents the **total centroid**:

```text
M * Zref_ddot = Fz_reference + Fz_other - M*g
zu_reference = (M*Zref - mf*zf_reference) / mu

length = zu - zf
Fleg = Fz_reference - mf*(g + zf_reference_ddot)
       + K*(length_reference - length)
       + B*(length_reference_dot - length_dot)

mf * zf_ddot = Fshoe - Fleg - mf*g
mu * zu_ddot = Fleg + Fz_other - mu*g
Z = (mf*zf + mu*zu) / M
```

`SolverSemiImplicit` advances the native bodies. Leg forces are equal and opposite.
They cancel from total vertical momentum. This is a **two-slider robot fixture**,
not a free spatial human leg: the prescribed guides supply the nonvertical
reactions. The default `K=12000 N/m` and `B=250 N s/m` are scenario settings, not
identified human impedance.

Measured heel motion and heel-to-toe pitch guide the fixture. A fixed heel-to-shoe
center offset keeps COP and foot in the same X frame. A single height shift aligns
the lowest outsole with the ground at the measured 50 N touchdown threshold.
This is a bench registration, not an anatomical shoe fit. Optical-clock C1 Hermite
curves provide consistent pose, velocity, and acceleration; the code does not
differentiate piecewise-linear optical data at the solver rate.

### Initial COM conditions are assumptions

Force alone does not identify absolute COM height or entry velocity. The default
profile uses height 1 m and vertical velocity 0 m/s at **89.9915 s**, the padded
start, not at touchdown. This is a repeatable initial condition, not measured COM.
It must not be interpreted as a qualified human trajectory.

For a separate sensitivity experiment, `--initial-vz -0.244957105` in the exporter
approximately gives equal COM height at the start and end of this one contact.
That is an **equal-height single-stance assumption**, not measured COM or a
periodic gait correction. A local pelvis-marker velocity proxy gives a different
value. Keep these scenarios in separate profiles and do not adjust measured forces
to make them agree. An initial-condition change correctly fails the fixed-input
shoe-comparison audit.

## Compare controlled shoe scenarios

Run the baseline before requesting comparisons:

```bash
uv run -m newton.examples impedance_instron --viewer null --test \
  --output outputs/impedance_instron/baseline
uv run -m newton.examples impedance_instron --viewer null --test \
  --shoe-stiffness-scale 0.7 --compare outputs/impedance_instron/baseline \
  --output outputs/impedance_instron/softer
uv run -m newton.examples impedance_instron --viewer null --test \
  --shoe-stiffness-scale 1.3 --compare outputs/impedance_instron/baseline \
  --output outputs/impedance_instron/stiffer
```

A scale changes the artifact's shear modulus and Pasternak coupling together.
It is a **synthetic material sensitivity**, not another identified commercial shoe.
Geometry, loading, controller, registration, and initial state stay fixed.

Each output directory contains `trace.csv`, `summary.json`, and `report.html`.
The offline report plots measured versus simulated loading, COP, COM response,
and separate power channels. It checks input hashes, processed reference, runtime,
fixture mass, controller, contact settings, timestep, and initial conditions before
showing numerical comparison deltas. Failed or incomplete runs remain visible.

`--mode replay` prescribes vertical motion too. Its extra vertical guide work is
reported separately; prescribed COM movement is not a shoe benefit. In impedance
mode both vertical states can depart from their references. With the current
marker-proxy registration, strict replay reaches about **18.3 kN** and fails the
engineering peak-force bound. Its saved report retains that failure. This is not
a validated displacement-replay fixture; do not interpret those loads as human
running forces or relax the limit to make the test pass.

Active source work,
passive damper loss, track work, pitch-drive work, contact work, and rig energy
balance are distinct. None is a metabolic-cost estimate.

For a finer integration check, repeat with `--substeps 128` into a new directory.
Do not treat a different timestep as a fixed-settings shoe comparison.

## Observed engineering checks

The supplied default running profile gives approximately:

| Scenario | Peak shoe force | Peak compression | Positive active-source work |
| --- | ---: | ---: | ---: |
| 0.7 stiffness scale | 1.883 kN | 18.42 mm | 71.28 J |
| Identified baseline | 1.832 kN | 13.13 mm | 75.58 J |
| 1.3 stiffness scale | 1.992 kN | 12.06 mm | 76.27 J |

All three complete the example without controller saturation. These are different
outcomes under one fixed controller, not equal-task shoe-efficiency rankings.
The softer case exceeds the original full-foot compression amplitude and is an
additional extrapolation. None is independent human validation.

Halving the native integration timestep changes baseline peak force by about
0.04 N and positive active-source work by about 0.03 J. The rig's energy residual
falls from about 0.15% to 0.06% of absolute power throughput. These are numerical
checks for this one scenario, not broad dynamic-model validation.

## Record a view and run regression tests

```bash
uv run --extra examples -m newton.examples impedance_instron \
  --viewer gl --headless --num-frames 120 --test \
  --record-gif outputs/impedance_instron/baseline/stance.gif \
  --screenshot docs/images/examples/example_impedance_instron.jpg
uv run --extra dev -m unittest newton.tests.test_impedance_instron
```

The GIF plays slowly for inspection. Engineering tests cover native vertical force
balance, reciprocal controller power, mass-consistent COM references, interpolation,
shared COP origins, signed work, and comparison rejection. They do not validate a
human response. The example's `test_final()` checks the real supplied-data run.

## Important limits

- Only normal shoe response affects the free dynamics. Measured horizontal force
  is context, not fitted shear/traction. Fore-aft travel and pitch are prescribed.
- The rigid last omits toe joints, arch motion, muscles, and tendons.
- The normal-compression shoe fit has failed peak/hysteresis validation gates;
  stride-rate tilted loading is not independently validated. See the
  [shoe acquisition protocol](../digital_shoe/ACQUISITION_PROTOCOL.md).
- `shoe_contact_work` is boundary work, not a closed-cycle foam hysteresis test.
  Rig energy closure is a numerical check, not human validation.
- The source data, footwear geometry, and derivatives are restricted to internal
  fork use. Do not redistribute them upstream; see [asset provenance](../../ASSET_PROVENANCE.md).
