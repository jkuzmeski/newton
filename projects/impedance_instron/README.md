# Impedance Instron: pitch-driven mechanical ankle

The default example uses **measured running pitch and kinetics**, a rigid shoe
last, a compliant Digital Shoe, and a virtual COM. There is no human skeleton or
muscle simulation. The mechanical ankle rotates about one fixed mounting point.
**Measured marker XYZ trajectories do not drive the vertical controller.**

This is a representative cross-shoe engineering example. The capture shoe was
**not the modeled Puma**. Matching-shoe data will be collected later; this is not
same-shoe or human-performance validation.

## Run the supplied-data example

From this worktree, with the two generated local inputs already present:

```bash
uv run --extra examples -m newton.examples impedance_instron \
  --viewer gl --render-fps 20

uv run --extra examples -m newton.examples impedance_instron \
  --viewer null --test
```

The defaults are:

- `outputs/impedance_instron/stance_pitch.json`: real running inputs and calibrated
  heel-triangle rotations.
- `DigitalInstron/digital_shoe_showcase/digital_shoe.json`: the identified shoe.
- `outputs/impedance_instron/pitch_baseline/`: trace CSV, summary JSON, and offline HTML.

The orange point is the mechanical ankle. The blue point is the **mass-weighted
COM**, not merely the upper inertial body. The gray mesh is a rigid Instron last;
the colored columns supply the shoe's effective compliance. The last mesh itself
has no ground collision. This is not an anatomical insertion/contact solve between
a human foot and a shoe upper.

One measured window plays and then holds. There is no fabricated swing trajectory.
Missing inputs fail explicitly; there is no synthetic-motion or proxy-shoe fallback.

## Rebuild the inputs

The calibrated shoe may be reused unchanged. If it is missing, follow
[Digital Shoe](../digital_shoe/README.md):

```bash
uv run -m projects.digital_instron_v2.export_digital_shoe \
  --manifest DigitalInstron/manifest_v2.json \
  --output DigitalInstron/digital_shoe_showcase

uv run -m projects.impedance_instron.profile \
  --source-worktree /home/jo31399/newton-worktrees/foot-contact \
  --window-start 90 --window-end 95 --side left --stance-index 0 \
  --pitch-source heel-cluster --calibration "Cal 101.v3d.c3d" \
  --output outputs/impedance_instron/stance_pitch.json
```

The exporter refuses to overwrite an existing profile. Select a new output path
for another experiment, then pass it with `--profile`.

The source worktree contains the local S001 data and public `projects.gait_c3d`
adapters. Its own `uv` environment performs the offline extraction with an
isolated `ezc3d==1.7.2` dependency. The example runtime loads only the two JSON
inputs. It does not open C3D files, import the gait pipeline, or load a human model.

### The data are running, not the walking lead-in

The source is S001 `Trial 101.v3d.c3d` on `jkuzmeski/foot-contact`.
Its earlier 8-14 second reference is walking and is **not used** here.

- Explicit classification window: 90-95 s, with a steady 3 m/s belt
  command/reference (not independently measured belt speed).
- Selected left contact: **90.0115-90.3065 s**, duration 0.295 s.
- Preceding/following flight: 0.0535 / 0.0645 s; cadence about 167-168 steps/min.
- Exported force window: **89.9915-90.3265 s**, including 20 ms flight padding.
- 671 force samples at 2000 Hz; 65 original 100 Hz angle knots include at least
  150 ms of extra optical context on either side of the exported window.

Both feet use the same physical force platform during running. Event-specific
marker/COP checks assign the selected foot; walking plate-to-foot labels are not
reused. The reference is the sum of both measured platform signals, including
unloaded-platform noise. Both original force/moment channels are retained.
Opposite-foot support is zero by an explicit airborne inference. The diagnostic
`unassigned_*` channels are already included in the sum and must not be added again.

The v2 export changes the pitch reconstruction, not the measured forces. All old
array fields except `pitch_rad` reproduce the v1 profile exactly. Raw marker XYZ
arrays remain as context/provenance but are not used by the angle-only controller.

## How pitch is reconstructed

User-confirmed placement:

- `LHEE`, `LHEE2`, `LHEE3`: a triangle on the heel.
- `LTOE`: on the shoe upper over the **second metatarsal head**, not the toe tip.
- `LHLX`: hallux; it is not included in the rigid rearfoot fit.

The existing Cal 101 interval 0.5-1.0 s supplies a static heel template. A proper,
no-scale Kabsch fit maps that triangle into each running frame. The static vector
from the heel cluster toward LTOE, projected into ground XY, defines forward.
The transformed forward vector gives world sagittal pitch, with **+Y positive
rotation lowering the +X toe**. Full 3D fitted rotations are retained as evidence,
but only pitch is commanded in this rig. This is foot pitch relative to the
ground, not anatomical ankle flexion relative to a shank.

Static flat-foot standing and unchanged Cal/Trial marker placement remain explicit
mechanical-reference assumptions. This does not independently calibrate a sole
axis or the markers to the separate Puma last. All source hashes, proper rotations,
fit residuals, marker positions, and assumptions are in `pitch_reference` and
provenance. The supplied context's maximum frame RMS is about 0.625 mm; that is
geometric consistency, not proof of anatomical attachment or absence of gap filling.

The controller smooths the optical angle knots with a second-difference penalty
(default approximate 12 Hz cutoff), then builds a **C2 natural cubic**. Position,
velocity, and acceleration come from the same curve. Extra context keeps spline
end conditions outside the measured window. `--pitch-cutoff 0` disables smoothing
but retains C2 interpolation. This is offline command preparation, not a causal
online filter or new measured bandwidth. Both raw and applied angles are plotted.
The measured force reference is already 20 Hz filtered; the simulated force is
not filtered to hide contact dynamics.

## Mechanical ankle and vertical controller

Default ankle mount in oriented shoe coordinates: **(-0.075, 0, 0.105) m**.
Use `--ankle-mount X Y Z` to change it. It is a mechanical design choice, not an
inferred anatomical ankle center. All shoe vertices and column anchors use this
same transform:

```text
p_world = p_ankle + R_y(pitch) * (p_shoe_local - ankle_mount)
```

The fixture's 2 kg effective inertia is lumped at the ankle mount. The remaining
mass is the upper inertial body; total source mass is about 81.93 kg. Newton's
`SolverSemiImplicit` integrates both vertical states. Their force balance is:

```text
mf * ankle_z_ddot = Fshoe - Fleg - mf*g
mu * upper_z_ddot = Fleg + Fother - mu*g
COM_z = (mf*ankle_z + mu*upper_z) / (mf + mu)
```

`Fleg` is an equal-and-opposite internal generalized force. It uses measured force
feedforward and an impedance around a **constant vertical gap**, not around measured
marker translations. Nominal gains are K=12000 N/m and B=250 N s/m. They are rig
settings, not identified human impedance. The force-integrated COM reference maps
consistently to both masses; it is not imposed on their actual vertical motion.

A quintic schedule fades K and B to zero over the final 40 ms before measured
toe-off. A bounded internal release force compensates fixture gravity. Optional
`--unload-acceleration` adds lift acceleration; the default is zero. This release
phase is necessary to avoid a constant virtual spring continuing to press the
shoe into the floor after stance. It is an explicit controller policy, not a
measured lift trajectory. Changing-stiffness work and retraction work are included
in active-source power; no released virtual-spring energy is credited to the shoe.

The ankle track is fixed by default (`--ankle-x 0 --track-speed 0`). The upper
fore-aft guide retains the source force-integrated COM reference. Horizontal motion
and pitch are prescribed robot axes; guide reactions/work are explicit. This is
a two-slider rig, not a free spatial human leg. It does not predict traction or
braking/propulsive performance. The pitch motor is ideal: its demanded torque and
work are recorded, but no real motor saturation or electrical-efficiency model is used.

### Initial conditions remain assumptions

The source COM reference starts at height 1 m and vertical velocity 0 m/s at the
padded start, 89.9915 s. Neither is measured COM. `--initial-vz` in the exporter
creates a separate declared scenario. One free-fall estimate places the ankle for
threshold touchdown using pitch, its fixed mount, and that assumed entry velocity.
This is initialization only, not XYZ replay. Do not adjust measured forces to make
an assumed COM trajectory close periodically.

## Equilibrium-point controller

`--control equilibrium` replaces the scheduled impedance with a commanded
equilibrium trajectory, in the equilibrium-point form of Hogan impedance control:

```text
f = k(t) * (L0(t) - L) + b(t) * (L0dot(t) - Ldot)
```

`L0(t)` is the equilibrium (virtual) leg length. The deviation `L0 - L` is not an
error to be removed; it is how force is produced. Nothing else drives the leg.
There is **no measured-force feedforward** and **no engage or release schedule**.
The leg is unilateral, so stance ends when the command stops loading it rather
than on a clock. `--no-leg-unilateral` restores a leg that can also pull.

The damping ratio is commanded, not the raw damper: `b(t) = 2*zeta(t)*sqrt(k(t)*m_eff)`
with `m_eff` the upper mass. Scaling `k` and `b` by a common gain, as the legacy
schedule does, drops the damping ratio like `sqrt(gain)`; commanding `zeta`
directly does not.

Source power reduces exactly to `raw*L0dot + 0.5*kdot*e^2`, plus a term for any
force-limit intervention: work done by moving the equilibrium point and by
changing stiffness. The **pre-clamp** force carries the equilibrium term. Using
the applied force misreports every limited sample and credits a released leg with
work it never did. `newton/tests/test_impedance_equilibrium.py` checks this
against an independently computed `P_body + dE/dt + D` in every branch.

### Why the legacy reference is circular

The legacy rest length comes from twice-integrated capture-trial force, so the
equilibrium already encodes the capture shoe's own response. Driving a different
shoe with it is circular, and it is why the tracking error drifts through stance
and needs a fade to hide the residual. A commanded equilibrium removes that.

### Solving the command

`projects.impedance_instron.optimize` searches the command with CMA-ES against a
**task-level** cost. Measured force histories are never tracked. The cost has three
task terms and a set of rig constraints:

- **Stance impulse.** The velocity change over stance, which is measured impulse
  divided by mass. Not an absolute velocity, which would depend on the assumed
  initial condition.
- **Momentum history.** The same velocity change sampled at nine fractions of
  contact, each run normalised by its own contact duration. Matching only the
  endpoint leaves the arrival time of the impulse free; see below.
- **Effort and dissipation.** Mean squared actuator force, plus energy dissipated
  in the virtual damper. Effort alone penalises force magnitude, not waste.

```bash
uv run --extra examples -m projects.impedance_instron.optimize \
  --max-evaluations 1500 --seed 7 --weight-momentum 20 --weight-dissipation 1.0 \
  --command-output outputs/impedance_instron/command.json

uv run --extra examples -m newton.examples impedance_instron --viewer gl \
  --control equilibrium --control-params outputs/impedance_instron/command.json
```

A solved command records the knot counts it was solved at, so replaying it does not
require repeating the knot flags. Like the profile and the shoe artifact, it is
generated output and is not committed. The search takes roughly ten minutes on one
GPU.

### Two cost gaps found by inspecting the result

Both were found by overlaying the simulated ground reaction on the measured one
with `projects.impedance_instron.explain`, not by looking at the cost value.

**Timing.** With only the total impulse and the stance duration constrained, the
solved command delivered the right impulse at the wrong time: peak force at 75.5%
of stance against a measured 47.0%, and a vertical force RMS error of 569 N. The
cost never said *when* within stance the impulse arrives, and nothing else pinned
it down. Adding the momentum-history term moved peak force to 47.4%.

**Dissipation.** The effort term sees `mean(f^2)` only, so burning energy in the
virtual damper was free. The search raised damper dissipation from 63 J at the
analytic seed to 519 J, then to 654 J once the timing term was added. That is about
1 kW over a 0.3 s stance, near the whole metabolic power of the subject. Adding the
dissipation term cut it to 25 J with no loss of timing accuracy.

Neither gap was a failure of the controller structure. Both were under-specified
objectives, and the search exploited the slack exactly as it should.

### One solved result, for reference

Supplied running stance, 6/6/3 knots, 15 parameters, 1500 evaluations, momentum
weight 20 and dissipation weight 1.0.

| quantity | measured | solved | legacy schedule |
| --- | --- | --- | --- |
| time of peak vertical force [% stance] | 47.0 | 47.4 | 50.3 |
| peak vertical force [N] | 1762.6 | 1791.9 | 1884.9 |
| vertical force RMS error [N] | -- | 292.8 | 224.4 |
| vertical impulse error [%] | -- | +3.2 | +5.8 |
| COM vertical velocity RMS error [m/s] | -- | 0.249 | 0.160 |
| damper dissipation [J] | -- | 24.8 | 74.5 |

The legacy column is **given** the measured ground reaction as feedforward. The
solved command is not, and still matches peak timing eight times more closely.

What is still wrong: the impulse centroid sits at 52.7% against a measured 45.2%,
and the force RMS and COM velocity errors remain above the legacy controller. Gross
timing is correct; fine waveform shape is not. Nine checkpoints constrain when the
force arrives, not its detailed profile.

### On parameter count

Six length knots won under the earlier, incomplete costs. Once damper dissipation is
penalised that reverses: nine knots gives the lowest vertical force RMS error of any
run tried, 209.5 N, below the legacy schedule's 224.5 N.

| knots | dissipation term | damper [J] | peak at [% stance] | force RMS [N] |
| --- | --- | --- | --- | --- |
| 9/6/3 | absent | 1445.2 | 28.4 | 394.5 |
| 9/6/3 | present | 169.6 | 51.8 | 209.5 |

So the apparent degeneracy of the richer command was largely a symptom of the missing
energy term, not a property of the knot count. **No conclusion about the intrinsic
dimensionality of the equilibrium command is supported by these runs.**

### No single command wins

The solved commands trace a trade-off rather than a ranking. The 6-knot command with
momentum weight 20 matches the time of peak force almost exactly and dissipates the
least. The 9-knot command with momentum weight 60 tracks the force waveform and the
COM best but undershoots the vertical impulse.

| | 6/6/3, momentum 20 | 9/6/3, momentum 60 | legacy schedule |
| --- | --- | --- | --- |
| time of peak force [% stance], measured 47.0 | 47.4 | 51.8 | 50.3 |
| impulse centroid [% stance], measured 45.2 | 52.7 | 49.4 | 50.1 |
| vertical force RMS error [N] | 292.8 | 209.5 | 224.4 |
| vertical impulse error [%] | +3.2 | -4.4 | +5.8 |
| COM vertical velocity RMS error [m/s] | 0.249 | 0.235 | 0.160 |
| damper dissipation [J] | 24.8 | 169.6 | 74.5 |

Both beat the legacy schedule on peak timing and on impulse, and the 9-knot command
also beats it on force RMS. The legacy column is **given** the measured ground
reaction as feedforward; neither solved command is. No solved command yet matches the
legacy COM velocity tracking, and none should be presented as the single answer until
the cost weights are justified rather than chosen.

**These are rig settings solved for one shoe on one measured stance, not identified
human leg impedance and not a validated material or anatomical result.** The cost
weights are a declared engineering choice. Cross-shoe comparison of solved commands
is the intended use; that comparison has not been run here.

## Shoe side and winding

The original artifact is not modified. `orientation.py` makes a detached in-memory
copy. The supplied defaults explicitly interpret the baked source as right-sided
and choose a left fixture (`--source-shoe-side right --shoe-side left`). This follows
the read-only audit, not the artifact's misleading `*_left` label.

A Y reflection transforms every bed/visual/fixture coordinate and corresponding
neighbor direction. Triangle winding is preserved through reflection. A separate
repair of the known inverted last winding requires exact geometry and source-hash
matches; unknown meshes are not guessed. These transformations preserve sagittal
normal-force mechanics, verified by native tests. Source files, material parameters,
and original validation records stay unchanged.

This resolves the fixture's explicit engineering convention and known winding
fault, **not independent anatomical side certification**. Metadata retains that
boundary. A Y reflection cannot repair a sagittal pitch or height error.

## Compare shoe scenarios and inspect results

```bash
uv run -m newton.examples impedance_instron --viewer null --test \
  --output outputs/impedance_instron/pitch_baseline
uv run -m newton.examples impedance_instron --viewer null --test \
  --shoe-stiffness-scale 0.7 --compare outputs/impedance_instron/pitch_baseline \
  --output outputs/impedance_instron/pitch_softer
uv run -m newton.examples impedance_instron --viewer null --test \
  --shoe-stiffness-scale 1.3 --compare outputs/impedance_instron/pitch_baseline \
  --output outputs/impedance_instron/pitch_stiffer
```

A scale changes shear modulus and Pasternak coupling, not geometry or loading.
It is a synthetic sensitivity, not another identified commercial shoe. These are
fixed-controller output comparisons: COM endpoints can differ, so lower actuator
work alone is not an equal-task efficiency benefit. The report checks the profile, controller, release schedule, mount, orientation, processing,
timestep, and initial-state settings before showing fixed-scenario deltas.

`report.html`, `trace.csv`, and `summary.json` include forces, ankle torque, raw/applied
pitch, free ankle/COM response, release engagement, rigid-last ground clearance,
separate work channels, and rig energy closure. Measured COP remains in its original
heel-origin frame and is shown as separate context, not falsely registered to the
fixed-ankle rig. Active source work is not metabolic cost or motor electrical energy.

The supplied-data default completes loading and unloading without controller
saturation or rigid-last ground penetration. Peak force is about 2.08 kN and peak
compression about 18.9 mm: the compression is an extrapolation beyond the original
full-foot test amplitude. Force shape and impulse are not an exact human match.
The source shoe fit still fails peak/hysteresis gates; no performance validation
is implied by a passing engineering example.

## Retained legacy and negative checks

The v1 loader and marker-trajectory experiment remain available:

```bash
uv run -m newton.examples impedance_instron --reference-mode markers \
  --profile outputs/impedance_instron/stance.json --viewer null --test \
  --output outputs/impedance_instron/legacy_markers
```

Add `--mode replay` only with `--reference-mode markers` for strict XYZ replay.
Its known ~18.3 kN overload and rigid-last ground penetration remain failures;
do not relax the checks to make it pass. `--unload-duration 0` is a useful negative
pitch-mode check: it leaves substantial post-toe-off load and is also unqualified.
These are not fixed-settings shoe comparisons against the new mode.

## CUDA graph replay

The rollout is launch bound: one 64-substep frame issues 1024 kernel launches and
192 device copies, while the actual work of 910 foam columns and two rigid bodies is
negligible. Every kernel therefore reads its sample index from a one-element device
array, so one frame captures into a CUDA graph and replays with a single
`wp.capture_launch`. On an RTX A6000 a complete 2881-substep stance drops from about
6.1 s to about 0.51 s with the default controller, and from about 6.7 s to about
0.52 s with `--control equilibrium`.

```bash
uv run --extra examples -m newton.examples impedance_instron --viewer null --no-graph
```

`--no-graph` restores the plain per-substep launches. Capture is CUDA only; a CPU run,
or a device that refuses capture, falls back to the same plain path with a warning.

Captured and uncaptured runs agree bit for bit until the shoe touches down. After
touchdown they differ in the last bits, exactly like two uncaptured runs of the same
build: the foundation sums its column wrench with float atomics, and the GPU summation
order is not reproducible. This is a property of the contact accumulation, not of the
capture. `newton/tests/test_impedance_graph.py` therefore compares the two paths over
the contact-free part of the stance, where the whole substep sequence is exactly
reproducible.

## Record and test

```bash
uv run --extra examples -m newton.examples impedance_instron \
  --viewer gl --headless --test --num-frames 120 \
  --record-gif outputs/impedance_instron/pitch_baseline/stance.gif \
  --screenshot docs/images/examples/example_impedance_instron.jpg
uv run --extra dev -m unittest newton.tests.test_impedance_pitch \
  newton.tests.test_impedance_pitch_profile newton.tests.test_impedance_orientation \
  newton.tests.test_impedance_instron newton.tests.test_impedance_graph \
  newton.tests.test_digital_shoe
```

Use `--device cpu` for a CPU run and `--substeps 128` for timestep refinement.
The tests cover supplied-data execution, no marker-XYZ feedthrough, proper heel
rotations, C2 derivatives, native force/power balance, release, side reflection,
source integrity, and comparison rejection. Local input tests skip when restricted
data are absent. Generated inputs/reports/recordings stay ignored. See
[asset provenance](../../ASSET_PROVENANCE.md) and the
[acquisition protocol](../digital_shoe/ACQUISITION_PROTOCOL.md) before redistribution
or stronger dynamic validation claims.
