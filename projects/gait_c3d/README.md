# Instrumented-treadmill C3D gait pipeline

The staged path from measured-load reconstruction through predictive torque- and
muscle-driven gait is defined in
[`FORWARD_DYNAMICS_ROADMAP.md`](FORWARD_DYNAMICS_ROADMAP.md). That document is
the implementation and scientific-QC contract for the flagship pipeline.

This project reconstructs one measured stride from the S001 instrumented-treadmill
trial with the public `newton.opensim` scaling, marker-placement, inverse-kinematics,
inverse-dynamics, and Static Optimization APIs. It is an analysis pipeline, not a
synthetic example. Raw acquisitions remain immutable and every cache and generated
artifact is written outside the Newton checkout.

## Inputs and outputs

The default immutable input directory is
`/home/jo31399/newton-data/gait/incoming` and must contain:

- `Cal 101.v3d.c3d`
- `Trial 101.v3d.c3d`
- `LeftBelt101.txt`
- `RightBelt101.txt`
- `Speedchange101.txt`

The default generated-data directory is
`/home/jo31399/newton-data/gait/processed/trial_101/latest`. The command rejects an output
path inside the repository. Each run is built in a sibling staging directory and
replaces `latest` from a completed staging directory only after every artifact and QC file succeeds, so a
failed or `--skip-static-optimization` run cannot mix stale outputs. A first run
reads the 376 MB dynamic C3D and writes `trial_cache.npz` with the complete marker
and two-platform force/COP/free-torque signals. Cache reuse validates its schema,
extraction configuration, Trial C3D hash, and all three belt-export hashes.
`--rebuild-cache` explicitly rebuilds it; sources are never modified.

Run from the repository root with the worktree environment:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m projects.gait_c3d.pipeline
```

Useful deterministic/staged options are:

```bash
# Re-extract both C3Ds, then run every stage.
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m projects.gait_c3d.pipeline \
  --rebuild-cache

# Validate through inverse dynamics without the slower muscle optimization.
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m projects.gait_c3d.pipeline \
  --skip-static-optimization

# Change stride search or coarse Static Optimization resolution.
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m projects.gait_c3d.pipeline \
  --search-time 20 --so-nodes 12
```

Use `--help` for input/output, stride search time, coarse node count, and Warp
device options. CPU is the deterministic default.
Static Optimization runs unless it is *explicitly* disabled; its default 12 nodes
are selected directly from the 100 Hz stride grid and resampled to every stride
frame. Both inverse dynamics and Static Optimization consume the same archived,
sanitized external-wrench frames.

## Frames and treadmill conversion

The acquisition lab is **+Z up, -Y forward, +X left**. OpenSim's native ground
frame is **+X forward, +Y up, +Z right**. The proper rotation applied to positions,
forces, COP, and free torques is

```text
R_lab_to_OpenSim = [[ 0, -1,  0],
                    [ 0,  0,  1],
                    [-1,  0,  0]]
```

The two belt metric exports are required to contain identical values. Their row
indices (zero based) are registered to C3D seconds with the documented anchors:

```text
row:     0, 1356, 22244, 43139, 52098, 53223
second:  0, 4.68, 74.69, 144.70, 174.71, 178.46
```

Piecewise-linear registration is followed by trapezoidal integration. The source
export does not state a unit, so treating `SPEEDCHANGE` as **m/s** is an explicit,
protocol-derived assumption rather than recorded source metadata. Positive speed `u` is forward-equivalent travel. For
each marker and COP:

```text
x_overground = x_treadmill + s_relative
v_overground = v_treadmill + u
s_relative(t) = s_absolute(t) - s_absolute(segment start)
```

This implementation is explicitly restricted to the verified level/aligned Trial
101: force-platform corner Z coordinates must be planar and the treadmill long axis
must be lab Y with participant-forward -Y. Inclined treadmill reuse is rejected
because that requires estimating Jung's treadmill-frame orientation `R_TR^G`.

`analysis.npz` retains both absolute integrated travel from registered time zero
and the relative travel whose origin is the selected stride start. This prevents a
viewer-friendly local origin from destroying the acquisition's absolute travel.

## Markers, forces, and stride

The pipeline strips the C3D subject prefix, uses
`newton.opensim.GAIT2354_VICON_ALIASES`, and synthesizes `V.Sacral` and `Top.Head`
with `GAIT2354_VIRTUAL_MARKERS`. All 35 gait2354 markers are required. The static
trial scales and places the bundled gait2354 model using a mass derived from a
robust stable interval of summed calibration +Z force. Marker relocation round-trip
error is labeled as circular consistency, while a later 0.61–0.65 s
calibration window supplies an adjacent temporal-consistency gate, not an
independent validation trial.

Force-platform data come from ezc3d's force-platform extraction at 2000 Hz. A
20 Hz fourth-order zero-phase low-pass is applied jointly to raw force and moment
about the platform surface center. COP and vertical free torque are then derived
from the filtered wrench so `M = (P - O) × F + T` remains true. The pipeline does
not filter ezc3d's separately derived COP or `Tz` channels. Lab force is already
ground-on-foot GRF. COP is converted from mm to m, free torque from Nmm to Nm,
and unloaded samples are thresholded at 50 N (force and free torque are zeroed;
COP is marked unavailable). A wrench-identity gate and a COP-to-assigned-foot
proximity gate reject boundary artifacts. The lab +X platform loads `calcn_l`;
lab -X loads `calcn_r`. COP forward position receives the same relative belt
displacement as markers.

Contacts are vertical-GRF threshold crossings. The default segment is the first
complete left heel-strike-to-left-heel-strike stride at or after 20 seconds.
Sequential IK warm-starts each frame from the preceding solution. The generated
GRF context MOT and ExternalLoads XML are parsed back and sampled once on the ID
grid; the sanitized wrenches actually passed to ID are archived separately. A
paired constant-speed treadmill/overground ID solve gates frame equivalence, while
FK equivalence is checked independently to numerical precision.

## Main artifacts

The output directory includes:

- `S001_scaled.osim` and its static TRC;
- `trial_ik.mot`, padded `trial_ik_dynamics_context.mot`, legacy adapter context, and `ik_marker_residuals.sto`;
- stride `trial_grf.mot`, `trial_grf_context.mot/.xml`, and the exact
  `trial_grf_id_sampled.mot` used by ID;
- `trial_id.sto` and paired `trial_id_treadmill_frame.sto`;
- coarse `trial_static_optimization.sto` and
  `trial_static_optimization_resampled.sto` (unless skipped);
- `analysis.npz` for project-local replay/inspection;
- `qc_summary.json`, with explicit frame labels and all assumptions.

The `analysis.npz` schema is:

| field | shape | meaning |
|---|---:|---|
| `times` | `[N]` | C3D seconds for the inclusive stride |
| `coords` | `[N,Q]` | OpenSim native coordinate values |
| `coordinate_names` | `[Q]` | coordinate column names |
| `target_markers`, `predicted_markers` | `[N,M,3]` | overground target and FK marker positions [m] |
| `marker_names` | `[M]` | marker column names |
| `grf`, `cop`, `free_torque` | `[N,2,3]` | OpenSim-frame loads; foot order left, right |
| `contact` | `[N,2]` | thresholded left/right contacts |
| `foot_names` | `[2]` | explicit foot ordering |
| `belt_speed` | `[N]` | assumed forward-equivalent belt speed [m/s] |
| `belt_displacement_absolute` | `[N]` | integrated travel from registered time zero [m] |
| `belt_displacement_relative` | `[N]` | integrated travel relative to stride start [m] |
| `com` | `[N,3]` | whole-body OpenSim-frame center of mass [m] |
| `activations` | `[N,U]` | resampled muscle activations; `[N,0]` when skipped |
| `muscle_names` | `[U]` | activation columns |
| `id_coordinates`, `id_speeds`, `id_accelerations` | `[N,Q]` | filtered coordinate state used by inverse dynamics [m or rad and time derivatives] |
| `id_generalized_forces` | `[N,Q]` | inverse-dynamics forces/moments |
| `id_names` | `[Q]` | inverse-dynamics column coordinate names |
| `id_external_wrenches` | `[N,2,9]` | exact sanitized `[F P T]` wrenches used by inverse dynamics |
| `id_external_bodies` | `[2]` | bodies receiving the archived external wrenches |

`qc_summary.json` records SHA-256 hashes of every raw source, axis/frame semantics,
metric anchors and unit assumptions, mass source, static and dynamic marker errors,
coordinate-range violations, treadmill-versus-overground stance heel speeds,
braking/propulsion longitudinal-force signs, true friction ratios, constant-speed
and FK/ID frame-equivalence gates, exact sampled-load differences, dimensionally
normalized pelvis residual force/moment resultants, and per-coordinate/unit Static
Optimization reserve statistics. A kinetics/reserve failure remains visible rather
than being hidden by a successful animation.

## Torque-reconstruction diagnostic

After producing a complete pipeline artifact, verify the inverse-to-forward
dynamics bridge with:

```bash
uv run --extra examples --extra opensim -m projects.gait_c3d.torque_reconstruction
```

The diagnostic first evaluates every frame pointwise using the exact filtered
`q`, `qdot`, and `qddot` retained by inverse dynamics, its generalized forces,
and its archived sanitized external wrenches. It then runs a trimmed open-loop
RK4 rollout with linearly interpolated copies of the measured wrenches and ID
torques. Outputs are written to a sibling directory rather than modifying the
staged source artifact.

This is an engineering reconstruction, not predictive contact dynamics. It
replays measured ground reactions and intentionally reports uncorrected
open-loop drift without a tracking controller or hidden residual forces.

## Stage 1 measured-load engineering diagnostics

The roadmap's non-predictive integration harness is separate from the pointwise
torque-reconstruction artifact:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python   -m projects.gait_c3d.measured_load_diagnostics   --data-dir /home/jo31399/newton-data/gait/processed/trial_101/latest   --output-dir /home/jo31399/newton-data/gait/processed/trial_101/stage1_engineering_measured_load_tracking
```

The canonical run performs 1.0, 0.5, and 0.25 ms convergence, conditionally adds
0.125 ms, schedules 25/50/100 ms restarts from every source frame, compares load
and interpolation variants, and runs bounded non-root tracking with every root
command forced to zero. It archives controller components, work/energy balance,
mass conditioning, range violations, marker/state error, and unavailable source
boundaries. This is an expensive engineering diagnostic and is never labeled
predictive gait. Use repeated `--section` options and
`--restart-start-limit N` only for explicitly incomplete probes.

## Stage 2 prescribed-motion predictive contact

Create a source-bound bilateral contact sidecar from frozen stance-tangent
geometry, then evaluate contact without passing measured loads to the contact
model:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python   -m projects.gait_c3d.predictive_contact init   --model /home/jo31399/newton-data/gait/processed/trial_101/latest/S001_scaled.osim   --analysis /home/jo31399/newton-data/gait/processed/trial_101/latest/analysis.npz   --output /home/jo31399/newton-data/gait/processed/trial_101/stage2_contact_sidecar.json   --body-height 1.695898298375747

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python   -m projects.gait_c3d.predictive_contact evaluate   --data-dir /home/jo31399/newton-data/gait/processed/trial_101/latest   --sidecar /home/jo31399/newton-data/gait/processed/trial_101/stage2_contact_sidecar.json   --output-dir /home/jo31399/newton-data/gait/processed/trial_101/stage2_prescribed_contact
```

The initial sidecar is intentionally uncalibrated. Measured GRF, COP, impulse,
timing, and free moment are validation targets read only after contact evaluation.
A finite prescribed replay is infrastructure, not an FD-1 result; all declared
contact gates and the later held-out calibration must pass.

### Preliminary bounded normal-contact fit

The first calibration stage fits only left-side vertical force with six bounded
parameters. It keeps the right side held out and writes full prescribed QC to a
separate artifact:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python   -m projects.gait_c3d.contact_calibration   --data-dir /home/jo31399/newton-data/gait/processed/trial_101/latest   --sidecar /home/jo31399/newton-data/gait/processed/trial_101/stage2_contact_sidecar.json   --output-dir /home/jo31399/newton-data/gait/processed/trial_101/stage2_normal_contact_calibration   --max-nfev 40   --prescribed-qc-output-dir /home/jo31399/newton-data/gait/processed/trial_101/stage2_prescribed_contact_calibrated
```

This fit adjusts ground height, four bilateral role-shared vertical center
offsets, and log stiffness. It records every evaluation and penalizes training-
side penetration above 20 mm. It does not fit horizontal force, timing, COP, free
moment, or the held-out side, so optimizer convergence cannot be called complete
Stage 2 calibration.

## Official OpenSim RRA reference

OpenSim is the executable residual-reduction reference. The optional official
Python bindings are needed only for `run`; preparation and parsing remain usable
without them. A development environment can install the reference runtime with
`uv pip install opensim` without adding it as a Newton package dependency.

```bash
out=/home/jo31399/newton-data/gait/processed/trial_101/opensim_rra_official_reference_fy4

.venv/bin/python -m projects.gait_c3d.opensim_rra_reference prepare   --data-dir /home/jo31399/newton-data/gait/processed/trial_101/latest   --output-dir "$out" --initial-time 20.60 --final-time 21.66   --tool-name trial101_official_opensim_rra_fy4 --fy-optimal-force 4

.venv/bin/python -m projects.gait_c3d.opensim_rra_reference run   --output-dir "$out"
.venv/bin/python -m projects.gait_c3d.opensim_rra_reference summarize   --output-dir "$out"
```

The adapter generates the pinned gait2354 RRA residual/reserve ForceSet and CMC
tasks, places spatial residuals at the scaled pelvis COM, invokes official
`RRATool`, and archives CMC states, adjusted kinematics/model, pErr, controls,
Actuation results, COM adjustment, and unapplied mass recommendation. Locked MTP
pErr remains diagnostic and is excluded from the production gate. The S001
reference uses FY optimal force 4 N; OpenSim's upstream default is 8 N and both
values remain explicit.

### Import accepted RRA motion for contact

After an official RRA summary passes, publish its adjusted model and q/u/udot on
the corrected contact-target grid. Original ID generalized forces are deliberately
excluded and must be regenerated:

```bash
.venv/bin/python -m projects.gait_c3d.rra_adjusted_contact_input   --rra-reference /home/jo31399/newton-data/gait/processed/trial_101/opensim_rra_official_reference_fy4   --data-dir /home/jo31399/newton-data/gait/processed/trial_101/latest   --output /home/jo31399/newton-data/gait/processed/trial_101/rra_adjusted_contact_input
```

### Official MocoInverse reference

Prepare MocoInverse from the hash-sealed accepted RRA result. The adapter converts
legacy short coordinate labels/degrees to absolute state paths/radians, welds
unsupported MTP joints, follows the pinned official ModelProcessor order, and
preserves sealed failures as reusable failed-guess artifacts:

```bash
rra=/home/jo31399/newton-data/gait/processed/trial_101/opensim_rra_official_reference_fy4
out=/home/jo31399/newton-data/gait/processed/trial_101/opensim_moco_inverse_reference

.venv/bin/python -m projects.gait_c3d.opensim_moco_inverse_reference prepare   "$rra" "$out" --mesh-interval 0.05 --max-iterations 1000
.venv/bin/python -m projects.gait_c3d.opensim_moco_inverse_reference run "$out"
.venv/bin/python -m projects.gait_c3d.opensim_moco_inverse_reference summarize "$out"
```

MocoInverse prescribes the accepted RRA motion. It is a muscle-redundancy
reference, not predictive forward dynamics. Mesh refinement and reserve QC are
required before interpretation.

### Official 3-D Moco contact topology

`projects.gait_c3d.opensim_moco_contact_reference` pins the current official
example3DWalking topology: six spheres per foot, with four on the calcaneus and
two on the toe body. It generates matching official XML and Newton augmentation
specs, MocoContactTrackingGoal groups with toe alternative frames, and independent
COP/free-moment validation. Measured ExternalLoads are reference-only and are
never added to the predictive contact model.

## Official OpenSim to Newton contact parity

Validate the Newton-native SmoothSphereHalfSpace port against official OpenSim
on the same sidecar, state, frame, and all stride samples:

```bash
.venv/bin/python -m projects.gait_c3d.opensim_contact_parity   --sidecar /home/jo31399/newton-data/gait/processed/trial_101/stage2_initial_contact_sidecar.json   --output-dir /home/jo31399/newton-data/gait/processed/trial_101/opensim_newton_contact_parity   --full-frames --device cpu
```

The mixed component gate is `abs(delta) <= atol + 1e-4 * vector_scale`, with
`atol=1e-3 N` for force and `1e-4 N*m` for torque. The comparison reads only q
and qdot; measured loads are not inputs. Pre-existing model contact is rejected
so both runtimes evaluate exactly the sidecar elements.

## Stage 4 residual and model sensitivity

Run the frozen preliminary timing and inertial audit without accepting a timing
change:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python   -m projects.gait_c3d.residual_sensitivity   --data-dir /home/jo31399/newton-data/gait/processed/trial_101/latest   --output-dir /home/jo31399/newton-data/gait/processed/trial_101/stage4_residual_sensitivity_preliminary
```

The artifact evaluates every integer wrench lag from -20 to +20 ms on one common
non-extrapolated interval in one batched inverse-dynamics call. It archives all
root residuals, vector-resultant RMS/peak normalization, qualified-but-not-
accepted lags, and every body mass, COM, and inertia. A lower residual at one lag
is sensitivity evidence only; timing remains unchanged until an independent
synchronization mechanism and the full residual/model closure stage are accepted.

## Visualization

After the pipeline completes, launch the exact-FK replay with measured markers,
marker residuals, activation-colored muscles, COM trail, bilateral COP/GRF, and a
pelvis-following overground camera:

```bash
uv run --extra examples --extra opensim -m projects.gait_c3d.viewer \
  --download-geometry
```

Add `--show-treadmill-ghost` to render a laterally offset gray skeleton in the
original treadmill frame while the primary skeleton advances overground. The
prepared non-interactive previews are kept with the private derived artifacts:

```text
visualization/flagship_c3d_overground.png
visualization/flagship_c3d_overground.gif
visualization/flagship_c3d_treadmill_comparison.gif
```

The virtual-origin sign follows Jung and Lee, *Treadmill-to-Overground Mapping of
Marker Trajectory for Treadmill-Based Continuous Gait Analysis*, Sensors 21(3),
786 (2021), [doi:10.3390/s21030786](https://doi.org/10.3390/s21030786). Their
published method estimates belt displacement from a marker chain. This dataset
has no belt marker chain, so the implementation uses the supplied registered belt
metric and records that adaptation explicitly in QC.

## Exact human-shoe integration

The pipeline also emits a padded mapped-IK context so the existing exact
human-shoe replay can identify a complete right stance on stationary overground.
Prepare and run that adapter with:

```bash
uv run --extra examples --extra opensim -m projects.gait_c3d.human_shoe \
  --replay-dt 0.001
```

This creates `S001_scaled_with_shoe_contacts.osim`, an external experiment JSON,
contact windows, replay CSV/JSON, and `human_shoe_integration_qc.json`. Visualize
the mapped shoe replay with:

```bash
uv run --extra examples -m projects.human_shoe.replay_viewer \
  --experiment /home/jo31399/newton-data/gait/processed/trial_101/latest/human_shoe_overground_experiment.json \
  --replay-dt 0.001
```

The shoe replay uses exact OpenSim FK with the mapped motion; it no longer treats
treadmill foot motion as motion over stationary ground. The baseline gait2354 shoe
anchors are reused on the scaled S001 model without independent subject/shoe
registration. `human_shoe_integration_qc.json` applies explicit 10% peak-force and
5% impulse-error gates; the current replay fails both overall validation because
peak error is 12.6% and impulse error is 71.1%. It remains a prescribed-kinematics
load replay with no shoe-force feedback into the human trajectory.

## Reproducibility and dependency note

C3D force-platform decoding uses [ezc3d](https://github.com/pyomeca/ezc3d), while
all model, scaling, marker IK, ExternalLoads, inverse-dynamics, forward-kinematics,
and muscle optimization operations use the public `newton.opensim` surface. The
cache validates its schema, extraction configuration, dynamic-C3D hash, and all
belt-source hashes; QC records every source hash. Outputs carry original C3D seconds rather than a renumbered time base, and
selection/filtering rules are fixed
by CLI values recorded in QC.
