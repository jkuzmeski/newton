# Impedance Instron worktree summary

This records the full worktree progression, not only the latest GPU change.
The existing branch history runs from `ab11605e` through `2b5c8902`; the additional
source changes include the retained legacy improvements, the new two-stiffness
workflow, shared shoe mechanics, contact/geometry repairs and resident calibration.

**Current entry point:** `python -m projects.impedance_instron`.
**Current viewer:** `python -m newton.examples impedance_stiffness`.
The six-action experiment is historical. Its scores are not comparable to the
active two-stiffness tracking loss. The active policy's old **1.1056** loss also
predates the contact repair and is **not current physical qualification**.

## 1. Earlier branch work retained

| Commit | Contribution |
|---|---|
| `ab11605e` | Runnable measured-running impedance rig, portable inputs, viewer and mechanical comparison reports. |
| `e8ff9e52` | Calibrated heel-cluster pitch at a fixed mechanical ankle, explicit release/work accounting and shoe-side conventions. |
| `f03ac31a` | Separate unilateral ground support from signed shear transfer; fit Maxwell relaxation time and include peak/loop errors. |
| `b6d2c896` | Identify the whole shoe with a relaxing passive surround, using Warp forward kernels instead of a mismatched fixed outer bed. |
| `aa385d3e` | Pin neighbor coupling to material modulus and thickness, set effective Poisson ratio to zero and remove the hidden outer support. |
| `ffc689c2` | Add the second Ogden–Hill term to span the two fixture strain ranges; keep one shared material and unchanged gates. |
| `8a48e1ac` | Equilibrium-point leg/ankle controls, explicit actuator power accounting and captured CUDA frame replay. |
| `c0fc1523` | Isolated world-batched foundations, per-world materials and fixed-order reductions instead of atomic totals. |
| `bfaaa96e` | Self-contained CMA-ES command search, three-tier task objective and detailed measured/commanded/achieved overlays. |
| `2b5c8902` | Closed-loop PPO on the batched rig, frozen normalization/commands, optional ONNX export and offline training dashboard. |

These steps explain the implementation history. They do not make the retired
momentum/work objective or its prescribed-pitch defaults part of the active rig.
Historical timing measurements also belong to their recorded workloads: CUDA
frame replay reduced the earlier single rollout from 6.7 s to 0.517 s; batched
foundation reduction reduced a substep from 0.170 to 0.077 ms at 64 worlds and
0.419 to 0.179 ms at 256 worlds. These are not the resident-calibration benchmark.

### Legacy improvements included alongside the replacement

The earlier working changes are retained, rather than discarded because the
controller contract later changed:

- `env.py` supports optional six-action leg/ankle residual control. It records
  achieved pitch, charges both actuators' source work, and restores ankle commands
  from checkpoints. The three-action prescribed-pitch mode remains available.
- Residual commands have per-frame slew limits and substep ramps. Equilibrium and
  stiffness derivatives reflect the applied, clipped commands. Observations carry
  the applied residual and distinguish episode phase from detected-contact phase.
  The legacy dense momentum reward and its criterion use the same touchdown datum.
- `train.py` retains complete substep waveforms by default, stamps the actual
  per-world material identity, and writes separate frozen-evaluation records and
  archives per material. Evaluation does not silently overwrite another shoe's
  diagnostics or reconstruct an ankle from a stale command-line flag.
- `dashboard.py` groups evaluations by run/material, filters either identity,
  overlays selected waveforms, marks coarse samples, and warns about stale material
  archives or differing measured references. It uses stdlib HTTP and inline SVG.
- `summary.py` provides an offline illustrated report and machine-readable figure
  data, with archive/material provenance and trace-resolution diagnostics. It now
  defaults to [LEGACY_REPORT.md](LEGACY_REPORT.md); unrelated/current documents
  receive no historical figures or numerical tables. Archived-run checks require
  matching provenance. It is not the active controller's report generator.
- `profile.py` adds sealed v3 pelvis-marker data on its own optical clock without
  breaking older profiles. `variability.py` measures across-stride force, timing,
  heel pitch and optical pelvis spread, and distinguishes measured pelvis motion
  from the force-integrated COM surrogate. The retained active input contains
  83 stances from a steady commanded 3 m/s run.
- `mcclough.py` retains separately labeled published-foam comparison materials and
  regression checks. Its compression fits and anchored loss comparison are not
  identified intact-shoe replacements; its shear, rate and structural limitations
  remain explicit. These materials do not replace the default shoe.
- Additional regression coverage protects ankle work, residual rates, contact
  clocks, material identity, graph replay, reports, profile data and variability.
  Legacy commands warn on invocation and point to the replacement workflow.

Legacy contact-aligned charts and rewards must not be read as the active scoring
clock. Old source APIs remain for one deprecation cycle. The local archive locator
is `outputs/impedance_instron/LEGACY_ARCHIVE.json`.

## 2. Active two-stiffness workflow

`simple/reference.py`, `simple/rig.py`, `simple/policy.py` and `simple/report.py`
implement a separate, small workflow. They do not import the retired environment,
optimizer, objective or trainer. `simple/example.py` supplies the registered viewer
and headless final-state check; the repository example gallery includes its image.

- The only online controls are absolute log leg stiffness [N/m] and ankle
  rotational stiffness [N·m/rad]. A rate-bounded C2 quintic frame interpolation
  makes stiffness and its derivative continuous.
- Offline inverse dynamics constructs equilibrium length/angle schedules from
  optical motion, measured platform wrench and declared rig masses/inertia.
  It solves `B*e_dot + K*e = inverse_load`, including moving-equilibrium damping.
  Those inverse loads are not runtime force/torque feedforward.
- The reference freezes schedules, initial state, processing assumptions, source
  identities and scales. Training/evaluation do not rebuild inverse dynamics.
  Missing input files fail explicitly.
- Reward is only the fixed-time integral of normalized pelvis-height and wrapped
  ground-relative foot-pitch squared error. The upper lump represents the measured
  pelvis centroid, not true whole-body COM. Pitch is not anatomical ankle flexion.
  Flight samples remain scored; contact detection never resets the scoring clock.
- GRF, source work, damper work, compression and safety are evaluation channels,
  not momentum/work/force reward tiers. Only valid deterministic evaluations can
  become `best.pt`; `last.pt` is separate. Finite invalid trajectories can still
  supply training data without being presented as qualified best policies.
- Checkpoints restore the complete experiment, including observation scales and
  rig/foundation source fingerprints. Material-only replacement requires identical
  geometry. Source-update consent permits re-evaluation, not silent inheritance of
  old scores, refitting, or relaxed geometry/reference checks.

Defaults are 120 frames/s, 64 substeps/frame and CUDA graph replay where available.
Nominal stiffnesses are 12000 N/m and 4000 N·m/rad. The allowed ranges are
3000–48000 N/m and 1000–16000 N·m/rad. Fixed damping ratios are 0.25 and 0.5;
damping follows `B_leg = 2*zeta_leg*sqrt(K_leg*m_upper)` and
`B_ankle = 2*zeta_ankle*sqrt(K_ankle*I_foot)`. The bilateral leg has a signed force
limit; the ankle reacts against the world, not a modeled shank.

### Policy result: preserve the before/after distinction

The historical 100-iteration, 32-world run selected iteration 100. Before the
contact repair, GPU loss was 1.105597 against nominal 6.509888. Those numbers and
the earlier replay/test record are labeled historical in [REPORT.md](REPORT.md).

Re-evaluating the unchanged policy after the repair, and again after consolidation,
gave the same recorded metrics:

| Corrected-contact replay, 0.375 s | Result |
|---|---:|
| Tracking loss | 7.468390 |
| Pelvis-height RMS error | 20.574 mm |
| Foot-pitch RMS error | 0.6024 degrees |
| Peak vertical GRF | 2617.625 N |
| Existing mechanical safety flags | None |

This is a finite corrected-physics replay, not improved tracking or retraining.
The report marks `checkpoint_scores_applicable: false`. No weights or foam
parameters were adjusted to hide the loss increase. A new source fingerprint
requires a new evaluation even when the source change is intended to preserve
numerical behavior.

## 3. Shared mechanics and repaired geometry

### One law, several boundary/dataflow adapters

`projects.digital_shoe.material` owns the two-term Ogden–Hill pressure and Maxwell
recurrence. NumPy fitting and compiled Warp forward/autodiff use the same Python
expression bodies; host fitting retains float64 precision.
`digital_shoe.contact` owns unilateral reaction, material-pinned symmetric neighbor
coupling, passive balance, anchored bristle history, contact points and wrench
mapping. Mutable runtime arrays, tape-safe histories and periodic fitting are
separate storage/scheduling adapters, not separate active laws.

The differentiable foundation now uses the forward bristle law and records the
forces actually applied. The old raw `foundation_apply_diff` kernel retains its
historical smooth-friction behavior only as a deprecated compatibility surface.
The smoothing option and fixture-subset Laplacian input are also deprecated.
Active fitting uses `Trial.surround`; old public material/import names remain.
Fixed-order cycle-force summation removes atomic scheduling noise in the forward
objective. It does not certify all reverse-mode reductions or optimizer paths.

### Passive flight and external ground wrench

Shared rendering reconstructs endpoints from scalar compression and the current
carrier pose. Neither the `z_free` pressure reference nor friction bristle anchors
are physical material endpoints. Rendering is read-only and cannot stretch a
column beyond rest length except endpoint-rounding tolerance.

The corrected final frame has zero passive columns incorrectly pinned near the
floor, instead of 299. Passive compression and external ground force are zero in
final flight. Across the captured frames, maximum passive length/rest ratio was
1.0000064, within float32 endpoint-rounding tolerance.

Carried-shoe consumers explicitly declare the ground plane. Nonnegative local
external pressure sets friction capacity, COP and the complete carrier force and
moment at the contact surface. Signed neighbor transfer stays a separate
`column_force` diagnostic; clamping it would invent support. Generic bench callers
retain their top-coordinate/load-transfer convention unless they opt into the
external-plane path.

### Full-bed replay and one intrinsic shoe for both fixtures

Portable Instron replay now retains all 910 columns, as identification does.
Full foot drives 611 and leaves 299 passive; rearfoot drives 62 and leaves 848
passive. Restoring the surround changed full-foot replay peak from 1876.910 N to
2371.010 N, against exported reference 2361.539 N. Peak discrepancy fell from
18.5% to 0.40%. Warm waveform NRMSE/exported peak is 0.0052% for full foot and
0.0375% for rearfoot. The 32/64-sweep checks changed the waveforms by less than
0.009% and 0.083% of peak, respectively.

Rearfoot and full foot now use array-identical intrinsic rest endpoints and
lengths. Legacy zero-bottom rearfoot records are rebased by shifting both anchor
and free top, not by flattening the shoe or changing rest lengths. The native
export geometry uses the same datum, and the punch visual sits over its heel
patch. Full-foot forces were unchanged; maximum rearfoot frame-force change was
0.0001831 N. Rearfoot peak remains 1001.942 N.

These are geometry and replay-consistency repairs, not new experimental gates,
new fitting, or gap-aware flat-punch/upper-last collision solves.

## 4. Preserve the two-term fit; move forward calibration onto the GPU

The existing material has two Ogden–Hill terms plus one Maxwell branch: six fitted
constants. Neighbor coupling remains derived from equilibrium shear modulus times
local thickness; effective Poisson ratio remains zero and the outer support bond
default remains zero. The second term and default calibration are retained.

The earlier two-term calibration reduced training loss from 0.006414 to 0.001811
and passed all six same-protocol held-out gates, versus two with one term. That
is a local calibration result, not broad material validation. The fixtures still
prefer about 3.8-fold different equilibrium moduli when fitted separately. The
first exponent is weakly identified, and equilibrium fraction/relaxation time
remain inseparable at the available single loading rate. Single-start fitting
remains the default; multi-start remains opt-in.

`digital_shoe.CalibrationWorkspace` now retains geometry, compression, Maxwell
fields and scratch buffers on the device. `core.predict()` caches work per
trial/device/thread/stream, detects input edits, and releases buffers with the
trial. Only small parameter/settings blocks upload per evaluation. Full-field
blending and convergence reductions run in Warp; CUDA graphs replay 25-sweep
chunks. CPU and remainder sweeps use the same kernels eagerly.

The bounded SciPy optimizer, objective, parameter meanings, warm-start policy,
sweep caps, tolerances and stopping decisions are unchanged. Scalar convergence
checks and the final force curve return to the host; large fields do not.

### Paired RTX A6000 benchmark

Training cycles 90–98, 501 frames × 910 columns per fixture; five parameter
candidates per sequence and five warm repeats. Values are warm median sequence
time divided by five. Setup/compilation are excluded and first-call timings are
stored separately. Environment: Warp 1.17.0.dev20260807, NumPy 2.5.0, SciPy 1.17.1.

| Work | Saved implementation | Resident | Speedup |
|---|---:|---:|---:|
| Rearfoot forward evaluation | 275.2 ms | 110.3 ms | 2.49× |
| Full-foot forward evaluation | 148.9 ms | 57.0 ms | 2.61× |
| Short unchanged SciPy fit | 11.89 s | 4.75 s | 2.50× |

The short fit requested `max_nfev=5` and performed 56 forward evaluations per trial
in each backend, including callback/selection work. It returned identical six
fitted values and mean-square residual 0.0017966006912530895. Training candidate
curves and held-out cycles 99–100 were bitwise identical to the saved implementation.
The diagnostic fit candidate was not installed as a new default artifact.

At the baseline parameter, rearfoot/full-foot D2H payload fell from about 29.18 MB
to 2376/2188 bytes, more than 99.99% less. Python kernel launches fell from 2019/797
to 25/25, with 77/30 graph replays. Each graph still performs the same physical
sweeps. `.numpy()` timing includes synchronization, not only transfer time.

## 5. Qualification limits remain explicit

- **Stopping estimate:** the resident path deliberately preserves the legacy
  interval-ratio estimate. It is not a certified remaining-error bound. Matching
  the old stopping decisions does not repair or qualify their convergence logic.
- **Differentiable fitting:** shared forward/autodiff laws and gradient checks do
  not solve the separate five-versus-six-parameter fitting mismatch. This work
  does not replace SciPy with a new differentiable optimizer.
- **Upper contact:** the live rig still uses idealized rigid backing over the
  projected fixture footprint, with one-sided passive retention. It does not solve
  contact against the displayed rigid last. Retained fixture clearances have
  median 4.902 mm and maximum 30.474 mm; point-clamped seating also disagrees with
  the rigid mesh. A coherent upper registration/contact model remains unresolved.
  Changing rest lengths or subtracting one gap from one force cannot fix it.
- **Scientific scope:** pelvis centroid is not COM; foot pitch is not shank-relative
  ankle angle. Reduced-model inverse-dynamics residuals are substantial, but do not
  prove that the two rewarded targets are unreachable. Recording and modeled shoes
  differ. This is not same-shoe, physiological-stiffness, metabolic or material
  transfer validation. Optical preprocessing and shoe-side registration assumptions
  remain disclosed.
- **Numerics:** contact gradients are piecewise and checked away from transitions.
  Exact parity applies to the tested build and workloads, not all devices, software
  versions or reverse-mode optimizer trajectories. Replay convergence and mechanical
  safety do not certify an anatomical upper or predictive human performance.

## 6. Reproduce and inspect

Run from the repository root in its existing environment. The required profile,
variability and shoe files are local under `outputs/impedance_instron/inputs/`.
They are not supplied by a source checkout. Preparing/training creates a new
experiment; use explicit output paths if keeping earlier runs.

```bash
# Active workflow.
uv run --no-sync -m projects.impedance_instron prepare
uv run --no-sync -m projects.impedance_instron run
uv run --no-sync -m projects.impedance_instron train --iterations 100 --worlds 32 \
  --output outputs/impedance_instron/simple/training_current

# Re-evaluate the retained pre-repair policy with explicit source-update consent.
uv run --no-sync -m projects.impedance_instron evaluate \
  outputs/impedance_instron/simple/training/best.pt \
  --allow-physics-update --output outputs/impedance_instron/simple/physics_updated

# View the actual rig. Use --viewer null --test for a headless check.
uv run --no-sync -m newton.examples impedance_stiffness --viewer gl --render-fps 30

# Same full shoe, two bench fixtures.
uv run --no-sync -m projects.digital_shoe.showcase \
  --mode instron --fixture fullfoot_last --viewer gl
uv run --no-sync -m projects.digital_shoe.showcase \
  --mode instron --fixture rearfoot_punch --viewer gl

# Normal calibration automatically uses the resident forward path.
uv run --no-sync -m projects.digital_instron_v2.workflow \
  --manifest DigitalInstron/manifest_v2.json

# Paired execution benchmark; requires the retained local pre-change source.
uv run --no-sync -m projects.digital_instron_v2.profile_calibration \
  --baseline-core outputs/impedance_instron/gpu_forward_calibration/before/projects/digital_instron_v2/core.py \
  --repeats 5 --fit-evaluations 5 \
  --output outputs/impedance_instron/gpu_forward_calibration/profile_final
```

For strict replay of a checkpoint from the current source, omit
`--allow-physics-update`. Add `--material /path/to/same-geometry-material.json`
to `evaluate` for an explicit material-only comparison. The viewer also accepts
`--checkpoint` and `--allow-physics-update`. Headless fixture audits use
`--viewer null --num-frames 180 --test`. Active reports are offline HTML plus
full-resolution NPZ traces; display decimation does not filter the saved data.

### Recorded validation evidence

These are completed stage-specific runs, not one combined unique-test count or
a claim about a later merge-validation run:

| Stage | Recorded checks | Local evidence under `outputs/impedance_instron/` |
|---|---|---|
| Passive/contact repair | 110 tests; CPU/CUDA, eager/graph and full 45-frame policy replay | `passive_attachment_fix/FIXES.md`, `final_tests.log`, `corrected/summary.json` |
| Shared-law/full-bed consolidation | 163 targeted tests in disjoint groups; both 180-frame fixture replays and other mechanical examples | `consolidation/RESULTS.md`, fixture replay JSON, `impedance/summary.json` |
| Rearfoot datum repair | 5 failing-before/passing-after tests; 9 focused plus 32 existing tests; both fixture replays | `rearfoot_same_shoe/RESULTS.md`, `comparison.json` |
| Resident calibration | 62 targeted tests; final 14-test workspace/cache rerun; training and held-out parity | `gpu_forward_calibration/RESULTS.md`, `profile_final/summary.json`, `heldout_parity.json` |

The recorded scoped hooks passed with `uv-lock` deliberately skipped to preserve
the environment. API generation produced no generated Newton-page changes.
No required dependency was added. Frozen input/artifact/checkpoint hashes were
checked around the repairs and GPU migration.

The generated reports, recordings, fitted artifacts, checkpoints, source snapshots
and participant/footwear inputs remain local ignored data. This source summary
links to that evidence; it does not distribute it or claim it will be committed.
See [README.md](README.md), [REPORT.md](REPORT.md),
[shared mechanics](../digital_shoe/CONSOLIDATION.md), and
[Digital Instron](../digital_instron_v2/README.md) for the detailed contracts.

## Source-integration validation

Before committing this worktree, the full project-specific suite sweep ran 590
checks: 530 passed and 60 skipped, with no remaining failures. The groups were
456 impedance checks (60 skipped), 47 Digital Shoe checks, and 87 Digital Instron
checks. The skips belong to optional or legacy input-dependent checks; they are
not counted as passes.

The broad run first exposed a legacy reporting mismatch. The repair separates
`LEGACY_REPORT.md` and archived artifacts from the current report; numerical
replay tolerances were not relaxed. Its archived smoke report renders all ten
figures and reproduces the original peak-timing pairs.

`uvx pre-commit run -a` passed, including the lockfile check. API generation and
the Towncrier draft completed. The current 45-frame OpenGL headless example passed
and regenerated the registered image; its checksum and restricted derivative
status are recorded in `ASSET_PROVENANCE.md`.

Detailed validation logs remain local under
`outputs/impedance_instron/merge_main/`. These are source-integration checks, not
new material identification, policy retraining or anatomical validation.
