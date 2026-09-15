# Independent impedance gains and shoe-material perturbations

This experiment follows the [movement-intent response workflow](RESPONSE.md).
It keeps nominal movement and reduced-rig inverse-dynamics assistance frozen,
while varying one impedance coefficient at a time. Shoe material is a separate
whole-episode environmental perturbation. The original RL environment, policy
weights, reward, geometry and recorded movement are not modified.

## Run and inspect

Use the existing worktree environment and retained reference/shoe inputs:

```bash
uv run --no-sync -m projects.impedance_instron sensitivity --device cuda:0
```

Open `outputs/impedance_instron/simple/sensitivity/report.html` in a browser.
The overview is figure-first, organized around material effects, independent gain
changes, and position/velocity recovery. Gray is the baseline; colored curves show
variants. Separate material force-difference plots reveal changes that can hide in
overlapping absolute GRF curves. Push/dwell shading, engineering bands and true
terminal markers explain the recovery screen. Terminal scatter plots are labeled
as two-component projections, not the full all-channel recovery test.

Exact parameter/result tables are collapsed supporting detail. The index links to
per-case raw plots in `pages/`; standalone SVG figures are in `figures/`. No server
or CDN is needed. Full numeric records and terminal states are saved as NPZ files in
`cases/`. `summary.json` contains paired motion, recovery, force, compression,
work and safety metrics. `manifest.json` freezes the experiment inputs/settings.
Do not use an existing nonempty output directory for another run.

A five-case nominal-controller/original-material smoke run excludes all sweeps:

```bash
uv run --no-sync -m projects.impedance_instron sensitivity \
  --device cuda:0 --gain-multipliers --modulus-multipliers --relaxation-multipliers \
  --output outputs/impedance_instron/simple/sensitivity_smoke
```

To compare an actual alternative material artifact, supply it explicitly. The
entire non-material physical description must match, including unknown physical
fields. A different mesh, column bed, fixture or coordinate frame is not a
material perturbation.

```bash
uv run --no-sync -m projects.impedance_instron sensitivity \
  --device cuda:0 --material /path/to/same-geometry-material.json \
  --output outputs/impedance_instron/simple/sensitivity_material
```

Repeat `--material` for multiple alternatives. Imported artifacts are not newly
validated by this experiment. Their source qualification is distinct from
validating transfer to this rig or gait trial.

## Redraw figures without rerunning the experiment

```bash
uv run --no-sync -m projects.impedance_instron report \
  outputs/impedance_instron/simple/sensitivity
```

Use `--overview-only` for a quick sensitivity-overview update that retains its
existing case pages. The command also accepts saved `response` suites. It checks
saved input and trace hashes, and writes only HTML/SVG presentation assets plus
`render_metadata.json`. It does not change metrics, snapshots, manifests or
recorded simulation-source fingerprints. Old results keep their original
qualification even when the current renderer or physics source has changed.

## What changes independently

The default controller is the separated `intent` law:

```text
F_leg = F_ID + K_leg * (Lref - L) + B_leg * (Lrefdot - Ldot)
tau = tau_ID + K_pitch * wrap(pitchref - pitch) + B_pitch * (pitchrefdot - pitchdot)
```

The nominal load stays the same at the reference position and velocity, even when
a gain changes. It remains explicit runtime assistance from recorded reduced-model
ID. This is not an anatomical or unassisted predictive controller.

There are nine default controller settings: nominal, plus half/double each of:

- Leg stiffness [N/m].
- Foot-pitch stiffness [N·m/rad].
- Leg damping [N·s/m].
- Foot-pitch damping [N·m·s/rad].

Only one coefficient changes in a setting. The others retain the frozen nominal
values. Duplicate and unit multipliers are omitted. If a nominal damper is zero,
multiplying it cannot create damping, so unchanged duplicate settings are omitted.
The default counts below assume the retained nonzero nominal damping values. In particular, changing stiffness does NOT recompute damping. These are
one-at-a-time sensitivity tests, not a joint optimizer or proof that interactions
are negligible. Use `--modes intent equilibrium` to add the old equilibrium-law
comparison without changing the existing `response` command.

## Material as a perturbation

The default material family contains the original artifact plus four synthetic
variants:

| Variant family | Factors | Changed quantity |
|---|---|---|
| Modulus | 0.75, 1.25 | Both instantaneous Ogden-Hill shear moduli |
| Relaxation time | 0.5, 2.0 | Maxwell relaxation time |

Modulus scaling preserves the term exponents, relative term weights, equilibrium
fraction, Poisson parameter and Maxwell relaxation time. It changes the material
law's amplitude, not merely the final shoe force. The existing neighbor coupling
is derived from equilibrium shear modulus and column thickness, so it changes
consistently with modulus. Reported derived quantities are recomputed, rather than
leaving stale fitted-modulus or coupling numbers in the artifact.

Relaxation-time scaling changes only the material Maxwell time. It does not
change the controller damping or `RigConfig.outer_relaxation_s`, which is the
numerical passive-surround relaxation setting. A longer Maxwell time is not
universally equivalent to greater damping; the response depends on loading rate
and history.

These synthetic artifacts are **declared parameter-sensitivity hypotheses**.
They are not newly calibrated shoes, confidence intervals, or measured material
uncertainty. Inherited fitted-success claims are removed from synthetic artifacts;
the original artifact is retained unchanged as the source snapshot. The shared
constitutive law is not changed or refitted.

All material parameters remain constant during a rollout, with fresh material
history at reset. Instantaneously changing material under a loaded foot would
require a separate history/energy-transfer model; that is not implemented here.
Masses, shoe geometry, mounting, ground, friction settings, initial body state,
measured targets and ID loads stay fixed across material comparisons. Fixed
controller gains do not mean fixed actuator force: feedback responds to the
material-induced motion change. These are closed-loop shoe/controller sensitivity
results, not isolated prescribed-deformation material tests.

## Two different pairing rules

1. **Permanent material sensitivity:** compare an unpushed changed-material run
   with the unpushed original-material run at the same controller setting.
   This measures material-induced motion, load and work changes.
2. **Transient push recovery:** compare a pushed run with the unpushed run using
   the SAME controller and SAME material. This measures response to the push
   without mixing in the material's change to nominal motion.

The pulse is applied to the upper body in four separate directions: forward,
backward, upward and downward. Each has 150 N peak and a 40 ms raised-cosine
profile, giving a signed 3 N·s impulse along the selected axis. Exact
interval-average forces preserve that impulse on the solver clock.

No pulse or material result is compared with a different controller baseline.
An invalid baseline makes the pair unqualified, even if the disturbed run itself
passes its mechanical checks. Invalid/failed cases remain visible.

## Default coverage: 97 cases

- All nine controllers on all five materials without a push: 45 cases.
- Four push directions for all controllers on the original material: 36 cases.
- Four push directions for the four changed materials at the nominal controller:
  16 cases.

This staged design tests material-only response across gains and push response
across materials without pretending to test every interaction. Use
`--full-factorial` for all 225 controller/material/push combinations. Counts grow
when adding controller modes, multipliers or imported materials. The old
24-case ground-height response experiment remains a separate unchanged workflow.

## What "returned" means

The recovery screen compares the complete planar body motion with its matching
unpushed baseline:

- Foot and upper-body X/Z positions and velocities.
- Foot pitch and angular velocity.
- Leg length and length rate (redundant but useful actuator coordinates).

Default engineering tolerances are 1 mm for positions/length, 1 mrad for pitch,
0.01 m/s for linear/length rates, and 0.01 rad/s for pitch rate. Both position and
velocity must remain inside their bands through the final 50 ms dwell AND at the
true final state. These are declared engineering screening values, not identified
human repeatability thresholds.

The default push begins at measured contact phase 0.25. On the retained 0.375 s
reference this leaves about 0.216 s after the pulse. The declared minimum observed
window is 0.15 s; it is a finite-window requirement, not proof of sufficient time
for every coupled mode to settle. No repeated stride, terminal hold or extra
measured motion is invented to extend it.

The report distinguishes:

- `unperturbed_baseline`: no perturbation and no recovery test for the reference case.
- `returned_within_window`: the declared sampled position/velocity screen passed.
- `not_returned_within_window`: enough observation time, but the screen did not pass.
- `insufficient_window`: not enough observed post-pulse time for the declared screen.
- `invalid_pair`: missing/nonfinite data, incompatible clocks/endpoints, or failed
  physical qualification prevents a valid comparison.
- `persistent_material_change`: the changed material remains present; it is not
  a removed transient from which recovery to the original material is expected.

True final velocities are computed from the terminal Newton body state, not the
last pre-integration trace sample. The traces and terminal times must agree between
pairs. No interpolation or phase alignment hides a late contact or a partial run.
A sampled return of body motion does not establish recovery of foam, Maxwell or
friction history, asymptotic stability, passivity, or general robustness.

## Work, contact and qualification

The reports retain contact force/impulse, compression, clearance and limits beside
motion and recovery. A small displacement response is not automatically desirable
if it costs excessive force or work. Material affects a coupled system; changing
one coefficient can improve one output while worsening another.

Delivered body work is separate from idealized controller source work, damping
work and virtual spring storage. Source-work totals depend on controller
realization and cannot establish motor/electrical/metabolic efficiency or hardware
energy recovery. The contact power is carrier wrench power, not a complete foam
thermodynamic ledger.

The upper mass is a pelvis-motion proxy, the pitch actuator reacts against the
world, and the upper shoe support is idealized. The recorded and modeled shoes
differ. Passing the existing mechanical checks does not validate anatomical
registration, the material perturbation range, or same-shoe gait prediction.

## Frozen-input replay

```bash
uv run --no-sync outputs/impedance_instron/simple/sensitivity/replay.py \
  outputs/impedance_instron/simple/sensitivity_replay --device cuda:0
```

Replay reads the exact saved material snapshots rather than regenerating or
refitting variants. It verifies reference/material hashes, the configuration seal,
manifest/summary consistency and source fingerprints. Tampered physical inputs or
settings cannot be accepted by a source-update flag. Explicit
`--allow-physics-update` permits a changed implementation as a new experiment;
all cases are evaluated again and old scores/validity are not inherited.

The files record the native runtime and device. They are not a complete archived
software environment or a promise of bitwise replay across devices or versions.

## Initial retained-input results

The default 97-case CUDA suite completed with all 97 existing physical checks and
paired comparisons valid. There were nine original-material quiet baselines,
36 permanent-material comparisons and 52 directional push comparisons.
None of the 52 pushes passed the declared full planar body position/velocity
return screen within the 0.2155 s post-pulse window. This does not establish
instability or prove that recovery would never occur. It is a fixed-clock,
finite-window screen with demanding position AND velocity bands; the controller
has no direct upper-body forward-position target.

At the nominal controller, the quiet material comparisons gave:

| Material change | Peak compression [mm] | Peak vertical GRF [N] | Peak paired pelvis-height change [mm] |
|---|---:|---:|---:|
| Original | 26.366 | 2339.60 | 0.000 |
| Both shear moduli x0.75 | 28.448 | 2352.91 | 3.210 |
| Both shear moduli x1.25 | 24.800 | 2321.21 | 2.673 |
| Maxwell time x0.5 | 26.713 | 2345.20 | 0.814 |
| Maxwell time x2.0 | 25.818 | 2331.46 | 1.013 |

These are closed-loop responses to synthetic hypotheses, not new material fits.
The softer modulus increased compression as expected, but did not lower peak GRF
in this particular moving rig. Do not replace the actual response measurement
with a general "softer means lower force" assumption.

Independent damping also exposed a tradeoff. For the original-material forward
push, doubling only leg damping reduced peak pelvis-height deviation from
1.648 to 1.261 mm, while terminal forward displacement increased from 9.890 to
10.012 mm. Nominal terminal forward velocity differed from its unpushed baseline
by 0.04337 m/s. A smaller vertical excursion is not complete recovery.

A fresh 97-case replay used the exact frozen material snapshots and settings.
All tracking scores, recovery metrics and physical/pair classifications matched
exactly on this CUDA run, with no inherited results. This is observed same-build
replay, not a general bitwise guarantee.

The selected combined regression command passed 173 tests. Repository-wide and
changed-file pre-commit checks passed. The original response controller, material
law and its saved source fingerprints were unchanged by this additive workflow.
Local evidence is in `outputs/impedance_instron/simple/sensitivity_validation.json`
and the suite's manifests, summaries and numeric traces.

## Tests

```bash
uv run --no-sync -m unittest \
  newton.tests.test_impedance_material_variants \
  newton.tests.test_impedance_recovery \
  newton.tests.test_impedance_sensitivity \
  newton.tests.test_impedance_sensitivity_report \
  newton.tests.test_impedance_sensitivity_cli
```
