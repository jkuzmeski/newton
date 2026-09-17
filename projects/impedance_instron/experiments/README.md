# Frozen-controller shoe experiments

This workflow changes shoes, not the fitted controller. It keeps all four
equilibrium splines, fixed gains, initial leg state, and timing unchanged. A
frozen feedback law can still produce different forces in a different shoe.

The current model covers one planar leg and a 0.36 s stance. It is not a
full-body runner or a model of controller adaptation. Motion termination is a
valid experiment outcome; numerical failure is reported separately.

See [the experiment plan](../EXPERIMENT_PLAN.md) for the 31 primary conditions:
baseline, 24 material variants, four thickness variants, a uniform rectangular
bed, and a flat original-footprint companion. The six geometry variants also
have separately labeled initial-clearance-matched stance diagnostics.

## Create, qualify, run, and inspect

Use the worktree's `uv` environment, CUDA, and local accepted baseline bundle.
Choose a new output directory. Creation does not fit or simulate anything.

```bash
uv run --no-sync -m projects.impedance_instron.experiments create \
  --output outputs/impedance_instron/shoe_campaign
uv run --no-sync -m projects.impedance_instron.experiments qualify \
  outputs/impedance_instron/shoe_campaign
uv run --no-sync -m projects.impedance_instron.experiments run \
  outputs/impedance_instron/shoe_campaign
uv run --no-sync -m projects.impedance_instron.experiments hysteresis \
  outputs/impedance_instron/shoe_campaign
uv run --no-sync -m projects.impedance_instron.experiments report \
  outputs/impedance_instron/shoe_campaign --individual
```

`qualify` executes the existing baseline CPU/GPU/contact and mixed-world checks.
It never calls the fit stage. `run` requires that evidence unless explicitly
marked `--diagnostic`. Variant fit errors are descriptive: variants need not
match the old shoe's measured motion to count as valid experiments.

`run` evaluates native and half timestep. Material conditions share a GPU batch
with distinct per-world materials. Geometries use separate engines. The large
rectangle uses the existing general foundation path rather than changing its
resolution to fit the fused path. Contact stiffness per area stays fixed even
where rectangular boundary-cell areas differ.

Every independent rollout starts with zero material/friction history. The
primary series keeps the original ground plane. Clearance-matched geometry
runs change only plane height and report that offset; they do not settle the
leg or retune the controller. Never mix the two protocols in a ranking.

## Small runnable example

After creation and qualification, select a few conditions first:

```bash
uv run --no-sync -m projects.impedance_instron.experiments run \
  outputs/impedance_instron/shoe_campaign \
  --case material_mu1_0.8 --case length_1.1 --case rectangle
uv run --no-sync -m projects.impedance_instron.experiments hysteresis \
  outputs/impedance_instron/shoe_campaign \
  --case baseline --case material_mu1_0.8 --case length_1.1 --case rectangle
uv run --no-sync -m projects.impedance_instron.experiments report \
  outputs/impedance_instron/shoe_campaign --individual
```

The unchanged baseline is included automatically in selected stance runs.
Use `--primary-only` to defer clearance-matched diagnostics. `--device cpu`
is a reference path, not the recommended campaign throughput path. A
`run --diagnostic` smoke run is explicitly unqualified evidence, not an accepted
campaign or a substitute for baseline qualification.

## Paper-informed compression comparison

Use a separate three-condition campaign for the unchanged baseline and the
FF LEAP / FF TURBO PLUS **compression-matched surrogates**:

```bash
uv run --no-sync -m projects.impedance_instron.experiments create \
  --suite paper_compression \
  --output outputs/impedance_instron/paper_compression_example
uv run --no-sync -m projects.impedance_instron.experiments run \
  outputs/impedance_instron/paper_compression_example --primary-only --diagnostic
uv run --no-sync -m projects.impedance_instron.experiments hysteresis \
  outputs/impedance_instron/paper_compression_example
uv run --no-sync -m projects.impedance_instron.experiments report \
  outputs/impedance_instron/paper_compression_example --individual
```

Choose a new output directory; creation refuses to overwrite a sealed campaign.
The default `sensitivity` suite remains the original 31-condition experiment.
This example uses explicitly diagnostic runs. Native/half-step agreement does not
replace full numerical qualification or establish physical validation.

The presets in [paper_materials.py](paper_materials.py) reproduce our existing
two-term approximations to Tables 1 and 2 of
[McCulloch, Delp and Kuhl](https://arxiv.org/abs/2602.12694v2), not the authors'
complete CANN models. They retain the source shoe geometry, fixtures, friction,
controller, equilibrium fraction, and relaxation time. Equilibrium moduli from
the compression fits are divided by the baseline equilibrium fraction to obtain
the stored instantaneous moduli. The tied Pasternak coefficients are then
recomputed from the new equilibrium moduli and unchanged thicknesses.

The paper does not identify Maxwell parameters. Our shared baseline relaxation
settings are an explicit assumption, not measurements of either named foam.
The compression fits also do not reproduce the paper's separate nonlinear shear
response. These cases do not represent complete ASICS shoes or their running
performance.

The report retains the source compression points, fit errors, and strain-domain
evidence. The authors' compression tables average conditioned loading/unloading
curves at 0.25/s and cover stretch 1.0 to 0.4, or **0 to 60% compression**.
Exceeding 60% is flagged as extrapolation; the simulation's existing safety caps
are not changed. A count of steps exceeding the range means at least one column
exceeded it at each such step, not that all columns did. The parameter fits are
approximate, especially near initial compression. The version-of-record URL is
recorded, but the source used here is the authors' manuscript because automated
publisher access was blocked.

## Rearfoot and fullfoot hysteresis

Every primary shoe has two independent bench tests: `rearfoot_punch` and
`fullfoot_last`. These are the actual saved fixtures. The user confirmed
fullfoot, not a forefoot-only last. Missing requested fixtures are reported;
another fixture is never silently substituted.

The sealed default is a **synthetic prescribed-displacement** protocol, not a
measured Instron replay: 10 mm peak depth, 0.20 s haversine cycles, three warmup
cycles plus the final fourth cycle. The displacement protocol is identical
across shoes. Material histories carry between conditioning cycles, but reset
between independent tests. Native/half-step benches are saved separately.

Each condition retains force-displacement histories, first/final loops, work
input/return/net values, closure checks, and termination/cap diagnostics. A
nonclosed loop's net work is not pure material dissipation. Bench tests still
run when that shoe's stance terminates. The report overlays the unchanged-shoe
loop for the same fixture, not the other fixture or a different loading rate.

## Actuator torque, power, and work

The report derives these metrics from saved native and half-step stance traces.
No controller fitting or simulation rerun is needed. Refresh an existing campaign:

```bash
uv run --no-sync -m projects.impedance_instron.experiments report \
  outputs/impedance_instron/shoe_experiments_v1
```

The torque mapping follows [the planar coordinates](../cartesian/mechanics.py)
and [the saved GPU actuator channels](../cartesian/gpu/engine.py): knee then
ankle torque, multiplied by their **relative** angular velocities. The report
shows torque and power overlays, peak/RMS loads, and cumulative produced,
absorbed, and net work. The stance slider also moves the plots' time cursor.

Hip point power is `Fx*vx + Fz*vz`. All-actuator produced and absorbed work sum
hip-x, hip-z, knee, and ankle contributions separately. Combining power before
splitting its sign would hide simultaneous production and absorption. These are
mechanical controller metrics, not muscle work or metabolic cost. An ankle-work
reduction can shift demand to the knee or external hip support.

[The analysis](actuation.py) uses full saved resolution before plot downsampling.
It integrates piecewise-linear power with explicit zero-crossing splits.
Absorbed work is a positive magnitude; net work is produced minus absorbed.
RMS uses time-weighted trapezoidal integration of squared load. Work covers
observed force support only, without extrapolation to an unsampled terminal
state. Stopped prefixes cannot be ranked as full-stance improvements.

The campaign metric selector compares each shoe against the current baseline
primary replay. The main table shows absolute and percent changes from baseline.
Percent change is `100 * (condition - baseline) / abs(baseline)` on matching
saved support; zero or near-zero baselines and incomplete comparisons show no
percentage. Half-step changes remain in row tooltips and downloads, alongside
the visible timestep-sensitivity diagnostics. Native/half-step effects use
common support across all four traces. Sign changes and effects no larger than their observed timestep change
are highlighted. This is numerical sensitivity, not an error bound, confidence
interval, or new qualification gate. Primary and clearance-matched protocols
remain separate.

`actuation.csv` exports values and shared-support diagnostics for both protocols.
`actuation.json` also contains display curves; the original `trace.npz` files
retain full-resolution samples. `results.csv` and `results.json` include the
main primary-protocol actuator summaries. Existing run outputs are not changed.

## Integrity and outputs

`plan.json` seals controller values, original input hashes, condition hashes,
source identity, requested fixtures, and bench protocol. Creation never edits
the accepted baseline. Changed sealed inputs or source require a new campaign.
Completed stance outputs have file hashes and can be resumed. Interrupted
partial outputs are preserved and must not be silently overwritten.

Open `report.html` for the offline comparison. Choose the left (solid) and right
(comparison) conditions. Overlay mode reuses the native leg, rigid last, solved
springs, foot-detail view, and deformation map. The selected comparison simulation
replaces mocap in the faded layer. Comparison opacity is adjustable from 5% to
100%; it changes visibility, not physics. Side-by-side mode shows both native
animations. Play and the time slider sit directly between the animation region
and the two shoe heat maps. Both heat maps always remain side by side in either
animation mode, with a shared compression unit and scale. They update from the
native renderer's acknowledged saved-frame times, not an independent timer.

`report --individual` builds the native pages used by the experiment viewer. Keep
`report.html` together with its `conditions/` directory; the native pages load
locally through relative paths and need no web server or external network. Each
native page remains a self-contained report that can also be opened directly.
The standalone native report retains its recorded-reference option; the embedded
experiment comparison does not display mocap.

`results.json` and `results.csv` retain every planned condition, including not-run,
terminated, and unresolved cases. Native reports include audited spring replays
where available. No terminated trace is padded with frozen states to look like a
complete run. Colors show compression, not force; the pair shares the mm scale,
and strain uses each column's own rest length. CAD meshes are undeformed context,
not a deformation prediction for the modified foundation.
