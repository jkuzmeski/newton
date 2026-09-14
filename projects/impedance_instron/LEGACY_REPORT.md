# Impedance Instron: controller rebuild and closed-loop policy

**Historical report template.** This page belongs to the retired multi-action,
momentum/work controller experiments. It is not the current two-stiffness report.
Read `projects/impedance_instron/REPORT.md` for the active workflow.

The fixed tables in `summary.py` were transcribed from the historical report with
this title. They are retained for reproducibility, not as current scientific
claims. Waveforms and timing checks come from the supplied artifacts. Missing
artifacts produce placeholders; a timing disagreement is not suppressed.

To render an archived run, supply its output directory explicitly:

    uv run -m projects.impedance_instron.summary --data-directory /path/to/legacy/outputs --output outputs/impedance_instron/legacy_report

A local `outputs/impedance_instron/LEGACY_ARCHIVE.json` may identify the archived
`outputs` directory and original report under `source_snapshot`. The archive is
optional and is not a repository dependency. Do not substitute current `simple/`
artifacts for these historical runs. Use `--report /path/to/historical/REPORT.md`
to render the original narrative instead of this abbreviated template.

## 2. Historical controller

### 2.1 Equilibrium-point impedance, both axes

The retired formulation controlled leg and ankle equilibria, stiffness and damping.
The historical stiff-limit table compares the compliant ankle with prescribed pitch
as ankle stiffness rises from 100 to 3,000,000 N m/rad. Figure G reproduces that fixed
table, not a convergence test of the current implementation.

### 2.3 Closed-loop policy

The archived ankle-v2 evaluation contains commanded leg length, achieved length,
stiffness and damping ratio. Figure H displays those channels when available.
The current policy does not have this historical multi-action interface.

## 3. Historical results

### 3.1 Against the measured stance

The historical comparison used S001 Trial 101, left stance at source times
90.0115-90.3065 s. `legacy_compare/trace.csv` contains the scheduled-controller
run and reference force. `eval_j/trace.csv` contains the open-loop command-J run.
`policy_ankle_v2*.eval.npz` contains the ankle-policy evaluation.

Peak force timing uses each run's detected-contact window, with shoe Fz above
`CONTACT_FORCE_FRACTION = 0.02` of the profile's own body weight. The measured peak
is located inside that same window, not a different clock. The historical table is:

| run | run peak [% stance] | measured peak in same window [% stance] |
| --- | ---: | ---: |
| legacy schedule, fed measured GRF | 47.73 | 44.22 |
| open-loop command J | 40.66 | 40.28 |
| policy with ankle v2 | 43.95 | 42.69 |

The renderer checks these pairs against loaded traces. Its existing 0.5-point
tolerance is unchanged; a coarse trace also receives its existing one-sample
quantisation allowance. These values must not be applied to a different run.

### 3.2 Task tolerances

Historical momentum excursion was measured outside a 0.044 m/s deadband.
The recorded excursion was 0.868 for command J, 1.190 for prescribed-pitch policy,
1.330 for prescribed pitch with three times the training budget, and 0.315 for
ankle v2. A broken-reward prescribed-pitch run recorded 1.951.

**The historical ankle policy was still outside the task tolerance.** These
numbers do not qualify it as a frozen measuring instrument or describe the current
tracking loss. Figure F reads historical training logs; fixed final-value lines
are not reconstructed learning curves.

## 4. Historical contact-clock correction

The old reward and acceptance criterion once used different definitions of when
stance began. The historical correction used detected touchdown for both.
The renderer retains the shared contact-force fraction for its timing checks.

## 5. Infrastructure

Figure J reproduces historical measurements, not a benchmark of today's code:
rollout 6.7 to 0.517 s, 64-world foundation substep 0.170 to 0.077 ms, and contact
metric block 49 to 10 microseconds. Hardware and workload changes require a new
benchmark; these fixed numbers do not establish a current speedup.

## 6. Material mapping, and what it cannot do

Figure I retains the old material comparison over 2-60 percent strain. It is not
human validation, a lateral-stability result, or evidence that a policy transfers
between shoes. Bare foam data and an effective intact-shoe law are different
experimental objects. Historical material curves and the controller figures must
not be interpreted as current two-stiffness calibration or validation results.
