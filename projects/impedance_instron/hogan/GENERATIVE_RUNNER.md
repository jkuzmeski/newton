# Generative runner with variable impedance

## Latest full run (2026-10-08)

The [frozen F01 model](baselines/generative_runner_f01_20261008.json) describes
one 66 kg runner at 3.65 m/s, with the Puma Fast-R Nitro Elite 3 digital shoe.
The original run is `outputs/impedance_instron/generative_fit_lm_flightcom_20261008`.
See the [pipeline guide](../README.md) for commands.

| Split | Model | Mean loss | Hip RMSE fwd / up [mm] | Angle RMSE [deg] | Fx / Fz RMSE [N] | Mean absolute peak Fz error [N] | Contact error [ms] |
|---|---|---:|---:|---:|---:|---:|---:|
| Train (98) | seed | 120.8 | 84 / 51 | 12.0 | 211 / 349 | 775 | -1 |
| Train (98) | fitted | 16.1 | 47 / 19 | 6.2 | 108 / 128 | 58 | -39 |
| Held-out (9) | seed | 125.4 | 79 / 52 | 11.8 | 214 / 359 | 866 | -2 |
| Held-out (9) | fitted | **14.8** | **42 / 18** | **6.5** | **105 / 119** | **26** | -42 |

LM ran 15 iterations and 195,608 candidate-stance rollouts in about 82 minutes
on an RTX A4000 Laptop GPU. Training objective fell from 42.2 to 10.03.
These are free-prediction fitting results, not physiological validation.

## Runtime boundary

[`runner.simulate`](runner.py) accepts a frozen `Runner`, physical `Chain`,
`Shoe`, initial `State`, known `Task`, horizon, and integration configuration.
It accepts no measured trajectory, GRF/COP target, reference clock, or
inverse-dynamics plan.

The six mechanical coordinates are hip x/z, pelvis tilt, hip, knee, and ankle.
Only the last three receive actuation; floating-base actuation is exactly zero.
The state also carries oscillator phase, filtered simulated normal load, and
three activation-filtered torques. Fourteen features generate bounded
equilibrium angle, stiffness, and damping:

- bias and sine/cosine of the first three phase harmonics;
- simulated normal load and its first-harmonic modulation;
- hip-over-ankle offset, pelvis lean, speed error, and task-speed offset.

Joint command is `K * (q_eq - q_joint) - D * qdot_joint`, with torque magnitude,
slew, and first-order response bounds. Optional intrinsic damping bypasses the
activation lag and defaults to zero. With `immediate_damping` (`--immediate-damping`),
only the spring torque passes through the response and slew bound and the
scheduled `D` acts without lag; saved models keep the fully lagged command.
Phase advances autonomously with bounded simulated-load modulation; it never
snaps to observed touchdown.

The equilibrium moves internally and can supply work. This is an active
effective-actuation model, not a passivity or muscle-identification claim.
Schema-1 models still load with zero weights on the added features.

## Preparation and initialization

[`prepare_dataset`](../cartesian/prepare_dataset.py) preserves the selected
windows and split. The latest dataset uses the corrected mass and belt speed
from `data/F01/FR3_1/stance_selection.json`. Three of 110 windows were excluded
because the relaxed shoe starts below ground.

[`identify.load_trials`](identify.py) re-solves hip and knee from measured hip
and ankle positions, keeping the absolute foot angle. This position-only step
corrects the exported joint-angle/FK mismatch; it does not fit force.

Prediction starts at the end of a three-frame observed-position prefix.
Backward quadratic differentiation supplies joint rates. Where available,
`flight_velocity_m_s` comes from the preceding stride's plate-force integration
and is matched to the model's whole-body COM, not directly to the hip. Members
without that preceding stride retain the three-frame velocity estimate.
Initial phase uses a fixed hip-angle/rate convention; torque and load memory
start at zero.

The loader checks ankle FK agreement and flight at the prediction origin.
Loaded clearance and COP footprint conflicts are reported, not gated.
Incompatible inputs stop fitting before training; `--allow-incompatible` is
an explicitly unvalidated diagnostic override. Upstream filtering may be
noncausal: the prefix boundary applies to the supplied prepared positions.

## Identification and execution

[`least_squares.fit_lm`](least_squares.py) learns one shared parameter set,
using training stances only. There are 119 active parameters at one speed, or
129 with multiple training speeds. Direct task-speed weights and cadence-speed
slope stay fixed for single-speed training.

The residual vector contains hip, angle, and GRF sample errors scaled by
0.02 m, 0.05 rad, and 100 N, plus offset regularization. With
`--force-filter-hz`, the GRF errors use the observed simulated force and a
second block holds the weighted physical minus observed force. Reported
[`identify.score`](identify.py) also includes peak force, impulse, contact
duration, and effort; those additional terms do not enter the LM residual.
Evaluation observations never select parameters.

Forward-difference Jacobians and the damping ladder use persistent batched
CUDA rollouts. [`gpu_residuals.GpuResiduals`](gpu_residuals.py) integrates all
training stances and candidates of one shoe concurrently; every world keeps its
trial's exact adjusted timestep. After each accepted step an observer writes the
weighted `residuals` entries on the device; targets never feed the dynamics. The
Jacobian difference, `J^T J`, and `J^T r` are also formed there, so only
per-world screens, a few metrics, and the small damped normal-equation solve
reach the host. [`gpu_runner.GpuBatch`](gpu_runner.py) records full traces for
evaluation and generation, grouped by shoe instance.

Elastic-Coulomb ground beds advance with the lean kernels in
[`gpu_shoe.py`](gpu_shoe.py): they evaluate the shared material, Maxwell,
surround, and friction laws but keep only physics history, reduce the carrier
wrench in the shared fixed order, and skip a world's shoe while every column
clears the ground by 0.5 mm, both before touchdown and after toe-off. A lifted
shoe counts the skipped steps and replays their unloaded history updates before
contact resumes. The kernels reproduce the generic fused-foundation traces
bitwise. Chain and actuator arithmetic use float64; shared shoe physics use
float32. The CPU backend remains the numerical reference. Both backends compute
rollout diagnostics with the same summary function. The shoe allocates only its
prescribed carrier and physical foundation state; report geometry comes directly
from the artifact, without a second mesh or spring-replay simulation.

By default, the finite-difference rollouts and their own reference use the same
shoe kernels compiled with fast math, which changes only float32 intrinsic
rounding. Costs, damping-ladder proposals, `J^T r`, and accepted steps always use
exact rollouts. `identify fit --exact-jacobian` integrates the differences
exactly as well.

On the same GPU, one full-dataset LM iteration (119 Jacobian and 5 ladder
candidates over 98 stances) now takes about 15 s instead of about 245 s, or
about 20 s with `--exact-jacobian`. With exact differences the iteration history
agrees with the saved run to float roundoff in the reductions; with fast-math
differences the first three costs agree to within 4e-5 relative.

Each fit writes `runner.json`, split metrics and provenance in `summary.json`,
`trace_NNN.npz`, reference-free `scenario_NNN.json`, and `report.html`.
[`generate`](generate.py) needs only the frozen model, scenario, and hash-verified
shoe artifact. A scenario contains chain, shoe, initial state, task, horizon,
and numerical settings, never observations.

Datasets accept `peak_hip_stance_dataset_1` or `generative_runner_dataset_1`.
Members specify `id`, `split`, `reference`, profile/shoe paths (or
`shared_assets`), mount, pitch, and task speed. Paths are dataset-relative.
Use one fixed physical subject profile; hold out entire sessions/conditions
when testing transfer. Changing the shoe for a mechanical comparison must keep
the runner and initial state fixed.
Profiles contain thigh/shank/foot masses, local COM offsets, sagittal inertias,
and inertial provenance. Historical Cartesian gains and search limits are
discarded on load, not reused as runner parameters.

## GRF targets and force ripple (2026-10-09)

The fitted vertical and fore-aft GRF showed a stance ripple against the original
F01 targets: 10-40 Hz Fz RMS of 137, 74, and 42 N in the first 30 %, middle
40 %, and last 30 % of contact over the 107 saved stances, against 14, 6, and
3 N in the target. Two causes combine.

**The original targets were over-smoothed.** Those Visual3D exports filtered the
plate force with a zero-lag second-order Butterworth at 6 Hz (Winter-corrected),
a walking setting, and zeroed both axes where filtered Fz fell below about 1 N.
Reprocessing the raw `data/FR3_Metabolic.c3d` plate reproduces that export
within 20 N RMS. The raw force carries a 1185 N impact peak about 40 ms after
contact and 191/78/43 N of 10-40 Hz Fz content, as much as the simulation. The
6 Hz filter moved the 50 N touchdown 21.7 ms early and lengthened contact by
27 ms, so most of the earlier "contact ends about 40 ms early" was the filter.

The F01 exports now use 20 Hz, the running convention (`9336e058`), with the
measured 66.5 kg body mass (`0567b02d`). The 20 Hz export matches the raw plate
within 12 N RMS, keeps the impact peak, and matches the raw contact timing to
within 1 ms. Its 10-40 Hz Fz content is 150/45/10 N.

**Part of the simulated vibration is a model artifact.** Differential impulse
responses at mid-stance show a ~70 Hz foot-pitch mode on the foam (damping
ratio 0.05-0.10) and a 20-27 Hz leg-on-foam mode (0.1-0.25). The foam's
Maxwell branch (5 ms relaxation, 0.695 equilibrium fraction) adds little
damping, and the fully lagged command keeps only `1 / (1 + (w T)^2)` of `D`
(about 1/6 at 20 Hz and 1/57 at 70 Hz for T = 17 ms) while its lagged spring
adds `-K T / (1 + (w T)^2)`. Halving the timestep, smoothing the sensory
features, removing phase feedback, or lifting the slew bound leaves the
ripple unchanged.

Use both remedies:

- `--force-filter-hz 20` scores simulated GRF as the target was measured, in
  every GRF residual, peak, impulse, and contact term on CPU and CUDA. The
  dynamics never see it. The residual also penalizes the physical minus observed
  force, which the target cannot constrain (`--force-vibration-weight`,
  default 1). Against the 6 Hz targets, LM without that penalty let the ripple
  grow to 307/172/65 N.
- `--immediate-damping` lets the scheduled damping act without the lag.

Refits on the 20 Hz dataset with `--force-filter-hz 20`, warm-started from the
frozen model for 15 LM iterations, and the frozen model itself, all scored the
same way (held-out mean loss and GRF RMSE as measured; physical simulated Fz
content over all 107 stances):

| Fit | Held-out loss | Fx / Fz RMSE [N] | 10-40 Hz load / mid / late [N] | 40-150 Hz mid [N] |
|---|---:|---:|---:|---:|
| Raw plate force | - | - | 191 / 78 / 43 | 13 |
| Frozen model, no refit | 11.66 | 108 / 118 | 152 / 82 / 46 | 37 |
| Refit, lagged damping | 12.00 | 102 / 111 | 202 / 69 / 48 | 12 |
| Refit, immediate damping | **11.48** | **103 / 107** | 209 / 65 / 34 | **5** |

The combination generalizes best and removes the foot ringing; the remaining
stance content is comparable to the raw plate's. On held-out stances, as
measured, contact ends about 15 ms early, peak Fz is about 80 N low, and the
propulsive Fx peak is about 160 N short of the measured 370 N. The frozen model
shows similar gaps, and refitting with immediate damping does not close them.
The unfitted seed with immediate damping bottoms out the shoe on 4 training
stances, so start such fits from a completed model with `--model`.

## Known limitations

- Against the 20 Hz targets, simulated contact ends about 15 ms early, and peak
  vertical and propulsive force are about 80 N and 160 N low.
- Impedance decomposition is not uniquely identified: equivalent K, D, and
  equilibrium schedules can produce similar motion and torque.
- One modeled leg and a lumped rest-of-body omit bilateral running and
  independent trunk/forefoot motion. Each call resets shoe history.
- Body inertia, registration, worn-shoe correspondence, and force-export
  history remain incompletely qualified. A left-shoe artifact is used with
  right-foot sagittal observations.
- No explicit muscles, tendons, transport delays, or metabolic cost are modeled.
  Transfer across speed/footwear and sustained running are not certified.

The saved full-run results and original measurements are preserved; cleanup
does not refit parameters or conceal these limitations.
