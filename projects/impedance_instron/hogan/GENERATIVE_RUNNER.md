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
activation lag and defaults to zero. Phase advances autonomously with bounded
simulated-load modulation; it never snaps to observed touchdown.

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
0.02 m, 0.05 rad, and 100 N, plus offset regularization. Reported
[`identify.score`](identify.py) also includes peak force, impulse, contact
duration, and effort; those additional terms do not enter the LM residual.
Evaluation observations never select parameters.

Forward-difference Jacobians and the damping ladder use persistent batched
CUDA rollouts. The small damped normal-equation solve runs on the host.
[`gpu_runner.GpuBatch`](gpu_runner.py) groups by shoe instance and exact adjusted
timestep. Chain and actuator arithmetic use float64; shared shoe physics use
float32. The CPU backend remains the numerical reference.
Both backends compute rollout diagnostics with the same summary function.
The shoe allocates only its prescribed carrier and physical foundation state;
report geometry comes directly from the artifact, without a second mesh or
spring-replay simulation.

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

## Known limitations

- Contact ends about 40 ms early. A small 10-20 Hz force wiggle remains.
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
