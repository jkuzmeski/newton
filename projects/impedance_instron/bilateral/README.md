# Bilateral runner: two legs, neural impedance policy, and 3D

This package extends the single-leg [Hogan runner](../hogan/GENERATIVE_RUNNER.md)
in three stages:

1. **Two legs in the sagittal plane.** A floating pelvis carrying only the
   trunk, arms, and head (HAT) and two explicit legs, each with its own
   persistent shoe.
2. **A neural impedance policy trained with PPO.** A proprioceptive MLP
   outputs bounded equilibrium, stiffness, and damping per joint, replacing the
   linear feature schedule.
3. **3D.** A Newton articulation with three-axis hips, hinge knees and ankles,
   and both shoes applying their full 3D wrench.

The same rule as the single-leg runner applies throughout: no measured
trajectory, force, or reference clock enters the dynamics or the policy inputs.
Measurements are used only for rewards, scores, terminations, initial states,
and the critic's training-only inputs.

## Data

```console
uv run --no-sync -m projects.impedance_instron prepare-bilateral --output outputs/bilateral/dataset_6step --profile PROFILE.json --shoe LEFT_SHOE.json
uv run --no-sync -m projects.impedance_instron prepare-bilateral-3d --planar-dataset outputs/bilateral/dataset_6step --output outputs/bilateral/dataset_3d
```

[`data.py`](data.py) cuts FR3_1 and FR3_2 into continuous segments. Each one
runs from the mid-hip height peak before the first contact to the peak after
the last, spanning six alternating contacts (about 2 s). The single plate force
goes to the foot that the single-leg dataset's COP-proximity rule identifies.
Running has no double support, so a segment is rejected only if loaded samples
cannot be attributed. Each trial has one contiguous evaluation block with a
one-segment guard on either side. With the default seed this gives 50 training
segments, 6 evaluation segments, and no rejections.

Both thighs attach at the mid-hip center. Each leg's hip and knee are found by
regularized inverse kinematics toward its measured hip-to-ankle vector, keeping
the measured foot angle ([`BilateralChain.fit_leg`](mechanics.py)). Pelvic
rotation and obliquity move the sagittal hip centers by up to 5 cm from the
mid-hip; near full knee extension, the projected hip-to-ankle distance can
exceed the fixed segment lengths. In both cases the measured joint angles keep
the knee from snapping straight, and the ankle error left over (1.6 to 2.3 cm
at its peak) is recorded per segment.

The F01 shoe artifact is a left shoe. [`shoes.py`](shoes.py) mirrors it for
the right foot: y becomes -y, the -y and +y neighbor slots swap, and triangle
winding is reversed. In the sagittal plane the mirrored shoe's wrench equals
the original's to float32 rounding. It is a geometric assumption, not a
measured right shoe.

[`data3d.py`](data3d.py) reuses the planar windows and splits. Static frames
are built from ASIS/PSIS, femoral epicondyles, malleoli, and the
heel-to-metatarsal heading (the shoe is level in the static trial), then each
segment is tracked by a rigid fit of its marker cluster. Hip angles are
anatomical. Knee and ankle hinges are zero in each trial's static pose and are
refined per frame against the measured joint centers and shoe frame. The
omitted knee and subtalar rotation leaves about 1 cm RMS ankle error and 5 to
7 degrees RMS shoe-orientation error. Each trial keeps its own calibration, so
world geometry follows its segment's trial.

## Planar mechanics and environment

[`mechanics.py`](mechanics.py) has 9 coordinates:
`[x, z, pelvis, hip_r, knee_r, ankle_r, hip_l, knee_l, ankle_l]`. Angle
conventions match the single-leg chain, and a test checks that a massless left
leg reduces to it. The HAT uses Winter's proportions (COM 0.33 m above the hip,
radius of gyration 0.26 m); its mass is the subject mass minus both legs. These
are anthropometric estimates.

[`env.py`](env.py) integrates batched CUDA worlds using the double-precision
kernels in [`gpu.py`](gpu.py) and the lean elastic-Coulomb shoe kernels, with
one foundation per foot. Shoe history persists across contacts, and a
terminated world's history is cleared on its own. The policy runs at 500 Hz;
the physics substep stays at 1.25e-4 s, and the torque cap acts on every
substep. Actuation has no torque lag and no slew limit by default, in the planar
and the 3D environment alike: `response_time_s=0` applies the impedance torque
directly, and `torque_slew_limit=False` removes the rate bound. Both options stay
for reproducing older runs, which saved their own values in the run config.
[`RESULTS.md`](RESULTS.md) compares the fitted single-leg runner's 5.3 ms lag, no
lag, and the unfitted seed's 25 ms, which turned the 0.0097 kg m^2 foot into a
lightly damped ~6 Hz resonator ("Torque lag and slew").
Scheduled damping is applied immediately (`immediate_damping`), following the
GRF ripple study. A NumPy/CPU-shoe
reference ([`reference.py`](reference.py)) reproduces a CUDA world through
contact to 2e-5.

## Neural impedance policy (PPO)

```console
uv run --no-sync --extra torch-cu12 -m projects.impedance_instron train-policy --dataset outputs/bilateral/dataset_6step --output outputs/bilateral/ppo_run1
uv run --no-sync --extra torch-cu12 -m projects.impedance_instron evaluate-bilateral --model outputs/bilateral/ppo_run1/policy.pt --dataset outputs/bilateral/dataset_6step --output outputs/bilateral/eval_ppo1
```

- **Actor** ([`policy.py`](policy.py)): 32 proprioceptive inputs, 2 x 256 ELU
  hidden layers, and 18 outputs (equilibrium, stiffness, and damping for 6
  joints), 78,866 parameters. The inputs are pelvis height, lean, speed
  error and rates; joint angles and rates; filtered simulated foot loads;
  filtered torques; ankle positions relative to the hip; and the internal
  oscillator's sine and cosine. The outputs are logits that pass through the
  single-leg runner's sigmoid bounds. The output bias starts at that runner's
  unfitted engineering seed, not a fitted model.
- **Critic**: additionally sees reference context (target pose and force,
  remaining segment time). It is used only during training.
- **Training** ([`ppo.py`](ppo.py)): a self-contained torch PPO with
  clipped-surrogate updates, GAE, value bootstrapping at time limits, running
  observation normalization, and a KL-adaptive learning rate. There is no
  symmetry loss (runs 1-3 had one).
- **Episodes**: each starts in flight from a random measured training state
  and continues through many contacts. An episode ends on a screen failure, on
  departing the reference (0.15 m height, 0.6 rad pelvis, or 0.35 m forward),
  or at the end of the segment.
- **Reward**: exponential kernels on pose, joint velocity, pelvis position,
  per-foot GRF (both averaged over the control interval), and effort, weighted
  0.35 / 0.1 / 0.2 / 0.25 / 0.1. Optional terms follow the ProtoMotions motion
  tracker:
  - pooled position and velocity kernels on the knee, ankle, heel, and toe
    relative to the pelvis (`--body-position-weight`, `--body-velocity-weight`);
  - a binary foot-contact match penalty (`--contact-match-penalty`);
  - an action-smoothness penalty (`--action-smoothness-penalty`);
  - the previous action as an input (`--previous-action-obs`).

  The policy inputs stay free of reference data: unlike ProtoMotions, the
  actor gets no future target frames.
- **Evaluation** ([`rollout.py`](rollout.py)): deterministic open-loop
  rollouts from each segment's three-frame prefix, initialized like the
  single-leg runner (including the plate-integrated COM velocity). Scoring
  extends `identify.score` to two feet, with 20 Hz force observation and the
  vibration penalty.

## Diagnosis tools

- [`ankle_diagnosis.py`](ankle_diagnosis.py) splits a policy's ankle error into
  a command error (q_eq ? q_ref) and a following error (q ? q_eq) for each
  phase: stance, early swing, late swing, and the 30 ms before touchdown. It
  also relates the ankle state at touchdown to the impact peak.
- [`ankle_interventions.py`](ankle_interventions.py) replays a frozen policy
  with one ankle channel overridden: q_eq, K, or D held for 60 ms after
  touchdown, or the swing ankle aimed at the measured angle.
- [`compare.py`](compare.py) scores models on the single-leg stances that lie
  outside every bilateral training segment.
- [`report.py`](report.py) evaluates a trained run and writes its graphs
  (training curves, segments, ankle phases, touchdown impedance) to
  `<run>/report/` (`report run --run-dir ... --dataset ...`). It also compares
  an experiment's runs (`report experiment --dir ...`). Grid jobs run both
  automatically.

## Linear Hogan baseline

```console
uv run --no-sync -m projects.impedance_instron fit-bilateral --dataset outputs/bilateral/dataset_6step --output outputs/bilateral/linear_c1 --horizon-s 0.45
```

[`linear.py`](linear.py) shares one set of single-leg feature weights
between both legs. Each leg uses its own load and hip-over-ankle feature, and
the left leg is half a cycle behind. That gives the same 119 active
parameters, fitted by forward-difference LM on deterministic training rollouts.
Starting from the unfitted seed, every rollout fails early. Fitting therefore
uses a horizon curriculum (0.45 s, 0.8 s, 1.2 s), each stage warm-started from
the previous one. This model is a comparison only; it never initializes the
policy.

## 3D articulation

```console
uv run --no-sync --extra torch-cu12 -m projects.impedance_instron train-policy --kind 3d --dataset outputs/bilateral/dataset_3d --output outputs/bilateral/ppo3d
uv run --no-sync --extra torch-cu12 -m projects.impedance_instron evaluate-bilateral-3d --model outputs/bilateral/ppo3d/policy.pt --dataset outputs/bilateral/dataset_3d --output outputs/bilateral/eval_ppo3d
uv run --no-sync --extra torch-cu12 -m projects.impedance_instron endurance-bilateral-3d --policy outputs/bilateral/ppo3d/policy.pt --dataset outputs/bilateral/dataset_3d_static --output outputs/bilateral/endurance_ppo3d --distance-m 1000 --worlds 8
```

`endurance-bilateral-3d` runs the deterministic policy open-ended from dataset
flight starts until each world falls or covers the distance, and reports
distance, fall cause, speed, heading and lateral drift, cadence, and stance
GRF peaks.

- **Model** ([`model3d.py`](model3d.py)): seven bodies per world. Each foot
  body has its origin at the ankle and uses the intrinsic shoe axes, so the
  shoe foundation attaches directly. Each world shifts each shoe along the
  foot's z axis so the ankle sits at its trial and side's static height;
  the shared shoe geometry uses their mean. Masses, COMs, and sagittal inertias match
  the planar profile; frontal and axial inertias are engineering assumptions.
  The model has no collision shapes.
- **Environment** ([`env3d.py`](env3d.py)): uses
  `SolverFeatherstone` and refreshes the mass matrix every second substep
  (0.25 ms). The shoes run the lean contact kernels with every column as a
  flight candidate, because the 3D foot rolls and yaws. Ten joints are driven
  by the same impedance law, giving 55 observations (including the offset from
  the pacer and the heading) and 30 actions. The mirror
  operation also negates lateral components.
- **Throughput**: the articulated solver dominates the cost. 3D training runs
  well below the planar environment's roughly 45k samples/s.

## Running on the LLSC grid

See [grid/README.md](grid/README.md): immutable code snapshots, datasets on `Z:`, and
throttled H100 array jobs for the experiments in [grid/experiments.py](grid/experiments.py).

## Results

See [`RESULTS.md`](RESULTS.md).

## Limitations

- The sagittal model attaches both hips at one point; pelvic rotation and
  obliquity remain as recorded ankle errors.
- The right shoe is a mirror of the left artifact.
- Impedance decomposition is not unique. The policy's K and D are effective
  values, not muscle properties.
- The HAT and the frontal and axial inertias are anthropometric assumptions.
- One subject at one speed; held-out segments come from the same sessions.
- In 3D, knees and ankles are hinges, there is no subtalar joint, and the
  trunk is rigid on the pelvis.
