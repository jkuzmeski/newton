# Movement intent and impedance response

This opt-in experiment separates the nominal movement load from the response to
motion error. It does not replace the two-stiffness PPO environment or train a new
policy. The existing `run`, `train`, `evaluate` and viewer commands keep their
controller contract.

## Run the experiment

Use the existing worktree environment and retained inputs. No new dependency is
required. If needed, first prepare the frozen reference with the command in the
[project README](README.md#one-workflow).

```bash
uv run --no-sync -m projects.impedance_instron response --device cuda:0
```

For a smaller comparison, run one stiffness level. Choose a fresh output directory
for each experiment; a nonempty output directory is rejected.

```bash
uv run --no-sync -m projects.impedance_instron response \
  --device cpu --stiffness-multipliers 1 \
  --push-force-x-n 150 --push-duration-s 0.04 --ground-offset-m 0.005 \
  --output outputs/impedance_instron/simple/response_nominal
```

Open `outputs/impedance_instron/simple/response/report.html` directly in a browser.
The report is figure-first: nominal-controller equivalence, the coupled stiffness
sweep and paired push response appear before collapsed numeric detail. The report
needs no server or CDN. Redraw saved results without simulation with
`uv run --no-sync -m projects.impedance_instron report outputs/impedance_instron/simple/response`. The output also saves full-resolution NPZ
traces, a JSON summary, input snapshots and a replay script. Saved inputs do not
archive the full Newton/Warp software environment. Source identities remain part
of the experiment record. Replay checks the saved input hashes and source
identities. After a source change, use explicit `--allow-physics-update` on the
saved replay script to run a new experiment; old results are not inherited.

```bash
uv run --no-sync outputs/impedance_instron/simple/response/replay.py \
  outputs/impedance_instron/simple/response_replay --device cuda:0
```

## Two explicit controller modes

### `equilibrium`: preserve the old movement program

```text
F = K * (L0 - L) + B * (L0dot - Ldot)
tau = Ka * wrap(angle0 - angle) + Ba * (angle0dot - angledot)
```

`L0` and `angle0` are the frozen inverse-dynamics equilibrium schedules. Changing
K changes both the disturbance response and the load produced along the measured
motion. This is a useful comparison, not a clean separation of the two effects.

The response experiment uses fixed K and independently fixed B in this mode.
The normal RL environment still uses its original variable K and fixed damping
ratios. These are distinct experiments.

### `intent`: separate nominal load and feedback

```text
F = F_ID + K * (Lref - L) + B * (Lrefdot - Ldot)
tau = tau_ID + Ka * wrap(angleref - angle) + Ba * (anglerefdot - angledot)
```

`Lref` is the leg length reconstructed from measured motion, NOT the stored
`Reference.leg_length_m` field. That legacy field contains the equilibrium `L0`.
The measured length and rate are in the reference's inverse-dynamics provenance.
`angleref` is the processed measured heel-cluster pitch.

At the reference position and velocity, feedback is zero. The raw actuator load
is therefore `F_ID` or `tau_ID` independently of K and B. Around that state, K
controls axial force per leg-length displacement and B controls axial force per
relative length rate; the pitch channel has the angular analogue. This is not a
specified Cartesian or coupled-system impedance: rotation of the loaded leg axis
also changes the force direction. Actuator limits still apply and are reported
separately.

**This mode explicitly uses nominal inverse-dynamics load feedforward at runtime.**
The old controller does not. This is a transparent engineering controller, not a
claim of unaided predictive motion, identified human impedance, or an anatomical
ankle controller. The nominal load remains tied to the recorded stride. No
inverse dynamics is recomputed during a rollout.

The implementation shares the Newton integrator, shoe material, contact mechanics,
limits and motion-only tracking score with the original rig. Neither mode directly
sets the body positions or velocities. The pitch actuator still reacts against
the world rather than a modeled shank.

## Paired tests

The default suite runs both modes at 0.5x, 1x and 2x nominal stiffness. Nominal K
is 12,000 N/m for the leg and 4,000 N·m/rad for pitch. Both stiffnesses change
together in this first sweep; it does not identify their separate contributions.
Damping is resolved once from the nominal reference settings and held identical
across all gains and modes. This avoids changing damping while testing stiffness.
The Python API accepts independently selected fixed damping values.

Each mode/gain has four cases:

1. An unperturbed rollout.
2. An upper-body forward push, with a raised-cosine envelope and 150 N peak.
   It starts at measured contact phase 0.35 and lasts 40 ms: a 3 N·s impulse.
3. A static ground plane 5 mm higher.
4. A static ground plane 5 mm lower.

The push uses interval-average forces so its discrete impulse matches the
specified continuous pulse, including partial solver intervals. Its work is
recorded separately from actuator source work.

The ground offsets are **different stationary planes from episode start**, not
a moving platform or a sudden terrain step. Initial body states, measured targets,
nominal loads, foam geometry and material stay unchanged. Both the contact solver
and last-clearance checks use the selected plane. No touchdown reset, reference
retiming, initial-height adjustment, or per-case inverse dynamics hides the
response.

## How to read the result

Compare each disturbed rollout to its own unperturbed mode/gain, not to a different
controller's motion. The report records displacement response, touchdown changes,
tracking loss, contact and actuator work, and physical safety results. Fore-aft
motion matters even though the original reward only measures pelvis height and
foot pitch.

A smaller disturbance deviation is not sufficient to call a controller better.
It can require more force or work, increase contact loading, or violate limits.
Invalid results remain visible. A single short stride does not establish recovery,
settling time, frequency response, closed-loop stability, or robustness to unseen
conditions. Ground-offset comparisons can also include a changed touchdown time.

Source work includes nominal-load power, moving-target power and any force-limit
intervention. Spring energy in `intent` mode uses displacement from the movement
reference; the nominal load is a separate active source. Positive damping
coefficients dissipate relative-motion energy, but moving references and nominal
loads can supply energy. **Passivity is not established.**

The two modes use different spring-energy and damping-reference definitions.
They can deliver almost the same body force while reporting very different source
work. Compare delivered actuator body work as well as the storage/damping/source
terms. A smaller source-work number across these controller realizations is NOT
proof of lower hardware energy use or higher efficiency.

The recorded pelvis is not whole-body COM, heel-to-shoe registration is assumed,
and reduced-model inverse-dynamics residuals remain substantial. The modeled shoe
is not the recording shoe. The rigid backing over the projected fixture footprint
is still an idealized upper interface. These tests do not repair or validate those
assumptions.

## Initial retained-input checks

The original zero-action baseline was captured before the shared-kernel change.
A fresh CUDA replay after the change gave exactly the same checked pelvis-height,
pitch, GRF, leg-force and pitch-torque traces, with tracking loss 9.61679195.
This is a software-regression observation on this run, not a cross-device or
cross-version bitwise guarantee.

At nominal gains, the intent and equilibrium laws produce nearly the same motion,
as expected from the offline construction. They need not be bitwise identical:
the frozen equilibrium offset is Hermite-interpolated between ID knots, whereas
the nominal ID load is linearly interpolated.

For the nominal intent mode, the 3 N·s forward push produced a peak pelvis-height
deviation of 1.234 mm and a terminal upper-body forward deviation of 8.086 mm.
Using 128 rather than 64 substeps per frame changed those values to 1.235 mm and
8.081 mm. Both runs passed the existing safety checks. This is a local time-step
sensitivity check, not proof of convergence for all gains or contact conditions.
The residual forward displacement also shows why height/pitch tracking alone is
not a disturbance-recovery test.

The full default sweep completed all 24 cases. Twenty-three passed the existing
safety checks. The old equilibrium controller at half stiffness with the lowered
plane failed the rigid-last clearance check. Its minimum clearance was -2.029 mm
against the existing -2.000 mm limit. Refinement to 128 and 256 substeps gave
-2.026 mm in both cases, still failing. This is a small margin, not a broad
robustness claim. That failed case remains in the report; it is not a qualified
comparison. All 12 intent-mode cases passed these
checks, which do not establish physiological validity or general robustness.

For the forward push, paired intent-mode results were:

| Stiffness multiplier | Peak pelvis-height deviation [mm] | Peak pitch deviation [mrad] | Final upper-body X deviation [mm] |
|---|---:|---:|---:|
| 0.5 | 1.031 | 0.451 | 7.976 |
| 1.0 | 1.234 | 0.281 | 8.086 |
| 2.0 | 1.201 | 0.197 | 8.364 |

Damping stayed identical. Higher K reduced pitch excursion here, but did not
monotonically reduce pelvis excursion or terminal forward displacement. This is a
coupled, moving system, not a static one-dimensional spring test. It is evidence
for keeping response metrics separate, not for automatically choosing a gain.

Local evidence is saved in
`outputs/impedance_instron/simple/response_integration_validation.json`,
`outputs/impedance_instron/simple/response_validation.json`, and the paired report
outputs. A fresh full saved-input replay reproduced all 24 tracking scores and
safety classifications exactly on this CUDA run. No prior validity was inherited;
every case was evaluated again. This does not promise bitwise GPU reproducibility
in general.

The selected regression checks passed 105 tests (81 existing/shared and 24 new).
Repository-wide and changed-file pre-commit checks passed. No policy
weights, measured reference, geometry or material were fitted for these checks.

## Next decisions

The independent gain, directional push, material-perturbation and finite-window
recovery experiment is now available in [SENSITIVITY.md](SENSITIVITY.md). It is
additive; the original 24-case experiment and RL controller above stay unchanged.

Use these paired responses to choose reasonable stiffness, damping and force
bounds before increasing the RL action space. Then test the two impedance channels
independently and vary push direction, timing and amplitude. A later policy can
adapt bounded impedance or movement intent, but it must retain separate load/work
accounting and demonstrate disturbance response rather than only nominal tracking.

## Implementation and tests

- [`simple/reference.py`](simple/reference.py): frozen motion and inverse-dynamics data.
- [`simple/rig.py`](simple/rig.py): shared mechanics and original RL controller.
- [`simple/response_control.py`](simple/response_control.py): opt-in fixed-gain controller and disturbances.
- [`simple/response.py`](simple/response.py): paired experiment and offline report.

```bash
uv run --no-sync -m unittest \
  newton.tests.test_impedance_response_control \
  newton.tests.test_impedance_response \
  newton.tests.test_impedance_response_cli
```
