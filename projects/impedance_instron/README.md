# Two-stiffness impedance Instron

**Two online controls. One kinematic tracking score.**

See [WORKTREE_SUMMARY.md](WORKTREE_SUMMARY.md) for the consolidated implementation
history, measured results, runnable examples and remaining qualification limits.

The policy changes leg spring stiffness [N/m] and ankle rotational stiffness
[N·m/rad]. It observes motion and force. It does not command pelvis position,
foot angle, force, equilibrium offsets, or damping independently.

The target is measured **pelvis-centroid height** and heel-cluster **foot pitch
relative to the ground**. The upper inertial body represents the pelvis centroid;
it is not whole-body COM or an anatomical pelvis. Foot pitch is not ankle flexion
relative to a shank. Ground-reaction force and actuator work are **evaluation
metrics only**. There is no momentum, impulse, work, or GRF term in the reward.

## One workflow

Use the worktree environment without changing its installed dependencies:

```bash
# Build and freeze the offline inverse-dynamics reference.
uv run --no-sync -m projects.impedance_instron prepare

# Run the fixed nominal-stiffness baseline and write an offline HTML report.
uv run --no-sync -m projects.impedance_instron run

# Train only the two stiffness outputs; save the best valid tracking policy.
uv run --no-sync -m projects.impedance_instron train --iterations 100 --worlds 32

# Restore the complete frozen experiment, not current CLI defaults.
uv run --no-sync -m projects.impedance_instron evaluate \
  outputs/impedance_instron/simple/training/best.pt

# Explicitly change material, not geometry, reference, normalization or rig settings.
uv run --no-sync -m projects.impedance_instron evaluate \
  outputs/impedance_instron/simple/training/best.pt \
  --material /path/to/same-geometry-different-material.json \
  --output outputs/impedance_instron/simple/material_comparison
```

The retained local inputs are in `outputs/impedance_instron/inputs/`:
`reference_profile.json` (sealed v3 optical pelvis/heel data),
`stride_variability.json` (83 stances at steady commanded 3 m/s), and
`digital_shoe.json` (two-term Ogden–Hill shoe fit). Missing inputs fail explicitly.
Generated inputs/outputs are local data, not distributed with the source.

The prepared `simple/reference.json` embeds the measured targets, frozen schedules,
initial state, processing assumptions, scales, source hashes and inverse-dynamics
residuals. Rebuilding is an explicit new experiment. Training and evaluation do
not rerun inverse dynamics. Input changes must not silently modify a frozen run.

## View the actual rig

```bash
uv run --no-sync -m newton.examples impedance_stiffness --viewer gl --render-fps 30
uv run --no-sync -m newton.examples impedance_stiffness --viewer null --test
```

Use `--checkpoint outputs/impedance_instron/simple/training/best.pt` to view a
frozen policy. Blue is the simulated upper mass. Orange is the measured height
target (offset sideways for visibility, not a force or constraint).

After a physics source change, explicitly re-evaluate old weights rather than
silently treating the checkpoint's old scores as current:

```bash
uv run --no-sync -m newton.examples impedance_stiffness \
  --checkpoint outputs/impedance_instron/simple/training/best.pt \
  --allow-physics-update --viewer gl --render-fps 30
```

This flag permits source-only changes. Geometry, material, reference and frozen
settings checks remain enforced. The weights are not retrained or overwritten.
The viewer report records both source fingerprints and marks old scores as
inapplicable. New checkpoints fingerprint the foundation runtime and its shared material/contact dependencies.
For a headless policy report, use the same explicit consent:

```bash
uv run --no-sync -m projects.impedance_instron evaluate \
  outputs/impedance_instron/simple/training/best.pt \
  --allow-physics-update --output outputs/impedance_instron/simple/physics_updated
```

### Passive shoe and contact geometry

Passive outer columns share compression through the existing neighbor graph.
Their undeformed geometry follows the carrier in flight. The viewer reconstructs
relative deformation from compression and the current pose; the contact solver's
`z_free` pressure reference and friction bristle anchors are not physical endpoint
positions. Plane projection cannot lengthen a displayed column beyond its rest
length. Rendering does not advance the foam or friction histories.

The active rig declares a physical ground plane at `z=0`. Local nonnegative ground
reaction sets friction capacity, center of pressure and the external wrench at
the contact surface. Signed neighbor-transfer traction remains a separate
`column_force` diagnostic; clamping it would invent support. The complete ground
force and moment reach the carrier in the massless-shoe approximation. Generic
bench-fixture foundation callers retain their existing convention unless they
explicitly supply `FoundationConfig.ground_height_m`.

The upper interface remains an **idealized rigid backing over the projected
fixture footprint**, not solved contact with the displayed rigid-last mesh.
`carrier_bond` is a one-sided retention bound on the free surround, not a literal
bond of every outer top to the foot. Fixture clearances and the known discrepancy
between point-clamped seating samples and the rigid mesh require a coherent upper
contact/registration model. They are not repaired by changing foam rest lengths,
subtracting a gap from one force calculation, or retuning the material. The report
records this limitation and measured fixture-clearance statistics under
`physics.last_support`; these ground-contact fixes do not validate last seating.

The baseline report is `outputs/impedance_instron/simple/baseline/report.html`.
It opens offline with no server or CDN. It shows separate stiffness axes in their
proper units, both damping values, kinematic targets, resulting GRF, source work,
and compression. NPZ files retain every solver sample. Report curves can be
decimated for display; the underlying saved trace is not filtered to hide force
oscillations.

## Offline construction, online impedance

Offline inverse dynamics uses the recorded optical pelvis/heel motion, declared
rig masses/inertia and measured platform wrench. Its force/torque are NOT runtime
feedforward commands. They construct equilibrium schedules with nominal K and B:

```text
B_nom * e_dot + K_nom * e = inverse_dynamics_load
L0 = nominal_leg_length + e_leg
angle0 = measured_foot_pitch + e_angle
```

The moving-equilibrium damping term is included; dividing load by stiffness alone
would not satisfy this impedance law. Marker mapping, geometric sole registration,
optical smoothing, and inverse-dynamics residuals are recorded. Human motion need
not satisfy this reduced model, especially its unactuated fore-aft balance.
Residuals diagnose full-motion consistency; they do not prove that the selected
pelvis-height/foot-pitch targets are unreachable.

During simulation:

```text
F_leg = K_leg*(L0-L) + B_leg*(L0_dot-L_dot)
tau_ankle = K_ankle*(angle0-angle) + B_ankle*(angle0_dot-angle_dot)
B_leg = 2*zeta_leg*sqrt(K_leg*m_upper)
B_ankle = 2*zeta_ankle*sqrt(K_ankle*I_foot)
```

The equilibrium schedules and damping ratios are frozen. The policy outputs two
bounded absolute log-stiffness coordinates. A rate-bounded quintic interpolation
keeps K and Kdot continuous at frame boundaries. Damping follows the declared
rule; it is not a third or fourth control. These ratios describe the selected
impedance convention, not identified coupled-system modal damping.

The mechanical leg spring can push and pull, with a signed actuator force limit.
The ankle reacts against the world, not a modeled shank. No prescribed pitch,
pelvis position servo, release schedule, or auxiliary residual actuator overrides
the dynamics. The identified foundation and its passive outer region are retained;
this is not a new anatomical foot-to-shoe attachment model.

## The entire reward

```text
error(t) = 0.5 * ((pelvis_z-target_z)/pelvis_scale)^2
         + 0.5 * (wrapped(foot_pitch-target_pitch)/pitch_scale)^2
return = -integral(error(t), dt)
```

Scales come from across-stride optical variability, not force-integrated COM.
They are scales, not hard human-repeatability thresholds. A single measured stride
is the target, not the ensemble mean. No claim that a score of one is an acceptance
boundary is made.

Reference time is ordinary simulation time. Nominal phase is
`(time - measured_touchdown)/measured_contact_duration`. There is no learned phase,
contact-dependent clock reset, toe-off alignment, or per-rollout time stretching.
Flight and release samples remain in the error. Actual contact timing is recorded
as a diagnostic, not used to hide tracking error.

PPO uses complete, undiscounted episode returns from this same error. Finite
trajectories can train even when physical qualification fails; only valid
deterministic evaluations can become `best.pt`. `last.pt` is distinct. Mechanical
safety checks do not become weighted reward tiers. If no valid policy is found,
training reports that fact rather than labeling an invalid checkpoint "best".

## Evaluation and limits

- GRF and work are consequences, not optimized targets. Positive, negative and
  signed source work are reported separately per actuator, without a metabolic
  efficiency objective. Variable-stiffness work and damper work remain visible.
- Tracking true COM acceleration would constrain net GRF. Pelvis height is only a
  surrogate, and finite motion errors can accompany substantial force differences.
- The recording shoe is not the modeled Puma. This is a representative engineering
  experiment, not same-shoe validation or an estimate of physiological stiffness.
- Material comparisons restore all frozen settings and verify geometry identity.
  Keeping the same policy does not prevent feedback from compensating for material
  changes; stiffness commands and work are therefore important evaluation outputs.
- Frozen replay checks reference, rig configuration, input identities and both rig
  and foundation source. Explicit source-update evaluation records the mismatch
  and does not inherit the checkpoint's old physical qualification.
  It is tested within the current Newton/Warp build. It is not a complete archived
  software environment or a promise of identical results across solver versions.
- The source optical file is a Visual3D export. Upstream gap filling and soft-tissue
  artifact remain possible. Differentiation for offline inverse dynamics depends
  on the disclosed smoothing; it is not a new direct acceleration measurement.

## Code and tests

The active path is `simple/reference.py`, `simple/rig.py`, `simple/policy.py`,
`simple/report.py`, and the optional `simple/example.py` viewer. It does not import
the legacy environment, optimizer, objective or trainer.

```bash
uv run --no-sync -m unittest \
  newton.tests.test_impedance_simple_reference \
  newton.tests.test_impedance_simple_rig \
  newton.tests.test_impedance_simple_policy \
  newton.tests.test_impedance_simple_report
```

The old six-action experiments are retired from the active workflow. Their source
snapshot and generated runs are archived outside the project; see
`outputs/impedance_instron/LEGACY_ARCHIVE.json`. Old source entry points remain for
one deprecation cycle to avoid deleting existing API symbols without notice.
They are not the implementation of this controller and their old scores are not
comparable to the new tracking loss.
