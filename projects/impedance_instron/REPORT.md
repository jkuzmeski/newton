# Two-stiffness workflow: implementation and validation

This replaces the retired six-action/momentum/work-optimized experiment narrative.
The preserved prior source and outputs are located by
`outputs/impedance_instron/LEGACY_ARCHIVE.json`.

**Status:** the baseline and 1.1056-loss training result below are historical,
measured **before the passive-shoe and external-ground-wrench repair**. They do
not qualify the current physics. Re-evaluating the same unchanged weights after
the repair gives loss **7.46839**, not 1.1056. The repaired run passes the existing
mechanical safety checks; that is not tracking, transfer or upper-contact
validation. See [WORKTREE_SUMMARY.md](WORKTREE_SUMMARY.md) for the full worktree
history and current evidence.

## Contract

- Learned outputs: leg stiffness and ankle rotational stiffness only.
- Frozen: inverse-dynamics equilibrium length/angle, damping ratios, reference
  processing, rig settings and policy observation scales.
- Reward: fixed-time integral of normalized measured pelvis-centroid height and
  foot-pitch squared error. Pelvis maps to the upper lump, not true whole-body COM.
- Evaluation only: GRF, actuator source/damper work, contact and compression.
- No prescribed pelvis/pitch state, force feedforward, momentum/work objective,
  per-shoe equilibrium optimization, learned phase, or retrospective time warp.

## Offline inverse dynamics

The selected optical profile covers the source stance 90.0115–90.3065 s, with
padding giving a 0.375 s episode. Measured 2% BW contact is 0.038–0.364 s locally.
Optical smoothing is the declared 12 Hz second-difference filter plus C2 cubic.
Nominal stiffnesses are 12000 N/m and 4000 N·m/rad. Damping ratios are 0.25 and 0.5.

The construction uses measured platform force and moment OFFLINE, and solves
`B*e_dot + K*e = inverse_load` for equilibrium offsets. It does not use
force-integrated COM as a target or initial state. The full reference stores raw
optical knots, processing, geometry mapping, initial conditions and residuals.

Full-motion consistency residuals are substantial: upper orthogonal force RMS
295.44 N (X 284.85 N, Z 78.40 N), and foot balance RMS 306.16 N (X 163.42 N, Z 258.90 N).
These diagnose the proxy/mounting/model mismatch for the FULL recorded motion.
They are not a proof or lower bound for the rewarded pelvis-height/foot-pitch
error, because other translations are free to differ.

## Baseline (historical, pre-contact-repair)

The first native nominal run completed 45 frames × 64 substeps, 0.375 s, without safety
flags. CPU tracking loss was 6.50981, pelvis height RMS 19.16 mm, and pitch RMS
0.01289 rad (0.738°). GPU 8-world graph replay gave loss 6.50989 in each world.
Peak vertical GRF was 2184.66 N. Simulated touchdown was 0.08294 s, later than the
reference 0.038 s; the clock was NOT shifted to hide that error.

These are implementation baselines, not fitted-human or same-shoe validation.
Subsequent training results and full-resolution traces live under
`outputs/impedance_instron/simple/`. See `README.md` for reproducible commands.

## Short training and frozen replay (historical, pre-contact-repair)

A 100-iteration, 32-world PPO integration run (seed 0) selected iteration 100.
With that earlier source, strict frozen GPU replay reproduced saved tracking loss
1.1055970700613216 exactly, against nominal 6.509887522334793. Both had no physical
safety flags under the earlier implementation. These scores are not applicable
after the contact repair.

In that earlier run, pelvis RMS improved 19.16 → 7.71 mm and pitch RMS
0.738 → 0.655 degrees. Peak GRF increased 2184.67 → 2424.06 N: force is deliberately
an evaluation output, not an optimized waveform. This was a short known-shoe run,
not convergence or transfer validation.

At that implementation stage, the active regression suite passed 48 tests.
A real checkpoint restoration check also confirmed exact CPU replay, explicit
synthetic material-only substitution,
and repeated CUDA graph replay with identical full traces. CPU/GPU scores differ
slightly due to numerical execution; cross-device bitwise identity is not claimed.
The OpenGL headless example completed its test and produced the registered 320×320
screenshot. Reports and full-resolution traces are linked from the local offline
landing page at http://127.0.0.1:8000/ when the stdlib server is running.

Reused offline/geometry/foundation regressions: 65 tests run, 62 passed and 3
legacy input-dependent tests skipped. Active feature tests had no skips.

The historical integration review approved the fixed-clock two-stiffness software contract.
It predates the contact repair and is not current physical qualification.

## Re-evaluation after the contact repair

The retained policy was re-evaluated for the same 0.375 s episode with explicit
`--allow-physics-update` consent. The repair and subsequent shared-law
consolidation reports agree: tracking loss 7.468390, pelvis RMS 20.574 mm,
foot-pitch RMS 0.6024 degrees, and peak vertical GRF 2617.625 N. The run is finite,
uses captured CUDA replay, and has no existing mechanical safety flags. The
checkpoint report sets `checkpoint_scores_applicable` to `false`.

No policy weights, material, source geometry or measured reference were retuned.
All 299 passive columns now follow the carrier in final flight. External ground
pressure, friction capacity, COP and the complete ground wrench use the declared
contact plane; signed internal neighbor transfer remains a separate diagnostic.

The upper interface is still idealized backing over the fixture footprint, not
solved gap-aware contact with the rigid-last mesh. Good tracking after retraining,
material transfer, anatomical registration and same-shoe validation remain
unestablished. Source changes require new evaluation; the opt-in does not carry
old qualification forward or relax other frozen-input checks.

Evidence is local:
`outputs/impedance_instron/passive_attachment_fix/FIXES.md` and
`outputs/impedance_instron/consolidation/impedance/summary.json`.
Use the re-evaluation commands in [README.md](README.md).
