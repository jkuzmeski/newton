# Treadmill-to-overground conversion plan

Status: phases P1 through P5 are implemented; see `README.md`. P6 is not
started.

Goal: convert the treadmill trials (`Trial 101.v3d.c3d` + `tm0001.txt`) into a
world-referenced overground motion, so the fitted subject walks and runs across
the ground instead of marching in place.

## 1. What the data actually contains

Measured from the local S001 and S014 source captures used to validate this
implementation:

- `tm0001.txt` is a D-Flow controller log at 300 Hz. `Time` is controller
  uptime, not trial time and not wall clock.
- `leftbelt_distance` and `rightbelt_distance` are **bit-identical** in both
  subjects. The protocol is **tied belt**, not split belt.
- Distance is the controller's own integral of speed. It matches a
  piecewise-linear re-integration to 2e-6 m over 345.5 m (6 ppb). There is no
  drift to remove.
- Protocol in both subjects: stand, ramp at +0.5 m/s^2 to 1.5 m/s, 67 s walk,
  ramp to 3.0 m/s, 67 s run, ramp down to 1.0 m/s, 26 s walk, ramp to zero.
  Total travel 345.55 m.
- `platform_pitch` and `platform_roll` never leave +/-0.004. The platform is
  level for these trials, so no incline rotation is needed yet.
- Synchronization: no trigger channel and no C3D events. Trial durations agree
  within 12 ms, and three independent estimators bound the offset to
  -0.025 s .. +0.070 s. Use `offset = 0` with +/-0.05 s uncertainty.
- Lab fore-aft is **+Y** (belt surface travels +Y, subject faces -Y). With the
  pipeline defaults (`up=+Z`, `forward=-Y`) the overground shift is a growing
  **+X** in the Newton world frame.

## 2. What carries over from the paper

Jung & Lee, *Sensors* 2021, 21(3), 786
(https://doi.org/10.3390/s21030786).

Carry over:

- The **virtual origin** formulation. A virtual origin moves backward with the
  belt, and overground position = lab position - virtual origin position
  (their Eq. 6-7). This is exactly the transform we need.
- Their finding that the map is a pure time-varying **translation**. Joint
  angles are invariant; only the root translation and the fore-aft velocity
  change.

Do **not** carry over:

- Their belt measurement method. They glue a chain of 14 markers on the belt
  surface and track it optically, because a consumer treadmill gives no speed
  signal. Our C3D has no belt-surface markers, so their estimator cannot run.
- Their re-indexing and sag-projection steps (Eq. 1, 4, 5). Those only exist to
  clean up the optical belt-marker chain.

Our replacement input is better than theirs: directly logged belt speed and
distance channels. The implementation integrates the piecewise-linear speed and
uses the distance channel to verify tied-belt operation. Their own validated
accuracy was 0.3-1.0% of travel distance; our belt command tracks the real belt
to better than 1% when checked against stance-foot marker speed.

The `Platform:` markers (FLeft, FRight, ORight, BLeft, BRight) give the paper's
treadmill frame {TR}. We use them to define the belt travel axis and to check
lab-vs-belt yaw, not to measure belt motion.

## 3. Where it goes: after IK, not before

Apply the shift **after** the IK solve, to the fitted free-root translation.

Reason it is exact: joint 0 is a `FREE` joint with identity parent and child
transforms, so `joint_q[0:3]` is literally the pelvis world position. Forward
kinematics is equivariant under a root translation. Shifting every marker
before IK and shifting the root after IK give the **same joint angles** in exact
arithmetic.

Reason after-IK is better in practice:

| | before IK | after IK |
|---|---|---|
| IK seed distance | up to 345 m from target, fixed 40 LM iterations | unchanged |
| float32 conditioning | residuals become differences of large numbers | unchanged |
| marker residual gates | must be relaxed | bit-identical |
| static calibration risk | can silently bias segment lengths | none |
| velocity | correct | correct (belt speed enters `joint_qd` for free) |

So: solve IK in the original bounded lab frame, then translate.

## 4. Design

### 4.1 New module `projects/gait_c3d/treadmill.py`

- `load_treadmill_log(path) -> TreadmillLog`: parse the tab-separated log,
  keep the raw 300 Hz samples, no filtering (the signal is noise-free;
  filtering only smears the 8 ramp corners).
- `belt_motion_for_frames(log, frame_count, rate=100.0, offset=0.0,
  side="auto") -> BeltMotion` with per-frame distance, offset, speed, and
  coverage arrays.
  Interpolate on **time**, never on sample index. Linear interpolation for
  speed is exact because the reference is piecewise linear; integrate the
  piecewise-linear speed exactly for distance. Measured accuracy on the 100 Hz
  grid: max per-frame error 1.3e-5 m, 2.4e-5 m cumulative over 345.55 m.
- Tied-belt guard: raise if `max|left - right| > 1 mm`. A single virtual origin
  is only valid for tied belts. Split belt needs a per-foot formulation and is
  out of scope for v1.
- Belt axis: by default, derive subject-backward from the declared C3D forward
  axis. An explicit laboratory travel vector remains available for other rigs.

### 4.2 Integration in `native_motion_fit.fit_c3d_marker_motion()`

Between the solve and `finite_difference_joint_qd()`:

1. `joint_q[:, q0:q0+3] += d(t)` for the free root only.
2. Add the same `d(t)` to `targets` and `predictions`, so replay overlays stay
   in one frame. Residual statistics are differences, so they do not change.
3. Let `finite_difference_joint_qd()` run on the shifted coordinates. The belt
   velocity then lands in the root linear DOFs automatically and correctly.

### 4.3 Artifact schema

- Add a `treadmill` block to the sealed manifest: source file and sha256, log
  sample rate, sync offset, belt axis in Newton frame, side policy, total
  distance, applied stage, and the tied-belt check result.
- Bump `gait_native_motion_artifact_1` -> `_3`. Version 2 introduced the
  treadmill block but retained float32 coordinates; version 3 stores coordinates
  in float64. Keep the loader accepting versions 1 and 2.
- Do **not** fold the shift into `registration` (a single 4x4 cannot express a
  growing translation) and do **not** reuse `ground.global_offset_m` (pinned at
  atol 2e-6 by existing tests).
- Keep the exactly-two-files rule for the motion directory. No sidecar file.
- Store the root translation in float64 in the artifact. float32 spacing at
  345 m is ~3e-5 m, which is coarser than the 1e-4 m marker gates.

### 4.4 CLI and replay

- `example_native_motion_fit.py`: `--treadmill-log PATH`, `--belt-offset`,
  `--belt-side {auto,left,right,mean}`, and `--no-overground` to keep the old
  behavior. Default: overground on when a log is found beside the C3D or in the
  subject bundle.
- The replay camera is fixed and Newton has no follow camera, so the subject
  leaves the view within seconds. Add a simple per-frame camera offset in the
  example, or offer `--camera follow`.

## 5. Phases

1. **P1 Belt input. Done.** `projects/gait_c3d/treadmill.py` with
   `newton/tests/test_gait_treadmill.py`.
2. **P2 Replay demo. Done** through P3; the example replays the overground
   artifact directly.
3. **P3 Pipeline integration. Done.** Post-IK shift in
   `fit_c3d_marker_motion()`, sealed manifest block, schema
   `gait_native_motion_artifact_3`, CLI flags, README.
4. **P4 Validation gates. Done** as unit tests, plus the measured S001 numbers
   in section 6.
5. **P5 Camera. Done.** The replay camera follows an overground motion;
   `--camera fixed` holds the view.
6. **P6 (future) Force plates.** Decode analog GRF, translate the COP/point of
   application by the same `-d(t)`, leave force and free moment unchanged,
   recompute moments about a fixed origin, gate on Fz.

## 6. Validation gates

Measured on S001 frames 800-1400 (6 s of 1.5 m/s walking):

| gate | target | measured |
|---|---|---|
| stance heel fore-aft speed | < 50 mm/s | 20 mm/s left, 27 mm/s right |
| root travel vs belt travel | within 1% | 9.038 m vs 8.985 m (0.6%) |
| mean root fore-aft velocity | within 1% of command | 1.509 m/s vs 1.500 m/s |
| joint-angle invariance | bit-identical | bit-identical, unit test |
| marker residual invariance | bit-identical | bit-identical, unit test |
| belt resampling error | < 1e-4 m per frame | 1.3e-5 m maximum |

The 0.6% surplus travel is the subject drifting forward on the belt, which is
real motion and not an artifact of the transform.

Static safety: the transform is only reachable through
`fit_c3d_marker_motion()`, which fits dynamic trials. The static calibration
path never receives a belt argument.

## 7. Risks and limits

1. **The belt channel is a command, not an encoder.** Holds are exactly
   1.500000 m/s. Stance-foot marker speed says the real belt tracks it to
   +0.15% / 0.0% / -0.27% at the three speeds. So expect up to ~1% residual
   scale error, comparable to the paper's own 0.3-1.0%. Optional refinement:
   estimate a per-segment scale factor from stance-foot ZUPT and record it in
   the manifest as a correction, off by default.
2. **Sync uncertainty is +/-0.05 s.** During constant speed this is only a
   constant origin shift and does not matter. It only matters at the 8 ramp
   corners.
3. **Tied belt assumption.** Guarded by an explicit check. Split belt needs a
   per-foot virtual origin.
4. **Non-inertial frame during ramps.** Constant belt speed is a Galilean
   boost, so dynamics are safe. During the 4 ramps (+/-0.5 m/s^2) the mapped
   frame accelerates. Run inverse dynamics in the lab frame, or restrict
   dynamic analysis to the constant-speed windows.
5. **Data hygiene.** `opensim_subject --overwrite` preserves subject-local
   `.c3d` captures and `.txt` treadmill logs while replacing generated output.
