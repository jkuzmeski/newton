# Bilateral runner results (F01, 3.65 m/s)

All scores come from deterministic, open-loop free runs started at each
segment's three-frame observed prefix. No reference input enters the dynamics
or the policy. "Held-out" means the evaluation block of
`outputs/bilateral/dataset_6step`, which no training run used.

## Multi-stride segments (2 s, 6 contacts)

| model | split | segments | failed | mean loss | peak Fz error [N] | contact duration error [ms] |
|---|---|---:|---:|---:|---:|---:|
| PPO run 1 | train | 50 | 0 | 50.2 | 360 | −8.7 |
| PPO run 1 | held-out | 6 | 0 | 46.4 | 336 | −9.3 |
| PPO run 3 (it 500) | held-out | 6 | 0 | 45.1 | 327 | −5.3 |
| Bilateral linear Hogan (LM, 1.2 s curriculum) | train | 50 | 50 | 1440 | – | – |
| Bilateral linear Hogan | held-out | 6 | 6 | 1512 | – | – |

The neural policy completes every 2-s, six-contact free run, train and
held-out. The 119-parameter shared linear schedule, fitted with LM through a
horizon curriculum (0.45, 0.8, 1.2 s), still fails every full segment.

## Comparison with the single-leg runner

The frozen single-leg runner (`generative_runner_score_20261010.json`)
re-scores exactly: 10.3085 train and 10.1617 held-out. All 9 of its held-out
stances, however, lie inside bilateral training segments.
[`compare.py`](compare.py) therefore scores every model on the 18 single-leg
stances outside all bilateral training. These are held-out for the bilateral
models but in-sample for the single-leg runner, which favors the single-leg
runner. A bilateral model's loss here is the single-leg runner's loss
restricted to the right leg and foot.

| model | loss | root | angles | GRF | peak | hip / knee / ankle RMSE [deg] | Fz RMSE [N] |
|---|---:|---:|---:|---:|---:|---|---:|
| single-leg runner (in-sample) | 9.8 | 3.31 | 5.23 | 0.96 | 0.11 | 8.1 / 4.5 / 4.7 | 90 |
| PPO run 1 | 27.7 | 0.41 | 15.05 | 7.61 | 4.14 | 7.0 / 6.2 / 19.0 | 340 |
| PPO run 2 (it 600, per-term kernels) | 34.3 | 0.25 | 11.61 | 7.59 | 14.02 | 7.1 / 7.2 / 15.2 | 342 |
| PPO run 3 (it 500, + force penalty) | 26.0 | 0.26 | 11.96 | 4.50 | 8.36 | 8.6 / 7.0 / 14.8 | 255 |
| bilateral linear | 59.3 | 0.86 | 15.98 | 5.63 | 33.43 | 16.2 / 7.1 / 9.2 | 295 |

The bilateral policies track the hip trajectory about 8x better (root term 0.3-0.4
vs 3.3) and match the hip and knee angles. Two terms account for their
higher loss: the ankle angle and the impact force.

## Ankle diagnosis (PPO run 1)

[`ankle_diagnosis.py`](ankle_diagnosis.py) splits the ankle error
`q - q_ref` into a command error `q_eq - q_ref` and a following error
`q - q_eq`, by gait phase, over the 50 training segments.

| phase | ankle RMSE [deg] | command bias [deg] | following bias [deg] | median K [N m/rad] | median D [N m s/rad] |
|---|---:|---:|---:|---:|---:|
| stance | 15-18 | −20 to −24 | +10 to +12 | 280 | 7.3 |
| early swing | 13-15 | **+40 to +43** | **−38 to −43** | 280-310 | **23** |
| late swing | **32-42** | **+22 to +33** | +1 | 220-250 | 6.5-7.1 |
| last 30 ms before touchdown | 4-5 | +6 | −3 to −7 | 250-263 | 4.7-6.8 |

- **The ankle reaches the right angle at touchdown.** The error is in swing.
- **In early swing the policy overdrives q_eq.** It aims about 40 deg ahead
  of the reference, and the ankle lags q_eq by almost the same amount.
- **In late swing the ankle follows q_eq** (bias about 1 deg), but q_eq is
  23-34 deg too dorsiflexed. The ankle then snaps back at 6-12 rad/s in the
  last 100 ms, against 1-4 rad/s measured.

### Ruled out: clearance and actuator lag

- **Toe clearance.** In the policy's own swings (middle two thirds), setting
  only the ankle to the measured angle and keeping every other joint as
  simulated gives zero scuffing frames in 200 swings. The median minimum
  clearance is 28 mm, against 30 mm as simulated.
- **Actuator lag.** The foot's inertia about the ankle is 0.0097 kg m^2, so the
  impedance and the first-order torque lag set its response. Runs 1-3 used the
  unfitted seed's 25 ms response time; the fitted single-leg runner uses 5.3 ms.

  | response time | K, D | step overshoot | 10-90 rise | gain and phase at 6 Hz |
  |---|---|---:|---:|---|
  | 25 ms | 250, 6 | 20 % | 38 ms | 1.17, ?84 deg |
  | 25 ms | 250, 23 | 0 % | 150 ms | (overdamped) |
  | 5.3 ms | 250, 6 | 0 % | 38 ms | 0.85, ?49 deg |
  | 5.3 ms | 400, 6 | 3 % | 21 ms | 0.98, ?33 deg |

  The 25 ms lag explains the early-swing overdrive. With the corrected
  5.3 ms actuator, however, run 4 at iteration 500 commands even more swing
  dorsiflexion (late-swing q_eq bias +53 deg against +24 deg for run 1 at
  iteration 500). The dorsiflexed swing is therefore a learned strategy, not
  lag compensation. The 5.3 ms value stays because it is the fitted one.

### The dorsiflexed swing is a landing strategy

Across 202 swing-to-contact pairs, more late-swing dorsiflexion goes with a
smaller next impact peak:

| | correlation with next peak excess |
|---|---:|
| late-swing ankle error | ?0.39 |
| late-swing q_eq error | ?0.38 |
| touchdown ankle error | ?0.59 |
| touchdown ankle D | ?0.30 |

The more dorsiflexed half of the swings land with a 622 N excess, against
737 N for the other half. The policy lands the heel with the foot dorsiflexed
and lets the ankle plantarflex under load. It uses the ankle as the shock
absorber because the heel arrives too fast. It trades swing-ankle tracking for
a smaller force error, which the reward weighs more.

### What sets the impact peak

Intervention runs ([`ankle_interventions.py`](ankle_interventions.py), 50
training segments):

| intervention | mean loss | failed | touchdown peak excess [N] | swing shoe > 50 N |
|---|---:|---:|---:|---:|
| none | 50.2 | 0 | 860 | 0.13 % |
| ankle q_eq held 60 ms after touchdown | 53.7 | 0 | 795 | 0.13 % |
| ankle K held 60 ms after touchdown | 49.6 | 0 | 827 | 0.13 % |
| ankle D held 60 ms after touchdown | 49.9 | 0 | 849 | 0.13 % |
| swing ankle aimed at the measured angle | 1928 | 50 | 748 | 1.02 % |

Fixing any ankle impedance channel at touchdown barely changes the spike. When
the swing ankle is aimed at the measured angle, every run fails the shoe
compression screen, on average 160 ms after a touchdown. That happens in
stance, a state this policy never trained in, not at impact.

The landing itself (250 contacts each, excluding the first from
initialization):

| at touchdown | heel vx [m/s] | heel vz [m/s] | foot rate [rad/s] | ankle rate [rad/s] | peak Fz, first 60 ms [N] |
|---|---:|---:|---:|---:|---:|
| measured | 0.80 | ?0.78 | ?6.8 | ?1.06 | 1150 |
| simulated | 1.08 | ?1.27 | ?4.6 | +0.88 | 1916 |

The heel lands 0.49 m/s faster downward. Contributions to that excess:

| coordinate | knee | pelvis | root descent | ankle | hip |
|---|---:|---:|---:|---:|---:|
| excess heel vz [m/s] | ?0.26 | ?0.14 | ?0.10 | ?0.07 | +0.09 |

The joint-angle and joint-rate rewards never see the heel's velocity. Run 5
adds ProtoMotions-style leg-point position and velocity kernels for this
reason.

## Runs 4 and 5 (stopped early)

Both runs train from scratch with run 1's pooled kernels and no force penalty:

- **Run 4:** run 1 without the symmetry loss, with the fitted 5.3 ms torque
  response.
- **Run 5:** run 4 plus leg-point position (0.3) and velocity (0.1) kernels,
  contact-match penalty 0.1, action smoothness 0.02, and the previous action as
  an input.

Both were stopped on request: run 4 at iteration 640 (last checkpoint 600) and
run 5 at iteration 540 (last checkpoint 500). Neither has been evaluated with
free rollouts. Training metrics at matched iterations:

| run | iteration | episode length [s] | termination fraction | pose kernel | force kernel | position kernel |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 500 | 0.76 | 0.51 | 0.15 | 0.67 | 0.61 |
| 4 | 500 | 0.60 | 0.74 | 0.21 | 0.68 | 0.65 |
| 5 | 500 | 0.80 | 0.51 | 0.29 | 0.61 | 0.56 |

Run 4's iteration-500 checkpoint fails all 50 training free runs (run 1's
fails 22 of 50 at the same iteration). It also commands more swing
dorsiflexion (late-swing q_eq bias +53 deg). Run 5 has the best pose
tracking at matched iterations.

To resume:

```console
uv run --no-sync --extra torch-cu12 -m projects.impedance_instron train-policy --dataset outputs/bilateral/dataset_6step --output outputs/bilateral/ppo_run5b --resume outputs/bilateral/ppo_run5/checkpoint.pt --reward-kernels pooled --force-penalty 0 --body-position-weight 0.3 --body-velocity-weight 0.1 --contact-match-penalty 0.1 --action-smoothness-penalty 0.02 --previous-action-obs
```
