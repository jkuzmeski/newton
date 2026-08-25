# Multi-rate and relaxation acquisition protocol

The current two tests both have an observed cycle period near 0.5 s. Their names
contain `140ms` and `185ms`, but those strings are not the measured cycle period.
Do not use them as rate metadata.

## Objective

Identify rate dependence independently of geometry and reserve entire physical
conditions for blind validation. The new data must distinguish equilibrium
stiffness, relaxation times, hysteresis, amplitude dependence, temperature, and
repeatability.

## Required metadata

Record specimen ID, shoe model, side, size, condition, machine, load cell,
calibration ID, operator, date, temperature, humidity, fixture geometry, control
mode, waveform, commanded and observed amplitude/rate, sample rate,
preconditioning, recovery, repeats, channel names, units, signs, saturation
limits, complete cycles, partial cycles, source hash, and processing decisions.

## Cyclic matrix

For each fixture, hold amplitude fixed and acquire complete stabilized blocks at:

```text
cycle period: 0.10, 0.25, 0.50, 1.0, 2.0, and 10.0 s
```

Use at least two independent blocks per period. Randomize or balance the order.
Log recovery time between blocks. Use a sample rate high enough to retain at
least 500 samples per cycle and to resolve the machine response. Record the
observed period from time rather than trusting a filename or command setting.

Repeat the matrix at low, nominal, and high compression amplitudes after the
single-amplitude rate study is complete.

## Relaxation matrix

For the same fixture and shoe, run hold-relaxation tests at low compression,
plateau, and densification levels. Use a controlled ramp and at least a 60 s
hold unless pilot data show that all relevant modes settle sooner. Sample the
ramp and early hold at high rate. A lower late-hold rate is acceptable if the
source time channel remains authoritative and continuous.

## Splits

Do not split adjacent cycles from one block for the final validation claim.
Reserve at least:

- one complete rate,
- one amplitude,
- one independent repeat or test day, and
- one independent mechanical test such as a guided drop or controlled rocker.

Keep these acquisitions sealed until model selection and thresholds are frozen.

## Predeclared metrics

Report every condition using peak-force error, active-region force NRMSE,
absorbed and returned energy, hysteresis error, relaxation error at fixed times,
and timing error. For impact tests also report peak force, impulse, maximum
compression, contact duration, and rebound. Do not silently change a threshold
after seeing the result.

## Quality control

Reject or explicitly quarantine nonmonotone clocks, missing units, NaN/Inf,
overload or saturation, incomplete selected cycles, undocumented sign changes,
and inconsistent calibration metadata. Preserve failed acquisitions and their
reason. Hash raw files before processing.
