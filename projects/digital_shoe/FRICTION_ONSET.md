<!--
SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
SPDX-License-Identifier: Apache-2.0
-->

# Supplied-signal friction observations

The old Cartesian-controller onset replay and comparison commands have been
retired. The independent signal-processing helper
[`friction_observation.py`](friction_observation.py) remains available with its
tests. It does not import a controller or integrate body dynamics.

## Match observations; do not filter applied force

`observe_friction_comparison()` compares supplied high-rate predictions and a
measured reference under a declared observation policy:

1. Requires `unfiltered_grf_target_n` and explicit filter provenance. This historical
   name means **pre-20 Hz**, not raw sensor data: the force was already tare-corrected,
   pooled, Hann-filtered and episode-assigned.
2. Applies a normalized Hann kernel to the full-rate prediction before native-clock
   resampling. Its physical duration comes from source metadata. The native 21-sample
   kernel spans 10 ms at 2 kHz; the fine-grid kernel is a declared approximation.
3. Excludes unsupported force endpoints instead of inventing a terminal force.
4. Reconstructs reference and prediction with the same fourth-order, forward/backward
   20 Hz Butterworth filter, common force clock, and odd-padding rule.
5. Retains signed Fx **and** Fz. Negative filtered Fz is a signal-processing artifact,
   not tensile normal contact in the simulation.
6. Defines stance from the separate pre-20 Hz measured normal signal, not clipped
   low-pass output.

No filter is applied to forces inside the shared contact or body integration
paths. Supplied raw curves and chatter diagnostics remain separate from the
observed signals. A partial prediction does not become complete merely because
it overlaps part of the reference.

Hann filtering on the shorter simulation window cannot reconstruct unavailable
full-trial context. Its zero-padding approximation and the Butterworth endpoint
padding are recorded. This matches the known linear processing, not an independently
identified force-platform transfer function. Do not infer material friction from
filtered Fx/Fz ratios near touchdown.

## Helper usage

For supplied reference and prediction arrays:

```python
from projects.digital_shoe.friction_observation import observe_friction_comparison

observations = observe_friction_comparison(
    reference,
    prediction_time_s,
    prediction_forces_n,
    forward_sign=1,
)
```

The helper validates time support, force shapes, and declared filter metadata
before returning observed signals and raw diagnostics. An explicit `filter_spec`
and Hann duration can be supplied for synthetic inputs. It does not score or rank
controller candidates. Raw and observed peaks must remain labelled separately;
a lower observation error is not an independent material calibration.
