# S001 calibrated default subject

This is the tracked default static-calibration realization of the S001 base
subject. It uses raw bilateral PSIS orientation, CODA/Bell--Brand hip
regression, source-joint-normalized segment geometry, hallux toe endpoints, a
posterior-pelvis-to-head torso calibration, three bounded
rotational torso axes, native marker-supported knee limits, and flat-foot
contacts. The source `s001_base` template remains separate so future subjects
can be rebuilt from the same base.

Dynamic native motion artifacts produced by `native_motion_fit` are stored in
this bundle under `motions/<trial>_native_motion/` by default. This keeps the
subject model, calibration, marker layout, and fitted trials together.
