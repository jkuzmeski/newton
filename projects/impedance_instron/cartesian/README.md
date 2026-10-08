# Runner inputs and shoe attachment

This directory retains only the infrastructure used by the
[generative Hogan pipeline](../README.md):

- `visual3d.py`, `prepare_visual3d.py`, `prepare_dataset.py`: export ingestion,
  endpoint registration, and peak-to-peak observations.
- `data.py`, `profile.py`: validation/loading of existing observation and inertial schemas.
- `shoe.py`: rigid attachment to the shared Digital Shoe runtime.
- `gpu/foundation.py`: its batched CUDA adapter.

The directory name and existing data schemas are retained so saved datasets
remain usable. Cartesian controller models, splines, searches, and adjoint
experiments are removed; runner dynamics live exclusively in `hogan/`.
