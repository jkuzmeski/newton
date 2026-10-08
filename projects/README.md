# Shoe and runner project map

| Package | Responsibility | Start here |
|---|---|---|
| [Digital Instron](digital_instron_v2/README.md) | Prepare bench data, identify material, validate and export it | `uv run --no-sync -m projects.digital_instron_v2 --help` |
| [Digital Shoe](digital_shoe/README.md) | Portable artifacts, shared material/contact laws, shoe-only examples | `uv run --no-sync -m projects.digital_shoe --help` |
| [Impedance Instron](impedance_instron/README.md) | Shared generative Hogan runner, LM fit, frozen prediction | `uv run --no-sync -m projects.impedance_instron --help` |
| [Gait C3D](gait_c3d/README.md) | Separate motion/data adapters | See its input guide |

## Data flow

Bench measurements and geometry feed Digital Instron identification.
Its exported Digital Shoe artifact supplies the same material/contact laws
to standalone examples and the Hogan runner. Visual3D observations feed
offline runner identification; the frozen runner generates motion without
those observations.

## Where to edit

- `digital_shoe/material.py`, `contact.py`, `runtime.py`: shared physical laws.
- `digital_shoe/artifact.py`: portable records and validation.
- `digital_instron_v2/`: bench calibration, validation, and artifact export.
- `impedance_instron/hogan/runner.py`, `mechanics.py`: causal runner and chain.
- `impedance_instron/hogan/gpu_runner.py`, `gpu_mechanics.py`: CUDA equivalents.
- `impedance_instron/hogan/identify.py`, `least_squares.py`: observations and LM.
- `impedance_instron/hogan/generate.py`, `fit_report.py`: frozen prediction and report.
- `impedance_instron/cartesian/`: retained reference/profile preparation and
  shoe attachment, not an alternate controller pipeline.

The project routers forward options to the selected tool. No command shows
help; it does not launch training. Impedance `fit` also writes its HTML report.
The retired equilibrium-spline and reference-tracking workflows are removed.

## Provenance and structural checks

The preserved local full run is
`outputs/impedance_instron/generative_fit_lm_flightcom_20261008`.
Restricted motion and shoe data are not included in a source-only checkout;
see [asset provenance](../ASSET_PROVENANCE.md). Newton's public controllers and
shared Digital Shoe constitutive laws are unchanged.

Run the existing AST duplicate check without importing the projects:

```console
uv run --no-sync python scripts/check_shoe_project_duplicates.py --check
```

[SHARED_FUNCTIONS.md](SHARED_FUNCTIONS.md) describes shared computations and
non-interchangeable copies.
