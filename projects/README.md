# Shoe and Instron project map

Three packages share the same shoe mechanics. Choose the package by the task,
not by the age or size of a source file.

| Package | Responsibility | Start here |
|---|---|---|
| [`digital_instron_v2`](digital_instron_v2/README.md) | Prepare bench data, identify material, validate and export it | `uv run --no-sync -m projects.digital_instron_v2 --help` |
| [`digital_shoe`](digital_shoe/README.md) | Own portable artifacts, shared material/contact laws and shoe-only examples | `uv run --no-sync -m projects.digital_shoe --help` |
| [`impedance_instron`](impedance_instron/README.md) | Run the current two-stiffness experiment using the identified shoe | `uv run --no-sync -m projects.impedance_instron --help` |

[`gait_c3d`](gait_c3d/README.md) supplies separate motion/data adapters. It is not
another shoe constitutive or contact implementation.

## Dependency and data flow

```text
bench measurements + geometry
        |
        v
Digital Instron preparation / identification / validation
        |                           |
        | portable artifact         | calls shared mechanics
        v                           v
Digital Shoe artifact + material/contact/runtime/calibration
        |
        v
Impedance Instron current experiment (simple/)
```

Digital Shoe does not import the fitting project or the impedance controller.
Digital Instron and the impedance experiment both call its shared laws. Artifact
production flows from Instron to Shoe; code reuse points back to the shared Shoe
package. Those are different directions for different purposes.

## Common tasks

All commands run from the repository root in its environment. Existing
module-specific commands remain supported with the same flags and defaults.

| Task | Package command | Existing equivalent |
|---|---|---|
| Historical averaged-cycle fit, without a held-out split | `python -m projects.digital_instron_v2 fit ...` | `python -m projects.digital_instron_v2.workflow ...` |
| Fit training cycles and check held-out cycles | `python -m projects.digital_instron_v2 validate ...` | `python -m projects.digital_instron_v2.phase1 ...` |
| Fit and validate dynamic bench replay | `python -m projects.digital_instron_v2 replay ...` | `python -m projects.digital_instron_v2.phase2 ...` |
| Identify, validate and export a portable shoe | `python -m projects.digital_instron_v2 export ...` | `python -m projects.digital_instron_v2.export_digital_shoe ...` |
| View source-backed mechanical examples | `python -m projects.digital_instron_v2 view ...` | `python -m projects.digital_instron_v2.example ...` |
| Profile resident forward calibration against a saved baseline | `python -m projects.digital_instron_v2 profile ...` | `python -m projects.digital_instron_v2.profile_calibration ...` |
| View an artifact-only shoe/Instron/drop/rocker | `python -m projects.digital_shoe view ...` | `python -m projects.digital_shoe.showcase ...` |
| Rebuild an artifact's offline report | `python -m projects.digital_shoe report ...` | `python -m projects.digital_shoe.report ...` |
| Record the artifact examples | `python -m projects.digital_shoe record ...` | `python -m projects.digital_shoe.record_gifs ...` |
| Check an acquisition manifest | `python -m projects.digital_shoe check-acquisition ...` | `python -m projects.digital_shoe.acquisition ...` |
| Prepare/run/train/evaluate the active impedance experiment | `python -m projects.impedance_instron <command> ...` | Current implementation in `simple/` |
| View the active impedance rig | `python -m newton.examples impedance_stiffness ...` | `python -m projects.impedance_instron.simple.example ...` |

Use `uv run --no-sync` before these commands. A package invoked with no command
shows help; it does not start a fit or simulation. A command followed by `--help`
shows its existing tool's options. The routers do not duplicate parsers or silently
change training splits, optimizers, numerical settings or output locations.

## Where to edit

### Shared shoe mechanics: `digital_shoe/`

- `material.py`: the shared two-term Ogden–Hill/Maxwell expressions, including
  NumPy and Warp adapters of the same source.
- `contact.py`: unilateral reaction, neighbor coupling, passive balance, bristles,
  contact kinematics and wrench mapping.
- `runtime.py`: mutable live simulation and shared low-level operations.
- `calibration.py`: resident forward-calibration workspace and graph scheduling.
- `artifact.py`, JSON schemas: portable records and validation.
- `rendering.py`, `showcase.py`, `report.py`, `record_gifs.py`: presentation and
  runnable demonstrations. Report diagrams remain beside their reader.
- `acquisition.py`, `ACQUISITION_PROTOCOL.md`: input metadata and future data plans.

The public package facade imports named exports only when requested. Importing
acquisition/provenance tools or asking for package help does not load the physics
stack. Artifact loading and actual mechanics still require their normal dependencies.

### Identification and validation: `digital_instron_v2/`

- `geometry.py`, `frame_qc.py`, `cycle_windows.py`: input preparation.
- `core.py`, `workflow.py`: fitting problem, bounded optimizer and legacy workflow.
- `phase1.py`, `validation.py`: train/hold-out protocol and official metrics.
- `phase2.py`: dynamic bench replay validation.
- `dynamics.py`, `example.py`: source-backed boundary geometry and demonstrations.
- `scenario_common.py`: shared scenario defaults, quaternion operations and
  the pure attachment-PD expression used by both forward and tape-safe adapters.
- `dynamics_diff.py`, `inverse_id.py`, `scenarios_diff.py`: tape-safe adapters,
  implicit gradients and research scenarios; they use the shared Shoe laws.
- `export_digital_shoe.py`: artifact production.
- `profile_calibration.py`, `plot_backends.py`, `plot_phase2.py`: performance and
  optional visualization tools.

The versioned package name and existing module paths are retained. A directory
reorganization is not a reason to rename public material records or Warp kernels.

### Current experiment versus retired code: `impedance_instron/`

- `__main__.py` and `simple/`: current fixed-clock, two-stiffness workflow.
- `profile.py`, `orientation.py`, `trajectory.py`, `variability.py`: preparation
  and geometry helpers used by the current workflow. They are **not legacy**.
- `mcclough.py`: optional published-material analysis through the shared law.
- `legacy/`: retired controller, optimizer, environment, trainer and reports.
- Old root module names such as `env.py` and `train.py`: compatibility aliases,
  not second copies of the retired implementations.
- `REPORT.md`: current results and qualification boundary.
- `LEGACY_REPORT.md`: historical report template.
- `WORKTREE_SUMMARY.md`: cross-generation implementation history and evidence.

Read [`legacy/README.md`](impedance_instron/legacy/README.md) before using an old
controller or checkpoint. Do not attach historical figures/scores to current runs.

## Data, provenance and compatibility

Generated artifacts, checkpoints, traces and report outputs stay under local
ignored data directories. They are not moved by this cleanup or supplied by a
source checkout. See [`ASSET_PROVENANCE.md`](../ASSET_PROVENANCE.md) for restrictions.
The local legacy archive is optional historical evidence, not the implementation.

Keep active `simple/` code, its preparation helpers and the shared physics source
files in place: their exact bytes/paths participate in frozen identities. Old
module imports and CLI commands continue to work through aliases. A moved legacy
implementation has a new defining module/file and may get a new diagnostic source
hash; old generic pickle lookups remain supported, but new pickles are not promised
to load in a checkout that predates `legacy/`. Checkpoint checks are not bypassed.

This organization pass does not change the two-term law, calibrated material,
contact mechanics, fitting objective/optimizer, reference, policy, or unresolved
upper-last seating assumptions. Advanced fitting/replay changes should be reviewed
and measured separately.

## Duplicate-function review

Use the AST-based review tool to find exact/renamed function clones and near
matches without importing or running the projects:

```bash
uv run --no-sync python scripts/check_shoe_project_duplicates.py --check \
  --output outputs/impedance_instron/organization_review/duplicates.json
```

`projects/shoe_duplicate_review.json` records the remaining exact groups and why
they are retained. The check fails if an additional exact/renamed algorithm copy
appears outside those reviewed groups. Near matches are candidates for human
review, not automatic replacement: opposite loop-area signs, different
interpolation boundaries, nonfinite-display versus strict JSON, and tape-safe
state layouts are not equivalent just because their code looks similar.

See [SHARED_FUNCTIONS.md](SHARED_FUNCTIONS.md) for the actual consolidations, remaining reviewed copies and non-interchangeable near matches.
