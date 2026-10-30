# Digital Instron cleanup report

## Deleted

- Removed the 2025 phase-1 experiment inputs and outputs: the legacy `manifest_v2.json`, 2025 rearfoot/full-foot raw CSVs, `DigitalInstron/processed/` averaged/split traces and phase-2 dynamic traces, and their summary JSON.

- Kept `validation.py` because the shared `projects.digital_shoe.mechanics_report.figures` imports its held-out metrics; removed its obsolete phase-specific framing.

- Removed `projects.digital_instron_v2.dynamics` and `example` plus `test_digital_instron_dynamics.py`; these were demonstrations, not pipeline steps. Removed the package `view` subcommand because `projects.digital_shoe.showcase` already owns the artifact-only Instron/drop/rocker viewer.

- Removed phase-1 validation tests and stale backend/phase-2 figures, along with obsolete research notes `MODEL_FINDINGS.md` and `GEOMETRY_AND_MATERIAL_REPORT.md`.

- Removed historical fit-start files under `FoamSweep/fit/history/`, both `*_grounded_columns` comparison fits, the old fit animation, and kernel profiling JSON. Kept only the current per-shoe fit JSON and registration PNGs, processed sweeps, and regenerated `report.html`.

- Removed obsolete STL-derived `.obj` source for a prior prep path; the current shoe `source.glb` assets and prepared left/right grids remain.

## Simplified

- The package command now exposes only `fit` and `export`. Sweep splitting, shoe preparation, and report building remain explicit module commands since they are distinct data-production steps without wrappers in the package CLI.

- `sweep_fit` now starts from the calibrated material and the configured seeded Latin-hypercube starts only. Removed `--start`, hidden `--fix`, corresponding API parameters, and per-residual `history` records from fit output. This makes the documented fit command independent of files in `fit/history/`; its default constants remain six random starts, seed 0, 80 evaluations, and 8 substeps.

- `sweep_report` no longer reads/compares old-law or grounded-column fits, bound-20 backups, or optional kernel-profile files. It reports current fitted results and replays traces as before. Regenerated the committed HTML from the kept fit files.

- `geometry` now contains only `ColumnGrid`, raycasting/grid sampling, and neighbor indexing needed by shoe preparation, fitting and export; removed old mesh posing/surface and rearfoot-fixture helpers.

- Rewrote `projects/digital_instron_v2/README.md`, `DigitalInstron/README.md`, and `FoamSweep/README.md` to document the remaining pipeline only.

## Kept items and reason

- `sweep_split.py`, `cycle_windows.py`, `frame_qc.py`: generate the current train/validate/repeat traces from raw Instron sweep CSVs; tested by deterministic file reproduction.

- `shoe_prep.py`, `geometry.py`, `registration.py`: turn current midsole scans and the shoe last into canonical column grids and a fitted last pose.

- `core.py`, `last_replay.py`, `sweep_fit.py`: shared material, trace replay and current material fit used to produce `material_fit.json`.

- `sweep_report.py`: builds the current fit report, including fresh replay figures and diagnostics.

- `export_digital_shoe.py`: builds the portable shoe artifact, validation report and GIF outputs consumed downstream.

- `validation.py`: retained despite its location because `projects.digital_shoe.mechanics_report.figures` imports its trace metrics; its old phase-1 description is gone.

- Current shoe source scans (`source.glb`), prepared grids, current fits, current showcase/export examples, raw FoamSweep source data and processed sweep outputs remain.

## Verification

- `uv sync --extra dev` - passed in this worktree.

- `uv run --extra dev -m unittest newton.tests.test_digital_instron_core newton.tests.test_digital_instron_project newton.tests.test_digital_instron_registration newton.tests.test_digital_instron_shoe_prep newton.tests.test_digital_instron_sweep_fit newton.tests.test_digital_instron_sweep_report newton.tests.test_digital_instron_sweep_split` - **27 tests passed**.

- `uvx pre-commit run --files ...` for all changed Python/docs files - passed (Ruff, formatter, typo checks and repository hooks).

- `uv run --extra dev -m projects.digital_instron_v2 --help` and `... fit --help` - passed; only `fit` and `export` are listed at package level; fit help contains no `--start` or `--fix`.

- `uv run --extra dev -m projects.digital_instron_v2.sweep_split` - passed for Dev3Elite and FR3 (30 conditions each). Follow-up `git diff --exit-code -- DigitalInstron/FoamSweep/processed` passed, so processed outputs are bit-for-bit unchanged from the committed files.

- `uv run --extra dev -m projects.digital_instron_v2.sweep_report --out .scratch_report.html` - passed; produced a 1.66 MB standalone report. The generated `DigitalInstron/FoamSweep/fit/report.html` was rebuilt after final edits; it contains no old-law, grounded-columns, bound-20, kernel-performance, or phase-1 comparisons.

- Export gate: `uv run --extra dev -m projects.digital_instron_v2 export --shoe DigitalInstron/puma-fast-r-nitro-elite-3 --fit DigitalInstron/FoamSweep/fit/puma-fast-r-nitro-elite-3/material_fit.json --sweep DigitalInstron/FoamSweep/processed/FR3 --output .scratch_export_final` - passed on CUDA. Exported `digital_shoe.json` has identical `column_bed` and `constitutive_model` (material) objects and the same top-level keys as `DigitalInstron/digital_shoe_showcase/digital_shoe.json`.

## Parent decisions / notes

- `ASSET_PROVENANCE.md` is parent-owned and was not edited. It likely needs entries removed or revised for the deleted 2025 phase-1 inputs and removed OBJ/research figures.

- `pyproject.toml`, `.pre-commit-config.yaml`, root README, `projects/README.md`, changelog and asset provenance files were not edited per instruction. Check whether project catalogs, asset checksums, or changelog references should be updated.

- Kept `validation.py` because another current report component imports it. Removing or relocating this shared helper would require a change to `projects/digital_shoe`, which is outside this worktree assignment.

