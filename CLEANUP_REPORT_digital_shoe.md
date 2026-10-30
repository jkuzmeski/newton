# Digital Shoe cleanup report

## Deleted

- Standalone `friction_*` comparison/observation/solver experiment modules, the sample image, and their seven test modules. This experiment was independent of `ColumnBed`; the retained ground-bristle friction is part of the production column bed.
- Acquisition-manifest validator, its example/schema, and `ACQUISITION_PROTOCOL.md`; the acquisition plan is not part of the shoe-artifact-to-runner pipeline.
- `provenance.py` and its source-fingerprint test; no consumer called `physics_source_identity`.
- Mechanics report generators, provenance manifest, report narrative, and reproduction docs. The report depended on a different pinned impedance artifact and generated restricted local assets, not the shoe showcase artifact.
- Old workflow diagram source/rendering (`methods.mmd`/`methods.svg`) and stale report narrative describing fitting paths that no longer exist.
- Consolidation note and friction guide pages (`CONSOLIDATION.md`, `FRICTION.md`, `FRICTION_COLUMN.md`, `FRICTION_CONTINUITY.md`, `FRICTION_ONSET.md`). The kept code and concise README now describe supported mechanics.
- Tests dedicated to deleted tools and the deleted source fingerprint; removed the acquisition test from `test_digital_shoe.py`.

## Simplified

- `projects/digital_shoe/README.md` now documents only the artifact report, Instron/drop/rocker demos, GIF recording, and shared runtime.
- Package CLI keeps only `view`, `report`, and `record`; package exports no longer expose the removed friction solver or source fingerprint.
- Validation report methods content now describes the current two-surface bed, two-term Ogden-Hill/Maxwell/Jenkin material, and validation limits without the removed workflow diagram or stale derivation claims.
- Removed unused `contact.bristle_step`; kept `cone_viscous_scale`, Pasternak coupling, contact kinematics, and column-balance helper because `column_bed.py` uses them.
- Added explicit `__all__` declarations for artifact, material, and runtime modules.
- Preserved the exact `projects.digital_shoe.mechanics_report.figures` import path as a helper-only 62-line module containing `BLUE`, `GREEN`, `INK`, `ORANGE`, `cells`, `decorate`, and `style`. `projects.digital_instron_v2.sweep_report` needs these helpers; no import change is required.
- Updated a fixture test to select its existing default Instron validation curve and compare its rendered start-of-cycle pose (3 mm preload offset).

## Kept

- `artifact.py`, schema, and `load_artifact`/`validate_artifact` plus `MODEL_TYPE`, `SCHEMA_VERSION`: define and load the portable shoe artifact used by exporters and runner consumers.
- `column_bed.py`, `contact.py`, `material.py`, `runtime.py`: shared production shoe mechanics required by Digital Instron and impedance runner; numerical laws were not changed.
- `rendering.py`, `figures.py` helper module, `report.py`: rendering/report APIs consumed by demos, exporter, and sweep report.
- `showcase.py` and `record_gifs.py`: the required Instron, drop, rocker, GIF, and validation-report workflows.
- `digital_shoe.schema.json`: documents the artifact contract.
- `newton/tests/test_digital_shoe.py`, `test_digital_shoe_consumers.py`, `test_digital_shoe_rearfoot_geometry.py`, `test_digital_shoe_material.py`, and `test_digital_shoe_column_bed_contact.py`: retained artifact, material, geometry, runtime, and production contact tests. The deleted fingerprint-only test was not retained.

## Verification

- `uv sync --extra dev` — passed.
- `uv run --extra dev -m newton.tests -k test_digital_shoe` — passed, 43 tests.
- `uv run --extra dev -m newton.tests -k test_digital_instron_sweep_report` — passed, 2 tests (consumer import/style helper check).
- `uv run --extra dev -m newton.tests -k test_digital_shoe_column_bed_contact` — passed, 12 CUDA tests; production column-bed contact results unchanged.
- `uvx pre-commit run --files ...` for all changed Python/docs — passed on second run (first run applied Ruff formatting fixes).
- `uv run --extra dev -m projects.digital_shoe.report DigitalInstron/digital_shoe_showcase/digital_shoe.json --output DigitalInstron/digital_shoe_showcase/validation_report_cleanup.html` — passed; temporary HTML was removed.
- `uv run --extra examples -m projects.digital_shoe.showcase ... --mode {instron,drop,rocker} --viewer null --num-frames 4` — all three headless smoke runs passed. With `--test`, four frames correctly fail the built-in final gates because they do not complete the required warmup/impact/load cycle; no physics change was made.
- `uv run --extra examples -m projects.digital_shoe.record_gifs --artifact ... --output .../cleanup_gifs` — passed; generated 30-frame Instron, 240-frame drop, 72-frame rocker GIFs and report, then temporary outputs were removed.

## Parent decisions / notes

- Root `README.md` is parent-owned and still advertises deleted `projects.digital_shoe friction`, `friction_example.jpg`, and `FRICTION.md` around lines 1053–1058. Please remove/update that block. No requested root README/pyproject/changelog files were changed.
- `mechanics_report.figures` remains as a module path only because an in-scope downstream consumer imports its style helpers; all report-generation assets/content were deleted. If the parent prefers relocating the style helper, update `projects/digital_instron_v2/sweep_report.py` to `from projects.digital_shoe.<new_module> import BLUE, GREEN, INK, ORANGE, cells, decorate, style`.
- Assumption: acquisition planning and the standalone friction comparisons are not mission-critical pipeline stages because the task’s ordered pipeline excludes them and repository scans found no external consumers of those modules. The production column-bed ground friction stays intact.
