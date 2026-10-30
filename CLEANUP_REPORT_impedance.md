# Impedance pipeline cleanup report

## Deleted

- `hogan/presentation.py` and `hogan/report_assets/report.css` / `report.js`: only used by the external presentation layer and no longer needed by native fit-report HTML.
- Optional formatter compatibility in `fit_report.py`: removed `rebuild_motion_from_saved_traces` and the `--saved-traces` option; native reports render directly from fit output.
- The full-run test fixture in `test_impedance_pipeline.py` that depended on the missing `outputs/hogan_reset/fit_score_20261029` directory and removed `generative_runner_score_20261029.json` model. Kept command-router tests.
- `Runner` schema-1 loading and its compatibility test. New profile-loading tests reject old controller keys rather than silently accepting them.
- `raw_ground_deprecated` ankle-angle mode and its test case.
- `--allow-incompatible` fitting override and its test. Fitting now always stops on failed compatibility.
- The unsupported `report_spec` / `report_html` placeholder roles from material-study output; no in-repository report builder consumed them.

## Simplified

- Replaced `projects/impedance_instron/README.md` and `hogan/MATERIALS.md` with short guides for the active preparation, fit/evaluate, material-study, and report-site workflows. The README explains that `impedance_instron` is retained as a historical package name and describes the active `cartesian/` directory as Visual3D/runner input code, not a Cartesian controller.
- Updated `hogan/GENERATIVE_RUNNER.md` to describe the current score model and current schema; removed the 2026-10-08 baseline section, stale model links, and historical controller-field acceptance wording.
- Kept the `cartesian/` import path and serialized reference/profile schema names because live preparation, runner, and report code depend on them. Reworded package and reference docs to make their role clear; renamed the shoe-adapter test from `test_impedance_cartesian_shoe.py` to `test_impedance_shoe.py`.
- Dropped obsolete impedance/limits provenance entries from `hogan/profiles/f01_66_5kg.json`; retained its inertial provenance and numeric inputs.
- Kept `data/F01` intact. The prepare path consumes the source time/force/COP/moment files, marker exports, stance selections, and static/dynamic Visual3D files; no input-file deletion was needed.

## Kept

- `cartesian/visual3d.py`, `prepare_visual3d.py`, `prepare_dataset.py`, `data.py`, `profile.py`, and `shoe.py`: required for the Visual3D-to-runner input and contact path.
- Hogan CPU/CUDA mechanics, runner, identification, generation, and fit-report code: required to fit, evaluate, generate, and inspect runner results.
- The committed slider-shoe score model and its provenance: required by the evaluation gate and default material study.
- Paper-material, geometry, material-sensitivity studies, tornado plotting, and the committed report-site data/build: explicitly in the current material-study/report pipeline.
- Visual3D input guidance and focused tests for preparation, shoe contact, runner physics, fit/evaluate/generation, and the material studies.

## Verification

- `uv sync --extra dev` — passed.
- Baseline pre-cleanup evaluate command — train mean loss `10.537715730923786`; held-out mean loss `10.427736690837257`.
- Post-cleanup evaluate command with the prescribed model/settings — train `10.537715730923786` (98 trials, zero failed); held-out `10.427736690837257` (9 trials, zero failed). Per-trial scores unchanged by the cleanup.
- README `prepare` command with `--exclude` — wrote 107 members. All 107 `reference.npz` SHA-256 digests matched the external reference dataset byte-for-byte.
- `uv run --no-sync -m projects.impedance_instron fit` with `--limit-per-split 1 --iterations 1` — passed; wrote fit summary and `report.html`.
- `uv run --no-sync -m newton.tests -k test_impedance_` — passed, 120 tests.
- `uv run --no-sync python projects/impedance_instron/hogan/report_site/_build/build.py` — passed; rebuilt `index.html` from committed JSON.
- `uvx pre-commit run --files <changed owned files>` — passed all hooks. The first run applied one Ruff cleanup; the subsequent run passed.

## Uncertainties / parent decisions

- Per the ownership addendum, I did not edit `projects/README.md`; it still describes `impedance_instron/cartesian/` as retained reference/profile preparation and mentions the obsolete `outputs/impedance_instron/generative_fit_lm_flightcom_20261008` directory. The project index should be updated to match this cleanup.
- Per the ownership addendum, I did not edit `ASSET_PROVENANCE.md`; the updated profile JSON and rebuilt report-site `index.html` may require its recorded byte sizes / hashes to be refreshed. The report-site build left the checked-in page unchanged at its recorded 221,817 bytes.
- I did not delete `hogan/material_sensitivity/RESULTS.md`, `tornado_data.json`, or `tornado_marathon065.png`: material sensitivity and `plot_tornado.py` are named in the current mission pipeline and the data is used by that plotting script. The results describe an older completed study, so the parent should decide whether it remains as a historical research record or should be reduced to the current report-page summary.
- I did not delete any F01 export files: the supported current path uses FR3_1 and FR3_2, but the other exports remain useful source records and the user specifically limited data cleanup to files proven unread by preparation.
- No changelog fragment was added, as directed. The parent should decide whether the workflow cleanup warrants a fragment.
## Second-pass cleanup

- Removed the dead `_signal` parser and the duplicate joint-center sample-count check; `_joint_centers` already enforces that marker-clock invariant.
- Removed the unused `rate` parameter from `_time_file`; declared rates remain checked by `_clock_rate`.
- Removed duplicated type/shape checks where parsed Visual3D headers or validity masks already enforce the same property, and dropped the `immediate_damping` bool-only guard (the option is normalized to bool).
- Kept the final `data.validate` boundary because it checks the complete prepared reference contract, not just individual input arrays.
- Second-pass gates: 120 impedance tests passed; the 107 prepared reference archives matched the external reference dataset byte-for-byte; evaluation remained train `10.537715730923786` and held-out `10.427736690837257`; report-site build passed.
