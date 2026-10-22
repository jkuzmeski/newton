# SuperMimic

SuperMimic trains a shoe-conditioned runner in Newton to track a human running
reference. `reference.py` builds the runner's body parameters and reference
windows from a subject's Visual3D exports.

```bash
uv run --extra dev --extra torch-cu12 -m projects.supermimic.reference \
    --subject projects/supermimic/subjects/F01.json
```

The output folder holds `reference.npz`, `body_params.json`, `subject.json` and
`report.json`. `collect.py`, `tracker.py` and `analysis.py` read the body and
shoe from it.

## Adding a runner

1. Export each running trial with Visual3D into its own folder, as for F01
   (`data/F01/Export_FR3_ImpedanceInstron.v3s`). The subject needs the same
   marker set and model. Each trial folder holds its own static calibration
   (`static_all_targets.txt`, `static_joint_centers.txt`), `model_mass.txt`,
   the motion and plate exports, and `visual3d_manifest.json`, which records
   the axes, rates and `belt_speed_m_s`. All trials of a subject must share
   one belt speed.
2. Add the fitted shoe to the column-bed dataset (`SHOE_DATASET` in
   `reference.py`) if it is not there yet; see [Adding a shoe](#adding-a-shoe).
3. Write `projects/supermimic/subjects/<name>.json`:

   ```json
   {
     "name": "F01",
     "data": "data/F01",
     "sex": "male",
     "stature_m": 1.65,
     "weighed_in": "puma-fast-r-nitro-elite-3",
     "trials": {"FR3_1": "puma-fast-r-nitro-elite-3", "FR3_2": "puma-fast-r-nitro-elite-3"}
   }
   ```

   - `data` is relative to the repository root.
   - `sex` selects the de Leva (1996) table (`male` or `female`).
   - `weighed_in` is the shoe worn when `model_mass.txt` was weighed. Use
     `null` for barefoot. Its mass is moved from the pelvis to the feet.
   - `trials` maps each trial folder to the shoe worn. One reference holds a
     single shoe.
4. Run `reference` with `--subject` and check `report.json`: the marker fit
   errors, the contact counts against the raw plate, and the midstance foot
   speed against the belt speed.

## Adding a shoe

A shoe needs a midsole scan, an Instron foam sweep on the full-foot last and,
if known, the mass of one shoe. Run from the repository root:

1. Turn the RTINGS midsole scan into column grids:
   `uv run --no-sync python -m projects.digital_instron_v2.shoe_prep <scan>.glb --slug <slug>`
   writes `DigitalInstron/<slug>/{left,right}/column_grid.npz`.
2. Split the sweep and queue the foam fit on the grid (one H100 per seed; see
   `grid/launch.py` for the grid setup):

   ```bash
   uv run --no-sync python -m projects.supermimic.grid.shoe submit <slug> \
       --sweep <test>.steps.tracking.csv --label <Label> --commit <sha>
   ```

   The sweep must follow the October 2026 protocol (`sweep_split.py`: stance
   times 160-280 ms at each force target). Its traces land in
   `DigitalInstron/FoamSweep/processed/<Label>`. The printed `local_run_dir`
   is the run folder; the logs are in `Z:/newton_grid/supermimic/logs`.
3. When the job has finished, copy the dataset to a new version folder and add
   the shoe to it, on the same commit the fit ran on (the export replays the
   validate traces with the local column bed):

   ```bash
   uv run --no-sync python -m projects.supermimic.grid.shoe add <local_run_dir> \
       --dataset Z:/newton_grid/data/dataset_3d_column_bed_v<N> --mass-kg 0.18 --mass-source "<scale or URL>"
   ```

   This exports the lowest-loss fit as `assets/<slug>_{left,right}.json` (the
   right shoe is the mirror of the left), writes the validation report and fit
   to `reports/<slug>/`, and enters the shoe in `manifest.json`. Its mount and
   static pitch are carried from the shoe worn in the motion trials through the
   shared last. Without `--mass-kg` the shoe weighs `DEFAULT_SHOE_MASS_KG`
   (0.2 kg). Check the printed validate-trace RMSE and the report.
4. Point `SHOE_DATASET` in `reference.py` at the new version.
