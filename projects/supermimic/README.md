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
   `reference.py`) if it is not there yet.
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
