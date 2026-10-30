# Digital Instron data

This directory contains the midsole scans, Instron foam sweeps, prepared column
beds, fitted material files and exported Digital Shoe artifacts.

## Pipeline

1. Split the raw `FoamSweep` tracking CSVs into train/validate traces:
   `uv run --extra dev -m projects.digital_instron_v2.sweep_split`
2. Prepare midsole scans with `projects.digital_instron_v2.shoe_prep`.
3. Fit each processed sweep with `projects.digital_instron_v2 fit` and build
   `FoamSweep/fit/report.html` with `projects.digital_instron_v2.sweep_report`.
4. Export a fitted artifact with `projects.digital_instron_v2 export`.

See `FoamSweep/README.md` and `projects/digital_instron_v2/README.md` for the
commands and data layout. `digital_shoe_showcase/` contains the reference
artifacts and viewer reports.
