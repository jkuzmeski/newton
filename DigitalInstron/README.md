# Digital Instron

This folder holds the Instron bench data for the Digital Instron project: the
rearfoot punch and full-foot shoe-last trials, the per-shoe foam sweeps, and the
shoe-last geometry.

The full-foot analysis uses the mirrored left-foot asset
`Instron Shoe Last Size 9 6drop merged attachment 1 left.STL`; the original
right-foot STL is retained unchanged as its source.

## Model

The repaired midsole mesh is sampled on a 5 mm column grid. Each column uses its
raycast thickness as its rest length. The column bed
(`projects/digital_shoe/column_bed.py`) gives each column a bottom on the platen
or ground and a top pressed by the last, and integrates pressure over the
columns' finite areas. Released columns carry no tension.

## Command

Fit one shoe's material to its foam sweep:

```powershell
uv run --extra dev -m projects.digital_instron_v2 fit `
  --shoe DigitalInstron/<slug> --sweep <processed sweep folder> --out <output folder>
```

See `projects/digital_instron_v2/README.md` for the scenes and export.

## Source layout

- `FoamSweep/README.md`: foam sweep data, fitted materials, report, and how to reproduce them
- `manifest_v2.json`: averaged traces, geometry, and initial values
- `../ASSET_PROVENANCE.md`: checksums and redistribution status for project data
- `projects/digital_instron_v2/core.py`: material record and metrics
- `projects/digital_instron_v2/geometry.py`: material-column geometry
- `projects/digital_instron_v2/sweep_fit.py`: fitting command
- `MODEL_FINDINGS.md` and `GEOMETRY_AND_MATERIAL_REPORT.md`: research record for
  the earlier one-unknown-per-column law
