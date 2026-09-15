<!--
SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
SPDX-License-Identifier: Apache-2.0
-->

# Two-term footwear mechanics report

Read [REPORT.md](REPORT.md) for the audited material, contact, geometry, solver,
and validation narrative. It covers Digital Shoe, Digital Instron, and the
current impedance rig. It uses only the selected **two-term** artifact.

## Rebuild the offline report

From the repository root, in its existing environment:

```bash
uv run --no-sync -m projects.digital_shoe.mechanics_report.figures \
  --artifact outputs/impedance_instron/inputs/digital_shoe.json \
  --output outputs/footwear_contact_material_report/figures
uv run --no-sync -m projects.digital_shoe.mechanics_report \
  --artifact outputs/impedance_instron/inputs/digital_shoe.json \
  --output outputs/footwear_contact_material_report/report.html
```

Open `outputs/footwear_contact_material_report/report.html`. All six figures are
embedded in the HTML; it requires no server, JavaScript, or CDN. PNG and SVG
versions stay in the sibling `figures/` directory. Use browser Print / Save as
PDF for a print copy. `--help` does not initialize the simulation or plotting
stack. Figure generation uses optional `matplotlib`; rendering uses optional
`markdown-it-py`. Both were already installed in the audited project environment.
These tools do not install packages or add a required project dependency.

The selected artifact and source hashes are pinned in [sources.json](sources.json).
An artifact change, disabled term, nonzero Poisson ratio, or physics-source drift
fails explicitly. Review the narrative and evidence before deliberately updating
the manifest. A successful redraw is not a new refit, fresh bench measurement,
or validation of a changed model. The renderer also checks generated figure
provenance so it cannot silently embed figures from another run.

The figure command calculates material-point and bristle illustrations with the
actual shared laws. It separately redraws the artifact's saved measured and
predicted bench curves. Metadata distinguishes these cases and retains both
stored and freshly recomputed metrics, including small numerical differences.

## Local data boundary

A source checkout contains the narrative, generators, provenance manifest, and
synthetic tests. It does **not** contain the input artifact, generated geometry
figures, measured curves, HTML/ZIP output, raw measurements, or run logs. The
report's links to figures and preparation logs require the local generated
package. The Markdown's numerical results describe the identified internal
research snapshot; they are not a substitute for the omitted data.

The output directory is ignored. Missing local input fails rather than falling
back to a different artifact. No command overwrites or refits the shoe artifact.
Source links in generated HTML resolve against the checkout. Rebuilding into
another output directory is supported when its `figures/` directory has first
been generated there.

Follow [ASSET_PROVENANCE.md](../../../ASSET_PROVENANCE.md) before external or
upstream publication. Committing these tools does not clear the measurements,
geometry, or generated derivatives for redistribution.

## Tests

```bash
uv run --no-sync -m unittest newton.tests.test_footwear_mechanics_report
```

The synthetic tests do not require local footwear measurements. The optional
renderer tests skip clearly when `markdown-it-py` is unavailable. The real-data
build commands above provide the reusable end-to-end example.
