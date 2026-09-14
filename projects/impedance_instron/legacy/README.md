# Retired impedance experiments

This package holds the retired three-/six-action controllers and their reports.
The active two-stiffness workflow remains in [`../simple/`](../simple/) and runs
with `uv run --no-sync -m projects.impedance_instron`.

## Implementation map

| Modules | Retired function |
|---|---|
| `control`, `example` | Equilibrium command parameterization and the earlier rig |
| `objective`, `cmaes`, `optimize` | Task/work objective and command-trajectory search |
| `env`, `train` | Residual-action batch environment and PPO/checkpoint tools |
| `report`, `explain`, `dashboard`, `summary` | Historical trace, overlay, training and narrative reports |

The root modules with these names are compatibility aliases to the same module
objects. Old imports, including private helpers, remain available. Old module
commands still call the same entry points and retain their existing warnings:

```bash
uv run --no-sync -m projects.impedance_instron.train --help
uv run --no-sync -m projects.impedance_instron.example --help
uv run --no-sync -m projects.impedance_instron.summary --help
```

Explicit `projects.impedance_instron.legacy.*` imports and module commands also
work. Importing this package alone does not load the retired runtime. The root
`profile`, `orientation`, `trajectory`, and `variability` modules remain active
shared preparation helpers. `mcclough` remains a separate material-analysis tool.

## Compatibility limits

- The implementation's `__file__` and classes' `__module__` now point here. Old
  pickle globals still resolve through the root aliases. New generic pickles use
  the `legacy` module path and need a checkout that contains this package.
- Relocating imports changes the retired example's source hash. Its report audit
  still checks that hash. No historical hash, checkpoint, or JSON seal is rewritten.
- CLI support means a fresh `python -m ...` process. Running an already imported
  old alias through in-process `runpy.run_module` can fail its loader-name check.
- Material/contact laws, optimization loops, command defaults, and checkpoint
  formats are unchanged. Historical scores do not qualify the active controller.

## Historical data and narrative

[`../LEGACY_REPORT.md`](../LEGACY_REPORT.md) remains the historical renderer's
default template. [`../REPORT.md`](../REPORT.md) describes the current workflow;
the legacy renderer must not add old figures to it.

The optional local `outputs/impedance_instron/LEGACY_ARCHIVE.json` points to an
immutable historical source snapshot and generated data outside the project.
That snapshot is separate from this maintained compatibility implementation.
Do not move current `simple/` artifacts into the historical data paths.

To render archived outputs, pass their directory explicitly:

```bash
uv run --no-sync -m projects.impedance_instron.summary \
  --data-directory /path/to/legacy/outputs \
  --output outputs/impedance_instron/legacy_report
```

No archived data is distributed with the source. Missing artifacts remain visible
as missing data in the report rather than being replaced by current results.
