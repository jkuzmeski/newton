# Digital Shoe

`projects.digital_shoe` is the path-independent runtime and presentation layer
for an intact shoe identified from mechanical test data. It does not depend on
gait, C3D, OpenSim, or a human model.

The project answers one narrow question:

> Can an effective nonlinear, rate-dependent shoe model be identified from
> controlled Instron measurements and deployed unchanged in a live simulation?

The checked data support a research proof of concept. They do not yet validate
new rates, temperatures, impacts, or shoes.

## Architecture

The dependency points in one direction:

```text
projects.digital_instron_v2 (data, geometry, fitting)
                    |
                    v
          digital_shoe.json
                    |
                    v
projects.digital_shoe (strict loader, runtime, report, demos)
```

The runtime artifact contains the full 910-column shoe bed, baked calibrated
midsole and shoe-last visual meshes, all six constitutive constants, neighbor
topology, coordinate semantics, held-out curves, metrics, and source hashes. It
contains no absolute file paths. The runtime never falls back to hidden
parameters when loading a named shoe.

## Identify and export

From the repository root:

```bash
uv run -m projects.digital_instron_v2.export_digital_shoe \
  --manifest DigitalInstron/manifest_v2.json \
  --output DigitalInstron/digital_shoe_showcase
```

This command fits cycles 90–98, evaluates held-out cycles 99–100, and writes:

```text
DigitalInstron/digital_shoe_showcase/digital_shoe.json
DigitalInstron/digital_shoe_showcase/validation_report.html
```

The output directory is intentionally ignored. The measurements, footwear
geometry, fitted artifact, and their derivatives are not cleared for upstream
redistribution. See `ASSET_PROVENANCE.md`.

## Mechanical demonstrations

All three scenes consume only `digital_shoe.json`.

```bash
# Displacement-controlled validation against held-out Instron curves.
uv run --extra examples -m projects.digital_shoe.showcase \
  --artifact DigitalInstron/digital_shoe_showcase/digital_shoe.json \
  --mode instron --viewer gl

# An 80 kg free six-DOF body-weight drop carried by the calibrated shoe last.
uv run --extra examples -m projects.digital_shoe.showcase \
  --artifact DigitalInstron/digital_shoe_showcase/digital_shoe.json \
  --mode drop --viewer gl

# A controlled rigid rocker that moves contact and COP heel-to-toe.
uv run --extra examples -m projects.digital_shoe.showcase \
  --artifact DigitalInstron/digital_shoe_showcase/digital_shoe.json \
  --mode rocker --viewer gl
```

Record all three audited GIF loops and rebuild the report with the GIF bytes
embedded directly in the HTML:

```bash
uv run --extra examples -m projects.digital_shoe.record_gifs \
  --artifact DigitalInstron/digital_shoe_showcase/digital_shoe.json
```

The command writes `instron.gif`, `drop.gif`, and `rocker.gif` beside the
artifact, then rewrites `validation_report.html` as a portable single file. It
records every display frame at 720 px. Instron and rocker play at 12 FPS. The
drop is sampled at 240 Hz for one second and plays at 24 FPS, making its impact
and rebound about ten times slower than real time. The report places each animation on its own
full-width row.
Add `--clear-kernel-cache` if a prior interrupted Warp compilation left a
missing `.ptx` cache entry.

Use `--viewer null --num-frames N --test` for a headless audit. Useful minimum
runs are 180 frames for Instron, 60 for drop, and 80 for rocker.

The scenes do not use proxy shoe boxes. The Virtual Instron uses the posed
shoe-last and calibrated midsole meshes. The drop is a free six-DOF 80 kg body
carried by the calibrated full-foot last; it renders the last above the exposed
colored springs and does not render the shoe surface. The rocker also renders
only its spring bed, so spring length and contact-color changes remain visible.

The drop demo adds 5 N·s/m per-column normal damping for impact stability. That
value is a scenario parameter and was not identified by the current
normal-compression tests. Friction is also not claimed as fitted. The 80 kg,
40 mm drop reaches about 20 mm column compression, slightly beyond the original
17.9 mm full-foot test amplitude, so it is an extrapolative mechanical showcase
rather than a validated impact prediction.

### Warp cache recovery

The OpenGL warning about falling back from MSAA is harmless. An error that says
Warp could not open a generated `.ptx` file means a kernel-cache write was
interrupted. Clear the compiled cache once and rerun:

```bash
uv run python -c "import warp as wp; wp.clear_kernel_cache()"
```

The prescribed Instron and rocker modes avoid articulation initialization, so
they do not compile the articulation module merely to set a rigid carrier pose.

## Current held-out result

| Fixture | Peak error | Active RMSE | Hysteresis error |
|---|---:|---:|---:|
| Rearfoot punch | about 12.6% | about 6.7% | about 15.5% |
| Full-foot last | about 14.7% | about 4.1% | about 17.2% |

The active-force RMSE gates pass. Peak and hysteresis gates fail. The HTML report
keeps the overall research status and every failed gate visible.

The held-out cycles are adjacent cycles from the same approximately 0.5 s runs.
This tests implementation consistency and local repeatability, not broad dynamic
generalization.

## New data

`ACQUISITION_PROTOCOL.md` defines the next multi-rate and relaxation experiment.
Use `acquisition_manifest.example.json` as a starting point and validate it with:

```bash
uv run -m projects.digital_shoe.acquisition path/to/acquisition_manifest.json
```

## Tests

```bash
uv run --extra dev -m unittest newton.tests.test_digital_shoe
uv run --extra dev -m unittest newton.tests.test_digital_instron_core
uv run --extra dev -m unittest newton.tests.test_digital_instron_dynamics
```
