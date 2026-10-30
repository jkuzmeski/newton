# Shoe-to-runner pipeline

This directory holds the research code that turns a running shoe into a
simulated midsole and fits a simulated runner on it.

| Package | Role |
|---|---|
| [`digital_instron_v2`](digital_instron_v2/README.md) | Split Instron sweeps, prepare midsole scans, fit the foam material, export the shoe |
| [`digital_shoe`](digital_shoe/README.md) | The `digital_shoe.json` format, the column-bed contact model, the validation report and demos |
| [`impedance_instron`](impedance_instron/README.md) | Prepare F01 running data, fit and evaluate the Hogan runner, compare shoe materials |

Data lives in [`DigitalInstron/`](../DigitalInstron/README.md) (shoes and Instron
sweeps) and `data/F01/` (Visual3D running exports). Restricted assets are listed
in [asset provenance](../ASSET_PROVENANCE.md).

## Run the pipeline

Run every command from the repository root after `uv sync --extra dev --extra examples`.

1. Split the raw Instron sweeps into train, validation and repeat traces:

   ```console
   uv run -m projects.digital_instron_v2.sweep_split
   ```

2. Fit a shoe's foam material and build the fit report. The committed fits are
   in `DigitalInstron/FoamSweep/fit/`; see
   [FoamSweep](../DigitalInstron/FoamSweep/README.md) to rerun them.

   ```console
   uv run -m projects.digital_instron_v2 fit --shoe DigitalInstron/puma-fast-r-nitro-elite-3 --sweep DigitalInstron/FoamSweep/processed/FR3 --out <output folder>
   uv run -m projects.digital_instron_v2.sweep_report
   ```

3. Export the fitted shoe:

   ```console
   uv run -m projects.digital_instron_v2 export --shoe DigitalInstron/puma-fast-r-nitro-elite-3 --fit DigitalInstron/FoamSweep/fit/puma-fast-r-nitro-elite-3/material_fit.json --sweep DigitalInstron/FoamSweep/processed/FR3 --output DigitalInstron/digital_shoe_showcase
   ```

4. View the shoe on the virtual Instron (`--mode drop` and `--mode rocker` also work):

   ```console
   uv run --extra examples -m projects.digital_shoe.showcase --artifact DigitalInstron/digital_shoe_showcase/digital_shoe.json --mode instron --viewer gl
   ```

5. Prepare the runner dataset, then evaluate the committed runner baseline. The
   [runner README](impedance_instron/README.md) lists the full commands, the
   expected losses (10.5377 train, 10.4277 held-out) and how to fit a new runner.

6. Rebuild the report page from its committed data:

   ```console
   uv run python projects/impedance_instron/hogan/report_site/_build/build.py
   ```

## Tests

```console
uv run --extra dev -m newton.tests -k test_digital_instron_
uv run --extra dev -m newton.tests -k test_digital_shoe
uv run --extra dev -m newton.tests -k test_impedance_
```
