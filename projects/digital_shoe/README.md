# Digital Shoe

Digital Shoe owns the portable `digital_shoe.json` artifact, the shared shoe
mechanics, and three artifact-only demonstrations. It does not fit measurements
or control an impedance runner.

## Pipeline

```text
Instron sweeps + midsole scans -> Digital Instron fit -> digital_shoe.json
                                                       -> validation report
                                                       -> Instron / drop / rocker
```

The artifact carries the whole sampled midsole column bed, visual meshes,
fitted material, held-out curves, and provenance. Digital Instron and impedance
runner consumers use the same material law and `ColumnBed` runtime.

## Rebuild the validation report

From the repository root, reuse the exported artifact and any GIF recordings:

```bash
uv run --extra dev -m projects.digital_shoe.report \
  DigitalInstron/digital_shoe_showcase/digital_shoe.json \
  --output DigitalInstron/digital_shoe_showcase/validation_report.html \
  --media-dir DigitalInstron/digital_shoe_showcase
```

The report is standalone HTML. It shows the held-out Instron curves, fitted
parameters, model assumptions, GIFs when present, and source hashes. A failed
validation gate stays visible; adjacent-cycle agreement does not establish
performance at unseen rates or impacts.

## Run the demonstrations

```bash
uv run --extra examples -m projects.digital_shoe.showcase \
  --artifact DigitalInstron/digital_shoe_showcase/digital_shoe.json \
  --mode instron --viewer gl
uv run --extra examples -m projects.digital_shoe.showcase \
  --artifact DigitalInstron/digital_shoe_showcase/digital_shoe.json \
  --mode drop --viewer gl
uv run --extra examples -m projects.digital_shoe.showcase \
  --artifact DigitalInstron/digital_shoe_showcase/digital_shoe.json \
  --mode rocker --viewer gl
```

Record the three headless GIF loops and rebuild the report:

```bash
uv run --extra examples -m projects.digital_shoe.record_gifs \
  --artifact DigitalInstron/digital_shoe_showcase/digital_shoe.json
```

For a short headless check, add `--viewer null --num-frames 4 --test` to the
showcase command. `--help` lists the supported modes and recording options.

## Shared mechanics

`material.py` defines the Ogden–Hill pressure and Maxwell recurrence used
by the host fitting code and Warp runtime. `contact.py`, `runtime.py`, and
`column_bed.py` implement the two-surface column solve, ground bristles, and
material settings. `rendering.py` supplies the display geometry and colors.
The report and GIF recorder live in `report.py` and `record_gifs.py`.
