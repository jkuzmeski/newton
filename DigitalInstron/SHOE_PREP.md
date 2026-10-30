# Shoe preparation: midsole `.glb` → canonical left/right column beds

Module: `projects/digital_instron_v2/shoe_prep.py`. Tests: `newton/tests/test_digital_instron_shoe_prep.py`.

```bash
# any mix of files and folders (every *.glb inside); prints "<slug>: ready -> <dir>" per shoe,
# exit code 1 if any shoe is rejected
uv run --extra dev python -m projects.digital_instron_v2.shoe_prep path/to/<slug>-3d-internal.glb DigitalInstron/<slug>/source.glb path/to/folder --out-root DigitalInstron
```

Slug: the file stem without `-3d-internal`, or the parent folder name for `source.glb`. `--slug`/`--product-name` need exactly one file.

## Pipeline

1. **Load and validate one body** (`load_midsole`). Read the scene and record each node, geometry, face count, and material in `shoe.json → mesh_qc.scene_parts`. Apply the node transforms. Guess the units from the longest extent (0.15–0.40 → m, 150–400 → mm; anything else is rejected). Weld the UV-seam duplicate vertices (3.0 M → 0.5 M). Drop degenerate and duplicate faces. Split into connected bodies. Reject the scan if bodies other than the largest hold more than 0.1 % of the faces. Fill single-triangle and single-quad holes, which is cheap and safe. Record open and non-manifold edges before and after filling. Reject the scan if the winding is inconsistent. Invert the mesh if its signed volume is negative. An open mesh is **not** rejected.
2. **Canonical frame** (`_canonical_mesh`). The frame is +x heel→toe, +z up, and +y = z × x. So +y is **medial on a right shoe** and lateral on a left shoe. x/y are centered on the bounding box and z_min = 0, the same convention as `load_mesh`.
   - Axes: length = longest extent, up = thinnest extent. Reject the scan if height > 0.8 × width.
   - Heel/toe: the shared `registration.canonical_frame(kind="shoe")`. The heel must be thicker **and** narrower, each by at least 3 mm, or the scan is rejected.
   - Up sign (owned here): the toe spring of the sole mid-surface, i.e. the toe band minus the lowest forefoot section. It must be at least 5 mm in absolute value. A negative value means the scan is upside down, and the frame is flipped (recorded as `up_flipped_from_source`).
3. **Foot side** (`foot_side`). The medial outline has the deeper arch indentation, measured as the largest gap to the convex hull between 30 % and 70 % of the length. The deeper side must be at least 1.5× the other and at least 3 mm deeper. Secondary cue: if the toe tip is more than 5 mm off the heel centerline, it must lean medially. Otherwise the scan is rejected.
4. **Columns** (`sample_columns`). These are exactly the `geometry.build_column_grid` columns (5 mm lattice, first/last hit, normal threshold 0.1, 5 mm minimum), plus one extra ray pass for QC:
   - footprint = valid columns with enclosed gaps filled;
   - **gap** = a footprint point whose ray hit the mesh but gave no valid column, and the ray has an odd hit count (open surface) or a hit span ≥ 5 mm (a steep face failing the normal check). Thin columns with paired hits are real and stay dropped. Rays with no hits are real through-holes and stay empty;
   - **suspect** = an isolated thickness spike: more than 5 mm below all of its neighbors or above all of them, with at least 5 valid neighbors;
   - gaps and suspects are linearly interpolated from trusted columns. If more than 2 % of the columns need repair, the scan is rejected ("footprint coverage compromised"). Counts and repaired positions go into `column_qc`.
   - each measured column is then averaged over a 9×9 ray grid across its 5 mm cell. Bottom and top are the means over the rays that cross material, and `area_m2` is 25 mm² times the fraction of those rays. An outline cell whose center ray grazes the sloped sidewall keeps its covered area and mean thickness, not the thin center value. Repaired columns keep the full cell area. `column_qc` records `partial_cells` and `covered_area_fraction`.
5. **Left/right** (`mirror_grid`). The opposite foot is the exact y-mirror of the sampled grid. `mirrored_from_source` is recorded per side. Each side gets a 4×4 `source_to_canonical` matrix (in the fragment: `midsole_to_canonical`) that maps `source.glb` file units to that side's frame in meters, mirror included.
6. **Gates** (US men's 9, the RTINGS test size): column count 600–1400; max thickness 5–70 mm; median thickness 15–50 mm; footprint length 255–330 mm; width 80–130 mm; heel-to-forefoot drop 0–20 mm (median thickness at 8–18 % of the length minus 70–80 %). If a gate fails, the figure is still written and `shoe.json` gets `geometry_status: "rejected"` with the reason.
7. **Determinism and provenance**. The npz files have fixed zip timestamps. JSON uses sorted keys. The PNG has no software or date metadata. Two runs are **byte-identical** (checked on both shoes and in the tests). `provenance_sha256` = sha256(source sha256, code sha256 of `shoe_prep.py` + `geometry.py` + `registration.py`, parameters). Each side also records the content hash `grid_sha256`.

Output: `<out-root>/<slug>/{source.glb, shoe.json, qc_top.png, left|right/{column_grid.npz, preflight.json, manifest_geometry_fragment.json}}`. The npz keys are `uv_m, bottom_m, top_m, area_m2, spacing_m, thickness_axis`; `area_m2` is per column. The fragment drops `midsole_mirror_axis`; the mirror is part of `midsole_to_canonical`. Every output is computed and written into a staging folder `<out-root>/.<slug>.*` first. Only then is each file moved into place with `os.replace`, and `shoe.json` goes last. A rejected shoe has its stale side files removed, but only after its new `shoe.json`/figure are staged. If a lock makes one `os.replace` fail partway through, the files already moved are complete and the others are unchanged. Each `preflight.json` carries `provenance_sha256`, so a mixed state can be detected against `shoe.json`.

## Results on the two shoes

| | FR3 | Deviate |
|---|---|---|
| scene | 1 mesh (1 000 000 faces) | same |
| open edges before / after hole fill | 0 / 0 | 4 (1 mm) / 0 |
| detected side | right | right |
| columns | 917 (1 gap, 1 spike repaired) | 944 (1 gap) |
| thickness min / median / max | 5.3 / 31.6 / 44.0 mm | 5.2 / 32.3 / 49.7 mm |
| heel / forefoot / drop | 34.8 / 32.6 / +2.1 mm | 38.7 / 31.0 / +7.7 mm |
| footprint length × width | 293.7 × 113.9 mm | 287.6 × 108.4 mm |

## Caveats

- Side detection uses an outline heuristic (arch indentation), checked on only 2 real shoes. Both scans are right shoes, so a real left scan has only been tested synthetically. Shoes with a straight or symmetric outline will be rejected. That is the intended outcome, but expect some rejections.
- Heel/toe uses the shared cue (heel thicker and narrower). Zero-drop or wide-heel shoes will be rejected rather than guessed.
- The units guess assumes mm or m. The size gates assume US men's 9.
- z = 0 is the lowest point (the `load_mesh` convention), not a fitted platen support plane. Registration fits the support plane from the grid bottom.
- Repairs interpolate top/bottom. FR3 repairs: one heel-tip rim spike and one steep-wall ray in the midfoot. Deviate repairs: one steep toe ray.
- `source.glb` is copied into the output folder (110 MB each).
