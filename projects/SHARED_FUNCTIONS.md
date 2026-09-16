# Shared-function boundaries

| Computation | Owner | Consumers |
|---|---|---|
| Shoe material, contact, foundation runtime | `digital_shoe/` | Calibration, standalone shoe examples, twelve-point controller |
| Camera look-at calculation | `digital_shoe.rendering.camera_look_at` | Showcase and Instron example |
| Quaternion and attachment PD helpers | `digital_instron_v2.scenario_common` | Forward and differentiable calibration adapters |
| XYZ rotation and ray-origin setup | `digital_instron_v2.geometry` | Mesh loading and ray queries |
| Cubic basis and derivative polygons | `impedance_instron.cartesian.spline` | CPU reference and GPU controller setup |
| Rigid foot/shoe attachment | `impedance_instron.cartesian.shoe` | CPU qualification, GPU dynamics and contact replay |

The retired controller stacks and their duplicate-review exemptions are gone.
The retained spline helpers and Shoe implementation were extracted without
changing their computation. New source identities require new qualification.
Shared material/contact laws remain unchanged.

Run the structural duplicate check without importing the projects:

```bash
uv run --no-sync python scripts/check_shoe_project_duplicates.py --check \
  --output /tmp/shoe-duplicates.json
```

`shoe_duplicate_review.json` records reviewed remaining groups. Strict JSON,
nonfinite display JSON, opposite hysteresis signs, different interpolation
boundaries, and mutable versus differentiable buffers must not be merged only
because their implementations look similar.
