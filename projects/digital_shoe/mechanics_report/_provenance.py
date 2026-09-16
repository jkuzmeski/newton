# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify the immutable inputs used by the two-term mechanics report."""

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = Path(__file__).with_name("sources.json")
DEFAULT_ARTIFACT = ROOT / "outputs/impedance_instron/baseline12/digital_shoe.json"
DEFAULT_OUTPUT = ROOT / "outputs/footwear_contact_material_report"


def load_verified_artifact(path: Path, *, manifest_path: Path = MANIFEST, root: Path = ROOT) -> dict:
    """Require the audited artifact bytes, two active terms, and unchanged physics.

    Args:
        path: Selected local artifact; relative paths resolve from ``root``.
        manifest_path: Frozen artifact and source checksums for this narrative.
        root: Checkout root for source paths recorded in the manifest.

    Returns:
        Parsed artifact data without importing the simulation or plotting stack.
    """
    path = Path(path)
    if not path.is_absolute():
        path = root / path
    manifest = json.loads(Path(manifest_path).read_text())
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest["artifact_sha256"]:
        raise ValueError("Artifact changed: review this fixed two-term report before rebuilding")
    artifact = json.loads(raw)
    parameters = artifact["constitutive_model"]["parameters"]
    moduli = [
        parameters.get(name, 0.0) for name in ("instantaneous_shear_modulus_pa", "instantaneous_shear_modulus_2_pa")
    ]
    if not all(math.isfinite(value) and value > 0.0 for value in moduli):
        raise ValueError("This report requires two positive moduli in the selected two-term model")
    if parameters.get("effective_poisson_ratio") != 0.0:
        raise ValueError("This report requires the selected zero-Poisson model")
    for source, expected in manifest["files"].items():
        if hashlib.sha256((root / source).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Source changed: {source}; review the audit before rebuilding")
    return artifact
