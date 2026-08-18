# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Human shoe project contract scaffolding and attachment adapter."""

from .adapter import (
    OSIM_LOCAL_TO_Z_UP_JUMP_BASIS,
    AttachedSoleGeometry,
    FootContactReference,
    ResolvedHumanShoeAttachment,
    attach_sole_geometry,
    resolve_attachment,
)
from .contracts import (
    DigitalInstronFitContract,
    DigitalInstronGridContract,
    DigitalInstronIndenterContract,
    DigitalInstronManifestContract,
    DigitalInstronTrialContract,
    FootShoeAttachmentContract,
    HumanShoeExperimentContract,
    load_experiment,
    load_manifest,
)

__all__ = [
    "OSIM_LOCAL_TO_Z_UP_JUMP_BASIS",
    "AttachedSoleGeometry",
    "DigitalInstronFitContract",
    "DigitalInstronGridContract",
    "DigitalInstronIndenterContract",
    "DigitalInstronManifestContract",
    "DigitalInstronTrialContract",
    "FootContactReference",
    "FootShoeAttachmentContract",
    "HumanShoeExperimentContract",
    "ResolvedHumanShoeAttachment",
    "attach_sole_geometry",
    "load_experiment",
    "load_manifest",
    "resolve_attachment",
]
