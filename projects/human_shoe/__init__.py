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
from .fidelity import BodyFidelityError, PoseFidelityReport, compare_imported_state
from .preparation import PreparedAttachedSole, make_human_shoe_foundation_config, prepare_attached_sole

__all__ = [
    "OSIM_LOCAL_TO_Z_UP_JUMP_BASIS",
    "AttachedSoleGeometry",
    "BodyFidelityError",
    "DigitalInstronFitContract",
    "DigitalInstronGridContract",
    "DigitalInstronIndenterContract",
    "DigitalInstronManifestContract",
    "DigitalInstronTrialContract",
    "FootContactReference",
    "FootShoeAttachmentContract",
    "HumanShoeExperimentContract",
    "PoseFidelityReport",
    "PreparedAttachedSole",
    "ResolvedHumanShoeAttachment",
    "attach_sole_geometry",
    "compare_imported_state",
    "load_experiment",
    "load_manifest",
    "make_human_shoe_foundation_config",
    "prepare_attached_sole",
    "resolve_attachment",
]
