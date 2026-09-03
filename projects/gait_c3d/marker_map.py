# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Validate and apply exact C3D marker-label aliases for gait protocols."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from .c3d_adapter import C3DMarkerTrajectory

MARKER_MAP_SCHEMA_VERSION = "gait_c3d_marker_map_1"

# The insertion order matches the canonical S001 native marker layout. Virtual
# targets have fixed source recipes; an alias file can rename only raw inputs.
NATIVE_MARKER_SOURCES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "Sternum": ("STRN",),
        "R.Acromium": ("RSHO",),
        "L.Acromium": ("LSHO",),
        "Top.Head": ("LFHD", "RFHD", "LBHD", "RBHD"),
        "R.ASIS": ("RASI",),
        "L.ASIS": ("LASI",),
        "V.Sacral": ("LPSI", "RPSI"),
        "R.Thigh.Centroid": ("RTH2", "RTH3", "RTH4"),
        "R.Knee.Lat": ("RKNE",),
        "R.Knee.Med": ("RMKNE",),
        "R.Shank.Centroid": ("RTIB2", "RTIB3", "RTIB4"),
        "R.Ankle.Lat": ("RANK",),
        "R.Ankle.Med": ("RMANK",),
        "R.Heel": ("RHEE",),
        "R.Toe.Lat": ("RMTH5",),
        "R.Toe.Med": ("RMTH1",),
        "R.Toe.Tip": ("RHLX",),
        "L.Thigh.Centroid": ("LTH2", "LTH3", "LTH4"),
        "L.Knee.Lat": ("LKNE",),
        "L.Knee.Med": ("LMKNE",),
        "L.Shank.Centroid": ("LTIB2", "LTIB3", "LTIB4"),
        "L.Ankle.Lat": ("LANK",),
        "L.Ankle.Med": ("LMANK",),
        "L.Heel": ("LHEE",),
        "L.Toe.Lat": ("LMTH5",),
        "L.Toe.Med": ("LMTH1",),
        "L.Toe.Tip": ("LHLX",),
    }
)


def _ordered_unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


CALIBRATION_MARKER_SOURCES = ("C7", "CLAV", "T10")
"""Optional torso landmarks used when building calibrated subject geometry."""

CANONICAL_C3D_MARKERS = _ordered_unique(
    source for sources in (*NATIVE_MARKER_SOURCES.values(), CALIBRATION_MARKER_SOURCES) for source in sources
)
"""Allowed canonical raw C3D labels in S001 native-layout and calibration order."""

CANONICAL_C3D_LABELS = CANONICAL_C3D_MARKERS
"""Alias for :data:`CANONICAL_C3D_MARKERS`."""

_CANONICAL_C3D_MARKER_SET = frozenset(CANONICAL_C3D_MARKERS)


@dataclass(frozen=True, slots=True)
class MarkerMapIssue:
    """One independently actionable marker-map validation issue."""

    code: str
    """Stable machine-readable issue code."""

    message: str
    """Concise human-readable issue description."""

    canonical: str | None = None
    """Canonical C3D label associated with the issue, if any."""

    source: str | None = None
    """Exact source C3D label associated with the issue, if any."""


class MarkerMapError(ValueError):
    """Report all marker-map issues found during one validation pass."""

    def __init__(self, issues: Iterable[MarkerMapIssue]):
        self.issues = tuple(issues)
        if not self.issues:
            raise ValueError("MarkerMapError requires at least one issue")
        lines = [f"marker mapping failed ({len(self.issues)} issue{'s' if len(self.issues) != 1 else ''})"]
        lines.extend(f"  - {issue.message}" for issue in self.issues)
        super().__init__("\n".join(lines))


@dataclass(frozen=True, slots=True)
class C3DMarkerMap:
    """A versioned immutable map from canonical labels to exact source labels.

    Entries omitted from :attr:`markers` use identity mapping.
    """

    markers: Mapping[str, str] = field(default_factory=dict)
    """Canonical-to-source label aliases; omitted canonical labels are identity."""

    schema_version: str = MARKER_MAP_SCHEMA_VERSION
    """Marker-map JSON schema version."""

    def __post_init__(self) -> None:
        issues, aliases = _validate_marker_map(self.schema_version, self.markers)
        if issues:
            raise MarkerMapError(issues)
        object.__setattr__(self, "markers", MappingProxyType(aliases))

    def source_for(self, canonical: str) -> str:
        """Return the exact configured source for one canonical label."""
        if canonical not in _CANONICAL_C3D_MARKER_SET:
            raise KeyError(f"unknown canonical C3D marker: {canonical!r}")
        return self.markers.get(canonical, canonical)

    def as_dict(self) -> dict[str, Any]:
        """Return a mutable JSON-compatible representation."""
        return {
            "schema_version": self.schema_version,
            "markers": dict(self.markers),
        }


@dataclass(frozen=True, slots=True)
class MarkerMapResolution:
    """One canonical label resolved to an exact source trajectory column."""

    canonical: str
    """Canonical gait C3D label."""

    source: str
    """Exact decoded source label."""

    source_index: int
    """Source column index in the raw trajectory."""

    @property
    def is_identity(self) -> bool:
        """Return whether canonical and source labels are identical."""
        return self.canonical == self.source


@dataclass(frozen=True, slots=True)
class MarkerMapValidation:
    """Immutable coverage report for one trajectory and marker map."""

    required: tuple[str, ...]
    """Caller-provided required canonical labels in caller order."""

    resolved: tuple[MarkerMapResolution, ...]
    """Available canonical labels in canonical protocol order."""

    unused_source_labels: tuple[str, ...]
    """Decoded source labels not selected by the canonical protocol."""

    issues: tuple[MarkerMapIssue, ...]
    """All validation issues found in one pass."""

    @property
    def is_valid(self) -> bool:
        """Return whether the requested marker set can be applied."""
        return not self.issues

    def raise_for_errors(self) -> None:
        """Raise one grouped error if validation found any issues."""
        if self.issues:
            raise MarkerMapError(self.issues)


def required_c3d_sources(attachment_names: Iterable[str]) -> tuple[str, ...]:
    """Return ordered canonical sources needed by native marker attachments.

    Sources follow attachment order and each fixed virtual-marker recipe order.
    Repeated sources are included only at their first occurrence.

    Args:
        attachment_names: Native MJCF marker-site names.

    Returns:
        Canonical raw C3D labels required to construct the attachments.
    """
    if isinstance(attachment_names, str):
        attachment_names = (attachment_names,)
    names = tuple(attachment_names)
    invalid = [name for name in names if not isinstance(name, str) or name not in NATIVE_MARKER_SOURCES]
    if invalid:
        raise ValueError(f"unknown native marker attachment names: {invalid}")
    return _ordered_unique(source for name in names for source in NATIVE_MARKER_SOURCES[name])


def _validate_marker_map(
    schema_version: object,
    markers: object,
) -> tuple[tuple[MarkerMapIssue, ...], dict[str, str]]:
    issues = []
    aliases: dict[str, str] = {}
    if schema_version != MARKER_MAP_SCHEMA_VERSION:
        issues.append(
            MarkerMapIssue(
                "unsupported_schema_version",
                f"schema_version must be {MARKER_MAP_SCHEMA_VERSION!r}, got {schema_version!r}",
            )
        )
    if not isinstance(markers, Mapping):
        issues.append(MarkerMapIssue("invalid_markers", "markers must be a JSON object"))
        return tuple(issues), aliases

    for canonical, source in markers.items():
        if not isinstance(canonical, str) or canonical not in _CANONICAL_C3D_MARKER_SET:
            issues.append(
                MarkerMapIssue(
                    "unknown_canonical_marker",
                    f"unknown canonical marker key: {canonical!r}",
                    canonical=canonical if isinstance(canonical, str) else None,
                )
            )
            continue
        if not isinstance(source, str) or not source or source != source.strip():
            issues.append(
                MarkerMapIssue(
                    "invalid_source_label",
                    f"source label for {canonical!r} must be a nonempty exact string without outer whitespace",
                    canonical=canonical,
                    source=source if isinstance(source, str) else None,
                )
            )
            continue
        aliases[canonical] = source

    source_owners: dict[str, str] = {}
    for canonical in CANONICAL_C3D_MARKERS:
        source = aliases.get(canonical, canonical)
        previous = source_owners.get(source)
        if previous is not None:
            issues.append(
                MarkerMapIssue(
                    "duplicate_source_label",
                    f"canonical markers {previous!r} and {canonical!r} both resolve to source {source!r}",
                    canonical=canonical,
                    source=source,
                )
            )
        else:
            source_owners[source] = canonical
    return tuple(issues), aliases


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise MarkerMapError((MarkerMapIssue("duplicate_json_key", f"duplicate JSON key: {key!r}"),))
        value[key] = item
    return value


def _marker_map_from_json(value: object) -> C3DMarkerMap:
    if not isinstance(value, dict):
        raise MarkerMapError((MarkerMapIssue("invalid_document", "marker map must be a JSON object"),))
    allowed_fields = {"schema_version", "markers"}
    issues = [
        MarkerMapIssue("unknown_field", f"unknown marker-map field: {name!r}")
        for name in value
        if name not in allowed_fields
    ]
    if "schema_version" not in value:
        issues.append(MarkerMapIssue("missing_field", "marker map is missing 'schema_version'"))
    if "markers" not in value:
        issues.append(MarkerMapIssue("missing_field", "marker map is missing 'markers'"))
    validation_issues, aliases = _validate_marker_map(value.get("schema_version"), value.get("markers"))
    issues.extend(validation_issues)
    if issues:
        raise MarkerMapError(issues)
    return C3DMarkerMap(aliases, str(value["schema_version"]))


def load_c3d_marker_map(path: str | os.PathLike) -> C3DMarkerMap:
    """Load and validate one versioned marker alias map.

    Args:
        path: Marker-map JSON file.

    Returns:
        Validated immutable marker map.
    """
    marker_map_path = Path(path).expanduser().resolve()
    try:
        value = json.loads(
            marker_map_path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except MarkerMapError:
        raise
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise MarkerMapError((MarkerMapIssue("invalid_json", f"marker map is not valid JSON: {error}"),)) from error
    return _marker_map_from_json(value)


def save_c3d_marker_map(marker_map: C3DMarkerMap, path: str | os.PathLike) -> Path:
    """Atomically save one marker alias map as deterministic JSON.

    Args:
        marker_map: Validated marker map.
        path: Destination JSON file.

    Returns:
        Absolute destination path.
    """
    if not isinstance(marker_map, C3DMarkerMap):
        raise TypeError("marker_map must be a C3DMarkerMap")
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output.parent, delete=False) as stream:
        staged = Path(stream.name)
        json.dump(marker_map.as_dict(), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    try:
        os.replace(staged, output)
    except BaseException:
        staged.unlink(missing_ok=True)
        raise
    return output


def validate_c3d_marker_map(
    trajectory: C3DMarkerTrajectory,
    marker_map: C3DMarkerMap | None = None,
    *,
    required: Iterable[str] = (),
) -> MarkerMapValidation:
    """Validate exact source coverage and return all required-label issues.

    Args:
        trajectory: Raw decoded C3D marker trajectory.
        marker_map: Marker aliases, or ``None`` for the S001 identity profile.
        required: Canonical labels required by the caller.

    Returns:
        Immutable coverage and grouped-issue report.
    """
    if marker_map is None:
        marker_map = S001_MARKER_MAP
    if not isinstance(marker_map, C3DMarkerMap):
        raise TypeError("marker_map must be a C3DMarkerMap")
    if isinstance(required, str):
        required = (required,)
    required_values = tuple(required)
    issues = []
    valid_required = []
    seen_required: set[str] = set()
    for canonical in required_values:
        if not isinstance(canonical, str) or canonical not in _CANONICAL_C3D_MARKER_SET:
            issues.append(
                MarkerMapIssue(
                    "unknown_required_marker",
                    f"unknown required canonical marker: {canonical!r}",
                    canonical=canonical if isinstance(canonical, str) else None,
                )
            )
        elif canonical in seen_required:
            issues.append(
                MarkerMapIssue(
                    "duplicate_required_marker",
                    f"required canonical marker is repeated: {canonical!r}",
                    canonical=canonical,
                )
            )
        else:
            seen_required.add(canonical)
            valid_required.append(canonical)

    source_index = {name: index for index, name in enumerate(trajectory.marker_names)}
    resolved = []
    for canonical in CANONICAL_C3D_MARKERS:
        source = marker_map.source_for(canonical)
        column = source_index.get(source)
        if column is not None:
            resolved.append(MarkerMapResolution(canonical, source, column))
    resolved_canonical = {item.canonical for item in resolved}
    for canonical in valid_required:
        if canonical not in resolved_canonical:
            source = marker_map.source_for(canonical)
            issues.append(
                MarkerMapIssue(
                    "missing_source_label",
                    f"required marker {canonical!r} resolves to missing exact source label {source!r}",
                    canonical=canonical,
                    source=source,
                )
            )

    used_sources = {item.source for item in resolved}
    unused = tuple(name for name in trajectory.marker_names if name not in used_sources)
    return MarkerMapValidation(
        tuple(required_values),
        tuple(resolved),
        unused,
        tuple(issues),
    )


def apply_c3d_marker_map(
    trajectory: C3DMarkerTrajectory,
    marker_map: C3DMarkerMap | None = None,
    *,
    required: Iterable[str] = (),
) -> C3DMarkerTrajectory:
    """Return an independent trajectory with exact source aliases canonicalized.

    Explicit aliases are authoritative. A same-named canonical raw column is
    not used as a fallback when its configured source is missing. Raw labels
    outside the canonical protocol are retained unless they would shadow an
    explicitly mapped canonical output.

    Args:
        trajectory: Raw decoded C3D marker trajectory.
        marker_map: Marker aliases, or ``None`` for the S001 identity profile.
        required: Canonical labels whose sources must all be present.

    Returns:
        Canonicalized marker trajectory with copied arrays and provenance.
    """
    if marker_map is None:
        marker_map = S001_MARKER_MAP
    validation = validate_c3d_marker_map(trajectory, marker_map, required=required)
    validation.raise_for_errors()
    canonical_by_source = {item.source: item.canonical for item in validation.resolved}
    explicit_targets = frozenset(marker_map.markers)
    source_columns = []
    output_names = []
    for column, source in enumerate(trajectory.marker_names):
        canonical = canonical_by_source.get(source)
        if canonical is not None:
            source_columns.append(column)
            output_names.append(canonical)
        elif source not in explicit_targets:
            source_columns.append(column)
            output_names.append(source)

    return C3DMarkerTrajectory(
        times=trajectory.times.copy(),
        positions=np.take(trajectory.positions, source_columns, axis=1).copy(),
        valid=np.take(trajectory.valid, source_columns, axis=1).copy(),
        marker_names=tuple(output_names),
        rate=trajectory.rate,
        first_frame=trajectory.first_frame,
        lab_to_newton=trajectory.lab_to_newton.copy(),
        source_file=trajectory.source_file,
        source_sha256=trajectory.source_sha256,
    )


def load_subject_c3d_marker_map(
    subject_dir: str | os.PathLike,
    manifest: Mapping[str, Any],
) -> tuple[C3DMarkerMap | None, Path | None, dict[str, Any] | None]:
    """Load and verify the optional marker map declared by a subject bundle.

    Args:
        subject_dir: Compiled subject bundle root.
        manifest: Parsed subject manifest.

    Returns:
        Marker map, resolved path, and mapping metadata, or three ``None`` values.
    """
    artifacts = manifest.get("artifacts")
    value = artifacts.get("marker_map") if isinstance(artifacts, Mapping) else None
    if value is None:
        return None, None, None
    if not isinstance(value, str) or not value:
        raise ValueError("subject marker-map artifact must be a safe relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("subject marker-map artifact must be a safe relative path")
    root = Path(subject_dir).expanduser().resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("subject marker-map artifact escapes its bundle") from error
    if not path.is_file():
        raise FileNotFoundError(f"subject marker-map artifact is missing: {path}")
    metadata = manifest.get("marker_mapping")
    if metadata is not None:
        if not isinstance(metadata, dict):
            raise ValueError("subject marker-map metadata is invalid")
        expected = metadata.get("sha256")
        strip_prefix = metadata.get("strip_prefix", True)
        if metadata.get("file") != value or not isinstance(expected, str) or not isinstance(strip_prefix, bool):
            raise ValueError("subject marker-map metadata is invalid")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("subject marker-map hash mismatch")
    return load_c3d_marker_map(path), path, metadata


S001_MARKER_MAP = C3DMarkerMap()
"""Identity marker map for the canonical S001 gait C3D protocol."""

S001_IDENTITY_MARKER_MAP = S001_MARKER_MAP
"""Explicit alias for the canonical S001 identity marker-map profile."""
