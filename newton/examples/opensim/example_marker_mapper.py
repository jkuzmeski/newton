# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Visually map exact C3D labels to marker roles on a native MJCF subject.

Command: ``python -m newton.examples marker_mapper``
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from projects.gait_c3d.c3d_adapter import C3DMarkerTrajectory, read_c3d_markers
from projects.gait_c3d.marker_map import (
    CALIBRATION_MARKER_SOURCES,
    NATIVE_MARKER_SOURCES,
    C3DMarkerMap,
    MarkerMapError,
    apply_c3d_marker_map,
    load_c3d_marker_map,
    required_c3d_sources,
    save_c3d_marker_map,
    validate_c3d_marker_map,
)
from projects.gait_c3d.native_motion_fit import marker_attachments_from_model, marker_positions_from_joint_q

_DEFAULT_SOURCE = "<use canonical label>"


def _subject_model_path(value: str | None) -> Path:
    """Resolve an MJCF file or a compiled subject bundle."""
    if value is None:
        return Path(__file__).resolve().parents[3] / "projects/gait_c3d/assets/s001_calibrated/model/subject.xml"
    path = Path(value).expanduser().resolve()
    if path.is_dir():
        path = path / "model" / "subject.xml"
    if not path.is_file():
        raise FileNotFoundError(f"subject MJCF is missing: {path}")
    return path


def _target_for_sources(attachment_names: tuple[str, ...]) -> dict[str, str]:
    """Return the native MJCF target for each canonical C3D source."""
    return {source: name for name in attachment_names for source in NATIVE_MARKER_SOURCES.get(name, ())}


def _normalized_label(value: str) -> str:
    """Normalize a label for display-only suggestions and filtering."""
    return "".join(character.casefold() for character in value if character.isalnum())


def _unique_name_suggestions(canonical: tuple[str, ...], sources: tuple[str, ...]) -> dict[str, str]:
    """Suggest unambiguous normalized-name matches without applying them."""
    suggestions = {}
    for target in canonical:
        target_key = _normalized_label(target)
        matches = [
            source
            for source in sources
            if target_key in _normalized_label(source) or _normalized_label(source) in target_key
        ]
        if len(matches) == 1:
            suggestions[target] = matches[0]
    duplicate_sources = {source for source in suggestions.values() if list(suggestions.values()).count(source) > 1}
    return {target: source for target, source in suggestions.items() if source not in duplicate_sources}


def _fit_display_registration(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit a proper row-vector rigid map used only by the visual overlay."""
    if len(source) == 0:
        return np.eye(3), np.zeros(3)
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    rotation = np.eye(3)
    if len(source) >= 3 and np.linalg.matrix_rank(source - source_center) >= 2:
        left, _, right_transpose = np.linalg.svd((source - source_center).T @ (target - target_center))
        rotation = left @ right_transpose
        if np.linalg.det(rotation) < 0.0:
            left[:, -1] *= -1.0
            rotation = left @ right_transpose
    return rotation, target_center - source_center @ rotation


class Example:
    """Edit an exact C3D label map while viewing its MJCF marker targets."""

    @staticmethod
    def create_parser():
        """Create the visual marker-mapping command line."""
        return _create_parser()

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.subject_path = _subject_model_path(args.subject)
        newton.use_coord_layout_targets = True
        builder = newton.ModelBuilder()
        builder.add_mjcf(str(self.subject_path), floating=False, parse_sites=True)
        self.model = builder.finalize(device=args.device)
        self.state = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        self.attachments = marker_attachments_from_model(self.model)
        self.attachment_names = tuple(attachment.name for attachment in self.attachments)
        self.target_positions = marker_positions_from_joint_q(self.model, self.attachments, self.model.joint_q.numpy())
        self.target_index = {name: index for index, name in enumerate(self.attachment_names)}
        self.source_target = _target_for_sources(self.attachment_names)
        self.required = required_c3d_sources(self.attachment_names)
        self.roles = (*self.required, *CALIBRATION_MARKER_SOURCES)
        self.demo = args.c3d is None
        self.raw_markers, default_map = self._load_markers(args)
        if len(self.raw_markers.times) == 0:
            raise ValueError("marker_mapper requires at least one C3D frame")
        self.map_path = (
            Path(args.marker_map).expanduser().resolve() if args.marker_map else Path("marker_map.json").resolve()
        )
        self.marker_map = (
            load_c3d_marker_map(self.map_path) if args.marker_map and self.map_path.is_file() else default_map
        )
        self.aliases = dict(self.marker_map.markers)
        self.edited_roles: set[str] = set()
        self.assignments = {}
        self._refresh_assignments()
        self.selected_target = 0
        self.selected_source = self.assignments.get(self.roles[0], _DEFAULT_SOURCE)
        self.source_filter = "" if self.selected_source != _DEFAULT_SOURCE else self.roles[0]
        self.frame = min(max(args.frame, 0), len(self.raw_markers.times) - 1)
        self.status = "Ready. Choose a target and its exact C3D label, then save."
        self.dirty = False
        self._fit_display_alignment()
        self._update_visuals()
        self.viewer.set_model(self.model)
        self.viewer.picking_enabled = False
        self.viewer.set_camera(pos=wp.vec3(2.0, -2.0, 1.15), pitch=-5.0, yaw=135.0)

    def _load_markers(self, args) -> tuple[C3DMarkerTrajectory, C3DMarkerMap]:
        """Load a real C3D or create the complete visual demonstration."""
        if args.c3d:
            return (
                read_c3d_markers(
                    args.c3d,
                    up_axis=args.c3d_up_axis,
                    forward_axis=args.c3d_forward_axis,
                    strip_prefix=not args.keep_c3d_prefix,
                ),
                C3DMarkerMap(),
            )
        source_names = tuple(f"LAB_{name}" for name in self.roles)
        aliases = dict(zip(self.roles, source_names, strict=True))
        positions = []
        for canonical in self.roles:
            if canonical in CALIBRATION_MARKER_SOURCES:
                sternum = self.target_positions[self.target_index["Sternum"]]
                top_head = self.target_positions[self.target_index["Top.Head"]]
                sacral = self.target_positions[self.target_index["V.Sacral"]]
                target = {
                    "C7": 0.75 * top_head + 0.25 * sternum,
                    "CLAV": sternum,
                    "T10": 0.5 * sternum + 0.5 * sacral,
                }[canonical]
                group = (canonical,)
            else:
                target = self.target_positions[self.target_index[self.source_target[canonical]]]
                group = NATIVE_MARKER_SOURCES[self.source_target[canonical]]
            member = group.index(canonical)
            angle = 2.0 * np.pi * member / len(group)
            radius = 0.018 if len(group) > 1 else 0.0
            positions.append(target + radius * np.asarray((np.cos(angle), np.sin(angle), 0.0)))
        frame = np.asarray(positions, dtype=np.float32)
        return (
            C3DMarkerTrajectory(
                times=np.asarray((0.0,), dtype=np.float64),
                positions=frame[None, :, :],
                valid=np.ones((1, len(source_names)), dtype=bool),
                marker_names=source_names,
                rate=100.0,
                first_frame=0,
                lab_to_newton=np.eye(3),
                source_file="marker_mapper_demo.c3d",
                source_sha256="0" * 64,
            ),
            C3DMarkerMap(aliases),
        )

    def _refresh_assignments(self) -> None:
        """Resolve configured aliases and exact identity labels for this C3D."""
        source_names = set(self.raw_markers.marker_names)
        self.assignments = {
            canonical: self.marker_map.source_for(canonical)
            for canonical in self.roles
            if self.marker_map.source_for(canonical) in source_names
        }

    def _current_map(
        self,
        assignments: dict[str, str] | None = None,
        edited_roles: set[str] | None = None,
    ) -> C3DMarkerMap:
        """Build a map while preserving loaded aliases that were not edited."""
        assignments = self.assignments if assignments is None else assignments
        edited_roles = self.edited_roles if edited_roles is None else edited_roles
        aliases = dict(self.aliases)
        for canonical in edited_roles:
            source = assignments.get(canonical)
            if source is None or source == canonical:
                aliases.pop(canonical, None)
            else:
                aliases[canonical] = source
        return C3DMarkerMap(aliases)

    def _assign(self, source: str) -> None:
        """Assign one source label or revert the selected role to identity."""
        canonical = self.roles[self.selected_target]
        assignments = dict(self.assignments)
        if source == _DEFAULT_SOURCE:
            if canonical in self.raw_markers.marker_names:
                assignments[canonical] = canonical
            else:
                assignments.pop(canonical, None)
        else:
            owner = next((name for name, value in assignments.items() if value == source and name != canonical), None)
            if owner is not None:
                self.status = f"{source} is already assigned to {owner}. Choose another source first."
                return
            assignments[canonical] = source
        edited_roles = self.edited_roles | {canonical}
        try:
            self._current_map(assignments, edited_roles)
        except MarkerMapError as error:
            self.status = str(error)
            return
        self.assignments = assignments
        self.edited_roles = edited_roles
        self.selected_source = self.assignments.get(canonical, _DEFAULT_SOURCE)
        self.dirty = True
        self.status = "Unsaved changes"
        self._update_visuals()

    def _eligible_suggestions(self) -> dict[str, str]:
        """Return suggestions that do not conflict with current assignments."""
        suggestions = _unique_name_suggestions(self.roles, self.raw_markers.marker_names)
        used = set(self.assignments.values())
        eligible = {}
        for canonical in self.roles:
            source = suggestions.get(canonical)
            if canonical in self.assignments or source is None or source in used:
                continue
            candidate = self.assignments | {canonical: source}
            try:
                self._current_map(candidate, self.edited_roles | {canonical})
            except MarkerMapError:
                continue
            eligible[canonical] = source
            used.add(source)
        return eligible

    def _accept_suggestions(self) -> None:
        """Apply unique normalized-name suggestions after an explicit click."""
        suggestions = self._eligible_suggestions()
        self.assignments.update(suggestions)
        self.edited_roles.update(suggestions)
        self.dirty = bool(suggestions) or self.dirty
        self.status = f"Applied {len(suggestions)} unique name suggestions. Review them before saving."
        self.selected_source = self.assignments.get(self.roles[self.selected_target], _DEFAULT_SOURCE)
        self._update_visuals()

    def _move_target(self, offset: int, *, unmapped_only: bool = False) -> None:
        """Select another target, optionally skipping completed assignments."""
        for step in range(1, len(self.roles) + 1):
            index = (self.selected_target + offset * step) % len(self.roles)
            if not unmapped_only or self.roles[index] not in self.assignments:
                self.selected_target = index
                break
        canonical = self.roles[self.selected_target]
        self.selected_source = self.assignments.get(canonical, _DEFAULT_SOURCE)
        if self.selected_source == _DEFAULT_SOURCE:
            self.source_filter = canonical
        self._update_visuals()

    def _fit_display_alignment(self) -> None:
        """Lock display alignment from the current reviewed mapping state."""
        frame_positions = self.raw_markers.positions[self.frame].astype(np.float64)
        frame_valid = self.raw_markers.valid[self.frame]
        source_index = {name: index for index, name in enumerate(self.raw_markers.marker_names)}
        paired_source = []
        paired_target = []
        for canonical in self.required:
            source = self.assignments.get(canonical)
            if source is None:
                continue
            column = source_index[source]
            if frame_valid[column]:
                paired_source.append(frame_positions[column])
                paired_target.append(self.target_positions[self.target_index[self.source_target[canonical]]])
        if paired_source:
            self.display_rotation, self.display_translation = _fit_display_registration(
                np.asarray(paired_source), np.asarray(paired_target)
            )
        else:
            self.display_rotation = np.eye(3)
            source_center = frame_positions[frame_valid].mean(axis=0) if np.any(frame_valid) else np.zeros(3)
            self.display_translation = self.target_positions.mean(axis=0) - source_center

    def _update_visuals(self) -> None:
        """Update point colors and lines under the locked display alignment."""
        frame_positions = self.raw_markers.positions[self.frame].astype(np.float64)
        frame_valid = self.raw_markers.valid[self.frame]
        source_index = {name: index for index, name in enumerate(self.raw_markers.marker_names)}
        aligned = frame_positions @ self.display_rotation + self.display_translation
        selected_canonical = self.roles[self.selected_target]
        selected_source = self.assignments.get(selected_canonical)
        source_colors = np.tile((0.42, 0.42, 0.45), (len(aligned), 1))
        for source in self.assignments.values():
            column = source_index[source]
            if frame_valid[column]:
                source_colors[column] = (0.15, 0.90, 0.25)
        if selected_source is not None and frame_valid[source_index[selected_source]]:
            source_colors[source_index[selected_source]] = (1.0, 0.55, 0.05)
        source_radii = np.where(frame_valid, 0.011, 0.0).astype(np.float32)
        target_colors = np.asarray(
            [
                (0.10, 0.75, 0.98)
                if all(source in self.assignments for source in NATIVE_MARKER_SOURCES[name])
                else (0.95, 0.15, 0.10)
                for name in self.attachment_names
            ],
            dtype=np.float32,
        )
        if selected_canonical in self.source_target:
            target_colors[self.target_index[self.source_target[selected_canonical]]] = (1.0, 0.55, 0.05)
        starts = []
        ends = []
        for canonical, source in self.assignments.items():
            column = source_index[source]
            if frame_valid[column] and canonical in self.source_target:
                starts.append(aligned[column])
                ends.append(self.target_positions[self.target_index[self.source_target[canonical]]])
        self.source_points = wp.array(aligned.astype(np.float32), dtype=wp.vec3, device=self.model.device)
        self.source_radii = wp.array(source_radii, dtype=wp.float32, device=self.model.device)
        self.source_colors = wp.array(source_colors.astype(np.float32), dtype=wp.vec3, device=self.model.device)
        self.target_points = wp.array(self.target_positions.astype(np.float32), dtype=wp.vec3, device=self.model.device)
        self.target_colors = wp.array(target_colors, dtype=wp.vec3, device=self.model.device)
        self.line_starts = wp.array(
            np.asarray(starts, dtype=np.float32).reshape(-1, 3), dtype=wp.vec3, device=self.model.device
        )
        self.line_ends = wp.array(
            np.asarray(ends, dtype=np.float32).reshape(-1, 3), dtype=wp.vec3, device=self.model.device
        )

    def _save(self) -> None:
        """Save the current aliases and report required-marker coverage."""
        try:
            marker_map = self._current_map()
        except MarkerMapError as error:
            self.status = str(error)
            return
        try:
            saved = save_c3d_marker_map(marker_map, self.map_path)
        except OSError as error:
            self.status = f"Could not save marker map: {error}"
            return
        validation = validate_c3d_marker_map(self.raw_markers, marker_map, required=self.required)
        missing = len(validation.issues)
        optional_missing = sum(name not in self.assignments for name in CALIBRATION_MARKER_SOURCES)
        self.marker_map = marker_map
        self.aliases = dict(marker_map.markers)
        self.edited_roles.clear()
        self.dirty = False
        if missing:
            self.status = f"Saved draft: {saved} ({missing} required labels unresolved)"
        elif optional_missing:
            self.status = f"Saved usable map: {saved} ({optional_missing} optional torso roles unresolved)"
        else:
            self.status = f"Saved complete map: {saved}"

    def gui(self, ui):
        """Draw the interactive marker assignment panel."""
        required_mapped = sum(name in self.assignments for name in self.required)
        ui.text(f"Required {required_mapped}/{len(self.required)}; all roles {len(self.assignments)}/{len(self.roles)}")
        ui.text(f"Frame {self.frame + 1}/{len(self.raw_markers.times)}")
        ui.separator()
        target_labels = [
            f"{name} (calibration only; no MJCF site)"
            if name in CALIBRATION_MARKER_SOURCES
            else f"{name}  ->  {self.source_target[name]}"
            for name in self.roles
        ]
        changed, target = ui.combo("MJCF target role", self.selected_target, target_labels, 12)
        if changed:
            self.selected_target = target
            canonical = self.roles[target]
            self.selected_source = self.assignments.get(canonical, _DEFAULT_SOURCE)
            if self.selected_source == _DEFAULT_SOURCE:
                self.source_filter = canonical
            self._update_visuals()
        changed, self.source_filter = ui.input_text("Filter C3D labels", self.source_filter)
        filter_text = _normalized_label(self.source_filter)
        source_labels = [name for name in self.raw_markers.marker_names if filter_text in _normalized_label(name)]
        if self.selected_source != _DEFAULT_SOURCE and self.selected_source not in source_labels:
            source_labels.insert(0, self.selected_source)
        source_options = [_DEFAULT_SOURCE, *source_labels]
        source_index = source_options.index(self.selected_source) if self.selected_source in source_options else 0
        changed, source_index = ui.combo("C3D source label", source_index, source_options, 16)
        if changed:
            self._assign(source_options[source_index])
        if ui.button("Previous"):
            self._move_target(-1)
        ui.same_line()
        if ui.button("Next"):
            self._move_target(1)
        ui.same_line()
        if ui.button("Next missing"):
            self._move_target(1, unmapped_only=True)
        suggestions = self._eligible_suggestions()
        if suggestions and ui.button(f"Apply {len(suggestions)} name suggestions"):
            self._accept_suggestions()
        if len(self.raw_markers.times) > 1:
            changed, frame = ui.slider_int("C3D frame", self.frame, 0, len(self.raw_markers.times) - 1)
            if changed:
                self.frame = frame
                self._update_visuals()
        if ui.button("Fit and lock display from current map"):
            self._fit_display_alignment()
            self._update_visuals()
            self.status = "Display alignment fitted and locked. It will not change while editing."
        if ui.button("Save marker map *" if self.dirty else "Save marker map"):
            self._save()
        ui.text_wrapped(self.status)
        ui.separator()
        ui.text("Orange: selected")
        ui.text("Green + lines: mapped C3D")
        ui.text("Blue: complete MJCF target recipe")
        ui.text("Red: incomplete MJCF target recipe")

    def step(self):
        """Keep the neutral mapping view static."""

    def render(self):
        """Render the MJCF, C3D markers, targets, and current assignments."""
        self.viewer.begin_frame(float(self.raw_markers.times[self.frame]))
        self.viewer.log_state(self.state)
        self.viewer.log_points("/marker_mapper/c3d", self.source_points, self.source_radii, self.source_colors)
        self.viewer.log_points("/marker_mapper/mjcf", self.target_points, 0.014, self.target_colors)
        if len(self.line_starts):
            self.viewer.log_lines("/marker_mapper/assignments", self.line_starts, self.line_ends, (0.15, 0.90, 0.25))
        else:
            self.viewer.log_lines("/marker_mapper/assignments", None, None, None)
        self.viewer.end_frame()

    def test_final(self):
        """Verify that the mapper keeps finite visual geometry and a complete demo."""
        if not np.all(np.isfinite(self.target_positions)):
            raise ValueError("MJCF marker targets must be finite")
        if len(self.roles) != len(set(self.roles)):
            raise ValueError("canonical marker roles must be unique")
        radii = self.source_radii.numpy()
        if np.any(radii[~self.raw_markers.valid[self.frame]] != 0.0):
            raise ValueError("invalid C3D samples must be hidden")
        if self.demo:
            if set(self.assignments) != set(self.roles):
                raise ValueError("the visual marker-mapping demonstration must be complete")
            canonical = apply_c3d_marker_map(self.raw_markers, self._current_map(), required=self.required)
            if not set(self.roles).issubset(canonical.marker_names):
                raise ValueError("the demonstration map did not produce every canonical role")


def _create_parser():
    """Create the visual marker-mapping command line."""
    parser = newton.examples.create_parser()
    parser.add_argument("--subject", help="Compiled subject bundle or MJCF file; defaults to calibrated S001")
    parser.add_argument("--c3d", help="C3D marker trial; omit for the built-in visual mapping demonstration")
    parser.add_argument("--marker-map", help="Marker-map JSON to load and update; defaults to ./marker_map.json")
    parser.add_argument("--frame", type=int, default=0, help="Initial C3D frame")
    parser.add_argument("--c3d-up-axis", default="+Z", help="Lab axis that points upward")
    parser.add_argument("--c3d-forward-axis", default="-Y", help="Lab axis that points subject-forward")
    parser.add_argument(
        "--keep-c3d-prefix",
        action="store_true",
        help="Keep subject prefixes such as Person01:LASI when matching exact labels",
    )
    return parser


def create_parser():
    """Create the visual marker-mapping command line."""
    return _create_parser()


if __name__ == "__main__":
    viewer, args = newton.examples.init(create_parser())
    example = Example(viewer, args)
    newton.examples.run(example, args)
