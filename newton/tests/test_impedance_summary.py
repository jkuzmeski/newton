# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the illustrated impedance report: figures, Markdown subset, axes and payload."""

import json
import os
import re
import tempfile
import unittest
import xml.etree.ElementTree as ElementTree
from html.parser import HTMLParser
from pathlib import Path

import numpy as np

from projects.impedance_instron.summary import (
    CONTACT_FORCE_FRACTION,
    DEFAULT_PROFILE_NAME,
    DEFAULT_REPORT,
    FIGURE_BUILDERS,
    FIGURE_PLACEMENT,
    INFRASTRUCTURE,
    MOMENTUM_EXCURSION,
    REPORT_PEAK_TIMING_PCT,
    STIFF_LIMIT_K_THETA,
    STIFF_LIMIT_SERIES,
    Series,
    axis_values,
    build_figures,
    cumulative_impulse,
    discover_archives,
    group_by_material,
    load_npz_text,
    load_npz_trace,
    load_profile_context,
    load_sources,
    optional_token,
    parse_blocks,
    parse_evaluation_record,
    peak_percent,
    peak_timing_comparison,
    percent_series,
    plot,
    render_block,
    render_page,
    sample_step_statistics,
    summary_payload,
    write_summary,
)

_NONFINITE = re.compile(r"\b(nan|-?inf|infinity)\b", re.IGNORECASE)

_LEGACY_REPORT_FIXTURE = "\n\n".join(
    [f"# {FIGURE_PLACEMENT[0][0]}", "Synthetic historical report. Local dashboard: http://localhost:8050."]
    + [
        f"## {heading}\n\nSynthetic section for figures {', '.join(letters)}."
        for heading, letters in FIGURE_PLACEMENT[1:]
    ]
)

_TRACE_COLUMNS = (
    "time_s",
    "reference_fz_n",
    "shoe_fz_n",
    "reference_fx_n",
    "shoe_fx_n",
    "com_z_m",
    "reference_com_z_m",
    "com_vz_m_s",
    "reference_com_vz_m_s",
)


def _shape(time_s: np.ndarray) -> np.ndarray:
    """Return a smooth unit stance pulse over a time axis."""
    return np.sin(np.pi * np.clip((time_s - 0.05) / 0.30, 0.0, 1.0)) ** 2


def _write_csv(path: Path, scale: float = 1.0, samples: int = 400) -> None:
    """Write a synthetic controller trace CSV with the pinned column names."""
    path.parent.mkdir(parents=True, exist_ok=True)
    time_s = np.linspace(0.0, 0.375, samples)
    pulse = _shape(time_s)
    columns = {
        "time_s": time_s,
        "reference_fz_n": 1760.0 * pulse,
        "shoe_fz_n": 1760.0 * scale * pulse,
        "reference_fx_n": -200.0 * np.gradient(pulse),
        "shoe_fx_n": -200.0 * scale * np.gradient(pulse),
        "com_z_m": 1.0 - 0.05 * pulse,
        "reference_com_z_m": 1.0 - 0.045 * pulse,
        "com_vz_m_s": np.gradient(1.0 - 0.05 * pulse, time_s),
        "reference_com_vz_m_s": np.gradient(1.0 - 0.045 * pulse, time_s),
    }
    rows = [",".join(_TRACE_COLUMNS)]
    for index in range(time_s.size):
        rows.append(",".join(f"{float(columns[name][index]):.9g}" for name in _TRACE_COLUMNS))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_npz(path: Path, samples: int = 46, substep_dt: bool = False, artifact: str | None = None) -> None:
    """Write a synthetic evaluation archive with a declared contact window.

    Args:
        path: Destination archive.
        samples: Number of samples, coarse at 46 frame boundaries and full at 2881 substeps.
        substep_dt: Also write the archive's own ``substep_dt_s`` field.
        artifact: Material token stored beside the channels, as the trainer now writes it.
    """
    time_s = np.linspace(0.0, 0.375, samples)
    pulse = _shape(time_s)
    extra: dict[str, object] = {"substep_dt_s": float(time_s[1] - time_s[0])} if substep_dt else {}
    if artifact is not None:
        extra["artifact"] = np.array(artifact)
    np.savez_compressed(
        path,
        **extra,
        iteration=np.int64(2500),
        time_s=time_s,
        shoe_fz_n=1800.0 * pulse,
        reference_fz_n=1760.0 * pulse,
        shoe_fx_n=-210.0 * np.gradient(pulse),
        reference_fx_n=-200.0 * np.gradient(pulse),
        com_z_m=1.0 - 0.048 * pulse,
        reference_com_z_m=1.0 - 0.045 * pulse,
        com_vz_m_s=np.gradient(1.0 - 0.048 * pulse, time_s),
        reference_com_vz_m_s=np.gradient(1.0 - 0.045 * pulse, time_s),
        leg_length_m=0.95 - 0.04 * pulse,
        commanded_length_m=0.95 - 0.03 * pulse,
        stiffness_n_m=24000.0 + 3000.0 * pulse,
        damping_ratio=0.3 + 0.1 * pulse,
        contact_start_s=0.05,
        contact_end_s=0.35,
    )


def _eval_line(iteration: int, momentum: float, objective: float, artifact: str | None = None) -> str:
    """Build one deterministic-evaluation record in the pinned training-log format.

    Args:
        iteration: Iteration of the record.
        momentum: Momentum excursion in multiples of the deadband.
        objective: Work objective [J].
        artifact: Material token appended after ``trace``, as the trainer now writes it.
    """
    line = (
        f"eval iteration={iteration} objective_j={objective:.3f} feasible=1 on_task=0 "
        f"excursion_duration=0.000 excursion_impulse=0.000 excursion_momentum={momentum:.3f} "
        f"violation_total=0.000 eval_return=-40.000 peak_fz_n=1809.629 peak_fz_ref_n=1762.554 "
        f"peak_time_pct=43.945 peak_time_ref_pct=42.690 fz_rms_n=218.011 impulse_err_pct=2.493 "
        f"com_vz_rms=0.141 com_z_rms_mm=24.270 contact_ms=301.042 peak_compression_mm=29.596"
    )
    if artifact is None:
        return line
    return f"{line} trace=/path/policy_ankle_v2_{artifact}.eval.npz artifact={artifact}"


def _write_profile(path: Path, mass_kg: float = 81.93121179999996) -> None:
    """Write the fields of the stance profile that the page reads."""
    path.write_text(
        json.dumps(
            {
                "mass_kg": mass_kg,
                "source_time_s": [89.9715, 89.9725],
                "provenance": {
                    "com_surrogate": {"gravity_m_s2": 9.80665},
                    "running": {"selected_stance_source_s": [90.0115, 90.3065]},
                },
            }
        ),
        encoding="utf-8",
    )


def _write_sources(directory: Path, samples: int = 400, archive_samples: int = 46) -> None:
    """Populate a directory with every artifact the figures read.

    Args:
        directory: Destination directory.
        samples: Substep count of the CSV traces.
        archive_samples: Sample count of the policy evaluation archive.
    """
    _write_profile(directory / "stance_planar_context.json")
    _write_csv(directory / "legacy_compare" / "trace.csv", scale=1.07, samples=samples)
    _write_csv(directory / "eval_j" / "trace.csv", scale=1.02, samples=samples)
    _write_npz(directory / "policy_ankle_v2.eval.npz", samples=archive_samples)
    lines = ["Warp 1.17.0 initialized:"]
    lines += [_eval_line(50 * step, 1.6 - 0.02 * step, 440.0 - 2.0 * step) for step in range(1, 11)]
    (directory / "train_ankle_v2.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # The prescribed-pitch runs predate evaluation logging, so they carry iteration lines only.
    iteration = (
        "iter  600 | return mean=   -84.085 med=   -82.989 best=   -80.682 | "
        "pi=-0.0046 v=0.8690 ent=+2.040 | ev=+0.992 kl=0.00499 clip=0.058"
    )
    for name in ("train_v3.log", "train_v5.log"):
        (directory / name).write_text(f"{iteration}\n", encoding="utf-8")


def _assert_xml(case: unittest.TestCase, markup: str) -> ElementTree.Element:
    """Parse markup as XML and fail the case when it is not well formed."""
    try:
        return ElementTree.fromstring(markup)
    except ElementTree.ParseError as error:  # pragma: no cover - only on a rendering regression
        case.fail(f"markup is not well-formed XML: {error}")
    return ElementTree.Element("unreachable")  # pragma: no cover


class FigureTest(unittest.TestCase):
    """Cover figure rendering against complete and missing artifacts."""

    @classmethod
    def setUpClass(cls):
        """Render every figure once from a synthetic but complete artifact set."""
        cls._temporary = tempfile.TemporaryDirectory()
        cls.directory = Path(cls._temporary.name)
        _write_sources(cls.directory)
        cls.sources = load_sources(cls.directory)
        cls.figures = build_figures(cls.sources)

    @classmethod
    def tearDownClass(cls):
        """Remove the synthetic artifacts."""
        cls._temporary.cleanup()

    def test_every_figure_is_well_formed_xml(self):
        """Check every figure parses as XML and carries a caption and an SVG panel."""
        for letter, title, _ in FIGURE_BUILDERS:
            with self.subTest(figure=letter):
                markup = self.figures[letter]
                root = _assert_xml(self, markup)
                self.assertEqual(root.tag, "figure")
                self.assertIsNotNone(root.find("figcaption"))
                self.assertTrue(root.findall(".//svg"), f"figure {letter} has no SVG panel")
                self.assertIn(title, markup)

    def test_no_figure_emits_a_nonfinite_token(self):
        """Check no figure writes nan or inf into a coordinate, tick or label."""
        for letter, _, _ in FIGURE_BUILDERS:
            with self.subTest(figure=letter):
                self.assertIsNone(_NONFINITE.search(self.figures[letter]))

    def test_every_figure_caption_states_a_conclusion(self):
        """Check every caption exists and is a sentence, not a bare label."""
        for letter, _, _ in FIGURE_BUILDERS:
            with self.subTest(figure=letter):
                caption = _assert_xml(self, self.figures[letter]).find("figcaption")
                text = "".join(caption.itertext())
                self.assertGreater(len(text), 120)
                self.assertIn("Conclude", text)

    def test_vertical_force_caption_states_the_shared_convention(self):
        """Check figure B names the shared contact fraction rather than a private threshold."""
        caption = "".join(_assert_xml(self, self.figures["B"]).find("figcaption").itertext())
        self.assertIn("CONTACT_FORCE_FRACTION", caption)
        self.assertIn(str(CONTACT_FORCE_FRACTION), caption)
        self.assertIn("LEGACY_REPORT.md section 3.1", caption)
        self.assertNotIn("erratum", caption)

    def test_complete_sources_draw_no_placeholder(self):
        """Check a complete artifact set draws every figure for real."""
        for letter, _, _ in FIGURE_BUILDERS:
            with self.subTest(figure=letter):
                self.assertNotIn("Not drawn:", self.figures[letter])

    def test_missing_files_render_placeholders(self):
        """Check absent artifacts produce labelled placeholders instead of raising."""
        with tempfile.TemporaryDirectory() as empty:
            sources = load_sources(empty)
            figures = build_figures(sources)
        self.assertEqual(len(figures), len(FIGURE_BUILDERS))
        for letter in ("B", "C", "D", "E", "F", "H"):
            with self.subTest(figure=letter):
                self.assertIn("Not drawn:", figures[letter])
                root = _assert_xml(self, figures[letter])
                self.assertEqual(root.tag, "figure")
        for letter in ("A", "G", "I", "J"):
            with self.subTest(figure=letter):
                self.assertNotIn("Not drawn:", figures[letter])

    def test_contact_window_falls_back_to_the_force_threshold(self):
        """Check a CSV trace without a declared window is windowed from its force."""
        archive = self.sources.run("ankle_v2")
        legacy = self.sources.run("legacy")
        self.assertEqual(archive.contact, (0.05, 0.35))
        self.assertIsNotNone(legacy.contact)
        self.assertLess(legacy.contact[0], legacy.contact[1])

    def test_percent_axis_spans_the_contact_window(self):
        """Check the percent abscissa reaches zero at touchdown and 100 at toe-off."""
        series = percent_series(self.sources.run("ankle_v2"), "fz_n")
        self.assertIsNotNone(series)
        self.assertLessEqual(float(np.min(series.x)), 0.0)
        self.assertGreaterEqual(float(np.max(series.x)), 100.0)

    def test_cumulative_impulse_ends_at_the_integral(self):
        """Check the impulse curve is monotone and ends at the trapezoidal integral."""
        run = self.sources.run("ankle_v2")
        series = cumulative_impulse(run)
        self.assertIsNotNone(series)
        time_s = run.trace["time_s"]
        inside = (time_s >= run.contact[0]) & (time_s <= run.contact[1])
        expected = np.trapezoid(run.trace["shoe_fz_n"][inside], time_s[inside])
        self.assertAlmostEqual(float(series.y[-1]), float(expected), places=6)
        self.assertTrue(np.all(np.diff(series.y) >= -1.0e-9))


def _legacy_artifact_directory(base: Path) -> Path:
    """Resolve optional historical outputs and reject unrelated report provenance.

    A relocated archive must retain its original report. An in-place archive must
    retain the published table in its saved summary. Actual trace values are checked
    by the tests, never used to decide whether an archive should be skipped.
    """
    pointer = base / "LEGACY_ARCHIVE.json"
    if pointer.is_file():
        manifest = json.loads(pointer.read_text(encoding="utf-8"))
        directory = Path(manifest["archive"]) / "outputs"
        report = Path(manifest["source_snapshot"]) / "projects/impedance_instron/REPORT.md"
        if not directory.is_dir() or not report.is_file():
            raise unittest.SkipTest("the relocated historical archive is not available")
        headings = re.findall(r"^#{1,6}\s+(.*?)\s*$", report.read_text(encoding="utf-8"), re.M)
        if FIGURE_PLACEMENT[0][0] not in headings:
            raise ValueError("archive report is not the historical momentum/work report")
        return directory
    saved = base / "report/summary.json"
    if not saved.is_file():
        raise unittest.SkipTest("historical report provenance is absent; current outputs are not legacy artifacts")
    payload = json.loads(saved.read_text(encoding="utf-8"))
    published = payload.get("peak_timing_pct", {}).get("report_section_3_1", {})
    for key, own, measured in REPORT_PEAK_TIMING_PCT:
        entry = published.get(key, {})
        if entry.get("published_run_peak_pct") != own or entry.get("published_measured_peak_pct") != measured:
            raise ValueError("archive summary does not identify the historical peak-timing table")
    return base


class RealArtifactTest(unittest.TestCase):
    """Check optional historical artifacts, not the current simple-run output tree."""

    @classmethod
    def setUpClass(cls):
        """Resolve a provenance-qualified historical archive when it is available."""
        cls.directory = _legacy_artifact_directory(Path("outputs/impedance_instron"))

    def require_files(self, *names: str) -> None:
        """Skip only when the historical files needed by a test are absent."""
        missing = [name for name in names if not (self.directory / name).is_file()]
        if missing:
            self.skipTest(f"historical artifacts are absent: {', '.join(missing)}")

    def test_prescribed_logs_carry_no_evaluation_records(self):
        """Check the historical prescribed-pitch logs contain no evaluation records."""
        self.require_files("train_v3.log", "train_v5.log")
        sources = load_sources(self.directory)
        for name in ("train_v3.log", "train_v5.log"):
            with self.subTest(log=name):
                self.assertEqual(sources.eval_counts[name], 0)

    def test_published_peak_timings_reproduce(self):
        """Reproduce the historical peak timings without relaxing their tolerance."""
        self.require_files(DEFAULT_PROFILE_NAME, "legacy_compare/trace.csv", "eval_j/trace.csv")
        if not any(self.directory.glob("policy_ankle_v2*.eval.npz")):
            self.skipTest("historical policy evaluation archive is absent")
        self.assertTrue(discover_archives(self.directory), "historical policy evaluation is unreadable")
        comparison = peak_timing_comparison(load_sources(self.directory))
        self.assertEqual(set(comparison), {key for key, _, _ in REPORT_PEAK_TIMING_PCT})
        for key, entry in comparison.items():
            with self.subTest(run=key):
                self.assertTrue(
                    entry["agrees"],
                    f"{key}: {entry['run_peak_pct']} vs published {entry['published_run_peak_pct']}",
                )

    def test_compliant_ankle_log_ends_at_the_reported_excursion(self):
        """Check the historical final excursion still matches the published 0.315."""
        self.require_files("train_ankle_v2.log")
        sources = load_sources(self.directory)
        self.assertTrue(sources.evaluations)
        self.assertAlmostEqual(sources.evaluations[-1]["excursion_momentum"], 0.315, places=3)


class ArchiveProvenanceTest(unittest.TestCase):
    """Separate optional historical runs from current or mismatched output trees."""

    def test_current_outputs_are_not_historical_provenance(self):
        """Reject an existing current output directory as evidence of an old run."""
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "simple").mkdir()
            _write_sources(base)
            with self.assertRaisesRegex(unittest.SkipTest, "historical report provenance is absent"):
                _legacy_artifact_directory(base)

    def test_mismatched_saved_report_is_rejected(self):
        """Fail rather than skip an archive whose saved table has different provenance."""
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "report").mkdir()
            (base / "report/summary.json").write_text('{"report": "two-stiffness"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "historical peak-timing table"):
                _legacy_artifact_directory(base)

    def test_archive_pointer_resolves_original_output_directory(self):
        """Use relocated outputs only with the original historical report source."""
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            archived = base / "archive"
            (archived / "outputs").mkdir(parents=True)
            source = archived / "source/projects/impedance_instron"
            source.mkdir(parents=True)
            report = source / "REPORT.md"
            report.write_text(_LEGACY_REPORT_FIXTURE, encoding="utf-8")
            (base / "LEGACY_ARCHIVE.json").write_text(
                json.dumps({"archive": str(archived), "source_snapshot": str(archived / "source")}), encoding="utf-8"
            )
            self.assertEqual(_legacy_artifact_directory(base), archived / "outputs")
            report.write_text("# Current two-stiffness report\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "historical momentum/work report"):
                _legacy_artifact_directory(base)

    def test_valid_provenance_does_not_hide_wrong_trace_values(self):
        """Keep numerical disagreement visible after a historical table is identified."""
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            _write_sources(base)
            (base / "report").mkdir()
            # The synthetic pulses deliberately differ from the published historical peaks.
            saved = summary_payload(load_sources(base), {})
            (base / "report/summary.json").write_text(json.dumps(saved), encoding="utf-8")
            resolved = _legacy_artifact_directory(base)
            comparison = peak_timing_comparison(load_sources(resolved))
            self.assertEqual(set(comparison), {key for key, _, _ in REPORT_PEAK_TIMING_PCT})
            self.assertTrue(all(not entry["agrees"] for entry in comparison.values()))


class ContactDefinitionTest(unittest.TestCase):
    """Cover the single definition of when stance began."""

    def test_constant_matches_the_environment(self):
        """Check the page's contact fraction equals env.CONTACT_FORCE_FRACTION, its source of truth."""
        source = Path("projects/impedance_instron/env.py").read_text(encoding="utf-8")
        found = re.search(r"^CONTACT_FORCE_FRACTION\s*=\s*([0-9.eE+-]+)", source, re.M)
        self.assertIsNotNone(found, "env.py no longer defines CONTACT_FORCE_FRACTION")
        self.assertEqual(float(found.group(1)), CONTACT_FORCE_FRACTION)

    def test_threshold_comes_from_the_profile_mass(self):
        """Check the threshold is the fraction of the profile's own body weight, not a literal."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / DEFAULT_PROFILE_NAME
            _write_profile(path, mass_kg=70.0)
            context = load_profile_context(path)
        self.assertAlmostEqual(context["body_weight_n"], 70.0 * 9.80665)
        self.assertAlmostEqual(context["contact_threshold_n"], CONTACT_FORCE_FRACTION * 70.0 * 9.80665)
        self.assertAlmostEqual(context["stance_window_s"][0], 0.04, places=6)

    def test_missing_profile_leaves_no_threshold(self):
        """Check a missing profile removes the threshold instead of inventing a window."""
        with tempfile.TemporaryDirectory() as directory:
            sources = load_sources(directory)
        self.assertIsNone(sources.threshold_n)
        self.assertIn("stance profile", sources.missing)

    def test_peak_percent_locates_a_maximum_in_its_window(self):
        """Check the peak locator reports the position of the maximum inside the window."""
        time_s = np.linspace(0.0, 1.0, 101)
        values = np.where(np.isclose(time_s, 0.25), 10.0, 1.0)
        peak = peak_percent(time_s, values, (0.0, 1.0))
        self.assertIsNotNone(peak)
        self.assertAlmostEqual(peak[0], 10.0)
        self.assertAlmostEqual(peak[1], 25.0, places=6)
        self.assertIsNone(peak_percent(time_s, values, (0.5, 0.5)))


class ResolutionTest(unittest.TestCase):
    """Cover how a series that is sampled far more coarsely than the others is drawn."""

    def _figure(self, samples: int, substep_dt: bool = False) -> tuple[str, object]:
        """Render figure B with the archive at ``samples`` and the CSV traces at 400 substeps."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            _write_sources(path, samples=max(samples, 400), archive_samples=samples)
            if substep_dt:
                _write_npz(path / "policy_ankle_v2.eval.npz", samples=samples, substep_dt=True)
            sources = load_sources(path)
            return build_figures(sources)["B"], sources

    def test_coarse_archive_is_dashed_with_its_samples_drawn(self):
        """Check a 46-sample archive is never drawn as a solid line beside 400-sample traces."""
        markup, sources = self._figure(samples=46)
        run = sources.run("ankle_v2")
        self.assertTrue(run.coarse)
        self.assertIn('stroke-dasharray="2 3"', markup)
        self.assertGreater(markup.count("<circle"), 20)
        self.assertIn("samples drawn", markup)
        caption = "".join(_assert_xml(self, markup).find("figcaption").itertext())
        self.assertIn("DASHED WITH ITS SAMPLES MARKED", caption)
        self.assertIn("sampling artefact", caption)
        self.assertIn("nothing was resampled or interpolated", caption)

    def test_full_resolution_archive_is_a_plain_line(self):
        """Check a full-resolution archive drops the dashes, the markers and the warning."""
        markup, sources = self._figure(samples=2881, substep_dt=True)
        run = sources.run("ankle_v2")
        self.assertFalse(run.coarse)
        self.assertNotIn("<circle", markup)
        self.assertNotIn('stroke-dasharray="2 3"', markup)
        caption = "".join(_assert_xml(self, markup).find("figcaption").itertext())
        self.assertIn("native resolution", caption)
        self.assertIn("contact transitions", caption)
        self.assertNotIn("sampling artefact", caption)

    def test_archive_reports_its_own_substep_interval(self):
        """Check the archive's ``substep_dt_s`` field is preferred over the time axis."""
        _, sources = self._figure(samples=2881, substep_dt=True)
        run = sources.run("ankle_v2")
        self.assertAlmostEqual(run.sample_interval_s, 0.375 / 2880.0, places=9)

    def test_coarse_series_keeps_every_sample(self):
        """Check the coarse series is drawn from its own samples, never interpolated onto a finer grid."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            _write_sources(path)
            sources = load_sources(path)
            series = percent_series(sources.run("ankle_v2"), "fz_n")
            samples = sources.run("ankle_v2").sample_count
        self.assertLessEqual(series.x.size, samples)
        self.assertTrue(series.markers)

    def test_step_statistics_describe_the_sampling(self):
        """Check the step statistics report spacing, jump size and contact transitions."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            _write_sources(path)
            sources = load_sources(path)
            coarse = sample_step_statistics(sources.run("ankle_v2"))
            fine = sample_step_statistics(sources.run("legacy"))
        self.assertEqual(coarse["contact_transitions"], 2)
        self.assertEqual(fine["contact_transitions"], 2)
        # The same waveform sampled 63 times more finely must show a much smaller jump per sample.
        self.assertGreater(coarse["max_step_n"], 5.0 * fine["max_step_n"])
        self.assertGreater(coarse["sample_interval_ms"], fine["sample_interval_ms"])


class MaterialTokenTest(unittest.TestCase):
    """Cover the optional material token that identifies which shoe produced a record."""

    REAL_LINE = (
        "eval iteration=400 objective_j=77.400 feasible=1 on_task=0 excursion_duration=0.000 "
        "excursion_impulse=0.000 excursion_momentum=1.190 violation_total=0.000 eval_return=-27.407 "
        "peak_fz_n=1050.000 peak_fz_ref_n=1000.000 peak_time_pct=50.000 peak_time_ref_pct=50.000 "
        "fz_rms_n=35.482 impulse_err_pct=5.000 com_vz_rms=0.100 com_z_rms_mm=9.854 contact_ms=138.000 "
        "peak_compression_mm=12.000 trace=/path/policy_ankle_v2_digital_shoe-c2e5666d.eval.npz "
        "artifact=digital_shoe-c2e5666d"
    )

    def test_record_with_a_token_parses_every_field(self):
        """Check the new trailing key parses without disturbing any earlier field."""
        record = parse_evaluation_record(self.REAL_LINE)
        self.assertIsNotNone(record)
        self.assertEqual(record["artifact"], "digital_shoe-c2e5666d")
        self.assertEqual(record["iteration"], 400.0)
        self.assertAlmostEqual(record["excursion_momentum"], 1.190)
        self.assertAlmostEqual(record["peak_compression_mm"], 12.0)
        self.assertEqual(record["trace"], "/path/policy_ankle_v2_digital_shoe-c2e5666d.eval.npz")

    def test_record_without_a_token_is_unchanged(self):
        """Check a log written before the key still parses and names no material."""
        record = parse_evaluation_record(_eval_line(50, 0.3, 350.0))
        self.assertIsNotNone(record)
        self.assertNotIn("artifact", record)

    def test_malformed_token_is_ignored(self):
        """Check an empty or repeated token is dropped instead of being trusted."""
        self.assertIsNone(optional_token(f"{self.REAL_LINE} artifact=second"))
        self.assertIsNone(optional_token(self.REAL_LINE.replace("artifact=digital_shoe-c2e5666d", "artifact=")))
        self.assertIsNone(optional_token("eval iteration=1"))

    def test_records_group_by_material_without_merging(self):
        """Check a sweep keeps one group per token, and untagged records stay separate."""
        records = [
            parse_evaluation_record(_eval_line(50, 0.3, 350.0, artifact="shoe-a")),
            parse_evaluation_record(_eval_line(100, 0.4, 340.0, artifact="shoe-b")),
            parse_evaluation_record(_eval_line(150, 0.5, 330.0, artifact="shoe-a")),
            parse_evaluation_record(_eval_line(200, 0.6, 320.0)),
        ]
        grouped = group_by_material(records)
        self.assertEqual(list(grouped), ["shoe-a", "shoe-b", None])
        self.assertEqual([len(group) for group in grouped.values()], [2, 1, 1])

    def test_training_figure_draws_one_curve_per_material(self):
        """Check a two-material log is drawn as two labelled curves, not one averaged curve."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            _write_sources(path)
            lines = [_eval_line(50 * step, 1.6 - 0.02 * step, 440.0 - step, artifact="shoe-a") for step in range(1, 6)]
            lines += [_eval_line(50 * step, 1.5 - 0.02 * step, 430.0 - step, artifact="shoe-b") for step in range(1, 6)]
            (path / "train_ankle_v2.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
            sources = load_sources(path)
            markup = build_figures(sources)["F"]
        self.assertEqual(set(sources.materials), {"shoe-a", "shoe-b"})
        _assert_xml(self, markup)
        self.assertIn("material shoe-a", markup)
        self.assertIn("material shoe-b", markup)
        caption = "".join(_assert_xml(self, markup).find("figcaption").itertext())
        self.assertIn("2 materials", caption)
        self.assertIn("averaging them would destroy", caption)

    def test_archive_token_is_read_and_shown(self):
        """Check the token stored in the archive names the run and reaches the payload."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            _write_sources(path)
            _write_npz(path / "policy_ankle_v2.eval.npz", artifact="digital_shoe-c2e5666d")
            sources = load_sources(path)
            payload = summary_payload(sources, build_figures(sources))
        self.assertEqual(sources.run("ankle_v2").material, "digital_shoe-c2e5666d")
        self.assertIn("digital_shoe-c2e5666d", sources.run("ankle_v2").label)
        self.assertEqual(payload["material_tokens"]["runs"]["ankle_v2"], "digital_shoe-c2e5666d")

    def test_sidecar_token_is_read_when_the_archive_has_none(self):
        """Check the JSON sidecar supplies the token for an archive written without one."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            _write_sources(path)
            sidecar = path / "policy_ankle_v2.eval.json"
            sidecar.write_text(json.dumps({"artifact": "shoe-from-sidecar"}), encoding="utf-8")
            sources = load_sources(path)
        self.assertEqual(sources.run("ankle_v2").material, "shoe-from-sidecar")

    def test_numeric_channels_survive_a_text_entry(self):
        """Check a text entry in the archive does not cost the page its numeric channels."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            _write_sources(path)
            _write_npz(path / "policy_ankle_v2.eval.npz", artifact="shoe-a")
            trace = load_npz_trace(path / "policy_ankle_v2.eval.npz")
            text = load_npz_text(path / "policy_ankle_v2.eval.npz")
        self.assertIn("shoe_fz_n", trace)
        self.assertNotIn("artifact", trace)
        self.assertEqual(text["artifact"], "shoe-a")


class ArchiveDiscoveryTest(unittest.TestCase):
    """Cover finding the policy archive instead of assuming one file name."""

    def test_tokenised_archive_supersedes_the_bare_one(self):
        """Check a tokenised archive is drawn and the older bare-named file is not."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            _write_sources(path)
            _write_npz(path / "policy_ankle_v2_shoe-a.eval.npz", samples=2881, substep_dt=True, artifact="shoe-a")
            sources = load_sources(path)
            discovered = discover_archives(path)
        self.assertEqual(sources.ankle_keys, ("ankle_v2",))
        self.assertTrue(sources.run("ankle_v2").source.endswith("policy_ankle_v2_shoe-a.eval.npz"))
        self.assertEqual(sources.run("ankle_v2").material, "shoe-a")
        self.assertEqual(sources.run("ankle_v2").sample_count, 2881)
        self.assertEqual([entry["tokenised"] for entry in discovered], [True, False])
        self.assertEqual(len(sources.superseded), 1)

    def test_bare_archive_is_used_when_it_is_the_only_one(self):
        """Check a directory written before tokenised names still renders its policy curve."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            _write_sources(path)
            sources = load_sources(path)
        self.assertEqual(sources.ankle_keys, ("ankle_v2",))
        self.assertTrue(sources.run("ankle_v2").source.endswith("policy_ankle_v2.eval.npz"))
        self.assertFalse(sources.superseded)

    def test_newest_archive_wins_within_one_material(self):
        """Check a rerun of one material supersedes its own earlier archive, by modification time."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            _write_sources(path)
            old_archive = path / "policy_ankle_v2_shoe-a.eval.npz"
            _write_npz(old_archive, samples=46, artifact="shoe-a")
            os.utime(old_archive, (1.0, 1.0))
            new_archive = path / "policy_ankle_v2_shoe-a-rerun.eval.npz"
            _write_npz(new_archive, samples=2881, substep_dt=True, artifact="shoe-a")
            sources = load_sources(path)
        self.assertEqual(sources.ankle_keys, ("ankle_v2",))
        self.assertTrue(sources.run("ankle_v2").source.endswith("policy_ankle_v2_shoe-a-rerun.eval.npz"))

    def test_several_materials_become_several_runs(self):
        """Check a material sweep draws one policy curve per material rather than choosing one."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            _write_sources(path)
            for token in ("shoe-a", "shoe-b"):
                _write_npz(path / f"policy_ankle_v2_{token}.eval.npz", samples=400, artifact=token)
            sources = load_sources(path)
            markup = build_figures(sources)["B"]
        self.assertEqual(len(sources.ankle_keys), 2)
        self.assertEqual({sources.run(key).material for key in sources.ankle_keys}, {"shoe-a", "shoe-b"})
        _assert_xml(self, markup)
        self.assertIn("material shoe-a", markup)
        self.assertIn("material shoe-b", markup)

    def test_payload_names_the_file_each_curve_used(self):
        """Check summary.json states the discovery rule and the file behind every policy curve."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            _write_sources(path)
            _write_npz(path / "policy_ankle_v2_shoe-a.eval.npz", samples=2881, substep_dt=True, artifact="shoe-a")
            sources = load_sources(path)
            payload = summary_payload(sources, build_figures(sources))
            page = render_page(_LEGACY_REPORT_FIXTURE, build_figures(sources), sources)
        archives = payload["archives"]
        self.assertIn("policy_ankle_v2*.eval.npz", archives["rule"])
        self.assertTrue(archives["used_by_run"]["ankle_v2"].endswith("policy_ankle_v2_shoe-a.eval.npz"))
        used = {entry["path"]: entry["used"] for entry in archives["discovered"]}
        self.assertEqual(sum(used.values()), 1)
        self.assertIn("policy_ankle_v2_shoe-a.eval.npz", page)
        self.assertIn("superseded by a tokenised archive", page)

    def test_missing_archive_leaves_placeholders(self):
        """Check a directory with no policy archive renders placeholders, not an error."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            _write_sources(path)
            (path / "policy_ankle_v2.eval.npz").unlink()
            sources = load_sources(path)
            figures = build_figures(sources)
        self.assertEqual(sources.ankle_keys, ())
        self.assertIn("policy_ankle_v2", sources.missing)
        self.assertIn("Not drawn:", figures["H"])
        _assert_xml(self, figures["B"])


class AxisTest(unittest.TestCase):
    """Cover the axis mapping that a log scale needs."""

    def test_log_axis_drops_zero_and_negative_samples(self):
        """Check a log axis maps non-positive values to nan before projection."""
        mapped = axis_values(np.array([-3.0, 0.0, 1.0, 100.0]), log=True)
        self.assertTrue(np.isnan(mapped[0]) and np.isnan(mapped[1]))
        self.assertAlmostEqual(float(mapped[2]), 0.0)
        self.assertAlmostEqual(float(mapped[3]), 2.0)

    def test_log_plot_writes_no_nan_coordinate(self):
        """Check a log-log panel with zero and negative input still renders finite paths."""
        x = np.array([0.0, -1.0, 1.0, 10.0, 100.0])
        y = np.array([-5.0, 0.0, 1.0, 0.1, 0.01])
        markup = plot(
            "zeros and negatives",
            [Series("mixed", x, y, "#2563eb")],
            xlabel="x",
            ylabel="y",
            xlog=True,
            ylog=True,
        )
        _assert_xml(self, markup)
        self.assertIsNone(_NONFINITE.search(markup))
        drawn = re.findall(r"[ML](-?\d+\.\d+),(-?\d+\.\d+)", markup)
        self.assertTrue(drawn)
        for horizontal, vertical in drawn:
            self.assertTrue(np.isfinite(float(horizontal)) and np.isfinite(float(vertical)))

    def test_log_plot_without_positive_samples_explains_itself(self):
        """Check a log panel with nothing positive reports the reason instead of failing."""
        markup = plot(
            "all non-positive",
            [Series("mixed", np.array([1.0, 2.0]), np.array([0.0, -1.0]), "#2563eb")],
            xlabel="x",
            ylabel="y",
            ylog=True,
        )
        _assert_xml(self, markup)
        self.assertIn("logarithmic", markup)

    def test_plot_without_series_explains_itself(self):
        """Check an empty panel renders an explanation rather than raising."""
        markup = plot("nothing", [], xlabel="x", ylabel="y")
        _assert_xml(self, markup)
        self.assertIn("No drawable samples", markup)


class MarkdownTest(unittest.TestCase):
    """Cover the Markdown subset REPORT.md actually uses."""

    def test_heading_levels_round_trip(self):
        """Check every ATX heading level renders at its own depth with an anchor."""
        blocks = parse_blocks("# One\n\n## Two\n\n### Three\n")
        self.assertEqual([block["level"] for block in blocks], [1, 2, 3])
        rendered = "".join(render_block(block) for block in blocks)
        self.assertIn('<h1 id="one">One</h1>', rendered)
        self.assertIn('<h2 id="two">Two</h2>', rendered)
        self.assertIn('<h3 id="three">Three</h3>', rendered)

    def test_table_renders_with_alignment(self):
        """Check a pipe table keeps its header, rows and per-column alignment."""
        text = "| a | b | c |\n| :--- | :---: | ---: |\n| 1 | 2 | 3 |\n"
        block = parse_blocks(text)[0]
        self.assertEqual(block["align"], ["left", "center", "right"])
        rendered = render_block(block)
        _assert_xml(self, rendered)
        self.assertIn('<th style="text-align:center">b</th>', rendered)
        self.assertIn('<td style="text-align:right">3</td>', rendered)

    def test_bold_and_inline_code_render(self):
        """Check bold spans, inline code and HTML-unsafe characters are handled."""
        rendered = render_block(parse_blocks("**bold** and `k < b` and 5 > 4\n")[0])
        _assert_xml(self, rendered)
        self.assertIn("<b>bold</b>", rendered)
        self.assertIn("<code>k &lt; b</code>", rendered)
        self.assertIn("5 &gt; 4", rendered)

    def test_lists_and_rules_and_code_blocks_render(self):
        """Check bullet lists, numbered lists, rules and indented code all parse."""
        text = "* one\n* two\n\n1. first\n2. second\n\n---\n\n    f = k * x\n"
        kinds = [block["type"] for block in parse_blocks(text)]
        self.assertEqual(kinds, ["list", "list", "rule", "code"])
        rendered = "".join(render_block(block) for block in parse_blocks(text))
        self.assertIn("<ul><li>one</li><li>two</li></ul>", rendered)
        self.assertIn("<ol><li>first</li><li>second</li></ol>", rendered)
        self.assertIn("<pre><code>f = k * x</code></pre>", rendered)

    def test_list_continuation_lines_join_their_item(self):
        """Check an indented continuation line stays part of the preceding list item."""
        block = parse_blocks("1. **Shear ordering inverts.** Their pairs\n   imply nu = +0.28.\n")[0]
        self.assertEqual(len(block["items"]), 1)
        self.assertIn("imply nu = +0.28.", block["items"][0])


class PageTest(unittest.TestCase):
    """Cover the assembled page and the JSON payload."""

    @classmethod
    def setUpClass(cls):
        """Render a controlled historical document against synthetic artifacts."""
        cls._temporary = tempfile.TemporaryDirectory()
        cls.directory = Path(cls._temporary.name)
        _write_sources(cls.directory)
        cls.report_text = _LEGACY_REPORT_FIXTURE
        cls.report_path = cls.directory / "historical_report.md"
        cls.report_path.write_text(cls.report_text, encoding="utf-8")
        cls.sources = load_sources(cls.directory)
        cls.figures = build_figures(cls.sources)
        cls.page = render_page(cls.report_text, cls.figures, cls.sources)

    @classmethod
    def tearDownClass(cls):
        """Remove the synthetic artifacts."""
        cls._temporary.cleanup()

    def test_page_contains_every_report_heading(self):
        """Check every fixture heading appears in the rendered page."""
        headings = re.findall(r"^#{1,6}\s+(.*?)\s*$", self.report_text, re.M)
        self.assertTrue(headings)
        for heading in headings:
            with self.subTest(heading=heading):
                self.assertIn(f">{heading}<", self.page)

    def test_page_places_every_figure_in_a_section(self):
        """Check every figure is anchored to a heading in the historical fixture."""
        headings = set(re.findall(r"^#{1,6}\s+(.*?)\s*$", self.report_text, re.M))
        for heading, letters in FIGURE_PLACEMENT:
            with self.subTest(heading=heading):
                self.assertIn(heading, headings, f"figures {letters} have no matching section")
        self.assertNotIn("Figures without a matching section", self.page)

    def test_figures_follow_the_report_section_order(self):
        """Check the figures follow historical section order."""
        order = [letter for heading, letters in FIGURE_PLACEMENT for letter in letters]
        positions = [self.page.index(f'id="figure-{letter.lower()}"') for letter in order]
        self.assertEqual(positions, sorted(positions))

    def test_page_is_self_contained_and_nonfinite_free(self):
        """Check the page needs no network and carries no nan or inf token."""
        self.assertTrue(self.page.startswith("<!DOCTYPE html>"))
        self.assertGreater(len(self.page), 50000)

        class Resources(HTMLParser):
            def __init__(self):
                super().__init__()
                self.tags = []

            def handle_starttag(self, tag, attrs):
                self.tags.append((tag, dict(attrs)))

        parsed = Resources()
        parsed.feed(self.page)
        self.assertIn("http://localhost:8050", self.page)
        for tag, attrs in parsed.tags:
            self.assertNotIn(tag, ("script", "iframe", "object", "embed", "link"))
            if tag in ("img", "image", "use", "video", "audio", "source"):
                for name in ("src", "href", "xlink:href"):
                    self.assertFalse(re.match(r"(?:https?:)?//", attrs.get(name, "")))
        self.assertNotRegex(self.page, r"(?i)@import|url\(\s*['\"]?(?:https?:)?//")
        self.assertIsNone(_NONFINITE.search(self.page))

    def test_payload_carries_the_plotted_numbers(self):
        """Check summary.json holds the hardcoded tables and the measured run statistics."""
        payload = summary_payload(self.sources, self.figures)
        serialized = json.loads(json.dumps(payload, allow_nan=False))
        for label, value in MOMENTUM_EXCURSION:
            self.assertAlmostEqual(serialized["momentum_excursion"][label], value)
        self.assertEqual(serialized["stiff_limit"]["k_theta_n_m_per_rad"], list(STIFF_LIMIT_K_THETA))
        for label, values in STIFF_LIMIT_SERIES:
            self.assertEqual(serialized["stiff_limit"][label], list(values))
        changes = {entry["change"]: entry for entry in serialized["infrastructure"]}
        for label, unit, before, after in INFRASTRUCTURE:
            self.assertEqual(changes[label]["unit"], unit)
            self.assertAlmostEqual(changes[label]["speedup"], before / after)
        self.assertAlmostEqual(
            serialized["contact_definition"]["threshold_n"],
            CONTACT_FORCE_FRACTION * 81.93121179999996 * 9.80665,
        )
        timing = serialized["peak_timing_pct"]
        self.assertEqual(set(timing["detected_contact"]), {"measured", "legacy", "command_j", "ankle_v2"})
        self.assertEqual(set(timing["report_section_3_1"]), {key for key, _, _ in REPORT_PEAK_TIMING_PCT})
        for key, published_run, published_measured in REPORT_PEAK_TIMING_PCT:
            with self.subTest(run=key):
                entry = timing["report_section_3_1"][key]
                self.assertAlmostEqual(entry["published_run_peak_pct"], published_run)
                self.assertAlmostEqual(entry["published_measured_peak_pct"], published_measured)
        for key in ("legacy", "command_j", "ankle_v2"):
            with self.subTest(run=key):
                self.assertIsNotNone(timing["detected_contact"][key])
                self.assertIsNotNone(timing["fixed_stance_window"][key])
        self.assertAlmostEqual(serialized["runs"]["ankle_v2"]["contact_ms"], 300.0)
        self.assertEqual(serialized["runs"]["ankle_v2"]["samples"], 46)
        self.assertEqual(len(serialized["training"]["ankle_v2"]["iteration"]), 10)
        self.assertEqual(len(serialized["materials"]["strain_pct"]), 240)
        self.assertFalse(serialized["missing_sources"])

    def test_payload_series_match_the_drawn_series(self):
        """Check the JSON waveforms repeat the values the figures plot."""
        payload = summary_payload(self.sources, self.figures)
        drawn = percent_series(self.sources.run("ankle_v2"), "fz_n")
        stored = payload["waveforms"]["ankle_v2"]["fz_n"]
        self.assertEqual(len(stored["y"]), drawn.y.size)
        self.assertAlmostEqual(stored["y"][10], float(drawn.y[10]), places=5)

    def test_write_summary_produces_both_artifacts(self):
        """Check the entry point writes a non-trivial page and a loadable payload."""
        with tempfile.TemporaryDirectory() as output:
            written = write_summary(output, self.directory, self.report_path)
            html = written["html"].read_text(encoding="utf-8")
            payload = json.loads(written["json"].read_text(encoding="utf-8"))
        self.assertGreater(len(html), 50000)
        self.assertEqual(html.count("<figure "), len(FIGURE_BUILDERS))
        self.assertEqual(set(payload["figures"]), {letter for letter, _, _ in FIGURE_BUILDERS})

    def test_default_report_is_explicitly_historical(self):
        """Keep the legacy renderer default separate from the current report."""
        self.assertEqual(Path(DEFAULT_REPORT).name, "LEGACY_REPORT.md")
        with tempfile.TemporaryDirectory() as output:
            written = write_summary(output, self.directory)
            page = written["html"].read_text(encoding="utf-8")
            payload = json.loads(written["json"].read_text(encoding="utf-8"))
        self.assertIn("Historical report template", page)
        self.assertIn("Historical momentum/work experiments", page)
        self.assertEqual(page.count("<figure "), len(FIGURE_BUILDERS))
        self.assertNotIn("Figures without a matching section", page)
        self.assertEqual(payload["report"]["kind"], "legacy-momentum-work")

    def test_current_report_does_not_receive_legacy_figures(self):
        """Render the current document without attaching old scientific figures or tables."""
        current = Path("projects/impedance_instron/REPORT.md")
        report = current.read_text(encoding="utf-8")
        page = render_page(report, self.figures, self.sources)
        self.assertIn("Legacy figures omitted", page)
        self.assertNotIn("<figure ", page)
        self.assertNotIn("Figures without a matching section", page)
        self.assertNotIn("Figures and their data", page)
        with tempfile.TemporaryDirectory() as output:
            written = write_summary(output, self.directory, current)
            self.assertEqual(written["html"].read_text(encoding="utf-8"), page)
            payload = json.loads(written["json"].read_text(encoding="utf-8"))
        self.assertEqual(payload["figures"], {})
        self.assertEqual(payload["report"]["kind"], "unillustrated")
        self.assertNotIn("momentum_excursion", payload)
        self.assertNotIn("stiff_limit", payload)

    def test_unrelated_heading_does_not_receive_legacy_figures(self):
        """Avoid placing legacy figures into an unrelated document with a reused subheading."""
        for heading, _ in FIGURE_PLACEMENT:
            with self.subTest(heading=heading):
                report = f"# Another experiment\n\n## {heading}\n\nUnrelated measurements.\n"
                page = render_page(report, self.figures, self.sources)
                self.assertIn("Unrelated measurements.", page)
                self.assertNotIn("<figure ", page)

    def test_missing_report_still_writes_a_page(self):
        """Check a missing REPORT.md degrades to a stub page instead of raising."""
        with tempfile.TemporaryDirectory() as output:
            written = write_summary(output, self.directory, Path(output) / "absent.md")
            html = written["html"].read_text(encoding="utf-8")
        self.assertIn("is missing", html)
        self.assertIn("Figures without a matching section", html)


if __name__ == "__main__":
    unittest.main()
