# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the offline impedance training dashboard: parsing, overlays and rendering."""

import re
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import numpy as np

from projects.impedance_instron.dashboard import (
    MAX_POINTS,
    create_server,
    downsample,
    eval_groups,
    load_runs,
    parse_evaluation,
    parse_frozen,
    parse_iteration,
    parse_log,
    parse_selection,
    plot,
    refresh_url,
    render,
    render_directory,
    run_colors,
)

ITERATION_LINE = (
    "iter  400 | return mean=  -103.502 med=   -99.251 best=   -91.009 | "
    "pi=-0.0044 v=312.2670 ent=+2.902 | ev=-0.123 kl=0.00415 clip=0.050"
)
FROZEN_LINE = "frozen: episodes=64 mean=-21.7848 median=-21.7848 best=-21.7848 worst=-21.7849"
EVAL_LINE = (
    "eval iteration=400 objective_j=77.400 feasible=1 on_task=0 excursion_duration=0.000 "
    "excursion_impulse=0.000 excursion_momentum=1.190 violation_total=0.000 eval_return=-84.085 "
    "peak_fz_n=1791.923 peak_fz_ref_n=1762.554 peak_time_pct=47.352 peak_time_ref_pct=46.954 "
    "fz_rms_n=292.760 impulse_err_pct=3.189 com_vz_rms=0.249 com_z_rms_mm=41.516 contact_ms=308.073 "
    "peak_compression_mm=20.744 trace=outputs/impedance_instron/policy_v5.eval.npz "
    "artifact=digital_shoe-c2e5666d"
)
SHORT_EVAL_LINE = EVAL_LINE[: EVAL_LINE.index(" peak_fz_n=")]
NOISE = (
    "Warp 1.17.0.dev20260807 initialized:",
    "   CUDA Toolkit 12.9, Driver 13.2",
    "Module newton._src.geometry.bvh 8b81f2f load on device 'cuda:0' took 2.70 ms  (cached)",
    "Traceback (most recent call last):",
    '  File "train.py", line 912, in main',
    "RuntimeError: CUDA out of memory",
    "onnx: outputs/impedance_instron/policy_v1.onnx",
    "",
)
_NONFINITE = re.compile(r"\b(nan|-?inf|infinity)\b", re.IGNORECASE)


def _eval_line(iteration, *, on_task=1, objective="12.500", momentum="0.000", trace=None, artifact=None):
    """Build one deterministic-evaluation record in the pinned log format."""
    line = (
        f"eval iteration={iteration} objective_j={objective} feasible=1 on_task={on_task} "
        f"excursion_duration=0.000 excursion_impulse=0.000 excursion_momentum={momentum} "
        f"violation_total=0.000 eval_return=-84.085 peak_fz_n=1791.923 peak_fz_ref_n=1762.554 "
        f"peak_time_pct=47.352 peak_time_ref_pct=46.954 fz_rms_n=292.760 impulse_err_pct=3.189 "
        f"com_vz_rms=0.249 com_z_rms_mm=41.516 contact_ms=308.073 peak_compression_mm=20.744"
    )
    if trace is not None:
        line = f"{line} trace={trace}"
    if artifact is not None:
        line = f"{line} artifact={artifact}"
    return line


def _write_trace(path, *, iteration=30, samples=400, artifact=None, reference_scale=1.0):
    """Write a synthetic evaluation trace archive using the documented schema."""
    time_s = np.linspace(0.0, 0.4, samples)
    shape = np.sin(np.pi * np.clip((time_s - 0.05) / 0.3, 0.0, 1.0)) ** 2
    extra = {} if artifact is None else {"artifact": np.asarray(artifact)}
    np.savez_compressed(
        path,
        **extra,
        time_s=time_s,
        shoe_fz_n=1800.0 * shape,
        reference_fz_n=1760.0 * reference_scale * shape,
        shoe_fx_n=-200.0 * np.gradient(shape),
        reference_fx_n=-190.0 * np.gradient(shape),
        com_z_m=1.0 - 0.05 * shape,
        reference_com_z_m=1.0 - 0.045 * shape,
        com_vz_m_s=np.gradient(1.0 - 0.05 * shape, time_s),
        reference_com_vz_m_s=np.gradient(1.0 - 0.045 * shape, time_s),
        leg_length_m=0.95 - 0.04 * shape,
        commanded_length_m=0.95 - 0.03 * shape,
        stiffness_n_m=24000.0 + 3000.0 * shape,
        damping_ratio=0.3 + 0.1 * shape,
        contact_start_s=np.asarray([0.05]),
        contact_end_s=np.asarray([0.35]),
        iteration=np.asarray([float(iteration)]),
    )
    return path


def _log_text(
    iterations, *, frozen=None, noise=True, evaluate=0, on_task=1, objective="12.500", trace=None, artifacts=None
):
    """Build a synthetic training log with optional noise, eval records and a frozen line."""
    lines = list(NOISE) if noise else []
    for index in iterations:
        lines.append(
            f"iter {index:4d} | return mean= {-200.0 + index:9.3f} med= {-205.0 + index:9.3f} "
            f"best= {-190.0 + index:9.3f} | pi=-0.0044 v={1000.0 / index:.4f} ent=+2.902 | "
            f"ev=-0.123 kl=0.00415 clip=0.050"
        )
        if evaluate and index % evaluate == 0:
            momentum = "0.000" if on_task else "1.190"
            artifact = artifacts[(index // evaluate - 1) % len(artifacts)] if artifacts else None
            lines.append(
                _eval_line(
                    index, on_task=on_task, objective=objective, momentum=momentum, trace=trace, artifact=artifact
                )
            )
    if frozen is not None:
        lines.append(frozen)
    return "\n".join(lines) + "\n"


def _svg_fragments(page):
    """Return every inline SVG element found in a rendered page."""
    return re.findall(r"<svg\b.*?</svg>", page, flags=re.DOTALL)


def _series_coordinates(section, color):
    """Return the drawn coordinates of the data path with the given stroke colour."""
    paths = re.findall(rf'<path d="([^"]+)" fill="none" stroke="{color}"', section)
    return [pair for path in paths for pair in re.findall(r"[ML](-?[\d.]+),(-?[\d.]+)", path)]


class TestParsing(unittest.TestCase):
    """Verify defensive parsing of real training log text."""

    def test_parse_iteration_fields(self):
        """Parse a real iteration line into the documented field values."""
        sample = parse_iteration(ITERATION_LINE)
        self.assertIsNotNone(sample)
        self.assertEqual(sample["iteration"], 400.0)
        self.assertAlmostEqual(sample["mean"], -103.502)
        self.assertAlmostEqual(sample["median"], -99.251)
        self.assertAlmostEqual(sample["best"], -91.009)
        self.assertAlmostEqual(sample["pi"], -0.0044)
        self.assertAlmostEqual(sample["v"], 312.2670)
        self.assertAlmostEqual(sample["ent"], 2.902)
        self.assertAlmostEqual(sample["ev"], -0.123)
        self.assertAlmostEqual(sample["kl"], 0.00415)
        self.assertAlmostEqual(sample["clip"], 0.050)

    def test_parse_frozen_line(self):
        """Parse the frozen evaluation line of a finished run."""
        metrics = parse_frozen(FROZEN_LINE)
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics["episodes"], 64.0)
        self.assertAlmostEqual(metrics["mean"], -21.7848)
        self.assertAlmostEqual(metrics["worst"], -21.7849)
        self.assertIsNotNone(parse_frozen("frozen outputs/policy_v1.pt: " + FROZEN_LINE.split(": ", 1)[1]))

    def test_malformed_lines_are_skipped(self):
        """Skip truncated, malformed and nonfinite lines without raising."""
        broken = (
            ITERATION_LINE[:57],
            "iter  400 | return mean=  -103.502 med=",
            "iter | return mean= x med= y best= z | pi=a v=b ent=c | ev=d kl=e clip=f",
            "iter  400 | return mean=  nan med=   -99.251 best=   -91.009 | "
            "pi=-0.0044 v=inf ent=+2.902 | ev=-0.123 kl=0.00415 clip=0.050",
            "frozen: episodes=64 mean=",
        )
        for line in broken:
            with self.subTest(line=line):
                self.assertIsNone(parse_iteration(line))
                self.assertIsNone(parse_frozen(line))

    def test_log_noise_is_ignored(self):
        """Ignore Warp module-load noise and tracebacks when parsing a log."""
        text = _log_text(range(1, 11), frozen=FROZEN_LINE)
        text += ITERATION_LINE[:40]  # A half-written final line, as seen during live training.
        samples, evaluations, frozen, metrics = parse_log(text)
        self.assertEqual(evaluations, [])
        self.assertEqual(len(samples), 10)
        self.assertEqual([sample["iteration"] for sample in samples], [float(i) for i in range(1, 11)])
        self.assertEqual(frozen, FROZEN_LINE)
        self.assertAlmostEqual(metrics["mean"], -21.7848)

    def test_parse_evaluation_fields(self):
        """Parse a real deterministic-evaluation line into the documented field values."""
        record = parse_evaluation(EVAL_LINE)
        self.assertIsNotNone(record)
        self.assertEqual(record["iteration"], 400.0)
        self.assertAlmostEqual(record["objective_j"], 77.400)
        self.assertEqual(record["feasible"], 1.0)
        self.assertEqual(record["on_task"], 0.0)
        self.assertAlmostEqual(record["excursion_duration"], 0.0)
        self.assertAlmostEqual(record["excursion_impulse"], 0.0)
        self.assertAlmostEqual(record["excursion_momentum"], 1.190)
        self.assertAlmostEqual(record["violation_total"], 0.0)
        self.assertAlmostEqual(record["eval_return"], -84.085)
        self.assertAlmostEqual(record["peak_fz_n"], 1791.923)
        self.assertAlmostEqual(record["peak_fz_ref_n"], 1762.554)
        self.assertAlmostEqual(record["peak_time_pct"], 47.352)
        self.assertAlmostEqual(record["peak_time_ref_pct"], 46.954)
        self.assertAlmostEqual(record["fz_rms_n"], 292.760)
        self.assertAlmostEqual(record["impulse_err_pct"], 3.189)
        self.assertAlmostEqual(record["com_vz_rms"], 0.249)
        self.assertAlmostEqual(record["com_z_rms_mm"], 41.516)
        self.assertAlmostEqual(record["contact_ms"], 308.073)
        self.assertAlmostEqual(record["peak_compression_mm"], 20.744)
        self.assertEqual(record["trace"], "outputs/impedance_instron/policy_v5.eval.npz")
        nan_record = parse_evaluation(EVAL_LINE.replace("objective_j=77.400", "objective_j=nan"))
        self.assertIsNotNone(nan_record)
        self.assertTrue(np.isnan(nan_record["objective_j"]))

    def test_malformed_evaluation_lines_are_skipped(self):
        """Skip truncated, valueless and non-boolean evaluation lines without raising."""
        broken = (
            EVAL_LINE[:60],
            "eval iteration=400 objective_j=77.400 feasible=1 on_task=0",
            EVAL_LINE.replace("on_task=0", "on_task=2"),
            EVAL_LINE.replace("iteration=400", "iteration=four"),
            EVAL_LINE.replace("eval ", "evaluation "),
            EVAL_LINE + " on_task=1",
            "eval",
            "eval iteration 400 objective_j 77.400",
        )
        for line in broken:
            with self.subTest(line=line):
                self.assertIsNone(parse_evaluation(line))

    def test_truncated_physical_block_keeps_the_required_fields(self):
        """Keep the required block and drop a half-written physical tail."""
        record = parse_evaluation(EVAL_LINE[: EVAL_LINE.index("fz_rms_n=")].strip())
        self.assertIsNotNone(record)
        self.assertAlmostEqual(record["objective_j"], 77.400)
        self.assertNotIn("peak_fz_n", record)
        self.assertNotIn("trace", record)
        short = parse_evaluation(SHORT_EVAL_LINE)
        self.assertIsNotNone(short)
        self.assertAlmostEqual(short["excursion_momentum"], 1.190)
        self.assertNotIn("peak_fz_n", short)

    def test_evaluation_records_survive_noise_and_truncation(self):
        """Collect eval records from a noisy log whose last line is half written."""
        text = _log_text(range(1, 21), evaluate=5)
        text += "eval iteration=25 objective_j=12.5 feas"  # The live tail of the file.
        samples, evaluations, frozen, _ = parse_log(text)
        self.assertEqual(len(samples), 20)
        self.assertEqual([record["iteration"] for record in evaluations], [5.0, 10.0, 15.0, 20.0])
        self.assertIsNone(frozen)


class TestRuns(unittest.TestCase):
    """Verify run discovery, overlay rendering and the run legend."""

    def test_multiple_runs_are_overlaid_with_correct_statistics(self):
        """Overlay two runs and report their per-run legend statistics."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train_v1.log").write_text(_log_text(range(1, 21), frozen=FROZEN_LINE))
            (root / "train_v2.log").write_text(_log_text(range(1, 6)))
            (root / "notes.txt").write_text("ignored, this is not a training log")
            runs = load_runs(root)
            self.assertEqual([run.name for run in runs], ["v1", "v2"])
            self.assertEqual([run.iteration_count for run in runs], [20, 5])
            self.assertEqual(runs[0].status, "finished")
            self.assertEqual(runs[1].status, "running")
            self.assertAlmostEqual(runs[0].final("mean"), -180.0)
            self.assertAlmostEqual(runs[0].best_return(), -170.0)
            page = render(runs, root, refresh=10)
            self.assertIn("v1", page)
            self.assertIn("v2", page)
            self.assertIn("20 iters", page)
            self.assertIn("5 iters", page)
            self.assertIn("-180", page)
            self.assertIn("-170", page)
            self.assertIn(">finished<", page)
            self.assertIn(">running<", page)
            self.assertIn(FROZEN_LINE, page)
            # One colour per run, so the two overlaid mean curves differ in stroke.
            self.assertIn('stroke="#1967b3"', page)
            self.assertIn('stroke="#b55214"', page)

    def test_page_refreshes_and_covers_every_metric(self):
        """Emit a meta refresh tag and one chart per reported metric."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train_v1.log").write_text(_log_text(range(1, 31)))
            page = render_directory(root, refresh=7)
            self.assertIn('<meta http-equiv="refresh" content="7; url=/">', page)
            self.assertNotIn("<script", page)
            self.assertNotIn("http://cdn", page)
            for title in (
                "Task excursions",
                "Work objective",
                "Episode return",
                "Policy entropy",
                "Explained variance",
                "Value loss",
                "Approximate KL divergence",
                "Clip fraction",
            ):
                self.assertIn(title, page)
            # Without eval records only the training-health charts draw.
            self.assertEqual(len(_svg_fragments(page)), 7)
            self.assertEqual(page.count("No samples yet."), 10)
            self.assertIn("No waveforms for the current selection.", page)

    def test_empty_directory_renders_a_page(self):
        """Render a readable page when the directory holds no training logs."""
        with tempfile.TemporaryDirectory() as directory:
            page = render_directory(directory)
            self.assertIn("No <code>train_*.log</code> logs were found", page)
            self.assertIn("</html>", page)
            self.assertIsNone(_NONFINITE.search(page))


class TestCharts(unittest.TestCase):
    """Verify chart geometry, log scaling and downsampling."""

    def test_log_axis_guards_zero_and_negative_values(self):
        """Drop nonpositive samples from the log axis without emitting nan or inf."""
        x = np.arange(6, dtype=float)
        y = np.array([0.0, -5.0, 1.0e-3, 1.0e4, 0.0, 2.0])
        section = plot("Value loss", "log scale", [("v1", x, y, "#1967b3", False)], log=True)
        self.assertIsNone(_NONFINITE.search(section))
        coordinates = _series_coordinates(section, "#1967b3")
        self.assertTrue(coordinates)
        for px, py in coordinates:
            self.assertTrue(np.isfinite(float(px)))
            self.assertTrue(np.isfinite(float(py)))
        empty = plot("Value loss", "log scale", [("v1", x, np.zeros(6), "#1967b3", False)], log=True)
        self.assertIn("No positive samples", empty)
        self.assertIsNone(_NONFINITE.search(empty))

    def test_single_sample_and_constant_series_stay_finite(self):
        """Keep SVG coordinates finite for degenerate single-point and flat series."""
        for x, y in ((np.array([3.0]), np.array([1.0])), (np.arange(4.0), np.full(4, -2.0))):
            section = plot("Metric", "unit", [("v1", x, y, "#1967b3", False)])
            with self.subTest(size=len(x)):
                self.assertIsNone(_NONFINITE.search(section))

    def test_downsampling_keeps_endpoints(self):
        """Thin long series by a fixed stride while keeping the first and last sample."""
        x = np.arange(5000.0)
        y = np.sin(x)
        thin_x, thin_y = downsample(x, y)
        self.assertLessEqual(len(thin_x), MAX_POINTS + 1)
        self.assertEqual(len(thin_x), len(thin_y))
        self.assertEqual(thin_x[0], x[0])
        self.assertEqual(thin_x[-1], x[-1])
        short_x, short_y = downsample(x[:10], y[:10])
        self.assertEqual(len(short_x), 10)
        self.assertEqual(len(short_y), 10)

    def test_rendered_svg_is_well_formed_xml(self):
        """Parse every rendered inline SVG as well-formed XML."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train_v1.log").write_text(_log_text(range(1, 2001), frozen=FROZEN_LINE, evaluate=25))
            (root / "train_v2.log").write_text(_log_text(range(1, 51), evaluate=10, on_task=0))
            page = render_directory(root)
            fragments = _svg_fragments(page)
            self.assertEqual(len(fragments), 17)
            for fragment in fragments:
                element = ElementTree.fromstring(fragment)
                self.assertTrue(element.tag.endswith("svg"))
            self.assertIsNone(_NONFINITE.search(page))


class TestEvaluationCharts(unittest.TestCase):
    """Verify the task-excursion and work-objective charts and the on-task badge."""

    def _page(self, **kwargs):
        """Render a one-run page from a synthetic log built with the given options."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train_v9.log").write_text(_log_text(range(1, 31), **kwargs))
            return render_directory(root)

    def test_page_sections_follow_the_required_order(self):
        """Order the page: table, waveforms, physical metrics, excursions, training health."""
        page = self._page(evaluate=5, on_task=0)
        order = [
            page.index("<th>Run</th>"),
            page.index("Waveforms"),
            page.index("Physical evaluation metrics"),
            page.index("Vertical force RMS error"),
            page.index("Task excursions"),
            page.index("Work objective"),
            page.index("Training health"),
            page.index("Episode return"),
            page.index("Clip fraction"),
        ]
        self.assertEqual(order, sorted(order))

    def test_excursion_chart_marks_the_on_task_limit(self):
        """Mark the on-task limit at zero on the task-excursion chart."""
        page = self._page(evaluate=5, on_task=0)
        excursions = page[page.index("Task excursions") : page.index("Work objective")]
        self.assertIn(">on task<", excursions)
        self.assertIn('stroke="#14663f"', excursions)
        for channel in ("duration", "impulse", "momentum"):
            self.assertIn(f"v9 {channel}", excursions)

    def test_on_task_badge_reports_both_states(self):
        """Show ON TASK for on_task=1 and OFF TASK for on_task=0."""
        on_page = self._page(evaluate=5, on_task=1)
        self.assertIn('<span class="task ontask">ON TASK</span>', on_page)
        self.assertNotIn("OFF TASK", on_page)
        off_page = self._page(evaluate=5, on_task=0)
        self.assertIn('<span class="task offtask">OFF TASK</span>', off_page)
        self.assertNotIn(">ON TASK<", off_page)

    def test_summary_table_reports_objective_and_total_excursion(self):
        """Report the latest objective and the summed latest excursion in the table."""
        page = self._page(evaluate=5, on_task=0, objective="77.400")
        self.assertIn("<th>Latest objective [J]</th>", page)
        self.assertIn("<th>Latest total excursion</th>", page)
        self.assertIn("<td>77.4</td>", page)
        self.assertIn("<td>1.19</td>", page)

    def test_run_without_evaluation_records_renders_gracefully(self):
        """Render the evaluation charts and table cells when a run logs no eval record."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train_old.log").write_text(_log_text(range(1, 21), frozen=FROZEN_LINE))
            (root / "train_new.log").write_text(_log_text(range(1, 21), evaluate=5))
            runs = load_runs(root)
            self.assertEqual([len(run.evaluations) for run in runs], [4, 0])
            self.assertIsNone(runs[1].latest_excursion_total())
            self.assertEqual(runs[1].task_status, "no eval")
            page = render(runs, root)
            self.assertIn('<span class="task noeval">no eval</span>', page)
            self.assertIn("\u2014", page)
            excursions = page[page.index("Task excursions") : page.index("Work objective")]
            self.assertIn("new duration", excursions)
            self.assertNotIn("old duration", excursions)
            self.assertIsNone(_NONFINITE.search(page))

    def test_nan_objective_stays_out_of_the_page(self):
        """Keep an undefined objective out of the HTML and out of the SVG coordinates."""
        page = self._page(evaluate=5, objective="nan")
        self.assertIsNone(_NONFINITE.search(page))
        objective = page[page.index("Work objective") : page.index("Episode return")]
        self.assertIn("No samples yet.", objective)
        self.assertEqual(_svg_fragments(objective), [])
        for fragment in _svg_fragments(page):
            ElementTree.fromstring(fragment)

    def test_mixed_finite_and_nan_objective_draws_finite_coordinates(self):
        """Drop undefined objective samples but still draw the finite ones."""
        x = np.arange(4.0)
        y = np.array([1.0, float("nan"), 3.0, 4.0])
        section = plot("Work objective", "J", [("v9", x, y, "#1967b3", False)])
        self.assertIsNone(_NONFINITE.search(section))
        coordinates = _series_coordinates(section, "#1967b3")
        self.assertEqual(len(coordinates), 3)
        for px, py in coordinates:
            self.assertTrue(np.isfinite(float(px)))
            self.assertTrue(np.isfinite(float(py)))


class TestWaveforms(unittest.TestCase):
    """Verify waveform overlays loaded from the evaluation trace archive."""

    def _render_with_trace(self, root, *, write=True, corrupt=None, iteration=30):
        """Write a run whose latest eval points at a trace archive, then render the page."""
        trace = root / "policy_v9.eval.npz"
        if write:
            _write_trace(trace, iteration=iteration)
        if corrupt == "garbage":
            trace.write_bytes(b"PK\x03\x04 not really an archive")
        elif corrupt == "truncated":
            data = trace.read_bytes()
            trace.write_bytes(data[: len(data) // 3])
        elif corrupt == "empty":
            trace.write_bytes(b"")
        text = _log_text(range(1, 31), evaluate=10, trace=str(trace))
        (root / "train_v9.log").write_text(text)
        return render_directory(root)

    def test_waveforms_render_from_a_trace_archive(self):
        """Overlay simulated and measured waveforms from the latest trace archive."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            page = self._render_with_trace(root)
            self.assertIn("Waveforms: 1 run overlaid", page)
            self.assertIn("v9 at iteration 30", page)
            self.assertNotIn("No waveforms for the current selection.", page)
            for title in (
                "Vertical ground reaction force",
                "Fore-aft ground reaction force",
                "COM height",
                "COM vertical velocity",
                "Leg length",
                "Commanded stiffness",
                "Commanded damping ratio",
            ):
                self.assertIn(title, page)
            waveforms = page[page.index("Waveforms:") : page.index("Physical evaluation metrics")]
            self.assertIn("<title>v9</title>", waveforms)
            self.assertIn("<title>measured</title>", waveforms)
            self.assertIn("v9 commanded L0", waveforms)
            self.assertIn(">contact start<", waveforms)
            self.assertIn(">contact end<", waveforms)
            self.assertIn("Contact [%]", waveforms)
            self.assertEqual(len(_svg_fragments(waveforms)), 7)
            # The vertical force chart leads the waveform group.
            self.assertLess(waveforms.index("Vertical ground reaction force"), waveforms.index("COM height"))
            for fragment in _svg_fragments(page):
                ElementTree.fromstring(fragment)
            self.assertIsNone(_NONFINITE.search(page))

    def test_broken_archives_degrade_to_no_waveforms(self):
        """Degrade to a no-waveforms message for missing or half-written archives."""
        for case in (None, "garbage", "truncated", "empty"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                page = self._render_with_trace(root, write=case is not None, corrupt=case)
                self.assertIn("No waveforms for the current selection.", page)
                self.assertIn("Physical evaluation metrics", page)
                self.assertIn("</html>", page)
                self.assertIsNone(_NONFINITE.search(page))

    def test_trace_is_found_next_to_the_logs(self):
        """Resolve a trace path that no longer matches the written directory."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_trace(root / "policy_v9.eval.npz")
            (root / "train_v9.log").write_text(
                _log_text(range(1, 31), evaluate=10, trace="outputs/impedance_instron/policy_v9.eval.npz")
            )
            runs = load_runs(root)
            self.assertIsNotNone(runs[0].trace)
            self.assertIn("Waveforms: 1 run overlaid", render(runs, root))

    def test_physical_charts_and_table_report_measured_errors(self):
        """Draw measured reference lines and report the physical errors in the table."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            page = self._render_with_trace(root)
            physical = page[page.index("Physical evaluation metrics") : page.index("Task excursions")]
            self.assertEqual(len(_svg_fragments(physical)), 8)
            for title in (
                "Peak vertical force",
                "Time of peak force",
                "Vertical force RMS error",
                "Impulse error",
                "COM vertical velocity RMS error",
                "COM height RMS error",
                "Contact duration",
            ):
                self.assertIn(title, physical)
            self.assertIn(">measured peak force<", physical)
            self.assertIn(">measured peak timing<", physical)
            self.assertIn(">measured impulse<", physical)
            self.assertIn(">measured contact<", physical)
            self.assertIn("<th>Peak force (error vs measured)</th>", page)
            self.assertIn("1791.9 N (1.67 %)", page)  # 1791.923 against the measured 1762.554 N.
            self.assertIn("<td>0.398</td>", page)  # 47.352 % minus the measured 46.954 %.
            table = page[page.index("<th>Run</th>") : page.index("Waveforms")]
            self.assertIn("<th>Contact error [ms]</th>", page)
            self.assertEqual(len(re.findall(r"<td>[-\d.]+</td>", table)) > 0, True)

    def test_run_without_trace_still_reports_physical_metrics(self):
        """Keep the physical charts and table when a run logs no trace path."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train_v9.log").write_text(_log_text(range(1, 31), evaluate=10))
            runs = load_runs(root)
            self.assertIsNone(runs[0].trace)
            page = render(runs, root)
            self.assertIn("No waveforms for the current selection.", page)
            physical = page[page.index("Physical evaluation metrics") : page.index("Task excursions")]
            self.assertEqual(len(_svg_fragments(physical)), 8)
            self.assertIn("\u2014", page)  # The contact-duration error needs a measured trace.
            self.assertIsNone(_NONFINITE.search(page))


class TestMaterialToken(unittest.TestCase):
    """Verify the optional artifact material token and its presentation."""

    REAL_LINE = (
        "eval iteration=400 objective_j=77.400 feasible=1 on_task=0 excursion_duration=0.000 "
        "excursion_impulse=0.000 excursion_momentum=1.190 violation_total=0.000 eval_return=-27.407 "
        "peak_fz_n=1050.000 peak_fz_ref_n=1000.000 peak_time_pct=50.000 peak_time_ref_pct=50.000 "
        "fz_rms_n=35.482 impulse_err_pct=5.000 com_vz_rms=0.100 com_z_rms_mm=9.854 contact_ms=138.000 "
        "peak_compression_mm=12.000 trace=/path/policy_ankle_v2_digital_shoe-c2e5666d.eval.npz "
        "artifact=digital_shoe-c2e5666d"
    )

    def test_parse_line_with_artifact_token(self):
        """Parse the trailing artifact token and keep every earlier field."""
        record = parse_evaluation(self.REAL_LINE)
        self.assertIsNotNone(record)
        self.assertEqual(record["artifact"], "digital_shoe-c2e5666d")
        self.assertEqual(record["trace"], "/path/policy_ankle_v2_digital_shoe-c2e5666d.eval.npz")
        self.assertEqual(record["iteration"], 400.0)
        self.assertAlmostEqual(record["eval_return"], -27.407)
        self.assertAlmostEqual(record["peak_fz_n"], 1050.0)
        self.assertAlmostEqual(record["contact_ms"], 138.0)
        self.assertAlmostEqual(record["peak_compression_mm"], 12.0)

    def test_artifact_token_is_optional(self):
        """Parse a record without the token and report no material for it."""
        record = parse_evaluation(EVAL_LINE[: EVAL_LINE.index(" artifact=")])
        self.assertIsNotNone(record)
        self.assertNotIn("artifact", record)
        self.assertIsNone(parse_evaluation(self.REAL_LINE + " artifact=other"))  # A duplicate key is malformed.
        empty = parse_evaluation(self.REAL_LINE.replace("artifact=digital_shoe-c2e5666d", "artifact="))
        self.assertIsNotNone(empty)
        self.assertNotIn("artifact", empty)

    def test_logs_without_the_token_are_unchanged(self):
        """Keep the old presentation for a log that carries no material token."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train_v9.log").write_text(_log_text(range(1, 31), evaluate=10))
            runs = load_runs(root)
            self.assertEqual(runs[0].materials, [None])
            self.assertIsNone(runs[0].material)
            page = render(runs, root)
            self.assertNotIn("sweep:", page)
            self.assertNotIn("material <code>", page)
            self.assertIn("<th>Shoe material</th>", page)
            self.assertIsNone(_NONFINITE.search(page))

    def test_single_material_is_named_in_the_table_and_legend(self):
        """Name the material of a single-material run in the table and legend."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train_v9.log").write_text(
                _log_text(range(1, 31), evaluate=10, artifacts=["digital_shoe-c2e5666d"])
            )
            runs = load_runs(root)
            self.assertEqual(runs[0].materials, ["digital_shoe-c2e5666d"])
            page = render(runs, root)
            self.assertIn("<code>digital_shoe-c2e5666d</code>", page)
            self.assertIn("material <code>digital_shoe-c2e5666d</code>", page)
            self.assertNotIn("sweep:", page)

    def test_material_sweep_is_grouped_not_collapsed(self):
        """Draw one curve per material token and mark the log as a sweep."""
        tokens = ["shoe-aaaa1111", "shoe-bbbb2222", "shoe-cccc3333"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train_v9.log").write_text(_log_text(range(1, 61), evaluate=5, artifacts=tokens))
            runs = load_runs(root)
            self.assertEqual(runs[0].materials, tokens)
            groups = eval_groups(runs)
            self.assertEqual([group.token for group in groups], tokens)
            self.assertEqual([group.label for group in groups], [f"v9 [{token}]" for token in tokens])
            self.assertEqual(len({group.color for group in groups}), 3)
            self.assertEqual(sum(len(group.records) for group in groups), 12)
            page = render(runs, root)
            self.assertIn("sweep: 3 materials", page)
            objective = page[page.index("Work objective") : page.index("Training health")]
            for token in tokens:
                self.assertIn(f"v9 [{token}]", objective)
            self.assertEqual(len(re.findall(r'fill="none" stroke="#', objective)), 3)
            for fragment in _svg_fragments(page):
                ElementTree.fromstring(fragment)

    def test_waveform_heading_names_the_material(self):
        """Name the material in the waveform heading and flag a stale archive."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "policy_v9.eval.npz"
            _write_trace(trace, artifact="shoe-aaaa1111")
            (root / "train_v9.log").write_text(
                _log_text(range(1, 31), evaluate=10, trace=str(trace), artifacts=["shoe-aaaa1111"])
            )
            page = render_directory(root)
            self.assertIn("Waveforms: 1 run overlaid", page)
            self.assertIn("v9 [shoe-aaaa1111] at iteration 30", page)
            self.assertNotIn("may be stale", page)

    def test_stale_archive_token_is_reported(self):
        """Warn when the archive token differs from the logged material token."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "policy_v9.eval.npz"
            _write_trace(trace, artifact="shoe-old00000")
            (root / "train_v9.log").write_text(
                _log_text(range(1, 31), evaluate=10, trace=str(trace), artifacts=["shoe-new11111"])
            )
            page = render_directory(root)
            self.assertIn("may be stale", page)
            self.assertIn("shoe-old00000", page)

    def test_archive_token_is_used_when_the_log_has_none(self):
        """Fall back to the archive token when the log line carries no material."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "policy_v9.eval.npz"
            _write_trace(trace, artifact="shoe-aaaa1111")
            (root / "train_v9.log").write_text(_log_text(range(1, 31), evaluate=10, trace=str(trace)))
            runs = load_runs(root)
            self.assertEqual(runs[0].trace["artifact"], "shoe-aaaa1111")
            self.assertIn("v9 [shoe-aaaa1111] at iteration 30", render(runs, root))


class TestSelection(unittest.TestCase):
    """Verify server-side run and material selection through the query string."""

    TOKENS = ("shoe-aaaa1111", "shoe-bbbb2222")

    def _directory(self, root):
        """Write three runs, two of them carrying different material tokens."""
        (root / "train_v1.log").write_text(_log_text(range(1, 21), evaluate=5, artifacts=[self.TOKENS[0]]))
        (root / "train_v2.log").write_text(_log_text(range(1, 21), evaluate=5, artifacts=[self.TOKENS[1]]))
        (root / "train_v3.log").write_text(_log_text(range(1, 21), evaluate=5, artifacts=list(self.TOKENS)))
        return load_runs(root)

    def _rows(self, page):
        """Return the run names listed in the summary table body."""
        body = page[page.index("<tbody>") : page.index("</tbody>")]
        return re.findall(r'<i class="swatch"[^>]*></i>([\w.\-]+)</th>', body)

    def test_parse_selection_reads_only_known_keys(self):
        """Read run and material values and ignore junk keys, blanks and duplicates."""
        selection = parse_selection("?run=a&run=b&run=a&material=m1&sort=x&run=")
        self.assertEqual(selection.runs, ("a", "b"))
        self.assertEqual(selection.materials, ("m1",))
        self.assertTrue(selection.active)
        self.assertFalse(parse_selection("").active)
        self.assertFalse(parse_selection("?").active)
        self.assertEqual(parse_selection("nonsense").runs, ())

    def test_no_query_shows_every_run(self):
        """Render every run, exactly as an unfiltered page did before selection existed."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = self._directory(root)
            plain, empty_query = render(runs, root), render(runs, root, query="")
            self.assertEqual(len(plain), len(empty_query))
            self.assertEqual(self._rows(plain), ["v1", "v2", "v3"])
            self.assertIn("Showing 3 of 3 runs", plain)
            self.assertIn("Showing all 3 runs", plain)
            self.assertNotIn("Filtered view", plain)
            health = plain[plain.index("Training health") :]
            self.assertEqual(len(re.findall(r'fill="none" stroke="#', health)), 24)  # 3 runs, 7 charts, 2 return lines.

    def test_single_run_filter(self):
        """Draw only the requested run and say which runs are hidden."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = self._directory(root)
            page = render(runs, root, query="run=v2")
            self.assertEqual(self._rows(page), ["v2"])
            self.assertIn("Showing 1 of 3 runs", page)
            self.assertIn("Filtered view: showing 1 of 3 runs.", page)
            self.assertIn("Hidden: v1, v3.", page)
            self.assertIn("2 run rows are hidden", page)
            health = page[page.index("Training health") :]
            self.assertEqual(len(re.findall(r'fill="none" stroke="#', health)), 8)
            self.assertNotIn("v1 mean", page)
            self.assertNotIn("v3 mean", page)

    def test_runs_and_material_filters_combine(self):
        """Combine a two-run selection with a material filter using AND."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = self._directory(root)
            page = render(runs, root, query=f"run=v1&run=v3&material={self.TOKENS[0]}")
            self.assertEqual(self._rows(page), ["v1", "v3"])  # v2 fails the run filter.
            self.assertIn(f"Material filter: {self.TOKENS[0]}.", page)
            objective = page[page.index("Work objective") : page.index("Training health")]
            self.assertIn(f"v3 [{self.TOKENS[0]}]", objective)
            self.assertNotIn(f"v3 [{self.TOKENS[1]}]", objective)
            # The material filter alone keeps every run that ran that foam.
            material_only = render(runs, root, query=f"material={self.TOKENS[1]}")
            self.assertEqual(self._rows(material_only), ["v2", "v3"])

    def test_refresh_url_preserves_the_selection(self):
        """Keep the active query string in the meta refresh target."""
        self.assertEqual(refresh_url(parse_selection("")), "/")
        selection = parse_selection("run=v1&run=v2&material=shoe-aaaa1111")
        self.assertEqual(refresh_url(selection), "/?run=v1&run=v2&material=shoe-aaaa1111")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = self._directory(root)
            page = render(runs, root, refresh=10, query="run=v1&run=v2")
            self.assertIn('<meta http-equiv="refresh" content="10; url=/?run=v1&amp;run=v2">', page)
            single = render(runs, root, refresh=7, query="run=v2")
            self.assertIn('<meta http-equiv="refresh" content="7; url=/?run=v2">', single)
            self.assertIn('<meta http-equiv="refresh" content="10; url=/">', render(runs, root))

    def test_unknown_names_are_ignored(self):
        """Ignore a stale run name instead of failing the request."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = self._directory(root)
            page = render(runs, root, query="run=v1&run=deleted_run")
            self.assertEqual(self._rows(page), ["v1"])
            self.assertIn("Showing 1 of 3 runs", page)

    def test_empty_selection_still_offers_the_form(self):
        """Render a readable page with the form when nothing matches."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = self._directory(root)
            for query in ("run=__none__", "run=ghost", "material=shoe-zzzz9999"):
                with self.subTest(query=query):
                    page = render(runs, root, query=query)
                    self.assertIn("No runs match this selection", page)
                    self.assertIn('<form method="get" class="filter">', page)
                    self.assertIn('<a href="/">all runs</a>', page)
                    self.assertIn("Showing 0 of 3 runs", page)
                    self.assertIn("</html>", page)
                    self.assertIsNone(_NONFINITE.search(page))

    def test_form_lists_every_run_with_its_checked_state(self):
        """Offer one checkbox per run and per material, ticking the active ones."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = self._directory(root)
            page = render(runs, root, query=f"run=v2&material={self.TOKENS[1]}")
            for name in ("v1", "v2", "v3"):
                self.assertIn(f'<input type="checkbox" name="run" value="{name}"', page)
            self.assertIn('<input type="checkbox" name="run" value="v2" checked>', page)
            self.assertIn('<input type="checkbox" name="run" value="v1">', page)
            for token in self.TOKENS:
                self.assertIn(f'<input type="checkbox" name="material" value="{token}"', page)
            self.assertIn(f'<input type="checkbox" name="material" value="{self.TOKENS[1]}" checked>', page)
            self.assertIn('<a href="/?run=__none__">none</a>', page)

    def test_handler_applies_the_query_string(self):
        """Filter over HTTP and keep the query in the served refresh tag."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._directory(root)
            server = create_server(root, host="127.0.0.1", port=0, refresh=10)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            try:
                base = f"http://127.0.0.1:{server.server_address[1]}/"
                with opener.open(base + "?run=v1&run=v3", timeout=10) as response:
                    page = response.read().decode("utf-8")
                self.assertIn('<meta http-equiv="refresh" content="10; url=/?run=v1&amp;run=v3">', page)
                self.assertEqual(self._rows(page), ["v1", "v3"])
                with opener.open(base, timeout=10) as response:
                    self.assertIn("Showing 3 of 3 runs", response.read().decode("utf-8"))
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=10)


class TestWaveformOverlay(unittest.TestCase):
    """Verify the single overlaid waveform block."""

    def _runs(self, root, *, names=("v1", "v2"), samples=(400, 400), iterations=(30, 30), reference_scale=(1.0, 1.0)):
        """Write one log and archive per run, then load the runs."""
        for name, count, iteration, scale in zip(names, samples, iterations, reference_scale, strict=True):
            trace = root / f"policy_{name}.eval.npz"
            _write_trace(trace, iteration=iteration, samples=count, reference_scale=scale)
            (root / f"train_{name}.log").write_text(
                _log_text(range(1, iteration + 1), evaluate=max(iteration // 3, 1), trace=str(trace))
            )
        return load_runs(root)

    def _waveforms(self, page):
        """Slice out the waveform article of a rendered page."""
        return page[page.index("<h2>Waveforms") : page.index("Physical evaluation metrics")]

    def _panel(self, page, title):
        """Slice out one waveform panel by title."""
        section = self._waveforms(page)
        start = section.index(f"<h3>{title} ")
        return section[start : section.index("</section>", start)]

    def test_one_block_holds_every_run(self):
        """Render one waveform block with seven panels, whatever the run count."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = self._runs(
                root,
                names=("v1", "v2", "v3"),
                samples=(400, 400, 400),
                iterations=(30, 60, 90),
                reference_scale=(1.0, 1.0, 1.0),
            )
            page = render(runs, root)
            self.assertEqual(page.count("<h2>Waveforms"), 1)
            self.assertEqual(len(_svg_fragments(self._waveforms(page))), 7)
            self.assertIn("Waveforms: 3 runs overlaid", page)

    def test_each_panel_overlays_every_run_with_one_reference(self):
        """Draw one curve per run and a single shared measured reference."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            page = render(
                self._runs(
                    root,
                    names=("v1", "v2", "v3"),
                    samples=(400, 400, 400),
                    iterations=(30, 30, 30),
                    reference_scale=(1.0, 1.0, 1.0),
                ),
                root,
            )
            panel = self._panel(page, "Vertical ground reaction force")
            self.assertEqual(len(re.findall(r'fill="none" stroke="#', panel)), 4)  # Three runs plus one reference.
            self.assertEqual(panel.count("<title>measured</title>"), 1)
            for name in ("v1", "v2", "v3"):
                self.assertIn(f"<title>{name}</title>", panel)

    def test_curve_colours_match_the_other_charts(self):
        """Use the same colour for a run in the waveforms, the table and the training charts."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = self._runs(root)
            page = render(runs, root)
            colors = run_colors(runs)
            panel = self._panel(page, "Vertical ground reaction force")
            health = page[page.index("Training health") :]
            for name in ("v1", "v2"):
                color = colors[name]
                self.assertIn(f'stroke="{color}" stroke-width="2"><title>{name}</title>', panel)
                self.assertIn(f'stroke="{color}" stroke-width="2"><title>{name}</title>', health)
                self.assertIn(f'<i class="swatch" style="background:{color}"></i>{name}</th>', page)

    def test_coarse_archive_stays_dashed_with_marked_samples(self):
        """Keep a coarse trace dashed and marked inside the overlay."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = self._runs(
                root, names=("coarse", "full"), samples=(46, 2881), iterations=(30, 30), reference_scale=(1.0, 1.0)
            )
            page = render(runs, root)
            panel = self._panel(page, "Vertical ground reaction force")
            colors = run_colors(runs)
            self.assertIn(f'stroke="{colors["coarse"]}" stroke-width="2" stroke-dasharray="2 3"', panel)
            self.assertIn(f'fill="{colors["coarse"]}" stroke="none"><title>coarse samples</title>', panel)
            self.assertIn(f'stroke="{colors["full"]}" stroke-width="2"><title>full</title>', panel)
            self.assertNotIn(f'fill="{colors["full"]}" stroke="none"', panel)
            self.assertIn("dashed with its samples marked", page)

    def test_empty_case_renders_one_message(self):
        """Say once that the selection has no waveforms, not once per run."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("v1", "v2", "v3"):
                (root / f"train_{name}.log").write_text(_log_text(range(1, 21), evaluate=5))
            page = render(load_runs(root), root)
            self.assertEqual(page.count("No waveforms for the current selection."), 1)
            self.assertEqual(len(_svg_fragments(self._waveforms(page))), 0)

    def test_header_names_the_runs_and_their_iterations(self):
        """Name every drawn run and the evaluation iteration it came from."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            page = render(
                self._runs(
                    root, names=("v1", "v2"), samples=(400, 400), iterations=(30, 90), reference_scale=(1.0, 1.0)
                ),
                root,
            )
            self.assertIn("v1 at iteration 30", page)
            self.assertIn("v2 at iteration 90", page)
            self.assertIn("per cent of the contact window", page)
            self.assertIn("CONTACT_FORCE_FRACTION = 0.02", page)

    def test_disagreeing_references_are_reported(self):
        """Warn when selected runs were scored against different measured references."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = self._runs(
                root, names=("v1", "v2"), samples=(400, 400), iterations=(30, 30), reference_scale=(1.0, 1.5)
            )
            page = render(runs, root)
            self.assertIn("Selected runs carry different measured references", page)
            self.assertIn("Vertical ground reaction force: v2", page)
        with tempfile.TemporaryDirectory() as directory:
            agreeing = Path(directory)
            page = render(
                self._runs(
                    agreeing, names=("v1", "v2"), samples=(400, 400), iterations=(30, 30), reference_scale=(1.0, 1.0)
                ),
                agreeing,
            )
            self.assertNotIn("different measured references", self._waveforms(page))

    def test_coarse_archive_is_not_reported_as_disagreeing(self):
        """Ignore a sparse archive in the reference check, since it only misses peaks."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = self._runs(
                root,
                names=("coarse", "full"),
                samples=(46, 2881),
                iterations=(30, 30),
                reference_scale=(1.0, 1.0),
            )
            page = render(runs, root)
            self.assertNotIn("different measured references", page)

    def test_overlay_follows_the_selection(self):
        """Draw only the selected runs in the overlay."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = self._runs(
                root,
                names=("v1", "v2", "v3"),
                samples=(400, 400, 400),
                iterations=(30, 30, 30),
                reference_scale=(1.0, 1.0, 1.0),
            )
            page = render(runs, root, query="run=v2")
            self.assertIn("Waveforms: 1 run overlaid", page)
            panel = self._panel(page, "Vertical ground reaction force")
            self.assertEqual(len(re.findall(r'fill="none" stroke="#', panel)), 2)  # One run plus the reference.
            self.assertIn("<title>v2</title>", panel)
            self.assertNotIn("<title>v1</title>", panel)


class TestServer(unittest.TestCase):
    """Verify the stdlib HTTP handler on a loopback ephemeral port."""

    def test_handler_serves_the_dashboard(self):
        """Serve the rendered dashboard over HTTP and reject unknown paths."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train_v1.log").write_text(_log_text(range(1, 21), frozen=FROZEN_LINE))
            server = create_server(root, host="127.0.0.1", port=0, refresh=5)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            # Bypass any ambient proxy so the test only touches loopback.
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            try:
                url = f"http://127.0.0.1:{server.server_address[1]}/"
                with opener.open(url, timeout=10) as response:
                    page = response.read().decode("utf-8")
                self.assertEqual(response.status, 200)
                self.assertIn("Impedance Instron training", page)
                self.assertIn("v1", page)
                self.assertIn('content="5; url=/"', page)
                with self.assertRaises(urllib.error.HTTPError):
                    opener.open(url + "missing", timeout=10)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=10)


if __name__ == "__main__":
    unittest.main()
