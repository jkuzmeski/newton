# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test dependency-light SVG figures without a simulator or plotting backend."""

import re
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from projects.impedance_instron.simple.figures import (
    BASELINE_COLOR,
    GAIN_COLORS,
    PALETTE,
    Curve,
    extract_svg,
    line_figure,
    scatter_figure,
)


def svg_elements(figure, class_name):
    root = ET.fromstring(extract_svg(figure))
    return [node for node in root.iter() if class_name in node.get("class", "").split()]


def figure_curve(**kwargs):
    args = {
        "label": "Baseline",
        "time": np.arange(4.0),
        "value": np.array([0.0, -2.0, 3.0, 1.0]),
        "color": BASELINE_COLOR,
    }
    args.update(kwargs)
    return Curve(**args)


class TestImpedanceFigures(unittest.TestCase):
    def test_raw_samples_are_not_resampled(self):
        """Preserve every saved raw sample, including narrow spikes and repeated times."""
        time = np.repeat(np.arange(6000.0), 2)
        values = np.zeros_like(time)
        values[6789] = 123.0
        figure = line_figure([figure_curve(time=time, value=values)], title="Force", y_label="Force [N]")
        path = svg_elements(figure, "raw-trace")[0]
        self.assertEqual(int(path.get("data-sample-count")), len(time))
        self.assertEqual(len(re.findall(r"[ML]", path.get("d"))), len(time))
        coordinates = np.asarray(re.findall(r"[ML]([^, ]+),([^ ]+)", path.get("d")), dtype=float)
        self.assertEqual(coordinates[0, 0], coordinates[1, 0])
        self.assertLess(coordinates[6789, 1], coordinates[6788, 1])
        self.assertAlmostEqual(coordinates[6788, 1], coordinates[6790, 1])

    def test_nonfinite_gaps_and_isolated_samples(self):
        """Break paths at missing times or values and display isolated finite samples."""
        time = np.array([0, 1, 2, 3, np.nan, 5, 6, 7], dtype=float)
        value = np.array([0, 1, np.nan, 2, 3, 4, np.inf, 5], dtype=float)
        figure = line_figure([figure_curve(time=time, value=value)], title="Position", y_label="Position [mm]")
        path = svg_elements(figure, "raw-trace")[0]
        self.assertEqual(re.findall(r"[ML]", path.get("d")), ["M", "L", "M", "M", "M"])
        self.assertEqual(len(svg_elements(figure, "isolated-sample")), 3)
        self.assertIn("3 unavailable samples; gaps shown", figure)
        self.assertNotRegex(path.get("d").lower(), r"nan|inf")

    def test_exact_push_dwell_and_tolerance(self):
        """Render exact push and final-dwell intervals with a separate tolerance band."""
        figure = line_figure(
            [figure_curve()],
            title="Response",
            y_label="Displacement [mm]",
            zero=True,
            band=(-0.25, 0.25),
            spans=(
                {"start": 0.5, "end": 1.0, "label": "Push", "color": GAIN_COLORS[1]},
                {"start": 2.5, "end": 3.0, "label": "Final dwell", "color": GAIN_COLORS[2]},
            ),
        )
        spans = svg_elements(figure, "plot-span")
        self.assertEqual([(s.get("data-start"), s.get("data-end")) for s in spans], [("0.5", "1"), ("2.5", "3")])
        self.assertAlmostEqual(float(spans[0].get("width")), float(spans[1].get("width")))
        band = svg_elements(figure, "tolerance-band")[0]
        self.assertEqual((band.get("data-low"), band.get("data-high")), ("-0.25", "0.25"))
        self.assertEqual(len(svg_elements(figure, "zero-y")), 1)
        self.assertIn("Final dwell", figure)

    def test_true_terminal_marker_is_separate(self):
        """Draw a supplied post-integration terminal marker without extending the raw path."""
        figure = line_figure(
            [figure_curve(terminal_time=4.0, terminal_value=8.0)], title="Terminal", y_label="Velocity [mm/s]"
        )
        path = svg_elements(figure, "raw-trace")[0]
        self.assertEqual(len(re.findall(r"[ML]", path.get("d"))), 4)
        last_x, last_y = map(float, re.findall(r"[ML]([^, ]+),([^ ]+)", path.get("d"))[-1])
        marker = svg_elements(figure, "terminal-marker")[0]
        self.assertGreater(float(marker.get("cx")), last_x)
        self.assertLess(float(marker.get("cy")), last_y)
        self.assertIn("true terminal state (4, 8)", figure)
        self.assertIn("not last trace sample", figure)

    def test_terminal_is_never_inferred(self):
        """Do not relabel the last pre-integration sample as the true terminal state."""
        figure = line_figure([figure_curve()], title="Trace", y_label="Position [mm]")
        self.assertEqual(svg_elements(figure, "terminal-marker"), [])
        invalid = line_figure(
            [figure_curve(terminal_time=4.0, terminal_value=np.nan)], title="Trace", y_label="Position [mm]"
        )
        self.assertEqual(svg_elements(invalid, "terminal-marker"), [])
        self.assertIn("terminal unavailable", invalid)

    def test_trace_and_legend_share_color_and_dash(self):
        """Use the same explicit color and dash pattern in the trace and its legend."""
        figure = line_figure([figure_curve(color="#0072b2", dashed=True)], title="Reference", y_label="Pitch [mrad]")
        trace = svg_elements(figure, "raw-trace")[0]
        legend = svg_elements(figure, "legend-item")[0]
        key = next(iter(legend))
        self.assertEqual(trace.get("stroke"), key.get("stroke"))
        self.assertEqual(trace.get("stroke-dasharray"), key.get("stroke-dasharray"))
        self.assertEqual(trace.get("stroke-dasharray"), "7 4")
        self.assertGreaterEqual(len(PALETTE), 8)
        self.assertEqual(len(set(PALETTE)), len(PALETTE))

    def test_unavailable_trace_has_no_fabricated_data(self):
        """Label all-nonfinite data unavailable without fabricated finite points or tick statistics."""
        figure = line_figure(
            [figure_curve(time=np.arange(3.0), value=np.array([np.nan, np.inf, None]))],
            title="Missing",
            y_label="Force [N]",
            zero=True,
        )
        self.assertIn("Unavailable: no finite paired samples", figure)
        self.assertIn("trace unavailable", figure)
        self.assertEqual(svg_elements(figure, "raw-trace"), [])
        self.assertEqual(svg_elements(figure, "grid"), [])
        self.assertEqual(svg_elements(figure, "zero-line"), [])

    def test_unqualified_terminal_is_cross_and_labeled(self):
        """Keep an unqualified run visible with an explicit label and terminal cross."""
        figure = line_figure(
            [figure_curve(qualified=False, terminal_time=4, terminal_value=2)],
            title="Failed run",
            y_label="Position [mm]",
        )
        marker = svg_elements(figure, "terminal-marker")[0]
        self.assertTrue(marker.tag.endswith("path"))
        self.assertIn("unqualified", marker.get("class"))
        self.assertIn("Baseline — unqualified", figure)
        self.assertEqual(len(svg_elements(figure, "raw-trace")), 1)

    def test_scatter_exact_narrow_acceptance_box(self):
        """Keep a narrow position-velocity acceptance rectangle at its exact data extent."""
        figure = scatter_figure(
            [{"label": "Run", "x": 1000.0, "y": -2000.0, "color": GAIN_COLORS[0]}],
            title="Terminal phase plane",
            x_label="Position [mm]",
            y_label="Velocity [mm/s]",
            x_band=(-0.1, 0.1),
            y_band=(-0.2, 0.2),
        )
        box = svg_elements(figure, "acceptance-box")[0]
        self.assertEqual((box.get("data-x-low"), box.get("data-x-high")), ("-0.1", "0.1"))
        self.assertEqual((box.get("data-y-low"), box.get("data-y-high")), ("-0.2", "0.2"))
        self.assertLess(float(box.get("width")), 1.0)
        self.assertLess(float(box.get("height")), 1.0)
        self.assertAlmostEqual(float(box.get("width")), 622 * 0.2 / (1000.1 * 1.14), places=8)
        self.assertEqual(len(svg_elements(figure, "zero-x")), 1)
        self.assertEqual(len(svg_elements(figure, "zero-y")), 1)

    def test_scatter_preserves_invalid_point_identity(self):
        """Show finite failures as crosses and missing terminal pairs as unavailable legend entries."""
        figure = scatter_figure(
            [
                {"label": "Unstable", "x": 5, "y": -3, "color": GAIN_COLORS[1], "qualified": False},
                {"label": "Missing state", "x": np.nan, "y": 0, "color": GAIN_COLORS[0]},
                {"label": "No velocity", "x": 1, "color": GAIN_COLORS[2]},
            ],
            title="Recovery",
            x_label="Position [mm]",
            y_label="Velocity [mm/s]",
        )
        markers = svg_elements(figure, "scatter-point")
        self.assertEqual(len(markers), 1)
        self.assertTrue(markers[0].tag.endswith("path"))
        self.assertIn("Unstable — unqualified", figure)
        self.assertIn("Missing state — unavailable", figure)
        self.assertIn("No velocity — unavailable", figure)

    def test_scatter_grouped_legend_keeps_individual_points(self):
        """Group a large scatter legend without losing individual case labels or links."""
        points = [
            {
                "label": f"Case {index}: individual terminal state",
                "legend_label": f"Direction {index % 4}",
                "x": index,
                "y": -index,
                "color": GAIN_COLORS[index % 4],
                "href": f"pages/case-{index}.html",
            }
            for index in range(52)
        ]
        figure = scatter_figure(points, title="Terminal pairs", x_label="Position [mm]", y_label="Velocity [mm/s]")
        self.assertEqual(len(svg_elements(figure, "scatter-point")), 52)
        self.assertEqual(len(svg_elements(figure, "legend-item")), 4)
        self.assertIn("Case 51: individual terminal state", figure)
        self.assertIn('href="pages/case-51.html"', figure)
        root = ET.fromstring(extract_svg(figure))
        self.assertEqual(len(list(root.iter("{http://www.w3.org/2000/svg}a"))), 52)

    def test_scatter_grouping_does_not_hide_invalid_cases(self):
        """Keep unqualified groups and nonfinite case identities explicit in grouped legends."""
        points = [
            {"label": "Good", "legend_label": "Forward", "x": 1, "y": 2, "color": "blue"},
            {"label": "Failed", "legend_label": "Forward", "x": 2, "y": 3, "color": "blue", "qualified": False},
            {"label": "Missing case", "legend_label": "Forward", "x": None, "y": 3, "color": "blue"},
        ]
        figure = scatter_figure(points, title="Terminal pairs", x_label="Position [mm]", y_label="Velocity [mm/s]")
        self.assertEqual(len(svg_elements(figure, "legend-item")), 3)
        self.assertIn("Forward — unqualified", figure)
        self.assertIn("Missing case — unavailable", figure)
        self.assertIn("Failed — unqualified", figure)
        self.assertEqual(len(svg_elements(figure, "scatter-point")), 2)

    def test_escape_labels_captions_colors_and_links(self):
        """Escape all caller text and reject executable or external-file link schemes."""
        malicious = '<script>alert("x")</script>'
        figure = line_figure(
            [figure_curve(label=malicious, color='red" onload="alert(1)')],
            title=malicious,
            y_label="Force [N] & <units>",
            caption=malicious,
        )
        root = ET.fromstring(extract_svg(figure))
        self.assertNotIn("<script>", figure)
        self.assertIn("&lt;script&gt;", figure)
        self.assertTrue(all("onload" not in element.attrib for element in root.iter()))
        points = [
            {"label": "Safe", "x": 0, "y": 0, "color": "blue", "href": 'run.html?a=1&b="two"'},
            {"label": "Bad", "x": 1, "y": 1, "color": "red", "href": "javascript:alert(1)"},
            {"label": "File", "x": 2, "y": 2, "color": "red", "href": "file:///etc/passwd"},
        ]
        scatter = scatter_figure(points, title="Links", x_label="Position [mm]", y_label="Velocity [mm/s]")
        self.assertNotIn("javascript:", scatter)
        self.assertNotIn("file:///", scatter)
        self.assertIn('href="run.html?a=1&amp;b=&quot;two&quot;"', scatter)
        ET.fromstring(extract_svg(scatter))

    def test_units_accessibility_and_standalone_svg(self):
        """Keep units, accessible descriptions, legends, and styling in the standalone SVG."""
        figure = line_figure(
            [figure_curve()], title="Motion", x_label="Time [ms]", y_label="Pitch [mrad]", caption="Raw samples only."
        )
        standalone = extract_svg(figure)
        root = ET.fromstring(standalone)
        self.assertEqual(root.get("role"), "img")
        self.assertEqual(root.get("aria-label"), "Motion")
        self.assertEqual(root.find("{http://www.w3.org/2000/svg}title").text, "Motion")
        self.assertIn("Raw samples only.", root.find("{http://www.w3.org/2000/svg}desc").text)
        self.assertIn("Time [ms]", standalone)
        self.assertIn("Pitch [mrad]", standalone)
        self.assertIn('class="plot-legend"', standalone)
        self.assertIn('class="figure-caption"', figure)
        self.assertNotIn("<script", figure)
        self.assertNotIn("<figure", standalone)

    def test_absolute_height_uses_data_range_without_zero(self):
        """Fit absolute-height overlays tightly unless an explicit zero baseline is requested."""
        curve = figure_curve(value=np.array([1000.0, 1001.0, 1000.5, 999.0]))
        fitted = line_figure([curve], title="Height", y_label="Height [mm]")
        zeroed = line_figure([curve], title="Height", y_label="Height [mm]", zero=True)
        fitted_path = svg_elements(fitted, "raw-trace")[0]
        zeroed_path = svg_elements(zeroed, "raw-trace")[0]
        fitted_y = np.asarray(re.findall(r"[ML][^, ]+,([^ ]+)", fitted_path.get("d")), dtype=float)
        zeroed_y = np.asarray(re.findall(r"[ML][^, ]+,([^ ]+)", zeroed_path.get("d")), dtype=float)
        self.assertIn(">1000.5</text>", fitted)
        self.assertIn(">999.5</text>", fitted)
        self.assertGreater(np.ptp(fitted_y), 200)
        self.assertLess(np.ptp(zeroed_y), 1)
        self.assertEqual(svg_elements(fitted, "zero-y"), [])
        self.assertEqual(len(svg_elements(zeroed, "zero-y")), 1)

    def test_validate_shapes_and_bounds(self):
        """Reject mismatched raw arrays and nonfinite or reversed interval bounds."""
        with self.assertRaises(ValueError):
            line_figure([figure_curve(value=np.zeros(2))], title="Bad", y_label="Force [N]")
        with self.assertRaises(ValueError):
            line_figure([figure_curve()], title="Bad", y_label="Force [N]", band=(1, -1))
        with self.assertRaises(ValueError):
            line_figure(
                [figure_curve()],
                title="Bad",
                y_label="Force [N]",
                spans=({"start": 0, "end": np.inf, "label": "Bad", "color": "gray"},),
            )
        with self.assertRaises(ValueError):
            scatter_figure([], title="Bad", x_label="Position [mm]", y_label="Velocity [mm/s]", y_band=(0, np.nan))
        with self.assertRaises(ValueError):
            extract_svg("<p>No figure</p>")


if __name__ == "__main__":
    unittest.main()
