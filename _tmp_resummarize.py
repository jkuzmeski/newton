"""Regenerate summary.json and summary.md from an existing rows.json, without simulating."""

import json
import sys
from pathlib import Path

from projects.impedance_instron.bilateral.eval_shoes import markdown, summarize

for directory in [Path(a) for a in sys.argv[1:]]:
    rows = json.loads((directory / "rows.json").read_text(encoding="utf-8"))
    report = summarize(rows)
    (directory / "summary.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    (directory / "summary.md").write_text(markdown(rows, report), encoding="utf-8")
    print(directory.name, "rewrote", len(rows), "rows")
