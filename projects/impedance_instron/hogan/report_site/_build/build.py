"""Assemble report_site/index.html from _build/template.html and the exported JSON files.

The JSON files were exported from local run outputs and are committed beside the
page, so this rebuild needs nothing else. CPU only.
"""

import json
from pathlib import Path

import numpy as np

SITE = Path(__file__).resolve().parents[1]


def compact_motion(mo):
    """Arrays-only form of motion.json read by the page's player (3 decimals for positions)."""

    def r3(a):
        return np.round(np.asarray(a, float), 3).tolist()

    out = {"trial": mo["trial"], "dt": mo["frame_dt_s"], "speed": mo["speed_m_s"], "c": {}}
    for cond, v in mo["conditions"].items():
        shoe = v["shoe"]
        frames = [
            [
                *r3(f["hip"]),
                *r3(f["knee"]),
                *r3(f["ankle"]),
                *r3(f["toe"]),
                *r3(f["com"]),
                round(f["foot"], 4),
                round(f["grf"][0]),
                round(f["grf"][1]),
                round(f["comp"], 3),
            ]
            for f in v["frames"]
        ]
        out["c"][cond] = {
            "last": r3(shoe["last"]),
            "mid": r3(shoe["midsole"]),
            "rb": r3(shoe["row_bottom"]),
            "rt": r3(shoe["row_top"]),
            "f": frames,
            "loss": round(v["loss"], 2),
            "peakc": round(v["peak_compression_fraction"], 3),
            "peakfz": round(v["peak_grf_n"][1]),
            "dur": round(v["contact_duration_s"], 3),
        }
    reference = mo["conditions"]["original__reference"]["frames"]
    out["meas"] = [
        [
            f["t"],
            *r3(f["meas"]["hip"]),
            *r3(f["meas"]["knee"]),
            *r3(f["meas"]["ankle"]),
            *r3(f["meas"]["toe"]),
            *r3(f["meas"]["com"]),
            round(f["meas"]["foot"], 4),
            round(f["grf_meas"][0]),
            round(f["grf_meas"][1]),
        ]
        for f in reference
        if "meas" in f
    ]
    return out


def compact_instron(ins):
    """Held-out, repeat and loop data used by the Instron section (training split dropped)."""
    return {
        "rises_ms": ins["rises_ms"],
        "peaks_n": ins["peaks_n"],
        "gate": ins["gate"],
        "loops": ins["loops"],
        "conditions": {c: {k: v for k, v in s.items() if k != "train"} for c, s in ins["conditions"].items()},
    }


def main():
    def read(name):
        return json.loads((SITE / name).read_text(encoding="utf-8"))

    def dump(obj):
        return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)

    page = (SITE / "_build/template.html").read_text(encoding="utf-8")
    page = (
        page.replace("__DATA__", dump(read("data.json")))
        .replace("__MO__", dump(compact_motion(read("motion.json"))))
        .replace("__IN__", dump(compact_instron(read("instron.json"))))
    )
    (SITE / "index.html").write_text(page, encoding="utf-8")
    print("index.html", len(page), "chars")


if __name__ == "__main__":
    main()
