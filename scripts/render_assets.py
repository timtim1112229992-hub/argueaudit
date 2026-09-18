# -*- coding: utf-8 -*-
"""P9: produce the tables and figures in the visual asset register.

Reads the model output from results.json and rebuilds the derived frames, which
is fast because the expensive part of the pipeline is the bootstrap and that
work is already persisted. Nothing here recomputes an estimate; if a figure and
the results file disagree, the figure is wrong.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from argueaudit import figures, tables  # noqa: E402
from argueaudit.config import SETTINGS  # noqa: E402
from argueaudit.derive import (distinctness, field_frame, mark_provenance,  # noqa: E402
                               stage_frame)
from argueaudit.ingest import load  # noqa: E402
from argueaudit.rubric import adjudicate, code_frame  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    out_root = Path(args.output_dir) if args.output_dir else SETTINGS.output_dir
    results_path = out_root / "results.json"
    if not results_path.exists():
        print(f"{results_path} not found. Run scripts/run_pipeline.py first.")
        return 1
    results = json.loads(results_path.read_text(encoding="utf-8"))["results"]

    corpus = load(Path(args.data_dir) if args.data_dir else None)
    fields = field_frame(corpus.submissions)
    stages = stage_frame(corpus.submissions)
    coded = mark_provenance(adjudicate(code_frame(fields)))
    dist = distinctness(fields)

    assets = out_root / "assets"
    written = figures.render_all(stages, dist, results, assets)
    doc = tables.render_all(coded, fields, stages, dist, results, assets)

    print(f"figures written to {assets}")
    for path in written:
        print(f"   {path.name}")
    print(f"\ntables written")
    print(f"   {doc.name}")
    for path in sorted(assets.glob("table_*.csv")):
        print(f"   {path.name}")

    # The custody guarantee applies to rendered assets as well as to the data files.
    import re
    han = re.compile(r"[\u4e00-\u9fff]")
    leaked = [p.name for p in assets.glob("table_*.csv")
              if han.search(p.read_text(encoding="utf-8"))]
    print(f"\nrendered tables carrying Han characters: {leaked or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
