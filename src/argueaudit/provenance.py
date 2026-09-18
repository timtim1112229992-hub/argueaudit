"""The release package.

A code availability statement that cannot be checked is not a disclosure. What
leaves this machine is the column map, the digests that fix the analysed record
set, the coding decisions as data, and a manifest describing the run. No pupil
text is included, and the writer refuses to emit anything that carries it.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import PROVENANCE_DIR, SETTINGS
from .lexicon import RULESET_VERSION

PACKAGES = ("numpy", "pandas", "scipy", "statsmodels", "matplotlib", "openpyxl")
ALLOWED_KEYS = {"schema", "generated_utc", "source", "code", "environment", "settings",
                "column_map", "tables", "counts", "notes"}

# Columns that carry pupil-authored text and must never reach a written file.
FORBIDDEN_COLUMNS = ("raw", "norm")


def _git(*args: str) -> str | None:
    try:
        root = Path(__file__).resolve().parents[2]
        out = subprocess.run(("git", *args), cwd=root, capture_output=True,
                             text=True, timeout=15)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def _versions() -> dict:
    found = {}
    for name in PACKAGES:
        try:
            found[name] = __import__(name).__version__
        except Exception:
            found[name] = None
    return found


def build_manifest(ingest_meta: dict, notes: str = "") -> dict:
    return {
        "schema": "argueaudit/provenance/1",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": ingest_meta.get("source"),
        "code": {"commit": _git("rev-parse", "HEAD"),
                 "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
                 "dirty": bool(_git("status", "--porcelain"))},
        "environment": {"python": sys.version.split()[0],
                        "implementation": platform.python_implementation(),
                        "packages": _versions()},
        "settings": {"seed": SETTINGS.seed,
                     "analytic_groups": SETTINGS.analytic_groups,
                     "n_stages": SETTINGS.n_stages,
                     "bootstrap_replicates": SETTINGS.bootstrap_replicates,
                     "permutation_replicates": SETTINGS.permutation_replicates,
                     "min_content_chars": SETTINGS.min_content_chars,
                     "min_propositional_chars": SETTINGS.min_propositional_chars,
                     "threshold_sweep": list(SETTINGS.threshold_sweep),
                     "confidence": SETTINGS.confidence,
                     "ruleset_version": RULESET_VERSION},
        "column_map": ingest_meta.get("column_map"),
        "tables": ingest_meta.get("tables"),
        "counts": ingest_meta.get("counts"),
        "notes": notes,
    }


def _assert_releasable(manifest: dict) -> None:
    extra = set(manifest) - ALLOWED_KEYS
    if extra:
        raise ValueError(f"manifest carries keys outside the release policy: {sorted(extra)}")
    for name, entry in (manifest.get("tables") or {}).items():
        unexpected = set(entry) - {"n_records", "aggregate_sha256", "record_sha256"}
        if unexpected:
            raise ValueError(f"digest block for {name} carries {sorted(unexpected)}")
        for digest in entry["record_sha256"]:
            if not (len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)):
                raise ValueError(f"digest block for {name} contains a non-digest value")


def releasable_decisions(coded: pd.DataFrame) -> pd.DataFrame:
    """The coding decisions with every pupil-authored column removed.

    The determinism requirement is that a clean re-execution reproduces the
    reported figures without repeating any judgement. That is met by releasing
    the decisions and the entry digests, which together let a reader check the
    arithmetic without reading a child's writing.
    """
    drop = [c for c in FORBIDDEN_COLUMNS if c in coded.columns]
    out = coded.drop(columns=drop)
    leaked = [c for c in out.columns if c in FORBIDDEN_COLUMNS]
    if leaked:
        raise ValueError(f"release frame still carries {leaked}")
    return out


def write_manifest(manifest: dict, directory: Path | None = None) -> Path:
    _assert_releasable(manifest)
    directory = directory or PROVENANCE_DIR
    directory.mkdir(parents=True, exist_ok=True)

    tables = manifest.pop("tables")
    digests = {"schema": "argueaudit/digests/1",
               "tables": {k: {"n_records": v["n_records"],
                              "aggregate_sha256": v["aggregate_sha256"],
                              "record_sha256": v["record_sha256"]} for k, v in tables.items()}}
    (directory / "record_digests.json").write_text(
        json.dumps(digests, indent=1, sort_keys=True), encoding="utf-8")
    (directory / "column_map.json").write_text(
        json.dumps({"schema": "argueaudit/columnmap/1", "columns": manifest["column_map"]},
                   indent=1, sort_keys=True), encoding="utf-8")

    manifest["tables"] = {k: {"n_records": v["n_records"],
                              "aggregate_sha256": v["aggregate_sha256"]}
                          for k, v in tables.items()}
    path = directory / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    return path


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
