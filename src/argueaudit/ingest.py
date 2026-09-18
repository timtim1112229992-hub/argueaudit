"""Reading the archived snapshot and fixing the analysed record set (D1, D3, D7).

The digest is taken over the records as read, before any transformation, so that
the register identifies the same artefacts a replication would draw. Row counts
are carried forward at every step, which is how D7 is satisfied: no entry is
dropped without the drop appearing in the count trail.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import COLUMN_MAP, OPTIONAL_TABLES, SETTINGS, SYNTHETIC_DIR, TABLES


@dataclass
class Corpus:
    submissions: pd.DataFrame
    groups: pd.DataFrame
    progress: pd.DataFrame
    meta: dict
    # Optional tables describing the instructional setting; absent when the run
    # is against the synthetic corpus.
    extra: dict = None

    @property
    def counts(self) -> dict:
        return self.meta["counts"]


def _digest_frame(df: pd.DataFrame) -> tuple[str, list[str]]:
    """A digest per record and one over the set, computed on a canonical form."""
    per = []
    for _, row in df.iterrows():
        payload = json.dumps({k: ("" if pd.isna(v) else str(v)) for k, v in row.items()},
                             ensure_ascii=False, sort_keys=True)
        per.append(hashlib.sha256(payload.encode("utf-8")).hexdigest())
    aggregate = hashlib.sha256("".join(sorted(per)).encode("utf-8")).hexdigest()
    return aggregate, per


def _resolve(directory: Path, filename: str) -> Path | None:
    """Find a table under either of the two suffixes the project uses.

    The restricted corpus arrives as workbooks. The synthetic stand-in ships as
    CSV instead, because a file that is published should be readable in a diff,
    and a binary blob in a repository is something reviewers take on trust.
    Resolving the suffix here keeps that decision out of the rest of the code.
    """
    path = Path(directory) / filename
    if path.exists():
        return path
    alternative = path.with_suffix(".csv")
    return alternative if alternative.exists() else None


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.suffix == ".csv" else pd.read_excel(path)


def load(data_dir: Path | None = None) -> Corpus:
    directory = data_dir or SETTINGS.data_dir or SYNTHETIC_DIR
    synthetic = (data_dir or SETTINGS.data_dir) is None

    frames, digests, counts = {}, {}, {}
    for name, filename in TABLES.items():
        path = _resolve(directory, filename)
        if path is None and name in OPTIONAL_TABLES:
            counts[f"{name}.read"] = 0
            continue
        if path is None:
            raise FileNotFoundError(
                f"{filename} not found in {directory}. Set ARGUEAUDIT_DATA_DIR to the "
                f"restricted corpus, or generate the synthetic corpus first.")
        df = _read(path)
        keep = [c for c in COLUMN_MAP[name] if c in df.columns]
        missing = set(COLUMN_MAP[name]) - set(df.columns)
        df = df[keep]
        counts[f"{name}.read"] = len(df)
        aggregate, per = _digest_frame(df)
        digests[name] = {"n_records": len(df), "aggregate_sha256": aggregate,
                         "record_sha256": per}
        if missing:
            digests[name]["n_records"] = len(df)
        frames[name] = df

    groups = frames["groups"].rename(columns={"id": "group_id"})
    sub = frames["submissions"].merge(groups[["group_id", "group_number"]],
                                      on="group_id", how="left")
    counts["submissions.joined"] = len(sub)
    unmatched = int(sub["group_number"].isna().sum())

    # D3: the ten staffed groups and the seven task stages.
    analytic = sub[sub["group_number"].between(1, SETTINGS.analytic_groups)].copy()
    analytic = analytic[analytic["stage"].between(0, SETTINGS.n_stages - 1)]
    analytic["group_number"] = analytic["group_number"].astype(int)
    analytic["stage"] = analytic["stage"].astype(int)
    analytic = analytic.sort_values(["group_number", "stage"]).reset_index(drop=True)
    counts["submissions.analytic"] = len(analytic)
    counts["submissions.excluded_reserve"] = len(sub) - len(analytic)

    meta = {"source": "synthetic" if synthetic else "restricted",
            "column_map": COLUMN_MAP,
            "tables": digests,
            "counts": counts,
            "unmatched_group_ids": unmatched}

    # Restrict the setting tables to the same ten groups, so every count the
    # paper reports is drawn from the same record set as the argument coding.
    staffed = set(groups.loc[groups["group_number"].between(1, SETTINGS.analytic_groups),
                             "group_id"])
    extra = {}
    for name in OPTIONAL_TABLES:
        if name not in frames:
            continue
        df = frames[name]
        kept = df[df["group_id"].isin(staffed)].copy()
        counts[f"{name}.analytic"] = len(kept)
        extra[name] = kept

    return Corpus(submissions=analytic, groups=groups,
                  progress=frames["progress"], meta=meta, extra=extra)
