"""Expanding the stored payload into a field-level frame (D2, D4, D6).

One row per group, per stage, per registered field. The unmodified string is
carried beside the normalised one, because presence is tested on the normalised
form while substantive coding must see what was actually written. Linguistic
features are computed here so that coding and modelling both draw on the same
measured quantities rather than recomputing them independently.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import re
import unicodedata

import pandas as pd

from .config import (CONTROL_FIELDS, FIELDS, PRESET_MIN_GROUPS, PRESET_MODAL_SHARE,
                     SETTINGS, STAGE_KEY)
from .lexicon import EXEMPLARS, MARKER_SETS, count_markers

# Characters that carry no content but can make a field look populated.
_CONTROL = re.compile(r"[\u0000-\u001f\u007f\u200b-\u200f\ufeff]")
_SPACE = re.compile(r"[\s\u3000]+")

# A run of underscores is the slot a sentence frame leaves for the pupil to fill.
# An entry submitted with the run intact is a frame that was never completed, and
# the stored completion value counts it as done.
_BLANK_SLOT = re.compile(r"[_＿]{2,}")


def parse_form(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def as_text(value) -> str:
    """Render a stored value as the text a reader would have seen."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(value, ensure_ascii=False)


def normalise(text: str) -> str:
    """D4: whitespace and control characters removed for presence testing."""
    return _SPACE.sub("", _CONTROL.sub("", text)).strip()


def _char_profile(text: str) -> dict:
    han = digit = latin = punct = 0
    for ch in text:
        if unicodedata.name(ch, "").startswith("CJK"):
            han += 1
        elif ch.isdigit():
            digit += 1
        elif ch.isalpha():
            latin += 1
        elif not ch.isspace():
            punct += 1
    return {"n_han": han, "n_digit": digit, "n_latin": latin, "n_punct": punct}


def _similarity(text: str, exemplar: str | None) -> float | None:
    if not exemplar or not text:
        return None
    return difflib.SequenceMatcher(None, text, exemplar).ratio()


def field_frame(submissions: pd.DataFrame) -> pd.DataFrame:
    """D2: one row per group, stage and registered field."""
    rows = []
    for _, rec in submissions.iterrows():
        form = parse_form(rec.get("form_data_json"))
        for stage, key, fmt, demand, analytic in FIELDS:
            if int(rec["stage"]) != stage:
                continue
            raw = as_text(form.get(key))
            norm = normalise(raw)
            exemplar = EXEMPLARS.get((stage, key))
            row = {
                "group_number": int(rec["group_number"]),
                "stage": stage,
                "stage_key": STAGE_KEY[stage],
                "field": key,
                "response_format": fmt,
                "demand": demand,
                "analytic": analytic,
                "raw": raw,
                "norm": norm,
                "populated": bool(norm),
                "n_chars": len(norm),
                "entry_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                "has_exemplar": exemplar is not None,
                "exemplar_similarity": _similarity(norm, exemplar),
                "n_blank_slots": len(_BLANK_SLOT.findall(norm)),
                "n_chars_outside_slots": len(_BLANK_SLOT.sub("", norm)),
            }
            row.update(_char_profile(norm))
            for which in MARKER_SETS:
                row[f"mk_{which}"] = count_markers(norm, which)
            # Type-token ratio over characters, which is the available unit for
            # short unsegmented Chinese entries.
            row["ttr"] = (len(set(norm)) / len(norm)) if norm else None
            rows.append(row)

    frame = pd.DataFrame(rows)
    expected = len(submissions["group_number"].unique()) * len(FIELDS)
    frame.attrs["expected_rows"] = expected
    return frame


def control_frame(submissions: pd.DataFrame) -> pd.DataFrame:
    """The fields registered as predictors in the plan, with their variance."""
    rows = []
    for _, rec in submissions.iterrows():
        form = parse_form(rec.get("form_data_json"))
        for stage, key in CONTROL_FIELDS:
            if int(rec["stage"]) != stage:
                continue
            rows.append({"group_number": int(rec["group_number"]), "stage": stage,
                         "field": key, "value": normalise(as_text(form.get(key)))})
    return pd.DataFrame(rows)


def stage_frame(submissions: pd.DataFrame) -> pd.DataFrame:
    """Stored completion per group and stage, for the stage-position control."""
    out = submissions[["group_number", "stage", "completion_rate", "is_completed"]].copy()
    out["stage_key"] = out["stage"].map(STAGE_KEY)
    return out.sort_values(["group_number", "stage"]).reset_index(drop=True)


def durations(submissions: pd.DataFrame) -> pd.DataFrame:
    """Time spent on each stage, derived from consecutive completion stamps.

    The archive records when a stage was completed but not when it was entered,
    so duration is the interval since the previous stage completed. The first
    stage has no predecessor and is left missing rather than assigned the
    session start, which would silently attribute joining time to task time.
    """
    out = submissions[["group_number", "stage", "completed_at", "updated_at"]].copy()
    stamp = pd.to_datetime(out["completed_at"], errors="coerce", utc=True)
    out["stamp"] = stamp.fillna(pd.to_datetime(out["updated_at"], errors="coerce", utc=True))
    out = out.sort_values(["group_number", "stage"])
    out["duration_s"] = (out.groupby("group_number")["stamp"].diff()
                         .dt.total_seconds())
    out.loc[out["duration_s"] <= 0, "duration_s"] = pd.NA
    return out[["group_number", "stage", "duration_s"]].reset_index(drop=True)


def mark_provenance(frame: pd.DataFrame) -> pd.DataFrame:
    """Flag entries the pupils did not compose.

    Coding stays blind: a coder sees one entry and judges its text. Provenance is
    a different question, answerable only across the cohort, and it is settled
    here. Where most groups submitted the identical string, that string came from
    the instrument, and crediting it as pupil content would record the software's
    words as a child's.
    """
    out = frame.copy()
    out["is_modal_value"] = False
    out["field_instrument_supplied"] = False
    for (stage, fieldname), grp in out.groupby(["stage", "field"]):
        populated = grp[grp["populated"]]
        if populated.empty:
            continue
        counts = populated["norm"].value_counts()
        modal_value, modal_n = counts.index[0], counts.iloc[0]
        share = modal_n / len(populated)
        supplied = bool(share >= PRESET_MODAL_SHARE and modal_n >= PRESET_MIN_GROUPS)
        idx = grp.index
        out.loc[idx, "field_instrument_supplied"] = supplied
        out.loc[idx, "is_modal_value"] = grp["norm"].eq(modal_value) & grp["populated"]
    # Composed content: written by the pupils, carrying a proposition, and not the
    # string the instrument put in front of every group.
    if "substantive" in out.columns:
        out["composed"] = (out["substantive"]
                           & ~(out["field_instrument_supplied"] & out["is_modal_value"]))
    return out


def distinctness(frame: pd.DataFrame) -> pd.DataFrame:
    """How many distinct entries ten independent groups produced per field.

    A low ratio means the entries are being reproduced rather than composed,
    which is the measure that distinguishes a field pupils wrote from a field
    they copied out of the interface.
    """
    rows = []
    for (stage, fieldname), grp in frame.groupby(["stage", "field"], sort=True):
        populated = grp[grp["populated"]]
        n = len(populated)
        distinct = populated["norm"].nunique()
        sim = populated["exemplar_similarity"].dropna()
        modal_n = int(populated["norm"].value_counts().iloc[0]) if n else 0
        modal_share = (modal_n / n) if n else None
        rows.append({
            "stage": stage, "field": fieldname,
            "response_format": grp["response_format"].iloc[0],
            "demand": grp["demand"].iloc[0],
            "n_populated": n, "n_distinct": distinct,
            "distinctness": (distinct / n) if n else None,
            "modal_share": modal_share,
            "instrument_supplied": bool(modal_share is not None
                                        and modal_share >= PRESET_MODAL_SHARE
                                        and modal_n >= PRESET_MIN_GROUPS),
            "n_with_unfilled_slot": int((populated["n_blank_slots"] > 0).sum()),
            "mean_chars": populated["n_chars"].mean() if n else None,
            "mean_exemplar_similarity": sim.mean() if len(sim) else None,
        })
    return pd.DataFrame(rows).sort_values(["stage", "field"]).reset_index(drop=True)
