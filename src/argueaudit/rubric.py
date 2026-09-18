"""Presence coding, substantive coding and the warrant rubric (P3, P4).

Two rulesets are applied. They are independent in construction, not merely in
execution: ruleset A decides from the character composition of an entry, ruleset
B decides from its marker content. Neither is derived from the other, so their
agreement is informative about the boundary being drawn rather than tautological.

They are rulesets, not people. This module does not claim human double coding,
and the agreement it reports is agreement between two operationalisations of the
same construct. Every decision is stored as data with the ruleset version that
produced it, so a clean re-execution reproduces the coding exactly.

Blinding is structural. A coder receives the entry text and nothing else: no
group identifier, no stored completion value, no field label and no indication
of which genre the interface expected.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import pandas as pd

from .config import SETTINGS
from .lexicon import (CONNECTIVE, INFERENTIAL, PERCEPTUAL, PROPERTY, RULESET_VERSION,
                      SAMPLE, mask_false_friends)

# Ordered warrant levels. The rubric is declared before coding runs and the
# threshold cannot be moved after the outcome is known except through the
# declared sensitivity sweep.
ABSENT, TOKEN, ASSERTION, LINKED, LINKED_SPECIFIC = 0, 1, 2, 3, 4
LEVEL_NAMES = {ABSENT: "absent", TOKEN: "token", ASSERTION: "assertion",
               LINKED: "linked", LINKED_SPECIFIC: "linked_specific"}

_DIGITS_ONLY = re.compile(r"^[0-9]+$")
_PUNCT_ONLY = re.compile(r"^[^\w\u4e00-\u9fff]+$")
_BLANK_SLOT = re.compile(r"[_＿]{2,}")

# Boilerplate a sentence frame supplies around its slots. An entry made only of
# frame and empty slots was submitted without being filled in, and crediting it
# would count the instrument's own words as the pupil's.
_FRAME_STEMS = ("一开始我觉得", "通过实验", "我发现", "我看到", "现在我更确定",
                "现在我觉得")


def _strip_frame(text: str) -> str:
    """What remains of an entry once the slots and the frame's own words are removed."""
    residue = _BLANK_SLOT.sub("", text)
    for stem in _FRAME_STEMS:
        residue = residue.replace(stem, "")
    return residue.strip("，。、,.!?！？ ")


def _han_count(text: str) -> int:
    return sum(1 for c in text if unicodedata.name(c, "").startswith("CJK"))


def _any(text: str, markers) -> bool:
    return any(m in mask_false_friends(text) for m in markers)


@dataclass(frozen=True)
class Decision:
    populated: bool
    substantive: bool
    level: int
    reason: str


# --------------------------------------------------------------------------
# Ruleset A: character composition
# --------------------------------------------------------------------------
def code_a(text: str, min_content: int | None = None,
           min_prop: int | None = None) -> Decision:
    """Decide from what the entry is made of.

    An entry that carries no Chinese characters cannot assert a proposition in
    this corpus, whatever its length. A bare numeric string is the clearest case:
    it occupies the field without saying anything.
    """
    min_content = SETTINGS.min_content_chars if min_content is None else min_content
    min_prop = SETTINGS.min_propositional_chars if min_prop is None else min_prop

    if not text:
        return Decision(False, False, ABSENT, "empty after normalisation")
    if _DIGITS_ONLY.match(text):
        return Decision(True, False, TOKEN, "numeric string, no propositional content")
    if _PUNCT_ONLY.match(text):
        return Decision(True, False, TOKEN, "punctuation only")
    if _BLANK_SLOT.search(text):
        residue = _strip_frame(text)
        if _han_count(residue) < min_content:
            return Decision(True, False, TOKEN,
                            "sentence frame submitted with its slots unfilled")

    han = _han_count(text)
    if han < min_content or len(text) < min_prop:
        return Decision(True, False, TOKEN, "below the propositional length threshold")

    if _any(text, CONNECTIVE):
        specific = _any(text, SAMPLE) and _any(text, PROPERTY)
        return Decision(True, True, LINKED_SPECIFIC if specific else LINKED,
                        "relates one proposition to another")
    return Decision(True, True, ASSERTION, "asserts a proposition without relating it")


# --------------------------------------------------------------------------
# Ruleset B: marker content
# --------------------------------------------------------------------------
def code_b(text: str, min_content: int | None = None,
           min_prop: int | None = None) -> Decision:
    """Decide from what the entry refers to.

    Content is credited when the entry names something in the task world, or
    performs an observable, inferential or connective move. An entry naming
    nothing and doing nothing is a token however many characters it runs to.
    """
    min_content = SETTINGS.min_content_chars if min_content is None else min_content
    min_prop = SETTINGS.min_propositional_chars if min_prop is None else min_prop

    if not text:
        return Decision(False, False, ABSENT, "empty after normalisation")

    # Judge a frame on what the pupil added to it, not on what it arrived with.
    subject = _strip_frame(text) if _BLANK_SLOT.search(text) else text
    if not subject:
        return Decision(True, False, TOKEN, "nothing written outside the supplied frame")

    refers = _any(subject, SAMPLE) or _any(subject, PROPERTY)
    acts = _any(subject, PERCEPTUAL) or _any(subject, INFERENTIAL) or _any(subject, CONNECTIVE)
    text = subject
    if not (refers or acts):
        return Decision(True, False, TOKEN, "names nothing and performs no epistemic move")
    if _han_count(text) < min_content or len(text) < min_prop:
        return Decision(True, False, TOKEN, "below the propositional length threshold")

    if _any(text, CONNECTIVE):
        return Decision(True, True, LINKED_SPECIFIC if (refers and acts) else LINKED,
                        "carries a connective relating evidence to a claim")
    return Decision(True, True, ASSERTION, "refers to the task world without connecting")


CODERS = {"A": code_a, "B": code_b}


def code_frame(frame: pd.DataFrame, min_content: int | None = None,
               min_prop: int | None = None) -> pd.DataFrame:
    """Apply both rulesets to every entry and store the decisions."""
    out = frame.copy()
    for name, coder in CODERS.items():
        decisions = [coder(t or "", min_content, min_prop) for t in out["norm"]]
        out[f"populated_{name}"] = [d.populated for d in decisions]
        out[f"substantive_{name}"] = [d.substantive for d in decisions]
        out[f"level_{name}"] = [d.level for d in decisions]
        out[f"reason_{name}"] = [d.reason for d in decisions]

    out["agree_substantive"] = out["substantive_A"] == out["substantive_B"]
    out["agree_level"] = out["level_A"] == out["level_B"]
    out["ruleset_version"] = RULESET_VERSION
    return out


def adjudicate(coded: pd.DataFrame) -> pd.DataFrame:
    """P4: resolve the entries on which the two rulesets differ.

    The rule is declared rather than discretionary: where the rulesets disagree
    on substance, the stricter decision stands, because the study's claim is that
    a component is absent and the conservative direction for that claim is to
    credit substance wherever either ruleset finds it. Taking the stricter
    reading makes the headline harder to obtain, not easier.
    """
    out = coded.copy()
    out["substantive"] = out["substantive_A"] & out["substantive_B"]
    out["level"] = out[["level_A", "level_B"]].min(axis=1)
    out["populated"] = out["populated_A"]
    out["adjudicated"] = ~out["agree_substantive"] | ~out["agree_level"]
    out["adjudication_rule"] = out["adjudicated"].map(
        {True: "rulesets differed, stricter decision retained", False: ""})
    unresolved = out["substantive"].isna().sum() + out["level"].isna().sum()
    out.attrs["unresolved"] = int(unresolved)
    return out
