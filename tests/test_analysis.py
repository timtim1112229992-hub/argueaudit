# -*- coding: utf-8 -*-
"""Tests for the coding rules and the estimators.

The rules decide what counts as a written justification, so they are tested
against the cases that decide the paper's headline: an empty field, a field
holding a number, a sentence frame submitted unfilled, and a real warrant.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from argueaudit import models  # noqa: E402
from argueaudit.derive import mark_provenance, normalise  # noqa: E402
from argueaudit.lexicon import count_markers, mask_false_friends  # noqa: E402
from argueaudit.rubric import (ABSENT, ASSERTION, LINKED, LINKED_SPECIFIC, TOKEN,
                               code_a, code_b)  # noqa: E402


# ---------------------------------------------------------------- normalisation
def test_normalise_strips_whitespace_and_control_characters():
    assert normalise("  \u200b 我看到\u3000水 \n") == "我看到水"
    assert normalise("") == ""
    assert normalise("\u200b\u200b") == ""


# ---------------------------------------------------------------- ruleset A
@pytest.mark.parametrize("text,populated,substantive,level", [
    ("", False, False, ABSENT),
    ("123456", True, False, TOKEN),            # the entry the corpus actually holds
    ("。。。", True, False, TOKEN),
    ("好", True, False, TOKEN),                 # one character carries no proposition
    ("我看到1号土的水最先流出来", True, True, ASSERTION),
    ("因为1号土渗水快所以适合种西瓜", True, True, LINKED_SPECIFIC),
])
def test_ruleset_a(text, populated, substantive, level):
    d = code_a(text)
    assert (d.populated, d.substantive, d.level) == (populated, substantive, level)


def test_ruleset_a_rejects_an_unfilled_sentence_frame():
    frame = normalise("一开始我觉得____。通过实验，我看到____。现在我更确定____。")
    d = code_a(frame)
    assert d.populated and not d.substantive
    assert "slots unfilled" in d.reason


def test_ruleset_a_credits_a_frame_the_pupil_filled_in():
    filled = normalise("一开始我觉得1号土渗水快。通过实验，我看到水最先流出来。")
    assert code_a(filled).substantive


# ---------------------------------------------------------------- ruleset B
def test_ruleset_b_rejects_text_that_names_nothing():
    d = code_b("gayandlala")
    assert d.populated and not d.substantive
    assert "names nothing" in d.reason


def test_ruleset_b_credits_reference_to_the_task_world():
    assert code_b("3号土很黏").substantive


def test_the_two_rulesets_are_not_the_same_rule():
    """They must be able to disagree, or their agreement means nothing."""
    text = "我根据资料进行了反思"           # prose, but names nothing in the task
    assert code_a(text).substantive is True
    assert code_b(text).substantive is False


# ---------------------------------------------------------------- false friends
def test_connective_marker_is_not_matched_across_a_word_boundary():
    """验证 appears inside 实验证据 by accident and must not be counted."""
    assert "验证" in "我根据实验证据进行了反思"
    assert count_markers("我根据实验证据进行了反思", "connective") == 0
    assert count_markers("这验证了我的猜想", "connective") == 1


def test_masking_preserves_length():
    assert len(mask_false_friends("我根据实验证据反思")) == len("我根据实验证据反思")


# ---------------------------------------------------------------- estimators
def test_clopper_pearson_matches_known_values():
    lo, hi = models.clopper_pearson(0, 10)
    assert lo == 0.0
    assert hi == pytest.approx(0.30850, abs=1e-4)
    lo, hi = models.clopper_pearson(10, 10)
    assert hi == 1.0
    assert lo == pytest.approx(0.69150, abs=1e-4)


def test_rule_of_three_bound():
    assert models.rule_of_three(10) == pytest.approx(1 - 0.05 ** 0.1, abs=1e-12)
    assert models.rule_of_three(10) == pytest.approx(0.25893, abs=1e-4)


def test_exact_mcnemar_on_a_complete_split():
    # Ten groups completed one component and none completed the other.
    assert models._exact_mcnemar(0, 10) == pytest.approx(2 / 1024)
    assert models._exact_mcnemar(0, 0) == 1.0


def test_krippendorff_alpha_is_one_for_identical_varying_codings():
    a = [0, 1, 0, 1, 1, 0]
    assert models.krippendorff_alpha(a, a, "nominal") == pytest.approx(1.0, abs=1e-9)


def test_krippendorff_alpha_is_undefined_without_variation():
    """Two coders who both say 'yes' to everything have agreed on nothing."""
    assert np.isnan(models.krippendorff_alpha([1, 1, 1], [1, 1, 1], "nominal"))


# ---------------------------------------------------------------- provenance rule
def _frame(values, field="changeReason", stage=6):
    return pd.DataFrame([
        {"group_number": i + 1, "stage": stage, "field": field, "norm": v,
         "populated": bool(v), "substantive": bool(v)}
        for i, v in enumerate(values)])


def test_a_string_most_groups_share_is_marked_instrument_supplied():
    out = mark_provenance(_frame(["同一句"] * 9 + ["自己写的"]))
    assert out["field_instrument_supplied"].all()
    assert out["composed"].sum() == 1


def test_a_single_entry_is_not_instrument_supplied_by_arithmetic():
    """One populated entry has a modal share of one and tells us nothing."""
    out = mark_provenance(_frame([""] * 9 + ["只有一组写了理由"]))
    assert not out["field_instrument_supplied"].any()
    assert out["composed"].sum() == 1


def test_distinct_entries_are_all_treated_as_composed():
    out = mark_provenance(_frame([f"第{i}组的理由" for i in range(10)]))
    assert out["composed"].sum() == 10
