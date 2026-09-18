"""Marker inventories, instrument exemplars and the coding ruleset version.

The corpus is Simplified Chinese written by nine and ten year olds, so markers
are matched as substrings rather than through a segmenter: a segmenter trained
on adult prose introduces its own error into short, irregular pupil text, and the
quantities of interest here are marker presence and count, not syntax.

Every list below is instrument or language material. No pupil text appears in
this module or anywhere else in the repository.
"""
from __future__ import annotations

# Bump when any rule affecting a coding decision changes. Written into the run
# manifest so a stored decision can always be traced to the rules that produced it.
RULESET_VERSION = "argueaudit/rules/1.0.0"

# Sentences the interface displayed beside the free-text boxes, transcribed from
# the system workflow documentation. These are instrument text.
EXEMPLARS = {
    (2, "guess"): "我觉得3号土渗水更快",
    (5, "seeBox"): "我看到1号土的水最先流出来",
    (5, "thinkBox"): "这可能说明1号土渗水性强",
}

# Perceptual and report markers: the language of stating what was observed.
PERCEPTUAL = ("看到", "看见", "观察", "发现", "摸", "闻", "流出", "流得", "倒",
              "捏", "尝试", "试验", "变", "出来", "最先", "最后", "最快", "最慢")

# Matching substrings in unsegmented Chinese will occasionally catch a marker
# that straddles a word boundary and means nothing there. The clearest case in
# this corpus is 实验证据, a noun phrase for experimental evidence, inside which
# the connective 验证 appears by accident. Such spans are masked before counting.
FALSE_FRIENDS = ("实验证据",)


def mask_false_friends(text: str) -> str:
    for span in FALSE_FRIENDS:
        text = text.replace(span, "\u3000" * len(span))
    return text

# Inferential and modal markers: the language of saying what an observation means.
INFERENTIAL = ("可能", "说明", "也许", "大概", "应该", "表明", "证明", "我觉得",
               "我们觉得", "认为", "猜", "估计", "会是", "代表", "意味")

# Connective markers: the language of relating one proposition to another, which
# is what a warrant requires and what the justification fields asked for.
CONNECTIVE = ("因为", "所以", "由于", "因此", "支持", "证明了", "符合", "一致",
              "说明了", "验证", "对应", "和猜想", "跟猜想", "与猜想", "这说明")

# Specificity markers: reference to a particular sample or a particular property.
SAMPLE = ("1号", "2号", "3号", "4号", "一号", "二号", "三号", "四号")
PROPERTY = ("渗水", "黏", "粘", "沙", "壤", "颗粒", "粗", "细", "颜色", "气味",
            "成团", "保水", "透水", "吸水")

MARKER_SETS = {"perceptual": PERCEPTUAL, "inferential": INFERENTIAL,
               "connective": CONNECTIVE, "sample": SAMPLE, "property": PROPERTY}


def count_markers(text: str, which: str) -> int:
    """Total occurrences of any marker in the named set."""
    text = mask_false_friends(text)
    return sum(text.count(m) for m in MARKER_SETS[which])


def has_marker(text: str, which: str) -> bool:
    text = mask_false_friends(text)
    return any(m in text for m in MARKER_SETS[which])
