"""Runtime configuration and the field register.

The restricted corpus is located through the environment and its path is never
recorded in the repository. The field register below is the analytical spine of
the study: it records, for every stored field, what the interface asked for and
how it asked, so that completion can be read against response format rather than
against reasoning demand alone.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_DIR = REPO_ROOT / "synthetic"
PROVENANCE_DIR = REPO_ROOT / "provenance"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs"

DATA_DIR_ENV = "ARGUEAUDIT_DATA_DIR"
OUTPUT_DIR_ENV = "ARGUEAUDIT_OUTPUT_DIR"

TABLES = {
    "submissions": "03_stage_submissions.xlsx",
    "groups": "02_groups.xlsx",
    "progress": "08_group_progress.xlsx",
    "decisions": "04_agent_decisions.xlsx",
    "qa": "05_agent_qa_messages.xlsx",
}

# The last two describe the instructional setting rather than the argument, and
# a run without them still answers every research question. They are optional so
# that the synthetic corpus, which carries no agent log, remains runnable.
OPTIONAL_TABLES = ("decisions", "qa")

# Columns the analysis is permitted to read, recorded in the provenance package
# so a reader can see the extent of access without seeing any value.
COLUMN_MAP = {
    "submissions": ["id", "group_id", "stage", "stage_key", "is_completed",
                    "completion_rate", "completed_at", "updated_at", "form_data_json"],
    "groups": ["id", "group_number", "current_stage", "status", "created_at"],
    "progress": ["group_id", "group_number", "completed_stage_count", "avg_completion_pct"],
    # Only the structural columns. The agent's message and hint text is not read
    # at all, because the count of interventions answers the question and the
    # wording does not.
    "decisions": ["id", "group_id", "stage", "action", "trigger_source", "created_at"],
    "qa": ["id", "group_id", "stage", "on_topic", "source", "created_at"],
}

STAGE_KEY = {0: "pretest", 1: "observe", 2: "guess", 3: "sticky_exp",
             4: "drain_exp", 5: "evidence", 6: "conclusion"}

# Response format, taken from the instrument documentation rather than inferred
# from the data, so that the empirical distinctness measure remains an
# independent check on it rather than a restatement.
#   menu     : the pupil selects from presented options
#   exemplar : free text, with a completed model sentence displayed beside the box
#   frame    : a sentence frame containing slots the pupil is to fill
#   preset   : the field arrives carrying a value the pupil did not compose
#   open     : free text, with an instruction but no model sentence
#
# The preset and frame classifications are made on documentary grounds and then
# checked against the data: a field whose modal value occupies most of the cohort
# is not being composed, whatever the interface intended.
#
# Demand is the epistemic act the field calls for.
#   describe : report what was observed or done
#   claim    : commit to a conjecture
#   interpret: say what an observation might mean
#   justify  : relate evidence to a claim, or give grounds for one
FIELDS = (
    # stage, key, format, demand, analytic
    (1, "soilRecords",   "menu",     "describe",  True),
    (2, "guess",         "exemplar", "claim",     True),
    (2, "reason",        "open",     "justify",   True),
    (3, "finding",       "open",     "describe",  True),
    (3, "findingReason", "open",     "justify",   True),
    (4, "finding",       "open",     "describe",  True),
    (4, "guessCompare",  "open",     "justify",   True),
    (5, "seeBox",        "exemplar", "describe",  True),
    (5, "thinkBox",      "exemplar", "interpret", True),
    (5, "explain",       "open",     "justify",   True),
    (6, "after",         "frame",    "interpret", True),
    (6, "changeReason",  "preset",   "justify",   True),
)

# A field is reported as instrument-supplied when one string accounts for at
# least this share of its populated entries, and does so across at least this
# many groups. The count requirement matters: a field with a single entry has a
# modal share of one by arithmetic, which says nothing about where it came from.
PRESET_MODAL_SHARE = 0.8
PRESET_MIN_GROUPS = 3

# The three fields that carry the argument at the evidence stage. The component
# census in E1 is defined over exactly these.
COMPONENTS = (("observation", 5, "seeBox"),
              ("interpretation", 5, "thinkBox"),
              ("justification", 5, "explain"))

# Fields the analysis plan registered as predictors. Each is checked for variance
# before any model is fitted, and a constant field is reported as constant
# rather than entered into a test that cannot estimate anything.
CONTROL_FIELDS = ((5, "linkedEvidence"), (5, "evidenceStrength"), (5, "evidenceSource"))


@dataclass(frozen=True)
class Settings:
    """Analysis parameters. Every value here is written into the run manifest."""

    seed: int = 20260625
    analytic_groups: int = 10          # G11 and G12 were provisioned but unstaffed
    n_stages: int = 7
    bootstrap_replicates: int = 10000
    permutation_replicates: int = 10000

    # Substantive coding thresholds. The primary values are declared here before
    # the coding runs; the sensitivity sweep in M9 moves them.
    min_content_chars: int = 2         # content characters for a substantive entry
    min_propositional_chars: int = 4   # below this an entry cannot carry a proposition

    # The sweep for M9. Each pair is (min_content_chars, min_propositional_chars).
    threshold_sweep: tuple = ((1, 1), (1, 2), (2, 4), (3, 6), (4, 8), (6, 12))

    # A proportion resting on ten groups is reported with an exact interval, and
    # a zero count additionally with the rule-of-three bound.
    confidence: float = 0.95

    @property
    def data_dir(self) -> Path | None:
        raw = os.environ.get(DATA_DIR_ENV)
        return Path(raw) if raw else None

    @property
    def output_dir(self) -> Path:
        raw = os.environ.get(OUTPUT_DIR_ENV)
        return Path(raw) if raw else DEFAULT_OUTPUT_DIR

    @property
    def using_synthetic(self) -> bool:
        return self.data_dir is None


SETTINGS = Settings()
