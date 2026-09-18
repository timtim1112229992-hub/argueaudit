# -*- coding: utf-8 -*-
"""Tests that stop pupil writing leaving this machine.

The custody undertaking is that only the column map, entry digests, coding
decisions and run manifest are releasable. These tests fail if any artefact the
pipeline writes carries a child's words, and they fail loudly rather than
warning, because a warning in a test suite is a leak with a note attached.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from argueaudit.provenance import (ALLOWED_KEYS, FORBIDDEN_COLUMNS,  # noqa: E402
                                   _assert_releasable, releasable_decisions)

OUTPUTS = ROOT / "outputs"
PROVENANCE = ROOT / "provenance"

# Any run of Han characters long enough to be a sentence a pupil wrote.
HAN_RUN = re.compile(r"[\u4e00-\u9fff]{4,}")

# Han strings the repository is allowed to contain: instrument text and markers,
# which are the software's words, not the children's.
ALLOWED_HAN = (
    "\u6211\u770b\u5230\u0031\u53f7\u571f\u7684\u6c34\u6700\u5148\u6d41\u51fa\u6765",
    "\u8fd9\u53ef\u80fd\u8bf4\u660e\u0031\u53f7\u571f\u6e17\u6c34\u6027\u5f3a",
    "\u6211\u89c9\u5f97\u0033\u53f7\u571f\u6e17\u6c34\u66f4\u5feb",
)


def _permitted(run: str) -> bool:
    """Instrument text, or a fragment of it.

    The exemplars carry digits, so a run of Han characters extracted from one is
    a fragment rather than a whole match. Matching on substrings is what keeps
    the check honest without making it vacuous.
    """
    return any(run in allowed for allowed in ALLOWED_HAN)


def test_release_frame_drops_every_pupil_text_column():
    frame = pd.DataFrame({"raw": ["\u6211\u770b\u5230\u6c34"],
                          "norm": ["\u6211\u770b\u5230\u6c34"],
                          "entry_sha256": ["a" * 64], "substantive": [True]})
    out = releasable_decisions(frame)
    for column in FORBIDDEN_COLUMNS:
        assert column not in out.columns
    assert "entry_sha256" in out.columns


@pytest.mark.skipif(not (OUTPUTS / "coding_decisions.csv").exists(),
                    reason="pipeline has not been run")
def test_written_decisions_carry_no_entry_text():
    frame = pd.read_csv(OUTPUTS / "coding_decisions.csv")
    for column in FORBIDDEN_COLUMNS:
        assert column not in frame.columns, f"{column} reached the written decisions"
    blob = frame.astype(str).to_csv(index=False)
    leaked = [m for m in HAN_RUN.findall(blob) if not _permitted(m)]
    assert not leaked, f"pupil text in coding_decisions.csv: {leaked[:5]}"


@pytest.mark.skipif(not (OUTPUTS / "results.json").exists(),
                    reason="pipeline has not been run")
def test_results_file_carries_no_entry_text():
    blob = (OUTPUTS / "results.json").read_text(encoding="utf-8")
    leaked = [m for m in HAN_RUN.findall(blob) if not _permitted(m)]
    assert not leaked, f"pupil text in results.json: {leaked[:5]}"


@pytest.mark.skipif(not (PROVENANCE / "run_manifest.json").exists(),
                    reason="pipeline has not been run")
def test_manifest_holds_only_permitted_keys():
    manifest = json.loads((PROVENANCE / "run_manifest.json").read_text(encoding="utf-8"))
    assert set(manifest) <= ALLOWED_KEYS
    blob = json.dumps(manifest, ensure_ascii=False)
    assert not [m for m in HAN_RUN.findall(blob) if not _permitted(m)]
    # The path to the restricted corpus is not a disclosure the manifest makes.
    assert "original_data" not in blob


@pytest.mark.skipif(not (PROVENANCE / "record_digests.json").exists(),
                    reason="pipeline has not been run")
def test_digest_register_is_digests_and_nothing_else():
    register = json.loads((PROVENANCE / "record_digests.json").read_text(encoding="utf-8"))
    for name, entry in register["tables"].items():
        assert set(entry) == {"n_records", "aggregate_sha256", "record_sha256"}
        assert len(entry["record_sha256"]) == entry["n_records"]
        for digest in entry["record_sha256"]:
            assert re.fullmatch(r"[0-9a-f]{64}", digest), f"{name} holds a non-digest"


def test_manifest_writer_refuses_an_unexpected_key():
    with pytest.raises(ValueError, match="outside the release policy"):
        _assert_releasable({"schema": "x", "entries": ["\u6211\u770b\u5230\u6c34"]})


def test_manifest_writer_refuses_a_digest_block_that_is_not_digests():
    bad = {"schema": "x", "tables": {"submissions": {
        "n_records": 1, "aggregate_sha256": "a" * 64,
        "record_sha256": ["\u6211\u770b\u5230\u6c34"]}}}
    with pytest.raises(ValueError, match="non-digest"):
        _assert_releasable(bad)


def _tracked() -> list[str] | None:
    """Paths git is tracking, or None where this is not a working repository."""
    try:
        out = subprocess.run(("git", "ls-files"), cwd=ROOT, capture_output=True,
                             text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.split() if out.returncode == 0 else None


def test_no_excluded_file_is_tracked():
    """The ignore rules are a policy; this is the check that they were obeyed.

    A file already tracked before its pattern was added to .gitignore stays
    tracked and is pushed on the next commit without complaint, which is the
    ordinary way data reaches a public repository. Reading the index directly
    catches that, where reading the ignore file cannot.
    """
    tracked = _tracked()
    if tracked is None:
        pytest.skip("not a git working tree")

    forbidden_dirs = ("outputs/", "data/", "raw/", "raw_data/", "original_data/",
                      "results/", "figures/", "artefacts/", "artifacts/", "logs/")
    forbidden_types = (".xlsx", ".xls", ".csv", ".png", ".eps", ".pdf", ".docx",
                       ".pptx", ".parquet", ".pkl", ".sqlite", ".npy", ".h5")

    offenders = []
    for path in tracked:
        if path.startswith(forbidden_dirs):
            offenders.append(path)
        elif path.endswith(forbidden_types) and not path.startswith("synthetic/"):
            offenders.append(path)
    assert not offenders, f"excluded files are tracked: {offenders}"


def test_only_csv_is_tracked_under_the_synthetic_corpus():
    """The one carve-out in the ignore rules, held to its stated width."""
    tracked = _tracked()
    if tracked is None:
        pytest.skip("not a git working tree")
    stray = [p for p in tracked
             if p.startswith("synthetic/") and not p.endswith(".csv")]
    assert not stray, f"the synthetic exception is being used for {stray}"


def test_released_text_names_no_author_and_no_working_label():
    """R13, R20 and R26, checked rather than remembered.

    Attribution travels with the paper. An archive that names a contributor, or
    that carries the working label or a finding from an unpublished manuscript,
    defeats anonymised review for everyone on it.
    """
    banned = re.compile(
        r"Paper ?[123][ab]\b|orcid|@[\w.-]+\.(?:com|edu|ac|org)|"
        r"\bHuang ?Hai\b|\bLai\b|Absent Justification|Reasoning Bridge",
        re.IGNORECASE)
    checked = 0
    for path in list(ROOT.rglob("*.py")) + list(ROOT.rglob("*.md")) + \
            list(ROOT.rglob("*.toml")) + list(PROVENANCE.rglob("*.json")):
        if "__pycache__" in path.parts or ".git" in path.parts:
            continue
        if OUTPUTS in path.parents:
            continue
        # This file has to spell the banned strings out in order to look for
        # them, so scanning it would report itself and nothing else.
        if path.resolve() == Path(__file__).resolve():
            continue
        checked += 1
        hit = banned.search(path.read_text(encoding="utf-8", errors="replace"))
        assert not hit, f"{path.relative_to(ROOT)} carries {hit.group(0)!r}"
    assert checked > 10, "the scan found almost nothing and is probably misdirected"


def test_no_source_file_contains_pupil_text():
    """The lexicons and exemplars are instrument text; nothing else may be Han."""
    for path in (ROOT / "src").rglob("*.py"):
        blob = path.read_text(encoding="utf-8")
        for run in HAN_RUN.findall(blob):
            assert _permitted(run) or len(run) <= 8, (
                f"{path.name} carries a long Han string: {run}")


def _messages() -> list[str] | None:
    """Every commit message in the repository, subject and body."""
    # A textual separator rather than a null byte, which this platform refuses
    # to carry through an argument list.
    mark = "-----end-of-commit-message-----"
    try:
        out = subprocess.run(["git", "log", f"--format=%s%n%b%n{mark}"], cwd=ROOT,
                             capture_output=True, text=True, encoding="utf-8",
                             errors="replace", check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return [m.strip() for m in out.stdout.split(mark) if m.strip()]


def test_no_commit_credits_a_tool_as_an_author():
    """A commit trailer is the one place an automated editor signs its work.

    Version control writes the trailer without being asked, it renders on the
    forge as co-authorship, and it survives every later check that looks only at
    the working tree. Tooling does not become a contributor by having been
    present, so the history is inspected rather than trusted.
    """
    messages = _messages()
    if messages is None:
        pytest.skip("not a git working tree")
    offenders = [m.splitlines()[0] for m in messages
                 if re.search(r"^\s*co-authored-by\s*:", m, re.I | re.M)]
    assert not offenders, f"commits credit a co-author: {offenders}"


def test_every_commit_carries_the_account_of_record():
    """One identity across the history, and it discloses nobody.

    A single commit made under a personal name or an institutional address
    undoes anonymised review for the whole archive, and it is the sort of thing
    that happens when a machine's global configuration is picked up silently.
    """
    try:
        out = subprocess.run(["git", "log", "--format=%an <%ae>|%cn <%ce>"],
                             cwd=ROOT, capture_output=True, text=True,
                             encoding="utf-8", errors="replace", check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("not a git working tree")
    identities = {part for line in out.stdout.splitlines() if line.strip()
                  for part in line.split("|")}
    assert len(identities) == 1, f"the history carries several identities: {identities}"
    only = identities.pop()
    assert "@users.noreply." not in only, (
        "the forge substituted a private address for the account of record, so "
        f"the history does not match the declared one: {only}")


def test_no_commit_message_carries_a_measurement():
    """A message is repository content, and findings do not belong in it.

    Every other guard here reads the working tree, which is exactly what a
    commit message is not. A quantity from the restricted session recorded in a
    message is published the moment the branch is pushed and cannot be recalled,
    so messages are held to describing the change and not its result.
    """
    messages = _messages()
    if messages is None:
        pytest.skip("not a git working tree")
    # Three digits is past any version number or exit code a message needs, and
    # a decimal fraction in a message is almost always an estimate.
    measurement = re.compile(r"(?<![\w.])\d{3,}(?![\w.])|\d+\.\d+")
    offenders = []
    for message in messages:
        hit = measurement.search(message)
        if hit:
            offenders.append(f"{message.splitlines()[0]!r} -> {hit.group(0)!r}")
    assert not offenders, f"commit messages carry measurements: {offenders}"
