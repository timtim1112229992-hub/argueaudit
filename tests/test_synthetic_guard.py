# -*- coding: utf-8 -*-
"""The synthetic corpus must not be able to pass itself off as the real one.

The corpus directory is supplied through an environment variable, and its
absence is deliberately not an error, because the package has to stay runnable
by someone who will never hold the restricted data. The cost of that choice is
that a shell which has lost the variable will run the whole pipeline against
invented records and write over the real output, announcing the substitution in
one field that nobody reads twice. These tests pin the two places where that
would do lasting damage: overwriting results computed from the corpus, and
publishing a provenance package that describes records which do not exist.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from argueaudit import pipeline                          # noqa: E402
from argueaudit.config import DATA_DIR_ENV               # noqa: E402
from argueaudit.ingest import load                       # noqa: E402


def _held(tmp_path: Path, source: str) -> None:
    (tmp_path / "results.json").write_text(
        json.dumps({"corpus": {"source": source}}), encoding="utf-8")


def test_a_run_without_the_variable_reads_the_synthetic_corpus(monkeypatch):
    monkeypatch.delenv(DATA_DIR_ENV, raising=False)
    corpus = load()
    assert corpus.meta["source"] == "synthetic"


def test_the_synthetic_corpus_carries_no_agent_log(monkeypatch):
    """Its absence is why the instructional context model reports non-estimable."""
    monkeypatch.delenv(DATA_DIR_ENV, raising=False)
    corpus = load()
    assert not corpus.extra


def test_synthetic_run_refuses_to_overwrite_restricted_results(tmp_path):
    _held(tmp_path, "restricted")
    with pytest.raises(SystemExit) as excinfo:
        pipeline._refuse_synthetic_overwrite(tmp_path, "synthetic", allow=False)
    assert DATA_DIR_ENV in str(excinfo.value)


def test_the_refusal_can_be_overridden_deliberately(tmp_path):
    _held(tmp_path, "restricted")
    pipeline._refuse_synthetic_overwrite(tmp_path, "synthetic", allow=True)


def test_a_restricted_run_is_never_refused(tmp_path):
    _held(tmp_path, "restricted")
    pipeline._refuse_synthetic_overwrite(tmp_path, "restricted", allow=False)


def test_a_synthetic_run_over_synthetic_results_is_never_refused(tmp_path):
    _held(tmp_path, "synthetic")
    pipeline._refuse_synthetic_overwrite(tmp_path, "synthetic", allow=False)


def test_an_empty_output_directory_is_never_refused(tmp_path):
    pipeline._refuse_synthetic_overwrite(tmp_path, "synthetic", allow=False)


def test_unreadable_previous_results_do_not_block_a_run(tmp_path):
    (tmp_path / "results.json").write_text("{ not json", encoding="utf-8")
    pipeline._refuse_synthetic_overwrite(tmp_path, "synthetic", allow=False)


def test_publishing_provenance_from_synthetic_data_is_refused(tmp_path, monkeypatch):
    monkeypatch.delenv(DATA_DIR_ENV, raising=False)
    before = (ROOT / "provenance" / "run_manifest.json").read_text(encoding="utf-8")
    with pytest.raises(SystemExit) as excinfo:
        pipeline.run(output_dir=tmp_path, publish_provenance=True)
    assert DATA_DIR_ENV in str(excinfo.value)
    after = (ROOT / "provenance" / "run_manifest.json").read_text(encoding="utf-8")
    assert before == after, "the refusal came too late and the manifest was rewritten"


def test_the_published_manifest_describes_the_restricted_corpus():
    """What is committed under provenance/ must not be a synthetic run's output."""
    manifest = json.loads(
        (ROOT / "provenance" / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["source"] == "restricted"
