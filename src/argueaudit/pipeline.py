"""Run order P1 to P10, and the completion gate G1 to G6.

The gate is evaluated by the code rather than asserted in prose. A gate that
fails is reported as failed and the reason is carried into the results file, so
that the drafting decision rests on the pipeline's own account of itself.
"""
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import models
from .config import COMPONENTS, DATA_DIR_ENV, SETTINGS
from .derive import (control_frame, distinctness, durations, field_frame,
                     mark_provenance, stage_frame)
from .ingest import load
from .provenance import build_manifest, releasable_decisions, write_manifest
from .rubric import adjudicate, code_frame


def _jsonable(obj):
    if isinstance(obj, models.Result):
        return {"name": obj.name, "estimand": obj.estimand, "estimable": obj.estimable,
                "note": obj.note, "values": _jsonable(obj.values)}
    if is_dataclass(obj):
        return _jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return None if np.isnan(value) else value
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    if isinstance(obj, pd.DataFrame):
        return _jsonable(obj.to_dict(orient="records"))
    return obj


def _refuse_synthetic_overwrite(out_dir: Path, source: str, allow: bool) -> None:
    """Stop a run on made-up data replacing results computed from the corpus.

    The corpus directory is supplied through an environment variable, and its
    absence is deliberately not an error: the package has to stay runnable by
    someone who will never hold the restricted data. The cost of that choice is
    that any shell which has lost the variable will run the whole pipeline
    against the synthetic stand-in and write over the real output, announcing
    the substitution in a single field that nobody reads twice. This refuses.
    """
    if source != "synthetic" or allow:
        return
    previous = out_dir / "results.json"
    if not previous.exists():
        return
    try:
        held = json.loads(previous.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return
    if (held.get("corpus") or {}).get("source") != "restricted":
        return
    raise SystemExit(
        f"refusing to overwrite results computed from the restricted corpus with a "
        f"synthetic run. Set {DATA_DIR_ENV} to the corpus, choose another output "
        f"directory, or pass allow_overwrite=True if this is what you intend.")


def run(data_dir: Path | None = None, output_dir: Path | None = None,
        publish_provenance: bool | None = None, allow_overwrite: bool = False) -> dict:
    """Execute P1 to P10.

    `publish_provenance` defaults to publishing only when the run read the
    restricted corpus. A provenance package is a claim about which records were
    analysed, so one describing invented records is worse than none at all.
    """
    out_dir = Path(output_dir or SETTINGS.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SETTINGS.seed)

    # P1 ingest -------------------------------------------------------------
    corpus = load(data_dir)
    source = corpus.meta["source"]
    _refuse_synthetic_overwrite(out_dir, source, allow_overwrite)
    if publish_provenance and source == "synthetic":
        raise SystemExit(
            f"refusing to publish a provenance package describing synthetic records. "
            f"Set {DATA_DIR_ENV} to the restricted corpus first.")

    # P2 derive -------------------------------------------------------------
    fields = field_frame(corpus.submissions)
    controls = control_frame(corpus.submissions)
    stages = stage_frame(corpus.submissions)
    stage_durations = durations(corpus.submissions)
    dist = distinctness(fields)
    corpus.meta["counts"]["fields.derived"] = len(fields)
    corpus.meta["counts"]["fields.expected"] = fields.attrs["expected_rows"]

    # P3 code, P4 adjudicate ------------------------------------------------
    coded = mark_provenance(adjudicate(code_frame(fields)))
    corpus.meta["counts"]["entries.coded"] = len(coded)
    corpus.meta["counts"]["entries.instrument_supplied"] = int(
        (coded["field_instrument_supplied"] & coded["is_modal_value"]).sum())
    corpus.meta["counts"]["entries.adjudicated"] = int(coded["adjudicated"].sum())

    # P5 census, P6 model, P7 sensitivity, P8 resample ----------------------
    results = {
        "M1": models.component_census(coded, COMPONENTS),
        "M2": models.cochran_and_pairwise(coded, COMPONENTS),
        "M3": models.agreement(coded, rng),
        "M4": models.genre(coded, rng),
        "M5": models.stage_comparison(stages),
        "M6": models.evidence_linking(controls, coded),
        "M7": models.competing_explanations(stages, stage_durations),
        "M8": models.specificity(coded, COMPONENTS),
        "M9": models.threshold_sweep(fields, COMPONENTS, code_frame, adjudicate),
        "M12": models.format_and_demand(coded, rng),
        "M13": models.completion_inflation(coded, dist),
        "M14": models.instructional_context(corpus.extra),
    }

    # The completion figure, computed from the artefacts and set beside the
    # materialised view, because the two disagree and the paper must say which
    # it reports.
    recomputed = float(stages["completion_rate"].mean())
    view = corpus.progress[corpus.progress["group_number"].between(1, SETTINGS.analytic_groups)]
    view_mean = float(view["avg_completion_pct"].mean()) if len(view) else float("nan")
    per_group = stages.groupby("group_number")["completion_rate"].mean() * 100
    completion = {
        "recomputed_mean_pct": recomputed * 100,
        "materialised_view_mean_pct": view_mean,
        "difference_pct_points": recomputed * 100 - view_mean,
        "view_reproduced_by_rounding_each_group_to_integer":
            float(per_group.round(0).mean()),
        "note": ("the view stores each group's average as a whole number, so the "
                 "mean of those stored values is not the mean of the artefacts"),
    }

    # G1 to G6 --------------------------------------------------------------
    gates = _gates(coded, results, corpus)

    payload = {
        "corpus": {"source": corpus.meta["source"],
                   "n_submissions": int(len(corpus.submissions)),
                   "n_groups": int(corpus.submissions["group_number"].nunique()),
                   "n_stages": int(corpus.submissions["stage"].nunique()),
                   "counts": corpus.meta["counts"]},
        "completion": completion,
        "distinctness": _jsonable(dist),
        "results": {k: _jsonable(v) for k, v in results.items()},
        "gates": gates,
    }

    (out_dir / "results.json").write_text(
        json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    releasable_decisions(coded).to_csv(out_dir / "coding_decisions.csv",
                                       index=False, encoding="utf-8")
    dist.to_csv(out_dir / "field_distinctness.csv", index=False, encoding="utf-8")

    # P10 provenance --------------------------------------------------------
    if publish_provenance is None:
        publish_provenance = source == "restricted"
    if publish_provenance:
        manifest = build_manifest(corpus.meta, notes=(
            "Two procedures registered in the analysis plan are reported as "
            "non-estimable rather than substituted. The pipeline records which, and on "
            "what grounds, in the results payload it writes alongside this manifest."))
        write_manifest(manifest)
    return payload


def _gates(coded: pd.DataFrame, results: dict, corpus) -> dict:
    coded_analytic = coded[coded["analytic"]]
    g1 = bool(coded_analytic["substantive"].notna().all()
              and coded_analytic["level"].notna().all())

    alpha_sub = results["M3"].values.get("alpha_substantive")
    raw_sub = results["M3"].values.get("raw_agreement_substantive", 0.0)
    # Alpha is undefined when one ruleset finds no variation at all; in that case
    # the raw agreement carries the check, and the undefined coefficient is
    # reported as undefined rather than as a failure to code.
    g2 = bool((alpha_sub is not None and not np.isnan(alpha_sub) and alpha_sub >= 0.80)
              or raw_sub >= 0.90)

    census = results["M1"].values["table"]
    zero_rows = [r for r in census if r["n_substantive"] == 0]
    g3 = all("substantive_upper_rule_of_three" in r for r in zero_rows)

    g4 = bool(results["M7"].values and results["M6"].values)
    g5 = bool(results["M9"].values.get("sweep"))
    counts = corpus.meta["counts"]
    g6 = counts.get("fields.derived") == counts.get("fields.expected")

    detail = {
        "G1_coding_complete": g1,
        "G2_agreement_threshold": g2,
        "G3_zero_count_bounded": g3,
        "G4_competing_explanations_tested": g4,
        "G5_threshold_sweep_complete": g5,
        "G6_row_counts_reconcile": g6,
    }
    detail["all_passed"] = all(detail.values())
    detail["notes"] = {
        "G2": (f"alpha_substantive={alpha_sub}, raw agreement={raw_sub:.3f}; "
               "agreement is between two rulesets, not two people"),
        "G6": f"derived {counts.get('fields.derived')} of {counts.get('fields.expected')} expected field rows",
    }
    return detail
