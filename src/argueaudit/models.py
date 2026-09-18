"""The pre-specified estimators (M1 to M12).

Two of the procedures named in the analysis plan cannot be run on this corpus,
and are reported as non-estimable rather than replaced by something that looks
like a result. The evidence-linking test has a constant predictor, and the genre
regression would model an interface template rather than pupil language. In both
cases the reason is recorded and an estimable substitute is fitted alongside.

With ten groups, exact procedures are preferred to asymptotic ones throughout.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field as dc_field

import numpy as np
import pandas as pd
from scipy import stats

from .config import SETTINGS


@dataclass
class Result:
    name: str
    estimand: str
    estimable: bool = True
    values: dict = dc_field(default_factory=dict)
    note: str = ""


# --------------------------------------------------------------------------
# M1: exact binomial intervals
# --------------------------------------------------------------------------
def clopper_pearson(k: int, n: int, conf: float | None = None) -> tuple[float, float]:
    conf = SETTINGS.confidence if conf is None else conf
    if n == 0:
        return (float("nan"), float("nan"))
    a = 1 - conf
    lo = 0.0 if k == 0 else stats.beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - a / 2, k + 1, n - k)
    return (float(lo), float(hi))


def rule_of_three(n: int, conf: float | None = None) -> float:
    """M11: upper bound on a rate whose observed count is zero.

    Reported so that an absence carries a quantified precision instead of being
    stated as a bare zero.

    The value returned is the exact one-sided bound, 1 - (1 - conf) ** (1 / n).
    The familiar rule of three, 3 / n, is the large-sample approximation to it
    and the two part company at the sizes this package works with: for ten units
    the approximation gives 0.300 against an exact 0.259. The exact form is used
    because the approximation is anti-conservative here, and the function keeps
    its established name so that the results schema stays stable. Anything
    reporting this quantity should describe it as the exact bound.
    """
    conf = SETTINGS.confidence if conf is None else conf
    return float(1 - (1 - conf) ** (1 / n)) if n else float("nan")


def component_census(coded: pd.DataFrame, components) -> Result:
    rows = []
    for label, stage, fieldname in components:
        sel = coded[(coded["stage"] == stage) & (coded["field"] == fieldname)]
        n = len(sel)
        pop, sub = int(sel["populated"].sum()), int(sel["substantive"].sum())
        entry = {"component": label, "field": f"S{stage}.{fieldname}", "n": n,
                 "n_populated": pop, "populated_rate": pop / n if n else float("nan"),
                 "n_substantive": sub, "substantive_rate": sub / n if n else float("nan")}
        entry["populated_ci"] = clopper_pearson(pop, n)
        entry["substantive_ci"] = clopper_pearson(sub, n)
        if sub == 0 and n:
            entry["substantive_upper_rule_of_three"] = rule_of_three(n)
        rows.append(entry)
    return Result("M1", "populated and substantive rate per argument component",
                  values={"table": rows})


# --------------------------------------------------------------------------
# M2: within-group comparison across components
# --------------------------------------------------------------------------
def _exact_mcnemar(b: int, c: int) -> float:
    """Two-sided exact McNemar, conditioning on the discordant total."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return float(min(1.0, 2 * tail))


def cochran_and_pairwise(coded: pd.DataFrame, components, outcome="substantive") -> Result:
    wide = {}
    for label, stage, fieldname in components:
        sel = coded[(coded["stage"] == stage) & (coded["field"] == fieldname)]
        wide[label] = sel.set_index("group_number")[outcome].astype(int)
    mat = pd.DataFrame(wide).sort_index()

    col_sums = mat.sum(axis=0)
    values = {"outcome": outcome, "matrix_column_totals": col_sums.to_dict(),
              "n_groups": int(len(mat))}

    # Cochran Q is undefined when every row total is 0 or k; report the reason.
    row_tot = mat.sum(axis=1)
    k = mat.shape[1]
    informative = ((row_tot > 0) & (row_tot < k)).sum()
    if informative == 0:
        values["cochran_q"] = None
        values["cochran_note"] = ("every group has the same outcome on all components, "
                                  "so Q has no discordant rows to work with")
    else:
        g = col_sums.to_numpy(dtype=float)
        L = row_tot.to_numpy(dtype=float)
        q = ((k - 1) * (k * (g ** 2).sum() - g.sum() ** 2)) / (k * g.sum() - (L ** 2).sum())
        values["cochran_q"] = float(q)
        values["cochran_df"] = k - 1
        values["cochran_p"] = float(stats.chi2.sf(q, k - 1))
        values["cochran_note"] = "asymptotic reference distribution, reported with the exact pairwise tests"

    pairs = []
    for a, b in itertools.combinations(mat.columns, 2):
        x, y = mat[a].to_numpy(), mat[b].to_numpy()
        n01 = int(((x == 0) & (y == 1)).sum())
        n10 = int(((x == 1) & (y == 0)).sum())
        pairs.append({"pair": f"{a} vs {b}", "discordant_01": n01, "discordant_10": n10,
                      "exact_p": _exact_mcnemar(n01, n10)})
    values["pairwise_mcnemar"] = pairs
    return Result("M2", "within-group difference in completion across components",
                  values=values)


# --------------------------------------------------------------------------
# M3: agreement between the two rulesets
# --------------------------------------------------------------------------
def krippendorff_alpha(a, b, level="nominal") -> float:
    """Alpha for two complete codings of the same units."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    n = len(a)
    if n == 0:
        return float("nan")

    values = np.concatenate([a, b])
    if len(np.unique(values)) == 1:
        # Perfect agreement with no variation: alpha is undefined, not 1.
        return float("nan")

    def delta(x, y):
        if level == "nominal":
            return 0.0 if x == y else 1.0
        return float((x - y) ** 2)

    observed = np.mean([delta(x, y) for x, y in zip(a, b)])
    expected = np.mean([delta(x, y) for x, y in itertools.product(values, values)])
    expected = expected * (2 * n) / (2 * n - 1)
    return float(1 - observed / expected) if expected else float("nan")


def agreement(coded: pd.DataFrame, rng: np.random.Generator,
              replicates: int | None = None) -> Result:
    replicates = replicates or SETTINGS.bootstrap_replicates
    sel = coded[coded["analytic"]]
    values = {"n_entries": int(len(sel)),
              "raw_agreement_substantive": float(sel["agree_substantive"].mean()),
              "raw_agreement_level": float(sel["agree_level"].mean()),
              "n_disagreements": int((~sel["agree_substantive"]).sum()
                                     + (~sel["agree_level"]).sum())}
    values["alpha_substantive"] = krippendorff_alpha(
        sel["substantive_A"].astype(int), sel["substantive_B"].astype(int), "nominal")
    values["alpha_level"] = krippendorff_alpha(
        sel["level_A"], sel["level_B"], "ordinal")

    # Cluster bootstrap over groups, since entries within a group are not
    # independent draws.
    groups = sel["group_number"].unique()
    boot_sub, boot_lev = [], []
    for _ in range(min(replicates, 2000)):
        pick = rng.choice(groups, size=len(groups), replace=True)
        rep = pd.concat([sel[sel["group_number"] == g] for g in pick])
        boot_sub.append(krippendorff_alpha(rep["substantive_A"].astype(int),
                                           rep["substantive_B"].astype(int), "nominal"))
        boot_lev.append(krippendorff_alpha(rep["level_A"], rep["level_B"], "ordinal"))
    for key, arr in (("alpha_substantive_ci", boot_sub), ("alpha_level_ci", boot_lev)):
        clean = [v for v in arr if not math.isnan(v)]
        values[key] = (float(np.percentile(clean, 2.5)),
                       float(np.percentile(clean, 97.5))) if clean else (float("nan"),) * 2
    values["note"] = ("agreement between two deterministic rulesets, not between "
                      "human coders")
    return Result("M3", "agreement between the two coding rulesets", values=values)


# --------------------------------------------------------------------------
# M4: genre separation, refitted as template adherence
# --------------------------------------------------------------------------
def genre(frame: pd.DataFrame, rng: np.random.Generator) -> Result:
    """The planned negative binomial model is not fitted, and here is why.

    A regression of marker counts on field type presumes the entries are
    independent compositions. They are not: the ten observation entries take four
    distinct values and the ten interpretation entries take four, each closely
    tracking the model sentence the interface displayed. A fitted coefficient
    would describe the instrument's template, not the pupils' language. What is
    estimable, and what is reported instead, is how closely each field's entries
    reproduce the exemplar, and whether the two fields differ in marker content
    once that reproduction is acknowledged.
    """
    see = frame[(frame["stage"] == 5) & (frame["field"] == "seeBox") & frame["populated"]]
    think = frame[(frame["stage"] == 5) & (frame["field"] == "thinkBox") & frame["populated"]]

    values = {
        "observation": {"n": len(see), "n_distinct": int(see["norm"].nunique()),
                        "mean_exemplar_similarity": float(see["exemplar_similarity"].mean()),
                        "mk_perceptual": float(see["mk_perceptual"].mean()),
                        "mk_inferential": float(see["mk_inferential"].mean())},
        "interpretation": {"n": len(think), "n_distinct": int(think["norm"].nunique()),
                           "mean_exemplar_similarity": float(think["exemplar_similarity"].mean()),
                           "mk_perceptual": float(think["mk_perceptual"].mean()),
                           "mk_inferential": float(think["mk_inferential"].mean())},
    }

    # Paired exact permutation on the marker contrast, pairing within group.
    merged = see.set_index("group_number")[["mk_perceptual", "mk_inferential"]].join(
        think.set_index("group_number")[["mk_perceptual", "mk_inferential"]],
        lsuffix="_see", rsuffix="_think")
    d = ((merged["mk_inferential_think"] - merged["mk_perceptual_think"])
         - (merged["mk_inferential_see"] - merged["mk_perceptual_see"])).dropna().to_numpy()
    n = len(d)
    if n and np.any(d != 0):
        observed = d.mean()
        count = 0
        for signs in itertools.product((1, -1), repeat=n):
            if abs((d * np.array(signs)).mean()) >= abs(observed) - 1e-12:
                count += 1
        values["paired_permutation"] = {"n_pairs": n, "mean_difference": float(observed),
                                        "exact_p": count / (2 ** n)}
    else:
        values["paired_permutation"] = {"n_pairs": n, "note": "no within-pair variation"}

    values["planned_model"] = "negative binomial with length offset and group random effect"
    values["planned_model_fitted"] = False
    values["planned_model_reason"] = (
        "entries reproduce a displayed exemplar, so marker counts index the "
        "instrument's template rather than pupil language")
    return Result("M4", "whether observation and interpretation entries differ in genre",
                  estimable=False, values=values,
                  note="reported as template adherence; the planned regression is not identified")


# --------------------------------------------------------------------------
# M5: stage position control
# --------------------------------------------------------------------------
def stage_comparison(stages: pd.DataFrame, focal: int = 5) -> Result:
    wide = stages.pivot(index="group_number", columns="stage", values="completion_rate")
    rows = []
    for other in sorted(c for c in wide.columns if c != focal):
        pair = wide[[focal, other]].dropna()
        d = (pair[focal] - pair[other]).to_numpy()
        nz = d[d != 0]
        if len(nz) == 0:
            rows.append({"against_stage": int(other), "n_pairs": int(len(d)),
                         "mean_difference": float(d.mean()), "exact_p": 1.0,
                         "note": "identical completion in every group"})
            continue
        # Exact sign test, which is valid with the heavy ties these values carry.
        pos = int((nz > 0).sum())
        p = float(min(1.0, 2 * stats.binom.sf(max(pos, len(nz) - pos) - 1, len(nz), 0.5)))
        rows.append({"against_stage": int(other), "n_pairs": int(len(d)),
                     "n_nonzero": int(len(nz)), "n_focal_higher": pos,
                     "mean_difference": float(d.mean()), "exact_p": p})
    return Result("M5", "completion at the evidence stage against every other stage",
                  values={"focal_stage": focal, "comparisons": rows,
                          "stage_means": wide.mean().to_dict()})


# --------------------------------------------------------------------------
# M6: the planned association test, and the estimable question behind it
# --------------------------------------------------------------------------
def evidence_linking(controls: pd.DataFrame, coded: pd.DataFrame) -> Result:
    values = {}
    for fieldname in controls["field"].unique():
        sel = controls[controls["field"] == fieldname]
        counts = sel["value"].value_counts().to_dict()
        values[fieldname] = {"n": int(len(sel)), "n_distinct": int(sel["value"].nunique()),
                             "value_counts": counts}
    constant = [f for f, v in values.items() if v["n_distinct"] < 2]
    return Result(
        "M6", "association between the evidence-linking control and a substantive warrant",
        estimable=not constant,
        values={"controls": values, "constant_fields": constant},
        note=("the planned Fisher exact test has no estimand: "
              f"{', '.join(constant)} take a single value in every group, which is the "
              "signature of an interface default rather than a recorded pupil choice")
        if constant else "")


# --------------------------------------------------------------------------
# M7: competing explanations
# --------------------------------------------------------------------------
def competing_explanations(stages: pd.DataFrame, durations: pd.DataFrame | None) -> Result:
    values = {}
    by_stage = stages.groupby("stage")["completion_rate"].mean()
    rho, p = stats.spearmanr(by_stage.index.to_numpy(), by_stage.to_numpy())
    values["sequence_position"] = {"spearman_rho": float(rho), "p": float(p),
                                   "n": int(len(by_stage))}
    if durations is not None and len(durations):
        merged = stages.merge(durations, on=["group_number", "stage"], how="inner").dropna(
            subset=["duration_s", "completion_rate"])
        if len(merged) > 2 and merged["duration_s"].nunique() > 1:
            rho2, p2 = stats.spearmanr(merged["duration_s"], merged["completion_rate"])
            values["stage_duration"] = {"spearman_rho": float(rho2), "p": float(p2),
                                        "n": int(len(merged))}
        else:
            values["stage_duration"] = {"note": "insufficient variation in recorded duration"}
    else:
        values["stage_duration"] = {"note": "no usable stage duration in the archive"}
    return Result("M7", "time on stage and sequence position as accounts of the pattern",
                  values=values)


# --------------------------------------------------------------------------
# M8: specificity
# --------------------------------------------------------------------------
def specificity(frame: pd.DataFrame, components) -> Result:
    rows = []
    for label, stage, fieldname in components:
        sel = frame[(frame["stage"] == stage) & (frame["field"] == fieldname)]
        pop = sel[sel["populated"]]
        rows.append({"component": label, "n_populated": int(len(pop)),
                     "mean_sample_refs": float(pop["mk_sample"].mean()) if len(pop) else None,
                     "mean_property_refs": float(pop["mk_property"].mean()) if len(pop) else None})
    return Result("M8", "reference to particular samples and properties by component",
                  values={"table": rows})


# --------------------------------------------------------------------------
# M9: threshold sensitivity
# --------------------------------------------------------------------------
def threshold_sweep(frame: pd.DataFrame, components, code_frame, adjudicate) -> Result:
    rows = []
    for min_content, min_prop in SETTINGS.threshold_sweep:
        coded = adjudicate(code_frame(frame, min_content, min_prop))
        entry = {"min_content_chars": min_content, "min_propositional_chars": min_prop}
        for label, stage, fieldname in components:
            sel = coded[(coded["stage"] == stage) & (coded["field"] == fieldname)]
            entry[f"{label}_substantive"] = int(sel["substantive"].sum())
        rows.append(entry)
    return Result("M9", "substantive rate across the declared threshold sweep",
                  values={"sweep": rows})


# --------------------------------------------------------------------------
# M10: cluster bootstrap
# --------------------------------------------------------------------------
def cluster_bootstrap(frame: pd.DataFrame, statistic, rng: np.random.Generator,
                      replicates: int | None = None) -> dict:
    replicates = replicates or SETTINGS.bootstrap_replicates
    groups = frame["group_number"].unique()
    draws = []
    for _ in range(replicates):
        pick = rng.choice(groups, size=len(groups), replace=True)
        rep = pd.concat([frame[frame["group_number"] == g] for g in pick])
        value = statistic(rep)
        if value is not None and not (isinstance(value, float) and math.isnan(value)):
            draws.append(value)
    if not draws:
        return {"point": float("nan"), "ci": (float("nan"), float("nan")), "n_replicates": 0}
    return {"point": float(statistic(frame)),
            "ci": (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))),
            "n_replicates": len(draws)}


# --------------------------------------------------------------------------
# M12: the estimable question, response format against epistemic demand
# --------------------------------------------------------------------------
def format_and_demand(coded: pd.DataFrame, rng: np.random.Generator) -> Result:
    """Completion read against how the interface asked, not only what it asked for.

    This is the contrast the corpus can actually support, and it is the one the
    evidence-linking test was reaching for. Fields are grouped by response format
    and by epistemic demand, and the populated rate is reported for each cell with
    an exact interval.
    """
    sel = coded[coded["analytic"]].copy()
    rows = []
    for (fmt, demand), grp in sel.groupby(["response_format", "demand"]):
        n, k = len(grp), int(grp["populated"].sum())
        rows.append({"response_format": fmt, "demand": demand, "n": n, "n_populated": k,
                     "rate": k / n, "ci": clopper_pearson(k, n),
                     "fields": sorted({f"S{s}.{f}" for s, f in
                                       zip(grp["stage"], grp["field"])})})

    # The central contrast: free-text fields asking for description against
    # free-text fields asking for justification.
    open_desc = sel[(sel["response_format"] == "open") & (sel["demand"] == "describe")]
    open_just = sel[(sel["response_format"] == "open") & (sel["demand"] == "justify")]
    kd, nd = int(open_desc["populated"].sum()), len(open_desc)
    kj, nj = int(open_just["populated"].sum()), len(open_just)
    table = [[kd, nd - kd], [kj, nj - kj]]
    odds, p = stats.fisher_exact(table)

    contrast = {"open_describe": {"n": nd, "n_populated": kd, "rate": kd / nd if nd else None,
                                  "ci": clopper_pearson(kd, nd)},
                "open_justify": {"n": nj, "n_populated": kj, "rate": kj / nj if nj else None,
                                 "ci": clopper_pearson(kj, nj)},
                "fisher_exact_p": float(p), "odds_ratio": float(odds)}

    def diff(rep):
        a = rep[(rep["response_format"] == "open") & (rep["demand"] == "describe")]
        b = rep[(rep["response_format"] == "open") & (rep["demand"] == "justify")]
        if not len(a) or not len(b):
            return None
        return float(a["populated"].mean() - b["populated"].mean())

    contrast["cluster_bootstrap_difference"] = cluster_bootstrap(sel, diff, rng)

    # The second contrast, and the one that locates the deficit in time. Asking a
    # pupil why they think something, before any evidence exists, is a different
    # act from asking them to relate evidence back to what they thought forty
    # minutes earlier. Both are justification; only the second requires holding an
    # earlier commitment in mind while reading present data.
    prospective = sel[(sel["stage"] == 2) & (sel["field"] == "reason")]
    retrospective = sel[sel["field"].isin(["findingReason", "guessCompare", "explain"])]
    kp, np_ = int(prospective["populated"].sum()), len(prospective)
    kr, nr = int(retrospective["populated"].sum()), len(retrospective)
    odds_t, p_t = stats.fisher_exact([[kp, np_ - kp], [kr, nr - kr]])
    timing = {
        "prospective": {"fields": ["S2.reason"], "n": np_, "n_populated": kp,
                        "rate": kp / np_ if np_ else None, "ci": clopper_pearson(kp, np_)},
        "retrospective": {"fields": ["S3.findingReason", "S4.guessCompare", "S5.explain"],
                          "n": nr, "n_populated": kr,
                          "rate": kr / nr if nr else None, "ci": clopper_pearson(kr, nr)},
        "fisher_exact_p": float(p_t), "odds_ratio": float(odds_t),
    }
    return Result("M12", "populated rate by response format and epistemic demand",
                  values={"cells": rows, "central_contrast": contrast,
                          "timing_contrast": timing})


# --------------------------------------------------------------------------
# M13: what the recorded completion figure is counting
# --------------------------------------------------------------------------
def completion_inflation(coded: pd.DataFrame, dist: pd.DataFrame) -> Result:
    """The gap between a field being populated and a pupil having written something.

    The stored completion metric tests for a non-empty field. Where the interface
    supplies the value, or supplies a frame the pupil submits unfilled, that test
    is satisfied without any pupil composition. This quantifies how far the two
    diverge, field by field, which is the measurement caveat the completion
    figures in this paper have to carry.
    """
    rows = []
    for _, d in dist.iterrows():
        sel = coded[(coded["stage"] == d["stage"]) & (coded["field"] == d["field"])]
        n = len(sel)
        pop = int(sel["populated"].sum())
        sub = int(sel["substantive"].sum())
        com = int(sel["composed"].sum()) if "composed" in sel.columns else sub
        rows.append({
            "field": f'S{int(d["stage"])}.{d["field"]}',
            "response_format": d["response_format"], "demand": d["demand"],
            "n": n, "populated_rate": pop / n if n else None,
            "substantive_rate": sub / n if n else None,
            "composed_rate": com / n if n else None,
            "inflation": (pop - com) / n if n else None,
            "n_distinct": int(d["n_distinct"]), "modal_share": d["modal_share"],
            "instrument_supplied": bool(d["instrument_supplied"]),
            "n_with_unfilled_slot": int(d["n_with_unfilled_slot"]),
        })
    total_n = sum(r["n"] for r in rows)
    total_pop = sum(r["populated_rate"] * r["n"] for r in rows if r["populated_rate"])
    total_com = sum(r["composed_rate"] * r["n"] for r in rows if r["composed_rate"])
    return Result("M13", "recorded completion against composed content",
                  values={"by_field": rows,
                          "overall_populated_rate": total_pop / total_n if total_n else None,
                          "overall_composed_rate": total_com / total_n if total_n else None,
                          "overall_inflation_points":
                              (total_pop - total_com) / total_n if total_n else None})


# --------------------------------------------------------------------------
# M14: the instructional setting the argument was written in
# --------------------------------------------------------------------------
def instructional_context(extra: dict | None, focal_stage: int = 5) -> Result:
    """Counts describing the support pupils received while they were writing.

    This is context rather than contribution. A decision-level audit of the
    agent's policy belongs to a separate study and nothing here characterises
    that policy. The counts exist so that a reader knows the justification field
    was left empty in a room where an agent was present and intervening, not in
    an unattended form. Only structural columns are read, so no wording composed
    by an agent or by a pupil enters the analysis.
    """
    if not extra or "decisions" not in extra or not len(extra["decisions"]):
        return Result("M14", "instructional support present during writing",
                      estimable=False, note="agent logs absent from this corpus")

    dec = extra["decisions"].copy()
    dec["stage"] = pd.to_numeric(dec["stage"], errors="coerce")
    focal = dec[dec["stage"] == focal_stage]
    probes = dec[dec["action"] == "probe"]

    values = {
        "n_decisions": int(len(dec)),
        "n_decisions_by_action":
            {str(k): int(v) for k, v in dec["action"].value_counts().items()},
        "n_groups_receiving_any_decision": int(dec["group_id"].nunique()),
        "decisions_per_group_mean": float(len(dec) / dec["group_id"].nunique()),
        "focal_stage": focal_stage,
        "n_decisions_focal_stage": int(len(focal)),
        "n_decisions_focal_by_action":
            {str(k): int(v) for k, v in focal["action"].value_counts().items()},
        "n_groups_probed_anywhere": int(probes["group_id"].nunique()),
        "n_groups_probed_at_focal_stage":
            int(probes[probes["stage"] == focal_stage]["group_id"].nunique()),
    }

    if "qa" in extra and len(extra["qa"]):
        qa = extra["qa"]
        values["n_pupil_questions"] = int(len(qa))
        values["n_groups_asking"] = int(qa["group_id"].nunique())
        if "on_topic" in qa.columns:
            values["n_questions_on_topic"] = int(
                qa["on_topic"].astype(str).str.lower().isin(["true", "1"]).sum())

    return Result("M14", "instructional support present during writing", values=values,
                  note=("counts only; the agent's policy is not characterised here and "
                        "no message text is read"))
