"""Rendering of the data-driven figures (P9).

Only figures whose content comes out of the analysis are produced here.
Conceptual diagrams, such as the coding workflow and the argument assembly model,
belong in a drawing tool rather than in code.

Labels are in English regardless of the language of the corpus, because the
figures are read by the audience of the paper rather than by the participants.
No axis label, tick or annotation carries pupil-authored text.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

STAGE_LABEL = {0: "S0\npretest", 1: "S1\nobserve", 2: "S2\nconjecture",
               3: "S3\ncohesion", 4: "S4\ndrainage", 5: "S5\nevidence",
               6: "S6\nconclusion"}
INK = "#2f4b7c"
MID = "#7a9cc6"
WARM = "#e07a3f"
GREY = "#c9c9c9"

plt.rcParams.update({"figure.dpi": 300, "font.size": 8, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False})


def _save(fig, out: Path, name: str) -> Path:
    """Write the raster copy for reading and a vector copy for submission.

    Journals that accept line art normally want EPS or TIFF at a resolution a
    raster export cannot reach without becoming unwieldy. Saving the vector form
    alongside the PNG avoids a later re-render, which would risk the figure and
    the results file drifting apart.
    """
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    fig.savefig(path)
    fig.savefig(path.with_suffix(".eps"))
    plt.close(fig)
    return path


def fig_6_1_completion_trajectory(stages: pd.DataFrame, out: Path) -> Path:
    """Completion across the seven stages, with each group drawn behind the mean."""
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    wide = stages.pivot(index="group_number", columns="stage", values="completion_rate")
    for _, row in wide.iterrows():
        ax.plot(row.index, row.to_numpy(), color=GREY, lw=.7, alpha=.75, zorder=1)
    mean = wide.mean()
    ax.plot(mean.index, mean.to_numpy(), color=INK, lw=1.8, marker="o", ms=4,
            zorder=3, label="cohort mean")
    ax.axvline(5, color=WARM, lw=1.0, ls="--", zorder=2)
    ax.annotate("evidence stage", xy=(5, .55), xytext=(4.05, .50), color=WARM,
                fontsize=7.5)
    ax.set_xticks(range(7))
    ax.set_xticklabels([STAGE_LABEL[s] for s in range(7)], fontsize=7)
    ax.set_ylim(.45, 1.04)
    ax.set_ylabel("stored completion rate")
    ax.set_title("Figure 6.1  Completion trajectory across the seven stages")
    ax.legend(frameon=False, loc="lower left", fontsize=7.5)
    return _save(fig, out, "figure_6_1_completion_trajectory.png")


def fig_6_2_component_contrast(census: list[dict], out: Path) -> Path:
    """Populated and substantive rates for the three argument components."""
    labels = [c["component"] for c in census]
    pop = [c["populated_rate"] for c in census]
    sub = [c["substantive_rate"] for c in census]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.bar(x - .19, pop, width=.36, color=MID, label="populated")
    ax.bar(x + .19, sub, width=.36, color=INK, label="substantive")

    for i, c in enumerate(census):
        lo, hi = c["substantive_ci"]
        ax.plot([x[i] + .19, x[i] + .19], [lo, hi], color="black", lw=.9, zorder=4)
        ax.plot([x[i] + .13, x[i] + .25], [hi, hi], color="black", lw=.9, zorder=4)
    zero = next((c for c in census if c["n_substantive"] == 0), None)
    if zero is not None:
        i = labels.index(zero["component"])
        hi = zero["substantive_ci"][1]
        ax.annotate(f'exact one-sided\nupper bound '
                    f'{zero["substantive_upper_rule_of_three"]:.3f}',
                    xy=(x[i] + .19, hi), xytext=(x[i] - .46, hi + .22),
                    fontsize=7, color=WARM, ha="left",
                    arrowprops=dict(arrowstyle="-", color=WARM, lw=.8))
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("proportion of the ten groups")
    ax.set_title("Figure 6.2  Populated against substantive completion, by component")
    ax.legend(frameon=False, ncol=2, fontsize=7.5, loc="upper center",
              bbox_to_anchor=(.5, -.12))
    return _save(fig, out, "figure_6_2_component_contrast.png")


def fig_6_3_provenance(dist: pd.DataFrame, m4: dict, out: Path) -> Path:
    """Distinct values per field, and how far the evidence-stage entries copy the exemplar."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.3),
                                   gridspec_kw={"width_ratios": [1.55, 1]})

    d = dist[dist["n_populated"] > 0].copy()
    d["label"] = "S" + d["stage"].astype(str) + "." + d["field"]
    d = d.sort_values("distinctness")
    colour = [WARM if v < .5 else MID for v in d["distinctness"]]
    ax1.barh(d["label"], d["distinctness"], color=colour, height=.66)
    ax1.axvline(.5, color="black", lw=.7, ls=":")
    ax1.set_xlim(0, 1.05)
    ax1.set_xlabel("distinct values as a fraction of populated entries")
    ax1.tick_params(axis="y", labelsize=6.8)
    ax1.set_title("Composition against reproduction", fontsize=8.5)

    names = ["observation", "interpretation"]
    sims = [m4["observation"]["mean_exemplar_similarity"],
            m4["interpretation"]["mean_exemplar_similarity"]]
    perc = [m4["observation"]["mk_perceptual"], m4["interpretation"]["mk_perceptual"]]
    infer = [m4["observation"]["mk_inferential"], m4["interpretation"]["mk_inferential"]]
    x = np.arange(2)
    ax2.bar(x - .2, perc, width=.38, color=MID, label="perceptual markers")
    ax2.bar(x + .2, infer, width=.38, color=INK, label="inferential markers")
    # A count of zero draws no bar, and an invisible bar reads as a missing
    # category rather than as the absence that it is, so every value is labelled.
    for xi, value in list(zip(x - .2, perc)) + list(zip(x + .2, infer)):
        ax2.text(xi, value + .06, f"{value:.1f}", ha="center", fontsize=7,
                 color="#555555")
    ax2.set_ylim(0, max(perc + infer) * 1.28)
    ax2b = ax2.twinx()
    ax2b.plot(x, sims, color=WARM, marker="D", ms=5, lw=1.2, label="exemplar similarity")
    ax2b.set_ylim(0, 1.0)
    ax2b.set_ylabel("mean similarity to displayed exemplar", color=WARM, fontsize=7.5)
    ax2b.tick_params(axis="y", colors=WARM, labelsize=7)
    ax2b.spines["top"].set_visible(False)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontsize=7.5)
    ax2.set_ylabel("mean markers per entry")
    ax2.set_title("Evidence-stage entries", fontsize=8.5)
    handles = ax2.get_legend_handles_labels()[0] + ax2b.get_legend_handles_labels()[0]
    labels_ = ax2.get_legend_handles_labels()[1] + ax2b.get_legend_handles_labels()[1]
    ax2.legend(handles, labels_, frameon=False, fontsize=6.6, loc="upper center",
               bbox_to_anchor=(.5, -.16))
    fig.suptitle("Figure 6.3  Provenance of the entries, and the marker separation it explains",
                 fontsize=9, y=1.02)
    return _save(fig, out, "figure_6_3_provenance.png")


def fig_6_4_threshold_sensitivity(sweep: list[dict], out: Path) -> Path:
    """The substantive count for each component across the declared threshold sweep."""
    fig, ax = plt.subplots(figsize=(5.6, 3.1))
    x = np.arange(len(sweep))
    # Observation and interpretation coincide at ten across the whole sweep. The
    # upper series is dashed so the one beneath it stays visible, rather than
    # appearing to be missing.
    series = (("observation_substantive", "observation", MID, "o", "-", 2),
              ("interpretation_substantive", "interpretation", INK, "s", "--", 3),
              ("justification_substantive", "justification", WARM, "D", "-", 2))
    for key, label, colour, marker, style, z in series:
        ax.plot(x, [row[key] for row in sweep], color=colour, marker=marker, ms=4.5,
                lw=1.3, ls=style, zorder=z, label=label)
    ax.annotate("observation and interpretation coincide", xy=(2.5, 10),
                xytext=(1.2, 8.3), fontsize=7, color="#555555",
                arrowprops=dict(arrowstyle="-", color="#999999", lw=.7))
    ax.set_xticks(x)
    ax.set_xticklabels([f'{r["min_content_chars"]}/{r["min_propositional_chars"]}'
                        for r in sweep], fontsize=7.5)
    ax.set_xlabel("threshold, content characters / propositional characters")
    ax.set_ylabel("groups coded substantive, of ten")
    ax.set_ylim(-.5, 10.6)
    ax.set_title("Figure 6.4  Sensitivity of the substantive rate to threshold choice")
    ax.legend(frameon=False, fontsize=7.5, loc="center right")
    return _save(fig, out, "figure_6_4_threshold_sensitivity.png")


def fig_6_5_timing_contrast(timing: dict, contrast: dict, out: Path) -> Path:
    """Two contrasts: format against demand, and justification before against after."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.2))

    for ax, pair, title, names in (
            (ax1, (contrast["open_describe"], contrast["open_justify"]),
             f'Free-text fields, exact p = {contrast["fisher_exact_p"]:.4f}',
             ("describe", "justify")),
            (ax2, (timing["prospective"], timing["retrospective"]),
             f'Justification, exact p = {timing["fisher_exact_p"]:.4f}',
             ("before the\nevidence", "after the\nevidence"))):
        x = np.arange(2)
        rates = [p["rate"] for p in pair]
        ax.bar(x, rates, width=.5, color=[MID, WARM])
        for i, p in enumerate(pair):
            lo, hi = p["ci"]
            ax.plot([x[i], x[i]], [lo, hi], color="black", lw=.9)
            ax.plot([x[i] - .06, x[i] + .06], [hi, hi], color="black", lw=.9)
            ax.plot([x[i] - .06, x[i] + .06], [lo, lo], color="black", lw=.9)
            # Counts sit inside the bar, clear of the tick labels underneath.
            # A short bar leaves no room below the lower error cap, so the
            # label moves above the upper cap instead of overprinting the bar.
            label = f'{p["n_populated"]}/{p["n"]}'
            if lo > .09:
                ax.text(x[i], .035, label, ha="center", fontsize=7,
                        color="white", fontweight="bold")
            else:
                ax.text(x[i], hi + .03, label, ha="center", fontsize=7,
                        color="black", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=7.5)
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=8)
    ax1.set_ylabel("populated rate")
    fig.suptitle("Figure 6.5  What the interface asked for, and when it asked",
                 fontsize=9, y=1.02)
    return _save(fig, out, "figure_6_5_timing_contrast.png")


def render_all(stages: pd.DataFrame, dist: pd.DataFrame, results: dict,
               out: Path) -> list[Path]:
    census = results["M1"]["values"]["table"]
    m12 = results["M12"]["values"]
    return [
        fig_6_1_completion_trajectory(stages, out),
        fig_6_2_component_contrast(census, out),
        fig_6_3_provenance(dist, results["M4"]["values"], out),
        fig_6_4_threshold_sensitivity(results["M9"]["values"]["sweep"], out),
        fig_6_5_timing_contrast(m12["timing_contrast"], m12["central_contrast"], out),
    ]
