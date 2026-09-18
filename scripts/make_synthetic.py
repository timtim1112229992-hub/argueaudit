# -*- coding: utf-8 -*-
"""Build the synthetic corpus that ships with the repository.

The restricted corpus is writing by nine and ten year olds and is not released,
so without a stand-in the published package would be code nobody outside the
project could execute. This writes a corpus with the same schema, the same field
register and the same shape, and nothing else in common.

The entries are assembled from a small English vocabulary by a seeded generator.
They are not translations, paraphrases or reconstructions of anything a child
wrote. The point is to exercise the code paths, not to imitate the data: a
stand-in that reproduced the findings would let a reader believe they had
replicated a result when they had replicated a random number seed. The rates
below were chosen to differ from the reported ones, and the run manifest records
`source: synthetic` for any run that reads this directory.

    python scripts/make_synthetic.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from argueaudit.config import FIELDS, STAGE_KEY, SYNTHETIC_DIR  # noqa: E402

SEED = 7
N_GROUPS = 12          # ten staffed, two provisioned and left empty
N_STAGES = 7

SUBJECTS = ("the first sample", "the second sample", "the third sample",
            "the fourth sample", "the water", "the jar")
VERBS = ("drained", "held together", "settled", "ran through", "clumped", "spread")
TAILS = ("quickly", "slowly", "more than the others", "in about a minute",
         "after we poured", "before the rest")
CONNECTIVES = ("because", "so", "which shows", "and that means")

# Menu fields store a selection rather than composed text.
MENU_OPTIONS = ("sandy", "clay", "loam", "mixed")

# A frame field arrives with its slots marked, and a preset field arrives
# carrying a value the respondent did not compose. Both are reproduced here so
# that the format analysis has something to find.
FRAME = "We thought ____ and we found ____"
PRESET = "Our idea changed after we tested it"


def _sentence(rng: np.random.Generator, reasoned: bool) -> str:
    core = f"{rng.choice(SUBJECTS)} {rng.choice(VERBS)} {rng.choice(TAILS)}"
    if reasoned:
        core += f" {rng.choice(CONNECTIVES)} {rng.choice(SUBJECTS)} {rng.choice(VERBS)}"
    return core


def _entry(rng: np.random.Generator, key: str, fmt: str, demand: str) -> str | None:
    """One stored field value, or None where the field was left empty.

    Populated rates vary by response format so that the format analysis is
    exercised, and they are deliberately unlike the observed ones.
    """
    rate = {"menu": 1.0, "exemplar": 0.95, "frame": 0.9, "preset": 1.0, "open": 0.6}[fmt]
    if rng.random() > rate:
        return None
    if fmt == "menu":
        return str(rng.choice(MENU_OPTIONS))
    if fmt == "preset":
        return PRESET
    if fmt == "frame":
        # Some groups return the frame with its slots still empty, which is the
        # case the composed-content measure has to separate from a real answer.
        return FRAME if rng.random() < .4 else FRAME.replace(
            "____", _sentence(rng, False), 1).replace("____", _sentence(rng, False), 1)
    return _sentence(rng, demand == "justify")


def build(rng: np.random.Generator) -> dict[str, pd.DataFrame]:
    groups = pd.DataFrame({
        "id": [f"g{n:02d}" for n in range(1, N_GROUPS + 1)],
        "group_number": list(range(1, N_GROUPS + 1)),
        "current_stage": [N_STAGES - 1] * 10 + [0, 0],
        "status": ["active"] * 10 + ["provisioned", "provisioned"],
        "created_at": ["2026-01-01T09:00:00"] * N_GROUPS,
    })

    by_stage: dict[int, list[tuple[str, str, str]]] = {}
    for stage, key, fmt, demand, analytic in FIELDS:
        by_stage.setdefault(stage, []).append((key, fmt, demand))
    # The control fields at the evidence stage, which the instrument submitted
    # with a fixed value for every group.
    controls = {"linkedEvidence": "yes", "evidenceStrength": "strong",
                "evidenceSource": "our test"}

    rows, submission_id = [], 0
    for _, group in groups.iterrows():
        if group["status"] != "active":
            continue
        for stage in range(N_STAGES):
            submission_id += 1
            form: dict[str, str] = {}
            for key, fmt, demand in by_stage.get(stage, []):
                value = _entry(rng, key, fmt, demand)
                if value is not None:
                    form[key] = value
            if stage == 5:
                form.update(controls)
            rows.append({
                "id": submission_id,
                "group_id": group["id"],
                "stage": stage,
                "stage_key": STAGE_KEY[stage],
                "is_completed": True,
                "completion_rate": round(float(rng.uniform(.6, 1.)), 2),
                "completed_at": f"2026-01-0{stage + 1}T10:00:00",
                "updated_at": f"2026-01-0{stage + 1}T10:05:00",
                "form_data_json": json.dumps(form, ensure_ascii=False),
            })

    submissions = pd.DataFrame(rows)
    per_group = (submissions.groupby("group_id")["completion_rate"].mean() * 100).round(0)
    progress = pd.DataFrame({
        "group_id": groups["id"],
        "group_number": groups["group_number"],
        "completed_stage_count": [N_STAGES] * 10 + [0, 0],
        "avg_completion_pct": [float(per_group.get(gid, 0.)) for gid in groups["id"]],
    })
    return {"02_groups.csv": groups,
            "03_stage_submissions.csv": submissions,
            "08_group_progress.csv": progress}


def main() -> int:
    rng = np.random.default_rng(SEED)
    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    for name, frame in build(rng).items():
        frame.to_csv(SYNTHETIC_DIR / name, index=False, encoding="utf-8")
        print(f"wrote {name}: {len(frame)} rows")
    print(f"\nsynthetic corpus in {SYNTHETIC_DIR}")
    print("the agent logs are not synthesised, so a run against this corpus reports")
    print("the instructional context model as non-estimable, which is correct")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
