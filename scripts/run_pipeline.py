# -*- coding: utf-8 -*-
"""Execute the pipeline and print the figures the paper will quote."""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from argueaudit.pipeline import run  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    payload = run(Path(args.data_dir) if args.data_dir else None,
                  Path(args.output_dir) if args.output_dir else None)

    c = payload["corpus"]
    print(f'corpus: {c["n_submissions"]} artefacts, {c["n_groups"]} groups, '
          f'{c["n_stages"]} stages')

    comp = payload["completion"]
    print(f'\ncompletion across the lesson')
    print(f'   recomputed from artefacts : {comp["recomputed_mean_pct"]:.4f}%')
    print(f'   materialised view         : {comp["materialised_view_mean_pct"]:.4f}%')
    print(f'   view reproduced by rounding each group: '
          f'{comp["view_reproduced_by_rounding_each_group_to_integer"]:.4f}%')

    print('\nM1 component census')
    for r in payload["results"]["M1"]["values"]["table"]:
        ci = r["substantive_ci"]
        bound = r.get("substantive_upper_rule_of_three")
        extra = f'   rule-of-three upper bound {bound:.4f}' if bound else ''
        print(f'   {r["component"]:<15} {r["field"]:<14} populated '
              f'{r["n_populated"]}/{r["n"]}  substantive {r["n_substantive"]}/{r["n"]}'
              f'  95% CI [{ci[0]:.4f}, {ci[1]:.4f}]{extra}')

    print('\nM2 pairwise exact McNemar')
    for p in payload["results"]["M2"]["values"]["pairwise_mcnemar"]:
        print(f'   {p["pair"]:<34} discordant {p["discordant_10"]}/{p["discordant_01"]}'
              f'  exact p = {p["exact_p"]:.5f}')

    print('\nM3 agreement between rulesets')
    m3 = payload["results"]["M3"]["values"]
    print(f'   raw agreement on substance : {m3["raw_agreement_substantive"]:.4f}')
    print(f'   raw agreement on level     : {m3["raw_agreement_level"]:.4f}')
    print(f'   alpha (substance)          : {m3["alpha_substantive"]}')
    print(f'   alpha (level, ordinal)     : {m3["alpha_level"]}')

    print('\nM6 evidence-linking control')
    m6 = payload["results"]["M6"]
    print(f'   estimable: {m6["estimable"]}')
    for f, v in m6["values"]["controls"].items():
        print(f'   {f:<18} distinct={v["n_distinct"]}  {v["value_counts"]}')

    print('\nM12 populated rate by response format and demand')
    for cell in payload["results"]["M12"]["values"]["cells"]:
        ci = cell["ci"]
        print(f'   {cell["response_format"]:<9} {cell["demand"]:<10} '
              f'{cell["n_populated"]:>2}/{cell["n"]:<3} = {cell["rate"]:.3f}'
              f'  [{ci[0]:.3f}, {ci[1]:.3f}]')
    cc = payload["results"]["M12"]["values"]["central_contrast"]
    print(f'   open/describe {cc["open_describe"]["rate"]:.3f} vs '
          f'open/justify {cc["open_justify"]["rate"]:.3f}, '
          f'Fisher exact p = {cc["fisher_exact_p"]:.6f}')
    tc = payload["results"]["M12"]["values"]["timing_contrast"]
    print(f'   prospective justification {tc["prospective"]["n_populated"]}/'
          f'{tc["prospective"]["n"]} = {tc["prospective"]["rate"]:.3f} vs '
          f'retrospective {tc["retrospective"]["n_populated"]}/'
          f'{tc["retrospective"]["n"]} = {tc["retrospective"]["rate"]:.3f}, '
          f'Fisher exact p = {tc["fisher_exact_p"]:.6f}')

    print('\nM13 recorded completion against composed content')
    m13 = payload["results"]["M13"]["values"]
    print(f'   {"field":<20} {"format":<9} {"demand":<10} {"pop":>5} {"subst":>6} '
          f'{"compos":>7} {"distinct":>9} {"modal":>6} {"slots":>6}')
    for r in m13["by_field"]:
        print(f'   {r["field"]:<20} {r["response_format"]:<9} {r["demand"]:<10} '
              f'{r["populated_rate"]:>5.2f} {r["substantive_rate"]:>6.2f} '
              f'{r["composed_rate"]:>7.2f} '
              f'{r["n_distinct"]:>9} {r["modal_share"]:>6.2f} {r["n_with_unfilled_slot"]:>6}')
    print(f'   overall populated {m13["overall_populated_rate"]:.4f}, '
          f'composed {m13["overall_composed_rate"]:.4f}, '
          f'gap {m13["overall_inflation_points"]:.4f}')

    print('\ngates')
    for k, v in payload["gates"].items():
        if k in ("notes",):
            continue
        print(f'   {k:<36} {v}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
