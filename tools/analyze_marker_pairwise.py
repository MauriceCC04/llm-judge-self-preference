#!/usr/bin/env python3
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def flatten(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        markers = r.get("markers") or {}
        for marker, m in markers.items():
            winner_role = m.get("winner_role")
            if winner_role not in {"llm", "programmatic", "tie"}:
                pref = m.get("preferred")
                if pref == "plan_a":
                    winner_role = r.get("left_plan_role")
                elif pref == "plan_b":
                    winner_role = r.get("right_plan_role")
                else:
                    winner_role = "tie"
            out.append(
                {
                    "record_id": r.get("record_id"),
                    "pair_id": r.get("pair_id"),
                    "judge": r.get("judge"),
                    "judge_model_family": r.get("judge_model_family"),
                    "source_model_family": r.get("source_model_family"),
                    "self_family_match": r.get("self_family_match"),
                    "fixture_id": r.get("fixture_id"),
                    "order": r.get("order"),
                    "run": r.get("run"),
                    "marker": marker,
                    "preferred": m.get("preferred"),
                    "winner_role": winner_role,
                    "llm_won": True if winner_role == "llm" else False if winner_role == "programmatic" else None,
                    "plan_a_score": m.get("plan_a_score"),
                    "plan_b_score": m.get("plan_b_score"),
                    "confidence": m.get("confidence"),
                    "rationale": m.get("rationale"),
                }
            )
    return out


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def summarize(rows: List[Dict[str, Any]], group_keys: List[str]) -> List[Dict[str, Any]]:
    groups = defaultdict(list)
    for r in rows:
        key = tuple(str(r.get(k)) for k in group_keys)
        groups[key].append(r)

    out = []
    for key, rs in sorted(groups.items()):
        n = len(rs)
        llm = sum(1 for r in rs if r.get("winner_role") == "llm")
        prog = sum(1 for r in rs if r.get("winner_role") == "programmatic")
        tie = sum(1 for r in rs if r.get("winner_role") == "tie")
        non_tie = llm + prog
        row = {k: v for k, v in zip(group_keys, key)}
        row.update(
            {
                "n": n,
                "llm_wins": llm,
                "programmatic_wins": prog,
                "ties": tie,
                "llm_win_rate_all": llm / n if n else None,
                "programmatic_win_rate_all": prog / n if n else None,
                "tie_rate": tie / n if n else None,
                "llm_win_rate_excluding_ties": llm / non_tie if non_tie else None,
                "llm_win_rate_ties_half": (llm + 0.5 * tie) / n if n else None,
            }
        )
        out.append(row)
    return out


def pair_run_marker_categories(flat: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by = defaultdict(dict)
    for r in flat:
        by[(r["pair_id"], r["judge"], r["run"], r["marker"])][r["order"]] = r

    out = []
    for (pair_id, judge, run, marker), d in by.items():
        if set(d) != {"AB", "BA"}:
            cat = "missing_order"
        else:
            ab = d["AB"]
            ba = d["BA"]
            if ab["winner_role"] == ba["winner_role"] and ab["winner_role"] in {"llm", "programmatic"}:
                cat = f"source_consistent_{ab['winner_role']}"
            elif ab["preferred"] == ba["preferred"] == "plan_a":
                cat = "position_consistent_plan_a"
            elif ab["preferred"] == ba["preferred"] == "plan_b":
                cat = "position_consistent_plan_b"
            elif ab["winner_role"] == "tie" and ba["winner_role"] == "tie":
                cat = "consistent_tie"
            else:
                cat = "other_inconsistent"

        ref = next(iter(d.values()))
        out.append(
            {
                "pair_id": pair_id,
                "judge": judge,
                "judge_model_family": ref.get("judge_model_family"),
                "source_model_family": ref.get("source_model_family"),
                "self_family_match": ref.get("self_family_match"),
                "run": run,
                "marker": marker,
                "category": cat,
                "fixture_id": ref.get("fixture_id"),
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--marker-judgments", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    rows = read_jsonl(Path(args.marker_judgments))
    flat = flatten(rows)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    write_csv(out / "marker_rows_flat.csv", flat)
    write_csv(out / "marker_overall_summary.csv", summarize(flat, ["marker"]))
    write_csv(out / "marker_by_judge_summary.csv", summarize(flat, ["marker", "judge"]))
    write_csv(out / "marker_by_judge_family_summary.csv", summarize(flat, ["marker", "judge_model_family"]))
    write_csv(out / "marker_by_source_family_summary.csv", summarize(flat, ["marker", "source_model_family"]))
    write_csv(out / "marker_by_judge_family_source_family_summary.csv", summarize(flat, ["marker", "judge_model_family", "source_model_family"]))
    write_csv(out / "marker_self_family_summary.csv", summarize(flat, ["marker", "self_family_match"]))
    write_csv(out / "marker_order_bias_summary.csv", summarize(flat, ["marker", "order"]))

    pr = pair_run_marker_categories(flat)
    write_csv(out / "marker_pair_run_consistency_rows.csv", pr)

    cats = []
    counts = defaultdict(Counter)
    for r in pr:
        counts[(r["marker"], r["judge"])][r["category"]] += 1
    for (marker, judge), ctr in sorted(counts.items()):
        total = sum(ctr.values())
        for cat, n in sorted(ctr.items()):
            cats.append({"marker": marker, "judge": judge, "category": cat, "n": n, "rate": n / total if total else None})
    write_csv(out / "marker_pair_run_consistency_by_judge.csv", cats)

    overall_cats = []
    counts2 = defaultdict(Counter)
    for r in pr:
        counts2[r["marker"]][r["category"]] += 1
    for marker, ctr in sorted(counts2.items()):
        total = sum(ctr.values())
        for cat, n in sorted(ctr.items()):
            overall_cats.append({"marker": marker, "category": cat, "n": n, "rate": n / total if total else None})
    write_csv(out / "marker_pair_run_consistency_summary.csv", overall_cats)

    integrity = {
        "marker_judgment_records": len(rows),
        "flat_marker_rows": len(flat),
        "unique_record_ids": len({r["record_id"] for r in rows}),
        "judges": dict(Counter(r.get("judge") for r in rows)),
        "orders": dict(Counter(r.get("order") for r in rows)),
        "runs": dict(Counter(str(r.get("run")) for r in rows)),
        "markers": dict(Counter(r.get("marker") for r in flat)),
        "pair_run_marker_units": len(pr),
        "marker_schema_versions": dict(Counter(r.get("marker_schema_version") for r in rows)),
    }
    (out / "marker_integrity_summary.json").write_text(json.dumps(integrity, indent=2, sort_keys=True))
    print(json.dumps(integrity, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
