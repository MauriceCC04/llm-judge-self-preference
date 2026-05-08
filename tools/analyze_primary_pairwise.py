import argparse, json, math, random
from pathlib import Path
from collections import Counter, defaultdict
import csv

def load_jsonl(p):
    with Path(p).open() as f:
        return [json.loads(x) for x in f if x.strip()]

def rate_summary(rows):
    vals = [1 if r.get("llm_won") is True else 0 for r in rows if r.get("llm_won") is not None]
    n = len(vals)
    wins = sum(vals)
    p = wins / n if n else float("nan")
    se = math.sqrt(p * (1 - p) / n) if n else float("nan")
    return {
        "n": n,
        "llm_wins": wins,
        "llm_win_rate": p,
        "wald_ci_low": p - 1.96 * se if n else float("nan"),
        "wald_ci_high": p + 1.96 * se if n else float("nan"),
    }

def group(rows, key):
    d = defaultdict(list)
    for r in rows:
        d[str(key(r) if callable(key) else r.get(key))].append(r)
    return d

def write_summary_csv(path, summaries):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in summaries for k in row.keys()})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in summaries:
            w.writerow(row)

def pair_run_categories(rows):
    by = defaultdict(dict)
    for r in rows:
        by[(r["pair_id"], r["judge"], r["run"])][r["order"]] = r

    out = []
    for (pair_id, judge, run), d in by.items():
        if set(d) != {"AB", "BA"}:
            cat = "missing_order"
            source_winner = ""
        else:
            ab = d["AB"]
            ba = d["BA"]
            if ab.get("winner_role") == ba.get("winner_role"):
                source_winner = ab.get("winner_role")
                cat = f"source_consistent_{source_winner}"
            elif ab.get("preferred") == ba.get("preferred") == "plan_a":
                cat = "position_consistent_plan_a"
                source_winner = ""
            elif ab.get("preferred") == ba.get("preferred") == "plan_b":
                cat = "position_consistent_plan_b"
                source_winner = ""
            else:
                cat = "other_inconsistent"
                source_winner = ""
        ref = next(iter(d.values()))
        out.append({
            "pair_id": pair_id,
            "judge": judge,
            "judge_model_family": ref.get("judge_model_family"),
            "source_model_family": ref.get("source_model_family"),
            "self_family_match": ref.get("self_family_match"),
            "run": run,
            "category": cat,
            "source_winner": source_winner,
            "fixture_id": ref.get("fixture_id"),
        })
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judgments", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    rows = load_jsonl(args.judgments)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    overall = [{"group": "overall", **rate_summary(rows)}]
    write_summary_csv(out / "row_level_summary.csv", overall)

    for name, key in [
        ("per_judge_summary.csv", "judge"),
        ("per_judge_family_summary.csv", "judge_model_family"),
        ("per_source_family_summary.csv", "source_model_family"),
        ("self_family_summary.csv", lambda r: str(r.get("self_family_match"))),
        ("order_bias_summary.csv", "order"),
        ("run_summary.csv", "run"),
        ("fixture_band_summary.csv", lambda r: str(r.get("fixture_id", "")).split("__")[0].replace("ab_", "")),
    ]:
        summaries = []
        for g, rs in sorted(group(rows, key).items()):
            summaries.append({"group": g, **rate_summary(rs)})
        write_summary_csv(out / name, summaries)

    pr = pair_run_categories(rows)
    with (out / "pair_run_consistency_rows.csv").open("w", newline="") as f:
        keys = list(pr[0].keys())
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(pr)

    summaries = []
    for g, rs in sorted(group(pr, "category").items()):
        summaries.append({"category": g, "n": len(rs)})
    write_summary_csv(out / "pair_run_consistency_summary.csv", summaries)

    by_judge_cat = []
    d = defaultdict(Counter)
    for r in pr:
        d[r["judge"]][r["category"]] += 1
    for judge, ctr in sorted(d.items()):
        total = sum(ctr.values())
        for cat, n in sorted(ctr.items()):
            by_judge_cat.append({"judge": judge, "category": cat, "n": n, "rate": n / total})
    write_summary_csv(out / "pair_run_consistency_by_judge.csv", by_judge_cat)

    print("records", len(rows))
    print("pair_run_units", len(pr))
    print("wrote", out)

if __name__ == "__main__":
    main()
