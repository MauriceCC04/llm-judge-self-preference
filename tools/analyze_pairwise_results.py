"""Analysis helpers for the Qwen/Gemma pairwise judgment study.

Reads pairwise JSONL files emitted by judge/harness.py and a matched-pairs
manifest. Produces descriptive win rates, bootstrap confidence intervals,
order-bias checks, source-family/self-family summaries, and score-gap sensitivity
without fabricating any results.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def _family(value: Any) -> str:
    s = str(value or "").lower()
    if "qwen" in s:
        return "qwen"
    if "gemma" in s:
        return "gemma"
    if "programmatic" in s:
        return "programmatic"
    return "unknown"


def _jsonl_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.glob("pairwise_*.jsonl")) + sorted(path.glob("*.pairwise.jsonl"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _load_pairs(path: Path) -> dict[str, dict[str, Any]]:
    return {p["pair_id"]: p for p in json.loads(path.read_text(encoding="utf-8"))}


def _bootstrap_ci(values: list[float], *, n_boot: int = 2000, seed: int = 0) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "ci_low": None, "ci_high": None}
    rng = random.Random(seed)
    means = []
    n = len(values)
    for _ in range(n_boot):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return {
        "n": n,
        "mean": sum(values) / n,
        "ci_low": means[int(0.025 * (n_boot - 1))],
        "ci_high": means[int(0.975 * (n_boot - 1))],
    }


def _summarize(rows: list[dict[str, Any]], key: str | None = None) -> dict[str, Any]:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        pref = row.get("prefers_llm")
        if pref is None:
            continue
        label = str(row.get(key) if key else "overall")
        groups[label].append(float(pref))
    return {label: _bootstrap_ci(values) for label, values in sorted(groups.items())}


def load_pairwise_records(judgments: Path, pairs_path: Path) -> list[dict[str, Any]]:
    pair_lookup = _load_pairs(pairs_path)
    rows: list[dict[str, Any]] = []
    for file in _jsonl_files(judgments):
        for rec in _load_jsonl(file):
            pair = pair_lookup.get(str(rec.get("pair_id") or ""), {})
            llm_id = pair.get("plan_a_id") or rec.get("plan_a_id")
            prog_id = pair.get("plan_b_id") or rec.get("plan_b_id")
            preferred_id = rec.get("preferred_id")
            prefers_llm = None
            if preferred_id == llm_id:
                prefers_llm = 1.0
            elif preferred_id == prog_id:
                prefers_llm = 0.0
            elif str(preferred_id).lower() == "tie":
                prefers_llm = 0.5
            source_model = pair.get("source_model_a") or rec.get("source_model_a")
            judge = rec.get("judge")
            source_family = _family(source_model)
            judge_family = _family(rec.get("judge_model_family") or judge)
            out = dict(rec)
            out.update({
                "llm_plan_id": llm_id,
                "programmatic_plan_id": prog_id,
                "prefers_llm": prefers_llm,
                "source_model_family": source_family,
                "judge_model_family": judge_family,
                "self_family_match": source_family in {"qwen", "gemma"} and source_family == judge_family,
                "structural_score_gap": pair.get("structural_score_gap", rec.get("structural_score_gap")),
            })
            rows.append(out)
    return rows


def run_analysis(judgments: Path, pairs: Path, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    rows = load_pairwise_records(judgments, pairs)
    by_order_counts = Counter(str(r.get("position") or r.get("order")) for r in rows)
    tie_rate = sum(1 for r in rows if r.get("prefers_llm") == 0.5) / len(rows) if rows else None
    p_prefers_left = None
    if rows:
        p_prefers_left = sum(1 for r in rows if r.get("preferred") == "plan_a") / len(rows)

    # Score-gap sensitivity: small/medium/high observed gap strata.
    gap_rows: list[dict[str, Any]] = []
    for row in rows:
        try:
            gap = abs(float(row.get("structural_score_gap")))
        except Exception:
            continue
        bucket = "gap_le_0.5" if gap <= 0.5 else "gap_0.5_to_1.5" if gap <= 1.5 else "gap_gt_1.5"
        row2 = dict(row)
        row2["score_gap_bucket"] = bucket
        gap_rows.append(row2)

    result = {
        "n_records": len(rows),
        "n_unique_pairs": len({r.get("pair_id") for r in rows}),
        "overall_llm_win_rate": _summarize(rows).get("overall", {}),
        "by_judge": _summarize(rows, "judge"),
        "by_source_family": _summarize(rows, "source_model_family"),
        "by_judge_family": _summarize(rows, "judge_model_family"),
        "by_self_family_match": _summarize(rows, "self_family_match"),
        "by_order": _summarize(rows, "position"),
        "by_score_gap_bucket": _summarize(gap_rows, "score_gap_bucket"),
        "order_bias_audit": {"p_prefers_left_position": p_prefers_left, "records_by_order": dict(by_order_counts)},
        "tie_rate": tie_rate,
        "notes": [
            "Tie outcomes are coded as 0.5 in descriptive win-rate summaries.",
            "Self-family analysis is interpretable only for source and judge families both represented in the final data.",
            "Rubric-level analysis should be joined from soft-eval/per-rubric outputs if those are collected.",
        ],
    }
    (output / "pairwise_analysis_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judgments", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = run_analysis(Path(args.judgments), Path(args.pairs), Path(args.output))
    print(json.dumps({k: result[k] for k in ["n_records", "n_unique_pairs", "overall_llm_win_rate", "tie_rate"]}, indent=2))


if __name__ == "__main__":
    main()
