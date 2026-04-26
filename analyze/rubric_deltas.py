"""analyze/rubric_deltas.py — H2 per-rubric paired contrasts with Holm correction."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

RUBRIC_IDS = [
    "goal_alignment",
    "plan_coherence",
    "explanation_quality",
    "caution_proportionality",
    "actionability",
]


def _extract_rubric_score(row: Any, rubric_id: str) -> float | None:
    rs = row.get("rubric_scores") or {}
    item = rs.get(rubric_id)
    if item is None:
        return None
    if isinstance(item, dict):
        val = item.get("score")
    else:
        val = item
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _judge_family(judge_name: str) -> str:
    name = (judge_name or "").lower()
    if "qwen" in name:
        return "qwen"
    if "llama" in name:
        return "llama"
    if "mistral" in name:
        return "mistral"
    return "unknown"


def _holm_correct(results: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    raw_pvalues = [float(results[rid].get("pvalue", 1.0) or 1.0) for rid in RUBRIC_IDS if rid in results]
    rubric_order = [rid for rid in RUBRIC_IDS if rid in results]
    n = len(raw_pvalues)
    if n == 0:
        return results

    sorted_idx = sorted(range(n), key=lambda i: raw_pvalues[i])
    holm_pvals = [1.0] * n
    prev = 0.0
    for rank, idx in enumerate(sorted_idx):
        corrected = raw_pvalues[idx] * (n - rank)
        corrected = max(prev, min(corrected, 1.0))
        holm_pvals[idx] = corrected
        prev = corrected

    for i, rid in enumerate(rubric_order):
        results[rid]["pvalue_holm"] = round(holm_pvals[i], 4)
        results[rid]["significant"] = holm_pvals[i] < 0.05
    return results


def _paired_results_from_deltas(grouped: dict[str, list[float]]) -> dict[str, dict[str, Any]]:
    from scipy import stats

    results: dict[str, dict[str, Any]] = {}
    for rid in RUBRIC_IDS:
        deltas = grouped.get(rid, [])
        if len(deltas) < 2:
            results[rid] = {
                "delta": float("nan"),
                "pvalue": float("nan"),
                "paired": True,
                "n_obs": len(deltas),
            }
            continue
        mean_delta = sum(deltas) / len(deltas)
        _, pval = stats.ttest_1samp(deltas, popmean=0.0)
        results[rid] = {
            "delta": round(float(mean_delta), 3),
            "pvalue": float(pval),
            "paired": True,
            "n_obs": len(deltas),
        }
    return _holm_correct(results)


def rubric_paired_contrasts(
    df: "pd.DataFrame",
    *,
    pairs_path: "Path | str | None" = None,
) -> dict[str, Any]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas + scipy required for rubric analysis") from exc

    if df.empty or "rubric_scores" not in df.columns:
        return {"overall": {}, "by_judge_family": {}, "by_judge": {}, "metadata": {}}

    if pairs_path is None:
        raise ValueError("rubric_paired_contrasts requires a matched-pairs manifest")

    pairs = json.loads(Path(pairs_path).read_text())
    llm_to_prog: dict[str, str] = {}
    for pair in pairs:
        if pair.get("arm_a") == "llm":
            llm_to_prog[str(pair["plan_a_id"])] = str(pair["plan_b_id"])
        if pair.get("arm_b") == "llm":
            llm_to_prog[str(pair["plan_b_id"])] = str(pair["plan_a_id"])

    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        plan_id = str(row.get("plan_id", "") or "")
        judge = str(row.get("judge", "") or "")
        arm = str(row.get("arm", "") or "")
        if not plan_id or not judge or arm not in {"llm", "programmatic"}:
            continue
        for rid in RUBRIC_IDS:
            score = _extract_rubric_score(row, rid)
            if score is None:
                continue
            records.append(
                {
                    "plan_id": plan_id,
                    "judge": judge,
                    "judge_family": _judge_family(judge),
                    "arm": arm,
                    "rubric_id": rid,
                    "score": score,
                }
            )

    if not records:
        return {"overall": {}, "by_judge_family": {}, "by_judge": {}, "metadata": {}}

    long_df = pd.DataFrame(records)
    plan_judge_scores: dict[tuple[str, str], dict[str, float]] = {}
    for _, row in long_df.iterrows():
        key = (str(row["plan_id"]), str(row["judge"]))
        plan_judge_scores.setdefault(key, {})
        plan_judge_scores[key][str(row["rubric_id"])] = float(row["score"])

    overall_grouped: dict[str, list[float]] = {rid: [] for rid in RUBRIC_IDS}
    by_family_grouped: dict[str, dict[str, list[float]]] = {}
    by_judge_grouped: dict[str, dict[str, list[float]]] = {}

    contributing_pairs: set[str] = set()
    for pair in pairs:
        llm_plan_id = ""
        prog_plan_id = ""
        if pair.get("arm_a") == "llm":
            llm_plan_id = str(pair["plan_a_id"])
            prog_plan_id = str(pair["plan_b_id"])
        elif pair.get("arm_b") == "llm":
            llm_plan_id = str(pair["plan_b_id"])
            prog_plan_id = str(pair["plan_a_id"])
        if not llm_plan_id or not prog_plan_id:
            continue

        judges = sorted({judge for plan_id, judge in plan_judge_scores if plan_id in {llm_plan_id, prog_plan_id}})
        for judge in judges:
            llm_scores = plan_judge_scores.get((llm_plan_id, judge), {})
            prog_scores = plan_judge_scores.get((prog_plan_id, judge), {})
            if not llm_scores or not prog_scores:
                continue
            family = _judge_family(judge)
            by_family_grouped.setdefault(family, {rid: [] for rid in RUBRIC_IDS})
            by_judge_grouped.setdefault(judge, {rid: [] for rid in RUBRIC_IDS})
            for rid in RUBRIC_IDS:
                if rid not in llm_scores or rid not in prog_scores:
                    continue
                delta = float(llm_scores[rid] - prog_scores[rid])
                overall_grouped[rid].append(delta)
                by_family_grouped[family][rid].append(delta)
                by_judge_grouped[judge][rid].append(delta)
                contributing_pairs.add(str(pair.get("pair_id") or ""))

    return {
        "overall": _paired_results_from_deltas(overall_grouped),
        "by_judge_family": {
            family: _paired_results_from_deltas(grouped)
            for family, grouped in sorted(by_family_grouped.items())
        },
        "by_judge": {
            judge: _paired_results_from_deltas(grouped)
            for judge, grouped in sorted(by_judge_grouped.items())
        },
        "metadata": {
            "paired": True,
            "n_pairs_with_any_rubric_observation": len([p for p in contributing_pairs if p]),
            "n_soft_eval_rows": int(len(long_df)),
        },
    }
