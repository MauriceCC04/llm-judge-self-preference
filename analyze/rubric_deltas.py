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
        item = item.get("score")
    try:
        return float(item)
    except (TypeError, ValueError):
        return None


def _holm_correct(results: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    raw = []
    order = []
    for rid in RUBRIC_IDS:
        if rid in results:
            p = results[rid].get("pvalue", float("nan"))
            try:
                p = float(p)
            except Exception:
                p = float("nan")
            raw.append(p)
            order.append(rid)

    finite = [(i, p) for i, p in enumerate(raw) if p == p]
    m = len(finite)
    if m == 0:
        for rid in order:
            results[rid]["pvalue_holm"] = float("nan")
            results[rid]["significant"] = False
        return results

    sorted_idx = sorted(finite, key=lambda x: x[1])
    corrected = [float("nan")] * len(raw)
    prev = 0.0
    for rank, (idx, p) in enumerate(sorted_idx):
        val = min(1.0, max(prev, p * (m - rank)))
        corrected[idx] = val
        prev = val

    for i, rid in enumerate(order):
        results[rid]["pvalue_holm"] = corrected[i]
        results[rid]["significant"] = bool(corrected[i] == corrected[i] and corrected[i] < 0.05)
    return results


def _pairs_from_manifest(pairs_path: Path | str) -> list[dict[str, str]]:
    pairs = json.loads(Path(pairs_path).read_text())
    out: list[dict[str, str]] = []
    for pair in pairs:
        llm_plan_id = ""
        prog_plan_id = ""
        if pair.get("arm_a") == "llm":
            llm_plan_id = str(pair.get("plan_a_id") or "")
            prog_plan_id = str(pair.get("plan_b_id") or "")
        elif pair.get("arm_b") == "llm":
            llm_plan_id = str(pair.get("plan_b_id") or "")
            prog_plan_id = str(pair.get("plan_a_id") or "")

        if llm_plan_id and prog_plan_id:
            out.append(
                {
                    "pair_id": str(pair.get("pair_id") or f"{llm_plan_id}__{prog_plan_id}"),
                    "llm_plan_id": llm_plan_id,
                    "programmatic_plan_id": prog_plan_id,
                }
            )
    return out


def _independent_results(df: "pd.DataFrame") -> dict[str, dict[str, Any]]:
    from scipy import stats

    results: dict[str, dict[str, Any]] = {}
    for rid in RUBRIC_IDS:
        llm_scores = []
        prog_scores = []
        for _, row in df.iterrows():
            score = _extract_rubric_score(row, rid)
            if score is None:
                continue
            arm = str(row.get("arm") or "")
            if arm == "llm":
                llm_scores.append(score)
            elif arm == "programmatic":
                prog_scores.append(score)

        if not llm_scores or not prog_scores:
            results[rid] = {
                "delta": float("nan"),
                "pvalue": float("nan"),
                "paired": False,
                "n_llm": len(llm_scores),
                "n_programmatic": len(prog_scores),
            }
            continue

        delta = float(sum(llm_scores) / len(llm_scores) - sum(prog_scores) / len(prog_scores))
        _, pvalue = stats.ttest_ind(llm_scores, prog_scores, equal_var=False)
        results[rid] = {
            "delta": round(delta, 3),
            "pvalue": float(pvalue),
            "paired": False,
            "n_llm": len(llm_scores),
            "n_programmatic": len(prog_scores),
        }

    return _holm_correct(results)


def _paired_results(df: "pd.DataFrame", pairs: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    from scipy import stats

    plan_scores: dict[tuple[str, str], dict[str, float]] = {}
    for _, row in df.iterrows():
        plan_id = str(row.get("plan_id") or "")
        judge = str(row.get("judge") or "")
        if not plan_id or not judge:
            continue
        key = (plan_id, judge)
        plan_scores.setdefault(key, {})
        for rid in RUBRIC_IDS:
            score = _extract_rubric_score(row, rid)
            if score is not None:
                plan_scores[key][rid] = score

    grouped: dict[str, list[float]] = {rid: [] for rid in RUBRIC_IDS}

    for pair in pairs:
        llm_id = pair["llm_plan_id"]
        prog_id = pair["programmatic_plan_id"]
        judges = sorted({j for (pid, j) in plan_scores if pid in {llm_id, prog_id}})
        for judge in judges:
            llm_scores = plan_scores.get((llm_id, judge), {})
            prog_scores = plan_scores.get((prog_id, judge), {})
            for rid in RUBRIC_IDS:
                if rid in llm_scores and rid in prog_scores:
                    grouped[rid].append(float(llm_scores[rid] - prog_scores[rid]))

    results: dict[str, dict[str, Any]] = {}
    for rid in RUBRIC_IDS:
        deltas = grouped[rid]
        if len(deltas) < 2:
            results[rid] = {
                "delta": float("nan"),
                "pvalue": float("nan"),
                "paired": True,
                "n_pairs": len(deltas),
            }
            continue

        delta = float(sum(deltas) / len(deltas))
        _, pvalue = stats.ttest_1samp(deltas, popmean=0.0)
        results[rid] = {
            "delta": round(delta, 3),
            "pvalue": float(pvalue),
            "paired": True,
            "n_pairs": len(deltas),
        }

    return _holm_correct(results)


def rubric_paired_contrasts(
    df: "pd.DataFrame",
    *,
    pairs_path: "Path | str | None" = None,
) -> dict[str, dict[str, Any]]:
    try:
        import pandas as pd  # noqa: F401
    except ImportError as exc:
        raise ImportError("pandas + scipy required for rubric analysis") from exc

    if df.empty or "rubric_scores" not in df.columns:
        return {}

    if pairs_path is None:
        return _independent_results(df)

    pairs = _pairs_from_manifest(pairs_path)
    if not pairs:
        raise ValueError("rubric_paired_contrasts requires a matched-pairs manifest")

    return _paired_results(df, pairs)