"""analyze/rubric_deltas.py — H2 per-rubric paired contrasts with Holm correction.

Study design note
-----------------
Soft-eval scores are produced for **every plan** independently.  To use a
paired contrast (as PREREGISTRATION §H2 requires) we join on ``pair_id`` from
the matched-pairs manifest: for each matched pair (llm_plan_id, prog_plan_id)
we compute Δ = score(llm) − score(prog) and test whether mean(Δ) > 0.

If the pairs manifest is not provided (e.g. during unit tests) we fall back to
an independent-samples t-test across all plans in each arm; the results will
then be labelled as unpaired in the returned dict.
"""
from __future__ import annotations

import json
import math
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
    """Pull a float score from a soft-eval DataFrame row."""
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


def rubric_paired_contrasts(
    df: "pd.DataFrame",  # type: ignore[name-defined]
    *,
    pairs_path: "Path | str | None" = None,
) -> dict[str, Any]:
    """H2: per-rubric paired contrasts, Holm-corrected.

    Parameters
    ----------
    df:
        Soft-eval DataFrame with columns: plan_id, judge, arm, rubric_scores.
    pairs_path:
        Path to ``matched_pairs.json``.  When provided, analysis is a true
        **paired** one-sample t-test on (llm_score − prog_score) for each
        matched pair.  When absent, falls back to an independent t-test and
        sets ``paired = False`` in each result dict.

    Returns
    -------
    dict mapping rubric_id → {delta, pvalue, pvalue_holm, significant, paired}
    """
    try:
        import pandas as pd
        import numpy as np
        from scipy import stats
    except ImportError as exc:
        raise ImportError("pandas + scipy required for rubric analysis") from exc

    if df.empty or "rubric_scores" not in df.columns:
        return {}

    # ── Build pairs lookup ─────────────────────────────────────────────────
    pairs_lookup: dict[str, str] | None = None   # llm_plan_id → prog_plan_id
    if pairs_path is not None:
        try:
            pairs = json.loads(Path(pairs_path).read_text())
            pairs_lookup = {p["plan_a_id"]: p["plan_b_id"] for p in pairs
                            if p.get("arm_a") == "llm"}
            # Also map B→A where B is LLM
            for p in pairs:
                if p.get("arm_b") == "llm":
                    pairs_lookup[p["plan_b_id"]] = p["plan_a_id"]
        except Exception:
            pairs_lookup = None

    # Build per-plan mean scores across judges (average out judge variance)
    plan_scores: dict[str, dict[str, list[float]]] = {}  # plan_id → {rubric: [scores]}
    for _, row in df.iterrows():
        pid = str(row.get("plan_id", ""))
        if not pid:
            continue
        plan_scores.setdefault(pid, {})
        for rid in RUBRIC_IDS:
            s = _extract_rubric_score(row, rid)
            if s is not None:
                plan_scores[pid].setdefault(rid, []).append(s)

    # Per-plan mean score per rubric
    plan_mean: dict[str, dict[str, float]] = {
        pid: {rid: sum(vs) / len(vs) for rid, vs in rubs.items()}
        for pid, rubs in plan_scores.items()
    }

    # Arm lookup
    arm_of: dict[str, str] = {}
    if "plan_id" in df.columns and "arm" in df.columns:
        for _, row in df[["plan_id", "arm"]].drop_duplicates().iterrows():
            arm_of[str(row["plan_id"])] = str(row.get("arm", "unknown"))

    results: dict[str, dict[str, Any]] = {}
    raw_pvalues: list[float] = []

    for rid in RUBRIC_IDS:
        if pairs_lookup is not None:
            # ── True paired test ──────────────────────────────────────────
            deltas: list[float] = []
            for llm_id, prog_id in pairs_lookup.items():
                llm_s = plan_mean.get(llm_id, {}).get(rid)
                prog_s = plan_mean.get(prog_id, {}).get(rid)
                if llm_s is not None and prog_s is not None:
                    deltas.append(llm_s - prog_s)

            if len(deltas) < 2:
                results[rid] = {"delta": float("nan"), "pvalue": float("nan"),
                                "paired": True, "n_pairs": len(deltas)}
                raw_pvalues.append(1.0)
                continue

            delta_arr = list(deltas)
            mean_delta = sum(delta_arr) / len(delta_arr)
            _, pval = stats.ttest_1samp(delta_arr, popmean=0)
            results[rid] = {
                "delta": round(float(mean_delta), 3),
                "pvalue": float(pval),
                "paired": True,
                "n_pairs": len(deltas),
            }
            raw_pvalues.append(float(pval))

        else:
            # ── Fallback: independent test ────────────────────────────────
            llm_scores = [
                plan_mean[pid][rid]
                for pid in plan_mean
                if arm_of.get(pid) == "llm" and rid in plan_mean[pid]
            ]
            prog_scores = [
                plan_mean[pid][rid]
                for pid in plan_mean
                if arm_of.get(pid) == "programmatic" and rid in plan_mean[pid]
            ]

            if len(llm_scores) < 2 or len(prog_scores) < 2:
                results[rid] = {"delta": float("nan"), "pvalue": float("nan"),
                                "paired": False}
                raw_pvalues.append(1.0)
                continue

            delta = float(sum(llm_scores) / len(llm_scores) -
                          sum(prog_scores) / len(prog_scores))
            _, pval = stats.ttest_ind(llm_scores, prog_scores)
            results[rid] = {
                "delta": round(delta, 3),
                "pvalue": float(pval),
                "paired": False,
                "n_llm": len(llm_scores),
                "n_prog": len(prog_scores),
            }
            raw_pvalues.append(float(pval))

    # ── Holm-Bonferroni correction ─────────────────────────────────────────
    n = len(raw_pvalues)
    sorted_idx = sorted(range(n), key=lambda i: raw_pvalues[i])
    holm_pvals = [1.0] * n
    prev = 0.0
    for rank, idx in enumerate(sorted_idx):
        corrected = raw_pvalues[idx] * (n - rank)
        corrected = max(prev, min(corrected, 1.0))
        holm_pvals[idx] = corrected
        prev = corrected

    for i, rid in enumerate(RUBRIC_IDS):
        if rid in results:
            results[rid]["pvalue_holm"] = round(holm_pvals[i], 4)
            results[rid]["significant"] = holm_pvals[i] < 0.05

    return results
