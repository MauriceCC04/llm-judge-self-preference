"""generate/study_manifest.py — structured frozen-study manifest."""
from __future__ import annotations

from typing import Any

from fixtures.spec import FIXTURE_IDS
from generate.constants import (
    ACTIVE_JUDGE_NAMES,
    EXPLAINER_MODEL_ID,
    LLM_SOURCE_MODELS,
    PAIRWISE_N_POSITIONS,
    PAIRWISE_N_RUNS,
    TARGET_LLM_PLAN_COUNT,
    TARGET_PAIRS,
    TARGET_PROGRAMMATIC_PLAN_COUNT,
    TARGET_TOTAL_PLAN_COUNT,
)
from generate.exact_counts import exact_count_summary

TRAILTRAINING_PIN_SHA = "3e7f1793ca051ba1aae05f1714d594691202ad7e"


def frozen_study_summary() -> dict[str, Any]:
    exact = exact_count_summary()
    return {
        "fixtures": len(FIXTURE_IDS),
        "fixture_ids": list(FIXTURE_IDS),
        "llm_source_models": list(LLM_SOURCE_MODELS),
        "explainer_model_id": EXPLAINER_MODEL_ID,
        "llm_plans": TARGET_LLM_PLAN_COUNT,
        "programmatic_plans": TARGET_PROGRAMMATIC_PLAN_COUNT,
        "total_plans": TARGET_TOTAL_PLAN_COUNT,
        "target_pairs": TARGET_PAIRS,
        "active_judges": list(ACTIVE_JUDGE_NAMES),
        "n_active_judges": len(ACTIVE_JUDGE_NAMES),
        "pairwise_runs": PAIRWISE_N_RUNS,
        "pairwise_positions": PAIRWISE_N_POSITIONS,
        "expected_pairwise_calls": TARGET_PAIRS * len(ACTIVE_JUDGE_NAMES) * PAIRWISE_N_RUNS * PAIRWISE_N_POSITIONS,
        "expected_soft_eval_calls": TARGET_TOTAL_PLAN_COUNT * len(ACTIVE_JUDGE_NAMES),
        "trailtraining_pin_sha": TRAILTRAINING_PIN_SHA,
        "exact_generation": exact,
    }


def render_study_summary_markdown() -> str:
    summary = frozen_study_summary()
    return "
".join([
        f"- Fixtures: {summary['fixtures']}",
        f"- LLM arm: {summary['llm_plans']} plans",
        f"- Programmatic arm: {summary['programmatic_plans']} plans",
        f"- Total plans: {summary['total_plans']}",
        f"- Target matched pairs: {summary['target_pairs']}",
        f"- Active judges: {summary['n_active_judges']} ({', '.join(summary['active_judges'])})",
        f"- Pairwise calls: {summary['expected_pairwise_calls']}",
        f"- Soft-eval calls: {summary['expected_soft_eval_calls']}",
        f"- trailtraining pin: {summary['trailtraining_pin_sha']}",
    ])
