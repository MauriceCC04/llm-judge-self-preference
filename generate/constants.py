"""generate/constants.py — immutable study-wide constants."""
from __future__ import annotations

EXPLAINER_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

LLM_SOURCE_MODELS: list[str] = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
]

PLAN_DAYS = 7
MATCH_TOLERANCE = 1.0
TARGET_PAIRS = 250

ACTIVE_JUDGE_NAMES: list[str] = [
    "llama_8b_judge",
    "qwen_7b_judge",
    "qwen_14b_judge",
    "qwen_32b_judge",
]

PAIRWISE_N_RUNS = 5
PAIRWISE_N_POSITIONS = 2
PILOT_PAIR_LIMIT = 30

DEFAULT_LLM_PLANS_PER_FIXTURE_PER_MODEL = 16
DEFAULT_PROGRAMMATIC_PLANS_PER_FIXTURE = 32
EXPECTED_LLM_PLAN_COUNT = 256
EXPECTED_PROGRAMMATIC_PLAN_COUNT = 256
EXPECTED_TOTAL_PLAN_COUNT = EXPECTED_LLM_PLAN_COUNT + EXPECTED_PROGRAMMATIC_PLAN_COUNT

MATCH_FEATURE_WEIGHTS: dict[str, float] = {
    "total_minutes": 2.0,
    "n_rest_days": 1.5,
    "n_hard_days": 1.5,
    "has_long_run": 1.0,
    "max_day_minutes": 1.0,
    "mean_day_minutes": 1.0,
    "narrative_chars": 0.5,
    "mean_workout_chars": 0.5,
    "mean_purpose_chars": 0.5,
    "n_data_notes": 0.5,
}

PAIRWISE_TEXT_CHAR_LIMITS: dict[str, int] = {
    "purpose": 160,
    "workout": 220,
    "readiness_rationale": 220,
    "snapshot_notes": 220,
    "risk_message": 180,
    "data_note": 140,
}

STYLE_AUDIT_Z_THRESHOLD = 0.2
STYLE_AUDIT_CRITICAL_FEATURES: set[str] = {
    "total_chars_narrative",
    "mean_workout_chars",
    "mean_purpose_chars",
    "n_citations",
    "n_claim_attributions",
}


def default_plans_per_fixture(arm: str) -> int:
    if arm == "llm":
        return DEFAULT_LLM_PLANS_PER_FIXTURE_PER_MODEL
    if arm == "programmatic":
        return DEFAULT_PROGRAMMATIC_PLANS_PER_FIXTURE
    raise ValueError(f"Unknown arm: {arm!r}")
