"""generate/constants.py — immutable study-wide constants."""
from __future__ import annotations

EXPLAINER_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

# Two local source families for the generation arm.
LLM_SOURCE_MODELS: list[str] = [
    "Qwen/Qwen2.5-7B-Instruct",
    "google/gemma-3-4b-it",
]

PLAN_DAYS = 7
MATCH_TOLERANCE = 1.0
TARGET_PAIRS = 256

STUDY_STYLE = "trailrunning"
DEFAULT_PRIMARY_GOAL = "to become a faster and more durable trail runner"

ACTIVE_JUDGE_NAMES: list[str] = [
    "qwen_7b_judge",
    "qwen_14b_judge",
    "gemma_4b_judge",
    "gemma_12b_judge",
]

PAIRWISE_N_RUNS = 5
PAIRWISE_N_POSITIONS = 2
PILOT_PAIR_LIMIT = 30
PAIRWISE_VIEW_DEFAULT = "raw_normalized"
PAIRWISE_VIEW_CHOICES: tuple[str, ...] = (
    "raw_normalized",
    "canonical_masked",
)

# Revised 32-cell study uses an oversampling profile:
#   * LLM arm: 6 plans per fixture per source model -> 12 raw attempts per cell total
#   * Programmatic arm: 10 raw attempts per fixture
DEFAULT_LLM_PLANS_PER_FIXTURE_PER_MODEL = 6
DEFAULT_PROGRAMMATIC_PLANS_PER_FIXTURE = 10

TARGET_LLM_PLAN_COUNT = 256
TARGET_PROGRAMMATIC_PLAN_COUNT = 256
TARGET_TOTAL_PLAN_COUNT = TARGET_LLM_PLAN_COUNT + TARGET_PROGRAMMATIC_PLAN_COUNT

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

DEFAULT_STYLE_GATE_SUMMARY_PATH = "results/style_audit_summary.json"


def default_plans_per_fixture(arm: str) -> int:
    if arm == "llm":
        return DEFAULT_LLM_PLANS_PER_FIXTURE_PER_MODEL
    if arm == "programmatic":
        return DEFAULT_PROGRAMMATIC_PLANS_PER_FIXTURE
    raise ValueError(f"Unknown arm: {arm!r}")
