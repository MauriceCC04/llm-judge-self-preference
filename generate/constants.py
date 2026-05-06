"""generate/constants.py — immutable study-wide constants."""
from __future__ import annotations

EXPLAINER_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

LLM_SOURCE_MODELS: list[str] = [
    "Qwen/Qwen2.5-7B-Instruct",
    "google/gemma-3-4b-it",
]

PLAN_DAYS = 7

# Structural-score tolerance. This is not the old TrailTraining score tolerance.
# A two-point tolerance yielded at least 250 same-cell matches on the current
# 1024-plan Qwen/Gemma/programmatic candidate pool while keeping small structural
# score gaps. Re-audit if the structural score version changes.
STRUCTURAL_MATCH_TOLERANCE = 2.0
MATCH_TOLERANCE = STRUCTURAL_MATCH_TOLERANCE
TARGET_PAIRS = 250
FULL_STUDY_MIN_PAIRS = 250
FULL_STUDY_EXACT_PAIRS = 250
FULL_STUDY_REQUIRED_JUDGES = 4
FULL_STUDY_REQUIRED_PAIRWISE_DOCUMENTS = 10000

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
PAIRWISE_VIEW_DEFAULT = "canonical_masked"
PAIRWISE_VIEW_CHOICES: tuple[str, ...] = (
    "raw_normalized",
    "canonical_masked",
)

DEFAULT_LLM_PLANS_PER_FIXTURE_PER_MODEL = 6
DEFAULT_PROGRAMMATIC_PLANS_PER_FIXTURE = 20

# Current retained candidate pool for the Qwen/Gemma study.
TARGET_LLM_PLAN_COUNT = 384
TARGET_PROGRAMMATIC_PLAN_COUNT = 640
TARGET_TOTAL_PLAN_COUNT = TARGET_LLM_PLAN_COUNT + TARGET_PROGRAMMATIC_PLAN_COUNT

# Structural-only feature distance weights. Do not include prose/narrative fields.
MATCH_FEATURE_WEIGHTS: dict[str, float] = {
    "total_minutes": 0.02,
    "n_rest_days": 2.0,
    "n_hard_days": 2.0,
    "n_active_days": 1.0,
    "n_long_runs": 2.0,
    "n_quality_days": 2.0,
    "max_day_minutes": 0.02,
    "mean_day_minutes": 0.02,
}


# Soft calipers improve target-pair selection without silently reducing full-study
# coverage. They are not hard exclusion rules. If the corresponding hard calipers
# are required, the current 1024-plan pool does not support 250 pairs and needs
# targeted programmatic top-up generation.
MATCH_SOFT_CALIPERS: dict[str, float] = {
    "total_minutes": 100.0,
    "max_day_minutes": 60.0,
    "n_rest_days": 1.0,
    "n_hard_days": 1.0,
    "n_long_runs": 1.0,
    "n_quality_days": 1.0,
}
MATCH_SOFT_CALIPER_PENALTY = 50.0

# Hard calipers are intentionally empty by default because current artifacts do
# not contain enough same-cell overlap to reach 250 pairs under strict calipers.
# Use CLI --caliper feature=value for pilot/sensitivity matching, and regenerate
# targeted programmatic top-ups before treating hard-caliper matching as primary.
MATCH_HARD_CALIPERS: dict[str, float] = {}

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
