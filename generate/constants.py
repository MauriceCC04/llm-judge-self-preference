"""generate/constants.py — immutable study-wide constants.

EXPLAINER_MODEL_ID is the single point of truth for which model fills narrative
fields in both arms.  It is asserted at generation time (test 8) so a drift
between arms is caught before any expensive generation run.
"""
from __future__ import annotations

# ── Explainer (shared across both arms — DO NOT change mid-study) ────────────
# Must match the panel table in PROMPT.md.
EXPLAINER_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

# ── LLM arm source models ────────────────────────────────────────────────────
LLM_SOURCE_MODELS: list[str] = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
]

# ── Plan sizes ───────────────────────────────────────────────────────────────
PLAN_DAYS = 7

# ── Matching ─────────────────────────────────────────────────────────────────
MATCH_TOLERANCE = 1.0   # |Δscore| ≤ 1 within (fixture_id, round(score)) bins
TARGET_PAIRS = 250
