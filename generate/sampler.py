"""generate/sampler.py — structural sampler for the programmatic arm.

``sample_machine_plan`` draws a random MachinePlanArtifact dict whose
structural marginals (hard days per week, rest days, session-type frequencies)
are controlled by ``StructuralSamplerConfig``.

Priors are either set manually or fitted from LLM-arm outputs via
``fit_sampler_config_from_plans``.  Fitting from LLM marginals prevents
"programmatic plans lost because they looked structurally weird."
"""
from __future__ import annotations

import datetime as dt
import random
from dataclasses import dataclass, field
from typing import Any

from trailtraining.llm.schemas import ensure_machine_plan_shape

# ── Valid session types (non-rest) ────────────────────────────────────────────

_HARD_TYPES = ("tempo", "intervals", "hills")
_EASY_TYPES = ("easy", "aerobic", "long", "strength", "cross")
_NON_REST_TYPES = _HARD_TYPES + _EASY_TYPES


@dataclass
class StructuralSamplerConfig:
    """Controls marginal distributions for the sampled machine plan."""

    plan_days: int = 7
    seed: int = 0

    # Hard days in the 7-day window: sampled from Binomial(plan_days, p_hard)
    p_hard_day: float = 0.25          # ~1.75 hard days per 7; guardrails cap at 3
    p_rest_day: float = 0.14          # ~1 rest per 7; guardrails enforce ≥1

    # Duration distributions by session type (mean minutes, std)
    duration_by_type: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        "easy": (45.0, 10.0),
        "aerobic": (50.0, 10.0),
        "long": (90.0, 20.0),
        "tempo": (50.0, 8.0),
        "intervals": (55.0, 8.0),
        "hills": (50.0, 8.0),
        "strength": (35.0, 8.0),
        "cross": (45.0, 10.0),
    })

    # Session-type frequencies among non-rest, non-hard days
    easy_type_probs: dict[str, float] = field(default_factory=lambda: {
        "easy": 0.45,
        "aerobic": 0.25,
        "long": 0.15,
        "strength": 0.08,
        "cross": 0.07,
    })

    # Fixed plan metadata (overridden at generation time from the fixture)
    today: str = "2026-03-17"
    plan_start: str = "2026-03-18"
    style: str = "trailrunning"
    primary_goal: str = "to become a faster trail runner"
    lifestyle_notes: str = ""
    readiness_status: str = "steady"  # "primed" | "steady" | "fatigued"


def _clamp_int(x: float, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(round(x))))


def _sample_session_type_non_hard(
    rng: random.Random, cfg: StructuralSamplerConfig
) -> str:
    types = list(cfg.easy_type_probs.keys())
    weights = [cfg.easy_type_probs[t] for t in types]
    return rng.choices(types, weights=weights, k=1)[0]


def _sample_duration(rng: random.Random, session_type: str, cfg: StructuralSamplerConfig) -> int:
    if session_type == "rest":
        return 0
    mean, std = cfg.duration_by_type.get(session_type, (45.0, 10.0))
    sampled = rng.gauss(mean, std)
    return _clamp_int(sampled, 20, 420)


def sample_machine_plan(
    cfg: StructuralSamplerConfig,
    *,
    combined: list[dict[str, Any]] | None = None,
    rollups: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a MachinePlanArtifact dict drawn from *cfg*.

    The dict is **not** yet guardrailed; callers must pipe it through
    ``apply_eval_coach_guardrails`` before passing to the explainer.

    Args:
        cfg: Sampler configuration (marginals + metadata).
        combined: Ignored here but accepted for signature compatibility.
        rollups: Ignored here but accepted for signature compatibility.
    """
    rng = random.Random(cfg.seed)
    n = cfg.plan_days

    # Step 1: decide which days are rest / hard / easy
    is_rest = [rng.random() < cfg.p_rest_day for _ in range(n)]
    is_hard = [
        (not r) and (rng.random() < cfg.p_hard_day)
        for r in is_rest
    ]

    # Ensure at least 1 rest day and at least 1 non-rest day in 7d window
    if not any(is_rest):
        pick = rng.randrange(n)
        is_rest[pick] = True
        is_hard[pick] = False
    if not any(not r for r in is_rest):
        pick = rng.randrange(n)
        is_rest[pick] = False

    base_date = dt.date.fromisoformat(cfg.plan_start)
    days: list[dict[str, Any]] = []

    for i in range(n):
        d = base_date + dt.timedelta(days=i)
        if is_rest[i]:
            days.append({
                "date": d.isoformat(),
                "session_type": "rest",
                "is_rest_day": True,
                "is_hard_day": False,
                "duration_minutes": 0,
                "target_intensity": "rest",
                "terrain": "n/a",
                "workout": "Rest day. No structured training.",
                "estimated_distance_km": None,
                "estimated_elevation_m": None,
            })
        elif is_hard[i]:
            st = rng.choice(list(_HARD_TYPES))
            dur = _sample_duration(rng, st, cfg)
            days.append({
                "date": d.isoformat(),
                "session_type": st,
                "is_rest_day": False,
                "is_hard_day": True,
                "duration_minutes": dur,
                "target_intensity": "threshold",
                "terrain": "trail",
                "workout": f"{dur} min {st} session.",
                "estimated_distance_km": None,
                "estimated_elevation_m": None,
            })
        else:
            st = _sample_session_type_non_hard(rng, cfg)
            dur = _sample_duration(rng, st, cfg)
            days.append({
                "date": d.isoformat(),
                "session_type": st,
                "is_rest_day": False,
                "is_hard_day": False,
                "duration_minutes": dur,
                "target_intensity": "easy",
                "terrain": "trail" if st in ("long", "aerobic", "easy") else "road",
                "workout": f"{dur} min {st} session.",
                "estimated_distance_km": None,
                "estimated_elevation_m": None,
            })

    total_min = sum(d["duration_minutes"] for d in days)
    return {
        "meta": {
            "today": cfg.today,
            "plan_start": cfg.plan_start,
            "plan_days": n,
            "style": cfg.style,
            "primary_goal": cfg.primary_goal,
            "lifestyle_notes": cfg.lifestyle_notes,
        },
        "readiness": {"status": cfg.readiness_status},
        "plan": {
            "weekly_totals": {
                "planned_distance_km": None,
                "planned_moving_time_hours": round(total_min / 60.0, 2),
                "planned_elevation_m": None,
            },
            "days": days,
        },
    }


def fit_sampler_config_from_plans(
    plan_dicts: list[dict[str, Any]],
    *,
    seed: int = 0,
) -> StructuralSamplerConfig:
    """Fit marginals from a list of TrainingPlanArtifact dicts (LLM-arm outputs).

    Returns a StructuralSamplerConfig whose priors match the LLM arm's
    structural summary statistics — the key control preventing structural
    confounding.
    """
    hard_counts: list[int] = []
    rest_counts: list[int] = []
    type_counts: dict[str, int] = {}
    durations_by_type: dict[str, list[float]] = {}

    for plan in plan_dicts:
        days = (plan.get("plan") or {}).get("days") or []
        if not days:
            continue
        n = len(days)
        hard_counts.append(sum(1 for d in days if d.get("is_hard_day")))
        rest_counts.append(sum(1 for d in days if d.get("is_rest_day")))
        for d in days:
            st = d.get("session_type") or "easy"
            if st == "rest":
                continue
            type_counts[st] = type_counts.get(st, 0) + 1
            dur = d.get("duration_minutes")
            if isinstance(dur, (int, float)) and float(dur) > 0:
                durations_by_type.setdefault(st, []).append(float(dur))

    n_plans = len(plan_dicts) or 1
    total_days = sum(
        len((p.get("plan") or {}).get("days") or []) for p in plan_dicts
    ) or 1

    p_hard = sum(hard_counts) / total_days if total_days else 0.25
    p_rest = sum(rest_counts) / total_days if total_days else 0.14

    # Easy-type probabilities (normalised)
    easy_total = sum(v for k, v in type_counts.items() if k in _EASY_TYPES) or 1
    easy_type_probs = {
        k: type_counts.get(k, 0) / easy_total
        for k in _EASY_TYPES
    }
    # Ensure all types have non-zero probability
    for k in _EASY_TYPES:
        easy_type_probs[k] = max(0.01, easy_type_probs.get(k, 0.01))
    # Re-normalise
    tot = sum(easy_type_probs.values())
    easy_type_probs = {k: v / tot for k, v in easy_type_probs.items()}

    # Duration distributions
    import math
    duration_by_type: dict[str, tuple[float, float]] = {}
    for st in _NON_REST_TYPES:
        vals = durations_by_type.get(st, [])
        if len(vals) >= 2:
            mean = sum(vals) / len(vals)
            std = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))
            duration_by_type[st] = (mean, max(5.0, std))
        elif len(vals) == 1:
            duration_by_type[st] = (vals[0], 10.0)
        else:
            # fall back to defaults
            pass

    cfg = StructuralSamplerConfig(seed=seed)
    cfg.p_hard_day = p_hard
    cfg.p_rest_day = p_rest
    cfg.easy_type_probs = easy_type_probs
    if duration_by_type:
        cfg.duration_by_type.update(duration_by_type)

    return cfg
