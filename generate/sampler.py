"""generate/sampler.py — structural sampler for the programmatic arm.

This version keeps the repo's existing structural-sampling approach but also
lets fixture metadata shape prompt-visible plan metadata and mild structural
priors so base vs peak contexts remain meaningfully different.
"""
from __future__ import annotations

import datetime as dt
import random
from dataclasses import dataclass, field, replace
from typing import Any

from trailtraining.llm.schemas import ensure_machine_plan_shape

_HARD_TYPES = ("tempo", "intervals", "hills")
_EASY_TYPES = ("easy", "aerobic", "long", "strength", "cross")
_NON_REST_TYPES = _HARD_TYPES + _EASY_TYPES


@dataclass
class StructuralSamplerConfig:
    """Controls marginal distributions for the sampled machine plan."""

    plan_days: int = 7
    seed: int = 0

    p_hard_day: float = 0.25
    p_rest_day: float = 0.14

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

    easy_type_probs: dict[str, float] = field(default_factory=lambda: {
        "easy": 0.45,
        "aerobic": 0.25,
        "long": 0.15,
        "strength": 0.08,
        "cross": 0.07,
    })

    today: str = "2026-03-17"
    plan_start: str = "2026-03-18"
    style: str = "trailrunning"
    primary_goal: str = "to become a faster trail runner"
    lifestyle_notes: str = ""
    readiness_status: str = "steady"

    # Keep these in the config so fixture phase can still influence the sampler,
    # but do not emit them into MachinePlanArtifact.meta because trailtraining's
    # schema forbids extra meta keys.
    block_label: str = "base development block"
    weeks_to_race: int = 12


def _clamp_int(x: float, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(round(x))))


def _sample_session_type_non_hard(rng: random.Random, cfg: StructuralSamplerConfig) -> str:
    types = list(cfg.easy_type_probs.keys())
    weights = [cfg.easy_type_probs[t] for t in types]
    return rng.choices(types, weights=weights, k=1)[0]


def _sample_duration(rng: random.Random, session_type: str, cfg: StructuralSamplerConfig) -> int:
    if session_type == "rest":
        return 0
    mean, std = cfg.duration_by_type.get(session_type, (45.0, 10.0))
    sampled = rng.gauss(mean, std)
    return _clamp_int(sampled, 20, 420)


def sampler_config_from_fixture_meta(
    fixture_meta: dict[str, Any],
    *,
    seed: int,
    plan_days: int,
    today: str,
    plan_start: str,
    base_cfg: StructuralSamplerConfig | None = None,
) -> StructuralSamplerConfig:
    """Return a fixture-aware sampler config.

    The config stays close to any fitted priors from the LLM arm but makes the
    training phase and readiness status visible in the structural sampler.
    """
    cfg = replace(base_cfg) if base_cfg is not None else StructuralSamplerConfig()
    cfg.plan_days = plan_days
    cfg.seed = seed
    cfg.today = today
    cfg.plan_start = plan_start
    cfg.style = str(fixture_meta.get("style", cfg.style) or cfg.style)
    cfg.primary_goal = str(fixture_meta.get("primary_goal", cfg.primary_goal) or cfg.primary_goal)
    cfg.lifestyle_notes = str(fixture_meta.get("lifestyle_notes", cfg.lifestyle_notes) or cfg.lifestyle_notes)
    cfg.block_label = str(fixture_meta.get("block_label", cfg.block_label) or cfg.block_label)
    cfg.readiness_status = str(fixture_meta.get("readiness_status", cfg.readiness_status) or cfg.readiness_status)
    cfg.weeks_to_race = int(fixture_meta.get("weeks_to_race", cfg.weeks_to_race) or cfg.weeks_to_race)

    race_phase = str(fixture_meta.get("race_phase", "base") or "base")
    if race_phase == "peak":
        cfg.p_rest_day = min(0.28, max(cfg.p_rest_day, 0.18))
        cfg.p_hard_day = max(0.18, min(cfg.p_hard_day, 0.24))
        cfg.easy_type_probs.update({
            "easy": 0.40,
            "aerobic": 0.18,
            "long": 0.13,
            "strength": 0.06,
            "cross": 0.23,
        })
        cfg.duration_by_type["long"] = (80.0, 14.0)
        cfg.duration_by_type["tempo"] = (46.0, 7.0)
    else:
        cfg.p_rest_day = min(0.22, max(cfg.p_rest_day, 0.14))
        cfg.p_hard_day = min(0.28, max(cfg.p_hard_day, 0.22))
        cfg.easy_type_probs.update({
            "easy": 0.34,
            "aerobic": 0.24,
            "long": 0.22,
            "strength": 0.10,
            "cross": 0.10,
        })
        cfg.duration_by_type["long"] = (96.0, 18.0)
        cfg.duration_by_type["hills"] = (52.0, 8.0)

    if cfg.readiness_status == "fatigued":
        cfg.p_rest_day = min(0.32, cfg.p_rest_day + 0.04)
        cfg.p_hard_day = max(0.14, cfg.p_hard_day - 0.05)
        cfg.duration_by_type["intervals"] = (46.0, 7.0)
        cfg.duration_by_type["tempo"] = (44.0, 7.0)
    elif cfg.readiness_status == "primed":
        cfg.p_hard_day = min(0.30, cfg.p_hard_day + 0.02)

    total = sum(cfg.easy_type_probs.values())
    if total > 0:
        cfg.easy_type_probs = {k: v / total for k, v in cfg.easy_type_probs.items()}
    return cfg


def sample_machine_plan(
    cfg: StructuralSamplerConfig,
    *,
    combined: list[dict[str, Any]] | None = None,
    rollups: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rng = random.Random(cfg.seed)
    n = cfg.plan_days

    is_rest = [rng.random() < cfg.p_rest_day for _ in range(n)]
    is_hard = [(not rest) and (rng.random() < cfg.p_hard_day) for rest in is_rest]

    if not any(is_rest):
        pick = rng.randrange(n)
        is_rest[pick] = True
        is_hard[pick] = False
    if not any(not rest for rest in is_rest):
        pick = rng.randrange(n)
        is_rest[pick] = False

    base_date = dt.date.fromisoformat(cfg.plan_start)
    days: list[dict[str, Any]] = []

    for i in range(n):
        current_date = base_date + dt.timedelta(days=i)
        if is_rest[i]:
            days.append({
                "date": current_date.isoformat(),
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
            continue

        if is_hard[i]:
            session_type = rng.choice(list(_HARD_TYPES))
            target_intensity = "threshold"
        else:
            session_type = _sample_session_type_non_hard(rng, cfg)
            target_intensity = "easy" if session_type not in {"tempo", "intervals", "hills"} else "threshold"

        duration = _sample_duration(rng, session_type, cfg)
        terrain = "trail" if session_type in {"easy", "aerobic", "long", "hills"} else "road"
        days.append({
            "date": current_date.isoformat(),
            "session_type": session_type,
            "is_rest_day": False,
            "is_hard_day": session_type in _HARD_TYPES,
            "duration_minutes": duration,
            "target_intensity": target_intensity,
            "terrain": terrain,
            "workout": f"{duration} min {session_type} session.",
            "estimated_distance_km": None,
            "estimated_elevation_m": None,
        })

    total_min = sum(day["duration_minutes"] for day in days)
    skeleton = {
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
    return ensure_machine_plan_shape(skeleton)


def fit_sampler_config_from_plans(
    plan_dicts: list[dict[str, Any]],
    *,
    seed: int = 0,
) -> StructuralSamplerConfig:
    hard_counts: list[int] = []
    rest_counts: list[int] = []
    type_counts: dict[str, int] = {}
    duration_observations: dict[str, list[float]] = {}

    for plan in plan_dicts:
        days = (plan.get("plan") or {}).get("days") or []
        if not days:
            continue
        hard_counts.append(sum(1 for day in days if day.get("is_hard_day")))
        rest_counts.append(sum(1 for day in days if day.get("is_rest_day")))
        for day in days:
            session_type = day.get("session_type") or "easy"
            if session_type == "rest":
                continue
            type_counts[session_type] = type_counts.get(session_type, 0) + 1
            duration = day.get("duration_minutes")
            if isinstance(duration, (int, float)) and float(duration) > 0:
                duration_observations.setdefault(session_type, []).append(float(duration))

    total_days = sum(len((plan.get("plan") or {}).get("days") or []) for plan in plan_dicts) or 1
    p_hard = sum(hard_counts) / total_days if total_days else 0.25
    p_rest = sum(rest_counts) / total_days if total_days else 0.14

    easy_total = sum(v for k, v in type_counts.items() if k in _EASY_TYPES) or 1
    easy_type_probs = {k: max(0.01, type_counts.get(k, 0) / easy_total) for k in _EASY_TYPES}
    easy_norm = sum(easy_type_probs.values())
    easy_type_probs = {k: v / easy_norm for k, v in easy_type_probs.items()}

    import math

    duration_by_type: dict[str, tuple[float, float]] = {}
    for session_type in _NON_REST_TYPES:
        values = duration_observations.get(session_type, [])
        if len(values) >= 2:
            avg = sum(values) / len(values)
            sd = math.sqrt(sum((v - avg) ** 2 for v in values) / len(values))
            duration_by_type[session_type] = (avg, max(5.0, sd))
        elif len(values) == 1:
            duration_by_type[session_type] = (values[0], 10.0)

    cfg = StructuralSamplerConfig(seed=seed)
    cfg.p_hard_day = p_hard
    cfg.p_rest_day = p_rest
    cfg.easy_type_probs = easy_type_probs
    if duration_by_type:
        cfg.duration_by_type.update(duration_by_type)
    return cfg
