from __future__ import annotations

from typing import Any

SESSION_TITLES = {
    "rest": "Rest day",
    "easy": "Easy run",
    "aerobic": "Aerobic run",
    "long": "Long run",
    "strength": "Strength session",
    "tempo": "Tempo session",
    "intervals": "Intervals session",
    "hills": "Hill session",
    "cross": "Cross-training",
}

SESSION_WORKOUTS = {
    "rest": "Rest day. No structured training.",
    "easy": "{duration} min easy run at conversational effort.",
    "aerobic": "{duration} min aerobic run at steady, controlled effort.",
    "long": "{duration} min long run at controlled aerobic effort.",
    "strength": "{duration} min strength and mobility session.",
    "tempo": "{duration} min tempo session at controlled hard effort.",
    "intervals": "{duration} min intervals session with controlled recoveries.",
    "hills": "{duration} min hill session with steady climbing efforts.",
    "cross": "{duration} min low-impact cross-training session.",
}

SESSION_PURPOSES = {
    "rest": "Absorb training and maintain freshness.",
    "easy": "Build aerobic consistency while keeping fatigue low.",
    "aerobic": "Develop aerobic endurance with controlled load.",
    "long": "Extend endurance and trail-specific durability.",
    "strength": "Support durability, stability, and injury resistance.",
    "tempo": "Develop sustainable threshold strength without excessive fatigue.",
    "intervals": "Develop controlled high-end aerobic power.",
    "hills": "Build climbing strength and trail-specific power.",
    "cross": "Add aerobic stimulus with reduced impact.",
}

HARD_TYPES = {"tempo", "intervals", "hills"}


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(float(value)))
    except Exception:
        return default


def normalize_programmatic_artifact_text(plan_obj: dict[str, Any]) -> dict[str, Any]:
    """Align human-facing text with trusted programmatic structure.

    This does not alter the intended structure except to enforce rest/hard
    consistency from session_type and is_rest_day.
    """
    plan = plan_obj.get("plan")
    if not isinstance(plan, dict):
        return plan_obj

    days = plan.get("days")
    if not isinstance(days, list):
        return plan_obj

    for day in days:
        if not isinstance(day, dict):
            continue

        session_type = str(day.get("session_type") or "").strip().lower()
        is_rest = bool(day.get("is_rest_day")) or session_type == "rest"

        if is_rest:
            session_type = "rest"
            day["session_type"] = "rest"
            day["is_rest_day"] = True
            day["is_hard_day"] = False
            day["duration_minutes"] = 0
            day["target_intensity"] = "rest"
            day["terrain"] = day.get("terrain") or "n/a"
        else:
            if session_type not in SESSION_TITLES:
                session_type = "easy"
                day["session_type"] = session_type
            day["is_rest_day"] = False
            day["is_hard_day"] = session_type in HARD_TYPES
            duration = max(1, _as_int(day.get("duration_minutes"), 30))
            day["duration_minutes"] = duration
            if not str(day.get("target_intensity") or "").strip() or day.get("target_intensity") == "rest":
                day["target_intensity"] = "threshold" if session_type in HARD_TYPES else "easy"

        duration = _as_int(day.get("duration_minutes"), 0)
        day["title"] = SESSION_TITLES.get(session_type, "Training session")
        day["workout"] = SESSION_WORKOUTS.get(
            session_type,
            "{duration} min training session.",
        ).format(duration=duration)
        day["purpose"] = SESSION_PURPOSES.get(
            session_type,
            "Complete the planned training while respecting current readiness.",
        )

    total_min = sum(_as_int(day.get("duration_minutes"), 0) for day in days if isinstance(day, dict))
    weekly_totals = plan.setdefault("weekly_totals", {})
    if isinstance(weekly_totals, dict):
        weekly_totals["planned_moving_time_hours"] = round(total_min / 60.0, 2)

    return plan_obj
