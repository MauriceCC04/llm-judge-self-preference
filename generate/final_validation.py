from __future__ import annotations

from typing import Any

from match.filtering import detect_plan_issues

_HARD_TYPES = {"tempo", "intervals", "hills"}
_CANONICAL_TITLES = {
    "rest": "Rest day",
    "easy": "Easy run",
    "aerobic": "Aerobic run",
    "long": "Long run",
    "strength": "Strength session",
    "cross": "Cross-training",
    "tempo": "Tempo session",
    "intervals": "Intervals session",
    "hills": "Hill session",
}
_CANONICAL_PURPOSES = {
    "rest": "Recovery and absorption.",
    "easy": "Build aerobic consistency with low stress.",
    "aerobic": "Build steady aerobic durability.",
    "long": "Build long-run durability and fueling resilience.",
    "strength": "Build strength and resilience for running economy.",
    "cross": "Maintain aerobic load with reduced impact.",
    "tempo": "Develop sustained threshold-like aerobic power.",
    "intervals": "Develop quality and repeatable running economy under load.",
    "hills": "Build climbing strength and uphill efficiency.",
}


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _contains_rest_day_text(*values: Any) -> bool:
    return any("rest day" in _as_text(value).strip().lower() for value in values)


def _canonical_nonrest_title(session_type: str) -> str:
    session_key = (session_type or "").strip().lower()
    return _CANONICAL_TITLES.get(session_key, session_key.replace("_", " ").title() or "Training session")


def _canonical_nonrest_workout(session_type: str, duration_minutes: int) -> str:
    session_key = (session_type or "").strip().lower()
    duration = max(0, int(duration_minutes or 0))
    if session_key == "easy":
        return f"{duration} min easy run on flat to rolling terrain."
    if session_key == "aerobic":
        return f"{duration} min steady aerobic run on mixed terrain."
    if session_key == "long":
        return f"{duration} min long run on rolling to hilly terrain."
    if session_key == "strength":
        return "Strength and mobility session."
    if session_key == "cross":
        return f"{duration} min cross-training session."
    if session_key == "tempo":
        return f"{duration} min tempo session."
    if session_key == "intervals":
        return f"{duration} min intervals session."
    if session_key == "hills":
        return f"{duration} min hill session."
    return f"{duration} min training session."


def normalize_final_plan_semantics(plan_obj: dict[str, Any]) -> dict[str, Any]:
    plan = _as_dict(plan_obj.get("plan"))
    days = [_as_dict(day) for day in _as_list(plan.get("days"))]
    normalized_days = 0

    for day in days:
        session_type = _as_text(day.get("session_type")).strip().lower()
        duration = int(day.get("duration_minutes") or 0)

        if session_type == "rest":
            changed = False
            if _as_text(day.get("title")).strip() != "Rest day":
                day["title"] = "Rest day"
                changed = True
            if _as_text(day.get("workout")).strip() != "Rest day. No structured training.":
                day["workout"] = "Rest day. No structured training."
                changed = True
            if _as_text(day.get("purpose")).strip() != _CANONICAL_PURPOSES["rest"]:
                day["purpose"] = _CANONICAL_PURPOSES["rest"]
                changed = True
            if duration != 0:
                day["duration_minutes"] = 0
                changed = True
            if day.get("is_rest_day") is not True:
                day["is_rest_day"] = True
                changed = True
            if day.get("is_hard_day") is not False:
                day["is_hard_day"] = False
                changed = True
            if _as_text(day.get("target_intensity")).strip().lower() != "rest":
                day["target_intensity"] = "rest"
                changed = True
            if _as_text(day.get("terrain")).strip().lower() != "n/a":
                day["terrain"] = "n/a"
                changed = True
            if changed:
                normalized_days += 1
            continue

        changed = False
        title = _as_text(day.get("title"))
        workout = _as_text(day.get("workout"))
        purpose = _as_text(day.get("purpose"))

        if _contains_rest_day_text(title) or not title.strip():
            day["title"] = _canonical_nonrest_title(session_type)
            changed = True

        if _contains_rest_day_text(workout) or not workout.strip():
            day["workout"] = _canonical_nonrest_workout(session_type, duration)
            changed = True

        if _contains_rest_day_text(purpose):
            day["purpose"] = _CANONICAL_PURPOSES.get(session_type, "")
            changed = True

        if day.get("is_rest_day") is not False:
            day["is_rest_day"] = False
            changed = True

        canonical_hard = session_type in _HARD_TYPES
        if bool(day.get("is_hard_day")) != canonical_hard:
            day["is_hard_day"] = canonical_hard
            changed = True

        if changed:
            normalized_days += 1

    return {
        "normalized_days": normalized_days,
        "issues_after_normalization": detect_plan_issues(plan_obj),
    }
