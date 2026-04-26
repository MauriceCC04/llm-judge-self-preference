from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def _safe_text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def extract_match_features(plan_obj: dict[str, Any]) -> dict[str, float]:
    plan = plan_obj.get("plan") or {}
    days = plan.get("days") or []
    durations = [float(d.get("duration_minutes") or 0) for d in days if isinstance(d, dict)]
    rest = sum(1 for d in days if isinstance(d, dict) and d.get("is_rest_day", False))
    hard = sum(1 for d in days if isinstance(d, dict) and d.get("is_hard_day", False))
    has_long = any((d.get("session_type") == "long") for d in days if isinstance(d, dict))
    purposes = [_safe_text(d.get("purpose")) for d in days if isinstance(d, dict)]
    workouts = [_safe_text(d.get("workout")) for d in days if isinstance(d, dict)]
    narrative = "".join(purposes + workouts)
    return {
        "total_minutes": float(sum(durations)),
        "n_rest_days": float(rest),
        "n_hard_days": float(hard),
        "has_long_run": float(1 if has_long else 0),
        "max_day_minutes": float(max(durations) if durations else 0.0),
        "mean_day_minutes": float(sum(durations) / len(durations)) if durations else 0.0,
        "narrative_chars": float(len(narrative)),
        "mean_workout_chars": float(sum(len(x) for x in workouts) / len(workouts)) if workouts else 0.0,
        "mean_purpose_chars": float(sum(len(x) for x in purposes) / len(purposes)) if purposes else 0.0,
        "n_data_notes": float(len(plan_obj.get("data_notes") or [])),
    }


def load_match_features(plan_path: Path) -> dict[str, float]:
    return extract_match_features(json.loads(plan_path.read_text(encoding="utf-8")))


def weighted_feature_distance(left: Mapping[str, float], right: Mapping[str, float], *, weights: Mapping[str, float]) -> float:
    return sum(abs(float(left.get(k, 0.0)) - float(right.get(k, 0.0))) * float(w) for k, w in weights.items())
