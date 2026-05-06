from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def _session_type(day: dict[str, Any]) -> str:
    return str(day.get("session_type") or "").strip().lower()


def extract_match_features(plan_obj: dict[str, Any]) -> dict[str, float]:
    """Extract structural-only features for distance matching.

    Deliberately excludes title text, workout prose, purpose prose, citations,
    claim_attributions, data_notes, explanation/rationale text, source labels,
    generation arm, and file names. Presentation leakage must be audited, not
    optimized into the primary match distance.
    """
    days = list(((plan_obj.get("plan") or {}).get("days") or []))
    durations = [float(day.get("duration_minutes") or 0) for day in days if isinstance(day, dict)]
    rest = sum(1 for day in days if isinstance(day, dict) and day.get("is_rest_day", False))
    hard = sum(1 for day in days if isinstance(day, dict) and day.get("is_hard_day", False))
    active = sum(1 for day in days if isinstance(day, dict) and not day.get("is_rest_day", False))
    long_runs = sum(1 for day in days if isinstance(day, dict) and _session_type(day) == "long")
    quality_types = {"interval", "intervals", "tempo", "threshold", "hill", "hills", "fartlek", "progression", "race", "time_trial", "quality"}
    quality = sum(
        1
        for day in days
        if isinstance(day, dict)
        and (day.get("is_hard_day", False) or _session_type(day) in quality_types)
    )
    return {
        "total_minutes": float(sum(durations)),
        "n_rest_days": float(rest),
        "n_hard_days": float(hard),
        "n_active_days": float(active),
        "n_long_runs": float(long_runs),
        "n_quality_days": float(quality),
        "max_day_minutes": float(max(durations) if durations else 0.0),
        "mean_day_minutes": float(sum(durations) / len(durations)) if durations else 0.0,
    }


def load_match_features(plan_path: Path) -> dict[str, float]:
    return extract_match_features(json.loads(plan_path.read_text(encoding="utf-8")))


def weighted_feature_distance(left: Mapping[str, float], right: Mapping[str, float], *, weights: Mapping[str, float]) -> float:
    return sum(abs(float(left.get(k, 0.0)) - float(right.get(k, 0.0))) * float(w) for k, w in weights.items())
