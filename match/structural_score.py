"""Source-neutral structural scoring for matched plan construction.

This module is intentionally independent of TrailTraining's full deterministic
quality scorer. The primary matching estimand requires a score that reflects
weekly training structure, not human-facing prose, citations, rationale length,
or source-identifying generation metadata.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

STRUCTURAL_SCORE_VERSION = "structural_score_v1.0.0"
PRESENTATION_EXCLUSION_NOTE = (
    "Structural score excludes title wording, workout prose richness, purpose "
    "prose richness, citations, claim_attributions, data_notes verbosity, "
    "rationale/explanation fields, source model, generation arm, and file names."
)

_SESSION_QUALITY_TYPES = {
    "interval",
    "intervals",
    "tempo",
    "threshold",
    "hill",
    "hills",
    "fartlek",
    "progression",
    "race",
    "time_trial",
    "quality",
}

_BAND_TOTAL_RANGES: dict[str, tuple[float, float]] = {
    "A1": (140.0, 300.0),
    "A2": (190.0, 390.0),
    "A3": (230.0, 480.0),
    "A4": (270.0, 580.0),
}


@dataclass(frozen=True)
class StructuralScoreReport:
    plan_id: str
    score: float
    components: dict[str, float]
    violations: list[str]
    features: dict[str, Any]
    score_version: str = STRUCTURAL_SCORE_VERSION
    presentation_fields_excluded: bool = True
    exclusion_note: str = PRESENTATION_EXCLUSION_NOTE

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _provenance_sidecar_for(plan_path: Path) -> Path:
    return plan_path.with_name(plan_path.name + ".provenance.json")


def _session_type(day: dict[str, Any]) -> str:
    return str(day.get("session_type") or "").strip().lower()


def _is_quality_day(day: dict[str, Any]) -> bool:
    return bool(day.get("is_hard_day", False)) or _session_type(day) in _SESSION_QUALITY_TYPES


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _range_penalty(value: float, low: float, high: float, under_scale: float, over_scale: float, cap: float) -> float:
    if value < low:
        return min(cap, (low - value) / under_scale)
    if value > high:
        return min(cap, (value - high) / over_scale)
    return 0.0


def _fixture_adjusted_total_range(provenance: dict[str, Any] | None) -> tuple[float, float]:
    provenance = provenance or {}
    band = str(provenance.get("athlete_band") or "").strip().upper()
    readiness = str(provenance.get("readiness") or "").strip().lower()
    recovery = str(provenance.get("recovery_capability") or "").strip().lower()
    phase = str(provenance.get("race_phase") or "").strip().lower()

    low, high = _BAND_TOTAL_RANGES.get(band, (180.0, 420.0))
    if readiness == "high":
        low += 20.0
        high += 40.0
    elif readiness == "low":
        high -= 20.0
    if recovery == "low":
        high -= 30.0
    elif recovery == "high":
        low += 10.0
        high += 20.0
    if phase == "peak":
        low += 10.0
        high += 30.0
    return max(0.0, low), max(low + 20.0, high)


def _expected_hard_range(provenance: dict[str, Any] | None) -> tuple[int, int]:
    provenance = provenance or {}
    band = str(provenance.get("athlete_band") or "").strip().upper()
    readiness = str(provenance.get("readiness") or "").strip().lower()
    recovery = str(provenance.get("recovery_capability") or "").strip().lower()
    phase = str(provenance.get("race_phase") or "").strip().lower()

    hard_min, hard_max = (0, 1)
    if band in {"A3", "A4"}:
        hard_min, hard_max = (1, 2)
    if readiness == "low":
        hard_max = max(0, hard_max - 1)
    if recovery == "low":
        hard_max = min(hard_max, 1)
    if phase == "peak" and readiness == "high" and recovery == "high":
        hard_min = max(hard_min, 1)
    if hard_min > hard_max:
        hard_min = hard_max
    return hard_min, hard_max


def _expected_rest_range(provenance: dict[str, Any] | None) -> tuple[int, int]:
    provenance = provenance or {}
    readiness = str(provenance.get("readiness") or "").strip().lower()
    recovery = str(provenance.get("recovery_capability") or "").strip().lower()

    rest_min, rest_max = (1, 2)
    if recovery == "low":
        rest_min, rest_max = (2, 3)
    if readiness == "low":
        rest_min = min(3, rest_min + 1)
        rest_max = min(4, rest_max + 1)
    return rest_min, rest_max


def extract_structural_features(plan_obj: dict[str, Any], provenance: dict[str, Any] | None = None) -> dict[str, Any]:
    """Extract source-neutral structural features only.

    The returned features deliberately avoid title, workout text, purpose text,
    explanation fields, data-note counts, citations, claim attributions, source
    model, generation arm, and file names.
    """
    provenance = provenance or {}
    days = list(((plan_obj.get("plan") or {}).get("days") or []))
    durations = [_safe_float(day.get("duration_minutes")) for day in days if isinstance(day, dict)]
    session_types = [_session_type(day) for day in days if isinstance(day, dict)]
    rest_flags = [bool(day.get("is_rest_day", False)) for day in days if isinstance(day, dict)]
    hard_flags = [bool(day.get("is_hard_day", False)) for day in days if isinstance(day, dict)]
    quality_flags = [_is_quality_day(day) for day in days if isinstance(day, dict)]

    total_minutes = float(sum(durations))
    active_days = sum(1 for flag in rest_flags if not flag)
    rest_days = sum(1 for flag in rest_flags if flag)
    hard_days = sum(1 for flag in hard_flags if flag)
    quality_days = sum(1 for flag in quality_flags if flag)
    long_runs = sum(1 for st in session_types if st == "long")
    max_day_minutes = max(durations) if durations else 0.0
    mean_day_minutes = total_minutes / len(days) if days else 0.0
    hard_indices = [idx for idx, flag in enumerate(hard_flags) if flag]
    rest_indices = [idx for idx, flag in enumerate(rest_flags) if flag]
    min_hard_spacing = min((b - a for a, b in zip(hard_indices, hard_indices[1:])), default=None)
    min_rest_spacing = min((b - a for a, b in zip(rest_indices, rest_indices[1:])), default=None)

    return {
        "plan_days": len(days),
        "expected_plan_days": int(provenance.get("plan_days") or (plan_obj.get("meta") or {}).get("plan_days") or 7),
        "athlete_band": provenance.get("athlete_band"),
        "readiness": provenance.get("readiness"),
        "recovery_capability": provenance.get("recovery_capability"),
        "race_phase": provenance.get("race_phase"),
        "style": provenance.get("style") or (plan_obj.get("meta") or {}).get("style"),
        "total_minutes": round(total_minutes, 3),
        "weekly_total_hours": round(total_minutes / 60.0, 3),
        "active_days": active_days,
        "rest_days": rest_days,
        "hard_days": hard_days,
        "quality_days": quality_days,
        "long_runs": long_runs,
        "max_day_minutes": round(max_day_minutes, 3),
        "mean_day_minutes": round(mean_day_minutes, 3),
        "session_type_sequence": session_types,
        "rest_day_indices": rest_indices,
        "hard_day_indices": hard_indices,
        "min_hard_spacing_days": min_hard_spacing,
        "min_rest_spacing_days": min_rest_spacing,
    }


def _component_scores(features: dict[str, Any], plan_obj: dict[str, Any], provenance: dict[str, Any] | None) -> tuple[dict[str, float], list[str]]:
    provenance = provenance or {}
    components: dict[str, float] = {
        "plan_length": 0.0,
        "rest_active_consistency": 0.0,
        "duration_plausibility": 0.0,
        "load_band_fit": 0.0,
        "hard_day_fit": 0.0,
        "rest_day_fit": 0.0,
        "long_run_fit": 0.0,
        "quality_day_fit": 0.0,
        "active_day_fit": 0.0,
        "hard_spacing_fit": 0.0,
    }
    violations: list[str] = []

    days = list(((plan_obj.get("plan") or {}).get("days") or []))
    expected_days = int(features["expected_plan_days"] or 7)
    actual_days = int(features["plan_days"] or 0)
    if actual_days != expected_days:
        components["plan_length"] -= abs(actual_days - expected_days) * 8.0
        violations.append("wrong_day_count")

    for idx, day in enumerate(days):
        if not isinstance(day, dict):
            components["rest_active_consistency"] -= 8.0
            violations.append(f"day_{idx}_not_object")
            continue
        duration = _safe_float(day.get("duration_minutes"))
        session_type = _session_type(day)
        is_rest = bool(day.get("is_rest_day", False))
        if is_rest and duration > 0:
            components["rest_active_consistency"] -= 10.0
            violations.append(f"rest_day_positive_duration:{idx}")
        if session_type == "rest" and not is_rest:
            components["rest_active_consistency"] -= 8.0
            violations.append(f"rest_session_not_rest_day:{idx}")
        if duration < 0 or duration > 240:
            components["duration_plausibility"] -= 8.0
            violations.append(f"impossible_duration:{idx}")

    low, high = _fixture_adjusted_total_range(provenance)
    total_minutes = float(features["total_minutes"])
    load_penalty = _range_penalty(total_minutes, low, high, under_scale=7.0, over_scale=10.0, cap=20.0)
    components["load_band_fit"] -= load_penalty
    if load_penalty:
        violations.append("weekly_load_outside_fixture_band")

    hard_min, hard_max = _expected_hard_range(provenance)
    hard_days = int(features["hard_days"])
    if hard_days < hard_min:
        components["hard_day_fit"] -= (hard_min - hard_days) * 8.0
        violations.append("too_few_hard_days")
    if hard_days > hard_max:
        components["hard_day_fit"] -= (hard_days - hard_max) * 8.0
        violations.append("too_many_hard_days")

    rest_min, rest_max = _expected_rest_range(provenance)
    rest_days = int(features["rest_days"])
    if rest_days < rest_min:
        components["rest_day_fit"] -= (rest_min - rest_days) * 5.0
        violations.append("too_few_rest_days")
    if rest_days > rest_max:
        components["rest_day_fit"] -= (rest_days - rest_max) * 5.0
        violations.append("too_many_rest_days")

    active_days = int(features["active_days"])
    if active_days < 3:
        components["active_day_fit"] -= (3 - active_days) * 8.0
        violations.append("too_few_active_days")

    long_runs = int(features["long_runs"])
    if long_runs < 1 and active_days >= 3:
        components["long_run_fit"] -= 8.0
        violations.append("missing_long_run")
    if long_runs > 1:
        components["long_run_fit"] -= (long_runs - 1) * 3.0
        violations.append("multiple_long_runs")

    quality_days = int(features["quality_days"])
    if quality_days > 2:
        components["quality_day_fit"] -= (quality_days - 2) * 4.0
        violations.append("too_many_quality_days")

    hard_indices = list(features.get("hard_day_indices") or [])
    for prev, cur in zip(hard_indices, hard_indices[1:]):
        if int(cur) - int(prev) < 2:
            components["hard_spacing_fit"] -= 5.0
            violations.append("hard_days_too_close")

    return {k: round(max(0.0, 100.0 + v), 3) for k, v in components.items()}, violations


def score_structural_plan(
    plan_path: Path,
    provenance_path: Path | None = None,
    rollups_path: Path | None = None,
) -> StructuralScoreReport:
    """Score a plan for source-neutral structural match quality.

    ``rollups_path`` is accepted for API stability but is not used by v1.0.0;
    the current score is based on plan structure plus fixture/provenance strata.
    """
    plan_path = Path(plan_path)
    plan_obj = _load_json(plan_path) or {}
    if provenance_path is None:
        sidecar = _provenance_sidecar_for(plan_path)
        provenance_path = sidecar if sidecar.exists() else None
    provenance = _load_json(Path(provenance_path)) if provenance_path else None

    features = extract_structural_features(plan_obj, provenance)
    components, violations = _component_scores(features, plan_obj, provenance)
    penalties = sum(100.0 - value for value in components.values())
    total_score = round(max(0.0, min(100.0, 100.0 - penalties)), 3)
    plan_id = str((provenance or {}).get("plan_id") or plan_path.stem)
    return StructuralScoreReport(
        plan_id=plan_id,
        score=total_score,
        components=components,
        violations=sorted(set(violations)),
        features=features,
    )


def score_structural_plan_obj(
    plan_obj: dict[str, Any],
    provenance: dict[str, Any] | None = None,
    *,
    plan_id: str = "<memory>",
) -> StructuralScoreReport:
    features = extract_structural_features(plan_obj, provenance)
    components, violations = _component_scores(features, plan_obj, provenance)
    penalties = sum(100.0 - value for value in components.values())
    return StructuralScoreReport(
        plan_id=plan_id,
        score=round(max(0.0, min(100.0, 100.0 - penalties)), 3),
        components=components,
        violations=sorted(set(violations)),
        features=features,
    )


def score_many(plan_paths: Iterable[Path]) -> list[StructuralScoreReport]:
    return [score_structural_plan(Path(path)) for path in plan_paths]
