"""Source-symmetric presentation normalization for saved plan artifacts.

This module is intentionally conservative: it canonicalizes judge-visible day
presentation from already-accepted structured fields, while preserving the
score-relevant structural projection of the artifact. It is meant to run after
artifact generation and before deterministic scoring, matching, and judging.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

NORMALIZER_VERSION = "artifact-presentation-normalizer-v1"
EXPECTED_PLAN_DAYS = 7

CANONICAL_TITLES: dict[str, str] = {
    "rest": "Rest day",
    "easy": "Easy run",
    "aerobic": "Aerobic run",
    "long": "Long run",
    "tempo": "Tempo run",
    "intervals": "Interval session",
    "hills": "Hill session",
    "strength": "Strength session",
    "cross": "Cross-training",
}

CANONICAL_WORKOUTS: dict[str, str] = {
    "rest": "Rest day. No structured training.",
    "easy": "Easy run at comfortable conversational effort.",
    "aerobic": "Aerobic run at steady controlled effort.",
    "long": "Long run. Build endurance at controlled effort.",
    "tempo": "Tempo run. Sustain controlled hard effort with adequate warm-up and cool-down.",
    "intervals": "Interval session. Use repeated faster efforts with easy recoveries.",
    "hills": "Hill session. Use controlled uphill efforts with easy recoveries.",
    "strength": "Strength session. Complete supportive strength and mobility work.",
    "cross": "Cross-training. Use low-impact aerobic training.",
}

CANONICAL_PURPOSES: dict[str, str] = {
    "rest": "Recover and maintain freshness.",
    "easy": "Support aerobic consistency without adding excessive fatigue.",
    "aerobic": "Develop aerobic durability while keeping effort controlled.",
    "long": "Build endurance and trail-specific durability.",
    "tempo": "Improve sustainable effort and race-specific rhythm.",
    "intervals": "Develop speed, economy, and pacing control.",
    "hills": "Build climbing strength, power, and durable mechanics.",
    "strength": "Support resilience, mechanics, and injury resistance.",
    "cross": "Maintain aerobic stimulus while reducing impact load.",
}

# Prompt-only structural scaffolding must never be judge-visible.
STRUCTURAL_PROMPT_LEAK_MARKERS: tuple[str, ...] = (
    "athlete band a1",
    "athlete band a2",
    "athlete band a3",
    "athlete band a4",
    "cell metadata:",
    "session_type='long'",
    'session_type="long"',
    "tempo/intervals/hills",
    "this cell expects",
    "weekly structure intent:",
)

_GENERIC_TITLE_RE = re.compile(r"^\s*day\s*\d+\s*$", re.IGNORECASE)
_ACTIVE_TOKEN_RE = re.compile(
    r"\b(run|running|tempo|threshold|interval|repetition|hill|long|strength|gym|lift|cross[- ]?training|bike|cycling|ride|swim|elliptical|workout)\b",
    re.IGNORECASE,
)

_PRIMARY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("rest", re.compile(r"\b(rest day|complete rest|no structured training|day off)\b", re.IGNORECASE)),
    ("cross", re.compile(r"\b(cross[- ]?training|bike|cycling|ride|swim|elliptical|low-impact aerobic)\b", re.IGNORECASE)),
    ("long", re.compile(r"\blong run\b", re.IGNORECASE)),
    ("intervals", re.compile(r"\b(interval|repetition|rep session|speed session)\b", re.IGNORECASE)),
    ("hills", re.compile(r"\b(hill session|hill repeats?|uphill repeats?|climb repeats?)\b", re.IGNORECASE)),
    ("strength", re.compile(r"\b(strength session|strength work|gym|lift|resistance|mobility circuit)\b", re.IGNORECASE)),
    ("tempo", re.compile(r"\b(tempo|threshold|sustained hard)\b", re.IGNORECASE)),
    ("easy", re.compile(r"\beasy run\b", re.IGNORECASE)),
    ("aerobic", re.compile(r"\b(aerobic run|steady aerobic|trail run)\b", re.IGNORECASE)),
)

_COMPATIBLE_PRIMARY_LABELS: dict[str, set[str]] = {
    "rest": {"rest"},
    "easy": {"easy", "aerobic"},
    "aerobic": {"easy", "aerobic"},
    # Long runs are normally aerobic, so aerobic/easy prose is compatible.
    "long": {"long", "aerobic", "easy"},
    "tempo": {"tempo"},
    "intervals": {"intervals"},
    "hills": {"hills"},
    "strength": {"strength"},
    "cross": {"cross"},
}

_HUMAN_DAY_FIELDS = ("title", "workout", "purpose")


@dataclass(frozen=True)
class FieldChange:
    path: str
    old: Any
    new: Any
    reason: str


@dataclass
class NormalizationResult:
    plan: dict[str, Any]
    changed_fields: list[FieldChange] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    structural_projection_before: dict[str, Any] = field(default_factory=dict)
    structural_projection_after: dict[str, Any] = field(default_factory=dict)

    @property
    def changed(self) -> bool:
        return bool(self.changed_fields)

    @property
    def structural_projection_unchanged(self) -> bool:
        return self.structural_projection_before == self.structural_projection_after

    def changed_field_payload(self) -> list[dict[str, Any]]:
        return [
            {"path": item.path, "old": item.old, "new": item.new, "reason": item.reason}
            for item in self.changed_fields
        ]


def stable_json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def artifact_sha256(plan_obj: dict[str, Any]) -> str:
    return hashlib.sha256(stable_json_dumps(plan_obj).encode("utf-8")).hexdigest()


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _clean_session_type(value: Any) -> str:
    return _as_text(value).strip().lower().replace(" ", "_").replace("-", "_")


def iter_plan_days(plan_obj: dict[str, Any]) -> Iterable[tuple[int, dict[str, Any]]]:
    days = _as_list(_as_dict(plan_obj.get("plan")).get("days"))
    for idx, day in enumerate(days):
        if isinstance(day, dict):
            yield idx, day


def score_relevant_projection(plan_obj: dict[str, Any]) -> dict[str, Any]:
    """Return a deep copy with judge-visible day prose removed.

    This deliberately keeps all non-day-prose fields and all day-level structure
    such as date, session_type, rest/hard flags, duration, intensity, and weekly
    totals. The normalizer asserts this projection is unchanged.
    """
    projected = copy.deepcopy(plan_obj)
    days = _as_list(_as_dict(projected.get("plan")).get("days"))
    for day in days:
        if isinstance(day, dict):
            for field_name in _HUMAN_DAY_FIELDS:
                day.pop(field_name, None)
    return projected


def canonical_title_for_day(day: dict[str, Any]) -> str | None:
    session_type = _clean_session_type(day.get("session_type"))
    if bool(day.get("is_rest_day")) or session_type == "rest":
        return CANONICAL_TITLES["rest"]
    return CANONICAL_TITLES.get(session_type)


def is_generic_day_title(title: Any) -> bool:
    return bool(_GENERIC_TITLE_RE.match(_as_text(title)))


def infer_primary_label(text: Any) -> str | None:
    raw = _as_text(text).strip()
    if not raw:
        return None
    first_sentence = re.split(r"(?<=[.!?])\s+", raw, maxsplit=1)[0]
    window = first_sentence[:160]
    for label, pattern in _PRIMARY_PATTERNS:
        if pattern.search(window):
            return label
    return None


def _is_label_compatible(session_type: str, label: str | None) -> bool:
    if label is None:
        return True
    return label in _COMPATIBLE_PRIMARY_LABELS.get(session_type, {session_type})


def _replace_field(day: dict[str, Any], path: str, field_name: str, new_value: str, reason: str, changes: list[FieldChange]) -> None:
    old_value = day.get(field_name)
    if old_value != new_value:
        changes.append(FieldChange(path=path, old=old_value, new=new_value, reason=reason))
        day[field_name] = new_value


def _field_needs_repair(value: Any, *, session_type: str, is_rest_day: bool) -> tuple[bool, str | None]:
    text = _as_text(value).strip()
    if not text:
        return True, "blank_human_facing_field"

    lower = text.lower()
    if is_rest_day or session_type == "rest":
        if _ACTIVE_TOKEN_RE.search(lower):
            return True, "rest_day_described_as_active"
        return False, None

    if "rest day" in lower:
        return True, "active_day_described_as_rest"

    label = infer_primary_label(text)
    if not _is_label_compatible(session_type, label):
        return True, f"field_primary_label_{label}_mismatches_session_type_{session_type}"

    return False, None


def collect_structural_prompt_leaks(value: Any, path: str = "$") -> list[tuple[str, str, str]]:
    hits: list[tuple[str, str, str]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            hits.extend(collect_structural_prompt_leaks(child, f"{path}.{key}"))
        return hits
    if isinstance(value, list):
        for idx, child in enumerate(value):
            hits.extend(collect_structural_prompt_leaks(child, f"{path}[{idx}]"))
        return hits
    if isinstance(value, str):
        lowered = value.strip().lower()
        for marker in STRUCTURAL_PROMPT_LEAK_MARKERS:
            if marker in lowered:
                hits.append((path, marker, value.strip()))
    return hits


def audit_plan_presentation(plan_obj: dict[str, Any], *, expected_days: int = EXPECTED_PLAN_DAYS) -> list[str]:
    """Return presentation/artifact-integrity issue codes after normalization."""
    issues: list[str] = []
    days = _as_list(_as_dict(plan_obj.get("plan")).get("days"))
    if len(days) != expected_days:
        issues.append(f"wrong_plan_day_count:{len(days)}!={expected_days}")

    for idx, day in iter_plan_days(plan_obj):
        day_num = idx + 1
        session_type = _clean_session_type(day.get("session_type"))
        is_rest_day = bool(day.get("is_rest_day")) or session_type == "rest"
        duration = int(day.get("duration_minutes") or 0)
        canonical_title = canonical_title_for_day(day)
        title = _as_text(day.get("title"))
        joined = " ".join(_as_text(day.get(field_name)).lower() for field_name in _HUMAN_DAY_FIELDS)

        if not session_type:
            issues.append(f"day_{day_num}:missing_session_type")
        elif session_type not in CANONICAL_TITLES:
            issues.append(f"day_{day_num}:unknown_session_type:{session_type}")

        if is_generic_day_title(title):
            issues.append(f"day_{day_num}:generic_day_title")
        if canonical_title is not None and title != canonical_title:
            issues.append(f"day_{day_num}:title_session_mismatch:{title!r}!={canonical_title!r}")

        if bool(day.get("is_rest_day")) and session_type != "rest":
            issues.append(f"day_{day_num}:rest_flag_nonrest_session_type:{session_type}")
        if session_type == "rest" and not bool(day.get("is_rest_day")):
            issues.append(f"day_{day_num}:rest_session_without_rest_flag")
        if session_type == "rest" and duration > 0:
            issues.append(f"day_{day_num}:rest_session_positive_duration:{duration}")
        if bool(day.get("is_hard_day")) and session_type == "rest":
            issues.append(f"day_{day_num}:rest_session_marked_hard")

        if is_rest_day:
            if _ACTIVE_TOKEN_RE.search(joined):
                issues.append(f"day_{day_num}:rest_session_described_as_active")
        else:
            if "rest day" in joined:
                issues.append(f"day_{day_num}:active_session_described_as_rest")
            for field_name in ("workout", "purpose"):
                label = infer_primary_label(day.get(field_name))
                if not _is_label_compatible(session_type, label):
                    issues.append(
                        f"day_{day_num}:{field_name}_primary_label_{label}_mismatches_session_type_{session_type}"
                    )

    for path, marker, _value in collect_structural_prompt_leaks(plan_obj):
        issues.append(f"structural_prompt_leak:{path}:{marker}")

    # Preserve stable order and remove duplicate issue codes.
    seen: set[str] = set()
    out: list[str] = []
    for issue in issues:
        if issue not in seen:
            seen.add(issue)
            out.append(issue)
    return out


def normalize_plan_artifact(plan_obj: dict[str, Any], *, expected_days: int = EXPECTED_PLAN_DAYS) -> NormalizationResult:
    """Canonicalize day-level presentation while preserving structure.

    The returned ``errors`` list contains issues that should normally quarantine
    an artifact rather than be silently fixed, such as non-7-day artifacts or
    structural rest/duration contradictions. Title and clear prose mismatches are
    repaired symmetrically for all source families.
    """
    before_projection = score_relevant_projection(plan_obj)
    out = copy.deepcopy(plan_obj)
    changes: list[FieldChange] = []
    warnings: list[str] = []
    errors: list[str] = []

    days = _as_list(_as_dict(out.get("plan")).get("days"))
    if len(days) != expected_days:
        errors.append(f"wrong_plan_day_count:{len(days)}!={expected_days}")

    if collect_structural_prompt_leaks(out):
        errors.append("structural_prompt_leak")

    for idx, day in iter_plan_days(out):
        day_num = idx + 1
        session_type = _clean_session_type(day.get("session_type"))
        is_rest_day_flag = bool(day.get("is_rest_day"))
        is_rest_day = is_rest_day_flag or session_type == "rest"
        duration = int(day.get("duration_minutes") or 0)

        if not session_type:
            errors.append(f"day_{day_num}:missing_session_type")
            continue
        if session_type not in CANONICAL_TITLES:
            warnings.append(f"day_{day_num}:unknown_session_type:{session_type}")
            continue

        if is_rest_day_flag and session_type != "rest":
            errors.append(f"day_{day_num}:rest_flag_nonrest_session_type:{session_type}")
        if session_type == "rest" and not is_rest_day_flag:
            errors.append(f"day_{day_num}:rest_session_without_rest_flag")
        if session_type == "rest" and duration > 0:
            errors.append(f"day_{day_num}:rest_session_positive_duration:{duration}")
        if session_type != "rest" and duration <= 0:
            warnings.append(f"day_{day_num}:active_session_nonpositive_duration:{duration}")
        if bool(day.get("is_hard_day")) and session_type == "rest":
            errors.append(f"day_{day_num}:rest_session_marked_hard")

        canonical_title = canonical_title_for_day(day)
        if canonical_title is not None:
            reason = "canonical_title_from_session_type"
            if is_generic_day_title(day.get("title")):
                reason = "generic_day_title"
            elif _as_text(day.get("title")) != canonical_title:
                reason = "title_session_mismatch"
            _replace_field(day, f"$.plan.days[{idx}].title", "title", canonical_title, reason, changes)

        for field_name, defaults in (("workout", CANONICAL_WORKOUTS), ("purpose", CANONICAL_PURPOSES)):
            needs_repair, reason = _field_needs_repair(
                day.get(field_name),
                session_type=session_type,
                is_rest_day=is_rest_day,
            )
            if needs_repair:
                _replace_field(
                    day,
                    f"$.plan.days[{idx}].{field_name}",
                    field_name,
                    defaults["rest" if is_rest_day else session_type],
                    reason or "presentation_repair",
                    changes,
                )

    after_projection = score_relevant_projection(out)
    if before_projection != after_projection:
        errors.append("score_relevant_projection_changed")

    # Remove duplicates while preserving first occurrence.
    deduped_errors: list[str] = []
    seen_errors: set[str] = set()
    for error in errors:
        if error not in seen_errors:
            seen_errors.add(error)
            deduped_errors.append(error)

    deduped_warnings: list[str] = []
    seen_warnings: set[str] = set()
    for warning in warnings:
        if warning not in seen_warnings:
            seen_warnings.add(warning)
            deduped_warnings.append(warning)

    return NormalizationResult(
        plan=out,
        changed_fields=changes,
        warnings=deduped_warnings,
        errors=deduped_errors,
        structural_projection_before=before_projection,
        structural_projection_after=after_projection,
    )


def build_normalization_provenance_payload(
    *,
    original_sha256: str,
    normalized_sha256: str,
    result: NormalizationResult,
    timestamp: str | None = None,
) -> dict[str, Any]:
    return {
        "normalizer_version": NORMALIZER_VERSION,
        "normalized_at": timestamp or datetime.now(tz=timezone.utc).isoformat(),
        "original_artifact_sha256": original_sha256,
        "normalized_artifact_sha256": normalized_sha256,
        "changed_fields": result.changed_field_payload(),
        "warnings": list(result.warnings),
        "errors": list(result.errors),
        "score_relevant_projection_unchanged": result.structural_projection_unchanged,
    }


def attach_normalization_metadata_to_provenance(
    provenance: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Store normalization metadata in provenance.runtime_metadata.

    This location survives the existing PlanProvenance round-trip because the
    current schema already preserves runtime_metadata. It also keeps normalizer
    details out of the judge-visible plan JSON.
    """
    out = copy.deepcopy(provenance)
    runtime_metadata = out.get("runtime_metadata")
    if not isinstance(runtime_metadata, dict):
        runtime_metadata = {}
    runtime_metadata["artifact_normalization"] = payload
    out["runtime_metadata"] = runtime_metadata
    return out
