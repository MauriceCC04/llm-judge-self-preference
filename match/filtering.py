"""Prefiltering for matchable plan records.

This filter only removes invalid or contradictory plans. Session-signature
repetition is audited but intentionally kept for primary matching: repeated
structural signatures are not source labels and dropping them before matching can
artificially destroy same-cell overlap. Duplicate-heavy cells should be handled
as a robustness/sensitivity analysis, not as a hard pre-matching exclusion.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


def _session_signature(record: dict[str, Any]) -> tuple[Any, ...]:
    plan_obj = record.get("plan_obj") or {}
    days = (plan_obj.get("plan") or {}).get("days") or []
    sig: list[tuple[str, int, bool, bool]] = []
    for day in days if isinstance(days, list) else []:
        if not isinstance(day, dict):
            continue
        sig.append(
            (
                str(day.get("session_type") or "").strip().lower(),
                int(float(day.get("duration_minutes") or 0)),
                bool(day.get("is_rest_day", False)),
                bool(day.get("is_hard_day", False)),
            )
        )
    if not sig:
        return ("missing_plan_obj", record.get("plan_id"))
    return tuple(sig)


def _contradiction(record: dict[str, Any]) -> str | None:
    plan_obj = record.get("plan_obj") or {}
    days = (plan_obj.get("plan") or {}).get("days") or []
    if not days and "plan_days" in record and int(record.get("plan_days") or 0) <= 0:
        return "invalid_plan_days"
    for idx, day in enumerate(days if isinstance(days, list) else []):
        if not isinstance(day, dict):
            return f"bad_day_{idx}"
        dur = float(day.get("duration_minutes") or 0)
        is_rest = bool(day.get("is_rest_day", False))
        st = str(day.get("session_type") or "").strip().lower()
        if is_rest and dur > 0 and st not in {"rest", "off", "recovery"}:
            return "rest_active_contradiction"
        if dur < 0 or dur > 360:
            return "impossible_duration"
    return None


def _duplicate_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record.get("arm"),
        record.get("source_model") if record.get("arm") == "llm" else "programmatic",
        record.get("fixture_id"),
        record.get("athlete_band"),
        record.get("readiness"),
        record.get("recovery_capability"),
        record.get("race_phase"),
        record.get("plan_days"),
        _session_signature(record),
    )


def filter_plan_records(plan_records: list[dict[str, Any]]) -> dict[str, Any]:
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    duplicate_groups: defaultdict[tuple[Any, ...], list[str]] = defaultdict(list)

    for record in plan_records:
        duplicate_groups[_duplicate_key(record)].append(str(record.get("plan_id") or ""))
        reason = _contradiction(record)
        if reason is None:
            kept.append(record)
        else:
            reasons[reason] += 1
            slim = {k: v for k, v in record.items() if k != "plan_obj"}
            slim["drop_reason"] = reason
            dropped.append(slim)

    repeated = {"|".join(map(str, key[:8])): ids for key, ids in duplicate_groups.items() if len(ids) > 1}
    n_duplicate_records = sum(len(ids) - 1 for ids in duplicate_groups.values() if len(ids) > 1)
    audit = {
        "n_input": len(plan_records),
        "n_kept": len(kept),
        "n_dropped": len(dropped),
        "drop_reasons": dict(sorted(reasons.items())),
        "session_signature_duplicate_policy": "audited_not_dropped_for_primary_matching",
        "n_session_signature_duplicate_groups": len(repeated),
        "n_session_signature_duplicate_extra_records": n_duplicate_records,
        "session_signature_duplicate_examples": dict(list(repeated.items())[:25]),
        "presentation_fields_used": False,
    }
    return {"kept_records": kept, "dropped_records": dropped, "audit": audit}
