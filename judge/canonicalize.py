"""judge/canonicalize.py — fixed-format pairwise rendering helpers."""
from __future__ import annotations

import copy
from typing import Any

_ALLOWED_SESSION_KEYS = (
    "date",
    "session_type",
    "is_rest_day",
    "is_hard_day",
    "duration_minutes",
    "target_intensity",
    "terrain",
)


def _canonical_day(day: dict[str, Any], idx: int) -> dict[str, Any]:
    rendered = {key: day.get(key) for key in _ALLOWED_SESSION_KEYS}
    rendered["day_index"] = idx + 1
    rendered["signal_count"] = len(day.get("signal_ids") or [])
    return rendered


def _risk_kinds(plan_obj: dict[str, Any]) -> list[str]:
    kinds: list[str] = []
    for risk in plan_obj.get("risks") or []:
        if not isinstance(risk, dict):
            continue
        kind = str(risk.get("kind") or risk.get("type") or "").strip()
        if kind:
            kinds.append(kind)
    return sorted(set(kinds))


def canonicalize_plan_for_pairwise(plan_obj: dict[str, Any]) -> dict[str, Any]:
    obj = copy.deepcopy(plan_obj)
    days = ((obj.get("plan") or {}).get("days") or [])
    canonical_days = [_canonical_day(day if isinstance(day, dict) else {}, idx) for idx, day in enumerate(days)]

    readiness = obj.get("readiness") or {}
    plan = obj.get("plan") or {}

    return {
        "meta": {
            "style": ((obj.get("meta") or {}).get("style") or ""),
            "primary_goal": ((obj.get("meta") or {}).get("primary_goal") or ""),
        },
        "plan": {
            "weekly_totals": copy.deepcopy(plan.get("weekly_totals") or {}),
            "days": canonical_days,
        },
        "readiness": {
            "status": readiness.get("status"),
            "signal_count": len(readiness.get("signal_ids") or []),
        },
        "risk_summary": {
            "n_risks": len(obj.get("risks") or []),
            "risk_kinds": _risk_kinds(obj),
        },
        "data_note_count": len(obj.get("data_notes") or []),
        "citation_count": len(obj.get("citations") or []),
        "claim_attribution_count": len(obj.get("claim_attributions") or []),
    }


def canonicalize_pair_for_pairwise(plan_a: dict[str, Any], plan_b: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    return canonicalize_plan_for_pairwise(plan_a), canonicalize_plan_for_pairwise(plan_b)
