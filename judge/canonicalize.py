"""judge/canonicalize.py — fixed-format pairwise rendering helpers."""
from __future__ import annotations

import copy
import re
from typing import Any

_WS_RE = re.compile(r"\s+")
_ALLOWED_SESSION_KEYS = (
    "date",
    "session_type",
    "is_rest_day",
    "is_hard_day",
    "duration_minutes",
    "target_intensity",
    "terrain",
)


def _clean_text(value: Any, *, max_chars: int = 120) -> str:
    if not isinstance(value, str):
        return ""
    text = _WS_RE.sub(" ", value.strip())
    return text[:max_chars].rstrip()


def _canonical_day(day: dict[str, Any], idx: int) -> dict[str, Any]:
    rendered = {key: day.get(key) for key in _ALLOWED_SESSION_KEYS}
    rendered["day_index"] = idx + 1
    rendered["session_summary"] = _clean_text(day.get("workout"), max_chars=120)
    rendered["purpose_summary"] = _clean_text(day.get("purpose"), max_chars=100)
    return rendered


def canonicalize_plan_for_pairwise(plan_obj: dict[str, Any]) -> dict[str, Any]:
    obj = copy.deepcopy(plan_obj)
    days = ((obj.get("plan") or {}).get("days") or [])
    canonical_days = [_canonical_day(day if isinstance(day, dict) else {}, idx) for idx, day in enumerate(days)]

    out = {
        "meta": {
            "style": ((obj.get("meta") or {}).get("style") or ""),
            "primary_goal": ((obj.get("meta") or {}).get("primary_goal") or ""),
        },
        "plan": {
            "weekly_totals": copy.deepcopy(((obj.get("plan") or {}).get("weekly_totals") or {})),
            "days": canonical_days,
        },
        "readiness": {
            "rationale": _clean_text(((obj.get("readiness") or {}).get("rationale")), max_chars=120),
        },
        "snapshot": {
            "notes": _clean_text(((obj.get("snapshot") or {}).get("notes")), max_chars=120),
        },
        "risks": [],
        "data_notes": [],
    }

    for risk in obj.get("risks") or []:
        if not isinstance(risk, dict):
            continue
        out["risks"].append({
            "kind": risk.get("kind") or risk.get("type") or "risk",
            "message": _clean_text(risk.get("message"), max_chars=100),
        })

    for note in obj.get("data_notes") or []:
        if isinstance(note, str):
            cleaned = _clean_text(note, max_chars=90)
            if cleaned:
                out["data_notes"].append(cleaned)
        elif isinstance(note, dict):
            cleaned = _clean_text(note.get("message"), max_chars=90)
            if cleaned:
                out["data_notes"].append({"message": cleaned})

    return out


def canonicalize_pair_for_pairwise(plan_a: dict[str, Any], plan_b: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    return canonicalize_plan_for_pairwise(plan_a), canonicalize_plan_for_pairwise(plan_b)
