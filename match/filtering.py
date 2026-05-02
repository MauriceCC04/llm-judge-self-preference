from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from typing import Any

_ACTIVE_TOKENS = (
    "run",
    "tempo",
    "interval",
    "hill",
    "long",
    "strength",
    "cross",
    "ride",
    "workout",
)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _lower(value: Any) -> str:
    return _as_text(value).strip().lower()


def stable_plan_json(plan_obj: dict[str, Any]) -> str:
    return json.dumps(plan_obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def exact_text_fingerprint(plan_obj: dict[str, Any]) -> str:
    return hashlib.sha256(stable_plan_json(plan_obj).encode("utf-8")).hexdigest()


def session_signature(plan_obj: dict[str, Any]) -> str:
    days = _as_list(_as_dict(plan_obj.get("plan")).get("days"))
    rows: list[str] = []
    for day in days:
        d = _as_dict(day)
        rows.append(
            "|".join(
                [
                    _as_text(d.get("session_type")),
                    str(int(d.get("duration_minutes") or 0)),
                    "1" if bool(d.get("is_rest_day")) else "0",
                    "1" if bool(d.get("is_hard_day")) else "0",
                    _as_text(d.get("target_intensity")),
                ]
            )
        )
    return "||".join(rows)


def compact_plan_summary(plan_obj: dict[str, Any]) -> dict[str, Any]:
    days = [_as_dict(day) for day in _as_list(_as_dict(plan_obj.get("plan")).get("days"))]
    return {
        "plan_days": len(days),
        "total_minutes": sum(int(day.get("duration_minutes") or 0) for day in days),
        "n_rest_days": sum(1 for day in days if bool(day.get("is_rest_day"))),
        "n_hard_days": sum(1 for day in days if bool(day.get("is_hard_day"))),
        "has_long_run": any(_as_text(day.get("session_type")) == "long" for day in days),
        "day_titles": [_as_text(day.get("title")) for day in days],
        "session_types": [_as_text(day.get("session_type")) for day in days],
    }


def detect_plan_issues(plan_obj: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    plan = _as_dict(plan_obj.get("plan"))
    days = [_as_dict(day) for day in _as_list(plan.get("days"))]

    if not days:
        issues.append("broken_artifact:no_plan_days")
        return issues

    seen_dates: Counter[str] = Counter()
    for day in days:
        seen_dates[_as_text(day.get("date"))] += 1
        session_type = _as_text(day.get("session_type"))
        duration = int(day.get("duration_minutes") or 0)
        title = _lower(day.get("title"))
        workout = _lower(day.get("workout"))
        purpose = _lower(day.get("purpose"))

        if duration < 0:
            issues.append("broken_artifact:negative_duration")

        if session_type == "":
            issues.append("broken_artifact:missing_session_type")

        if "rest day" in title and session_type != "rest":
            issues.append("human_contradiction:title_rest_day_but_nonrest_session_type")

        if "rest day" in title and duration > 0:
            issues.append("human_contradiction:title_rest_day_but_positive_duration")

        if session_type == "rest" and duration > 0:
            issues.append("human_contradiction:rest_session_with_positive_duration")

        if session_type == "rest":
            active_text = " ".join([title, workout, purpose])
            if any(tok in active_text for tok in _ACTIVE_TOKENS):
                issues.append("human_contradiction:rest_session_described_as_active")

        if session_type != "rest":
            joined = " ".join([title, workout, purpose])
            if "rest day" in joined and duration > 0:
                issues.append("human_contradiction:nonrest_session_described_as_rest_day")

    dup_dates = [d for d, n in seen_dates.items() if d and n > 1]
    if dup_dates:
        issues.append(f"broken_artifact:duplicate_dates:{','.join(sorted(dup_dates))}")

    out: list[str] = []
    seen: set[str] = set()
    for item in issues:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _scope_key(record: dict[str, Any], scope_fields: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(str(record.get(field) or "") for field in scope_fields)


def _scope_label(scope: tuple[str, ...], scope_fields: tuple[str, ...]) -> str:
    return ", ".join(f"{field}={value}" for field, value in zip(scope_fields, scope))


def filter_plan_records(
    records: list[dict[str, Any]],
    *,
    dedup_scope_fields: tuple[str, ...] = ("fixture_id", "arm"),
) -> dict[str, Any]:
    valid_records: list[dict[str, Any]] = []
    invalid_or_contradictory: list[dict[str, Any]] = []

    for record in sorted(
        records,
        key=lambda r: (
            str(r.get("fixture_id") or ""),
            str(r.get("arm") or ""),
            str(r.get("plan_id") or ""),
        ),
    ):
        plan_obj = record.get("plan_obj")
        if not isinstance(plan_obj, dict):
            invalid_or_contradictory.append(
                {
                    "plan_id": str(record.get("plan_id") or ""),
                    "fixture_id": str(record.get("fixture_id") or ""),
                    "arm": str(record.get("arm") or ""),
                    "issues": ["broken_artifact:missing_plan_obj"],
                }
            )
            continue

        issues = detect_plan_issues(plan_obj)
        if issues:
            invalid_or_contradictory.append(
                {
                    "plan_id": str(record.get("plan_id") or ""),
                    "fixture_id": str(record.get("fixture_id") or ""),
                    "arm": str(record.get("arm") or ""),
                    "issues": issues,
                }
            )
            continue

        enriched = dict(record)
        enriched["text_fingerprint"] = exact_text_fingerprint(plan_obj)
        enriched["session_signature"] = session_signature(plan_obj)
        valid_records.append(enriched)

    text_groups_raw: dict[tuple[tuple[str, ...], str], list[dict[str, Any]]] = defaultdict(list)
    for record in valid_records:
        text_groups_raw[(_scope_key(record, dedup_scope_fields), str(record["text_fingerprint"]))].append(record)

    exact_text_duplicate_groups: list[dict[str, Any]] = []
    dropped_plan_ids: set[str] = set()
    dropped_records: list[dict[str, Any]] = []

    for (scope, fingerprint), group in sorted(text_groups_raw.items(), key=lambda x: (x[0][0], x[0][1])):
        group_sorted = sorted(group, key=lambda r: str(r.get("plan_id") or ""))
        if len(group_sorted) <= 1:
            continue
        keep = group_sorted[0]
        drop = group_sorted[1:]
        exact_text_duplicate_groups.append(
            {
                "scope": _scope_label(scope, dedup_scope_fields),
                "text_fingerprint": fingerprint,
                "kept_plan_id": keep["plan_id"],
                "dropped_plan_ids": [r["plan_id"] for r in drop],
                "all_plan_ids": [r["plan_id"] for r in group_sorted],
            }
        )
        for record in drop:
            dropped_plan_ids.add(str(record["plan_id"]))
            dropped_records.append(
                {
                    "plan_id": str(record["plan_id"]),
                    "fixture_id": str(record.get("fixture_id") or ""),
                    "arm": str(record.get("arm") or ""),
                    "drop_reason": "exact_text_duplicate",
                    "canonical_plan_id": str(keep["plan_id"]),
                }
            )

    remaining = [r for r in valid_records if str(r["plan_id"]) not in dropped_plan_ids]

    sig_groups_raw: dict[tuple[tuple[str, ...], str], list[dict[str, Any]]] = defaultdict(list)
    for record in remaining:
        sig_groups_raw[(_scope_key(record, dedup_scope_fields), str(record["session_signature"]))].append(record)

    session_signature_duplicate_groups: list[dict[str, Any]] = []
    for (scope, sig), group in sorted(sig_groups_raw.items(), key=lambda x: (x[0][0], x[0][1])):
        group_sorted = sorted(group, key=lambda r: str(r.get("plan_id") or ""))
        if len(group_sorted) <= 1:
            continue
        keep = group_sorted[0]
        drop = group_sorted[1:]
        session_signature_duplicate_groups.append(
            {
                "scope": _scope_label(scope, dedup_scope_fields),
                "session_signature": sig,
                "kept_plan_id": keep["plan_id"],
                "dropped_plan_ids": [r["plan_id"] for r in drop],
                "all_plan_ids": [r["plan_id"] for r in group_sorted],
            }
        )
        for record in drop:
            dropped_plan_ids.add(str(record["plan_id"]))
            dropped_records.append(
                {
                    "plan_id": str(record["plan_id"]),
                    "fixture_id": str(record.get("fixture_id") or ""),
                    "arm": str(record.get("arm") or ""),
                    "drop_reason": "session_signature_duplicate",
                    "canonical_plan_id": str(keep["plan_id"]),
                }
            )

    kept_records = [r for r in valid_records if str(r["plan_id"]) not in dropped_plan_ids]

    for record in invalid_or_contradictory:
        dropped_records.append(
            {
                "plan_id": record["plan_id"],
                "fixture_id": record["fixture_id"],
                "arm": record["arm"],
                "drop_reason": "invalid_or_contradictory",
                "issues": record["issues"],
            }
        )

    reasons = Counter(str(item["drop_reason"]) for item in dropped_records)
    audit = {
        "n_input": len(records),
        "n_kept": len(kept_records),
        "n_dropped": len(dropped_records),
        "drop_reasons": dict(sorted(reasons.items())),
        "exact_text_duplicate_groups": exact_text_duplicate_groups,
        "session_signature_duplicate_groups": session_signature_duplicate_groups,
        "invalid_or_contradictory": invalid_or_contradictory,
    }
    return {"kept_records": kept_records, "dropped_records": dropped_records, "audit": audit}
