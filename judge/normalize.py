from __future__ import annotations

import copy
import re

_WS_RE = re.compile(r"\s+")


def _norm(text: str, *, max_chars: int) -> str:
    return _WS_RE.sub(" ", text.strip())[:max_chars].rstrip()


def normalize_plan_for_pairwise(plan_obj: dict, *, text_char_limits: dict[str, int]) -> dict:
    obj = copy.deepcopy(plan_obj)
    for day in ((obj.get("plan") or {}).get("days") or []):
        if not isinstance(day, dict):
            continue
        if isinstance(day.get("purpose"), str):
            day["purpose"] = _norm(day["purpose"], max_chars=text_char_limits["purpose"])
        if isinstance(day.get("workout"), str):
            day["workout"] = _norm(day["workout"], max_chars=text_char_limits["workout"])
    readiness = obj.get("readiness") or {}
    if isinstance(readiness.get("rationale"), str):
        readiness["rationale"] = _norm(readiness["rationale"], max_chars=text_char_limits["readiness_rationale"])
    snapshot = obj.get("snapshot") or {}
    if isinstance(snapshot.get("notes"), str):
        snapshot["notes"] = _norm(snapshot["notes"], max_chars=text_char_limits["snapshot_notes"])
    normalized_notes = []
    for note in obj.get("data_notes") or []:
        if isinstance(note, str):
            normalized_notes.append(_norm(note, max_chars=text_char_limits["data_note"]))
        elif isinstance(note, dict):
            note = dict(note)
            if isinstance(note.get("message"), str):
                note["message"] = _norm(note["message"], max_chars=text_char_limits["data_note"])
            normalized_notes.append(note)
    obj["data_notes"] = normalized_notes
    for risk in obj.get("risks") or []:
        if isinstance(risk, dict) and isinstance(risk.get("message"), str):
            risk["message"] = _norm(risk["message"], max_chars=text_char_limits["risk_message"])
    return obj


def normalize_pair_for_pairwise(plan_a: dict, plan_b: dict, *, text_char_limits: dict[str, int]) -> tuple[dict, dict]:
    return (
        normalize_plan_for_pairwise(plan_a, text_char_limits=text_char_limits),
        normalize_plan_for_pairwise(plan_b, text_char_limits=text_char_limits),
    )
