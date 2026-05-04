from __future__ import annotations

from typing import Any


_EASYISH_SESSION_TYPES = {"easy", "aerobic", "strength", "cross"}


def _clean_str(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _clean_key(value: Any) -> str:
    return _clean_str(value).lower()


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, (int, float)):
        return int(round(float(value)))
    return None


def _dedupe_lines(lines: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for line in lines:
        clean = " ".join(str(line or "").split()).strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        out.append(clean)
    return out


def build_structural_lifestyle_notes(
    fixture_meta: dict[str, Any],
    base_lifestyle_notes: str | None = None,
) -> str:
    fixture_meta = fixture_meta if isinstance(fixture_meta, dict) else {}

    base = _clean_str(fixture_meta.get("lifestyle_notes")) or _clean_str(base_lifestyle_notes)
    band = _clean_str(fixture_meta.get("athlete_band"))
    band_label = _clean_str(fixture_meta.get("athlete_band_label")) or band
    readiness = _clean_str(fixture_meta.get("readiness"))
    recovery = _clean_str(fixture_meta.get("recovery_capability"))
    phase = _clean_str(fixture_meta.get("race_phase"))
    summary = _clean_str(fixture_meta.get("athlete_profile_summary"))
    structure_hint = _clean_str(fixture_meta.get("structure_hint"))
    recent_volume_hint = _clean_str(fixture_meta.get("recent_volume_hint"))
    long_run_tolerance = _int_or_none(fixture_meta.get("long_run_tolerance_minutes"))

    lines: list[str] = []
    if base:
        lines.append(base)

    meta_bits: list[str] = []
    if band:
        meta_bits.append(f"athlete band {band}")
    if band_label and band_label != band:
        meta_bits.append(band_label)
    if readiness:
        meta_bits.append(f"readiness {readiness}")
    if recovery:
        meta_bits.append(f"recovery capacity {recovery}")
    if phase:
        meta_bits.append(f"phase {phase}")
    if meta_bits:
        lines.append("Cell metadata: " + ", ".join(meta_bits) + ".")

    if summary:
        lines.append(f"Athlete profile summary: {summary}")
    if recent_volume_hint:
        lines.append(f"Recent-volume context: {recent_volume_hint}.")
    if structure_hint:
        lines.append(f"Weekly structure intent: {structure_hint}.")
    if long_run_tolerance is not None:
        lines.append(
            f"Any long run should stay at or below about {long_run_tolerance} minutes unless the safety constraints require less."
        )

    lines.append(
        "Day titles and workout text must match the structured fields; do not use 'Rest day' text on active sessions."
    )
    lines.append(
        "Rest days must be semantically inert: session_type='rest', is_rest_day=true, duration_minutes=0, target_intensity='rest', terrain='n/a', title='Rest day', workout='Rest day. No structured training.', and purpose='Recover and maintain freshness.' Do not mention run, long, tempo, interval, hill, strength, cross-training, ride, or workout in rest-day title, workout, or purpose."
    )

    if band in {"A3", "A4"}:
        lines.append(
            "This cell expects a genuinely structured week, not an all-easy or all-rest template. Include at least one day with session_type='long'. Low readiness should reduce intensity or duration, not collapse the entire week to rest unless deterministic safety constraints explicitly require that."
        )

    if band == "A4":
        lines.append(
            "This athlete band should include at least one clearly harder quality session using session_type tempo/intervals/hills with is_hard_day=true when readiness is high, recovery capacity is high, or phase is peak. For low-readiness cells, make the quality session shorter and controlled rather than omitting quality entirely unless deterministic safety constraints explicitly prohibit it."
        )
    elif band == "A3" and phase == "peak" and readiness == "high":
        lines.append(
            "Because this is an advanced peak-phase high-readiness cell, include at least one clearly harder quality session using session_type tempo/intervals/hills with is_hard_day=true unless safety constraints make that unsafe."
        )
    elif band == "A2" and readiness == "high" and recovery == "high":
        lines.append(
            "This developing-recreational high-readiness/high-recovery cell should include at least one differentiated structural feature: either one long run or one clearly harder quality session."
        )
    elif band == "A1":
        lines.append("Simple conservative structure is acceptable here; do not force high-intensity work.")

    lines.append(
        "If you omit an expected long run or quality day for safety reasons, explain that explicitly in data_notes and keep the remaining days clearly differentiated."
    )
    return "\n".join(_dedupe_lines(lines))


def detect_understructured_plan(plan_obj: dict[str, Any], fixture_meta: dict[str, Any]) -> list[str]:
    fixture_meta = fixture_meta if isinstance(fixture_meta, dict) else {}
    band = _clean_str(fixture_meta.get("athlete_band"))
    readiness = _clean_str(fixture_meta.get("readiness"))
    recovery = _clean_str(fixture_meta.get("recovery_capability"))
    phase = _clean_str(fixture_meta.get("race_phase"))

    if not band:
        return []

    plan = plan_obj.get("plan") if isinstance(plan_obj, dict) else {}
    days = plan.get("days") if isinstance(plan, dict) else []
    if not isinstance(days, list):
        return []

    non_rest_days = [
        day
        for day in days
        if isinstance(day, dict)
        and not bool(day.get("is_rest_day"))
        and _clean_key(day.get("session_type")) != "rest"
    ]
    if not non_rest_days:
        issues = ["understructured:no_active_sessions"]
        if band in {"A3", "A4"}:
            issues.append("understructured:missing_long_run")
        if band == "A4" and (readiness == "high" or recovery == "high" or phase == "peak"):
            issues.append("understructured:missing_quality_for_A4")
        elif band == "A3" and phase == "peak" and readiness == "high":
            issues.append("understructured:missing_quality_for_A3_peak_high")
        elif band == "A2" and readiness == "high" and recovery == "high":
            issues.append("understructured:missing_long_or_quality_for_A2_high_high")
        return issues

    hard_count = sum(1 for day in non_rest_days if bool(day.get("is_hard_day")))
    long_count = sum(1 for day in non_rest_days if _clean_key(day.get("session_type")) == "long")
    active_types = {_clean_key(day.get("session_type")) for day in non_rest_days}
    collapsed_easy_template = bool(active_types) and active_types.issubset(_EASYISH_SESSION_TYPES)

    issues: list[str] = []

    if band in {"A3", "A4"} and long_count == 0:
        issues.append("understructured:missing_long_run")

    if band == "A4" and hard_count == 0 and (readiness == "high" or recovery == "high" or phase == "peak"):
        issues.append("understructured:missing_quality_for_A4")
    elif band == "A3" and phase == "peak" and readiness == "high" and hard_count == 0:
        issues.append("understructured:missing_quality_for_A3_peak_high")
    elif band == "A2" and readiness == "high" and recovery == "high" and hard_count == 0 and long_count == 0:
        issues.append("understructured:missing_long_or_quality_for_A2_high_high")

    if band in {"A3", "A4"} and collapsed_easy_template and hard_count == 0 and long_count == 0:
        issues.append("understructured:collapsed_easy_template")

    return issues
