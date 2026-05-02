from __future__ import annotations

from generate.structural_expectations import (
    build_structural_lifestyle_notes,
    detect_understructured_plan,
)


def _plan(days: list[dict]) -> dict:
    return {
        "plan": {
            "weekly_totals": {"planned_moving_time_hours": 6.0},
            "days": days,
        }
    }


def test_build_structural_lifestyle_notes_adds_fixture_specific_guidance() -> None:
    fixture_meta = {
        "athlete_band": "A4",
        "athlete_band_label": "High-volume competitive",
        "readiness": "high",
        "recovery_capability": "high",
        "race_phase": "peak",
        "lifestyle_notes": "Bias toward race specificity and freshness.",
        "athlete_profile_summary": "Strong training history and high long-run capacity.",
        "structure_hint": "highly structured week with the most race-specific options",
        "recent_volume_hint": "high recent volume with strong historical load tolerance",
        "long_run_tolerance_minutes": 180,
    }

    notes = build_structural_lifestyle_notes(fixture_meta)
    assert "Bias toward race specificity and freshness." in notes
    assert "athlete band A4" in notes
    assert "session_type='long'" in notes
    assert "tempo/intervals/hills" in notes
    assert "180 minutes" in notes


def test_detect_understructured_plan_flags_collapsed_a4_week() -> None:
    plan_obj = _plan(
        [
            {"session_type": "rest", "is_rest_day": True, "is_hard_day": False, "duration_minutes": 0},
            {"session_type": "easy", "is_rest_day": False, "is_hard_day": False, "duration_minutes": 60},
            {"session_type": "aerobic", "is_rest_day": False, "is_hard_day": False, "duration_minutes": 75},
            {"session_type": "easy", "is_rest_day": False, "is_hard_day": False, "duration_minutes": 45},
        ]
    )
    fixture_meta = {
        "athlete_band": "A4",
        "readiness": "high",
        "recovery_capability": "high",
        "race_phase": "peak",
    }

    issues = detect_understructured_plan(plan_obj, fixture_meta)
    assert "understructured:missing_long_run" in issues
    assert "understructured:missing_quality_for_A4" in issues
    assert "understructured:collapsed_easy_template" in issues


def test_detect_understructured_plan_allows_beginner_conservative_week() -> None:
    plan_obj = _plan(
        [
            {"session_type": "rest", "is_rest_day": True, "is_hard_day": False, "duration_minutes": 0},
            {"session_type": "easy", "is_rest_day": False, "is_hard_day": False, "duration_minutes": 30},
            {"session_type": "easy", "is_rest_day": False, "is_hard_day": False, "duration_minutes": 40},
        ]
    )
    fixture_meta = {
        "athlete_band": "A1",
        "readiness": "low",
        "recovery_capability": "low",
        "race_phase": "base",
    }

    assert detect_understructured_plan(plan_obj, fixture_meta) == []


def test_detect_understructured_plan_allows_a2_nonmaximal_week() -> None:
    plan_obj = _plan(
        [
            {"session_type": "rest", "is_rest_day": True, "is_hard_day": False, "duration_minutes": 0},
            {"session_type": "easy", "is_rest_day": False, "is_hard_day": False, "duration_minutes": 45},
            {"session_type": "aerobic", "is_rest_day": False, "is_hard_day": False, "duration_minutes": 75},
        ]
    )
    fixture_meta = {
        "athlete_band": "A2",
        "readiness": "low",
        "recovery_capability": "high",
        "race_phase": "base",
    }

    assert detect_understructured_plan(plan_obj, fixture_meta) == []
