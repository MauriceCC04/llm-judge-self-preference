from __future__ import annotations

from generate.programmatic_text_normalization import normalize_programmatic_artifact_text


def test_normalize_programmatic_artifact_text_fixes_active_rest_title() -> None:
    obj = {
        "plan": {
            "weekly_totals": {},
            "days": [
                {
                    "session_type": "long",
                    "is_rest_day": False,
                    "is_hard_day": False,
                    "duration_minutes": 90,
                    "target_intensity": "easy",
                    "terrain": "trail",
                    "title": "Rest day",
                    "workout": "Rest day.",
                    "purpose": "Rest day.",
                }
            ],
        }
    }

    out = normalize_programmatic_artifact_text(obj)
    day = out["plan"]["days"][0]
    assert day["session_type"] == "long"
    assert day["is_rest_day"] is False
    assert day["title"] == "Long run"
    assert "long run" in day["workout"].lower()
    assert "rest day" not in " ".join([day["title"], day["workout"], day["purpose"]]).lower()


def test_normalize_programmatic_artifact_text_fixes_rest_active_text() -> None:
    obj = {
        "plan": {
            "weekly_totals": {},
            "days": [
                {
                    "session_type": "rest",
                    "is_rest_day": True,
                    "is_hard_day": True,
                    "duration_minutes": 45,
                    "target_intensity": "threshold",
                    "terrain": "trail",
                    "title": "Tempo session",
                    "workout": "45 min tempo session.",
                    "purpose": "Quality.",
                }
            ],
        }
    }

    out = normalize_programmatic_artifact_text(obj)
    day = out["plan"]["days"][0]
    assert day["session_type"] == "rest"
    assert day["is_rest_day"] is True
    assert day["is_hard_day"] is False
    assert day["duration_minutes"] == 0
    assert day["title"] == "Rest day"
    assert "no structured training" in day["workout"].lower()


def test_normalize_programmatic_artifact_text_recomputes_weekly_hours() -> None:
    obj = {
        "plan": {
            "weekly_totals": {"planned_moving_time_hours": 999},
            "days": [
                {"session_type": "easy", "is_rest_day": False, "duration_minutes": 30},
                {"session_type": "rest", "is_rest_day": True, "duration_minutes": 99},
            ],
        }
    }

    out = normalize_programmatic_artifact_text(obj)
    assert out["plan"]["weekly_totals"]["planned_moving_time_hours"] == 0.5
