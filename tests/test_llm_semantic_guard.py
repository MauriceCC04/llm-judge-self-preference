from __future__ import annotations

from generate.trailtraining_compat import _normalize_rest_day_semantics
from match.filtering import detect_plan_issues


def test_normalize_rest_day_semantics_repairs_reused_rest_titles() -> None:
    plan_obj = {
        "plan": {
            "days": [
                {
                    "date": "2026-03-11",
                    "title": "Rest day",
                    "session_type": "easy",
                    "is_rest_day": False,
                    "is_hard_day": False,
                    "duration_minutes": 30,
                    "target_intensity": "easy",
                    "terrain": "flat",
                    "workout": "30 min easy run on flat to rolling.",
                    "purpose": "",
                },
                {
                    "date": "2026-03-12",
                    "title": "Rest day",
                    "session_type": "rest",
                    "is_rest_day": True,
                    "is_hard_day": False,
                    "duration_minutes": 0,
                    "target_intensity": "rest",
                    "terrain": "n/a",
                    "workout": "Rest day. No structured training.",
                    "purpose": "",
                },
            ]
        }
    }

    repaired = _normalize_rest_day_semantics(plan_obj)
    days = repaired["plan"]["days"]

    assert days[0]["title"] == "Easy run"
    assert days[0]["session_type"] == "easy"
    assert days[0]["duration_minutes"] == 30

    assert days[1]["title"] == "Rest day"
    assert days[1]["session_type"] == "rest"
    assert days[1]["duration_minutes"] == 0

    assert detect_plan_issues(repaired) == []
