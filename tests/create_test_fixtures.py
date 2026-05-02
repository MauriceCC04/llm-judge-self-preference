"""tests/create_test_fixtures.py — minimal fixture bundle for Gate-0 tests.

Called at the start of run_tests.py; cleaned up at exit.
Returns the path to the single minimal fixture directory.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any


def make_minimal_combined_summary(*, as_of: str = "2026-03-17") -> list[dict[str, Any]]:
    """35 daily records ending on *as_of*, with sleep and one activity each."""
    from datetime import date, timedelta

    end = date.fromisoformat(as_of)
    records = []
    for i in range(35):
        d = end - timedelta(days=34 - i)
        records.append(
            {
                "date": d.isoformat(),
                "sleep": {
                    "calendarDate": d.isoformat(),
                    "sleepTimeSeconds": 28800,
                    "restingHeartRate": 46,
                    "avgOvernightHrv": 68,
                },
                "activities": [
                    {
                        "id": 1000 + i,
                        "sport_type": "TrailRun",
                        "start_date_local": f"{d.isoformat()}T08:00:00",
                        "distance": 10000.0,
                        "moving_time": 3600,
                        "total_elevation_gain": 250.0,
                        "average_heartrate": 148,
                        "max_heartrate": 172,
                    }
                ],
            }
        )
    return records


def make_minimal_rollups(*, as_of: str = "2026-03-17") -> dict[str, Any]:
    return {
        "generated_at": f"{as_of}T00:00:00Z",
        "windows": {
            "7": {
                "window_days": 7,
                "start_date": "2026-03-11",
                "end_date": as_of,
                "sleep_days_with_data": 5,
                "activities": {
                    "count": 5,
                    "total_distance_km": 48.0,
                    "total_elevation_m": 1200.0,
                    "total_moving_time_hours": 4.8,
                    "total_training_load_hours": 5.5,
                    "average_heartrate_mean": 148.0,
                    "count_by_sport": {"TrailRun": 5},
                },
            },
            "28": {
                "window_days": 28,
                "start_date": "2026-02-18",
                "end_date": as_of,
                "sleep_days_with_data": 20,
                "activities": {
                    "count": 20,
                    "total_distance_km": 190.0,
                    "total_elevation_m": 4800.0,
                    "total_moving_time_hours": 19.2,
                    "total_training_load_hours": 22.0,
                    "average_heartrate_mean": 147.0,
                    "count_by_sport": {"TrailRun": 20},
                },
            },
        },
    }


def make_minimal_forecast(*, as_of: str = "2026-03-17") -> dict[str, Any]:
    return {
        "generated_at": f"{as_of}T00:00:00Z",
        "result": {
            "date": as_of,
            "readiness": {"score": 72.5, "status": "steady"},
            "overreach_risk": {"score": 38.0, "level": "moderate"},
            "inputs": {
                "as_of_date": as_of,
                "rhr_7d_mean_bpm": 46.0,
                "rhr_28d_mean_bpm": 45.0,
                "rhr_28d_std_bpm": 1.5,
                "rhr_delta_bpm": 1.0,
                "rhr_z": 0.67,
                "training_load_7d_hours": 5.5,
                "training_load_rolling7_mean_hours": 4.5,
                "training_load_rolling7_std_hours": 0.8,
                "training_load_delta_hours": 1.0,
                "training_load_z": 1.25,
                "atl_load_hours": 5.2,
                "ctl_load_hours": 4.8,
                "tsb_load_hours": -0.4,
                "sleep_7d_mean_hours": 7.8,
                "sleep_28d_mean_hours": 7.9,
                "sleep_28d_std_hours": 0.4,
                "sleep_delta_hours": -0.1,
                "sleep_z": -0.25,
                "hrv_7d_mean_ms": 68.0,
                "hrv_28d_mean_ms": 70.0,
                "hrv_28d_std_ms": 5.0,
                "hrv_delta_ms": -2.0,
                "hrv_z": -0.4,
                "recovery_capability_key": "load_sleep_resting_hr_hrv",
                "recovery_capability_label": "I have load + sleep + resting HR + HRV",
                "sleep_days_7d": 5,
                "resting_hr_days_7d": 5,
                "hrv_days_7d": 5,
                "notes": ["Minimal test fixture."],
            },
            "drivers": {
                "readiness": ["Load within comfortable range."],
                "overreach_risk": ["Load slightly elevated vs baseline."],
            },
        },
    }


def make_minimal_personal_data() -> dict[str, Any]:
    return {
        "userInfo": {},
        "biometricProfile": {},
        "derived_activity_profile": {
            "observed_window": {
                "as_of_date": "2026-03-17",
                "first_activity_date": "2024-03-17",
                "last_activity_date": "2026-03-16",
                "activity_span_days": 730,
            },
            "sports": {},
            "top_sports": {},
            "historical_capacities": {},
        },
    }


def make_minimal_fixture_meta() -> dict[str, Any]:
    return {
        "fixture_id": "ab_A2__r_high__rc_high__ph_base",
        "athlete_band": "A2",
        "athlete_band_label": "Developing recreational",
        "athlete_archetype": "developing_recreational_runner",
        "athlete_profile_summary": "Modest recent consistency, moderate but not high load, some structure tolerated.",
        "recent_volume_hint": "modest recent consistency",
        "structure_hint": "moderately structured week with limited hard load",
        "long_run_tolerance_minutes": 110,
        "readiness": "high",
        "recovery_capability": "high",
        "race_phase": "base",
        "readiness_status": "primed",
        "recovery_key": "load_sleep_resting_hr_hrv",
        "weeks_to_race": 20,
        "block_label": "base development block",
        "style": "trailrunning",
        "primary_goal": "to build consistent trail-running fitness with manageable structure",
        "lifestyle_notes": "Minimal test fixture.",
        "baseline_load_hours": 5.0,
        "baseline_distance_km": 40.0,
        "baseline_elevation_m": 1300.0,
        "base_speed_kmh": 8.1,
        "weight_kg": 69.0,
        "height_cm": 175.0,
        "years_in_sport": 2.8,
    }


def create_test_fixture(tmp_dir: Path | None = None) -> Path:
    """Create a minimal fixture bundle in *tmp_dir* (or a fresh tempdir).

    Returns the fixture directory path.
    Call ``cleanup_test_fixture(path)`` or rely on atexit to remove it.
    """
    if tmp_dir is None:
        tmp_dir = Path(tempfile.mkdtemp(prefix="jbs_test_fixture_"))

    fixture_dir = tmp_dir / "test_fixture"
    fixture_dir.mkdir(parents=True, exist_ok=True)

    def _write(name: str, obj: Any) -> None:
        (fixture_dir / name).write_text(
            json.dumps(obj, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    _write("combined_summary.json", make_minimal_combined_summary())
    _write("combined_rollups.json", make_minimal_rollups())
    _write("readiness_and_risk_forecast.json", make_minimal_forecast())
    _write("formatted_personal_data.json", make_minimal_personal_data())
    _write("fixture_meta.json", make_minimal_fixture_meta())

    return fixture_dir
