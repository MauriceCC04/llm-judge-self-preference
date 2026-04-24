"""fixtures/build.py — synthesise the 8 committed fixture bundles.

Run once (idempotent):
    python -m fixtures.build

Each bundle is written to ``fixtures/data/<fixture_id>/`` and contains:
    combined_summary.json              – 35 days of synthetic daily records
    combined_rollups.json              – pre-computed 7d / 28d rollups
    readiness_and_risk_forecast.json   – deterministic forecast artifact
    formatted_personal_data.json       – minimal personal profile stub
    fixture_meta.json                  – FixtureMeta serialised

The data is *synthetic*, not scraped, so there are no IRB concerns.
The values are chosen to be internally consistent (e.g. a "fatigued /
load_only" fixture has elevated RHR vs baseline and no sleep records).
"""
from __future__ import annotations

import json
import math
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from fixtures.spec import ALL_FIXTURE_SPECS, FixtureMeta, RecoveryCapabilityLevel, ReadinessLevel

DATA_DIR = Path(__file__).parent / "data"

# ── Numeric parameters per axis level ───────────────────────────────────────

_LOAD_HIGH = 6.5   # training_load_hours last7  (primed)
_LOAD_LOW = 3.0    # training_load_hours last7  (fatigued — elevated relative to baseline)
_BASELINE_LOAD = 4.0  # rolling-baseline load_h (same for both — controls ramp perception)

_RHR_NORMAL = 45   # bpm  (primed)
_RHR_ELEVATED = 52  # bpm  (fatigued)

_HRV_NORMAL = 70   # ms   (primed)
_HRV_LOW = 52      # ms   (fatigued — though fatigued fixtures may omit it)

_SLEEP_NORMAL = 7.8  # hours
_SLEEP_LOW = 6.2     # hours


def _isodate(d: date) -> str:
    return d.isoformat()


def _activity(day: date, *, moving_time_s: int = 3600, avg_hr: int = 148) -> dict[str, Any]:
    return {
        "id": abs(hash(day.isoformat())) % (10**9),
        "sport_type": "TrailRun",
        "start_date_local": f"{day.isoformat()}T08:00:00",
        "distance": 10000.0,
        "moving_time": moving_time_s,
        "total_elevation_gain": 250.0,
        "average_heartrate": avg_hr,
        "max_heartrate": 172,
    }


def _sleep_record(
    day: date,
    *,
    readiness: ReadinessLevel,
    recovery_capability: RecoveryCapabilityLevel,
) -> dict[str, Any] | None:
    """Return a sleep dict or None (when recovery_capability=='low')."""
    if recovery_capability == "low":
        return None  # no sleep/HRV data for load_only fixtures

    sleep_h = _SLEEP_NORMAL if readiness == "high" else _SLEEP_LOW
    rhr = _RHR_NORMAL if readiness == "high" else _RHR_ELEVATED
    hrv = _HRV_NORMAL if readiness == "high" else _HRV_LOW
    return {
        "calendarDate": day.isoformat(),
        "sleepTimeSeconds": int(sleep_h * 3600),
        "restingHeartRate": rhr,
        "avgOvernightHrv": hrv,
    }


def _build_combined_summary(spec: FixtureMeta, *, as_of: date) -> list[dict[str, Any]]:
    """35 days of daily records ending on *as_of*."""
    combined: list[dict[str, Any]] = []
    last7_load = _LOAD_HIGH if spec.readiness == "high" else _LOAD_LOW
    # We want the last-7 load to be noticeably different from the 28-day baseline.
    # For primed: last7 > baseline slightly (athlete training well)
    # For fatigued: last7 > baseline by more (overcooked)
    baseline_per_day = _BASELINE_LOAD / 7.0

    for i in range(35):
        day = as_of - timedelta(days=34 - i)
        is_recent = i >= 28  # last 7 days

        # Determine daily load — ramp up in last 7 for fatigued athlete
        if is_recent:
            target_week_load = last7_load
        else:
            target_week_load = _BASELINE_LOAD

        daily_load_h = (target_week_load / 7.0) * (0.9 + 0.2 * ((i % 3) / 2.0))
        moving_time_s = int(daily_load_h * 3600)

        # Rest days: every 4th day has no activity
        if i % 4 == 3:
            activities: list[dict[str, Any]] = []
        else:
            avg_hr = 155 if spec.readiness == "low" else 145
            activities = [_activity(day, moving_time_s=max(600, moving_time_s), avg_hr=avg_hr)]

        sleep = _sleep_record(
            day,
            readiness=spec.readiness,
            recovery_capability=spec.recovery_capability,
        )

        combined.append({"date": day.isoformat(), "sleep": sleep, "activities": activities})

    return combined


def _build_rollups(spec: FixtureMeta, *, as_of: date) -> dict[str, Any]:
    last7_load = _LOAD_HIGH if spec.readiness == "high" else _LOAD_LOW
    last7_dist = 55.0 if spec.readiness == "high" else 28.0
    baseline28_load = _BASELINE_LOAD * 4  # 4 weeks
    baseline28_dist = last7_dist * 3.8

    w7_start = as_of - timedelta(days=6)
    w28_start = as_of - timedelta(days=27)
    return {
        "generated_at": f"{as_of.isoformat()}T00:00:00Z",
        "windows": {
            "7": {
                "window_days": 7,
                "start_date": w7_start.isoformat(),
                "end_date": as_of.isoformat(),
                "sleep_days_with_data": 5 if spec.recovery_capability == "high" else 0,
                "activities": {
                    "count": 5,
                    "total_distance_km": round(last7_dist, 1),
                    "total_elevation_m": 1250.0 if spec.readiness == "high" else 650.0,
                    "total_moving_time_hours": round(last7_load / 1.15, 2),
                    "total_training_load_hours": round(last7_load, 2),
                    "average_heartrate_mean": 148.0 if spec.readiness == "high" else 156.0,
                    "count_by_sport": {"TrailRun": 5},
                },
            },
            "28": {
                "window_days": 28,
                "start_date": w28_start.isoformat(),
                "end_date": as_of.isoformat(),
                "sleep_days_with_data": 20 if spec.recovery_capability == "high" else 0,
                "activities": {
                    "count": 20,
                    "total_distance_km": round(baseline28_dist, 1),
                    "total_elevation_m": 5000.0,
                    "total_moving_time_hours": round(baseline28_load / 1.10, 2),
                    "total_training_load_hours": round(baseline28_load, 2),
                    "average_heartrate_mean": 147.0,
                    "count_by_sport": {"TrailRun": 20},
                },
            },
        },
    }


def _build_forecast(spec: FixtureMeta, *, as_of: date) -> dict[str, Any]:
    readiness_score = 78.0 if spec.readiness == "high" else 42.0
    readiness_status = spec.readiness_status  # "primed" or "fatigued"
    risk_score = 28.0 if spec.readiness == "high" else 68.0
    risk_level = "low" if spec.readiness == "high" else "high"

    rhr_7d = _RHR_NORMAL if spec.readiness == "high" else _RHR_ELEVATED
    rhr_28d = _RHR_NORMAL
    hrv_7d = _HRV_NORMAL if spec.readiness == "high" else _HRV_LOW
    hrv_28d = _HRV_NORMAL
    sleep_7d: float | None
    sleep_28d: float | None
    if spec.recovery_capability == "high":
        sleep_7d = _SLEEP_NORMAL if spec.readiness == "high" else _SLEEP_LOW
        sleep_28d = _SLEEP_NORMAL
    else:
        sleep_7d = None
        sleep_28d = None
        rhr_7d = None  # type: ignore[assignment]
        rhr_28d = None  # type: ignore[assignment]
        hrv_7d = None  # type: ignore[assignment]
        hrv_28d = None  # type: ignore[assignment]

    last7_load = _LOAD_HIGH if spec.readiness == "high" else _LOAD_LOW

    return {
        "generated_at": f"{as_of.isoformat()}T00:00:00Z",
        "result": {
            "date": as_of.isoformat(),
            "readiness": {"score": readiness_score, "status": readiness_status},
            "overreach_risk": {"score": risk_score, "level": risk_level},
            "inputs": {
                "as_of_date": as_of.isoformat(),
                "rhr_7d_mean_bpm": rhr_7d,
                "rhr_28d_mean_bpm": rhr_28d,
                "rhr_28d_std_bpm": 2.0 if rhr_28d else None,
                "rhr_delta_bpm": (rhr_7d - rhr_28d) if (rhr_7d and rhr_28d) else None,
                "rhr_z": ((rhr_7d - rhr_28d) / 2.0) if (rhr_7d and rhr_28d) else None,
                "training_load_7d_hours": round(last7_load, 2),
                "training_load_rolling7_mean_hours": _BASELINE_LOAD,
                "training_load_rolling7_std_hours": 0.8,
                "training_load_delta_hours": round(last7_load - _BASELINE_LOAD, 2),
                "training_load_z": round((last7_load - _BASELINE_LOAD) / 0.8, 3),
                "atl_load_hours": round(last7_load * 0.9, 3),
                "ctl_load_hours": round(_BASELINE_LOAD * 0.95, 3),
                "tsb_load_hours": round(_BASELINE_LOAD * 0.95 - last7_load * 0.9, 3),
                "sleep_7d_mean_hours": sleep_7d,
                "sleep_28d_mean_hours": sleep_28d,
                "sleep_28d_std_hours": 0.5 if sleep_28d else None,
                "sleep_delta_hours": ((sleep_7d - sleep_28d) if (sleep_7d and sleep_28d) else None),
                "sleep_z": None,
                "hrv_7d_mean_ms": hrv_7d,
                "hrv_28d_mean_ms": hrv_28d,
                "hrv_28d_std_ms": 5.0 if hrv_28d else None,
                "hrv_delta_ms": ((hrv_7d - hrv_28d) if (hrv_7d and hrv_28d) else None),
                "hrv_z": None,
                "recovery_capability_key": spec.recovery_key,
                "recovery_capability_label": (
                    "I have load + sleep + resting HR + HRV"
                    if spec.recovery_capability == "high"
                    else "I only have training data"
                ),
                "sleep_days_7d": 5 if spec.recovery_capability == "high" else 0,
                "resting_hr_days_7d": 5 if spec.recovery_capability == "high" else 0,
                "hrv_days_7d": 5 if spec.recovery_capability == "high" else 0,
                "notes": ["Synthetic fixture generated by fixtures/build.py."],
            },
            "drivers": {
                "readiness": (
                    ["Load is within comfortable range of 28-day baseline"]
                    if spec.readiness == "high"
                    else ["7d training load is elevated vs rolling baseline",
                          "Resting HR above 28d baseline"]
                ),
                "overreach_risk": (
                    ["Training load within sustainable range"]
                    if spec.readiness == "high"
                    else ["7d training load elevated", "Resting HR elevation signals fatigue"]
                ),
            },
        },
    }


def _build_personal_data(spec: FixtureMeta, *, as_of: date) -> dict[str, Any]:
    return {
        "userInfo": {},
        "biometricProfile": {
            "weight_kg": 68.0,
            "height_cm": 175.0,
        },
        "derived_activity_profile": {
            "observed_window": {
                "as_of_date": as_of.isoformat(),
                "first_activity_date": (as_of - timedelta(days=730)).isoformat(),
                "last_activity_date": (as_of - timedelta(days=1)).isoformat(),
                "activity_span_days": 730,
            },
            "sports": {
                "trailrunning": {
                    "sport_family": "running",
                    "claimed_years_sport": 2.0,
                    "activity_count": 180,
                    "total_distance_km": 2200.0,
                    "total_moving_time_hours": 280.0,
                    "total_elevation_m": 45000.0,
                    "total_training_load_hours": 310.0,
                }
            },
            "top_sports": {
                "90d": {"primary_sport_discipline": "trailrunning"},
                "365d": {"primary_sport_discipline": "trailrunning"},
            },
            "historical_capacities": {
                "90d": {
                    "all_sports": {"peak_7d_training_load_hours": 8.5},
                    "running_family": {"peak_7d_distance_km": 65.0},
                },
                "365d": {
                    "all_sports": {"peak_7d_training_load_hours": 10.2},
                    "running_family": {"peak_7d_distance_km": 80.0},
                },
            },
        },
        "profile_metadata": {
            "generated_at": f"{as_of.isoformat()}T00:00:00Z",
            "generated_from": ["synthetic_fixture"],
        },
    }


def build_fixture(spec: FixtureMeta, *, as_of: date, out_dir: Path) -> None:
    """Write all four JSON files + fixture_meta.json into *out_dir*."""
    out_dir.mkdir(parents=True, exist_ok=True)

    combined = _build_combined_summary(spec, as_of=as_of)
    rollups = _build_rollups(spec, as_of=as_of)
    forecast = _build_forecast(spec, as_of=as_of)
    personal = _build_personal_data(spec, as_of=as_of)

    def _write(name: str, obj: Any) -> None:
        (out_dir / name).write_text(
            json.dumps(obj, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    _write("combined_summary.json", combined)
    _write("combined_rollups.json", rollups)
    _write("readiness_and_risk_forecast.json", forecast)
    _write("formatted_personal_data.json", personal)
    _write("fixture_meta.json", spec.model_dump())

    print(f"  [built] {out_dir.name}/")


def build_all(out_root: Path | None = None) -> None:
    if out_root is None:
        out_root = DATA_DIR
    as_of = date(2026, 3, 17)  # frozen reference date
    print(f"Building 8 fixture bundles into {out_root}/")
    for spec in ALL_FIXTURE_SPECS:
        build_fixture(spec, as_of=as_of, out_dir=out_root / spec.fixture_id)
    print("Done.")


if __name__ == "__main__":
    build_all()
