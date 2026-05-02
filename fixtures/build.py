"""fixtures/build.py — synthesise the 32 committed fixture bundles.

Run once (idempotent):
    python -m fixtures.build

This revision upgrades the study from 8 to 32 fixture bundles by introducing
athlete_band as a first-class cell axis. Phase, readiness, recovery, and athlete
band now all change prompt-visible payloads.
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from statistics import mean
from typing import Any

from fixtures.spec import ALL_FIXTURE_SPECS, FIXTURE_BY_ID, FixtureMeta, PHASE_PROFILES

DATA_DIR = Path(__file__).parent / "data"


def _json_write(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _rest_day_for_week(week_idx: int, spec: FixtureMeta) -> int:
    if spec.readiness == "low":
        return [1, 3, 1, 2, 4][week_idx]
    if spec.race_phase == "peak":
        return [1, 1, 2, 3, 4][week_idx]
    return [2, 1, 2, 1, 3][week_idx]


def _long_run_day_for_week(week_idx: int, spec: FixtureMeta) -> int:
    return 5 if spec.race_phase == "base" else 4 + (week_idx % 2)


def _phase_training_multiplier(spec: FixtureMeta, day_idx: int) -> float:
    profile = PHASE_PROFILES[spec.race_phase]
    if day_idx < 28:
        return profile.baseline28_load_mult
    return profile.last7_load_mult


def _readiness_training_multiplier(spec: FixtureMeta, day_idx: int) -> float:
    if spec.readiness == "high":
        return 1.10 if day_idx >= 28 else 1.00
    return 0.90 if day_idx < 28 else 0.82


def _duration_minutes(
    spec: FixtureMeta,
    day_idx: int,
    weekday: int,
    *,
    is_rest: bool,
    is_long_run: bool,
) -> int:
    if is_rest:
        return 0
    baseline_minutes = (spec.baseline_load_hours * 60.0) / 5.5
    structure_mult = 0.82 + (weekday * 0.06)
    if is_long_run:
        structure_mult *= 1.60 * PHASE_PROFILES[spec.race_phase].long_run_emphasis
    if weekday == 2 and spec.readiness == "high":
        structure_mult *= 1.12
    if weekday == 2 and spec.readiness == "low":
        structure_mult *= 0.90
    duration = baseline_minutes * structure_mult
    duration *= _phase_training_multiplier(spec, day_idx)
    duration *= _readiness_training_multiplier(spec, day_idx)
    rounded = max(35, int(round(duration)))
    if is_long_run:
        long_cap = int(
            round(
                spec.long_run_tolerance_minutes
                * (1.05 if spec.race_phase == "base" else 0.95)
            )
        )
        return min(rounded, max(60, long_cap))
    return rounded


def _distance_km(
    spec: FixtureMeta,
    duration_minutes: int,
    *,
    is_rest: bool,
    is_long_run: bool,
) -> float | None:
    if is_rest:
        return None
    base_speed_kmh = spec.base_speed_kmh
    if spec.race_phase == "peak":
        base_speed_kmh *= 1.02
    if spec.readiness == "low":
        base_speed_kmh *= 0.94
    distance = (duration_minutes / 60.0) * base_speed_kmh
    distance *= PHASE_PROFILES[spec.race_phase].distance_mult
    if is_long_run:
        distance *= 1.08
    return round(distance, 1)


def _elevation_m(
    spec: FixtureMeta,
    duration_minutes: int,
    *,
    is_rest: bool,
    is_long_run: bool,
) -> float | None:
    if is_rest:
        return None
    climb_rate = 4.8 if spec.race_phase == "base" else 4.1
    if spec.readiness == "low":
        climb_rate *= 0.90
    elevation = duration_minutes * climb_rate * PHASE_PROFILES[spec.race_phase].elevation_mult
    if is_long_run:
        elevation *= 1.12
    return round(elevation, 1)


def _average_hr(spec: FixtureMeta, *, is_rest: bool, is_long_run: bool) -> int | None:
    if is_rest:
        return None
    base = 146 if spec.readiness == "high" else 153
    if is_long_run:
        base -= 2
    if spec.race_phase == "peak":
        base += 1
    return base


def _session_type(spec: FixtureMeta, weekday: int, *, is_rest: bool, is_long_run: bool) -> str:
    if is_rest:
        return "rest"
    if is_long_run:
        return "long"
    if weekday in (1, 2):
        return "hills" if spec.race_phase == "base" else "tempo"
    if weekday == 4:
        return "strength" if spec.readiness == "low" else "aerobic"
    return "easy"


def _sleep_record(
    spec: FixtureMeta,
    current_day: date,
    day_idx: int,
    *,
    is_rest: bool,
) -> dict[str, Any] | None:
    if spec.recovery_capability == "low":
        return None
    if spec.readiness == "high":
        sleep_hours = 7.8 if not is_rest else 8.1
        resting_hr = 45 if spec.race_phase == "base" else 46
        hrv = 71 if spec.race_phase == "base" else 69
    else:
        sleep_hours = 6.3 if not is_rest else 6.7
        resting_hr = 52 if spec.race_phase == "base" else 50
        hrv = 53 if spec.race_phase == "base" else 56
    if day_idx >= 28 and spec.race_phase == "peak" and spec.readiness == "high":
        sleep_hours += 0.2
    return {
        "calendarDate": current_day.isoformat(),
        "sleepTimeSeconds": int(round(sleep_hours * 3600)),
        "restingHeartRate": resting_hr,
        "avgOvernightHrv": hrv,
    }


def _activity_record(
    current_day: date,
    duration_minutes: int,
    distance_km: float | None,
    elevation_m: float | None,
    avg_hr: int | None,
) -> dict[str, Any]:
    return {
        "id": abs(hash((current_day.isoformat(), duration_minutes, distance_km))) % (10**9),
        "sport_type": "TrailRun",
        "start_date_local": f"{current_day.isoformat()}T07:30:00",
        "distance": None if distance_km is None else round(distance_km * 1000.0, 1),
        "moving_time": int(duration_minutes * 60),
        "total_elevation_gain": elevation_m,
        "average_heartrate": avg_hr,
        "max_heartrate": None if avg_hr is None else avg_hr + 22,
    }


def _build_combined_summary(spec: FixtureMeta, *, as_of: date) -> list[dict[str, Any]]:
    combined: list[dict[str, Any]] = []
    start_day = as_of - timedelta(days=34)
    for day_idx in range(35):
        current_day = start_day + timedelta(days=day_idx)
        week_idx = day_idx // 7
        weekday = day_idx % 7
        is_rest = weekday == _rest_day_for_week(week_idx, spec)
        is_long_run = (weekday == _long_run_day_for_week(week_idx, spec)) and not is_rest
        session_type = _session_type(spec, weekday, is_rest=is_rest, is_long_run=is_long_run)
        duration_minutes = _duration_minutes(
            spec,
            day_idx,
            weekday,
            is_rest=is_rest,
            is_long_run=is_long_run,
        )
        distance_km = _distance_km(spec, duration_minutes, is_rest=is_rest, is_long_run=is_long_run)
        elevation_m = _elevation_m(spec, duration_minutes, is_rest=is_rest, is_long_run=is_long_run)
        avg_hr = _average_hr(spec, is_rest=is_rest, is_long_run=is_long_run)
        activities = []
        if not is_rest:
            activities.append(
                _activity_record(current_day, duration_minutes, distance_km, elevation_m, avg_hr)
            )
        combined.append(
            {
                "date": current_day.isoformat(),
                "sleep": _sleep_record(spec, current_day, day_idx, is_rest=is_rest),
                "activities": activities,
                "notes": {
                    "session_type": session_type,
                    "phase_block": spec.block_label,
                    "athlete_band": spec.athlete_band,
                    "athlete_band_label": spec.athlete_band_label,
                },
            }
        )
    return combined


def _window_summary(
    records: list[dict[str, Any]],
    *,
    start_day: date,
    end_day: date,
) -> dict[str, Any]:
    in_window = [r for r in records if start_day <= date.fromisoformat(r["date"]) <= end_day]
    sleep_days = sum(1 for r in in_window if r.get("sleep"))
    activities = [a for r in in_window for a in (r.get("activities") or [])]
    distances = [float(a.get("distance") or 0.0) / 1000.0 for a in activities]
    elevations = [float(a.get("total_elevation_gain") or 0.0) for a in activities]
    moving_hours = [float(a.get("moving_time") or 0.0) / 3600.0 for a in activities]
    avg_hrs = [
        float(a.get("average_heartrate"))
        for a in activities
        if a.get("average_heartrate") is not None
    ]
    sport_counts: dict[str, int] = {}
    for activity in activities:
        sport = str(activity.get("sport_type") or "unknown")
        sport_counts[sport] = sport_counts.get(sport, 0) + 1
    return {
        "window_days": (end_day - start_day).days + 1,
        "start_date": start_day.isoformat(),
        "end_date": end_day.isoformat(),
        "sleep_days_with_data": sleep_days,
        "activities": {
            "count": len(activities),
            "total_distance_km": round(sum(distances), 1),
            "total_elevation_m": round(sum(elevations), 1),
            "total_moving_time_hours": round(sum(moving_hours), 2),
            "total_training_load_hours": round(sum(moving_hours), 2),
            "average_heartrate_mean": round(mean(avg_hrs), 1) if avg_hrs else None,
            "count_by_sport": sport_counts,
        },
    }


def _build_rollups(
    spec: FixtureMeta,
    *,
    as_of: date,
    combined: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "generated_at": f"{as_of.isoformat()}T00:00:00Z",
        "windows": {
            "7": _window_summary(combined, start_day=as_of - timedelta(days=6), end_day=as_of),
            "28": _window_summary(combined, start_day=as_of - timedelta(days=27), end_day=as_of),
        },
    }


def _mean_optional(values: list[float | None]) -> float | None:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None
    return round(mean(clean), 2)


def _build_forecast(
    spec: FixtureMeta,
    *,
    as_of: date,
    combined: list[dict[str, Any]],
    rollups: dict[str, Any],
) -> dict[str, Any]:
    recent = combined[-7:]
    historical = combined[-28:]
    sleep_7d = _mean_optional(
        [
            None if r.get("sleep") is None else float(r["sleep"]["sleepTimeSeconds"]) / 3600.0
            for r in recent
        ]
    )
    sleep_28d = _mean_optional(
        [
            None if r.get("sleep") is None else float(r["sleep"]["sleepTimeSeconds"]) / 3600.0
            for r in historical
        ]
    )
    rhr_7d = _mean_optional(
        [None if r.get("sleep") is None else r["sleep"].get("restingHeartRate") for r in recent]
    )
    rhr_28d = _mean_optional(
        [None if r.get("sleep") is None else r["sleep"].get("restingHeartRate") for r in historical]
    )
    hrv_7d = _mean_optional(
        [None if r.get("sleep") is None else r["sleep"].get("avgOvernightHrv") for r in recent]
    )
    hrv_28d = _mean_optional(
        [None if r.get("sleep") is None else r["sleep"].get("avgOvernightHrv") for r in historical]
    )

    load_7d = float(rollups["windows"]["7"]["activities"]["total_training_load_hours"])
    load_28d = float(rollups["windows"]["28"]["activities"]["total_training_load_hours"])
    rolling_7_mean = round(load_28d / 4.0, 2)

    if spec.readiness == "high":
        readiness_score = 77.0 if spec.race_phase == "base" else 81.0
        risk_score = 29.0 if spec.race_phase == "base" else 24.0
    else:
        readiness_score = 43.0 if spec.race_phase == "base" else 49.0
        risk_score = 69.0 if spec.race_phase == "base" else 58.0

    readiness_drivers = []
    risk_drivers = []
    if spec.readiness == "high":
        readiness_drivers.append("Recent load is compatible with current preparedness.")
        if spec.race_phase == "peak":
            readiness_drivers.append("Freshness indicators fit a race-specific sharpening phase.")
        else:
            readiness_drivers.append(
                "Load supports continued aerobic development without obvious strain."
            )
        risk_drivers.append("Recent training remains within sustainable range for this athlete profile.")
    else:
        readiness_drivers.append(
            "Recent signals indicate meaningful fatigue relative to the athlete baseline."
        )
        risk_drivers.append("Readiness context suggests reduced tolerance for extra hard volume.")
        if spec.recovery_capability == "high":
            risk_drivers.append(
                "Recovery signals do not fully support aggressive progression right now."
            )
        else:
            risk_drivers.append(
                "Recovery instrumentation is limited, increasing uncertainty around strain."
            )

    readiness_drivers.append(f"Athlete band context: {spec.athlete_band_label}.")
    readiness_drivers.extend(PHASE_PROFILES[spec.race_phase].forecast_driver_notes)
    risk_drivers.append(f"Athlete profile summary: {spec.athlete_profile_summary}")
    risk_drivers.extend(PHASE_PROFILES[spec.race_phase].forecast_driver_notes)

    return {
        "generated_at": f"{as_of.isoformat()}T00:00:00Z",
        "result": {
            "date": as_of.isoformat(),
            "readiness": {"score": readiness_score, "status": spec.readiness_status},
            "overreach_risk": {
                "score": risk_score,
                "level": "low" if risk_score < 40 else "moderate" if risk_score < 60 else "high",
            },
            "inputs": {
                "as_of_date": as_of.isoformat(),
                "rhr_7d_mean_bpm": rhr_7d,
                "rhr_28d_mean_bpm": rhr_28d,
                "rhr_28d_std_bpm": 2.0 if rhr_28d is not None else None,
                "rhr_delta_bpm": None
                if (rhr_7d is None or rhr_28d is None)
                else round(rhr_7d - rhr_28d, 2),
                "rhr_z": None
                if (rhr_7d is None or rhr_28d is None)
                else round((rhr_7d - rhr_28d) / 2.0, 3),
                "training_load_7d_hours": round(load_7d, 2),
                "training_load_rolling7_mean_hours": rolling_7_mean,
                "training_load_rolling7_std_hours": 0.8,
                "training_load_delta_hours": round(load_7d - rolling_7_mean, 2),
                "training_load_z": round((load_7d - rolling_7_mean) / 0.8, 3),
                "atl_load_hours": round(load_7d * 0.92, 3),
                "ctl_load_hours": round(rolling_7_mean * 0.97, 3),
                "tsb_load_hours": round((rolling_7_mean * 0.97) - (load_7d * 0.92), 3),
                "sleep_7d_mean_hours": sleep_7d,
                "sleep_28d_mean_hours": sleep_28d,
                "sleep_28d_std_hours": 0.45 if sleep_28d is not None else None,
                "sleep_delta_hours": None
                if (sleep_7d is None or sleep_28d is None)
                else round(sleep_7d - sleep_28d, 2),
                "hrv_7d_mean_ms": hrv_7d,
                "hrv_28d_mean_ms": hrv_28d,
                "hrv_28d_std_ms": 5.0 if hrv_28d is not None else None,
                "hrv_delta_ms": None
                if (hrv_7d is None or hrv_28d is None)
                else round(hrv_7d - hrv_28d, 2),
                "recovery_capability_key": spec.recovery_key,
                "recovery_capability_label": (
                    "I have load + sleep + resting HR + HRV"
                    if spec.recovery_capability == "high"
                    else "I only have training data"
                ),
                "sleep_days_7d": int(rollups["windows"]["7"]["sleep_days_with_data"]),
                "resting_hr_days_7d": int(rollups["windows"]["7"]["sleep_days_with_data"]),
                "hrv_days_7d": int(rollups["windows"]["7"]["sleep_days_with_data"]),
                "weeks_to_race": spec.weeks_to_race,
                "block_label": spec.block_label,
                "notes": [
                    "Synthetic fixture generated by fixtures/build.py.",
                    f"Athlete band: {spec.athlete_band} ({spec.athlete_band_label}).",
                    f"Athlete profile: {spec.athlete_profile_summary}",
                    f"Phase context: {spec.block_label}.",
                ],
            },
            "drivers": {
                "readiness": readiness_drivers,
                "overreach_risk": risk_drivers,
            },
        },
    }


def _build_personal_data(spec: FixtureMeta, *, as_of: date) -> dict[str, Any]:
    base_load = spec.baseline_load_hours
    base_distance = spec.baseline_distance_km
    return {
        "userInfo": {},
        "biometricProfile": {
            "weight_kg": spec.weight_kg,
            "height_cm": spec.height_cm,
        },
        "derived_activity_profile": {
            "observed_window": {
                "as_of_date": as_of.isoformat(),
                "first_activity_date": (as_of - timedelta(days=780)).isoformat(),
                "last_activity_date": (as_of - timedelta(days=1)).isoformat(),
                "activity_span_days": 780,
            },
            "sports": {
                spec.style: {
                    "sport_family": "running",
                    "claimed_years_sport": spec.years_in_sport,
                    "activity_count": int(150 + (spec.years_in_sport * 30)),
                    "total_distance_km": round(base_distance * 52.0, 1),
                    "total_moving_time_hours": round(base_load * 52.0, 1),
                    "total_elevation_m": round(spec.baseline_elevation_m * 52.0, 1),
                    "total_training_load_hours": round(base_load * 56.0, 1),
                }
            },
            "top_sports": {
                "90d": {"primary_sport_discipline": spec.style},
                "365d": {"primary_sport_discipline": spec.style},
            },
            "historical_capacities": {
                "90d": {
                    "all_sports": {"peak_7d_training_load_hours": round(base_load * 1.45, 2)},
                    "running_family": {"peak_7d_distance_km": round(base_distance * 1.35, 2)},
                },
                "365d": {
                    "all_sports": {"peak_7d_training_load_hours": round(base_load * 1.70, 2)},
                    "running_family": {"peak_7d_distance_km": round(base_distance * 1.55, 2)},
                },
            },
        },
        "profile_metadata": {
            "generated_at": f"{as_of.isoformat()}T00:00:00Z",
            "generated_from": ["synthetic_fixture"],
            "fixture_id": spec.fixture_id,
            "athlete_band": spec.athlete_band,
            "athlete_band_label": spec.athlete_band_label,
            "athlete_archetype": spec.athlete_archetype,
            "athlete_profile_summary": spec.athlete_profile_summary,
            "race_phase": spec.race_phase,
            "block_label": spec.block_label,
            "primary_goal": spec.primary_goal,
            "lifestyle_notes": spec.lifestyle_notes,
            "weeks_to_race": spec.weeks_to_race,
        },
    }


def build_fixture(spec: FixtureMeta, *, as_of: date, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    combined = _build_combined_summary(spec, as_of=as_of)
    rollups = _build_rollups(spec, as_of=as_of, combined=combined)
    forecast = _build_forecast(spec, as_of=as_of, combined=combined, rollups=rollups)
    personal = _build_personal_data(spec, as_of=as_of)

    _json_write(out_dir / "combined_summary.json", combined)
    _json_write(out_dir / "combined_rollups.json", rollups)
    _json_write(out_dir / "readiness_and_risk_forecast.json", forecast)
    _json_write(out_dir / "formatted_personal_data.json", personal)
    _json_write(out_dir / "fixture_meta.json", spec.model_dump())
    print(f"  [built] {out_dir.name}/")


def build_all(out_root: Path | None = None) -> None:
    root = out_root or DATA_DIR
    as_of = date(2026, 3, 17)
    print(f"Building {len(ALL_FIXTURE_SPECS)} fixture bundles into {root}/")
    for spec in ALL_FIXTURE_SPECS:
        build_fixture(spec, as_of=as_of, out_dir=root / spec.fixture_id)
    print("Done.")


def build_one(fixture_id: str, out_root: Path | None = None) -> None:
    root = out_root or DATA_DIR
    spec = FIXTURE_BY_ID[fixture_id]
    build_fixture(spec, as_of=date(2026, 3, 17), out_dir=root / fixture_id)


if __name__ == "__main__":
    build_all()
