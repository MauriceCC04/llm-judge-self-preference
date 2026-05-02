"""fixtures/spec.py — axis definitions and fixture metadata schema.

Revised study design:
    athlete_band         : A1 | A2 | A3 | A4
    readiness            : low | high
    recovery_capability  : low | high
    race_phase           : base | peak

Total fixture cells: 4 x 2 x 2 x 2 = 32
"""
from __future__ import annotations

from itertools import product
from typing import Literal

from pydantic import BaseModel


AthleteBand = Literal["A1", "A2", "A3", "A4"]
ReadinessLevel = Literal["low", "high"]
RecoveryCapabilityLevel = Literal["low", "high"]
RacePhase = Literal["base", "peak"]


class AthleteBandProfile(BaseModel):
    label: str
    athlete_archetype: str
    athlete_profile_summary: str
    recent_volume_hint: str
    structure_hint: str
    long_run_tolerance_minutes: int

    baseline_load_hours: float
    baseline_distance_km: float
    baseline_elevation_m: float
    base_speed_kmh: float

    weight_kg: float
    height_cm: float
    years_in_sport: float


class PhaseProfile(BaseModel):
    weeks_to_race: int
    block_label: str
    style: str = "trailrunning"
    primary_goal: str
    lifestyle_notes: str
    last7_load_mult: float
    baseline28_load_mult: float
    distance_mult: float
    elevation_mult: float
    long_run_emphasis: float
    forecast_driver_notes: list[str]


class FixtureMeta(BaseModel):
    fixture_id: str

    athlete_band: AthleteBand
    readiness: ReadinessLevel
    recovery_capability: RecoveryCapabilityLevel
    race_phase: RacePhase

    readiness_status: Literal["fatigued", "primed"]
    recovery_key: str
    weeks_to_race: int

    # Prompt-visible metadata
    athlete_band_label: str
    athlete_archetype: str
    athlete_profile_summary: str
    recent_volume_hint: str
    structure_hint: str
    long_run_tolerance_minutes: int

    block_label: str
    style: str
    primary_goal: str
    lifestyle_notes: str

    # Numeric profile fields used by fixtures/build.py
    baseline_load_hours: float
    baseline_distance_km: float
    baseline_elevation_m: float
    base_speed_kmh: float
    weight_kg: float
    height_cm: float
    years_in_sport: float


ATHLETE_BANDS: list[AthleteBand] = ["A1", "A2", "A3", "A4"]
READINESS_LEVELS: list[ReadinessLevel] = ["low", "high"]
RECOVERY_CAPABILITY_LEVELS: list[RecoveryCapabilityLevel] = ["low", "high"]
RACE_PHASES: list[RacePhase] = ["base", "peak"]

READINESS_TO_STATUS: dict[ReadinessLevel, Literal["fatigued", "primed"]] = {
    "low": "fatigued",
    "high": "primed",
}

RECOVERY_TO_KEY: dict[RecoveryCapabilityLevel, str] = {
    "low": "load_only",
    "high": "load_sleep_resting_hr_hrv",
}

ATHLETE_BAND_PROFILES: dict[AthleteBand, AthleteBandProfile] = {
    "A1": AthleteBandProfile(
        label="Beginner / low-volume",
        athlete_archetype="beginner_low_volume_runner",
        athlete_profile_summary="Low recent volume, lower long-run tolerance, simpler weekly structure, conservative loading.",
        recent_volume_hint="low recent volume",
        structure_hint="simple weekly structure with conservative loading",
        long_run_tolerance_minutes=85,
        baseline_load_hours=3.8,
        baseline_distance_km=28.0,
        baseline_elevation_m=850.0,
        base_speed_kmh=7.6,
        weight_kg=71.0,
        height_cm=176.0,
        years_in_sport=1.2,
    ),
    "A2": AthleteBandProfile(
        label="Developing recreational",
        athlete_archetype="developing_recreational_runner",
        athlete_profile_summary="Modest recent consistency, moderate but not high load, some structure tolerated.",
        recent_volume_hint="modest recent consistency",
        structure_hint="moderately structured week with limited hard load",
        long_run_tolerance_minutes=110,
        baseline_load_hours=5.0,
        baseline_distance_km=40.0,
        baseline_elevation_m=1300.0,
        base_speed_kmh=8.1,
        weight_kg=69.0,
        height_cm=175.0,
        years_in_sport=2.8,
    ),
    "A3": AthleteBandProfile(
        label="Advanced amateur",
        athlete_archetype="advanced_amateur_runner",
        athlete_profile_summary="Consistent runner, higher recent load, more robust weekly structure.",
        recent_volume_hint="consistent higher recent load",
        structure_hint="robust weekly structure with clearly differentiated sessions",
        long_run_tolerance_minutes=145,
        baseline_load_hours=6.4,
        baseline_distance_km=54.0,
        baseline_elevation_m=1900.0,
        base_speed_kmh=8.6,
        weight_kg=67.0,
        height_cm=174.0,
        years_in_sport=5.5,
    ),
    "A4": AthleteBandProfile(
        label="High-volume competitive",
        athlete_archetype="high_volume_competitive_runner",
        athlete_profile_summary="Strongest training history, highest recent load, highest long-run capacity, most race-specific structure allowed.",
        recent_volume_hint="high recent volume with strong historical load tolerance",
        structure_hint="highly structured week with the most race-specific options",
        long_run_tolerance_minutes=180,
        baseline_load_hours=8.0,
        baseline_distance_km=72.0,
        baseline_elevation_m=2600.0,
        base_speed_kmh=8.9,
        weight_kg=65.0,
        height_cm=173.0,
        years_in_sport=7.5,
    ),
}

PHASE_PROFILES: dict[RacePhase, PhaseProfile] = {
    "base": PhaseProfile(
        weeks_to_race=20,
        block_label="base development block",
        primary_goal="to build aerobic durability and resilient trail-running capacity",
        lifestyle_notes="Prioritize consistent aerobic development, hill strength, and durable weekly routine.",
        last7_load_mult=1.02,
        baseline28_load_mult=1.00,
        distance_mult=1.04,
        elevation_mult=1.08,
        long_run_emphasis=1.10,
        forecast_driver_notes=[
            "Current context resembles a development block rather than a taper.",
            "Long-run durability and uphill economy matter more than sharpening.",
        ],
    ),
    "peak": PhaseProfile(
        weeks_to_race=4,
        block_label="peak specificity block",
        primary_goal="to arrive race-ready for an upcoming trail event",
        lifestyle_notes="Bias toward race specificity, freshness, and avoiding unnecessary fatigue accumulation.",
        last7_load_mult=0.94,
        baseline28_load_mult=1.03,
        distance_mult=0.97,
        elevation_mult=1.02,
        long_run_emphasis=0.92,
        forecast_driver_notes=[
            "Current context is close enough to race day that freshness matters.",
            "Specificity and recovery discipline should outweigh extra volume.",
        ],
    ),
}


def fixture_id(
    athlete_band: AthleteBand,
    readiness: ReadinessLevel,
    recovery_capability: RecoveryCapabilityLevel,
    race_phase: RacePhase,
) -> str:
    return f"ab_{athlete_band}__r_{readiness}__rc_{recovery_capability}__ph_{race_phase}"


def build_fixture_meta(
    athlete_band: AthleteBand,
    readiness: ReadinessLevel,
    recovery_capability: RecoveryCapabilityLevel,
    race_phase: RacePhase,
) -> FixtureMeta:
    band = ATHLETE_BAND_PROFILES[athlete_band]
    phase = PHASE_PROFILES[race_phase]

    goal = phase.primary_goal
    if athlete_band == "A1":
        goal = "to build confidence, routine, and durable trail-running basics"
    elif athlete_band == "A2":
        goal = "to build consistent trail-running fitness with manageable structure"
    elif athlete_band == "A3":
        goal = phase.primary_goal
    elif athlete_band == "A4":
        goal = (
            "to convert strong training history into race-specific trail-running performance"
            if race_phase == "peak"
            else "to sustain high training quality while building race-relevant durability"
        )

    lifestyle_notes = (
        f"{phase.lifestyle_notes} Athlete profile: {band.label}. "
        f"Training history context: {band.athlete_profile_summary}"
    )

    return FixtureMeta(
        fixture_id=fixture_id(athlete_band, readiness, recovery_capability, race_phase),
        athlete_band=athlete_band,
        readiness=readiness,
        recovery_capability=recovery_capability,
        race_phase=race_phase,
        readiness_status=READINESS_TO_STATUS[readiness],
        recovery_key=RECOVERY_TO_KEY[recovery_capability],
        weeks_to_race=phase.weeks_to_race,
        athlete_band_label=band.label,
        athlete_archetype=band.athlete_archetype,
        athlete_profile_summary=band.athlete_profile_summary,
        recent_volume_hint=band.recent_volume_hint,
        structure_hint=band.structure_hint,
        long_run_tolerance_minutes=band.long_run_tolerance_minutes,
        block_label=phase.block_label,
        style=phase.style,
        primary_goal=goal,
        lifestyle_notes=lifestyle_notes,
        baseline_load_hours=band.baseline_load_hours,
        baseline_distance_km=band.baseline_distance_km,
        baseline_elevation_m=band.baseline_elevation_m,
        base_speed_kmh=band.base_speed_kmh,
        weight_kg=band.weight_kg,
        height_cm=band.height_cm,
        years_in_sport=band.years_in_sport,
    )


ALL_FIXTURE_SPECS: list[FixtureMeta] = [
    build_fixture_meta(ab, r, rc, ph)
    for ab, r, rc, ph in product(
        ATHLETE_BANDS,
        READINESS_LEVELS,
        RECOVERY_CAPABILITY_LEVELS,
        RACE_PHASES,
    )
]

FIXTURE_IDS: list[str] = [f.fixture_id for f in ALL_FIXTURE_SPECS]
FIXTURE_BY_ID: dict[str, FixtureMeta] = {f.fixture_id: f for f in ALL_FIXTURE_SPECS}

assert len(ALL_FIXTURE_SPECS) == 32, "Expected exactly 32 fixture bundles"
