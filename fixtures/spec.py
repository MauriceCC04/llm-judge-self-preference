"""fixtures/spec.py — axis definitions and fixture metadata schema.

The study conditions cross three binary axes:

    readiness           : low | high
    recovery_capability : low | high
    race_phase          : base | peak

Unlike the earlier version, ``race_phase`` is now prompt-visible and not merely
metadata.  It propagates into the fixture payloads via block labels, goals,
lifestyle notes, recent-load shape, and forecast language.
"""
from __future__ import annotations

from itertools import product
from typing import Literal

from pydantic import BaseModel

ReadinessLevel = Literal["low", "high"]
RecoveryCapabilityLevel = Literal["low", "high"]
RacePhase = Literal["base", "peak"]


class PhaseProfile(BaseModel):
    """Prompt-visible profile for the training phase."""

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
    """Frozen description of one fixture bundle's study context."""

    fixture_id: str
    readiness: ReadinessLevel
    recovery_capability: RecoveryCapabilityLevel
    race_phase: RacePhase

    readiness_status: Literal["fatigued", "primed"]
    recovery_key: str
    weeks_to_race: int

    # Prompt-visible metadata
    block_label: str
    style: str
    primary_goal: str
    lifestyle_notes: str
    athlete_archetype: str


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

PHASE_PROFILES: dict[RacePhase, PhaseProfile] = {
    "base": PhaseProfile(
        weeks_to_race=20,
        block_label="base development block",
        primary_goal="to become a faster and more durable trail runner",
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

ATHLETE_ARCHETYPES: tuple[str, ...] = (
    "developing_runner",
    "experienced_runner",
    "durable_high_volume_runner",
    "fatigue_prone_runner",
)


def fixture_id(r: ReadinessLevel, rc: RecoveryCapabilityLevel, ph: RacePhase) -> str:
    return f"r_{r}__rc_{rc}__ph_{ph}"


def archetype_for_fixture(
    readiness: ReadinessLevel,
    recovery_capability: RecoveryCapabilityLevel,
    race_phase: RacePhase,
) -> str:
    """Deterministic archetype assignment so fixtures are frozen and reproducible."""
    index = (
        READINESS_LEVELS.index(readiness) * 4
        + RECOVERY_CAPABILITY_LEVELS.index(recovery_capability) * 2
        + RACE_PHASES.index(race_phase)
    ) % len(ATHLETE_ARCHETYPES)
    return ATHLETE_ARCHETYPES[index]


def build_fixture_meta(
    readiness: ReadinessLevel,
    recovery_capability: RecoveryCapabilityLevel,
    race_phase: RacePhase,
) -> FixtureMeta:
    profile = PHASE_PROFILES[race_phase]
    return FixtureMeta(
        fixture_id=fixture_id(readiness, recovery_capability, race_phase),
        readiness=readiness,
        recovery_capability=recovery_capability,
        race_phase=race_phase,
        readiness_status=READINESS_TO_STATUS[readiness],
        recovery_key=RECOVERY_TO_KEY[recovery_capability],
        weeks_to_race=profile.weeks_to_race,
        block_label=profile.block_label,
        style=profile.style,
        primary_goal=profile.primary_goal,
        lifestyle_notes=profile.lifestyle_notes,
        athlete_archetype=archetype_for_fixture(readiness, recovery_capability, race_phase),
    )


ALL_FIXTURE_SPECS: list[FixtureMeta] = [
    build_fixture_meta(r, rc, ph)
    for r, rc, ph in product(READINESS_LEVELS, RECOVERY_CAPABILITY_LEVELS, RACE_PHASES)
]

FIXTURE_IDS: list[str] = [f.fixture_id for f in ALL_FIXTURE_SPECS]
FIXTURE_BY_ID: dict[str, FixtureMeta] = {f.fixture_id: f for f in ALL_FIXTURE_SPECS}

assert len(ALL_FIXTURE_SPECS) == 8, "Expected exactly 8 fixture bundles"
