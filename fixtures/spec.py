"""fixtures/spec.py — axis definitions and fixture metadata schema.

The 8 fixture bundles cross three binary axes:

    readiness          : low | high
    recovery_capability: low | high
    race_phase         : base | peak

That gives 2³ = 8 unique combinations, each committed as a subdirectory
under fixtures/data/.

A fixture bundle is a directory containing four JSON files in the exact
shapes that trailtraining expects:
    combined_summary.json
    combined_rollups.json
    readiness_and_risk_forecast.json
    formatted_personal_data.json

The ``FixtureMeta`` model describes the axis values for each bundle and
is serialised as ``fixture_meta.json`` inside the bundle directory.
"""
from __future__ import annotations

from itertools import product
from typing import Literal

from pydantic import BaseModel

# ── Axis types ──────────────────────────────────────────────────────────────

ReadinessLevel = Literal["low", "high"]
RecoveryCapabilityLevel = Literal["low", "high"]
RacePhase = Literal["base", "peak"]


class FixtureMeta(BaseModel):
    """Frozen description of one fixture bundle's axis values."""

    fixture_id: str                         # e.g. "r_low__rc_low__ph_base"
    readiness: ReadinessLevel
    recovery_capability: RecoveryCapabilityLevel
    race_phase: RacePhase

    # Derived convenience fields (set by build.py)
    readiness_status: Literal["fatigued", "primed"]   # maps readiness→forecast status
    recovery_key: str                                  # maps recovery_capability→capability key
    weeks_to_race: int                                 # maps race_phase→nominal weeks


# ── Axis values ─────────────────────────────────────────────────────────────

READINESS_LEVELS: list[ReadinessLevel] = ["low", "high"]
RECOVERY_CAPABILITY_LEVELS: list[RecoveryCapabilityLevel] = ["low", "high"]
RACE_PHASES: list[RacePhase] = ["base", "peak"]

# ── Axis → derived value maps ────────────────────────────────────────────────

READINESS_TO_STATUS: dict[ReadinessLevel, Literal["fatigued", "primed"]] = {
    "low": "fatigued",
    "high": "primed",
}

RECOVERY_TO_KEY: dict[RecoveryCapabilityLevel, str] = {
    "low": "load_only",               # no sleep/HRV data
    "high": "load_sleep_resting_hr_hrv",
}

PHASE_TO_WEEKS: dict[RacePhase, int] = {
    "base": 20,   # well into base block → develop phase
    "peak": 4,    # four weeks out → taper/race-ready
}


def fixture_id(r: ReadinessLevel, rc: RecoveryCapabilityLevel, ph: RacePhase) -> str:
    return f"r_{r}__rc_{rc}__ph_{ph}"


# ── Enumerate all 8 bundles ──────────────────────────────────────────────────

ALL_FIXTURE_SPECS: list[FixtureMeta] = [
    FixtureMeta(
        fixture_id=fixture_id(r, rc, ph),
        readiness=r,
        recovery_capability=rc,
        race_phase=ph,
        readiness_status=READINESS_TO_STATUS[r],
        recovery_key=RECOVERY_TO_KEY[rc],
        weeks_to_race=PHASE_TO_WEEKS[ph],
    )
    for r, rc, ph in product(READINESS_LEVELS, RECOVERY_CAPABILITY_LEVELS, RACE_PHASES)
]

FIXTURE_IDS: list[str] = [f.fixture_id for f in ALL_FIXTURE_SPECS]

assert len(ALL_FIXTURE_SPECS) == 8, "Expected exactly 8 fixture bundles"
