"""Supplemental regression checks for the HPC-alignment fixes.

These are intentionally lightweight and can be adapted into the repo's custom
Gate-0 runner if desired.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from fixtures.build import build_all
from fixtures.spec import ALL_FIXTURE_SPECS


def _signature(bundle_dir: Path) -> str:
    digest = hashlib.sha256()
    for name in (
        "combined_summary.json",
        "combined_rollups.json",
        "readiness_and_risk_forecast.json",
        "formatted_personal_data.json",
    ):
        digest.update((bundle_dir / name).read_bytes())
    return digest.hexdigest()


def test_fixture_payloads_are_all_unique(tmp_path: Path) -> None:
    build_all(tmp_path)
    signatures = [_signature(tmp_path / spec.fixture_id) for spec in ALL_FIXTURE_SPECS]
    assert len(signatures) == len(set(signatures)) == 8


def test_rollups_match_combined_summary(tmp_path: Path) -> None:
    build_all(tmp_path)
    sample_dir = tmp_path / ALL_FIXTURE_SPECS[0].fixture_id
    combined = json.loads((sample_dir / "combined_summary.json").read_text(encoding="utf-8"))
    rollups = json.loads((sample_dir / "combined_rollups.json").read_text(encoding="utf-8"))
    recent = combined[-7:]
    expected_sleep_days = sum(1 for day in recent if day.get("sleep"))
    expected_activity_count = sum(len(day.get("activities") or []) for day in recent)
    assert rollups["windows"]["7"]["sleep_days_with_data"] == expected_sleep_days
    assert rollups["windows"]["7"]["activities"]["count"] == expected_activity_count
