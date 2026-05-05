from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from generate.normalize_corpus_for_judging import NormalizationError, normalize_corpus, normalize_day, normalize_plan


def _day(**overrides):
    day = {
        "date": "2026-03-11",
        "title": "Day 1",
        "session_type": "easy",
        "is_rest_day": False,
        "is_hard_day": False,
        "duration_minutes": 30,
        "target_intensity": "easy",
        "terrain": "trail",
        "workout": "30 min easy run.",
        "purpose": "Maintain aerobic rhythm.",
        "signal_ids": [],
        "estimated_distance_km": None,
        "estimated_elevation_m": None,
    }
    day.update(overrides)
    return day


def _plan(days=None):
    return {
        "meta": {"style": "trailrunning"},
        "plan": {
            "weekly_totals": {"planned_moving_time_hours": 0.5},
            "days": days if days is not None else [_day()],
        },
        "readiness": {},
        "recovery": {},
        "risks": [],
        "data_notes": [],
        "citations": [],
        "claim_attributions": [],
        "effective_constraints": {},
    }


def test_rest_day_remains_consistent():
    normalized, change = normalize_day(
        _day(
            title="Recovery",
            session_type="rest",
            is_rest_day=True,
            is_hard_day=True,
            duration_minutes=15,
            target_intensity="easy",
            workout="Jog if you feel good.",
            purpose="Recover and prepare.",
        ),
        plan_id="p1",
        day_idx=0,
    )
    assert normalized["title"] == "Rest day"
    assert normalized["workout"] == "Rest day. No structured training."
    assert normalized["purpose"] == "Absorb training and restore readiness."
    assert normalized["duration_minutes"] == 0
    assert normalized["is_rest_day"] is True
    assert normalized["is_hard_day"] is False
    assert normalized["session_type"] == "rest"
    assert change["changed_fields"]


def test_zero_minute_active_session_is_converted_to_rest():
    normalized, change = normalize_day(_day(session_type="easy", is_rest_day=False, duration_minutes=0), plan_id="p1", day_idx=0)
    assert normalized["session_type"] == "rest"
    assert normalized["is_rest_day"] is True
    assert normalized["is_hard_day"] is False
    assert normalized["duration_minutes"] == 0
    assert normalized["title"] == "Rest day"
    assert change["zero_duration_active_converted"] is True


def test_generic_titles_are_removed():
    normalized, change = normalize_day(_day(title="Day 1", session_type="tempo", duration_minutes=45), plan_id="p1", day_idx=0)
    assert normalized["title"] == "Tempo run"
    assert change["generic_title_replaced"] is True


def test_active_rest_wording_is_repaired():
    normalized, change = normalize_day(
        _day(title="Day 2", session_type="easy", duration_minutes=35, workout="No structured training today.", purpose="Recover and maintain freshness."),
        plan_id="p1",
        day_idx=1,
    )
    assert normalized["session_type"] == "easy"
    assert normalized["title"] == "Easy run"
    assert normalized["workout"] == "35 min easy run at relaxed effort."
    assert normalized["purpose"] == "Build aerobic consistency with low training stress."
    assert change["active_rest_wording_repaired"] is True


def test_unknown_active_session_type_fails_loudly():
    with pytest.raises(NormalizationError, match="unknown active session_type"):
        normalize_day(_day(session_type="fartlek", duration_minutes=40), plan_id="p1", day_idx=0)


def test_qwen_problematic_shape_is_normalized():
    plan_obj = _plan([
        _day(title="Day 1", session_type="easy", duration_minutes=0, workout="No structured training.", purpose="Recover."),
        _day(title="Day 2", session_type="easy", duration_minutes=35, workout="No structured training.", purpose="Recover and maintain freshness."),
        _day(title="Day 3", session_type="long", duration_minutes=90),
    ])
    normalized, report = normalize_plan(plan_obj, plan_id="qwen_problem")
    days = normalized["plan"]["days"]
    assert days[0]["session_type"] == "rest"
    assert days[0]["title"] == "Rest day"
    assert days[1]["title"] == "Easy run"
    assert days[1]["workout"] == "35 min easy run at relaxed effort."
    assert days[2]["title"] == "Long run"
    assert report["days_changed"] == 3


def _write_pair(plans_dir: Path, plan_id: str, plan_obj: dict, *, fixture_id: str = "ab_A1__r_low__rc_low__ph_base", band: str = "A1"):
    plan_path = plans_dir / f"{plan_id}.json"
    prov_path = plans_dir / f"{plan_id}.json.provenance.json"
    plan_path.write_text(json.dumps(plan_obj, sort_keys=True), encoding="utf-8")
    prov_path.write_text(json.dumps({
        "plan_id": plan_id,
        "fixture_id": fixture_id,
        "athlete_band": band,
        "readiness": "low",
        "recovery_capability": "low",
        "race_phase": "base",
        "arm": "llm",
        "explainer_model": "Qwen/Qwen2.5-3B-Instruct",
        "seed": 0,
        "generated_at": "2026-03-17T00:00:00Z",
        "plan_path": str(plan_path),
    }, sort_keys=True), encoding="utf-8")


def test_end_to_end_copy_does_not_mutate_input_and_writes_manifest_and_report(tmp_path):
    input_dir = tmp_path / "raw" / "plans"
    output_dir = tmp_path / "normalized" / "plans"
    input_dir.mkdir(parents=True)
    raw_plan = _plan([
        _day(title="Day 1", session_type="easy", duration_minutes=0),
        _day(title="Day 2", session_type="easy", duration_minutes=30),
    ])
    raw_plan_before = copy.deepcopy(raw_plan)
    _write_pair(input_dir, "plan_001", raw_plan)

    manifest = tmp_path / "normalized" / "manifest_sha256.txt"
    report = tmp_path / "normalized" / "normalization_report.json"
    report_jsonl = tmp_path / "normalized" / "normalization_report.jsonl"

    result = normalize_corpus(
        input_dir,
        output_dir,
        manifest_path=manifest,
        report_path=report,
        report_jsonl_path=report_jsonl,
        expected_plans=1,
        expected_cells=1,
        expected_plans_per_cell=1,
    )

    assert (output_dir / "plan_001.json").exists()
    assert (output_dir / "plan_001.json.provenance.json").exists()
    assert manifest.exists()
    assert report.exists()
    assert report_jsonl.exists()
    assert len(manifest.read_text(encoding="utf-8").strip().splitlines()) == 2
    assert json.loads((input_dir / "plan_001.json").read_text(encoding="utf-8")) == raw_plan_before
    out_plan = json.loads((output_dir / "plan_001.json").read_text(encoding="utf-8"))
    assert out_plan["plan"]["days"][0]["session_type"] == "rest"
    assert result["summary"]["zero_duration_active_converted"] == 1
    sidecar = json.loads((output_dir / "plan_001.json.provenance.json").read_text(encoding="utf-8"))
    assert sidecar["plan_id"] == "plan_001"
    assert sidecar["plan_path"] == str(output_dir / "plan_001.json")
