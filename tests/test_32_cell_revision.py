from __future__ import annotations

import json
from pathlib import Path

from analyze.load import load_judgments
from fixtures.spec import ALL_FIXTURE_SPECS, ATHLETE_BANDS
from generate.provenance import PlanProvenance
from match.filtering import filter_plan_records
from match.pair import greedy_pair


def _minimal_plan(
    *,
    title: str = "Easy run",
    session_type: str = "easy",
    duration_minutes: int = 45,
    purpose: str = "Aerobic maintenance",
    workout: str = "45 min easy run",
) -> dict:
    return {
        "meta": {
            "today": "2026-03-17",
            "plan_start": "2026-03-18",
            "plan_days": 7,
            "style": "trailrunning",
            "primary_goal": "test",
            "lifestyle_notes": "",
        },
        "snapshot": {
            "last7": {
                "distance_km": "20",
                "moving_time_hours": "2.0",
                "elevation_m": "300",
                "activity_count": "2",
                "sleep_hours_mean": "",
                "hrv_mean": "",
                "rhr_mean": "46",
            },
            "baseline28": {
                "distance_km": "18",
                "moving_time_hours": "1.8",
                "elevation_m": "250",
                "activity_count": "2",
                "sleep_hours_mean": "",
                "hrv_mean": "",
                "rhr_mean": "45",
            },
            "notes": "",
        },
        "readiness": {"status": "steady", "rationale": "Test.", "signal_ids": []},
        "plan": {
            "weekly_totals": {
                "planned_distance_km": None,
                "planned_moving_time_hours": 0.75,
                "planned_elevation_m": None,
            },
            "days": [
                {
                    "date": "2026-03-18",
                    "title": title,
                    "session_type": session_type,
                    "is_rest_day": session_type == "rest",
                    "is_hard_day": session_type in {"tempo", "intervals", "hills"},
                    "duration_minutes": duration_minutes,
                    "target_intensity": "rest" if session_type == "rest" else "easy",
                    "terrain": "trail" if session_type != "rest" else "n/a",
                    "workout": workout,
                    "purpose": purpose,
                    "signal_ids": [],
                    "estimated_distance_km": None,
                    "estimated_elevation_m": None,
                }
            ],
        },
        "recovery": {"actions": [], "signal_ids": []},
        "risks": [],
        "data_notes": [],
        "citations": [],
        "claim_attributions": [],
        "effective_constraints": {},
    }


def test_fixture_spec_expands_to_32_cells() -> None:
    assert len(ALL_FIXTURE_SPECS) == 32
    assert {spec.athlete_band for spec in ALL_FIXTURE_SPECS} == set(ATHLETE_BANDS)


def test_filter_drops_human_facing_rest_day_contradiction() -> None:
    records = [
        {
            "plan_id": "bad_001",
            "fixture_id": "ab_A1__r_low__rc_low__ph_base",
            "arm": "llm",
            "plan_obj": _minimal_plan(title="Rest day", session_type="easy", duration_minutes=40),
        }
    ]
    result = filter_plan_records(records)
    assert result["audit"]["n_kept"] == 0
    assert result["audit"]["drop_reasons"]["invalid_or_contradictory"] == 1


def test_filter_dedups_exact_text_within_cell_and_arm() -> None:
    plan = _minimal_plan()
    records = [
        {"plan_id": "llm_001", "fixture_id": "ab_A2__r_high__rc_high__ph_base", "arm": "llm", "plan_obj": plan},
        {"plan_id": "llm_002", "fixture_id": "ab_A2__r_high__rc_high__ph_base", "arm": "llm", "plan_obj": plan},
    ]
    result = filter_plan_records(records)
    assert result["audit"]["n_kept"] == 1
    assert result["audit"]["drop_reasons"]["exact_text_duplicate"] == 1


def test_filter_dedups_session_signature_within_cell_and_arm() -> None:
    plan_a = _minimal_plan(title="Easy run", purpose="Aerobic support", workout="45 min easy run")
    plan_b = _minimal_plan(
        title="Steady aerobic day",
        purpose="Keep routine consistent",
        workout="45 min comfortable running",
    )
    records = [
        {"plan_id": "llm_001", "fixture_id": "ab_A3__r_high__rc_low__ph_peak", "arm": "llm", "plan_obj": plan_a},
        {"plan_id": "llm_002", "fixture_id": "ab_A3__r_high__rc_low__ph_peak", "arm": "llm", "plan_obj": plan_b},
    ]
    result = filter_plan_records(records)
    assert result["audit"]["n_kept"] == 1
    assert result["audit"]["drop_reasons"]["session_signature_duplicate"] == 1


def test_greedy_pair_never_crosses_full_cell() -> None:
    plans = [
        {
            "plan_id": "llm_a1",
            "fixture_id": "ab_A1__r_low__rc_low__ph_base",
            "athlete_band": "A1",
            "readiness": "low",
            "recovery_capability": "low",
            "race_phase": "base",
            "plan_days": 7,
            "style": "trailrunning",
            "score": 80.0,
            "arm": "llm",
            "total_minutes": 300.0,
            "n_rest_days": 1.0,
            "n_hard_days": 1.0,
            "has_long_run": 1.0,
        },
        {
            "plan_id": "prog_a2",
            "fixture_id": "ab_A2__r_low__rc_low__ph_base",
            "athlete_band": "A2",
            "readiness": "low",
            "recovery_capability": "low",
            "race_phase": "base",
            "plan_days": 7,
            "style": "trailrunning",
            "score": 80.2,
            "arm": "programmatic",
            "total_minutes": 302.0,
            "n_rest_days": 1.0,
            "n_hard_days": 1.0,
            "has_long_run": 1.0,
        },
    ]
    assert greedy_pair(plans, tolerance=1.0) == []


def test_provenance_round_trip_includes_athlete_band(tmp_path: Path) -> None:
    path = tmp_path / "plan.json.provenance.json"
    prov = PlanProvenance(
        plan_id="plan_001",
        fixture_id="ab_A4__r_high__rc_high__ph_peak",
        athlete_band="A4",
        readiness="high",
        recovery_capability="high",
        race_phase="peak",
        plan_days=7,
        style="trailrunning",
        arm="llm",
        source_model="Qwen/Qwen2.5-7B-Instruct",
        explainer_model="Qwen/Qwen2.5-3B-Instruct",
        seed=0,
        generated_at="2026-03-17T00:00:00Z",
        plan_path=str(tmp_path / "plan.json"),
    )
    path.write_text(prov.model_dump_json(indent=2), encoding="utf-8")
    loaded = PlanProvenance.model_validate_json(path.read_text(encoding="utf-8"))
    assert loaded.athlete_band == "A4"
    assert loaded.race_phase == "peak"


def test_analysis_load_includes_athlete_band(tmp_path: Path) -> None:
    judgments_dir = tmp_path / "judgments"
    provenance_dir = tmp_path / "plans"
    judgments_dir.mkdir()
    provenance_dir.mkdir()

    for plan_id, arm in [("llm_001", "llm"), ("prog_001", "programmatic")]:
        (provenance_dir / f"{plan_id}.json").write_text(json.dumps(_minimal_plan()), encoding="utf-8")
        prov = PlanProvenance(
            plan_id=plan_id,
            fixture_id="ab_A3__r_high__rc_low__ph_base",
            athlete_band="A3",
            readiness="high",
            recovery_capability="low",
            race_phase="base",
            plan_days=7,
            style="trailrunning",
            arm=arm,  # type: ignore[arg-type]
            source_model="Qwen/Qwen2.5-7B-Instruct" if arm == "llm" else None,
            explainer_model="Qwen/Qwen2.5-3B-Instruct",
            seed=0,
            generated_at="2026-03-17T00:00:00Z",
            plan_path=str(provenance_dir / f"{plan_id}.json"),
        )
        (provenance_dir / f"{plan_id}.json.provenance.json").write_text(
            prov.model_dump_json(indent=2),
            encoding="utf-8",
        )

    row = {
        "pair_id": "pair_0001",
        "plan_a_id": "llm_001",
        "plan_b_id": "prog_001",
        "preferred": "plan_a",
        "preferred_id": "llm_001",
        "judge": "qwen_7b_judge",
        "position": "AB",
        "run": 0,
    }
    (judgments_dir / "pairwise_test.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    df = load_judgments(judgments_dir, provenance_dir, kind="pairwise")
    assert "athlete_band" in df.columns
    assert "readiness" in df.columns
    assert "recovery_capability" in df.columns
    assert "race_phase" in df.columns
    assert df.iloc[0]["athlete_band"] == "A3"
