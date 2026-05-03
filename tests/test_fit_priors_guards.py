from __future__ import annotations

import json
import pytest


def _plan_with_structure(band: str) -> dict:
    days = [
        {"date": "2026-03-18", "session_type": "rest", "is_rest_day": True, "is_hard_day": False, "duration_minutes": 0, "target_intensity": "rest", "terrain": "n/a", "workout": "Rest day."},
        {"date": "2026-03-19", "session_type": "long" if band in {"A3", "A4"} else "easy", "is_rest_day": False, "is_hard_day": False, "duration_minutes": 90, "target_intensity": "easy", "terrain": "trail", "workout": "90 min controlled trail run."},
        {"date": "2026-03-20", "session_type": "tempo" if band == "A4" else "easy", "is_rest_day": False, "is_hard_day": band == "A4", "duration_minutes": 45, "target_intensity": "threshold" if band == "A4" else "easy", "terrain": "trail", "workout": "45 min structured run."},
    ]
    return {"meta": {"style": "trailrunning"}, "plan": {"weekly_totals": {"planned_moving_time_hours": 2.25}, "days": days}}


def _write_bundle(plans_dir, *, plan_id: str, band: str):
    plan_path = plans_dir / f"{plan_id}.json"
    prov_path = plans_dir / f"{plan_id}.json.provenance.json"
    plan_path.write_text(json.dumps(_plan_with_structure(band)), encoding="utf-8")
    prov_path.write_text(json.dumps({"plan_id": plan_id, "arm": "llm", "plan_path": str(plan_path), "athlete_band": band, "readiness": "high", "recovery_capability": "high", "race_phase": "base"}), encoding="utf-8")


def test_fit_priors_aborts_on_missing_bands(tmp_path):
    from generate.fit_priors import fit_and_save
    plans_dir = tmp_path / "plans"
    plans_dir.mkdir()
    for idx in range(2):
        _write_bundle(plans_dir, plan_id=f"a1_{idx}", band="A1")
        _write_bundle(plans_dir, plan_id=f"a2_{idx}", band="A2")
    with pytest.raises(SystemExit):
        fit_and_save(plans_dir, tmp_path / "sampler.json", min_plans=1, min_plans_per_band=1)


def test_fit_priors_aborts_on_total_min_plans(tmp_path):
    from generate.fit_priors import fit_and_save
    plans_dir = tmp_path / "plans"
    plans_dir.mkdir()
    _write_bundle(plans_dir, plan_id="a1_0", band="A1")
    with pytest.raises(SystemExit):
        fit_and_save(plans_dir, tmp_path / "sampler.json", min_plans=30, min_plans_per_band=0)


def test_fit_priors_can_be_overridden_for_pilot_runs(tmp_path):
    from generate.fit_priors import fit_and_save
    plans_dir = tmp_path / "plans"
    plans_dir.mkdir()
    _write_bundle(plans_dir, plan_id="a1_0", band="A1")
    out = tmp_path / "sampler.json"
    fit_and_save(plans_dir, out, min_plans=30, min_plans_per_band=4, allow_unsafe_prior_fit=True)
    assert out.exists()
