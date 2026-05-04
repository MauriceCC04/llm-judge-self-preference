from __future__ import annotations

import json
from types import SimpleNamespace

import generate.run_generation as rg


def _minimal_plan() -> dict:
    return {
        "meta": {"style": "trailrunning"},
        "plan": {
            "weekly_totals": {},
            "days": [
                {
                    "date": "2026-03-11",
                    "title": "Easy run",
                    "session_type": "easy",
                    "is_rest_day": False,
                    "is_hard_day": False,
                    "duration_minutes": 30,
                    "target_intensity": "easy",
                    "terrain": "trail",
                    "workout": "30 min easy run.",
                    "purpose": "Maintain aerobic rhythm.",
                    "signal_ids": [],
                }
            ],
        },
    }


def test_run_llm_arm_tops_up_after_failed_candidate_seeds(monkeypatch, tmp_path):
    fixture_id = "ab_A4__r_low__rc_high__ph_base"
    fixture_dir = tmp_path / "fixtures" / "data" / fixture_id
    fixture_dir.mkdir(parents=True)

    monkeypatch.setattr(rg, "_ROOT", tmp_path)
    monkeypatch.setattr(
        rg,
        "_resolve_fixture_specs",
        lambda fixture_ids: [SimpleNamespace(fixture_id=fixture_id)],
    )

    import generate.llm_arm as llm_arm

    def fake_generate_llm_plan(*, output_dir, plan_id, seed, **kwargs):
        if seed in {0, 1}:
            raise RuntimeError(f"synthetic rejection seed={seed}")
        plan_path = output_dir / f"{plan_id}.json"
        prov_path = output_dir / f"{plan_id}.json.provenance.json"
        plan_obj = _minimal_plan()
        plan_path.write_text(json.dumps(plan_obj), encoding="utf-8")
        prov_path.write_text(json.dumps({"plan_id": plan_id}), encoding="utf-8")
        return json.dumps(plan_obj), str(plan_path), str(prov_path)

    monkeypatch.setattr(llm_arm, "generate_llm_plan", fake_generate_llm_plan)

    generated, skipped = rg.run_llm_arm(
        tmp_path / "plans",
        plans_per_fixture=2,
        source_model="Qwen/Qwen2.5-7B-Instruct",
        seed_offset=0,
        fixture_ids=[fixture_id],
        max_attempts_per_fixture=4,
    )

    assert generated == 2
    assert skipped == 0
    assert len(list((tmp_path / "plans").glob("*.json"))) == 4  # 2 plans + 2 provenance sidecars
    failures = (tmp_path / "plans" / "failed_plans.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(failures) == 2
