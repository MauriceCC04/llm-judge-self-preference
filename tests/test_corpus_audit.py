from __future__ import annotations

import json
from pathlib import Path

from tools.audit_plan_corpus import audit_corpus
from tools.normalize_plan_artifacts import normalize_corpus


def _plan(title: str = "Day 1", session_type: str = "easy"):
    days = []
    for idx in range(7):
        st = session_type if idx == 0 else "rest"
        days.append(
            {
                "date": f"2026-03-{18 + idx:02d}",
                "title": title if idx == 0 else "Rest day",
                "session_type": st,
                "is_rest_day": st == "rest",
                "is_hard_day": False,
                "duration_minutes": 0 if st == "rest" else 45,
                "target_intensity": "rest" if st == "rest" else "moderate",
                "workout": "Easy run. Keep it comfortable." if st != "rest" else "Rest day. No structured training.",
                "purpose": "Aerobic support." if st != "rest" else "Recover and maintain freshness.",
            }
        )
    return {"meta": {"plan_days": 7, "style": "trailrunning"}, "plan": {"days": days, "weekly_totals": {}}}


def _write_pair(root: Path, plan_id: str, plan_obj: dict, *, cell: str = "ab_A1__r_low__rc_low__ph_base") -> None:
    plan_path = root / f"{plan_id}.json"
    plan_path.write_text(json.dumps(plan_obj, indent=2), encoding="utf-8")
    prov = {
        "plan_id": plan_id,
        "fixture_id": cell,
        "athlete_band": "A1",
        "readiness": "low",
        "recovery_capability": "low",
        "race_phase": "base",
        "plan_days": 7,
        "style": "trailrunning",
        "arm": "llm",
        "source_model": "Qwen/Qwen2.5-7B-Instruct",
        "explainer_model": "Qwen/Qwen2.5-3B-Instruct",
        "seed": 0,
        "generated_at": "2026-05-05T00:00:00+00:00",
        "plan_path": str(plan_path),
        "runtime_metadata": {},
    }
    (root / f"{plan_id}.json.provenance.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")


def test_normalize_corpus_repairs_presentation_and_writes_sidecar_metadata_and_manifest(tmp_path: Path):
    plans = tmp_path / "full_qwen" / "plans"
    plans.mkdir(parents=True)
    _write_pair(plans, "p1", _plan(title="Day 1", session_type="easy"))

    audit = normalize_corpus(plans_dir=plans)

    assert audit["n_changed_plans"] == 1
    assert (plans.parent / "manifest_sha256.txt").exists()
    normalized = json.loads((plans / "p1.json").read_text(encoding="utf-8"))
    assert normalized["plan"]["days"][0]["title"] == "Easy run"
    prov = json.loads((plans / "p1.json.provenance.json").read_text(encoding="utf-8"))
    assert prov["runtime_metadata"]["artifact_normalization"]["normalizer_version"]

    final_audit = audit_corpus(
        plans_dir=plans,
        expected_plan_count=1,
        expected_sidecar_count=1,
        expected_cell_count=1,
        expected_per_cell=1,
        require_manifest=True,
    )
    assert final_audit["passed"]


def test_audit_detects_missing_sidecars_junk_debug_and_presentation_issues(tmp_path: Path):
    plans = tmp_path / "full_gemma3" / "plans"
    plans.mkdir(parents=True)
    (plans / ".DS_Store").write_text("junk", encoding="utf-8")
    (plans / "failed_plans.jsonl").write_text("{}\n", encoding="utf-8")
    (plans / "raw_failures").mkdir()
    (plans / "orphan.json").write_text(json.dumps(_plan(title="Day 1")), encoding="utf-8")

    audit = audit_corpus(plans_dir=plans, expected_plan_count=1, expected_sidecar_count=1, expected_cell_count=1, expected_per_cell=1)

    assert not audit["passed"]
    assert any(item.startswith("missing_sidecars") for item in audit["failures"])
    assert any(item.startswith("junk_files") for item in audit["failures"])
    assert any(item.startswith("debug_artifacts_in_plans") for item in audit["failures"])
