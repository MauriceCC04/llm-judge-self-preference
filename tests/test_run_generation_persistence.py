from __future__ import annotations

import json
import types


def _prepare_single_fixture(monkeypatch, tmp_path):
    import generate.run_generation as rg
    fixture_root = tmp_path / "fixtures" / "data" / "f0"
    fixture_root.mkdir(parents=True)
    monkeypatch.setattr(rg, "_ROOT", tmp_path)
    monkeypatch.setattr(rg, "_resolve_fixture_specs", lambda fixture_ids: [types.SimpleNamespace(fixture_id="f0")])
    return rg


def test_run_llm_arm_does_not_count_missing_files(monkeypatch, tmp_path):
    rg = _prepare_single_fixture(monkeypatch, tmp_path)
    import generate.llm_arm as llm_arm
    def fake_generate_llm_plan(**kwargs):
        output_dir = kwargs["output_dir"]
        plan_id = kwargs["plan_id"]
        return "{}", str(output_dir / f"{plan_id}.json"), str(output_dir / f"{plan_id}.json.provenance.json")
    monkeypatch.setattr(llm_arm, "generate_llm_plan", fake_generate_llm_plan)
    out = tmp_path / "out"
    generated, skipped = rg.run_llm_arm(out, plans_per_fixture=1, source_model="Qwen/Qwen2.5-7B-Instruct")
    assert generated == 0
    assert skipped == 0
    assert "plan artifact was not written" in (out / "failed_plans.jsonl").read_text(encoding="utf-8")


def test_run_programmatic_arm_does_not_count_missing_files(monkeypatch, tmp_path):
    rg = _prepare_single_fixture(monkeypatch, tmp_path)
    import generate.programmatic_arm as programmatic_arm
    def fake_generate_programmatic_plan(**kwargs):
        output_dir = kwargs["output_dir"]
        plan_id = kwargs["plan_id"]
        return "{}", str(output_dir / f"{plan_id}.json"), str(output_dir / f"{plan_id}.json.provenance.json")
    monkeypatch.setattr(programmatic_arm, "generate_programmatic_plan", fake_generate_programmatic_plan)
    out = tmp_path / "out"
    generated, skipped = rg.run_programmatic_arm(out, plans_per_fixture=1)
    assert generated == 0
    assert skipped == 0
    assert "plan artifact was not written" in (out / "failed_plans.jsonl").read_text(encoding="utf-8")


def test_run_generation_skips_only_complete_pairs(monkeypatch, tmp_path):
    rg = _prepare_single_fixture(monkeypatch, tmp_path)
    import generate.llm_arm as llm_arm
    def fake_generate_llm_plan(**kwargs):
        output_dir = kwargs["output_dir"]
        plan_id = kwargs["plan_id"]
        plan_path = output_dir / f"{plan_id}.json"
        prov_path = output_dir / f"{plan_id}.json.provenance.json"
        plan_path.write_text(json.dumps({"plan": {"days": []}}), encoding="utf-8")
        prov_path.write_text(json.dumps({"plan_id": plan_id}), encoding="utf-8")
        return "{}", str(plan_path), str(prov_path)
    monkeypatch.setattr(llm_arm, "generate_llm_plan", fake_generate_llm_plan)
    out = tmp_path / "out"
    out.mkdir()
    stale_plan_id = "f0__qwen2.5_7b_instruct__src_t070__exp_t000__s000"
    (out / f"{stale_plan_id}.json").write_text(json.dumps({"plan": {"days": []}}), encoding="utf-8")
    generated, skipped = rg.run_llm_arm(out, plans_per_fixture=1, source_model="Qwen/Qwen2.5-7B-Instruct")
    assert generated == 1
    assert skipped == 0
