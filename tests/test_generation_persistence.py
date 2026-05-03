from __future__ import annotations

import json
import pytest
from generate.persistence import verify_json_object_file, verify_plan_and_provenance


def _minimal_plan() -> dict:
    return {"meta": {"style": "trailrunning"}, "plan": {"weekly_totals": {}, "days": []}}


def test_verify_plan_and_provenance_accepts_complete_pair(tmp_path):
    plan_path = tmp_path / "plan_001.json"
    prov_path = tmp_path / "plan_001.json.provenance.json"
    plan_path.write_text(json.dumps(_minimal_plan()), encoding="utf-8")
    prov_path.write_text(json.dumps({"plan_id": "plan_001"}), encoding="utf-8")
    plan, prov = verify_plan_and_provenance(plan_path, prov_path, expected_plan_id="plan_001")
    assert "plan" in plan
    assert prov["plan_id"] == "plan_001"


def test_verify_plan_and_provenance_rejects_missing_sidecar(tmp_path):
    plan_path = tmp_path / "plan_001.json"
    plan_path.write_text(json.dumps(_minimal_plan()), encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        verify_plan_and_provenance(plan_path, tmp_path / "missing.json", expected_plan_id="plan_001")


def test_verify_plan_and_provenance_rejects_mismatched_plan_id(tmp_path):
    plan_path = tmp_path / "plan_001.json"
    prov_path = tmp_path / "plan_001.json.provenance.json"
    plan_path.write_text(json.dumps(_minimal_plan()), encoding="utf-8")
    prov_path.write_text(json.dumps({"plan_id": "other"}), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match expected"):
        verify_plan_and_provenance(plan_path, prov_path, expected_plan_id="plan_001")


def test_verify_json_object_file_rejects_invalid_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        verify_json_object_file(path, label="test artifact")
