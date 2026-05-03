from __future__ import annotations

from pathlib import Path
import pytest
from generate.artifact_leakage import assert_no_structural_prompt_leaks
from generate.structural_expectations import build_structural_lifestyle_notes


def test_structural_notes_contain_advanced_cell_requirements():
    notes = build_structural_lifestyle_notes({"athlete_band": "A4", "readiness": "high", "recovery_capability": "high", "race_phase": "peak"})
    assert "athlete band A4" in notes
    assert "session_type='long'" in notes
    assert "tempo/intervals/hills" in notes


def test_trailtraining_compat_does_not_drop_structural_notes():
    source = Path("generate/trailtraining_compat.py").read_text(encoding="utf-8")
    assert "prompt_lifestyle_notes = build_structural_lifestyle_notes(fixture_meta)" in source
    assert 'base_lifestyle_notes = str(fixture_meta.get("lifestyle_notes") or "").strip()' in source
    assert "lifestyle_notes=prompt_lifestyle_notes" in source
    assert "lifestyle_notes=base_lifestyle_notes" in source
    assert 'lifestyle_notes=""' not in source
    assert "_assert_no_structural_prompt_leaks" in source


def test_structural_prompt_leak_guard_rejects_scaffold_text(tmp_path):
    plan = {"meta": {}, "plan": {"days": [{"purpose": "This cell expects a genuinely structured week."}]}}
    with pytest.raises(ValueError, match="Structural prompt scaffold leaked"):
        assert_no_structural_prompt_leaks(plan_id="leaky", output_path=tmp_path / "leaky.json", plan_obj=plan)


def test_structural_notes_are_band_specific():
    notes = build_structural_lifestyle_notes({"athlete_band": "A1", "readiness": "high", "recovery_capability": "high", "race_phase": "peak"})
    assert "tempo/intervals/hills" not in notes
    assert "Simple conservative structure is acceptable" in notes
