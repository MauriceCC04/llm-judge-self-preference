from __future__ import annotations

from generate.artifact_leakage import assert_no_structural_prompt_leaks


def test_effective_constraints_lifestyle_notes_accept_base_notes_only(tmp_path):
    plan = {
        "meta": {"lifestyle_notes": "Minimal test fixture."},
        "effective_constraints": {"lifestyle_notes": "Minimal test fixture."},
        "plan": {"days": []},
    }

    assert_no_structural_prompt_leaks(
        plan_id="base_ok",
        output_path=tmp_path / "base_ok.json",
        plan_obj=plan,
    )


def test_effective_constraints_lifestyle_notes_reject_structural_scaffold(tmp_path):
    plan = {
        "meta": {"lifestyle_notes": "Minimal test fixture."},
        "effective_constraints": {
            "lifestyle_notes": "Cell metadata: athlete band A4. Include session_type='long'."
        },
        "plan": {"days": []},
    }

    try:
        assert_no_structural_prompt_leaks(
            plan_id="leak",
            output_path=tmp_path / "leak.json",
            plan_obj=plan,
        )
    except ValueError as exc:
        assert "effective_constraints.lifestyle_notes" in str(exc)
    else:
        raise AssertionError("Expected structural scaffold leakage to be rejected")
