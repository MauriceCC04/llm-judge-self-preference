from __future__ import annotations

from pathlib import Path

TEST_PATH = Path('tests/test_prompt_conditioning_regression.py')

NEW_CONTENT = '''from __future__ import annotations

from pathlib import Path

from generate.artifact_leakage import collect_structural_prompt_leaks
from generate.structural_expectations import build_structural_lifestyle_notes


def test_structural_notes_contain_advanced_cell_requirements():
    notes = build_structural_lifestyle_notes(
        {
            "athlete_band": "A4",
            "readiness": "high",
            "recovery_capability": "high",
            "race_phase": "peak",
        }
    )

    assert "athlete band A4" in notes
    assert "session_type='long'" in notes
    assert "tempo/intervals/hills" in notes


def test_trailtraining_compat_does_not_drop_structural_notes():
    source = Path("generate/trailtraining_compat.py").read_text(encoding="utf-8")

    # Rich fixture-derived notes must be constructed, even if the call is multiline.
    assert "prompt_lifestyle_notes = build_structural_lifestyle_notes(" in source
    assert "base_lifestyle_notes=base_lifestyle_notes" in source

    # Rich notes should reach the source-stage and explainer-stage prompt builders.
    assert "lifestyle_notes=prompt_lifestyle_notes" in source

    # Base notes, not rich prompt scaffolding, should be used for artifact-visible paths.
    assert "lifestyle_notes=base_lifestyle_notes" in source
    assert "lifestyle_notes=\"\"" not in source


def test_structural_prompt_leak_guard_rejects_scaffold_text():
    obj = {
        "meta": {"lifestyle_notes": "normal lifestyle note"},
        "data_notes": ["Cell metadata: athlete band A4, readiness high."],
    }
    hits = collect_structural_prompt_leaks(obj)
    assert hits


def test_structural_notes_are_band_specific():
    a1_notes = build_structural_lifestyle_notes(
        {
            "athlete_band": "A1",
            "readiness": "high",
            "recovery_capability": "high",
            "race_phase": "peak",
        }
    )
    a4_notes = build_structural_lifestyle_notes(
        {
            "athlete_band": "A4",
            "readiness": "high",
            "recovery_capability": "high",
            "race_phase": "peak",
        }
    )

    assert "tempo/intervals/hills" not in a1_notes
    assert "tempo/intervals/hills" in a4_notes
'''


def main() -> int:
    if not TEST_PATH.exists():
        raise SystemExit(f"Missing expected test file: {TEST_PATH}")
    TEST_PATH.write_text(NEW_CONTENT, encoding='utf-8')
    print(f"Updated {TEST_PATH} to match multiline prompt-conditioning implementation.")
    print("Run:")
    print("  pytest tests/test_prompt_conditioning_regression.py tests/test_effective_constraints_no_structural_leak.py -v")
    print("  python tests/run_tests.py")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
