#!/usr/bin/env python3
"""Repair structural prompt scaffold leakage through effective_constraints.lifestyle_notes.

Run from the root of llm-judge-self-preference after applying the previous
final_generation_fix_files package:

    python repair_effective_constraint_lifestyle_leak.py

Then rerun:

    python tests/run_tests.py
    pytest tests/test_generation_persistence.py tests/test_prompt_conditioning_regression.py \
           tests/test_run_generation_persistence.py tests/test_fit_priors_guards.py -v
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path.cwd()


def _read(path: str) -> str:
    p = ROOT / path
    if not p.exists():
        raise FileNotFoundError(f"Missing expected file: {p}")
    return p.read_text(encoding="utf-8")


def _write(path: str, text: str) -> None:
    (ROOT / path).write_text(text, encoding="utf-8")


def _replace_once(text: str, old: str, new: str, *, path: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected exactly one occurrence in {path}, found {count}:\n{old}")
    return text.replace(old, new, 1)


def patch_trailtraining_compat() -> None:
    path = "generate/trailtraining_compat.py"
    text = _read(path)

    # Make the rich prompt notes explicitly wrap only the base lifestyle notes,
    # then keep effective constraints/artifact serialization on the base notes.
    old = '    prompt_lifestyle_notes = build_structural_lifestyle_notes(fixture_meta)\n    base_lifestyle_notes = str(fixture_meta.get("lifestyle_notes") or "").strip()\n'
    new = '    base_lifestyle_notes = str(fixture_meta.get("lifestyle_notes") or "").strip()\n    prompt_lifestyle_notes = build_structural_lifestyle_notes(\n        fixture_meta,\n        base_lifestyle_notes=base_lifestyle_notes,\n    )\n'
    if old in text:
        text = text.replace(old, new, 1)
    elif "base_lifestyle_notes = str(fixture_meta.get(\"lifestyle_notes\") or \"\").strip()" not in text:
        old2 = '    prompt_lifestyle_notes = build_structural_lifestyle_notes(fixture_meta)\n'
        new2 = new
        text = _replace_once(text, old2, new2, path=path)

    old = '''    effective = derive_effective_constraints(
        det_forecast=data["forecast"],
        rollups=data["rollups"],
        cfg=constraint_config_from_env(),
        lifestyle_notes=prompt_lifestyle_notes,
    )'''
    new = '''    effective = derive_effective_constraints(
        det_forecast=data["forecast"],
        rollups=data["rollups"],
        cfg=constraint_config_from_env(),
        lifestyle_notes=base_lifestyle_notes,
    )'''
    if old in text:
        text = text.replace(old, new, 1)

    # Defensive artifact-level scrub: even if future code accidentally derives
    # effective constraints from rich notes, the saved judge-visible artifact must
    # only carry base lifestyle notes.
    marker = '''    obj = _finalize_training_plan_artifact(
        obj,
        combined=data["combined"],
        rollups=data["rollups"],
        deterministic_forecast=data["forecast"],
        effective=effective,
    )
    obj = _normalize_rest_day_semantics(obj)
'''
    replacement = '''    obj = _finalize_training_plan_artifact(
        obj,
        combined=data["combined"],
        rollups=data["rollups"],
        deterministic_forecast=data["forecast"],
        effective=effective,
    )
    if isinstance(obj.get("effective_constraints"), dict):
        obj["effective_constraints"]["lifestyle_notes"] = base_lifestyle_notes
    obj = _normalize_rest_day_semantics(obj)
'''
    if marker in text and replacement not in text:
        text = text.replace(marker, replacement, 1)

    _write(path, text)


def patch_programmatic_arm() -> None:
    path = "generate/programmatic_arm.py"
    text = _read(path)

    old = '''    prompt_lifestyle_notes = build_structural_lifestyle_notes(
        fixture_meta,
        base_lifestyle_notes=cfg.lifestyle_notes,
    )
    base_lifestyle_notes = str(fixture_meta.get("lifestyle_notes") or cfg.lifestyle_notes or "").strip()
'''
    new = '''    base_lifestyle_notes = str(fixture_meta.get("lifestyle_notes") or cfg.lifestyle_notes or "").strip()
    prompt_lifestyle_notes = build_structural_lifestyle_notes(
        fixture_meta,
        base_lifestyle_notes=base_lifestyle_notes,
    )
'''
    if old in text:
        text = text.replace(old, new, 1)
    elif "base_lifestyle_notes = str(fixture_meta.get(\"lifestyle_notes\") or cfg.lifestyle_notes or \"\").strip()" not in text:
        old2 = '''    prompt_lifestyle_notes = build_structural_lifestyle_notes(
        fixture_meta,
        base_lifestyle_notes=cfg.lifestyle_notes,
    )
'''
        text = _replace_once(text, old2, new, path=path)

    old = '''    effective = derive_effective_constraints(
        det_forecast=data["forecast"],
        rollups=data["rollups"],
        cfg=constraint_config_from_env(),
        lifestyle_notes=prompt_lifestyle_notes,
    )'''
    new = '''    effective = derive_effective_constraints(
        det_forecast=data["forecast"],
        rollups=data["rollups"],
        cfg=constraint_config_from_env(),
        lifestyle_notes=base_lifestyle_notes,
    )'''
    if old in text:
        text = text.replace(old, new, 1)

    marker = '''    obj = _finalize_training_plan_artifact(
        obj,
        combined=source_data.combined,
        rollups=source_data.rollups,
        deterministic_forecast=deterministic_forecast,
        effective=effective,
    )

    _assert_no_placeholder_leaks(
'''
    replacement = '''    obj = _finalize_training_plan_artifact(
        obj,
        combined=source_data.combined,
        rollups=source_data.rollups,
        deterministic_forecast=deterministic_forecast,
        effective=effective,
    )
    if isinstance(obj.get("effective_constraints"), dict):
        obj["effective_constraints"]["lifestyle_notes"] = artifact_lifestyle_notes

    _assert_no_placeholder_leaks(
'''
    if marker in text and replacement not in text:
        text = text.replace(marker, replacement, 1)

    _write(path, text)


def add_targeted_test() -> None:
    path = ROOT / "tests" / "test_effective_constraints_no_structural_leak.py"
    path.write_text(
        '''from __future__ import annotations\n\nfrom generate.artifact_leakage import assert_no_structural_prompt_leaks\n\n\ndef test_effective_constraints_lifestyle_notes_accept_base_notes_only(tmp_path):\n    plan = {\n        "meta": {"lifestyle_notes": "Minimal test fixture."},\n        "effective_constraints": {"lifestyle_notes": "Minimal test fixture."},\n        "plan": {"days": []},\n    }\n\n    assert_no_structural_prompt_leaks(\n        plan_id="base_ok",\n        output_path=tmp_path / "base_ok.json",\n        plan_obj=plan,\n    )\n\n\ndef test_effective_constraints_lifestyle_notes_reject_structural_scaffold(tmp_path):\n    plan = {\n        "meta": {"lifestyle_notes": "Minimal test fixture."},\n        "effective_constraints": {\n            "lifestyle_notes": "Cell metadata: athlete band A4. Include session_type='long'."\n        },\n        "plan": {"days": []},\n    }\n\n    try:\n        assert_no_structural_prompt_leaks(\n            plan_id="leak",\n            output_path=tmp_path / "leak.json",\n            plan_obj=plan,\n        )\n    except ValueError as exc:\n        assert "effective_constraints.lifestyle_notes" in str(exc)\n    else:\n        raise AssertionError("Expected structural scaffold leakage to be rejected")\n''',
        encoding="utf-8",
    )


def main() -> int:
    patch_trailtraining_compat()
    patch_programmatic_arm()
    add_targeted_test()
    print("Repaired effective_constraints.lifestyle_notes structural scaffold leakage.")
    print("Run:")
    print("  python tests/run_tests.py")
    print("  pytest tests/test_generation_persistence.py tests/test_prompt_conditioning_regression.py tests/test_run_generation_persistence.py tests/test_fit_priors_guards.py tests/test_effective_constraints_no_structural_leak.py -v")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
