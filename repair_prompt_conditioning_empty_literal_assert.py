from __future__ import annotations

from pathlib import Path

path = Path("tests/test_prompt_conditioning_regression.py")
if not path.exists():
    raise SystemExit(f"Missing {path}; run this from the llm-judge-self-preference repo root.")

text = path.read_text(encoding="utf-8")

# The prior repair accidentally left a Python string literal that evaluates to
# "lifestyle_notes=", which rejects all valid keyword uses. The intended check is
# only that the old empty-string bug is gone.
replacements = {
    'assert "lifestyle_notes=""" not in source': (
        'assert \'lifestyle_notes=""\' not in source\n'
        '    assert "lifestyle_notes=\\\'\\\'" not in source'
    ),
    "assert 'lifestyle_notes=' not in source": (
        'assert \'lifestyle_notes=""\' not in source\n'
        '    assert "lifestyle_notes=\\\'\\\'" not in source'
    ),
    'assert "lifestyle_notes=" not in source': (
        'assert \'lifestyle_notes=""\' not in source\n'
        '    assert "lifestyle_notes=\\\'\\\'" not in source'
    ),
}

changed = False
for old, new in replacements.items():
    if old in text:
        text = text.replace(old, new)
        changed = True

# Also normalize the test body if the earlier patch produced a slightly different
# but still fragile source-inspection assertion.
if not changed:
    old_line = '    assert "lifestyle_notes=" not in source\n'
    if old_line in text:
        text = text.replace(
            old_line,
            '    assert \'lifestyle_notes=""\' not in source\n'
            '    assert "lifestyle_notes=\\\'\\\'" not in source\n',
        )
        changed = True

if not changed:
    # Last-resort robust function rewrite.
    start_marker = "def test_trailtraining_compat_does_not_drop_structural_notes():"
    start = text.find(start_marker)
    if start == -1:
        raise SystemExit("Could not find test_trailtraining_compat_does_not_drop_structural_notes")
    next_def = text.find("\ndef ", start + 1)
    if next_def == -1:
        next_def = len(text)
    new_func = '''def test_trailtraining_compat_does_not_drop_structural_notes():
    source = Path("generate/trailtraining_compat.py").read_text(encoding="utf-8")

    assert "prompt_lifestyle_notes = build_structural_lifestyle_notes(" in source
    assert "base_lifestyle_notes=base_lifestyle_notes" in source
    assert "lifestyle_notes=prompt_lifestyle_notes" in source
    assert "lifestyle_notes=base_lifestyle_notes" in source
    assert "derive_effective_constraints(" in source
    assert "lifestyle_notes=base_lifestyle_notes" in source

    # Regression for the original bug: the source and explainer prompt calls must
    # not hard-code an empty lifestyle_notes string.
    assert 'lifestyle_notes=""' not in source
    assert "lifestyle_notes=''" not in source

'''
    text = text[:start] + new_func + text[next_def + 1 if next_def < len(text) else next_def:]
    changed = True

path.write_text(text, encoding="utf-8")
print("Updated prompt-conditioning test to reject only empty lifestyle_notes literals, not valid keyword uses.")
print("Run:")
print("  pytest tests/test_prompt_conditioning_regression.py tests/test_effective_constraints_no_structural_leak.py -v")
print("  python tests/run_tests.py")
