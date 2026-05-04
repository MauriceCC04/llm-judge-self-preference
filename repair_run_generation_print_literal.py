from __future__ import annotations

import py_compile
from pathlib import Path

path = Path('generate/run_generation.py')
if not path.exists():
    raise SystemExit(f'Missing expected file: {path}')

text = path.read_text(encoding='utf-8')
original = text

# v2 patcher accidentally emitted this as two physical lines:
#     print("
# === Done ===")
# which is a SyntaxError. Convert it back to an escaped newline inside the string.
text = text.replace('    print("\n=== Done ===")', '    print("\\n=== Done ===")')

# Same defensive repair for any similar section header that may have been emitted
# with a physical newline after print(".
text = text.replace('    print("\n=== Exact-count generation complete ===")', '    print("\\n=== Exact-count generation complete ===")')

if text != original:
    path.write_text(text, encoding='utf-8')
    print(f'Repaired newline string literal in {path}')
else:
    print(f'No broken newline string literal found in {path}; checking compile anyway')

try:
    py_compile.compile(str(path), doraise=True)
except py_compile.PyCompileError as exc:
    lines = path.read_text(encoding='utf-8').splitlines()
    lineno = getattr(exc.exc_value, 'lineno', None)
    if lineno:
        start = max(1, lineno - 5)
        end = min(len(lines), lineno + 5)
        print(f'Compile still failed around line {lineno}:')
        for i in range(start, end + 1):
            print(f'{i:04d}: {lines[i-1]}')
    raise

print('generate/run_generation.py compiles successfully')
