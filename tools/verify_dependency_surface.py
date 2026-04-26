"""tools/verify_dependency_surface.py — check frozen dependency contract."""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path

TRAILTRAINING_PIN_SHA = "3e7f1793ca051ba1aae05f1714d594691202ad7e"


def _read_pyproject_pin(pyproject_path: Path) -> str:
    text = pyproject_path.read_text(encoding="utf-8")
    marker = "trailtraining @ git+https://github.com/MauriceCC04/trailtraining.git@"
    start = text.index(marker) + len(marker)
    end = text.index('"', start)
    return text[start:end]


def build_report(pyproject_path: Path | None = None) -> dict[str, object]:
    pyproject = pyproject_path or Path("pyproject.toml")
    pin = _read_pyproject_pin(pyproject)
    report: dict[str, object] = {
        "pyproject_pin": pin,
        "expected_pin": TRAILTRAINING_PIN_SHA,
        "pin_matches": pin == TRAILTRAINING_PIN_SHA,
    }
    for module_name in ("trailtraining", "openai", "pydantic"):
        try:
            module = importlib.import_module(module_name)
            report[f"import_{module_name}"] = True
            report[f"path_{module_name}"] = str(getattr(module, "__file__", ""))
        except Exception:
            report[f"import_{module_name}"] = False
    try:
        importlib.import_module("vllm")
        report["import_vllm"] = True
    except Exception:
        report["import_vllm"] = False
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Verify the frozen dependency surface.")
    parser.add_argument("--pyproject", default="pyproject.toml")
    args = parser.parse_args(argv)
    report = build_report(Path(args.pyproject))
    print(report)
    if not report.get("pin_matches", False):
        raise SystemExit("pyproject trailtraining pin does not match the frozen study manifest.")
    for key in ("import_trailtraining", "import_openai", "import_pydantic"):
        if not report.get(key, False):
            raise SystemExit(f"Required import failed: {key}")


if __name__ == "__main__":
    main()
