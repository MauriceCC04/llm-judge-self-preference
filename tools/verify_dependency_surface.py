"""tools/verify_dependency_surface.py — check frozen dependency contract.

This revision validates both pyproject.toml and requirements-hpc.txt, because the
HPC runbook depends on the latter as the canonical install surface.
"""
from __future__ import annotations

import argparse
import importlib
import json
import re
from pathlib import Path
from typing import Any

TRAILTRAINING_PIN_SHA = "3e7f1793ca051ba1aae05f1714d594691202ad7e"
REQUIRED_HPC_PACKAGES = {
    "trailtraining",
    "openai",
    "pydantic",
    "numpy",
    "pandas",
    "statsmodels",
    "scipy",
    "matplotlib",
    "pytest",
    "pytest-cov",
    "huggingface_hub",
}



def _extract_trailtraining_pin(text: str) -> str | None:
    marker = "trailtraining @ git+https://github.com/MauriceCC04/trailtraining.git@"
    if marker not in text:
        return None
    start = text.index(marker) + len(marker)
    remainder = text[start:]
    match = re.match(r"([0-9a-fA-F]{40})", remainder)
    return match.group(1) if match else None



def _read_pyproject_pin(pyproject_path: Path) -> str | None:
    return _extract_trailtraining_pin(pyproject_path.read_text(encoding="utf-8"))



def _read_requirements_text(requirements_path: Path) -> str:
    return requirements_path.read_text(encoding="utf-8")



def _read_requirements_pin(requirements_path: Path) -> str | None:
    return _extract_trailtraining_pin(_read_requirements_text(requirements_path))



def _parse_requirement_names(requirements_path: Path) -> set[str]:
    packages: set[str] = set()
    for raw_line in _read_requirements_text(requirements_path).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if " @ git+" in line:
            packages.add(line.split("@", 1)[0].strip())
            continue
        match = re.match(r"([A-Za-z0-9_.-]+)", line)
        if match:
            packages.add(match.group(1).replace("-", "_"))
    return packages



def build_report(
    pyproject_path: Path | None = None,
    requirements_path: Path | None = None,
) -> dict[str, Any]:
    pyproject = pyproject_path or Path("pyproject.toml")
    requirements = requirements_path or Path("requirements-hpc.txt")

    pyproject_pin = _read_pyproject_pin(pyproject)
    requirements_pin = _read_requirements_pin(requirements)
    requirement_names = _parse_requirement_names(requirements)

    report: dict[str, Any] = {
        "pyproject_pin": pyproject_pin,
        "requirements_pin": requirements_pin,
        "expected_pin": TRAILTRAINING_PIN_SHA,
        "pyproject_pin_matches": pyproject_pin == TRAILTRAINING_PIN_SHA,
        "requirements_pin_matches": requirements_pin == TRAILTRAINING_PIN_SHA,
        "pins_match_each_other": pyproject_pin == requirements_pin,
        "requirements_present": sorted(requirement_names),
        "required_hpc_packages_present": sorted(REQUIRED_HPC_PACKAGES.issubset(requirement_names) and REQUIRED_HPC_PACKAGES or set()),
        "missing_hpc_packages": sorted(REQUIRED_HPC_PACKAGES - requirement_names),
    }

    for module_name in (
        "trailtraining",
        "openai",
        "pydantic",
        "huggingface_hub",
    ):
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

    report["dependency_surface_ok"] = bool(
        report["pyproject_pin_matches"]
        and report["requirements_pin_matches"]
        and report["pins_match_each_other"]
        and not report["missing_hpc_packages"]
        and all(report.get(key, False) for key in ("import_trailtraining", "import_openai", "import_pydantic", "import_huggingface_hub"))
    )
    return report



def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Verify the frozen dependency surface.")
    parser.add_argument("--pyproject", default="pyproject.toml")
    parser.add_argument("--requirements", default="requirements-hpc.txt")
    args = parser.parse_args(argv)

    report = build_report(Path(args.pyproject), Path(args.requirements))
    print(json.dumps(report, indent=2, sort_keys=True))

    if not report.get("dependency_surface_ok", False):
        raise SystemExit("Dependency surface check failed: review pins, required packages, or imports.")


if __name__ == "__main__":
    main()
