"""tools/verify_dependency_surface.py — check frozen dependency contract.

This helper validates both pyproject.toml and requirements-hpc.txt because the
HPC path depends on the latter as the canonical install surface.

Important policy choice in this revision:
- `huggingface_hub` is treated as a required *runtime import*
- but it is only a *recommended explicit requirement* in requirements-hpc.txt

That means the check will pass if the package is available transitively (for
example through the HF/vLLM stack), while still warning if it is not listed
explicitly in requirements-hpc.txt.
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
    "pytest_cov",
}

RECOMMENDED_EXPLICIT_HPC_PACKAGES = {
    "huggingface_hub",
}

REQUIRED_RUNTIME_IMPORTS = (
    "trailtraining",
    "openai",
    "pydantic",
    "huggingface_hub",
)

OPTIONAL_RUNTIME_IMPORTS = (
    "vllm",
    "torch",
)


def _extract_trailtraining_pin(text: str) -> str | None:
    marker = "trailtraining @ git+https://github.com/MauriceCC04/trailtraining.git@"
    if marker not in text:
        return None
    start = text.index(marker) + len(marker)
    remainder = text[start:]
    match = re.match(r"([0-9a-fA-F]{40})", remainder)
    return match.group(1) if match else None


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_pyproject_pin(pyproject_path: Path) -> str | None:
    return _extract_trailtraining_pin(_read_text(pyproject_path))


def _read_requirements_pin(requirements_path: Path) -> str | None:
    return _extract_trailtraining_pin(_read_text(requirements_path))


def _parse_requirement_names(requirements_path: Path) -> set[str]:
    packages: set[str] = set()
    for raw_line in _read_text(requirements_path).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if " @ git+" in line:
            packages.add(line.split("@", 1)[0].strip().replace("-", "_"))
            continue
        match = re.match(r"([A-Za-z0-9_.-]+)", line)
        if match:
            packages.add(match.group(1).replace("-", "_"))
    return packages


def _try_import(module_name: str) -> tuple[bool, str | None]:
    try:
        module = importlib.import_module(module_name)
        return True, str(getattr(module, "__file__", "")) or None
    except Exception:
        return False, None


def build_report(
    pyproject_path: Path | None = None,
    requirements_path: Path | None = None,
) -> dict[str, Any]:
    pyproject = pyproject_path or Path("pyproject.toml")
    requirements = requirements_path or Path("requirements-hpc.txt")

    pyproject_pin = _read_pyproject_pin(pyproject)
    requirements_pin = _read_requirements_pin(requirements)
    requirement_names = _parse_requirement_names(requirements)

    missing_required = sorted(REQUIRED_HPC_PACKAGES - requirement_names)
    missing_recommended = sorted(RECOMMENDED_EXPLICIT_HPC_PACKAGES - requirement_names)

    report: dict[str, Any] = {
        "pyproject_pin": pyproject_pin,
        "requirements_pin": requirements_pin,
        "expected_pin": TRAILTRAINING_PIN_SHA,
        "pyproject_pin_matches": pyproject_pin == TRAILTRAINING_PIN_SHA,
        "requirements_pin_matches": requirements_pin == TRAILTRAINING_PIN_SHA,
        "pins_match_each_other": pyproject_pin == requirements_pin,
        "requirements_present": sorted(requirement_names),
        "required_hpc_packages_present": sorted(REQUIRED_HPC_PACKAGES & requirement_names),
        "missing_hpc_packages": missing_required,
        "recommended_explicit_packages_present": sorted(
            RECOMMENDED_EXPLICIT_HPC_PACKAGES & requirement_names
        ),
        "missing_recommended_explicit_packages": missing_recommended,
    }

    for module_name in REQUIRED_RUNTIME_IMPORTS + OPTIONAL_RUNTIME_IMPORTS:
        ok, path = _try_import(module_name)
        report[f"import_{module_name}"] = ok
        if path:
            report[f"path_{module_name}"] = path

    warnings: list[str] = []
    if missing_recommended:
        warnings.append(
            "Recommended explicit requirements missing from requirements-hpc.txt: "
            + ", ".join(missing_recommended)
        )
    if report.get("import_huggingface_hub", False) and "huggingface_hub" in missing_recommended:
        warnings.append(
            "huggingface_hub is importable at runtime but not listed explicitly in "
            "requirements-hpc.txt. The environment is usable, but the frozen dependency "
            "surface would be clearer if it were listed explicitly."
        )

    hard_checks_ok = bool(
        report["pyproject_pin_matches"]
        and report["requirements_pin_matches"]
        and report["pins_match_each_other"]
        and not missing_required
        and all(report.get(f"import_{name}", False) for name in REQUIRED_RUNTIME_IMPORTS)
    )

    report["warnings"] = warnings
    report["dependency_surface_ok"] = hard_checks_ok
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Verify the frozen dependency surface.")
    parser.add_argument("--pyproject", default="pyproject.toml")
    parser.add_argument("--requirements", default="requirements-hpc.txt")
    args = parser.parse_args(argv)

    report = build_report(Path(args.pyproject), Path(args.requirements))
    print(json.dumps(report, indent=2, sort_keys=True))

    if not report.get("dependency_surface_ok", False):
        raise SystemExit(
            "Dependency surface check failed: review pins, required packages, or runtime imports."
        )


if __name__ == "__main__":
    main()