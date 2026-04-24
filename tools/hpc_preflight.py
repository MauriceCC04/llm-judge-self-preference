"""tools/hpc_preflight.py — HPC preflight checks.

Imported by run_preflight.sh and by Gate-0 test 20.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any


def estimate_dir_gb(path: Path) -> float:
    """Return disk usage of *path* in GiB; returns 0.0 if path doesn't exist."""
    if not path.exists():
        return 0.0
    total = 0
    try:
        for f in path.rglob("*"):
            if f.is_file():
                try:
                    total += f.stat().st_size
                except OSError:
                    pass
    except PermissionError:
        pass
    return total / (1024 ** 3)


def classify_storage(
    *,
    used_gb: float,
    quota_gb: float,
    soft_ceiling_gb: float,
) -> dict[str, Any]:
    return {
        "used_gb": round(used_gb, 2),
        "quota_gb": quota_gb,
        "soft_ceiling_gb": soft_ceiling_gb,
        "quota_ok": used_gb < quota_gb,
        "ceiling_ok": used_gb < soft_ceiling_gb,
        "available_gb": round(quota_gb - used_gb, 2),
    }


def check_imports() -> dict[str, bool]:
    required = [
        "trailtraining",
        "trailtraining.contracts",
        "trailtraining.llm.coach",
        "trailtraining.llm.soft_eval",
        "trailtraining.llm.eval",
        "trailtraining.llm.guardrails",
        "openai",
        "pydantic",
    ]
    results: dict[str, bool] = {}
    for mod in required:
        try:
            importlib.import_module(mod)
            results[mod] = True
        except ImportError:
            results[mod] = False
    return results


def build_preflight_report(
    *,
    home_quota_gb: float = 50.0,
    used_gb: float | None = None,
    has_gpu: bool = False,
    hf_cache_dir: Path | None = None,
) -> dict[str, Any]:
    import shutil

    if used_gb is None:
        home = Path.home()
        used_gb = estimate_dir_gb(home)

    storage = classify_storage(
        used_gb=used_gb,
        quota_gb=home_quota_gb,
        soft_ceiling_gb=30.0,
    )

    import_results = check_imports()

    cache_dirs: dict[str, bool] = {}
    if hf_cache_dir:
        cache_dirs["hf_cache"] = Path(hf_cache_dir).exists()

    return {
        "quota_ok": storage["quota_ok"],
        "used_gb": storage["used_gb"],
        "available_gb": storage["available_gb"],
        "ceiling_ok": storage["ceiling_ok"],
        "has_gpu": has_gpu,
        "imports_ok": all(import_results.values()),
        "import_details": import_results,
        "cache_dirs_exist": cache_dirs,
        "storage": storage,
    }


def main() -> None:
    report = build_preflight_report()
    import json
    print(json.dumps(report, indent=2, default=str))
    if not report["imports_ok"]:
        print("\n[FAIL] Missing imports — fix before sbatch.", file=sys.stderr)
        sys.exit(1)
    if not report["quota_ok"]:
        print("\n[FAIL] Quota exceeded.", file=sys.stderr)
        sys.exit(1)
    print("\n[OK] Preflight passed.")


if __name__ == "__main__":
    main()
