"""tools/hpc_preflight.py — HPC preflight checks.

Imported by run_preflight.sh and by Gate-0 tests.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

try:
    from hpc.quota import SOFT_QUOTA_GB_DEFAULT, TOTAL_QUOTA_GB, hf_cache_root
except Exception:
    TOTAL_QUOTA_GB = 50.0
    SOFT_QUOTA_GB_DEFAULT = 30.0

    def hf_cache_root() -> Path:
        return Path.home() / ".cache" / "huggingface" / "hub"


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


def _resolve_hf_cache_dir(hf_cache_dir: Path | None = None) -> Path:
    if hf_cache_dir is not None:
        return Path(hf_cache_dir)
    return hf_cache_root()


def build_preflight_report(
    *,
    home_quota_gb: float = TOTAL_QUOTA_GB,
    used_gb: float | None = None,
    has_gpu: bool = False,
    hf_cache_dir: Path | None = None,
    soft_ceiling_gb: float = SOFT_QUOTA_GB_DEFAULT,
) -> dict[str, Any]:
    if used_gb is None:
        home = Path.home()
        used_gb = estimate_dir_gb(home)

    storage = classify_storage(
        used_gb=used_gb,
        quota_gb=home_quota_gb,
        soft_ceiling_gb=soft_ceiling_gb,
    )

    import_results = check_imports()
    cache_dir = _resolve_hf_cache_dir(hf_cache_dir)

    project_root_env = (os.getenv("PROJECT_ROOT") or "").strip()
    cwd = Path.cwd()
    cache_dirs = {
        "hf_cache": cache_dir.exists(),
    }

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
        "project_root_env": project_root_env or None,
        "cwd": str(cwd),
        "soft_ceiling_gb": soft_ceiling_gb,
        "hf_cache_dir": str(cache_dir),
    }


def main() -> None:
    report = build_preflight_report()
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
