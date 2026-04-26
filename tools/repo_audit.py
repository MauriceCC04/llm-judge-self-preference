"""tools/repo_audit.py — code-first readiness and alignment audit."""
from __future__ import annotations

import json
from pathlib import Path

from generate.study_manifest import frozen_study_summary


def build_audit() -> dict[str, object]:
    summary = frozen_study_summary()
    pyproject_text = Path("pyproject.toml").read_text(encoding="utf-8") if Path("pyproject.toml").exists() else ""
    readme_text = Path("README.md").read_text(encoding="utf-8") if Path("README.md").exists() else ""
    runbook_text = Path("HPC_SYNC_RUNBOOK.md").read_text(encoding="utf-8") if Path("HPC_SYNC_RUNBOOK.md").exists() else ""
    prereg_text = Path("PREREGISTRATION.md").read_text(encoding="utf-8") if Path("PREREGISTRATION.md").exists() else ""

    checks = {
        "pyproject_pin_present": summary["trailtraining_pin_sha"] in pyproject_text,
        "readme_512_synced": str(summary["total_plans"]) in readme_text,
        "runbook_512_synced": str(summary["total_plans"]) in runbook_text,
        "prereg_512_synced": str(summary["total_plans"]) in prereg_text,
        "canonical_pairwise_view_available": Path("judge/canonicalize.py").exists(),
        "cache_check_tool_available": Path("tools/check_model_cache.py").exists(),
        "dependency_verify_tool_available": Path("tools/verify_dependency_surface.py").exists(),
    }
    hpc_score = sum(bool(v) for k, v in checks.items() if k in {
        "pyproject_pin_present",
        "runbook_512_synced",
        "cache_check_tool_available",
        "dependency_verify_tool_available",
    })
    goal_score = sum(bool(v) for k, v in checks.items() if k in {
        "readme_512_synced",
        "prereg_512_synced",
        "canonical_pairwise_view_available",
    })
    return {
        "frozen_study_summary": summary,
        "checks": checks,
        "hpc_readiness_score": hpc_score,
        "goal_alignment_score": goal_score,
    }


def main() -> None:
    print(json.dumps(build_audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
