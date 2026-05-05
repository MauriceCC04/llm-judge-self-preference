from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from generate.artifact_normalization import audit_plan_presentation, collect_structural_prompt_leaks, is_generic_day_title

PROVENANCE_SUFFIX = ".json.provenance.json"
JUNK_NAMES = {".DS_Store", "__MACOSX", "__pycache__", ".pytest_cache"}
DEBUG_ARTIFACT_NAMES = {"raw_failures", "failed_plans.jsonl"}


def _json_read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def _is_plan_json(path: Path) -> bool:
    return path.is_file() and path.name.endswith(".json") and not path.name.endswith(PROVENANCE_SUFFIX)


def _plan_paths(plans_dir: Path) -> list[Path]:
    return sorted(path for path in plans_dir.iterdir() if _is_plan_json(path))


def _cell_key(provenance: dict[str, Any]) -> str:
    fixture_id = str(provenance.get("fixture_id") or "")
    if fixture_id:
        return fixture_id
    return "__".join(
        [
            f"ab_{provenance.get('athlete_band') or ''}",
            f"r_{provenance.get('readiness') or ''}",
            f"rc_{provenance.get('recovery_capability') or provenance.get('recovery_capacity') or ''}",
            f"ph_{provenance.get('race_phase') or provenance.get('phase') or ''}",
        ]
    )


def _sidecar_path(plan_path: Path, provenance_dir: Path) -> Path:
    return provenance_dir / f"{plan_path.name}.provenance.json"


def _find_junk(root: Path) -> list[str]:
    out: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.name.startswith("._") or path.name in JUNK_NAMES:
            out.append(str(path))
    return out


def _find_debug_artifacts(plans_dir: Path) -> list[str]:
    return [str(plans_dir / name) for name in sorted(DEBUG_ARTIFACT_NAMES) if (plans_dir / name).exists()]


def audit_corpus(
    *,
    plans_dir: Path,
    provenance_dir: Path | None = None,
    expected_plan_count: int | None = None,
    expected_sidecar_count: int | None = None,
    expected_cell_count: int | None = None,
    expected_per_cell: int | None = None,
    expected_days: int = 7,
    require_manifest: bool = False,
) -> dict[str, Any]:
    plans_dir = Path(plans_dir)
    provenance_dir = Path(provenance_dir or plans_dir)
    plan_paths = _plan_paths(plans_dir)
    sidecar_paths = sorted(provenance_dir.glob(f"*{PROVENANCE_SUFFIX}"))
    plan_stems = {path.stem for path in plan_paths}
    sidecar_plan_stems = {path.name[: -len(PROVENANCE_SUFFIX)] for path in sidecar_paths}

    missing_sidecars = sorted(plan_stems - sidecar_plan_stems)
    orphan_sidecars = sorted(sidecar_plan_stems - plan_stems)

    cell_counts: Counter[str] = Counter()
    by_band: Counter[str] = Counter()
    by_arm: Counter[str] = Counter()
    by_source_model: Counter[str] = Counter()
    day_count_issues: list[dict[str, Any]] = []
    presentation_issues: list[dict[str, Any]] = []
    structural_leaks: list[dict[str, Any]] = []
    generic_titles_by_plan: dict[str, int] = {}
    title_mismatch_by_plan: dict[str, int] = {}
    sidecar_normalization_missing: list[str] = []

    for plan_path in plan_paths:
        prov_path = _sidecar_path(plan_path, provenance_dir)
        provenance = _json_read(prov_path) if prov_path.exists() else {}
        plan_obj = _json_read(plan_path)
        plan_id = str(provenance.get("plan_id") or plan_path.stem)
        cell = _cell_key(provenance)
        cell_counts[cell] += 1
        by_band[str(provenance.get("athlete_band") or "")] += 1
        by_arm[str(provenance.get("arm") or "")] += 1
        by_source_model[str(provenance.get("source_model") or "")] += 1

        days = (plan_obj.get("plan") or {}).get("days") or []
        if len(days) != expected_days:
            day_count_issues.append({"plan_id": plan_id, "n_days": len(days)})

        generic_count = 0
        title_mismatch_count = 0
        for idx, day in enumerate(days):
            if not isinstance(day, dict):
                continue
            if is_generic_day_title(day.get("title")):
                generic_count += 1
        issues = audit_plan_presentation(plan_obj, expected_days=expected_days)
        title_mismatch_count = sum(1 for issue in issues if "title_session_mismatch" in issue)
        if generic_count:
            generic_titles_by_plan[plan_id] = generic_count
        if title_mismatch_count:
            title_mismatch_by_plan[plan_id] = title_mismatch_count
        if issues:
            presentation_issues.append({"plan_id": plan_id, "issues": issues})

        leaks = collect_structural_prompt_leaks(plan_obj)
        if leaks:
            structural_leaks.append(
                {"plan_id": plan_id, "hits": [{"path": p, "marker": m, "value": v} for p, m, v in leaks]}
            )

        runtime_metadata = provenance.get("runtime_metadata") if isinstance(provenance, dict) else None
        norm_meta = runtime_metadata.get("artifact_normalization") if isinstance(runtime_metadata, dict) else None
        if not norm_meta:
            sidecar_normalization_missing.append(plan_id)

    bad_cell_counts = {
        cell: count for cell, count in sorted(cell_counts.items()) if expected_per_cell is not None and count != expected_per_cell
    }
    manifest_path = plans_dir.parent / "manifest_sha256.txt"
    failures: list[str] = []
    if expected_plan_count is not None and len(plan_paths) != expected_plan_count:
        failures.append(f"plan_count:{len(plan_paths)}!={expected_plan_count}")
    if expected_sidecar_count is not None and len(sidecar_paths) != expected_sidecar_count:
        failures.append(f"sidecar_count:{len(sidecar_paths)}!={expected_sidecar_count}")
    if expected_cell_count is not None and len(cell_counts) != expected_cell_count:
        failures.append(f"cell_count:{len(cell_counts)}!={expected_cell_count}")
    if bad_cell_counts:
        failures.append(f"bad_cell_counts:{len(bad_cell_counts)}")
    if missing_sidecars:
        failures.append(f"missing_sidecars:{len(missing_sidecars)}")
    if orphan_sidecars:
        failures.append(f"orphan_sidecars:{len(orphan_sidecars)}")
    if day_count_issues:
        failures.append(f"wrong_day_counts:{len(day_count_issues)}")
    if presentation_issues:
        failures.append(f"presentation_issues:{len(presentation_issues)}")
    if structural_leaks:
        failures.append(f"structural_leaks:{len(structural_leaks)}")
    junk = _find_junk(plans_dir.parent)
    if junk:
        failures.append(f"junk_files:{len(junk)}")
    debug_artifacts = _find_debug_artifacts(plans_dir)
    if debug_artifacts:
        failures.append(f"debug_artifacts_in_plans:{len(debug_artifacts)}")
    if require_manifest and not manifest_path.exists():
        failures.append("missing_manifest")

    return {
        "plans_dir": str(plans_dir),
        "provenance_dir": str(provenance_dir),
        "n_plan_files": len(plan_paths),
        "n_sidecars": len(sidecar_paths),
        "n_cells": len(cell_counts),
        "cell_counts": dict(sorted(cell_counts.items())),
        "bad_cell_counts": bad_cell_counts,
        "by_band": dict(sorted(by_band.items())),
        "by_arm": dict(sorted(by_arm.items())),
        "by_source_model": dict(sorted(by_source_model.items())),
        "missing_sidecars": missing_sidecars,
        "orphan_sidecars": orphan_sidecars,
        "day_count_issues": day_count_issues,
        "presentation_issues": presentation_issues,
        "generic_titles_by_plan": generic_titles_by_plan,
        "title_mismatch_by_plan": title_mismatch_by_plan,
        "structural_leaks": structural_leaks,
        "sidecar_normalization_missing": sidecar_normalization_missing,
        "debug_artifacts_in_plans": debug_artifacts,
        "junk_files": junk,
        "manifest_path": str(manifest_path),
        "manifest_exists": manifest_path.exists(),
        "failures": failures,
        "passed": not failures,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit a normalized plan corpus before scoring/matching/judging.")
    parser.add_argument("--plans", required=True)
    parser.add_argument("--provenance", default=None)
    parser.add_argument("--expected-plan-count", type=int, default=None)
    parser.add_argument("--expected-sidecar-count", type=int, default=None)
    parser.add_argument("--expected-cell-count", type=int, default=32)
    parser.add_argument("--expected-per-cell", type=int, default=None)
    parser.add_argument("--expected-days", type=int, default=7)
    parser.add_argument("--require-manifest", action="store_true")
    parser.add_argument("--output", default=None)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    audit = audit_corpus(
        plans_dir=Path(args.plans),
        provenance_dir=Path(args.provenance) if args.provenance else None,
        expected_plan_count=args.expected_plan_count,
        expected_sidecar_count=args.expected_sidecar_count,
        expected_cell_count=args.expected_cell_count,
        expected_per_cell=args.expected_per_cell,
        expected_days=args.expected_days,
        require_manifest=args.require_manifest,
    )

    print("\n=== Plan corpus audit ===")
    print(f"Plans:       {audit['n_plan_files']}")
    print(f"Sidecars:    {audit['n_sidecars']}")
    print(f"Cells:       {audit['n_cells']}")
    print(f"By band:     {audit['by_band']}")
    print(f"Passed:      {audit['passed']}")
    if audit["failures"]:
        print("Failures:")
        for failure in audit["failures"]:
            print(f"  - {failure}")
    if args.output:
        _json_write(Path(args.output), audit)
        print(f"Audit: {args.output}")
    if args.strict and not audit["passed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
