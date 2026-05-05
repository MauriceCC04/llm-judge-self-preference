from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from generate.artifact_normalization import (
    NORMALIZER_VERSION,
    artifact_sha256,
    attach_normalization_metadata_to_provenance,
    audit_plan_presentation,
    build_normalization_provenance_payload,
    normalize_plan_artifact,
)

PLAN_SUFFIX = ".json"
PROVENANCE_SUFFIX = ".json.provenance.json"
DEBUG_ARTIFACT_NAMES = {"raw_failures", "failed_plans.jsonl"}
JUNK_NAMES = {".DS_Store", "__MACOSX", "__pycache__", ".pytest_cache"}


def _json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def _json_read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_plan_json(path: Path) -> bool:
    return path.is_file() and path.name.endswith(PLAN_SUFFIX) and not path.name.endswith(PROVENANCE_SUFFIX)


def _plan_paths(plans_dir: Path) -> list[Path]:
    return sorted(path for path in plans_dir.iterdir() if _is_plan_json(path))


def _provenance_path_for_plan(plan_path: Path, provenance_dir: Path) -> Path:
    return provenance_dir / f"{plan_path.name}.provenance.json"


def _relative_manifest_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def write_manifest(*, plans_dir: Path, provenance_dir: Path, manifest_path: Path) -> dict[str, Any]:
    root = manifest_path.parent.resolve()
    files: list[Path] = []
    for path in sorted(plans_dir.iterdir()):
        if _is_plan_json(path) or path.name.endswith(PROVENANCE_SUFFIX):
            files.append(path)
    if provenance_dir != plans_dir:
        for path in sorted(provenance_dir.glob(f"*{PROVENANCE_SUFFIX}")):
            files.append(path)

    lines: list[str] = []
    for path in sorted(set(files)):
        digest = path.read_bytes()
        lines.append(f"{__import__('hashlib').sha256(digest).hexdigest()}  {_relative_manifest_path(path.resolve(), root)}")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return {"path": str(manifest_path), "n_files": len(lines)}


def _archive_debug_artifacts(plans_dir: Path, archive_dir: Path, *, dry_run: bool) -> list[dict[str, str]]:
    moved: list[dict[str, str]] = []
    for name in sorted(DEBUG_ARTIFACT_NAMES):
        src = plans_dir / name
        if not src.exists():
            continue
        dest = archive_dir / name
        moved.append({"from": str(src), "to": str(dest)})
        if dry_run:
            continue
        archive_dir.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            suffix = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            dest = archive_dir / f"{src.name}.{suffix}"
        shutil.move(str(src), str(dest))
    return moved


def _remove_junk(root: Path, *, dry_run: bool) -> list[str]:
    removed: list[str] = []
    for path in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if path.name.startswith("._") or path.name in JUNK_NAMES:
            removed.append(str(path))
            if dry_run:
                continue
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
    return removed


def _quarantine_pair(plan_path: Path, provenance_path: Path, quarantine_dir: Path, *, dry_run: bool) -> dict[str, str]:
    out = {"plan": str(plan_path), "provenance": str(provenance_path), "quarantine_dir": str(quarantine_dir)}
    if dry_run:
        return out
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(plan_path), str(quarantine_dir / plan_path.name))
    if provenance_path.exists():
        shutil.move(str(provenance_path), str(quarantine_dir / provenance_path.name))
    return out


def normalize_corpus(
    *,
    plans_dir: Path,
    provenance_dir: Path | None = None,
    expected_days: int = 7,
    dry_run: bool = False,
    quarantine_dir: Path | None = None,
    archive_debug_dir: Path | None = None,
    cleanup: bool = True,
    manifest_path: Path | None = None,
    fail_on_errors: bool = False,
) -> dict[str, Any]:
    plans_dir = Path(plans_dir)
    provenance_dir = Path(provenance_dir or plans_dir)
    if not plans_dir.exists():
        raise FileNotFoundError(f"Missing plans directory: {plans_dir}")
    if not provenance_dir.exists():
        raise FileNotFoundError(f"Missing provenance directory: {provenance_dir}")

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_debug_dir = archive_debug_dir or plans_dir.parent / "debug_artifacts" / timestamp
    quarantine_dir = quarantine_dir or plans_dir.parent / "quarantine" / f"artifact_normalization_{timestamp}"
    manifest_path = manifest_path or plans_dir.parent / "manifest_sha256.txt"

    moved_debug: list[dict[str, str]] = []
    removed_junk: list[str] = []
    if cleanup:
        moved_debug = _archive_debug_artifacts(plans_dir, archive_debug_dir, dry_run=dry_run)
        removed_junk = _remove_junk(plans_dir.parent, dry_run=dry_run)

    plan_paths = _plan_paths(plans_dir)
    changed_plans: list[dict[str, Any]] = []
    error_plans: list[dict[str, Any]] = []
    missing_sidecars: list[str] = []
    presentation_issues_after: dict[str, list[str]] = {}
    cell_counts: Counter[str] = Counter()
    counts_by_band: Counter[str] = Counter()
    changed_field_reasons: Counter[str] = Counter()
    quarantined: list[dict[str, str]] = []

    for plan_path in plan_paths:
        provenance_path = _provenance_path_for_plan(plan_path, provenance_dir)
        if not provenance_path.exists():
            missing_sidecars.append(str(plan_path))
            continue

        plan_obj = _json_read(plan_path)
        provenance = _json_read(provenance_path)
        fixture_id = str(provenance.get("fixture_id") or "")
        athlete_band = str(provenance.get("athlete_band") or "")
        cell_counts[fixture_id] += 1
        counts_by_band[athlete_band] += 1

        original_sha = artifact_sha256(plan_obj)
        result = normalize_plan_artifact(plan_obj, expected_days=expected_days)
        after_issues = audit_plan_presentation(result.plan, expected_days=expected_days)
        if after_issues:
            presentation_issues_after[plan_path.name] = after_issues

        for item in result.changed_fields:
            changed_field_reasons[item.reason] += 1

        if result.errors or not result.structural_projection_unchanged:
            error_payload = {
                "plan_id": plan_path.stem,
                "plan_path": str(plan_path),
                "provenance_path": str(provenance_path),
                "errors": result.errors,
                "warnings": result.warnings,
                "changed_fields": result.changed_field_payload(),
            }
            error_plans.append(error_payload)
            quarantined.append(_quarantine_pair(plan_path, provenance_path, quarantine_dir, dry_run=dry_run))
            continue

        if result.changed:
            normalized_sha = artifact_sha256(result.plan)
            payload = build_normalization_provenance_payload(
                original_sha256=original_sha,
                normalized_sha256=normalized_sha,
                result=result,
            )
            normalized_provenance = attach_normalization_metadata_to_provenance(provenance, payload)
            changed_plans.append(
                {
                    "plan_id": plan_path.stem,
                    "plan_path": str(plan_path),
                    "provenance_path": str(provenance_path),
                    "n_changed_fields": len(result.changed_fields),
                    "changed_fields": result.changed_field_payload(),
                    "warnings": result.warnings,
                    "original_artifact_sha256": original_sha,
                    "normalized_artifact_sha256": normalized_sha,
                }
            )
            if not dry_run:
                _json_write(plan_path, result.plan)
                _json_write(provenance_path, normalized_provenance)

    manifest = None
    if not dry_run:
        manifest = write_manifest(plans_dir=plans_dir, provenance_dir=provenance_dir, manifest_path=manifest_path)

    audit = {
        "normalizer_version": NORMALIZER_VERSION,
        "plans_dir": str(plans_dir),
        "provenance_dir": str(provenance_dir),
        "dry_run": dry_run,
        "n_plan_files_seen": len(plan_paths),
        "n_changed_plans": len(changed_plans),
        "n_error_plans": len(error_plans),
        "n_missing_sidecars": len(missing_sidecars),
        "cell_counts": dict(sorted(cell_counts.items())),
        "counts_by_band": dict(sorted(counts_by_band.items())),
        "changed_field_reasons": dict(sorted(changed_field_reasons.items())),
        "changed_plans": changed_plans,
        "error_plans": error_plans,
        "missing_sidecars": missing_sidecars,
        "presentation_issues_after": presentation_issues_after,
        "moved_debug_artifacts": moved_debug,
        "removed_junk": removed_junk,
        "quarantined": quarantined,
        "manifest": manifest,
    }

    if fail_on_errors and (error_plans or missing_sidecars or presentation_issues_after):
        raise SystemExit(
            "Normalization found artifact errors: "
            f"errors={len(error_plans)} missing_sidecars={len(missing_sidecars)} "
            f"post_issues={len(presentation_issues_after)}"
        )
    return audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Normalize judge-visible day presentation in plan artifacts.")
    parser.add_argument("--plans", required=True, help="Directory containing plan JSON files.")
    parser.add_argument("--provenance", default=None, help="Directory containing provenance sidecars; defaults to --plans.")
    parser.add_argument("--expected-days", type=int, default=7)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-cleanup", action="store_true", help="Do not move debug artifacts or remove junk files.")
    parser.add_argument("--quarantine-dir", default=None)
    parser.add_argument("--archive-debug-dir", default=None)
    parser.add_argument("--manifest", default=None, help="Manifest output path; defaults to <plans parent>/manifest_sha256.txt.")
    parser.add_argument("--audit-output", default=None, help="Optional JSON audit output path.")
    parser.add_argument("--fail-on-errors", action="store_true")
    args = parser.parse_args(argv)

    audit = normalize_corpus(
        plans_dir=Path(args.plans),
        provenance_dir=Path(args.provenance) if args.provenance else None,
        expected_days=args.expected_days,
        dry_run=args.dry_run,
        quarantine_dir=Path(args.quarantine_dir) if args.quarantine_dir else None,
        archive_debug_dir=Path(args.archive_debug_dir) if args.archive_debug_dir else None,
        cleanup=not args.no_cleanup,
        manifest_path=Path(args.manifest) if args.manifest else None,
        fail_on_errors=args.fail_on_errors,
    )

    print("\n=== Artifact normalization ===")
    print(f"Plans seen:       {audit['n_plan_files_seen']}")
    print(f"Changed plans:    {audit['n_changed_plans']}")
    print(f"Error plans:      {audit['n_error_plans']}")
    print(f"Missing sidecars: {audit['n_missing_sidecars']}")
    print(f"Post-issues:      {len(audit['presentation_issues_after'])}")
    if audit.get("moved_debug_artifacts"):
        print(f"Moved debug artifacts: {len(audit['moved_debug_artifacts'])}")
    if audit.get("removed_junk"):
        print(f"Removed junk files/dirs: {len(audit['removed_junk'])}")
    if audit.get("manifest"):
        print(f"Manifest: {audit['manifest']['path']} ({audit['manifest']['n_files']} files)")

    if args.audit_output:
        _json_write(Path(args.audit_output), audit)
        print(f"Audit: {args.audit_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
