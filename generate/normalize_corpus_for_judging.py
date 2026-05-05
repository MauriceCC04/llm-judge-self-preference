"""Deterministically normalize retained plan corpora for judge-facing use."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from generate.persistence import verify_json_object_file, verify_plan_and_provenance

try:
    from match.filtering import detect_plan_issues
except Exception:  # pragma: no cover
    detect_plan_issues = None  # type: ignore[assignment]

REST_TITLE = "Rest day"
REST_WORKOUT = "Rest day. No structured training."
REST_PURPOSE = "Absorb training and restore readiness."

SESSION_TEXT: dict[str, tuple[str, str, str]] = {
    "easy": ("Easy run", "{duration} min easy run at relaxed effort.", "Build aerobic consistency with low training stress."),
    "aerobic": ("Aerobic run", "{duration} min steady aerobic run at controlled effort.", "Support aerobic development without excessive fatigue."),
    "long": ("Long run", "{duration} min long run at controlled aerobic effort.", "Build endurance and trail-specific durability."),
    "tempo": ("Tempo run", "{duration} min tempo-focused run at controlled hard effort.", "Develop sustained controlled effort while managing fatigue."),
    "intervals": ("Interval session", "{duration} min interval session with controlled recoveries.", "Develop higher-end aerobic power with structured recovery."),
    "hills": ("Hill session", "{duration} min hill-focused session with controlled climbing efforts.", "Build climbing strength and trail-specific power."),
    "strength": ("Strength session", "{duration} min strength and mobility session.", "Improve durability, resilience, and movement quality."),
    "cross": ("Cross-training", "{duration} min low-impact cross-training session.", "Maintain aerobic stimulus while reducing impact load."),
}

GENERIC_TITLE_RE = re.compile(r"^\s*day\s+\d+\s*$", re.I)
REST_WORDING_RE = re.compile(r"\b(no\s+structured\s+training|no\s+training|rest\s+day|recover\b|recovery\b|restore\b|freshness\b|maintain\s+freshness|absorb\s+training)\b", re.I)
LEAK_PATTERNS = (
    ("signal_placeholder_angle", re.compile(r">\s*\{\s*signal_id", re.I)),
    ("signal_placeholder_brace", re.compile(r"\{\s*signal_id\s*\}", re.I)),
    ("structural_prompt_expectation", re.compile(r"This cell expects", re.I)),
    ("structural_prompt_low_readiness", re.compile(r"Low readiness should reduce", re.I)),
    ("structural_prompt_required_long", re.compile(r"Include at least one day with session_type", re.I)),
)


class NormalizationError(RuntimeError):
    pass


def _duration(day: dict[str, Any], plan_id: str, idx: int) -> int:
    raw = day.get("duration_minutes")
    if isinstance(raw, bool):
        raise NormalizationError(f"{plan_id} day {idx}: duration_minutes must be numeric")
    try:
        value = int(raw)
    except Exception as exc:
        raise NormalizationError(f"{plan_id} day {idx}: invalid duration_minutes={raw!r}") from exc
    if value < 0:
        raise NormalizationError(f"{plan_id} day {idx}: negative duration_minutes={value}")
    return value


def is_generic_day_title(title: Any) -> bool:
    return bool(GENERIC_TITLE_RE.match(str(title or "")))


def has_rest_or_no_training_wording(text: Any) -> bool:
    return bool(REST_WORDING_RE.search(str(text or "")))


def normalize_day(day: dict[str, Any], *, plan_id: str = "<unknown>", day_idx: int = 0) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(day, dict):
        raise NormalizationError(f"{plan_id} day {day_idx}: day must be an object")
    before = copy.deepcopy(day)
    out = copy.deepcopy(day)
    session_before = str(out.get("session_type") or "").strip().lower()
    dur = _duration(out, plan_id, day_idx)
    is_rest_flag = bool(out.get("is_rest_day"))
    zero_active = dur == 0 and session_before != "rest" and not is_rest_flag
    rest_normalized = is_rest_flag or session_before == "rest" or dur == 0
    generic_title = is_generic_day_title(before.get("title"))
    active_rest_words = (not rest_normalized) and (has_rest_or_no_training_wording(before.get("workout")) or has_rest_or_no_training_wording(before.get("purpose")))

    if rest_normalized:
        out.update(
            session_type="rest",
            is_rest_day=True,
            is_hard_day=False,
            duration_minutes=0,
            target_intensity="rest",
            title=REST_TITLE,
            workout=REST_WORKOUT,
            purpose=REST_PURPOSE,
        )
    else:
        if session_before not in SESSION_TEXT:
            raise NormalizationError(f"{plan_id} day {day_idx}: unknown active session_type {session_before!r}")
        title, workout, purpose = SESSION_TEXT[session_before]
        out.update(session_type=session_before, title=title, workout=workout.format(duration=dur), purpose=purpose)

    changed = sorted(k for k in set(before) | set(out) if before.get(k) != out.get(k))
    change = {
        "index": day_idx,
        "date": str(before.get("date") or ""),
        "session_type_before": session_before,
        "session_type_after": str(out.get("session_type") or ""),
        "changed_fields": changed,
        "zero_duration_active_converted": zero_active,
        "generic_title_replaced": generic_title and "title" in changed,
        "active_rest_wording_repaired": active_rest_words and ("workout" in changed or "purpose" in changed),
    }
    return out, change


def normalize_plan(plan_obj: dict[str, Any], *, plan_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    out = copy.deepcopy(plan_obj)
    plan = out.get("plan")
    if not isinstance(plan, dict) or not isinstance(plan.get("days"), list) or not plan["days"]:
        raise NormalizationError(f"{plan_id}: plan.days must be a non-empty list")
    changes: list[dict[str, Any]] = []
    normalized_days = []
    for idx, day in enumerate(plan["days"]):
        normalized, change = normalize_day(day, plan_id=plan_id, day_idx=idx)
        normalized_days.append(normalized)
        if change["changed_fields"]:
            changes.append(change)
    plan["days"] = normalized_days
    totals = plan.get("weekly_totals")
    if isinstance(totals, dict) and "planned_moving_time_hours" in totals:
        totals["planned_moving_time_hours"] = round(sum(int(d.get("duration_minutes") or 0) for d in normalized_days) / 60.0, 2)
    return out, {"plan_id": plan_id, "days_seen": len(normalized_days), "days_changed": len(changes), "day_changes": changes}


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=str(path.parent), delete=False) as handle:
        tmp = Path(handle.name)
        handle.write(text)
        handle.flush()
    tmp.replace(path)


def _write_json(path: Path, obj: Any) -> None:
    _atomic_write(path, json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def plan_json_paths(input_dir: Path) -> list[Path]:
    return sorted(p for p in input_dir.glob("*.json") if p.is_file() and not p.name.endswith(".provenance.json"))


def sidecar_path_for_plan(plan_path: Path) -> Path:
    return plan_path.with_name(f"{plan_path.name}.provenance.json")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(root_dir: Path, manifest_path: Path) -> int:
    files = sorted(p for p in root_dir.glob("*.json*") if p.is_file())
    lines = [f"{_sha256_file(path)}  {path.name}" for path in files]
    _atomic_write(manifest_path, "\n".join(lines) + ("\n" if lines else ""))
    return len(lines)


def _fixture_from_prov(prov: dict[str, Any], plan_id: str) -> str:
    val = str(prov.get("fixture_id") or "").strip()
    return val or ("__".join(plan_id.split("__")[:4]) if "__" in plan_id else "")


def _band_from_prov(prov: dict[str, Any], fixture_id: str) -> str:
    val = str(prov.get("athlete_band") or "").strip()
    if val:
        return val
    match = re.search(r"(?:^|__)ab_(A[1-4])(?:__|$)", fixture_id)
    return match.group(1) if match else ""


def audit_normalized_corpus(
    plans_dir: Path,
    *,
    expected_plans: int | None = None,
    expected_cells: int | None = None,
    expected_plans_per_cell: int | None = None,
    expected_per_band: int | None = None,
) -> dict[str, Any]:
    plan_paths = plan_json_paths(plans_dir)
    sidecars = sorted(plans_dir.glob("*.json.provenance.json"))
    sidecar_by_plan = {p.name[: -len(".provenance.json")]: p for p in sidecars}
    plan_names = {p.name for p in plan_paths}
    missing_sidecars = sorted(name for name in plan_names if name not in sidecar_by_plan)
    orphan_sidecars = sorted(name for name in sidecar_by_plan if name not in plan_names)
    by_cell: Counter[str] = Counter()
    by_band: Counter[str] = Counter()
    leaks: list[dict[str, str]] = []
    bad: list[dict[str, Any]] = []
    generic_counts: list[int] = []
    active_zero = 0
    active_rest_words = 0
    no_active: list[str] = []
    adv_missing_long: list[str] = []
    a4_missing_quality: list[str] = []

    for plan_path in plan_paths:
        plan_id = plan_path.stem
        try:
            plan_obj, prov = verify_plan_and_provenance(plan_path, sidecar_path_for_plan(plan_path), expected_plan_id=plan_id)
        except Exception as exc:
            bad.append({"plan_id": plan_id, "issue": f"unreadable_pair:{exc}"})
            continue
        fixture = _fixture_from_prov(prov, plan_id)
        band = _band_from_prov(prov, fixture)
        if fixture:
            by_cell[fixture] += 1
        if band:
            by_band[band] += 1
        text = json.dumps(plan_obj, ensure_ascii=False)
        for leak_name, pattern in LEAK_PATTERNS:
            if pattern.search(text):
                leaks.append({"plan_id": plan_id, "leak": leak_name})
        days = [d for d in plan_obj.get("plan", {}).get("days", []) if isinstance(d, dict)]
        active_days = []
        has_long = False
        has_quality = False
        generic_count = 0
        for idx, day in enumerate(days):
            st = str(day.get("session_type") or "").strip().lower()
            dur = int(day.get("duration_minutes") or 0)
            is_rest = bool(day.get("is_rest_day")) or st == "rest"
            if is_generic_day_title(day.get("title")):
                generic_count += 1
            if not is_rest:
                active_days.append(day)
                if dur == 0:
                    active_zero += 1
                    bad.append({"plan_id": plan_id, "day_index": idx, "issue": "active_zero_min"})
                if has_rest_or_no_training_wording(day.get("workout")) or has_rest_or_no_training_wording(day.get("purpose")):
                    active_rest_words += 1
                    bad.append({"plan_id": plan_id, "day_index": idx, "issue": "active_rest_wording"})
            if st == "long" and not is_rest and dur > 0:
                has_long = True
            if st in {"tempo", "intervals", "hills"} and not is_rest and dur > 0:
                has_quality = True
        generic_counts.append(generic_count)
        if generic_count:
            bad.append({"plan_id": plan_id, "issue": "generic_day_titles", "count": generic_count})
        if not active_days:
            no_active.append(plan_id)
        if band in {"A3", "A4"} and not has_long:
            adv_missing_long.append(plan_id)
        if band == "A4" and not has_quality:
            a4_missing_quality.append(plan_id)
        if detect_plan_issues is not None:
            for issue in detect_plan_issues(plan_obj):
                bad.append({"plan_id": plan_id, "issue": issue})

    failures: list[str] = []
    if expected_plans is not None and len(plan_paths) != expected_plans:
        failures.append(f"plans {len(plan_paths)} != expected {expected_plans}")
    if len(sidecars) != len(plan_paths):
        failures.append(f"provenance {len(sidecars)} != plans {len(plan_paths)}")
    if missing_sidecars:
        failures.append(f"missing_sidecars {missing_sidecars}")
    if orphan_sidecars:
        failures.append(f"orphan_sidecars {orphan_sidecars}")
    if expected_cells is not None and len(by_cell) != expected_cells:
        failures.append(f"cells {len(by_cell)} != expected {expected_cells}")
    non_expected_cells: dict[str, int] = {}
    if expected_plans_per_cell is not None:
        non_expected_cells = {c: n for c, n in sorted(by_cell.items()) if n != expected_plans_per_cell}
        if non_expected_cells:
            failures.append(f"non_{expected_plans_per_cell}_cells {non_expected_cells}")
    if expected_per_band is not None:
        bad_bands = {b: n for b, n in sorted(by_band.items()) if n != expected_per_band}
        if bad_bands or len(by_band) != 4:
            failures.append(f"bad_band_counts {dict(sorted(by_band.items()))}")
    if leaks:
        failures.append(f"leaks {leaks[:10]}")
    if bad:
        failures.append(f"bad {bad[:20]}")
    if no_active:
        failures.append(f"no_active_days {no_active[:20]}")
    if adv_missing_long:
        failures.append(f"advanced_missing_long {adv_missing_long[:20]}")
    if a4_missing_quality:
        failures.append(f"a4_missing_quality {a4_missing_quality[:20]}")

    return {
        "plans": len(plan_paths),
        "provenance": len(sidecars),
        "cells": len(by_cell),
        "by_band": dict(sorted(by_band.items())),
        "cell_count_min": min(by_cell.values()) if by_cell else 0,
        "cell_count_max": max(by_cell.values()) if by_cell else 0,
        "non_expected_cells": non_expected_cells,
        "missing_sidecars": missing_sidecars,
        "orphan_sidecars": orphan_sidecars,
        "leaks": leaks,
        "bad": bad,
        "generic_titles_mean": sum(generic_counts) / len(generic_counts) if generic_counts else 0,
        "generic_titles_max": max(generic_counts) if generic_counts else 0,
        "active_zero_min": active_zero,
        "active_rest_wording_issues": active_rest_words,
        "no_active_days": no_active,
        "advanced_missing_long": adv_missing_long,
        "a4_missing_quality": a4_missing_quality,
        "failures": failures,
    }


def _copy_sidecar(src: Path, dst: Path, new_plan_path: Path) -> None:
    prov = verify_json_object_file(src, label="provenance sidecar")
    if "plan_path" in prov:
        prov["plan_path"] = str(new_plan_path)
    _write_json(dst, prov)


def normalize_corpus(
    input_dir: Path,
    output_dir: Path,
    *,
    manifest_path: Path,
    report_path: Path,
    report_jsonl_path: Path | None = None,
    overwrite: bool = False,
    expected_plans: int | None = None,
    expected_cells: int | None = None,
    expected_plans_per_cell: int | None = None,
    expected_per_band: int | None = None,
) -> dict[str, Any]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"input directory does not exist or is not a directory: {input_dir}")
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"output directory is non-empty; pass --overwrite to replace it: {output_dir}")
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_paths = plan_json_paths(input_dir)
    if expected_plans is not None and len(raw_paths) != expected_plans:
        raise NormalizationError(f"expected {expected_plans} plan JSONs, found {len(raw_paths)}")
    if report_jsonl_path:
        _atomic_write(Path(report_jsonl_path), "")

    summary = {
        "plans_processed": 0,
        "days_seen": 0,
        "days_normalized": 0,
        "zero_duration_active_converted": 0,
        "generic_titles_replaced": 0,
        "active_rest_wording_repairs": 0,
        "provenance_copied": 0,
        "manifest_entries": 0,
        "failures": [],
    }
    reports: list[dict[str, Any]] = []

    for plan_path in raw_paths:
        plan_id = plan_path.stem
        sidecar = sidecar_path_for_plan(plan_path)
        if not sidecar.exists():
            raise NormalizationError(f"missing provenance sidecar for {plan_path.name}: {sidecar.name}")
        plan_obj, _ = verify_plan_and_provenance(plan_path, sidecar, expected_plan_id=plan_id)
        normalized, plan_report = normalize_plan(plan_obj, plan_id=plan_id)
        out_plan = output_dir / plan_path.name
        out_sidecar = output_dir / sidecar.name
        _write_json(out_plan, normalized)
        _copy_sidecar(sidecar, out_sidecar, out_plan)
        verify_plan_and_provenance(out_plan, out_sidecar, expected_plan_id=plan_id)
        plan_report.update(input_path=str(plan_path), output_path=str(out_plan), provenance_path=str(out_sidecar))
        reports.append(plan_report)
        summary["plans_processed"] += 1
        summary["days_seen"] += int(plan_report["days_seen"])
        summary["days_normalized"] += int(plan_report["days_changed"])
        for change in plan_report["day_changes"]:
            summary["zero_duration_active_converted"] += int(bool(change["zero_duration_active_converted"]))
            summary["generic_titles_replaced"] += int(bool(change["generic_title_replaced"]))
            summary["active_rest_wording_repairs"] += int(bool(change["active_rest_wording_repaired"]))
        summary["provenance_copied"] += 1
        if report_jsonl_path:
            with Path(report_jsonl_path).open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(plan_report, ensure_ascii=False, sort_keys=True) + "\n")

    validation = audit_normalized_corpus(
        output_dir,
        expected_plans=expected_plans,
        expected_cells=expected_cells,
        expected_plans_per_cell=expected_plans_per_cell,
        expected_per_band=expected_per_band,
    )
    summary["failures"].extend(validation.get("failures") or [])
    summary["manifest_entries"] = write_manifest(output_dir, Path(manifest_path))
    if expected_plans is not None and summary["manifest_entries"] != expected_plans * 2:
        summary["failures"].append(f"manifest entry count {summary['manifest_entries']} != expected retained artifact hashes {expected_plans * 2}")
    report = {"summary": summary, "validation": validation, "plans": reports}
    _write_json(Path(report_path), report)
    if summary["failures"]:
        raise NormalizationError("normalized corpus validation failed: " + "; ".join(map(str, summary["failures"])))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a deterministic judge-facing normalized copy of a retained plan corpus.")
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--report-jsonl", default=None, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--expected-plans", type=int, default=None)
    parser.add_argument("--expected-cells", type=int, default=None)
    parser.add_argument("--expected-plans-per-cell", type=int, default=None)
    parser.add_argument("--expected-per-band", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        report = normalize_corpus(
            args.input_dir,
            args.output_dir,
            manifest_path=args.manifest,
            report_path=args.report,
            report_jsonl_path=args.report_jsonl,
            overwrite=args.overwrite,
            expected_plans=args.expected_plans,
            expected_cells=args.expected_cells,
            expected_plans_per_cell=args.expected_plans_per_cell,
            expected_per_band=args.expected_per_band,
        )
    except Exception as exc:
        print(f"[normalization failed] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    s = report["summary"]
    print("=== Normalization complete ===")
    print(f"  plans processed: {s['plans_processed']}")
    print(f"  days normalized: {s['days_normalized']}")
    print(f"  zero-duration active sessions converted: {s['zero_duration_active_converted']}")
    print(f"  generic titles replaced: {s['generic_titles_replaced']}")
    print(f"  active rest-wording repairs: {s['active_rest_wording_repairs']}")
    print(f"  provenance sidecars copied: {s['provenance_copied']}")
    print(f"  manifest entries: {s['manifest_entries']}")


if __name__ == "__main__":
    main()
