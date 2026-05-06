"""Structural matching diagnostics for the Qwen/Gemma study pool."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from generate.constants import MATCH_FEATURE_WEIGHTS, STRUCTURAL_MATCH_TOLERANCE, TARGET_PAIRS
from match.features import extract_match_features
from match.pair import greedy_pair
from match.structural_score import STRUCTURAL_SCORE_VERSION, score_structural_plan

_SOURCE_PATTERNS = [
    re.compile(r"\bqwen\b", re.I),
    re.compile(r"\bgemma\b", re.I),
    re.compile(r"\bprogrammatic\b", re.I),
    re.compile(r"\bllm[-_ ]?source\b", re.I),
    re.compile(r"\bsource_model\b", re.I),
]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _plan_paths(plans_dir: Path) -> list[Path]:
    return sorted(p for p in plans_dir.glob("*.json") if not p.name.endswith(".provenance.json"))


def _family(model: Any, arm: str | None = None) -> str:
    s = str(model or "").lower()
    if "qwen" in s:
        return "qwen"
    if "gemma" in s:
        return "gemma"
    if arm == "programmatic" or not s or s == "none":
        return "programmatic"
    return "other"


def _quantiles(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    x = sorted(values)
    return {
        "n": len(x),
        "min": round(x[0], 3),
        "p25": round(x[int(0.25 * (len(x) - 1))], 3),
        "median": round(x[int(0.50 * (len(x) - 1))], 3),
        "p75": round(x[int(0.75 * (len(x) - 1))], 3),
        "max": round(x[-1], 3),
    }


def _text_lengths(plan_obj: dict[str, Any]) -> dict[str, float]:
    days = list(((plan_obj.get("plan") or {}).get("days") or []))
    titles = [str(day.get("title") or "") for day in days if isinstance(day, dict)]
    workouts = [str(day.get("workout") or "") for day in days if isinstance(day, dict)]
    purposes = [str(day.get("purpose") or "") for day in days if isinstance(day, dict)]
    return {
        "title_chars": float(sum(len(x) for x in titles)),
        "workout_chars": float(sum(len(x) for x in workouts)),
        "purpose_chars": float(sum(len(x) for x in purposes)),
        "n_citations": float(len(plan_obj.get("citations") or [])),
        "n_claim_attributions": float(len(plan_obj.get("claim_attributions") or [])),
        "n_data_notes": float(len(plan_obj.get("data_notes") or [])),
    }


def _source_leaks(plan_obj: dict[str, Any]) -> list[str]:
    text = json.dumps(plan_obj, ensure_ascii=False)
    leaks: list[str] = []
    for pattern in _SOURCE_PATTERNS:
        if pattern.search(text):
            leaks.append(pattern.pattern)
    return leaks


def _session_signature(plan_obj: dict[str, Any]) -> str:
    days = list(((plan_obj.get("plan") or {}).get("days") or []))
    parts = []
    for day in days:
        if not isinstance(day, dict):
            continue
        parts.append(
            f"{day.get('session_type')}:{int(float(day.get('duration_minutes') or 0))}:"
            f"{int(bool(day.get('is_rest_day')))}:{int(bool(day.get('is_hard_day')))}"
        )
    return "|".join(parts)


def load_records(plans_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for plan_path in _plan_paths(plans_dir):
        prov_path = plan_path.with_name(plan_path.name + ".provenance.json")
        if not prov_path.exists():
            continue
        plan_obj = _load_json(plan_path)
        prov = _load_json(prov_path)
        report = score_structural_plan(plan_path, provenance_path=prov_path)
        rec = {
            "plan_id": prov.get("plan_id") or plan_path.stem,
            "fixture_id": prov.get("fixture_id"),
            "athlete_band": prov.get("athlete_band"),
            "readiness": prov.get("readiness"),
            "recovery_capability": prov.get("recovery_capability"),
            "race_phase": prov.get("race_phase"),
            "plan_days": prov.get("plan_days") or len((plan_obj.get("plan") or {}).get("days") or []),
            "style": prov.get("style") or (plan_obj.get("meta") or {}).get("style"),
            "arm": prov.get("arm"),
            "source_model": prov.get("source_model"),
            "source_family": _family(prov.get("source_model"), prov.get("arm")),
            "score": report.score,
            "structural_score": report.score,
            "structural_score_version": report.score_version,
            "violations": report.violations,
            "plan_path": str(plan_path),
            "session_signature": _session_signature(plan_obj),
            **extract_match_features(plan_obj),
            **{f"style_{k}": v for k, v in _text_lengths(plan_obj).items()},
        }
        rec["source_leaks"] = _source_leaks(plan_obj)
        records.append(rec)
    return records


def _distribution_by(records: list[dict[str, Any]], key: str, value: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for rec in records:
        grouped[str(rec.get(key) or "")].append(float(rec.get(value) or 0.0))
    return {k: _quantiles(v) for k, v in sorted(grouped.items())}


def _possible_counts(records: list[dict[str, Any]], tolerances: list[float]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for tol in tolerances:
        pairs = greedy_pair(records, tolerance=float(tol), feature_weights=MATCH_FEATURE_WEIGHTS)
        out[str(tol)] = {
            "n_pairs": len(pairs),
            "by_source_family": dict(Counter(_family(p.get("source_model_a"), "llm") for p in pairs)),
            "by_fixture": dict(sorted(Counter(p.get("fixture_id") for p in pairs).items())),
            "mean_gap": round(sum(float(p.get("structural_score_gap") or p.get("score_gap") or 0) for p in pairs) / len(pairs), 4) if pairs else None,
            "max_gap": max((float(p.get("structural_score_gap") or p.get("score_gap") or 0) for p in pairs), default=None),
        }
    return out


def _balance_for_pairs(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ["total_minutes", "n_hard_days", "n_long_runs", "n_rest_days", "n_quality_days", "n_active_days", "max_day_minutes"]
    rows = []
    for pair in pairs:
        row = {"pair_id": pair["pair_id"], "fixture_id": pair.get("fixture_id")}
        for key in keys:
            row[f"gap_{key}"] = pair.get("feature_gaps", {}).get(key)
        rows.append(row)
    summary = {}
    for key in keys:
        vals = [float(r[f"gap_{key}"]) for r in rows if r.get(f"gap_{key}") is not None]
        summary[key] = _quantiles(vals)
    return {"summary": summary, "rows": rows[:1000]}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({k for row in rows for k in row})
    lines = [",".join(keys)]
    for row in rows:
        vals = []
        for key in keys:
            val = row.get(key, "")
            text = json.dumps(val, ensure_ascii=False) if isinstance(val, (list, dict)) else str(val)
            vals.append('"' + text.replace('"', '""') + '"')
        lines.append(",".join(vals))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_matching_diagnostics(
    plans_dir: Path,
    output_dir: Path,
    *,
    tolerances: list[float] | None = None,
    target_pairs: int = TARGET_PAIRS,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tolerances = tolerances or [0.5, 1.0, STRUCTURAL_MATCH_TOLERANCE, 3.0]
    records = load_records(plans_dir)
    pairs = greedy_pair(records, tolerance=STRUCTURAL_MATCH_TOLERANCE, feature_weights=MATCH_FEATURE_WEIGHTS)

    duplicate_signatures = {
        sig: ids for sig, ids in _group_values(records, "session_signature", "plan_id").items() if len(ids) > 1
    }
    leaks = {rec["plan_id"]: rec["source_leaks"] for rec in records if rec.get("source_leaks")}

    payload = {
        "summary": {
            "n_records": len(records),
            "n_pairs_at_primary_tolerance": len(pairs),
            "target_pairs": target_pairs,
            "full_study_gate_pairs_ok": len(pairs) >= 250,
            "structural_score_version": STRUCTURAL_SCORE_VERSION,
        },
        "structural_score_distribution_by_arm": _distribution_by(records, "arm", "structural_score"),
        "structural_score_distribution_by_source_family": _distribution_by(records, "source_family", "structural_score"),
        "per_fixture_score_overlap": _per_fixture_overlap(records),
        "possible_match_counts_by_tolerance": _possible_counts(records, tolerances),
        "final_pair_balance": _balance_for_pairs(pairs),
        "style_distribution_by_arm": {k: _distribution_by(records, "arm", f"style_{k}") for k in ["title_chars", "workout_chars", "purpose_chars", "n_citations", "n_claim_attributions", "n_data_notes"]},
        "source_leak_audit": {"n_plans_with_leaks": len(leaks), "examples": dict(list(leaks.items())[:25])},
        "duplicate_session_signature_groups": {"n_groups": len(duplicate_signatures), "examples": dict(list(duplicate_signatures.items())[:25])},
    }
    (output_dir / "matching_diagnostics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(output_dir / "structural_scores.csv", records)
    _write_csv(output_dir / "primary_tolerance_pairs.csv", pairs)
    (output_dir / "matched_pairs_structural_preview.json").write_text(
        json.dumps(pairs, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return payload


def _group_values(records: list[dict[str, Any]], key: str, value: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    for rec in records:
        out[str(rec.get(key) or "")].append(str(rec.get(value) or ""))
    return out


def _per_fixture_overlap(records: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    by_fixture: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_fixture[str(rec.get("fixture_id") or "")].append(rec)
    for fid, rows in sorted(by_fixture.items()):
        llm = [float(r["structural_score"]) for r in rows if r.get("arm") == "llm"]
        prog = [float(r["structural_score"]) for r in rows if r.get("arm") == "programmatic"]
        if not llm or not prog:
            out[fid] = {"has_overlap": False, "n_llm": len(llm), "n_programmatic": len(prog)}
            continue
        lo = max(min(llm), min(prog))
        hi = min(max(llm), max(prog))
        out[fid] = {
            "has_overlap": lo <= hi,
            "overlap_low": round(lo, 3),
            "overlap_high": round(hi, 3),
            "n_llm": len(llm),
            "n_programmatic": len(prog),
        }
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plans", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tolerance", type=float, action="append", default=None)
    parser.add_argument("--target-pairs", type=int, default=TARGET_PAIRS)
    args = parser.parse_args(argv)
    result = run_matching_diagnostics(Path(args.plans), Path(args.output), tolerances=args.tolerance, target_pairs=args.target_pairs)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
