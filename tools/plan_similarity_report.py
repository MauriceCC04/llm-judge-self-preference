#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from statistics import mean, median
from typing import Any


def load_plan(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_plan_json(path: str) -> bool:
    base = os.path.basename(path)
    return base.endswith(".json") and not base.endswith(".provenance.json")


def fixture_cell_from_filename(path: str) -> str:
    base = os.path.basename(path).replace(".json", "")
    parts = base.split("__")
    return "__".join(parts[:3]) if len(parts) >= 3 else base


def normalize_text(s: str) -> str:
    return " ".join((s or "").split()).strip().lower()


def canonical_json_text(obj: dict[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def exact_text_fingerprint(obj: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json_text(obj).encode("utf-8")).hexdigest()


def day_signature(day: dict[str, Any]) -> tuple[Any, ...]:
    return (
        day.get("session_type"),
        day.get("duration_minutes"),
        day.get("is_rest_day"),
        day.get("is_hard_day"),
        day.get("target_intensity"),
    )


def session_signature(obj: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(day_signature(day) for day in obj.get("plan", {}).get("days", []))


def compact_structure_string(obj: dict[str, Any]) -> str:
    parts: list[str] = []
    for d in obj.get("plan", {}).get("days", []):
        st = str(d.get("session_type", "?"))
        dur = str(d.get("duration_minutes", "?"))
        rest = "R" if d.get("is_rest_day") else "N"
        hard = "H" if d.get("is_hard_day") else "E"
        ti = str(d.get("target_intensity", "?"))
        parts.append(f"{st}:{dur}:{rest}:{hard}:{ti}")
    return "|".join(parts)


def narrative_text(obj: dict[str, Any]) -> str:
    parts: list[str] = []
    for d in obj.get("plan", {}).get("days", []):
        parts.extend(
            [
                normalize_text(str(d.get("title", ""))),
                normalize_text(str(d.get("workout", ""))),
                normalize_text(str(d.get("purpose", ""))),
            ]
        )
    for item in obj.get("data_notes", []) or []:
        parts.append(normalize_text(str(item)))
    for item in (obj.get("recovery", {}) or {}).get("actions", []) or []:
        parts.append(normalize_text(str(item)))
    for item in obj.get("risks", []) or []:
        parts.append(normalize_text(str(item.get("message", ""))))
    return "\n".join(p for p in parts if p)


def ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return math.nan
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = (len(sorted_vals) - 1) * p
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return sorted_vals[lo]
    w = idx - lo
    return sorted_vals[lo] * (1 - w) + sorted_vals[hi] * w


def format_pct(x: float) -> str:
    return f"{100.0 * x:.2f}%"


def write_csv(path: str, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analyze(plans_dir: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    plan_paths = sorted(p for p in glob.glob(os.path.join(plans_dir, "*.json")) if is_plan_json(p))
    if not plan_paths:
        raise SystemExit(f"No plan JSON files found in {plans_dir}")

    records: list[dict[str, Any]] = []
    by_text: dict[str, list[str]] = defaultdict(list)
    by_sig: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    by_text_within_cell: dict[tuple[str, str], list[str]] = defaultdict(list)
    by_sig_within_cell: dict[tuple[str, tuple[Any, ...]], list[str]] = defaultdict(list)

    struct_strings: dict[str, str] = {}
    narrative_strings: dict[str, str] = {}

    for path in plan_paths:
        obj = load_plan(path)
        cell = fixture_cell_from_filename(path)
        text_fp = exact_text_fingerprint(obj)
        sig = session_signature(obj)
        struct_str = compact_structure_string(obj)
        narr_str = narrative_text(obj)
        file_name = os.path.basename(path)

        records.append(
            {
                "file": file_name,
                "cell": cell,
                "day_count": len(obj.get("plan", {}).get("days", [])),
                "rest_days": sum(1 for d in obj.get("plan", {}).get("days", []) if d.get("is_rest_day")),
                "hard_days": sum(1 for d in obj.get("plan", {}).get("days", []) if d.get("is_hard_day")),
                "total_minutes": sum(int(d.get("duration_minutes", 0) or 0) for d in obj.get("plan", {}).get("days", [])),
                "text_fp": text_fp,
                "sig_hash": hashlib.sha256(repr(sig).encode("utf-8")).hexdigest()[:16],
            }
        )
        by_text[text_fp].append(file_name)
        by_sig[sig].append(file_name)
        by_text_within_cell[(cell, text_fp)].append(file_name)
        by_sig_within_cell[(cell, sig)].append(file_name)
        struct_strings[file_name] = struct_str
        narrative_strings[file_name] = narr_str

    total = len(records)
    exact_text_dup_groups = [sorted(v) for v in by_text.values() if len(v) > 1]
    exact_sig_dup_groups = [sorted(v) for v in by_sig.values() if len(v) > 1]
    cell_text_dup_groups = [
        {"cell": cell, "files": sorted(v), "group_size": len(v)}
        for (cell, _fp), v in by_text_within_cell.items()
        if len(v) > 1
    ]
    cell_sig_dup_groups = [
        {"cell": cell, "files": sorted(v), "group_size": len(v)}
        for (cell, _sig), v in by_sig_within_cell.items()
        if len(v) > 1
    ]

    exact_text_dup_plans = sum(len(g) for g in exact_text_dup_groups)
    exact_text_extra = sum(len(g) - 1 for g in exact_text_dup_groups)
    exact_sig_dup_plans = sum(len(g) for g in exact_sig_dup_groups)
    exact_sig_extra = sum(len(g) - 1 for g in exact_sig_dup_groups)

    # Nearest-neighbor similarity (global and within-cell)
    global_struct_nn: list[dict[str, Any]] = []
    global_narr_nn: list[dict[str, Any]] = []
    cell_struct_nn: list[dict[str, Any]] = []
    cell_narr_nn: list[dict[str, Any]] = []

    file_to_cell = {r["file"]: r["cell"] for r in records}
    files = [r["file"] for r in records]
    cell_to_files: dict[str, list[str]] = defaultdict(list)
    for f in files:
        cell_to_files[file_to_cell[f]].append(f)

    def best_match(source: str, candidates: list[str], text_map: dict[str, str]) -> tuple[str | None, float]:
        best_name = None
        best_score = -1.0
        src = text_map[source]
        for other in candidates:
            if other == source:
                continue
            score = ratio(src, text_map[other])
            if score > best_score:
                best_name = other
                best_score = score
        return best_name, max(best_score, 0.0)

    for f in files:
        other, score = best_match(f, files, struct_strings)
        global_struct_nn.append({"file": f, "nearest_file": other or "", "similarity": round(score, 6)})
        other, score = best_match(f, files, narrative_strings)
        global_narr_nn.append({"file": f, "nearest_file": other or "", "similarity": round(score, 6)})

        same_cell = cell_to_files[file_to_cell[f]]
        other, score = best_match(f, same_cell, struct_strings)
        cell_struct_nn.append({"file": f, "cell": file_to_cell[f], "nearest_file": other or "", "similarity": round(score, 6)})
        other, score = best_match(f, same_cell, narrative_strings)
        cell_narr_nn.append({"file": f, "cell": file_to_cell[f], "nearest_file": other or "", "similarity": round(score, 6)})

    def summarize_scores(rows: list[dict[str, Any]]) -> dict[str, float]:
        vals = sorted(float(r["similarity"]) for r in rows)
        return {
            "mean": round(mean(vals), 6),
            "median": round(median(vals), 6),
            "p25": round(percentile(vals, 0.25), 6),
            "p75": round(percentile(vals, 0.75), 6),
            "max": round(max(vals), 6),
        }

    per_cell_rows: list[dict[str, Any]] = []
    for cell, group_files in sorted(cell_to_files.items()):
        group_records = [r for r in records if r["cell"] == cell]
        text_counts = Counter(r["text_fp"] for r in group_records)
        sig_counts = Counter(r["sig_hash"] for r in group_records)
        exact_text_groups = sum(1 for c in text_counts.values() if c > 1)
        exact_sig_groups = sum(1 for c in sig_counts.values() if c > 1)
        text_dup_plans = sum(c for c in text_counts.values() if c > 1)
        sig_dup_plans = sum(c for c in sig_counts.values() if c > 1)
        per_cell_rows.append(
            {
                "cell": cell,
                "plan_count": len(group_records),
                "exact_text_duplicate_groups": exact_text_groups,
                "exact_text_duplicate_plan_pct": round(text_dup_plans / len(group_records), 6),
                "exact_signature_duplicate_groups": exact_sig_groups,
                "exact_signature_duplicate_plan_pct": round(sig_dup_plans / len(group_records), 6),
                "mean_total_minutes": round(mean(r["total_minutes"] for r in group_records), 2),
            }
        )

    summary = {
        "plans_dir": plans_dir,
        "total_plans": total,
        "exact_text_duplicate_groups": len(exact_text_dup_groups),
        "exact_text_duplicate_plans": exact_text_dup_plans,
        "exact_text_duplicates_pct_of_plans": round(exact_text_dup_plans / total, 6),
        "exact_text_extra_duplicates": exact_text_extra,
        "exact_text_extra_duplicates_pct_of_plans": round(exact_text_extra / total, 6),
        "exact_session_signature_duplicate_groups": len(exact_sig_dup_groups),
        "exact_session_signature_duplicate_plans": exact_sig_dup_plans,
        "exact_session_signature_duplicates_pct_of_plans": round(exact_sig_dup_plans / total, 6),
        "exact_session_signature_extra_duplicates": exact_sig_extra,
        "exact_session_signature_extra_duplicates_pct_of_plans": round(exact_sig_extra / total, 6),
        "global_structure_nn_similarity": summarize_scores(global_struct_nn),
        "within_cell_structure_nn_similarity": summarize_scores(cell_struct_nn),
        "global_narrative_nn_similarity": summarize_scores(global_narr_nn),
        "within_cell_narrative_nn_similarity": summarize_scores(cell_narr_nn),
    }

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(out_dir, "exact_text_duplicate_groups.json"), "w", encoding="utf-8") as f:
        json.dump(exact_text_dup_groups, f, indent=2)
    with open(os.path.join(out_dir, "exact_session_signature_duplicate_groups.json"), "w", encoding="utf-8") as f:
        json.dump(exact_sig_dup_groups, f, indent=2)
    with open(os.path.join(out_dir, "cell_exact_text_duplicate_groups.json"), "w", encoding="utf-8") as f:
        json.dump(sorted(cell_text_dup_groups, key=lambda x: (x["cell"], -x["group_size"], x["files"])), f, indent=2)
    with open(os.path.join(out_dir, "cell_exact_session_signature_duplicate_groups.json"), "w", encoding="utf-8") as f:
        json.dump(sorted(cell_sig_dup_groups, key=lambda x: (x["cell"], -x["group_size"], x["files"])), f, indent=2)

    write_csv(os.path.join(out_dir, "plan_inventory.csv"), records, ["file", "cell", "day_count", "rest_days", "hard_days", "total_minutes", "text_fp", "sig_hash"])
    write_csv(os.path.join(out_dir, "per_cell_summary.csv"), per_cell_rows, ["cell", "plan_count", "exact_text_duplicate_groups", "exact_text_duplicate_plan_pct", "exact_signature_duplicate_groups", "exact_signature_duplicate_plan_pct", "mean_total_minutes"])
    write_csv(os.path.join(out_dir, "global_structure_nearest_neighbor.csv"), global_struct_nn, ["file", "nearest_file", "similarity"])
    write_csv(os.path.join(out_dir, "within_cell_structure_nearest_neighbor.csv"), cell_struct_nn, ["file", "cell", "nearest_file", "similarity"])
    write_csv(os.path.join(out_dir, "global_narrative_nearest_neighbor.csv"), global_narr_nn, ["file", "nearest_file", "similarity"])
    write_csv(os.path.join(out_dir, "within_cell_narrative_nearest_neighbor.csv"), cell_narr_nn, ["file", "cell", "nearest_file", "similarity"])

    print("SUMMARY")
    print(f"Total plans: {summary['total_plans']}")
    print(f"Exact text duplicate groups: {summary['exact_text_duplicate_groups']}")
    print(f"Plans in exact text duplicate groups: {summary['exact_text_duplicate_plans']} ({format_pct(summary['exact_text_duplicates_pct_of_plans'])})")
    print(f"Extra exact text duplicates beyond first copy: {summary['exact_text_extra_duplicates']} ({format_pct(summary['exact_text_extra_duplicates_pct_of_plans'])})")
    print(f"Exact session-signature duplicate groups: {summary['exact_session_signature_duplicate_groups']}")
    print(f"Plans in exact signature duplicate groups: {summary['exact_session_signature_duplicate_plans']} ({format_pct(summary['exact_session_signature_duplicates_pct_of_plans'])})")
    print(f"Extra exact signature duplicates beyond first copy: {summary['exact_session_signature_extra_duplicates']} ({format_pct(summary['exact_session_signature_extra_duplicates_pct_of_plans'])})")
    print()
    print("NEAREST-NEIGHBOR SIMILARITY")
    print("Global structure:", summary["global_structure_nn_similarity"])
    print("Within-cell structure:", summary["within_cell_structure_nn_similarity"])
    print("Global narrative:", summary["global_narrative_nn_similarity"])
    print("Within-cell narrative:", summary["within_cell_narrative_nn_similarity"])
    print()
    print(f"Wrote analysis files to: {out_dir}")



def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze plan similarity and duplicate rates.")
    parser.add_argument("--plans-dir", required=True, help="Directory containing plan JSON files.")
    parser.add_argument("--out-dir", required=True, help="Directory to write summary outputs.")
    args = parser.parse_args()
    analyze(args.plans_dir, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
