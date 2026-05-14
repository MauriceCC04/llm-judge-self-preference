#!/usr/bin/env python3
"""Build leakage-filtered sensitivity datasets and summaries.

This script does NOT claim that existing judgments were made on cleaned prompts.
It identifies source-asymmetric presentation artifacts in frozen judge-facing inputs,
flags/excludes affected matched pairs, and writes analysis-ready sensitivity datasets
from the already-completed primary and marker judgment outputs.

Typical use from repo root:

  python tools/build_leakage_filtered_results.py \
    --frozen-root artifacts/gen_src_t070_exp_t000/frozen_primary_v1 \
    --output-dir results/primary_t000_scrubbed_v1/leakage_filtered

Outputs include:
  - leakage_flags_by_pair.csv/json
  - leakage_flags_by_record.csv
  - included_clean_pair_ids.txt / excluded_flagged_pair_ids.txt
  - primary_clean_pairwise.jsonl / primary_flagged_pairwise.jsonl
  - marker_clean_pairwise.jsonl / marker_flagged_pairwise.jsonl
  - summary CSVs for primary, marker, and AB/BA pair-run consistency
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

PATTERNS: Dict[str, re.Pattern[str]] = {
    "trailrun_pipe": re.compile(r"Trail\s*Run\s*\||TrailRun\s*\|", re.I),
    "avg_hr": re.compile(r"\bavg\s*HR\b|\bavgHR\b", re.I),
    "km_unit": re.compile(r"\b\d+(?:\.\d+)?\s*km\b|\bkilomet(?:er|re)s?\b", re.I),
    "elevation_or_vertical": re.compile(r"\b\d+(?:\.\d+)?\s*m\+\b|\bm\+\b|\belev(?:ation)?\b|\bvertical\b|\bvert\b", re.I),
    "ellipsis": re.compile(r"…|\.\.\."),
    "activity_like_fragment": re.compile(
        r"Trail\s*Run\s*\||TrailRun\s*\||\bavg\s*HR\b|\bavgHR\b|"
        r"\b\d+(?:\.\d+)?\s*km\b|\bkilomet(?:er|re)s?\b|\b\d+(?:\.\d+)?\s*m\+\b",
        re.I,
    ),
}

MINUTES_RE = re.compile(r"\b(\d{1,3})\s*min(?:ute)?s?\b", re.I)
MARKER_MANIFEST = Path("eval_manifest/pairwise_eval_manifest.jsonl")
JUDGE_INPUT_DIR = Path("eval_manifest/judge_inputs")
PRIMARY_COMBINED = Path("judgments_eval_t000_scrubbed_v1_staged/pairwise_all_judges_canonical_masked_scrubbed_v1_t000.jsonl")
MARKER_COMBINED = Path("marker_eval_t000_scrubbed_v1_staged/marker_all_judges_canonical_masked_scrubbed_v1_t000.jsonl")
PRIMARY_GLOB = "judgments_eval_t000_scrubbed_v1_staged/pairwise_*_judge_canonical_masked_scrubbed_v1_t000.jsonl"
MARKER_GLOB = "marker_eval_t000_scrubbed_v1_staged/marker_*_judge_canonical_masked_scrubbed_v1_t000.jsonl"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except Exception as e:
                    raise ValueError(f"Bad JSONL at {path}:{line_no}: {e}") from e
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = sorted({k for row in materialized for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(materialized)


def find_judge_input(frozen_root: Path, judge_input_path: str) -> Path:
    p = Path(judge_input_path)
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    candidates.append(frozen_root / p)
    candidates.append(frozen_root / JUDGE_INPUT_DIR / p.name)
    marker = "frozen_primary_v1/"
    if marker in judge_input_path:
        rel = judge_input_path.split(marker, 1)[1]
        candidates.append(frozen_root / rel)
    marker2 = "artifacts/gen_src_t070_exp_t000/frozen_primary_v1/"
    if marker2 in judge_input_path:
        rel = judge_input_path.split(marker2, 1)[1]
        candidates.append(frozen_root / rel)
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Could not resolve judge_input_path={judge_input_path!r}; tried {candidates}")


def traverse_strings(obj: Any, path: str = "") -> Iterable[Tuple[str, str]]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            child = f"{path}.{k}" if path else str(k)
            yield from traverse_strings(v, child)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from traverse_strings(v, f"{path}[{i}]")
    elif isinstance(obj, str):
        yield path, obj


def plan_side_flags(side_obj: Dict[str, Any]) -> Tuple[Dict[str, int], List[Dict[str, str]]]:
    """Return leakage flags and compact examples for a single judge-facing plan side."""
    flags: Dict[str, int] = {}
    examples: List[Dict[str, str]] = []
    all_text = json.dumps(side_obj, ensure_ascii=False)
    for name, pat in PATTERNS.items():
        m = pat.search(all_text)
        if m:
            flags[name] = 1
            examples.append({
                "flag": name,
                "context": all_text[max(0, m.start() - 80): min(len(all_text), m.end() + 120)],
            })

    # Flag duration text inconsistencies at day level. This catches activity-log fragments where
    # structured duration_minutes says one thing but summary text embeds another minute value.
    days = (((side_obj.get("plan") or {}).get("days")) or [])
    for day in days:
        if not isinstance(day, dict):
            continue
        dur = day.get("duration_minutes")
        try:
            dur_int = int(dur) if dur is not None else None
        except Exception:
            dur_int = None
        if dur_int is None:
            continue
        text = " ".join(str(day.get(k, "")) for k in ("title", "workout_summary", "purpose_summary", "notes"))
        nums = [int(x) for x in MINUTES_RE.findall(text)]
        # A single mismatching embedded duration is sufficient. This is intentionally conservative:
        # it flags for sensitivity exclusion rather than editing or imputing.
        if nums and dur_int not in nums:
            flags["duration_text_inconsistency"] = 1
            examples.append({
                "flag": "duration_text_inconsistency",
                "context": f"duration_minutes={dur_int}; embedded_minutes={nums[:5]}; text={text[:200]}",
            })
            break
    if flags:
        flags["any_presentation_artifact"] = 1
    return flags, examples[:5]


def load_available_combined(frozen_root: Path, combined: Path, glob_pat: str, combined_name: str) -> Tuple[List[Dict[str, Any]], Path]:
    p = frozen_root / combined
    if p.exists():
        return read_jsonl(p), p
    parts = sorted(q for q in frozen_root.glob(glob_pat) if "all_judges" not in q.name and "failures" not in q.name)
    rows: List[Dict[str, Any]] = []
    for q in parts:
        rows.extend(read_jsonl(q))
    if rows:
        out = frozen_root / combined.parent / combined_name
        write_jsonl(out, rows)
        return rows, out
    return [], p


def proportion_ci_wald(wins: int, n: int) -> Tuple[Optional[float], Optional[float]]:
    if n <= 0:
        return None, None
    p = wins / n
    se = math.sqrt(max(0.0, p * (1.0 - p) / n))
    return p - 1.96 * se, p + 1.96 * se


def summarize_primary(rows: List[Dict[str, Any]], group_fields: List[str]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, ...], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        key = tuple(str(r.get(k)) for k in group_fields)
        groups[key].append(r)
    out: List[Dict[str, Any]] = []
    for key, rs in sorted(groups.items()):
        n = len(rs)
        llm = sum(1 for r in rs if r.get("winner_role") == "llm" or r.get("llm_won") is True)
        prog = sum(1 for r in rs if r.get("winner_role") == "programmatic" or r.get("llm_won") is False)
        ties = sum(1 for r in rs if r.get("winner_role") == "tie")
        lo, hi = proportion_ci_wald(llm, n)
        row: Dict[str, Any] = {k: v for k, v in zip(group_fields, key)}
        row.update({
            "n": n,
            "llm_wins": llm,
            "programmatic_wins": prog,
            "ties": ties,
            "llm_win_rate": llm / n if n else None,
            "wald_ci_low": lo,
            "wald_ci_high": hi,
        })
        out.append(row)
    return out


def flatten_markers(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    flat: List[Dict[str, Any]] = []
    for r in rows:
        for marker, m in (r.get("markers") or {}).items():
            flat.append({
                "record_id": r.get("record_id"),
                "pair_id": r.get("pair_id"),
                "judge": r.get("judge"),
                "judge_model_family": r.get("judge_model_family"),
                "source_model_family": r.get("source_model_family"),
                "self_family_match": r.get("self_family_match"),
                "fixture_id": r.get("fixture_id"),
                "order": r.get("order"),
                "run": r.get("run"),
                "llm_plan_id": r.get("llm_plan_id"),
                "programmatic_plan_id": r.get("programmatic_plan_id"),
                "marker": marker,
                "preferred": m.get("preferred"),
                "winner_role": m.get("winner_role"),
                "llm_won": True if m.get("winner_role") == "llm" else False if m.get("winner_role") == "programmatic" else None,
                "plan_a_score": m.get("plan_a_score"),
                "plan_b_score": m.get("plan_b_score"),
                "confidence": m.get("confidence"),
                "has_normalization_warning": bool(m.get("normalization_warnings")),
            })
    return flat


def summarize_marker(flat: List[Dict[str, Any]], group_fields: List[str]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, ...], List[Dict[str, Any]]] = defaultdict(list)
    for r in flat:
        key = tuple(str(r.get(k)) for k in group_fields)
        groups[key].append(r)
    out: List[Dict[str, Any]] = []
    for key, rs in sorted(groups.items()):
        n = len(rs)
        llm = sum(1 for r in rs if r.get("winner_role") == "llm")
        prog = sum(1 for r in rs if r.get("winner_role") == "programmatic")
        tie = sum(1 for r in rs if r.get("winner_role") == "tie")
        non_tie = llm + prog
        row: Dict[str, Any] = {k: v for k, v in zip(group_fields, key)}
        row.update({
            "n": n,
            "llm_wins": llm,
            "programmatic_wins": prog,
            "ties": tie,
            "tie_rate": tie / n if n else None,
            "llm_win_rate_all": llm / n if n else None,
            "llm_win_rate_excluding_ties": llm / non_tie if non_tie else None,
            "llm_win_rate_ties_half": (llm + 0.5 * tie) / n if n else None,
        })
        out.append(row)
    return out


def primary_pair_run_categories(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by: Dict[Tuple[Any, Any, Any], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for r in rows:
        by[(r.get("pair_id"), r.get("judge"), r.get("run"))][str(r.get("order"))] = r
    cats: List[Dict[str, Any]] = []
    for (pair_id, judge, run), d in sorted(by.items()):
        if set(d) != {"AB", "BA"}:
            cat = "missing_order"
            ref = next(iter(d.values()))
        else:
            ab, ba = d["AB"], d["BA"]
            ref = ab
            if ab.get("winner_role") == ba.get("winner_role") and ab.get("winner_role") in {"llm", "programmatic"}:
                cat = f"source_consistent_{ab.get('winner_role')}"
            elif ab.get("preferred") == ba.get("preferred") == "plan_a":
                cat = "position_consistent_plan_a"
            elif ab.get("preferred") == ba.get("preferred") == "plan_b":
                cat = "position_consistent_plan_b"
            else:
                cat = "other_inconsistent"
        cats.append({
            "pair_id": pair_id,
            "judge": judge,
            "run": run,
            "category": cat,
            "judge_model_family": ref.get("judge_model_family"),
            "source_model_family": ref.get("source_model_family"),
            "self_family_match": ref.get("self_family_match"),
            "fixture_id": ref.get("fixture_id"),
        })
    return cats


def marker_pair_run_categories(flat: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by: Dict[Tuple[Any, Any, Any, Any], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for r in flat:
        by[(r.get("pair_id"), r.get("judge"), r.get("run"), r.get("marker"))][str(r.get("order"))] = r
    cats: List[Dict[str, Any]] = []
    for (pair_id, judge, run, marker), d in sorted(by.items()):
        if set(d) != {"AB", "BA"}:
            cat = "missing_order"
            ref = next(iter(d.values()))
        else:
            ab, ba = d["AB"], d["BA"]
            ref = ab
            if ab.get("winner_role") == ba.get("winner_role") and ab.get("winner_role") in {"llm", "programmatic"}:
                cat = f"source_consistent_{ab.get('winner_role')}"
            elif ab.get("preferred") == ba.get("preferred") == "plan_a":
                cat = "position_consistent_plan_a"
            elif ab.get("preferred") == ba.get("preferred") == "plan_b":
                cat = "position_consistent_plan_b"
            elif ab.get("winner_role") == ba.get("winner_role") == "tie":
                cat = "consistent_tie"
            else:
                cat = "other_inconsistent"
        cats.append({
            "pair_id": pair_id,
            "judge": judge,
            "run": run,
            "marker": marker,
            "category": cat,
            "judge_model_family": ref.get("judge_model_family"),
            "source_model_family": ref.get("source_model_family"),
            "self_family_match": ref.get("self_family_match"),
            "fixture_id": ref.get("fixture_id"),
        })
    return cats


def summarize_categories(rows: List[Dict[str, Any]], group_fields: List[str]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, ...], Counter] = defaultdict(Counter)
    totals: Counter = Counter()
    for r in rows:
        key = tuple(str(r.get(k)) for k in group_fields)
        groups[key][str(r.get("category"))] += 1
        totals[key] += 1
    out: List[Dict[str, Any]] = []
    for key, ctr in sorted(groups.items()):
        total = totals[key]
        for cat, n in sorted(ctr.items()):
            row: Dict[str, Any] = {k: v for k, v in zip(group_fields, key)}
            row.update({"category": cat, "n": n, "rate": n / total if total else None})
            out.append(row)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen-root", required=True, help="Path to frozen_primary_v1 directory")
    ap.add_argument("--output-dir", required=True, help="Where to write leakage-filtered outputs")
    ap.add_argument("--strict-clean", action="store_true", help="Exclude pairs if either side has any artifact. Default also behaves this way.")
    args = ap.parse_args()

    frozen_root = Path(args.frozen_root).resolve()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    manifest_path = frozen_root / MARKER_MANIFEST
    manifest = read_jsonl(manifest_path)
    if len(manifest) != 10000:
        print(f"WARNING: expected 10000 manifest rows, got {len(manifest)}")

    # Record-level and pair-level leakage flags from final judge-facing inputs.
    pair_flags: Dict[str, Dict[str, Any]] = {}
    record_flags: List[Dict[str, Any]] = []
    plan_flags: Dict[str, Counter] = defaultdict(Counter)
    examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for r in manifest:
        pair_id = str(r.get("pair_id"))
        rec_id = str(r.get("record_id"))
        jp = find_judge_input(frozen_root, str(r.get("judge_input_path")))
        data = json.loads(jp.read_text(encoding="utf-8"))
        if pair_id not in pair_flags:
            pair_flags[pair_id] = {
                "pair_id": pair_id,
                "llm_plan_id": r.get("llm_plan_id"),
                "programmatic_plan_id": r.get("programmatic_plan_id"),
                "source_model_family": r.get("source_model_family"),
                "fixture_id": r.get("fixture_id"),
            }
        rec: Dict[str, Any] = {
            "record_id": rec_id,
            "pair_id": pair_id,
            "judge": r.get("judge"),
            "order": r.get("order"),
            "run": r.get("run"),
            "llm_plan_id": r.get("llm_plan_id"),
            "programmatic_plan_id": r.get("programmatic_plan_id"),
        }
        for side, role_key in (("plan_a", "left_plan_role"), ("plan_b", "right_plan_role")):
            role = str(r.get(role_key))
            pid = str(r.get("llm_plan_id") if role == "llm" else r.get("programmatic_plan_id"))
            side_obj = data.get(side) or {}
            flags, exs = plan_side_flags(side_obj)
            prefix = f"{role}_"
            any_side = int(bool(flags.get("any_presentation_artifact")))
            rec[f"{prefix}any_presentation_artifact"] = any_side
            pair_flags[pair_id][f"{prefix}any_presentation_artifact"] = max(int(pair_flags[pair_id].get(f"{prefix}any_presentation_artifact", 0)), any_side)
            for name in sorted(PATTERNS) + ["duration_text_inconsistency"]:
                val = int(bool(flags.get(name)))
                rec[f"{prefix}{name}"] = val
                pair_flags[pair_id][f"{prefix}{name}"] = max(int(pair_flags[pair_id].get(f"{prefix}{name}", 0)), val)
                if val:
                    plan_flags[pid][f"{role}_{name}"] = 1
            for ex in exs:
                if len(examples[ex["flag"]]) < 12:
                    examples[ex["flag"]].append({
                        "record_id": rec_id,
                        "pair_id": pair_id,
                        "role": role,
                        "side": side,
                        "plan_id": pid,
                        "context": ex["context"],
                    })
        rec["any_presentation_artifact"] = int(bool(rec.get("llm_any_presentation_artifact") or rec.get("programmatic_any_presentation_artifact")))
        record_flags.append(rec)

    # Finalize pair-level any flag.
    for pf in pair_flags.values():
        pf["any_presentation_artifact"] = int(bool(pf.get("llm_any_presentation_artifact") or pf.get("programmatic_any_presentation_artifact")))

    pair_rows = sorted(pair_flags.values(), key=lambda x: x["pair_id"])
    clean_pair_ids = {r["pair_id"] for r in pair_rows if int(r.get("any_presentation_artifact", 0)) == 0}
    flagged_pair_ids = {r["pair_id"] for r in pair_rows if int(r.get("any_presentation_artifact", 0)) == 1}

    write_csv(out / "leakage_flags_by_pair.csv", pair_rows)
    (out / "leakage_flags_by_pair.json").write_text(json.dumps(pair_rows, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(out / "leakage_flags_by_record.csv", record_flags)
    (out / "leakage_examples.json").write_text(json.dumps(examples, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    (out / "included_clean_pair_ids.txt").write_text("\n".join(sorted(clean_pair_ids)) + "\n", encoding="utf-8")
    (out / "excluded_flagged_pair_ids.txt").write_text("\n".join(sorted(flagged_pair_ids)) + "\n", encoding="utf-8")

    # Load judgments.
    primary_rows, primary_source = load_available_combined(
        frozen_root, PRIMARY_COMBINED, PRIMARY_GLOB, "pairwise_all_judges_canonical_masked_scrubbed_v1_t000.jsonl"
    )
    marker_rows, marker_source = load_available_combined(
        frozen_root, MARKER_COMBINED, MARKER_GLOB, "marker_all_judges_canonical_masked_scrubbed_v1_t000.jsonl"
    )

    def is_clean(r: Dict[str, Any]) -> bool:
        return str(r.get("pair_id")) in clean_pair_ids

    primary_clean = [r for r in primary_rows if is_clean(r)]
    primary_flagged = [r for r in primary_rows if not is_clean(r)]
    marker_clean = [r for r in marker_rows if is_clean(r)]
    marker_flagged = [r for r in marker_rows if not is_clean(r)]

    write_jsonl(out / "primary_clean_pairwise.jsonl", primary_clean)
    write_jsonl(out / "primary_flagged_pairwise.jsonl", primary_flagged)
    if marker_rows:
        write_jsonl(out / "marker_clean_pairwise.jsonl", marker_clean)
        write_jsonl(out / "marker_flagged_pairwise.jsonl", marker_flagged)

    # Add split labels for comparisons.
    primary_split = []
    for r in primary_rows:
        x = dict(r)
        x["leakage_split"] = "clean" if is_clean(r) else "flagged"
        primary_split.append(x)
    marker_split = []
    for r in marker_rows:
        x = dict(r)
        x["leakage_split"] = "clean" if is_clean(r) else "flagged"
        marker_split.append(x)

    # Summaries.
    summary_dir = out / "summaries"
    write_csv(summary_dir / "primary_clean_overall.csv", summarize_primary(primary_clean, ["judge_view"] if primary_clean and "judge_view" in primary_clean[0] else []))
    for name, fields in [
        ("primary_clean_by_judge.csv", ["judge"]),
        ("primary_clean_by_judge_family.csv", ["judge_model_family"]),
        ("primary_clean_by_source_family.csv", ["source_model_family"]),
        ("primary_clean_by_self_family.csv", ["self_family_match"]),
        ("primary_clean_by_order.csv", ["order"]),
        ("primary_by_leakage_split.csv", ["leakage_split"]),
        ("primary_by_leakage_split_and_judge.csv", ["leakage_split", "judge"]),
    ]:
        rows = primary_split if "leakage_split" in fields else primary_clean
        write_csv(summary_dir / name, summarize_primary(rows, fields))

    pr_clean = primary_pair_run_categories(primary_clean)
    pr_all = primary_pair_run_categories(primary_split)
    write_csv(summary_dir / "primary_clean_pair_run_consistency.csv", summarize_categories(pr_clean, []))
    write_csv(summary_dir / "primary_clean_pair_run_consistency_by_judge.csv", summarize_categories(pr_clean, ["judge"]))
    # For all split pair runs, attach split by pair_id.
    for row in pr_all:
        row["leakage_split"] = "clean" if row["pair_id"] in clean_pair_ids else "flagged"
    write_csv(summary_dir / "primary_pair_run_consistency_by_leakage_split.csv", summarize_categories(pr_all, ["leakage_split"]))

    if marker_rows:
        flat_clean = flatten_markers(marker_clean)
        flat_split = flatten_markers(marker_split)
        for f in flat_split:
            f["leakage_split"] = "clean" if str(f.get("pair_id")) in clean_pair_ids else "flagged"
        for name, fields in [
            ("marker_clean_overall_by_marker.csv", ["marker"]),
            ("marker_clean_by_marker_judge.csv", ["marker", "judge"]),
            ("marker_clean_by_marker_judge_family.csv", ["marker", "judge_model_family"]),
            ("marker_clean_by_marker_source_family.csv", ["marker", "source_model_family"]),
            ("marker_clean_by_marker_self_family.csv", ["marker", "self_family_match"]),
            ("marker_clean_by_marker_order.csv", ["marker", "order"]),
        ]:
            write_csv(summary_dir / name, summarize_marker(flat_clean, fields))
        write_csv(summary_dir / "marker_by_leakage_split_and_marker.csv", summarize_marker(flat_split, ["leakage_split", "marker"]))
        write_csv(summary_dir / "marker_by_leakage_split.csv", summarize_marker(flat_split, ["leakage_split"]))

        mpr_clean = marker_pair_run_categories(flat_clean)
        mpr_split = marker_pair_run_categories(flat_split)
        for row in mpr_split:
            row["leakage_split"] = "clean" if row["pair_id"] in clean_pair_ids else "flagged"
        write_csv(summary_dir / "marker_clean_pair_run_consistency_by_marker.csv", summarize_categories(mpr_clean, ["marker"]))
        write_csv(summary_dir / "marker_pair_run_consistency_by_leakage_split.csv", summarize_categories(mpr_split, ["leakage_split"]))
        write_csv(summary_dir / "marker_pair_run_consistency_by_leakage_split_marker.csv", summarize_categories(mpr_split, ["leakage_split", "marker"]))

    # Integrity + limitations report.
    flag_counter = Counter()
    for pf in pair_rows:
        for k, v in pf.items():
            if k.endswith("artifact") or k in ["llm_trailrun_pipe", "llm_avg_hr", "llm_km_unit", "llm_elevation_or_vertical", "llm_ellipsis", "llm_activity_like_fragment", "llm_duration_text_inconsistency", "programmatic_trailrun_pipe", "programmatic_avg_hr", "programmatic_km_unit", "programmatic_elevation_or_vertical", "programmatic_ellipsis", "programmatic_activity_like_fragment", "programmatic_duration_text_inconsistency"]:
                if isinstance(v, int) and v:
                    flag_counter[k] += 1
    integrity = {
        "frozen_root": str(frozen_root),
        "manifest_records": len(manifest),
        "matched_pairs_seen": len(pair_rows),
        "clean_pairs": len(clean_pair_ids),
        "flagged_pairs": len(flagged_pair_ids),
        "primary_source": str(primary_source),
        "primary_records_total": len(primary_rows),
        "primary_records_clean": len(primary_clean),
        "primary_records_flagged": len(primary_flagged),
        "marker_source": str(marker_source),
        "marker_records_total": len(marker_rows),
        "marker_records_clean": len(marker_clean),
        "marker_records_flagged": len(marker_flagged),
        "marker_decisions_clean": len(marker_clean) * 9 if marker_clean else 0,
        "pair_flag_counts": dict(flag_counter),
        "note": "Clean/flagged splits are post-hoc sensitivity datasets based on presentation artifacts in judge-facing inputs. They do not change the prompts originally seen by judges.",
    }
    (out / "leakage_filtered_integrity.json").write_text(json.dumps(integrity, indent=2, sort_keys=True), encoding="utf-8")
    report = [
        "# Leakage-filtered sensitivity dataset\n",
        "\n",
        "This output excludes matched pairs whose final judge-facing inputs contained searched presentation artifacts. ",
        "It is a post-hoc sensitivity dataset, not a rerun on scrubbed prompts.\n\n",
        f"- Matched pairs seen: {len(pair_rows)}\n",
        f"- Clean pairs retained: {len(clean_pair_ids)}\n",
        f"- Flagged pairs excluded: {len(flagged_pair_ids)}\n",
        f"- Primary records retained: {len(primary_clean)} / {len(primary_rows)}\n",
        f"- Marker records retained: {len(marker_clean)} / {len(marker_rows)}\n",
        f"- Marker decisions retained: {len(marker_clean) * 9 if marker_clean else 0}\n",
        "\n## Pair-level flag counts\n\n",
    ]
    for k, v in sorted(flag_counter.items()):
        report.append(f"- {k}: {v}\n")
    report.append("\n## Use in report\n\n")
    report.append(
        "Use the clean-pair summaries as a robustness/sensitivity analysis: results after excluding pairs with detected presentation leakage. "
        "Do not describe these as judgments made on scrubbed prompts unless a new evaluation pass is run on scrubbed-v2 inputs.\n"
    )
    (out / "README_LEAKAGE_FILTERED.md").write_text("".join(report), encoding="utf-8")

    print(json.dumps(integrity, indent=2, sort_keys=True))
    if len(clean_pair_ids) == 0:
        raise SystemExit("No clean pairs retained; check pattern definitions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
