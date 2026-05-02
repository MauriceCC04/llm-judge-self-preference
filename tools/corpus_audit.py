from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from match.filtering import compact_plan_summary, filter_plan_records


def _load_records(plans_dir: Path, provenance_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for prov_path in sorted(provenance_dir.glob("*.provenance.json")):
        prov = json.loads(prov_path.read_text(encoding="utf-8"))
        plan_id = str(prov.get("plan_id") or "")
        if not plan_id:
            continue
        plan_path = plans_dir / f"{plan_id}.json"
        if not plan_path.exists():
            continue
        plan_obj = json.loads(plan_path.read_text(encoding="utf-8"))
        records.append(
            {
                "plan_id": plan_id,
                "fixture_id": str(prov.get("fixture_id") or ""),
                "athlete_band": str(prov.get("athlete_band") or ""),
                "readiness": str(prov.get("readiness") or ""),
                "recovery_capability": str(prov.get("recovery_capability") or ""),
                "race_phase": str(prov.get("race_phase") or ""),
                "arm": str(prov.get("arm") or ""),
                "score": prov.get("deterministic_score"),
                "plan_obj": plan_obj,
            }
        )
    return records


def run_corpus_audit(
    *,
    plans_dir: Path,
    provenance_dir: Path,
    sample_per_cell: int = 1,
) -> dict[str, Any]:
    records = _load_records(plans_dir, provenance_dir)
    filtered = filter_plan_records(records)

    by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in filtered["kept_records"]:
        by_cell[str(record["fixture_id"])].append(record)

    sampled_cells: list[dict[str, Any]] = []
    for fixture_id, group in sorted(by_cell.items()):
        group_sorted = sorted(group, key=lambda r: (str(r.get("arm") or ""), str(r.get("plan_id") or "")))
        for record in group_sorted[:sample_per_cell]:
            sampled_cells.append(
                {
                    "fixture_id": fixture_id,
                    "athlete_band": record.get("athlete_band"),
                    "readiness": record.get("readiness"),
                    "recovery_capability": record.get("recovery_capability"),
                    "race_phase": record.get("race_phase"),
                    "arm": record.get("arm"),
                    "plan_id": record.get("plan_id"),
                    "score": record.get("score"),
                    "summary": compact_plan_summary(record["plan_obj"]),
                }
            )

    kept_by_cell = {
        fixture_id: {
            "total_kept": len(group),
            "by_arm": {
                arm: sum(1 for record in group if str(record.get("arm") or "") == arm)
                for arm in sorted({str(record.get("arm") or "") for record in group})
            },
        }
        for fixture_id, group in sorted(by_cell.items())
    }

    return {
        "n_loaded": len(records),
        "n_kept": len(filtered["kept_records"]),
        "n_dropped": len(filtered["dropped_records"]),
        "kept_by_cell": kept_by_cell,
        "sampled_cells": sampled_cells,
        "exact_text_duplicate_groups": filtered["audit"]["exact_text_duplicate_groups"],
        "session_signature_duplicate_groups": filtered["audit"]["session_signature_duplicate_groups"],
        "invalid_or_contradictory": filtered["audit"]["invalid_or_contradictory"],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Audit plan corpus by cell, duplicates, and contradictions.")
    parser.add_argument("--plans", default="plans/")
    parser.add_argument("--provenance", default=None)
    parser.add_argument("--sample-per-cell", type=int, default=1)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    plans_dir = Path(args.plans)
    provenance_dir = Path(args.provenance or args.plans)
    result = run_corpus_audit(
        plans_dir=plans_dir,
        provenance_dir=provenance_dir,
        sample_per_cell=args.sample_per_cell,
    )

    print("\\n=== Corpus audit ===")
    print(f"Loaded:  {result['n_loaded']}")
    print(f"Kept:    {result['n_kept']}")
    print(f"Dropped: {result['n_dropped']}")
    print(f"Exact-text duplicate groups: {len(result['exact_text_duplicate_groups'])}")
    print(f"Session-signature duplicate groups: {len(result['session_signature_duplicate_groups'])}")
    print(f"Invalid/contradictory plans: {len(result['invalid_or_contradictory'])}")

    for sample in result["sampled_cells"]:
        summary = sample["summary"]
        print(
            f"- {sample['fixture_id']} [{sample['arm']}] {sample['plan_id']} "
            f"(score={sample['score']}, min={summary['total_minutes']}, "
            f"rest={summary['n_rest_days']}, hard={summary['n_hard_days']}, long={summary['has_long_run']})"
        )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\\n[Saved] {out_path}")


if __name__ == "__main__":
    main()
