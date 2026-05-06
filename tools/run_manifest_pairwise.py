#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from compat.trailtraining_client import install_trailtraining_client_compat
from judge.panel import get_judge
from trailtraining.llm.soft_eval import SoftEvalConfig, compare_plans


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def existing_record_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                rid = row.get("record_id")
                if rid:
                    out.add(str(rid))
            except Exception:
                continue
    return out


def winner_role(preferred: str, row: dict[str, Any]) -> str:
    preferred = str(preferred or "tie").lower()
    if preferred == "tie":
        return "tie"
    if preferred == "plan_a":
        return str(row.get("left_plan_role") or "unknown")
    if preferred == "plan_b":
        return str(row.get("right_plan_role") or "unknown")
    return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--judge", required=True)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--failures", default=None, type=Path)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    install_trailtraining_client_compat(default_stage="judge")

    judge = get_judge(args.judge)
    rows = [r for r in load_jsonl(args.manifest) if r.get("judge") == args.judge]
    if args.limit is not None:
        rows = rows[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    failure_path = args.failures or args.output.with_name(args.output.stem + "_failures.jsonl")
    done = existing_record_ids(args.output)

    cfg = SoftEvalConfig(
        enabled=True,
        model=judge.model_id,
        reasoning_effort="none",
        skip_synthesis=True,
        parallel_batches=False,
        temperature=args.temperature,
    )

    print(f"judge={args.judge}")
    print(f"model_id={judge.model_id}")
    print(f"manifest={args.manifest}")
    print(f"rows_for_judge={len(rows)}")
    print(f"already_done={len(done)}")
    print(f"output={args.output}")

    attempted = 0
    written = 0
    failed = 0

    for row in rows:
        record_id = str(row["record_id"])
        if record_id in done:
            continue

        input_path = Path(row["judge_input_path"])
        if not input_path.exists():
            append_jsonl(failure_path, {
                **row,
                "error": f"missing judge_input_path: {input_path}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            failed += 1
            continue

        payload = json.loads(input_path.read_text(encoding="utf-8"))
        plan_a = payload["plan_a"]
        plan_b = payload["plan_b"]

        attempted += 1
        try:
            result = compare_plans(plan_a, plan_b, rollups=None, cfg=cfg)
            preferred = str(result.get("preferred", "tie") or "tie").lower()
            role = winner_role(preferred, row)
            append_jsonl(args.output, {
                **row,
                "judge_model_id": judge.model_id,
                "judge_temperature": args.temperature,
                "preferred": preferred,
                "winner_role": role,
                "llm_won": True if role == "llm" else False if role == "programmatic" else None,
                "reasoning": str(result.get("reasoning", "") or ""),
                "plan_a_advantages": result.get("plan_a_advantages", []),
                "plan_b_advantages": result.get("plan_b_advantages", []),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            done.add(record_id)
            written += 1
            if written % 25 == 0:
                print(f"written={written} attempted={attempted} failed={failed}", flush=True)
        except Exception as exc:
            append_jsonl(failure_path, {
                **row,
                "judge_model_id": judge.model_id,
                "judge_temperature": args.temperature,
                "error": f"{type(exc).__name__}: {exc}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            failed += 1

    print(json.dumps({
        "judge": args.judge,
        "rows_for_judge": len(rows),
        "attempted": attempted,
        "written": written,
        "failed": failed,
        "output": str(args.output),
        "failures": str(failure_path),
    }, indent=2))


if __name__ == "__main__":
    main()
