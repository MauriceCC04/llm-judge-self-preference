"""Summarize an HPC generation run from stderr/stdout and output plans.

Usage:
    python tools/summarize_generation_run.py \
      --err err/generate_hpc_qwen_verify_fix4_all_486934.err \
      --out out/generate_hpc_qwen_verify_fix4_all_486934.out \
      --plans-dir plans_qwen_verify_fix4_all
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path
from typing import Any

STAGE_RE = re.compile(
    r"stage=(?P<stage>[^\s]+)\s+finish=(?P<finish>[^\s]+)\s+prompt_tokens=(?P<prompt>\d+)\s+completion_tokens=(?P<completion>\d+)"
)
DONE_RE = re.compile(r"Generated:\s+(?P<generated>\d+)")
FAILED_RE = re.compile(r"FAILED:")
REPAIR_RE = re.compile(r"Machine-plan parse/validation failed; attempting one structured repair")
PLACEHOLDER_MARKERS = (">{signal_id", "{signal_id")


def _load_text(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _collect_placeholder_hits(obj: Any, path: str = "$") -> list[tuple[str, str]]:
    hits: list[tuple[str, str]] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            hits.extend(_collect_placeholder_hits(value, f"{path}.{key}"))
        return hits
    if isinstance(obj, list):
        for idx, value in enumerate(obj):
            hits.extend(_collect_placeholder_hits(value, f"{path}[{idx}]"))
        return hits
    if isinstance(obj, str):
        text = obj.strip()
        if any(marker in text for marker in PLACEHOLDER_MARKERS):
            hits.append((path, text))
    return hits


def summarize(err_text: str, out_text: str, plans_dir: Path | None) -> dict[str, Any]:
    stage_records: list[dict[str, Any]] = []
    for m in STAGE_RE.finditer(err_text):
        stage_records.append(
            {
                "stage": m.group("stage"),
                "finish": m.group("finish"),
                "prompt_tokens": int(m.group("prompt")),
                "completion_tokens": int(m.group("completion")),
            }
        )

    plans: list[dict[str, Any]] = []
    placeholder_total = 0
    bad_durations: list[tuple[str, str, int]] = []
    if plans_dir is not None and plans_dir.exists():
        for path_str in sorted(glob.glob(str(plans_dir / "*.json"))):
            if path_str.endswith(".provenance.json"):
                continue
            path = Path(path_str)
            obj = json.loads(path.read_text(encoding="utf-8"))
            hits = _collect_placeholder_hits(obj)
            placeholder_total += len(hits)
            for day in obj.get("plan", {}).get("days", []):
                dur = day.get("duration_minutes")
                if isinstance(dur, int) and (dur < 0 or dur > 420):
                    bad_durations.append((path.name, str(day.get("date")), dur))
            plans.append(
                {
                    "file": path.name,
                    "placeholder_hits": len(hits),
                    "days": len(obj.get("plan", {}).get("days", [])),
                }
            )

    generated = None
    m = DONE_RE.search(out_text)
    if m:
        generated = int(m.group("generated"))

    return {
        "generated": generated,
        "failed_lines": len(FAILED_RE.findall(out_text)),
        "source_repairs": len(REPAIR_RE.findall(err_text)),
        "stage_counts": {
            "machine_plan": sum(1 for r in stage_records if r["stage"] == "machine_plan"),
            "machine_plan_repair": sum(1 for r in stage_records if r["stage"] in {"machine_plan_repair", "trailtraining_machine_plan_v2"}),
            "plan_explanation": sum(1 for r in stage_records if r["stage"] == "plan_explanation"),
        },
        "non_stop_finishes": [r for r in stage_records if r["finish"] != "stop"],
        "plans": plans,
        "placeholder_hits_total": placeholder_total,
        "bad_durations": bad_durations,
        "raw_failures_present": bool(plans_dir and (plans_dir / "raw_failures").exists()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--err", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plans-dir", type=Path, required=True)
    args = parser.parse_args()

    summary = summarize(
        err_text=_load_text(args.err),
        out_text=_load_text(args.out),
        plans_dir=args.plans_dir,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
