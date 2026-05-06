"""Build a source-masked pairwise evaluation manifest.

One manifest row is one judge-facing pairwise comparison instance:
matched pair × judge model × run index × AB/BA order.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

from generate.constants import ACTIVE_JUDGE_NAMES, PAIRWISE_N_RUNS

SOURCE_KEYS = {
    "source_model",
    "source_family",
    "arm",
    "generation_condition",
    "generation_pipeline",
    "runtime_backend",
    "runtime_metadata",
    "explainer_model",
    "actual_explainer_model",
    "explainer_model_verified",
}
SOURCE_PATTERNS = [re.compile(r"qwen", re.I), re.compile(r"gemma", re.I), re.compile(r"programmatic", re.I), re.compile(r"source_model", re.I)]


def _family(model: Any, arm: str | None = None) -> str:
    s = str(model or "").lower()
    if "qwen" in s:
        return "qwen"
    if "gemma" in s:
        return "gemma"
    if arm == "programmatic" or not s or s == "none":
        return "programmatic"
    return "other"


def mask_plan_for_judge(plan: dict[str, Any]) -> dict[str, Any]:
    """Return a judge-facing plan object with provenance/source fields removed."""
    def clean(obj: Any) -> Any:
        if isinstance(obj, dict):
            out: dict[str, Any] = {}
            for key, value in obj.items():
                if key in SOURCE_KEYS:
                    continue
                if key == "meta" and isinstance(value, dict):
                    out[key] = {k: clean(v) for k, v in value.items() if k not in SOURCE_KEYS}
                else:
                    out[key] = clean(value)
            return out
        if isinstance(obj, list):
            return [clean(v) for v in obj]
        return obj

    return clean(plan)


def audit_masked_payload(payload: dict[str, Any]) -> dict[str, Any]:
    text = json.dumps(payload, ensure_ascii=False)
    hits = sorted({pattern.pattern for pattern in SOURCE_PATTERNS if pattern.search(text)})
    return {"passed": not hits, "source_pattern_hits": hits}


def _load_plan(plans_dir: Path, plan_id: str) -> dict[str, Any]:
    return json.loads((plans_dir / f"{plan_id}.json").read_text(encoding="utf-8"))


def build_manifest(
    *,
    plans_dir: Path,
    pairs_path: Path,
    output_dir: Path,
    judge_names: list[str] | None = None,
    n_runs: int = PAIRWISE_N_RUNS,
    orders: list[str] | None = None,
    max_pairs: int | None = None,
    seed: int = 0,
    write_judge_inputs: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    judge_names = judge_names or list(ACTIVE_JUDGE_NAMES)
    orders = orders or ["AB", "BA"]
    pairs = json.loads(pairs_path.read_text(encoding="utf-8"))
    if max_pairs is not None:
        pairs = pairs[:max_pairs]

    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    masking_failures: list[dict[str, Any]] = []
    judge_input_dir = output_dir / "judge_inputs"
    if write_judge_inputs:
        judge_input_dir.mkdir(parents=True, exist_ok=True)

    for pair in pairs:
        raw_a = _load_plan(plans_dir, pair["plan_a_id"])
        raw_b = _load_plan(plans_dir, pair["plan_b_id"])
        masked_a = mask_plan_for_judge(raw_a)
        masked_b = mask_plan_for_judge(raw_b)
        pair_mask_audit = audit_masked_payload({"plan_a": masked_a, "plan_b": masked_b})
        pair_orders = list(orders)
        rng.shuffle(pair_orders)
        for judge_name in judge_names:
            judge_family = _family(judge_name)
            source_family = _family(pair.get("source_model_a"), pair.get("arm_a"))
            for run in range(n_runs):
                run_orders = list(pair_orders)
                rng.shuffle(run_orders)
                for order in run_orders:
                    left, right = (masked_a, masked_b) if order == "AB" else (masked_b, masked_a)
                    judge_payload = {"plan_a": left, "plan_b": right}
                    record_id = f"{pair['pair_id']}__{judge_name}__r{run:02d}__{order}"
                    if not pair_mask_audit["passed"]:
                        masking_failures.append({"record_id": record_id, **pair_mask_audit})
                    if write_judge_inputs:
                        (judge_input_dir / f"{record_id}.json").write_text(json.dumps(judge_payload, indent=2, ensure_ascii=False), encoding="utf-8")
                    rows.append(
                        {
                            "record_id": record_id,
                            "pair_id": pair["pair_id"],
                            "fixture_id": pair.get("fixture_id"),
                            "judge": judge_name,
                            "judge_model_family": judge_family,
                            "run": run,
                            "order": order,
                            "position": order,
                            "left_plan_role": "llm" if order == "AB" else "programmatic",
                            "right_plan_role": "programmatic" if order == "AB" else "llm",
                            "llm_plan_id": pair["plan_a_id"],
                            "programmatic_plan_id": pair["plan_b_id"],
                            "source_model_family": source_family,
                            "self_family_match": source_family in {"qwen", "gemma"} and source_family == judge_family,
                            "structural_score_gap": pair.get("structural_score_gap", pair.get("score_gap")),
                            "structural_score_version": pair.get("structural_score_version"),
                            "judge_input_path": str(judge_input_dir / f"{record_id}.json") if write_judge_inputs else None,
                        }
                    )

    manifest_path = output_dir / "pairwise_eval_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    audit = {
        "n_pairs": len(pairs),
        "n_judges": len(judge_names),
        "n_runs": n_runs,
        "orders": sorted(set(orders)),
        "expected_pairwise_documents": len(rows),
        "exact_10000_gate_passed": len(rows) == 10000 and len(pairs) == 250 and len(judge_names) == 4 and n_runs == 5 and set(orders) == {"AB", "BA"},
        "masking_gate_passed": not masking_failures,
        "masking_failures": masking_failures[:50],
        "manifest_path": str(manifest_path),
    }
    (output_dir / "pairwise_eval_manifest_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    return audit


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plans", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--judge", action="append", default=None)
    parser.add_argument("--runs", type=int, default=PAIRWISE_N_RUNS)
    parser.add_argument("--order", action="append", choices=["AB", "BA"], default=None)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--write-judge-inputs", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(build_manifest(plans_dir=Path(args.plans), pairs_path=Path(args.pairs), output_dir=Path(args.output), judge_names=args.judge, n_runs=args.runs, orders=args.order, max_pairs=args.max_pairs, seed=args.seed, write_judge_inputs=args.write_judge_inputs), indent=2))


if __name__ == "__main__":
    main()
