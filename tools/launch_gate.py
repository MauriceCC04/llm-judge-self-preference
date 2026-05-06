"""Full-study launch gate for the 10,000 pairwise-judgment Qwen/Gemma design."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from generate.constants import ACTIVE_JUDGE_NAMES, FULL_STUDY_EXACT_PAIRS, FULL_STUDY_MIN_PAIRS, FULL_STUDY_REQUIRED_JUDGES, FULL_STUDY_REQUIRED_PAIRWISE_DOCUMENTS, PAIRWISE_N_RUNS
from tools.build_pairwise_eval_manifest import audit_masked_payload, mask_plan_for_judge


def _family(model_or_name: Any, arm: str | None = None) -> str:
    s = str(model_or_name or "").lower()
    if "qwen" in s:
        return "qwen"
    if "gemma" in s:
        return "gemma"
    if arm == "programmatic" or not s or s == "none":
        return "programmatic"
    return "other"


def _load_plan(plans_dir: Path, plan_id: str) -> dict[str, Any]:
    return json.loads((plans_dir / f"{plan_id}.json").read_text(encoding="utf-8"))


def _judge_models() -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    try:
        from judge.panel import get_judge
        for name in ACTIVE_JUDGE_NAMES:
            spec = get_judge(name)
            model_id = str(getattr(spec, "model_id", "") or "")
            out.append({"name": name, "model_id": model_id, "family": _family(model_id or name)})
    except Exception:
        for name in ACTIVE_JUDGE_NAMES:
            out.append({"name": name, "model_id": name, "family": _family(name)})
    return out


def _structural_matching_audit(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    gaps = [float(p.get("structural_score_gap", p.get("score_gap", 999))) for p in pairs]
    version_ok = all(p.get("structural_score_version") for p in pairs)
    same_cell_ok = all(p.get("fixture_id") for p in pairs)
    return {
        "passed": bool(pairs) and version_ok and same_cell_ok and max(gaps, default=999) <= 2.0,
        "version_present": version_ok,
        "same_cell_metadata_present": same_cell_ok,
        "max_structural_score_gap": max(gaps, default=None),
        "mean_structural_score_gap": sum(gaps) / len(gaps) if gaps else None,
    }


def _source_masking_audit(plans_dir: Path, pairs: list[dict[str, Any]], limit: int = 50) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    for pair in pairs[:limit]:
        payload = {
            "plan_a": mask_plan_for_judge(_load_plan(plans_dir, pair["plan_a_id"])),
            "plan_b": mask_plan_for_judge(_load_plan(plans_dir, pair["plan_b_id"])),
        }
        audit = audit_masked_payload(payload)
        if not audit["passed"]:
            failures.append({"pair_id": pair.get("pair_id"), **audit})
    return {"passed": not failures, "checked_pairs": min(limit, len(pairs)), "failures": failures[:20]}


def _style_gate(style_audit_path: Path | None, require_style_audit: bool) -> dict[str, Any]:
    if not require_style_audit:
        return {"passed": True, "skipped": True, "note": "style audit explicitly skipped"}
    if style_audit_path is None or not style_audit_path.exists():
        return {"passed": False, "reason": "style audit path missing"}
    payload = json.loads(style_audit_path.read_text(encoding="utf-8"))
    gate = payload.get("gate", payload)
    return {"passed": bool(gate.get("passed", False)), "gate": gate}


def run_launch_gate(
    *,
    plans_dir: Path,
    pairs_path: Path,
    style_audit_path: Path | None = None,
    n_runs: int = PAIRWISE_N_RUNS,
    orders: list[str] | None = None,
    require_style_audit: bool = True,
) -> dict[str, Any]:
    orders = orders or ["AB", "BA"]
    pairs = json.loads(pairs_path.read_text(encoding="utf-8")) if pairs_path.exists() else []
    judges = _judge_models()
    judge_families = {j["family"] for j in judges}
    source_families = {_family(p.get("source_model_a"), p.get("arm_a")) for p in pairs}
    expected_pairwise_documents = len(pairs) * len(judges) * int(n_runs) * len(set(orders))

    checks = {
        "matched_pairs_exactly_250": len(pairs) == FULL_STUDY_EXACT_PAIRS,
        "expected_pairwise_documents_exactly_10000": expected_pairwise_documents == FULL_STUDY_REQUIRED_PAIRWISE_DOCUMENTS,
        "both_source_families_present": {"qwen", "gemma"}.issubset(source_families),
        "exactly_4_qwen_gemma_judges": len(judges) == FULL_STUDY_REQUIRED_JUDGES and all(j["family"] in {"qwen", "gemma"} for j in judges),
        "five_repeats": int(n_runs) == 5,
        "both_ab_ba_orders": set(orders) == {"AB", "BA"},
    }
    structural = _structural_matching_audit(pairs)
    masking = _source_masking_audit(plans_dir, pairs)
    style = _style_gate(style_audit_path, require_style_audit)
    checks["structural_matching_audit_passes"] = structural["passed"]
    checks["source_masking_audit_passes"] = masking["passed"]
    checks["style_audit_passes"] = style["passed"]

    return {
        "passed": all(checks.values()),
        "checks": checks,
        "n_pairs": len(pairs),
        "judges": judges,
        "judge_families": sorted(judge_families),
        "source_families": sorted(source_families),
        "n_runs": n_runs,
        "orders": sorted(set(orders)),
        "expected_pairwise_documents": expected_pairwise_documents,
        "structural_matching_audit": structural,
        "source_masking_audit": masking,
        "style_audit": style,
        "requirement_formula": "250 matched pairs x 4 judges x 5 runs x 2 positions = 10,000 pairwise judgments",
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plans", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--style-audit", default=None)
    parser.add_argument("--runs", type=int, default=PAIRWISE_N_RUNS)
    parser.add_argument("--order", action="append", choices=["AB", "BA"], default=None)
    parser.add_argument("--skip-style-audit", action="store_true")
    args = parser.parse_args(argv)
    result = run_launch_gate(plans_dir=Path(args.plans), pairs_path=Path(args.pairs), style_audit_path=Path(args.style_audit) if args.style_audit else None, n_runs=args.runs, orders=args.order, require_style_audit=not args.skip_style_audit)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(0 if result["passed"] else 2)


if __name__ == "__main__":
    main()
