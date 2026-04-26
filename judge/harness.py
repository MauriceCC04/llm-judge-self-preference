"""judge/harness.py — pairwise + soft-eval harness with crash-safe resume."""
from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Optional

from compat.trailtraining_client import install_trailtraining_client_compat
from generate.constants import PAIRWISE_TEXT_CHAR_LIMITS, PAIRWISE_VIEW_DEFAULT
from judge.canonicalize import canonicalize_pair_for_pairwise
from judge.normalize import normalize_pair_for_pairwise

log = logging.getLogger(__name__)
_COMPARE_PARSE_SENTINEL = "Could not parse comparison response."


def _load_plan(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_rollups(rollups_path: Path | None) -> Optional[dict[str, Any]]:
    if rollups_path and rollups_path.exists():
        return json.loads(rollups_path.read_text(encoding="utf-8"))
    return None


def _resolve_rollups_for_fixture(
    fixture_id: str | None,
    *,
    rollups_path: Path | None,
    fixtures_dir: Path | None,
    cache: dict[str, Optional[dict[str, Any]]],
) -> Optional[dict[str, Any]]:
    if fixture_id and fixtures_dir:
        if fixture_id not in cache:
            fixture_rollups = fixtures_dir / fixture_id / "combined_rollups.json"
            cache[fixture_id] = _load_rollups(fixture_rollups)
        return cache[fixture_id]
    return _load_rollups(rollups_path)


def _fixture_id_from_provenance(plan_id: str, provenance_dir: Path | None) -> str:
    if provenance_dir is None:
        return ""
    provenance_path = provenance_dir / f"{plan_id}.json.provenance.json"
    if not provenance_path.exists():
        return ""
    try:
        return str(json.loads(provenance_path.read_text(encoding="utf-8")).get("fixture_id", "") or "")
    except Exception:
        return ""


def _prepare_pairwise_view(
    plan_a: dict[str, Any],
    plan_b: dict[str, Any],
    *,
    pairwise_view: str,
    normalize_inputs: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    out_a, out_b = plan_a, plan_b
    if normalize_inputs:
        out_a, out_b = normalize_pair_for_pairwise(
            out_a,
            out_b,
            text_char_limits=PAIRWISE_TEXT_CHAR_LIMITS,
        )
    if pairwise_view == "canonical_masked":
        out_a, out_b = canonicalize_pair_for_pairwise(out_a, out_b)
    elif pairwise_view != "raw_normalized":
        raise ValueError(f"Unknown pairwise_view: {pairwise_view!r}")
    return out_a, out_b


def run_pairwise_harness(
    pairs: list[dict[str, Any]],
    plans_dir: Path,
    judge: Any,
    rollups_path: Optional[Path],
    output_path: Path,
    *,
    fixtures_dir: Optional[Path] = None,
    n_runs: int = 3,
    n_positions: int = 2,
    normalize_inputs: bool = True,
    pairwise_view: str = PAIRWISE_VIEW_DEFAULT,
    schema_failures_path: Optional[Path] = None,
) -> None:
    install_trailtraining_client_compat(default_stage="judge")
    from judge.outputs import PairwiseWriter, SchemaFailWriter
    from trailtraining.llm.soft_eval import SoftEvalConfig, compare_plans

    writer = PairwiseWriter(output_path)
    fail_writer = SchemaFailWriter(schema_failures_path or output_path.parent / "schema_failures.jsonl")
    rollups_cache: dict[str, Optional[dict[str, Any]]] = {}

    cfg = SoftEvalConfig(
        enabled=True,
        model=judge.model_id,
        reasoning_effort="none",
        skip_synthesis=True,
        parallel_batches=False,
    )
    if not cfg.enabled:
        raise RuntimeError("SoftEvalConfig.enabled must be True to call compare_plans.")

    positions = ["AB", "BA"][:n_positions]

    for pair in pairs:
        pair_id = pair["pair_id"]
        fixture_id = str(pair.get("fixture_id", "") or "")
        plan_a_path = plans_dir / f"{pair['plan_a_id']}.json"
        plan_b_path = plans_dir / f"{pair['plan_b_id']}.json"
        if not plan_a_path.exists() or not plan_b_path.exists():
            log.warning("Missing plan files for pair %s — skipping", pair_id)
            continue

        raw_plan_a = _load_plan(plan_a_path)
        raw_plan_b = _load_plan(plan_b_path)
        rollups = _resolve_rollups_for_fixture(
            fixture_id,
            rollups_path=rollups_path,
            fixtures_dir=fixtures_dir,
            cache=rollups_cache,
        )

        for run in range(n_runs):
            for position in positions:
                stub = {
                    "pair_id": pair_id,
                    "judge": judge.name,
                    "run": run,
                    "position": position,
                    "pairwise_view": pairwise_view,
                }
                if writer.exists(stub):
                    continue

                plan_a, plan_b = (raw_plan_a, raw_plan_b) if position == "AB" else (raw_plan_b, raw_plan_a)
                plan_a, plan_b = _prepare_pairwise_view(
                    plan_a,
                    plan_b,
                    pairwise_view=pairwise_view,
                    normalize_inputs=normalize_inputs,
                )

                try:
                    result = compare_plans(plan_a, plan_b, rollups=rollups, cfg=cfg)
                except Exception as exc:
                    log.warning("compare_plans raised pair=%s run=%d pos=%s: %s", pair_id, run, position, exc)
                    fail_writer.append(
                        {
                            "pair_id": pair_id,
                            "judge": judge.name,
                            "call_type": "compare_plans",
                            "run": run,
                            "position": position,
                            "pairwise_view": pairwise_view,
                            "error": str(exc),
                            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                        }
                    )
                    continue

                preferred_raw = result.get("preferred", "tie")
                reasoning = (result.get("reasoning") or "").strip()
                if preferred_raw == "tie" and reasoning == _COMPARE_PARSE_SENTINEL:
                    fail_writer.append(
                        {
                            "pair_id": pair_id,
                            "judge": judge.name,
                            "call_type": "compare_plans",
                            "run": run,
                            "position": position,
                            "pairwise_view": pairwise_view,
                            "error": "compare_plans returned schema-failure sentinel",
                            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                        }
                    )
                    continue

                if position == "AB":
                    preferred_id = (
                        pair["plan_a_id"]
                        if preferred_raw == "plan_a"
                        else pair["plan_b_id"]
                        if preferred_raw == "plan_b"
                        else "tie"
                    )
                else:
                    preferred_id = (
                        pair["plan_b_id"]
                        if preferred_raw == "plan_a"
                        else pair["plan_a_id"]
                        if preferred_raw == "plan_b"
                        else "tie"
                    )

                writer.append(
                    {
                        **stub,
                        "preferred": preferred_raw,
                        "preferred_id": preferred_id,
                        "plan_a_id": pair["plan_a_id"],
                        "plan_b_id": pair["plan_b_id"],
                        "reasoning": reasoning,
                        "fixture_id": fixture_id,
                        "normalized_inputs": normalize_inputs,
                        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                    }
                )


def run_soft_eval_harness(
    plan_ids: list[str],
    plans_dir: Path,
    judge: Any,
    rollups_path: Optional[Path],
    output_path: Path,
    *,
    fixtures_dir: Optional[Path] = None,
    provenance_dir: Optional[Path] = None,
    schema_failures_path: Optional[Path] = None,
) -> None:
    install_trailtraining_client_compat(default_stage="judge")
    from judge.outputs import SchemaFailWriter, SoftEvalWriter
    from trailtraining.llm.constraints import ConstraintConfig, evaluate_training_plan_quality
    from trailtraining.llm.soft_eval import SoftEvalConfig, evaluate_training_plan_soft

    writer = SoftEvalWriter(output_path)
    fail_writer = SchemaFailWriter(schema_failures_path or output_path.parent / "schema_failures.jsonl")
    rollups_cache: dict[str, Optional[dict[str, Any]]] = {}

    cfg = SoftEvalConfig(
        enabled=True,
        model=judge.model_id,
        reasoning_effort="none",
        skip_synthesis=True,
        parallel_batches=False,
    )
    if not cfg.enabled:
        raise RuntimeError("SoftEvalConfig.enabled must be True to call evaluate_training_plan_soft.")

    det_cfg = ConstraintConfig(min_signal_ids_per_day=0)

    for plan_id in plan_ids:
        stub = {"plan_id": plan_id, "judge": judge.name}
        if writer.exists(stub):
            continue
        plan_path = plans_dir / f"{plan_id}.json"
        if not plan_path.exists():
            log.warning("Missing plan %s — skipping", plan_id)
            continue

        fixture_id = _fixture_id_from_provenance(plan_id, provenance_dir or plans_dir)
        rollups = _resolve_rollups_for_fixture(
            fixture_id,
            rollups_path=rollups_path,
            fixtures_dir=fixtures_dir,
            cache=rollups_cache,
        )
        plan_obj = _load_plan(plan_path)
        det_report = evaluate_training_plan_quality(plan_obj, rollups, det_cfg)

        try:
            assessment = evaluate_training_plan_soft(plan_obj, det_report, rollups, cfg)
            writer.append(
                {
                    **stub,
                    "fixture_id": fixture_id,
                    "overall_score": assessment.get("overall_score"),
                    "rubric_scores": assessment.get("rubric_scores", {}),
                    "marker_results": assessment.get("marker_results", []),
                    "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                }
            )
        except Exception as exc:
            log.warning("Failure plan=%s judge=%s: %s", plan_id, judge.name, exc)
            fail_writer.append(
                {
                    "plan_id": plan_id,
                    "judge": judge.name,
                    "call_type": "soft_eval",
                    "error": str(exc),
                    "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                }
            )


def _pair_consistency(records: list[dict[str, Any]]) -> float:
    by_pair: dict[str, list[str]] = defaultdict(list)
    for record in records:
        preferred_id = str(record.get("preferred_id") or "")
        if preferred_id and preferred_id != "tie":
            by_pair[str(record.get("pair_id") or "")].append(preferred_id)
    scores: list[float] = []
    for preferred_ids in by_pair.values():
        if not preferred_ids:
            continue
        counts = Counter(preferred_ids)
        scores.append(max(counts.values()) / len(preferred_ids))
    return float(mean(scores)) if scores else float("nan")


def _swap_agreement(records: list[dict[str, Any]]) -> float:
    grouped: dict[tuple[str, int], dict[str, str]] = defaultdict(dict)
    for record in records:
        pair_id = str(record.get("pair_id") or "")
        run = int(record.get("run", -1))
        position = str(record.get("position") or "")
        preferred_id = str(record.get("preferred_id") or "")
        if preferred_id and preferred_id != "tie" and position in {"AB", "BA"}:
            grouped[(pair_id, run)][position] = preferred_id
    agreements: list[float] = []
    for choices in grouped.values():
        if "AB" in choices and "BA" in choices:
            agreements.append(1.0 if choices["AB"] == choices["BA"] else 0.0)
    return float(mean(agreements)) if agreements else float("nan")


def check_pilot_bias_gate(
    output_path: Path,
    *,
    threshold: float = 0.2,
    min_records: int = 20,
    max_tie_rate: float = 0.5,
    min_pair_consistency: float = 0.6,
    min_swap_agreement: float = 0.6,
) -> dict[str, Any]:
    from vendor_patches.resume_jsonl import load_all

    records = load_all(output_path)
    n = len(records)
    if n < min_records:
        return {
            "passed": False,
            "p_prefers_a": float("nan"),
            "n": n,
            "message": f"Too few records ({n} < {min_records}) for reliable bias check.",
        }

    n_prefers_a = sum(1 for record in records if record.get("preferred") == "plan_a")
    n_ties = sum(1 for record in records if str(record.get("preferred_id") or record.get("preferred") or "") == "tie")
    p_prefers_a = n_prefers_a / n
    tie_rate = n_ties / n
    bias = abs(p_prefers_a - 0.5)
    pair_consistency = _pair_consistency(records)
    swap_agreement = _swap_agreement(records)

    bias_ok = bias < threshold
    tie_ok = tie_rate <= max_tie_rate
    consistency_ok = pair_consistency == pair_consistency and pair_consistency >= min_pair_consistency
    swap_ok = swap_agreement == swap_agreement and swap_agreement >= min_swap_agreement
    passed = bias_ok and tie_ok and consistency_ok and swap_ok

    failures: list[str] = []
    if not bias_ok:
        failures.append(f"|P(prefer_a)-0.5|={bias:.3f} >= {threshold}")
    if not tie_ok:
        failures.append(f"tie_rate={tie_rate:.3f} > {max_tie_rate}")
    if not consistency_ok:
        failures.append(f"within_pair_consistency={pair_consistency:.3f} < {min_pair_consistency}")
    if not swap_ok:
        failures.append(f"ab_ba_agreement={swap_agreement:.3f} < {min_swap_agreement}")

    return {
        "passed": passed,
        "p_prefers_a": round(p_prefers_a, 4),
        "bias_magnitude": round(bias, 4),
        "tie_rate": round(tie_rate, 4),
        "within_pair_consistency": round(pair_consistency, 4) if pair_consistency == pair_consistency else float("nan"),
        "ab_ba_agreement": round(swap_agreement, 4) if swap_agreement == swap_agreement else float("nan"),
        "threshold": threshold,
        "n": n,
        "message": (
            "PASS: pilot gate passed"
            if passed
            else "FAIL: " + "; ".join(failures)
        ),
    }
