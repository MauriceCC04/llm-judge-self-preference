"""match/pair.py — source-neutral structural-score-based greedy pairing."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from generate.constants import MATCH_FEATURE_WEIGHTS
from match.features import extract_match_features, weighted_feature_distance
from match.filtering import filter_plan_records
from match.structural_score import STRUCTURAL_SCORE_VERSION, score_structural_plan


class ScoringError(RuntimeError):
    """Raised when a scorer cannot produce a study-valid score."""


class MatchingCoverageError(RuntimeError):
    """Raised when matching falls below the configured coverage floor."""


def _source_family(source_model: Any) -> str:
    text = str(source_model or "").lower()
    if "qwen" in text:
        return "qwen"
    if "gemma" in text:
        return "gemma"
    if "programmatic" in text:
        return "programmatic"
    return "unknown"


def _round_robin_by_source_family(llm_plans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for plan in llm_plans:
        buckets.setdefault(_source_family(plan.get("source_model")), []).append(plan)
    if len(buckets) <= 1:
        return sorted(llm_plans, key=lambda p: (p["fixture_id"], p["score"], p["plan_id"]))
    for family in list(buckets):
        buckets[family] = sorted(buckets[family], key=lambda p: (p["fixture_id"], p["score"], p["plan_id"]))
    families = sorted(buckets)
    out: list[dict[str, Any]] = []
    while any(buckets.values()):
        for family in families:
            if buckets[family]:
                out.append(buckets[family].pop(0))
    return out


def _score_bin(score: float) -> int:
    return int(math.floor(float(score) + 0.5))


def _normalize_temperature_value(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), 6)
    except Exception:
        return None


def _condition_key(plan_record: dict[str, Any]) -> tuple[str, float | None, float | None]:
    return (
        str(plan_record.get("generation_condition") or ""),
        _normalize_temperature_value(plan_record.get("source_temperature")),
        _normalize_temperature_value(plan_record.get("explainer_temperature")),
    )


def _assert_single_generation_condition(plan_records: list[dict[str, Any]]) -> None:
    by_arm: dict[str, set[tuple[str, float | None, float | None]]] = {}
    for record in plan_records:
        arm = str(record.get("arm") or "")
        by_arm.setdefault(arm, set()).add(_condition_key(record))

    problematic = {arm: sorted(list(keys)) for arm, keys in by_arm.items() if len(keys) > 1}
    if problematic:
        raise ValueError(
            "Mixed generation conditions detected in the same plans/provenance directory. "
            "Keep each temperature condition in its own artifact directory, or set "
            "allow_mixed_generation_conditions=True explicitly. "
            f"Observed conditions: {problematic}"
        )


def _old_quality_score(
    plan_path: Path,
    rollups_path: Path | None = None,
    *,
    strict: bool = True,
) -> float | None:
    """Return the legacy TrailTraining quality score as a diagnostic only."""
    try:
        import json as _json
        from trailtraining.llm.constraints import ConstraintConfig, evaluate_training_plan_quality

        plan_obj = _json.loads(Path(plan_path).read_text(encoding="utf-8"))
        rollups = (
            _json.loads(Path(rollups_path).read_text(encoding="utf-8"))
            if rollups_path and Path(rollups_path).exists()
            else None
        )
        cfg = ConstraintConfig(min_signal_ids_per_day=0)
        report = evaluate_training_plan_quality(plan_obj, rollups, cfg)
        return float(report["score"])
    except Exception as exc:
        if strict:
            raise ScoringError(
                f"Legacy deterministic scorer failed for {plan_path}: {type(exc).__name__}: {exc}"
            ) from exc
        return None


def score_plan(
    plan_path: Path,
    rollups_path: Path | None = None,
    *,
    strict: bool = True,
    provenance_path: Path | None = None,
) -> float:
    """Primary matching score: source-neutral structural score.

    This replaces the legacy TrailTraining quality score for matching. The legacy
    score may still be computed separately as ``old_quality_score`` for audit and
    sensitivity reporting, but must not control primary pair construction.
    """
    try:
        return float(score_structural_plan(plan_path, provenance_path=provenance_path, rollups_path=rollups_path).score)
    except Exception as exc:
        if strict:
            raise ScoringError(
                f"Structural scorer failed for {plan_path}: {type(exc).__name__}: {exc}"
            ) from exc
        return _heuristic_score(plan_path)


def _heuristic_score(plan_path: Path) -> float:
    """Last-resort structural fallback used only when strict=False."""
    try:
        data = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    except Exception:
        return 0.0

    days = (data.get("plan") or {}).get("days") or []
    if not days:
        return 0.0

    total_min = sum(float(d.get("duration_minutes") or 0) for d in days if isinstance(d, dict))
    n_active = sum(1 for d in days if isinstance(d, dict) and not d.get("is_rest_day", False))
    n_rest = sum(1 for d in days if isinstance(d, dict) and d.get("is_rest_day", False))
    n_hard = sum(1 for d in days if isinstance(d, dict) and d.get("is_hard_day", False))
    n_long = sum(1 for d in days if isinstance(d, dict) and str(d.get("session_type") or "").lower() == "long")

    score = 70.0
    score -= abs(len(days) - 7) * 8.0
    score += min(total_min / 20.0, 15.0)
    score += min(n_active, 5) * 2.0
    score += min(n_rest, 3) * 1.5
    score += min(n_hard, 2) * 2.0
    score += 4.0 if n_long >= 1 else -8.0
    return round(max(0.0, min(100.0, score)), 3)


def _same_required_cell(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Return True when two records are eligible for same-cell causal matching."""
    if right.get("fixture_id") != left.get("fixture_id"):
        return False
    if right.get("athlete_band") != left.get("athlete_band"):
        return False
    if right.get("readiness") != left.get("readiness"):
        return False
    if right.get("recovery_capability") != left.get("recovery_capability"):
        return False
    if right.get("race_phase") != left.get("race_phase"):
        return False
    if int(right.get("plan_days") or 0) != int(left.get("plan_days") or 0):
        return False
    if str(right.get("style") or "") != str(left.get("style") or ""):
        return False
    return True


def _candidate_adjacency(
    llm_plans: list[dict[str, Any]],
    prog_plans: list[dict[str, Any]],
    *,
    tolerance: float,
    feature_weights: dict[str, float],
    require_same_score_bin: bool,
) -> tuple[list[list[int]], dict[tuple[int, int], tuple[float, float]]]:
    adjacency: list[list[int]] = [[] for _ in llm_plans]
    edge_meta: dict[tuple[int, int], tuple[float, float]] = {}
    for i, lp in enumerate(llm_plans):
        ranked: list[tuple[float, float, str, int]] = []
        for j, pp in enumerate(prog_plans):
            if not _same_required_cell(lp, pp):
                continue
            if require_same_score_bin and pp["score_bin"] != lp["score_bin"]:
                continue
            gap = abs(float(lp["score"]) - float(pp["score"]))
            if gap > tolerance:
                continue
            distance = weighted_feature_distance(lp, pp, weights=feature_weights)
            edge_meta[(i, j)] = (gap, distance)
            ranked.append((gap, distance, str(pp.get("plan_id") or ""), j))
        adjacency[i] = [j for _, _, _, j in sorted(ranked)]
    return adjacency, edge_meta


def _hopcroft_karp(adjacency: list[list[int]], n_right: int) -> dict[int, int]:
    """Maximum-cardinality bipartite matching, left index -> right index.

    The previous matcher was locally greedy. With dense same-cell candidate sets,
    a locally best early pair can consume the only viable programmatic plan for a
    later LLM plan, leaving study-valid matches unused. Hopcroft-Karp enforces
    cardinality first; sorted adjacency gives deterministic low-gap/low-distance
    preference among equal-cardinality solutions.
    """
    from collections import deque

    n_left = len(adjacency)
    pair_u = [-1] * n_left
    pair_v = [-1] * n_right
    dist = [0] * n_left

    def bfs() -> bool:
        q: deque[int] = deque()
        found = False
        for u in range(n_left):
            if pair_u[u] == -1:
                dist[u] = 0
                q.append(u)
            else:
                dist[u] = -1
        while q:
            u = q.popleft()
            for v in adjacency[u]:
                u2 = pair_v[v]
                if u2 == -1:
                    found = True
                elif dist[u2] == -1:
                    dist[u2] = dist[u] + 1
                    q.append(u2)
        return found

    def dfs(u: int) -> bool:
        for v in adjacency[u]:
            u2 = pair_v[v]
            if u2 == -1 or (dist[u2] == dist[u] + 1 and dfs(u2)):
                pair_u[u] = v
                pair_v[v] = u
                return True
        dist[u] = -1
        return False

    while bfs():
        for u in range(n_left):
            if pair_u[u] == -1:
                dfs(u)
    return {u: v for u, v in enumerate(pair_u) if v != -1}


def greedy_pair(
    plans: list[dict[str, Any]],
    *,
    tolerance: float = 2.0,
    feature_weights: dict[str, float] | None = None,
    require_same_score_bin: bool = False,
) -> list[dict[str, Any]]:
    """Build same-cell LLM-vs-programmatic pairs using structural scores.

    Historical name kept for API compatibility. The implementation now uses a
    maximum-cardinality bipartite matcher, not local greedy search. This is the
    correct objective for the full-study gate: maximize valid same-cell matches
    under structural tolerance, then use structural feature distance only to
    order candidate edges deterministically.
    """
    feature_weights = feature_weights or MATCH_FEATURE_WEIGHTS
    llm_plans = [p.copy() for p in plans if p["arm"] == "llm"]
    prog_plans = [p.copy() for p in plans if p["arm"] == "programmatic"]

    for p in llm_plans + prog_plans:
        p["score_bin"] = _score_bin(float(p["score"]))

    llm_sorted = _round_robin_by_source_family(llm_plans)
    prog_sorted = sorted(prog_plans, key=lambda p: (p["fixture_id"], p["score"], p["plan_id"]))

    adjacency, edge_meta = _candidate_adjacency(
        llm_sorted,
        prog_sorted,
        tolerance=tolerance,
        feature_weights=feature_weights,
        require_same_score_bin=require_same_score_bin,
    )
    matching = _hopcroft_karp(adjacency, len(prog_sorted))

    ordered = sorted(
        matching.items(),
        key=lambda uv: (
            str(llm_sorted[uv[0]].get("fixture_id") or ""),
            _source_family(llm_sorted[uv[0]].get("source_model")),
            float(edge_meta[(uv[0], uv[1])][0]),
            float(edge_meta[(uv[0], uv[1])][1]),
            str(llm_sorted[uv[0]].get("plan_id") or ""),
        ),
    )
    pairs: list[dict[str, Any]] = []
    for pair_idx, (llm_idx, prog_idx) in enumerate(ordered):
        gap, distance = edge_meta[(llm_idx, prog_idx)]
        pairs.append(_make_pair_record(pair_idx, llm_sorted[llm_idx], prog_sorted[prog_idx], gap, distance, feature_weights))
    return pairs


def _make_pair_record(
    pair_idx: int,
    lp: dict[str, Any],
    best_prog: dict[str, Any],
    best_gap: float,
    best_distance: float,
    feature_weights: dict[str, float],
) -> dict[str, Any]:
    return {
        "pair_id": f"pair_{pair_idx:04d}",
        "plan_a_id": lp["plan_id"],
        "plan_b_id": best_prog["plan_id"],
        "fixture_id": lp["fixture_id"],
        "athlete_band": lp.get("athlete_band"),
        "readiness": lp.get("readiness"),
        "recovery_capability": lp.get("recovery_capability"),
        "race_phase": lp.get("race_phase"),
        "plan_days": lp.get("plan_days"),
        "score_gap": round(best_gap, 3),
        "structural_score_gap": round(best_gap, 3),
        "match_distance": round(best_distance, 3),
        "score_a": lp["score"],
        "score_b": best_prog["score"],
        "structural_score_a": lp["score"],
        "structural_score_b": best_prog["score"],
        "structural_score_version": lp.get("structural_score_version") or STRUCTURAL_SCORE_VERSION,
        "old_quality_score_a": lp.get("old_quality_score"),
        "old_quality_score_b": best_prog.get("old_quality_score"),
        "source_model_a": lp.get("source_model"),
        "source_model_b": best_prog.get("source_model"),
        "explainer_model_a": lp.get("explainer_model"),
        "explainer_model_b": best_prog.get("explainer_model"),
        "actual_explainer_model_a": lp.get("actual_explainer_model"),
        "actual_explainer_model_b": best_prog.get("actual_explainer_model"),
        "explainer_verified_a": lp.get("explainer_model_verified"),
        "explainer_verified_b": best_prog.get("explainer_model_verified"),
        "source_temperature_a": lp.get("source_temperature"),
        "source_temperature_b": best_prog.get("source_temperature"),
        "explainer_temperature_a": lp.get("explainer_temperature"),
        "explainer_temperature_b": best_prog.get("explainer_temperature"),
        "generation_condition_a": lp.get("generation_condition"),
        "generation_condition_b": best_prog.get("generation_condition"),
        "score_bin": lp["score_bin"],
        "score_bin_a": lp["score_bin"],
        "score_bin_b": best_prog["score_bin"],
        "match_rule": "same_full_cell_same_plan_days_structural_score_tolerance_max_cardinality_weighted_structural_distance",
        "arm_a": "llm",
        "arm_b": "programmatic",
        "feature_gaps": _feature_gaps(lp, best_prog, feature_weights),
    }


def _feature_gaps(left: dict[str, Any], right: dict[str, Any], weights: dict[str, float]) -> dict[str, float]:
    return {
        key: round(abs(float(left.get(key, 0.0)) - float(right.get(key, 0.0))), 3)
        for key in sorted(weights)
    }


def _select_balanced_target_pairs(pairs: list[dict[str, Any]], target_pairs: int) -> list[dict[str, Any]]:
    """Cap to target while preserving Qwen/Gemma balance where possible."""
    if target_pairs <= 0 or len(pairs) <= target_pairs:
        return pairs
    buckets: dict[str, list[dict[str, Any]]] = {}
    for pair in pairs:
        buckets.setdefault(_source_family(pair.get("source_model_a")), []).append(pair)
    for family in list(buckets):
        buckets[family] = sorted(
            buckets[family],
            key=lambda p: (float(p.get("structural_score_gap", p.get("score_gap", 999))), float(p.get("match_distance", 999)), p.get("pair_id", "")),
        )
    families = sorted(buckets)
    selected: list[dict[str, Any]] = []
    while len(selected) < target_pairs and any(buckets.values()):
        for family in families:
            if len(selected) >= target_pairs:
                break
            if buckets[family]:
                selected.append(buckets[family].pop(0))
    selected = selected[:target_pairs]
    for idx, pair in enumerate(selected):
        pair["pair_id"] = f"pair_{idx:04d}"
    return selected


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def _build_matching_audit(
    *,
    pairs: list[dict[str, Any]],
    plan_records: list[dict[str, Any]],
    target_pairs: int,
) -> dict[str, Any]:
    n = len(pairs)
    gaps = [float(p["structural_score_gap"]) for p in pairs]
    mean_gap = sum(gaps) / n if n else float("nan")
    max_gap = max(gaps) if gaps else float("nan")
    p95_gap = sorted(gaps)[int(0.95 * (n - 1))] if n else float("nan")

    by_fixture: dict[str, int] = {}
    by_athlete_band: dict[str, int] = {}
    by_source_family: dict[str, int] = {}
    llm_conditions: set[str] = set()
    programmatic_conditions: set[str] = set()

    for p in pairs:
        fid = str(p["fixture_id"])
        by_fixture[fid] = by_fixture.get(fid, 0) + 1
        band = str(p.get("athlete_band") or "")
        if band:
            by_athlete_band[band] = by_athlete_band.get(band, 0) + 1
        family = _model_family(p.get("source_model_a"))
        by_source_family[family] = by_source_family.get(family, 0) + 1
        llm_conditions.add(str(p.get("generation_condition_a") or ""))
        programmatic_conditions.add(str(p.get("generation_condition_b") or ""))

    coverage_ratio = (n / target_pairs) if target_pairs else float("nan")
    coverage_ok = bool(target_pairs and n >= min(target_pairs, 250))
    score_values_by_arm: dict[str, list[float]] = {}
    for record in plan_records:
        score_values_by_arm.setdefault(str(record.get("arm") or ""), []).append(float(record.get("score") or 0.0))

    return {
        "n_pairs": n,
        "target_pairs": target_pairs,
        "coverage_ratio": round(coverage_ratio, 4) if coverage_ratio == coverage_ratio else float("nan"),
        "coverage_ok": coverage_ok,
        "mean_structural_score_gap": round(mean_gap, 4) if mean_gap == mean_gap else float("nan"),
        "max_structural_score_gap": round(max_gap, 4) if max_gap == max_gap else float("nan"),
        "p95_structural_score_gap": round(p95_gap, 4) if p95_gap == p95_gap else float("nan"),
        "pairs_by_fixture": dict(sorted(by_fixture.items())),
        "pairs_by_athlete_band": dict(sorted(by_athlete_band.items())),
        "pairs_by_source_family": dict(sorted(by_source_family.items())),
        "llm_generation_conditions": sorted(x for x in llm_conditions if x),
        "programmatic_generation_conditions": sorted(x for x in programmatic_conditions if x),
        "structural_score_version": STRUCTURAL_SCORE_VERSION,
        "primary_matching_score": "source_neutral_structural_score",
        "legacy_quality_score_role": "diagnostic_only",
        "presentation_fields_excluded": True,
        "match_rule": "same_full_cell_same_plan_days_structural_score_tolerance_min_weighted_structural_distance",
        "score_distribution_by_arm": {
            arm: _distribution(values) for arm, values in sorted(score_values_by_arm.items())
        },
    }


def _distribution(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "min": round(ordered[0], 3),
        "p25": round(ordered[int(0.25 * (len(ordered) - 1))], 3),
        "median": round(ordered[int(0.50 * (len(ordered) - 1))], 3),
        "p75": round(ordered[int(0.75 * (len(ordered) - 1))], 3),
        "max": round(ordered[-1], 3),
    }


def _model_family(model: Any) -> str:
    s = str(model or "").lower()
    if "qwen" in s:
        return "qwen"
    if "gemma" in s:
        return "gemma"
    if not s or s == "none":
        return "programmatic"
    return "other"


def build_matched_pairs(
    plans_dir: Path,
    provenance_dir: Path,
    fixtures_dir: Path,
    output_path: Path,
    *,
    tolerance: float = 2.0,
    target_pairs: int = 256,
    strict_scoring: bool = True,
    scoring_failures_path: Path | None = None,
    feature_weights: dict[str, float] | None = None,
    fail_below_target_ratio: float | None = None,
    allow_mixed_generation_conditions: bool = False,
    compute_old_quality_score: bool = False,
    require_same_score_bin: bool = False,
) -> list[dict[str, Any]]:
    from generate.provenance import PlanProvenance

    plan_records: list[dict[str, Any]] = []
    scoring_failures: list[dict[str, Any]] = []

    for prov_path in sorted(provenance_dir.glob("*.provenance.json")):
        prov = PlanProvenance.model_validate_json(prov_path.read_text(encoding="utf-8"))
        plan_path = Path(prov.plan_path)
        if not plan_path.exists():
            candidate = plans_dir / f"{prov.plan_id}.json"
            if candidate.exists():
                plan_path = candidate
            else:
                continue

        fixture_rollups = fixtures_dir / prov.fixture_id / "combined_rollups.json"
        rollups_path = fixture_rollups if fixture_rollups.exists() else None

        try:
            structural_report = score_structural_plan(plan_path, provenance_path=prov_path, rollups_path=rollups_path)
            score = float(structural_report.score)
        except Exception as exc:
            scoring_failures.append(
                {
                    "plan_id": prov.plan_id,
                    "fixture_id": prov.fixture_id,
                    "arm": prov.arm,
                    "plan_path": str(plan_path),
                    "rollups_path": str(rollups_path) if rollups_path else None,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            continue

        old_quality_score = None
        if compute_old_quality_score:
            old_quality_score = _old_quality_score(plan_path, rollups_path, strict=False)

        prov.deterministic_score = score
        prov_path.write_text(prov.model_dump_json(indent=2), encoding="utf-8")

        plan_obj = json.loads(plan_path.read_text(encoding="utf-8"))
        plan_records.append(
            {
                "plan_id": prov.plan_id,
                "fixture_id": prov.fixture_id,
                "athlete_band": prov.athlete_band,
                "readiness": prov.readiness,
                "recovery_capability": prov.recovery_capability,
                "race_phase": prov.race_phase,
                "plan_days": prov.plan_days or len((plan_obj.get("plan") or {}).get("days") or []),
                "style": prov.style or str((plan_obj.get("meta") or {}).get("style") or ""),
                "arm": prov.arm,
                "score": score,
                "structural_score": score,
                "structural_score_version": structural_report.score_version,
                "structural_violations": structural_report.violations,
                "old_quality_score": old_quality_score,
                "score_bin": _score_bin(score),
                "plan_path": str(plan_path),
                "source_model": prov.source_model,
                "explainer_model": prov.explainer_model,
                "actual_explainer_model": prov.actual_explainer_model,
                "explainer_model_verified": prov.explainer_model_verified,
                "source_temperature": prov.source_temperature,
                "explainer_temperature": prov.explainer_temperature,
                "generation_condition": prov.generation_condition,
                "plan_obj": plan_obj,
                **extract_match_features(plan_obj),
            }
        )

    if scoring_failures:
        fail_path = scoring_failures_path or output_path.with_name("scoring_failures.json")
        _write_json(fail_path, scoring_failures)
        if strict_scoring:
            raise ScoringError(
                f"{len(scoring_failures)} plans failed structural scoring; see {fail_path}"
            )

    prefilter = filter_plan_records(plan_records)
    _write_json(output_path.with_name("matching_prefilter_audit.json"), prefilter["audit"])
    plan_records = [
        {k: v for k, v in record.items() if k != "plan_obj"}
        for record in prefilter["kept_records"]
    ]

    if not allow_mixed_generation_conditions:
        _assert_single_generation_condition(plan_records)

    pairs_all = greedy_pair(
        plan_records,
        tolerance=tolerance,
        feature_weights=feature_weights,
        require_same_score_bin=require_same_score_bin,
    )
    pairs = _select_balanced_target_pairs(pairs_all, target_pairs)
    audit = _build_matching_audit(pairs=pairs, plan_records=plan_records, target_pairs=target_pairs)
    audit["n_candidate_pairs_before_target_cap"] = len(pairs_all)
    audit["target_cap_applied"] = len(pairs_all) > len(pairs)
    _print_audit(audit)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(pairs, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    _write_json(output_path.with_name("matching_audit.json"), audit)

    if fail_below_target_ratio is not None and audit["coverage_ratio"] < fail_below_target_ratio:
        raise MatchingCoverageError(
            f"Matching coverage {audit['coverage_ratio']:.3f} is below required ratio "
            f"{fail_below_target_ratio:.3f}. See {output_path.with_name('matching_audit.json')}"
        )

    print(f"\n[Saved] {output_path}  ({len(pairs)} pairs)")
    return pairs


def _print_audit(audit: dict[str, Any]) -> None:
    print("\n-- Matching audit --------------------------------")
    print(f"  Pairs yielded:    {audit['n_pairs']}  (target: {audit['target_pairs']})")
    print(f"  Mean score gap:   {audit['mean_structural_score_gap']:.3f}")
    print(f"  P95 score gap:    {audit['p95_structural_score_gap']:.3f}")
    print(f"  Max score gap:    {audit['max_structural_score_gap']:.3f}")
    print(f"  Coverage ratio:   {audit['coverage_ratio']:.3f}")
    print(f"  Coverage OK:      {audit['coverage_ok']}")
    print(f"  Score version:    {audit['structural_score_version']}")
    for fid, cnt in sorted((audit.get('pairs_by_fixture') or {}).items()):
        print(f"    {fid}: {cnt} pairs")
    if audit.get("pairs_by_source_family"):
        print(f"  Source families:  {audit['pairs_by_source_family']}")
    if audit.get("llm_generation_conditions"):
        print(f"  LLM conditions:   {audit['llm_generation_conditions']}")
    if audit.get("programmatic_generation_conditions"):
        print(f"  Prog conditions:  {audit['programmatic_generation_conditions']}")
    if not audit["coverage_ok"]:
        print("\n  WARNING: coverage below full-study gate. Do not launch full judging.")
    print("--------------------------------------------------\n")
