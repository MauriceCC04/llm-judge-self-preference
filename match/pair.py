"""match/pair.py — deterministic-score-based greedy pairing."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from generate.constants import MATCH_FEATURE_WEIGHTS
from match.features import load_match_features, weighted_feature_distance


class ScoringError(RuntimeError):
    """Raised when the deterministic scorer cannot produce a study-valid score."""


class MatchingCoverageError(RuntimeError):
    """Raised when matching falls below the configured coverage floor."""


def _score_bin(score: float) -> int:
    return int(math.floor(float(score) + 0.5))


def greedy_pair(
    plans: list[dict[str, Any]],
    *,
    tolerance: float = 1.0,
    feature_weights: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    feature_weights = feature_weights or MATCH_FEATURE_WEIGHTS
    llm_plans = [p.copy() for p in plans if p["arm"] == "llm"]
    prog_plans = [p.copy() for p in plans if p["arm"] == "programmatic"]

    for p in llm_plans + prog_plans:
        p["score_bin"] = _score_bin(float(p["score"]))

    llm_sorted = sorted(llm_plans, key=lambda p: (p["fixture_id"], p["score_bin"], p["score"]))
    prog_sorted = sorted(prog_plans, key=lambda p: (p["fixture_id"], p["score_bin"], p["score"]))

    used_llm: set[str] = set()
    used_prog: set[str] = set()
    pairs: list[dict[str, Any]] = []
    pair_idx = 0

    for lp in llm_sorted:
        if lp["plan_id"] in used_llm:
            continue

        best_prog = None
        best_gap = float("inf")
        best_distance = float("inf")

        for pp in prog_sorted:
            if pp["plan_id"] in used_prog:
                continue
            if pp["fixture_id"] != lp["fixture_id"]:
                continue
            if pp["score_bin"] != lp["score_bin"]:
                continue

            gap = abs(float(lp["score"]) - float(pp["score"]))
            if gap > tolerance:
                continue
            distance = weighted_feature_distance(lp, pp, weights=feature_weights)
            if (distance, gap) < (best_distance, best_gap):
                best_gap = gap
                best_distance = distance
                best_prog = pp

        if best_prog is not None:
            pairs.append(
                {
                    "pair_id": f"pair_{pair_idx:04d}",
                    "plan_a_id": lp["plan_id"],
                    "plan_b_id": best_prog["plan_id"],
                    "fixture_id": lp["fixture_id"],
                    "score_gap": round(best_gap, 3),
                    "match_distance": round(best_distance, 3),
                    "score_a": lp["score"],
                    "score_b": best_prog["score"],
                    "source_model_a": lp.get("source_model"),
                    "source_model_b": best_prog.get("source_model"),
                    "explainer_model_a": lp.get("explainer_model"),
                    "explainer_model_b": best_prog.get("explainer_model"),
                    "actual_explainer_model_a": lp.get("actual_explainer_model"),
                    "actual_explainer_model_b": best_prog.get("actual_explainer_model"),
                    "explainer_verified_a": lp.get("explainer_model_verified"),
                    "explainer_verified_b": best_prog.get("explainer_model_verified"),
                    "score_bin": lp["score_bin"],
                    "score_bin_a": lp["score_bin"],
                    "score_bin_b": best_prog["score_bin"],
                    "match_rule": "same_fixture_same_score_bin_within_tolerance_min_weighted_distance",
                    "arm_a": "llm",
                    "arm_b": "programmatic",
                }
            )
            used_llm.add(lp["plan_id"])
            used_prog.add(best_prog["plan_id"])
            pair_idx += 1

    return pairs


def score_plan(
    plan_path: Path,
    rollups_path: Path | None = None,
    *,
    strict: bool = True,
) -> float:
    try:
        import json as _json
        from trailtraining.llm.constraints import ConstraintConfig, evaluate_training_plan_quality

        plan_obj = _json.loads(Path(plan_path).read_text())
        rollups = (
            _json.loads(Path(rollups_path).read_text())
            if rollups_path and Path(rollups_path).exists()
            else None
        )
        cfg = ConstraintConfig(min_signal_ids_per_day=0)
        report = evaluate_training_plan_quality(plan_obj, rollups, cfg)
        return float(report["score"])
    except Exception as exc:
        if strict:
            raise ScoringError(
                f"Deterministic scorer failed for {plan_path}: {type(exc).__name__}: {exc}"
            ) from exc
        return _heuristic_score(plan_path)


def _heuristic_score(plan_path: Path) -> float:
    import hashlib

    try:
        data = json.loads(Path(plan_path).read_text())
    except Exception:
        return 0.0

    days = (data.get("plan") or {}).get("days") or []
    if not days:
        return 0.0

    total_min = sum(d.get("duration_minutes") or 0 for d in days)
    n_active = sum(1 for d in days if not d.get("is_rest_day", False))
    has_long = any(d.get("session_type") == "long" for d in days)

    score = min(total_min / 50.0, 10.0)
    score += (n_active / max(len(days), 1)) * 3
    score += 1.0 if has_long else 0.0

    pid = str(plan_path.stem)
    jitter = int(hashlib.md5(pid.encode()).hexdigest()[:4], 16) / 65535 * 0.01
    return round(score + jitter, 4)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def _build_matching_audit(
    *,
    pairs: list[dict[str, Any]],
    target_pairs: int,
) -> dict[str, Any]:
    n = len(pairs)
    gaps = [float(p["score_gap"]) for p in pairs]
    mean_gap = sum(gaps) / n if n else float("nan")
    by_fixture: dict[str, int] = {}
    for p in pairs:
        fid = str(p["fixture_id"])
        by_fixture[fid] = by_fixture.get(fid, 0) + 1
    coverage_ratio = (n / target_pairs) if target_pairs else float("nan")
    coverage_ok = bool(target_pairs and n >= int(target_pairs * 0.8))
    return {
        "n_pairs": n,
        "target_pairs": target_pairs,
        "coverage_ratio": round(coverage_ratio, 4) if coverage_ratio == coverage_ratio else float("nan"),
        "coverage_ok": coverage_ok,
        "mean_score_gap": round(mean_gap, 4) if mean_gap == mean_gap else float("nan"),
        "pairs_by_fixture": dict(sorted(by_fixture.items())),
        "match_rule": "same_fixture_same_score_bin_within_tolerance_min_weighted_distance",
    }


def build_matched_pairs(
    plans_dir: Path,
    provenance_dir: Path,
    fixtures_dir: Path,
    output_path: Path,
    *,
    tolerance: float = 1.0,
    target_pairs: int = 250,
    strict_scoring: bool = True,
    scoring_failures_path: Path | None = None,
    feature_weights: dict[str, float] | None = None,
    fail_below_target_ratio: float | None = None,
) -> list[dict[str, Any]]:
    from generate.provenance import PlanProvenance

    plan_records: list[dict[str, Any]] = []
    scoring_failures: list[dict[str, Any]] = []

    for prov_path in sorted(provenance_dir.glob("*.provenance.json")):
        prov = PlanProvenance.model_validate_json(prov_path.read_text())
        plan_path = Path(prov.plan_path)
        if not plan_path.exists():
            continue

        fixture_rollups = fixtures_dir / prov.fixture_id / "combined_rollups.json"
        rollups_path = fixture_rollups if fixture_rollups.exists() else None

        try:
            score = score_plan(plan_path, rollups_path, strict=strict_scoring)
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

        prov.deterministic_score = score
        prov_path.write_text(prov.model_dump_json(indent=2), encoding="utf-8")

        plan_records.append(
            {
                "plan_id": prov.plan_id,
                "fixture_id": prov.fixture_id,
                "arm": prov.arm,
                "score": score,
                "score_bin": _score_bin(score),
                "plan_path": str(plan_path),
                "source_model": prov.source_model,
                "explainer_model": prov.explainer_model,
                "actual_explainer_model": prov.actual_explainer_model,
                "explainer_model_verified": prov.explainer_model_verified,
                **load_match_features(plan_path),
            }
        )

    if scoring_failures:
        fail_path = scoring_failures_path or output_path.with_name("scoring_failures.json")
        _write_json(fail_path, scoring_failures)
        if strict_scoring:
            raise ScoringError(
                f"{len(scoring_failures)} plans failed deterministic scoring; see {fail_path}"
            )

    pairs = greedy_pair(plan_records, tolerance=tolerance, feature_weights=feature_weights)
    audit = _build_matching_audit(pairs=pairs, target_pairs=target_pairs)
    _print_audit(audit)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(pairs, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    _write_json(output_path.with_name("matching_audit.json"), audit)

    if fail_below_target_ratio is not None and audit["coverage_ratio"] < fail_below_target_ratio:
        raise MatchingCoverageError(
            f"Matching coverage {audit['coverage_ratio']:.3f} is below required ratio {fail_below_target_ratio:.3f}. "
            f"See {output_path.with_name('matching_audit.json')}"
        )

    print(f"\n[Saved] {output_path}  ({len(pairs)} pairs)")
    return pairs


def _print_audit(audit: dict[str, Any]) -> None:
    print("\n── Matching audit ──────────────────────────────")
    print(f"  Pairs yielded:    {audit['n_pairs']}  (target: {audit['target_pairs']})")
    print(f"  Mean score gap:   {audit['mean_score_gap']:.3f}")
    print(f"  Coverage ratio:   {audit['coverage_ratio']:.3f}")
    print(f"  Coverage OK:      {audit['coverage_ok']}")
    for fid, cnt in sorted((audit.get('pairs_by_fixture') or {}).items()):
        print(f"    {fid}: {cnt} pairs")
    if not audit["coverage_ok"]:
        print("\n  ⚠ Coverage below 80% — retune sampler priors or generate more programmatic plans.")
    print("────────────────────────────────────────────────\n")
