"""match/pair.py — deterministic-score-based greedy pairing."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def greedy_pair(
    plans: list[dict[str, Any]],
    *,
    tolerance: float = 1.0,
) -> list[dict[str, Any]]:
    """Pair LLM plans with programmatic plans within |Δscore| <= tolerance.

    Each entry in *plans* must have:
        plan_id, fixture_id, score (float), arm ("llm" | "programmatic")

    Returns a list of pair dicts:
        pair_id, plan_a_id, plan_b_id, fixture_id, score_gap, score_a, score_b
    """
    llm_plans = [p for p in plans if p["arm"] == "llm"]
    prog_plans = [p for p in plans if p["arm"] == "programmatic"]

    llm_sorted = sorted(llm_plans, key=lambda p: (p["fixture_id"], p["score"]))
    prog_sorted = sorted(prog_plans, key=lambda p: (p["fixture_id"], p["score"]))

    used_llm: set[str] = set()
    used_prog: set[str] = set()
    pairs: list[dict[str, Any]] = []
    pair_idx = 0

    for lp in llm_sorted:
        if lp["plan_id"] in used_llm:
            continue
        best_prog = None
        best_gap = float("inf")
        for pp in prog_sorted:
            if pp["plan_id"] in used_prog:
                continue
            if pp["fixture_id"] != lp["fixture_id"]:
                continue
            gap = abs(lp["score"] - pp["score"])
            if gap <= tolerance and gap < best_gap:
                best_gap = gap
                best_prog = pp

        if best_prog is not None:
            pairs.append({
                "pair_id": f"pair_{pair_idx:04d}",
                "plan_a_id": lp["plan_id"],
                "plan_b_id": best_prog["plan_id"],
                "fixture_id": lp["fixture_id"],
                "score_gap": round(best_gap, 3),
                "score_a": lp["score"],
                "score_b": best_prog["score"],
                "arm_a": "llm",
                "arm_b": "programmatic",
            })
            used_llm.add(lp["plan_id"])
            used_prog.add(best_prog["plan_id"])
            pair_idx += 1

    return pairs


def score_plan(plan_path: Path, rollups_path: Path | None = None) -> float:
    """Return a quality score for *plan_path*.

    Calls ``evaluate_training_plan_quality`` from trailtraining when available.
    Falls back to ``_heuristic_score`` so matching works offline without a GPU
    — useful for integration tests and dry-runs.
    """
    try:
        import json as _json
        from trailtraining.llm.constraints import (
            ConstraintConfig,
            evaluate_training_plan_quality,
        )

        plan_obj = _json.loads(Path(plan_path).read_text())
        rollups = (
            _json.loads(Path(rollups_path).read_text())
            if rollups_path and Path(rollups_path).exists()
            else None
        )
        cfg = ConstraintConfig(min_signal_ids_per_day=0)
        report = evaluate_training_plan_quality(plan_obj, rollups, cfg)
        return float(report["score"])
    except Exception:
        return _heuristic_score(plan_path)


def _heuristic_score(plan_path: Path) -> float:
    """Deterministic structural heuristic score (no LLM required).

    Returns a float in roughly the same range as trailtraining's quality score
    so greedy pairing still works during offline tests.

    Signals used:
    - Total scheduled minutes (normalised to 0–10)
    - Fraction of days that are non-rest (plan utilisation)
    - Presence of a long run day (+1 bonus)
    """
    import json
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

    score = min(total_min / 50.0, 10.0)        # normalise minutes to [0,10]
    score += (n_active / max(len(days), 1)) * 3  # utilisation bonus [0,3]
    score += 1.0 if has_long else 0.0           # long-run bonus

    # Add tiny deterministic jitter from plan_id so identical plans get
    # different scores, preventing degenerate tie-breaking.
    pid = str(plan_path.stem)
    jitter = int(hashlib.md5(pid.encode()).hexdigest()[:4], 16) / 65535 * 0.01
    return round(score + jitter, 4)


def build_matched_pairs(
    plans_dir: Path,
    provenance_dir: Path,
    fixtures_dir: Path,
    output_path: Path,
    *,
    tolerance: float = 1.0,
    target_pairs: int = 250,
) -> list[dict[str, Any]]:
    from generate.provenance import PlanProvenance

    plan_records: list[dict[str, Any]] = []
    for prov_path in sorted(provenance_dir.glob("*.provenance.json")):
        prov = PlanProvenance.model_validate_json(prov_path.read_text())
        plan_path = Path(prov.plan_path)
        if not plan_path.exists():
            continue
        fixture_rollups = fixtures_dir / prov.fixture_id / "combined_rollups.json"
        rollups_path = fixture_rollups if fixture_rollups.exists() else None
        try:
            score = score_plan(plan_path, rollups_path)
        except Exception as exc:
            print(f"  [warn] scoring failed for {prov.plan_id}: {exc}")
            score = 0.0
        plan_records.append({
            "plan_id": prov.plan_id,
            "fixture_id": prov.fixture_id,
            "arm": prov.arm,
            "score": score,
            "plan_path": str(plan_path),
        })

    pairs = greedy_pair(plan_records, tolerance=tolerance)
    _print_audit(pairs, plan_records, target_pairs=target_pairs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(pairs, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\n[Saved] {output_path}  ({len(pairs)} pairs)")
    return pairs


def _print_audit(
    pairs: list[dict[str, Any]],
    plans: list[dict[str, Any]],
    *,
    target_pairs: int,
) -> None:
    n = len(pairs)
    gaps = [p["score_gap"] for p in pairs]
    mean_gap = sum(gaps) / n if n else 0.0
    by_fixture: dict[str, int] = {}
    for p in pairs:
        by_fixture[p["fixture_id"]] = by_fixture.get(p["fixture_id"], 0) + 1

    print("\n── Matching audit ──────────────────────────────")
    print(f"  Pairs yielded:    {n}  (target: {target_pairs})")
    print(f"  Mean score gap:   {mean_gap:.3f}")
    print(f"  Coverage OK:      {n >= int(target_pairs * 0.8)}")
    for fid, cnt in sorted(by_fixture.items()):
        print(f"    {fid}: {cnt} pairs")
    if n < int(target_pairs * 0.8):
        print("\n  ⚠ Coverage below 80% — retune sampler priors.")
    print("────────────────────────────────────────────────\n")
