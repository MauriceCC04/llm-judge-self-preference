from __future__ import annotations

import copy
import json
from pathlib import Path

from match.pair import greedy_pair
from match.structural_score import STRUCTURAL_SCORE_VERSION, score_structural_plan, score_structural_plan_obj
from tools.build_pairwise_eval_manifest import audit_masked_payload, build_manifest, mask_plan_for_judge


def _plan(duration_shift: int = 0, hard: bool = False, rest_day: int = 6) -> dict:
    days = []
    sessions = ["easy", "easy", "tempo" if hard else "easy", "easy", "long", "strength", "rest"]
    for i, session in enumerate(sessions):
        is_rest = i == rest_day
        dur = 0 if is_rest else [35, 45, 50, 40, 90, 25, 0][i] + duration_shift
        days.append(
            {
                "date": f"2026-03-{18+i:02d}",
                "session_type": "rest" if is_rest else session,
                "duration_minutes": dur,
                "is_rest_day": is_rest,
                "is_hard_day": bool(hard and session == "tempo" and not is_rest),
                "target_intensity": "rest" if is_rest else ("moderate" if hard and session == "tempo" else "easy"),
                "title": f"Human-facing title {i}",
                "purpose": "Short purpose text.",
                "workout": "Short workout text.",
                "signal_ids": [],
            }
        )
    return {
        "meta": {"plan_days": 7, "style": "trailrunning", "primary_goal": "goal"},
        "plan": {"days": days, "weekly_totals": {"planned_moving_time_hours": sum(d["duration_minutes"] for d in days) / 60.0}},
        "citations": [],
        "claim_attributions": [],
        "data_notes": [],
    }


def _prov(plan_id: str = "p", arm: str = "llm", source_model: str | None = "Qwen/Qwen2.5-7B-Instruct") -> dict:
    return {
        "plan_id": plan_id,
        "fixture_id": "ab_A2__r_high__rc_low__ph_peak",
        "athlete_band": "A2",
        "readiness": "high",
        "recovery_capability": "low",
        "race_phase": "peak",
        "plan_days": 7,
        "style": "trailrunning",
        "arm": arm,
        "source_model": source_model,
    }


def test_structural_score_ignores_presentation_fields() -> None:
    plan = _plan()
    mutated = copy.deepcopy(plan)
    mutated["citations"] = ["many citations"] * 10
    mutated["claim_attributions"] = [{"claim": "x", "source": "y"}]
    mutated["data_notes"] = ["verbose data note"] * 10
    mutated["meta"]["primary_goal"] = "A dramatically different human-facing goal sentence"
    for day in mutated["plan"]["days"]:
        day["title"] = "Very long title that should not matter"
        day["purpose"] = "Verbose purpose prose. " * 20
        day["workout"] = "Verbose workout prose. " * 20
    a = score_structural_plan_obj(plan, _prov(), plan_id="a")
    b = score_structural_plan_obj(mutated, _prov(), plan_id="b")
    assert a.score == b.score
    assert a.features == b.features


def test_structural_score_changes_when_structure_changes() -> None:
    base = score_structural_plan_obj(_plan(), _prov(), plan_id="base")
    changed = score_structural_plan_obj(_plan(duration_shift=30, hard=True, rest_day=99), _prov(), plan_id="changed")
    assert base.score != changed.score
    assert base.features["total_minutes"] != changed.features["total_minutes"]
    assert base.features["rest_days"] != changed.features["rest_days"]
    assert base.features["hard_days"] != changed.features["hard_days"]


def test_structural_score_report_has_version_components_and_note(tmp_path: Path) -> None:
    path = tmp_path / "p.json"
    prov_path = tmp_path / "p.json.provenance.json"
    path.write_text(json.dumps(_plan()), encoding="utf-8")
    prov_path.write_text(json.dumps(_prov()), encoding="utf-8")
    report = score_structural_plan(path, provenance_path=prov_path)
    assert report.score_version == STRUCTURAL_SCORE_VERSION
    assert report.components
    assert report.features
    assert report.presentation_fields_excluded is True
    assert "prose" in report.exclusion_note


def test_programmatic_and_llm_use_same_structural_function() -> None:
    llm = score_structural_plan_obj(_plan(), _prov(arm="llm", source_model="google/gemma-3-4b-it"), plan_id="llm")
    prog = score_structural_plan_obj(_plan(), _prov(arm="programmatic", source_model=None), plan_id="prog")
    assert llm.score == prog.score


def test_greedy_pair_produces_many_pairs_when_structural_scores_overlap() -> None:
    rows = []
    for i in range(20):
        rows.append({"plan_id": f"llm_{i}", "fixture_id": "f", "athlete_band": "A2", "readiness": "high", "recovery_capability": "low", "race_phase": "peak", "plan_days": 7, "style": "trailrunning", "arm": "llm", "score": 90 + (i % 3), "total_minutes": 250 + i, "n_rest_days": 2, "n_hard_days": 1, "n_active_days": 5, "n_long_runs": 1, "n_quality_days": 1, "max_day_minutes": 90, "mean_day_minutes": 35})
        rows.append({"plan_id": f"prog_{i}", "fixture_id": "f", "athlete_band": "A2", "readiness": "high", "recovery_capability": "low", "race_phase": "peak", "plan_days": 7, "style": "trailrunning", "arm": "programmatic", "score": 90 + (i % 3) + 0.5, "total_minutes": 252 + i, "n_rest_days": 2, "n_hard_days": 1, "n_active_days": 5, "n_long_runs": 1, "n_quality_days": 1, "max_day_minutes": 91, "mean_day_minutes": 36})
    pairs = greedy_pair(rows, tolerance=1.0)
    assert len(pairs) >= 18
    assert len({p["plan_a_id"] for p in pairs}) == len(pairs)
    assert len({p["plan_b_id"] for p in pairs}) == len(pairs)


def test_judge_facing_payload_masks_source_labels() -> None:
    plan = _plan()
    plan["source_model"] = "Qwen/Qwen2.5-7B-Instruct"
    plan["arm"] = "llm"
    plan["generation_condition"] = "gen_qwen"
    masked = mask_plan_for_judge(plan)
    text = json.dumps(masked).lower()
    assert "qwen" not in text
    assert "source_model" not in text
    assert audit_masked_payload({"plan_a": masked, "plan_b": masked})["passed"]


def test_eval_manifest_order_randomization_and_seed_reproducible(tmp_path: Path) -> None:
    plans = tmp_path / "plans"
    plans.mkdir()
    for pid, arm, src in [("llm_0", "llm", "Qwen/Qwen2.5-7B-Instruct"), ("prog_0", "programmatic", None)]:
        (plans / f"{pid}.json").write_text(json.dumps(_plan()), encoding="utf-8")
        (plans / f"{pid}.json.provenance.json").write_text(json.dumps(_prov(pid, arm, src)), encoding="utf-8")
    pairs = [{"pair_id": "pair_0000", "plan_a_id": "llm_0", "plan_b_id": "prog_0", "fixture_id": "f", "arm_a": "llm", "arm_b": "programmatic", "source_model_a": "Qwen/Qwen2.5-7B-Instruct", "structural_score_gap": 0.2, "structural_score_version": STRUCTURAL_SCORE_VERSION}]
    pair_path = tmp_path / "pairs.json"
    pair_path.write_text(json.dumps(pairs), encoding="utf-8")
    a = build_manifest(plans_dir=plans, pairs_path=pair_path, output_dir=tmp_path / "out_a", judge_names=["qwen_7b_judge"], n_runs=2, orders=["AB", "BA"], seed=123)
    b = build_manifest(plans_dir=plans, pairs_path=pair_path, output_dir=tmp_path / "out_b", judge_names=["qwen_7b_judge"], n_runs=2, orders=["AB", "BA"], seed=123)
    assert a["expected_pairwise_documents"] == 4
    assert (tmp_path / "out_a" / "pairwise_eval_manifest.jsonl").read_text() == (tmp_path / "out_b" / "pairwise_eval_manifest.jsonl").read_text().replace("out_b", "out_a")


def test_structural_matcher_uses_max_cardinality_not_local_greedy() -> None:
    def rec(pid: str, arm: str, score: float, minutes: int) -> dict:
        return {
            "plan_id": pid,
            "fixture_id": "f",
            "athlete_band": "A2",
            "readiness": "high",
            "recovery_capability": "low",
            "race_phase": "peak",
            "plan_days": 7,
            "style": "trailrunning",
            "arm": arm,
            "score": score,
            "total_minutes": minutes,
            "n_rest_days": 2,
            "n_hard_days": 1,
            "n_active_days": 5,
            "n_long_runs": 1,
            "n_quality_days": 1,
            "max_day_minutes": 90,
            "mean_day_minutes": 35,
        }

    rows = [
        rec("llm_0", "llm", 90.0, 250),
        rec("llm_1", "llm", 90.0, 500),
        rec("prog_0", "programmatic", 90.0, 250),
        rec("prog_1", "programmatic", 90.0, 250),
    ]
    # Make llm_1 ineligible for prog_1 by changing its fixture cell.
    rows[3]["fixture_id"] = "different_cell"
    pairs = greedy_pair(rows, tolerance=0.1)
    assert len(pairs) == 1

    # Restore the same cell but leave prog_1 structurally farther from llm_0.
    # A local greedy matcher would choose llm_0->prog_0 first and strand llm_1.
    rows[3]["fixture_id"] = "f"
    rows[3]["total_minutes"] = 1000
    pairs = greedy_pair(rows, tolerance=0.1)
    assert len(pairs) == 2
    assert len({p["plan_a_id"] for p in pairs}) == 2
    assert len({p["plan_b_id"] for p in pairs}) == 2
