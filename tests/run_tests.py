"""tests/run_tests.py — Gate-0 local test harness.

Runs on CPU, no GPU, no network.  Uses MockLLMClient for all LLM calls.
Exit code 0 = all tests passed; 1 = one or more failed.

Usage::

    python tests/run_tests.py

Matches the SuperCLIP pattern: @test decorator, per-test pass/fail, exit-1 on
any failure.  All temporary directories are cleaned up on exit.
"""
from __future__ import annotations

import atexit
import functools
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable

# Add project root to path
_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_PROJECT_ROOT))

# ── Test runner machinery ─────────────────────────────────────────────────────

_tests: list[tuple[str, Callable[[], None]]] = []
_results: dict[str, bool] = {}
_tmp_dirs: list[Path] = []


def _cleanup() -> None:
    for d in _tmp_dirs:
        shutil.rmtree(d, ignore_errors=True)


atexit.register(_cleanup)


def _make_tmp() -> Path:
    p = Path(tempfile.mkdtemp(prefix="jbs_gate0_"))
    _tmp_dirs.append(p)
    return p


def test(name: str) -> Callable[[Callable[[], None]], Callable[[], None]]:
    def decorator(fn: Callable[[], None]) -> Callable[[], None]:
        @functools.wraps(fn)
        def wrapper() -> None:
            fn()
        _tests.append((name, wrapper))
        return wrapper
    return decorator


def _run_all() -> bool:
    passed = 0
    failed = 0
    print(f"\n{'─' * 60}")
    print(f"  judge-bias-study Gate-0 tests  ({len(_tests)} tests)")
    print(f"{'─' * 60}\n")

    for name, fn in _tests:
        try:
            fn()
            _results[name] = True
            print(f"  ✅  {name}")
            passed += 1
        except Exception as exc:
            _results[name] = False
            print(f"  ❌  {name}")
            print(f"       {type(exc).__name__}: {exc}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'─' * 60}")
    print(f"  {passed} passed / {failed} failed")
    print(f"{'─' * 60}\n")
    return failed == 0


# ── Test fixtures (shared across tests) ──────────────────────────────────────

_FIXTURE_DIR: Path | None = None
_TMP_ROOT: Path | None = None


def _get_fixture() -> Path:
    global _FIXTURE_DIR, _TMP_ROOT
    if _FIXTURE_DIR is None:
        _TMP_ROOT = _make_tmp()
        from tests.create_test_fixtures import create_test_fixture
        _FIXTURE_DIR = create_test_fixture(_TMP_ROOT)
    return _FIXTURE_DIR


def _patch_llm(monkeypatch_dict: dict | None = None) -> None:
    """Monkey-patch all LLM client factories to return MockLLMClient."""
    from tests.mock_llm_client import MockLLMClient
    import trailtraining.llm.shared as shared_mod
    import trailtraining.llm.soft_eval as soft_eval_mod
    shared_mod.make_openrouter_client = lambda: MockLLMClient()
    soft_eval_mod.make_openrouter_client = lambda: MockLLMClient()
    try:
        import trailtraining.llm.coach as coach_mod
        coach_mod._make_openrouter_client = lambda: MockLLMClient()
    except Exception:
        pass
    try:
        import trailtraining.llm.revise as revise_mod
        revise_mod._make_openrouter_client = lambda: MockLLMClient()
    except Exception:
        pass
    # Patch our own env-var client factory used in the fallback explainer path
    try:
        import generate.programmatic_arm as prog_arm_mod
        prog_arm_mod._make_client_from_env = lambda: MockLLMClient()
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# Test 1 — Create synthetic fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@test("01 create_synthetic_fixtures")
def test_01_create_fixtures() -> None:
    fixture = _get_fixture()
    assert fixture.exists(), "Fixture dir not created"
    required = [
        "combined_summary.json",
        "combined_rollups.json",
        "readiness_and_risk_forecast.json",
        "formatted_personal_data.json",
    ]
    for name in required:
        assert (fixture / name).exists(), f"Missing {name}"

    combined = json.loads((fixture / "combined_summary.json").read_text())
    assert isinstance(combined, list) and len(combined) == 35
    rollups = json.loads((fixture / "combined_rollups.json").read_text())
    assert "7" in rollups["windows"]
    assert "28" in rollups["windows"]


# ═══════════════════════════════════════════════════════════════════════════════
# Test 2 — Contract imports
# ═══════════════════════════════════════════════════════════════════════════════

@test("02 contract_imports")
def test_02_contract_imports() -> None:
    from trailtraining.contracts import (
        TrainingPlanArtifact,
        MachinePlanArtifact,
        PlanExplanationArtifact,
        EvaluationReportArtifact,
        SoftAssessmentArtifact,
    )
    # Verify extra="forbid" is honoured
    import pydantic
    with _assert_raises(pydantic.ValidationError):
        TrainingPlanArtifact.model_validate({"unexpected_field": 1})


def _assert_raises(exc_type: type) -> "ContextManager":
    import contextlib

    class CM:
        def __enter__(self): return self
        def __exit__(self, tp, val, tb):
            if tp is None:
                raise AssertionError(f"Expected {exc_type.__name__} to be raised")
            if not issubclass(tp, exc_type):
                return False  # re-raise
            return True  # suppress

    return CM()


# ═══════════════════════════════════════════════════════════════════════════════
# Test 3 — Provenance round-trip
# ═══════════════════════════════════════════════════════════════════════════════

@test("03 provenance_roundtrip")
def test_03_provenance() -> None:
    from generate.provenance import PlanProvenance
    tmp = _make_tmp()
    p = PlanProvenance(
        plan_id="plan_001",
        fixture_id="r_low__rc_low__ph_base",
        arm="llm",
        source_model="meta-llama/Llama-3.1-8B-Instruct",
        explainer_model="Qwen/Qwen2.5-3B-Instruct",
        seed=42,
        generated_at="2026-03-17T00:00:00Z",
        plan_path=str(tmp / "plan.json"),
    )
    path = tmp / "plan.json.provenance.json"
    path.write_text(p.model_dump_json(), encoding="utf-8")
    loaded = PlanProvenance.model_validate_json(path.read_text())
    assert loaded.plan_id == "plan_001"
    assert loaded.arm == "llm"
    assert loaded.seed == 42


# ═══════════════════════════════════════════════════════════════════════════════
# Test 4 — Sampler shape
# ═══════════════════════════════════════════════════════════════════════════════

@test("04 sampler_shape")
def test_04_sampler() -> None:
    from generate.sampler import StructuralSamplerConfig, sample_machine_plan
    from trailtraining.llm.schemas import ensure_machine_plan_shape
    import json

    fixture = _get_fixture()
    combined = json.loads((fixture / "combined_summary.json").read_text())
    rollups = json.loads((fixture / "combined_rollups.json").read_text())

    cfg = StructuralSamplerConfig(plan_days=7, seed=0)
    skeleton = sample_machine_plan(cfg, combined=combined, rollups=rollups)
    # Must pass schema validation
    shaped = ensure_machine_plan_shape(skeleton)
    assert len(shaped["plan"]["days"]) == 7


# ═══════════════════════════════════════════════════════════════════════════════
# Test 5 — Guardrails normalise violations
# ═══════════════════════════════════════════════════════════════════════════════

@test("05 guardrails_normalise_violations")
def test_05_guardrails() -> None:
    from trailtraining.llm.guardrails import apply_eval_coach_guardrails
    from datetime import date, timedelta

    bad_days = []
    base = date(2026, 3, 18)
    for i in range(7):
        d = base + timedelta(days=i)
        bad_days.append({
            "date": d.isoformat(),
            "session_type": "intervals",  # all hard — violates max_hard_per_7d
            "is_rest_day": False,
            "is_hard_day": True,
            "duration_minutes": 60,
            "target_intensity": "threshold",
            "terrain": "trail",
            "workout": f"Hard session {i+1}",
            "purpose": "Quality.",
            "signal_ids": ["s1"],
        })

    plan_obj = {
        "meta": {"lifestyle_notes": ""},
        "plan": {
            "weekly_totals": {"planned_moving_time_hours": 7.0,
                              "planned_distance_km": None, "planned_elevation_m": None},
            "days": bad_days,
        },
        "data_notes": [],
    }
    apply_eval_coach_guardrails(plan_obj, rollups=None)

    days = plan_obj["plan"]["days"]
    # After guardrails: no more than 2 consecutive hard days
    consec = 0
    max_consec = 0
    for d in days:
        if d["is_hard_day"]:
            consec += 1
            max_consec = max(max_consec, consec)
        else:
            consec = 0
    assert max_consec <= 2, f"max consecutive hard = {max_consec} after guardrails"
    # At least one rest day
    rest_count = sum(1 for d in days if d["is_rest_day"])
    assert rest_count >= 1, "no rest day after guardrails"


# ═══════════════════════════════════════════════════════════════════════════════
# Test 6 — Programmatic arm end-to-end (mock LLM)
# ═══════════════════════════════════════════════════════════════════════════════

@test("06 programmatic_arm_end_to_end")
def test_06_programmatic_arm() -> None:
    _patch_llm()
    from generate.programmatic_arm import generate_programmatic_plan
    from trailtraining.contracts import TrainingPlanArtifact
    import json

    fixture = _get_fixture()
    tmp = _make_tmp()

    result_json, _, _prov = generate_programmatic_plan(
        fixture_dir=fixture,
        output_dir=tmp,
        plan_id="prog_test_001",
        seed=0,
    )
    art = TrainingPlanArtifact.model_validate(json.loads(result_json))
    assert art.plan.weekly_totals.planned_moving_time_hours >= 0


# ═══════════════════════════════════════════════════════════════════════════════
# Test 7 — LLM arm wrapper (mock LLM)
# ═══════════════════════════════════════════════════════════════════════════════

@test("07 llm_arm_wrapper")
def test_07_llm_arm() -> None:
    _patch_llm()
    from generate.llm_arm import generate_llm_plan
    from trailtraining.contracts import TrainingPlanArtifact
    import json

    fixture = _get_fixture()
    tmp = _make_tmp()

    result_json, saved_path, provenance_path = generate_llm_plan(
        fixture_dir=fixture,
        output_dir=tmp,
        plan_id="llm_test_001",
        source_model="meta-llama/Llama-3.1-8B-Instruct",
        seed=7,
    )
    art = TrainingPlanArtifact.model_validate(json.loads(result_json))
    assert art.meta.style == "trailrunning"
    assert Path(provenance_path).exists(), "Provenance sidecar not written"


# ═══════════════════════════════════════════════════════════════════════════════
# Test 8 — Explainer identity check
# ═══════════════════════════════════════════════════════════════════════════════

@test("08 explainer_identity_check")
def test_08_explainer_identity() -> None:
    from generate import constants

    llm_explainer = constants.EXPLAINER_MODEL_ID
    prog_explainer = constants.EXPLAINER_MODEL_ID  # same constant must be used

    assert llm_explainer == prog_explainer, (
        f"Explainer model mismatch: llm={llm_explainer!r} prog={prog_explainer!r}. "
        "This is a study-design fatal error — same explainer must be used for both arms."
    )
    assert llm_explainer  # must not be empty


# ═══════════════════════════════════════════════════════════════════════════════
# Test 9 — Deterministic scoring
# ═══════════════════════════════════════════════════════════════════════════════

@test("09 deterministic_scoring")
def test_09_deterministic_scoring() -> None:
    import json
    from trailtraining.llm.eval import evaluate_training_plan_quality_file
    from trailtraining.llm.constraints import ConstraintConfig

    # Write a minimal valid plan to a temp file
    tmp = _make_tmp()
    from tests.mock_llm_client import _training_plan_payload
    plan = _training_plan_payload()
    plan_path = tmp / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    rollups = json.loads((_get_fixture() / "combined_rollups.json").read_text())
    rollups_path = tmp / "combined_rollups.json"
    rollups_path.write_text(json.dumps(rollups), encoding="utf-8")

    cfg = ConstraintConfig(min_signal_ids_per_day=0)
    report, _ = evaluate_training_plan_quality_file(
        str(plan_path),
        rollups_path=str(rollups_path),
        cfg=cfg,
    )
    score = report["score"]
    assert isinstance(score, (int, float)), f"Score not numeric: {score!r}"
    assert 0 <= score <= 100, f"Score out of range: {score}"


# ═══════════════════════════════════════════════════════════════════════════════
# Test 10 — Matching logic
# ═══════════════════════════════════════════════════════════════════════════════

@test("10 matching_logic")
def test_10_matching() -> None:
    from match.pair import greedy_pair

    # 20 plans with known scores — pairs within |Δ| ≤ 1
    plans = [
        {"plan_id": f"llm_{i}", "fixture_id": "f0", "score": float(i), "arm": "llm"}
        for i in range(10)
    ] + [
        {"plan_id": f"prog_{i}", "fixture_id": "f0", "score": float(i) + 0.5, "arm": "programmatic"}
        for i in range(10)
    ]

    pairs = greedy_pair(plans, tolerance=1.0)
    assert len(pairs) >= 8, f"Expected ≥8 pairs, got {len(pairs)}"

    # No plan appears in two pairs
    used_ids: set[str] = set()
    for p in pairs:
        assert p["plan_a_id"] not in used_ids, f"{p['plan_a_id']} appears in multiple pairs"
        assert p["plan_b_id"] not in used_ids, f"{p['plan_b_id']} appears in multiple pairs"
        used_ids.add(p["plan_a_id"])
        used_ids.add(p["plan_b_id"])


# ═══════════════════════════════════════════════════════════════════════════════
# Test 11 — JSONL resume correctness
# ═══════════════════════════════════════════════════════════════════════════════

@test("11 jsonl_resume_correctness")
def test_11_jsonl_resume() -> None:
    from vendor_patches.resume_jsonl import ResumeJsonl

    tmp = _make_tmp()
    path = tmp / "test.jsonl"
    writer = ResumeJsonl(path, key_fields=("pair_id", "run"))

    # Write 50 records
    for i in range(50):
        writer.append({"pair_id": f"p{i:03d}", "run": 0, "verdict": "plan_a"})

    assert len(writer) == 50

    # Simulate crash at record 30: create a new ResumeJsonl that re-reads from disk
    writer2 = ResumeJsonl(path, key_fields=("pair_id", "run"))
    assert len(writer2) == 50  # all 50 already on disk

    # Attempt to write duplicates of records 0–29 — all should be skipped
    for i in range(30):
        written = writer2.append({"pair_id": f"p{i:03d}", "run": 0, "verdict": "plan_a"})
        assert not written, f"Record {i} should have been a duplicate"

    # Add 10 new records (50–59)
    for i in range(50, 60):
        writer2.append({"pair_id": f"p{i:03d}", "run": 0, "verdict": "plan_b"})

    # Reload and verify 60 distinct records, zero duplicates
    writer3 = ResumeJsonl(path, key_fields=("pair_id", "run"))
    assert len(writer3) == 60, f"Expected 60 distinct records, got {len(writer3)}"

    # Verify no duplicate keys on disk
    from vendor_patches.resume_jsonl import load_all
    records = load_all(path)
    keys = [(r["pair_id"], r["run"]) for r in records]
    assert len(keys) == len(set(keys)), "Duplicate keys found in JSONL"


# ═══════════════════════════════════════════════════════════════════════════════
# Test 12 — Pairwise harness loop (mock LLM)
# ═══════════════════════════════════════════════════════════════════════════════

@test("12 pairwise_harness_loop")
def test_12_pairwise_harness() -> None:
    _patch_llm()
    from judge.harness import run_pairwise_harness
    from vendor_patches.resume_jsonl import load_all

    tmp = _make_tmp()
    fixture = _get_fixture()

    # Build 4 minimal plan stubs
    import json
    from tests.mock_llm_client import _training_plan_payload
    plan = _training_plan_payload()
    plans_dir = tmp / "plans"
    plans_dir.mkdir()
    plan_ids = []
    for i in range(4):
        pid = f"plan_{i:03d}"
        plan_ids.append(pid)
        (plans_dir / f"{pid}.json").write_text(json.dumps(plan), encoding="utf-8")

    pairs = [
        {"pair_id": f"pair_{i}", "plan_a_id": plan_ids[i*2], "plan_b_id": plan_ids[i*2+1],
         "fixture_id": "test_fixture", "score_gap": 0.5}
        for i in range(2)
    ]
    pairs_file = tmp / "matched_pairs.json"
    pairs_file.write_text(json.dumps(pairs), encoding="utf-8")

    from judge.panel import JudgeSpec
    judge = JudgeSpec(
        name="test_judge",
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        quant="fp16",
        disk_gb=15,
        time_hours=6,
        role="judge",
    )

    output_file = tmp / "judgments_test_judge.jsonl"
    run_pairwise_harness(
        pairs=pairs,
        plans_dir=plans_dir,
        judge=judge,
        rollups_path=fixture / "combined_rollups.json",
        output_path=output_file,
        n_runs=2,
        n_positions=2,  # AB and BA
    )

    records = load_all(output_file)
    # 2 pairs × 2 runs × 2 positions = 8 records
    assert len(records) == 8, f"Expected 8 records, got {len(records)}"
    for r in records:
        assert "pair_id" in r and "judge" in r and "position" in r and "preferred" in r


# ═══════════════════════════════════════════════════════════════════════════════
# Test 13 — Per-rubric harness (mock LLM)
# ═══════════════════════════════════════════════════════════════════════════════

@test("13 per_rubric_harness")
def test_13_per_rubric_harness() -> None:
    _patch_llm()
    from judge.harness import run_soft_eval_harness
    from vendor_patches.resume_jsonl import load_all
    import json
    from tests.mock_llm_client import _training_plan_payload

    tmp = _make_tmp()
    fixture = _get_fixture()

    plans_dir = tmp / "plans"
    plans_dir.mkdir()
    plan = _training_plan_payload()
    plan_ids = []
    for i in range(4):
        pid = f"plan_{i:03d}"
        plan_ids.append(pid)
        (plans_dir / f"{pid}.json").write_text(json.dumps(plan), encoding="utf-8")

    from judge.panel import JudgeSpec
    judge = JudgeSpec(
        name="test_judge",
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        quant="fp16",
        disk_gb=15,
        time_hours=6,
        role="judge",
    )

    output_file = tmp / "soft_eval_test_judge.jsonl"
    run_soft_eval_harness(
        plan_ids=plan_ids,
        plans_dir=plans_dir,
        judge=judge,
        rollups_path=fixture / "combined_rollups.json",
        output_path=output_file,
    )

    records = load_all(output_file)
    # 4 plans × 1 judge = 4 records
    assert len(records) == 4, f"Expected 4 records, got {len(records)}"
    for r in records:
        assert "plan_id" in r and "judge" in r


# ═══════════════════════════════════════════════════════════════════════════════
# Test 14 — Schema-invalid response handling
# ═══════════════════════════════════════════════════════════════════════════════

@test("14 schema_invalid_response_handling")
def test_14_schema_failures() -> None:
    from vendor_patches.resume_jsonl import load_all
    import json
    from tests.mock_llm_client import _training_plan_payload

    tmp = _make_tmp()
    fixture = _get_fixture()

    # Monkey-patch to return malformed JSON
    import trailtraining.llm.shared as shared_mod
    import trailtraining.llm.soft_eval as soft_eval_mod
    from tests.mock_llm_client import _FakeResponse

    def _bad_client():
        class _Bad:
            class responses:
                @staticmethod
                def create(**kwargs): return _FakeResponse("NOT JSON AT ALL }{")
        return _Bad()

    shared_mod.make_openrouter_client = _bad_client
    soft_eval_mod.make_openrouter_client = _bad_client
    try:
        import trailtraining.llm.coach as coach_mod
        coach_mod._make_openrouter_client = _bad_client
    except Exception:
        pass

    plans_dir = tmp / "plans"
    plans_dir.mkdir()
    plan = _training_plan_payload()
    pid = "plan_000"
    (plans_dir / f"{pid}.json").write_text(json.dumps(plan), encoding="utf-8")

    from judge.panel import JudgeSpec
    judge = JudgeSpec(name="bad_judge", model_id="x", quant="fp16", disk_gb=1, time_hours=1, role="judge")

    from judge.harness import run_soft_eval_harness
    schema_failures_file = tmp / "schema_failures.jsonl"
    output_file = tmp / "soft_eval_bad.jsonl"

    # Should NOT raise; schema failure must be logged instead
    run_soft_eval_harness(
        plan_ids=[pid],
        plans_dir=plans_dir,
        judge=judge,
        rollups_path=fixture / "combined_rollups.json",
        output_path=output_file,
        schema_failures_path=schema_failures_file,
    )

    # schema_failures.jsonl must have at least one entry
    failures = load_all(schema_failures_file)
    assert len(failures) >= 1, "Schema failure not logged to schema_failures.jsonl"

    # Restore mock
    _patch_llm()


# ═══════════════════════════════════════════════════════════════════════════════
# Test 15 — Analysis load
# ═══════════════════════════════════════════════════════════════════════════════

@test("15 analysis_load")
def test_15_analysis_load() -> None:
    from analyze.load import load_judgments
    from vendor_patches.resume_jsonl import ResumeJsonl
    import json
    from tests.mock_llm_client import _training_plan_payload

    tmp = _make_tmp()

    # Write some provenance sidecars
    from generate.provenance import PlanProvenance
    for i in range(4):
        pid = f"plan_{i:03d}"
        arm = "llm" if i % 2 == 0 else "programmatic"
        prov = PlanProvenance(
            plan_id=pid,
            fixture_id="r_low__rc_low__ph_base",
            arm=arm,
            source_model="meta-llama/Llama-3.1-8B-Instruct" if arm == "llm" else None,
            explainer_model="Qwen/Qwen2.5-3B-Instruct",
            seed=i,
            generated_at="2026-03-17T00:00:00Z",
            plan_path=str(tmp / f"{pid}.json"),
        )
        (tmp / f"{pid}.json.provenance.json").write_text(prov.model_dump_json(), encoding="utf-8")

    # Write pairwise judgments
    pairs_jsonl = tmp / "judgments_test.jsonl"
    writer = ResumeJsonl(pairs_jsonl, key_fields=("pair_id", "judge", "run", "position"))
    for i in range(2):
        writer.append({
            "pair_id": f"pair_{i}",
            "plan_a_id": f"plan_{i*2:03d}",
            "plan_b_id": f"plan_{i*2+1:03d}",
            "judge": "test_judge",
            "run": 0,
            "position": "AB",
            "preferred": "plan_a",
            "fixture_id": "r_low__rc_low__ph_base",
        })

    df = load_judgments(
        judgments_dir=tmp,
        provenance_dir=tmp,
        kind="pairwise",
    )
    assert len(df) == 2
    required_cols = {"pair_id", "judge", "preferred", "arm_a", "arm_b"}
    missing = required_cols - set(df.columns)
    assert not missing, f"Missing columns: {missing}"


# ═══════════════════════════════════════════════════════════════════════════════
# Test 16 — Mixed-effects fit on synthetic data
# ═══════════════════════════════════════════════════════════════════════════════

@test("16 mixed_effects_fit")
def test_16_mixed_effects() -> None:
    try:
        import pandas  # noqa: F401
        import statsmodels  # noqa: F401
    except ImportError:
        print("     [skip] pandas/statsmodels not installed")
        return

    from analyze.models import fit_h1_model
    import pandas as pd
    import numpy as np

    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame({
        "pair_id": [f"p{i}" for i in range(n)],
        "judge": rng.choice(["j1", "j2", "j3"], n),
        "fixture_id": rng.choice(["f1", "f2"], n),
        "prefers_llm": rng.integers(0, 2, n),
        "position": rng.choice(["AB", "BA"], n),
        "run": rng.integers(0, 3, n),
    })

    result = fit_h1_model(df)
    assert "coef_is_llm" in result
    assert "pvalue_is_llm" in result
    assert isinstance(result["coef_is_llm"], float)


# ═══════════════════════════════════════════════════════════════════════════════
# Test 17 — Position-bias audit
# ═══════════════════════════════════════════════════════════════════════════════

@test("17 position_bias_audit")
def test_17_position_bias() -> None:
    from analyze.load import detect_position_bias
    import pandas as pd
    import numpy as np

    try:
        import pandas  # noqa: F401
    except ImportError:
        print("     [skip] pandas not installed")
        return

    rng = np.random.default_rng(99)
    n = 60
    # Inject strong position_a bias (70% prefer plan_a regardless of content)
    df = pd.DataFrame({
        "judge": "biased_judge",
        "position": rng.choice(["AB", "BA"], n),
        "preferred": ["plan_a"] * 43 + ["plan_b"] * 17,  # ~72% plan_a > 0.2 threshold
    })
    result = detect_position_bias(df, threshold=0.2)
    assert result["biased"] is True, "Should detect bias when P(prefer_A) > 0.7"
    assert result["judge"] == "biased_judge"


# ═══════════════════════════════════════════════════════════════════════════════
# Test 18 — SLURM scripts parse
# ═══════════════════════════════════════════════════════════════════════════════

@test("18 slurm_scripts_parse")
def test_18_slurm_parse() -> None:
    slurm_dir = _PROJECT_ROOT / "slurm"
    if not slurm_dir.exists():
        print("     [skip] slurm/ not yet created")
        return

    scripts = list(slurm_dir.glob("*.sh"))
    assert scripts, "No .sh scripts found in slurm/"

    for script in scripts:
        result = subprocess.run(
            ["bash", "-n", str(script)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"bash -n failed for {script.name}:\n{result.stderr}"
        )

    # Every script (except common.sh) must source common.sh
    common = slurm_dir / "common.sh"
    for script in scripts:
        if script.name == "common.sh":
            continue
        text = script.read_text(encoding="utf-8")
        assert "common.sh" in text, f"{script.name} does not source common.sh"


# ═══════════════════════════════════════════════════════════════════════════════
# Test 19 — common.sh resolves PROJECT_ROOT from script location
# ═══════════════════════════════════════════════════════════════════════════════

@test("19 common_sh_project_root")
def test_19_common_sh() -> None:
    slurm_dir = _PROJECT_ROOT / "slurm"
    common = slurm_dir / "common.sh"
    if not common.exists():
        print("     [skip] slurm/common.sh not yet created")
        return

    # Source from /tmp and check PROJECT_ROOT resolves to the actual repo root
    result = subprocess.run(
        ["bash", "-c",
         f"source {common} 2>/dev/null; echo $PROJECT_ROOT"],
        capture_output=True,
        text=True,
        cwd="/tmp",
    )
    output = result.stdout.strip()
    assert output, "PROJECT_ROOT is empty after sourcing common.sh"
    assert Path(output).exists(), f"PROJECT_ROOT={output!r} does not exist"


# ═══════════════════════════════════════════════════════════════════════════════
# Test 20 — Preflight schema
# ═══════════════════════════════════════════════════════════════════════════════

@test("20 preflight_schema")
def test_20_preflight() -> None:
    from tools.hpc_preflight import build_preflight_report

    report = build_preflight_report(
        home_quota_gb=50,
        used_gb=10,
        has_gpu=False,  # CPU-only test environment
        hf_cache_dir=_make_tmp(),
    )
    required_keys = {
        "quota_ok",
        "used_gb",
        "available_gb",
        "imports_ok",
        "cache_dirs_exist",
    }
    missing = required_keys - set(report.keys())
    assert not missing, f"Preflight report missing keys: {missing}"


# ═══════════════════════════════════════════════════════════════════════════════
# Test 21 — Preflight storage estimator
# ═══════════════════════════════════════════════════════════════════════════════

@test("21 preflight_storage_estimator")
def test_21_storage_estimator() -> None:
    from tools.hpc_preflight import classify_storage

    # Within quota
    ok = classify_storage(used_gb=25, quota_gb=50, soft_ceiling_gb=30)
    assert ok["quota_ok"] is True
    assert ok["ceiling_ok"] is True

    # Exceeds soft ceiling (> 30 GB) but within quota
    over_ceil = classify_storage(used_gb=32, quota_gb=50, soft_ceiling_gb=30)
    assert over_ceil["quota_ok"] is True
    assert over_ceil["ceiling_ok"] is False

    # Nonexistent directory: must not crash
    from tools.hpc_preflight import estimate_dir_gb
    result = estimate_dir_gb(Path("/nonexistent/does/not/exist"))
    assert result == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Test 22 — vLLM server lifecycle (mock subprocess)
# ═══════════════════════════════════════════════════════════════════════════════

@test("22 vllm_server_lifecycle")
def test_22_vllm_lifecycle() -> None:
    from judge.vllm_server import VllmServer, VllmServerError
    import threading

    tmp = _make_tmp()

    # Build a tiny mock HTTP server that responds 200 to /health
    import http.server
    import socketserver

    PORT = 0  # OS picks a free port

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
        def log_message(self, *args): pass  # silence

    httpd = socketserver.TCPServer(("127.0.0.1", PORT), _Handler)
    httpd.allow_reuse_address = True
    actual_port = httpd.server_address[1]

    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    try:
        # VllmServer should successfully poll this mock server
        server = VllmServer(
            model_id="mock/model",
            port=actual_port,
            max_model_len=8192,
            log_dir=tmp,
        )
        # health_poll returns True when the server responds 200
        healthy = server.health_poll(timeout_s=5, interval_s=0.1)
        assert healthy, "health_poll should return True for a running server"

        # shutdown should not raise (no subprocess to kill in this mock)
        server.shutdown()
    finally:
        httpd.shutdown()




# ═══════════════════════════════════════════════════════════════════════════════
# Test 23 — fit_priors round-trip
# ═══════════════════════════════════════════════════════════════════════════════

@test("23 fit_priors_roundtrip")
def test_23_fit_priors() -> None:
    import json
    from datetime import datetime, timezone
    from generate.fit_priors import fit_and_save, load_sampler_config
    from generate.provenance import PlanProvenance
    from tests.mock_llm_client import _training_plan_payload

    tmp = _make_tmp()
    plans_dir = tmp / "plans"
    plans_dir.mkdir()

    # Write 8 mock plans with llm-arm provenance
    for i in range(8):
        plan = _training_plan_payload()
        pid = f"plan_{i:03d}"
        plan_path = plans_dir / f"{pid}.json"
        plan_path.write_text(json.dumps(plan), encoding="utf-8")
        prov = PlanProvenance(
            plan_id=pid, fixture_id="f0", arm="llm",
            source_model="meta-llama/Llama-3.1-8B-Instruct",
            explainer_model="Qwen/Qwen2.5-3B-Instruct",
            seed=i,
            generated_at=datetime.now(tz=timezone.utc).isoformat(),
            plan_path=str(plan_path),
        )
        (plans_dir / f"{pid}.json.provenance.json").write_text(prov.model_dump_json())

    config_path = tmp / "sampler_config.json"
    fit_and_save(plans_dir, config_path, min_plans=5)
    assert config_path.exists(), "sampler_config.json not written"

    cfg = load_sampler_config(config_path)
    assert 0 < cfg.p_hard_day < 1, f"p_hard_day out of range: {cfg.p_hard_day}"
    assert 0 < cfg.p_rest_day < 1, f"p_rest_day out of range: {cfg.p_rest_day}"
    assert abs(sum(cfg.easy_type_probs.values()) - 1.0) < 0.01, "easy_type_probs don't sum to 1"


# ═══════════════════════════════════════════════════════════════════════════════
# Test 24 — run_generation resume (skip existing plans)
# ═══════════════════════════════════════════════════════════════════════════════

@test("24 run_generation_resume")
def test_24_run_generation_resume() -> None:
    import json
    from generate.run_generation import _plan_exists, _write_failure
    from pathlib import Path

    tmp = _make_tmp()

    # Plan that exists should be skipped
    (tmp / "plan_abc.json").write_text(json.dumps({"meta": {}}))
    assert _plan_exists(tmp, "plan_abc") is True
    assert _plan_exists(tmp, "plan_xyz") is False

    # _write_failure writes to failures JSONL
    _write_failure(tmp / "failed_plans.jsonl", "plan_xyz", RuntimeError("test error"))
    lines = (tmp / "failed_plans.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["plan_id"] == "plan_xyz"
    assert "test error" in entry["error"]


# ═══════════════════════════════════════════════════════════════════════════════
# Test 25 — figures module imports and CSV output
# ═══════════════════════════════════════════════════════════════════════════════

@test("25 figures_csv_output")
def test_25_figures_csv() -> None:
    from analyze.figures import save_rubric_heatmap_csv

    try:
        import pandas as pd
    except ImportError:
        print("     [skip] pandas not installed")
        return

    tmp = _make_tmp()
    rubric_results = {
        "goal_alignment": {"delta": 2.5, "pvalue": 0.03, "pvalue_holm": 0.15, "significant": False},
        "explanation_quality": {"delta": 5.1, "pvalue": 0.001, "pvalue_holm": 0.005, "significant": True},
        "plan_coherence": {"delta": 1.2, "pvalue": 0.12, "pvalue_holm": 0.36, "significant": False},
    }
    out = tmp / "h2.csv"
    save_rubric_heatmap_csv(rubric_results, out)
    assert out.exists()
    df = pd.read_csv(out)
    assert set(df.columns) >= {"rubric", "delta", "pvalue_holm", "significant"}
    assert len(df) == 3


# ═══════════════════════════════════════════════════════════════════════════════
# Test 26 — run_generation.sh parses as bash
# ═══════════════════════════════════════════════════════════════════════════════

@test("26 run_generation_sh_parses")
def test_26_gen_sh_parse() -> None:
    script = _PROJECT_ROOT / "slurm" / "run_generation.sh"
    if not script.exists():
        print("     [skip] slurm/run_generation.sh not found")
        return
    result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, f"bash -n failed:\n{result.stderr}"
    text = script.read_text()
    assert "common.sh" in text, "run_generation.sh does not source common.sh"



# ═══════════════════════════════════════════════════════════════════════════════
# Test 27 — H1 logistic model returns finite result on synthetic data
# ═══════════════════════════════════════════════════════════════════════════════

@test("27 h1_logistic_model")
def test_27_h1_logistic() -> None:
    try:
        import pandas as pd
        import numpy as np
    except ImportError:
        print("     [skip] pandas not installed")
        return

    from analyze.models import fit_h1_model
    import math

    rng = np.random.default_rng(7)
    n = 160
    # Set prefers_llm directly so the injected rate is unambiguous (60%)
    df = pd.DataFrame({
        "pair_id": [f"p{i}" for i in range(n)],
        "judge": rng.choice(["j1", "j2", "j3"], n),
        "fixture_id": rng.choice(["f1", "f2"], n),
        # Inject exactly 60% LLM preference by construction
        "prefers_llm": ([1] * int(n * 0.6) + [0] * int(n * 0.4)),
    })

    result = fit_h1_model(df)

    assert "coef_is_llm" in result, "missing coef_is_llm"
    assert "pvalue_is_llm" in result, "missing pvalue_is_llm"
    assert "prob_llm" in result, "missing prob_llm"
    assert isinstance(result["n_obs"], int) and result["n_obs"] == n
    # prob_llm = mean predicted probability; should be near injected 0.6
    prob = result["prob_llm"]
    assert not math.isnan(prob), "prob_llm is NaN"
    assert 0.55 <= prob <= 0.65, f"Expected ~0.60 (±0.05), got {prob}"


# ═══════════════════════════════════════════════════════════════════════════════
# Test 28 — pilot bias gate: pass and fail cases
# ═══════════════════════════════════════════════════════════════════════════════

@test("28 pilot_bias_gate")
def test_28_pilot_gate() -> None:
    from judge.harness import check_pilot_bias_gate
    from vendor_patches.resume_jsonl import ResumeJsonl

    tmp = _make_tmp()
    jpath = tmp / "pilot.jsonl"
    writer = ResumeJsonl(jpath, key_fields=("pair_id", "judge", "run", "position"))

    # Write 40 records with no position bias (50/50)
    for i in range(40):
        writer.append({
            "pair_id": f"pair_{i:03d}", "judge": "test_judge",
            "run": 0, "position": "AB",
            "preferred": "plan_a" if i < 20 else "plan_b",
        })

    result = check_pilot_bias_gate(jpath, threshold=0.2)
    assert result["passed"] is True, f"Should pass (50/50): {result}"

    # Write heavily biased records
    jpath2 = tmp / "biased.jsonl"
    writer2 = ResumeJsonl(jpath2, key_fields=("pair_id", "judge", "run", "position"))
    for i in range(40):
        writer2.append({
            "pair_id": f"pair_{i:03d}", "judge": "biased_judge",
            "run": 0, "position": "AB",
            "preferred": "plan_a" if i < 35 else "plan_b",  # 87.5% plan_a
        })

    result2 = check_pilot_bias_gate(jpath2, threshold=0.2)
    assert result2["passed"] is False, f"Should fail (87.5% plan_a): {result2}"


# ═══════════════════════════════════════════════════════════════════════════════
# Test 29 — run_analysis produces summary.json and summary.md
# ═══════════════════════════════════════════════════════════════════════════════

@test("29 run_analysis_produces_outputs")
def test_29_run_analysis() -> None:
    try:
        import pandas as pd
    except ImportError:
        print("     [skip] pandas not installed")
        return

    from analyze.run_analysis import main as run_analysis
    from vendor_patches.resume_jsonl import ResumeJsonl
    from generate.provenance import PlanProvenance
    import json
    from datetime import datetime, timezone
    from tests.mock_llm_client import _training_plan_payload

    tmp = _make_tmp()
    judgments_dir = tmp / "judgments"
    plans_dir = tmp / "plans"
    out_dir = tmp / "results"
    judgments_dir.mkdir()
    plans_dir.mkdir()

    # Write 4 plan files + provenance
    for i in range(4):
        pid = f"plan_{i:03d}"
        plan = _training_plan_payload()
        (plans_dir / f"{pid}.json").write_text(json.dumps(plan))
        arm = "llm" if i % 2 == 0 else "programmatic"
        prov = PlanProvenance(
            plan_id=pid, fixture_id="r_low__rc_low__ph_base", arm=arm,
            source_model="meta-llama/Llama-3.1-8B-Instruct" if arm == "llm" else None,
            explainer_model="Qwen/Qwen2.5-3B-Instruct",
            seed=i, generated_at=datetime.now(tz=timezone.utc).isoformat(),
            plan_path=str(plans_dir / f"{pid}.json"),
        )
        (plans_dir / f"{pid}.json.provenance.json").write_text(prov.model_dump_json())

    # Write pairwise judgment JSONL
    pairwise = judgments_dir / "pairwise_test_judge.jsonl"
    writer = ResumeJsonl(pairwise, key_fields=("pair_id", "judge", "run", "position"))
    for i in range(20):
        writer.append({
            "pair_id": f"pair_{i % 2:03d}",
            "plan_a_id": "plan_000", "plan_b_id": "plan_001",
            "judge": "qwen_7b_judge", "run": i // 4, "position": "AB" if i % 2 == 0 else "BA",
            "preferred": "plan_a", "preferred_id": "plan_000",
            "arm_a": "llm", "arm_b": "programmatic",
            "fixture_id": "r_low__rc_low__ph_base",
        })

    run_analysis([
        "--judgments", str(judgments_dir),
        "--plans", str(plans_dir),
        "--output", str(out_dir),
    ])

    assert (out_dir / "summary.json").exists(), "summary.json not written"
    assert (out_dir / "summary.md").exists(), "summary.md not written"
    assert (out_dir / "h1_logistic_result.json").exists(), "h1 result not written"
    assert (out_dir / "position_bias_audit.csv").exists(), "position bias audit not written"

    summary = json.loads((out_dir / "summary.json").read_text())
    assert "h1" in summary and "h2" in summary and "h3" in summary and "h4" in summary


# ═══════════════════════════════════════════════════════════════════════════════
# Test 30 — llm_arm writes to output_dir, not fixture_dir
# ═══════════════════════════════════════════════════════════════════════════════

@test("30 llm_arm_output_separation")
def test_30_llm_arm_separation() -> None:
    """Verify generate_llm_plan writes plan to output_dir, not fixture_dir."""
    _patch_llm()
    from generate.llm_arm import generate_llm_plan
    from pathlib import Path

    fixture = _get_fixture()
    tmp = _make_tmp()
    output_dir = tmp / "plans"

    _, plan_path, prov_path = generate_llm_plan(
        fixture_dir=fixture,
        output_dir=output_dir,
        plan_id="sep_test_001",
        source_model="meta-llama/Llama-3.1-8B-Instruct",
        seed=99,
    )

    # Plan must be in output_dir, not fixture_dir
    assert Path(plan_path).parent.resolve() == output_dir.resolve(), (
        f"Plan written to wrong dir: {plan_path} (expected {output_dir})"
    )
    assert Path(prov_path).parent.resolve() == output_dir.resolve()

    # fixture_dir must be clean — no new JSON files written there
    fixture_json = list(fixture.glob("*.json"))
    expected_fixture_files = {
        "combined_summary.json", "combined_rollups.json",
        "readiness_and_risk_forecast.json", "formatted_personal_data.json",
    }
    written = {f.name for f in fixture_json} - expected_fixture_files
    assert not written, f"Unexpected files written to fixture_dir: {written}"



# ═══════════════════════════════════════════════════════════════════════════════
# Test 31 — load_judgments populates source_model_a / source_model_b
# ═══════════════════════════════════════════════════════════════════════════════

@test("31 load_judgments_source_model_join")
def test_31_source_model_join() -> None:
    try:
        import pandas as pd
    except ImportError:
        print("     [skip] pandas not installed")
        return

    import json
    from analyze.load import load_judgments
    from generate.provenance import PlanProvenance
    from vendor_patches.resume_jsonl import ResumeJsonl
    from datetime import datetime, timezone

    tmp = _make_tmp()
    plans_dir = tmp / "plans"
    judgments_dir = tmp / "judgments"
    plans_dir.mkdir(); judgments_dir.mkdir()

    # Write two plans with different arms / source models
    for pid, arm, model in [
        ("plan_llm_001", "llm", "meta-llama/Llama-3.1-8B-Instruct"),
        ("plan_prog_001", "programmatic", None),
    ]:
        prov = PlanProvenance(
            plan_id=pid, fixture_id="fix1", arm=arm,
            source_model=model,
            explainer_model="Qwen/Qwen2.5-3B-Instruct",
            seed=0, generated_at=datetime.now(tz=timezone.utc).isoformat(),
            plan_path=str(plans_dir / f"{pid}.json"),
        )
        (plans_dir / f"{pid}.json.provenance.json").write_text(prov.model_dump_json())

    # Write a pairwise judgment
    jpath = judgments_dir / "pairwise_test.jsonl"
    writer = ResumeJsonl(jpath, key_fields=("pair_id", "judge", "run", "position"))
    writer.append({
        "pair_id": "pair_0001",
        "plan_a_id": "plan_llm_001", "plan_b_id": "plan_prog_001",
        "judge": "qwen_7b_judge", "run": 0, "position": "AB",
        "preferred": "plan_a", "preferred_id": "plan_llm_001",
    })

    df = load_judgments(judgments_dir, plans_dir, kind="pairwise")
    assert not df.empty
    assert "source_model_a" in df.columns
    assert "source_model_b" in df.columns
    assert df["source_model_a"].iloc[0] == "meta-llama/Llama-3.1-8B-Instruct"
    assert df["arm_a"].iloc[0] == "llm"
    assert df["arm_b"].iloc[0] == "programmatic"
    assert df["prefers_llm"].iloc[0] == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Test 32 — score_plan fallback (heuristic, no trailtraining eval)
# ═══════════════════════════════════════════════════════════════════════════════

@test("32 score_plan_heuristic_fallback")
def test_32_score_plan_heuristic() -> None:
    from match.pair import _heuristic_score
    from tests.mock_llm_client import _training_plan_payload
    import json

    tmp = _make_tmp()
    plan = _training_plan_payload()
    ppath = tmp / "plan_001.json"
    ppath.write_text(json.dumps(plan))

    score = _heuristic_score(ppath)
    assert isinstance(score, float), f"Expected float, got {type(score)}"
    assert 0.0 < score < 25.0, f"Heuristic score out of plausible range: {score}"

    # Two different plan_ids should get different scores (jitter)
    ppath2 = tmp / "plan_002.json"
    ppath2.write_text(json.dumps(plan))
    score2 = _heuristic_score(ppath2)
    assert score != score2, "Jitter should produce different scores for different plan IDs"


# ═══════════════════════════════════════════════════════════════════════════════
# Test 33 — cli.py subcommand parsing (all 5 subcommands parse without error)
# ═══════════════════════════════════════════════════════════════════════════════

@test("33 cli_subcommand_parsing")
def test_33_cli_parse() -> None:
    from cli import build_parser

    parser = build_parser()

    cases = [
        ["generate", "--arm", "llm", "--n", "5"],
        ["generate", "--arm", "programmatic", "--sampler-config", "cfg.json"],
        ["fit-priors", "--plans", "plans/", "--output", "cfg.json"],
        ["match", "--plans", "plans/", "--output", "pairs.json"],
        ["judge", "--judge", "qwen_7b_judge", "--pilot"],
        ["analyze", "--judgments", "j/", "--plans", "p/", "--output", "r/"],
    ]

    for argv in cases:
        args = parser.parse_args(argv)
        assert hasattr(args, "func"), f"No func registered for: {argv}"



# ═══════════════════════════════════════════════════════════════════════════════
# Test 34 — ResumeJsonl O(1) append: 1 500 writes complete quickly
# ═══════════════════════════════════════════════════════════════════════════════

@test("34 resume_jsonl_append_performance")
def test_34_jsonl_perf() -> None:
    import time
    from vendor_patches.resume_jsonl import ResumeJsonl

    tmp = _make_tmp() / "perf.jsonl"
    writer = ResumeJsonl(tmp, key_fields=("pair_id", "run", "pos"))
    t0 = time.time()
    for i in range(1500):
        writer.append({"pair_id": f"p{i:04d}", "run": 0, "pos": "AB", "v": "x"})
    elapsed = time.time() - t0

    # Should complete in < 3 s on any reasonable disk (was ~39 s with O(n²) impl)
    assert elapsed < 3.0, f"1500 appends took {elapsed:.2f}s — likely O(n²) regression"
    assert len(writer) == 1500

    # Dedup: re-appending should be no-ops
    for i in range(10):
        writer.append({"pair_id": f"p{i:04d}", "run": 0, "pos": "AB", "v": "x"})
    assert len(writer) == 1500, "Dedup failed after re-append"


# ═══════════════════════════════════════════════════════════════════════════════
# Test 35 — rubric_paired_contrasts: independent fallback + paired path
# ═══════════════════════════════════════════════════════════════════════════════

@test("35 rubric_paired_contrasts_correctness")
def test_35_rubric_contrasts() -> None:
    try:
        import pandas as pd
        import numpy as np
    except ImportError:
        print("     [skip] pandas not installed")
        return

    from analyze.rubric_deltas import rubric_paired_contrasts, RUBRIC_IDS

    rng = np.random.default_rng(42)
    n = 40  # 40 paired plans

    def _make_row(pid, arm, delta_bias=0.0):
        base_score = rng.uniform(5, 8)
        scores = {rid: {"score": float(base_score + delta_bias + rng.normal(0, 0.3))}
                  for rid in RUBRIC_IDS}
        return {"plan_id": pid, "arm": arm, "judge": "j1", "rubric_scores": scores}

    rows = []
    for i in range(n):
        rows.append(_make_row(f"llm_{i:03d}", "llm", delta_bias=1.5))   # LLM gets +1.5
        rows.append(_make_row(f"prog_{i:03d}", "programmatic", delta_bias=0.0))
    df = pd.DataFrame(rows)

    # ── Independent fallback (no pairs_path) ──
    result_ind = rubric_paired_contrasts(df)
    assert set(result_ind.keys()) == set(RUBRIC_IDS)
    for rid, stats in result_ind.items():
        assert not stats.get("paired", True), f"Expected paired=False for {rid}"
        assert "pvalue_holm" in stats

    # ── Paired path with a synthetic pairs manifest ──
    import json, tempfile
    pairs = [{"plan_a_id": f"llm_{i:03d}", "plan_b_id": f"prog_{i:03d}",
               "arm_a": "llm", "arm_b": "programmatic", "pair_id": f"pair_{i:04d}"}
              for i in range(n)]
    pairs_file = _make_tmp() / "pairs.json"
    pairs_file.write_text(json.dumps(pairs))

    result_pair = rubric_paired_contrasts(df, pairs_path=pairs_file)
    for rid, stats in result_pair.items():
        assert stats.get("paired") is True, f"Expected paired=True for {rid}"
        assert stats.get("n_pairs", 0) == n, f"Expected {n} pairs for {rid}"
        # With +1.5 bias, delta should be clearly positive
        assert stats["delta"] > 0.5, f"Expected positive delta for {rid}, got {stats['delta']}"


# ═══════════════════════════════════════════════════════════════════════════════
# Test 36 — _make_client_from_env honours TRAILTRAINING_LLM_BASE_URL
# ═══════════════════════════════════════════════════════════════════════════════

@test("36 make_client_from_env")
def test_36_client_env() -> None:
    import os
    from unittest.mock import patch

    # With TRAILTRAINING_LLM_BASE_URL set, should not raise
    with patch.dict(os.environ, {"TRAILTRAINING_LLM_BASE_URL": "http://127.0.0.1:8773/v1"}):
        from generate.programmatic_arm import _make_client_from_env
        # Re-import to pick up the real function (not mocked by _patch_llm)
        import importlib, generate.programmatic_arm as mod
        importlib.reload(mod)
        client = mod._make_client_from_env()
        # Check the base_url was set correctly
        assert "8773" in str(client.base_url), f"base_url wrong: {client.base_url}"

    # Without any env vars, should raise RuntimeError
    clean_env = {k: v for k, v in os.environ.items()
                 if k not in ("TRAILTRAINING_LLM_BASE_URL",
                               "OPENROUTER_API_KEY",
                               "TRAILTRAINING_OPENROUTER_API_KEY")}
    with patch.dict(os.environ, clean_env, clear=True):
        importlib.reload(mod)
        try:
            mod._make_client_from_env()
            assert False, "Expected RuntimeError when no endpoint configured"
        except RuntimeError:
            pass



# ═══════════════════════════════════════════════════════════════════════════════
# Test 34 — rubric_paired_contrasts with synthetic paired data
# ═══════════════════════════════════════════════════════════════════════════════

@test("37 rubric_paired_contrasts")
def test_34_rubric_contrasts() -> None:
    try:
        import pandas as pd
        import numpy as np
    except ImportError:
        print("     [skip] pandas not installed")
        return

    from analyze.rubric_deltas import rubric_paired_contrasts

    rng = np.random.default_rng(42)
    n = 40
    # LLM arm gets consistently higher explanation_quality (+3 signal)
    records = []
    for i in range(n):
        arm = "llm" if i % 2 == 0 else "programmatic"
        expl_boost = 3.0 if arm == "llm" else 0.0
        records.append({
            "plan_id": f"plan_{i:03d}",
            "judge": "test_judge",
            "arm": arm,
            "rubric_scores": {
                "goal_alignment":       {"score": float(rng.integers(5, 10))},
                "plan_coherence":       {"score": float(rng.integers(5, 10))},
                "explanation_quality":  {"score": float(rng.integers(5, 8)) + expl_boost},
                "caution_proportionality": {"score": float(rng.integers(5, 10))},
                "actionability":        {"score": float(rng.integers(5, 10))},
            },
        })

    df = pd.DataFrame(records)
    results = rubric_paired_contrasts(df)

    assert set(results.keys()) >= {"explanation_quality", "plan_coherence"},         f"Expected rubric keys, got: {list(results.keys())}"

    exp_delta = results["explanation_quality"]["delta"]
    assert exp_delta > 1.5, f"explanation_quality delta should be ~3, got {exp_delta}"

    # All rubrics should have pvalue_holm assigned
    for rid, stats in results.items():
        assert "pvalue_holm" in stats, f"{rid} missing pvalue_holm"
        assert "significant" in stats, f"{rid} missing significant"

    # explanation_quality should be the largest delta
    deltas = {r: abs(stats.get("delta", 0)) for r, stats in results.items() if stats.get("delta") == stats.get("delta")}
    assert max(deltas, key=deltas.get) == "explanation_quality",         f"Largest delta should be explanation_quality, got: {max(deltas, key=deltas.get)}"


# ═══════════════════════════════════════════════════════════════════════════════
# Test 35 — add_same_family_column correctly identifies same-family pairs
# ═══════════════════════════════════════════════════════════════════════════════

@test("38 add_same_family_column")
def test_35_same_family() -> None:
    try:
        import pandas as pd
    except ImportError:
        print("     [skip] pandas not installed")
        return

    from analyze.models import add_same_family_column

    df = pd.DataFrame([
        # Qwen judge, Qwen source → same family
        {"judge": "qwen_7b_judge", "source_model_a": "Qwen/Qwen2.5-7B-Instruct"},
        # Llama judge, Qwen source → different family
        {"judge": "llama_8b_judge", "source_model_a": "Qwen/Qwen2.5-7B-Instruct"},
        # Llama judge, Llama source → same family
        {"judge": "llama_8b_judge", "source_model_a": "meta-llama/Llama-3.1-8B-Instruct"},
        # Mistral judge, Llama source → different family
        {"judge": "mistral_7b_judge", "source_model_a": "meta-llama/Llama-3.1-8B-Instruct"},
    ])

    result = add_same_family_column(df)

    assert "same_family" in result.columns
    assert result["same_family"].iloc[0] == 1, "Qwen/Qwen should be same family"
    assert result["same_family"].iloc[1] == 0, "Llama/Qwen should be different family"
    assert result["same_family"].iloc[2] == 1, "Llama/Llama should be same family"
    assert result["same_family"].iloc[3] == 0, "Mistral/Llama should be different family"


# ═══════════════════════════════════════════════════════════════════════════════
# Test 36 — detect_position_bias handles preferred_id format
# ═══════════════════════════════════════════════════════════════════════════════

@test("39 detect_position_bias_preferred_id")
def test_36_position_bias_preferred_id() -> None:
    try:
        import pandas as pd
    except ImportError:
        print("     [skip] pandas not installed")
        return

    from analyze.load import detect_position_bias

    # Records using preferred_id (plan ID) rather than "plan_a"/"plan_b" labels
    df = pd.DataFrame([
        {"judge": "j", "preferred_id": "plan_llm_001", "plan_a_id": "plan_llm_001"},
        {"judge": "j", "preferred_id": "plan_llm_001", "plan_a_id": "plan_llm_001"},
        {"judge": "j", "preferred_id": "plan_prog_001", "plan_a_id": "plan_llm_001"},  # chose plan_b (position B)
        {"judge": "j", "preferred_id": "plan_llm_001", "plan_a_id": "plan_llm_001"},
    ])

    result = detect_position_bias(df, threshold=0.2)
    # 3/4 = 0.75 prefer position_a → |0.75-0.5| = 0.25 ≥ 0.2 → biased
    assert result["biased"] is True, f"Expected biased, got {result}"
    assert abs(result["p_prefers_a"] - 0.75) < 0.01

    # Balanced case
    df2 = pd.DataFrame([
        {"judge": "j", "preferred_id": "plan_llm_001", "plan_a_id": "plan_llm_001"},
        {"judge": "j", "preferred_id": "plan_prog_001", "plan_a_id": "plan_llm_001"},
        {"judge": "j", "preferred_id": "plan_llm_001", "plan_a_id": "plan_llm_001"},
        {"judge": "j", "preferred_id": "plan_prog_001", "plan_a_id": "plan_llm_001"},
    ])
    result2 = detect_position_bias(df2, threshold=0.2)
    assert result2["biased"] is False, f"Expected unbiased (50/50), got {result2}"


# ═══════════════════════════════════════════════════════════════════════════════
# Test 37 — score_plan falls back to heuristic for unreadable files
# ═══════════════════════════════════════════════════════════════════════════════

@test("40 score_plan_fallback_missing_file")
def test_37_score_plan_fallback() -> None:
    from match.pair import score_plan, _heuristic_score
    import json

    tmp = _make_tmp()

    # Missing file → heuristic returns 0.0
    s = score_plan(tmp / "nonexistent.json")
    assert s == 0.0, f"Expected 0.0 for missing file, got {s}"

    # Valid plan → real scorer or heuristic, both must be float ≥ 0
    from tests.mock_llm_client import _training_plan_payload
    p = tmp / "plan.json"
    p.write_text(json.dumps(_training_plan_payload()))
    s2 = score_plan(p)
    assert isinstance(s2, float) and s2 >= 0.0, f"Expected float≥0, got {s2}"

    # _heuristic_score with a rich plan returns > 0
    h = _heuristic_score(p)
    assert h > 0.0, f"Heuristic should be > 0 for non-empty plan, got {h}"


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ok = _run_all()
    sys.exit(0 if ok else 1)
