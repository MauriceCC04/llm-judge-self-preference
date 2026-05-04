from __future__ import annotations

import re
from pathlib import Path

ROOT = Path.cwd()


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def write(path: str, text: str) -> None:
    (ROOT / path).write_text(text, encoding="utf-8")


def replace_once(text: str, old: str, new: str, path: str) -> str:
    n = text.count(old)
    if n != 1:
        raise RuntimeError(f"Expected exactly one occurrence in {path}, found {n}:\n{old}")
    return text.replace(old, new, 1)


def regex_replace_once(text: str, pattern: str, replacement: str, path: str) -> str:
    new_text, n = re.subn(pattern, replacement, text, count=1, flags=re.DOTALL)
    if n != 1:
        raise RuntimeError(f"Expected exactly one regex occurrence in {path}, found {n}:\n{pattern}")
    return new_text


# ---------------------------------------------------------------------------
# 1. Strengthen structural validator against near-rest advanced plans.
# ---------------------------------------------------------------------------
path = "generate/structural_expectations.py"
text = read(path)

old_advanced_line = '''    if band in {"A3", "A4"}:
        lines.append(
            "This cell expects a genuinely structured week, not an all-easy or all-rest template. Include at least one day with session_type='long'. Low readiness should reduce intensity or duration, not collapse the entire week to rest unless deterministic safety constraints explicitly require that."
        )
'''
new_advanced_line = '''    if band in {"A3", "A4"}:
        lines.append(
            "This cell expects a genuinely structured week, not an all-easy or all-rest template. Include at least one day with session_type='long' and at least three active non-rest training days. Low readiness should reduce intensity or duration, not collapse the entire week to rest unless deterministic safety constraints explicitly require that."
        )
'''
if old_advanced_line in text:
    text = replace_once(text, old_advanced_line, new_advanced_line, path)
elif "at least three active non-rest training days" not in text:
    raise RuntimeError("Could not find advanced-cell structural prompt line to strengthen.")

insert_after = '''    issues: list[str] = []

'''
min_active_block = '''    min_active_days: int | None = None
    if band in {"A3", "A4"}:
        min_active_days = 3 if readiness == "low" else 4
    elif band == "A2" and readiness == "high" and recovery == "high":
        min_active_days = 2

    if min_active_days is not None and len(non_rest_days) < min_active_days:
        issues.append(f"understructured:too_few_active_days:{len(non_rest_days)}<{min_active_days}")

'''
if "understructured:too_few_active_days" not in text:
    text = replace_once(text, insert_after, insert_after + min_active_block, path)
write(path, text)


# ---------------------------------------------------------------------------
# 2. Add fixture top-up / retry logic to generate.run_generation.
# ---------------------------------------------------------------------------
path = "generate/run_generation.py"
text = read(path)

new_run_llm_arm = r'''def run_llm_arm(
    output_dir: Path,
    *,
    plans_per_fixture: int,
    source_model: str,
    seed_offset: int = 0,
    fixture_ids: list[str] | None = None,
    source_temperature: float = 0.7,
    explainer_temperature: float = 0.0,
    max_attempts_per_fixture: int | None = None,
) -> tuple[int, int]:
    from generate.llm_arm import generate_llm_plan

    specs = _resolve_fixture_specs(fixture_ids)
    output_dir.mkdir(parents=True, exist_ok=True)
    failures_path = output_dir / "failed_plans.jsonl"
    generated = 0
    skipped = 0
    attempts_per_fixture = max_attempts_per_fixture or plans_per_fixture
    if attempts_per_fixture < plans_per_fixture:
        raise ValueError(
            "max_attempts_per_fixture must be >= plans_per_fixture "
            f"({attempts_per_fixture} < {plans_per_fixture})"
        )
    generation_condition = build_llm_generation_condition(
        source_temperature=source_temperature,
        explainer_temperature=explainer_temperature,
    )

    for idx, spec in enumerate(specs, start=1):
        fixture_dir = _ROOT / "fixtures" / "data" / spec.fixture_id
        if not fixture_dir.exists():
            print(f"  [warn] fixture dir not found: {fixture_dir}", file=sys.stderr)
            continue

        accepted_for_fixture = 0
        for candidate_idx in range(attempts_per_fixture):
            if accepted_for_fixture >= plans_per_fixture:
                break

            actual_seed = seed_offset + candidate_idx
            plan_id = build_llm_plan_id(
                fixture_id=spec.fixture_id,
                source_model=source_model,
                seed=actual_seed,
                source_temperature=source_temperature,
                explainer_temperature=explainer_temperature,
            )
            if _plan_outputs_exist(output_dir, plan_id):
                skipped += 1
                accepted_for_fixture += 1
                continue

            print(
                f"  [fixture {idx}/{len(specs)} attempt {candidate_idx + 1}/{attempts_per_fixture} "
                f"accepted {accepted_for_fixture}/{plans_per_fixture}] {plan_id} ...",
                end="",
                flush=True,
            )
            try:
                result = generate_llm_plan(
                    fixture_dir=fixture_dir,
                    output_dir=output_dir,
                    plan_id=plan_id,
                    source_model=source_model,
                    seed=actual_seed,
                    source_temperature=source_temperature,
                    explainer_temperature=explainer_temperature,
                    generation_condition=generation_condition,
                )
                _verify_generated_result(plan_id, result)
                print(" OK")
                generated += 1
                accepted_for_fixture += 1
            except Exception as exc:  # pragma: no cover - failure logging path
                print(f" FAILED: {exc}")
                _write_failure(failures_path, plan_id, exc)

        if accepted_for_fixture < plans_per_fixture:
            print(
                f"  [warn] fixture {spec.fixture_id} accepted {accepted_for_fixture}/"
                f"{plans_per_fixture} valid plans after {attempts_per_fixture} attempts",
                file=sys.stderr,
            )
    return generated, skipped

'''

new_run_programmatic_arm = r'''def run_programmatic_arm(
    output_dir: Path,
    *,
    plans_per_fixture: int,
    sampler_config_path: Path | None = None,
    seed_offset: int = 0,
    fixture_ids: list[str] | None = None,
    explainer_temperature: float = 0.0,
    max_attempts_per_fixture: int | None = None,
) -> tuple[int, int]:
    from generate.fit_priors import load_sampler_config
    from generate.programmatic_arm import generate_programmatic_plan

    specs = _resolve_fixture_specs(fixture_ids)
    output_dir.mkdir(parents=True, exist_ok=True)
    failures_path = output_dir / "failed_plans.jsonl"
    base_cfg = load_sampler_config(sampler_config_path) if sampler_config_path and sampler_config_path.exists() else None
    generated = 0
    skipped = 0
    attempts_per_fixture = max_attempts_per_fixture or plans_per_fixture
    if attempts_per_fixture < plans_per_fixture:
        raise ValueError(
            "max_attempts_per_fixture must be >= plans_per_fixture "
            f"({attempts_per_fixture} < {plans_per_fixture})"
        )
    generation_condition = build_programmatic_generation_condition(
        explainer_temperature=explainer_temperature,
    )

    for idx, spec in enumerate(specs, start=1):
        fixture_dir = _ROOT / "fixtures" / "data" / spec.fixture_id
        if not fixture_dir.exists():
            print(f"  [warn] fixture dir not found: {fixture_dir}", file=sys.stderr)
            continue

        accepted_for_fixture = 0
        for candidate_idx in range(attempts_per_fixture):
            if accepted_for_fixture >= plans_per_fixture:
                break

            actual_seed = seed_offset + candidate_idx
            plan_id = build_programmatic_plan_id(
                fixture_id=spec.fixture_id,
                seed=actual_seed,
                explainer_temperature=explainer_temperature,
            )
            if _plan_outputs_exist(output_dir, plan_id):
                skipped += 1
                accepted_for_fixture += 1
                continue

            print(
                f"  [fixture {idx}/{len(specs)} attempt {candidate_idx + 1}/{attempts_per_fixture} "
                f"accepted {accepted_for_fixture}/{plans_per_fixture}] {plan_id} ...",
                end="",
                flush=True,
            )
            try:
                result = generate_programmatic_plan(
                    fixture_dir=fixture_dir,
                    output_dir=output_dir,
                    plan_id=plan_id,
                    seed=actual_seed,
                    sampler_cfg=base_cfg,
                    explainer_temperature=explainer_temperature,
                    generation_condition=generation_condition,
                )
                _verify_generated_result(plan_id, result)
                print(" OK")
                generated += 1
                accepted_for_fixture += 1
            except Exception as exc:  # pragma: no cover - failure logging path
                print(f" FAILED: {exc}")
                _write_failure(failures_path, plan_id, exc)

        if accepted_for_fixture < plans_per_fixture:
            print(
                f"  [warn] fixture {spec.fixture_id} accepted {accepted_for_fixture}/"
                f"{plans_per_fixture} valid plans after {attempts_per_fixture} attempts",
                file=sys.stderr,
            )
    return generated, skipped

'''

text = regex_replace_once(
    text,
    r"def run_llm_arm\(.*?\n\ndef run_programmatic_arm\(",
    new_run_llm_arm + "def run_programmatic_arm(",
    path,
)
text = regex_replace_once(
    text,
    r"def run_programmatic_arm\(.*?\n\ndef _split_fixture_ids\(",
    new_run_programmatic_arm + "def _split_fixture_ids(",
    path,
)

parser_anchor = '''    parser.add_argument("--explainer-temperature", type=float, default=0.0)
'''
parser_insert = '''    parser.add_argument("--explainer-temperature", type=float, default=0.0)
    parser.add_argument(
        "--max-attempts-per-fixture",
        type=int,
        default=None,
        help=(
            "Maximum candidate seeds to try per fixture while filling --plans-per-fixture "
            "valid saved artifacts. Defaults to --plans-per-fixture."
        ),
    )
'''
if "--max-attempts-per-fixture" not in text:
    text = replace_once(text, parser_anchor, parser_insert, path)

text = replace_once(
    text,
    '''            explainer_temperature=args.explainer_temperature,
        )
''',
    '''            explainer_temperature=args.explainer_temperature,
            max_attempts_per_fixture=args.max_attempts_per_fixture,
        )
''',
    path,
)
text = replace_once(
    text,
    '''            explainer_temperature=args.explainer_temperature,
        )

    elapsed''',
    '''            explainer_temperature=args.explainer_temperature,
            max_attempts_per_fixture=args.max_attempts_per_fixture,
        )

    elapsed''',
    path,
)
write(path, text)


# ---------------------------------------------------------------------------
# 3. Wire the retry option through the Slurm wrapper.
# ---------------------------------------------------------------------------
path = "slurm/run_generation_hpc.sh"
text = read(path)

anchor = '''EXPLAINER_TEMPERATURE="${EXPLAINER_TEMPERATURE:-0.0}"
PRIOR_FIT_MIN_PLANS="${PRIOR_FIT_MIN_PLANS:-30}"
'''
insert = '''EXPLAINER_TEMPERATURE="${EXPLAINER_TEMPERATURE:-0.0}"
MAX_ATTEMPTS_PER_FIXTURE="${MAX_ATTEMPTS_PER_FIXTURE:-${GENERATION_MAX_ATTEMPTS_PER_FIXTURE:-}}"
PRIOR_FIT_MIN_PLANS="${PRIOR_FIT_MIN_PLANS:-30}"
'''
if "MAX_ATTEMPTS_PER_FIXTURE" not in text:
    text = replace_once(text, anchor, insert, path)

anchor = '''echo "  explainer_temperature:   ${EXPLAINER_TEMPERATURE}"
if [[ "${GENERATION_ARM}" == "llm" ]]; then
'''
insert = '''echo "  explainer_temperature:   ${EXPLAINER_TEMPERATURE}"
echo "  max_attempts_per_fixture:${MAX_ATTEMPTS_PER_FIXTURE:-default}"
if [[ "${GENERATION_ARM}" == "llm" ]]; then
'''
if "max_attempts_per_fixture" not in text:
    text = replace_once(text, anchor, insert, path)

fixture_args_anchor = '''FIXTURE_ARGS=()
if [[ -n "${FIXTURE_IDS}" ]]; then
    FIXTURE_ARGS+=(--fixture-id "${FIXTURE_IDS}")
fi

if [[ "${GENERATION_ARM}" == "llm" ]]; then
'''
fixture_args_insert = '''FIXTURE_ARGS=()
if [[ -n "${FIXTURE_IDS}" ]]; then
    FIXTURE_ARGS+=(--fixture-id "${FIXTURE_IDS}")
fi
ATTEMPT_ARGS=()
if [[ -n "${MAX_ATTEMPTS_PER_FIXTURE:-}" ]]; then
    ATTEMPT_ARGS+=(--max-attempts-per-fixture "${MAX_ATTEMPTS_PER_FIXTURE}")
fi

if [[ "${GENERATION_ARM}" == "llm" ]]; then
'''
if "ATTEMPT_ARGS" not in text:
    text = replace_once(text, fixture_args_anchor, fixture_args_insert, path)

text = replace_once(
    text,
    '''        "${FIXTURE_ARGS[@]}" \\
        --output "${PLANS_DIR}"
''',
    '''        "${FIXTURE_ARGS[@]}" \\
        "${ATTEMPT_ARGS[@]}" \\
        --output "${PLANS_DIR}"
''',
    path,
)
text = replace_once(
    text,
    '''        "${FIXTURE_ARGS[@]}" \\
        --output "${PLANS_DIR}"
fi
''',
    '''        "${FIXTURE_ARGS[@]}" \\
        "${ATTEMPT_ARGS[@]}" \\
        --output "${PLANS_DIR}"
fi
''',
    path,
)
write(path, text)


# ---------------------------------------------------------------------------
# 4. Add regression tests.
# ---------------------------------------------------------------------------
path = "tests/test_structural_expectations.py"
text = read(path)
struct_test = '''

def test_detect_understructured_plan_flags_too_few_active_days_for_advanced_week() -> None:
    plan_obj = _plan(
        [
            {"session_type": "long", "is_rest_day": False, "is_hard_day": False, "duration_minutes": 90},
            {"session_type": "tempo", "is_rest_day": False, "is_hard_day": True, "duration_minutes": 45},
            {"session_type": "rest", "is_rest_day": True, "is_hard_day": False, "duration_minutes": 0},
            {"session_type": "rest", "is_rest_day": True, "is_hard_day": False, "duration_minutes": 0},
            {"session_type": "rest", "is_rest_day": True, "is_hard_day": False, "duration_minutes": 0},
            {"session_type": "rest", "is_rest_day": True, "is_hard_day": False, "duration_minutes": 0},
            {"session_type": "rest", "is_rest_day": True, "is_hard_day": False, "duration_minutes": 0},
        ]
    )
    fixture_meta = {
        "athlete_band": "A4",
        "readiness": "low",
        "recovery_capability": "high",
        "race_phase": "base",
    }

    issues = detect_understructured_plan(plan_obj, fixture_meta)
    assert "understructured:too_few_active_days:2<3" in issues


def test_structural_lifestyle_notes_request_multiple_active_advanced_days() -> None:
    notes = build_structural_lifestyle_notes(
        {
            "athlete_band": "A4",
            "readiness": "low",
            "recovery_capability": "high",
            "race_phase": "base",
        }
    )
    assert "at least three active non-rest training days" in notes
'''
if "test_detect_understructured_plan_flags_too_few_active_days_for_advanced_week" not in text:
    text = text.rstrip() + struct_test + "\n"
write(path, text)

path = "tests/test_run_generation_topup.py"
if not (ROOT / path).exists():
    write(path, '''from __future__ import annotations

import json
from types import SimpleNamespace

import generate.run_generation as rg


def _minimal_plan() -> dict:
    return {
        "meta": {"style": "trailrunning"},
        "plan": {
            "weekly_totals": {},
            "days": [
                {
                    "date": "2026-03-11",
                    "title": "Easy run",
                    "session_type": "easy",
                    "is_rest_day": False,
                    "is_hard_day": False,
                    "duration_minutes": 30,
                    "target_intensity": "easy",
                    "terrain": "trail",
                    "workout": "30 min easy run.",
                    "purpose": "Maintain aerobic rhythm.",
                    "signal_ids": [],
                }
            ],
        },
    }


def test_run_llm_arm_tops_up_after_failed_candidate_seeds(monkeypatch, tmp_path):
    fixture_id = "ab_A4__r_low__rc_high__ph_base"
    fixture_dir = tmp_path / "fixtures" / "data" / fixture_id
    fixture_dir.mkdir(parents=True)

    monkeypatch.setattr(rg, "_ROOT", tmp_path)
    monkeypatch.setattr(
        rg,
        "_resolve_fixture_specs",
        lambda fixture_ids: [SimpleNamespace(fixture_id=fixture_id)],
    )

    import generate.llm_arm as llm_arm

    def fake_generate_llm_plan(*, output_dir, plan_id, seed, **kwargs):
        if seed in {0, 1}:
            raise RuntimeError(f"synthetic rejection seed={seed}")
        plan_path = output_dir / f"{plan_id}.json"
        prov_path = output_dir / f"{plan_id}.json.provenance.json"
        plan_obj = _minimal_plan()
        plan_path.write_text(json.dumps(plan_obj), encoding="utf-8")
        prov_path.write_text(json.dumps({"plan_id": plan_id}), encoding="utf-8")
        return json.dumps(plan_obj), str(plan_path), str(prov_path)

    monkeypatch.setattr(llm_arm, "generate_llm_plan", fake_generate_llm_plan)

    generated, skipped = rg.run_llm_arm(
        tmp_path / "plans",
        plans_per_fixture=2,
        source_model="Qwen/Qwen2.5-7B-Instruct",
        seed_offset=0,
        fixture_ids=[fixture_id],
        max_attempts_per_fixture=4,
    )

    assert generated == 2
    assert skipped == 0
    assert len(list((tmp_path / "plans").glob("*.json"))) == 4  # 2 plans + 2 provenance sidecars
    failures = (tmp_path / "plans" / "failed_plans.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(failures) == 2
''')

print("Patched active-day guard and validated top-up generation.")
print("Run:")
print("  python tests/run_tests.py")
print("  pytest tests/test_structural_expectations.py tests/test_run_generation_topup.py -v")
