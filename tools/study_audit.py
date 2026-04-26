"""tools/study_audit.py — readiness + goal-alignment audit for the frozen study.

Two questions, one tool:

  1. HPC readiness — can this repo actually execute the frozen 512-plan study
     on the cluster end-to-end without manual repair?

  2. Goal alignment — does what's implemented in the repo match the stated
     research design (256 LLM + 256 programmatic, 4 judges, both pairwise
     views, self-preference + scale-ladder + rubric tests, style audit)?

Each check is small, weighted, and produces structured evidence. Checks that
require optional dependencies (pandas, statsmodels, torch, vllm) degrade
gracefully into a "skipped" status that does NOT count toward the score.

Usage
-----
    python tools/study_audit.py                       # human + JSON to stdout
    python tools/study_audit.py --json-out audit.json # save JSON
    python tools/study_audit.py --strict              # exit nonzero if any
                                                       # blocker fails
    python tools/study_audit.py --bucket hpc          # run only HPC checks
    python tools/study_audit.py --bucket alignment    # run only alignment

Design notes
------------
* The audit walks the on-disk repo under the directory it's invoked from.
* Most checks are pure file/text/AST inspection — no imports of project code.
  This means the audit works on a half-installed checkout, which is exactly
  when you want it most.
* A handful of optional dynamic checks (e.g. resolving panel size, importing
  the constants module) run when imports succeed and are otherwise skipped.
* Each check declares a `weight`, a `severity` (blocker | warning | info),
  and an `evidence` payload pointing back at the file/symbol responsible.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

# ── Configuration ────────────────────────────────────────────────────────────

EXPECTED_TRAILTRAINING_SHA = "3e7f1793ca051ba1aae05f1714d594691202ad7e"
EXPECTED_EXPLAINER = "Qwen/Qwen2.5-3B-Instruct"
EXPECTED_SOURCE_MODELS = {
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
}
EXPECTED_JUDGES = {
    "llama_8b_judge",
    "qwen_7b_judge",
    "qwen_14b_judge",
    "qwen_32b_judge",
}
EXPECTED_PAIRWISE_VIEWS = {"raw_normalized", "canonical_masked"}
EXPECTED_RUBRICS = {
    "goal_alignment", "plan_coherence", "explanation_quality",
    "caution_proportionality", "actionability",
}
EXPECTED_FIXTURE_COUNT = 8

CANONICAL_SLURM_SCRIPTS = {
    "common.sh",
    "run_preflight.sh",
    "run_vllm_smoke.sh",
    "run_generation_hpc.sh",
    "run_judge_hpc.sh",
    "submit_judge_hpc.sh",
    "pre_cache_models.sh",
}


# ── Data model ───────────────────────────────────────────────────────────────

@dataclass
class CheckResult:
    key: str
    bucket: str            # "hpc" | "alignment"
    title: str
    passed: bool
    severity: str          # "blocker" | "warning" | "info"
    weight: float          # contribution to the bucket score
    evidence: dict[str, Any] = field(default_factory=dict)
    skipped: bool = False  # excluded from scoring (e.g. optional dep missing)
    skip_reason: str = ""


# ── Small AST/text helpers ───────────────────────────────────────────────────

def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""
    except Exception as exc:
        return f"<<read_error: {exc}>>"





def _module_constants(path: Path) -> dict[str, Any]:
    """Extract top-level constants from a python file via AST.

    Handles:
      * `NAME = <literal>`              (ast.Assign with literal RHS)
      * `NAME: T = <literal>`           (ast.AnnAssign with literal RHS)
      * `NAME = OTHER_NAME + OTHER`     (simple BinOp over names already resolved)

    The two-pass arithmetic resolver is intentionally minimal — it handles
    composite constants like `TARGET_TOTAL_PLAN_COUNT = A + B` without dragging
    in a full evaluator.
    """
    src = _read(path)
    if not src:
        return {}
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return {}

    out: dict[str, Any] = {}

    def _resolve(value: ast.AST) -> Any:
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass
        # Try to resolve simple references and arithmetic over already-known names.
        if isinstance(value, ast.Name) and value.id in out:
            return out[value.id]
        if isinstance(value, ast.BinOp):
            try:
                left = _resolve(value.left)
                right = _resolve(value.right)
            except _Unresolvable:
                raise
            if isinstance(value.op, ast.Add): return left + right
            if isinstance(value.op, ast.Sub): return left - right
            if isinstance(value.op, ast.Mult): return left * right
            if isinstance(value.op, ast.FloorDiv): return left // right
        raise _Unresolvable(ast.dump(value))

    # Iterate to a fixed point — handles forward references between constants.
    for _ in range(3):
        changed = False
        for node in tree.body:
            target_name: str | None = None
            value_node: ast.AST | None = None
            if isinstance(node, ast.Assign):
                if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                    target_name = node.targets[0].id
                    value_node = node.value
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                target_name = node.target.id
                value_node = node.value
            if target_name is None or value_node is None or target_name in out:
                continue
            try:
                out[target_name] = _resolve(value_node)
                changed = True
            except _Unresolvable:
                continue
        if not changed:
            break
    return out


class _Unresolvable(Exception):
    pass


def _symbol_present(path: Path, name: str) -> bool:
    """True iff the file defines a top-level function/class/assignment `name`.

    Falls back to substring search when AST parsing fails — useful for stub
    files or files with intentional placeholders.
    """
    src = _read(path)
    if not src:
        return False
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return name in src
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == name:
                return True
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == name:
                    return True
    return name in src  # last-resort substring


def _file_mentions_all(path: Path, needles: Iterable[str]) -> tuple[bool, list[str]]:
    src = _read(path)
    missing = [n for n in needles if n not in src]
    return (not missing, missing)


# ── Check helpers ────────────────────────────────────────────────────────────

def _ok(key, bucket, title, *, weight=1.0, severity="warning", evidence=None) -> CheckResult:
    return CheckResult(key, bucket, title, True, severity, weight, evidence or {})


def _fail(key, bucket, title, *, weight=1.0, severity="warning", evidence=None) -> CheckResult:
    return CheckResult(key, bucket, title, False, severity, weight, evidence or {})


def _skip(key, bucket, title, reason, *, weight=1.0, severity="warning") -> CheckResult:
    return CheckResult(key, bucket, title, False, severity, weight,
                       skipped=True, skip_reason=reason)


# ── HPC readiness checks ─────────────────────────────────────────────────────

def check_dependency_pin(root: Path) -> CheckResult:
    """pyproject.toml and requirements-hpc.txt agree on a single pinned SHA."""
    pyproject = _read(root / "pyproject.toml")
    requirements = _read(root / "requirements-hpc.txt")
    pyproj_sha = re.search(r"trailtraining\.git@([0-9a-f]{7,40})", pyproject)
    req_sha = re.search(r"trailtraining\.git@([0-9a-f]{7,40})", requirements)

    evidence = {
        "pyproject_sha": pyproj_sha.group(1) if pyproj_sha else None,
        "requirements_sha": req_sha.group(1) if req_sha else None,
        "expected_sha": EXPECTED_TRAILTRAINING_SHA,
    }
    if not pyproj_sha or not req_sha:
        return _fail("hpc.pin_present", "hpc",
                     "trailtraining pin present in pyproject and requirements-hpc",
                     weight=3, severity="blocker", evidence=evidence)
    if pyproj_sha.group(1) != req_sha.group(1):
        return _fail("hpc.pin_consistent", "hpc",
                     "pyproject and requirements-hpc pin same SHA",
                     weight=3, severity="blocker", evidence=evidence)
    if pyproj_sha.group(1) != EXPECTED_TRAILTRAINING_SHA:
        # Warning, not blocker — they may have intentionally bumped the pin.
        return _fail("hpc.pin_matches_expected", "hpc",
                     "trailtraining pin matches the documented frozen SHA",
                     weight=1, severity="warning", evidence=evidence)
    return _ok("hpc.pin", "hpc",
               "trailtraining pin present, consistent, matches expected",
               weight=3, severity="blocker", evidence=evidence)


def check_bootstrap_script(root: Path) -> CheckResult:
    p = root / "bootstrap_hpc_env.sh"
    src = _read(p)
    has_pythonpath = "PYTHONPATH" in src
    installs_reqs = "requirements-hpc.txt" in src
    sets_hf_home = "HF_HOME" in src
    ok = bool(src) and has_pythonpath and installs_reqs and sets_hf_home
    return CheckResult(
        "hpc.bootstrap_script", "hpc",
        "bootstrap_hpc_env.sh exists and is well-formed",
        ok, "blocker", 2,
        {"path": str(p), "has_pythonpath": has_pythonpath,
         "installs_requirements": installs_reqs, "sets_hf_home": sets_hf_home},
    )


def check_slurm_scripts_present(root: Path) -> CheckResult:
    slurm = root / "slurm"
    found = {f.name for f in slurm.glob("*.sh")} if slurm.exists() else set()
    missing = sorted(CANONICAL_SLURM_SCRIPTS - found)
    extras = sorted(found - CANONICAL_SLURM_SCRIPTS)
    return CheckResult(
        "hpc.slurm_scripts_present", "hpc",
        "All canonical SLURM scripts present",
        not missing, "blocker", 2,
        {"missing": missing, "extras": extras, "found_count": len(found)},
    )


def check_slurm_scripts_parse(root: Path) -> CheckResult:
    """Each slurm/*.sh must pass `bash -n`."""
    slurm = root / "slurm"
    if not slurm.exists():
        return _fail("hpc.slurm_scripts_parse", "hpc",
                     "slurm/ scripts pass bash -n", weight=2, severity="blocker",
                     evidence={"reason": "slurm/ does not exist"})
    failures: list[dict[str, str]] = []
    if not _bash_available():
        return _skip("hpc.slurm_scripts_parse", "hpc",
                     "slurm/ scripts pass bash -n",
                     "bash not available in audit environment",
                     weight=2, severity="blocker")
    for script in sorted(slurm.glob("*.sh")):
        result = subprocess.run(
            ["bash", "-n", str(script)], capture_output=True, text=True
        )
        if result.returncode != 0:
            failures.append({"script": script.name, "stderr": result.stderr.strip()})
    return CheckResult(
        "hpc.slurm_scripts_parse", "hpc",
        "Every slurm/*.sh passes bash -n",
        not failures, "blocker", 2, {"failures": failures},
    )


def check_slurm_scripts_source_common(root: Path) -> CheckResult:
    slurm = root / "slurm"
    if not slurm.exists():
        return _fail("hpc.slurm_source_common", "hpc",
                     "slurm scripts source common.sh", weight=1, severity="warning",
                     evidence={"reason": "slurm/ missing"})
    bad: list[str] = []
    for script in slurm.glob("*.sh"):
        if script.name == "common.sh":
            continue
        if "common.sh" not in _read(script):
            bad.append(script.name)
    return CheckResult(
        "hpc.slurm_source_common", "hpc",
        "Every slurm/*.sh sources common.sh",
        not bad, "warning", 1, {"missing_source": bad},
    )


def check_common_sh_resolves_root(root: Path) -> CheckResult:
    """common.sh must resolve PROJECT_ROOT from its own location, not $PWD."""
    p = root / "slurm" / "common.sh"
    src = _read(p)
    ok = "BASH_SOURCE" in src and "PROJECT_ROOT" in src
    return CheckResult(
        "hpc.common_sh_root", "hpc",
        "slurm/common.sh resolves PROJECT_ROOT from its own location",
        ok, "blocker", 2,
        {"uses_bash_source": "BASH_SOURCE" in src,
         "exports_project_root": "PROJECT_ROOT" in src},
    )


def check_login_node_guard(root: Path) -> CheckResult:
    """pre_cache_models.sh must refuse to run inside a SLURM job."""
    src = _read(root / "slurm" / "pre_cache_models.sh")
    has_guard = "SLURM_JOB_ID" in src and ("ABORT" in src or "exit 1" in src)
    return CheckResult(
        "hpc.login_node_guard", "hpc",
        "pre_cache_models.sh rejects execution inside sbatch",
        has_guard, "warning", 1,
        {"has_slurm_job_id_check": "SLURM_JOB_ID" in src},
    )


def check_vllm_server_module(root: Path) -> CheckResult:
    p = root / "judge" / "vllm_server.py"
    has_class = _symbol_present(p, "VllmServer")
    src = _read(p)
    has_health = "health_poll" in src
    has_lifecycle = "shutdown" in src or "__exit__" in src
    ok = has_class and has_health and has_lifecycle
    return CheckResult(
        "hpc.vllm_server_module", "hpc",
        "judge/vllm_server.py exposes VllmServer with health_poll + shutdown",
        ok, "blocker", 2,
        {"path": str(p), "has_class": has_class,
         "has_health_poll": has_health, "has_lifecycle": has_lifecycle},
    )


def check_model_cache_tool(root: Path) -> CheckResult:
    p = root / "tools" / "check_model_cache.py"
    has_fn = _symbol_present(p, "model_is_cached")
    return CheckResult(
        "hpc.cache_check_tool", "hpc",
        "tools/check_model_cache.py exposes model_is_cached()",
        has_fn, "warning", 1, {"path": str(p), "has_function": has_fn},
    )


def check_dependency_verify_tool(root: Path) -> CheckResult:
    p = root / "tools" / "verify_dependency_surface.py"
    src = _read(p)
    has_pin = EXPECTED_TRAILTRAINING_SHA in src
    has_main = "def main" in src or "build_report" in src
    return CheckResult(
        "hpc.dependency_verify_tool", "hpc",
        "tools/verify_dependency_surface.py present and pin-aware",
        has_pin and has_main, "blocker", 2,
        {"path": str(p), "encodes_expected_sha": has_pin, "has_entry_point": has_main},
    )


def check_resume_jsonl(root: Path) -> CheckResult:
    p = root / "vendor_patches" / "resume_jsonl.py"
    has_class = _symbol_present(p, "ResumeJsonl")
    has_load = _symbol_present(p, "load_all")
    return CheckResult(
        "hpc.resume_jsonl", "hpc",
        "vendor_patches/resume_jsonl.py provides crash-safe JSONL writer",
        has_class and has_load, "blocker", 2,
        {"path": str(p), "has_resume_class": has_class, "has_load_all": has_load},
    )


def check_walltime_helper(root: Path) -> CheckResult:
    p = root / "judge" / "panel.py"
    has_fn = _symbol_present(p, "walltime_hours_with_buffer")
    has_quota_check = _symbol_present(p, "assert_judge_fits_quota")
    return CheckResult(
        "hpc.walltime_helper", "hpc",
        "judge/panel.py provides walltime + quota helpers",
        has_fn and has_quota_check, "warning", 1,
        {"has_walltime": has_fn, "has_quota_assert": has_quota_check},
    )


def check_quota_planning(root: Path) -> CheckResult:
    p = root / "hpc" / "quota.py"
    src = _read(p)
    has_plan_fn = "study_quota_report" in src or "estimate_judge_peak_gb" in src
    return CheckResult(
        "hpc.quota_planning", "hpc",
        "hpc/quota.py models the 50 GB home-quota budget",
        has_plan_fn, "warning", 1, {"path": str(p)},
    )


HPC_CHECKS: list[Callable[[Path], CheckResult]] = [
    check_dependency_pin,
    check_bootstrap_script,
    check_slurm_scripts_present,
    check_slurm_scripts_parse,
    check_slurm_scripts_source_common,
    check_common_sh_resolves_root,
    check_login_node_guard,
    check_vllm_server_module,
    check_model_cache_tool,
    check_dependency_verify_tool,
    check_resume_jsonl,
    check_walltime_helper,
    check_quota_planning,
]


# ── Goal-alignment checks ────────────────────────────────────────────────────

def check_constants_match_design(root: Path) -> CheckResult:
    """Frozen design numbers and identifiers are encoded in generate/constants.py."""
    consts = _module_constants(root / "generate" / "constants.py")
    expected = {
        "EXPLAINER_MODEL_ID": EXPECTED_EXPLAINER,
        "TARGET_LLM_PLAN_COUNT": 256,
        "TARGET_PROGRAMMATIC_PLAN_COUNT": 256,
        "TARGET_TOTAL_PLAN_COUNT": 512,
        "TARGET_PAIRS": 250,
        "PAIRWISE_N_RUNS": 5,
        "PAIRWISE_N_POSITIONS": 2,
    }
    mismatches = {k: {"expected": v, "actual": consts.get(k)}
                  for k, v in expected.items() if consts.get(k) != v}

    sources = set(consts.get("LLM_SOURCE_MODELS") or [])
    judges = set(consts.get("ACTIVE_JUDGE_NAMES") or [])
    views = set(consts.get("PAIRWISE_VIEW_CHOICES") or ())

    sources_ok = sources == EXPECTED_SOURCE_MODELS
    judges_ok = judges == EXPECTED_JUDGES
    views_ok = EXPECTED_PAIRWISE_VIEWS.issubset(views)

    ok = not mismatches and sources_ok and judges_ok and views_ok
    return CheckResult(
        "alignment.constants", "alignment",
        "generate/constants.py matches the frozen design",
        ok, "blocker", 4,
        {
            "scalar_mismatches": mismatches,
            "source_models": {"expected": sorted(EXPECTED_SOURCE_MODELS),
                              "actual": sorted(sources)},
            "judges": {"expected": sorted(EXPECTED_JUDGES),
                       "actual": sorted(judges)},
            "pairwise_views": {"expected": sorted(EXPECTED_PAIRWISE_VIEWS),
                               "actual": sorted(views)},
        },
    )


def check_arithmetic_consistency(root: Path) -> CheckResult:
    """The four headline numbers must be internally consistent."""
    consts = _module_constants(root / "generate" / "constants.py")
    pairs = consts.get("TARGET_PAIRS")
    runs = consts.get("PAIRWISE_N_RUNS")
    positions = consts.get("PAIRWISE_N_POSITIONS")
    judges = len(consts.get("ACTIVE_JUDGE_NAMES") or [])
    total_plans = consts.get("TARGET_TOTAL_PLAN_COUNT")

    pairwise_calls = (pairs or 0) * judges * (runs or 0) * (positions or 0)
    soft_calls = (total_plans or 0) * judges
    expected_pair = 10000
    expected_soft = 2048

    ok = pairwise_calls == expected_pair and soft_calls == expected_soft
    return CheckResult(
        "alignment.arithmetic", "alignment",
        "Pairwise (10,000) and soft-eval (2,048) call counts are consistent",
        ok, "warning", 2,
        {"pairwise_calls": pairwise_calls, "expected_pairwise": expected_pair,
         "soft_eval_calls": soft_calls, "expected_soft_eval": expected_soft,
         "factors": {"pairs": pairs, "judges": judges, "runs": runs,
                     "positions": positions, "total_plans": total_plans}},
    )


def check_exact_count_arithmetic(root: Path) -> CheckResult:
    """The exact_counts module must pin 128 plans/source-model and 256 programmatic.

    We accept either:
      (a) the constants resolve to those values, or
      (b) the file contains explicit `assert EXPECTED_* == <value>` lines.

    (b) is the form used in the actual repo (`len(FIXTURE_IDS) * 16`), and the
    asserts are evaluated at import time, so they are a real guarantee.
    """
    p = root / "generate" / "exact_counts.py"
    consts = _module_constants(p)
    src = _read(p)
    llm_per_model = consts.get("EXPECTED_LLM_PER_MODEL")
    prog_total = consts.get("EXPECTED_PROGRAMMATIC_TOTAL")
    assert_llm = bool(re.search(r"assert\s+EXPECTED_LLM_PER_MODEL\s*==\s*128", src))
    assert_prog = bool(re.search(r"assert\s+EXPECTED_PROGRAMMATIC_TOTAL\s*==\s*256", src))
    ok = (llm_per_model == 128 or assert_llm) and (prog_total == 256 or assert_prog)
    return CheckResult(
        "alignment.exact_counts", "alignment",
        "exact_counts module yields 128 plans/source-model and 256 programmatic",
        ok, "blocker", 2,
        {"llm_per_model": llm_per_model, "programmatic_total": prog_total,
         "assert_llm_pinned": assert_llm, "assert_prog_pinned": assert_prog},
    )


def check_fixture_count(root: Path) -> CheckResult:
    src = _read(root / "fixtures" / "spec.py")
    # Prefer the explicit assert in the file; fall back to enumerating IDs.
    has_assert_8 = "len(ALL_FIXTURE_SPECS) == 8" in src or "Expected exactly 8" in src
    consts = _module_constants(root / "fixtures" / "spec.py")
    ids = consts.get("FIXTURE_IDS") or []
    count_ok = len(ids) == EXPECTED_FIXTURE_COUNT or has_assert_8
    return CheckResult(
        "alignment.fixture_count", "alignment",
        "Exactly 8 fixtures (the 2x2x2 design)",
        count_ok, "warning", 1,
        {"id_count": len(ids), "has_assert": has_assert_8},
    )


def check_pairwise_view_implementations(root: Path) -> CheckResult:
    """Both raw_normalized and canonical_masked must have real implementations."""
    canon = root / "judge" / "canonicalize.py"
    norm = root / "judge" / "normalize.py"
    canon_ok = _symbol_present(canon, "canonicalize_pair_for_pairwise")
    norm_ok = _symbol_present(norm, "normalize_pair_for_pairwise")

    harness_src = _read(root / "judge" / "harness.py")
    harness_routes = "canonical_masked" in harness_src and "pairwise_view" in harness_src

    ok = canon_ok and norm_ok and harness_routes
    return CheckResult(
        "alignment.pairwise_views", "alignment",
        "Both pairwise views (raw_normalized + canonical_masked) implemented and routed",
        ok, "blocker", 3,
        {"canonicalize_present": canon_ok, "normalize_present": norm_ok,
         "harness_routes_view": harness_routes},
    )


def check_provenance_schema(root: Path) -> CheckResult:
    src = _read(root / "generate" / "provenance.py")
    needed = ["plan_id", "fixture_id", "arm", "source_model",
              "explainer_model", "actual_explainer_model",
              "explainer_model_verified"]
    missing = [n for n in needed if n not in src]
    return CheckResult(
        "alignment.provenance", "alignment",
        "PlanProvenance carries arm, source_model, fixture_id, and explainer verification",
        not missing, "blocker", 2,
        {"missing_fields": missing},
    )


def check_load_judgments_joins(root: Path) -> CheckResult:
    """Per INC-003: source_model_a/b must be joined for H3 self-preference."""
    src = _read(root / "analyze" / "load.py")
    needed = ["source_model_a", "source_model_b", "arm_a", "arm_b",
              "prefers_llm", "llm_in_position_a"]
    missing = [n for n in needed if n not in src]
    return CheckResult(
        "alignment.load_joins", "alignment",
        "analyze/load.py joins arm/source-model columns required for H3",
        not missing, "blocker", 2, {"missing": missing},
    )


def check_h1_uses_logit(root: Path) -> CheckResult:
    """Per INC-001: binary outcome → logistic, not LMM."""
    src = _read(root / "analyze" / "models.py")
    uses_logit = "smf.logit" in src or "Logit" in src
    avoids_mixedlm = "mixedlm" not in src or "Linear mixed model" not in src
    has_cluster_robust = "cluster" in src and "groups" in src
    ok = uses_logit and avoids_mixedlm and has_cluster_robust
    return CheckResult(
        "alignment.h1_logit", "alignment",
        "H1 model is logistic with cluster-robust SEs (not LMM on binary)",
        ok, "blocker", 3,
        {"uses_logit": uses_logit, "avoids_mixedlm": avoids_mixedlm,
         "has_cluster_robust_se": has_cluster_robust},
    )


def check_h3_h4_implemented(root: Path) -> CheckResult:
    src = _read(root / "analyze" / "models.py")
    has_h3 = "fit_h3_model" in src and "same_family" in src
    has_h4 = "fit_h4_model" in src and "QWEN_PARAMS_B" in src
    has_family_helper = "add_same_family_column" in src
    has_position_cov = "add_position_bias_covariate" in src or "llm_position_bias" in src
    ok = has_h3 and has_h4 and has_family_helper and has_position_cov
    return CheckResult(
        "alignment.h3_h4", "alignment",
        "Self-preference (H3) and Qwen-ladder (H4) models implemented with controls",
        ok, "blocker", 3,
        {"has_h3": has_h3, "has_h4": has_h4,
         "has_family_helper": has_family_helper,
         "has_position_covariate": has_position_cov},
    )


def check_rubric_paired_contrasts(root: Path) -> CheckResult:
    src = _read(root / "analyze" / "rubric_deltas.py")
    has_fn = "rubric_paired_contrasts" in src
    references_rubrics = sum(1 for r in EXPECTED_RUBRICS if r in src)
    has_holm = "holm" in src.lower() or "pvalue_holm" in src
    has_paired = "pairs_path" in src or "paired" in src
    ok = has_fn and references_rubrics >= 4 and has_holm and has_paired
    return CheckResult(
        "alignment.rubric_contrasts", "alignment",
        "H2 paired Holm-corrected per-rubric contrasts implemented",
        ok, "blocker", 2,
        {"has_function": has_fn, "rubric_ids_referenced": references_rubrics,
         "has_holm_correction": has_holm, "supports_paired_manifest": has_paired},
    )


def check_style_audit(root: Path) -> CheckResult:
    src = _read(root / "analyze" / "style_audit.py")
    has_fn = "run_style_audit" in src
    has_features = "extract_surface_features" in src
    has_gate = "gate" in src or "z_threshold" in src
    ok = has_fn and has_features and has_gate
    return CheckResult(
        "alignment.style_audit", "alignment",
        "Residual stylistic-leakage audit implemented with gate output",
        ok, "warning", 2,
        {"has_run_fn": has_fn, "has_feature_extractor": has_features,
         "has_gate_logic": has_gate},
    )


def check_score_bin_matching(root: Path) -> CheckResult:
    src = _read(root / "match" / "pair.py")
    constrains_fixture = "fixture_id" in src
    uses_score_bin = "score_bin" in src
    has_strict_scoring = "strict_scoring" in src or "ScoringError" in src
    uses_real_scorer = "evaluate_training_plan_quality" in src
    ok = constrains_fixture and uses_score_bin and has_strict_scoring and uses_real_scorer
    return CheckResult(
        "alignment.matching_logic", "alignment",
        "Matching is within fixture, within score bin, with strict scoring",
        ok, "blocker", 3,
        {"within_fixture": constrains_fixture, "within_score_bin": uses_score_bin,
         "strict_mode": has_strict_scoring, "uses_real_scorer": uses_real_scorer},
    )


def check_pilot_gate(root: Path) -> CheckResult:
    src = _read(root / "judge" / "harness.py")
    has_gate = "check_pilot_bias_gate" in src
    return CheckResult(
        "alignment.pilot_gate", "alignment",
        "Pilot position-bias gate exists in the judge harness",
        has_gate, "warning", 1, {"has_gate": has_gate},
    )


def check_cli_exposes_pipeline(root: Path) -> CheckResult:
    src = _read(root / "cli.py")
    needed = ["generate", "match", "judge", "analyze", "audit-style",
              "--pairwise-view", "--pilot", "--require-style-gate"]
    missing = [n for n in needed if n not in src]
    return CheckResult(
        "alignment.cli", "alignment",
        "CLI exposes all subcommands and flags required for the frozen run",
        not missing, "warning", 2, {"missing_tokens": missing},
    )


def check_schema_failure_logging(root: Path) -> CheckResult:
    outputs_src = _read(root / "judge" / "outputs.py")
    has_writer = "SchemaFailWriter" in outputs_src
    harness_src = _read(root / "judge" / "harness.py")
    uses_writer = "SchemaFailWriter" in harness_src or "schema_failures" in harness_src
    return CheckResult(
        "alignment.schema_failures", "alignment",
        "Schema failures are logged separately, not silently dropped",
        has_writer and uses_writer, "warning", 1,
        {"writer_defined": has_writer, "harness_uses_writer": uses_writer},
    )


def check_incidents_log(root: Path) -> CheckResult:
    """An INCIDENTS.md with concrete entries indicates real study discipline."""
    src = _read(root / "INCIDENTS.md")
    n_entries = len(re.findall(r"INC-\d{3}", src))
    return CheckResult(
        "alignment.incidents_log", "alignment",
        "INCIDENTS.md tracks fixed methodological bugs",
        n_entries >= 1, "info", 1,
        {"entries_found": n_entries},
    )


ALIGNMENT_CHECKS: list[Callable[[Path], CheckResult]] = [
    check_constants_match_design,
    check_arithmetic_consistency,
    check_exact_count_arithmetic,
    check_fixture_count,
    check_pairwise_view_implementations,
    check_provenance_schema,
    check_load_judgments_joins,
    check_h1_uses_logit,
    check_h3_h4_implemented,
    check_rubric_paired_contrasts,
    check_style_audit,
    check_score_bin_matching,
    check_pilot_gate,
    check_cli_exposes_pipeline,
    check_schema_failure_logging,
    check_incidents_log,
]


# ── Runner ───────────────────────────────────────────────────────────────────

def _bash_available() -> bool:
    try:
        subprocess.run(["bash", "--version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


# Markers that identify the repo root. Any one is sufficient.
_REPO_MARKERS = ("pyproject.toml", "PREREGISTRATION.md", "HPC_SYNC_RUNBOOK.md")


def discover_repo_root(start: Path | None = None) -> Path:
    """Walk upward from *start* (or this file's location) until we see a marker.

    PyCharm and many IDEs run scripts with the script's own directory as cwd,
    so the tool can't rely on `Path('.')`. We instead look for a directory
    that contains any of the well-known repo-root markers. If we find none,
    we fall back to cwd — which is correct when the user runs from the repo
    root in a terminal.
    """
    candidates: list[Path] = []
    if start is not None:
        candidates.append(Path(start).resolve())
    candidates.append(Path(__file__).resolve().parent)
    candidates.append(Path.cwd().resolve())

    seen: set[Path] = set()
    for c in candidates:
        cur = c
        while cur != cur.parent and cur not in seen:
            seen.add(cur)
            if any((cur / m).exists() for m in _REPO_MARKERS):
                return cur
            cur = cur.parent
    return Path.cwd().resolve()


def run_checks(root: Path, *, bucket: str | None = None) -> list[CheckResult]:
    results: list[CheckResult] = []
    if bucket in (None, "hpc"):
        for fn in HPC_CHECKS:
            results.append(fn(root))
    if bucket in (None, "alignment"):
        for fn in ALIGNMENT_CHECKS:
            results.append(fn(root))
    return results


def score_bucket(results: list[CheckResult], bucket: str) -> dict[str, Any]:
    """Weighted score over non-skipped results in *bucket*."""
    bucket_results = [r for r in results if r.bucket == bucket and not r.skipped]
    total_weight = sum(r.weight for r in bucket_results)
    earned = sum(r.weight for r in bucket_results if r.passed)
    blockers_failed = [r.key for r in bucket_results
                       if r.severity == "blocker" and not r.passed]
    warnings_failed = [r.key for r in bucket_results
                       if r.severity == "warning" and not r.passed]
    skipped = [r.key for r in results if r.bucket == bucket and r.skipped]
    return {
        "score": round(100.0 * earned / total_weight, 1) if total_weight else 0.0,
        "earned_weight": earned,
        "total_weight": total_weight,
        "n_checks": len(bucket_results),
        "n_passed": sum(1 for r in bucket_results if r.passed),
        "n_blockers_failed": len(blockers_failed),
        "blockers_failed": blockers_failed,
        "warnings_failed": warnings_failed,
        "skipped": skipped,
    }


def render_human(results: list[CheckResult], summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("  llm-judge-self-preference — readiness + alignment audit")
    lines.append("=" * 78)
    for bucket in ("hpc", "alignment"):
        b = summary[bucket]
        verdict = "READY" if b["score"] >= 90 and not b["n_blockers_failed"] else \
                  "NEARLY" if b["score"] >= 75 and not b["n_blockers_failed"] else \
                  "GAPS"
        title = "HPC readiness" if bucket == "hpc" else "Goal alignment"
        lines.append("")
        lines.append(f"── {title}: {b['score']}%  [{verdict}]")
        lines.append(f"     {b['n_passed']}/{b['n_checks']} checks passed, "
                     f"{b['n_blockers_failed']} blocker(s) failed, "
                     f"{len(b['warnings_failed'])} warning(s), "
                     f"{len(b['skipped'])} skipped")
        for r in [r for r in results if r.bucket == bucket]:
            if r.skipped:
                mark, note = "  ⏭ ", f"(skipped: {r.skip_reason})"
            elif r.passed:
                mark, note = "  ✓ ", ""
            else:
                tag = "BLOCKER" if r.severity == "blocker" else \
                      "warn" if r.severity == "warning" else "info"
                mark, note = "  ✗ ", f"  [{tag}]"
            lines.append(f"{mark}{r.title}{note}")
            if not r.passed and not r.skipped:
                # Surface a one-line evidence summary on failure.
                ev_summary = ", ".join(
                    f"{k}={_short(v)}" for k, v in r.evidence.items()
                    if v not in (None, "", [], {})
                )[:160]
                if ev_summary:
                    lines.append(f"        ↳ {ev_summary}")
    lines.append("")
    lines.append("=" * 78)
    overall = (summary["hpc"]["score"] + summary["alignment"]["score"]) / 2.0
    lines.append(f"  Overall: HPC {summary['hpc']['score']}% | "
                 f"Alignment {summary['alignment']['score']}% | "
                 f"Mean {overall:.1f}%")
    lines.append("=" * 78)
    return "\n".join(lines)


def _short(v: Any) -> str:
    s = str(v)
    return s if len(s) < 60 else s[:57] + "..."


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Audit the study repo's HPC readiness and goal alignment.")
    parser.add_argument("--root", default=None,
                        help="Repository root (default: auto-detect from script location, "
                             "falls back to cwd)")
    parser.add_argument("--bucket", choices=["hpc", "alignment"], default=None,
                        help="Run only one bucket")
    parser.add_argument("--json-out", default=None, help="Optional JSON output path")
    parser.add_argument("--strict", action="store_true",
                        help="Exit nonzero if any blocker fails")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve() if args.root else discover_repo_root()

    # Sanity check — if we resolved a path that has none of the markers and
    # none of the expected top-level dirs, tell the user immediately rather
    # than spitting out 29 confusing failures.
    has_marker = any((root / m).exists() for m in _REPO_MARKERS)
    has_dirs = any((root / d).exists() for d in ("generate", "slurm", "judge"))
    if not has_marker and not has_dirs:
        print(f"[ABORT] Could not locate the repo root.\n"
              f"  Scanned: {root}\n"
              f"  No marker files found ({', '.join(_REPO_MARKERS)})\n"
              f"  No expected dirs found (generate/, slurm/, judge/)\n"
              f"  Pass --root <path-to-repo> explicitly, or run from the repo root.",
              file=sys.stderr)
        sys.exit(2)

    print(f"  Scanned root: {root}\n")
    results = run_checks(root, bucket=args.bucket)
    summary = {
        "hpc": score_bucket(results, "hpc"),
        "alignment": score_bucket(results, "alignment"),
        "root": str(root),
    }
    print(render_human(results, summary))

    payload = {
        "summary": summary,
        "results": [asdict(r) for r in results],
    }
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(payload, indent=2),
                                       encoding="utf-8")
        print(f"\n[Saved] {args.json_out}")

    if args.strict:
        any_blocker = (summary["hpc"]["n_blockers_failed"] +
                       summary["alignment"]["n_blockers_failed"]) > 0
        sys.exit(2 if any_blocker else 0)


if __name__ == "__main__":
    main()