"""cli.py — judge-bias-study command-line interface."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(_ROOT))


def _split_fixture_ids(raw_values: list[str] | None) -> list[str] | None:
    if not raw_values:
        return None
    fixture_ids: list[str] = []
    for raw in raw_values:
        fixture_ids.extend(part.strip() for part in raw.split(",") if part.strip())
    return fixture_ids or None


def _load_style_gate_summary(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Required style-gate summary not found at {path}. Run `python cli.py audit-style ...` first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    gate = payload.get("gate")
    if not isinstance(gate, dict):
        raise RuntimeError(f"Style-gate summary at {path} does not contain a `gate` object.")
    return payload


def _ensure_style_gate(args: argparse.Namespace, summary_path: Path) -> dict:
    from analyze.style_audit import run_style_audit
    from generate.constants import STYLE_AUDIT_CRITICAL_FEATURES, STYLE_AUDIT_Z_THRESHOLD

    if summary_path.exists():
        return _load_style_gate_summary(summary_path)

    output_dir = summary_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_style_audit(
        plans_dir=Path(args.plans),
        provenance_dir=Path(args.provenance or args.plans),
        pairs_path=Path(args.pairs),
        output_dir=output_dir,
        critical_features=STYLE_AUDIT_CRITICAL_FEATURES,
        z_threshold=STYLE_AUDIT_Z_THRESHOLD,
    )
    return result


def cmd_generate(args: argparse.Namespace) -> None:
    from generate.constants import default_plans_per_fixture
    from generate.run_generation import run_llm_arm, run_programmatic_arm

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    plans_per_fixture = args.n if args.n is not None else default_plans_per_fixture(args.arm)
    fixture_ids = _split_fixture_ids(args.fixture_id)

    print(f"Arm: {args.arm}")
    print(f"Plans per fixture: {plans_per_fixture}")
    print(f"Seed offset: {args.seed_offset}")
    print(f"Fixture subset: {fixture_ids or 'ALL'}")
    print(f"Explainer temperature: {args.explainer_temperature}")

    if args.arm == "llm":
        if not args.source_model:
            raise SystemExit("--source-model is required for the llm arm")
        print(f"Source model: {args.source_model}")
        print(f"Source temperature: {args.source_temperature}")
        n_gen, n_skip = run_llm_arm(
            output_dir=output_dir,
            plans_per_fixture=plans_per_fixture,
            source_model=args.source_model,
            seed_offset=args.seed_offset,
            fixture_ids=fixture_ids,
            source_temperature=args.source_temperature,
            explainer_temperature=args.explainer_temperature,
        )
    else:
        n_gen, n_skip = run_programmatic_arm(
            output_dir=output_dir,
            plans_per_fixture=plans_per_fixture,
            sampler_config_path=Path(args.sampler_config) if args.sampler_config else None,
            seed_offset=args.seed_offset,
            fixture_ids=fixture_ids,
            explainer_temperature=args.explainer_temperature,
        )
    print(f"Generated {n_gen} plans, skipped {n_skip} (already existed) -> {output_dir}")


def cmd_fit_priors(args: argparse.Namespace) -> None:
    from generate.fit_priors import fit_and_save

    fit_and_save(
        plans_dir=Path(args.plans),
        output_path=Path(args.output),
        min_plans=args.min_plans,
        seed=args.seed,
    )


def cmd_match(args: argparse.Namespace) -> None:
    from generate.constants import MATCH_TOLERANCE, TARGET_PAIRS
    from match.pair import build_matched_pairs

    plans_dir = Path(args.plans)
    fixtures_dir = _ROOT / "fixtures" / "data"
    output_path = Path(args.output)
    pairs = build_matched_pairs(
        plans_dir=plans_dir,
        provenance_dir=plans_dir,
        fixtures_dir=fixtures_dir,
        output_path=output_path,
        tolerance=MATCH_TOLERANCE,
        target_pairs=TARGET_PAIRS,
        allow_mixed_generation_conditions=args.allow_mixed_generation_conditions,
    )
    print(f"{len(pairs)} pairs written to {output_path}")


def cmd_judge(args: argparse.Namespace) -> None:
    from generate.constants import (
        DEFAULT_STYLE_GATE_SUMMARY_PATH,
        PAIRWISE_N_POSITIONS,
        PAIRWISE_N_RUNS,
        PILOT_PAIR_LIMIT,
    )
    from generate.temperature import format_temperature_tag
    from judge.harness import check_pilot_bias_gate, run_pairwise_harness, run_soft_eval_harness
    from judge.panel import assert_judge_fits_quota, get_judge

    judge = get_judge(args.judge)
    assert_judge_fits_quota(judge)

    if args.require_style_gate:
        summary_path = Path(args.style_gate_summary or DEFAULT_STYLE_GATE_SUMMARY_PATH)
        summary = _ensure_style_gate(args, summary_path)
        gate = summary["gate"]
        if not gate.get("passed", False):
            flagged = ", ".join(gate.get("flagged_critical_features", [])) or "unknown critical features"
            raise SystemExit(f"Style gate failed at {summary_path}: {flagged}")
        print(f"Style gate passed: {summary_path}")

    plans_dir = Path(args.plans)
    pairs_file = Path(args.pairs)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    fixtures_dir = _ROOT / "fixtures" / "data"
    judge_temp_tag = format_temperature_tag(args.judge_temperature)

    if not args.skip_pairwise:
        pairs = json.loads(pairs_file.read_text(encoding="utf-8")) if pairs_file.exists() else []
        if args.pilot:
            pairs = pairs[:PILOT_PAIR_LIMIT]
            print(f"Pilot mode: {len(pairs)} pairs")
        else:
            if args.pair_offset:
                pairs = pairs[args.pair_offset :]
            if args.pair_limit is not None:
                pairs = pairs[: args.pair_limit]
            print(f"Pairwise subset: {len(pairs)} pairs (offset={args.pair_offset}, limit={args.pair_limit})")

        pairwise_out = output_dir / f"pairwise_{judge.name}_{args.pairwise_view}_{judge_temp_tag}.jsonl"
        run_pairwise_harness(
            pairs=pairs,
            plans_dir=plans_dir,
            judge=judge,
            rollups_path=None,
            fixtures_dir=fixtures_dir,
            output_path=pairwise_out,
            n_runs=PAIRWISE_N_RUNS,
            n_positions=PAIRWISE_N_POSITIONS,
            pairwise_view=args.pairwise_view,
            judge_temperature=args.judge_temperature,
        )

        if args.pilot:
            gate = check_pilot_bias_gate(pairwise_out)
            print(f"Pilot bias gate: {gate['message']}")
            if not gate["passed"]:
                print("  Judge excluded from H1/H2 — do not proceed to a full run.")
                raise SystemExit(2)
    else:
        print("Skipping pairwise harness")

    if args.skip_soft_eval:
        print("Skipping soft-eval harness")
    else:
        plan_ids = [
            plan_path.stem
            for plan_path in plans_dir.glob("*.json")
            if not plan_path.name.endswith(".provenance.json")
        ]
        if args.plan_offset:
            plan_ids = plan_ids[args.plan_offset :]
        if args.plan_limit is not None:
            plan_ids = plan_ids[: args.plan_limit]
        print(f"Soft-eval subset: {len(plan_ids)} plans (offset={args.plan_offset}, limit={args.plan_limit})")
        run_soft_eval_harness(
            plan_ids=plan_ids,
            plans_dir=plans_dir,
            judge=judge,
            rollups_path=None,
            fixtures_dir=fixtures_dir,
            provenance_dir=plans_dir,
            output_path=output_dir / f"softeval_{judge.name}_{judge_temp_tag}.jsonl",
            judge_temperature=args.judge_temperature,
        )
    print(f"Judge run complete: {judge.name}")


def cmd_audit_style(args: argparse.Namespace) -> None:
    from analyze.style_audit import run_style_audit
    from generate.constants import STYLE_AUDIT_CRITICAL_FEATURES, STYLE_AUDIT_Z_THRESHOLD

    result = run_style_audit(
        plans_dir=Path(args.plans),
        provenance_dir=Path(args.provenance or args.plans),
        pairs_path=Path(args.pairs),
        output_dir=Path(args.output),
        critical_features=STYLE_AUDIT_CRITICAL_FEATURES,
        z_threshold=STYLE_AUDIT_Z_THRESHOLD,
    )
    gate = result["gate"]
    print(f"Style gate passed: {gate['passed']}")
    if not gate["passed"]:
        print(f"Flagged critical features: {gate['flagged_critical_features']}")
        raise SystemExit(2)


def cmd_analyze(args: argparse.Namespace) -> None:
    from analyze.run_analysis import main as run_analysis_main

    argv = [
        "--judgments",
        args.judgments,
        "--plans",
        args.plans,
        "--output",
        args.output,
        "--pairs",
        args.pairs,
    ]
    if args.provenance:
        argv.extend(["--provenance", args.provenance])
    if args.pairwise_view:
        argv.extend(["--pairwise-view", args.pairwise_view])
    run_analysis_main(argv)


def build_parser() -> argparse.ArgumentParser:
    from generate.constants import ACTIVE_JUDGE_NAMES, LLM_SOURCE_MODELS, PAIRWISE_VIEW_CHOICES, PAIRWISE_VIEW_DEFAULT

    parser = argparse.ArgumentParser(prog="judge-bias-study", description="LLM-judge self-preference study pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Generate plans for one arm")
    gen.add_argument("--arm", choices=["llm", "programmatic"], required=True)
    gen.add_argument("--n", type=int, default=None, help="Plans per fixture (per model for llm arm)")
    gen.add_argument("--output", default="plans/")
    gen.add_argument("--sampler-config", default=None)
    gen.add_argument("--seed-offset", type=int, default=0)
    gen.add_argument("--source-model", choices=LLM_SOURCE_MODELS, default=None)
    gen.add_argument("--source-temperature", type=float, default=0.7)
    gen.add_argument("--explainer-temperature", type=float, default=0.0)
    gen.add_argument("--fixture-id", action="append", default=None, help="Optional fixture subset; repeat or pass comma-separated ids")
    gen.set_defaults(func=cmd_generate)

    fp = sub.add_parser("fit-priors", help="Fit sampler priors from LLM-arm outputs")
    fp.add_argument("--plans", default="plans/")
    fp.add_argument("--output", default="sampler_config.json")
    fp.add_argument("--min-plans", type=int, default=30)
    fp.add_argument("--seed", type=int, default=0)
    fp.set_defaults(func=cmd_fit_priors)

    mat = sub.add_parser("match", help="Score plans and build matched_pairs.json")
    mat.add_argument("--plans", default="plans/")
    mat.add_argument("--output", default="matched_pairs.json")
    mat.add_argument("--allow-mixed-generation-conditions", action="store_true")
    mat.set_defaults(func=cmd_match)

    jdg = sub.add_parser("judge", help="Run pairwise and or soft-eval for one judge")
    jdg.add_argument("--judge", required=True, choices=ACTIVE_JUDGE_NAMES)
    jdg.add_argument("--plans", default="plans/")
    jdg.add_argument("--pairs", default="matched_pairs.json")
    jdg.add_argument("--output", default="judgments/")
    jdg.add_argument("--pilot", action="store_true")
    jdg.add_argument("--pair-offset", type=int, default=0)
    jdg.add_argument("--pair-limit", type=int, default=None)
    jdg.add_argument("--plan-offset", type=int, default=0)
    jdg.add_argument("--plan-limit", type=int, default=None)
    jdg.add_argument("--skip-pairwise", action="store_true")
    jdg.add_argument("--skip-soft-eval", action="store_true")
    jdg.add_argument("--require-style-gate", action="store_true")
    jdg.add_argument("--style-gate-summary", default=None)
    jdg.add_argument("--provenance", default=None)
    jdg.add_argument("--pairwise-view", choices=PAIRWISE_VIEW_CHOICES, default=PAIRWISE_VIEW_DEFAULT)
    jdg.add_argument("--judge-temperature", type=float, default=0.0)
    jdg.set_defaults(func=cmd_judge)

    sty = sub.add_parser("audit-style", help="Run paired surface-form leakage audit and gate")
    sty.add_argument("--plans", default="plans/")
    sty.add_argument("--provenance", default=None)
    sty.add_argument("--pairs", default="matched_pairs.json")
    sty.add_argument("--output", default="results/")
    sty.set_defaults(func=cmd_audit_style)

    ana = sub.add_parser("analyze", help="Run full analysis pipeline")
    ana.add_argument("--judgments", default="judgments/")
    ana.add_argument("--plans", default="plans/")
    ana.add_argument("--provenance", default=None)
    ana.add_argument("--pairs", default="matched_pairs.json")
    ana.add_argument("--output", default="results/")
    ana.add_argument("--pairwise-view", default=None)
    ana.set_defaults(func=cmd_analyze)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()