"""cli.py — judge-bias-study command-line interface."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(_ROOT))


def cmd_generate(args: argparse.Namespace) -> None:
    from generate.constants import default_plans_per_fixture
    from generate.run_generation import run_llm_arm, run_programmatic_arm

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    plans_per_fixture = args.n if args.n is not None else default_plans_per_fixture(args.arm)

    print(f"Arm: {args.arm}")
    print(f"Plans per fixture: {plans_per_fixture}")

    if args.arm == "llm":
        n_gen, n_skip = run_llm_arm(output_dir=output_dir, plans_per_fixture=plans_per_fixture)
    else:
        n_gen, n_skip = run_programmatic_arm(
            output_dir=output_dir,
            plans_per_fixture=plans_per_fixture,
            sampler_config_path=Path(args.sampler_config) if args.sampler_config else None,
        )
    print(f"Generated {n_gen} plans, skipped {n_skip} (already existed) → {output_dir}")


def cmd_fit_priors(args: argparse.Namespace) -> None:
    from generate.fit_priors import fit_and_save
    fit_and_save(plans_dir=Path(args.plans), output_path=Path(args.output), min_plans=args.min_plans, seed=args.seed)


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
    )
    print(f"{len(pairs)} pairs written to {output_path}")


def cmd_judge(args: argparse.Namespace) -> None:
    import json
    from generate.constants import PAIRWISE_N_POSITIONS, PAIRWISE_N_RUNS, PILOT_PAIR_LIMIT
    from judge.harness import check_pilot_bias_gate, run_pairwise_harness, run_soft_eval_harness
    from judge.panel import assert_judge_fits_quota, get_judge

    judge = get_judge(args.judge)
    assert_judge_fits_quota(judge)

    plans_dir = Path(args.plans)
    pairs_file = Path(args.pairs)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = json.loads(pairs_file.read_text()) if pairs_file.exists() else []
    if args.pilot:
        pairs = pairs[:PILOT_PAIR_LIMIT]
        print(f"Pilot mode: {len(pairs)} pairs")

    fixtures_dir = _ROOT / "fixtures" / "data"
    pairwise_out = output_dir / f"pairwise_{judge.name}.jsonl"
    run_pairwise_harness(
        pairs=pairs,
        plans_dir=plans_dir,
        judge=judge,
        rollups_path=None,
        fixtures_dir=fixtures_dir,
        output_path=pairwise_out,
        n_runs=PAIRWISE_N_RUNS,
        n_positions=PAIRWISE_N_POSITIONS,
    )

    if args.pilot:
        gate = check_pilot_bias_gate(pairwise_out)
        print(f"Pilot bias gate: {gate['message']}")
        if not gate["passed"]:
            print("  ⚠ Judge excluded from H1/H2 — do NOT proceed to full run.")
            sys.exit(2)

    plan_ids = [p.stem for p in plans_dir.glob("*.json") if not p.name.endswith(".provenance.json")]
    run_soft_eval_harness(
        plan_ids=plan_ids,
        plans_dir=plans_dir,
        judge=judge,
        rollups_path=None,
        fixtures_dir=fixtures_dir,
        provenance_dir=plans_dir,
        output_path=output_dir / f"softeval_{judge.name}.jsonl",
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
        sys.exit(2)


def cmd_analyze(args: argparse.Namespace) -> None:
    from analyze.run_analysis import main as run_analysis_main
    argv = ["--judgments", args.judgments, "--plans", args.plans, "--output", args.output, "--pairs", args.pairs]
    if args.provenance:
        argv.extend(["--provenance", args.provenance])
    run_analysis_main(argv)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="judge-bias-study", description="LLM-judge self-preference study pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Generate plans (llm or programmatic arm)")
    gen.add_argument("--arm", choices=["llm", "programmatic"], required=True)
    gen.add_argument("--n", type=int, default=None, help="Plans per fixture (per model for llm arm); default uses frozen study config")
    gen.add_argument("--output", default="plans/")
    gen.add_argument("--sampler-config", default=None)
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
    mat.set_defaults(func=cmd_match)

    jdg = sub.add_parser("judge", help="Run pairwise + soft-eval for one judge")
    jdg.add_argument("--judge", required=True)
    jdg.add_argument("--plans", default="plans/")
    jdg.add_argument("--pairs", default="matched_pairs.json")
    jdg.add_argument("--output", default="judgments/")
    jdg.add_argument("--pilot", action="store_true")
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
    ana.set_defaults(func=cmd_analyze)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
