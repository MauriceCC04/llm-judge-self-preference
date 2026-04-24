"""cli.py — judge-bias-study command-line interface.

Usage::

    python cli.py generate --arm llm [--fixture r_low__rc_low__ph_base] [--n 20]
    python cli.py generate --arm programmatic [--n 40] [--sampler-config sampler_config.json]
    python cli.py match   [--plans plans/] [--output matched_pairs.json]
    python cli.py judge   --judge qwen_7b_judge [--pilot] [--plans plans/] [--pairs matched_pairs.json]
    python cli.py analyze [--judgments judgments/] [--plans plans/] [--output results/]

For HPC batch runs prefer the SLURM scripts in slurm/ directly.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(_ROOT))


# ── Sub-commands ──────────────────────────────────────────────────────────────

def cmd_generate(args: argparse.Namespace) -> None:
    """Delegate to the batch generation orchestrator."""
    from generate.run_generation import run_llm_arm, run_programmatic_arm

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Arm: {args.arm}")
    print(f"Plans per fixture: {args.n}")

    if args.arm == "llm":
        n_gen, n_skip = run_llm_arm(
            output_dir=output_dir,
            plans_per_fixture=args.n,
        )
    else:
        n_gen, n_skip = run_programmatic_arm(
            output_dir=output_dir,
            plans_per_fixture=args.n,
            sampler_config_path=Path(args.sampler_config) if args.sampler_config else None,
        )

    print(f"\nGenerated {n_gen} plans, skipped {n_skip} (already existed) → {output_dir}")


def cmd_fit_priors(args: argparse.Namespace) -> None:
    """Fit sampler priors from LLM-arm outputs."""
    from generate.fit_priors import fit_and_save

    fit_and_save(
        plans_dir=Path(args.plans),
        output_path=Path(args.output),
        min_plans=args.min_plans,
        seed=args.seed,
    )


def cmd_match(args: argparse.Namespace) -> None:
    """Score all plans and build matched_pairs.json."""
    from match.pair import build_matched_pairs
    from generate.constants import MATCH_TOLERANCE, TARGET_PAIRS

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
    """Run pairwise + soft-eval for one judge (requires running vLLM server)."""
    import json
    from judge.panel import get_judge
    from judge.harness import run_pairwise_harness, run_soft_eval_harness, check_pilot_bias_gate

    judge = get_judge(args.judge)
    plans_dir = Path(args.plans)
    pairs_file = Path(args.pairs)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = json.loads(pairs_file.read_text()) if pairs_file.exists() else []
    if args.pilot:
        pairs = pairs[:30]
        print(f"Pilot mode: {len(pairs)} pairs")

    # Find rollups for context
    fixtures_dir = _ROOT / "fixtures" / "data"
    rollups_path: Path | None = None
    for pair in pairs[:1]:
        candidate = fixtures_dir / pair.get("fixture_id", "") / "combined_rollups.json"
        if candidate.exists():
            rollups_path = candidate
            break

    pairwise_out = output_dir / f"pairwise_{judge.name}.jsonl"
    run_pairwise_harness(
        pairs=pairs,
        plans_dir=plans_dir,
        judge=judge,
        rollups_path=rollups_path,
        output_path=pairwise_out,
        n_runs=3,
        n_positions=2,
    )

    if args.pilot:
        gate = check_pilot_bias_gate(pairwise_out)
        print(f"\nPilot bias gate: {gate['message']}")
        if not gate["passed"]:
            print("  ⚠ Judge excluded from H1/H2 — do NOT proceed to full run.")
            sys.exit(2)

    plan_ids = [
        p.stem for p in plans_dir.glob("*.json")
        if not p.name.endswith(".provenance.json")
    ]
    run_soft_eval_harness(
        plan_ids=plan_ids,
        plans_dir=plans_dir,
        judge=judge,
        rollups_path=rollups_path,
        output_path=output_dir / f"softeval_{judge.name}.jsonl",
    )
    print(f"Judge run complete: {judge.name}")


def cmd_analyze(args: argparse.Namespace) -> None:
    """Run the full analysis pipeline — delegates to analyze/run_analysis.py."""
    from analyze.run_analysis import main as run_analysis_main

    run_analysis_main([
        "--judgments", args.judgments,
        "--plans",     args.plans,
        "--output",    args.output,
        *(["--pairs", args.pairs] if hasattr(args, "pairs") else []),
    ])


# ── Parser ────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="judge-bias-study",
        description="LLM-judge self-preference study pipeline",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # generate
    gen = sub.add_parser("generate", help="Generate plans (llm or programmatic arm)")
    gen.add_argument("--arm", choices=["llm", "programmatic"], required=True)
    gen.add_argument("--n", type=int, default=20, help="Plans per fixture (per model for llm arm)")
    gen.add_argument("--output", default="plans/", help="Output directory")
    gen.add_argument("--sampler-config", default=None, help="Fitted priors JSON (programmatic arm)")
    gen.set_defaults(func=cmd_generate)

    # fit-priors
    fp = sub.add_parser("fit-priors", help="Fit sampler priors from LLM-arm outputs")
    fp.add_argument("--plans", default="plans/", help="Directory with LLM-arm plan files + provenance")
    fp.add_argument("--output", default="sampler_config.json", help="Output config JSON")
    fp.add_argument("--min-plans", type=int, default=30)
    fp.add_argument("--seed", type=int, default=0)
    fp.set_defaults(func=cmd_fit_priors)

    # match
    mat = sub.add_parser("match", help="Score plans and build matched_pairs.json")
    mat.add_argument("--plans", default="plans/", help="Directory with plan JSON + provenance files")
    mat.add_argument("--output", default="matched_pairs.json", help="Output manifest")
    mat.set_defaults(func=cmd_match)

    # judge
    jdg = sub.add_parser("judge", help="Run pairwise + soft-eval for one judge")
    jdg.add_argument("--judge", required=True, help="Judge name, e.g. qwen_7b_judge")
    jdg.add_argument("--plans", default="plans/", help="Directory with plan JSON files")
    jdg.add_argument("--pairs", default="matched_pairs.json", help="Matched pairs manifest")
    jdg.add_argument("--output", default="judgments/", help="Output directory")
    jdg.add_argument("--pilot", action="store_true", help="Pilot mode: first 30 pairs only")
    jdg.set_defaults(func=cmd_judge)

    # analyze
    ana = sub.add_parser("analyze", help="Run full analysis pipeline → summary.json + figures")
    ana.add_argument("--judgments", default="judgments/", help="Judgments directory")
    ana.add_argument("--plans",     default="plans/",     help="Plans directory (provenance)")
    ana.add_argument("--pairs",     default="matched_pairs.json", help="Matched pairs manifest")
    ana.add_argument("--output",    default="results/",   help="Output directory")
    ana.set_defaults(func=cmd_analyze)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
