"""generate/exact_counts.py — exact-count generation orchestrator.

This revision keeps the module importable by downstream tools while upgrading
the study profile to the 32-cell oversampling design.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from fixtures.spec import FIXTURE_IDS
from generate.run_generation import run_llm_arm, run_programmatic_arm

LLM_BASE_PER_FIXTURE_PER_MODEL = 6
PROGRAMMATIC_BASE_PER_FIXTURE = 10

EXPECTED_LLM_PER_MODEL = len(FIXTURE_IDS) * LLM_BASE_PER_FIXTURE_PER_MODEL
EXPECTED_PROGRAMMATIC_TOTAL = len(FIXTURE_IDS) * PROGRAMMATIC_BASE_PER_FIXTURE

assert EXPECTED_LLM_PER_MODEL == 192
assert EXPECTED_PROGRAMMATIC_TOTAL == 320


def exact_count_summary() -> dict[str, Any]:
    return {
        "n_fixtures": len(FIXTURE_IDS),
        "llm": {
            "base_per_fixture_per_model": LLM_BASE_PER_FIXTURE_PER_MODEL,
            "expected_per_model": EXPECTED_LLM_PER_MODEL,
            "expected_total_two_models": EXPECTED_LLM_PER_MODEL * 2,
        },
        "programmatic": {
            "base_per_fixture": PROGRAMMATIC_BASE_PER_FIXTURE,
            "expected_total": EXPECTED_PROGRAMMATIC_TOTAL,
        },
    }


def run_exact_llm(*, output_dir: Path, source_model: str) -> tuple[int, int]:
    return run_llm_arm(
        output_dir=output_dir,
        plans_per_fixture=LLM_BASE_PER_FIXTURE_PER_MODEL,
        source_model=source_model,
        seed_offset=0,
        fixture_ids=None,
    )


def run_exact_programmatic(
    *,
    output_dir: Path,
    sampler_config_path: Path | None = None,
) -> tuple[int, int]:
    return run_programmatic_arm(
        output_dir=output_dir,
        plans_per_fixture=PROGRAMMATIC_BASE_PER_FIXTURE,
        sampler_config_path=sampler_config_path,
        seed_offset=0,
        fixture_ids=None,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run exact-count study generation shards.")
    parser.add_argument("--arm", choices=["llm", "programmatic"], required=True)
    parser.add_argument("--output", default="plans/")
    parser.add_argument("--source-model", default=None)
    parser.add_argument("--sampler-config", default=None)
    parser.add_argument("--print-summary", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.print_summary:
        print(exact_count_summary())

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.arm == "llm":
        if not args.source_model:
            parser.error("--source-model is required when --arm llm")
        print(f"=== Exact LLM-arm generation for {args.source_model} ===")
        print(
            f"Base shard: {LLM_BASE_PER_FIXTURE_PER_MODEL} per fixture across all "
            f"{len(FIXTURE_IDS)} fixtures"
        )
        generated, skipped = run_exact_llm(output_dir=output_dir, source_model=args.source_model)
        expected = EXPECTED_LLM_PER_MODEL
    else:
        sampler_config_path = Path(args.sampler_config) if args.sampler_config else None
        print("=== Exact programmatic-arm generation ===")
        print(
            f"Base shard: {PROGRAMMATIC_BASE_PER_FIXTURE} per fixture across all "
            f"{len(FIXTURE_IDS)} fixtures"
        )
        generated, skipped = run_exact_programmatic(
            output_dir=output_dir,
            sampler_config_path=sampler_config_path,
        )
        expected = EXPECTED_PROGRAMMATIC_TOTAL

    print("\\n=== Exact-count generation complete ===")
    print(f"Generated this run: {generated}")
    print(f"Skipped (already existed): {skipped}")
    print(f"Expected logical total for this study branch: {expected}")


if __name__ == "__main__":
    main()
