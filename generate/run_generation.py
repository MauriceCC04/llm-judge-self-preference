"""generate/run_generation.py — batch generation orchestrator.

Key fixes in this version:
- the LLM arm now targets exactly one source model per job
- fixture subsets can be selected for HPC sharding and exact-count top-ups
- seed offsets are wired through consistently
- generation temperatures are now first-class experiment variables
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from generate.persistence import verify_plan_and_provenance
from generate.temperature import (
    build_llm_generation_condition,
    build_llm_plan_id,
    build_programmatic_generation_condition,
    build_programmatic_plan_id,
)

_ROOT = Path(__file__).parent.parent
log = logging.getLogger(__name__)


def _configure_logging() -> None:
    level_name = (os.getenv("TRAILTRAINING_LOG_LEVEL") or "WARNING").strip().upper()
    level = getattr(logging, level_name, logging.WARNING)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    log.info("Configured logging at level=%s", level_name)


def _plan_outputs_exist(output_dir: Path, plan_id: str) -> bool:
    plan_path = output_dir / f"{plan_id}.json"
    provenance_path = output_dir / f"{plan_id}.json.provenance.json"
    if not (plan_path.exists() and provenance_path.exists()):
        return False
    try:
        verify_plan_and_provenance(plan_path, provenance_path, expected_plan_id=plan_id)
    except Exception:
        return False
    return True


def _verify_generated_result(plan_id: str, result: tuple[str, str, str]) -> None:
    _plan_json, plan_path, provenance_path = result
    verify_plan_and_provenance(Path(plan_path), Path(provenance_path), expected_plan_id=plan_id)


def _write_failure(failures_path: Path, plan_id: str, exc: Exception) -> None:
    failures_path.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "plan_id": plan_id,
        "error": str(exc),
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }

    for attr_name in (
        "stage",
        "raw_text_path",
        "request_json_path",
        "prompt_text_path",
        "failure_json_path",
    ):
        value = getattr(exc, attr_name, None)
        if value:
            entry[attr_name] = str(value)

    cause = getattr(exc, "__cause__", None)
    if cause is not None:
        entry["cause"] = str(cause)

    with failures_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, default=str) + "\n")

def _resolve_fixture_specs(fixture_ids: Iterable[str] | None):
    from fixtures.spec import ALL_FIXTURE_SPECS, FIXTURE_BY_ID

    if not fixture_ids:
        return list(ALL_FIXTURE_SPECS)

    resolved = []
    for fixture_id in fixture_ids:
        if fixture_id not in FIXTURE_BY_ID:
            valid = ", ".join(sorted(FIXTURE_BY_ID))
            raise KeyError(f"Unknown fixture id {fixture_id!r}. Valid ids: {valid}")
        resolved.append(FIXTURE_BY_ID[fixture_id])
    return resolved


def run_llm_arm(
    output_dir: Path,
    *,
    plans_per_fixture: int,
    source_model: str,
    seed_offset: int = 0,
    fixture_ids: list[str] | None = None,
    source_temperature: float = 0.7,
    explainer_temperature: float = 0.0,
) -> tuple[int, int]:
    from generate.llm_arm import generate_llm_plan

    specs = _resolve_fixture_specs(fixture_ids)
    output_dir.mkdir(parents=True, exist_ok=True)
    failures_path = output_dir / "failed_plans.jsonl"
    generated = 0
    skipped = 0
    total = len(specs) * plans_per_fixture
    generation_condition = build_llm_generation_condition(
        source_temperature=source_temperature,
        explainer_temperature=explainer_temperature,
    )

    for idx, spec in enumerate(specs, start=1):
        fixture_dir = _ROOT / "fixtures" / "data" / spec.fixture_id
        if not fixture_dir.exists():
            print(f"  [warn] fixture dir not found: {fixture_dir}", file=sys.stderr)
            continue
        for seed in range(plans_per_fixture):
            actual_seed = seed_offset + seed
            plan_id = build_llm_plan_id(
                fixture_id=spec.fixture_id,
                source_model=source_model,
                seed=actual_seed,
                source_temperature=source_temperature,
                explainer_temperature=explainer_temperature,
            )
            progress = ((idx - 1) * plans_per_fixture) + seed + 1
            if _plan_outputs_exist(output_dir, plan_id):
                skipped += 1
                continue
            print(f"  [{progress}/{total}] {plan_id} ...", end="", flush=True)
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
            except Exception as exc:  # pragma: no cover - failure logging path
                print(f" FAILED: {exc}")
                _write_failure(failures_path, plan_id, exc)
    return generated, skipped


def run_programmatic_arm(
    output_dir: Path,
    *,
    plans_per_fixture: int,
    sampler_config_path: Path | None = None,
    seed_offset: int = 0,
    fixture_ids: list[str] | None = None,
    explainer_temperature: float = 0.0,
) -> tuple[int, int]:
    from generate.fit_priors import load_sampler_config
    from generate.programmatic_arm import generate_programmatic_plan

    specs = _resolve_fixture_specs(fixture_ids)
    output_dir.mkdir(parents=True, exist_ok=True)
    failures_path = output_dir / "failed_plans.jsonl"
    base_cfg = load_sampler_config(sampler_config_path) if sampler_config_path and sampler_config_path.exists() else None
    generated = 0
    skipped = 0
    total = len(specs) * plans_per_fixture
    generation_condition = build_programmatic_generation_condition(
        explainer_temperature=explainer_temperature,
    )

    for idx, spec in enumerate(specs, start=1):
        fixture_dir = _ROOT / "fixtures" / "data" / spec.fixture_id
        if not fixture_dir.exists():
            print(f"  [warn] fixture dir not found: {fixture_dir}", file=sys.stderr)
            continue
        for seed in range(plans_per_fixture):
            actual_seed = seed_offset + seed
            plan_id = build_programmatic_plan_id(
                fixture_id=spec.fixture_id,
                seed=actual_seed,
                explainer_temperature=explainer_temperature,
            )
            progress = ((idx - 1) * plans_per_fixture) + seed + 1
            if _plan_outputs_exist(output_dir, plan_id):
                skipped += 1
                continue
            print(f"  [{progress}/{total}] {plan_id} ...", end="", flush=True)
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
            except Exception as exc:  # pragma: no cover - failure logging path
                print(f" FAILED: {exc}")
                _write_failure(failures_path, plan_id, exc)
    return generated, skipped


def _split_fixture_ids(raw_values: list[str] | None) -> list[str] | None:
    if not raw_values:
        return None
    fixture_ids: list[str] = []
    for raw in raw_values:
        fixture_ids.extend(part.strip() for part in raw.split(",") if part.strip())
    return fixture_ids or None


def build_parser() -> argparse.ArgumentParser:
    from generate.constants import LLM_SOURCE_MODELS

    parser = argparse.ArgumentParser(description="Batch generation orchestrator for both study arms.")
    parser.add_argument("--arm", choices=["llm", "programmatic"], required=True)
    parser.add_argument("--plans-per-fixture", type=int, default=None)
    parser.add_argument("--output", default="plans/")
    parser.add_argument("--sampler-config", default=None)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument(
        "--source-model",
        default=None,
        help=f"Required for llm arm. One of: {', '.join(LLM_SOURCE_MODELS)}",
    )
    parser.add_argument("--source-temperature", type=float, default=0.7)
    parser.add_argument("--explainer-temperature", type=float, default=0.0)
    parser.add_argument(
        "--fixture-id",
        action="append",
        default=None,
        help="Optional fixture subset. Repeat or pass a comma-separated list for sharded HPC runs.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    from generate.constants import default_plans_per_fixture

    _configure_logging()

    parser = build_parser()
    args = parser.parse_args(argv)
    output_dir = Path(args.output)
    plans_per_fixture = args.plans_per_fixture or default_plans_per_fixture(args.arm)
    fixture_ids = _split_fixture_ids(args.fixture_id)

    print(f"=== Generation: {args.arm} arm ===")
    print(f"  plans_per_fixture:     {plans_per_fixture}")
    print(f"  output_dir:            {output_dir}")
    print(f"  seed_offset:           {args.seed_offset}")
    print(f"  fixture_ids:           {fixture_ids or 'ALL'}")
    print(f"  explainer_temperature: {args.explainer_temperature}")
    if args.arm == "llm":
        if not args.source_model:
            parser.error("--source-model is required when --arm llm")
        print(f"  source_model:          {args.source_model}")
        print(f"  source_temperature:    {args.source_temperature}")

    t0 = datetime.now(tz=timezone.utc)
    if args.arm == "llm":
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

    elapsed = (datetime.now(tz=timezone.utc) - t0).total_seconds()
    print("\n=== Done ===")
    print(f"  Generated: {n_gen}")
    print(f"  Skipped (already existed): {n_skip}")
    print(f"  Elapsed: {elapsed:.0f} s")


if __name__ == "__main__":
    main()
