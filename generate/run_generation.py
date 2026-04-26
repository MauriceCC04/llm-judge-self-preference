"""generate/run_generation.py — batch generation orchestrator."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).parent.parent


def _plan_exists(output_dir: Path, plan_id: str) -> bool:
    return (output_dir / f"{plan_id}.json").exists()


def _write_failure(failures_path: Path, plan_id: str, exc: Exception) -> None:
    failures_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "plan_id": plan_id,
        "error": str(exc),
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }
    with failures_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, default=str) + "\n")


def run_llm_arm(output_dir: Path, *, plans_per_fixture: int, seed_offset: int = 0) -> tuple[int, int]:
    from fixtures.spec import ALL_FIXTURE_SPECS
    from generate.constants import LLM_SOURCE_MODELS
    from generate.llm_arm import generate_llm_plan

    output_dir.mkdir(parents=True, exist_ok=True)
    failures_path = output_dir / "failed_plans.jsonl"
    generated = 0
    skipped = 0
    total = len(ALL_FIXTURE_SPECS) * len(LLM_SOURCE_MODELS) * plans_per_fixture
    done = 0
    for spec in ALL_FIXTURE_SPECS:
        fixture_dir = _ROOT / "fixtures" / "data" / spec.fixture_id
        if not fixture_dir.exists():
            print(f"  [warn] fixture dir not found: {fixture_dir}", file=sys.stderr)
            continue
        for model in LLM_SOURCE_MODELS:
            model_tag = model.split("/")[-1].replace("-", "_").lower()
            for seed in range(plans_per_fixture):
                plan_id = f"{spec.fixture_id}__{model_tag}__s{seed_offset + seed:03d}"
                done += 1
                if _plan_exists(output_dir, plan_id):
                    skipped += 1
                    continue
                print(f"  [{done}/{total}] {plan_id} ...", end="", flush=True)
                try:
                    generate_llm_plan(fixture_dir=fixture_dir, output_dir=output_dir, plan_id=plan_id, source_model=model, seed=seed_offset + seed)
                    print(" OK")
                    generated += 1
                except Exception as exc:
                    print(f" FAILED: {exc}")
                    _write_failure(failures_path, plan_id, exc)
    return generated, skipped


def run_programmatic_arm(output_dir: Path, *, plans_per_fixture: int, sampler_config_path: Path | None = None, seed_offset: int = 0) -> tuple[int, int]:
    from fixtures.spec import ALL_FIXTURE_SPECS
    from generate.programmatic_arm import generate_programmatic_plan
    from generate.sampler import StructuralSamplerConfig

    output_dir.mkdir(parents=True, exist_ok=True)
    failures_path = output_dir / "failed_plans.jsonl"
    base_cfg: StructuralSamplerConfig | None = None
    if sampler_config_path and sampler_config_path.exists():
        from generate.fit_priors import load_sampler_config
        base_cfg = load_sampler_config(sampler_config_path)
    generated = 0
    skipped = 0
    total = len(ALL_FIXTURE_SPECS) * plans_per_fixture
    done = 0
    for spec in ALL_FIXTURE_SPECS:
        fixture_dir = _ROOT / "fixtures" / "data" / spec.fixture_id
        if not fixture_dir.exists():
            print(f"  [warn] fixture dir not found: {fixture_dir}", file=sys.stderr)
            continue
        for seed in range(plans_per_fixture):
            plan_id = f"{spec.fixture_id}__prog__s{seed_offset + seed:03d}"
            done += 1
            if _plan_exists(output_dir, plan_id):
                skipped += 1
                continue
            if base_cfg is not None:
                import copy
                cfg = copy.copy(base_cfg)
                cfg.seed = seed_offset + seed
                cfg.readiness_status = spec.readiness_status
            else:
                cfg = StructuralSamplerConfig(plan_days=7, seed=seed_offset + seed, readiness_status=spec.readiness_status)
            print(f"  [{done}/{total}] {plan_id} ...", end="", flush=True)
            try:
                generate_programmatic_plan(fixture_dir=fixture_dir, output_dir=output_dir, plan_id=plan_id, seed=seed_offset + seed, sampler_cfg=cfg)
                print(" OK")
                generated += 1
            except Exception as exc:
                print(f" FAILED: {exc}")
                _write_failure(failures_path, plan_id, exc)
    return generated, skipped


def main() -> None:
    from generate.constants import default_plans_per_fixture

    parser = argparse.ArgumentParser(description="Batch generation orchestrator for both study arms.")
    parser.add_argument("--arm", choices=["llm", "programmatic"], required=True)
    parser.add_argument("--plans-per-fixture", type=int, default=None)
    parser.add_argument("--output", default="plans/")
    parser.add_argument("--sampler-config", default=None)
    parser.add_argument("--seed-offset", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output)
    plans_per_fixture = args.plans_per_fixture or default_plans_per_fixture(args.arm)
    print(f"=== Generation: {args.arm} arm ===")
    print(f"  plans_per_fixture: {plans_per_fixture}")
    print(f"  output_dir:        {output_dir}")
    print(f"  seed_offset:       {args.seed_offset}")
    t0 = datetime.now(tz=timezone.utc)

    if args.arm == "llm":
        n_gen, n_skip = run_llm_arm(output_dir=output_dir, plans_per_fixture=plans_per_fixture, seed_offset=args.seed_offset)
    else:
        n_gen, n_skip = run_programmatic_arm(output_dir=output_dir, plans_per_fixture=plans_per_fixture, sampler_config_path=Path(args.sampler_config) if args.sampler_config else None, seed_offset=args.seed_offset)

    elapsed = (datetime.now(tz=timezone.utc) - t0).total_seconds()
    print("\n=== Done ===")
    print(f"  Generated: {n_gen}")
    print(f"  Skipped (already existed): {n_skip}")
    print(f"  Elapsed: {elapsed:.0f} s")


if __name__ == "__main__":
    main()
