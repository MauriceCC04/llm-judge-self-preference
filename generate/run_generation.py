"""generate/run_generation.py — batch generation orchestrator.

Run this script (or the equivalent SLURM job) to generate all ~320 plans for
one arm.  It loops over (fixture × source_model × seed) for the LLM arm, and
over (fixture × seed) for the programmatic arm.

Usage::

    # LLM arm — 20 plans per fixture (8 fixtures × 20 = 160 per source model,
    #            2 source models → 320 total)
    python -m generate.run_generation \\
        --arm llm \\
        --plans-per-fixture 20 \\
        --output plans/

    # Programmatic arm — fit priors first, then generate
    python -m generate.fit_priors --plans-dir plans/ --output sampler_config.json
    python -m generate.run_generation \\
        --arm programmatic \\
        --plans-per-fixture 40 \\
        --sampler-config sampler_config.json \\
        --output plans/

Design notes
------------
* Both arms write to the SAME output directory; provenance sidecars record arm.
* The ResumeJsonl-style idiom is replicated here: if a plan file already exists
  for a given (plan_id) the generation step is skipped.  Re-running after a
  crash resumes from the first missing plan.
* Progress is printed to stdout; errors are caught per-plan and written to
  failed_plans.jsonl so a single bad plan does not abort the run.
"""
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


def run_llm_arm(
    output_dir: Path,
    *,
    plans_per_fixture: int,
    seed_offset: int = 0,
) -> tuple[int, int]:
    """Generate LLM-arm plans.  Returns (n_generated, n_skipped)."""
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
                    generate_llm_plan(
                        fixture_dir=fixture_dir,
                        output_dir=output_dir,
                        plan_id=plan_id,
                        source_model=model,
                        seed=seed_offset + seed,
                    )
                    print(" OK")
                    generated += 1
                except Exception as exc:
                    print(f" FAILED: {exc}")
                    _write_failure(failures_path, plan_id, exc)

    return generated, skipped


def run_programmatic_arm(
    output_dir: Path,
    *,
    plans_per_fixture: int,
    sampler_config_path: Path | None = None,
    seed_offset: int = 0,
) -> tuple[int, int]:
    """Generate programmatic-arm plans.  Returns (n_generated, n_skipped)."""
    from fixtures.spec import ALL_FIXTURE_SPECS
    from generate.programmatic_arm import generate_programmatic_plan
    from generate.sampler import StructuralSamplerConfig

    output_dir.mkdir(parents=True, exist_ok=True)
    failures_path = output_dir / "failed_plans.jsonl"

    # Load fitted sampler config if provided
    base_cfg: StructuralSamplerConfig | None = None
    if sampler_config_path and sampler_config_path.exists():
        from generate.fit_priors import load_sampler_config
        base_cfg = load_sampler_config(sampler_config_path)
        print(f"  Loaded sampler priors from {sampler_config_path}")
    else:
        print("  [warn] No sampler_config.json — using default priors.", file=sys.stderr)

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

            # Build per-fixture sampler config (override readiness from fixture meta)
            if base_cfg is not None:
                import copy
                cfg = copy.copy(base_cfg)
                cfg.seed = seed_offset + seed
                cfg.readiness_status = spec.readiness_status
            else:
                cfg = StructuralSamplerConfig(
                    plan_days=7,
                    seed=seed_offset + seed,
                    readiness_status=spec.readiness_status,
                )

            print(f"  [{done}/{total}] {plan_id} ...", end="", flush=True)
            try:
                generate_programmatic_plan(
                    fixture_dir=fixture_dir,
                    output_dir=output_dir,
                    plan_id=plan_id,
                    seed=seed_offset + seed,
                    sampler_cfg=cfg,
                )
                print(" OK")
                generated += 1
            except Exception as exc:
                print(f" FAILED: {exc}")
                _write_failure(failures_path, plan_id, exc)

    return generated, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch generation orchestrator for both study arms."
    )
    parser.add_argument(
        "--arm", choices=["llm", "programmatic"], required=True,
        help="Which arm to generate."
    )
    parser.add_argument(
        "--plans-per-fixture", type=int, default=20,
        help=(
            "Plans per fixture.  LLM arm: per source model per fixture.  "
            "Programmatic arm: per fixture.  "
            "Default 20 → LLM: 8×2×20=320 total; prog: 8×40=320 total."
        ),
    )
    parser.add_argument(
        "--output", default="plans/",
        help="Output directory for plan JSON files and provenance sidecars."
    )
    parser.add_argument(
        "--sampler-config", default=None,
        help="(Programmatic arm) Path to fitted sampler_config.json."
    )
    parser.add_argument(
        "--seed-offset", type=int, default=0,
        help="Add this offset to all seed values (for resuming with fresh seeds)."
    )
    args = parser.parse_args()

    output_dir = Path(args.output)

    print(f"=== Generation: {args.arm} arm ===")
    print(f"  plans_per_fixture: {args.plans_per_fixture}")
    print(f"  output_dir:        {output_dir}")
    print(f"  seed_offset:       {args.seed_offset}")
    t0 = datetime.now(tz=timezone.utc)

    if args.arm == "llm":
        n_gen, n_skip = run_llm_arm(
            output_dir=output_dir,
            plans_per_fixture=args.plans_per_fixture,
            seed_offset=args.seed_offset,
        )
    else:
        n_gen, n_skip = run_programmatic_arm(
            output_dir=output_dir,
            plans_per_fixture=args.plans_per_fixture,
            sampler_config_path=Path(args.sampler_config) if args.sampler_config else None,
            seed_offset=args.seed_offset,
        )

    elapsed = (datetime.now(tz=timezone.utc) - t0).total_seconds()
    print(f"\n=== Done ===")
    print(f"  Generated: {n_gen}")
    print(f"  Skipped (already existed): {n_skip}")
    print(f"  Elapsed: {elapsed:.0f} s")
    failures_path = output_dir / "failed_plans.jsonl"
    if failures_path.exists():
        with failures_path.open() as fh:
            n_fail = sum(1 for line in fh if line.strip())
        print(f"  Failures logged: {n_fail}  (see {failures_path})")


if __name__ == "__main__":
    main()
