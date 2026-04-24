"""generate/fit_priors.py — fit StructuralSamplerConfig priors from LLM arm outputs.

The study protocol requires:
    1. Generate ~100 LLM plans first (one seed per fixture, both source models).
    2. Fit marginals: hard_days/week, rest_days/week, duration mean/std per
       session_type, session_type frequency.
    3. Feed the fitted config into the programmatic arm sampler so both arms
       share structural summary statistics.

This eliminates "judges prefer LLM plans because they looked structurally
different" as an alternative explanation for H1.

Usage::

    python -m generate.fit_priors \\
        --plans-dir plans/ \\
        --output sampler_config.json \\
        [--min-plans 50]

The output JSON can be loaded back via StructuralSamplerConfig.model_validate().
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_plan_dicts(plans_dir: Path, arm: str = "llm") -> list[dict]:
    """Load all LLM-arm plan dicts from *plans_dir* using provenance sidecars."""
    plan_dicts = []
    for prov_path in sorted(plans_dir.glob("*.provenance.json")):
        try:
            prov = json.loads(prov_path.read_text())
        except Exception:
            continue
        if prov.get("arm") != arm:
            continue
        plan_path = Path(prov.get("plan_path", ""))
        if not plan_path.exists():
            # Try relative to plans_dir
            plan_path = plans_dir / f"{prov.get('plan_id', '')}.json"
        if not plan_path.exists():
            continue
        try:
            plan_dicts.append(json.loads(plan_path.read_text()))
        except Exception:
            continue
    return plan_dicts


def fit_and_save(
    plans_dir: Path,
    output_path: Path,
    *,
    min_plans: int = 30,
    seed: int = 0,
) -> None:
    """Fit sampler priors and write sampler_config.json."""
    from generate.sampler import fit_sampler_config_from_plans

    plan_dicts = _load_plan_dicts(plans_dir, arm="llm")

    if len(plan_dicts) < min_plans:
        print(
            f"[warn] Only {len(plan_dicts)} LLM plans found in {plans_dir} "
            f"(min_plans={min_plans}).  Priors may be noisy.",
            file=sys.stderr,
        )

    if not plan_dicts:
        print("[abort] No LLM-arm plans found — cannot fit priors.", file=sys.stderr)
        sys.exit(1)

    cfg = fit_sampler_config_from_plans(plan_dicts, seed=seed)

    # Serialise to JSON (dataclass → dict manually for readability)
    cfg_dict = {
        "plan_days": cfg.plan_days,
        "seed": cfg.seed,
        "p_hard_day": round(cfg.p_hard_day, 4),
        "p_rest_day": round(cfg.p_rest_day, 4),
        "easy_type_probs": {k: round(v, 4) for k, v in cfg.easy_type_probs.items()},
        "duration_by_type": {
            k: [round(m, 1), round(s, 1)]
            for k, (m, s) in cfg.duration_by_type.items()
        },
        "today": cfg.today,
        "plan_start": cfg.plan_start,
        "style": cfg.style,
        "primary_goal": cfg.primary_goal,
        "lifestyle_notes": cfg.lifestyle_notes,
        "readiness_status": cfg.readiness_status,
        "_meta": {
            "fitted_from_n_plans": len(plan_dicts),
            "source": str(plans_dir),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(cfg_dict, indent=2), encoding="utf-8")

    print(f"Fitted priors from {len(plan_dicts)} LLM-arm plans.")
    print(f"  p_hard_day:  {cfg.p_hard_day:.4f}  (~{cfg.p_hard_day * 7:.1f} hard/week)")
    print(f"  p_rest_day:  {cfg.p_rest_day:.4f}  (~{cfg.p_rest_day * 7:.1f} rest/week)")
    print(f"  easy_type distribution: {cfg.easy_type_probs}")
    print(f"[Saved] {output_path}")


def load_sampler_config(config_path: Path) -> "StructuralSamplerConfig":  # noqa: F821
    """Load a fitted StructuralSamplerConfig from a JSON file."""
    from generate.sampler import StructuralSamplerConfig

    raw = json.loads(config_path.read_text())
    # Convert duration_by_type from [[mean, std], ...] to {type: (mean, std)}
    if "duration_by_type" in raw:
        raw["duration_by_type"] = {
            k: tuple(v) for k, v in raw["duration_by_type"].items()
        }
    # Drop metadata key
    raw.pop("_meta", None)
    return StructuralSamplerConfig(**{
        k: v for k, v in raw.items()
        if k in StructuralSamplerConfig.__dataclass_fields__
    })


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit StructuralSamplerConfig priors from LLM arm plan outputs."
    )
    parser.add_argument(
        "--plans-dir", default="plans/",
        help="Directory containing plan JSON files and provenance sidecars."
    )
    parser.add_argument(
        "--output", default="sampler_config.json",
        help="Output path for the fitted config JSON."
    )
    parser.add_argument(
        "--min-plans", type=int, default=30,
        help="Warn (but don't abort) if fewer than this many plans are found."
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Seed for the fitted sampler config."
    )
    args = parser.parse_args()

    fit_and_save(
        plans_dir=Path(args.plans_dir),
        output_path=Path(args.output),
        min_plans=args.min_plans,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
