"""Fit StructuralSamplerConfig priors from LLM arm outputs.

Abort by default if the retained LLM corpus is tiny or band-skewed, so the
programmatic arm does not inherit a collapsed LLM-arm distribution.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

from generate.structural_expectations import detect_understructured_plan
from match.filtering import detect_plan_issues

REQUIRED_ATHLETE_BANDS = ("A1", "A2", "A3", "A4")


def _env_flag(name: str) -> bool:
    return (os.getenv(name) or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _load_plan_bundles(plans_dir: Path, arm: str = "llm") -> list[dict]:
    bundles = []
    for prov_path in sorted(plans_dir.glob("*.provenance.json")):
        try:
            prov = json.loads(prov_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if prov.get("arm") != arm:
            continue
        plan_path = Path(prov.get("plan_path", ""))
        if not plan_path.exists():
            plan_path = plans_dir / f"{prov.get('plan_id', '')}.json"
        if not plan_path.exists():
            continue
        try:
            bundles.append({"plan": json.loads(plan_path.read_text(encoding="utf-8")), "provenance": prov})
        except Exception:
            continue
    return bundles


def _fixture_meta_from_provenance(provenance: dict) -> dict:
    return {
        "athlete_band": provenance.get("athlete_band"),
        "readiness": provenance.get("readiness"),
        "recovery_capability": provenance.get("recovery_capability"),
        "race_phase": provenance.get("race_phase"),
    }


def fit_and_save(
    plans_dir: Path,
    output_path: Path,
    *,
    min_plans: int = 30,
    min_plans_per_band: int = 4,
    seed: int = 0,
    allow_unsafe_prior_fit: bool | None = None,
) -> None:
    from generate.sampler import fit_sampler_config_from_plans

    bundles = _load_plan_bundles(plans_dir, arm="llm")
    valid_bundles: list[dict] = []
    dropped_invalid = 0
    dropped_understructured = 0

    for bundle in bundles:
        plan = bundle.get("plan")
        provenance = bundle.get("provenance") or {}
        if not isinstance(plan, dict):
            continue
        if detect_plan_issues(plan):
            dropped_invalid += 1
            continue
        fixture_meta = _fixture_meta_from_provenance(provenance)
        if detect_understructured_plan(plan, fixture_meta):
            dropped_understructured += 1
            continue
        valid_bundles.append(bundle)

    if dropped_invalid:
        print(f"[warn] Excluding {dropped_invalid} invalid/contradictory LLM plans from prior fitting.", file=sys.stderr)
    if dropped_understructured:
        print(f"[warn] Excluding {dropped_understructured} structurally underexpressed LLM plans from prior fitting.", file=sys.stderr)

    if allow_unsafe_prior_fit is None:
        allow_unsafe_prior_fit = _env_flag("ALLOW_UNSAFE_PRIOR_FIT")

    valid_plan_dicts = [bundle["plan"] for bundle in valid_bundles]
    if not valid_plan_dicts:
        print("[abort] No valid LLM-arm plans found - cannot fit priors.", file=sys.stderr)
        sys.exit(1)

    valid_band_counts = Counter(
        str((bundle.get("provenance") or {}).get("athlete_band") or "")
        for bundle in valid_bundles
    )
    print(f"  per-band valid counts: {dict(valid_band_counts)}", file=sys.stderr)

    unsafe_reasons: list[str] = []
    if len(valid_plan_dicts) < min_plans:
        unsafe_reasons.append(
            f"only {len(valid_plan_dicts)} valid LLM plans found in {plans_dir} (min_plans={min_plans})"
        )
    under_by_band = {
        band: valid_band_counts.get(band, 0)
        for band in REQUIRED_ATHLETE_BANDS
        if valid_band_counts.get(band, 0) < min_plans_per_band
    }
    if under_by_band:
        unsafe_reasons.append(
            f"insufficient per-band coverage (min_plans_per_band={min_plans_per_band}): {under_by_band}"
        )

    if unsafe_reasons and not allow_unsafe_prior_fit:
        print(
            "[abort] Refusing to fit programmatic priors on an unsafe retained LLM sample: "
            + "; ".join(unsafe_reasons)
            + ". Generate more valid LLM plans, lower the thresholds intentionally, "
            + "or pass --allow-unsafe-prior-fit / ALLOW_UNSAFE_PRIOR_FIT=1.",
            file=sys.stderr,
        )
        sys.exit(1)
    if unsafe_reasons:
        print("[warn] Fitting priors despite unsafe retained LLM sample: " + "; ".join(unsafe_reasons), file=sys.stderr)

    cfg = fit_sampler_config_from_plans(valid_plan_dicts, seed=seed)
    cfg_dict = {
        "plan_days": cfg.plan_days,
        "seed": cfg.seed,
        "p_hard_day": round(cfg.p_hard_day, 4),
        "p_rest_day": round(cfg.p_rest_day, 4),
        "easy_type_probs": {k: round(v, 4) for k, v in cfg.easy_type_probs.items()},
        "duration_by_type": {k: [round(m, 1), round(s, 1)] for k, (m, s) in cfg.duration_by_type.items()},
        "today": cfg.today,
        "plan_start": cfg.plan_start,
        "style": cfg.style,
        "primary_goal": cfg.primary_goal,
        "lifestyle_notes": cfg.lifestyle_notes,
        "readiness_status": cfg.readiness_status,
        "_meta": {
            "loaded_llm_plans": len(bundles),
            "fitted_from_valid_llm_plans": len(valid_plan_dicts),
            "excluded_invalid_llm_plans": dropped_invalid,
            "excluded_understructured_llm_plans": dropped_understructured,
            "valid_band_counts": dict(valid_band_counts),
            "source": str(plans_dir),
            "allow_unsafe_prior_fit": bool(allow_unsafe_prior_fit),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(cfg_dict, indent=2), encoding="utf-8")
    print(f"Fitted priors from {len(valid_plan_dicts)} valid LLM-arm plans.")
    print(f"  p_hard_day:  {cfg.p_hard_day:.4f}  (~{cfg.p_hard_day * 7:.1f} hard/week)")
    print(f"  p_rest_day:  {cfg.p_rest_day:.4f}  (~{cfg.p_rest_day * 7:.1f} rest/week)")
    print(f"  easy_type distribution: {cfg.easy_type_probs}")
    print(f"[Saved] {output_path}")


def load_sampler_config(config_path: Path) -> "StructuralSamplerConfig":  # noqa: F821
    from generate.sampler import StructuralSamplerConfig
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if "duration_by_type" in raw:
        raw["duration_by_type"] = {k: tuple(v) for k, v in raw["duration_by_type"].items()}
    raw.pop("_meta", None)
    return StructuralSamplerConfig(**{k: v for k, v in raw.items() if k in StructuralSamplerConfig.__dataclass_fields__})


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit StructuralSamplerConfig priors from LLM arm plan outputs.")
    parser.add_argument("--plans-dir", default="plans/", help="Directory containing plan JSON files and provenance sidecars.")
    parser.add_argument("--output", default="sampler_config.json", help="Output path for the fitted config JSON.")
    parser.add_argument("--min-plans", type=int, default=30, help="Abort unless at least this many valid LLM plans remain after filtering.")
    parser.add_argument("--min-plans-per-band", type=int, default=4, help="Abort unless each A1-A4 band has at least this many valid plans.")
    parser.add_argument("--allow-unsafe-prior-fit", action="store_true", help="Permit fitting priors below total/per-band safety thresholds.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the fitted sampler config.")
    args = parser.parse_args()
    fit_and_save(
        plans_dir=Path(args.plans_dir),
        output_path=Path(args.output),
        min_plans=args.min_plans,
        min_plans_per_band=args.min_plans_per_band,
        seed=args.seed,
        allow_unsafe_prior_fit=args.allow_unsafe_prior_fit,
    )


if __name__ == "__main__":
    main()
