"""generate/programmatic_arm.py — programmatic arm plan generation.

Pipeline per plan:
    1. sample_machine_plan(cfg, fixture)
    2. ensure_machine_plan_shape(skeleton)
    3. apply_eval_coach_guardrails(skeleton, rollups, effective)
    4. run_training_plan_from_machine_plan OR fallback explainer call
"""
from __future__ import annotations

import json
import os
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from generate.constants import EXPLAINER_MODEL_ID, PLAN_DAYS
from generate.provenance import PlanProvenance
from generate.sampler import StructuralSamplerConfig, sample_machine_plan


def _load_fixture(fixture_dir: Path) -> dict[str, Any]:
    combined = json.loads((fixture_dir / "combined_summary.json").read_text())
    rollups_path = fixture_dir / "combined_rollups.json"
    rollups = json.loads(rollups_path.read_text()) if rollups_path.exists() else None
    forecast_path = fixture_dir / "readiness_and_risk_forecast.json"
    forecast = json.loads(forecast_path.read_text()) if forecast_path.exists() else None
    personal = json.loads((fixture_dir / "formatted_personal_data.json").read_text())
    meta_path = fixture_dir / "fixture_meta.json"
    fixture_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return {
        "combined": combined,
        "rollups": rollups,
        "forecast": forecast,
        "personal": personal,
        "fixture_meta": fixture_meta,
    }


def _next_day(iso_date: str) -> str:
    from datetime import date, timedelta
    d = date.fromisoformat(iso_date)
    return (d + timedelta(days=1)).isoformat()


def generate_programmatic_plan(
    fixture_dir: Path,
    output_dir: Path,
    plan_id: str,
    seed: int = 0,
    sampler_cfg: Optional[StructuralSamplerConfig] = None,
) -> tuple[str, str, str]:
    """Generate one programmatic-arm plan.

    Returns (plan_json_str, plan_path, provenance_path).
    """
    from trailtraining.llm.coach import CoachConfig
    from trailtraining.llm.constraints import (
        derive_effective_constraints,
        constraint_config_from_env,
    )
    from trailtraining.llm.guardrails import apply_eval_coach_guardrails
    from trailtraining.llm.schemas import ensure_machine_plan_shape

    fixture_dir = Path(fixture_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = _load_fixture(fixture_dir)
    fixture_meta = data["fixture_meta"]
    fixture_id = fixture_meta.get("fixture_id", fixture_dir.name)

    # Build StructuralSamplerConfig from fixture metadata
    if sampler_cfg is None:
        readiness_status = fixture_meta.get("readiness_status", "steady")
        last_date = data["combined"][-1]["date"] if data["combined"] else "2026-03-17"
        sampler_cfg = StructuralSamplerConfig(
            plan_days=PLAN_DAYS,
            seed=seed,
            readiness_status=readiness_status,
            today=last_date,
            plan_start=_next_day(last_date),
        )

    # 1. Sample skeleton
    skeleton = sample_machine_plan(sampler_cfg, combined=data["combined"], rollups=data["rollups"])

    # 2. Validate shape
    skeleton = ensure_machine_plan_shape(skeleton)

    # 3. Guardrails
    effective = derive_effective_constraints(
        det_forecast=data["forecast"],
        rollups=data["rollups"],
        cfg=constraint_config_from_env(),
    )
    apply_eval_coach_guardrails(skeleton, data["rollups"], effective=effective)

    # 4. Run explainer
    cfg = CoachConfig(
        model=EXPLAINER_MODEL_ID,
        reasoning_effort="none",
        temperature=0.0,
        plan_days=PLAN_DAYS,
    )

    source_data = types.SimpleNamespace(
        personal=data["personal"],
        combined=data["combined"],
        rollups=data["rollups"],
    )

    out_path = output_dir / f"{plan_id}.json"

    # Try PR-2 public helper first, then fall back to manual explainer call
    try:
        from trailtraining.llm.coach import run_training_plan_from_machine_plan
        plan_json, _ = run_training_plan_from_machine_plan(
            skeleton,
            cfg=cfg,
            source_data=source_data,
            deterministic_forecast=data["forecast"],
            effective=effective,
            output_path=str(out_path),
        )
    except (ImportError, AttributeError):
        plan_json = _run_explainer_directly(
            skeleton=skeleton,
            cfg=cfg,
            source_data=source_data,
            deterministic_forecast=data["forecast"],
            effective=effective,
            out_path=out_path,
        )

    # Write provenance sidecar
    prov = PlanProvenance(
        plan_id=plan_id,
        fixture_id=fixture_id,
        arm="programmatic",
        source_model=None,
        explainer_model=EXPLAINER_MODEL_ID,
        seed=seed,
        generated_at=datetime.now(tz=timezone.utc).isoformat(),
        plan_path=str(out_path),
    )
    prov_path = output_dir / f"{plan_id}.json.provenance.json"
    prov_path.write_text(prov.model_dump_json(indent=2), encoding="utf-8")

    return plan_json, str(out_path), str(prov_path)


def _make_client_from_env() -> "OpenAI":  # type: ignore[name-defined]
    """Build an OpenAI-compatible client from env vars without PR-1.

    Works with both OpenRouter (production) and a local vLLM server (HPC).
    Priority:
        TRAILTRAINING_LLM_BASE_URL  → local vLLM override (study HPC path)
        OPENROUTER_API_KEY          → openrouter production
    """
    from openai import OpenAI

    base_url = (os.getenv("TRAILTRAINING_LLM_BASE_URL") or "").strip()
    api_key = (
        os.getenv("OPENROUTER_API_KEY")
        or os.getenv("TRAILTRAINING_OPENROUTER_API_KEY")
        or ""
    ).strip()

    if base_url:
        return OpenAI(base_url=base_url, api_key=api_key or "dummy")

    if not api_key:
        raise RuntimeError(
            "No LLM endpoint configured.  Set TRAILTRAINING_LLM_BASE_URL "
            "(local vLLM) or OPENROUTER_API_KEY (OpenRouter production)."
        )
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


def _run_explainer_directly(
    *,
    skeleton: dict[str, Any],
    cfg: Any,
    source_data: Any,
    deterministic_forecast: Any,
    effective: Any,
    out_path: Path,
) -> str:
    """Fallback: call the explainer directly using the public coach machinery.

    This path is taken when PR-2 (run_training_plan_from_machine_plan) has not
    yet been merged into trailtraining.  It avoids calling private coach
    functions directly; instead it delegates to the public ``run_coach_brief``
    interface with a pre-built machine plan injected via environment.
    """
    import json
    from trailtraining.llm.coach import (
        _merge_machine_plan_and_explanations,
        _parse_plan_explanation,
        _finalize_training_plan_artifact,
        _apply_eval_coach_guardrails_compat,
    )
    from trailtraining.llm.presets import get_system_prompt
    from trailtraining.llm.schemas import PLAN_EXPLANATION_SCHEMA
    from trailtraining.llm.shared import (
        call_with_schema,
        recompute_planned_hours,
    )
    from trailtraining.llm.rubrics import default_primary_goal_for_style
    from trailtraining.util.state import save_json, _json_default
    import trailtraining.llm.coach_prompting as _coach_prompting

    resolved_goal = (cfg.primary_goal or "").strip() or default_primary_goal_for_style(cfg.style)
    detail_days = max(1, min(14, len(source_data.combined)))

    explainer_prompt = _coach_prompting.build_explainer_prompt_text(
        machine_plan=skeleton,
        personal=source_data.personal,
        rollups=source_data.rollups,
        combined=source_data.combined,
        deterministic_forecast=deterministic_forecast,
        style=cfg.style,
        primary_goal=resolved_goal,
        lifestyle_notes=cfg.lifestyle_notes,
        max_chars=cfg.max_chars,
        detail_days=detail_days,
        effective_constraints=effective,
    )

    explain_kwargs: dict[str, Any] = {
        "model": cfg.model,
        "instructions": get_system_prompt(cfg.style),
        "input": explainer_prompt,
        "reasoning": {"effort": cfg.reasoning_effort},
        "text": {"verbosity": cfg.verbosity},
    }
    if cfg.reasoning_effort == "none" and cfg.temperature is not None:
        explain_kwargs["temperature"] = cfg.temperature

    # Build client without depending on private _make_openrouter_client (PR-1)
    client = _make_client_from_env()
    explain_resp = call_with_schema(client, explain_kwargs, PLAN_EXPLANATION_SCHEMA)
    explain_text = getattr(explain_resp, "output_text", None) or str(explain_resp)
    explanation_obj = _parse_plan_explanation(
        explain_text, client, cfg, str(explain_kwargs.get("instructions") or "")
    )

    obj = _merge_machine_plan_and_explanations(
        skeleton,
        explanation_obj,
        resolved_goal=resolved_goal,
        lifestyle_notes=cfg.lifestyle_notes,
        deterministic_forecast=deterministic_forecast,
        effective=effective,
    )
    recompute_planned_hours(obj)
    _apply_eval_coach_guardrails_compat(obj, source_data.rollups, effective)
    obj = _finalize_training_plan_artifact(
        obj,
        combined=source_data.combined,
        rollups=source_data.rollups,
        deterministic_forecast=deterministic_forecast,
        effective=effective,
    )

    save_json(out_path, obj, compact=False)
    return json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default)
