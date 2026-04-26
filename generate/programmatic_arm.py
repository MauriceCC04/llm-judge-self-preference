"""generate/programmatic_arm.py — programmatic arm plan generation.

Pipeline per plan:
    1. load fixture bundle and prompt-visible metadata
    2. derive a fixture-aware structural sampler config
    3. sample machine plan
    4. apply guardrails
    5. run shared explainer
"""
from __future__ import annotations

import json
import os
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from compat.trailtraining_client import (
    describe_client_routing,
    install_trailtraining_client_compat,
    make_stage_client,
)
from generate.constants import EXPLAINER_MODEL_ID, PLAN_DAYS
from generate.provenance import PlanProvenance
from generate.sampler import StructuralSamplerConfig, sample_machine_plan, sampler_config_from_fixture_meta


def _load_fixture(fixture_dir: Path) -> dict[str, Any]:
    combined = json.loads((fixture_dir / "combined_summary.json").read_text(encoding="utf-8"))
    rollups_path = fixture_dir / "combined_rollups.json"
    rollups = json.loads(rollups_path.read_text(encoding="utf-8")) if rollups_path.exists() else None
    forecast_path = fixture_dir / "readiness_and_risk_forecast.json"
    forecast = json.loads(forecast_path.read_text(encoding="utf-8")) if forecast_path.exists() else None
    personal = json.loads((fixture_dir / "formatted_personal_data.json").read_text(encoding="utf-8"))
    meta_path = fixture_dir / "fixture_meta.json"
    fixture_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return {
        "combined": combined,
        "rollups": rollups,
        "forecast": forecast,
        "personal": personal,
        "fixture_meta": fixture_meta,
    }


def _next_day(iso_date: str) -> str:
    from datetime import date, timedelta

    return (date.fromisoformat(iso_date) + timedelta(days=1)).isoformat()


def _make_client_from_env():
    from openai import OpenAI

    base_url = (os.getenv("TRAILTRAINING_LLM_BASE_URL") or "").strip()
    api_key = (os.getenv("OPENROUTER_API_KEY") or os.getenv("TRAILTRAINING_OPENROUTER_API_KEY") or "").strip()
    if api_key.lower() == "dummy":
        api_key = ""
    if base_url:
        return OpenAI(base_url=base_url, api_key=api_key or "dummy")
    if not api_key:
        raise RuntimeError(
            "No LLM endpoint configured. Set TRAILTRAINING_LLM_BASE_URL "
            "(local vLLM) or OPENROUTER_API_KEY / TRAILTRAINING_OPENROUTER_API_KEY."
        )
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


def _sampler_cfg_from_fixture(
    data: dict[str, Any],
    *,
    seed: int,
    base_cfg: Optional[StructuralSamplerConfig],
) -> StructuralSamplerConfig:
    fixture_meta = data["fixture_meta"]
    combined = data["combined"]
    last_date = combined[-1]["date"] if combined else "2026-03-17"
    return sampler_config_from_fixture_meta(
        fixture_meta,
        seed=seed,
        plan_days=PLAN_DAYS,
        today=last_date,
        plan_start=_next_day(last_date),
        base_cfg=base_cfg,
    )


def generate_programmatic_plan(
    fixture_dir: Path,
    output_dir: Path,
    plan_id: str,
    seed: int = 0,
    sampler_cfg: Optional[StructuralSamplerConfig] = None,
) -> tuple[str, str, str]:
    install_trailtraining_client_compat()
    os.environ.setdefault("OPENROUTER_API_KEY", "dummy")

    from trailtraining.llm.coach import CoachConfig
    from trailtraining.llm.constraints import constraint_config_from_env, derive_effective_constraints
    from trailtraining.llm.guardrails import apply_eval_coach_guardrails
    from trailtraining.llm.schemas import ensure_machine_plan_shape

    fixture_dir = Path(fixture_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = _load_fixture(fixture_dir)
    fixture_meta = data["fixture_meta"]
    fixture_id = fixture_meta.get("fixture_id", fixture_dir.name)
    cfg = _sampler_cfg_from_fixture(data, seed=seed, base_cfg=sampler_cfg)

    skeleton = sample_machine_plan(cfg, combined=data["combined"], rollups=data["rollups"])
    skeleton = ensure_machine_plan_shape(skeleton)

    effective = derive_effective_constraints(
        det_forecast=data["forecast"],
        rollups=data["rollups"],
        cfg=constraint_config_from_env(),
    )
    apply_eval_coach_guardrails(skeleton, data["rollups"], effective=effective)

    coach_cfg = CoachConfig(
        model=EXPLAINER_MODEL_ID,
        reasoning_effort="none",
        temperature=0.0,
        plan_days=PLAN_DAYS,
        style=cfg.style,
        primary_goal=cfg.primary_goal,
        lifestyle_notes=cfg.lifestyle_notes,
    )

    source_data = types.SimpleNamespace(
        personal=data["personal"],
        combined=data["combined"],
        rollups=data["rollups"],
    )
    output_path = output_dir / f"{plan_id}.json"

    try:
        from trailtraining.llm.coach import run_training_plan_from_machine_plan

        plan_json, _ = run_training_plan_from_machine_plan(
            skeleton,
            cfg=coach_cfg,
            source_data=source_data,
            deterministic_forecast=data["forecast"],
            effective=effective,
            output_path=str(output_path),
        )
        runtime_backend = "trailtraining.llm.coach.run_training_plan_from_machine_plan"
    except (ImportError, AttributeError):
        plan_json = _run_explainer_directly(
            skeleton=skeleton,
            cfg=coach_cfg,
            source_data=source_data,
            deterministic_forecast=data["forecast"],
            effective=effective,
            out_path=output_path,
        )
        runtime_backend = "generate.programmatic_arm._run_explainer_directly"

    prov = PlanProvenance(
        plan_id=plan_id,
        fixture_id=fixture_id,
        arm="programmatic",
        source_model=None,
        explainer_model=EXPLAINER_MODEL_ID,
        actual_explainer_model=EXPLAINER_MODEL_ID,
        explainer_model_verified=True,
        generation_pipeline="programmatic_guardrailed_explainer",
        runtime_backend=runtime_backend,
        runtime_metadata={
            **describe_client_routing(),
            "actual_explainer_model": EXPLAINER_MODEL_ID,
            "fixture_block_label": fixture_meta.get("block_label"),
            "fixture_primary_goal": fixture_meta.get("primary_goal"),
            "fixture_weeks_to_race": fixture_meta.get("weeks_to_race"),
        },
        seed=seed,
        generated_at=datetime.now(tz=timezone.utc).isoformat(),
        plan_path=str(output_path),
    )
    provenance_path = output_dir / f"{plan_id}.json.provenance.json"
    provenance_path.write_text(prov.model_dump_json(indent=2), encoding="utf-8")
    return plan_json, str(output_path), str(provenance_path)


def _run_explainer_directly(
    *,
    skeleton: dict[str, Any],
    cfg: Any,
    source_data: Any,
    deterministic_forecast: Any,
    effective: Any,
    out_path: Path,
) -> str:
    import json

    import trailtraining.llm.coach_prompting as coach_prompting
    from trailtraining.llm.coach import (
        _apply_eval_coach_guardrails_compat,
        _finalize_training_plan_artifact,
        _merge_machine_plan_and_explanations,
        _parse_plan_explanation,
    )
    from trailtraining.llm.presets import get_system_prompt
    from trailtraining.llm.rubrics import default_primary_goal_for_style
    from trailtraining.llm.schemas import PLAN_EXPLANATION_SCHEMA
    from trailtraining.llm.shared import call_with_schema, recompute_planned_hours
    from trailtraining.util.state import _json_default, save_json

    resolved_goal = (cfg.primary_goal or "").strip() or default_primary_goal_for_style(cfg.style)
    detail_days = max(1, min(14, len(source_data.combined)))
    explainer_prompt = coach_prompting.build_explainer_prompt_text(
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

    client = make_stage_client(stage="explainer", model_id=EXPLAINER_MODEL_ID)
    response = call_with_schema(client, explain_kwargs, PLAN_EXPLANATION_SCHEMA)
    explain_text = getattr(response, "output_text", None) or str(response)
    explanation_obj = _parse_plan_explanation(
        explain_text,
        client,
        cfg,
        str(explain_kwargs.get("instructions") or ""),
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
