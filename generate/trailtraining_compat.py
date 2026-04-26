from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import trailtraining.llm.coach_prompting as _coach_prompting
from compat.trailtraining_client import (
    describe_client_routing,
    ensure_dual_endpoint_support,
    make_stage_client,
)
from generate.constants import PLAN_DAYS


def _load_fixture(fixture_dir: Path) -> dict[str, Any]:
    combined = json.loads((fixture_dir / "combined_summary.json").read_text(encoding="utf-8"))
    personal = json.loads((fixture_dir / "formatted_personal_data.json").read_text(encoding="utf-8"))
    rollups_path = fixture_dir / "combined_rollups.json"
    forecast_path = fixture_dir / "readiness_and_risk_forecast.json"
    meta_path = fixture_dir / "fixture_meta.json"
    return {
        "combined": combined,
        "personal": personal,
        "rollups": json.loads(rollups_path.read_text(encoding="utf-8")) if rollups_path.exists() else None,
        "forecast": json.loads(forecast_path.read_text(encoding="utf-8")) if forecast_path.exists() else None,
        "fixture_meta": json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {},
    }


def run_two_stage_generation_compat(
    *,
    fixture_dir: Path,
    output_path: Path,
    source_model: str,
    explainer_model: str,
    primary_goal: str,
    seed: int,
) -> tuple[str, dict[str, Any]]:
    from trailtraining.llm.coach import (
        CoachConfig,
        _apply_eval_coach_guardrails_compat,
        _finalize_training_plan_artifact,
        _merge_machine_plan_and_explanations,
        _parse_machine_plan,
        _parse_plan_explanation,
    )
    from trailtraining.llm.constraints import constraint_config_from_env, derive_effective_constraints
    from trailtraining.llm.presets import get_system_prompt
    from trailtraining.llm.rubrics import default_primary_goal_for_style
    from trailtraining.llm.schemas import MACHINE_PLAN_SCHEMA, PLAN_EXPLANATION_SCHEMA
    from trailtraining.llm.shared import call_with_schema, recompute_planned_hours
    from trailtraining.util.state import _json_default, save_json

    fixture_dir = Path(fixture_dir)
    output_path = Path(output_path)
    data = _load_fixture(fixture_dir)

    ensure_dual_endpoint_support(source_model=source_model, explainer_model=explainer_model)

    style = "trailrunning"
    resolved_goal = (primary_goal or "").strip() or default_primary_goal_for_style(style)
    detail_days = max(1, min(14, len(data["combined"])))

    effective = derive_effective_constraints(
        det_forecast=data["forecast"],
        rollups=data["rollups"],
        cfg=constraint_config_from_env(),
        lifestyle_notes="",
    )

    source_cfg = CoachConfig(
        model=source_model,
        reasoning_effort="none",
        temperature=0.7,
        plan_days=PLAN_DAYS,
        primary_goal=resolved_goal,
        style=style,
    )
    explainer_cfg = CoachConfig(
        model=explainer_model,
        reasoning_effort="none",
        temperature=0.0,
        plan_days=PLAN_DAYS,
        primary_goal=resolved_goal,
        style=style,
    )

    source_client = make_stage_client(stage="source", model_id=source_model)
    explainer_client = make_stage_client(stage="explainer", model_id=explainer_model)

    machine_prompt = _coach_prompting.build_machine_plan_prompt_text(
        personal=data["personal"],
        rollups=data["rollups"],
        combined=data["combined"],
        deterministic_forecast=data["forecast"],
        style=style,
        primary_goal=resolved_goal,
        lifestyle_notes="",
        max_chars=source_cfg.max_chars,
        detail_days=detail_days,
        plan_days=source_cfg.plan_days,
        effective_constraints=effective,
    )
    machine_kwargs: dict[str, Any] = {
        "model": source_model,
        "instructions": get_system_prompt(style),
        "input": machine_prompt,
        "reasoning": {"effort": source_cfg.reasoning_effort},
        "text": {"verbosity": source_cfg.verbosity},
    }
    if source_cfg.temperature is not None:
        machine_kwargs["temperature"] = source_cfg.temperature
    machine_resp = call_with_schema(source_client, machine_kwargs, MACHINE_PLAN_SCHEMA)
    machine_text = getattr(machine_resp, "output_text", None) or str(machine_resp)
    machine_obj = _parse_machine_plan(
        machine_text,
        source_client,
        source_cfg,
        str(machine_kwargs.get("instructions") or ""),
    )

    guarded_stub: dict[str, Any] = {
        "meta": dict((machine_obj.get("meta") or {})),
        "snapshot": {
            "last7": {
                "distance_km": "",
                "moving_time_hours": "",
                "elevation_m": "",
                "activity_count": "",
                "sleep_hours_mean": "",
                "hrv_mean": "",
                "rhr_mean": "",
            },
            "baseline28": {
                "distance_km": "",
                "moving_time_hours": "",
                "elevation_m": "",
                "activity_count": "",
                "sleep_hours_mean": "",
                "hrv_mean": "",
                "rhr_mean": "",
            },
            "notes": "",
        },
        "readiness": {
            "status": str(((machine_obj.get("readiness") or {}).get("status") or "steady")),
            "rationale": "",
            "signal_ids": [],
        },
        "plan": {
            "weekly_totals": dict(((machine_obj.get("plan") or {}).get("weekly_totals") or {})),
            "days": [
                {
                    "date": d.get("date"),
                    "title": "",
                    "session_type": d.get("session_type"),
                    "is_rest_day": d.get("is_rest_day"),
                    "is_hard_day": d.get("is_hard_day"),
                    "duration_minutes": d.get("duration_minutes"),
                    "target_intensity": d.get("target_intensity"),
                    "terrain": d.get("terrain"),
                    "workout": d.get("workout"),
                    "purpose": "",
                    "signal_ids": [],
                }
                for d in ((machine_obj.get("plan") or {}).get("days") or [])
                if isinstance(d, dict)
            ],
        },
        "recovery": {"actions": [], "signal_ids": []},
        "risks": [],
        "data_notes": [],
        "citations": [],
        "claim_attributions": [],
        "effective_constraints": None,
    }
    _apply_eval_coach_guardrails_compat(guarded_stub, data["rollups"], effective)
    machine_obj["plan"] = guarded_stub["plan"]

    explainer_prompt = _coach_prompting.build_explainer_prompt_text(
        machine_plan=machine_obj,
        personal=data["personal"],
        rollups=data["rollups"],
        combined=data["combined"],
        deterministic_forecast=data["forecast"],
        style=style,
        primary_goal=resolved_goal,
        lifestyle_notes="",
        max_chars=explainer_cfg.max_chars,
        detail_days=detail_days,
        effective_constraints=effective,
    )
    explain_kwargs: dict[str, Any] = {
        "model": explainer_model,
        "instructions": get_system_prompt(style),
        "input": explainer_prompt,
        "reasoning": {"effort": explainer_cfg.reasoning_effort},
        "text": {"verbosity": explainer_cfg.verbosity},
    }
    if explainer_cfg.temperature is not None:
        explain_kwargs["temperature"] = explainer_cfg.temperature
    explain_resp = call_with_schema(explainer_client, explain_kwargs, PLAN_EXPLANATION_SCHEMA)
    explain_text = getattr(explain_resp, "output_text", None) or str(explain_resp)
    explanation_obj = _parse_plan_explanation(
        explain_text,
        explainer_client,
        explainer_cfg,
        str(explain_kwargs.get("instructions") or ""),
    )

    obj = _merge_machine_plan_and_explanations(
        machine_obj,
        explanation_obj,
        resolved_goal=resolved_goal,
        lifestyle_notes="",
        deterministic_forecast=data["forecast"],
        effective=effective,
    )
    recompute_planned_hours(obj)
    _apply_eval_coach_guardrails_compat(obj, data["rollups"], effective)
    obj = _finalize_training_plan_artifact(
        obj,
        combined=data["combined"],
        rollups=data["rollups"],
        deterministic_forecast=data["forecast"],
        effective=effective,
    )

    save_json(output_path, obj, compact=False)
    runtime_metadata = {
        **describe_client_routing(),
        "source_model": source_model,
        "actual_explainer_model": explainer_model,
        "seed": seed,
    }
    return json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default), runtime_metadata
