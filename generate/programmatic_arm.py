"""generate/programmatic_arm.py — programmatic arm plan generation.

Pipeline per plan:
    1. load fixture bundle and prompt-visible metadata
    2. derive a fixture-aware structural sampler config
    3. sample machine plan
    4. apply guardrails
    5. run shared explainer with explicit explainer-stage routing
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
    make_client_from_env,
)
from generate.constants import EXPLAINER_MODEL_ID, PLAN_DAYS
from generate.provenance import PlanProvenance
from generate.sampler import (
    StructuralSamplerConfig,
    sample_machine_plan,
    sampler_config_from_fixture_meta,
)
from generate.temperature import build_programmatic_generation_condition

EXPLAINER_MAX_TOKENS = int(os.getenv("TRAILTRAINING_EXPLAINER_MAX_TOKENS", "12288"))

_PLACEHOLDER_LEAK_MARKERS = (
    ">{signal_id",
    "{signal_id",
)


def _collect_placeholder_leaks(value: Any, path: str = "$") -> list[tuple[str, str]]:
    hits: list[tuple[str, str]] = []

    if isinstance(value, dict):
        for key, child in value.items():
            hits.extend(_collect_placeholder_leaks(child, f"{path}.{key}"))
        return hits

    if isinstance(value, list):
        for idx, child in enumerate(value):
            hits.extend(_collect_placeholder_leaks(child, f"{path}[{idx}]"))
        return hits

    if isinstance(value, str):
        text = value.strip()
        if any(marker in text for marker in _PLACEHOLDER_LEAK_MARKERS):
            hits.append((path, text))
    return hits


def _assert_no_placeholder_leaks(*, plan_obj: dict[str, Any], out_path: Path) -> None:
    hits = _collect_placeholder_leaks(plan_obj)
    if not hits:
        return

    preview = "; ".join(f"{path}={value!r}" for path, value in hits[:10])
    raise ValueError(
        f"Placeholder leak detected in final artifact {out_path.name} "
        f"({len(hits)} hit(s)): {preview}"
    )


def _make_client_from_env() -> Any:
    """Compatibility helper retained for tests and fallback patching."""
    return make_client_from_env(stage="explainer", model_id=EXPLAINER_MODEL_ID)


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


def _extract_response_model_id(response: Any, fallback: str | None = None) -> str | None:
    candidates = [
        getattr(response, "model", None),
        getattr(getattr(response, "response", None), "model", None),
        (response.get("model") if isinstance(response, dict) else None),
    ]
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return fallback


def generate_programmatic_plan(
    fixture_dir: Path,
    output_dir: Path,
    plan_id: str,
    seed: int = 0,
    sampler_cfg: Optional[StructuralSamplerConfig] = None,
    explainer_temperature: float = 0.0,
    generation_condition: str | None = None,
) -> tuple[str, str, str]:
    install_trailtraining_client_compat(default_stage="judge")
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

    if generation_condition is None:
        generation_condition = build_programmatic_generation_condition(
            explainer_temperature=explainer_temperature,
        )

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
        temperature=explainer_temperature,
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

    plan_json, actual_explainer_model = _run_explainer_directly(
        skeleton=skeleton,
        cfg=coach_cfg,
        source_data=source_data,
        deterministic_forecast=data["forecast"],
        effective=effective,
        out_path=output_path,
    )

    explainer_verified = bool(actual_explainer_model and actual_explainer_model == EXPLAINER_MODEL_ID)

    prov = PlanProvenance(
        plan_id=plan_id,
        fixture_id=fixture_id,
        athlete_band=str(fixture_meta.get("athlete_band") or "") or None,
        readiness=str(fixture_meta.get("readiness") or "") or None,
        recovery_capability=str(fixture_meta.get("recovery_capability") or "") or None,
        race_phase=str(fixture_meta.get("race_phase") or "") or None,
        plan_days=PLAN_DAYS,
        style=str(fixture_meta.get("style") or "") or None,
        arm="programmatic",
        source_model=None,
        explainer_model=EXPLAINER_MODEL_ID,
        actual_explainer_model=actual_explainer_model,
        explainer_model_verified=explainer_verified,
        generation_pipeline="programmatic_guardrailed_explainer",
        runtime_backend="generate.programmatic_arm._run_explainer_directly",
        runtime_metadata={
            **describe_client_routing(),
            "actual_explainer_model": actual_explainer_model,
            "explainer_model_verified": explainer_verified,
            "athlete_band": fixture_meta.get("athlete_band"),
            "readiness": fixture_meta.get("readiness"),
            "recovery_capability": fixture_meta.get("recovery_capability"),
            "race_phase": fixture_meta.get("race_phase"),
            "fixture_block_label": fixture_meta.get("block_label"),
            "fixture_primary_goal": fixture_meta.get("primary_goal"),
            "fixture_weeks_to_race": fixture_meta.get("weeks_to_race"),
            "explainer_temperature": explainer_temperature,
            "generation_condition": generation_condition,
        },
        source_temperature=None,
        explainer_temperature=explainer_temperature,
        generation_condition=generation_condition,
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
) -> tuple[str, str | None]:
    import json

    import trailtraining.llm.coach_prompting as coach_prompting
    from trailtraining.llm.coach import (
        _apply_eval_coach_guardrails_compat,
        _build_deterministic_snapshot,
        _finalize_training_plan_artifact,
        _merge_machine_plan_and_explanations,
        _parse_plan_explanation,
    )
    from trailtraining.llm.presets import get_system_prompt
    from trailtraining.llm.rubrics import default_primary_goal_for_style
    from trailtraining.llm.schemas import PLAN_EXPLANATION_STAGE_SCHEMA
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
        "max_tokens": EXPLAINER_MAX_TOKENS,
    }
    if cfg.reasoning_effort == "none" and cfg.temperature is not None:
        explain_kwargs["temperature"] = cfg.temperature

    client = _make_client_from_env()
    response = call_with_schema(client, explain_kwargs, PLAN_EXPLANATION_STAGE_SCHEMA)
    actual_explainer_model = _extract_response_model_id(response, fallback=EXPLAINER_MODEL_ID)
    explain_text = getattr(response, "output_text", None) or str(response)
    explanation_obj = _parse_plan_explanation(
        explain_text,
        client,
        cfg,
        str(explain_kwargs.get("instructions") or ""),
    )
    explanation_obj["snapshot"] = _build_deterministic_snapshot(source_data.combined)

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

    _assert_no_placeholder_leaks(
        plan_obj=obj,
        out_path=out_path,
    )

    save_json(out_path, obj, compact=False)
    return json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default), actual_explainer_model
