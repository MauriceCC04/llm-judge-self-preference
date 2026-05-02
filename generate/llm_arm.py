"""generate/llm_arm.py — LLM arm plan generation with local two-stage compatibility."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from compat.trailtraining_client import install_trailtraining_client_compat
from generate.constants import EXPLAINER_MODEL_ID
from generate.provenance import PlanProvenance
from generate.temperature import build_llm_generation_condition
from generate.trailtraining_compat import run_two_stage_generation_compat


def _bool_strict(value: object) -> bool:
    return value is True


def generate_llm_plan(
    fixture_dir: Path,
    output_dir: Path,
    plan_id: str,
    source_model: str,
    seed: int = 0,
    source_temperature: float = 0.7,
    explainer_temperature: float = 0.0,
    generation_condition: str | None = None,
) -> tuple[str, str, str]:
    install_trailtraining_client_compat(default_stage="judge")
    os.environ.setdefault("OPENROUTER_API_KEY", "dummy")

    fixture_dir = Path(fixture_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta_path = fixture_dir / "fixture_meta.json"
    fixture_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    fixture_id = fixture_meta.get("fixture_id", fixture_dir.name)
    primary_goal = str(fixture_meta.get("primary_goal") or "to become a faster trail runner")

    if generation_condition is None:
        generation_condition = build_llm_generation_condition(
            source_temperature=source_temperature,
            explainer_temperature=explainer_temperature,
        )

    plan_path = output_dir / f"{plan_id}.json"

    plan_json, runtime_metadata = run_two_stage_generation_compat(
        fixture_dir=fixture_dir,
        output_path=plan_path,
        source_model=source_model,
        explainer_model=EXPLAINER_MODEL_ID,
        primary_goal=primary_goal,
        seed=seed,
        source_temperature=source_temperature,
        explainer_temperature=explainer_temperature,
    )

    runtime_metadata = dict(runtime_metadata or {})
    runtime_metadata.setdefault("TRAILTRAINING_TWO_STAGE_PLAN", "1")
    runtime_metadata.setdefault("source_temperature", source_temperature)
    runtime_metadata.setdefault("explainer_temperature", explainer_temperature)
    runtime_metadata.setdefault("generation_condition", generation_condition)
    runtime_metadata.setdefault("athlete_band", fixture_meta.get("athlete_band"))
    runtime_metadata.setdefault("readiness", fixture_meta.get("readiness"))
    runtime_metadata.setdefault("recovery_capability", fixture_meta.get("recovery_capability"))
    runtime_metadata.setdefault("race_phase", fixture_meta.get("race_phase"))

    actual_explainer_model_raw = runtime_metadata.get("actual_explainer_model")
    actual_explainer_model = (
        str(actual_explainer_model_raw).strip()
        if isinstance(actual_explainer_model_raw, str) and actual_explainer_model_raw.strip()
        else None
    )

    explainer_verified = bool(
        _bool_strict(runtime_metadata.get("explainer_model_verified"))
        and actual_explainer_model == EXPLAINER_MODEL_ID
    )
    runtime_metadata["explainer_model_verified"] = explainer_verified

    prov = PlanProvenance(
        plan_id=plan_id,
        fixture_id=fixture_id,
        athlete_band=str(fixture_meta.get("athlete_band") or "") or None,
        readiness=str(fixture_meta.get("readiness") or "") or None,
        recovery_capability=str(fixture_meta.get("recovery_capability") or "") or None,
        race_phase=str(fixture_meta.get("race_phase") or "") or None,
        plan_days=7,
        style=str(fixture_meta.get("style") or "") or None,
        arm="llm",
        source_model=source_model,
        explainer_model=EXPLAINER_MODEL_ID,
        actual_explainer_model=actual_explainer_model,
        explainer_model_verified=explainer_verified,
        generation_pipeline="llm_two_stage",
        runtime_backend="trailtraining.llm.coach.run_coach_brief",
        runtime_metadata=runtime_metadata,
        source_temperature=source_temperature,
        explainer_temperature=explainer_temperature,
        generation_condition=generation_condition,
        seed=seed,
        generated_at=datetime.now(tz=timezone.utc).isoformat(),
        plan_path=str(plan_path),
    )

    prov_path = output_dir / f"{plan_id}.json.provenance.json"
    prov_path.write_text(prov.model_dump_json(indent=2), encoding="utf-8")

    return plan_json, str(plan_path), str(prov_path)
