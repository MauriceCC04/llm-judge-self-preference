"""generate/llm_arm.py — LLM arm plan generation with local two-stage compatibility."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from generate.constants import EXPLAINER_MODEL_ID
from generate.provenance import PlanProvenance
from generate.trailtraining_compat import run_two_stage_generation_compat



def generate_llm_plan(
    fixture_dir: Path,
    output_dir: Path,
    plan_id: str,
    source_model: str,
    seed: int = 0,
) -> tuple[str, str, str]:
    fixture_dir = Path(fixture_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta_path = fixture_dir / "fixture_meta.json"
    fixture_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    fixture_id = fixture_meta.get("fixture_id", fixture_dir.name)
    plan_path = output_dir / f"{plan_id}.json"

    plan_json, runtime_metadata = run_two_stage_generation_compat(
        fixture_dir=fixture_dir,
        output_path=plan_path,
        source_model=source_model,
        explainer_model=EXPLAINER_MODEL_ID,
        primary_goal="to become a faster trail runner",
        seed=seed,
    )

    runtime_metadata = dict(runtime_metadata or {})
    runtime_metadata.setdefault("TRAILTRAINING_TWO_STAGE_PLAN", "1")

    actual_explainer_model = runtime_metadata.get("actual_explainer_model")
    explainer_verified = bool(runtime_metadata.get("explainer_model_verified", False))

    prov = PlanProvenance(
        plan_id=plan_id,
        fixture_id=fixture_id,
        arm="llm",
        source_model=source_model,
        explainer_model=EXPLAINER_MODEL_ID,
        actual_explainer_model=actual_explainer_model,
        explainer_model_verified=explainer_verified,
        generation_pipeline="llm_two_stage",
        runtime_backend="trailtraining.llm.coach.run_coach_brief",
        runtime_metadata=runtime_metadata,
        seed=seed,
        generated_at=datetime.now(tz=timezone.utc).isoformat(),
        plan_path=str(plan_path),
    )
    prov_path = output_dir / f"{plan_id}.json.provenance.json"
    prov_path.write_text(prov.model_dump_json(indent=2), encoding="utf-8")
    return plan_json, str(plan_path), str(prov_path)
