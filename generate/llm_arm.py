"""generate/llm_arm.py — LLM arm plan generation.

Generates one plan per call via run_coach_brief.  Input files (combined_summary,
rollups, etc.) are read from *fixture_dir*; output files are written to
*output_dir*.  The two directories are kept separate so fixture data is never
mutated.
"""
from __future__ import annotations

import json
import os
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from generate.constants import EXPLAINER_MODEL_ID, PLAN_DAYS
from generate.provenance import PlanProvenance


def generate_llm_plan(
    fixture_dir: Path,
    output_dir: Path,
    plan_id: str,
    source_model: str,
    seed: int = 0,
) -> tuple[str, str, str]:
    """Generate one LLM-arm plan via run_coach_brief.

    Input files are read from *fixture_dir* (combined_summary.json, etc.).
    The plan JSON and provenance sidecar are written to *output_dir*.

    The TRAILTRAINING_TWO_STAGE_PLAN=1 env var activates the two-stage pipeline
    (machine plan → guardrails → explainer) so both arms share the same explainer
    model path.

    Returns (plan_json_str, plan_path, provenance_path).
    """
    from trailtraining.llm.coach import CoachConfig, run_coach_brief
    import trailtraining.config as tt_config

    fixture_dir = Path(fixture_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load fixture metadata for provenance
    meta_path = fixture_dir / "fixture_meta.json"
    fixture_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    fixture_id = fixture_meta.get("fixture_id", fixture_dir.name)

    # Enable two-stage pipeline so the explainer is the same as programmatic arm
    os.environ["TRAILTRAINING_TWO_STAGE_PLAN"] = "1"

    plan_path = output_dir / f"{plan_id}.json"

    cfg = CoachConfig(
        model=source_model,
        reasoning_effort="none",
        temperature=0.7,
        plan_days=PLAN_DAYS,
        primary_goal="to become a faster trail runner",
    )

    # Build a runtime whose prompting_directory points to OUTPUT_DIR
    # so coach can write temp files there, while input files are read
    # from fixture_dir via explicit input_path= / summary_path= overrides.
    fake_paths = types.SimpleNamespace(
        prompting_directory=output_dir,   # writes go here
        processing_directory=output_dir,
        base_dir=output_dir,
        rhr_directory=output_dir,
        sleep_directory=output_dir,
        fit_directory=output_dir,
    )
    fake_runtime = types.SimpleNamespace(paths=fake_paths)

    # Patch ensure_directories to be a no-op (output_dir already exists)
    original_ensure = tt_config.ensure_directories
    tt_config.ensure_directories = lambda runtime=None: None
    try:
        plan_json, saved_path = run_coach_brief(
            prompt="training-plan",
            cfg=cfg,
            # Point coach at the fixture files explicitly
            summary_path=str(fixture_dir / "combined_summary.json"),
            personal_path=str(fixture_dir / "formatted_personal_data.json"),
            input_path=str(fixture_dir),
            output_path=str(plan_path),
            runtime=fake_runtime,
        )
    finally:
        tt_config.ensure_directories = original_ensure

    # Write provenance sidecar
    prov = PlanProvenance(
        plan_id=plan_id,
        fixture_id=fixture_id,
        arm="llm",
        source_model=source_model,
        explainer_model=EXPLAINER_MODEL_ID,
        seed=seed,
        generated_at=datetime.now(tz=timezone.utc).isoformat(),
        plan_path=str(plan_path),
    )
    prov_path = output_dir / f"{plan_id}.json.provenance.json"
    prov_path.write_text(prov.model_dump_json(indent=2), encoding="utf-8")

    return plan_json, str(plan_path), str(prov_path)
