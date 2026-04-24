"""generate/provenance.py — PlanProvenance sidecar.

Each generated plan (from either arm) gets a companion JSON file:
    <plan_path>.provenance.json

The sidecar is deliberately separate from TrainingMeta so the study can evolve
its metadata without touching the trailtraining contract.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

Arm = Literal["llm", "programmatic"]


class PlanProvenance(BaseModel):
    """Frozen record of how a plan was generated.

    Written alongside every plan JSON as ``<plan_path>.provenance.json``.
    Never modifies TrainingMeta — that is a product contract.
    """

    plan_id: str
    fixture_id: str
    arm: Arm

    # LLM arm only (None for programmatic arm)
    source_model: Optional[str] = None

    # Shared explainer model (identical for both arms — study invariant)
    explainer_model: str

    seed: int
    generated_at: str  # ISO-8601 UTC timestamp

    plan_path: str  # absolute path to the plan JSON

    # Deterministic score filled in after generation
    deterministic_score: Optional[float] = None
