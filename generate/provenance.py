"""generate/provenance.py — PlanProvenance sidecar."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

Arm = Literal["llm", "programmatic"]


class PlanProvenance(BaseModel):
    plan_id: str
    fixture_id: str
    arm: Arm
    source_model: Optional[str] = None
    explainer_model: str
    actual_explainer_model: Optional[str] = None
    explainer_model_verified: bool = False
    generation_pipeline: str = "unknown"
    runtime_backend: Optional[str] = None
    runtime_metadata: dict[str, Any] = Field(default_factory=dict)
    source_temperature: Optional[float] = None
    explainer_temperature: Optional[float] = None
    generation_condition: Optional[str] = None
    seed: int
    generated_at: str
    plan_path: str
    deterministic_score: Optional[float] = None