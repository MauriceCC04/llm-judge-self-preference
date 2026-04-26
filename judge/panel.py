"""judge/panel.py — judge panel definitions."""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel

from generate.constants import ACTIVE_JUDGE_NAMES

Role = Literal["judge", "llm_source", "explainer"]


class JudgeSpec(BaseModel):
    name: str
    model_id: str
    quant: Literal["fp16", "awq_int4"]
    disk_gb: float
    time_hours: float
    role: Role
    max_model_len: int = 8192


PANEL: list[JudgeSpec] = [
    JudgeSpec(name="llama_8b_source", model_id="meta-llama/Llama-3.1-8B-Instruct", quant="fp16", disk_gb=15.0, time_hours=6.0, role="llm_source"),
    JudgeSpec(name="qwen_7b_source", model_id="Qwen/Qwen2.5-7B-Instruct", quant="fp16", disk_gb=15.0, time_hours=6.0, role="llm_source"),
    JudgeSpec(name="qwen_3b_explainer", model_id="Qwen/Qwen2.5-3B-Instruct", quant="fp16", disk_gb=6.0, time_hours=4.0, role="explainer"),
    JudgeSpec(name="llama_8b_judge", model_id="meta-llama/Llama-3.1-8B-Instruct", quant="fp16", disk_gb=15.0, time_hours=6.0, role="judge"),
    JudgeSpec(name="qwen_7b_judge", model_id="Qwen/Qwen2.5-7B-Instruct", quant="fp16", disk_gb=15.0, time_hours=6.0, role="judge"),
    JudgeSpec(name="qwen_14b_judge", model_id="Qwen/Qwen2.5-14B-Instruct-AWQ", quant="awq_int4", disk_gb=8.0, time_hours=10.0, role="judge"),
    JudgeSpec(name="qwen_32b_judge", model_id="Qwen/Qwen2.5-32B-Instruct-AWQ", quant="awq_int4", disk_gb=18.0, time_hours=16.0, role="judge"),
    JudgeSpec(name="mistral_7b_judge", model_id="mistralai/Mistral-7B-Instruct-v0.3", quant="fp16", disk_gb=14.0, time_hours=6.0, role="judge"),
]

PANEL_BY_NAME: dict[str, JudgeSpec] = {j.name: j for j in PANEL}
JUDGES: list[JudgeSpec] = [j for j in PANEL if j.role == "judge"]
ACTIVE_JUDGES: list[JudgeSpec] = [PANEL_BY_NAME[name] for name in ACTIVE_JUDGE_NAMES]


def get_judge(name: str) -> JudgeSpec:
    if name not in PANEL_BY_NAME:
        raise KeyError(f"Unknown judge name: {name!r}. Valid names: {list(PANEL_BY_NAME)}")
    return PANEL_BY_NAME[name]


def get_active_judges(names: list[str] | None = None) -> list[JudgeSpec]:
    selected = names or ACTIVE_JUDGE_NAMES
    return [get_judge(name) for name in selected]
