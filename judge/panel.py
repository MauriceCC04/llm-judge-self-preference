"""judge/panel.py — judge panel definitions."""
from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel

from generate.constants import ACTIVE_JUDGE_NAMES

Role = Literal["judge", "llm_source", "explainer"]

_HOME_QUOTA_GB = 50.0
_MAX_MODEL_LEN = 8192


class JudgeSpec(BaseModel):
    name: str
    model_id: str
    quant: Literal["fp16", "awq_int4"]
    disk_gb: float
    time_hours: float
    role: Role
    max_model_len: int = _MAX_MODEL_LEN


PANEL: list[JudgeSpec] = [
    JudgeSpec(
        name="qwen_7b_source",
        model_id="Qwen/Qwen2.5-7B-Instruct",
        quant="fp16",
        disk_gb=15.0,
        time_hours=6.0,
        role="llm_source",
    ),
    JudgeSpec(
        name="qwen_3b_explainer",
        model_id="Qwen/Qwen2.5-3B-Instruct",
        quant="fp16",
        disk_gb=6.0,
        time_hours=4.0,
        role="explainer",
    ),
    JudgeSpec(
        name="qwen_7b_judge",
        model_id="Qwen/Qwen2.5-7B-Instruct",
        quant="fp16",
        disk_gb=15.0,
        time_hours=6.0,
        role="judge",
    ),
    JudgeSpec(
        name="qwen_14b_judge",
        model_id="Qwen/Qwen2.5-14B-Instruct-AWQ",
        quant="awq_int4",
        disk_gb=8.0,
        time_hours=10.0,
        role="judge",
    ),
    JudgeSpec(
        name="qwen_32b_judge",
        model_id="Qwen/Qwen2.5-32B-Instruct-AWQ",
        quant="awq_int4",
        disk_gb=18.0,
        time_hours=16.0,
        role="judge",
    ),
    JudgeSpec(
        name="gemma_4b_source",
        model_id="google/gemma-3-4b-it",
        quant="fp16",
        disk_gb=10.0,
        time_hours=6.0,
        role="llm_source",
        max_model_len=16384,
    ),
    JudgeSpec(
        name="gemma_4b_judge",
        model_id="google/gemma-3-4b-it",
        quant="fp16",
        disk_gb=10.0,
        time_hours=6.0,
        role="judge",
        max_model_len=16384,
    ),
    JudgeSpec(
        name="gemma_12b_judge",
        model_id="google/gemma-3-12b-it",
        quant="fp16",
        disk_gb=28.0,
        time_hours=12.0,
        role="judge",
        max_model_len=16384,
    ),
]

PANEL_BY_NAME: dict[str, JudgeSpec] = {judge.name: judge for judge in PANEL}
JUDGES: list[JudgeSpec] = [judge for judge in PANEL if judge.role == "judge"]
ACTIVE_JUDGES: list[JudgeSpec] = [PANEL_BY_NAME[name] for name in ACTIVE_JUDGE_NAMES]


def get_judge(name: str) -> JudgeSpec:
    if name not in PANEL_BY_NAME:
        valid = ", ".join(sorted(PANEL_BY_NAME))
        raise KeyError(f"Unknown judge name: {name!r}. Valid names: {valid}")
    return PANEL_BY_NAME[name]


def get_active_judges(names: list[str] | None = None) -> list[JudgeSpec]:
    selected = names or ACTIVE_JUDGE_NAMES
    return [get_judge(name) for name in selected]


def assert_judge_fits_quota(judge: JudgeSpec) -> None:
    if judge.disk_gb > _HOME_QUOTA_GB:
        raise RuntimeError(
            f"Judge {judge.name} requires {judge.disk_gb} GB on disk, exceeding the {_HOME_QUOTA_GB} GB quota."
        )
    if judge.max_model_len > _MAX_MODEL_LEN:
        raise RuntimeError(
            f"Judge {judge.name} requests max_model_len={judge.max_model_len}, above the supported {_MAX_MODEL_LEN}."
        )


def walltime_hours_with_buffer(judge: JudgeSpec, *, multiplier: float = 1.2) -> int:
    return int(math.ceil(judge.time_hours * multiplier))


def recommend_pair_shards(
    judge: JudgeSpec,
    *,
    n_pairs: int,
    n_runs: int,
    n_positions: int,
    mean_pairwise_seconds: float,
    n_soft_eval_plans: int = 0,
    mean_soft_eval_seconds: float = 0.0,
    target_utilization: float = 0.85,
    walltime_multiplier: float = 1.2,
) -> int:
    available_seconds = walltime_hours_with_buffer(judge, multiplier=walltime_multiplier) * 3600 * target_utilization
    projected_seconds = (
        (n_pairs * n_runs * n_positions * mean_pairwise_seconds)
        + (n_soft_eval_plans * mean_soft_eval_seconds)
    )
    if available_seconds <= 0:
        return 1
    return max(1, int(math.ceil(projected_seconds / available_seconds)))