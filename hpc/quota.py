from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from judge.panel import PANEL, get_active_judges

TOTAL_QUOTA_GB = 50.0


def _model_size_lookup() -> dict[str, float]:
    return {spec.model_id: spec.disk_gb for spec in PANEL}


def estimate_generation_peak_gb() -> float:
    sizes = _model_size_lookup()
    source_peak = max(sizes.get("meta-llama/Llama-3.1-8B-Instruct", 15.0), sizes.get("Qwen/Qwen2.5-7B-Instruct", 15.0))
    explainer = sizes.get("Qwen/Qwen2.5-3B-Instruct", 6.0)
    return float(source_peak + explainer)


def estimate_judge_peak_gb() -> float:
    judges = get_active_judges()
    return float(max(j.disk_gb for j in judges)) if judges else 0.0


def study_quota_report(total_quota_gb: float = TOTAL_QUOTA_GB) -> dict[str, Any]:
    generation_peak = estimate_generation_peak_gb()
    judge_peak = estimate_judge_peak_gb()
    return {
        "total_quota_gb": total_quota_gb,
        "generation_peak_gb": generation_peak,
        "judge_peak_gb": judge_peak,
        "generation_fits": generation_peak < total_quota_gb,
        "judge_fits": judge_peak < total_quota_gb,
        "note": "This assumes quota-safe sequencing and cache cleanup between jobs."
    }


def hf_cache_root() -> Path:
    return Path.home() / ".cache" / "huggingface" / "hub"


def purge_hf_model_cache(model_id: str, *, cache_root: Path | None = None) -> bool:
    cache_root = cache_root or hf_cache_root()
    dir_name = f"models--{model_id.replace('/', '--')}"
    path = cache_root / dir_name
    if not path.exists():
        return False
    shutil.rmtree(path, ignore_errors=True)
    return True
