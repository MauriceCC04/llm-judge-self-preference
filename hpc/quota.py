from __future__ import annotations

"""hpc/quota.py — quota and Hugging Face cache helpers for Bocconi HPC.

This module is intentionally import-light so it can be used from preflight
helpers before the full runtime stack is exercised.

Operational assumptions:
- cluster hard quota is 50 GB
- we keep a softer operational ceiling below that
- generation and judging are run sequentially
- only the models required for the current job should be cached
- model caches should be purged before new pre-cache steps and cleaned up after jobs
"""

import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from judge.panel import PANEL, get_active_judges

TOTAL_QUOTA_GB = 50.0
DEFAULT_SOFT_QUOTA_GB = 40.0
DEFAULT_HF_CACHE_ROOT = Path.home() / "hf_cache" / "hub"


@dataclass(frozen=True)
class QuotaReport:
    total_quota_gb: float
    soft_quota_gb: float
    generation_peak_gb: float
    judge_peak_gb: float
    generation_fits_hard_quota: bool
    judge_fits_hard_quota: bool
    generation_fits_soft_quota: bool
    judge_fits_soft_quota: bool
    note: str
    hf_cache_root: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _model_size_lookup() -> dict[str, float]:
    return {spec.model_id: float(spec.disk_gb) for spec in PANEL}


def _max_disk_gb(model_ids: Iterable[str], fallback_gb: float) -> float:
    sizes = _model_size_lookup()
    return max((sizes.get(model_id, fallback_gb) for model_id in model_ids), default=fallback_gb)


def estimate_generation_peak_gb() -> float:
    """Estimate generation-time model-cache pressure.

    The frozen-study generation path runs one explainer model alongside one
    source model at a time, so we budget for the largest source plus the
    explainer.
    """
    source_peak = _max_disk_gb(
        [
            "Qwen/Qwen2.5-7B-Instruct",
            "google/gemma-3-4b-it",
        ],
        fallback_gb=17.0,
    )
    explainer = _model_size_lookup().get("Qwen/Qwen2.5-3B-Instruct", 6.0)
    return float(source_peak + explainer)


def estimate_judge_peak_gb() -> float:
    """Estimate judge-time model-cache pressure for the active panel."""
    judges = get_active_judges()
    return float(max((float(j.disk_gb) for j in judges), default=0.0))


def resolve_soft_quota_gb(
    soft_quota_gb: float | None = None,
    *,
    env: dict[str, str] | None = None,
) -> float:
    env_map = env or os.environ
    if soft_quota_gb is not None:
        return float(soft_quota_gb)
    env_value = env_map.get("SOFT_QUOTA_GB")
    if env_value:
        try:
            return float(env_value)
        except ValueError:
            pass
    return DEFAULT_SOFT_QUOTA_GB


def hf_cache_root(*, env: dict[str, str] | None = None) -> Path:
    """Resolve the active Hugging Face hub cache.

    Priority:
    1. HF_HUB_CACHE
    2. HUGGINGFACE_HUB_CACHE
    3. repo default under ~/hf_cache/hub
    4. legacy ~/.cache/huggingface/hub if it already exists
    """
    env_map = env or os.environ
    for key in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        value = env_map.get(key)
        if value:
            return Path(value).expanduser()

    legacy = Path.home() / ".cache" / "huggingface" / "hub"
    if legacy.exists() and not DEFAULT_HF_CACHE_ROOT.exists():
        return legacy
    return DEFAULT_HF_CACHE_ROOT


def cached_model_dirs(*, cache_root: Path | None = None, env: dict[str, str] | None = None) -> list[Path]:
    """List cached Hugging Face model directories under the active cache root."""
    root = (cache_root or hf_cache_root(env=env)).expanduser()
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and path.name.startswith("models--")
    )


def study_quota_report(
    *,
    total_quota_gb: float = TOTAL_QUOTA_GB,
    soft_quota_gb: float | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    generation_peak = estimate_generation_peak_gb()
    judge_peak = estimate_judge_peak_gb()
    soft = resolve_soft_quota_gb(soft_quota_gb, env=env)

    report = QuotaReport(
        total_quota_gb=float(total_quota_gb),
        soft_quota_gb=float(soft),
        generation_peak_gb=generation_peak,
        judge_peak_gb=judge_peak,
        generation_fits_hard_quota=generation_peak < total_quota_gb,
        judge_fits_hard_quota=judge_peak < total_quota_gb,
        generation_fits_soft_quota=generation_peak < soft,
        judge_fits_soft_quota=judge_peak < soft,
        note=(
            "Assumes quota-safe sequencing with only the current job's model cache present, "
            "plus explicit purge-before-precache and post-job cleanup."
        ),
        hf_cache_root=str(hf_cache_root(env=env)),
    )
    return report.to_dict()


def purge_hf_model_cache(
    model_id: str,
    *,
    cache_root: Path | None = None,
    env: dict[str, str] | None = None,
) -> bool:
    root = (cache_root or hf_cache_root(env=env)).expanduser()
    dir_name = f"models--{model_id.replace('/', '--')}"
    path = root / dir_name
    if not path.exists():
        return False
    shutil.rmtree(path, ignore_errors=True)
    return not path.exists()


def purge_all_hf_model_caches(
    *,
    cache_root: Path | None = None,
    env: dict[str, str] | None = None,
) -> int:
    """Remove all cached Hugging Face model directories and return how many were removed."""
    removed = 0
    for path in cached_model_dirs(cache_root=cache_root, env=env):
        shutil.rmtree(path, ignore_errors=True)
        if not path.exists():
            removed += 1
    return removed