from __future__ import annotations

import importlib
import os
from contextlib import contextmanager
from typing import Any, Iterator

from openai import OpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_DEFAULT_PATCH_STAGE_ENV = "TRAILTRAINING_PATCH_DEFAULT_STAGE"


def _env(name: str) -> str:
    return (os.getenv(name) or "").strip()


def _normalize_stage(stage: str | None) -> str:
    value = (stage or "judge").strip().lower()
    if value not in {"source", "explainer", "judge"}:
        return "judge"
    return value


def _stage_base_url(stage: str) -> str:
    stage = _normalize_stage(stage)
    if stage == "source":
        return _env("TRAILTRAINING_SOURCE_LLM_BASE_URL") or _env("TRAILTRAINING_LLM_BASE_URL")
    if stage == "explainer":
        return _env("TRAILTRAINING_EXPLAINER_LLM_BASE_URL") or _env("TRAILTRAINING_LLM_BASE_URL")
    if stage == "judge":
        return _env("TRAILTRAINING_JUDGE_LLM_BASE_URL") or _env("TRAILTRAINING_LLM_BASE_URL")
    return _env("TRAILTRAINING_LLM_BASE_URL")


def _stage_api_key(stage: str) -> str:
    stage = _normalize_stage(stage)
    stage_key = _env(f"TRAILTRAINING_{stage.upper()}_API_KEY")
    if stage_key:
        return stage_key
    return _env("OPENROUTER_API_KEY") or _env("TRAILTRAINING_OPENROUTER_API_KEY") or ""


def has_local_base_url_config() -> bool:
    return bool(
        _env("TRAILTRAINING_LLM_BASE_URL")
        or _env("TRAILTRAINING_SOURCE_LLM_BASE_URL")
        or _env("TRAILTRAINING_EXPLAINER_LLM_BASE_URL")
        or _env("TRAILTRAINING_JUDGE_LLM_BASE_URL")
    )


def _make_openrouter_client_fallback(stage: str = "judge") -> Any:
    """Defer to trailtraining's current client factory when possible."""
    for module_name, attr in [
        ("trailtraining.llm.shared", "make_openrouter_client"),
        ("trailtraining.llm.soft_eval", "make_openrouter_client"),
        ("trailtraining.llm.coach", "_make_openrouter_client"),
    ]:
        try:
            module = importlib.import_module(module_name)
            factory = getattr(module, attr)
            return factory()
        except Exception:
            continue

    api_key = _stage_api_key(stage)
    if not api_key:
        raise RuntimeError(
            "No compatible LLM endpoint configured. Set a stage-specific or generic "
            "TRAILTRAINING_*_LLM_BASE_URL for local vLLM, or provide OPENROUTER_API_KEY / "
            "TRAILTRAINING_OPENROUTER_API_KEY."
        )
    headers: dict[str, str] = {}
    site_url = _env("TRAILTRAINING_OPENROUTER_SITE_URL")
    if site_url:
        headers["HTTP-Referer"] = site_url
    app_name = _env("TRAILTRAINING_OPENROUTER_APP_NAME") or "trailtraining"
    headers["X-OpenRouter-Title"] = app_name
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key, default_headers=headers)


def make_stage_client(*, stage: str, model_id: str | None = None) -> Any:
    del model_id
    stage = _normalize_stage(stage)
    base_url = _stage_base_url(stage)
    api_key = _stage_api_key(stage) or "dummy"
    if base_url:
        return OpenAI(base_url=base_url.rstrip("/"), api_key=api_key)
    return _make_openrouter_client_fallback(stage=stage)


def ensure_dual_endpoint_support(*, source_model: str, explainer_model: str) -> None:
    generic = _env("TRAILTRAINING_LLM_BASE_URL")
    src = _env("TRAILTRAINING_SOURCE_LLM_BASE_URL")
    exp = _env("TRAILTRAINING_EXPLAINER_LLM_BASE_URL")
    if generic and not src and not exp and source_model != explainer_model:
        raise RuntimeError(
            "LLM-arm generation requires different source and explainer models, but only a single "
            "TRAILTRAINING_LLM_BASE_URL is configured. For local vLLM, set both "
            "TRAILTRAINING_SOURCE_LLM_BASE_URL and TRAILTRAINING_EXPLAINER_LLM_BASE_URL, or use a "
            "single remote provider that can serve both models."
        )


@contextmanager
def temporary_patch_stage(stage: str) -> Iterator[None]:
    stage = _normalize_stage(stage)
    old_value = os.getenv(_DEFAULT_PATCH_STAGE_ENV)
    os.environ[_DEFAULT_PATCH_STAGE_ENV] = stage
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop(_DEFAULT_PATCH_STAGE_ENV, None)
        else:
            os.environ[_DEFAULT_PATCH_STAGE_ENV] = old_value


def install_trailtraining_client_compat(*, default_stage: str = "judge") -> None:
    """Patch trailtraining client factories only when local base URLs are configured."""
    if not has_local_base_url_config():
        return

    default_stage = _normalize_stage(default_stage)

    def _patched_factory() -> Any:
        stage = _normalize_stage(_env(_DEFAULT_PATCH_STAGE_ENV) or default_stage)
        return make_stage_client(stage=stage)

    module_specs = [
        ("trailtraining.llm.shared", "make_openrouter_client"),
        ("trailtraining.llm.soft_eval", "make_openrouter_client"),
        ("trailtraining.llm.coach", "_make_openrouter_client"),
        ("trailtraining.llm.revise", "_make_openrouter_client"),
    ]
    for module_name, attr in module_specs:
        try:
            module = importlib.import_module(module_name)
            setattr(module, attr, _patched_factory)
        except Exception:
            continue


def describe_client_routing() -> dict[str, Any]:
    return {
        "source_base_url": _stage_base_url("source") or None,
        "explainer_base_url": _stage_base_url("explainer") or None,
        "judge_base_url": _stage_base_url("judge") or None,
        "using_local_base_url": has_local_base_url_config(),
        "default_patch_stage": _normalize_stage(_env(_DEFAULT_PATCH_STAGE_ENV) or "judge"),
    }
