"""generate/temperature.py — helpers for temperature-aware artifact IDs and labels."""
from __future__ import annotations


def normalize_temperature(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


def format_temperature_tag(value: float | None) -> str:
    if value is None:
        return "na"
    normalized = normalize_temperature(value)
    scaled = int(round(float(normalized) * 100))
    return f"t{scaled:03d}"


def model_tag_from_source_model(source_model: str) -> str:
    return source_model.split("/")[-1].replace("-", "_").lower()


def build_llm_generation_condition(
    *,
    source_temperature: float,
    explainer_temperature: float,
) -> str:
    return f"gen_src_{format_temperature_tag(source_temperature)}__exp_{format_temperature_tag(explainer_temperature)}"


def build_programmatic_generation_condition(
    *,
    explainer_temperature: float,
) -> str:
    return f"gen_prog__exp_{format_temperature_tag(explainer_temperature)}"


def build_llm_plan_id(
    *,
    fixture_id: str,
    source_model: str,
    seed: int,
    source_temperature: float,
    explainer_temperature: float,
) -> str:
    model_tag = model_tag_from_source_model(source_model)
    return (
        f"{fixture_id}__{model_tag}"
        f"__src_{format_temperature_tag(source_temperature)}"
        f"__exp_{format_temperature_tag(explainer_temperature)}"
        f"__s{seed:03d}"
    )


def build_programmatic_plan_id(
    *,
    fixture_id: str,
    seed: int,
    explainer_temperature: float,
) -> str:
    return (
        f"{fixture_id}__prog"
        f"__exp_{format_temperature_tag(explainer_temperature)}"
        f"__s{seed:03d}"
    )