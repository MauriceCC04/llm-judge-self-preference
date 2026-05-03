"""Leak guards for prompt-only structural scaffolding."""
from __future__ import annotations

from pathlib import Path
from typing import Any


STRUCTURAL_PROMPT_LEAK_MARKERS = (
    "athlete band a1",
    "athlete band a2",
    "athlete band a3",
    "athlete band a4",
    "cell metadata:",
    "session_type='long'",
    "session_type=\"long\"",
    "tempo/intervals/hills",
    "this cell expects",
    "weekly structure intent:",
)


def collect_structural_prompt_leaks(value: Any, path: str = "$") -> list[tuple[str, str, str]]:
    """Return (path, marker, value) hits for prompt-only scaffold text."""
    hits: list[tuple[str, str, str]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            hits.extend(collect_structural_prompt_leaks(child, f"{path}.{key}"))
        return hits
    if isinstance(value, list):
        for idx, child in enumerate(value):
            hits.extend(collect_structural_prompt_leaks(child, f"{path}[{idx}]"))
        return hits
    if isinstance(value, str):
        text = value.strip()
        lowered = text.lower()
        for marker in STRUCTURAL_PROMPT_LEAK_MARKERS:
            if marker in lowered:
                hits.append((path, marker, text))
    return hits


def assert_no_structural_prompt_leaks(
    *,
    plan_id: str,
    output_path: Path,
    plan_obj: dict[str, Any],
) -> None:
    """Raise if prompt-only structural instructions leaked into a saved artifact."""
    hits = collect_structural_prompt_leaks(plan_obj)
    if not hits:
        return
    preview = "; ".join(
        f"{path} contains {marker!r}: {value!r}"
        for path, marker, value in hits[:10]
    )
    raise ValueError(
        f"Structural prompt scaffold leaked into final artifact {plan_id} at {output_path} "
        f"({len(hits)} hit(s)): {preview}"
    )
