"""Durable-output checks for generation pipelines."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def verify_json_object_file(path: Path, *, label: str) -> dict[str, Any]:
    """Return a parsed JSON object after verifying the file was durably written."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{label} was not written: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{label} path is not a file: {path}")
    if path.stat().st_size <= 0:
        raise ValueError(f"{label} is empty: {path}")
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"{label} is not valid JSON: {path}: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return obj


def verify_plan_and_provenance(
    plan_path: Path,
    provenance_path: Path,
    *,
    expected_plan_id: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify a generated plan and its provenance sidecar are present and coherent."""
    plan_path = Path(plan_path)
    provenance_path = Path(provenance_path)
    plan_obj = verify_json_object_file(plan_path, label="plan artifact")
    provenance_obj = verify_json_object_file(provenance_path, label="provenance sidecar")
    plan = plan_obj.get("plan")
    if not isinstance(plan, dict) or not isinstance(plan.get("days"), list):
        raise ValueError(f"plan artifact missing plan.days list: {plan_path}")
    expected = expected_plan_id or plan_path.stem
    actual = str(provenance_obj.get("plan_id") or "").strip()
    if actual != expected:
        raise ValueError(
            f"provenance plan_id {actual!r} does not match expected {expected!r}: {provenance_path}"
        )
    return plan_obj, provenance_obj
