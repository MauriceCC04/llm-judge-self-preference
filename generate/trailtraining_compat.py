from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import os

import trailtraining.llm.coach_prompting as _coach_prompting
from compat.trailtraining_client import (
    describe_client_routing,
    ensure_dual_endpoint_support,
    make_stage_client,
)
from generate.constants import PLAN_DAYS
from generate.structural_expectations import (
    build_structural_lifestyle_notes,
    detect_understructured_plan,
)
from match.filtering import detect_plan_issues

log = logging.getLogger(__name__)

SOURCE_MAX_TOKENS = int(os.getenv("TRAILTRAINING_SOURCE_MAX_TOKENS", "4096"))
EXPLAINER_MAX_TOKENS = int(os.getenv("TRAILTRAINING_EXPLAINER_MAX_TOKENS", "12288"))


class StructuredStageError(RuntimeError):
    def __init__(
        self,
        *,
        plan_id: str,
        stage: str,
        cause: Exception,
        raw_text_path: Path | None = None,
        request_json_path: Path | None = None,
        prompt_text_path: Path | None = None,
        failure_json_path: Path | None = None,
    ) -> None:
        super().__init__(f"[{stage}] {cause}")
        self.plan_id = plan_id
        self.stage = stage
        self.raw_text_path = str(raw_text_path) if raw_text_path else None
        self.request_json_path = str(request_json_path) if request_json_path else None
        self.prompt_text_path = str(prompt_text_path) if prompt_text_path else None
        self.failure_json_path = str(failure_json_path) if failure_json_path else None


_PLACEHOLDER_LEAK_MARKERS = (
    ">{signal_id",
    "{signal_id",
)


def _write_stage_failure_artifacts(
    *,
    output_path: Path,
    stage: str,
    prompt_text: str,
    request_kwargs: dict[str, Any],
    raw_text: str,
    exc: Exception,
) -> dict[str, Path]:
    root = output_path.parent / "raw_failures"
    root.mkdir(parents=True, exist_ok=True)

    stem = output_path.stem
    prompt_path = root / f"{stem}.{stage}.prompt.txt"
    request_path = root / f"{stem}.{stage}.request.json"
    raw_path = root / f"{stem}.{stage}.raw.txt"
    failure_path = root / f"{stem}.{stage}.failure.json"

    prompt_path.write_text(prompt_text or "", encoding="utf-8")
    request_path.write_text(
        json.dumps(request_kwargs, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    raw_path.write_text(raw_text or "", encoding="utf-8")
    failure_path.write_text(
        json.dumps(
            {
                "plan_id": stem,
                "stage": stage,
                "error": str(exc),
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                "raw_text_chars": len(raw_text or ""),
                "prompt_text_path": str(prompt_path),
                "request_json_path": str(request_path),
                "raw_text_path": str(raw_path),
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    return {
        "prompt_text_path": prompt_path,
        "request_json_path": request_path,
        "raw_text_path": raw_path,
        "failure_json_path": failure_path,
    }


def _parse_with_failure_dump(
    *,
    plan_id: str,
    output_path: Path,
    stage: str,
    prompt_text: str,
    request_kwargs: dict[str, Any],
    raw_text: str,
    parse_fn,
) -> dict[str, Any]:
    try:
        return parse_fn(raw_text)
    except Exception as exc:
        paths = _write_stage_failure_artifacts(
            output_path=output_path,
            stage=stage,
            prompt_text=prompt_text,
            request_kwargs=request_kwargs,
            raw_text=raw_text,
            exc=exc,
        )
        raise StructuredStageError(
            plan_id=plan_id,
            stage=stage,
            cause=exc,
            raw_text_path=paths["raw_text_path"],
            request_json_path=paths["request_json_path"],
            prompt_text_path=paths["prompt_text_path"],
            failure_json_path=paths["failure_json_path"],
        ) from exc


def _collect_placeholder_leaks(value: Any, path: str = "$") -> list[tuple[str, str]]:
    hits: list[tuple[str, str]] = []

    if isinstance(value, dict):
        for key, child in value.items():
            hits.extend(_collect_placeholder_leaks(child, f"{path}.{key}"))
        return hits

    if isinstance(value, list):
        for idx, child in enumerate(value):
            hits.extend(_collect_placeholder_leaks(child, f"{path}[{idx}]"))
        return hits

    if isinstance(value, str):
        text = value.strip()
        if any(marker in text for marker in _PLACEHOLDER_LEAK_MARKERS):
            hits.append((path, text))
    return hits


def _assert_no_placeholder_leaks(
    *,
    plan_id: str,
    output_path: Path,
    plan_obj: dict[str, Any],
) -> None:
    hits = _collect_placeholder_leaks(plan_obj)
    if not hits:
        return

    raw_text = json.dumps(plan_obj, indent=2, ensure_ascii=False, default=str)
    preview = "; ".join(f"{path}={value!r}" for path, value in hits[:10])
    exc = ValueError(
        f"Placeholder leak detected in final artifact ({len(hits)} hit(s)): {preview}"
    )

    paths = _write_stage_failure_artifacts(
        output_path=output_path,
        stage="final_validation",
        prompt_text="",
        request_kwargs={
            "validation": "placeholder_leak",
            "markers": list(_PLACEHOLDER_LEAK_MARKERS),
            "hits": [{"path": path, "value": value} for path, value in hits],
        },
        raw_text=raw_text,
        exc=exc,
    )

    raise StructuredStageError(
        plan_id=plan_id,
        stage="final_validation",
        cause=exc,
        raw_text_path=paths["raw_text_path"],
        request_json_path=paths["request_json_path"],
        prompt_text_path=paths["prompt_text_path"],
        failure_json_path=paths["failure_json_path"],
    ) from exc


def _load_fixture(fixture_dir: Path) -> dict[str, Any]:
    combined = json.loads((fixture_dir / "combined_summary.json").read_text(encoding="utf-8"))
    personal = json.loads((fixture_dir / "formatted_personal_data.json").read_text(encoding="utf-8"))
    rollups_path = fixture_dir / "combined_rollups.json"
    forecast_path = fixture_dir / "readiness_and_risk_forecast.json"
    meta_path = fixture_dir / "fixture_meta.json"
    return {
        "combined": combined,
        "personal": personal,
        "rollups": json.loads(rollups_path.read_text(encoding="utf-8")) if rollups_path.exists() else None,
        "forecast": json.loads(forecast_path.read_text(encoding="utf-8")) if forecast_path.exists() else None,
        "fixture_meta": json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {},
    }


def _extract_response_model_id(response: Any, fallback: str | None = None) -> str | None:
    candidates = [
        getattr(response, "model", None),
        getattr(getattr(response, "response", None), "model", None),
        (response.get("model") if isinstance(response, dict) else None),
    ]
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return fallback


def _build_deterministic_snapshot(combined: list[dict[str, Any]]) -> dict[str, Any]:
    days = [day for day in combined if isinstance(day, dict)]
    last7 = days[-7:]
    last28 = days[-28:]

    def _activity_items(window: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for day in window:
            activities = day.get("activities") or []
            if isinstance(activities, list):
                out.extend([a for a in activities if isinstance(a, dict)])
        return out

    def _sleep_items(window: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for day in window:
            sleep = day.get("sleep")
            if isinstance(sleep, dict):
                out.append(sleep)
        return out

    def _fmt_num(value: float | int | None, digits: int = 1) -> str:
        if value is None:
            return ""
        if isinstance(value, int):
            return str(value)
        text = f"{value:.{digits}f}"
        return text.rstrip("0").rstrip(".") if "." in text else text

    def _sum_field(items: list[dict[str, Any]], key: str, scale: float = 1.0) -> float | None:
        vals = [item.get(key) for item in items if isinstance(item.get(key), (int, float))]
        if not vals:
            return None
        return sum(float(v) for v in vals) / scale

    def _mean_field(items: list[dict[str, Any]], keys: tuple[str, ...]) -> float | None:
        vals: list[float] = []
        for item in items:
            for key in keys:
                val = item.get(key)
                if isinstance(val, (int, float)):
                    vals.append(float(val))
                    break
        if not vals:
            return None
        return sum(vals) / len(vals)

    def _snapshot_obj(window: list[dict[str, Any]]) -> dict[str, str]:
        activities = _activity_items(window)
        sleeps = _sleep_items(window)

        sleep_seconds = _mean_field(sleeps, ("duration", "total_sleep"))
        sleep_hours = sleep_seconds / 3600.0 if sleep_seconds is not None else None

        return {
            "distance_km": _fmt_num(_sum_field(activities, "distance", scale=1000.0)),
            "moving_time_hours": _fmt_num(_sum_field(activities, "moving_time", scale=3600.0)),
            "elevation_m": _fmt_num(_sum_field(activities, "total_elevation_gain")),
            "activity_count": str(len(activities)) if activities else "",
            "sleep_hours_mean": _fmt_num(sleep_hours),
            "hrv_mean": _fmt_num(_mean_field(sleeps, ("hrv", "average_hrv", "hrv_mean"))),
            "rhr_mean": _fmt_num(
                _mean_field(sleeps, ("resting_hr", "rhr", "resting_heart_rate"))
            ),
        }

    last7_obj = _snapshot_obj(last7)
    baseline28_obj = _snapshot_obj(last28)

    notes_parts: list[str] = []
    if last7_obj["distance_km"]:
        notes_parts.append(f"The athlete covered {last7_obj['distance_km']} km in the last 7 days.")
    if last7_obj["moving_time_hours"]:
        notes_parts.append(f"Moving time was {last7_obj['moving_time_hours']} hours.")
    if last7_obj["elevation_m"]:
        notes_parts.append(f"Elevation gain was {last7_obj['elevation_m']} meters.")
    if last7_obj["sleep_hours_mean"]:
        notes_parts.append(f"Average sleep was {last7_obj['sleep_hours_mean']} hours per night.")

    return {
        "last7": last7_obj,
        "baseline28": baseline28_obj,
        "notes": " ".join(notes_parts).strip(),
    }


def _canonical_title_for_session(session_type: str) -> str:
    mapping = {
        "rest": "Rest day",
        "easy": "Easy run",
        "aerobic": "Aerobic run",
        "long": "Long run",
        "strength": "Strength session",
        "tempo": "Tempo session",
        "intervals": "Intervals",
        "hills": "Hill session",
        "cross": "Cross-training",
    }
    key = (session_type or "").strip().lower()
    return mapping.get(key, key.replace("_", " ").title() or "Training session")


def _generic_workout_for_session(session_type: str, duration_minutes: int) -> str:
    st = (session_type or "").strip().lower()
    if st == "rest":
        return "Rest day. No structured training."
    dur = max(0, int(duration_minutes or 0))
    if st == "easy":
        return f"{dur} min easy run on flat to rolling terrain."
    if st == "aerobic":
        return f"{dur} min steady aerobic run on mixed terrain."
    if st == "long":
        return f"{dur} min long run at controlled aerobic effort."
    if st == "strength":
        return f"{dur} min strength and mobility session."
    if st == "tempo":
        return f"{dur} min tempo-focused run at controlled hard effort."
    if st == "intervals":
        return f"{dur} min interval session with controlled recoveries."
    if st == "hills":
        return f"{dur} min hill-focused session with steady climbing efforts."
    if st == "cross":
        return f"{dur} min low-impact cross-training session."
    return f"{dur} min {st or 'training'} session."


def _normalize_rest_day_semantics(plan_obj: dict[str, Any]) -> dict[str, Any]:
    plan = plan_obj.get("plan") or {}
    days = plan.get("days") or []
    for day in days:
        if not isinstance(day, dict):
            continue
        session_type = str(day.get("session_type") or "").strip().lower()
        is_rest = bool(day.get("is_rest_day")) or session_type == "rest"
        duration = int(day.get("duration_minutes") or 0)
        title = str(day.get("title") or "")
        workout = str(day.get("workout") or "")
        purpose = str(day.get("purpose") or "")
        joined = " ".join([title, workout, purpose]).lower()

        if is_rest:
            day["session_type"] = "rest"
            day["is_rest_day"] = True
            day["is_hard_day"] = False
            day["duration_minutes"] = 0
            day["target_intensity"] = day.get("target_intensity") or "rest"
            day["title"] = "Rest day"
            day["workout"] = "Rest day. No structured training."
            if "purpose" not in day or not str(day.get("purpose") or "").strip() or "rest day" in purpose.lower():
                day["purpose"] = "Absorb training and maintain freshness."
            continue

        if "rest day" in joined:
            day["title"] = _canonical_title_for_session(session_type)
            if "rest day" in workout.lower() or not workout.strip():
                day["workout"] = _generic_workout_for_session(session_type, duration)
            if "rest day" in purpose.lower():
                day["purpose"] = f"Complete the planned {_canonical_title_for_session(session_type).lower()} without adding extra fatigue."
    return plan_obj


def _assert_no_human_contradictions(*, plan_id: str, output_path: Path, plan_obj: dict[str, Any]) -> None:
    issues = detect_plan_issues(plan_obj)
    issues = [issue for issue in issues if issue.startswith("human_contradiction:")]
    if not issues:
        return

    raw_text = json.dumps(plan_obj, indent=2, ensure_ascii=False, default=str)
    exc = ValueError(f"Human-facing contradictions remain after normalization: {issues}")
    paths = _write_stage_failure_artifacts(
        output_path=output_path,
        stage="final_validation",
        prompt_text="",
        request_kwargs={
            "validation": "human_contradiction",
            "issues": issues,
        },
        raw_text=raw_text,
        exc=exc,
    )
    raise StructuredStageError(
        plan_id=plan_id,
        stage="final_validation",
        cause=exc,
        raw_text_path=paths["raw_text_path"],
        request_json_path=paths["request_json_path"],
        prompt_text_path=paths["prompt_text_path"],
        failure_json_path=paths["failure_json_path"],
    ) from exc


def _assert_not_understructured(
    *,
    plan_id: str,
    output_path: Path,
    plan_obj: dict[str, Any],
    fixture_meta: dict[str, Any],
) -> None:
    issues = detect_understructured_plan(plan_obj, fixture_meta)
    if not issues:
        return

    raw_text = json.dumps(plan_obj, indent=2, ensure_ascii=False, default=str)
    exc = ValueError(f"Plan is understructured for fixture cell: {issues}")
    paths = _write_stage_failure_artifacts(
        output_path=output_path,
        stage="final_validation",
        prompt_text="",
        request_kwargs={
            "validation": "understructured_plan",
            "issues": issues,
            "fixture_meta": fixture_meta,
        },
        raw_text=raw_text,
        exc=exc,
    )
    raise StructuredStageError(
        plan_id=plan_id,
        stage="final_validation",
        cause=exc,
        raw_text_path=paths["raw_text_path"],
        request_json_path=paths["request_json_path"],
        prompt_text_path=paths["prompt_text_path"],
        failure_json_path=paths["failure_json_path"],
    ) from exc


def run_two_stage_generation_compat(
    *,
    fixture_dir: Path,
    output_path: Path,
    source_model: str,
    explainer_model: str,
    primary_goal: str,
    seed: int,
    source_temperature: float = 0.7,
    explainer_temperature: float = 0.0,
) -> tuple[str, dict[str, Any]]:
    from trailtraining.llm.coach import (
        CoachConfig,
        _apply_eval_coach_guardrails_compat,
        _finalize_training_plan_artifact,
        _merge_machine_plan_and_explanations,
        _parse_machine_plan,
        _parse_plan_explanation,
    )
    from trailtraining.llm.constraints import constraint_config_from_env, derive_effective_constraints
    from trailtraining.llm.presets import get_system_prompt
    from trailtraining.llm.rubrics import default_primary_goal_for_style
    from trailtraining.llm.schemas import MACHINE_PLAN_SCHEMA, PLAN_EXPLANATION_STAGE_SCHEMA
    from trailtraining.llm.shared import call_with_schema, recompute_planned_hours
    from trailtraining.util.state import _json_default, save_json

    fixture_dir = Path(fixture_dir)
    output_path = Path(output_path)
    data = _load_fixture(fixture_dir)

    ensure_dual_endpoint_support(source_model=source_model, explainer_model=explainer_model)

    fixture_meta = data["fixture_meta"] if isinstance(data.get("fixture_meta"), dict) else {}
    style = str(fixture_meta.get("style") or "trailrunning").strip() or "trailrunning"
    resolved_goal = (
        (primary_goal or "").strip()
        or str(fixture_meta.get("primary_goal") or "").strip()
        or default_primary_goal_for_style(style)
    )
    prompt_lifestyle_notes = build_structural_lifestyle_notes(fixture_meta)
    detail_days = max(1, min(14, len(data["combined"])))

    effective = derive_effective_constraints(
        det_forecast=data["forecast"],
        rollups=data["rollups"],
        cfg=constraint_config_from_env(),
        lifestyle_notes=prompt_lifestyle_notes,
    )

    source_cfg = CoachConfig(
        model=source_model,
        reasoning_effort="none",
        temperature=source_temperature,
        plan_days=PLAN_DAYS,
        primary_goal=resolved_goal,
        style=style,
    )
    explainer_cfg = CoachConfig(
        model=explainer_model,
        reasoning_effort="none",
        temperature=explainer_temperature,
        plan_days=PLAN_DAYS,
        primary_goal=resolved_goal,
        style=style,
    )

    source_client = make_stage_client(stage="source", model_id=source_model)
    explainer_client = make_stage_client(stage="explainer", model_id=explainer_model)

    machine_prompt = _coach_prompting.build_machine_plan_prompt_text(
        personal=data["personal"],
        rollups=data["rollups"],
        combined=data["combined"],
        deterministic_forecast=data["forecast"],
        style=style,
        primary_goal=resolved_goal,
        lifestyle_notes="",
        max_chars=source_cfg.max_chars,
        detail_days=detail_days,
        plan_days=source_cfg.plan_days,
        effective_constraints=effective,
    )
    plan_id = output_path.stem

    machine_kwargs: dict[str, Any] = {
        "model": source_model,
        "instructions": get_system_prompt(style),
        "input": machine_prompt,
        "reasoning": {"effort": source_cfg.reasoning_effort},
        "text": {"verbosity": source_cfg.verbosity},
        "stage_name": "machine_plan",
        "max_tokens": SOURCE_MAX_TOKENS,
    }
    if source_cfg.temperature is not None:
        machine_kwargs["temperature"] = source_cfg.temperature

    machine_resp = call_with_schema(source_client, machine_kwargs, MACHINE_PLAN_SCHEMA)
    actual_source_model = _extract_response_model_id(machine_resp, fallback=source_model)
    machine_text = getattr(machine_resp, "output_text", None) or str(machine_resp)

    machine_obj = _parse_with_failure_dump(
        plan_id=plan_id,
        output_path=output_path,
        stage="machine_plan",
        prompt_text=machine_prompt,
        request_kwargs=machine_kwargs,
        raw_text=machine_text,
        parse_fn=lambda text: _parse_machine_plan(
            text,
            source_client,
            source_cfg,
            str(machine_kwargs.get("instructions") or ""),
        ),
    )

    guarded_stub: dict[str, Any] = {
        "meta": dict((machine_obj.get("meta") or {})),
        "snapshot": _build_deterministic_snapshot(data["combined"]),
        "readiness": {
            "status": str(((machine_obj.get("readiness") or {}).get("status") or "steady")),
            "rationale": "",
            "signal_ids": [],
        },
        "plan": {
            "weekly_totals": dict(((machine_obj.get("plan") or {}).get("weekly_totals") or {})),
            "days": [
                {
                    "date": d.get("date"),
                    "title": "",
                    "session_type": d.get("session_type"),
                    "is_rest_day": d.get("is_rest_day"),
                    "is_hard_day": d.get("is_hard_day"),
                    "duration_minutes": d.get("duration_minutes"),
                    "target_intensity": d.get("target_intensity"),
                    "terrain": d.get("terrain"),
                    "workout": d.get("workout"),
                    "purpose": "",
                    "signal_ids": [],
                }
                for d in ((machine_obj.get("plan") or {}).get("days") or [])
                if isinstance(d, dict)
            ],
        },
        "recovery": {"actions": [], "signal_ids": []},
        "risks": [],
        "data_notes": [],
        "citations": [],
        "claim_attributions": [],
        "effective_constraints": None,
    }
    _apply_eval_coach_guardrails_compat(guarded_stub, data["rollups"], effective)
    machine_obj["plan"] = guarded_stub["plan"]

    explainer_prompt = _coach_prompting.build_explainer_prompt_text(
        machine_plan=machine_obj,
        personal=data["personal"],
        rollups=data["rollups"],
        combined=data["combined"],
        deterministic_forecast=data["forecast"],
        style=style,
        primary_goal=resolved_goal,
        lifestyle_notes="",
        max_chars=explainer_cfg.max_chars,
        detail_days=detail_days,
        effective_constraints=effective,
    )
    explain_kwargs: dict[str, Any] = {
        "model": explainer_model,
        "instructions": get_system_prompt(style),
        "input": explainer_prompt,
        "reasoning": {"effort": explainer_cfg.reasoning_effort},
        "text": {"verbosity": "low"},
        "stage_name": "plan_explanation",
        "max_tokens": EXPLAINER_MAX_TOKENS,
    }
    if explainer_cfg.temperature is not None:
        explain_kwargs["temperature"] = explainer_cfg.temperature

    explain_resp = call_with_schema(explainer_client, explain_kwargs, PLAN_EXPLANATION_STAGE_SCHEMA)
    actual_explainer_model = _extract_response_model_id(explain_resp, fallback=None)
    explainer_model_verified = bool(
        actual_explainer_model is not None and actual_explainer_model == explainer_model
    )
    explain_text = getattr(explain_resp, "output_text", None) or str(explain_resp)

    explanation_obj = _parse_with_failure_dump(
        plan_id=plan_id,
        output_path=output_path,
        stage="plan_explanation",
        prompt_text=explainer_prompt,
        request_kwargs=explain_kwargs,
        raw_text=explain_text,
        parse_fn=lambda text: _parse_plan_explanation(
            text,
            explainer_client,
            explainer_cfg,
            str(explain_kwargs.get("instructions") or ""),
        ),
    )
    explanation_obj["snapshot"] = _build_deterministic_snapshot(data["combined"])

    obj = _merge_machine_plan_and_explanations(
        machine_obj,
        explanation_obj,
        resolved_goal=resolved_goal,
        lifestyle_notes="",
        deterministic_forecast=data["forecast"],
        effective=effective,
    )
    recompute_planned_hours(obj)
    _apply_eval_coach_guardrails_compat(obj, data["rollups"], effective)
    obj = _finalize_training_plan_artifact(
        obj,
        combined=data["combined"],
        rollups=data["rollups"],
        deterministic_forecast=data["forecast"],
        effective=effective,
    )
    obj = _normalize_rest_day_semantics(obj)

    _assert_no_placeholder_leaks(
        plan_id=plan_id,
        output_path=output_path,
        plan_obj=obj,
    )
    _assert_no_human_contradictions(
        plan_id=plan_id,
        output_path=output_path,
        plan_obj=obj,
    )
    _assert_not_understructured(
        plan_id=plan_id,
        output_path=output_path,
        plan_obj=obj,
        fixture_meta=fixture_meta,
    )

    save_json(output_path, obj, compact=False)
    runtime_metadata = {
        **describe_client_routing(),
        "source_model": source_model,
        "actual_source_model": actual_source_model,
        "actual_explainer_model": actual_explainer_model,
        "explainer_model_verified": explainer_model_verified,
        "source_temperature": source_temperature,
        "explainer_temperature": explainer_temperature,
        "seed": seed,
    }
    return json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default), runtime_metadata
