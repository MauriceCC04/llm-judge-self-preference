from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PLACEHOLDER_MARKERS = (
    ">{signal_id",
    "{signal_id",
)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


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
        if any(marker in text for marker in PLACEHOLDER_MARKERS):
            hits.append((path, text))
    return hits


def _artifact_to_machine_plan(artifact: dict[str, Any]) -> dict[str, Any]:
    meta = dict(_as_dict(artifact.get("meta")))
    readiness = _as_dict(artifact.get("readiness"))
    plan = _as_dict(artifact.get("plan"))
    weekly_totals = dict(_as_dict(plan.get("weekly_totals")))

    days: list[dict[str, Any]] = []
    for raw_day in plan.get("days", []):
        day = _as_dict(raw_day)
        days.append(
            {
                "date": day.get("date"),
                "session_type": day.get("session_type"),
                "duration_minutes": day.get("duration_minutes"),
                "target_intensity": day.get("target_intensity"),
                "terrain": day.get("terrain"),
                "workout": day.get("workout"),
                "estimated_distance_km": day.get("estimated_distance_km"),
                "estimated_elevation_m": day.get("estimated_elevation_m"),
            }
        )

    return {
        "meta": meta,
        "readiness": {"status": readiness.get("status")},
        "plan": {
            "weekly_totals": weekly_totals,
            "days": days,
        },
    }


def _load_fixture_bundle(fixture_dir: Path) -> dict[str, Any]:
    return {
        "combined": _load_json(fixture_dir / "combined_summary.json"),
        "personal": _load_json(fixture_dir / "formatted_personal_data.json"),
        "rollups": _load_json(fixture_dir / "combined_rollups.json") if (fixture_dir / "combined_rollups.json").exists() else None,
        "forecast": _load_json(fixture_dir / "readiness_and_risk_forecast.json") if (fixture_dir / "readiness_and_risk_forecast.json").exists() else None,
        "fixture_meta": _load_json(fixture_dir / "fixture_meta.json") if (fixture_dir / "fixture_meta.json").exists() else {},
    }


def _resolve_machine_plan(saved_plan: Path | None, machine_plan: Path | None) -> tuple[dict[str, Any], str]:
    if machine_plan is not None:
        return _load_json(machine_plan), str(machine_plan)
    if saved_plan is not None:
        artifact = _load_json(saved_plan)
        return _artifact_to_machine_plan(artifact), str(saved_plan)
    raise SystemExit("Provide either --saved-plan-json or --machine-plan-json.")


def build_replay_artifacts(
    *,
    fixture_dir: Path,
    out_dir: Path,
    saved_plan_json: Path | None,
    machine_plan_json: Path | None,
    max_chars: int,
    detail_days: int | None,
) -> int:
    from trailtraining.llm.coach_prompting import (
        _forecast_signal_rows,
        build_explainer_prompt_text,
    )
    from trailtraining.llm.constraints import constraint_config_from_env, derive_effective_constraints
    from trailtraining.llm.signals import build_retrieval_context

    data = _load_fixture_bundle(fixture_dir)
    machine_plan_obj, machine_plan_source = _resolve_machine_plan(saved_plan_json, machine_plan_json)

    style = str(data["fixture_meta"].get("style") or "trailrunning")
    primary_goal = str(data["fixture_meta"].get("primary_goal") or "to become a faster trail runner")
    lifestyle_notes = str(data["fixture_meta"].get("lifestyle_notes") or "")
    effective = derive_effective_constraints(
        det_forecast=data["forecast"],
        rollups=data["rollups"],
        cfg=constraint_config_from_env(),
        lifestyle_notes=lifestyle_notes,
    )

    if detail_days is None:
        detail_days = max(1, min(14, len(data["combined"])))

    prompt_text = build_explainer_prompt_text(
        machine_plan=machine_plan_obj,
        personal=data["personal"],
        rollups=data["rollups"],
        combined=data["combined"],
        deterministic_forecast=data["forecast"],
        style=style,
        primary_goal=primary_goal,
        lifestyle_notes=lifestyle_notes,
        max_chars=max_chars,
        detail_days=detail_days,
        effective_constraints=effective,
    )

    retrieval_ctx = build_retrieval_context(data["combined"], data["rollups"], retrieval_weeks=8)
    signal_registry = list(retrieval_ctx.get("signal_registry") or [])
    if isinstance(data["forecast"], dict):
        signal_registry.extend(_forecast_signal_rows(data["forecast"]))

    prompt_hits = _collect_placeholder_leaks(prompt_text, path="$.prompt")
    registry_hits = _collect_placeholder_leaks(signal_registry, path="$.signal_registry")
    machine_hits = _collect_placeholder_leaks(machine_plan_obj, path="$.machine_plan")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "explainer_prompt.txt").write_text(prompt_text, encoding="utf-8")
    _write_json(out_dir / "signal_registry.json", signal_registry)
    _write_json(out_dir / "machine_plan.json", machine_plan_obj)
    _write_json(
        out_dir / "audit.json",
        {
            "fixture_dir": str(fixture_dir),
            "machine_plan_source": machine_plan_source,
            "markers": list(PLACEHOLDER_MARKERS),
            "prompt_hits": [{"path": p, "value": v} for p, v in prompt_hits],
            "signal_registry_hits": [{"path": p, "value": v} for p, v in registry_hits],
            "machine_plan_hits": [{"path": p, "value": v} for p, v in machine_hits],
            "counts": {
                "prompt_hits": len(prompt_hits),
                "signal_registry_hits": len(registry_hits),
                "machine_plan_hits": len(machine_hits),
            },
        },
    )

    print(f"[saved] {out_dir / 'explainer_prompt.txt'}")
    print(f"[saved] {out_dir / 'signal_registry.json'}")
    print(f"[saved] {out_dir / 'machine_plan.json'}")
    print(f"[saved] {out_dir / 'audit.json'}")

    total_hits = len(prompt_hits) + len(registry_hits) + len(machine_hits)
    if total_hits:
        print(f"[warn] placeholder markers found: {total_hits}", file=sys.stderr)
        for group_name, hits in [
            ("prompt", prompt_hits),
            ("signal_registry", registry_hits),
            ("machine_plan", machine_hits),
        ]:
            for path, value in hits[:10]:
                print(f"  {group_name}: {path} = {value!r}", file=sys.stderr)
        return 2

    print("[ok] no placeholder markers found in prompt, signal registry, or machine plan")
    return 0


def validate_saved_plan(saved_plan_json: Path, out_dir: Path) -> int:
    obj = _load_json(saved_plan_json)
    hits = _collect_placeholder_leaks(obj)
    _write_json(
        out_dir / "saved_plan_leak_audit.json",
        {
            "saved_plan_json": str(saved_plan_json),
            "markers": list(PLACEHOLDER_MARKERS),
            "hits": [{"path": p, "value": v} for p, v in hits],
            "count": len(hits),
        },
    )
    print(f"[saved] {out_dir / 'saved_plan_leak_audit.json'}")
    if hits:
        print(f"[warn] placeholder markers found in saved plan: {len(hits)}", file=sys.stderr)
        for path, value in hits[:20]:
            print(f"  {path} = {value!r}", file=sys.stderr)
        return 3
    print("[ok] no placeholder markers found in saved plan")
    return 0


BAD_STAGE_EXAMPLE = {
    "snapshot": {
        "last7": {
            "distance_km": ">{signal_id: ",
            "moving_time_hours": ">{signal_id: ",
            "elevation_m": ">{signal_id: ",
            "activity_count": ">{signal_id: ",
            "sleep_hours_mean": ">{signal_id: ",
            "hrv_mean": ">{signal_id: ",
            "rhr_mean": ">{signal_id: ",
        },
        "baseline28": {
            "distance_km": ">{signal_id: ",
            "moving_time_hours": ">{signal_id: ",
            "elevation_m": ">{signal_id: ",
            "activity_count": ">{signal_id: ",
            "sleep_hours_mean": ">{signal_id: ",
            "hrv_mean": ">{signal_id: ",
            "rhr_mean": ">{signal_id: ",
        },
        "notes": "Bad canned stage artifact for leak-guard testing.",
    },
    "readiness_rationale": "Example rationale.",
    "readiness_signal_ids": ["forecast.readiness.status"],
    "day_explanations": [],
    "recovery": {"actions": [], "signal_ids": []},
    "risks": [],
    "data_notes": [],
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline replay + prompt hygiene audit for the bad fixture.")
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--saved-plan-json", type=Path)
    parser.add_argument("--machine-plan-json", type=Path)
    parser.add_argument("--max-chars", type=int, default=20000)
    parser.add_argument("--detail-days", type=int)
    parser.add_argument("--validate-saved-plan", action="store_true")
    parser.add_argument("--write-canned-bad-stage", action="store_true")
    args = parser.parse_args()

    rc = build_replay_artifacts(
        fixture_dir=args.fixture_dir,
        out_dir=args.out_dir,
        saved_plan_json=args.saved_plan_json,
        machine_plan_json=args.machine_plan_json,
        max_chars=args.max_chars,
        detail_days=args.detail_days,
    )

    if args.write_canned_bad_stage:
        _write_json(args.out_dir / "canned_bad_stage_output.json", BAD_STAGE_EXAMPLE)
        print(f"[saved] {args.out_dir / 'canned_bad_stage_output.json'}")

    if args.validate_saved_plan:
        if args.saved_plan_json is None:
            raise SystemExit("--validate-saved-plan requires --saved-plan-json")
        rc = max(rc, validate_saved_plan(args.saved_plan_json, args.out_dir))

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
