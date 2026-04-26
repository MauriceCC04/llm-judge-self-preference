"""analyze/style_audit.py — paired surface-form leakage audit."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

WORD_RE = re.compile(r"[A-Za-z0-9']+")


def _tokenize(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def _safe_text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _collect_narrative_text(plan_obj: dict[str, Any]) -> str:
    parts: list[str] = []
    plan = plan_obj.get("plan") or {}
    for day in plan.get("days") or []:
        parts.append(_safe_text(day.get("purpose")))
        parts.append(_safe_text(day.get("workout")))
    readiness = plan_obj.get("readiness") or {}
    parts.append(_safe_text(readiness.get("rationale")))
    snapshot = plan_obj.get("snapshot") or {}
    parts.append(_safe_text(snapshot.get("notes")))
    for risk in plan_obj.get("risks") or []:
        if isinstance(risk, dict):
            parts.append(_safe_text(risk.get("message")))
    for note in plan_obj.get("data_notes") or []:
        if isinstance(note, str):
            parts.append(note)
        elif isinstance(note, dict):
            parts.append(_safe_text(note.get("message")))
    return "".join(p for p in parts if p)


def extract_surface_features(plan_obj: dict[str, Any]) -> dict[str, float]:
    plan = plan_obj.get("plan") or {}
    days = plan.get("days") or []
    readiness = plan_obj.get("readiness") or {}
    recovery = plan_obj.get("recovery") or {}
    snapshot = plan_obj.get("snapshot") or {}
    risks = plan_obj.get("risks") or []
    citations = plan_obj.get("citations") or []
    claim_attributions = plan_obj.get("claim_attributions") or []
    data_notes = plan_obj.get("data_notes") or []

    purposes = [_safe_text(d.get("purpose")) for d in days if isinstance(d, dict)]
    workouts = [_safe_text(d.get("workout")) for d in days if isinstance(d, dict)]
    signal_counts = [len(d.get("signal_ids") or []) for d in days if isinstance(d, dict)]

    narrative = _collect_narrative_text(plan_obj)
    tokens = _tokenize(narrative)
    numeric_tokens = sum(1 for t in tokens if any(ch.isdigit() for ch in t))
    type_token_ratio = (len(set(tokens)) / len(tokens)) if tokens else 0.0

    return {
        "total_chars_narrative": float(len(narrative)),
        "mean_purpose_chars": float(sum(len(x) for x in purposes) / len(purposes)) if purposes else 0.0,
        "mean_workout_chars": float(sum(len(x) for x in workouts) / len(workouts)) if workouts else 0.0,
        "readiness_rationale_chars": float(len(_safe_text(readiness.get("rationale")))),
        "snapshot_notes_chars": float(len(_safe_text(snapshot.get("notes")))),
        "n_recovery_actions": float(len(recovery.get("actions") or [])),
        "n_risks": float(len(risks)),
        "n_data_notes": float(len(data_notes)),
        "n_citations": float(len(citations)),
        "n_claim_attributions": float(len(claim_attributions)),
        "numeric_token_ratio": float(numeric_tokens / len(tokens)) if tokens else 0.0,
        "type_token_ratio": float(type_token_ratio),
        "mean_signal_ids_per_day": float(sum(signal_counts) / len(signal_counts)) if signal_counts else 0.0,
    }


def _load_plan(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _sd(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = _mean(values)
    return math.sqrt(sum((x - mu) ** 2 for x in values) / (len(values) - 1))


def run_style_audit(
    plans_dir: Path,
    provenance_dir: Path,
    pairs_path: Path,
    output_dir: Path,
    *,
    critical_features: set[str] | None = None,
    z_threshold: float = 0.2,
) -> dict[str, Any]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required for style audit") from exc

    pairs = json.loads(Path(pairs_path).read_text(encoding="utf-8"))
    pair_rows: list[dict[str, Any]] = []

    for pair in pairs:
        llm_plan_id = pair["plan_a_id"] if pair.get("arm_a") == "llm" else pair["plan_b_id"]
        prog_plan_id = pair["plan_b_id"] if pair.get("arm_a") == "llm" else pair["plan_a_id"]
        llm_plan = _load_plan(plans_dir / f"{llm_plan_id}.json")
        prog_plan = _load_plan(plans_dir / f"{prog_plan_id}.json")
        llm_features = extract_surface_features(llm_plan)
        prog_features = extract_surface_features(prog_plan)

        row = {
            "pair_id": pair.get("pair_id", ""),
            "fixture_id": pair.get("fixture_id", ""),
            "llm_plan_id": llm_plan_id,
            "programmatic_plan_id": prog_plan_id,
        }
        for feat in sorted(llm_features.keys()):
            row[f"llm_{feat}"] = llm_features[feat]
            row[f"programmatic_{feat}"] = prog_features[feat]
            row[f"delta_{feat}"] = llm_features[feat] - prog_features[feat]
        pair_rows.append(row)

    pair_df = pd.DataFrame(pair_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_path = output_dir / "style_audit_pairwise.csv"
    pair_df.to_csv(pair_path, index=False)

    summary_rows: list[dict[str, Any]] = []
    feature_names = sorted(k[len("delta_"):] for k in pair_df.columns if k.startswith("delta_"))
    flagged = 0
    for feat in feature_names:
        deltas = [float(v) for v in pair_df[f"delta_{feat}"].dropna().tolist()]
        mean_delta = _mean(deltas) if deltas else float("nan")
        sd_delta = _sd(deltas) if deltas else float("nan")
        standardized = (mean_delta / sd_delta) if deltas and sd_delta not in (0.0, float("nan")) else 0.0
        is_flagged = abs(standardized) >= z_threshold
        if is_flagged:
            flagged += 1
        summary_rows.append(
            {
                "feature": feat,
                "mean_delta": mean_delta,
                "sd_delta": sd_delta,
                "standardized_delta": standardized,
                "pvalue": float("nan"),
                "n_pairs": len(deltas),
                "flagged": is_flagged,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = output_dir / "style_audit_summary.csv"
    summary_json = output_dir / "style_audit_summary.json"
    summary_df.to_csv(summary_csv, index=False)
    summary_payload = {
        "n_pairs": int(len(pair_df)),
        "n_features": int(len(summary_rows)),
        "n_flagged_features": int(flagged),
        "features": summary_rows,
    }
    summary_json.write_text(json.dumps(summary_payload, indent=2, default=str), encoding="utf-8")

    critical_features = critical_features or set()
    flagged_features = [r["feature"] for r in summary_rows if r["flagged"]]
    flagged_critical = [f for f in flagged_features if f in critical_features]
    gate = {
        "passed": len(flagged_critical) == 0,
        "flagged_features": flagged_features,
        "flagged_critical_features": flagged_critical,
        "z_threshold": z_threshold,
    }

    return {
        "n_pairs": int(len(pair_df)),
        "n_features": int(len(summary_rows)),
        "n_flagged_features": int(flagged),
        "pairwise_csv": str(pair_path),
        "summary_csv": str(summary_csv),
        "summary_json": str(summary_json),
        "gate": gate,
    }
