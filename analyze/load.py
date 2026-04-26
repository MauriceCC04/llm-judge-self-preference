"""analyze/load.py — load JSONL judgments + provenance sidecars → DataFrame.

Provenance sidecars supply arm, fixture_id, source_model, and pair-level LLM
identity columns used by the H3/H4 self-preference analyses.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

Kind = Literal["pairwise", "soft_eval"]


def _load_provenance_index(provenance_dir: Path) -> dict[str, dict[str, Any]]:
    """Return {plan_id: prov_dict} for all *.provenance.json files found."""
    index: dict[str, dict[str, Any]] = {}
    for prov_path in sorted(provenance_dir.glob("*.provenance.json")):
        try:
            prov = json.loads(prov_path.read_text())
            index[str(prov["plan_id"])] = prov
        except Exception:
            pass
    return index


def _get_prov(
    index: dict[str, dict[str, Any]],
    plan_id: str,
    field: str,
    default: str = "unknown",
) -> str:
    return index.get(str(plan_id), {}).get(field, default)


def _pick_llm_side(row: Any) -> str:
    if row.get("arm_a") == "llm":
        return "a"
    if row.get("arm_b") == "llm":
        return "b"
    return ""


def _llm_in_position_a(row: Any) -> int:
    llm_side = row.get("llm_side", "")
    pos = row.get("position", "")
    if not llm_side or pos not in {"AB", "BA"}:
        return 0
    if pos == "AB":
        return 1 if llm_side == "a" else 0
    return 1 if llm_side == "b" else 0


def _referenced_plan_ids(
    df: "pd.DataFrame",
    *,
    kind: Kind,
) -> set[str]:
    ids: set[str] = set()
    if df.empty:
        return ids

    if kind == "pairwise":
        if "plan_a_id" in df.columns:
            ids.update(str(x) for x in df["plan_a_id"].dropna().tolist())
        if "plan_b_id" in df.columns:
            ids.update(str(x) for x in df["plan_b_id"].dropna().tolist())
    elif kind == "soft_eval" and "plan_id" in df.columns:
        ids.update(str(x) for x in df["plan_id"].dropna().tolist())

    return ids


def collect_invalid_plan_ids(
    prov_index: dict[str, dict[str, Any]],
    *,
    referenced_plan_ids: set[str],
    expected_explainer: str,
    require_verified_explainer: bool = False,
) -> tuple[set[str], list[dict[str, Any]]]:
    invalid: set[str] = set()
    exclusions: list[dict[str, Any]] = []

    for plan_id in sorted(referenced_plan_ids):
        prov = prov_index.get(str(plan_id))
        reason: str | None = None

        if prov is None:
            reason = "missing_provenance"
            explainer_model = None
            actual_explainer_model = None
            explainer_model_verified = None
        else:
            explainer_model = prov.get("explainer_model")
            actual_explainer_model = prov.get("actual_explainer_model")
            explainer_model_verified = prov.get("explainer_model_verified")
            if explainer_model != expected_explainer:
                reason = "explainer_model_mismatch"
            elif actual_explainer_model and actual_explainer_model != expected_explainer:
                reason = "actual_explainer_model_mismatch"
            elif require_verified_explainer and not bool(explainer_model_verified):
                reason = "explainer_model_unverified"

        if reason is not None:
            invalid.add(str(plan_id))
            exclusions.append(
                {
                    "plan_id": str(plan_id),
                    "reason": reason,
                    "explainer_model": explainer_model,
                    "actual_explainer_model": actual_explainer_model,
                    "explainer_model_verified": explainer_model_verified,
                }
            )

    return invalid, exclusions


def apply_provenance_exclusions(
    df: "pd.DataFrame",
    *,
    kind: Kind,
    invalid_plan_ids: set[str],
) -> "pd.DataFrame":
    if df.empty or not invalid_plan_ids:
        return df

    if kind == "pairwise":
        if "plan_a_id" not in df.columns or "plan_b_id" not in df.columns:
            return df
        mask = (~df["plan_a_id"].astype(str).isin(invalid_plan_ids)) & (
            ~df["plan_b_id"].astype(str).isin(invalid_plan_ids)
        )
        return df.loc[mask].copy()

    if kind == "soft_eval" and "plan_id" in df.columns:
        return df.loc[~df["plan_id"].astype(str).isin(invalid_plan_ids)].copy()

    return df


def load_judgments(
    judgments_dir: Path,
    provenance_dir: Path,
    *,
    kind: Kind = "pairwise",
    expected_explainer: str | None = None,
    require_verified_explainer: bool = False,
    return_exclusions: bool = False,
) -> "pd.DataFrame | tuple[pd.DataFrame, list[dict[str, Any]]]":
    """Join judgment JSONL records with provenance sidecars."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required for analyze/load.py") from exc

    prov_index = _load_provenance_index(provenance_dir)

    records: list[dict[str, Any]] = []
    for jpath in sorted(judgments_dir.glob("*.jsonl")):
        if "schema_failures" in jpath.name:
            continue
        if kind == "pairwise" and "softeval" in jpath.name:
            continue
        if kind == "soft_eval" and "pairwise" in jpath.name:
            continue

        with jpath.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    if not records:
        empty_df = pd.DataFrame()
        if return_exclusions:
            return empty_df, []
        return empty_df

    df = pd.DataFrame(records)

    if kind == "pairwise":
        if "plan_a_id" not in df.columns or "plan_b_id" not in df.columns:
            if return_exclusions:
                return df, []
            return df

        def _get(plan_id: str, field: str, default: str = "unknown") -> str:
            return prov_index.get(str(plan_id), {}).get(field, default)

        df["arm_a"] = df["plan_a_id"].apply(lambda x: _get(x, "arm"))
        df["arm_b"] = df["plan_b_id"].apply(lambda x: _get(x, "arm"))
        df["source_model_a"] = df["plan_a_id"].apply(lambda x: _get(x, "source_model", ""))
        df["source_model_b"] = df["plan_b_id"].apply(lambda x: _get(x, "source_model", ""))

        if "fixture_id" not in df.columns or df["fixture_id"].isna().all():
            df["fixture_id"] = df["plan_a_id"].apply(lambda x: _get(x, "fixture_id"))
        else:
            df["fixture_id"] = df["fixture_id"].fillna(
                df["plan_a_id"].apply(lambda x: _get(x, "fixture_id"))
            )

        df["llm_side"] = df.apply(_pick_llm_side, axis=1)
        df["llm_plan_id"] = df.apply(
            lambda r: r["plan_a_id"]
            if r["llm_side"] == "a"
            else r["plan_b_id"]
            if r["llm_side"] == "b"
            else "",
            axis=1,
        )
        df["programmatic_plan_id"] = df.apply(
            lambda r: r["plan_b_id"]
            if r["llm_side"] == "a"
            else r["plan_a_id"]
            if r["llm_side"] == "b"
            else "",
            axis=1,
        )
        df["llm_source_model"] = df.apply(
            lambda r: r["source_model_a"]
            if r["llm_side"] == "a"
            else r["source_model_b"]
            if r["llm_side"] == "b"
            else "",
            axis=1,
        )
        df["llm_in_position_a"] = df.apply(_llm_in_position_a, axis=1)

        def _preferred_plan_id(row: Any) -> str:
            preferred_id = row.get("preferred_id")
            if preferred_id:
                return str(preferred_id)
            preferred = row.get("preferred", "")
            if preferred == "plan_a":
                return str(row.get("plan_a_id", ""))
            if preferred == "plan_b":
                return str(row.get("plan_b_id", ""))
            return ""

        df["preferred_plan_id"] = df.apply(_preferred_plan_id, axis=1)
        df["prefers_llm"] = (
            (df["preferred_plan_id"] != "")
            & (df["preferred_plan_id"] != "tie")
            & (df["preferred_plan_id"] == df["llm_plan_id"])
        ).astype(int)

    elif kind == "soft_eval":
        if "plan_id" in df.columns:
            df["arm"] = df["plan_id"].apply(lambda x: _get_prov(prov_index, x, "arm"))
            df["fixture_id"] = df["plan_id"].apply(lambda x: _get_prov(prov_index, x, "fixture_id"))
            df["source_model"] = df["plan_id"].apply(lambda x: _get_prov(prov_index, x, "source_model", ""))

    exclusions: list[dict[str, Any]] = []
    if expected_explainer:
        referenced_plan_ids = _referenced_plan_ids(df, kind=kind)
        invalid_plan_ids, exclusions = collect_invalid_plan_ids(
            prov_index,
            referenced_plan_ids=referenced_plan_ids,
            expected_explainer=expected_explainer,
            require_verified_explainer=require_verified_explainer,
        )
        df = apply_provenance_exclusions(df, kind=kind, invalid_plan_ids=invalid_plan_ids)

    if return_exclusions:
        return df, exclusions
    return df


def detect_position_bias(
    df: "pd.DataFrame",
    *,
    threshold: float = 0.2,
) -> dict[str, Any]:
    """Detect position bias for a single-judge DataFrame."""
    judge_name = df["judge"].iloc[0] if "judge" in df.columns and len(df) else "unknown"
    n = len(df)
    if n == 0:
        return {"biased": False, "p_prefers_a": 0.5, "judge": judge_name, "n": 0}

    if "preferred" in df.columns and df["preferred"].isin(["plan_a", "plan_b"]).all():
        n_pos_a = int((df["preferred"] == "plan_a").sum())
    elif "preferred_id" in df.columns and "plan_a_id" in df.columns:
        n_pos_a = int((df["preferred_id"] == df["plan_a_id"]).sum())
    else:
        return {
            "biased": False,
            "p_prefers_a": 0.5,
            "judge": judge_name,
            "n": n,
            "threshold": threshold,
            "note": "cannot determine position preference",
        }

    p = n_pos_a / n
    return {
        "biased": abs(p - 0.5) >= threshold,
        "p_prefers_a": round(p, 4),
        "judge": judge_name,
        "n": n,
        "threshold": threshold,
    }
