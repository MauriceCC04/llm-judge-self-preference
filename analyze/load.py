"""analyze/load.py — load JSONL judgments + provenance sidecars → DataFrame.

Provenance sidecars supply arm, fixture_id, source_model, and (for pairwise)
source_model_a / source_model_b — the columns needed by add_same_family_column()
for the H3 self-preference test.
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
            index[prov["plan_id"]] = prov
        except Exception:
            pass
    return index


def load_judgments(
    judgments_dir: Path,
    provenance_dir: Path,
    *,
    kind: Kind = "pairwise",
) -> "pd.DataFrame":  # type: ignore[name-defined]  # noqa: F821
    """Join judgment JSONL records with provenance sidecars.

    Parameters
    ----------
    judgments_dir:
        Directory containing *.jsonl judgment files.
        Schema-failure files (matching ``schema_failures*.jsonl``) are skipped.
    provenance_dir:
        Directory containing *.provenance.json sidecars for all plans.
    kind:
        ``'pairwise'`` — records have pair_id, plan_a_id, plan_b_id.
        ``'soft_eval'`` — records have plan_id and per-rubric scores.

    Returns
    -------
    pd.DataFrame
        All judgment fields plus columns joined from provenance:

        Pairwise:
            arm_a, arm_b, fixture_id (if not already in the record),
            source_model_a, source_model_b,
            prefers_llm (position-corrected binary: 1 = LLM plan preferred)

        Soft-eval:
            arm, fixture_id, source_model
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required for analyze/load.py") from exc

    prov_index = _load_provenance_index(provenance_dir)

    # Load all JSONL records, skipping schema-failure files
    records: list[dict[str, Any]] = []
    for jpath in sorted(judgments_dir.glob("*.jsonl")):
        if "schema_failures" in jpath.name:
            continue
        # Filter by kind: pairwise files contain "pairwise", soft_eval files contain "softeval"
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
        return pd.DataFrame()

    df = pd.DataFrame(records)

    if kind == "pairwise":
        if "plan_a_id" not in df.columns:
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

        # Position-corrected preference: 1 = LLM arm plan was preferred
        def _prefers_llm(row: Any) -> int:
            preferred_id = row.get("preferred_id") or row.get("preferred", "")
            if not preferred_id or preferred_id == "tie":
                return 0
            if preferred_id == row.get("plan_a_id"):
                winning_arm = row.get("arm_a", "unknown")
            else:
                winning_arm = row.get("arm_b", "unknown")
            return 1 if winning_arm == "llm" else 0

        df["prefers_llm"] = df.apply(_prefers_llm, axis=1)

    elif kind == "soft_eval":
        if "plan_id" in df.columns:
            df["arm"] = df["plan_id"].apply(lambda x: _get_prov(prov_index, x, "arm"))
            df["fixture_id"] = df["plan_id"].apply(lambda x: _get_prov(prov_index, x, "fixture_id"))
            df["source_model"] = df["plan_id"].apply(lambda x: _get_prov(prov_index, x, "source_model", ""))

    return df


def _get_prov(index: dict[str, dict[str, Any]], plan_id: str, field: str, default: str = "unknown") -> str:
    return index.get(str(plan_id), {}).get(field, default)


def detect_position_bias(
    df: "pd.DataFrame",  # type: ignore[name-defined]  # noqa: F821
    *,
    threshold: float = 0.2,
) -> dict[str, Any]:
    """Detect position bias for a single-judge DataFrame.

    A judge is position-biased if |P(prefer the plan served as plan_a) − 0.5|
    exceeds *threshold* (PREREGISTRATION: 0.2).

    The raw ``preferred`` column from the harness is either ``"plan_a"`` /
    ``"plan_b"`` (position labels) or an actual plan ID.  Both cases are
    handled:

    * If the value is ``"plan_a"`` / ``"plan_b"``, use it directly.
    * Otherwise check whether ``preferred_id == plan_a_id``.
    """
    judge_name = df["judge"].iloc[0] if "judge" in df.columns and len(df) else "unknown"
    n = len(df)
    if n == 0:
        return {"biased": False, "p_prefers_a": 0.5, "judge": judge_name, "n": 0}

    # Determine whether position-A plan was preferred
    if "preferred" in df.columns and df["preferred"].isin(["plan_a", "plan_b"]).all():
        n_pos_a = int((df["preferred"] == "plan_a").sum())
    elif "preferred_id" in df.columns and "plan_a_id" in df.columns:
        n_pos_a = int((df["preferred_id"] == df["plan_a_id"]).sum())
    else:
        # Can't determine — assume no bias
        return {"biased": False, "p_prefers_a": 0.5, "judge": judge_name, "n": n,
                "threshold": threshold, "note": "cannot determine position preference"}

    p = n_pos_a / n
    return {
        "biased": abs(p - 0.5) >= threshold,
        "p_prefers_a": round(p, 4),
        "judge": judge_name,
        "n": n,
        "threshold": threshold,
    }
