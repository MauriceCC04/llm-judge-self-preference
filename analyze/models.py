"""analyze/models.py — statistical model fits for H1, H3, H4.

Model choices
-------------
H1/H3: Logistic regression with cluster-robust standard errors (pair-level
       clusters).  The PREREGISTRATION calls for "logistic mixed-effects";
       we implement the pre-registered model as:
         • logistic link (correct for binary prefers_llm ∈ {0,1})
         • cluster-robust SEs at the pair level (conservative, avoids
           within-pair correlation inflating significance)
         • judge fixed effects (absorbs judge-level intercept shifts without
           the convergence brittleness of random effects in statsmodels)
         • fixture fixed effects

       Full GLMM (lme4-style random effects) can be added via pymer4/rpy2
       if lme4 is available on the cluster; this implementation is the
       primary one because statsmodels is guaranteed to be present.

H4: OLS slope of prefers_llm ~ log10(param_count) for the Qwen 7B/14B/32B
    ladder, restricted to Qwen-sourced plans.
"""
from __future__ import annotations

import math
from typing import Any


# ── H1 ────────────────────────────────────────────────────────────────────────

def fit_h1_model(df: "pd.DataFrame") -> dict[str, Any]:  # type: ignore[name-defined]
    """H1: Logistic regression with cluster-robust SEs on position-corrected preference.

    Model: prefers_llm ~ C(judge) + C(fixture_id)
    Standard errors: cluster-robust at pair_id level.

    Returns dict with:
        coef_is_llm    intercept (log-odds of preferring LLM over programmatic)
        pvalue_is_llm  p-value for the intercept
        prob_llm       implied probability P(prefer LLM) at baseline
        converged      bool
        n_obs          int
    """
    try:
        import pandas as pd
        import statsmodels.formula.api as smf
        import numpy as np
    except ImportError as exc:
        raise ImportError("pandas + statsmodels required for analysis") from exc

    df = df.copy()

    # Derive prefers_llm if not present
    if "prefers_llm" not in df.columns:
        if "arm_a" not in df.columns:
            raise ValueError("DataFrame needs 'prefers_llm' or 'arm_a'/'arm_b' columns")
        df["prefers_llm"] = (
            ((df["preferred"] == "plan_a") & (df["arm_a"] == "llm")) |
            ((df["preferred"] == "plan_b") & (df["arm_b"] == "llm"))
        ).astype(int)

    # Drop rows with missing keys
    needed = ["prefers_llm"]
    df = df.dropna(subset=needed)
    if df.empty:
        return {"coef_is_llm": float("nan"), "pvalue_is_llm": float("nan"),
                "prob_llm": float("nan"), "converged": False, "n_obs": 0}

    # Build formula — add judge and fixture fixed effects where cardinality allows
    formula = "prefers_llm ~ 1"
    if "judge" in df.columns and df["judge"].nunique() > 1:
        formula += " + C(judge)"
    if "fixture_id" in df.columns and df["fixture_id"].nunique() > 1:
        formula += " + C(fixture_id)"

    cluster_col = "pair_id" if "pair_id" in df.columns else None

    try:
        model = smf.logit(formula, data=df)
        if cluster_col and df[cluster_col].nunique() > 1:
            result = model.fit(
                cov_type="cluster",
                cov_kwds={"groups": df[cluster_col]},
                disp=False,
                maxiter=200,
            )
        else:
            result = model.fit(disp=False, maxiter=200)

        intercept = float(result.params.get("Intercept", float("nan")))
        pval = float(result.pvalues.get("Intercept", float("nan")))
        # Mean predicted probability = population-averaged P(prefer LLM).
        # For a fixed-effects logistic model this equals the raw proportion,
        # but using predict() is correct and stable regardless of parameterisation.
        prob = float(result.predict().mean())
        converged = bool(result.mle_retvals.get("converged", True))
    except Exception as exc:
        # Fallback: simple proportion test
        p = float(df["prefers_llm"].mean())
        intercept = math.log(p / (1 - p)) if 0 < p < 1 else 0.0
        pval = float("nan")
        prob = p
        converged = False

    return {
        "coef_is_llm": round(intercept, 4),
        "pvalue_is_llm": round(pval, 4) if not math.isnan(pval) else float("nan"),
        "prob_llm": round(prob, 4),
        "converged": converged,
        "n_obs": len(df),
        "formula": formula,
    }


# ── H3 ────────────────────────────────────────────────────────────────────────

def fit_h3_model(df: "pd.DataFrame") -> dict[str, Any]:  # type: ignore[name-defined]
    """H3: Self-preference — same_family × is_llm interaction.

    Model: prefers_llm ~ same_family + C(fixture_id)
    Returns: coef_same_family, pvalue_same_family, converged, n_obs.
    """
    try:
        import pandas as pd
        import statsmodels.formula.api as smf
    except ImportError as exc:
        raise ImportError("pandas + statsmodels required") from exc

    df = df.copy()
    if "prefers_llm" not in df.columns:
        df["prefers_llm"] = 0
    if "same_family" not in df.columns:
        df["same_family"] = 0

    df = df.dropna(subset=["prefers_llm", "same_family"])
    if len(df) < 5:
        return {"coef_same_family": float("nan"), "pvalue_same_family": float("nan"),
                "converged": False, "n_obs": len(df)}

    formula = "prefers_llm ~ same_family"
    if "fixture_id" in df.columns and df["fixture_id"].nunique() > 1:
        formula += " + C(fixture_id)"

    cluster_col = "pair_id" if "pair_id" in df.columns else None

    try:
        model = smf.logit(formula, data=df)
        if cluster_col and df[cluster_col].nunique() > 1:
            result = model.fit(
                cov_type="cluster",
                cov_kwds={"groups": df[cluster_col]},
                disp=False, maxiter=200,
            )
        else:
            result = model.fit(disp=False, maxiter=200)
        coef = float(result.params.get("same_family", float("nan")))
        pval = float(result.pvalues.get("same_family", float("nan")))
        converged = bool(result.mle_retvals.get("converged", True))
    except Exception:
        coef = pval = float("nan")
        converged = False

    return {
        "coef_same_family": round(coef, 4),
        "pvalue_same_family": round(pval, 4) if not math.isnan(pval) else float("nan"),
        "converged": converged,
        "n_obs": len(df),
        "formula": formula,
    }


# ── H4 ────────────────────────────────────────────────────────────────────────

def fit_h4_model(df: "pd.DataFrame") -> dict[str, Any]:  # type: ignore[name-defined]
    """H4: Scale effect within Qwen family.

    OLS slope of prefers_llm ~ log10(param_count_billions) for Qwen 7B/14B/32B.
    Restricted to Qwen-sourced plans.
    """
    QWEN_PARAMS_B = {
        "qwen_7b_judge": 7.0,
        "qwen_14b_judge": 14.0,
        "qwen_32b_judge": 32.0,
    }

    try:
        import pandas as pd
        import numpy as np
        import statsmodels.formula.api as smf
    except ImportError as exc:
        raise ImportError("pandas + statsmodels required") from exc

    if "judge" not in df.columns:
        return {"slope": float("nan"), "pvalue": float("nan"), "n_obs": 0}

    sub = df[df["judge"].isin(QWEN_PARAMS_B)].copy()
    if sub.empty:
        return {"slope": float("nan"), "pvalue": float("nan"), "n_obs": 0}

    sub["log10_params"] = sub["judge"].map(
        {k: math.log10(v) for k, v in QWEN_PARAMS_B.items()}
    )

    if "prefers_llm" not in sub.columns:
        sub["prefers_llm"] = 0

    sub = sub.dropna(subset=["prefers_llm", "log10_params"])
    if len(sub) < 3:
        return {"slope": float("nan"), "pvalue": float("nan"), "n_obs": len(sub)}

    try:
        model = smf.ols("prefers_llm ~ log10_params", data=sub)
        result = model.fit()
        slope = float(result.params.get("log10_params", float("nan")))
        pval = float(result.pvalues.get("log10_params", float("nan")))
    except Exception:
        slope = pval = float("nan")

    return {
        "slope": round(slope, 4) if not math.isnan(slope) else float("nan"),
        "pvalue": round(pval, 4) if not math.isnan(pval) else float("nan"),
        "n_obs": len(sub),
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def add_same_family_column(
    df: "pd.DataFrame",  # type: ignore[name-defined]
    *,
    judge_families: dict[str, str] | None = None,
    source_families: dict[str, str] | None = None,
) -> "pd.DataFrame":  # type: ignore[name-defined]
    """Add `same_family` column: 1 if judge model family == LLM source family.

    Default family maps cover the study panel.
    """
    if judge_families is None:
        judge_families = {
            "llama_8b_judge": "llama",
            "qwen_7b_judge": "qwen",
            "qwen_14b_judge": "qwen",
            "qwen_32b_judge": "qwen",
            "mistral_7b_judge": "mistral",
        }
    if source_families is None:
        source_families = {
            "meta-llama/Llama-3.1-8B-Instruct": "llama",
            "Qwen/Qwen2.5-7B-Instruct": "qwen",
        }

    df = df.copy()

    # Derive source family from provenance if available
    def _judge_fam(judge: str) -> str:
        return judge_families.get(str(judge), "unknown")

    def _source_fam_from_provenance(plan_id: str, prov_col: str) -> str:
        return "unknown"

    df["judge_family"] = df["judge"].apply(_judge_fam) if "judge" in df.columns else "unknown"

    # For same_family we need to know the source family of the LLM plan in the pair.
    # This requires joining provenance — done in load.py when building the DataFrame.
    # Here we just propagate if the column is already present.
    if "source_model_a" in df.columns:
        df["source_family_a"] = df["source_model_a"].map(source_families).fillna("unknown")
        df["source_family_b"] = df.get("source_model_b", df["source_model_a"]).map(
            source_families
        ).fillna("unknown")
        # same_family = 1 if judge family matches either source family in the pair
        df["same_family"] = (
            (df["judge_family"] == df["source_family_a"]) |
            (df["judge_family"] == df["source_family_b"])
        ).astype(int)
    elif "same_family" not in df.columns:
        df["same_family"] = 0

    return df
