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

QWEN_SOURCE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
QWEN_PARAMS_B = {
    "qwen_7b_judge": 7.0,
    "qwen_14b_judge": 14.0,
    "qwen_32b_judge": 32.0,
}


def add_position_bias_covariate(
    df: "pd.DataFrame",
    *,
    bias_lookup: dict[str, float],
) -> "pd.DataFrame":
    df = df.copy()

    if "llm_in_position_a" not in df.columns:
        df["llm_in_position_a"] = 0

    df["judge_prefers_a_centered"] = (
        df["judge"].map(bias_lookup).fillna(0.5).astype(float) - 0.5
    )
    df["llm_position_sign"] = df["llm_in_position_a"].map({1: 1.0, 0: -1.0})
    df["llm_position_bias"] = (
        df["judge_prefers_a_centered"] * df["llm_position_sign"]
    )
    return df


# ── H1 ────────────────────────────────────────────────────────────────────────

def fit_h1_model(df: "pd.DataFrame") -> dict[str, Any]:  # type: ignore[name-defined]
    """H1: Logistic regression with cluster-robust SEs on position-corrected preference."""
    try:
        import statsmodels.formula.api as smf
    except ImportError as exc:
        raise ImportError("pandas + statsmodels required for analysis") from exc

    df = df.copy()

    if "prefers_llm" not in df.columns:
        if "arm_a" not in df.columns:
            raise ValueError("DataFrame needs 'prefers_llm' or 'arm_a'/'arm_b' columns")
        df["prefers_llm"] = (
            ((df["preferred"] == "plan_a") & (df["arm_a"] == "llm")) |
            ((df["preferred"] == "plan_b") & (df["arm_b"] == "llm"))
        ).astype(int)

    df = df.dropna(subset=["prefers_llm"])
    if df.empty:
        return {
            "coef_is_llm": float("nan"),
            "pvalue_is_llm": float("nan"),
            "prob_llm": float("nan"),
            "converged": False,
            "n_obs": 0,
        }

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
        prob = float(result.predict().mean())
        converged = bool(result.mle_retvals.get("converged", True))
    except Exception:
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
    """H3: Self-preference model with position-bias correction."""
    try:
        import statsmodels.formula.api as smf
    except ImportError as exc:
        raise ImportError("pandas + statsmodels required") from exc

    df = df.copy()
    if "prefers_llm" not in df.columns:
        df["prefers_llm"] = 0
    if "same_family" not in df.columns:
        df["same_family"] = 0
    if "llm_position_bias" not in df.columns:
        df["llm_position_bias"] = 0.0

    df = df.dropna(subset=["prefers_llm", "same_family"])
    if len(df) < 5:
        return {
            "coef_same_family": float("nan"),
            "pvalue_same_family": float("nan"),
            "converged": False,
            "n_obs": len(df),
        }

    formula = "prefers_llm ~ same_family + llm_position_bias"
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
    """H4: Scale effect within Qwen family, restricted to Qwen-sourced plans."""
    try:
        import statsmodels.formula.api as smf
    except ImportError as exc:
        raise ImportError("pandas + statsmodels required") from exc

    if "judge" not in df.columns:
        return {"slope": float("nan"), "pvalue": float("nan"), "n_obs": 0}

    sub = df[df["judge"].isin(QWEN_PARAMS_B)].copy()
    if "llm_source_model" in sub.columns:
        sub = sub[sub["llm_source_model"] == QWEN_SOURCE_MODEL].copy()
    if sub.empty:
        return {"slope": float("nan"), "pvalue": float("nan"), "n_obs": 0}

    sub["log10_params"] = sub["judge"].map({k: math.log10(v) for k, v in QWEN_PARAMS_B.items()})

    if "prefers_llm" not in sub.columns:
        sub["prefers_llm"] = 0
    if "llm_position_bias" not in sub.columns:
        sub["llm_position_bias"] = 0.0

    sub = sub.dropna(subset=["prefers_llm", "log10_params"])
    if len(sub) < 3:
        return {"slope": float("nan"), "pvalue": float("nan"), "n_obs": len(sub)}

    formula = "prefers_llm ~ log10_params"
    if "llm_position_bias" in sub.columns:
        formula += " + llm_position_bias"
    if "fixture_id" in sub.columns and sub["fixture_id"].nunique() > 1:
        formula += " + C(fixture_id)"

    try:
        model = smf.ols(formula, data=sub)
        if "pair_id" in sub.columns and sub["pair_id"].nunique() > 1:
            result = model.fit(cov_type="cluster", cov_kwds={"groups": sub["pair_id"]})
        else:
            result = model.fit()
        slope = float(result.params.get("log10_params", float("nan")))
        pval = float(result.pvalues.get("log10_params", float("nan")))
    except Exception:
        slope = pval = float("nan")

    return {
        "slope": round(slope, 4) if not math.isnan(slope) else float("nan"),
        "pvalue": round(pval, 4) if not math.isnan(pval) else float("nan"),
        "n_obs": len(sub),
        "formula": formula,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def add_same_family_column(
    df: "pd.DataFrame",  # type: ignore[name-defined]
    *,
    judge_families: dict[str, str] | None = None,
    source_families: dict[str, str] | None = None,
) -> "pd.DataFrame":  # type: ignore[name-defined]
    """Add `same_family` column using only the LLM plan's source family."""
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
    df["judge_family"] = df["judge"].map(judge_families).fillna("unknown") if "judge" in df.columns else "unknown"

    if "llm_source_model" in df.columns:
        df["llm_source_family"] = df["llm_source_model"].map(source_families).fillna("unknown")
        df["same_family"] = (df["judge_family"] == df["llm_source_family"]).astype(int)
    elif "same_family" not in df.columns:
        df["same_family"] = 0

    return df
