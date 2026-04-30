"""analyze/models.py — statistical model fits for H1, H3, H4."""
from __future__ import annotations

import math
from typing import Any

QWEN_SOURCE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
QWEN_PARAMS_B = {
    "qwen_7b_judge": 7.0,
    "qwen_14b_judge": 14.0,
    "qwen_32b_judge": 32.0,
}


def _safe_logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _bootstrap_preference_rate(
    df: "pd.DataFrame",
    *,
    n_boot: int = 2000,
    seed: int = 0,
    cluster_col: str = "pair_id",
) -> dict[str, float]:
    import numpy as np
    import pandas as pd

    if df.empty:
        return {
            "prob": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "pvalue_gt_half": float("nan"),
            "se_prob": float("nan"),
        }

    observed_prob = float(df["prefers_llm"].mean())

    if cluster_col not in df.columns or df[cluster_col].nunique() <= 1:
        draws = np.random.default_rng(seed).binomial(1, observed_prob, size=(n_boot, len(df)))
        boot_probs = draws.mean(axis=1)
    else:
        rng = np.random.default_rng(seed)
        grouped = [group["prefers_llm"].to_numpy(dtype=float) for _, group in df.groupby(cluster_col, sort=True)]
        n_clusters = len(grouped)
        boot_probs = np.empty(n_boot, dtype=float)
        for i in range(n_boot):
            sampled_indices = rng.integers(0, n_clusters, size=n_clusters)
            sampled = [grouped[idx] for idx in sampled_indices]
            total = sum(arr.sum() for arr in sampled)
            denom = sum(arr.size for arr in sampled)
            boot_probs[i] = float(total / denom) if denom else float("nan")

    ci_low, ci_high = np.quantile(boot_probs, [0.025, 0.975])
    pvalue_gt_half = float(np.mean(boot_probs <= 0.5))
    se_prob = float(np.nanstd(boot_probs, ddof=1))
    return {
        "prob": round(observed_prob, 4),
        "ci_low": round(float(ci_low), 4),
        "ci_high": round(float(ci_high), 4),
        "pvalue_gt_half": round(pvalue_gt_half, 4),
        "se_prob": round(se_prob, 4),
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


def fit_h1_model(df: "pd.DataFrame") -> dict[str, Any]:
    """H1 primary analysis: cluster bootstrap on P(prefer LLM)."""
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
            "prob_llm_ci_low": float("nan"),
            "prob_llm_ci_high": float("nan"),
            "se": float("nan"),
            "converged": False,
            "n_obs": 0,
            "method": "cluster_bootstrap_pairwise_rate",
        }

    boot = _bootstrap_preference_rate(df, cluster_col="pair_id" if "pair_id" in df.columns else "__none__")
    prob = float(boot["prob"])
    coef = _safe_logit(prob)
    se = float(boot["se_prob"]) / max(prob * (1 - prob), 1e-6)

    return {
        "coef_is_llm": round(coef, 4),
        "pvalue_is_llm": boot["pvalue_gt_half"],
        "prob_llm": boot["prob"],
        "prob_llm_ci_low": boot["ci_low"],
        "prob_llm_ci_high": boot["ci_high"],
        "se": round(se, 4),
        "converged": True,
        "n_obs": len(df),
        "method": "cluster_bootstrap_pairwise_rate",
        "null_hypothesis": "P(prefer LLM) <= 0.5",
    }


def fit_h3_model(df: "pd.DataFrame") -> dict[str, Any]:
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


def fit_h4_model(df: "pd.DataFrame") -> dict[str, Any]:
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


def add_same_family_column(
    df: "pd.DataFrame",
    *,
    judge_families: dict[str, str] | None = None,
    source_families: dict[str, str] | None = None,
) -> "pd.DataFrame":
    if judge_families is None:
        judge_families = {
            "qwen_7b_judge": "qwen",
            "qwen_14b_judge": "qwen",
            "gemma_4b_judge": "gemma",
            "gemma_12b_judge": "gemma",
        }
    if source_families is None:
        source_families = {
            "Qwen/Qwen2.5-7B-Instruct": "qwen",
            "google/gemma-3-4b-it": "gemma",
        }

    def _infer_family(model_id: object) -> str:
        if not model_id:
            return "unknown"
        s = str(model_id)
        if s in source_families:
            return source_families[s]
        s_lower = s.lower()
        if "qwen" in s_lower:
            return "qwen"
        if "llama" in s_lower:
            return "llama"
        if "mistral" in s_lower:
            return "mistral"
        return "unknown"

    def _pick_llm_source_model(row: object) -> str:
        r = row

        if "llm_source_model" in r and r["llm_source_model"]:
            return str(r["llm_source_model"])

        arm_a = r["arm_a"] if "arm_a" in r else None
        arm_b = r["arm_b"] if "arm_b" in r else None

        if arm_a == "llm" and "source_model_a" in r:
            return str(r["source_model_a"] or "")
        if arm_b == "llm" and "source_model_b" in r:
            return str(r["source_model_b"] or "")

        if "source_model_a" in r and r["source_model_a"]:
            return str(r["source_model_a"])
        if "source_model_b" in r and r["source_model_b"]:
            return str(r["source_model_b"])

        return ""

    df = df.copy()

    if "judge" in df.columns:
        df["judge_family"] = df["judge"].map(judge_families).fillna("unknown")
    else:
        df["judge_family"] = "unknown"

    df["llm_source_model_inferred"] = df.apply(_pick_llm_source_model, axis=1)
    df["llm_source_family"] = df["llm_source_model_inferred"].apply(_infer_family)
    df["same_family"] = (df["judge_family"] == df["llm_source_family"]).astype(int)

    return df
