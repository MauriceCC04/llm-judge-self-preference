"""analyze/figures.py — forest plots, rubric heatmaps, position-bias audit.

All functions are matplotlib-based and write to files so they can be called
from a headless HPC node (no display required).

Requires: matplotlib, pandas, numpy (all in the [analysis] extra).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

QWEN_SOURCE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def _require_matplotlib() -> "types.ModuleType":  # type: ignore[name-defined]
    try:
        import matplotlib
        matplotlib.use("Agg")
        return matplotlib
    except ImportError as exc:
        raise ImportError("matplotlib is required for figures: pip install matplotlib") from exc


def save_position_bias_audit(
    df: "pd.DataFrame",  # type: ignore[name-defined]
    output_path: Path,
) -> None:
    """Write per-judge position-bias statistics as a CSV and a bar chart."""
    try:
        import pandas as pd
    except ImportError:
        return

    from analyze.load import detect_position_bias

    records = []
    for _, group in df.groupby("judge"):
        bias = detect_position_bias(group)
        records.append(bias)

    result_df = pd.DataFrame(records)
    csv_path = Path(str(output_path).replace(".png", ".csv")).with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(csv_path, index=False)
    print(f"[Saved] {csv_path}")

    try:
        _require_matplotlib()
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        judges = result_df["judge"].tolist()
        p_vals = result_df["p_prefers_a"].tolist()
        colors = ["#d62728" if b else "#2ca02c" for b in result_df["biased"]]

        ax.barh(judges, p_vals, color=colors, edgecolor="black", linewidth=0.5)
        ax.axvline(0.5, color="black", linewidth=1.5, linestyle="--", label="No bias (0.5)")
        ax.axvline(0.7, color="#d62728", linewidth=1, linestyle=":", alpha=0.7)
        ax.axvline(0.3, color="#d62728", linewidth=1, linestyle=":", alpha=0.7)

        ax.set_xlabel("P(prefer position-A plan)")
        ax.set_title("Position-bias audit by judge\n(red = biased, threshold |p−0.5| ≥ 0.2)")
        ax.set_xlim(0.2, 0.8)
        ax.legend(fontsize=8)
        fig.tight_layout()

        png_path = Path(output_path).with_suffix(".png")
        png_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        print(f"[Saved] {png_path}")
    except Exception as exc:
        print(f"[warn] position-bias chart failed: {exc}")


def save_h1_forest_plot(
    h1_result: dict[str, Any],
    output_path: Path,
    *,
    per_judge_results: list[dict[str, Any]] | None = None,
) -> None:
    """Forest plot showing overall H1 estimate and per-judge estimates."""
    try:
        _require_matplotlib()
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"[warn] forest plot skipped: {exc}")
        return

    fig, ax = plt.subplots(figsize=(7, 4 + len(per_judge_results or []) * 0.5))

    rows: list[dict[str, Any]] = []
    if per_judge_results:
        rows.extend(per_judge_results)
    rows.append({**h1_result, "label": "Overall (mixed-effects)", "is_overall": True})

    labels = [r.get("label", r.get("judge", "?")) for r in rows]
    coefs = [float(r.get("coef_is_llm", 0.0) or 0.0) for r in rows]
    ses = [float(r.get("se", 0.1) or 0.1) for r in rows]
    ci_lo = [c - 1.96 * s for c, s in zip(coefs, ses)]
    ci_hi = [c + 1.96 * s for c, s in zip(coefs, ses)]

    y = list(range(len(rows)))
    colors = ["#d62728" if r.get("is_overall") else "#1f77b4" for r in rows]

    for i in range(len(rows)):
        ax.plot([ci_lo[i], ci_hi[i]], [y[i], y[i]], color=colors[i], linewidth=2)
        ax.scatter([coefs[i]], [y[i]], color=colors[i], s=60, zorder=5)

    ax.axvline(0.0, color="black", linewidth=1.5, linestyle="--", label="No preference")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Log-odds coefficient for LLM plan preference\n(positive = prefer LLM)")
    ax.set_title("H1: Forest plot — judge-level and pooled LLM-plan preference")
    ax.legend(fontsize=8)
    fig.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[Saved] {out}")


def save_rubric_heatmap(
    rubric_results: dict[str, Any],
    output_path: Path,
) -> None:
    """Heatmap of per-rubric score deltas (LLM − programmatic)."""
    try:
        _require_matplotlib()
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"[warn] rubric heatmap skipped: {exc}")
        return

    rubric_ids = list(rubric_results.keys())
    if not rubric_ids:
        return

    deltas = [float(rubric_results[r].get("delta", 0.0) or 0.0) for r in rubric_ids]
    pvals_holm = [float(rubric_results[r].get("pvalue_holm", 1.0) or 1.0) for r in rubric_ids]
    significant = [bool(rubric_results[r].get("significant", False)) for r in rubric_ids]

    fig, ax = plt.subplots(figsize=(8, max(3, len(rubric_ids) * 0.7)))
    colors = ["#d62728" if s else "#aec7e8" for s in significant]
    bars = ax.barh(rubric_ids, deltas, color=colors, edgecolor="black", linewidth=0.5)
    ax.axvline(0.0, color="black", linewidth=1.5, linestyle="--")

    for i, (bar, pv, sig) in enumerate(zip(bars, pvals_holm, significant)):
        label = f"p={pv:.3f}{'*' if sig else ''}"
        x = bar.get_width()
        ax.text(
            x + (0.5 if x >= 0 else -0.5),
            i,
            label,
            va="center",
            ha="left" if x >= 0 else "right",
            fontsize=8,
        )

    ax.set_xlabel("Score delta: mean(LLM) − mean(programmatic)\n(red = Holm-significant)")
    ax.set_title("H2: Per-rubric score differences (LLM arm vs programmatic arm)")
    fig.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[Saved] {out}")


def save_rubric_heatmap_csv(
    rubric_results: dict[str, Any],
    output_path: Path,
) -> None:
    """Write rubric delta table as CSV."""
    try:
        import pandas as pd
    except ImportError:
        return

    rows = [{"rubric": rid, **stats} for rid, stats in rubric_results.items()]
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[Saved] {out}")


def save_h4_scale_curve(
    df: "pd.DataFrame",  # type: ignore[name-defined]
    output_path: Path,
) -> None:
    """Scatter + regression line for H4 (param count vs preference rate)."""
    try:
        _require_matplotlib()
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        print(f"[warn] H4 scale curve skipped: {exc}")
        return

    qwen_params = {
        "qwen_7b_judge": 7e9,
        "qwen_14b_judge": 14e9,
        "qwen_32b_judge": 32e9,
    }

    try:
        sub = df[df["judge"].isin(qwen_params)].copy()
        if "llm_source_model" in sub.columns:
            sub = sub[sub["llm_source_model"] == QWEN_SOURCE_MODEL].copy()
        if sub.empty:
            return

        sub["log_params"] = sub["judge"].map({k: math.log10(v / 1e9) for k, v in qwen_params.items()})
        sub["param_b"] = sub["judge"].map({k: v / 1e9 for k, v in qwen_params.items()})

        by_judge = sub.groupby(["judge", "log_params", "param_b"])["prefers_llm"].mean().reset_index()

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(by_judge["param_b"], by_judge["prefers_llm"], s=80, zorder=5)

        for _, row in by_judge.iterrows():
            ax.annotate(
                f"{row['param_b']:.0f}B",
                (row["param_b"], row["prefers_llm"]),
                textcoords="offset points",
                xytext=(5, 3),
                fontsize=9,
            )

        if len(by_judge) >= 2:
            x = by_judge["param_b"].values
            y = by_judge["prefers_llm"].values
            try:
                m, b = np.polyfit(x, y, 1)
                xr = np.linspace(x.min() * 0.8, x.max() * 1.1, 100)
                ax.plot(xr, m * xr + b, "r--", alpha=0.6, label=f"slope={m:.4f}/B")
                ax.legend(fontsize=8)
            except Exception:
                pass

        ax.axhline(0.5, color="black", linewidth=1, linestyle=":", alpha=0.5)
        ax.set_xlabel("Judge parameter count (billions)")
        ax.set_ylabel("P(prefer LLM plan)")
        ax.set_title("H4: Qwen scale ladder — self-preference vs parameter count")
        fig.tight_layout()

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"[Saved] {out}")
    except Exception as exc:
        print(f"[warn] H4 scale curve failed: {exc}")


def save_all_figures(
    df_pair: "pd.DataFrame",  # type: ignore[name-defined]
    df_soft: "pd.DataFrame",  # type: ignore[name-defined]
    h1_result: dict[str, Any],
    h2_result: dict[str, Any],
    output_dir: Path,
) -> None:
    """Generate the full figure pack into *output_dir*."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_position_bias_audit(df_pair, output_dir / "position_bias_audit.png")
    save_h1_forest_plot(h1_result, output_dir / "h1_forest.png")
    save_rubric_heatmap(h2_result, output_dir / "h2_rubric_heatmap.png")
    save_rubric_heatmap_csv(h2_result, output_dir / "h2_rubric_deltas.csv")

    if not df_pair.empty and "prefers_llm" in df_pair.columns:
        save_h4_scale_curve(df_pair, output_dir / "h4_scale_curve.png")