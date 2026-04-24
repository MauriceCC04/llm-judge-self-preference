"""analyze/run_analysis.py — complete paper-ready analysis pipeline.

Produces all tables and figures from the raw JSONL judgment files.

Usage::

    python -m analyze.run_analysis \
        --judgments judgments/ \
        --plans     plans/ \
        --fixtures  fixtures/data/ \
        --output    results/

Output layout::

    results/
    ├── summary.json                  ← machine-readable summary of all H1–H4
    ├── summary.md                    ← human-readable markdown summary
    ├── h1_logistic_result.json       ← full H1 model output
    ├── h1_forest.png                 ← H1 forest plot
    ├── h2_rubric_deltas.json         ← full H2 per-rubric table
    ├── h2_rubric_deltas.csv
    ├── h2_rubric_heatmap.png
    ├── h3_self_preference.json       ← H3 model output
    ├── h4_scale_result.json          ← H4 slope
    ├── h4_scale_curve.png
    ├── position_bias_audit.csv
    ├── position_bias_audit.png
    ├── schema_failure_rates.csv      ← judge reliability
    ├── pair_coverage.csv             ← matching audit
    ├── style_audit_pairwise.csv      ← style leakage features by matched pair
    ├── style_audit_summary.csv       ← paired style leakage summary
    └── style_audit_summary.json      ← machine-readable style leakage summary
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _require_pandas() -> None:
    try:
        import pandas  # noqa: F401
    except ImportError:
        print(
            "ERROR: pandas is required for analysis.
"
            "Install with: pip install -e '.[analysis]'",
            file=sys.stderr,
        )
        sys.exit(1)


# ── Individual analysis steps ─────────────────────────────────────────────────

def step_position_bias(df_pair: "pd.DataFrame", out: Path) -> dict:  # type: ignore[name-defined]
    from analyze.figures import save_position_bias_audit
    from analyze.load import detect_position_bias
    import pandas as pd

    records = []
    for judge_name, group in df_pair.groupby("judge"):
        bias = detect_position_bias(group)
        records.append(bias)

    result_df = pd.DataFrame(records)
    result_df.to_csv(out / "position_bias_audit.csv", index=False)

    try:
        save_position_bias_audit(df_pair, out / "position_bias_audit.png")
    except Exception as exc:
        print(f"  [warn] position-bias chart: {exc}")

    biased = result_df[result_df["biased"]]["judge"].tolist()
    print(f"  Position-bias audit: {len(records)} judges, {len(biased)} biased: {biased}")
    return {"biased_judges": biased, "audit": records}


def step_h1(df_pair: "pd.DataFrame", out: Path) -> dict:  # type: ignore[name-defined]
    from analyze.models import fit_h1_model
    from analyze.figures import save_h1_forest_plot

    result = fit_h1_model(df_pair)
    (out / "h1_logistic_result.json").write_text(json.dumps(result, indent=2, default=str))

    try:
        save_h1_forest_plot(result, out / "h1_forest.png")
    except Exception as exc:
        print(f"  [warn] H1 forest plot: {exc}")

    status = "SUPPORTED" if (
        result.get("prob_llm", 0.5) > 0.5
        and (result.get("pvalue_is_llm", 1.0) or 1.0) < 0.05
    ) else "NOT SUPPORTED"
    print(f"  H1: P(prefer LLM)={result.get('prob_llm', '?')}  "
          f"p={result.get('pvalue_is_llm', '?')}  [{status}]")
    return result


def step_h2(
    df_soft: "pd.DataFrame",
    out: Path,
    *,
    pairs_path: Path,
) -> dict:  # type: ignore[name-defined]
    from analyze.rubric_deltas import rubric_paired_contrasts
    from analyze.figures import save_rubric_heatmap, save_rubric_heatmap_csv

    if df_soft.empty:
        print("  H2: no soft-eval data")
        return {}

    if not pairs_path.exists():
        raise FileNotFoundError(
            f"H2 requires a matched-pairs manifest, but {pairs_path} does not exist."
        )

    result = rubric_paired_contrasts(df_soft, pairs_path=pairs_path)
    (out / "h2_rubric_deltas.json").write_text(json.dumps(result, indent=2, default=str))
    save_rubric_heatmap_csv(result, out / "h2_rubric_deltas.csv")

    try:
        save_rubric_heatmap(result, out / "h2_rubric_heatmap.png")
    except Exception as exc:
        print(f"  [warn] H2 heatmap: {exc}")

    exp_delta = result.get("explanation_quality", {}).get("delta", float("nan"))
    coh_delta = result.get("plan_coherence", {}).get("delta", float("nan"))
    exp_sig = result.get("explanation_quality", {}).get("significant", False)

    try:
        h2_supported = float(exp_delta) > float(coh_delta) and exp_sig
        status = "SUPPORTED" if h2_supported else "NOT SUPPORTED"
    except Exception:
        status = "?"

    paired_flag = result.get("explanation_quality", {}).get("paired")
    n_pairs = result.get("explanation_quality", {}).get("n_pairs")
    print(
        f"  H2: explanation_quality Δ={exp_delta}  plan_coherence Δ={coh_delta}  "
        f"paired={paired_flag}  n_pairs={n_pairs}  [{status}]"
    )
    return result


def step_h3(df_pair: "pd.DataFrame", out: Path) -> dict:  # type: ignore[name-defined]
    from analyze.models import fit_h3_model, add_same_family_column

    if "same_family" not in df_pair.columns:
        df_pair = add_same_family_column(df_pair)

    result = fit_h3_model(df_pair)
    (out / "h3_self_preference.json").write_text(json.dumps(result, indent=2, default=str))

    coef = result.get("coef_same_family", float("nan"))
    pval = result.get("pvalue_same_family", float("nan"))
    try:
        status = "SUPPORTED" if float(coef) > 0 and float(pval) < 0.05 else "NOT SUPPORTED"
    except Exception:
        status = "?"
    print(f"  H3: coef_same_family={coef}  p={pval}  [{status}]")
    return result


def step_h4(df_pair: "pd.DataFrame", out: Path) -> dict:  # type: ignore[name-defined]
    from analyze.models import fit_h4_model
    from analyze.figures import save_h4_scale_curve

    result = fit_h4_model(df_pair)
    (out / "h4_scale_result.json").write_text(json.dumps(result, indent=2, default=str))

    try:
        save_h4_scale_curve(df_pair, out / "h4_scale_curve.png")
    except Exception as exc:
        print(f"  [warn] H4 scale curve: {exc}")

    slope = result.get("slope", float("nan"))
    pval = result.get("pvalue", float("nan"))
    try:
        status = "SUPPORTED" if float(slope) > 0 and float(pval) < 0.05 else "NOT SUPPORTED"
    except Exception:
        status = "?"
    print(f"  H4: slope={slope}  p={pval}  n={result.get('n_obs')}  [{status}]")
    return result


def step_schema_failures(judgments_dir: Path, out: Path) -> None:
    import pandas as pd
    from vendor_patches.resume_jsonl import load_all

    records = []
    for jpath in sorted(judgments_dir.glob("schema_failures*.jsonl")):
        for r in load_all(jpath):
            records.append(r)

    if not records:
        print("  Schema failures: none recorded")
        return

    df = pd.DataFrame(records)
    by_judge = df.groupby("judge").size().reset_index(name="n_failures")
    by_judge.to_csv(out / "schema_failure_rates.csv", index=False)
    print(f"  Schema failures: {len(records)} total across {by_judge['judge'].nunique()} judges")


def step_pair_coverage(pairs_file: Path, out: Path) -> None:
    import pandas as pd

    if not pairs_file.exists():
        return

    pairs = json.loads(pairs_file.read_text())
    df = pd.DataFrame(pairs)
    if df.empty:
        return

    coverage = df.groupby("fixture_id").size().reset_index(name="n_pairs")
    coverage["mean_gap"] = df.groupby("fixture_id")["score_gap"].mean().values
    coverage.to_csv(out / "pair_coverage.csv", index=False)
    print(f"  Pair coverage: {len(pairs)} total pairs, "
          f"mean gap={df['score_gap'].mean():.3f}")


def step_style_audit(
    *,
    plans_dir: Path,
    provenance_dir: Path,
    pairs_path: Path,
    out: Path,
) -> dict:
    from analyze.style_audit import run_style_audit

    if not pairs_path.exists():
        print("  Style audit: skipped (matched pairs manifest missing)")
        return {}

    result = run_style_audit(
        plans_dir=plans_dir,
        provenance_dir=provenance_dir,
        pairs_path=pairs_path,
        output_dir=out,
    )
    flagged = result.get("n_flagged_features", 0)
    total = result.get("n_features", 0)
    print(f"  Style audit: {flagged}/{total} features flagged")
    return result


# ── Markdown summary ──────────────────────────────────────────────────────────

def write_markdown_summary(
    h1: dict, h2: dict, h3: dict, h4: dict,
    bias_audit: dict,
    style_audit: dict,
    out_path: Path,
) -> None:
    def _fmt(v: object) -> str:
        if isinstance(v, float) and (v != v):  # nan
            return "—"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    def _supported(cond: bool) -> str:
        return "✅ **Supported**" if cond else "❌ Not supported"

    h1_prob = h1.get("prob_llm", float("nan"))
    h1_pval = h1.get("pvalue_is_llm", 1.0)
    h1_supported = isinstance(h1_prob, float) and h1_prob > 0.5 and isinstance(h1_pval, float) and h1_pval < 0.05

    exp_delta = h2.get("explanation_quality", {}).get("delta", float("nan"))
    coh_delta = h2.get("plan_coherence", {}).get("delta", float("nan"))
    exp_sig = h2.get("explanation_quality", {}).get("significant", False)
    h2_paired = h2.get("explanation_quality", {}).get("paired", False)
    h2_pairs = h2.get("explanation_quality", {}).get("n_pairs", "—")
    try:
        h2_supported = float(exp_delta) > float(coh_delta) and exp_sig
    except Exception:
        h2_supported = False

    h3_coef = h3.get("coef_same_family", float("nan"))
    h3_pval = h3.get("pvalue_same_family", 1.0)
    try:
        h3_supported = float(h3_coef) > 0 and float(h3_pval) < 0.05
    except Exception:
        h3_supported = False

    h4_slope = h4.get("slope", float("nan"))
    h4_pval = h4.get("pvalue", 1.0)
    try:
        h4_supported = float(h4_slope) > 0 and float(h4_pval) < 0.05
    except Exception:
        h4_supported = False

    biased = bias_audit.get("biased_judges", [])
    style_flagged = style_audit.get("n_flagged_features", 0)
    style_n = style_audit.get("n_features", 0)

    md = f"""# Analysis summary

## Position-bias audit

Judges with |P(prefer_A) − 0.5| ≥ 0.2 (excluded from H1/H2): {biased if biased else 'none'}

---

## H1 — Pairwise LLM preference  {_supported(h1_supported)}

| Metric | Value |
|---|---|
| P(prefer LLM plan) | {_fmt(h1_prob)} |
| Log-odds intercept | {_fmt(h1.get("coef_is_llm"))} |
| p-value (cluster-robust) | {_fmt(h1_pval)} |
| N observations | {h1.get("n_obs", "—")} |
| Converged | {h1.get("converged", "—")} |

---

## H2 — Per-rubric gap (mechanism)  {_supported(h2_supported)}

Paired analysis: {h2_paired}  
Matched pairs contributing: {h2_pairs}

| Rubric | Δ (LLM − prog) | p (Holm) | Significant |
|---|---|---|---|
"""
    for rid, stats in sorted(h2.items()):
        md += f"| {rid} | {_fmt(stats.get('delta'))} | {_fmt(stats.get('pvalue_holm'))} | {'✅' if stats.get('significant') else '—'} |
"

    md += f"""
**H2 verdict:** explanation_quality Δ={_fmt(exp_delta)}, plan_coherence Δ={_fmt(coh_delta)}.  
{"Mechanism hypothesis supported: explanation gap > coherence gap." if h2_supported else "Mechanism hypothesis not supported."}

---

## H3 — Self-preference  {_supported(h3_supported)}

| Metric | Value |
|---|---|
| coef (same_family) | {_fmt(h3_coef)} |
| p-value | {_fmt(h3_pval)} |
| N | {h3.get("n_obs", "—")} |

---

## H4 — Scale effect (Qwen ladder)  {_supported(h4_supported)}

| Metric | Value |
|---|---|
| Slope (per log10 B params) | {_fmt(h4_slope)} |
| p-value | {_fmt(h4_pval)} |
| N (Qwen judges on Qwen-sourced plans) | {h4.get("n_obs", "—")} |

---

## Style leakage audit

| Metric | Value |
|---|---|
| Flagged features | {style_flagged} |
| Total audited features | {style_n} |
| Summary JSON | style_audit_summary.json |
"""
    out_path.write_text(md, encoding="utf-8")
    print(f"[Saved] {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Full analysis pipeline → paper outputs")
    parser.add_argument("--judgments", default="judgments/")
    parser.add_argument("--plans", default="plans/")
    parser.add_argument("--fixtures", default="fixtures/data/")
    parser.add_argument("--pairs", default="matched_pairs.json")
    parser.add_argument("--output", default="results/")
    args = parser.parse_args(argv)

    _require_pandas()

    from analyze.load import load_judgments
    from analyze.models import add_position_bias_covariate, add_same_family_column

    judgments_dir = Path(args.judgments)
    plans_dir = Path(args.plans)
    pairs_path = Path(args.pairs)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print("=== Analysis pipeline ===
")

    print("Loading pairwise judgments ...")
    df_pair = load_judgments(judgments_dir, plans_dir, kind="pairwise")
    print(f"  {len(df_pair)} pairwise records
")

    print("Loading soft-eval judgments ...")
    df_soft = load_judgments(judgments_dir, plans_dir, kind="soft_eval")
    print(f"  {len(df_soft)} soft-eval records
")

    print("── Position-bias audit ─────────────────────────────────")
    bias_audit = step_position_bias(df_pair, out)

    biased = bias_audit.get("biased_judges", [])
    df_pair_clean = df_pair[~df_pair["judge"].isin(biased)].copy() if not df_pair.empty else df_pair
    df_soft_clean = df_soft[~df_soft["judge"].isin(biased)].copy() if not df_soft.empty else df_soft
    if biased:
        print(f"  Excluded {biased} from H1/H2 (position bias)")

    bias_lookup = {
        r["judge"]: float(r["p_prefers_a"])
        for r in bias_audit.get("audit", [])
        if "judge" in r and "p_prefers_a" in r
    }

    df_pair_h3h4 = add_position_bias_covariate(df_pair, bias_lookup=bias_lookup)
    df_pair_h3h4 = add_same_family_column(df_pair_h3h4)

    print("
── H1: Pairwise LLM preference ─────────────────────────")
    h1 = step_h1(df_pair_clean, out)

    print("
── H2: Per-rubric gap ──────────────────────────────────")
    h2 = step_h2(df_soft_clean, out, pairs_path=pairs_path)

    print("
── H3: Self-preference ─────────────────────────────────")
    h3 = step_h3(df_pair_h3h4, out)

    print("
── H4: Scale effect ────────────────────────────────────")
    h4 = step_h4(df_pair_h3h4, out)

    print("
── Schema failures ─────────────────────────────────────")
    step_schema_failures(judgments_dir, out)

    print("
── Pair coverage ───────────────────────────────────────")
    step_pair_coverage(pairs_path, out)

    print("
── Style leakage audit ─────────────────────────────────")
    style_audit = step_style_audit(
        plans_dir=plans_dir,
        provenance_dir=plans_dir,
        pairs_path=pairs_path,
        out=out,
    )

    summary = {
        "h1": h1,
        "h2": h2,
        "h3": h3,
        "h4": h4,
        "bias_audit": bias_audit,
        "style_audit": style_audit,
        "n_pairwise": len(df_pair),
        "n_soft_eval": len(df_soft),
        "biased_judges_excluded": biased,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    write_markdown_summary(h1, h2, h3, h4, bias_audit, style_audit, out / "summary.md")

    print(f"
=== Done. Results in {out}/ ===")


if __name__ == "__main__":
    main()
