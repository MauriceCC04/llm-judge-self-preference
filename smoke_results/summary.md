# Analysis summary

## Provenance exclusions

Excluded plans due to missing provenance / explainer mismatch checks: 0

---

## Position-bias audit

Judges with |P(prefer_A) − 0.5| ≥ 0.2 (excluded from H1/H2): ['qwen_7b_judge']

---

## H1 — Pairwise LLM preference  ❌ Not supported

| Metric | Value |
|---|---|
| P(prefer LLM plan) | — |
| 95% CI | [—, —] |
| One-sided p-value for P(prefer LLM) > 0.5 | — |
| Descriptive log-odds at observed rate | None |
| N observations | 0 |
| Method | — |

---

## H2 — Per-rubric gap (mechanism)  ❌ Not supported

Paired analysis: True  
Matched judge-pair deltas contributing (explanation_quality): —

| Rubric | Δ (LLM − prog) | p (Holm) | Significant |
|---|---|---|---|

**H2 verdict:** explanation_quality Δ=—, plan_coherence Δ=—.  
Mechanism hypothesis not supported.

---

## H3 — Self-preference  ❌ Not supported

| Metric | Value |
|---|---|
| coef (same_family) | — |
| p-value | — |
| N | 80 |

### H3 by judge family

- **qwen**: coef=—, p=—, n=80

---

## H4 — Scale effect (Qwen ladder)  ❌ Not supported

| Metric | Value |
|---|---|
| Slope (per log10 B params) | — |
| p-value | — |
| N (Qwen judges on Qwen-sourced plans) | 0 |

---

## Style leakage audit

| Metric | Value |
|---|---|
| Flagged features | 4 |
| Total audited features | 13 |
| Interpretation | No flagged leakage above the audit threshold, not proof of zero leakage |
| Summary JSON | style_audit_summary.json |
