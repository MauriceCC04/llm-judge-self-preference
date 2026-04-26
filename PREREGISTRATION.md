# PREREGISTRATION.md

**Status: FROZEN** — do not modify after the first generation run begins.

---

## Study title

LLM-judge self-preference on marathon training plans: a controlled experiment on
Bocconi Jupiter I HPC.

## Research question

Controlling for deterministic plan quality, do LLM judges systematically favour
LLM-generated training plans over programmatically generated ones, and does this
effect vary by judge model family and scale?

---

## Hypotheses (pre-registered)

### H1 — Pairwise LLM preference (primary)

After position-swap correction, judges prefer LLM-arm plans to programmatic-arm
plans at a rate above 50%.

- **Operationalisation:** position-corrected binary outcome
  `prefers_llm ∈ {0, 1}` for each (pair, judge, run, position) record.
- **Test:** logistic mixed-effects model (see §Statistical models).
- **Decision rule:** H1 is supported if the estimated probability of preferring
  the LLM plan is > 0.5 and the corresponding p-value < 0.05.

### H2 — Per-rubric gap (mechanism hypothesis)

The LLM-preference gap is larger on `explanation_quality` than on
`plan_coherence`.

- **Operationalisation:** per-rubric soft-eval scores from
  `evaluate_training_plan_soft(skip_synthesis=True)`.
- **Test:** paired contrast on rubric score difference (LLM − programmatic),
  Holm-corrected across all 5 rubrics.
- **Decision rule:** H2 is supported if `explanation_quality` has a
  significantly positive contrast and its Holm-corrected p-value is smaller than
  `plan_coherence`'s.

### H3 — Self-preference (family axis)

`P(prefer LLM plan | judge family = source family) >
 P(prefer LLM plan | judge family ≠ source family)`.

- **Operationalisation:** binary `same_family` indicator (1 if judge model
  family matches source model family, 0 otherwise).
  - Llama judge + Llama source = same_family=1
  - Qwen judge + Qwen source = same_family=1
- **Test:** interaction term `same_family` in the mixed-effects model (H3 model,
  see §Statistical models).
- **Decision rule:** positive and significant `same_family` coefficient (p < 0.05).

### H4 — Scale effect (Qwen ladder)

Within the Qwen2.5 family, self-preference grows with judge parameter count
across the 7B / 14B / 32B ladder.

- **Operationalisation:** OLS slope of `prefers_llm ~ log(parameter_count)` for
  the three Qwen judges, restricted to Qwen-sourced plans.
- **Test:** positive slope with p < 0.05.
- **Decision rule:** positive slope coefficient on `log_params`.

---

## Study design parameters (frozen)

### Fixtures

| Parameter | Value |
|---|---|
| Number of fixture bundles | 8 |
| Axes | readiness × recovery_capability × race_phase |
| Levels per axis | low/high × low/high × base/peak |
| Fixture format | 4 committed JSON files per bundle |
| Fixture generation | deterministic (fixtures/build.py, frozen date 2026-03-17) |

### Generation

| Parameter | Value |
|---|---|
| Plans per arm (target) | ~256 (8 fixtures × 16 plans/fixture/source × 2 source models for LLM; 8 × 32 for programmatic) |
| Total plans (target) | ~512 |
| LLM arm source models | meta-llama/Llama-3.1-8B-Instruct, Qwen/Qwen2.5-7B-Instruct |
| Programmatic arm | sampler → guardrails → shared explainer |
| Shared explainer model | **Qwen/Qwen2.5-3B-Instruct** (constant `EXPLAINER_MODEL_ID`) |
| Two-stage pipeline | `TRAILTRAINING_TWO_STAGE_PLAN=1` |
| Endpoint configuration | two vLLMs: `TRAILTRAINING_SOURCE_LLM_BASE_URL` + `TRAILTRAINING_EXPLAINER_LLM_BASE_URL` |

**Critical invariant:** `EXPLAINER_MODEL_ID` must be identical for both arms.
Any divergence is a study-fatal writing-style confound. Test 8 in `run_tests.py`
enforces this at Gate 0.

### Matching

| Parameter | Value |
|---|---|
| Scoring function | `evaluate_training_plan_quality_file` |
| Binning | `(fixture_id, round(score))` |
| Tolerance | `|Δscore| ≤ 1.0` |
| Target pairs | 250 |
| Pair manifest | `matched_pairs.json` |
| Pre-judging audit | mean gap, per-fixture coverage; abort if coverage < 80% of target |

### Judging

| Parameter | Value |
|---|---|
| Active judges | 4 (`llama_8b_judge`, `qwen_7b_judge`, `qwen_14b_judge`, `qwen_32b_judge`) |
| Pairwise calls | 250 pairs × 4 judges × 5 runs × 2 positions = **10 000** |
| Per-rubric calls | ~512 plans × 4 judges ≈ **2 048** |
| `PAIRWISE_N_RUNS` | 5 (single source of truth: `generate.constants`) |
| `PAIRWISE_N_POSITIONS` | 2 (mandatory AB and BA) |
| Pilot gate | reject any judge with `|P(prefer position_a) − 0.5| > 0.2` on a 30-pair pilot |
| Dedup key (pairwise) | `(pair_id, judge, run, position)` |
| Dedup key (per-rubric) | `(plan_id, judge)` |
| `max_model_len` cap | 8192 tokens for all judges |

### Judge panel

| Judge name | Model | Quant | Disk (GiB) | Active? |
|---|---|---|---:|---|
| `llama_8b_judge` | meta-llama/Llama-3.1-8B-Instruct | FP16 | 15 | ✓ |
| `qwen_7b_judge` | Qwen/Qwen2.5-7B-Instruct | FP16 | 15 | ✓ |
| `qwen_14b_judge` | Qwen/Qwen2.5-14B-Instruct-AWQ | AWQ-INT4 | 8 | ✓ |
| `qwen_32b_judge` | Qwen/Qwen2.5-32B-Instruct-AWQ | AWQ-INT4 | 18 | ✓ |
| `mistral_7b_judge` | mistralai/Mistral-7B-Instruct-v0.3 | FP16 | 14 | reserved (not in active set) |

The Mistral judge remains in `judge.panel.PANEL` for completeness but is **not**
included in `ACTIVE_JUDGE_NAMES` and is therefore excluded from H1/H2/H3/H4
analyses unless explicitly added before generation begins.

---

## Statistical model specifications (frozen)

### H1 model

```
prefers_llm ~ 1 + (1 | pair_id) + (1 | judge) + fixture_id
```

- Link: logistic (binomial family)
- Random intercepts for pair and judge
- Fixture as fixed effect
- Cluster-robust SEs at the pair level
- Implementation: `statsmodels.formula.mixedlm` or equivalent

### H3 model

```
prefers_llm ~ same_family + (1 | judge) + fixture_id
```

- `same_family`: 1 if judge model family == source model family
- Random intercept for judge
- Implementation: `analyze.models.fit_h3_model`

### H4 model

```
prefers_llm ~ log(param_count)   [OLS, Qwen judges only]
```

- Three-point ladder: 7B, 14B, 32B
- Restricted to Qwen-sourced plans
- Implementation: `analyze.models.fit_h4_model`

### H2 model

- Per-rubric paired t-test (LLM rubric score − programmatic rubric score)
- Holm correction across 5 rubrics
- Implementation: `analyze.rubric_deltas.rubric_paired_contrasts`

---

## Exclusion criteria (pre-registered)

1. **Schema-invalid responses:** records in `schema_failures.jsonl` are excluded
   from all analyses. They are retained as data on judge reliability. This now
   explicitly includes `compare_plans` rows where the upstream JSON-parse
   sentinel `"Could not parse comparison response."` is detected by the harness.
2. **Oversized prompts:** any pair where `len(prompt) > 8192 tokens` after
   context assembly is excluded. Logged in `data_notes`.
3. **Position-biased judges:** any judge failing the pilot gate
   (`|P(prefer_position_a) − 0.5| > 0.2`) is excluded from H1/H2 analysis but
   retained in H3/H4 with a bias-correction covariate.
4. **Unmatched plans:** plans that did not receive a matching partner within
   the score tolerance are excluded from the pairwise analysis but retained for
   the per-rubric analysis.
5. **Explainer mismatch:** any plan whose provenance sidecar shows
   `explainer_model ≠ EXPLAINER_MODEL_ID` is excluded from all arms. This
   would indicate a violation of the study invariant.

---

## Pre-registration timestamp

This file must be committed to version control **before** the first `sbatch`
generation job is submitted. The commit hash serves as the pre-registration
timestamp.

```
# Command to verify pre-registration commit predates first generation job:
git log --oneline PREREGISTRATION.md
git log --oneline --grep="generate" | head -5
```
