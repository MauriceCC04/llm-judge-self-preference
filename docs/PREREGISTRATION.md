# Preregistration / Frozen Analysis Plan

This document records the frozen design and analysis plan for the Qwen/Gemma training-plan judge-bias study. Because primary results now exist, this should be treated as the frozen design and analysis protocol for the primary experiment plus a transparent post-freeze record of the secondary marker-level extension.

## Research question

Controlling for deterministic structural plan quality, do LLM judges systematically favor LLM-generated training plans over programmatically generated plans, and does this effect vary by judge model family, model size, rubric/marker, or whether the plan matches the judge's own family?

## Hypotheses

Primary hypotheses:

1. LLM judges may prefer LLM-source plans over structurally matched programmatic plans.
2. Preference may vary by judge model family and model size.
3. Self-family preference may occur when the judge family matches the LLM source family.
4. Presentation-sensitive criteria may show different effects than structural training-plan criteria.

Null expectations:

1. After structural matching and source masking, LLM-source and programmatic plans may be preferred at equal rates.
2. Self-family match may not affect preferences once source labels are masked.
3. Apparent preference may be driven by order/position effects rather than source preference.

## Frozen design

Source plans:

- `Qwen/Qwen2.5-7B-Instruct`, 192 plans.
- `google/gemma-3-4b-it`, 192 plans.
- Programmatic controls, 640 plans.

Matching pool:

```text
artifacts/gen_src_t070_exp_t000/matching_pool/plans
```

Frozen matched set:

```text
artifacts/gen_src_t070_exp_t000/frozen_primary_v1/matched_pairs.json
```

Matched-pair target and achieved count:

```text
250 structurally matched LLM-vs-programmatic pairs
```

Judges:

- `qwen_7b_judge`
- `qwen_14b_judge`
- `gemma_4b_judge`
- `gemma_12b_judge`

Primary evaluation:

```text
250 matched pairs × 4 judges × 5 repeated runs × 2 AB/BA orders = 10,000 pairwise judgments
```

Primary view:

```text
canonical_masked_scrubbed_v1
```

Primary temperature:

```text
eval_t000
```

## Structural matching principle

Matching uses source-neutral structural quality rather than presentation-rich TrailTraining quality scores. This is necessary because equal deterministic score is not equivalent to equal plan quality if the score rewards explanation richness, title wording, citations, or prose style.

Structural matching should account for:

- fixture/cell identity
- athlete band
- readiness
- recovery capability
- race phase
- plan days
- total duration
- active/rest day counts
- hard-day count
- quality-day count
- long-run count
- hard/rest spacing
- impossible durations and contradictions

Structural matching must exclude:

- title wording
- workout prose richness
- purpose prose richness
- rationale length
- citations
- claim attributions
- data notes
- source model names
- generation-arm labels
- file naming artifacts

Frozen matching achieved:

```text
matched pairs: 250
mean structural score gap: 0.3076
p95 structural score gap: 2.0
max structural score gap: 2.0
coverage_ok: true
```

## Source masking

Judge-facing artifacts must not expose:

- source family
- model name
- arm label
- provenance metadata
- file path or file name
- fixture ID
- explicit `llm` or `programmatic` labels

Metadata required for analysis is preserved outside the judge prompt.

## Primary outcome

Primary binary outcome:

```text
llm_won = true if the judge chooses the LLM-source plan over the matched programmatic plan
```

Primary estimands:

- overall LLM win rate
- per-judge LLM win rate
- judge-family LLM win rate
- source-family LLM win rate
- self-family LLM win rate where source and judge families overlap
- order/position effect
- AB/BA pair-run source consistency

## Primary analysis plan

Report:

1. Raw row-level LLM win rate.
2. Per-judge and per-family win rates.
3. AB vs BA order rates.
4. Pair-run AB/BA consistency:
   - `source_consistent_llm`
   - `source_consistent_programmatic`
   - `position_consistent_plan_a`
   - `position_consistent_plan_b`
5. Bootstrap confidence intervals clustered by matched pair.
6. Sensitivity by source family, judge family, band, fixture/cell, and structural-score gap.

Order effects are central. If AB and BA disagree, interpretation should privilege pair-run consistency over row-level rates.

## Secondary explicit marker-level extension

After the primary pairwise study, an explicit marker-level evaluation was added to support marker-specific conclusions.

Markers:

- `plan_coherence`
- `training_specificity`
- `load_progression`
- `recovery_safety`
- `quality_session_design`
- `endurance_development`
- `readiness_alignment`
- `clarity_actionability`
- `explanation_quality`

Each marker response records:

- `preferred`: `plan_a`, `plan_b`, or `tie`
- `plan_a_score`: integer 1-5
- `plan_b_score`: integer 1-5
- `confidence`: 0.0-1.0
- `rationale`

Marker-level analysis should report:

1. marker-level LLM win rates, ties included and ties excluded
2. marker × judge
3. marker × judge family
4. marker × source family
5. marker × self-family match
6. marker × order
7. marker-level AB/BA pair-run consistency
8. repaired marker-field counts

Current uploaded marker status is partial:

```text
qwen_14b_judge: complete
qwen_7b_judge: complete
gemma_12b_judge: pending/missing
gemma_4b_judge: pending/missing
```

Therefore marker-level claims must currently be labeled Qwen-only until Gemma marker passes are complete.

## Current primary results to disclose

Primary pairwise results are complete:

```text
LLM wins: 3,653 / 10,000 = 36.53%
Programmatic wins: 6,347 / 10,000 = 63.47%
```

Per judge:

```text
gemma_12b_judge: 29.60% LLM win rate
gemma_4b_judge:  33.44% LLM win rate
qwen_14b_judge:  41.40% LLM win rate
qwen_7b_judge:   41.68% LLM win rate
```

Order effect:

```text
AB, LLM as Plan A: 57.54% LLM win rate
BA, LLM as Plan B: 15.52% LLM win rate
```

Pair-run consistency:

```text
source_consistent_programmatic: 1,973 / 5,000 = 39.46%
source_consistent_llm:           626 / 5,000 = 12.52%
position_consistent_plan_a:     2,251 / 5,000 = 45.02%
position_consistent_plan_b:       150 / 5,000 = 3.00%
```

Interpretation:

- The primary result does not support broad LLM-over-programmatic preference.
- Programmatic plans are preferred more often overall and source-consistently.
- Position bias is large and must be modeled and discussed.

## Current marker results to disclose cautiously

Qwen-only explicit marker results show programmatic plans favored over LLM-source plans on all nine markers. Qwen-only marker self-family evidence is weak and inconsistent.

Do not claim full four-judge marker-level results until Gemma marker files are complete.

## Threats to validity

- Equal structural score is not full plan equality.
- Structural matching cannot remove all surface/style confounds.
- Programmatic plans may include LLM-written explanation fields.
- Source masking may not eliminate all stylistic leakage.
- Strong position effects complicate row-level interpretation.
- Local judge models may not generalize to external judges.
- Marker-level Qwen-only results are partial until Gemma marker runs are complete.
- Some marker responses required deterministic repair of blank `preferred` values from scores.
- Local HPC quota constraints affected caching and model ordering, though not the frozen manifest.

## Confirmatory vs exploratory claims

Confirmatory for the primary pairwise experiment:

- full 10,000-record overall LLM-vs-programmatic preference
- per-judge primary win rates
- order-bias audit
- AB/BA pair-run consistency

Secondary/extension:

- explicit marker-level judgments

Exploratory:

- keyword-coded rationale marker analysis from primary free-text rationales
- any analysis not pre-specified above
