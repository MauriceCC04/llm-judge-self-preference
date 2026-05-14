# Preregistration and Final Analysis Plan

This document records the preregistered/intended design and the final completed analysis state for the Qwen/Gemma LLM judge self-preference study. Because the study is now complete, this file also records deviations, post hoc sensitivity analyses, and final limitations.

## Research question

Controlling for deterministic structural training-plan quality, do LLM judges systematically prefer LLM-generated training plans over programmatically generated plans, and does this effect vary by judge model family, model size, explicit evaluation marker, or whether the source plan family matches the judge family?

## Hypotheses

### H1: Overall LLM preference

LLM judges may prefer LLM-generated plans over structurally matched programmatic plans.

**Final result:** not supported. In the primary overall pairwise evaluation, LLM-source plans won 36.53% of judgments.

### H2: Judge-family heterogeneity

Preference may vary by judge model family and model size.

**Final result:** supported as heterogeneity, but not as an LLM-preference effect. Primary row-level LLM win rates varied by judge from 29.60% to 41.68%. Explicit marker-level results also varied strongly by judge and marker.

### H3: Self-family preference

Judges may prefer plans produced by their own model family.

**Final result:** not supported. Marker-level self-family rows were not more favorable to LLM-source plans overall. Crossed judge-family/source-family results also did not show a simple own-family preference pattern.

### H4: Marker-specific effects

LLM-source plans may win on presentation or explanation markers while losing on structural training markers.

**Final result:** not supported in the completed explicit marker pass. Programmatic plans outperformed LLM-source plans on every explicit marker, including `explanation_quality`.

## Final study design

### Source corpora

| Source | Count |
|---|---:|
| Qwen LLM plans | 192 |
| Gemma LLM plans | 192 |
| Programmatic plans | 640 |
| Total matching pool | 1,024 |

Source models:

- `Qwen/Qwen2.5-7B-Instruct`
- `google/gemma-3-4b-it`

No Llama source family is part of the completed study.

### Matching

Plans were matched using source-neutral structural scoring rather than the older TrailTraining quality score as the primary matching criterion. Structural matching excludes presentation/prose quality as a matching criterion.

Final matched set:

- 250 matched LLM-vs-programmatic pairs
- Same/equivalent fixture constraints preserved
- Qwen-source pairs: 134
- Gemma-source pairs: 116
- Mean structural score gap: 0.3076
- P95 structural score gap: 2.0
- Max structural score gap: 2.0

### Primary pairwise evaluation

Primary design:

```text
250 matched pairs x 4 judges x 5 repeated runs x 2 AB/BA positions = 10,000 pairwise judgment records
```

Judges:

- `qwen_7b_judge`
- `qwen_14b_judge`
- `gemma_4b_judge`
- `gemma_12b_judge`

Judge-facing artifacts used `canonical_masked_scrubbed_v1` and excluded explicit model/source/provenance metadata.

### Explicit marker evaluation

Secondary explicit marker design:

```text
10,000 marker records x 9 markers = 90,000 marker decisions
```

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

Each marker has a categorical preference (`plan_a`, `plan_b`, or `tie`) and numeric 1-5 scores for each plan.

## Primary analysis endpoints

### Overall pairwise endpoint

Outcome: whether the judge selected the LLM-source plan.

Primary summary:

- Overall LLM win rate
- Per-judge LLM win rate
- Judge-family and source-family breakdowns
- Self-family match breakdown
- AB/BA order-bias audit
- Pair-run consistency:
  - `source_consistent_llm`
  - `source_consistent_programmatic`
  - `position_consistent_plan_a`
  - `position_consistent_plan_b`

### Marker-level endpoint

Outcome: whether the judge selected the LLM-source plan for each marker.

Primary marker summaries:

- LLM marker win rate excluding ties
- LLM marker win rate with ties counted as half
- Tie rate
- Marker x judge
- Marker x judge family
- Marker x source family
- Marker x judge family x source family
- Marker x self-family match
- Marker-level AB/BA consistency

## Final primary results

Overall pairwise:

| Outcome | Count | Rate |
|---|---:|---:|
| LLM wins | 3,653 / 10,000 | 36.53% |
| Programmatic wins | 6,347 / 10,000 | 63.47% |

Per judge:

| Judge | LLM win rate |
|---|---:|
| `gemma_12b_judge` | 29.60% |
| `gemma_4b_judge` | 33.44% |
| `qwen_14b_judge` | 41.40% |
| `qwen_7b_judge` | 41.68% |

Pair-run consistency:

| Category | Count | Rate |
|---|---:|---:|
| `source_consistent_programmatic` | 1,973 / 5,000 | 39.46% |
| `source_consistent_llm` | 626 / 5,000 | 12.52% |
| `position_consistent_plan_a` | 2,251 / 5,000 | 45.02% |
| `position_consistent_plan_b` | 150 / 5,000 | 3.00% |

## Final marker results

Marker-level LLM win rates excluding ties:

| Marker | LLM win rate excluding ties |
|---|---:|
| `training_specificity` | 48.54% |
| `clarity_actionability` | 40.62% |
| `readiness_alignment` | 39.84% |
| `recovery_safety` | 39.07% |
| `plan_coherence` | 36.00% |
| `load_progression` | 35.11% |
| `quality_session_design` | 33.44% |
| `endurance_development` | 32.81% |
| `explanation_quality` | 32.21% |

Programmatic plans outperformed LLM-source plans on every explicit marker.

Marker-level pair-run consistency across all 45,000 marker pair-run units:

| Category | Count | Rate |
|---|---:|---:|
| `source_consistent_programmatic` | 10,800 | 24.00% |
| `source_consistent_llm` | 3,262 | 7.25% |
| `position_consistent_plan_a` | 12,233 | 27.18% |
| `position_consistent_plan_b` | 4,879 | 10.84% |
| `consistent_tie` | 2,725 | 6.06% |
| `other_inconsistent` | 11,101 | 24.67% |

## Leakage-filtered sensitivity analysis

A post hoc audit found source-asymmetric presentation artifacts in some LLM-source judge-facing plans. Examples include `TrailRun |`, `avgHR`, `km`, elevation strings, ellipses, and duration-text inconsistencies. These artifacts were absent from programmatic plans in the scan.

To avoid overclaiming, a leakage-filtered sensitivity dataset was constructed by excluding contaminated pairs.

- Clean pairs retained: 103
- Flagged pairs excluded: 147
- Clean primary records: 4,120
- Clean marker decisions: 37,080

Clean-pair primary result:

- LLM wins: 1,419 / 4,120 = 34.44%
- Programmatic wins: 2,701 / 4,120 = 65.56%

Clean-pair marker results also favored programmatic plans on every marker.

This analysis is post hoc and exclusion-based. It is not a scrubbed-v2 rerun.

## Deviations and post hoc additions

1. The original deterministic TrailTraining quality score was not used as the primary matching score because diagnostics showed it was unsuitable for source-neutral structural matching.
2. A source-neutral structural score was introduced for matching.
3. Explicit marker-level evaluation was added after the primary pairwise run to support marker-specific conclusions.
4. A post hoc leakage audit identified source-asymmetric presentation artifacts.
5. A leakage-filtered sensitivity dataset was created by excluding flagged pairs.
6. The marker runner was patched to repair a small number of blank `preferred` fields and to enforce compact JSON for Gemma retries.

## Validity considerations

The study supports strong descriptive conclusions for the frozen Qwen/Gemma dataset, but the following limitations must be emphasized:

- Equal structural score is not full equality between plans.
- Source masking removed explicit metadata but did not fully remove all presentation-level artifacts.
- Large AB/BA position bias is present in both primary and marker-level evaluations.
- Programmatic plans may contain LLM-written explanation fields, which complicates interpretation of `explanation_quality`.
- The leakage-filtered clean subset is smaller and less source-family-balanced than the full dataset.
- Self-family inference is limited to Qwen and Gemma families represented in both source and judge sets.
- Results are specific to the training-plan domain and the available local judge models.

## Final interpretation

The completed study does not support the hypothesis that Qwen/Gemma LLM judges systematically prefer LLM-source plans over structurally matched programmatic plans. Programmatic plans win more often overall, win more often on every explicit marker, and remain favored in leakage-filtered sensitivity analyses. The study also does not support a robust self-family preference effect. The report should frame position bias and source-asymmetric presentation leakage as central threats to validity that were diagnosed and partially addressed through AB/BA analysis and leakage-filtered sensitivity analysis.
