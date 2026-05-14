# Leakage-Filtered Sensitivity Analysis

This document describes the post hoc leakage audit and the leakage-filtered sensitivity dataset.

## Why this analysis exists

After completing the primary and marker-level evaluations, a post hoc audit found that source masking removed explicit metadata but did not fully remove presentation-level artifacts. Some LLM-source judge-facing plans contained activity-log-like fragments and other presentation cues not found in programmatic plans.

Examples of flagged artifacts:

```text
TrailRun |
avgHR
km
m+ / elevation / vertical
ellipsis characters
duration-text inconsistencies
```

These cues could influence judge preferences or indirectly reveal the LLM-source arm even when explicit metadata is hidden.

## Important interpretation boundary

The leakage-filtered results are **not** a scrubbed-v2 rerun. Judges in the original evaluation saw the original `canonical_masked_scrubbed_v1` prompts. The leakage-filtered analysis is an exclusion-based sensitivity analysis that removes contaminated matched pairs from analysis after the fact.

Use this wording:

```text
We constructed a leakage-filtered sensitivity dataset by excluding matched pairs whose judge-facing inputs contained source-asymmetric presentation artifacts. This tests whether the main conclusions are robust to excluding contaminated pairs, but it does not replace a true scrubbed-v2 rerun.
```

## Final leakage split

- Total matched pairs: 250
- Clean pairs retained: 103
- Flagged pairs excluded: 147
- Primary clean records: 4,120
- Primary flagged records: 5,880
- Marker clean records: 4,120
- Marker flagged records: 5,880
- Clean marker decisions: 37,080

## Flag counts

| Flag | Pairs |
|---|---:|
| Any presentation artifact | 147 |
| Any LLM-side artifact | 147 |
| Any programmatic-side artifact | 0 |
| LLM ellipsis | 109 |
| LLM activity-like fragment | 91 |
| LLM `km` unit | 91 |
| LLM `TrailRun` pipe | 56 |
| LLM `avgHR` | 51 |
| LLM duration-text inconsistency | 40 |
| LLM elevation/vertical | 7 |

## Clean-pair primary results

| Split | Records | LLM wins | Programmatic wins | LLM win rate |
|---|---:|---:|---:|---:|
| Clean pairs | 4,120 | 1,419 | 2,701 | 34.44% |
| Flagged pairs | 5,880 | 2,234 | 3,646 | 37.99% |

The main conclusion does not reverse after excluding flagged pairs. The clean subset is slightly less favorable to LLM-source plans than the flagged subset.

## Clean-pair primary by judge

| Judge | Clean LLM win rate |
|---|---:|
| `gemma_12b_judge` | 27.67% |
| `gemma_4b_judge` | 32.04% |
| `qwen_7b_judge` | 34.37% |
| `qwen_14b_judge` | 43.69% |

## Clean-pair pair-run consistency

| Category | Count | Rate |
|---|---:|---:|
| `source_consistent_programmatic` | 875 | 42.48% |
| `source_consistent_llm` | 234 | 11.36% |
| `position_consistent_plan_a` | 861 | 41.80% |
| `position_consistent_plan_b` | 90 | 4.37% |

Source-consistent programmatic preferences remain far more common than source-consistent LLM preferences in the clean subset.

## Clean-pair marker results

LLM marker win rate excluding ties:

| Marker | Clean-pair LLM win rate |
|---|---:|
| `training_specificity` | 48.19% |
| `readiness_alignment` | 41.45% |
| `clarity_actionability` | 40.06% |
| `recovery_safety` | 39.82% |
| `load_progression` | 37.48% |
| `plan_coherence` | 36.57% |
| `quality_session_design` | 34.03% |
| `explanation_quality` | 32.30% |
| `endurance_development` | 29.39% |

Programmatic plans still outperform LLM-source plans on every explicit marker in the clean-pair sensitivity subset.

## Caveats

- The clean subset is smaller than the full dataset: 103 pairs rather than 250.
- The clean subset is less source-family-balanced than the full dataset.
- Exclusion changes the estimand: it asks about artifact-free pairs, not the whole generated population.
- A true scrubbed-v2 rerun would be stronger, but was not completed for the current report.

## Report-ready wording

```text
A post hoc leakage audit found source-asymmetric presentation artifacts in 147 of 250 matched pairs. We therefore constructed a leakage-filtered sensitivity dataset by excluding any pair whose judge-facing inputs contained these artifacts. This retained 103 matched pairs, corresponding to 4,120 primary judgments and 37,080 marker decisions. The qualitative findings were unchanged: programmatic plans continued to outperform LLM-source plans overall and across all explicit markers. These filtered results should be interpreted as an exclusion-based sensitivity analysis, not as evidence from a rerun on newly scrubbed prompts.
```
