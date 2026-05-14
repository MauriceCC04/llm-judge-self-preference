# Final Results Summary

This summary describes the completed Qwen/Gemma study as of the final frozen primary and marker-level artifacts.

## Dataset integrity

### Primary overall pairwise evaluation

- Records: 10,000
- Unique record IDs: 10,000
- Judges: 2,500 each
- Orders: 5,000 AB / 5,000 BA
- Runs: 2,000 each for indices 0-4

### Explicit marker-level evaluation

- Marker records: 10,000
- Explicit marker decisions: 90,000
- Markers per record: 9
- Bad marker fields: 0
- Wrong marker-count records: 0
- Normalization warnings: 31 / 90,000 = 0.034%

## Primary overall result

| Outcome | Count | Rate |
|---|---:|---:|
| LLM-source plan preferred | 3,653 | 36.53% |
| Programmatic plan preferred | 6,347 | 63.47% |

The main primary result is that programmatic plans were preferred more often than LLM-source plans.

## Primary result by judge

| Judge | LLM win rate |
|---|---:|
| `gemma_12b_judge` | 29.60% |
| `gemma_4b_judge` | 33.44% |
| `qwen_14b_judge` | 41.40% |
| `qwen_7b_judge` | 41.68% |

## Primary result by family/source

| Group | LLM win rate |
|---|---:|
| Gemma judges | 31.52% |
| Qwen judges | 41.54% |
| Gemma-source LLM plans | 38.64% |
| Qwen-source LLM plans | 34.70% |
| Self-family false | 36.34% |
| Self-family true | 36.72% |

The row-level self-family difference is negligible and does not support a strong self-preference claim.

## Order bias

| Order | LLM win rate |
|---|---:|
| AB, LLM as Plan A | 57.54% |
| BA, LLM as Plan B | 15.52% |

The order effect is large and must be treated as a central result/threat to validity.

## Primary pair-run consistency

One pair-run unit is `pair_id x judge x run`.

| Category | Count | Rate |
|---|---:|---:|
| `source_consistent_programmatic` | 1,973 | 39.46% |
| `source_consistent_llm` | 626 | 12.52% |
| `position_consistent_plan_a` | 2,251 | 45.02% |
| `position_consistent_plan_b` | 150 | 3.00% |

Source-consistent programmatic preferences are much more common than source-consistent LLM preferences, but position-consistent behavior is also very common.

## Explicit marker-level result

LLM marker win rates excluding ties:

| Marker | LLM wins | Programmatic wins | Ties | LLM win rate excluding ties |
|---|---:|---:|---:|---:|
| `training_specificity` | 4,368 | 4,630 | 1,002 | 48.54% |
| `clarity_actionability` | 2,710 | 3,962 | 3,328 | 40.62% |
| `readiness_alignment` | 2,863 | 4,324 | 2,813 | 39.84% |
| `recovery_safety` | 3,103 | 4,839 | 2,058 | 39.07% |
| `plan_coherence` | 2,835 | 5,040 | 2,125 | 36.00% |
| `load_progression` | 3,218 | 5,947 | 835 | 35.11% |
| `quality_session_design` | 2,783 | 5,539 | 1,678 | 33.44% |
| `endurance_development` | 3,046 | 6,239 | 715 | 32.81% |
| `explanation_quality` | 2,578 | 5,425 | 1,997 | 32.21% |

Programmatic plans outperform LLM-source plans on every explicit marker.

## Marker-level pair-run consistency

Across 45,000 marker pair-run units:

| Category | Count | Rate |
|---|---:|---:|
| `source_consistent_programmatic` | 10,800 | 24.00% |
| `source_consistent_llm` | 3,262 | 7.25% |
| `position_consistent_plan_a` | 12,233 | 27.18% |
| `position_consistent_plan_b` | 4,879 | 10.84% |
| `consistent_tie` | 2,725 | 6.06% |
| `other_inconsistent` | 11,101 | 24.67% |

## Leakage-filtered sensitivity result

Post hoc leakage audit:

- 147 of 250 matched pairs flagged for LLM-side presentation artifacts
- 103 clean pairs retained
- 0 programmatic-side presentation artifact flags in the scan

Clean-pair primary result:

| Split | Records | LLM wins | Programmatic wins | LLM win rate |
|---|---:|---:|---:|---:|
| Clean pairs | 4,120 | 1,419 | 2,701 | 34.44% |
| Flagged pairs | 5,880 | 2,234 | 3,646 | 37.99% |

Clean-pair marker results excluding ties:

| Marker | LLM win rate excluding ties |
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

The clean-pair sensitivity results do not reverse the conclusion.

## Concise final conclusion

The completed Qwen/Gemma study does not find evidence that LLM judges systematically prefer LLM-source training plans over structurally matched programmatic plans. Programmatic plans win more often overall, win more often on every explicit marker, and remain favored in leakage-filtered sensitivity analyses. The strongest caveats are large position bias and post hoc presentation leakage, both of which should be central in the report.
