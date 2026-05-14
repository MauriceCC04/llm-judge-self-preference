# Explicit Marker-Level Evaluation

This document describes the completed explicit marker-level evaluation.

## Purpose

The primary pairwise judgment records contain overall preferences but do not by themselves provide explicit rubric/marker-level preferences. A secondary marker-level pass was added to answer whether judges favored LLM-source or programmatic plans on specific criteria such as recovery safety, training specificity, or explanation quality.

## Design

The marker-level evaluation uses the same frozen primary manifest and judge-facing inputs as the primary pairwise study:

```text
artifacts/gen_src_t070_exp_t000/frozen_primary_v1/eval_manifest/pairwise_eval_manifest.jsonl
artifacts/gen_src_t070_exp_t000/frozen_primary_v1/eval_manifest/judge_inputs/
```

Completed design:

```text
250 pairs x 4 judges x 5 runs x 2 orders = 10,000 marker records
10,000 marker records x 9 markers = 90,000 marker decisions
```

## Markers

| Marker | Meaning |
|---|---|
| `plan_coherence` | Whether the week fits together coherently with compatible sessions and sensible sequencing |
| `training_specificity` | Specificity to athlete, trail-running context, terrain, hills, race phase, and endurance demands |
| `load_progression` | Appropriateness of total volume, duration, stress distribution, and load progression |
| `recovery_safety` | Rest-day placement, hard-day spacing, injury-risk control, and safety |
| `quality_session_design` | Purposefulness and feasibility of intervals, tempo, hills, progression runs, and long runs |
| `endurance_development` | Support for aerobic/endurance development, easy volume, and long-run stimulus |
| `readiness_alignment` | Respect for readiness, recovery capacity, and fatigue constraints |
| `clarity_actionability` | Whether the athlete can understand and execute the plan |
| `explanation_quality` | Helpfulness and specificity of plan rationale; presentation-sensitive |

## Output schema

Each marker record has:

```json
{
  "record_id": "...",
  "pair_id": "...",
  "judge": "...",
  "order": "AB or BA",
  "run": 0,
  "markers": {
    "recovery_safety": {
      "preferred": "plan_a | plan_b | tie",
      "winner_role": "llm | programmatic | tie",
      "llm_won": true,
      "plan_a_score": 4,
      "plan_b_score": 3,
      "confidence": 0.8,
      "rationale": "..."
    }
  }
}
```

## Final output files

```text
artifacts/gen_src_t070_exp_t000/frozen_primary_v1/marker_eval_t000_scrubbed_v1_staged/marker_gemma_4b_judge_canonical_masked_scrubbed_v1_t000.jsonl
artifacts/gen_src_t070_exp_t000/frozen_primary_v1/marker_eval_t000_scrubbed_v1_staged/marker_gemma_12b_judge_canonical_masked_scrubbed_v1_t000.jsonl
artifacts/gen_src_t070_exp_t000/frozen_primary_v1/marker_eval_t000_scrubbed_v1_staged/marker_qwen_7b_judge_canonical_masked_scrubbed_v1_t000.jsonl
artifacts/gen_src_t070_exp_t000/frozen_primary_v1/marker_eval_t000_scrubbed_v1_staged/marker_qwen_14b_judge_canonical_masked_scrubbed_v1_t000.jsonl
artifacts/gen_src_t070_exp_t000/frozen_primary_v1/marker_eval_t000_scrubbed_v1_staged/marker_all_judges_canonical_masked_scrubbed_v1_t000.jsonl
```

## Integrity results

- 10,000 marker rows
- 90,000 marker decisions
- 2,500 rows per judge
- 5,000 AB / 5,000 BA
- 2,000 per run index 0-4
- 9 markers per record
- 0 bad marker fields
- 0 records with wrong marker count
- 31 repaired `preferred` fields, all flagged
- 26 categorical preference / score-direction mismatches

## Final marker results

LLM win rate excluding ties:

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

## Interpretation

Programmatic plans outperform LLM-source plans on every explicit marker. This includes `explanation_quality`, which means the marker-level evaluation does not support a story where LLM-source plans lose structurally but win on explanation/presentation.

## Position bias

Position effects remain large at the marker level. Row-level marker rates should always be paired with marker-level pair-run consistency.

Across all marker pair-run units:

| Category | Count | Rate |
|---|---:|---:|
| `source_consistent_programmatic` | 10,800 | 24.00% |
| `source_consistent_llm` | 3,262 | 7.25% |
| `position_consistent_plan_a` | 12,233 | 27.18% |
| `position_consistent_plan_b` | 4,879 | 10.84% |
| `consistent_tie` | 2,725 | 6.06% |
| `other_inconsistent` | 11,101 | 24.67% |

## Runner notes

The marker runner was patched during execution to handle model-specific issues:

- Qwen 7B blank `preferred` fields repaired from numeric scores.
- Gemma JSON decode failures recovered by compact JSON prompt and `response_format`.
- All failed rows were rerun from missing-record manifests.

These operational patches are reflected in the final data via warnings and provenance files.
