# Data Processing Log

This document records the current data-processing state for the `llm-judge-self-preference` project.

## Study question

The study asks whether LLM judges systematically prefer LLM-generated trail-training plans over programmatically generated plans when deterministic structural plan quality is controlled.

The current frozen study is Qwen/Gemma only:

- Source families: Qwen and Gemma.
- Judge families: Qwen and Gemma.
- No Llama source or judge is included in the current frozen study.

## Data generation corpora

Final corpora used for matching:

| Arm | Path | Plans | Notes |
|---|---|---:|---|
| Qwen LLM | `artifacts/gen_src_t070_exp_t000/full_qwen/plans` | 192 | Balanced across 32 cells |
| Gemma LLM | `artifacts/gen_src_t070_exp_t000/full_gemma3/plans` | 192 | Balanced across 32 cells |
| Programmatic | `artifacts/gen_src_t070_exp_t000/full_programmatic/plans` | 640 | 20 per cell across 32 cells |
| Matching pool | `artifacts/gen_src_t070_exp_t000/matching_pool/plans` | 1,024 | 192 + 192 + 640 |

Original generation audits reported the intended cell/band balance:

```text
Qwen:
  plans: 192
  cells: 32
  by_band: A1=48, A2=48, A3=48, A4=48

Gemma:
  plans: 192
  cells: 32
  by_band: A1=48, A2=48, A3=48, A4=48

Programmatic:
  plans: 640
  cells: 32
  by_band: A1=160, A2=160, A3=160, A4=160
```

## Original matching blocker

The initial matching run used a deterministic score that did not produce enough overlap between LLM-source and programmatic plans.

Observed initial score distributions:

```text
Qwen: mostly 97 and 100
Gemma: mostly 85, 97, and 100
Programmatic: mostly around 30, with only a few 97/100
```

The old matcher produced only about 30 pairs under strict matching. This was not adequate for the full study. Relaxing tolerances did not solve the issue because the primary problem was score non-overlap.

Interpretation: the original deterministic score appeared unsuitable as the primary matching score because it may have included presentation/explanation quality instead of only structural plan quality.

## Structural matching repair

The matching strategy was repaired by using a source-neutral structural score and a canonical masked judge view. The matching score excludes source-identifying/presentation fields such as title wording, prose richness, explanation length, citations, data notes, source model, and generation arm.

## Judge-facing JSON masking and field-removal policy

The raw plan JSON files and provenance sidecars were not sent directly to judge models. Pairwise evaluation used derived judge-facing JSON files under:

`artifacts/gen_src_t070_exp_t000/frozen_primary_v1/eval_manifest/judge_inputs/`

These files present plans only as neutral `plan_a` and `plan_b` objects. Source-identifying and analysis-only metadata were excluded from the judge-facing payload.

Excluded from judge-facing plan payloads:

- source model names, including Qwen and Gemma identifiers
- generation arm labels such as `llm`, `programmatic`, `prog`, or source-family labels
- provenance sidecars and provenance fields
- original file names and file paths
- plan IDs containing source/model tokens
- fixture IDs and cell IDs when they could reveal generation or matching structure
- match metadata such as structural score, score gap, feature gaps, and pair ID
- judge model identity
- source-family and self-family labels
- run index and evaluation metadata
- other fields used only for analysis rather than plan interpretation

Retained outside the judge prompt, in manifest and judgment-output metadata:

- `record_id`
- `pair_id`
- `order`
- `run`
- `left_plan_role`
- `right_plan_role`
- `llm_plan_id`
- `programmatic_plan_id`
- `source_model_family`
- `judge_model_family`
- `self_family_match`
- `fixture_id`
- `structural_score_gap`
- `feature_gaps`
- `canonical_masking_version`

This separation is central to the design. Judges see source-masked plan content, while the analysis pipeline retains source metadata externally to estimate LLM-vs-programmatic preference, judge-family effects, source-family effects, self-family effects, and order bias.

The scrubbed manifest/audit should be treated as the source of truth for the judge-facing view. The raw plan/provenance files remain preserved for reproducibility and auditing but are not the prompt payload used in pairwise judging.

Final matching state:

```text
matched pairs: 250
coverage_ok: true
coverage_ratio: 1.0
pairs_by_source_family:
  qwen: 134
  gemma: 116
pairs_by_athlete_band:
  A1: 60
  A2: 55
  A3: 74
  A4: 61
mean_structural_score_gap: 0.3076
p95_structural_score_gap: 2.0
max_structural_score_gap: 2.0
```

Final prefilter audit:

```text
n_input: 1024
n_kept: 1023
n_dropped: 1
drop_reasons:
  impossible_duration: 1
```

## Frozen primary inputs

Frozen primary directory:

```text
artifacts/gen_src_t070_exp_t000/frozen_primary_v1/
```

Important frozen files:

```text
matched_pairs.json
matching_audit.json
matching_prefilter_audit.json
eval_manifest/pairwise_eval_manifest.jsonl
eval_manifest/judge_inputs/*.json
```

The frozen eval manifest contains:

```text
records: 10,000
unique record_ids: 10,000
pairs: 250
judges:
  qwen_7b_judge: 2,500
  qwen_14b_judge: 2,500
  gemma_4b_judge: 2,500
  gemma_12b_judge: 2,500
orders:
  AB: 5,000
  BA: 5,000
runs:
  0: 2,000
  1: 2,000
  2: 2,000
  3: 2,000
  4: 2,000
```

The scrubbed judge inputs passed masking checks. A later audit found:

```text
judge input files: 10,000
parse errors: 0
schema-bad count: 0
source/model-family leakage terms: 0 occurrences
```

## Primary pairwise judging execution

Primary pairwise judging was run on HPC in staged model-cache order due to storage limits.

Storage constraint:

```text
home quota: 50 GB
not enough space to keep all judge models cached simultaneously
```

Staged execution order included:

1. `gemma_4b_judge`
2. `gemma_12b_judge`
3. `qwen_7b_judge`
4. `qwen_14b_judge`

Final primary output directory:

```text
artifacts/gen_src_t070_exp_t000/frozen_primary_v1/judgments_eval_t000_scrubbed_v1_staged/
```

Final files:

```text
pairwise_gemma_4b_judge_canonical_masked_scrubbed_v1_t000.jsonl
pairwise_gemma_12b_judge_canonical_masked_scrubbed_v1_t000.jsonl
pairwise_qwen_7b_judge_canonical_masked_scrubbed_v1_t000.jsonl
pairwise_qwen_14b_judge_canonical_masked_scrubbed_v1_t000.jsonl
pairwise_all_judges_canonical_masked_scrubbed_v1_t000.jsonl
```

Final primary integrity:

```text
total rows: 10,000
duplicate record_ids: 0
2,500 rows per judge
5,000 AB / 5,000 BA
2,000 rows per run index
```

## Primary results

Overall row-level result:

```text
LLM wins: 3,653 / 10,000 = 36.53%
Programmatic wins: 6,347 / 10,000 = 63.47%
```

By judge:

| Judge | LLM wins | Programmatic wins | LLM win rate |
|---|---:|---:|---:|
| `gemma_4b_judge` | 836 | 1,664 | 33.44% |
| `gemma_12b_judge` | 740 | 1,760 | 29.60% |
| `qwen_7b_judge` | 1,042 | 1,458 | 41.68% |
| `qwen_14b_judge` | 1,035 | 1,465 | 41.40% |

Order effect:

| Order | LLM win rate |
|---|---:|
| AB, LLM as Plan A | 57.54% |
| BA, LLM as Plan B | 15.52% |

AB/BA pair-run consistency:

| Category | Count | Rate |
|---|---:|---:|
| `source_consistent_llm` | 626 | 12.52% |
| `source_consistent_programmatic` | 1,973 | 39.46% |
| `position_consistent_plan_a` | 2,251 | 45.02% |
| `position_consistent_plan_b` | 150 | 3.00% |

Interpretation: the primary result does not support a broad LLM-source preference. Programmatic plans are preferred more often, and position bias is large.

## Exploratory marker-rationale analysis

A post-hoc exploratory analysis keyword-coded the primary judgments' free-text rationales and advantage lists. This was not a formal marker-level judging pass.

Exploratory result: programmatic plans were more often credited with winning advantages on most training-relevant marker-like categories, including endurance/base, quality/speed, recovery/safety, and variety/balance.

This analysis is useful for hypothesis generation but should be labeled exploratory.

## Explicit marker-level rerun status

A formal marker-level rerun was started after the primary study completed. It uses the same frozen manifest and judge inputs but asks each judge for explicit per-marker preferences and 1--5 scores.

Markers:

```text
plan_coherence
training_specificity
load_progression
recovery_safety
quality_session_design
endurance_development
readiness_alignment
clarity_actionability
explanation_quality
```

Expected full marker dataset:

```text
10,000 marker records
x 9 markers
= 90,000 explicit marker decisions
```

Current status from latest session:

- Qwen 14B marker smoke succeeded.
- Qwen 14B full marker run produced 701 valid rows with 0 failures.
- These 701 rows landed in a nested incorrect path:

```text
artifacts/gen_src_t070_exp_t000/frozen_primary_v1/marker_eval_t000_scrubbed_v1_staged/
  pairwise_qwen_14b_judge_canonical_masked_scrubbed_v1_t000.jsonl/
    marker_qwen_14b_judge_canonical_masked_scrubbed_v1_t000.jsonl
```

- The next step is to patch the marker SLURM script so it passes `--output-dir "$OUTPUT_DIR"`, move the 701 rows to the correct location, and resume Qwen 14B to 2,500 rows.

Correct target path:

```text
artifacts/gen_src_t070_exp_t000/frozen_primary_v1/marker_eval_t000_scrubbed_v1_staged/marker_qwen_14b_judge_canonical_masked_scrubbed_v1_t000.jsonl
```

## Work not yet complete

Still required:

- Complete Qwen 14B explicit marker pass to 2,500 rows.
- Run explicit marker pass for Qwen 7B, Gemma 12B, and Gemma 4B.
- Combine marker files into `marker_all_judges_canonical_masked_scrubbed_v1_t000.jsonl`.
- Run marker analysis.
- Run bootstrap confidence intervals and inferential modeling.
- Populate final research report with actual marker-level results.
- Complete temperature-sensitivity evaluation if still desired.

