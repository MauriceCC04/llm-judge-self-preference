# Data Processing Log — `llm-judge-self-preference`

This document records the data-processing and experiment-preparation steps taken during the Qwen/Gemma training-plan judge-bias study. It is intended to be maintained as the study continues.

## Project objective

Study whether LLM judges systematically prefer LLM-generated training plans over programmatically generated training plans when deterministic plan quality is controlled.

The core research question is:

> Controlling for deterministic structural plan quality, do LLM judges systematically favor LLM-generated training plans over programmatically generated plans, and does this effect vary by judge model family, model size, rubric, or whether the plan matches the judge's own family?

The key validity principle is that equal deterministic score is necessary but not sufficient. The experiment must control structural quality without letting presentation polish, explanation richness, citations, data-note verbosity, title wording, model names, or generation artifacts identify the source arm.

## Current frozen design

Current study design:

* LLM source families:
  * `Qwen/Qwen2.5-7B-Instruct`
  * `google/gemma-3-4b-it`
* Programmatic control plans generated from structural variation logic.
* Shared explainer model used for narrative/explanation fields where applicable.
* Matching target: exactly 250 structurally matched LLM-vs-programmatic pairs for the primary full study.
* Active local judge set:
  * `qwen_7b_judge`
  * `qwen_14b_judge`
  * `gemma_4b_judge`
  * `gemma_12b_judge`
* Pairwise evaluation target:
  * 250 matched pairs x 4 judge models x 5 repeats x 2 AB/BA orders = 10,000 pairwise judgment records.
* Primary judge-facing view after repair:
  * `canonical_masked` with scrub version `canonical_masked_scrubbed_v1`.
* Primary frozen input directory after local preparation:
  * `artifacts/gen_src_t070_exp_t000/frozen_primary_v1/`
* Primary manifest for HPC judging:
  * `artifacts/gen_src_t070_exp_t000/frozen_primary_v1/eval_manifest/pairwise_eval_manifest.jsonl`

The 10,000-document requirement means 10,000 pairwise judgment records. It does not mean 10,000 generated training plans.

## Repository and environment notes

### Working repository

HPC repository path:

```bash
/mnt/beegfsstudents/home/3202029/llm-judge-self-preference
```

Related TrailTraining repository path:

```bash
/mnt/beegfsstudents/home/3202029/trailtraining
```

Required environment variables for scripts that import both repositories:

```bash
export REPO_ROOT=/mnt/beegfsstudents/home/3202029/llm-judge-self-preference
export TRAILTRAINING_REPO=/mnt/beegfsstudents/home/3202029/trailtraining
export PYTHONPATH="${REPO_ROOT}:${TRAILTRAINING_REPO}/src:${PYTHONPATH:-}"
```

Important HPC environment issue observed:

* `conda activate judge-bias` failed in an interactive login shell with:

```text
CondaError: Run 'conda init' before 'conda activate'
```

Safer workaround:

* Use the environment Python directly when possible:

```bash
/home/3202029/.conda/envs/judge-bias/bin/python
```

or ensure the correct module and shell initialization are loaded in SLURM jobs.


## Qwen source-generation processing

### Early Qwen pilot/top-up observations

A Qwen pilot/top-up generation run produced valid plans but also repeated validation failures for understructured cells, especially:

```text
ab_A4__r_low__rc_high__ph_base
```

Common failure causes:

```text
understructured:no_active_sessions
understructured:missing_long_run
understructured:missing_quality_for_A4
```

A successful advanced pilot later produced 8 valid plans across 8 advanced cells.

Example successful advanced pilot audit:

```text
plans 8
provenance 8
leaks []
bad []
```

Observations:

* Qwen generation could succeed, but some low/high/base and A4 cells required extra attempts.
* Final validation guards were working: understructured plans were rejected and logged.

### Full Qwen generation

Full Qwen generation was launched for all 32 fixture cells.

Important initial issue:

* A first check from the home directory showed no `full_qwen/plans` directory because commands were run from the wrong working directory.
* After checking the correct repository path, outputs were found.

The full Qwen run initially produced:

```text
total provenance: 190
cells: 32
by_band: {'A1': 46, 'A2': 48, 'A3': 48, 'A4': 48}
non-6 cells:
  ab_A1__r_low__rc_high__ph_base: 4/6
missing_cells_csv:
ab_A1__r_low__rc_high__ph_base
```

A top-up was run for the missing Qwen cell:

```text
ab_A1__r_low__rc_high__ph_base
```

The top-up overgenerated the cell to 10 plans. Extra plans were then quarantined.

Plans kept:

```text
ab_A1__r_low__rc_high__ph_base__qwen2.5_7b_instruct__src_t070__exp_t000__s013.json
ab_A1__r_low__rc_high__ph_base__qwen2.5_7b_instruct__src_t070__exp_t000__s015.json
ab_A1__r_low__rc_high__ph_base__qwen2.5_7b_instruct__src_t070__exp_t000__s016.json
ab_A1__r_low__rc_high__ph_base__qwen2.5_7b_instruct__src_t070__exp_t000__s022.json
ab_A1__r_low__rc_high__ph_base__qwen2.5_7b_instruct__src_t070__exp_t000__s028.json
ab_A1__r_low__rc_high__ph_base__qwen2.5_7b_instruct__src_t070__exp_t000__s039.json
```

Quarantined extras:

```text
ab_A1__r_low__rc_high__ph_base__qwen2.5_7b_instruct__src_t070__exp_t000__s047.json
ab_A1__r_low__rc_high__ph_base__qwen2.5_7b_instruct__src_t070__exp_t000__s049.json
ab_A1__r_low__rc_high__ph_base__qwen2.5_7b_instruct__src_t070__exp_t000__s053.json
ab_A1__r_low__rc_high__ph_base__qwen2.5_7b_instruct__src_t070__exp_t000__s060.json
```

Quarantine destination:

```text
artifacts/gen_src_t070_exp_t000/full_qwen/plans/quarantine_overgenerated_ab_A1__r_low__rc_high__ph_base_20260504_195455
```

Final Qwen corpus audit:

```text
plans 192
provenance 192
cells 32
by_band {'A1': 48, 'A2': 48, 'A3': 48, 'A4': 48}
cell count min/max 6 6
non_6_cells {}
missing_sidecars []
leaks []
bad []
generic_titles_mean 1.9635416666666667
generic_titles_max 6
```

Conclusion:

* Qwen source corpus is balanced and complete: 32 cells x 6 plans = 192 plans.
* Sidecar provenance files are complete.
* Basic leakage and validity checks passed.

## Gemma source-generation processing

### Hugging Face cache and access issue

Initial Gemma model caching failed because the Hugging Face token was not configured.

Observed failure:

```text
GatedRepoError: 401 Client Error
Access to model google/gemma-2-9b-it is restricted. You must have access to it and be authenticated to access it. Please log in.
```

After setting the Hugging Face token, model caching succeeded:

```text
Downloading google/gemma-2-9b-it ...
Done google/gemma-2-9b-it
Downloading google/gemma-2-2b-it ...
Done google/gemma-2-2b-it
Model cached: google/gemma-2-9b-it
Model cached: google/gemma-2-2b-it
```

Later, because of quota constraints, some cached Qwen models were removed. This caused later failures:

```text
Model not found in cache: Qwen/Qwen2.5-3B-Instruct
Model not found in cache: google/gemma-3-4b-it
```

Important note:

* The explainer model cache must remain available if generation depends on it.
* Avoid deleting cached models unless the next job's source and explainer models are known and re-cached.

### Gemma pilot generation

Gemma pilot used:

```text
source model: google/gemma-3-4b-it
explainer model: Qwen/Qwen2.5-3B-Instruct
```

The advanced pilot produced 8 valid plans over 8 selected A3/A4 cells.

Pilot result:

```text
Generated: 8
plans 8
provenance 8
leaks []
bad []
```

Observed rejected attempts included:

```text
understructured:missing_long_run
understructured:collapsed_easy_template
```

Conclusion:

* Gemma 3 generation was viable, but some cells showed repetitive/easy-template tendencies.

### Full Gemma generation and top-ups

Initial full Gemma generation produced:

```text
plans: 180
provenance: 180
total: 180
cells: 30
by_band: {'A1': 48, 'A2': 36, 'A3': 48, 'A4': 48}
cell count min/max: 6 6
```

Missing cells:

```text
ab_A2__r_high__rc_high__ph_base
ab_A2__r_high__rc_high__ph_peak
```

Additional issue discovered:

```text
non_7_day_plans [('ab_A3__r_low__rc_high__ph_base__gemma_3_4b_it__src_t070__exp_t000__s003.json', 18)]
count 1
```

A3 replacement was run separately.

A2 high/high base and peak top-ups were difficult. One run with 120 attempts produced only:

```text
ab_A2__r_high__rc_high__ph_base: accepted 1/6
ab_A2__r_high__rc_high__ph_peak: accepted 0/6
```

Common failure:

```text
understructured:missing_long_or_quality_for_A2_high_high
```

Later top-up strategy:

* A2 base: exactly 5 more plans.
* A2 peak: exactly 6 plans.
* A3 replacement run separately.

Final Gemma corpus audit:

```text
plans 192
provenance 192
total 192
cells 32
by_band {'A1': 48, 'A2': 48, 'A3': 48, 'A4': 48}
cell count min/max 6 6
non_6_cells {}
missing_sidecars []
bad_lengths []
leaks []
bad []
generic_titles_mean 2.6197916666666665
generic_titles_max 6
```

Conclusion:

* Gemma 3 source corpus is balanced and complete: 32 cells x 6 plans = 192 plans.
* Known problematic A2 and A3 cells were corrected through targeted top-up/replacement.

## Programmatic-plan generation processing

### Initial programmatic generation failure

First full programmatic run failed because the sampler attempted to auto-fit priors from too few LLM plans:

```text
[ABORT] Refusing to auto-fit sampler priors from only 0 LLM plans (< 30).
Provide SAMPLER_CONFIG, generate more LLM plans, or set ALLOW_TINY_PRIOR_FIT=1 to override.
```

Existing sampler configs found:

```text
./artifacts/pilot16/sampler_config.json
./artifacts/pilot_yield/sampler_config.json
./artifacts/postfix_llm_sanity/sampler_config.json
```

The `pilot_yield` sampler config was based on 24 plans; `pilot16` and `postfix_llm_sanity` were based on 8 plans.

Interpretation:

* The programmatic generation script needed an explicit sampler config or a corrected source path for fitting priors.
* Auto-fitting from 0 LLM plans was correctly blocked.

### Programmatic generation before final fix

A programmatic run produced 640 plans and 640 provenance files, but initial quality audits showed narrative/session contradictions:

```text
generic_titles 3224
title_session_mismatches 35
rest_active_contradictions 24
unique_session_sequences 106 / 640
```

Examples included:

* `session_type` was aerobic but text said Long session.
* `session_type` was long but text said Rest day.
* Active sessions described as rest days.

Conclusion:

* The first programmatic corpus was structurally present but not acceptable due to source-identifying or contradictory narrative artifacts.
* Programmatic explanation/title generation needed correction.

### Fixed programmatic generation

A fixed programmatic job completed successfully:

```text
JOBID=489430
State: COMPLETED
Elapsed: 04:32:07
Generated: 640
Skipped: 0
```

The generation logs still included some failed attempts due to structured explanation parse failures, for example:

```text
Unterminated string starting at ...
Expecting ',' delimiter ...
```

However, retries succeeded and the final accepted corpus was complete.

Final programmatic corpus audit:

```text
plans 640
provenance 640
cells 32
by_band {'A1': 160, 'A2': 160, 'A3': 160, 'A4': 160}
cell count min/max 20 20
non_20_cells {}
missing_sidecars []
bad_lengths []
leaks []
bad_json []
generic_titles 0
title_session_mismatches 0
rest_active_contradictions 0
unique_session_sequences 134 / 640
```

Structural summary:

```text
duration_by_band {
  'A1': {'n': 160, 'mean': 232.34, 'min': 115, 'max': 353},
  'A2': {'n': 160, 'mean': 263.48, 'min': 137, 'max': 417},
  'A3': {'n': 160, 'mean': 292.21, 'min': 138, 'max': 525},
  'A4': {'n': 160, 'mean': 323.52, 'min': 138, 'max': 577}
}
hard_by_band {
  'A1': {'n': 160, 'mean': 0.72, 'min': 0, 'max': 2},
  'A2': {'n': 160, 'mean': 0.75, 'min': 0, 'max': 2},
  'A3': {'n': 160, 'mean': 0.99, 'min': 0, 'max': 3},
  'A4': {'n': 160, 'mean': 1.07, 'min': 0, 'max': 3}
}
long_by_band {
  'A1': {'n': 160, 'mean': 0.66, 'min': 0, 'max': 2},
  'A2': {'n': 160, 'mean': 0.89, 'min': 0, 'max': 2},
  'A3': {'n': 160, 'mean': 0.93, 'min': 0, 'max': 3},
  'A4': {'n': 160, 'mean': 1.02, 'min': 0, 'max': 3}
}
rest_by_band {
  'A1': {'n': 160, 'mean': 1.81, 'min': 1, 'max': 4},
  'A2': {'n': 160, 'mean': 1.59, 'min': 1, 'max': 3},
  'A3': {'n': 160, 'mean': 1.52, 'min': 1, 'max': 3},
  'A4': {'n': 160, 'mean': 1.49, 'min': 1, 'max': 3}
}
```

Conclusion:

* Programmatic corpus is complete and internally consistent by the applied checks.
* It contains 20 plans per cell, across 32 cells, totaling 640 plans.

## Final corpus state before matching repair

Final corpus audits:

```text
=== qwen ===
plans 192
provenance 192
cells 32
by_band {'A1': 48, 'A2': 48, 'A3': 48, 'A4': 48}
cell_count_min_max (6, 6)
non_expected_cells {}
missing_sidecars []
bad []

=== gemma3 ===
plans 192
provenance 192
cells 32
by_band {'A1': 48, 'A2': 48, 'A3': 48, 'A4': 48}
cell_count_min_max (6, 6)
non_expected_cells {}
missing_sidecars []
bad []

=== programmatic ===
plans 640
provenance 640
cells 32
by_band {'A1': 160, 'A2': 160, 'A3': 160, 'A4': 160}
cell_count_min_max (20, 20)
non_expected_cells {}
missing_sidecars []
bad []
```

Total available plans:

```text
Qwen LLM:        192
Gemma 3 LLM:     192
Programmatic:    640
Total:          1024
```

## Matching-pool construction

A matching pool was constructed by copying Qwen, Gemma 3, and programmatic plan files plus provenance sidecars into:

```text
artifacts/gen_src_t070_exp_t000/matching_pool/plans
```

Result:

```text
copied_plans 1024
pool artifacts/gen_src_t070_exp_t000/matching_pool/plans
```

## Initial matching attempt and blocker

Initial matching command:

```bash
python cli.py match \
  --plans artifacts/gen_src_t070_exp_t000/matching_pool/plans \
  --output artifacts/gen_src_t070_exp_t000/matching_pool/matched_pairs.json \
  --allow-mixed-generation-conditions
```

First failure:

```text
ModuleNotFoundError: No module named 'pydantic'
```

After switching to an environment where `pydantic` was importable, matching failed again:

```text
match.pair.ScoringError: 1024 plans failed deterministic scoring; see artifacts/gen_src_t070_exp_t000/matching_pool/scoring_failures.json
```

Inspecting `scoring_failures.json` showed the real cause:

```text
ModuleNotFoundError: No module named 'trailtraining'
```

Interpretation:

* The scorer was being run with an environment that had `pydantic` but not the `trailtraining` package on `PYTHONPATH`.
* This was an environment/path issue, not a plan-validity issue.

Required environment fix:

```bash
export REPO_ROOT=/mnt/beegfsstudents/home/3202029/llm-judge-self-preference
export TRAILTRAINING_REPO=/mnt/beegfsstudents/home/3202029/trailtraining
export PYTHONPATH="${REPO_ROOT}:${TRAILTRAINING_REPO}/src:${PYTHONPATH:-}"
```

### Matching output after environment fix

After the environment issue was corrected, matching completed but yielded too few pairs:

```text
Pairs yielded:    30  (target: 256)
Mean score gap:   0.000
Coverage ratio:   0.117
Coverage OK:      False
```

Matched-pair output was saved to:

```text
artifacts/gen_src_t070_exp_t000/matching_pool/matched_pairs.json
```

Additional audit files:

```text
artifacts/gen_src_t070_exp_t000/matching_pool/matching_audit.json
artifacts/gen_src_t070_exp_t000/matching_pool/matching_prefilter_audit.json
```

Conclusion:

* The 30-pair matched set was not usable for the full study.
* Full pairwise judging was blocked from this matched set.

### Matching prefilter audit

The matching prefilter audit showed:

```text
n_input: 1024
n_kept: 936
n_dropped: 88
drop_reasons: session_signature_duplicate=88
exact_text_duplicate_groups: none
invalid_or_contradictory: none
```

Interpretation:

* Prefiltering removed some duplicate session signatures.
* No exact text duplicates were found.
* No invalid or contradictory plans were flagged by the prefilter.
* The prefilter was not the main cause of low pair yield.

### Score non-overlap diagnosis

Observed deterministic score distributions by source:

Qwen:

```text
n: 192
min: 10
p25: 97
median: 100
p75: 100
max: 100
bins: mostly 97 and 100
```

Gemma 3:

```text
n: 192
min: 10
p25: 85
median: 100
p75: 100
max: 100
bins: mostly 85, 97, 100
```

Programmatic:

```text
n: 640
min: 0
p25: 30
median: 30
p75: 30
max: 100
bins:
  0: 26
  12: 6
  15: 17
  27: 77
  30: 471
  97: 7
  100: 20
  plus a few scattered values
```

Relaxing matching constraints did not solve the issue:

* With same score bin required:
  * tolerance 0.25 to 3.0 produced only about 31 pairs.
* Without same score bin:
  * tolerance 0.25 to 2.0 produced only about 31 pairs.
  * tolerance 3.0 produced about 35 pairs.

Interpretation:

* The problem was deterministic score non-overlap, not same-bin matching.
* Most LLM plans scored 97-100.
* Most programmatic plans scored around 30.
* The existing deterministic score was unsuitable as the primary matching score for this study.

### Working hypothesis about the scoring issue

The existing TrailTraining quality score appeared likely to be contaminated by presentation/explanation quality rather than measuring only structural plan quality.

Possible presentation-sensitive fields:

* title wording
* workout prose
* purpose/rationale prose
* explanation richness
* claim attributions
* citations
* data-note completeness
* narrative length
* LLM-style polish

Why this was a problem:

* The study needs to control structural quality, not presentation quality.
* If the matching score rewards LLM-style prose, matching will either fail or condition on a source-identifying confound.
* Equal score under a presentation-contaminated scorer would not isolate structural quality.

## Structural matching repair

### Code changes applied

A structural-score repair was implemented in patches through v4. The important methodological changes were:

1. Added a source-neutral structural scorer.
2. Switched the primary matching score from TrailTraining's full deterministic quality score to structural score.
3. Removed narrative/prose length from the primary matching distance.
4. Changed session-signature duplicates from hard drops to audit/sensitivity metadata, because hard-dropping duplicates reduced match coverage below the full-study target.
5. Changed matching from simple greedy pairing to target-cardinality matching with structural distance/soft-caliper cost.
6. Preserved same-cell constraints for causal identification:
   * same fixture/cell
   * same athlete band
   * same readiness
   * same recovery capability
   * same race phase
   * same style
   * same plan days
7. Added structural matching diagnostics and launch gates.
8. Added a canonical masked judge-facing view and later a scrubbed canonical-mask post-processing step.

Structural score exclusions:

* source family/model labels
* generation arm
* file names
* title wording
* prose richness
* citations
* claim attributions
* data-note verbosity
* rationale/explanation length
* LLM-written narrative fields

Structural score included source-neutral plan structure:

* plan length
* rest-day count
* active-day count
* hard-day count
* quality-day count
* long-run count
* total duration
* max day duration
* mean day duration
* session types
* durations
* rest/hard flags
* target intensity where normalized
* fixture/cell constraints

### Matching repair iterations

#### v1/v2 repair attempt

Initial structural-score repair produced a structurally plausible overlap and 250+ possible matches in diagnostics, but actual matching remained below target locally because session-signature duplicates were still hard-dropped before matching.

Observed failure after v2:

```text
match.pair.MatchingCoverageError: Matching coverage 0.891 is below required ratio 0.977
```

Interpretation:

* The structural score fixed the score-overlap problem.
* The remaining coverage failure came from pre-match duplicate dropping and matching selection behavior.

#### v3 repair

v3 changed session-signature duplicate handling from hard-drop to audit-only for primary matching while continuing to drop invalid or contradictory plans.

Validated result on the available artifact pool:

```text
kept: 1023
dropped: 1
candidate_pairs: 258
selected_pairs: 250
by_source:
  Qwen/Qwen2.5-7B-Instruct: 130
  google/gemma-3-4b-it: 120
```

v3 also aligned the full-study gate to exactly:

```text
250 pairs x 4 judges x 5 runs x 2 orders = 10,000 records
```

#### v4 repair

v4 added cleaner judge-facing masking and improved matching selection:

* `canonical_masked` became the default judge-facing view.
* Source/model/provenance-like strings were scrubbed from judge inputs.
* Presentation-heavy fields were removed from primary judge-facing inputs:
  * citations
  * claim attributions
  * data notes
  * rationale/explanation/reasoning fields
  * long notes fields
* Matching selection used target-cardinality min-cost matching with soft-caliper penalties.
* Optional hard calipers were added for feasibility and sensitivity checks.

The current matched set remained usable but still had non-trivial structural tail gaps. This was accepted as a defensible primary input after canonical masking and a scrubbed manifest step, with planned sensitivity analyses for high-gap pairs.

## Final matched-pair state

Final primary matched-pair file:

```text
artifacts/gen_src_t070_exp_t000/matching_pool/matched_pairs.json
```

Current matched-pair audit:

```text
matched pairs: 250
Qwen-source pairs: 130
Gemma-source pairs: 120
fixture cells covered: 32/32
structural score gap mean: approximately 0.255
structural score gap max: 2.0
```

From judge-input-derived feature gaps after v4/scrubbed manifest:

```text
total_minutes gap:
  mean approximately 64.8
  median approximately 45.0
  p95 approximately 186.9
  max approximately 312.0

max_day_minutes gap:
  mean approximately 23.7
  median approximately 17.0
  p95 approximately 63.0
  max approximately 166.0

hard-day gap:
  mean approximately 0.50
  median 0
  max 2

long-run gap:
  mean approximately 0.57
  median 0
  max 3

quality-day gap:
  mean approximately 0.50
  median 0
  max 2

rest-day gap:
  mean approximately 0.36
  median 0
  max 3
```

Interpretation:

* The 30-pair blocker is resolved.
* The full-study pair count is achieved.
* The matched set is balanced across source families and covers all cells.
* Structural score gaps are low.
* Some feature-level tail gaps remain and should be handled in sensitivity analyses rather than ignored.

Recommended sensitivity analyses:

* Exclude largest total-duration-gap pairs.
* Exclude largest max-day-duration-gap pairs.
* Exclude pairs with hard/long/quality/rest-day gaps greater than 1.
* Exclude duplicate-heavy session-signature cells or duplicated LLM session signatures.
* Report score-gap sensitivity.
* Report fixture/band subgroup sensitivity.

## Eval manifest construction

### Canonical masked manifest

A full pairwise evaluation manifest was built from the 250 matched pairs.

Primary design count:

```text
250 matched pairs x 4 judges x 5 runs x 2 orders = 10,000 records
```

Observed manifest checks before final scrub:

```text
judge_view: canonical_masked
expected_pairwise_documents: 10000
exact_10000_gate_passed: true
masking_gate_passed: true
masking_failures: []
manifest_path: artifacts/gen_src_t070_exp_t000/matching_pool/eval_manifest/pairwise_eval_manifest.jsonl
```

An uploaded manifest bundle was independently audited and contained:

```text
manifest records: 10,000
judge-input JSON files: 10,000
unique pairs: 250
records per pair: 40 exactly
qwen_7b_judge: 2,500 records
qwen_14b_judge: 2,500 records
gemma_4b_judge: 2,500 records
gemma_12b_judge: 2,500 records
AB records: 5,000
BA records: 5,000
run indices 0-4: 2,000 records each
judge_view: canonical_masked for all records
self-family records: 5,000
non-self-family records: 5,000
duplicate manifest keys: 0
```

Judge-facing schema in representative files contained only:

```text
plan_a
plan_b
```

with day-level fields limited to:

```text
day_index
session_type
duration_minutes
is_rest_day
is_hard_day
target_intensity
workout_summary
purpose_summary
```

This was a major improvement over raw plan JSON.

### Residual canonicalization artifacts found

Audit of the first canonical masked manifest found a small number of non-source-label artifacts:

```text
Cite: / Citations: / signal_id / load.last7 remnants
readiness status wording
```

These were not direct Qwen/Gemma/programmatic leaks, but they were provenance/data-style cues that could distract judges or raise reviewer concerns.

Affected examples included citation-like strings inside `workout_summary`, such as:

```text
Citations: [{"signal_id": "load.last7.distance_..."}]
```

Decision:

* Do not rematch or regenerate.
* Fix by non-destructive post-processing of judge-facing text summaries only.
* Preserve structural fields, pair metadata, order, judge, run, source-family metadata, and original manifest auditability.

## Scrubbed canonical masked manifest

A post-processing scrubber was applied to the canonical masked manifest.

Scrubbed manifest directory:

```text
artifacts/gen_src_t070_exp_t000/matching_pool/eval_manifest_scrubbed_v1/
```

Scrubbed manifest path:

```text
artifacts/gen_src_t070_exp_t000/matching_pool/eval_manifest_scrubbed_v1/pairwise_eval_manifest.jsonl
```

Scrub version:

```text
canonical_masked_scrubbed_v1
```

Fields edited by the scrubber:

* `purpose_summary`
* `workout_summary`

Fields preserved:

* all structural fields
* `pair_id`
* `record_id`
* `fixture_id`
* `judge`
* `run`
* `order`
* source-family metadata outside the judge prompt
* self-family metadata outside the judge prompt
* structural score gap
* feature gaps

Patterns scrubbed from judge-facing files:

```text
Qwen
Gemma
programmatic
source
generated by
fixture
provenance
model
Cite:
Citations:
signal_id
load.last7
readiness status
current readiness
```

Independent verifier output after scrub:

```text
records: 10000
detected judge key: judge
pairs: 250 min/max records per pair: 40 40
judges: Counter({'qwen_7b_judge': 2500, 'qwen_14b_judge': 2500, 'gemma_4b_judge': 2500, 'gemma_12b_judge': 2500})
orders: Counter({'AB': 5000, 'BA': 5000})
runs: Counter({0: 2000, 1: 2000, 2: 2000, 3: 2000, 4: 2000})
views: Counter({'canonical_masked': 10000})
PASS: scrubbed manifest is ready for judging
```

Example manifest row keys after scrub:

```text
canonical_masking_version = canonical_masked_scrubbed_v1
feature_gaps = {...}
fixture_id = ab_A1__r_high__rc_high__ph_base
judge = qwen_7b_judge
judge_input_path = artifacts/gen_src_t070_exp_t000/matching_pool/eval_manifest_scrubbed_v1/judge_inputs/pair_0000__qwen_7b_judge__r00__AB.json
judge_model_family = qwen
judge_view = canonical_masked
left_plan_role = llm
llm_plan_id = ...
order = AB
pair_id = pair_0000
position = AB
programmatic_plan_id = ...
record_id = pair_0000__qwen_7b_judge__r00__AB
right_plan_role = programmatic
run = 0
self_family_match = False
source_model_family = gemma
structural_score_gap = 0.0
structural_score_version = structural_score_v1.0.0
```

Conclusion:

* The scrubbed manifest is the primary evaluation input.
* The original canonical masked manifest is retained as an audit trail and possible sensitivity condition.
* Source/provenance leakage scan passed on the scrubbed manifest.
* The full 10,000-record gate passed.

## Frozen primary input

The scrubbed manifest was frozen into:

```text
artifacts/gen_src_t070_exp_t000/frozen_primary_v1/
```

Freeze commands:

```bash
mkdir -p artifacts/gen_src_t070_exp_t000/frozen_primary_v1

cp artifacts/gen_src_t070_exp_t000/matching_pool/matched_pairs.json \
  artifacts/gen_src_t070_exp_t000/frozen_primary_v1/

cp -R artifacts/gen_src_t070_exp_t000/matching_pool/eval_manifest_scrubbed_v1 \
  artifacts/gen_src_t070_exp_t000/frozen_primary_v1/eval_manifest

cp artifacts/gen_src_t070_exp_t000/matching_pool/matching_audit.json \
  artifacts/gen_src_t070_exp_t000/frozen_primary_v1/ || true

cp artifacts/gen_src_t070_exp_t000/matching_pool/matching_prefilter_audit.json \
  artifacts/gen_src_t070_exp_t000/frozen_primary_v1/ || true

find artifacts/gen_src_t070_exp_t000/frozen_primary_v1 -type f -print0 \
  | sort -z \
  | xargs -0 shasum -a 256 \
  > artifacts/gen_src_t070_exp_t000/frozen_primary_v1/SHA256SUMS.txt
```

Primary manifest from this point forward:

```text
artifacts/gen_src_t070_exp_t000/frozen_primary_v1/eval_manifest/pairwise_eval_manifest.jsonl
```

The older unsanitized or non-scrubbed manifest should not be used as the primary evaluation input.

## Smoke manifest

An 8-record smoke-test manifest was created from the frozen primary manifest:

```bash
mkdir -p artifacts/gen_src_t070_exp_t000/frozen_primary_v1/smoke_test

head -n 8 artifacts/gen_src_t070_exp_t000/frozen_primary_v1/eval_manifest/pairwise_eval_manifest.jsonl \
  > artifacts/gen_src_t070_exp_t000/frozen_primary_v1/smoke_test/smoke_manifest.jsonl
```

Expected local check:

```bash
wc -l artifacts/gen_src_t070_exp_t000/frozen_primary_v1/smoke_test/smoke_manifest.jsonl
```

Expected output:

```text
8 artifacts/gen_src_t070_exp_t000/frozen_primary_v1/smoke_test/smoke_manifest.jsonl
```

Interpretation:

* Creating the smoke manifest is a local file operation and prints no output unless checked with `wc`, `head`, or similar commands.
* Actual smoke judging should run where the local judge models are available, typically on an HPC GPU/compute node.

## Current status

The study is ready to move to HPC for smoke judging and then full pairwise judging.

Completed gates:

```text
final Qwen corpus complete: yes
final Gemma corpus complete: yes
final programmatic corpus complete: yes
matching pool complete: yes
structural matching repair complete: yes
matched pairs: 250
Qwen and Gemma source plans represented: yes
four Qwen/Gemma judges configured in manifest: yes
five runs configured: yes
AB and BA orders configured: yes
expected pairwise records: 10,000
exact 10,000 gate passed: yes
canonical masked judge inputs created: yes
scrubbed canonical manifest created: yes
source/provenance masking verifier passed: yes
frozen primary input prepared: yes
smoke manifest prepared: yes
```

Remaining before full judging:

```text
transfer frozen_primary_v1 to HPC
verify checksums on HPC
run 8-record smoke judging job on GPU/compute node
confirm compare_plans executes and output parses
only then launch full 10,000-record judging job
```

## HPC transfer and preflight

Copy frozen primary input to HPC:

```bash
rsync -av \
  artifacts/gen_src_t070_exp_t000/frozen_primary_v1/ \
  <USER>@<HPC_HOST>:/path/to/llm-judge-self-preference/artifacts/gen_src_t070_exp_t000/frozen_primary_v1/
```

Sync patched repo code to HPC if needed:

```bash
rsync -av \
  --exclude ".venv" \
  --exclude "__pycache__" \
  --exclude ".git" \
  ./ \
  <USER>@<HPC_HOST>:/path/to/llm-judge-self-preference/
```

On HPC login node, only run lightweight checks:

```bash
export REPO_ROOT=/mnt/beegfsstudents/home/3202029/llm-judge-self-preference
export TRAILTRAINING_REPO=/mnt/beegfsstudents/home/3202029/trailtraining
export PY=/home/3202029/.conda/envs/judge-bias/bin/python
export PYTHONPATH="${REPO_ROOT}:${TRAILTRAINING_REPO}/src:${PYTHONPATH:-}"
cd "$REPO_ROOT"

"$PY" --version
"$PY" - <<'PY'
import pydantic
import trailtraining
print("pydantic ok")
print("trailtraining ok")
PY
```

Verify frozen files survived transfer:

```bash
sha256sum -c artifacts/gen_src_t070_exp_t000/frozen_primary_v1/SHA256SUMS.txt
```

If macOS produced `shasum -a 256` hashes, `sha256sum -c` on Linux should still accept the file format if it is standard two-column output. If not, regenerate checksums on HPC and keep both checksum files.

## HPC smoke judging

Smoke manifest:

```text
artifacts/gen_src_t070_exp_t000/frozen_primary_v1/smoke_test/smoke_manifest.jsonl
```

Smoke-test requirements:

```text
8 attempted records
0 missing judge input files
models load from cache
compare_plans completes
winner labels parse
record_id/pair_id/judge/run/order preserved
errors are logged instead of crashing the whole job
```

Do not launch the full 10,000-record job until the smoke test succeeds.

## Full pairwise judging

Primary full manifest:

```text
artifacts/gen_src_t070_exp_t000/frozen_primary_v1/eval_manifest/pairwise_eval_manifest.jsonl
```

Expected full output:

```text
10,000 pairwise judgment records
```

Full job must not be run on the login node. Use a GPU/compute node or SLURM batch job.

The judging harness must use the repository's existing `compare_plans` function from `soft_eval.py` for pairwise comparison.

Each output record should preserve at minimum:

```text
record_id
pair_id
judge
judge_model_family
run
order
position
left_plan_role
right_plan_role
llm_plan_id
programmatic_plan_id
source_model_family
self_family_match
structural_score_gap
structural_score_version
winner / chosen plan
raw judge response or parse metadata
parse_status / error field if applicable
```

## Post-judging integrity checks

Before interpreting results, run checks that verify:

```text
result rows == 10000
no duplicate record_id
all manifest record_ids present
all 250 pairs represented
all four judges represented
AB records == 5000
BA records == 5000
runs 0-4 balanced
winner labels valid
parse failures counted
refusals/errors counted
missing judge input files == 0
```

Only after these checks pass should preference rates be interpreted.

## Analysis plan

Primary descriptive outputs:

* Overall LLM win rate.
* LLM win rate by judge model.
* LLM win rate by judge family.
* LLM win rate by source family.
* LLM win rate by order/position.
* Self-family vs non-self-family win rate.
* Parse/error/refusal rate.

Primary uncertainty estimates:

* Bootstrap confidence interval for overall LLM win rate.
* Bootstrap confidence intervals by judge model.
* Bootstrap confidence intervals by judge family.
* Cluster/bootstrap sensitivity by pair ID where possible.

Primary bias/validity audits:

* AB vs BA order-bias audit.
* Score-gap sensitivity.
* Fixture/band subgroup analysis.
* Source-family analysis.
* Judge-family and judge-size analysis.
* Self-family analysis only where both source and judge families are present.

Required sensitivity analyses:

* Exclude largest structural feature-gap pairs.
* Exclude pairs with high total-duration gaps.
* Exclude pairs with hard/long/quality/rest-day gaps greater than 1.
* Exclude duplicate-heavy session-signature groups or cells.
* Compare scrubbed canonical masked primary view against the original canonical masked view only as a sensitivity condition, not as the primary input.
* Report whether conclusions change under these exclusions.

Interpretation cautions:

* Equal structural score is not full equality.
* The structural score controls major plan structure, not every meaningful training-quality dimension.
* Programmatic plans may still contain LLM-written explanation fields in the underlying corpora, though the primary judge view masks/scrubs much of this.
* The canonical masked/scrubbed view improves causal identification but may reduce real-world presentation variation.
* Self-family preference should not be claimed for any family without both source plans and judge models from that family.

## Documentation and reproducibility notes

Documentation should reflect the current state:

* The initial 30-pair TrailTraining-score match was rejected.
* Structural matching replaced the presentation-contaminated primary score.
* Session-signature duplicates are audited and used in sensitivity analysis rather than hard-dropped before primary matching.
* The primary full-study input is `frozen_primary_v1/eval_manifest/pairwise_eval_manifest.jsonl`.
* The primary judge view is `canonical_masked` with `canonical_masked_scrubbed_v1`.
* Full-study launch requires exactly 10,000 pairwise judgment records.
* Full judging must be blocked if the manifest does not contain exactly 250 pairs x 4 judges x 5 runs x 2 orders.
* The older 30-pair matched set and any non-scrubbed manifest are not primary study inputs.

Known errors to prevent:

* `ModuleNotFoundError: No module named 'pydantic'`
* `ModuleNotFoundError: No module named 'trailtraining'`
* broken `conda activate` on HPC
* model not found in cache
* accidental deletion of cached models
* using the wrong Python interpreter
* matching on only 30 pairs and proceeding anyway
* treating TrailTraining quality score as structural-only without proof
* mixing incompatible generation/evaluation temperature conditions in one directory
* proceeding with fewer than exactly 10,000 pairwise records for the full study
* generating judge prompts with source-identifying labels
* configuring or assuming non-Qwen/Gemma judges without explicit support and approval

