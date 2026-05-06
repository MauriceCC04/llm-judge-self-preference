Data Processing Log — llm-judge-self-preference

This document records the data-processing and experiment-preparation steps taken during the Qwen/Gemma training-plan judge-bias study. It is intended to be appended as the study continues.

Project objective

Study whether LLM judges systematically prefer LLM-generated training plans over programmatically generated training plans when deterministic plan quality is controlled.

Current frozen design:

* LLM source families:
    * Qwen/Qwen2.5-7B-Instruct
    * google/gemma-3-4b-it
* Programmatic control plans generated from structural variation logic.
* Shared explainer model used for narrative/explanation fields where applicable.
* Matching target: at least 250 structurally matched LLM-vs-programmatic pairs.
* Full pairwise evaluation target:
    * 250 matched pairs × 4 judge models × 5 repeats × 2 AB/BA orders = 10,000 pairwise judgment records.

Repository and environment notes

Working repository

HPC repository path:

/mnt/beegfsstudents/home/3202029/llm-judge-self-preference

Related TrailTraining repository path:

/mnt/beegfsstudents/home/3202029/trailtraining

Required environment variables for scripts that import both repositories:

export REPO_ROOT=/mnt/beegfsstudents/home/3202029/llm-judge-self-preference
export TRAILTRAINING_REPO=/mnt/beegfsstudents/home/3202029/trailtraining
export PYTHONPATH="${REPO_ROOT}:${TRAILTRAINING_REPO}/src:${PYTHONPATH:-}"

Important HPC environment issue observed:

* conda activate judge-bias failed in interactive login shell with:

CondaError: Run 'conda init' before 'conda activate'

Safer workaround:

* Use the environment Python directly when possible:

/home/3202029/.conda/envs/judge-bias/bin/python

or ensure the correct module and shell initialization are loaded in SLURM jobs.

Git state observed early in the session

Initial repository state showed local untracked outputs and divergence from origin/main:

?? err/
?? out/
?? plans_qwen_llm_fix4_n32/
?? slurm-487111.out
?? slurm-487112.out
?? slurm-487151.out

Divergence:

> bbffb8c (origin/main, origin/HEAD) more
> 491b219 Fix LLM generation conditioning and validation guards
< 02e9082 (HEAD -> main) Fix LLM generation conditioning and validation guards

Interpretation:

* Local branch and remote branch had distinct commits.
* HPC output directories and SLURM logs were untracked.
* Generated artifacts needed to be treated separately from source-control changes.

Qwen source-generation processing

Early Qwen pilot/top-up observations

A Qwen pilot/top-up generation run produced valid plans but also repeated validation failures for understructured cells, especially:

ab_A4__r_low__rc_high__ph_base

Common failure causes:

understructured:no_active_sessions
understructured:missing_long_run
understructured:missing_quality_for_A4

A successful advanced pilot later produced 8 valid plans across 8 advanced cells.

Example successful advanced pilot audit:

plans 8
provenance 8
leaks []
bad []

Observations:

* Qwen generation could succeed, but some low/high/base and A4 cells required extra attempts.
* Final validation guards were working: understructured plans were rejected and logged.

Full Qwen generation

Full Qwen generation was launched for all 32 fixture cells.

Important initial issue:

* A first check from the home directory showed no full_qwen/plans directory because commands were run from the wrong working directory.
* After checking the correct repository path, outputs were found.

The full Qwen run initially produced:

total provenance: 190
cells: 32
by_band: {'A1': 46, 'A2': 48, 'A3': 48, 'A4': 48}
non-6 cells:
  ab_A1__r_low__rc_high__ph_base: 4/6
missing_cells_csv:
ab_A1__r_low__rc_high__ph_base

A top-up was run for the missing Qwen cell:

ab_A1__r_low__rc_high__ph_base

The top-up overgenerated the cell to 10 plans. Extra plans were then quarantined.

Plans kept:

ab_A1__r_low__rc_high__ph_base__qwen2.5_7b_instruct__src_t070__exp_t000__s013.json
ab_A1__r_low__rc_high__ph_base__qwen2.5_7b_instruct__src_t070__exp_t000__s015.json
ab_A1__r_low__rc_high__ph_base__qwen2.5_7b_instruct__src_t070__exp_t000__s016.json
ab_A1__r_low__rc_high__ph_base__qwen2.5_7b_instruct__src_t070__exp_t000__s022.json
ab_A1__r_low__rc_high__ph_base__qwen2.5_7b_instruct__src_t070__exp_t000__s028.json
ab_A1__r_low__rc_high__ph_base__qwen2.5_7b_instruct__src_t070__exp_t000__s039.json

Quarantined extras:

ab_A1__r_low__rc_high__ph_base__qwen2.5_7b_instruct__src_t070__exp_t000__s047.json
ab_A1__r_low__rc_high__ph_base__qwen2.5_7b_instruct__src_t070__exp_t000__s049.json
ab_A1__r_low__rc_high__ph_base__qwen2.5_7b_instruct__src_t070__exp_t000__s053.json
ab_A1__r_low__rc_high__ph_base__qwen2.5_7b_instruct__src_t070__exp_t000__s060.json

Quarantine destination:

artifacts/gen_src_t070_exp_t000/full_qwen/plans/quarantine_overgenerated_ab_A1__r_low__rc_high__ph_base_20260504_195455

Final Qwen corpus audit:

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

Conclusion:

* Qwen source corpus is balanced and complete: 32 cells × 6 plans = 192 plans.
* Sidecar provenance files are complete.
* Basic leakage and validity checks passed.

Gemma source-generation processing

Hugging Face cache and access issue

Initial Gemma model caching failed because the Hugging Face token was not configured.

Observed failure:

GatedRepoError: 401 Client Error
Access to model google/gemma-2-9b-it is restricted. You must have access to it and be authenticated to access it. Please log in.

After setting the Hugging Face token, model caching succeeded:

Downloading google/gemma-2-9b-it ...
Done google/gemma-2-9b-it
Downloading google/gemma-2-2b-it ...
Done google/gemma-2-2b-it
Model cached: google/gemma-2-9b-it
Model cached: google/gemma-2-2b-it

Later, because of quota constraints, some cached Qwen models were removed. This caused later failures:

Model not found in cache: Qwen/Qwen2.5-3B-Instruct
Model not found in cache: google/gemma-3-4b-it

Important note:

* The explainer model cache must remain available if generation depends on it.
* Avoid deleting cached models unless the next job’s source and explainer models are known and re-cached.

Gemma pilot generation

Gemma pilot used:

source model: google/gemma-3-4b-it
explainer model: Qwen/Qwen2.5-3B-Instruct

The advanced pilot produced 8 valid plans over 8 selected A3/A4 cells.

Pilot result:

Generated: 8
plans 8
provenance 8
leaks []
bad []

Observed rejected attempts included:

understructured:missing_long_run
understructured:collapsed_easy_template

Conclusion:

* Gemma 3 generation was viable, but some cells showed repetitive/easy-template tendencies.

Full Gemma generation and top-ups

Initial full Gemma generation produced:

plans: 180
provenance: 180
total: 180
cells: 30
by_band: {'A1': 48, 'A2': 36, 'A3': 48, 'A4': 48}
cell count min/max: 6 6

Missing cells:

ab_A2__r_high__rc_high__ph_base
ab_A2__r_high__rc_high__ph_peak

Additional issue discovered:

non_7_day_plans [('ab_A3__r_low__rc_high__ph_base__gemma_3_4b_it__src_t070__exp_t000__s003.json', 18)]
count 1

A3 replacement was run separately.

A2 high/high base and peak top-ups were difficult. One run with 120 attempts produced only:

ab_A2__r_high__rc_high__ph_base: accepted 1/6
ab_A2__r_high__rc_high__ph_peak: accepted 0/6

Common failure:

understructured:missing_long_or_quality_for_A2_high_high

Later top-up strategy:

* A2 base: exactly 5 more plans.
* A2 peak: exactly 6 plans.
* A3 replacement run separately.

Final Gemma corpus audit:

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

Conclusion:

* Gemma 3 source corpus is now balanced and complete: 32 cells × 6 plans = 192 plans.
* Known problematic A2 and A3 cells were corrected through targeted top-up/replacement.

Programmatic-plan generation processing

Initial programmatic generation failure

First full programmatic run failed because the sampler attempted to auto-fit priors from too few LLM plans:

[ABORT] Refusing to auto-fit sampler priors from only 0 LLM plans (< 30).
Provide SAMPLER_CONFIG, generate more LLM plans, or set ALLOW_TINY_PRIOR_FIT=1 to override.

Existing sampler configs found:

./artifacts/pilot16/sampler_config.json
./artifacts/pilot_yield/sampler_config.json
./artifacts/postfix_llm_sanity/sampler_config.json

The pilot_yield sampler config was based on 24 plans; pilot16 and postfix_llm_sanity were based on 8 plans.

Interpretation:

* The programmatic generation script needed an explicit sampler config or a corrected source path for fitting priors.
* Auto-fitting from 0 LLM plans was correctly blocked.

Programmatic generation before final fix

A programmatic run produced 640 plans and 640 provenance files, but initial quality audits showed narrative/session contradictions:

generic_titles 3224
title_session_mismatches 35
rest_active_contradictions 24
unique_session_sequences 106 / 640

Examples included:

* session_type was aerobic but text said Long session.
* session_type was long but text said Rest day.
* active sessions described as rest days.

Conclusion:

* The first programmatic corpus was structurally present but not acceptable due to source-identifying or contradictory narrative artifacts.
* Programmatic explanation/title generation needed correction.

Fixed programmatic generation

A fixed programmatic job completed successfully:

JOBID=489430
State: COMPLETED
Elapsed: 04:32:07
Generated: 640
Skipped: 0

The generation logs still included some failed attempts due to structured explanation parse failures, for example:

Unterminated string starting at ...
Expecting ',' delimiter ...

However, retries succeeded and the final accepted corpus was complete.

Final programmatic corpus audit:

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

Structural summary:

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

Conclusion:

* Programmatic corpus is now complete and internally consistent by the applied checks.
* It contains 20 plans per cell, across 32 cells, totaling 640 plans.

Current corpus state

Final corpus audits:

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

Total available plans:

Qwen LLM:        192
Gemma 3 LLM:     192
Programmatic:    640
Total:          1024

Matching-pool construction

A matching pool was constructed by copying Qwen, Gemma 3, and programmatic plan files plus provenance sidecars into:

artifacts/gen_src_t070_exp_t000/matching_pool/plans

Result:

copied_plans 1024
pool artifacts/gen_src_t070_exp_t000/matching_pool/plans

Matching attempt and scoring failures

Initial matching command:

python cli.py match \
  --plans artifacts/gen_src_t070_exp_t000/matching_pool/plans \
  --output artifacts/gen_src_t070_exp_t000/matching_pool/matched_pairs.json \
  --allow-mixed-generation-conditions

First failure:

ModuleNotFoundError: No module named 'pydantic'

After switching to an environment where pydantic was importable, matching failed again:

match.pair.ScoringError: 1024 plans failed deterministic scoring; see artifacts/gen_src_t070_exp_t000/matching_pool/scoring_failures.json

Inspecting scoring_failures.json showed the real cause:

ModuleNotFoundError: No module named 'trailtraining'

Interpretation:

* The scorer was being run with an environment that had pydantic but not the trailtraining package on PYTHONPATH.
* This was an environment/path issue, not a plan-validity issue.

Required environment fix:

export REPO_ROOT=/mnt/beegfsstudents/home/3202029/llm-judge-self-preference
export TRAILTRAINING_REPO=/mnt/beegfsstudents/home/3202029/trailtraining
export PYTHONPATH="${REPO_ROOT}:${TRAILTRAINING_REPO}/src:${PYTHONPATH:-}"

Matching output after environment fix

After the environment issue was corrected, matching completed but yielded too few pairs:

Pairs yielded:    30  (target: 256)
Mean score gap:   0.000
Coverage ratio:   0.117
Coverage OK:      False

Matched-pair output was saved to:

artifacts/gen_src_t070_exp_t000/matching_pool/matched_pairs.json

Additional audit files:

artifacts/gen_src_t070_exp_t000/matching_pool/matching_audit.json
artifacts/gen_src_t070_exp_t000/matching_pool/matching_prefilter_audit.json

Conclusion:

* The 30-pair matched set is not usable for the full study.
* Full pairwise judging must not be run from this matched set.

Matching prefilter audit

The matching prefilter audit showed:

n_input: 1024
n_kept: 936
n_dropped: 88
drop_reasons: session_signature_duplicate=88
exact_text_duplicate_groups: none
invalid_or_contradictory: none

Interpretation:

* Prefiltering removed some duplicate session signatures.
* No exact text duplicates were found.
* No invalid or contradictory plans were flagged by the prefilter.
* The prefilter is not the main cause of low pair yield.

Main current blocker: score non-overlap

Observed deterministic score distributions by source:

Qwen

n: 192
min: 10
p25: 97
median: 100
p75: 100
max: 100
bins: mostly 97 and 100

Gemma 3

n: 192
min: 10
p25: 85
median: 100
p75: 100
max: 100
bins: mostly 85, 97, 100

Programmatic

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

Relaxing matching constraints did not solve the issue:

* With same score bin required:
    * tolerance 0.25 to 3.0 produced only about 31 pairs.
* Without same score bin:
    * tolerance 0.25 to 2.0 produced only about 31 pairs.
    * tolerance 3.0 produced about 35 pairs.

Interpretation:

* The problem is deterministic score non-overlap, not same-bin matching.
* Most LLM plans score 97–100.
* Most programmatic plans score around 30.
* Existing deterministic score appears unsuitable as the primary matching score.

Working hypothesis about the scoring issue

The current deterministic scorer may be contaminated by presentation/explanation quality rather than measuring only structural plan quality.

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

Why this is a problem:

* The study needs to control structural quality, not presentation quality.
* If the matching score rewards LLM-style prose, matching will either fail or condition on a source-identifying confound.
* Equal score under a presentation-contaminated scorer would not isolate structural quality.

Required fix:

* Implement or validate a source-neutral structural score for matching.
* Structural score should use only fields such as session type, duration, rest/hard flags, long-run count, hard-day spacing, rest-day spacing, total load, and fixture/cell constraints.
* Structural score should explicitly exclude narrative, explanation, citations, provenance source labels, and model-family identifiers.

Proposed next processing step: structural matching repair

Before full judging, complete the following:

1. Inspect current matcher and deterministic scoring code.
2. Add a diagnostic proving whether current score changes when only presentation fields are edited.
3. Implement a source-neutral structural score.
4. Update matcher or add alternate structural-matching path.
5. Produce matching diagnostics:
    * structural score distribution by arm/model
    * per-cell overlap
    * possible pair counts by tolerance
    * final pair coverage by cell and source family
    * feature balance across matched pairs
6. Add tests:
    * structural score ignores prose/title changes
    * structural score changes when structural fields change
    * no source labels leak into judge-facing pair artifacts
    * pair generation is reproducible with seed
7. Rerun matching.
8. Do not proceed to full pairwise judging unless at least 250 structurally matched pairs are available.

Full-study launch gate

Full pairwise evaluation should remain blocked until all gates pass:

matched_pairs >= 250
expected_pairwise_documents >= 10,000
exactly 4 valid Qwen/Gemma judge models configured
5 repeats configured
both AB and BA orderings configured
Qwen and Gemma source plans represented in matched pairs
structural matching audit passes
source masking audit passes
no source-identifying labels in judge-facing prompts

Expected full-study count:

250 matched pairs × 4 judges × 5 repeats × 2 positions = 10,000 pairwise judgments

Sync/back-up notes

The user wanted to sync HPC outputs back to laptop before committing.

Items to sync:

* source code changes
* scripts
* docs
* artifacts/gen_src_t070_exp_t000/full_qwen/plans
* artifacts/gen_src_t070_exp_t000/full_gemma3/plans
* artifacts/gen_src_t070_exp_t000/full_programmatic/plans
* matching pool audits
* relevant SLURM logs in out/ and err/
* sampler configs and run configs
* not necessarily large model caches

Recommended principle:

* Commit code and lightweight metadata/audits.
* Store large generated corpora and raw outputs according to project artifact policy, not necessarily directly in Git.

Open issues and TODOs

Immediate

* Inspect current matching and scoring implementation.
* Prove whether current deterministic scorer is presentation-sensitive.
* Implement structural score.
* Rerun matching using structural score.
* Achieve at least 250 matched LLM-vs-programmatic pairs.

Before judging

* Build judge-facing pair artifacts with source masking.
* Randomize AB/BA order and record order metadata externally.
* Confirm exactly 4 active Qwen/Gemma judge models.
* Confirm full-study expected count equals 10,000 pairwise judgment records.
* Add launch gate that refuses underpowered or source-leaky evaluation.

Analysis pipeline

* Compute LLM win rate overall.
* Compute win rate by judge model.
* Compute win rate by judge family and size.
* Compute source-family effects.
* Compute self-family effect only where source and judge family overlap.
* Audit AB/BA order bias.
* Compute bootstrap confidence intervals.
* Add score-gap and fixture/band sensitivity analyses.

Documentation

* Update README.
* Update HPC runbook.
* Update preregistration/methods document.
* Document all known HPC environment pitfalls.
* Add reproducibility commands.
* Add report scaffold with placeholders for actual results.

Append-only processing log

Use this section for future entries.

YYYY-MM-DD — Step title

Objective:

Command(s):

# commands here

Inputs:

Outputs:

Audit checks:

# counts/checks here

Interpretation:

Decision / next step:

Risks / caveats: