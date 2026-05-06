# Data Processing Log — `llm-judge-self-preference`

This document records what data was produced, how it was processed, why each processing decision was made, what audits were run, and what remains to be done for the Qwen/Gemma LLM-judge self-preference study.

The current study asks:

> Controlling for deterministic structural plan quality, do LLM judges systematically prefer LLM-generated training plans over programmatically generated plans, and does this effect vary by judge model family, model size, rubric, or whether the plan matches the judge's own family?

The central validity rule is:

> Equal deterministic score is necessary but not sufficient. Matching must control source-neutral plan structure while avoiding source-identifying presentation artifacts such as prose richness, citations, claim attributions, data notes, title style, source/model labels, or file/provenance strings.

---

## Current frozen primary input

The primary evaluation input is now:

```text
artifacts/gen_src_t070_exp_t000/frozen_primary_v1/eval_manifest/pairwise_eval_manifest.jsonl
```

The primary judge-facing view is:

```text
canonical_masked
canonical_masking_version = canonical_masked_scrubbed_v1
```

The full-study design is exactly:

```text
250 matched pairs x 4 judges x 5 runs x 2 AB/BA positions = 10,000 pairwise judgment records
```

The 10,000-document requirement means 10,000 pairwise evaluation records. It does not mean 10,000 generated training plans.

---

## Current frozen design

### Source plan families

LLM-source plan families:

```text
Qwen/Qwen2.5-7B-Instruct
google/gemma-3-4b-it
```

Programmatic controls:

```text
programmatic structural generator with varied hard-day count, rest placement, long-run/quality placement, readiness, recovery capability, race phase, and related plan-logic constraints
```

### Judge models

Configured local judge names in the scrubbed manifest:

```text
qwen_7b_judge
qwen_14b_judge
gemma_4b_judge
gemma_12b_judge
```

Judge-family balance in the final manifest:

```text
qwen judge-family records: 5,000
gemma judge-family records: 5,000
```

Self-family metadata is preserved outside the judge prompt:

```text
self_family_match records: 5,000 true / 5,000 false
```

---

## Processing-stage overview: what, how, why, audit result

| Stage | What was done | How it was done | Why it was done | Audit result / status |
|---|---|---|---|---|
| Environment setup | Fixed import/runtime assumptions. | Used explicit `REPO_ROOT`, `TRAILTRAINING_REPO`, and `PYTHONPATH`; preferred direct env Python on HPC. | Avoided `pydantic`, `trailtraining`, and broken `conda activate` failures. | Environment issues diagnosed; HPC preflight commands documented. |
| Qwen generation | Produced balanced Qwen source plans. | Ran full generation, audited per cell, topped up missing cell, quarantined extras. | Needed 6 Qwen plans per 32 fixture cells. | 192 plans, 192 provenance, 32 cells, 6/cell. |
| Gemma generation | Produced balanced Gemma 3 source plans. | Fixed model-cache/auth issues, ran full generation, targeted difficult A2/A3 top-ups/replacement. | Needed 6 Gemma plans per 32 fixture cells. | 192 plans, 192 provenance, 32 cells, 6/cell. |
| Programmatic generation | Produced programmatic controls. | Initial generator audit found contradictions; fixed generation/explainer logic; reran full job. | Needed structurally varied non-LLM controls without obvious empty-text artifacts. | 640 plans, 640 provenance, 32 cells, 20/cell; no generic titles or rest-active contradictions in final audit. |
| Matching-pool construction | Combined Qwen, Gemma, and programmatic corpora. | Copied plan JSON and provenance sidecars into `matching_pool/plans`. | Created one source directory for scoring and matching. | 1,024 plans total. |
| Initial matching attempt | Tried existing matcher and TrailTraining quality score. | Ran `cli.py match`; fixed env failures; inspected matching and score audits. | Needed to test whether existing scorer/matcher could construct the full study. | Only ~30 pairs; rejected. |
| Score diagnosis | Diagnosed why matching failed. | Compared score distributions by source and relaxed tolerances. | Determined whether low coverage was due to matching constraints or score non-overlap. | Programmatic plans clustered near 30; LLM plans near 97-100; score unsuitable. |
| Structural-score repair | Replaced primary matching score with source-neutral structural score. | Implemented structural score excluding prose/presentation fields; kept same-cell constraints. | Matching on presentation-contaminated score would condition on the confound under study. | Structural overlap recovered; 250 matched pairs obtained. |
| Duplicate handling | Changed session-signature duplicates from hard drops to audit/sensitivity. | v3 repair retained duplicates for primary matching but reports them for robustness checks. | Hard-dropping duplicate signatures reduced coverage below the full-study gate despite no exact text duplicates. | Coverage restored to 250 pairs. |
| v4 matching selection | Improved pair selection quality. | Used target-cardinality matching with structural distance and soft-caliper penalties. | Avoid arbitrary feasible matches and reduce structural tails where possible. | 250 pairs with low structural score gap; remaining feature-gap tails documented. |
| Eval manifest | Built the 10,000-record pairwise manifest. | Crossed 250 pairs with 4 judges, 5 runs, and AB/BA order; wrote judge-input files. | Needed one row per actual judge-facing pairwise comparison. | 10,000 records, 10,000 judge input JSON files. |
| Canonical masking | Removed source/provenance-heavy raw plan fields from judge prompts. | Used `canonical_masked` view containing only normalized day structure plus short summaries. | Prevented source leakage and reduced presentation-style confounds. | Explicit Qwen/Gemma/programmatic/source/provenance/model labels absent. |
| Scrubbing | Removed residual citation/signal/readiness artifacts. | Non-destructive post-processing of only `purpose_summary` and `workout_summary`; wrote `eval_manifest_scrubbed_v1`. | Residual strings were not source labels but could act as data/provenance cues. | Verifier passed: 10,000 rows, 250 pairs, 4 judges, no leak patterns. |
| Freezing | Froze primary manifest and inputs. | Copied scrubbed manifest, judge inputs, matched pairs, and audits to `frozen_primary_v1`; generated SHA256 checksums. | Prevent accidental input drift before HPC judging. | Frozen primary input prepared. |
| Smoke manifest | Created 8-record smoke test. | `head -n 8` from frozen primary manifest. | Test model loading, `compare_plans`, parsing, and metadata before full GPU job. | Smoke manifest prepared; actual judging should run on HPC/GPU. |

---

## Repository and environment processing

### What was done

The run environment was diagnosed and documented because several data-processing steps depend on both this repository and the external TrailTraining package.

### How it was done

The expected HPC paths were established:

```bash
export REPO_ROOT=/mnt/beegfsstudents/home/3202029/llm-judge-self-preference
export TRAILTRAINING_REPO=/mnt/beegfsstudents/home/3202029/trailtraining
export PYTHONPATH="${REPO_ROOT}:${TRAILTRAINING_REPO}/src:${PYTHONPATH:-}"
```

A safer Python invocation was preferred over relying on interactive `conda activate`:

```bash
/home/3202029/.conda/envs/judge-bias/bin/python
```

### Why it was done

Earlier processing attempts failed with environment errors:

```text
ModuleNotFoundError: No module named 'pydantic'
ModuleNotFoundError: No module named 'trailtraining'
CondaError: Run 'conda init' before 'conda activate'
```

These were not plan-data problems. They were interpreter/PYTHONPATH problems that could falsely appear as scoring or matching failures.

### Audit / result

The data-processing documentation now treats environment setup as a first-class reproducibility dependency. Full matching and judging should not be interpreted unless the same Python/path setup is confirmed.

---

## Qwen source-plan processing

### What was done

A balanced Qwen source corpus was generated for all 32 fixture cells.

### How it was done

The full generation was run, then audited by cell. One cell was underfilled:

```text
ab_A1__r_low__rc_high__ph_base
```

A targeted top-up was run for that cell. The top-up overgenerated the cell to 10 plans. Six were retained and extras were quarantined rather than silently deleted.

Quarantined extras were moved to:

```text
artifacts/gen_src_t070_exp_t000/full_qwen/plans/quarantine_overgenerated_ab_A1__r_low__rc_high__ph_base_20260504_195455
```

### Why it was done

The study design requires balanced coverage across fixture cells. Keeping 6 plans per cell avoids overrepresenting a particular athlete/readiness/recovery/race-phase cell. Quarantining extras preserves auditability while preventing accidental inclusion.

### Audit / result

Final Qwen corpus:

```text
plans: 192
provenance: 192
cells: 32
by_band: A1=48, A2=48, A3=48, A4=48
cell count min/max: 6/6
missing sidecars: none
leaks: none
bad plans: none
```

### Caveats

Some Qwen attempts failed validation for understructured plans, including no active sessions, missing long run, or missing quality session in advanced cells. These failures are useful evidence that validation guards were active.

---

## Gemma source-plan processing

### What was done

A balanced Gemma 3 source corpus was generated for all 32 fixture cells.

### How it was done

Generation used:

```text
source model: google/gemma-3-4b-it
explainer model: Qwen/Qwen2.5-3B-Instruct where applicable
```

Initial model access/cache issues were fixed by configuring Hugging Face access and ensuring the relevant models were cached. Difficult cells were topped up:

```text
ab_A2__r_high__rc_high__ph_base
ab_A2__r_high__rc_high__ph_peak
```

A bad A3 plan with an incorrect day count was replaced.

### Why it was done

Gemma source plans are needed to test whether judge preference varies by source family and to define Gemma self-family analysis when Gemma judges are present. Targeted top-ups preserve the same 6-per-cell balance used for Qwen.

### Audit / result

Final Gemma corpus:

```text
plans: 192
provenance: 192
cells: 32
by_band: A1=48, A2=48, A3=48, A4=48
cell count min/max: 6/6
missing sidecars: none
bad lengths: none
leaks: none
bad plans: none
```

### Caveats

Gemma generation showed difficulty in some A2 high-readiness/high-recovery cells, often failing understructure checks such as missing long or quality work. These failures were handled by retries/top-ups rather than weakening validation.

---

## Programmatic control-plan processing

### What was done

A complete programmatic control corpus was generated with 20 plans per fixture cell.

### How it was done

The first full programmatic attempt exposed generation/audit problems:

```text
generic_titles: high
title_session_mismatches: present
rest_active_contradictions: present
```

The programmatic generation and explanation/title logic were fixed, then the full job was rerun.

### Why it was done

Programmatic plans provide the non-LLM control arm. They must vary structural properties while avoiding obvious artifacts such as empty text, generic titles, rest-active contradictions, or source-identifying format differences. Otherwise judges might prefer or reject them for artifacts rather than plan quality.

### Audit / result

Final programmatic corpus:

```text
plans: 640
provenance: 640
cells: 32
by_band: A1=160, A2=160, A3=160, A4=160
cell count min/max: 20/20
bad lengths: none
leaks: none
bad JSON: none
generic_titles: 0
title_session_mismatches: 0
rest_active_contradictions: 0
unique session sequences: 134/640
```

Structural distribution summary:

```text
A1 duration mean: 232.34
A2 duration mean: 263.48
A3 duration mean: 292.21
A4 duration mean: 323.52
A1 hard-day mean: 0.72
A2 hard-day mean: 0.75
A3 hard-day mean: 0.99
A4 hard-day mean: 1.07
```

### Caveats

The programmatic corpus still has a finite number of unique session signatures. Session-signature duplication is not treated as invalid, but it is retained for duplicate/sensitivity analyses.

---

## Matching-pool construction

### What was done

The final Qwen, Gemma, and programmatic corpora were combined into a single matching pool.

### How it was done

Plan JSON files and provenance sidecars were copied into:

```text
artifacts/gen_src_t070_exp_t000/matching_pool/plans
```

### Why it was done

A single pool gives the matcher one directory containing both treatment arms and all provenance required for same-cell matching. It also produces a single audit trail for prefiltering, scoring, and matching.

### Audit / result

Matching pool:

```text
Qwen: 192
Gemma 3: 192
Programmatic: 640
Total: 1024
cells: 32
```

---

## Initial matching attempt and failure diagnosis

### What was done

The existing matcher was first run using the repository's original deterministic score path.

### How it was done

Command used:

```bash
python cli.py match \
  --plans artifacts/gen_src_t070_exp_t000/matching_pool/plans \
  --output artifacts/gen_src_t070_exp_t000/matching_pool/matched_pairs.json \
  --allow-mixed-generation-conditions
```

After environment fixes, the matcher produced only about 30 matched pairs.

### Why it was done

Before adding new scoring logic, the existing pipeline needed to be tested. If it could already produce valid same-cell structural matches, no new scorer would be needed.

### Audit / result

Initial matching result:

```text
Pairs yielded: 30
Target: 256
Coverage ratio: 0.117
Coverage OK: false
```

Prefilter audit:

```text
n_input: 1024
n_kept: 936
n_dropped: 88
drop reason: session_signature_duplicate=88
exact text duplicate groups: none
invalid or contradictory: none
```

Score distributions showed the true blocker:

```text
Qwen: mostly 97-100
Gemma: mostly 85, 97, 100
Programmatic: mostly around 30
```

Relaxing score tolerance did not solve the problem.

### Why this result was rejected

The low pair count was not due to invalid plans. It was due to non-overlap in the original deterministic score. The original score appeared likely to reward presentation/explanation quality as well as structure. Matching on that score would either fail or condition on exactly the presentation style confound the study is trying to evaluate.

---

## Structural-score matching repair

### What was done

A source-neutral structural matching score replaced the original TrailTraining quality score as the primary matching score.

### How it was done

A structural scorer was implemented and used by the matcher. The report includes:

```text
structural score
component scores
violations
extracted structural features
score version
explicit note that presentation fields are excluded
```

The structural score uses source-neutral features such as:

```text
session_type
duration_minutes
is_rest_day
is_hard_day
target_intensity where normalized
weekly total duration
long-run count
hard-day count
quality-day count
rest-day count
active-day count
max day duration
plan length
fixture/cell constraints
```

The structural score explicitly excludes:

```text
title wording
workout prose richness
purpose prose richness
rationale/explanation fields
citations
claim_attributions
data_notes verbosity
source model name
generation arm
file naming artifacts
```

### Why it was done

The causal target is source preference conditional on structural plan quality. Presentation quality must not define the matching score, because that would either prevent matches or control away/source-code the very confound being studied.

### Same-cell matching constraints

Pairs must remain within the same fixture/cell when possible. The primary matching kept:

```text
same fixture_id
same athlete_band
same readiness
same recovery_capability
same race_phase
same style
same plan_days
```

### Duplicate handling decision

Session-signature duplicates were changed from hard drops to audit-only for primary matching.

Why:

* Hard-dropping session-signature duplicates reduced coverage below 250 pairs.
* There were no exact text duplicate groups.
* Duplicate signatures are a robustness/sensitivity issue, not a reason to destroy the primary sample size.

Planned sensitivity:

* Repeat analysis excluding duplicate-heavy cells or duplicated session signatures.

### Audit / result

After repair:

```text
kept records: 1023
dropped records: 1
candidate pairs: 258
selected pairs: 250
Qwen-source pairs: 130
Gemma-source pairs: 120
```

The 30-pair blocker was resolved.

---

## Matched-pair construction and audit

### What was done

A final 250-pair matched set was selected from the matching pool.

### How it was done

Matching used:

```text
same-cell eligibility
structural score tolerance
weighted structural feature distance
target-cardinality selection
soft-caliper penalties in v4 selection
source-family balancing where feasible
```

Final output:

```text
artifacts/gen_src_t070_exp_t000/matching_pool/matched_pairs.json
```

### Why it was done

The full study requires exactly 250 matched pairs to reach 10,000 pairwise judgment records with the configured judges/repeats/orders.

### Audit / result

Final matched set:

```text
matched pairs: 250
Qwen-source pairs: 130
Gemma-source pairs: 120
fixture cells covered: 32/32
mean structural score gap: approximately 0.255
max structural score gap: 2.0
```

### Known caveat

Structural score gaps are low, but individual feature gaps still have tails:

```text
abs_total_minutes gap p95: approximately 186.9
abs_total_minutes gap max: approximately 312.0
abs_max_day gap p95: approximately 63.0
abs_max_day gap max: approximately 166.0
```

This does not invalidate the run, but it requires sensitivity analyses that exclude high-gap pairs.

---

## Pairwise evaluation manifest construction

### What was done

A pairwise evaluation manifest was built from the matched pairs.

### How it was done

The manifest crossed:

```text
250 matched pairs
4 judge models
5 run indices
2 positions/orders: AB and BA
```

Command pattern:

```bash
python cli.py build-eval-manifest \
  --plans artifacts/gen_src_t070_exp_t000/matching_pool/plans \
  --pairs artifacts/gen_src_t070_exp_t000/matching_pool/matched_pairs.json \
  --output artifacts/gen_src_t070_exp_t000/matching_pool/eval_manifest \
  --seed 20260506 \
  --judge-view canonical_masked \
  --write-judge-inputs
```

### Why it was done

Each manifest row corresponds to one actual judge-facing pairwise comparison instance. This is the unit of the 10,000-document requirement.

### Audit / result

Canonical masked manifest:

```text
records: 10,000
judge input JSON files: 10,000
unique pairs: 250
records per pair: 40
AB: 5,000
BA: 5,000
runs 0-4: 2,000 each
four judges: 2,500 records each
```

---

## Canonical masking

### What was done

The judge-facing plan representation was changed from raw plan JSON to `canonical_masked`.

### How it was done

The judge-facing JSON contains only:

```text
plan_a
plan_b
```

Each plan day contains only:

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

The canonical view removes or excludes:

```text
source model labels
source-family labels
generation arm labels
file paths
provenance metadata
citations
claim_attributions
data_notes
long rationale/explanation fields
raw fixture metadata
```

### Why it was done

Judges should compare plan content, not metadata, file names, model names, or explanation/citation richness. Canonical masking reduces direct source leakage and presentation-style confounds.

### Audit / result

Explicit source/model/provenance labels were not found in the canonical masked judge inputs:

```text
Qwen: 0
Gemma: 0
programmatic: 0
source/model/provenance labels: 0
```

---

## Scrubbed canonical manifest

### What was done

A non-destructive scrubber was applied to the canonical masked manifest to remove residual citation/signal/readiness wording from judge-facing summaries.

### How it was done

Only text summary fields were edited:

```text
purpose_summary
workout_summary
```

The scrubber removed or normalized:

```text
Cite:
Citations:
signal_id
load.last7
readiness status
current readiness
```

The scrubbed manifest was written to:

```text
artifacts/gen_src_t070_exp_t000/matching_pool/eval_manifest_scrubbed_v1/
```

### Why it was done

The residual artifacts were not direct Qwen/Gemma/programmatic labels, but they looked like provenance/data-note artifacts. Removing them makes the primary judge prompt cleaner without changing structural fields or matched-pair metadata.

### What was preserved

The scrubber preserved:

```text
record_id
pair_id
fixture_id
judge
judge_model_family
run
order
left/right role metadata
source_model_family outside the prompt
self_family_match outside the prompt
structural_score_gap
feature_gaps
all structural day fields
```

### Audit / result

Verifier output after scrub:

```text
records: 10000
detected judge key: judge
pairs: 250 min/max records per pair: 40 40
judges: qwen_7b_judge=2500, qwen_14b_judge=2500, gemma_4b_judge=2500, gemma_12b_judge=2500
orders: AB=5000, BA=5000
runs: 0=2000, 1=2000, 2=2000, 3=2000, 4=2000
views: canonical_masked=10000
PASS: scrubbed manifest is ready for judging
```

Conclusion:

```text
eval_manifest_scrubbed_v1 is the primary evaluation input.
```

---

## Frozen primary input

### What was done

The scrubbed manifest and matched pairs were frozen into a primary input directory.

### How it was done

Commands:

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

### Why it was done

Freezing prevents accidental drift between the local verified input and the HPC execution input. Checksums make transfer verification possible.

### Result

Primary frozen manifest:

```text
artifacts/gen_src_t070_exp_t000/frozen_primary_v1/eval_manifest/pairwise_eval_manifest.jsonl
```

---

## Smoke manifest

### What was done

An 8-record smoke manifest was created.

### How it was done

Command:

```bash
mkdir -p artifacts/gen_src_t070_exp_t000/frozen_primary_v1/smoke_test

head -n 8 artifacts/gen_src_t070_exp_t000/frozen_primary_v1/eval_manifest/pairwise_eval_manifest.jsonl \
  > artifacts/gen_src_t070_exp_t000/frozen_primary_v1/smoke_test/smoke_manifest.jsonl
```

### Why it was done

The smoke manifest is a cheap test of the actual judging harness before launching the full 10,000-record GPU job.

It should verify:

```text
judge model loads from cache
compare_plans executes
winner parses
record_id/pair_id/judge/run/order are preserved
errors are logged instead of crashing the whole job
```

### Result

Smoke manifest path:

```text
artifacts/gen_src_t070_exp_t000/frozen_primary_v1/smoke_test/smoke_manifest.jsonl
```

Creating the smoke manifest is only a file operation. It prints nothing by default.

---

## HPC handoff

### What to transfer

Transfer the frozen primary input and patched code to HPC.

```bash
rsync -av \
  artifacts/gen_src_t070_exp_t000/frozen_primary_v1/ \
  <USER>@<HPC_HOST>:/path/to/llm-judge-self-preference/artifacts/gen_src_t070_exp_t000/frozen_primary_v1/
```

If code is not yet synced:

```bash
rsync -av \
  --exclude ".venv" \
  --exclude "__pycache__" \
  --exclude ".git" \
  ./ \
  <USER>@<HPC_HOST>:/path/to/llm-judge-self-preference/
```

### Why this is done

The full 10,000-record judging job requires local Qwen/Gemma judge models and GPU/compute resources. It should not be run on the laptop or HPC login node.

### HPC preflight

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

Verify frozen files after transfer:

```bash
sha256sum -c artifacts/gen_src_t070_exp_t000/frozen_primary_v1/SHA256SUMS.txt
```

### Execution sequence

1. Run the 8-record smoke judging job on a GPU/compute node.
2. Inspect smoke output for valid parsed judgments.
3. Only then launch the full 10,000-record job.

---

## Post-judging processing plan

After full judging completes, do not interpret win rates until integrity checks pass.

Required integrity checks:

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

Primary descriptive outputs:

```text
overall LLM win rate
LLM win rate by judge model
LLM win rate by judge family
LLM win rate by source family
AB vs BA order-bias audit
self-family vs non-self-family win rate
parse/error/refusal rate
```

Primary uncertainty estimates:

```text
bootstrap confidence interval for overall LLM win rate
per-judge bootstrap confidence intervals
clustered or pair-level bootstrap where feasible
```

Required sensitivity analyses:

```text
exclude largest total-duration-gap pairs
exclude largest max-day-gap pairs
exclude pairs with hard/long/quality/rest-day gap > 1
exclude duplicate-heavy session-signature groups/cells
compare scrubbed canonical primary view against original canonical masked view only as sensitivity
score-gap and fixture/band subgroup analyses
```


