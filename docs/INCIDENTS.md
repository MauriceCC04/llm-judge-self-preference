# Incidents and Operational Decisions

This file records failures, root causes, fixes, and methodological decisions that affect reproducibility or validity. Append new incidents rather than deleting history.

## INC-001 - General incident-log policy

Date: ongoing
Phase: all
Severity: informational

Whenever a pipeline failure, environment issue, artifact-integrity issue, methodological correction, or documentation-changing decision occurs, add an incident entry with:

- date
- phase
- severity
- symptoms
- root cause
- fix or decision
- impact on validity/reproducibility

## INC-017 - Retained corpora expanded beyond earlier 512-plan documentation

Date: 2026-05-06
Phase: generation/documentation
Severity: informational

**Symptoms:** Earlier documentation described a frozen 512-plan baseline. Actual retained corpora are now 192 Qwen plans, 192 Gemma 3 plans, and 640 programmatic plans, for 1024 matching-pool candidates.

**Root cause:** The operational pool was expanded to support valid same-cell matching and source-family analysis.

**Fix/decision:** Documentation must refer to the current 1024-candidate pool. The estimand remains a matched LLM-vs-programmatic pairwise preference effect.

**Impact:** Generated-plan count changed, but the judge-evaluation target remains at least 250 matched pairs and 10,000 judge-facing evaluation documents.

## INC-018 - Programmatic arm initially had presentation contradictions and flattened band structure

Date: 2026-05-05 to 2026-05-06
Phase: programmatic generation
Severity: blocking until fixed

**Symptoms:** Early programmatic artifacts showed title/session mismatches, rest-active contradictions, many generic titles, and weak A3/A4 load differentiation.

**Root cause:** The programmatic arm allowed the explainer to fill human-facing fields inconsistently with trusted structural fields. The sampler's phase adjustment also overwrote some band-specific duration settings.

**Fix/decision:** Add deterministic programmatic artifact text normalization and patch the sampler so race phase tunes band-specific parameters rather than erasing them.

**Validation:** Final programmatic audit showed:

- 640 plans and 640 provenance.
- 32 cells, 20 plans per cell.
- no missing sidecars.
- no bad JSON.
- no wrong plan lengths.
- no structural prompt leaks.
- generic_titles=0.
- title_session_mismatches=0.
- rest_active_contradictions=0.
- plausible duration/load gradient from A1 to A4.

**Impact:** The fixed programmatic corpus is eligible as a candidate pool for structural matching.

## INC-019 - Conda activation and Python path issues on HPC

Date: 2026-05-06
Phase: matching/environment
Severity: blocking until fixed

**Symptoms:** Matching failed with:

```text
ModuleNotFoundError: No module named 'pydantic'
ModuleNotFoundError: No module named 'trailtraining'
CondaError: Run 'conda init' before 'conda activate'
```

**Root cause:** Interactive shell environment did not reliably activate `judge-bias`; Python could not see the sibling `trailtraining/src` repo.

**Fix/decision:** Use the env Python directly and export `PYTHONPATH`:

```bash
export REPO_ROOT=/mnt/beegfsstudents/home/<USER_ID>/llm-judge-self-preference
export TRAILTRAINING_REPO=/mnt/beegfsstudents/home/<USER_ID>/trailtraining
export PYTHONPATH="${REPO_ROOT}:${TRAILTRAINING_REPO}/src:${PYTHONPATH:-}"
export PY=/home/<USER_ID>/.conda/envs/judge-bias/bin/python
```

**Impact:** Future HPC commands and SLURM jobs must use this robust pattern.

## INC-020 - Gated Gemma model and Hugging Face token failure

Date: 2026-05-05
Phase: model caching
Severity: blocking until fixed

**Symptoms:** Caching Gemma failed with 401 Unauthorized and gated repository errors.

**Root cause:** Hugging Face token was not configured and model terms were not accessible for the session.

**Fix/decision:** Set `HF_TOKEN`, accept model terms, export Hugging Face cache variables, and re-run caching. Verify with `tools/check_model_cache.py`.

**Impact:** The runbook now explicitly documents gated-model and cache requirements.

## INC-021 - Model cache cleanup caused downstream model-not-found failures

Date: 2026-05-04 to 2026-05-05
Phase: generation/model cache
Severity: operational

**Symptoms:** Later jobs failed instantly with `Model not found in cache` after earlier jobs deleted cached Qwen explainer/source models.

**Root cause:** Cleanup flags removed cached weights needed by subsequent jobs. Quota pressure made cache management delicate.

**Fix/decision:** Set cleanup flags intentionally. Before jobs, verify exactly the needed source/explainer/judge models are cached. Use `lquota` and delete only models that are not needed for the next job.

**Impact:** The runbook now requires cache verification and warns against accidental cleanup.

## INC-022 - Old deterministic quality score failed as primary matching score

Date: 2026-05-06
Phase: matching
Severity: blocking

**Symptoms:** The old matcher produced only 30 pairs from the 1024-candidate matching pool:

```text
Pairs yielded: 30
Target: 256
Coverage ratio: 0.117
Coverage OK: False
```

The prefilter kept 936/1024 records and dropped only 88 session-signature duplicates. Therefore prefiltering was not the main blocker.

Score diagnostics showed severe non-overlap:

- Qwen median score about 100.
- Gemma 3 median score about 100.
- Programmatic median score about 30.

Relaxing same-score-bin matching did not materially improve pair count.

**Root cause:** The existing TrailTraining quality score appears to reward or penalize presentation/explanation richness rather than measuring only source-neutral structural quality. This is incompatible with the study design because presentation richness is a possible source cue and confound.

**Fix/decision:** Do not proceed to full judging with the 30-pair matched set. Implement a source-neutral structural matching score that excludes prose, explanations, citations, source labels, and file-name artifacts. Keep the old quality score only as a diagnostic unless tests prove it is structural-only.

**Impact:** Structural-score implementation and matching diagnostics are now mandatory before judge evaluation.

## INC-023 - 10,000-document requirement clarified

Date: 2026-05-06
Phase: evaluation planning
Severity: validity-critical

**Symptoms:** The phrase `500 plans x 5 runs x 4 models = 10,000 eval documents` was ambiguous.

**Clarification:** For pairwise judging, the correct formula is:

```text
250 matched pairs x 2 left-right orders x 5 runs x 4 judge models = 10,000 evaluation documents
```

Preferred:

```text
256 matched pairs x 2 orders x 5 runs x 4 judge models = 10,240 evaluation documents
```

**Fix/decision:** Full judge launchers must compute and assert `n_eval_documents >= 10000` and must refuse full launch if `matched_pairs < 250`.

**Impact:** The current 30-pair old-score match yields only 1,200 documents and is not sufficient.
