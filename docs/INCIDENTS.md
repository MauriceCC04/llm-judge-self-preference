# INCIDENTS.md — Decision log

Append a new entry whenever something breaks, whenever a design decision is
revised, or whenever an unexpected result requires an explanation.  Never
edit existing entries.

Format:
```
## INC-NNN — <short title>
Date: YYYY-MM-DD
Phase: gate_N | generation | matching | judging | analysis
Severity: blocking | degraded | informational
```

---

## INC-000 — Study initialised

Date: 2026-04-23
Phase: setup
Severity: informational

Initial repo created.  PREREGISTRATION.md committed before any generation run.
Gate-0 (22 tests) passing on CPU without GPU or API keys.

Three upstream PRs opened against `trailtraining`:
- PR-1: `TRAILTRAINING_LLM_BASE_URL` env override in `make_openrouter_client`
- PR-2: Public `run_training_plan_from_machine_plan()` helper
- PR-3: `util/resume_jsonl.py` append-only JSONL with dedup

Until PRs are merged, `vendor_patches/resume_jsonl.py` is used directly and
`generate/programmatic_arm.py` falls back to `_run_explainer_directly()`.

---

<!-- Append new incidents below this line -->

## INC-001 — H1/H3 used linear mixed model on binary outcome

Date: 2026-04-24
Phase: analysis
Severity: blocking

**Root cause:** `analyze/models.py` used `smf.mixedlm` (linear mixed model,
Gaussian likelihood) for `prefers_llm ∈ {0,1}`. This is methodologically
wrong: coefficients are unbounded and SEs are unreliable for binary data.

**Fix:** Replaced with `smf.logit` (logistic regression) with cluster-robust
standard errors clustered at `pair_id`.  `prob_llm` is now computed as
`result.predict().mean()` (population-averaged probability) rather than
the logit-inverse of the bare intercept, which depended on the arbitrary
reference-level choice for judge and fixture dummies.

**Impact:** Previous test 16 (`mixed_effects_fit`) was exercising the wrong
model class.  Test was updated to call `fit_h1_model` directly.  H3/H4 were
similarly fixed (H4 remains OLS since `prefers_llm` averaged over a pair is
treated as continuous for the scale-trend test).

---

## INC-002 — llm_arm.py wrote temp files into fixture_dir

Date: 2026-04-24
Phase: generation
Severity: blocking

**Root cause:** `generate/llm_arm.py` set `fake_paths.prompting_directory =
fixture_dir`.  Any temp files the trailtraining coach wrote went into the
committed fixture directory, potentially corrupting it.  The `ensure_directories`
call would also create unwanted subdirs there.

**Fix:** Changed `fake_paths.*` to point to `output_dir`.  Added a
`ensure_directories = lambda: None` patch so the already-extant output dir
is not re-created.  Test 30 (`llm_arm_output_separation`) asserts that after
a generation call the fixture directory contains only its original five JSON
files and nothing else.

---

## INC-003 — analyze/load.py did not populate source_model_a/b

Date: 2026-04-24
Phase: analysis
Severity: degraded

**Root cause:** `load_judgments` for pairwise records only joined `arm_a` and
`arm_b` from provenance.  `add_same_family_column()` in `models.py` requires
`source_model_a` and `source_model_b` to determine H3 self-preference; these
were always empty, forcing `same_family = 0` for every row.

**Fix:** `load_judgments` now also joins `source_model_a` and `source_model_b`
from provenance sidecars.  The kind-based file filter (pairwise vs soft_eval
files) was also added so the two record types are not mixed.

