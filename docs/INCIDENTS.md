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


---

## INC-004 — Bocconi login-shell activation leaked to the wrong Python

Date: 2026-04-29
Phase: setup
Severity: blocking

**Root cause:** The initial HPC setup used plain `conda activate judge-bias`
on a shell where the conda hook had not been loaded.  Activation failed with
`CondaError: Run 'conda init' before 'conda activate'`, but subsequent `pip`
commands still ran against the system Miniconda interpreter and user site.
That produced mixed signals such as `cp313` wheels and imports from
`~/.local/lib/python3.13/...` even though the intended environment was Python
3.11.

**Fix:** The HPC runbook now requires `eval "$(conda shell.bash hook)"` before
`conda activate`, followed by explicit `which python` / `python --version`
verification and `export PYTHONNOUSERSITE=1`.

**Impact:** This was a pure environment reproducibility failure.  It did not
reflect a code bug in the repository, but it could easily corrupt package
installs and invalidate later debugging.

---

## INC-005 — `vllm` install on Jupiter I required explicit CUDA setup

Date: 2026-04-29
Phase: setup
Severity: blocking

**Root cause:** Installing `vllm` before loading CUDA caused pip build
resolution to fail with `AssertionError: CUDA_HOME is not set`.

**Fix:** The documented install order is now: activate the correct conda env,
`module load cuda/12.4`, export `CUDA_HOME` from `which nvcc`, then install
`torch` and `vllm`.

**Impact:** Without this change, a user could have a seemingly healthy Python
setup while still being unable to start the real local-vLLM path used by the
SLURM scripts.

---

## INC-006 — stale SLURM wrapper scripts broke Gate-0 parse tests

Date: 2026-04-29
Phase: gate_0
Severity: blocking

**Root cause:** The checkout used during HPC setup contained thin wrapper
variants of `run_generation.sh`, `run_judge.sh`, and `submit_judge.sh` that did
not source `common.sh`.  Gate-0 therefore failed with errors such as
`run_generation.sh does not source common.sh`, later
`run_judge.sh does not source common.sh`, and then
`submit_judge.sh does not source common.sh`.

**Fix:** The runbook now treats these failures as a stale-checkout signal and
requires synchronising to a passing revision before proceeding.

**Impact:** This looked like an HPC launch problem, but it was actually a
repository/version consistency problem.

---

## INC-007 — direct `cli.py generate` on HPC failed without stage-specific endpoints

Date: 2026-04-29
Phase: generation
Severity: blocking

**Root cause:** Generation was first attempted through direct `python cli.py
 generate ...` commands on the login node rather than through the HPC wrapper.
The source stage hit `401 Missing Authentication header`, while the
programmatic arm failed with a missing explainer endpoint, because the
stage-specific `TRAILTRAINING_*_LLM_BASE_URL` variables were not managed by the
user shell.

**Fix:** The runbook now explicitly distinguishes the canonical HPC path
(wrapper-driven) from the direct-CLI path (valid only when the user has already
started the local vLLM endpoints and exported the stage-specific URLs).

**Impact:** This was an operational reproducibility failure that could be
misread as an API-key issue or a generation bug.

---

## INC-008 — HF access, Xet download, and quota all blocked model caching

Date: 2026-04-29
Phase: generation
Severity: blocking

**Root cause:** Several independent issues appeared in sequence during model
pre-caching on Bocconi:

1. gated Llama access initially failed because the HF account was not yet on the
   authorized list for `meta-llama/Llama-3.1-8B-Instruct`
2. after access was granted, the download failed through the Xet backend with
   `File reconstruction error: Internal Writer Error: Background writer channel closed`
3. after disabling Xet, cache writes failed because the default cache location
   under `/home/<USER_ID>` hit `Disk quota exceeded`
4. moving the cache to `/mnt/beegfsstudents/home/<USER_ID>/hf_cache` solved the
   `/home` problem but still exhausted the 50 GB BeegFS home quota once the
   partial Llama cache reached roughly 26 GB and overall usage passed the limit

**Fix:** The runbook and HPC troubleshooting guide now require:

- explicit HF login for gated models
- `HF_HUB_DISABLE_XET=1` during pre-cache on this cluster
- cache relocation away from `/home/<USER_ID>`
- quota checks before large downloads
- a documented one-model-at-a-time fallback when no larger scratch/project path
  is available

**Impact:** This became the main HPC reproducibility risk for the real study,
not because the code was wrong, but because model-cache storage assumptions were
not explicit enough.
