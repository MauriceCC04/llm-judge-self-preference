# INCIDENTS.md — Decision log

Append a new entry whenever something breaks, whenever a design decision is
revised, or whenever an unexpected result requires an explanation. Never edit
existing entries.

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

Initial repo created. `docs/PREREGISTRATION.md` committed before any generation
run. Gate-0 passing on CPU without GPU or API keys.

Three upstream PRs opened against `trailtraining`:
- PR-1: `TRAILTRAINING_LLM_BASE_URL` env override in `make_openrouter_client`
- PR-2: Public `run_training_plan_from_machine_plan()` helper
- PR-3: `util/resume_jsonl.py` append-only JSONL with dedup

Until PRs are merged, `vendor_patches/resume_jsonl.py` is used directly and
`generate/programmatic_arm.py` falls back to `_run_explainer_directly()`.

---

## INC-001 — H1/H3 used linear mixed model on binary outcome

Date: 2026-04-24
Phase: analysis
Severity: blocking

**Root cause:** `analyze/models.py` used `smf.mixedlm` for a binary outcome.

**Fix:** Replaced with logistic regression and cluster-robust standard errors.

**Impact:** H1/H3/H4 fitting logic and related tests were corrected.

---

## INC-002 — llm_arm.py wrote temp files into fixture_dir

Date: 2026-04-24
Phase: generation
Severity: blocking

**Root cause:** temp outputs were pointed at the fixture directory.

**Fix:** temp files now point at output directories instead.

**Impact:** fixture directories are no longer contaminated by generation temp
artifacts.

---

## INC-003 — analyze/load.py did not populate source_model_a/b

Date: 2026-04-24
Phase: analysis
Severity: degraded

**Root cause:** pairwise provenance joins omitted `source_model_a/b`.

**Fix:** load path now joins them from provenance sidecars.

**Impact:** self-family preference columns became meaningful.

---

## INC-004 — Bocconi login-shell activation leaked to the wrong Python

Date: 2026-04-29
Phase: setup
Severity: blocking

**Root cause:** plain `conda activate` was used before loading the shell hook.

**Fix:** runbook now requires `eval "$(conda shell.bash hook)"` before
activation, plus interpreter verification and `PYTHONNOUSERSITE=1`.

**Impact:** removed a major reproducibility hazard in HPC setup.

---

## INC-005 — `vllm` install on Jupiter I required explicit CUDA setup

Date: 2026-04-29
Phase: setup
Severity: blocking

**Root cause:** `vllm` was installed before loading CUDA.

**Fix:** documented install order: activate env, load CUDA, export
`CUDA_HOME`, install `torch`, then install `vllm`.

**Impact:** made the real local-vLLM HPC path installable.

---

## INC-006 — stale SLURM wrapper scripts broke Gate-0 parse tests

Date: 2026-04-29
Phase: gate_0
Severity: blocking

**Root cause:** stale thin wrapper variants did not source `common.sh`.

**Fix:** runbook now treats these parse failures as a stale-checkout signal.

**Impact:** reduced false debugging of environment issues that were actually
checkout-version issues.

---

## INC-007 — direct `cli.py generate` on HPC failed without stage-specific endpoints

Date: 2026-04-29
Phase: generation
Severity: blocking

**Root cause:** generation was attempted through direct CLI calls on the login
node without the local-vLLM wrapper-managed stage URLs.

**Fix:** runbook now clearly separates the wrapper-driven canonical HPC path
from the direct-CLI path.

**Impact:** avoided misreading wrapper/endpoint mistakes as model or API bugs.

---

## INC-008 — HF access, Xet download, and quota all blocked model caching

Date: 2026-04-29
Phase: generation
Severity: blocking

**Root cause:** gated-model access, Xet download behavior, and quota limits all
caused pre-caching failures in sequence.

**Fix:** documented HF login, `HF_HUB_DISABLE_XET=1`, cache relocation, and
one-required-model-set discipline.

**Impact:** made model-cache management the explicit operational bottleneck to
plan around.

---

## INC-009 — explainer request exceeded model context length

Date: 2026-05-01
Phase: generation
Severity: blocking

**Root cause:** after increasing structured output caps, the explainer request
used `12288` output tokens on a server with only `16384` total context, while
the prompt already contained about `4097` tokens.

**Fix:** increased the explainer server context budget and froze the validated
structured settings:

- `TRAILTRAINING_STRUCTURED_MAX_TOKENS=12288`
- `TRAILTRAINING_SOURCE_MAX_TOKENS=4096`
- `TRAILTRAINING_EXPLAINER_MAX_TOKENS=12288`
- `VLLM_SOURCE_MAX_MODEL_LEN=16384`
- `VLLM_EXPLAINER_MAX_MODEL_LEN=24576`

**Impact:** removed the immediate 400-context-overflow blocker from the Qwen
source + shared-explainer path.

---

## INC-010 — explainer truncation surfaced as malformed JSON, not as the true root cause

Date: 2026-05-01
Phase: generation
Severity: blocking

**Root cause:** a pathological explainer output hit the completion cap with
`finish=length`, and the downstream parser raised JSON errors such as
`Expecting ',' delimiter`.

**Fix:** moved to the working structured-output request path with validated
budgets and treated `finish=length` as a truncation signal rather than as
primary evidence that the model was incapable.

**Impact:** the previously failing fixture was recovered, and the 8-fixture
pilot later completed cleanly.

---

## INC-011 — placeholder leakage contaminated saved final artifacts

Date: 2026-05-01
Phase: generation
Severity: blocking

**Root cause:** one saved final plan contained `>{signal_id` placeholder-like
content in `snapshot.last7.*` and `snapshot.baseline28.*`. Prompt and machine
plan replay showed the contamination arose downstream of the initial prompt
construction.

**Fix:** final `snapshot` construction was changed to use deterministic
structured signal data rather than trusting explainer text, and final-validation
placeholder leak checks were added.

**Impact:** the bad fixture stopped leaking placeholders, the repaired 8-fixture
pilot had zero placeholder hits, and the later 254-plan Qwen run also had zero
placeholder hits in saved plans.

---

## INC-012 — mock/test path diverged from the new explainer-stage schema

Date: 2026-05-01
Phase: gate_0
Severity: blocking

**Root cause:** after the deterministic-snapshot change, the mock explainer
stage still returned `snapshot` even though the stage schema no longer allowed
that field.

**Fix:** updated the mock client/server so
`trailtraining_plan_explanation_stage_v1` returns only stage-valid fields.

**Impact:** restored local Gate-0 after the real HPC path had already been fixed.

---

## INC-013 — manual file copying caused repo drift across HPC checkouts

Date: 2026-05-01
Phase: setup
Severity: degraded

**Root cause:** hand-copying files into the HPC repos left
`llm-judge-self-preference` and `trailtraining` temporarily out of sync, leading
to import errors such as missing `_build_deterministic_snapshot`.

**Fix:** resynchronised both repos from the desktop / GitHub versions, cleared
`__pycache__`, and switched back to repo-sync discipline instead of ad hoc file
copying.

**Impact:** removed a confusing class of false code-path failures.

---

## INC-014 — occasional machine-plan duration violations remained after the major fixes

Date: 2026-05-01
Phase: generation
Severity: degraded

**Root cause:** even after the main generation path was repaired, some source
machine plans still produced invalid `duration_minutes` values such as `480` or
`630`. Many such cases were repaired successfully, but a few attempts still
failed and had to be excluded.

**Fix:** relied on the existing structured repair path, excluded unrepaired
failures from the retained corpus, and added saved-plan duration validation to
the post-generation acceptance gate.

**Impact:** the pipeline became runnable at scale, but with some residual yield
loss rather than catastrophic artifact corruption.

---

## INC-015 — large-scale Qwen generation succeeded operationally but exposed corpus duplication

Date: 2026-05-01
Phase: generation
Severity: degraded

**Root cause:** once the Qwen generation path was operational, the retained LLM
pool showed substantial within-cell repetition:

- exact text duplicates
- repeated exact session-signature groups
- generic repeated structures
- some human-facing contradictions such as non-rest sessions titled `Rest day`

**Fix:** introduced retained-corpus filtering rules:

- drop human-facing contradictions
- collapse exact text duplicates to one representative
- collapse exact session-signature duplicates to one representative
- require a light human audit before matching and judging

**Impact:** shifted the main blocker from infrastructure failure to corpus
quality / study-validity discipline.

---

## INC-016 — `git clean -fd` deleted generated study artifacts

Date: 2026-05-02
Phase: setup
Severity: degraded

**Root cause:** generated plan directories and job logs were stored as untracked
files inside the repo working tree. `git clean -fd` therefore removed them.

**Fix:** documented artifact-safety discipline in the runbook and README:
completed corpora must be copied to a durable non-repo artifact directory or
back to the desktop before any reset / clean cycle.

**Impact:** this became an operational reproducibility rule for all future runs.
