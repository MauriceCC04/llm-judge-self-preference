# HPC pilot checklist

Use this checklist before launching the full frozen study.

## Goal

Run a **small real-endpoint pilot** on the cluster that validates:
- source/explainer routing
- judge routing
- provenance integrity
- matching outputs
- pairwise and soft-eval JSONL writing
- analysis loading and exclusions

## Important distinction

The HPC path uses **real local vLLM OpenAI-compatible servers** launched by the SLURM scripts.
It does **not** use `tools/mock_llm_server.py`.

That means the exact mock-server bugs discovered during local smoke testing should not reappear on HPC.

However, a related compatibility risk still exists: if a real endpoint returns an unexpected `model` or output shape, provenance checks or parsing can still fail.

## Preflight

Run these before any pilot:

```bash
python tests/run_tests.py
python -m fixtures.build
bash bootstrap_hpc_env.sh
bash slurm/run_preflight.sh
```

## Recommended first pilot

Keep the pilot deliberately small:
- 1 fixture
- 1 LLM source model
- 1 judge
- low pair limit
- low soft-eval plan limit
- `PAIRWISE_VIEW=canonical_masked`

## What to validate after generation

Inspect a few provenance files and confirm:
- `explainer_model == Qwen/Qwen2.5-3B-Instruct`
- `actual_explainer_model == Qwen/Qwen2.5-3B-Instruct`
- `explainer_model_verified == true`

If those fields are wrong, fix endpoint/model routing before scaling up.

## What to validate after judging

Confirm that the pilot produced:
- nonempty pairwise JSONL
- nonempty soft-eval JSONL
- no schema-failure explosion
- no unexpected endpoint or connection failures

## What to validate after analysis

Confirm that analysis shows:
- no provenance exclusions for good plans
- pair coverage based on actual pilot pairs
- interpretable outputs even if some judge-level rows are excluded for bias

## Judge routing note

The current code can route judge traffic through the generic `TRAILTRAINING_LLM_BASE_URL`.
Setting `TRAILTRAINING_JUDGE_LLM_BASE_URL` explicitly is optional, but recommended for clarity.

## Scaling rule

Do not scale to the frozen study until the small pilot is clean.

Scale only after:
- both generation arms work on real endpoints
- provenance fields are correct
- pairwise and soft-eval records are nonempty
- analysis loads without unexpected exclusions
- any position-bias exclusions are understood, not accidental
