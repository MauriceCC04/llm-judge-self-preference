# Local smoke test guide

This guide is for **local plumbing validation only**.

It is useful for:
- confirming CLI wiring
- testing generation, matching, judging, and analysis end to end
- verifying provenance files
- checking resume behavior
- catching OpenAI-compatible client-routing regressions

It is **not** a substitute for a real HPC pilot.

## 1. Run the standard checks first

```bash
python tests/run_tests.py
python -m fixtures.build
```

## 2. Start the local mock server

Run this from the repository root:

```bash
PYTHONPATH=. python tools/mock_llm_server.py --port 8765
```

Check that it is alive:

```bash
curl http://127.0.0.1:8765/health
```

Expected response:

```json
{"status": "ok"}
```

## 3. Export all stage-specific endpoints

In a second terminal:

```bash
export TRAILTRAINING_SOURCE_LLM_BASE_URL=http://127.0.0.1:8765/v1
export TRAILTRAINING_EXPLAINER_LLM_BASE_URL=http://127.0.0.1:8765/v1
export TRAILTRAINING_JUDGE_LLM_BASE_URL=http://127.0.0.1:8765/v1
export TRAILTRAINING_LLM_BASE_URL=http://127.0.0.1:8765/v1
export OPENROUTER_API_KEY=dummy
```

Confirm:

```bash
env | grep TRAILTRAINING_ | sort
```

## 4. Run a minimal smoke test

```bash
rm -rf smoke_plans smoke_judgments smoke_results smoke_artifacts
mkdir -p smoke_artifacts

python cli.py generate --arm llm --source-model meta-llama/Llama-3.1-8B-Instruct --n 2 --fixture-id r_low__rc_low__ph_base --output smoke_plans
python cli.py fit-priors --plans smoke_plans --output smoke_artifacts/smoke_sampler_config.json --min-plans 1
python cli.py generate --arm programmatic --n 2 --fixture-id r_low__rc_low__ph_base --sampler-config smoke_artifacts/smoke_sampler_config.json --output smoke_plans
python cli.py match --plans smoke_plans --output smoke_artifacts/smoke_pairs.json
python cli.py judge --judge qwen_7b_judge --plans smoke_plans --pairs smoke_artifacts/smoke_pairs.json --output smoke_judgments --pair-limit 2 --plan-limit 4 --pairwise-view canonical_masked
python cli.py analyze --judgments smoke_judgments --plans smoke_plans --pairs smoke_artifacts/smoke_pairs.json --output smoke_results --pairwise-view canonical_masked
```

## 5. Run a larger local smoke test

```bash
rm -rf smoke_plans smoke_judgments smoke_results smoke_artifacts
mkdir -p smoke_artifacts

python cli.py generate --arm llm --source-model meta-llama/Llama-3.1-8B-Instruct --n 4 \
  --fixture-id r_low__rc_low__ph_base --fixture-id r_high__rc_high__ph_peak \
  --output smoke_plans

python cli.py fit-priors --plans smoke_plans --output smoke_artifacts/smoke_sampler_config.json --min-plans 2

python cli.py generate --arm programmatic --n 4 \
  --fixture-id r_low__rc_low__ph_base --fixture-id r_high__rc_high__ph_peak \
  --sampler-config smoke_artifacts/smoke_sampler_config.json --output smoke_plans

python cli.py match --plans smoke_plans --output smoke_artifacts/smoke_pairs.json

python cli.py judge --judge qwen_7b_judge \
  --plans smoke_plans --pairs smoke_artifacts/smoke_pairs.json --output smoke_judgments \
  --pair-limit 8 --plan-limit 16 --pairwise-view canonical_masked

python cli.py analyze --judgments smoke_judgments --plans smoke_plans \
  --pairs smoke_artifacts/smoke_pairs.json --output smoke_results --pairwise-view canonical_masked
```

## 6. Check resume behavior

Run the same commands again without deleting outputs.

Expected behavior:
- generation should skip already-existing plans
- judging should resume cleanly
- analysis should still complete from saved artifacts

## 7. What counts as success

A successful local smoke run should show:
- nonzero LLM-arm plans generated
- nonzero programmatic-arm plans generated
- nonzero matched pairs
- nonzero pairwise and soft-eval records loaded by analysis
- no provenance exclusions caused by explainer identity mismatch

## 8. What not to over-interpret

Tiny smoke runs often show:
- low coverage
- judge position bias
- H1/H2 skipped because all rows were removed after bias exclusion
- `PerfectSeparationWarning`
- `H3` or `H4` unsupported

These do **not** mean the repo is broken. They usually mean the mock run is too small or too synthetic for inference.

## 9. Common failure modes

### `Connection error.` everywhere

Usually means one of the following:
- the mock server is not running
- you started a new shell and forgot to re-export stage-specific env vars
- `SOURCE` and `EXPLAINER` URLs were not exported, only the generic or judge URL

### Provenance exclusions during analysis

If `actual_explainer_model` is wrong in provenance files, analysis will drop plans.
The mock server must echo the requested model name so the explainer identity matches the frozen expected explainer.

### Files landing in the repo root

Relative `--output` paths write into the current working directory.
Use an explicit artifact directory such as `smoke_artifacts/`.
