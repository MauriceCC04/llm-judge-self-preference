# judge-bias-study

**LLM-judge self-preference on training plans** — a controlled experiment on whether LLM judges systematically prefer LLM-generated plans over programmatic plans when deterministic plan quality is controlled.

## Research question

Controlling for deterministic plan quality, do LLM judges systematically favor LLM-generated training plans over programmatically generated ones, and does this effect vary by judge family, judge scale, rubric, or whether the plan matches the judge’s own family?

## Frozen study summary

- Fixtures: 8
- LLM arm: 256 plans
- Programmatic arm: 256 plans
- Total plans: 512
- Target matched pairs: 250
- Active judges: 4 (`llama_8b_judge`, `qwen_7b_judge`, `qwen_14b_judge`, `qwen_32b_judge`)
- Pairwise calls: `250 × 4 × 5 × 2 = 10,000`
- Soft-eval calls: `512 × 4 = 2,048`
- Shared explainer: `Qwen/Qwen2.5-3B-Instruct`

## What this repository does

This repository builds a frozen evaluation pipeline around paired training plans:

1. **Fixtures** define athlete contexts and supporting rollups.
2. **Generation** creates two plan arms:
   - **LLM arm**: source model creates a machine plan, then a shared explainer model fills explanation fields.
   - **Programmatic arm**: structure is sampled programmatically, then the same shared explainer model fills explanation fields.
3. **Matching** pairs LLM and programmatic plans with similar deterministic scores.
4. **Judging** runs pairwise comparisons and soft-eval scoring.
5. **Analysis** estimates preference effects, self-family effects, rubric-level contrasts, position bias, and style leakage.

## Repository structure

- `fixtures/` — frozen fixture bundles and builder
- `generate/` — LLM arm, programmatic arm, provenance, structural sampling
- `match/` — deterministic-score pairing and matching audit outputs
- `judge/` — pairwise and soft-eval harnesses
- `analyze/` — data loading, model fitting, plots, summaries
- `slurm/` — HPC orchestration scripts for generation and judging
- `tools/` — local support utilities including the OpenAI-compatible mock server
- `tests/` — Gate-0 regression tests, including the local HTTP mock smoke path

## Operating modes

There are two supported ways to run the project.

### 1. Local smoke testing

Use the local mock server only to validate wiring, artifacts, resume behavior, and end-to-end plumbing.

A local smoke run can verify that:
- generation works for both arms
- provenance files are written correctly
- matching and judging produce outputs
- analysis can load and process those outputs

A local smoke run is **not** evidence for the research question.

Why not:
- mock responses are synthetic
- tiny runs often trigger position-bias exclusions that remove all rows from H1/H2
- small samples can trigger `PerfectSeparationWarning` and unsupported fits in H3/H4
- coverage and style audits are not meaningful at smoke scale

See [`docs/LOCAL_SMOKE_TEST.md`](docs/LOCAL_SMOKE_TEST.md).

### 2. Canonical HPC execution

Use the SLURM scripts and real vLLM OpenAI-compatible servers for any meaningful pilot or full study run.

The HPC path uses **real local vLLM endpoints**, not the local mock server.

Recommended order:

1. `python tests/run_tests.py`
2. `python -m fixtures.build`
3. `bash bootstrap_hpc_env.sh`
4. run the SLURM scripts in `slurm/`

Start with a **small real-endpoint pilot** before scaling to the frozen study.

See [`docs/HPC_PILOT_CHECKLIST.md`](docs/HPC_PILOT_CHECKLIST.md).

## Pairwise views

Two pairwise surfaces are supported:

- `raw_normalized` — normalized comparison surface
- `canonical_masked` — fixed-format rendering for stricter structure-focused comparison

Use `--pairwise-view canonical_masked` in the CLI or `PAIRWISE_VIEW=canonical_masked` in the judge script to run the masked control condition.

## Local smoke quick start

Start the mock server from the repository root:

```bash
PYTHONPATH=. python tools/mock_llm_server.py --port 8765
```

In another terminal:

```bash
export TRAILTRAINING_SOURCE_LLM_BASE_URL=http://127.0.0.1:8765/v1
export TRAILTRAINING_EXPLAINER_LLM_BASE_URL=http://127.0.0.1:8765/v1
export TRAILTRAINING_JUDGE_LLM_BASE_URL=http://127.0.0.1:8765/v1
export TRAILTRAINING_LLM_BASE_URL=http://127.0.0.1:8765/v1
export OPENROUTER_API_KEY=dummy
```

Minimal smoke run:

```bash
python cli.py generate --arm llm --source-model meta-llama/Llama-3.1-8B-Instruct --n 2 --fixture-id r_low__rc_low__ph_base --output smoke_plans
python cli.py fit-priors --plans smoke_plans --output smoke_artifacts/smoke_sampler_config.json --min-plans 1
python cli.py generate --arm programmatic --n 2 --fixture-id r_low__rc_low__ph_base --sampler-config smoke_artifacts/smoke_sampler_config.json --output smoke_plans
python cli.py match --plans smoke_plans --output smoke_artifacts/smoke_pairs.json
python cli.py judge --judge qwen_7b_judge --plans smoke_plans --pairs smoke_artifacts/smoke_pairs.json --output smoke_judgments --pair-limit 2 --plan-limit 4 --pairwise-view canonical_masked
python cli.py analyze --judgments smoke_judgments --plans smoke_plans --pairs smoke_artifacts/smoke_pairs.json --output smoke_results --pairwise-view canonical_masked
```

## Important routing notes

The code supports **stage-specific OpenAI-compatible endpoints**:

- `TRAILTRAINING_SOURCE_LLM_BASE_URL`
- `TRAILTRAINING_EXPLAINER_LLM_BASE_URL`
- `TRAILTRAINING_JUDGE_LLM_BASE_URL`
- `TRAILTRAINING_LLM_BASE_URL` (generic fallback)

For local smoke runs, export all four so the pipeline behaves consistently.

For HPC runs, generation already uses explicit source/explainer endpoints. Judge runs can rely on the generic endpoint, though setting `TRAILTRAINING_JUDGE_LLM_BASE_URL` explicitly is also fine for clarity.

## Output paths and where files go

The CLI writes exactly where you tell it to write.

Examples:
- `fit-priors --output smoke_sampler_config.json` writes `smoke_sampler_config.json` in the current working directory.
- `match --output smoke_pairs.json` writes `smoke_pairs.json` in the current working directory.
- `matching_audit.json` is written **next to** the pairs file.

To keep smoke artifacts organized, prefer explicit subdirectories such as:

```bash
python cli.py fit-priors --plans smoke_plans --output smoke_artifacts/smoke_sampler_config.json --min-plans 2
python cli.py match --plans smoke_plans --output smoke_artifacts/smoke_pairs.json
```

## Interpretation notes for small runs

If a tiny smoke run shows:
- `Coverage OK: False`
- `Position-bias audit: ... biased`
- `H1: skipped`
- `H2: no soft-eval data`
- many `PerfectSeparationWarning` lines

that usually means the local plumbing worked, but the sample is too small or too synthetic for meaningful inference.

## Recommended workflow

1. Keep `tests/run_tests.py` green.
2. Use local smoke runs to validate plumbing and resume behavior.
3. Run a **small HPC pilot** with real vLLM endpoints.
4. Only then scale to the frozen study.

## Notes on the local mock path

The local HTTP mock path matters because it validates the same client-routing layer used by the CLI.

The local mock server should:
- be launched from repo root or with repo-root `PYTHONPATH`
- echo the requested model in the response so provenance checks pass
- return a clean OpenAI-compatible response shape without duplicating output text

These details matter for local smoke runs, but they do **not** change the canonical HPC execution path, which uses real vLLM servers.
