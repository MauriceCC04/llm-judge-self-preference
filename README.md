## `README.md`


# judge-bias-study

**LLM-judge self-preference on training plans** — a controlled experiment for
Bocconi Jupiter I HPC.

---

## Research question

Controlling for deterministic plan quality, do LLM judges systematically favour
LLM-generated training plans over programmatically generated ones, and does this
effect vary by judge model family and scale?

## Study modes

This repository supports two distinct modes:

- **Baseline frozen study** — the main causal estimate with fixed generation and
  judging settings.
- **Temperature sensitivity sweeps** — controlled variation of generation and
  judge temperatures to test whether results depend on stochasticity.

Temperature control is implemented **in this repository**, not by modifying
`trailtraining`. `trailtraining` is treated as a pinned library dependency; the
experiment logic, provenance, matching discipline, and sweep orchestration live
here.

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
- Canonical baseline HPC path: `docs/HPC_RUNBOOK.md`
- Temperature sensitivity runs: `docs/TEMPERATURE_SWEEPS.md`

## Baseline temperatures

The default baseline study uses:

- LLM source generation temperature: `0.7`
- Shared explainer temperature: `0.0`
- Judge temperature: `0.0`

These settings should be treated as part of the frozen baseline configuration and
recorded in provenance and judgment outputs.

## Architecture

```text
8 frozen fixtures
        │
        ├── LLM arm          exact-count generation
        │   └── 16 plans / fixture / source model × 8 fixtures × 2 models = 256
        │
        └── Programmatic arm  exact-count generation
            └── 32 plans / fixture × 8 fixtures = 256
        │
        ▼
deterministic scoring → greedy pair matching (|Δscore| ≤ 1)
        │
        ▼
4 judges × (250 pairs × 5 runs × 2 positions + 512 plans × 1 soft-eval batch)
        │
        ▼
mixed-effects models + rubric contrasts + forest plots
```

## Pairwise views

Two pairwise surfaces are supported:

* `raw_normalized` — existing normalized comparison surface
* `canonical_masked` — fixed-format rendering for stricter structure-focused comparison

Use `--pairwise-view canonical_masked` in the judge CLI or `PAIRWISE_VIEW=canonical_masked`
in `slurm/run_judge_hpc.sh` to run the masked control condition.

## Temperature controls

Three temperature knobs are supported:

* `source_temperature` — affects only the **LLM arm source generator**
* `explainer_temperature` — affects the **shared explainer stage** used to fill
  explanation and narrative fields
* `judge_temperature` — affects **pairwise** and **soft-eval** judging

These should not be interpreted the same way:

* Varying **source temperature** changes the LLM arm generation process itself
  and may change both structure and surface form.
* Varying **judge temperature** changes evaluator stochasticity on fixed plans.
* Varying **explainer temperature** changes presentation style and should be
  treated cautiously because it can reintroduce stylistic leakage.

For the main study, keep `explainer_temperature = 0.0` and `judge_temperature = 0.0`.
Treat temperature sweeps as sensitivity analyses unless the study is explicitly
about stochastic generation or stochastic evaluation.

## Artifact layout and condition discipline

Different temperature conditions must be kept in distinct artifact directories.

Do **not** mix multiple generation conditions in the same `plans/` directory and
then run matching on that mixed directory. Matching should be performed on a
single generation condition at a time.

Recommended layout:

```text
artifacts/
  gen_src_t070_exp_t000/
    plans/
    matched_pairs.json
    matching_audit.json
    judgments/
      judge_t000/
      judge_t020/
      judge_t070/
    results/
      judge_t000/
      judge_t020/
      judge_t070/
```

This keeps:

* resume logic correct
* matching condition-pure
* provenance interpretable
* sensitivity analyses separable from the main estimate

## Quota-aware HPC model-cache discipline

On Bocconi, model weights are the main storage bottleneck. The repository is now
designed around a **one required model set per job** policy rather than “cache
everything first”.

Operationally this means:

- **judge jobs** cache exactly **one judge model**
- **programmatic generation jobs** cache exactly the **shared explainer**
- **LLM generation jobs** cache exactly the **shared explainer + one source model**
- unrelated model caches should not remain on disk between jobs
- `bash slurm/pre_cache_models.sh all` is **not** the normal quota-safe workflow

Planning budgets from `judge.panel` and `hpc.quota` are:

- `meta-llama/Llama-3.1-8B-Instruct`: **17 GB**
- `Qwen/Qwen2.5-7B-Instruct`: **15 GB**
- `Qwen/Qwen2.5-3B-Instruct`: **6 GB**
- `Qwen/Qwen2.5-14B-Instruct-AWQ`: **8 GB**
- `Qwen/Qwen2.5-32B-Instruct-AWQ`: **18 GB**

That yields:

- largest **generation** model set: `17 + 6 = 23 GB`
- largest **judge** model set: `18 GB`

See `docs/HPC_RUNBOOK.md` for the exact quota-safe caching commands.

## Baseline command examples

### Generate baseline plans

```bash
python cli.py generate \
  --arm llm \
  --source-model meta-llama/Llama-3.1-8B-Instruct \
  --source-temperature 0.7 \
  --explainer-temperature 0.0 \
  --output artifacts/gen_src_t070_exp_t000/plans

python cli.py generate \
  --arm programmatic \
  --explainer-temperature 0.0 \
  --output artifacts/gen_src_t070_exp_t000/plans
```

### Match within one generation condition

```bash
python cli.py match \
  --plans artifacts/gen_src_t070_exp_t000/plans \
  --output artifacts/gen_src_t070_exp_t000/matched_pairs.json
```

### Judge baseline plans

```bash
python cli.py judge \
  --judge qwen_7b_judge \
  --judge-temperature 0.0 \
  --plans artifacts/gen_src_t070_exp_t000/plans \
  --pairs artifacts/gen_src_t070_exp_t000/matched_pairs.json \
  --output artifacts/gen_src_t070_exp_t000/judgments/judge_t000
```

## Temperature sweep examples

### Judge-temperature sweep on fixed plans

```bash
python cli.py judge \
  --judge qwen_7b_judge \
  --judge-temperature 0.2 \
  --plans artifacts/gen_src_t070_exp_t000/plans \
  --pairs artifacts/gen_src_t070_exp_t000/matched_pairs.json \
  --output artifacts/gen_src_t070_exp_t000/judgments/judge_t020

python cli.py judge \
  --judge qwen_7b_judge \
  --judge-temperature 0.7 \
  --plans artifacts/gen_src_t070_exp_t000/plans \
  --pairs artifacts/gen_src_t070_exp_t000/matched_pairs.json \
  --output artifacts/gen_src_t070_exp_t000/judgments/judge_t070
```

### Source-temperature sweep

```bash
python cli.py generate \
  --arm llm \
  --source-model meta-llama/Llama-3.1-8B-Instruct \
  --source-temperature 0.3 \
  --explainer-temperature 0.0 \
  --output artifacts/gen_src_t030_exp_t000/plans
```

Then regenerate the programmatic arm explainer outputs at the same explainer
setting, rerun matching inside that new generation-condition directory, and only
then launch judging.

## Canonical run path

Use the code-first frozen study path:

1. `python tests/run_tests.py`
2. `python -m fixtures.build`
3. `bash bootstrap_hpc_env.sh`
4. follow `docs/HPC_RUNBOOK.md`

Do not use the older mixed runbook as the source of truth for the frozen study.

## Documentation map

* `docs/HPC_RUNBOOK.md` — baseline frozen-study HPC path
* `docs/HPC_TROUBLESHOOTING.md` — concrete Bocconi failure signatures and fixes
* `docs/TEMPERATURE_SWEEPS.md` — temperature sensitivity workflow and HPC usage
* `docs/EXPERIMENT_CONTROLS.md` — research and implementation design rules for
  provenance, condition isolation, and `trailtraining` compatibility
* `docs/INCIDENTS.md` — running log of incidents, root causes, fixes, and impacts
* `PREREGISTRATION.md` — frozen research design parameters and exclusion criteria
* `LOCAL_SMOKE_TEST.md` — local mock-server smoke test instructions
