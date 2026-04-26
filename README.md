# judge-bias-study

**LLM-judge self-preference on training plans** — a controlled experiment for
Bocconi Jupiter I HPC.

---

## Research question

Controlling for deterministic plan quality, do LLM judges systematically favour
LLM-generated training plans over programmatically generated ones, and does this
effect vary by judge model family and scale?

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
- Canonical HPC path: `HPC_SYNC_RUNBOOK.md`

## Architecture

```
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

- `raw_normalized` — existing normalized comparison surface
- `canonical_masked` — fixed-format rendering for stricter structure-focused comparison

Use `--pairwise-view canonical_masked` in the judge CLI or `PAIRWISE_VIEW=canonical_masked`
in `slurm/run_judge_hpc.sh` to run the masked control condition.

## Canonical run path

Use the code-first frozen study path:

1. `python tests/run_tests.py`
2. `python -m fixtures.build`
3. `bash bootstrap_hpc_env.sh`
4. follow `HPC_SYNC_RUNBOOK.md`

Do not use the older mixed runbook as the source of truth for the frozen study.
