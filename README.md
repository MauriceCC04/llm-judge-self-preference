# judge-bias-study

**LLM-judge self-preference on marathon training plans** — a controlled
experiment on Bocconi Jupiter I HPC.

[![Gate-0](https://img.shields.io/badge/Gate--0-22%2F22%20tests-brightgreen)]()

---

## Research question

Controlling for deterministic plan quality, do LLM judges systematically favour
LLM-generated training plans over programmatically generated ones, and does this
effect vary by judge model family and scale?

## Hypotheses

| | Hypothesis | Mechanism |
|---|---|---|
| H1 | Judges prefer LLM plans > 50% after position-swap correction | — |
| H2 | Preference gap is larger on `explanation_quality` than `plan_coherence` | Narrative quality drives the bias |
| H3 | Self-preference: P(prefer LLM \| same family) > P(prefer LLM \| different family) | In-group recognition |
| H4 | Within Qwen, self-preference grows with parameter count (7B → 14B → 32B) | Scale amplifies bias |

All four hypotheses are pre-registered in `PREREGISTRATION.md` before any
generation run begins.

---

## Architecture

```
8 frozen fixtures (readiness × recovery × phase)
        │
        ├── LLM arm          loop over (fixture × source_model × seed)
        │   └── run_coach_brief → TrainingPlanArtifact + .provenance.json
        │
        └── Programmatic arm  sampler priors fitted from LLM arm marginals
            └── sample → guardrails → shared explainer → TrainingPlanArtifact
                                         ↑
                              SAME model as LLM arm explainer
                              (EXPLAINER_MODEL_ID constant, tested at Gate 0)
        │
        ▼
deterministic scoring → greedy pair matching (|Δscore| ≤ 1)
        │
        ▼
5 judges × (250 pairs × 3 runs × 2 positions + 500 plans × 1 batch)
        │
        ▼
mixed-effects models + rubric contrasts + forest plots
```

**Critical invariant:** `EXPLAINER_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"` is
shared across both arms.  Test 8 in `tests/run_tests.py` asserts this at Gate 0.

---

## Cluster constraints (Bocconi Jupiter I)

| Resource | Limit | Mitigation |
|---|---|---|
| Home quota | 50 GB | One model on disk at a time; delete after each judge job |
| VRAM | 40 GB (MIG slice) | `--max-model-len 8192` cap; AWQ-INT4 for 14B/32B |
| Job slots | 2 running / 2 submitted | Sequential judge jobs |
| Bad node | `gnode04` | `--exclude=gnode04` on all `sbatch` calls |

---

## Quick start

### On a laptop (Gate 0)

```bash
git clone <this-repo> && cd judge-bias-study
pip install -e /path/to/trailtraining
pip install -e ".[dev]"

python tests/run_tests.py        # must show 22/22 before anything else
```

### On the HPC (Gates 1–4)

```bash
# 0. Pre-register (commit before first sbatch)
git add PREREGISTRATION.md && git commit -m "pre-registration"

# 1. Build fixtures (idempotent, already committed)
python -m fixtures.build

# 2. Gate 1 — preflight
sbatch --exclude=gnode04 --wrap='cd $PROJECT_ROOT && bash slurm/run_preflight.sh'

# 3. Gate 2 — vLLM smoke test
bash slurm/pre_cache_models.sh qwen_7b_source
sbatch --exclude=gnode04 --wrap='cd $PROJECT_ROOT && bash slurm/run_vllm_smoke.sh'

# 4. Generate LLM arm (all source models, all fixtures)
#    Run ~100 plans first so we can fit sampler priors before launching the rest
bash slurm/pre_cache_models.sh llama_8b_source
GENERATION_ARM=llm GENERATION_N=14 sbatch --exclude=gnode04 \
    --wrap='cd $PROJECT_ROOT && bash slurm/run_generation.sh'

# 5. Fit sampler priors and generate programmatic arm
python -m generate.fit_priors --plans-dir plans/ --output sampler_config.json
GENERATION_ARM=programmatic sbatch --exclude=gnode04 \
    --wrap='cd $PROJECT_ROOT && bash slurm/run_generation.sh'

# 6. Match
python cli.py match --plans plans/ --output matched_pairs.json

# 7. Gate 3 — pilot judge
bash slurm/pre_cache_models.sh qwen_7b_judge
JUDGE_NAME=qwen_7b_judge sbatch --exclude=gnode04 \
    --wrap='cd $PROJECT_ROOT && bash slurm/run_judge.sh'

# 8. Gate 4 — full judge runs (one at a time)
for judge in llama_8b_judge qwen_14b_judge qwen_32b_judge mistral_7b_judge; do
    bash slurm/pre_cache_models.sh $judge
    JUDGE_NAME=$judge sbatch --exclude=gnode04 \
        --wrap="cd $PROJECT_ROOT && bash slurm/run_judge.sh"
    # Wait for completion before submitting next
done

# 9. Analyse
pip install -e ".[analysis]"
python cli.py analyze --judgments judgments/ --plans plans/ --output results/
```

---

## Repository layout

```
judge-bias-study/
├── PREREGISTRATION.md          frozen before any generation run
├── HPC_RUNBOOK.md              gate sequence, disk discipline, failure modes
├── INCIDENTS.md                append-only decision log
├── pyproject.toml
├── cli.py                      generate / match / judge / analyze
├── fixtures/
│   ├── spec.py                 8 fixture axes × levels
│   ├── build.py                synthesise fixture bundles
│   └── data/                   8 committed fixture directories
├── generate/
│   ├── constants.py            EXPLAINER_MODEL_ID (study invariant)
│   ├── provenance.py           PlanProvenance sidecar
│   ├── sampler.py              StructuralSamplerConfig + sample_machine_plan
│   ├── programmatic_arm.py     sampler → guardrails → explainer → finalize
│   ├── llm_arm.py              run_coach_brief wrapper
│   ├── fit_priors.py           fit sampler priors from LLM arm outputs
│   └── run_generation.py       batch generation orchestrator
├── match/
│   └── pair.py                 score → bin → greedy-pair → manifest
├── judge/
│   ├── panel.py                JudgeSpec × 8 panel members
│   ├── vllm_server.py          spawn / health-poll / shutdown
│   ├── harness.py              pair × judge × run × swap loop (JSONL resume)
│   └── outputs.py              PairwiseWriter / SoftEvalWriter / SchemaFailWriter
├── analyze/
│   ├── load.py                 JSONL + provenance → DataFrame
│   ├── models.py               H1 / H3 / H4 mixed-effects fits
│   ├── rubric_deltas.py        H2 paired contrasts (Holm correction)
│   └── figures.py              forest plots, rubric heatmaps, position-bias audit
├── slurm/
│   ├── common.sh               cache paths, activate_env, preflight gate
│   ├── run_preflight.sh        Gate 1
│   ├── run_vllm_smoke.sh       Gate 2
│   ├── run_generation.sh       generation jobs
│   ├── run_judge.sh            Gate 4 workhorse
│   └── pre_cache_models.sh     login-node model pre-cache
├── tools/
│   ├── hpc_preflight.py        preflight report builder
│   ├── clean_caches.sh         recover from orphaned downloads
│   └── mock_llm_server.py      tiny OpenAI-compatible mock server
├── vendor_patches/
│   └── resume_jsonl.py         PR-3 (upstream pending merge)
└── tests/
    ├── run_tests.py            Gate 0 — 22 tests, CPU-only
    ├── create_test_fixtures.py minimal fixture builder
    └── mock_llm_client.py      dummy LLM client (schema-valid JSON)
```

---

## Upstream PRs

Three small PRs are open against `trailtraining`; the study works without them
via fallbacks, but merging them is cleaner.

| PR | File | Change |
|---|---|---|
| PR-1 | `llm/shared.py` | `TRAILTRAINING_LLM_BASE_URL` env override in `make_openrouter_client` |
| PR-2 | `llm/coach.py` | Public `run_training_plan_from_machine_plan()` helper |
| PR-3 | `util/resume_jsonl.py` | Append-only JSONL with dedup by key tuple |

See `patches/` for the full diffs.

---

## Design decisions

**Sidecar provenance, not a contract extension.** Extending `TrainingMeta` would
force contract churn on every downstream user. Sidecars let study metadata evolve
independently.

**Sampler priors fit from LLM arm marginals.** Without this, "judges prefer LLM
plans" has a valid alternative explanation of "programmatic plans looked
structurally weird." With matching priors, both arms share structural summary
statistics — the comparison is about narrative quality given matched structure.

**Three-point Qwen scale ladder.** The local-only constraint rules out 70B Llama.
The Qwen 7B/14B/32B ladder gives three points on a single tuning recipe, which
is statistically stronger than a two-point ladder.

**`skip_synthesis=True` for judging.** The synthesis LLM call is decorative for
this study. Skipping it halves the token cost per soft-eval call and removes one
source of variance.

---

## License

MIT
