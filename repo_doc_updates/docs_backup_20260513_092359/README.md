# llm-judge-self-preference

This repository supports a research study on whether LLM judges systematically prefer LLM-generated trail-training plans over programmatically generated plans when deterministic structural plan quality is controlled.

## Current frozen study state

The current frozen primary design is Qwen/Gemma only.

Source plan families:

- `Qwen/Qwen2.5-7B-Instruct`
- `google/gemma-3-4b-it`
- Programmatic structural-control plans

Primary judge set:

- `qwen_7b_judge`
- `qwen_14b_judge`
- `gemma_4b_judge`
- `gemma_12b_judge`

Primary frozen artifact root:

```text
artifacts/gen_src_t070_exp_t000/frozen_primary_v1/
```

Primary evaluation view:

```text
canonical_masked_scrubbed_v1
```

Primary temperature condition:

```text
eval_t000
```

## Study question

Controlling for deterministic structural plan quality, do LLM judges systematically favor LLM-generated training plans over programmatically generated plans, and does this effect vary by judge model family, model size, rubric/marker, or whether the plan matches the judge's own family?

## Corpora used for frozen primary matching

The full matching pool was built from:

| Corpus | Path | Count | Notes |
|---|---:|---:|---|
| Qwen LLM plans | `artifacts/gen_src_t070_exp_t000/full_qwen/plans` | 192 | 32 cells × 6 plans |
| Gemma 3 LLM plans | `artifacts/gen_src_t070_exp_t000/full_gemma3/plans` | 192 | 32 cells × 6 plans |
| Programmatic plans | `artifacts/gen_src_t070_exp_t000/full_programmatic/plans` | 640 | 32 cells × 20 plans |
| Matching pool | `artifacts/gen_src_t070_exp_t000/matching_pool/plans` | 1,024 | 192 + 192 + 640 |

The frozen matched set contains 250 structurally matched LLM-vs-programmatic pairs.

Matching diagnostics:

```text
matched pairs:                  250
target pairs:                   250
coverage_ok:                    true
coverage_ratio:                 1.0
mean structural score gap:       0.3076
p95 structural score gap:        2.0
max structural score gap:        2.0
pairs by source family:          gemma=116, qwen=134
pairs by athlete band:           A1=60, A2=55, A3=74, A4=61
```

## Primary pairwise evaluation

The full primary pairwise study is complete.

Required design:

```text
250 matched pairs × 4 judge models × 5 repeated runs × 2 AB/BA orders = 10,000 pairwise judgments
```

Observed primary output:

```text
artifacts/gen_src_t070_exp_t000/frozen_primary_v1/judgments_eval_t000_scrubbed_v1_staged/pairwise_all_judges_canonical_masked_scrubbed_v1_t000.jsonl
```

Integrity summary:

```text
records:              10,000
unique record IDs:    10,000
duplicate record IDs: 0
pairs:                250
judges:               2,500 each
orders:               5,000 AB / 5,000 BA
runs:                 2,000 each for run 0-4
```

Primary overall result:

| Outcome | Count | Rate |
|---|---:|---:|
| LLM wins | 3,653 | 36.53% |
| Programmatic wins | 6,347 | 63.47% |

By judge:

| Judge | LLM wins | Programmatic wins | LLM win rate |
|---|---:|---:|---:|
| `gemma_12b_judge` | 740 | 1,760 | 29.60% |
| `gemma_4b_judge` | 836 | 1,664 | 33.44% |
| `qwen_14b_judge` | 1,035 | 1,465 | 41.40% |
| `qwen_7b_judge` | 1,042 | 1,458 | 41.68% |

Order effects are large and must be reported:

| Order | LLM position | LLM wins | N | LLM win rate |
|---|---|---:|---:|---:|
| AB | Plan A | 2,877 | 5,000 | 57.54% |
| BA | Plan B | 776 | 5,000 | 15.52% |

AB/BA pair-run consistency categories:

| Category | Count | Rate among 5,000 pair-run units |
|---|---:|---:|
| `source_consistent_programmatic` | 1,973 | 39.46% |
| `source_consistent_llm` | 626 | 12.52% |
| `position_consistent_plan_a` | 2,251 | 45.02% |
| `position_consistent_plan_b` | 150 | 3.00% |

Interpretation should therefore emphasize both the overall programmatic advantage and the strong position effect.

## Explicit marker-level evaluation

A secondary explicit marker-level pass was added after the primary study. It asks the same judge models to rate each frozen AB/BA pair on nine markers:

- `plan_coherence`
- `training_specificity`
- `load_progression`
- `recovery_safety`
- `quality_session_design`
- `endurance_development`
- `readiness_alignment`
- `clarity_actionability`
- `explanation_quality`

Current uploaded/frozen marker state:

```text
qwen_14b_judge: 2,500 / 2,500 complete
qwen_7b_judge:  2,500 / 2,500 complete
gemma_12b_judge: missing from uploaded marker archive
gemma_4b_judge:  missing from uploaded marker archive
```

The current explicit marker archive therefore contains 5,000 marker records and 45,000 explicit marker decisions. The full four-judge marker study will contain 10,000 marker records and 90,000 explicit marker decisions once Gemma marker passes are complete.

Qwen-only marker result summary, excluding ties:

| Marker | Qwen-only LLM win rate excluding ties |
|---|---:|
| `training_specificity` | 44.14% |
| `clarity_actionability` | 41.37% |
| `readiness_alignment` | 38.08% |
| `quality_session_design` | 37.64% |
| `recovery_safety` | 37.15% |
| `load_progression` | 36.72% |
| `explanation_quality` | 34.76% |
| `plan_coherence` | 34.38% |
| `endurance_development` | 31.35% |

Current marker-level conclusion must be phrased as Qwen-only:

> Among Qwen judges only, explicit marker ratings do not show robust self-family preference. Qwen judges more often favor programmatic plans than LLM-source plans on every explicit marker, including explanation quality. These marker-level results are partial until Gemma marker files are added.

## Important reproducibility notes

Use the environment Python directly on HPC rather than relying on `conda activate`:

```bash
export REPO_ROOT=/mnt/beegfsstudents/home/<USER>/llm-judge-self-preference
export TRAILTRAINING_REPO=/mnt/beegfsstudents/home/<USER>/trailtraining
export PY=/home/<USER>/.conda/envs/judge-bias/bin/python
export PYTHONPATH="${REPO_ROOT}:${TRAILTRAINING_REPO}/src:${PYTHONPATH:-}"
cd "$REPO_ROOT"
```

Do not run full matching or judging jobs on the login node. Use SLURM for all GPU and large-scale evaluation work.

Because of quota constraints, cache and run one model at a time. Do not delete a cached model until its output has been synced and verified locally.

## Known artifact hygiene items

Before final archival:

- Regenerate the empty Gemma 4B primary pairwise `.sha256` file if still empty.
- Regenerate top-level `SHA256SUMS.txt` while excluding itself.
- Remove `.DS_Store` and `__MACOSX` metadata files from public archives.
- Move old smoke outputs, missing-manifest files, and pre-repair failure files to a provenance folder or exclude from clean release bundles.

## Current documentation map

- `README.md`: project overview and current state.
- `HPC_RUNBOOK.md`: HPC setup, cache, launch, sync, verification commands.
- `PREREGISTRATION.md`: frozen design and analysis plan.
- `INCIDENTS.md`: run issues, fixes, and reproducibility lessons.
- `TEMPERATURE_SWEEPS.md`: temperature-condition policy and future sensitivity runs.
- `MARKER_EVAL_RUNBOOK.md`: explicit marker-level evaluation procedure.
- `RESULTS_SUMMARY.md`: primary and current marker results.
- `ARTIFACT_MANIFEST.md`: key artifacts and integrity notes.
