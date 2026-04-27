# TEMPERATURE_SWEEPS.md

Temperature sensitivity workflow for `judge-bias-study`.

This document describes how to run controlled temperature sweeps **without**
changing `trailtraining`. `trailtraining` is treated as a pinned library
dependency; all experiment control lives in this repository.

These runs are **sensitivity analyses**, not the default baseline study path.
For the canonical baseline HPC workflow, see `docs/HPC_RUNBOOK.md`.

## 1. Purpose

Temperature sweeps help answer two different robustness questions:

1. **Generation sensitivity**  
   Does the observed preference pattern change when the LLM arm is generated with
   different source temperatures?

2. **Judge sensitivity**  
   Does the observed preference pattern change when the same fixed plans are
   judged at different judge temperatures?

These are not the same thing.

- Changing **source temperature** changes the generated plans themselves.
- Changing **judge temperature** changes stochasticity in the evaluator while
  keeping the plans fixed.
- Changing **explainer temperature** changes presentation and should be treated
  cautiously because it can reintroduce stylistic leakage.

## 2. Temperature knobs

This repository supports three knobs:

- `source_temperature`
- `explainer_temperature`
- `judge_temperature`

### Recommended interpretation

- `source_temperature`  
  Applies only to the LLM arm source generator.
- `explainer_temperature`  
  Applies to the shared explainer stage for both arms.
- `judge_temperature`  
  Applies to both pairwise and soft-eval judging.

## 3. Recommended baseline and sweep grids

### Baseline study

- `source_temperature = 0.7`
- `explainer_temperature = 0.0`
- `judge_temperature = 0.0`

### Recommended judge-temperature sweep

Keep the plans fixed and vary only:

- `judge_temperature ∈ {0.0, 0.2, 0.7}`

This is the cleanest robustness check because it isolates evaluator stochasticity.

### Recommended source-temperature sweep

Regenerate the LLM arm and rebuild matching for each source setting:

- `source_temperature ∈ {0.0, 0.3, 0.7}`
- `explainer_temperature = 0.0`

This is a generation-condition analysis, not just a “noise” analysis.

### Explainer-temperature guidance

Use `explainer_temperature = 0.0` unless you are explicitly studying whether
explanation-style stochasticity leaks source identity into the judged artifact.

## 4. Non-negotiable condition discipline

Do not mix multiple generation conditions in the same `plans/` directory.

Do not run matching on a directory that contains multiple generation conditions.

Do not pool generation conditions into the baseline estimate without
stratification.

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

  gen_src_t030_exp_t000/
    plans/
    matched_pairs.json
    matching_audit.json
    judgments/
      judge_t000/
````

## 5. Provenance and output expectations

Temperature conditions should be visible in:

* plan IDs
* provenance sidecars
* generation condition labels
* judgment JSONL rows
* judgment output filenames
* artifact directory names

This is required for:

* crash-safe resume
* condition-pure matching
* reliable downstream analysis
* auditability

## 6. Local command examples

## 6a. Baseline generation condition

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

## 6b. Match baseline condition

```bash
python cli.py match \
  --plans artifacts/gen_src_t070_exp_t000/plans \
  --output artifacts/gen_src_t070_exp_t000/matched_pairs.json
```

## 6c. Judge-temperature sweep on fixed plans

```bash
python cli.py judge \
  --judge qwen_7b_judge \
  --judge-temperature 0.0 \
  --plans artifacts/gen_src_t070_exp_t000/plans \
  --pairs artifacts/gen_src_t070_exp_t000/matched_pairs.json \
  --output artifacts/gen_src_t070_exp_t000/judgments/judge_t000

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

## 6d. Source-temperature sweep

```bash
python cli.py generate \
  --arm llm \
  --source-model meta-llama/Llama-3.1-8B-Instruct \
  --source-temperature 0.3 \
  --explainer-temperature 0.0 \
  --output artifacts/gen_src_t030_exp_t000/plans

python cli.py generate \
  --arm programmatic \
  --explainer-temperature 0.0 \
  --output artifacts/gen_src_t030_exp_t000/plans

python cli.py match \
  --plans artifacts/gen_src_t030_exp_t000/plans \
  --output artifacts/gen_src_t030_exp_t000/matched_pairs.json
```

Judge that new generation condition separately.

## 7. HPC submission pattern

The safest HPC pattern is:

1. One job per **generation condition**
2. One matching step per **generation condition**
3. Multiple judge jobs per **fixed matched plan set**

### Example: baseline generation condition

```bash
sbatch \
  --account=3202029 \
  --partition=stud \
  --qos=stud \
  --exclude=gnode04 \
  --export=ALL,GENERATION_ARM=llm,GENERATION_PROFILE=exact,LLM_SOURCE_MODEL=meta-llama/Llama-3.1-8B-Instruct,SOURCE_TEMPERATURE=0.7,EXPLAINER_TEMPERATURE=0.0,OUTPUT_DIR=artifacts/gen_src_t070_exp_t000/plans \
  --wrap="cd ${REPO_ROOT} && bash slurm/run_generation_hpc.sh"
```

### Example: judge-temperature sweep on fixed plans

```bash
JUDGE_TEMPERATURE=0.0 JUDGMENTS_DIR=artifacts/gen_src_t070_exp_t000/judgments/judge_t000 \
  bash slurm/submit_judge_hpc.sh qwen_7b_judge

JUDGE_TEMPERATURE=0.2 JUDGMENTS_DIR=artifacts/gen_src_t070_exp_t000/judgments/judge_t020 \
  bash slurm/submit_judge_hpc.sh qwen_7b_judge

JUDGE_TEMPERATURE=0.7 JUDGMENTS_DIR=artifacts/gen_src_t070_exp_t000/judgments/judge_t070 \
  bash slurm/submit_judge_hpc.sh qwen_7b_judge
```

Adjust your SLURM wrappers so they forward:

* `SOURCE_TEMPERATURE`
* `EXPLAINER_TEMPERATURE`
* `JUDGE_TEMPERATURE`
* condition-specific output directories

## 8. Recommended analysis strategy

## 8a. Judge-temperature sensitivity

Keep plans and matching fixed. Vary only `judge_temperature`.

Report separately:

* pairwise LLM win rate by judge temperature
* per-rubric soft-eval changes by judge temperature
* run-to-run variability by judge temperature
* self-family preference by judge temperature

## 8b. Source-temperature sensitivity

Regenerate LLM-arm plans, rerun matching, then rejudge.

Report separately:

* LLM win rate by generation condition
* matching coverage by generation condition
* score-gap distribution by generation condition
* any change in style-leakage audit by generation condition

Do not present this as just “evaluation randomness.” It changes the generated
artifacts themselves.

## 8c. Explainer-temperature sensitivity

Only run this if you are explicitly studying whether explanation stochasticity
changes judge preference or introduces source-identifying surface cues.

## 9. Reporting rules

For the paper or report:

* baseline estimates should come from the frozen baseline condition
* temperature sweeps should be reported as sensitivity analyses
* judge-temperature results and source-temperature results should be separated
* mixed-condition pooling should be avoided unless explicitly modeled

## 10. Common failure modes

### Mixed plans directory

If you place multiple generation conditions in one `plans/` directory, matching
can mix them and invalidate interpretation.

### Resume collisions

If temperature is not part of artifact identity, repeated runs can silently reuse
or overwrite previous results.

### Overclaiming from source-temperature sweeps

A source-temperature effect is not the same as a pure judge-bias effect.

### Explainer leakage

Nonzero explainer temperature may change surface form enough to introduce
detectable source cues.

## 11. Minimal recommended sweep set

If compute is limited, run only:

1. baseline frozen condition
2. judge-temperature sweep on one or two key judges
3. one additional source-temperature condition such as `0.3`

That gives you meaningful robustness evidence without exploding the matrix.

````
