# LLM Judge Self-Preference Study

This repository supports a controlled empirical study of whether LLM judges systematically prefer LLM-generated training plans over programmatically generated control plans when deterministic structural plan quality is controlled.

## Current Qwen/Gemma design

The current study uses only Qwen and Gemma source families:

- Qwen LLM source: `Qwen/Qwen2.5-7B-Instruct`
- Gemma LLM source: `google/gemma-3-4b-it`
- Programmatic controls: deterministic structural sampler plus the shared explainer for narrative fields

The current candidate pool is expected to contain:

| Corpus | Path | Plans |
|---|---:|---:|
| Qwen LLM source | `artifacts/gen_src_t070_exp_t000/full_qwen/plans` | 192 |
| Gemma 3 LLM source | `artifacts/gen_src_t070_exp_t000/full_gemma3/plans` | 192 |
| Programmatic controls | `artifacts/gen_src_t070_exp_t000/full_programmatic/plans` | 640 |
| Matching pool | `artifacts/gen_src_t070_exp_t000/matching_pool/plans` | 1024 |

The full pairwise-judgment requirement is:

```text
250 matched pairs x 4 judge models x 5 repeated runs x 2 AB/BA positions = 10,000 pairwise judgments
```

The preferred run uses at least 256 matched pairs, yielding at least 10,000 pairwise judgment records.

Do not count generated training plans as pairwise evaluation documents. One pairwise evaluation document is one judge-facing comparison for one matched pair, one judge model, one run index, and one AB/BA order.

## Why structural scoring is required

The legacy matcher called TrailTraining's full deterministic quality score and also used feature-distance terms derived from narrative/prose length. That made matching vulnerable to presentation confounds. The primary matching score is now `structural_score_v1.0.0`, which excludes:

- title wording
- workout and purpose prose richness
- citations and claim attributions
- data-note verbosity
- rationale/explanation fields
- source model name, generation arm, and file names

The score uses source-neutral structure only: session types, durations, rest/hard flags, active-day count, hard/rest spacing, long-run and quality-session counts, plan length, and fixture-cell constraints such as athlete band, readiness, recovery capability, and race phase.

## Structural matching command

```bash
export REPO_ROOT=/mnt/beegfsstudents/home/<USER_ID>/llm-judge-self-preference
export TRAILTRAINING_REPO=/mnt/beegfsstudents/home/<USER_ID>/trailtraining
export PY=/home/<USER_ID>/.conda/envs/judge-bias/bin/python
export PYTHONPATH="${REPO_ROOT}:${TRAILTRAINING_REPO}/src:${PYTHONPATH:-}"
cd "$REPO_ROOT"

$PY cli.py match \
  --plans artifacts/gen_src_t070_exp_t000/matching_pool/plans \
  --output artifacts/gen_src_t070_exp_t000/matching_pool/matched_pairs.json \
  --allow-mixed-generation-conditions \
  --tolerance 2.0 \
  --target-pairs 250 \
  --fail-below-target-ratio 1.0
```

The final full-study gate must fail if fewer than 250 pairs are produced.

## Diagnostics before judging

```bash
$PY cli.py match-diagnostics \
  --plans artifacts/gen_src_t070_exp_t000/matching_pool/plans \
  --output artifacts/gen_src_t070_exp_t000/matching_pool/structural_diagnostics

$PY cli.py build-eval-manifest \
  --plans artifacts/gen_src_t070_exp_t000/matching_pool/plans \
  --pairs artifacts/gen_src_t070_exp_t000/matching_pool/matched_pairs.json \
  --output artifacts/gen_src_t070_exp_t000/matching_pool/eval_manifest \
  --seed 20260506

$PY cli.py launch-gate \
  --plans artifacts/gen_src_t070_exp_t000/matching_pool/plans \
  --pairs artifacts/gen_src_t070_exp_t000/matching_pool/matched_pairs.json \
  --style-audit artifacts/gen_src_t070_exp_t000/matching_pool/results/style_audit_summary.json
```

The launch gate checks: at least 250 pairs, expected pairwise documents at least 10,000, both Qwen and Gemma source plans represented, exactly four Qwen/Gemma judges, five repeats, both AB and BA orders, structural matching audit, and source masking audit.

## Pairwise judging

Pairwise judging uses `compare_plans` from `trailtraining.llm.soft_eval` via `judge/harness.py`. Source-family metadata is retained only in output metadata; it is not placed in the judge-facing plan payload.

Run one job per judge:

```bash
for J in qwen_7b_judge qwen_14b_judge gemma_4b_judge gemma_12b_judge; do
  JUDGE_NAME="$J" \
  PLANS_DIR=artifacts/gen_src_t070_exp_t000/matching_pool/plans \
  PAIRS_FILE=artifacts/gen_src_t070_exp_t000/matching_pool/matched_pairs.json \
  JUDGMENTS_DIR=artifacts/gen_src_t070_exp_t000/matching_pool/judgments_t000 \
  PAIRWISE_VIEW=canonical_masked \
  JUDGE_TEMPERATURE=0.0 \
  sbatch slurm/run_judge_hpc.sh
done
```

Temperature sensitivity runs should use separate output directories such as `judgments_eval_t030` and must not be mixed with the primary `judgments_t000` results.

## Analysis

```bash
$PY tools/analyze_pairwise_results.py \
  --judgments artifacts/gen_src_t070_exp_t000/matching_pool/judgments_t000 \
  --pairs artifacts/gen_src_t070_exp_t000/matching_pool/matched_pairs.json \
  --output artifacts/gen_src_t070_exp_t000/matching_pool/results_t000
```

The analysis reports overall LLM win rate, bootstrap confidence intervals, per-judge and per-family summaries, self-family summaries where defined, order bias, and score-gap sensitivity. Do not fill report conclusions until real judge outputs exist.

## v4 validity tightening

After final artifact inspection, the primary risk shifted from pair count to residual structural and presentation confounding. The v4 patch therefore uses target-cardinality min-cost structural matching with soft caliper penalties and makes `canonical_masked` the default judge-facing view. Strict hard calipers are available through `cli.py match --caliper feature=value`, but the current candidate pool does not reach 250 pairs under the strict calipers recommended for a cleaner primary causal claim; use those calipers to size targeted programmatic top-up generation or to define a smaller pilot.
